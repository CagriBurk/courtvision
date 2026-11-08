"""
05_predict_today.py
-------------------
Fetches today's NBA games from nba_api and predicts them.

Live features are produced by the SAME pipeline as training
(feature_engineering.build_feature_matrix): today's games are appended to
the master table as rows without a result, so shift(1) makes their
features depend only on games already played.

Output:
  data/live/today_schedule.json
  data/live/live_features.parquet
  data/processed/master_games.parquet  (current season refreshed)
  data/processed/market_lines.parquet  (today's ESPN / DraftKings lines)
  reports/predictions_YYYY-MM-DD.html

Usage:
    python src/05_predict_today.py
    python src/05_predict_today.py --date 2026-02-22  # specific day
"""

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from espn_odds import update_today_lines
from feature_engineering import build_feature_matrix
from market_blend import build_lines, load_market_distribution
from modeling import TASKS, apply_calibration, explain_prediction
from nba_data_client import (
    NBA_API_AVAILABLE,
    fetch_player_season,
    fetch_season_wide,
    fetch_today_schedule,
)
from player_stats import get_injuries, match_player_ids
from prediction_tracker import save_predictions, fill_results
from season import current_season, nba_today, season_id, season_start_year

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
MODELS_DIR  = ROOT / "models"
PROC_DIR    = ROOT / "data" / "processed"
CACHE_DIR   = ROOT / "data" / "api_cache"
LIVE_DIR    = ROOT / "data" / "live"
REPORTS_DIR = ROOT / "reports"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Same type rules as 01_build_master (pyarrow rejects mixed types)."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ["game_id", "team_id_home", "team_id_away", "season_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# ---------------------------------------------------------------------------
# 1. Refresh the current season in the master table
# ---------------------------------------------------------------------------
def refresh_current_season(master: pd.DataFrame, season: str) -> pd.DataFrame:
    """Re-fetch `season` with one LeagueGameLog call and upsert it into master."""
    fresh = fetch_season_wide(season)
    if fresh.empty:
        print(f"  [UYARI] {season} çekilemedi, mevcut master kullanılıyor.")
        return master

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(CACHE_DIR / f"season_{season}.parquet", index=False)

    fresh = _normalize_ids(fresh)
    master = master[~master["game_id"].isin(fresh["game_id"])]
    combined = pd.concat([master, fresh], ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)
    combined.to_parquet(PROC_DIR / "master_games.parquet", index=False)
    print(f"  → Master güncellendi: {len(combined)} maç (son: {combined['game_date'].max().date()})")
    return combined


def refresh_current_players(season: str) -> pd.DataFrame | None:
    """Re-fetch the season's player box scores and upsert player_games.parquet."""
    path = PROC_DIR / "player_games.parquet"
    existing = pd.read_parquet(path) if path.exists() else None
    fresh = fetch_player_season(season)
    if fresh.empty:
        print(f"  [UYARI] {season} oyuncu verisi çekilemedi, mevcut veri kullanılıyor.")
        return existing

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(CACHE_DIR / f"players_{season}.parquet", index=False)
    if existing is not None:
        existing = existing[~existing["game_id"].astype(str).isin(fresh["game_id"])]
        fresh = pd.concat([existing, fresh], ignore_index=True)
    fresh.to_parquet(path, index=False)
    return fresh


# Injury report statuses treated as "will not play"
OUT_STATUSES = {"Out", "Doubtful"}


def build_absent_override(games: list[dict], player_games: pd.DataFrame,
                          injuries: dict, target_date: date) -> tuple[dict | None, list[str]]:
    """
    {(game_id, team_id): {player_id, ...}} of injured players for each upcoming
    game, matched by name against each team's current roster (players whose
    last appearance was for that team). None if the injury report is missing.
    """
    if not injuries:
        return None, []

    target_ts = pd.Timestamp(target_date)
    recent = player_games[
        (pd.to_datetime(player_games["game_date"]) < target_ts)
        & (pd.to_datetime(player_games["game_date"]) >= target_ts - pd.Timedelta(days=400))
    ]
    last_team = recent.sort_values("game_date").groupby("player_id").tail(1)

    override, unmatched = {}, []
    for g in games:
        for side in ("home", "away"):
            team_id = str(g[f"{side}_team_id"])
            abbr = g[f"{side}_team_abbr"]
            roster = last_team[last_team["team_id"].astype(str) == team_id]
            candidates = dict(zip(roster["player_id"].astype(str), roster["player_name"]))
            names = [p["name"] for p in injuries.get(abbr, []) if p.get("status") in OUT_STATUSES]
            ids, missed = match_player_ids(names, candidates)
            override[(str(g["game_id"]), team_id)] = ids
            unmatched += [f"{abbr}: {n}" for n in missed]
    return override, unmatched


# ---------------------------------------------------------------------------
# 2. Live features through the training pipeline
# ---------------------------------------------------------------------------
def build_pending_rows(games: list[dict], target_date: date) -> pd.DataFrame:
    """Today's games in master (wide) format, without a result."""
    sid = season_id(season_start_year(target_date))
    return pd.DataFrame([
        {
            "game_id":                str(g["game_id"]),
            "game_date":              pd.Timestamp(target_date),
            "season_id":              sid,
            "season_type":            "Regular Season",
            "team_id_home":           str(g["home_team_id"]),
            "team_abbreviation_home": g["home_team_abbr"],
            "team_name_home":         g["home_team_name"],
            "team_id_away":           str(g["away_team_id"]),
            "team_abbreviation_away": g["away_team_abbr"],
            "team_name_away":         g["away_team_name"],
            "target":                 np.nan,
        }
        for g in games
    ])


def compute_live_features(master: pd.DataFrame, games: list[dict], target_date: date,
                          player_games: pd.DataFrame | None = None,
                          absent_override: dict | None = None) -> pd.DataFrame:
    """Game-level feature rows for `games`, indexed by game_id."""
    target_ts = pd.Timestamp(target_date)
    history = master[master["game_date"] < target_ts]
    pending = build_pending_rows(games, target_date)

    # Rolling windows only need the previous season onwards (L10 + season-to-date);
    # Elo and H2H still use the full history.
    rolling_since = pd.Timestamp(f"{season_start_year(target_date) - 1}-08-01")
    features, _ = build_feature_matrix(
        pd.concat([history, pending], ignore_index=True),
        rolling_since=rolling_since,
        player_games=player_games,
        absent_override=absent_override,
    )
    return features[features["game_id"].isin(pending["game_id"])].set_index("game_id")


# ---------------------------------------------------------------------------
# 3. Predict
# ---------------------------------------------------------------------------
def recent_form(master: pd.DataFrame, team_id: str, before: date, n: int = 10) -> list[int]:
    """Last n results (1=W, 0=L) of a team before `before`, oldest first."""
    games = master[
        ((master["team_id_home"] == team_id) | (master["team_id_away"] == team_id))
        & (master["game_date"] < pd.Timestamp(before))
        & master["target"].notna()
    ].sort_values("game_date").tail(n)
    is_home = games["team_id_home"] == team_id
    won = games["target"].where(is_home, 1 - games["target"])
    return [int(w) for w in won]


def _num(v):
    """numpy scalar → plain float (None if missing) for JSON/parquet output."""
    return None if pd.isna(v) else float(v)


def _int(v):
    return None if pd.isna(v) else int(v)


def _lines_summary(g: dict, lines: dict) -> str:
    """'Skor 109-117 | Handikap: BOS -8.5 | Toplam 226.5' style one-liner."""
    parts = []
    if "projected" in lines:
        parts.append(f"Skor {lines['projected']['home']:.0f}-{lines['projected']['away']:.0f}")
    if "margin" in lines:
        fl = lines["margin"]["fair_line"]
        fav = g["home_team_abbr"] if fl > 0 else g["away_team_abbr"]
        parts.append(f"Handikap: {fav} -{abs(fl):.1f}")
    if "total" in lines:
        parts.append(f"Toplam {lines['total']['fair_line']:.1f}")
    return " | ".join(parts)


def _margin_label(g: dict, margin: float | None) -> str:
    """Expected home margin → 'DAL -7.5' (the favourite and its handicap)."""
    if margin is None:
        return "-"
    fav = g["home_team_abbr"] if margin >= 0 else g["away_team_abbr"]
    return f"{fav} -{abs(margin):.1f}"


def _market_summary(g: dict, info: dict) -> str:
    """'Piyasa DAL -7.5 / 222.5 · model DAL -12.5' style one-liner."""
    parts = [f"Piyasa {_margin_label(g, info['market_margin'])}"]
    if info.get("market_total") is not None:
        parts[0] += f" / {info['market_total']:.1f}"
    if info.get("model_fair_margin") is not None:
        parts.append(f"model {_margin_label(g, info['model_fair_margin'])}")
    return " · ".join(parts)


def load_line_models() -> tuple[dict, dict, str]:
    """
    Margin / total regressors and their error distributions (from 03 and 04),
    plus which model supplies the home win probability (metrics.json).
    Missing files simply switch the handicap / total output off.
    """
    dist_path = MODELS_DIR / "distribution.json"
    dists = json.loads(dist_path.read_text(encoding="utf-8")) if dist_path.exists() else {}
    models = {}
    for task in ("margin", "total"):
        path = MODELS_DIR / TASKS[task]["model_file"]
        if path.exists() and task in dists:
            with open(path, "rb") as f:
                models[task] = pickle.load(f)
            models[task].set_params(device="cpu")

    metrics_path = MODELS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    win_source = metrics.get("win_source", "classifier")
    if win_source == "margin" and "margin" not in models:
        win_source = "classifier"
    return models, dists, win_source


def predict_games(games: list[dict], live_feats: pd.DataFrame, model,
                  feature_columns: list, target_date: date,
                  calibration: dict | None = None,
                  line_models: dict | None = None, dists: dict | None = None,
                  win_source: str = "classifier", market: dict | None = None,
                  market_dist: dict | None = None) -> list[dict]:
    predictions = []
    line_models = line_models or {}
    market = market or {}

    for g in games:
        gid = str(g["game_id"])
        if gid not in live_feats.index:
            print(f"  [UYARI] {g['home_team_abbr']} vs {g['away_team_abbr']}: feature üretilemedi, atlanıyor.")
            continue
        row = live_feats.loc[gid]

        X_live = row.reindex(feature_columns).to_frame().T.astype(float)
        raw_prob = model.predict_proba(X_live)[0][1]
        prob_home = float(apply_calibration(calibration, raw_prob))
        # Modelin kendi gerekçesi (SHAP katkıları, okunur gruplara toplanmış)
        model_factors = explain_prediction(model, X_live, calibration)

        # Handikap / toplam: piyasa çizgisi varsa onun etrafında (market_blend),
        # yoksa modelin fark / toplam tahmini → barem olasılıkları
        reg_pred = {t: float(m.predict(X_live)[0]) for t, m in line_models.items()}
        lines, model_lines, market_info = build_lines(dists or {}, reg_pred, market.get(gid), market_dist)
        source = "classifier"
        prob_clf = prob_home
        if win_source == "margin" and model_lines and "margin" in model_lines:
            prob_home = model_lines["margin"]["p_home_win"]
            source = "margin"
        prob_away = 1.0 - prob_home
        winner    = "HOME" if prob_home > 0.5 else "AWAY"
        conf      = "YÜKSEK" if max(prob_home, prob_away) >= 0.65 else \
                    "ORTA"   if max(prob_home, prob_away) >= 0.55 else "DÜŞÜK"

        result = {
            **g,
            "date":             str(target_date),
            "prob_home":        prob_home,
            "prob_away":        prob_away,
            "predicted_winner": winner,
            "confidence":       conf,
            "model_factors":    json.dumps(model_factors, ensure_ascii=False),
            "win_source":       source,
            "prob_home_classifier": prob_clf,
            "home_games_played": _int(row.get("home_season_game_num")),
            "away_games_played": _int(row.get("away_season_game_num")),
            # Handikap / toplam (modeller yoksa None)
            "pred_margin":      reg_pred.get("margin"),
            "pred_total":       reg_pred.get("total"),
            "lines":            json.dumps(lines) if lines else None,
            "model_lines":      json.dumps(model_lines) if model_lines else None,
            # Piyasa (ESPN / DraftKings): çizgi, marjsız ev olasılığı, model ile kıyas
            "market":           json.dumps(market_info) if market_info else None,
            "prob_home_market": market_info["p_home"] if market_info else None,
            # Form özeti
            "home_form_L5":     _num(row.get("home_won_L5")),
            "home_form_L10":    _num(row.get("home_won_L10")),
            "home_b2b":         _int(row.get("home_is_back_to_back")) or 0,
            "home_rest":        _int(row.get("home_days_since_last_game")),
            "home_streak":      _int(row.get("home_current_streak")) or 0,
            "away_form_L5":     _num(row.get("away_won_L5")),
            "away_form_L10":    _num(row.get("away_won_L10")),
            "away_b2b":         _int(row.get("away_is_back_to_back")) or 0,
            "away_rest":        _int(row.get("away_days_since_last_game")),
            "away_streak":      _int(row.get("away_current_streak")) or 0,
            "h2h_games":        _int(row.get("h2h_n_games")) or 0,
            "h2h_home_wr":      _num(row.get("h2h_home_win_rate")),
            "h2h_pts_diff":     _num(row.get("h2h_pts_diff_mean")),
            # Key diff features — UI faktörleri + ileride analiz için
            "elo_diff":            _num(row.get("elo_diff")),
            "elo_home":            _num(row.get("elo_home")),
            "elo_away":            _num(row.get("elo_away")),
            "diff_plus_minus_L10": _num(row.get("diff_plus_minus_L10")),
            "diff_plus_minus_L5":  _num(row.get("diff_plus_minus_L5")),
            "diff_season_win_pct": _num(row.get("diff_season_win_pct")),
            "diff_season_pd":      _num(row.get("diff_season_pd")),
            "diff_srs_L10":        _num(row.get("diff_srs_L10")),
            "fatigue_diff_b2b":    _num(row.get("fatigue_diff_b2b")),
            # Oyuncu eksikliği (sakatlık raporundan)
            "home_missing_value_share": _num(row.get("home_missing_value_share")),
            "away_missing_value_share": _num(row.get("away_missing_value_share")),
            "home_n_missing_rotation":  _int(row.get("home_n_missing_rotation")),
            "away_n_missing_rotation":  _int(row.get("away_n_missing_rotation")),
        }
        print(f"  {g['home_team_abbr']} %{prob_home * 100:.1f} | {g['away_team_abbr']} %{prob_away * 100:.1f} [{conf}]"
              + (f"  | {_lines_summary(g, lines)}" if lines else "")
              + (f"  | {_market_summary(g, market_info)}" if market_info else ""))
        predictions.append(result)

    return predictions


# ---------------------------------------------------------------------------
# 4. HTML raporu üret
# ---------------------------------------------------------------------------
def _streak_str(s: int) -> str:
    if s > 0: return f"G{s}"
    if s < 0: return f"M{abs(s)}"
    return "-"


def generate_report(predictions: list[dict], target_date: date) -> str:
    date_str = target_date.strftime("%d %B %Y")

    cards = ""
    for p in predictions:
        winner_label = p["home_team_name"] if p["predicted_winner"] == "HOME" else p["away_team_name"]
        conf_color   = {"YÜKSEK": "#27ae60", "ORTA": "#e67e22", "DÜŞÜK": "#e74c3c"}.get(p["confidence"], "#555")

        home_b2b_badge = '<span style="color:red;font-weight:bold"> ⚡B2B</span>' if p["home_b2b"] else ""
        away_b2b_badge = '<span style="color:red;font-weight:bold"> ⚡B2B</span>' if p["away_b2b"] else ""

        home_form_pct = f"{p['home_form_L5']:.0%}" if p["home_form_L5"] is not None else "—"
        away_form_pct = f"{p['away_form_L5']:.0%}" if p["away_form_L5"] is not None else "—"
        lines_html = (f'<div class="h2h">{_lines_summary(p, json.loads(p["lines"]))}</div>'
                      if p.get("lines") else "")

        cards += f"""
<div class="card">
  <div class="matchup">{p['home_team_abbr']} (Ev) vs {p['away_team_abbr']} (Dep.)</div>
  <div class="prediction">
    <span class="team-name">{winner_label}</span> kazanır
    <span class="conf-badge" style="background:{conf_color}">{p['confidence']}</span>
  </div>
  <div class="prob-bar">
    <div class="home-bar" style="width:{p['prob_home']*100:.1f}%">{p['prob_home']:.1%}</div>
    <div class="away-bar" style="width:{p['prob_away']*100:.1f}%">{p['prob_away']:.1%}</div>
  </div>
  <div class="labels"><span>{p['home_team_abbr']}</span><span>{p['away_team_abbr']}</span></div>
  <div class="details">
    <div class="detail-col">
      <b>{p['home_team_name']}{home_b2b_badge}</b><br>
      Last 5 games: {home_form_pct} | Streak: {_streak_str(p['home_streak'])} | Rest: {p['home_rest'] or '?'} days
    </div>
    <div class="detail-col">
      <b>{p['away_team_name']}{away_b2b_badge}</b><br>
      Last 5 games: {away_form_pct} | Streak: {_streak_str(p['away_streak'])} | Rest: {p['away_rest'] or '?'} days
    </div>
  </div>
  <div class="h2h">
    H2H ({p['h2h_games']} maç): Ev sahibi win rate %{p['h2h_home_wr']:.1%} |
    Avg score diff: {p['h2h_pts_diff']:+.1f}
  </div>
  {lines_html}
</div>"""

    conf_colors = {"YÜKSEK": "#27ae60", "ORTA": "#e67e22", "DÜŞÜK": "#e74c3c"}
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{p['home_team_abbr']} vs {p['away_team_abbr']}</td>"
        f"<td>{p['prob_home']:.1%}</td>"
        f"<td>{p['prob_away']:.1%}</td>"
        f"<td><b>{p['home_team_abbr'] if p['predicted_winner'] == 'HOME' else p['away_team_abbr']}</b></td>"
        f"<td><span style='color:{conf_colors.get(p['confidence'], '#555')}'>{p['confidence']}</span></td>"
        "</tr>"
        for p in predictions
    )

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>NBA Tahminleri — {date_str}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e0e0e0; max-width: 900px; margin: 0 auto; padding: 30px 20px; }}
  h1 {{ color: #e94560; text-align: center; font-size: 2em; margin-bottom: 4px; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
  .card {{ background: #1a1a2e; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #2a2a4a; }}
  .matchup {{ font-size: 1.1em; color: #888; margin-bottom: 8px; }}
  .prediction {{ font-size: 1.5em; font-weight: bold; margin-bottom: 12px; }}
  .team-name {{ color: #e94560; }}
  .conf-badge {{ font-size: 0.6em; padding: 3px 10px; border-radius: 12px; color: white; margin-left: 10px; vertical-align: middle; }}
  .prob-bar {{ display: flex; height: 32px; border-radius: 8px; overflow: hidden; margin-bottom: 4px; }}
  .home-bar {{ background: #e94560; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.9em; }}
  .away-bar {{ background: #16213e; display: flex; align-items: center; justify-content: center; color: #aaa; font-size: 0.9em; }}
  .labels {{ display: flex; justify-content: space-between; font-size: 0.85em; color: #888; margin-bottom: 14px; }}
  .details {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.9em; padding: 12px 0; border-top: 1px solid #2a2a4a; }}
  .detail-col {{ line-height: 1.7; }}
  .h2h {{ font-size: 0.85em; color: #888; padding-top: 10px; border-top: 1px solid #2a2a4a; margin-top: 10px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1a2e; border-radius: 10px; overflow: hidden; margin-top: 20px; }}
  th {{ background: #16213e; padding: 12px; text-align: center; color: #e0e0e0; }}
  td {{ padding: 10px; text-align: center; border-bottom: 1px solid #2a2a4a; }}
  tr:hover td {{ background: #2a2a4a; }}
</style>
</head>
<body>
<h1>🏀 NBA TAHMİNLERİ</h1>
<div class="subtitle">{date_str} &nbsp;|&nbsp; {len(predictions)} maç &nbsp;|&nbsp; XGBoost Modeli</div>

{cards}

<h2 style="color:#e94560; margin-top:30px">📊 Özet Tablo</h2>
<table>
  <tr><th>Maç</th><th>Ev %</th><th>Dep. %</th><th>Tahmin</th><th>Güven</th></tr>
  {summary_rows}
</table>
</body>
</html>"""

    return html


def update_coupons(predictions: list[dict], dists: dict, target_date: date,
                   market_dist: dict | None = None) -> None:
    """Today's handicap coupons (Faz 6/6b): build, log (locked ones kept), live view for the API."""
    import coupon_tracker
    from coupons import build_coupons

    day_str = str(target_date)
    started = {str(p["game_id"]) for p in predictions if p.get("game_status", 1) != 1}
    market_margin = (market_dist or {}).get("margin")
    if "margin" not in dists and market_margin is None:
        day = {"coupons": [], "games": [], "n_legs": 0,
               "note": "Handikap dağılımı yok: önce 04 (ya da 09) çalıştırılmalı."}
    elif not predictions:
        day = {"coupons": [], "games": [], "n_legs": 0, "note": "Bugün programda maç yok."}
    else:
        # Locked coupons stay as they are; their games are not reused
        day = build_coupons(predictions, dists.get("margin"), market_margin,
                            coupon_tracker.locked_games(day_str, started))
    today = coupon_tracker.save_coupons(day_str, day, started) if day["coupons"] or predictions else []
    coupon_tracker.write_live_view(day_str, day, today)

    if today:
        print("\nKuponlar:")
        for c in today:
            lock = " [kilitli]" if c.get("locked") else ""
            legs = " · ".join(l["label"] for l in c["legs"])
            print(f"  {c['label']:8} %{c['p'] * 100:4.1f}  en az {c['min_odds']:.2f}{lock} | {legs}")
    elif day.get("note"):
        print(f"\nKupon yok: {day['note']}")


def write_live_meta(target_date: date, n_games: int) -> None:
    """data/live/meta.json — the API uses it to decide if predictions are fresh."""
    meta = {
        "date":         str(target_date),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_games":      n_games,
    }
    (LIVE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main(target_date: date | None = None):
    if not NBA_API_AVAILABLE:
        raise SystemExit("nba_api yüklü değil: pip install nba_api")

    if target_date is None:
        target_date = nba_today()

    print(f"\n{'='*60}")
    print(f"NBA Günlük Tahmin — {target_date}")
    print(f"{'='*60}\n")

    # Model yükle
    with open(MODELS_DIR / "xgb_best.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    model.set_params(device="cpu")  # inference CPU'da, CUDA device mismatch uyarısını kapat
    print(f"Model loaded. Feature count: {len(feature_columns)}")

    # 04_evaluate_model.py writes this only if it improved the holdout log-loss
    cal_path = MODELS_DIR / "calibration.json"
    calibration = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
    print(f"Kalibrasyon: {'açık (Platt)' if calibration else 'kapalı'}")

    # Handikap / toplam modelleri (03 --task margin/total + 04 sonrası)
    line_models, dists, win_source = load_line_models()
    print(f"Handikap/toplam: {', '.join(line_models) if line_models else 'model yok'} | "
          f"Kazanan olasılığı: {'fark modeli' if win_source == 'margin' else 'sınıflandırıcı'}\n")

    # Master tablo + güncel sezon
    master = _normalize_ids(pd.read_parquet(PROC_DIR / "master_games.parquet"))
    season = current_season(target_date)
    print(f"Güncel sezon yenileniyor ({season})...")
    master = refresh_current_season(master, season)
    player_games = refresh_current_players(season)

    # Maç programı
    games = fetch_today_schedule(target_date)
    live_path = LIVE_DIR / "live_features.parquet"
    if not games:
        print("Bugün programda maç yok.")
        pd.DataFrame().to_parquet(live_path, index=False)
        write_live_meta(target_date, 0)
        update_coupons([], dists, target_date)
        fill_results(master)
        return

    schedule_path = LIVE_DIR / "today_schedule.json"
    schedule_path.write_text(json.dumps(games, indent=2, ensure_ascii=False), encoding="utf-8")

    # Piyasa çizgileri (ESPN / DraftKings): tek istek, yalnız başlamamış maçlar güncellenir
    market = {}
    try:
        market = update_today_lines(target_date, games)
        print(f"Piyasa çizgisi: {len(market)}/{len(games)} maç")
    except Exception as e:
        print(f"  [UYARI] Piyasa çizgileri alınamadı: {e}")
    market_dist = load_market_distribution()
    if market_dist is None:
        print("  Piyasa dağılımı yok (09_market_eval.py çalıştırılmalı): çizgiler modelden")
    else:
        print(f"  Çizgiler piyasa merkezli (model payı: fark w={market_dist['blend']['margin']['w']:.2f}, "
              f"toplam w={market_dist['blend']['total']['w']:.2f})")

    # Feature'lar (eğitimle aynı pipeline) + tahminler.
    # Başlamış/bitmiş maçlar da tahmin edilir: feature'lar sadece önceki
    # günlerin maçlarına dayanır, gün içinde değişmez ve sızıntı yoktur.
    # Böylece dashboard gün boyu bütün maçları gösterir.
    absent_override = None
    if player_games is not None:
        absent_override, unmatched = build_absent_override(
            games, player_games, get_injuries(force_refresh=True), target_date
        )
        if absent_override is None:
            print("  [UYARI] Sakatlık raporu alınamadı — oyuncu eksikliği bilinmiyor (NaN)")
        else:
            n_out = sum(len(v) for v in absent_override.values())
            print(f"  Sakatlık raporu: {n_out} oyuncu eksik olarak eşleşti")
            if unmatched:
                print(f"  Eşleşmeyen isimler (kadroda son maçı yok olabilir): {', '.join(unmatched)}")

    print("\nFeature'lar hesaplanıyor (eğitim pipeline'ı)...")
    live_feats = compute_live_features(master, games, target_date, player_games, absent_override)

    print("\nTahminler hesaplanıyor...")
    predictions = predict_games(games, live_feats, model, feature_columns, target_date, calibration,
                                line_models, dists, win_source, market, market_dist)
    for p in predictions:
        p["home_form_seq"] = recent_form(master, str(p["home_team_id"]), target_date)
        p["away_form_seq"] = recent_form(master, str(p["away_team_id"]), target_date)

    pd.DataFrame(predictions).to_parquet(live_path, index=False)
    write_live_meta(target_date, len(predictions))

    # HTML raporu
    report_html = generate_report(predictions, target_date)
    report_path = REPORTS_DIR / f"predictions_{target_date.strftime('%Y-%m-%d')}.html"
    report_path.write_text(report_html, encoding="utf-8")

    # Takip kaydına sadece maç başlamadan yapılan tahminler girer (dürüst isabet oranı)
    pre_tip = [p for p in predictions if p.get("game_status", 1) == 1]
    if pre_tip:
        save_predictions(pre_tip)
    update_coupons(predictions, dists, target_date, market_dist)
    fill_results(master)  # sonuçları ve kuponları kapatır

    print(f"\n{'='*60}")
    print(f"Tamamlandı! Rapor: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Piped stdout (API subprocess) uses the Windows codepage; a player name
    # like "Porziņģis" must not crash the run.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Tahmin tarihi (YYYY-MM-DD), varsayılan: bugün (ABD Doğu saati)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    main(target_date=target)
