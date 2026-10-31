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
  reports/predictions_YYYY-MM-DD.html

Usage:
    python src/05_predict_today.py
    python src/05_predict_today.py --date 2026-02-22  # specific day
"""

import argparse
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_feature_matrix
from nba_data_client import NBA_API_AVAILABLE, fetch_season_wide, fetch_today_schedule
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


def compute_live_features(master: pd.DataFrame, games: list[dict], target_date: date) -> pd.DataFrame:
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


def predict_games(games: list[dict], live_feats: pd.DataFrame, model,
                  feature_columns: list, target_date: date) -> list[dict]:
    predictions = []

    for g in games:
        gid = str(g["game_id"])
        if gid not in live_feats.index:
            print(f"  [UYARI] {g['home_team_abbr']} vs {g['away_team_abbr']}: feature üretilemedi, atlanıyor.")
            continue
        row = live_feats.loc[gid]

        X_live = row.reindex(feature_columns).to_frame().T.astype(float)
        prob_home = float(model.predict_proba(X_live)[0][1])
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
        }
        print(f"  {g['home_team_abbr']} %{prob_home:.1%} | {g['away_team_abbr']} %{prob_away:.1%} [{conf}]")
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
    print(f"Model loaded. Feature count: {len(feature_columns)}\n")

    # Master tablo + güncel sezon
    master = _normalize_ids(pd.read_parquet(PROC_DIR / "master_games.parquet"))
    season = current_season(target_date)
    print(f"Güncel sezon yenileniyor ({season})...")
    master = refresh_current_season(master, season)

    # Maç programı
    games = fetch_today_schedule(target_date)
    live_path = LIVE_DIR / "live_features.parquet"
    if not games:
        print("Bugün programda maç yok.")
        pd.DataFrame().to_parquet(live_path, index=False)
        write_live_meta(target_date, 0)
        fill_results(master)
        return

    schedule_path = LIVE_DIR / "today_schedule.json"
    schedule_path.write_text(json.dumps(games, indent=2, ensure_ascii=False), encoding="utf-8")

    # Feature'lar (eğitimle aynı pipeline) + tahminler.
    # Başlamış/bitmiş maçlar da tahmin edilir: feature'lar sadece önceki
    # günlerin maçlarına dayanır, gün içinde değişmez ve sızıntı yoktur.
    # Böylece dashboard gün boyu bütün maçları gösterir.
    print("\nFeature'lar hesaplanıyor (eğitim pipeline'ı)...")
    live_feats = compute_live_features(master, games, target_date)

    print("\nTahminler hesaplanıyor...")
    predictions = predict_games(games, live_feats, model, feature_columns, target_date)
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
    fill_results(master)

    print(f"\n{'='*60}")
    print(f"Tamamlandı! Rapor: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Tahmin tarihi (YYYY-MM-DD), varsayılan: bugün (ABD Doğu saati)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    main(target_date=target)
