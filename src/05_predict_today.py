"""
05_predict_today.py
-------------------
Bugünün NBA maçlarını nba_api'den otomatik çeker,
rolling feature'ları hesaplar ve tahmin üretir.

Çıktı:
  data/live/today_schedule.json
  data/live/live_features.parquet
  reports/predictions_YYYY-MM-DD.html

Kullanım:
    python src/05_predict_today.py
    python src/05_predict_today.py --date 2026-02-22  # belirli bir gün
"""

import argparse
import json
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nba_api.stats.endpoints import scoreboardv3, teamgamelog
    from nba_api.stats.static import teams as nba_teams_static
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

from feature_engineering import compute_live_team_features
from elo_srs import get_current_elo
from prediction_tracker import save_predictions, fill_results

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
MODELS_DIR  = ROOT / "models"
PROC_DIR    = ROOT / "data" / "processed"
LIVE_DIR    = ROOT / "data" / "live"
REPORTS_DIR = ROOT / "reports"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_SEC       = 0.7
CURRENT_SEASON  = "2025-26"
MIN_GAMES_NEEDED = 5  # daha az maç varsa önceki sezonu ekle


# ---------------------------------------------------------------------------
# 1. Bugünkü maç programını çek
# ---------------------------------------------------------------------------
def fetch_today_schedule(target_date: date) -> list[dict]:
    date_str = target_date.strftime("%m/%d/%Y")
    print(f"  Maç programı çekiliyor: {date_str}...")

    sb = scoreboardv3.ScoreboardV3(game_date=date_str, league_id="00")
    time.sleep(SLEEP_SEC)

    dfs     = sb.get_data_frames()
    game_df = dfs[1]   # gameId, gameStatus, gameCode, gameStatusText ...
    team_df = dfs[2]   # gameId, teamId, teamTricode ... (home+away aynı tabloda)

    # gameStatus: 1=başlamamış, 2=devam ediyor, 3=bitti
    upcoming = game_df[game_df["gameStatus"] == 1]

    games = []
    seen_ids = set()
    for _, row in upcoming.iterrows():
        gid = str(row["gameId"])
        if gid in seen_ids:
            continue
        seen_ids.add(gid)

        # gameCode: '20260222/CLEOKC' → away=CLE, home=OKC (her zaman 3+3 karakter)
        try:
            code_part    = row["gameCode"].split("/")[1]
            away_tricode = code_part[:3]
            home_tricode = code_part[3:]
        except Exception:
            continue

        game_teams = team_df[team_df["gameId"] == row["gameId"]]
        home_row   = game_teams[game_teams["teamTricode"] == home_tricode]
        away_row   = game_teams[game_teams["teamTricode"] == away_tricode]
        if home_row.empty or away_row.empty:
            continue

        games.append({
            "game_id":      gid,
            "home_team_id": int(home_row.iloc[0]["teamId"]),
            "away_team_id": int(away_row.iloc[0]["teamId"]),
            "status":       str(row.get("gameStatusText", "")),
        })

    # Takım adlarını ekle
    all_teams = {t["id"]: t for t in nba_teams_static.get_teams()}
    for g in games:
        h = all_teams.get(g["home_team_id"], {})
        a = all_teams.get(g["away_team_id"], {})
        g["home_team_name"] = h.get("full_name", str(g["home_team_id"]))
        g["home_team_abbr"] = h.get("abbreviation", "???")
        g["away_team_name"] = a.get("full_name", str(g["away_team_id"]))
        g["away_team_abbr"] = a.get("abbreviation", "???")

    print(f"  → {len(games)} maç bulundu")
    return games


# ---------------------------------------------------------------------------
# 2. Takımın son maçlarını çek (mevcut sezon + gerekirse önceki sezon)
# ---------------------------------------------------------------------------
def fetch_team_history(team_id: int) -> pd.DataFrame:
    """
    Önce mevcut sezonu çek. Yeterli maç yoksa önceki sezondan
    offline team_game_log.parquet'ten tamamla.
    """
    # Mevcut sezon
    try:
        gl = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
            timeout=30,
        )
        time.sleep(SLEEP_SEC)
        current = gl.get_data_frames()[0]
    except Exception as e:
        print(f"    [UYARI] {team_id} için mevcut sezon çekilemedi: {e}")
        current = pd.DataFrame()

    if len(current) >= 20:
        return current

    # Yetersizse önceki sezondan offline veriden tamamla
    log_path = PROC_DIR / "team_game_log.parquet"
    if log_path.exists():
        offline = pd.read_parquet(log_path)
        offline = offline[offline["team_id"] == str(team_id)].copy()
        if not offline.empty:
            # Offline'dan son 20 maç → nba_api formatına çevir
            offline = offline.sort_values("game_date").tail(20)
            # Basit dönüşüm: offline sütunları → API sütun adları
            bridge = pd.DataFrame()
            bridge["GAME_DATE"]   = offline["game_date"].astype(str)
            bridge["WL"]          = offline["won"].map({1: "W", 0: "L"})
            bridge["PTS"]         = offline["pts_scored"]
            bridge["FGM"]         = offline.get("fgm", np.nan)
            bridge["FGA"]         = offline.get("fga", np.nan)
            bridge["FG_PCT"]      = offline.get("fg_pct", np.nan)
            bridge["FG3M"]        = offline.get("fg3m", np.nan)
            bridge["FG3A"]        = offline.get("fg3a", np.nan)
            bridge["FG3_PCT"]     = offline.get("fg3_pct", np.nan)
            bridge["FTM"]         = offline.get("ftm", np.nan)
            bridge["FTA"]         = offline.get("fta", np.nan)
            bridge["FT_PCT"]      = offline.get("ft_pct", np.nan)
            bridge["OREB"]        = offline.get("oreb", np.nan)
            bridge["DREB"]        = offline.get("dreb", np.nan)
            bridge["REB"]         = offline.get("reb", np.nan)
            bridge["AST"]         = offline.get("ast", np.nan)
            bridge["STL"]         = offline.get("stl", np.nan)
            bridge["BLK"]         = offline.get("blk", np.nan)
            bridge["TOV"]         = offline.get("tov", np.nan)
            bridge["PF"]          = offline.get("pf", np.nan)
            bridge["PLUS_MINUS"]  = offline.get("plus_minus", np.nan)
            bridge["MATCHUP"]     = offline["is_home"].map({1: "HOME vs.", 0: "@ AWAY"})

            combined = pd.concat([bridge, current], ignore_index=True)
            return combined

    return current


# ---------------------------------------------------------------------------
# 3. H2H feature'ları (offline master tablodan)
# ---------------------------------------------------------------------------
def compute_h2h_live(home_team_id: int, away_team_id: int) -> dict:
    master_path = PROC_DIR / "master_games.parquet"
    if not master_path.exists():
        return {"h2h_n_games": 0, "h2h_home_win_rate": 0.5,
                "h2h_home_win_rate_L5": 0.5, "h2h_pts_diff_mean": 0.0, "h2h_pts_diff_L5": 0.0}

    master = pd.read_parquet(master_path)
    h_tid  = str(home_team_id)
    a_tid  = str(away_team_id)

    prior = master[
        ((master["team_id_home"].astype(str) == h_tid) & (master["team_id_away"].astype(str) == a_tid)) |
        ((master["team_id_home"].astype(str) == a_tid) & (master["team_id_away"].astype(str) == h_tid))
    ].copy()

    if prior.empty:
        return {"h2h_n_games": 0, "h2h_home_win_rate": 0.5,
                "h2h_home_win_rate_L5": 0.5, "h2h_pts_diff_mean": 0.0, "h2h_pts_diff_L5": 0.0}

    wins, diffs = [], []
    for _, row in prior.iterrows():
        if str(row["team_id_home"]) == h_tid:
            wins.append(int(row.get("target", 0)))
            diffs.append((row.get("pts_home", 0) or 0) - (row.get("pts_away", 0) or 0))
        else:
            wins.append(1 - int(row.get("target", 0)))
            diffs.append((row.get("pts_away", 0) or 0) - (row.get("pts_home", 0) or 0))

    return {
        "h2h_n_games":          len(wins),
        "h2h_home_win_rate":    float(np.mean(wins)),
        "h2h_home_win_rate_L5": float(np.mean(wins[-5:])),
        "h2h_pts_diff_mean":    float(np.mean(diffs)),
        "h2h_pts_diff_L5":      float(np.mean(diffs[-5:])),
    }


# ---------------------------------------------------------------------------
# 4. Feature vektörü oluştur ve tahmin yap
# ---------------------------------------------------------------------------
def predict_games(games: list[dict], model, feature_columns: list, current_elo: dict) -> list[dict]:
    predictions = []

    for g in games:
        h_id = g["home_team_id"]
        a_id = g["away_team_id"]
        print(f"  {g['home_team_abbr']} vs {g['away_team_abbr']} ...", end=" ")

        h_log = fetch_team_history(h_id)
        a_log = fetch_team_history(a_id)

        today_ts = pd.Timestamp.today().normalize()
        home_feats = compute_live_team_features(h_log, today_ts) if not h_log.empty else {}
        away_feats = compute_live_team_features(a_log, today_ts) if not a_log.empty else {}

        h2h_feats = compute_h2h_live(h_id, a_id)

        # Game-level feature dict
        game_feats = {}
        for k, v in home_feats.items():
            game_feats[f"home_{k}"] = v
        for k, v in away_feats.items():
            game_feats[f"away_{k}"] = v
        game_feats.update(h2h_feats)
        game_feats["is_playoff"] = 0

        # Elo feature'ları — get_current_elo(master)'dan gelen güncel değerler
        h_elo = current_elo.get(str(h_id), 1500.0)
        a_elo = current_elo.get(str(a_id), 1500.0)
        e_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        game_feats["elo_home"]     = round(h_elo, 2)
        game_feats["elo_away"]     = round(a_elo, 2)
        game_feats["elo_diff"]     = round(h_elo - a_elo, 2)
        game_feats["elo_win_prob"] = round(e_home, 4)

        # Diff feature'lar
        diff_pairs = [
            ("home_won_L5",          "away_won_L5",          "diff_win_rate_L5"),
            ("home_won_L10",         "away_won_L10",         "diff_win_rate_L10"),
            ("home_pts_scored_L5",   "away_pts_scored_L5",   "diff_pts_scored_L5"),
            ("home_pts_scored_L10",  "away_pts_scored_L10",  "diff_pts_scored_L10"),
            ("home_pts_allowed_L5",  "away_pts_allowed_L5",  "diff_pts_allowed_L5"),
            ("home_pts_allowed_L10", "away_pts_allowed_L10", "diff_pts_allowed_L10"),
            ("home_plus_minus_L5",   "away_plus_minus_L5",   "diff_plus_minus_L5"),
            ("home_plus_minus_L10",  "away_plus_minus_L10",  "diff_plus_minus_L10"),
            ("home_fg_pct_L10",      "away_fg_pct_L10",      "diff_fg_pct_L10"),
            ("home_fg3_pct_L10",     "away_fg3_pct_L10",     "diff_fg3_pct_L10"),
            ("home_reb_L5",          "away_reb_L5",           "diff_reb_L5"),
            ("home_ast_L5",          "away_ast_L5",           "diff_ast_L5"),
            ("home_tov_L5",          "away_tov_L5",           "diff_tov_L5"),
            ("home_pts_fb_L5",       "away_pts_fb_L5",        "diff_pts_fb_L5"),
            ("home_pts_paint_L5",    "away_pts_paint_L5",     "diff_pts_paint_L5"),
            ("home_season_win_pct",  "away_season_win_pct",   "diff_season_win_pct"),
            ("home_season_pd_before","away_season_pd_before", "diff_season_pd"),
            ("home_current_streak",  "away_current_streak",   "diff_streak"),
            ("home_is_back_to_back", "away_is_back_to_back",  "fatigue_diff_b2b"),
            ("home_days_since_last_game","away_days_since_last_game","diff_rest_days"),
            # SRS / opp_quality — compute_live_team_features'ta yoksa NaN olur,
            # XGBoost native NaN handling ile tolere eder
            ("home_srs_L10",         "away_srs_L10",         "diff_srs_L10"),
            ("home_srs_L5",          "away_srs_L5",          "diff_srs_L5"),
            ("home_opp_quality_L10", "away_opp_quality_L10", "diff_opp_quality_L10"),
            ("home_opp_quality_L5",  "away_opp_quality_L5",  "diff_opp_quality_L5"),
        ]
        for hk, ak, dk in diff_pairs:
            game_feats[dk] = game_feats.get(hk, 0) - game_feats.get(ak, 0)

        # Feature vektörünü eğitim sırasına hizala
        X_live = pd.DataFrame([game_feats])
        for col in feature_columns:
            if col not in X_live.columns:
                X_live[col] = np.nan
        X_live = X_live[feature_columns]

        prob_home = float(model.predict_proba(X_live)[0][1])
        prob_away = 1.0 - prob_home
        winner    = "HOME" if prob_home > 0.5 else "AWAY"
        conf      = "YÜKSEK" if max(prob_home, prob_away) >= 0.65 else \
                    "ORTA"   if max(prob_home, prob_away) >= 0.55 else "DÜŞÜK"

        result = {
            **g,
            "prob_home":       prob_home,
            "prob_away":       prob_away,
            "predicted_winner": winner,
            "confidence":      conf,
            # Form özeti
            "home_form_L5":    home_feats.get("won_L5", None),
            "home_form_L10":   home_feats.get("won_L10", None),
            "home_b2b":        home_feats.get("is_back_to_back", 0),
            "home_rest":       home_feats.get("days_since_last_game", None),
            "home_streak":     home_feats.get("current_streak", 0),
            "away_form_L5":    away_feats.get("won_L5", None),
            "away_form_L10":   away_feats.get("won_L10", None),
            "away_b2b":        away_feats.get("is_back_to_back", 0),
            "away_rest":       away_feats.get("days_since_last_game", None),
            "away_streak":     away_feats.get("current_streak", 0),
            "h2h_games":       h2h_feats.get("h2h_n_games", 0),
            "h2h_home_wr":     h2h_feats.get("h2h_home_win_rate", 0.5),
            "h2h_pts_diff":    h2h_feats.get("h2h_pts_diff_mean", 0),
            # Key diff features — UI faktörleri + ileride analiz için
            "elo_diff":              game_feats.get("elo_diff", None),
            "elo_home":              game_feats.get("elo_home", None),
            "elo_away":              game_feats.get("elo_away", None),
            "diff_plus_minus_L10":   game_feats.get("diff_plus_minus_L10", None),
            "diff_plus_minus_L5":    game_feats.get("diff_plus_minus_L5", None),
            "diff_season_win_pct":   game_feats.get("diff_season_win_pct", None),
            "diff_season_pd":        game_feats.get("diff_season_pd", None),
            "diff_srs_L10":          game_feats.get("diff_srs_L10", None),
            "fatigue_diff_b2b":      game_feats.get("fatigue_diff_b2b", None),
        }
        print(f"{g['home_team_abbr']} %{prob_home:.1%} | {g['away_team_abbr']} %{prob_away:.1%} [{conf}]")
        predictions.append(result)

    return predictions


# ---------------------------------------------------------------------------
# 5. HTML raporu üret
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
      Son 5 maç: {home_form_pct} | Seri: {_streak_str(p['home_streak'])} | Dinlenme: {p['home_rest'] or '?'} gün
    </div>
    <div class="detail-col">
      <b>{p['away_team_name']}{away_b2b_badge}</b><br>
      Son 5 maç: {away_form_pct} | Seri: {_streak_str(p['away_streak'])} | Dinlenme: {p['away_rest'] or '?'} gün
    </div>
  </div>
  <div class="h2h">
    H2H ({p['h2h_games']} maç): Ev sahibi win rate %{p['h2h_home_wr']:.1%} |
    Ort. skor farkı: {p['h2h_pts_diff']:+.1f}
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


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main(target_date: date | None = None):
    if not NBA_API_AVAILABLE:
        raise SystemExit("nba_api yüklü değil: pip install nba_api")

    if target_date is None:
        target_date = date.today()

    print(f"\n{'='*60}")
    print(f"NBA Günlük Tahmin — {target_date}")
    print(f"{'='*60}\n")

    # Model yükle
    with open(MODELS_DIR / "xgb_best.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    model.set_params(device="cpu")  # inference CPU'da, CUDA device mismatch uyarısını kapat
    print(f"Model yüklendi. Feature sayısı: {len(feature_columns)}\n")

    # Güncel Elo değerlerini hesapla
    print("Elo rating hesaplanıyor (tüm tarihsel maçlardan)...")
    master = pd.read_parquet(PROC_DIR / "master_games.parquet")
    master["game_date"] = pd.to_datetime(master["game_date"])
    current_elo = get_current_elo(master)
    print(f"  → {len(current_elo)} takım için Elo hazır\n")

    # Maç programı
    games = fetch_today_schedule(target_date)
    if not games:
        print("Bugün maç yok veya tüm maçlar tamamlandı.")
        return

    schedule_path = LIVE_DIR / "today_schedule.json"
    schedule_path.write_text(json.dumps(games, indent=2, ensure_ascii=False), encoding="utf-8")

    # Tahminler
    print("\nTahminler hesaplanıyor...")
    predictions = predict_games(games, model, feature_columns, current_elo)

    # Live features kaydet
    live_df = pd.DataFrame(predictions)
    live_df.to_parquet(LIVE_DIR / "live_features.parquet", index=False)

    # HTML raporu
    report_html = generate_report(predictions, target_date)
    report_path = REPORTS_DIR / f"predictions_{target_date.strftime('%Y-%m-%d')}.html"
    report_path.write_text(report_html, encoding="utf-8")

    # Tahminleri kaydet
    for p in predictions:
        p["date"] = str(target_date)
    save_predictions(predictions)

    # Önceki günlerin sonuçlarını doldur (master tablodan)
    fill_results(master)

    print(f"\n{'='*60}")
    print(f"Tamamlandı! Rapor: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Tahmin tarihi (YYYY-MM-DD), varsayılan: bugün")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    main(target_date=target)
