"""
feature_engineering.py
-----------------------
Shared feature pipeline: 02_build_features.py (offline training) and
05_predict_today.py (live prediction) both go through build_feature_matrix().

All functions follow shift(1) rule: no feature uses current match
stats — only past match data.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from elo_srs import compute_elo, compute_opponent_quality, compute_srs


# ---------------------------------------------------------------------------
# 1. Team Game Log (Wide -> Tall conversion)
# ---------------------------------------------------------------------------
def build_team_game_log(master: pd.DataFrame) -> pd.DataFrame:
    """
    Splits each game into two rows: home perspective + away perspective.
    Output columns are consistent for both teams.
    """
    stat_map = {
        "pts":        ("pts_home",        "pts_away"),
        "fgm":        ("fgm_home",        "fgm_away"),
        "fga":        ("fga_home",        "fga_away"),
        "fg_pct":     ("fg_pct_home",     "fg_pct_away"),
        "fg3m":       ("fg3m_home",       "fg3m_away"),
        "fg3a":       ("fg3a_home",       "fg3a_away"),
        "fg3_pct":    ("fg3_pct_home",    "fg3_pct_away"),
        "ftm":        ("ftm_home",        "ftm_away"),
        "fta":        ("fta_home",        "fta_away"),
        "ft_pct":     ("ft_pct_home",     "ft_pct_away"),
        "oreb":       ("oreb_home",       "oreb_away"),
        "dreb":       ("dreb_home",       "dreb_away"),
        "reb":        ("reb_home",        "reb_away"),
        "ast":        ("ast_home",        "ast_away"),
        "stl":        ("stl_home",        "stl_away"),
        "blk":        ("blk_home",        "blk_away"),
        "tov":        ("tov_home",        "tov_away"),
        "pf":         ("pf_home",         "pf_away"),
        "plus_minus": ("plus_minus_home", "plus_minus_away"),
        # other_stats
        "pts_paint":      ("pts_paint_home",      "pts_paint_away"),
        "pts_2nd_chance": ("pts_2nd_chance_home", "pts_2nd_chance_away"),
        "pts_fb":         ("pts_fb_home",         "pts_fb_away"),
        "largest_lead":   ("largest_lead_home",   "largest_lead_away"),
        "total_turnovers":("total_turnovers_home","total_turnovers_away"),
        "team_rebounds":  ("team_rebounds_home",  "team_rebounds_away"),
        "pts_off_to":     ("pts_off_to_home",     "pts_off_to_away"),
    }

    id_cols = ["game_id", "game_date", "season_id", "season_type"]
    rows_home = []
    rows_away = []

    for _, row in master.iterrows():
        base = {c: row[c] for c in id_cols if c in row.index}
        # Upcoming games (live prediction) have no result yet
        target = row.get("target")
        home_won = int(target) if pd.notna(target) else np.nan

        # --- Ev sahibi satırı ---
        h = base.copy()
        h["team_id"]      = row.get("team_id_home")
        h["team_abbr"]    = row.get("team_abbreviation_home")
        h["opponent_id"]  = row.get("team_id_away")
        h["is_home"]      = 1
        h["won"]          = home_won
        h["pts_scored"]   = row.get("pts_home")
        h["pts_allowed"]  = row.get("pts_away")
        for col, (hcol, _) in stat_map.items():
            h[col] = row.get(hcol)
        # Periyot farklar: home perspektifinden pozitif = home önde
        h["qdiff_qtr1"]  = row.get("qdiff_qtr1")
        h["qdiff_qtr2"]  = row.get("qdiff_qtr2")
        h["qdiff_qtr3"]  = row.get("qdiff_qtr3")
        h["qdiff_qtr4"]  = row.get("qdiff_qtr4")
        h["went_to_ot"]  = row.get("went_to_ot")
        # lead_changes / times_tied maç geneli (iki taraf için aynı)
        h["lead_changes"] = row.get("lead_changes")
        h["times_tied"]   = row.get("times_tied")
        rows_home.append(h)

        # --- Deplasman satırı ---
        a = base.copy()
        a["team_id"]      = row.get("team_id_away")
        a["team_abbr"]    = row.get("team_abbreviation_away")
        a["opponent_id"]  = row.get("team_id_home")
        a["is_home"]      = 0
        a["won"]          = 1 - home_won
        a["pts_scored"]   = row.get("pts_away")
        a["pts_allowed"]  = row.get("pts_home")
        for col, (_, acol) in stat_map.items():
            a[col] = row.get(acol)
        # Deplasman için fark işaretleri ters
        a["qdiff_qtr1"]  = -row.get("qdiff_qtr1", 0) if pd.notna(row.get("qdiff_qtr1")) else np.nan
        a["qdiff_qtr2"]  = -row.get("qdiff_qtr2", 0) if pd.notna(row.get("qdiff_qtr2")) else np.nan
        a["qdiff_qtr3"]  = -row.get("qdiff_qtr3", 0) if pd.notna(row.get("qdiff_qtr3")) else np.nan
        a["qdiff_qtr4"]  = -row.get("qdiff_qtr4", 0) if pd.notna(row.get("qdiff_qtr4")) else np.nan
        a["went_to_ot"]  = row.get("went_to_ot")
        a["lead_changes"] = row.get("lead_changes")
        a["times_tied"]   = row.get("times_tied")
        rows_away.append(a)

    log = pd.DataFrame(rows_home + rows_away)
    log = log.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    log["game_date"] = pd.to_datetime(log["game_date"])
    return log


# ---------------------------------------------------------------------------
# 2. Per-team rolling feature hesaplama
# ---------------------------------------------------------------------------

# shift(1) ile hesaplanacak istatistik sütunları
ROLLING_STAT_COLS = [
    "won", "pts_scored", "pts_allowed", "plus_minus",
    "fg_pct", "fg3_pct", "ft_pct",
    "reb", "ast", "stl", "blk", "tov", "pf", "oreb", "dreb",
    "pts_paint", "pts_2nd_chance", "pts_fb",
    "largest_lead", "lead_changes", "times_tied",
    "total_turnovers", "team_rebounds", "pts_off_to",
    "went_to_ot",
    "qdiff_qtr1", "qdiff_qtr2", "qdiff_qtr3", "qdiff_qtr4",
]

ROLLING_WINDOWS = [5, 10]


def _streak(won_series: pd.Series) -> list:
    """Kazanma/kaybetme serisi hesaplar. Pozitif = galibiyet, negatif = mağlubiyet."""
    streaks = []
    current = 0
    for w in won_series.shift(1).fillna(0):
        if w == 1:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        streaks.append(current)
    return streaks


def compute_team_rolling_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """
    Her takım için rolling feature'ları hesaplar.
    KURALLAR:
      - shift(1): her feature sadece önceki maçların verilerini kullanır
      - min_periods=1: sezon başında bile feature üretilir
      - groupby('season_id'): cumulative stats sezon başında sıfırlanır
    """
    team_log = team_log.sort_values(["team_id", "game_date", "game_id"]).copy()
    result_frames = []

    for team_id, grp in team_log.groupby("team_id", sort=False):
        grp = grp.sort_values("game_date").copy()

        # ---- Yorgunluk / Takvim ----
        grp["days_since_last_game"] = (
            grp["game_date"].diff().dt.days.fillna(7.0)
        )
        grp["is_back_to_back"] = (grp["days_since_last_game"] == 1).astype(int)

        # Son 5 ve 7 takvim günündeki maç sayısı (mevcut maç hariç)
        # dates is sorted → count of past dates in [d - N days, d)
        dates = grp["game_date"].values
        left = np.searchsorted(dates, dates, side="left")
        for n in (5, 7):
            start = np.searchsorted(dates, dates - np.timedelta64(n, "D"), side="left")
            grp[f"games_in_last_{n}_days"] = left - start

        grp["prev_game_was_home"] = grp["is_home"].shift(1).fillna(0.5)

        # ---- Rolling istatistikler ----
        # Built in one go to avoid fragmenting the frame with ~60 inserts
        rolling_cols = {}
        for col in ROLLING_STAT_COLS:
            if col not in grp.columns:
                continue
            shifted = pd.to_numeric(grp[col], errors="coerce").shift(1)
            for w in ROLLING_WINDOWS:
                rolling_cols[f"{col}_L{w}"] = shifted.rolling(w, min_periods=1).mean()
        grp = pd.concat([grp, pd.DataFrame(rolling_cols, index=grp.index)], axis=1)

        # Türetilen rolling metrikler
        pace_raw = grp["pts_scored"] + grp["pts_allowed"]
        grp["pace_L5"]  = pace_raw.shift(1).rolling(5,  min_periods=1).mean()
        grp["pace_L10"] = pace_raw.shift(1).rolling(10, min_periods=1).mean()

        grp["point_diff_L5"]  = grp["plus_minus"].shift(1).rolling(5,  min_periods=1).mean()
        grp["point_diff_L10"] = grp["plus_minus"].shift(1).rolling(10, min_periods=1).mean()

        # ---- Ev/Deplasman özel win rate ----
        home_won = grp["won"].where(grp["is_home"] == 1)
        away_won = grp["won"].where(grp["is_home"] == 0)

        grp["home_win_rate_L10"] = (
            home_won.shift(1).rolling(10, min_periods=1).mean()
        )
        grp["away_win_rate_L10"] = (
            away_won.shift(1).rolling(10, min_periods=1).mean()
        )
        grp["home_win_rate_L10"] = grp["home_win_rate_L10"].ffill().fillna(0.5)
        grp["away_win_rate_L10"] = grp["away_win_rate_L10"].ffill().fillna(0.5)

        # ---- Sezon-to-date (sezon sınırında sıfırla) ----
        grp["season_game_num"] = grp.groupby("season_id").cumcount()  # 0-indexed

        season_wins_cumsum  = grp.groupby("season_id")["won"].cumsum()
        grp["season_wins_before"] = season_wins_cumsum.shift(1).fillna(0)
        grp["season_win_pct"] = (
            grp["season_wins_before"] / grp["season_game_num"].clip(lower=1)
        )

        season_pd_cumsum = grp.groupby("season_id")["plus_minus"].cumsum()
        grp["season_pd_before"] = season_pd_cumsum.shift(1).fillna(0)

        # ---- Seri (streak) ----
        grp["current_streak"] = _streak(grp["won"])

        # ---- L3 window (ultra-recent form) ----
        for col in ["won", "pts_scored", "pts_allowed", "plus_minus", "fg_pct", "fg3_pct"]:
            if col in grp.columns:
                grp[f"{col}_L3"] = grp[col].shift(1).rolling(3, min_periods=1).mean()

        # ---- Form trend: L3 - L10 (pozitif = form yükseliyor) ----
        for col in ["won", "plus_minus", "pts_scored", "pts_allowed"]:
            l3_col, l10_col = f"{col}_L3", f"{col}_L10"
            if l3_col in grp.columns and l10_col in grp.columns:
                grp[f"{col}_trend"] = grp[l3_col] - grp[l10_col]

        # ---- True Shooting % (pts / (2*(fga + 0.44*fta))) ----
        if "fga" in grp.columns and "fta" in grp.columns:
            pts_s = grp["pts_scored"].shift(1)
            fga_s = grp["fga"].shift(1)
            fta_s = grp["fta"].shift(1)
            denom = 2 * (fga_s + 0.44 * fta_s)
            ts_raw = pts_s / denom.where(denom > 0, np.nan)
            grp["ts_pct_L10"] = ts_raw.rolling(10, min_periods=1).mean()
            grp["ts_pct_L5"]  = ts_raw.rolling(5,  min_periods=1).mean()
        else:
            grp["ts_pct_L10"] = np.nan
            grp["ts_pct_L5"]  = np.nan

        # ---- Four Factors (Offensive and Defensive) ----
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        # ORB% = OREB / (OREB + Opp DREB) (Simplified to OREB / (OREB + DREB) for rolling as approx)
        # FTr = FTA / FGA
        if all(c in grp.columns for c in ["fgm", "fg3m", "fga", "tov", "fta", "oreb", "dreb"]):
            fgm_s = grp["fgm"].shift(1)
            fg3m_s = grp["fg3m"].shift(1)
            fga_s = grp["fga"].shift(1)
            tov_s = grp["tov"].shift(1)
            fta_s = grp["fta"].shift(1)
            oreb_s = grp["oreb"].shift(1)
            dreb_s = grp["dreb"].shift(1)

            efg_raw = (fgm_s + 0.5 * fg3m_s) / fga_s.where(fga_s > 0, np.nan)
            tov_pct_raw = tov_s / (fga_s + 0.44 * fta_s + tov_s).where((fga_s + 0.44 * fta_s + tov_s) > 0, np.nan)
            # Simplified ORB% approximation since we don't have opponent DREB perfectly aligned in this view easily
            # We use OREB / (OREB + DREB) as a proxy for offensive rebounding effort relative to total own rebounds
            orb_pct_raw = oreb_s / (oreb_s + dreb_s).where((oreb_s + dreb_s) > 0, np.nan)
            ftr_raw = fta_s / fga_s.where(fga_s > 0, np.nan)

            for w in [5, 10]:
                grp[f"efg_pct_L{w}"] = efg_raw.rolling(w, min_periods=1).mean()
                grp[f"tov_pct_L{w}"] = tov_pct_raw.rolling(w, min_periods=1).mean()
                grp[f"orb_pct_L{w}"] = orb_pct_raw.rolling(w, min_periods=1).mean()
                grp[f"ftr_L{w}"] = ftr_raw.rolling(w, min_periods=1).mean()
        else:
            for w in [5, 10]:
                grp[f"efg_pct_L{w}"] = np.nan
                grp[f"tov_pct_L{w}"] = np.nan
                grp[f"orb_pct_L{w}"] = np.nan
                grp[f"ftr_L{w}"] = np.nan

        # ---- Tutarlılık (std) ----
        for col in ["won", "pts_scored", "plus_minus"]:
            if col in grp.columns:
                grp[f"{col}_std_L10"] = (
                    grp[col].shift(1).rolling(10, min_periods=3).std().fillna(0)
                )

        # ---- Hücum etkinliği (skor oranı) ----
        if "pts_allowed" in grp.columns:
            total = grp["pts_scored"] + grp["pts_allowed"]
            off_raw = grp["pts_scored"] / total.where(total > 0, np.nan)
            grp["off_eff_L10"] = off_raw.shift(1).rolling(10, min_periods=1).mean()
            grp["off_eff_L5"]  = off_raw.shift(1).rolling(5,  min_periods=1).mean()

        # ---- Sezon fazı ----
        grp["season_pct_complete"] = grp["season_game_num"] / 82.0

        result_frames.append(grp)

    return pd.concat(result_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Head-to-Head özellikleri (sözlük önbellekleme ile O(n·k))
# ---------------------------------------------------------------------------
def compute_h2h_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Her maç için, aynı iki takım arasındaki önceki maçların istatistiklerini hesaplar.
    Kritik: pair_history ÖNCE okunur, maç kaydedildikten SONRA sözlüğe eklenir.
    Bu strict < date filter'ın sözlük versiyonudur.
    """
    master = master.sort_values("game_date").reset_index(drop=True)

    # Sözlük: frozenset({team_a_id, team_b_id}) → [{"home_won": ..., "pts_diff": ...}, ...]
    pair_history: dict = defaultdict(list)
    h2h_records = []

    for _, row in master.iterrows():
        h_tid = str(row["team_id_home"])
        a_tid = str(row["team_id_away"])
        key   = frozenset({h_tid, a_tid})

        hist = pair_history[key]
        n    = len(hist)

        if n == 0:
            h2h_records.append({
                "game_id":              row["game_id"],
                "h2h_n_games":          0,
                "h2h_home_win_rate":    0.5,
                "h2h_home_win_rate_L5": 0.5,
                "h2h_pts_diff_mean":    0.0,
                "h2h_pts_diff_L5":      0.0,
            })
        else:
            # Normalize every past game to the CURRENT home team's perspective
            wins = [
                g["home_won"] if g["home_team_id"] == h_tid else 1 - g["home_won"]
                for g in hist
            ]
            pts_diffs = [
                g["pts_diff"] if g["home_team_id"] == h_tid else -g["pts_diff"]
                for g in hist
            ]

            h2h_records.append({
                "game_id":              row["game_id"],
                "h2h_n_games":          n,
                "h2h_home_win_rate":    np.mean(wins),
                "h2h_home_win_rate_L5": np.mean(wins[-5:]),
                "h2h_pts_diff_mean":    np.mean(pts_diffs),
                "h2h_pts_diff_L5":      np.mean(pts_diffs[-5:]),
            })

        # Bu maçı sözlüğe ekle — sonraki maçlar için geçmiş olacak.
        # Kayıt bu maçın ev sahibi perspektifinden tutulur; okurken normalize edilir.
        if pd.isna(row.get("target")):
            continue  # upcoming game, no result yet
        home_won_this = int(row["target"])
        pts_diff_this = (
            (row.get("pts_home", 0) or 0) - (row.get("pts_away", 0) or 0)
        )
        pair_history[key].append({
            "home_won": home_won_this,
            "pts_diff": pts_diff_this,
            "home_team_id": h_tid,
        })

    h2h_df = pd.DataFrame(h2h_records)
    return master.merge(h2h_df, on="game_id", how="left")


# ---------------------------------------------------------------------------
# 4. Game-level feature matrix (Tall → Wide geri dönüşümü)
# ---------------------------------------------------------------------------

# Feature sütunları (home_/away_ prefix ile game seviyesine çevrilecek)
TEAM_FEATURE_COLS = (
    ["days_since_last_game", "is_back_to_back",
     "games_in_last_5_days", "games_in_last_7_days",
     "prev_game_was_home", "current_streak",
     "season_game_num", "season_win_pct", "season_pd_before",
     "season_pct_complete",
     "home_win_rate_L10", "away_win_rate_L10",
     "pace_L5", "pace_L10", "point_diff_L5", "point_diff_L10"]
    + [f"{col}_L{w}" for col in ROLLING_STAT_COLS for w in ROLLING_WINDOWS]
    # Yeni: L3 window
    + [f"{col}_L3" for col in ["won", "pts_scored", "pts_allowed", "plus_minus", "fg_pct", "fg3_pct"]]
    # Yeni: Form trend (L3 - L10)
    + [f"{col}_trend" for col in ["won", "plus_minus", "pts_scored", "pts_allowed"]]
    # Yeni: Verimlilik metrikleri
    + ["ts_pct_L10", "ts_pct_L5", "off_eff_L10", "off_eff_L5"]
    # Four Factors
    + [f"{metric}_L{w}" for metric in ["efg_pct", "tov_pct", "orb_pct", "ftr"] for w in [5, 10]]
    # Yeni: Tutarlılık (std)
    + ["won_std_L10", "pts_scored_std_L10", "plus_minus_std_L10"]
    # Elo/SRS/Rakip kalitesi (elo_srs.py'den gelir)
    + ["srs_L10", "srs_L5", "opp_quality_L10", "opp_quality_L5", "opp_pdiff_L10"]
    # Oyuncu eksikliği (compute_availability_features, oyuncu verisi varsa)
    + ["missing_min_share", "missing_value_share", "n_missing_rotation"]
)


def pivot_to_game_level(
    team_log_feats: pd.DataFrame,
    master_with_h2h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tall team log'u home/away prefix ile game seviyesine döndürür,
    H2H feature'larını ve diff feature'larını ekler.
    """
    # Mevcut feature sütunlarını filtrele
    available_feat_cols = [c for c in TEAM_FEATURE_COLS if c in team_log_feats.columns]

    home = team_log_feats[team_log_feats["is_home"] == 1][
        ["game_id"] + available_feat_cols
    ].copy()
    away = team_log_feats[team_log_feats["is_home"] == 0][
        ["game_id"] + available_feat_cols
    ].copy()

    home.columns = ["game_id"] + ["home_" + c for c in available_feat_cols]
    away.columns = ["game_id"] + ["away_" + c for c in available_feat_cols]

    game_feats = home.merge(away, on="game_id", how="inner")

    # H2H + Elo feature'larını ekle
    h2h_cols = [
        "game_id", "target",
        "h2h_n_games", "h2h_home_win_rate", "h2h_home_win_rate_L5",
        "h2h_pts_diff_mean", "h2h_pts_diff_L5",
        # Elo (elo_srs.py tarafından master_with_h2h'a merge edildi)
        "elo_home", "elo_away", "elo_diff", "elo_win_prob",
        "game_date", "season_id", "season_type",
        "team_id_home", "team_abbreviation_home",
        "team_id_away", "team_abbreviation_away",
    ]
    h2h_cols = [c for c in h2h_cols if c in master_with_h2h.columns]
    meta = master_with_h2h[h2h_cols]
    game_feats = game_feats.merge(meta, on="game_id", how="left")

    # Playoff flag
    game_feats["is_playoff"] = (game_feats.get("season_type", "") == "Playoffs").astype(int)

    # ---- Differential features ----
    diff_pairs = [
        ("won_L5",            "diff_win_rate_L5"),
        ("won_L10",           "diff_win_rate_L10"),
        ("pts_scored_L5",     "diff_pts_scored_L5"),
        ("pts_scored_L10",    "diff_pts_scored_L10"),
        ("pts_allowed_L5",    "diff_pts_allowed_L5"),
        ("pts_allowed_L10",   "diff_pts_allowed_L10"),
        ("plus_minus_L5",     "diff_plus_minus_L5"),
        ("plus_minus_L10",    "diff_plus_minus_L10"),
        ("fg_pct_L10",        "diff_fg_pct_L10"),
        ("fg3_pct_L10",       "diff_fg3_pct_L10"),
        ("reb_L5",            "diff_reb_L5"),
        ("ast_L5",            "diff_ast_L5"),
        ("tov_L5",            "diff_tov_L5"),
        ("pts_fb_L5",         "diff_pts_fb_L5"),
        ("pts_paint_L5",      "diff_pts_paint_L5"),
        ("season_win_pct",    "diff_season_win_pct"),
        ("season_pd_before",  "diff_season_pd"),
        ("current_streak",    "diff_streak"),
        ("is_back_to_back",   "fatigue_diff_b2b"),
        ("days_since_last_game", "diff_rest_days"),
        # Yeni diff feature'lar
        ("won_L3",           "diff_win_rate_L3"),
        ("plus_minus_L3",    "diff_plus_minus_L3"),
        ("pts_scored_L3",    "diff_pts_scored_L3"),
        ("pts_allowed_L3",   "diff_pts_allowed_L3"),
        ("won_trend",        "diff_won_trend"),
        ("plus_minus_trend", "diff_pm_trend"),
        ("pts_scored_trend", "diff_pts_trend"),
        ("ts_pct_L10",       "diff_ts_pct_L10"),
        ("off_eff_L10",      "diff_off_eff_L10"),
        ("won_std_L10",      "diff_won_std_L10"),
        # Four Factors
        ("efg_pct_L5",       "diff_efg_pct_L5"),
        ("efg_pct_L10",      "diff_efg_pct_L10"),
        ("tov_pct_L5",       "diff_tov_pct_L5"),
        ("tov_pct_L10",      "diff_tov_pct_L10"),
        ("orb_pct_L5",       "diff_orb_pct_L5"),
        ("orb_pct_L10",      "diff_orb_pct_L10"),
        ("ftr_L5",           "diff_ftr_L5"),
        ("ftr_L10",          "diff_ftr_L10"),
        # Elo/SRS/opp_quality diff'leri
        ("srs_L10",          "diff_srs_L10"),
        ("srs_L5",           "diff_srs_L5"),
        ("opp_quality_L10",  "diff_opp_quality_L10"),
        ("opp_quality_L5",   "diff_opp_quality_L5"),
        # Oyuncu eksikliği
        ("missing_min_share",   "diff_missing_min_share"),
        ("missing_value_share", "diff_missing_value_share"),
        ("n_missing_rotation",  "diff_n_missing_rotation"),
    ]

    for col, diff_name in diff_pairs:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in game_feats.columns and a_col in game_feats.columns:
            game_feats[diff_name] = game_feats[h_col] - game_feats[a_col]

    game_feats = game_feats.sort_values("game_date").reset_index(drop=True)
    return game_feats


# ---------------------------------------------------------------------------
# 5. Oyuncu eksikliği: bu maçta kim yok ve takım için ne kadar önemli?
# ---------------------------------------------------------------------------
AVAILABILITY_WINDOW = 10        # son N takım maçı → oyuncunun beklenen dakikası/katkısı
AVAILABILITY_MIN_GAMES = 2      # pencerede en az bu kadar maç oynamış olmalı
ROTATION_MINUTES = 20.0        # rotasyon oyuncusu eşiği (beklenen dakika)
TEAM_MINUTES = 240.0            # 5 oyuncu × 48 dakika

_BOX_COLS = ["min", "pts", "fgm", "fga", "ftm", "fta", "oreb", "dreb",
             "ast", "stl", "blk", "tov", "pf"]


def _game_score(pg: pd.DataFrame) -> pd.Series:
    """Hollinger Game Score: tek maçlık katkıyı tek sayıya indirir."""
    return (
        pg["pts"] + 0.4 * pg["fgm"] - 0.7 * pg["fga"] - 0.4 * (pg["fta"] - pg["ftm"])
        + 0.7 * pg["oreb"] + 0.3 * pg["dreb"] + pg["stl"] + 0.7 * pg["ast"]
        + 0.7 * pg["blk"] - 0.4 * pg["pf"] - pg["tov"]
    )


def compute_availability_features(
    team_games: pd.DataFrame,
    player_games: pd.DataFrame,
    absent_override: dict | None = None,
) -> pd.DataFrame:
    """
    For each team in each game: how much of the team's expected minutes and
    production is missing.

    Pool    = players with minutes in at least AVAILABILITY_MIN_GAMES of the
              team's previous AVAILABILITY_WINDOW games; expected minutes/value
              = their average over the games they played (games they missed
              would otherwise dilute a star returning from injury, or one whose
              window reaches into last season's final weeks).
    Absent  = training: on the roster (their next appearance, at or after this
              date, is for this team — so traded or departed players don't
              count) but no minutes in this game.
              live: names from the injury report, given as
              absent_override[(game_id, team_id)] = {player_id, ...}; the
              roster is then "last appearance was for this team".

    Rows without a box score and without an override get NaN.

    Returns game_id, team_id, missing_min_share, missing_value_share,
    n_missing_rotation.
    """
    absent_override = absent_override or {}
    cols = ["missing_min_share", "missing_value_share", "n_missing_rotation"]

    tg_all = team_games[["game_id", "game_date", "team_id"]].drop_duplicates(["game_id", "team_id"]).copy()
    tg_all["team_id"] = tg_all["team_id"].astype(str)
    tg_all["game_id"] = tg_all["game_id"].astype(str)
    tg_all["game_date"] = pd.to_datetime(tg_all["game_date"])

    pg = player_games.copy()
    for c in ["player_id", "team_id", "game_id"]:
        pg[c] = pg[c].astype(str)
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    pg = pg[pg["game_date"] >= tg_all["game_date"].min()]
    for c in _BOX_COLS:
        pg[c] = pd.to_numeric(pg[c], errors="coerce").fillna(0.0)
    pg = pg[pg["min"] > 0]
    pg["value"] = _game_score(pg)

    # Every player's appearances (any team), for roster membership
    timeline = {
        pid: (grp["game_date"].values, grp["team_id"].values)
        for pid, grp in pg.sort_values("game_date").groupby("player_id", sort=False)
    }

    frames = []
    for team_id, tg in tg_all.groupby("team_id", sort=False):
        tg = tg.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        n = len(tg)
        gidx = {gid: i for i, gid in enumerate(tg["game_id"])}
        tp = pg[(pg["team_id"] == team_id) & pg["game_id"].isin(gidx)]
        live_cols = {gidx[gid]: ids for (gid, tid), ids in absent_override.items()
                     if str(tid) == team_id and gid in gidx}

        result = np.full((n, len(cols)), np.nan)
        if tp.empty:
            frames.append(pd.DataFrame(result, columns=cols).assign(game_id=tg["game_id"], team_id=team_id))
            continue

        players = tp["player_id"].unique()
        pidx = {p: j for j, p in enumerate(players)}
        rows = tp["player_id"].map(pidx).values
        gcols = tp["game_id"].map(gidx).values
        minutes = np.zeros((len(players), n))
        value = np.zeros((len(players), n))
        minutes[rows, gcols] = tp["min"].values
        value[rows, gcols] = tp["value"].values
        has_box = np.zeros(n, dtype=bool)
        has_box[gcols] = True

        # Window averages over the previous AVAILABILITY_WINDOW team games
        idx = np.arange(n)
        lo = np.maximum(idx - AVAILABILITY_WINDOW, 0)
        win = (idx - lo).astype(float)
        cum_m = np.concatenate([np.zeros((len(players), 1)), np.cumsum(minutes, axis=1)], axis=1)
        cum_v = np.concatenate([np.zeros((len(players), 1)), np.cumsum(value, axis=1)], axis=1)
        cum_n = np.concatenate([np.zeros((len(players), 1)), np.cumsum(minutes > 0, axis=1)], axis=1)
        n_played = cum_n[:, idx] - cum_n[:, lo]
        pool = n_played >= AVAILABILITY_MIN_GAMES
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_min = np.where(pool, (cum_m[:, idx] - cum_m[:, lo]) / n_played, 0.0)
            exp_val = np.where(pool, (cum_v[:, idx] - cum_v[:, lo]) / n_played, 0.0)

        # Roster membership
        dates = tg["game_date"].values
        member_next = np.zeros((len(players), n), dtype=bool)
        member_last = np.zeros((len(players), n), dtype=bool)
        for p, j in pidx.items():
            p_dates, p_teams = timeline[p]
            k = np.searchsorted(p_dates, dates, side="left")
            member_next[j] = (k < len(p_dates)) & (p_teams[np.minimum(k, len(p_dates) - 1)] == team_id)
            member_last[j] = (k > 0) & (p_teams[np.maximum(k - 1, 0)] == team_id)

        member = member_next.copy()
        absent = pool & member_next & (minutes == 0)
        for i, ids in live_cols.items():
            member[:, i] = member_last[:, i]
            absent[:, i] = pool[:, i] & member_last[:, i] & np.isin(players, list(ids))

        missing_min = (absent * exp_min).sum(axis=0)
        missing_val = (absent * exp_val).sum(axis=0)
        team_val = ((pool & member) * exp_val).sum(axis=0)
        n_rot = (absent & (exp_min >= ROTATION_MINUTES)).sum(axis=0)

        result[:, 0] = missing_min / TEAM_MINUTES
        with np.errstate(invalid="ignore", divide="ignore"):
            result[:, 1] = np.where(team_val > 0, missing_val / team_val, np.nan)
        result[:, 2] = n_rot

        valid = (has_box | np.isin(idx, list(live_cols))) & (win > 0)
        result[~valid] = np.nan
        frames.append(pd.DataFrame(result, columns=cols).assign(game_id=tg["game_id"], team_id=team_id))

    return pd.concat(frames, ignore_index=True)[["game_id", "team_id"] + cols]


# ---------------------------------------------------------------------------
# 6. Full pipeline: master table → game-level feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(
    master: pd.DataFrame,
    rolling_since: pd.Timestamp | None = None,
    player_games: pd.DataFrame | None = None,
    absent_override: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single entry point used by BOTH training (02_build_features.py) and
    live prediction (05_predict_today.py), so the two can never drift apart.

    Live prediction appends upcoming games to `master` with target=NaN and
    no box score; shift(1) guarantees their features only use past games.

    rolling_since: compute team rolling features only on games from this
        date on (speed-up for live prediction). Must reach back at least one
        full season so L10 windows and season-to-date stats are unaffected.
        Elo and H2H always use the full history.

    player_games / absent_override: player box scores and, for upcoming
        games, injured players (see compute_availability_features). Without
        player_games the availability features are simply not produced.

    Returns (game-level features, team log with rolling features).
    """
    master = master.copy()
    master["game_date"] = pd.to_datetime(master["game_date"])

    recent = master if rolling_since is None else master[master["game_date"] >= rolling_since]
    team_log = build_team_game_log(recent)
    team_log_feats = compute_team_rolling_features(team_log)
    team_log_feats = compute_opponent_quality(team_log_feats)
    team_log_feats = compute_srs(team_log_feats)

    if player_games is not None and not player_games.empty:
        avail = compute_availability_features(
            team_log_feats[["game_id", "game_date", "team_id"]], player_games, absent_override
        )
        team_log_feats["team_id"] = team_log_feats["team_id"].astype(str)
        team_log_feats["game_id"] = team_log_feats["game_id"].astype(str)
        team_log_feats = team_log_feats.merge(avail, on=["game_id", "team_id"], how="left")

    elo_df = compute_elo(master)
    master_with_h2h = compute_h2h_features(master).merge(elo_df, on="game_id", how="left")

    features = pivot_to_game_level(team_log_feats, master_with_h2h)
    return features, team_log_feats
