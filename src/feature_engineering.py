"""
feature_engineering.py
-----------------------
Hem 02_build_features.py (offline eğitim) hem de api_server.py (live tahmin)
tarafından import edilen paylaşımlı modül.

Tüm fonksiyonlar shift(1) kuralına uyar: hiçbir feature mevcut maçın
istatistiklerini içermez — sadece geçmiş maçların verileri.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Team Game Log (Wide → Tall dönüşümü)
# ---------------------------------------------------------------------------
def build_team_game_log(master: pd.DataFrame) -> pd.DataFrame:
    """
    Her maçı iki satıra ayırır: ev sahibi perspektifi + deplasman perspektifi.
    Çıktı sütunları her iki takım için de tutarlıdır.
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

        # --- Ev sahibi satırı ---
        h = base.copy()
        h["team_id"]      = row.get("team_id_home")
        h["team_abbr"]    = row.get("team_abbreviation_home")
        h["opponent_id"]  = row.get("team_id_away")
        h["is_home"]      = 1
        h["won"]          = int(row.get("target", 0))
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
        a["won"]          = 1 - int(row.get("target", 0))
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
        dates = grp["game_date"].values
        g5, g7 = [], []
        for d in dates:
            d_ts = pd.Timestamp(d)
            past = grp["game_date"]
            g5.append(int(((past < d_ts) & (past >= d_ts - pd.Timedelta(days=5))).sum()))
            g7.append(int(((past < d_ts) & (past >= d_ts - pd.Timedelta(days=7))).sum()))
        grp["games_in_last_5_days"] = g5
        grp["games_in_last_7_days"] = g7

        grp["prev_game_was_home"] = grp["is_home"].shift(1).fillna(0.5)

        # ---- Rolling istatistikler ----
        for col in ROLLING_STAT_COLS:
            if col not in grp.columns:
                continue
            shifted = grp[col].shift(1)
            for w in ROLLING_WINDOWS:
                grp[f"{col}_L{w}"] = shifted.rolling(w, min_periods=1).mean()

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
            wins      = [g["home_won"]   for g in hist]
            pts_diffs = [g["pts_diff"]   for g in hist]

            h2h_records.append({
                "game_id":              row["game_id"],
                "h2h_n_games":          n,
                "h2h_home_win_rate":    np.mean(wins),
                "h2h_home_win_rate_L5": np.mean(wins[-5:]),
                "h2h_pts_diff_mean":    np.mean(pts_diffs),
                "h2h_pts_diff_L5":      np.mean(pts_diffs[-5:]),
            })

        # Bu maçı sözlüğe ekle — sonraki maçlar için geçmiş olacak
        # Perspektifi her zaman MEVCUT EV SAHİBİ takımı açısından normalize et
        home_won_this = int(row.get("target", 0))
        pts_diff_this = (
            (row.get("pts_home", 0) or 0) - (row.get("pts_away", 0) or 0)
        )
        # Eğer bu maçta h_tid deplasman ise işareti çevir
        # (pair_history perspektif her zaman alfabetik küçük team_id = "home" olarak normalize)
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
    # Yeni: Tutarlılık (std)
    + ["won_std_L10", "pts_scored_std_L10", "plus_minus_std_L10"]
    # Elo/SRS/Rakip kalitesi (elo_srs.py'den gelir)
    + ["srs_L10", "srs_L5", "opp_quality_L10", "opp_quality_L5", "opp_pdiff_L10"]
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
        # Elo/SRS/opp_quality diff'leri
        ("srs_L10",          "diff_srs_L10"),
        ("srs_L5",           "diff_srs_L5"),
        ("opp_quality_L10",  "diff_opp_quality_L10"),
        ("opp_quality_L5",   "diff_opp_quality_L5"),
    ]

    for col, diff_name in diff_pairs:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in game_feats.columns and a_col in game_feats.columns:
            game_feats[diff_name] = game_feats[h_col] - game_feats[a_col]

    game_feats = game_feats.sort_values("game_date").reset_index(drop=True)
    return game_feats


# ---------------------------------------------------------------------------
# 5. Live tahmin için tek takımın rolling feature'larını hesapla
# ---------------------------------------------------------------------------
def compute_live_team_features(
    team_game_log: pd.DataFrame,
    today: pd.Timestamp | None = None,
) -> dict:
    """
    nba_api TeamGameLog çıktısından (kronolojik sırada) bir sonraki maç
    için feature dict üretir.

    team_game_log: GAME_DATE, WL, PTS, FGM, FGA, FG_PCT, FG3M, FG3A,
                   FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST,
                   STL, BLK, TOV, PF, PLUS_MINUS, MATCHUP içeren DF.
    today: tahmin tarihi (varsayılan: bugün).
    """
    if today is None:
        today = pd.Timestamp.today().normalize()

    df = team_game_log.copy()
    df["game_date"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df = df.sort_values("game_date").reset_index(drop=True)

    if df.empty:
        return {}

    def _col(api_col, default=np.nan):
        """API sütunu yoksa NaN serisi döndür."""
        # Büyük/küçük harf varyantlarını dene
        for c in [api_col, api_col.lower(), api_col.upper()]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(default, index=df.index, dtype=float)

    df["is_home"]     = df["MATCHUP"].apply(lambda m: 0 if "@" in m else 1)
    df["won"]         = (df["WL"] == "W").astype(int)
    df["pts_scored"]  = _col("PTS")
    df["plus_minus"]  = _col("PLUS_MINUS")
    df["fg_pct"]      = _col("FG_PCT")
    df["fg3_pct"]     = _col("FG3_PCT")
    df["ft_pct"]      = _col("FT_PCT")
    df["fga"]         = _col("FGA")
    df["fta"]         = _col("FTA")
    df["reb"]         = _col("REB")
    df["ast"]         = _col("AST")
    df["stl"]         = _col("STL")
    df["blk"]         = _col("BLK")
    df["tov"]         = _col("TOV")
    df["pf"]          = _col("PF")

    feats = {}
    n = len(df)
    last_date = df["game_date"].iloc[-1]

    # Yorgunluk
    feats["days_since_last_game"] = (today - last_date).days
    feats["is_back_to_back"]      = int(feats["days_since_last_game"] == 1)
    feats["games_in_last_5_days"] = int(
        ((df["game_date"] > today - pd.Timedelta(days=5)) & (df["game_date"] < today)).sum()
    )
    feats["games_in_last_7_days"] = int(
        ((df["game_date"] > today - pd.Timedelta(days=7)) & (df["game_date"] < today)).sum()
    )
    feats["prev_game_was_home"] = int(df["is_home"].iloc[-1])

    # Rolling (tüm geçmiş = "önceki maçlar", mevcut maç yok)
    for col in ["won", "pts_scored", "plus_minus", "fg_pct", "fg3_pct",
                "ft_pct", "reb", "ast", "stl", "blk", "tov", "pf"]:
        for w in [5, 10]:
            tail = df[col].tail(w)
            feats[f"{col}_L{w}"] = float(tail.mean()) if len(tail) > 0 else 0.5

    feats["point_diff_L5"]  = float(df["plus_minus"].tail(5).mean())
    feats["point_diff_L10"] = float(df["plus_minus"].tail(10).mean())
    feats["pace_L5"]        = float((df["pts_scored"] * 2).tail(5).mean())  # yaklaşık pace
    feats["pace_L10"]       = float((df["pts_scored"] * 2).tail(10).mean())

    # Ev/deplasman win rate
    home_games = df[df["is_home"] == 1]["won"].tail(10)
    away_games = df[df["is_home"] == 0]["won"].tail(10)
    feats["home_win_rate_L10"] = float(home_games.mean()) if len(home_games) > 0 else 0.5
    feats["away_win_rate_L10"] = float(away_games.mean()) if len(away_games) > 0 else 0.5

    # Sezon-to-date
    feats["season_game_num"]  = n
    feats["season_win_pct"]   = float(df["won"].mean())
    feats["season_pd_before"] = float(df["plus_minus"].sum())

    # Streak
    current = 0
    for w in reversed(df["won"].tolist()):
        if current == 0:
            current = 1 if w == 1 else -1
        elif current > 0 and w == 1:
            current += 1
        elif current < 0 and w == 0:
            current -= 1
        else:
            break
    feats["current_streak"] = current

    # ---- Sezon fazı ----
    feats["season_pct_complete"] = n / 82.0

    # ---- L3 window ----
    for col in ["won", "pts_scored", "plus_minus", "fg_pct", "fg3_pct"]:
        if col in df.columns:
            feats[f"{col}_L3"] = float(df[col].tail(3).mean())

    # ---- Form trend (L3 - L10) ----
    for col in ["won", "plus_minus", "pts_scored", "pts_allowed"]:
        l3  = feats.get(f"{col}_L3",  feats.get(f"{col}_L5",  0.0))
        l10 = feats.get(f"{col}_L10", feats.get(f"{col}_L5",  0.0))
        feats[f"{col}_trend"] = l3 - l10

    # ---- True Shooting % ----
    pts_arr = df["pts_scored"].tail(10)
    fga_arr = df["fga"].tail(10)
    fta_arr = df["fta"].tail(10)
    denom   = 2 * (fga_arr + 0.44 * fta_arr)
    ts_vals = pts_arr.values / np.where(denom.values > 0, denom.values, np.nan)
    feats["ts_pct_L10"] = float(np.nanmean(ts_vals)) if len(ts_vals) > 0 else 0.55
    feats["ts_pct_L5"]  = float(np.nanmean(ts_vals[-5:])) if len(ts_vals) >= 1 else 0.55

    # ---- Tutarlılık (std) ----
    for col in ["won", "pts_scored", "plus_minus"]:
        if col in df.columns:
            vals = df[col].tail(10)
            feats[f"{col}_std_L10"] = float(vals.std()) if len(vals) >= 3 else 0.0

    # ---- Hücum etkinliği ----
    if "pts_allowed" in df.columns:
        pts_s   = df["pts_scored"].tail(10)
        pts_a   = df["pts_allowed"].tail(10)
        total   = pts_s + pts_a
        off_raw = pts_s / total.where(total > 0, np.nan)
        feats["off_eff_L10"] = float(off_raw.mean())
        feats["off_eff_L5"]  = float(off_raw.tail(5).mean())
    else:
        feats["off_eff_L10"] = 0.5
        feats["off_eff_L5"]  = 0.5

    return feats
