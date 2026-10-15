"""
elo_srs.py
----------
Elo rating ve SRS (Simple Rating System) hesaplama modülü.
02_build_features.py ve 05_predict_today.py tarafından import edilir.

Fonksiyonlar:
  compute_elo(master)              → game_id bazlı pre-game Elo değerleri
  compute_opponent_quality(tlog)   → rakip kalitesi rolling feature'ları
  compute_srs(tlog)                → rolling SRS tahmini
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Elo sabitleri
# ---------------------------------------------------------------------------
ELO_INIT       = 1500.0   # yeni takım başlangıç Elo'su
ELO_K          = 20.0     # güncelleme hızı (NBA için standart)
ELO_REGRESSION = 0.75     # sezon başı: yeni_elo = 0.75*eski + 0.25*1500


# ---------------------------------------------------------------------------
# 1. Elo Rating
# ---------------------------------------------------------------------------
def compute_elo(master: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm tarihsel maçlardan pre-game Elo hesaplar.

    Parametreler:
        master: game_id, game_date, season_id, team_id_home, team_id_away, target içeren DataFrame

    Döner:
        game_id, elo_home, elo_away, elo_diff, elo_win_prob sütunları olan DataFrame
        (her değer o maça GİRMEDEN önceki Elo'yu yansıtır)
    """
    master = master.sort_values("game_date").reset_index(drop=True)

    elo_table: dict[str, float] = {}  # team_id → güncel Elo
    last_season: dict[str, str] = {}  # team_id → son görülen season_id

    records = []

    for _, row in master.iterrows():
        h_id = str(row["team_id_home"])
        a_id = str(row["team_id_away"])
        season = str(row.get("season_id", ""))
        target = row.get("target", np.nan)

        # Yeni takımı başlat
        if h_id not in elo_table:
            elo_table[h_id] = ELO_INIT
            last_season[h_id] = season
        if a_id not in elo_table:
            elo_table[a_id] = ELO_INIT
            last_season[a_id] = season

        # Sezon değişimi → mean'e regresyon
        for tid in (h_id, a_id):
            if last_season[tid] != season:
                elo_table[tid] = (
                    ELO_REGRESSION * elo_table[tid]
                    + (1 - ELO_REGRESSION) * ELO_INIT
                )
                last_season[tid] = season

        h_elo = elo_table[h_id]
        a_elo = elo_table[a_id]

        # Beklenen ev sahibi galibiyet olasılığı (Elo formülü)
        e_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))

        records.append({
            "game_id":      str(row["game_id"]),
            "elo_home":     round(h_elo, 2),
            "elo_away":     round(a_elo, 2),
            "elo_diff":     round(h_elo - a_elo, 2),
            "elo_win_prob": round(e_home, 4),
        })

        # Elo'yu güncelle (sonuç biliniyorsa)
        if pd.notna(target):
            actual = float(target)
            elo_table[h_id] += ELO_K * (actual - e_home)
            elo_table[a_id] += ELO_K * ((1 - actual) - (1 - e_home))

    return pd.DataFrame(records)


def get_current_elo(master: pd.DataFrame) -> dict[str, float]:
    """
    Tüm maçlardan sonraki (güncel) Elo değerlerini döner.
    Live tahmin için: {team_id: elo}
    """
    master = master.sort_values("game_date").reset_index(drop=True)

    elo_table: dict[str, float] = {}
    last_season: dict[str, str] = {}

    for _, row in master.iterrows():
        h_id = str(row["team_id_home"])
        a_id = str(row["team_id_away"])
        season = str(row.get("season_id", ""))
        target = row.get("target", np.nan)

        for tid in (h_id, a_id):
            if tid not in elo_table:
                elo_table[tid] = ELO_INIT
                last_season[tid] = season
            if last_season[tid] != season:
                elo_table[tid] = (
                    ELO_REGRESSION * elo_table[tid]
                    + (1 - ELO_REGRESSION) * ELO_INIT
                )
                last_season[tid] = season

        h_elo = elo_table[h_id]
        a_elo = elo_table[a_id]
        e_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))

        if pd.notna(target):
            actual = float(target)
            elo_table[h_id] += ELO_K * (actual - e_home)
            elo_table[a_id] += ELO_K * ((1 - actual) - (1 - e_home))

    return elo_table


# ---------------------------------------------------------------------------
# 2. Rakip Kalitesi Rolling Features
# ---------------------------------------------------------------------------
def compute_opponent_quality(team_log_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Her takımın son N maçındaki rakiplerin kalitesini (season_win_pct ve
    point_diff_L10) hesaplar.

    Bu feature "güçlü takıma karşı kazanan" ile "zayıfa karşı kazanan"ı ayırt eder.

    Girdi: compute_team_rolling_features() çıktısı (team_id, opponent_id,
           game_date, season_win_pct, point_diff_L10 içermeli)
    """
    tlog = team_log_feats.sort_values(["team_id", "game_date"]).copy()

    # Lookup: (team_id, game_date) → season_win_pct, point_diff_L10
    # Aynı tarihte birden fazla maç olabilir → game_id'den bağımsız en son değeri al
    lookup_wr  = {}   # (team_id, game_date) → season_win_pct
    lookup_pdiff = {} # (team_id, game_date) → point_diff_L10

    for _, row in tlog.iterrows():
        key = (str(row["team_id"]), str(row["game_date"]))
        lookup_wr[key]    = row.get("season_win_pct", 0.5)
        lookup_pdiff[key] = row.get("point_diff_L10", 0.0)

    # Her satır için rakip kalitesini bul
    opp_wrs   = []
    opp_pdiffs = []

    for _, row in tlog.iterrows():
        opp_key = (str(row["opponent_id"]), str(row["game_date"]))
        opp_wrs.append(lookup_wr.get(opp_key, 0.5))
        opp_pdiffs.append(lookup_pdiff.get(opp_key, 0.0))

    tlog["opp_win_pct_now"]   = opp_wrs
    tlog["opp_pdiff_now"]     = opp_pdiffs

    # Rakip kalitesi rolling ortalaması (shift(1) uygulanmış)
    result_frames = []
    for team_id, grp in tlog.groupby("team_id", sort=False):
        grp = grp.sort_values("game_date").copy()

        grp["opp_quality_L10"] = (
            grp["opp_win_pct_now"].shift(1).rolling(10, min_periods=1).mean()
        )
        grp["opp_quality_L5"] = (
            grp["opp_win_pct_now"].shift(1).rolling(5, min_periods=1).mean()
        )
        grp["opp_pdiff_L10"] = (
            grp["opp_pdiff_now"].shift(1).rolling(10, min_periods=1).mean()
        )
        result_frames.append(grp)

    return pd.concat(result_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Rolling SRS (Simple Rating System)
# ---------------------------------------------------------------------------
def compute_srs(team_log_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling SRS tahmini:
        srs_L10 = kendi point_diff_L10 + rakibin point_diff_L10

    Statik (sezon-bütünü) SRS yerine rolling pencere kullanılır,
    çünkü takım gücü sezon içinde değişir.

    Girdi: compute_opponent_quality() çıktısı (opp_pdiff_L10 içermeli)
    """
    tlog = team_log_feats.copy()

    own_pdiff  = tlog.get("point_diff_L10", pd.Series(0.0, index=tlog.index))
    opp_pdiff  = tlog.get("opp_pdiff_L10",  pd.Series(0.0, index=tlog.index))

    tlog["srs_L10"] = own_pdiff.fillna(0) + opp_pdiff.fillna(0)
    tlog["srs_L5"]  = (
        tlog.get("point_diff_L5", own_pdiff).fillna(0)
        + tlog.get("opp_pdiff_L10", opp_pdiff).fillna(0)
    )

    return tlog
