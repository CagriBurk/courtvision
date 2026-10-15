"""
01_build_master.py
------------------
Kaggle CSV'leri + nba_api önbelleğini birleştirerek tek bir
master_games.parquet dosyası üretir.

Çıktı: data/processed/master_games.parquet
  - Bir satır = bir Regular Season maçı
  - ~2000–bugün arası tüm sezonlar
  - home + away sütunları yan yana (wide format)
  - target: 1 = ev sahibi kazandı, 0 = deplasman kazandı

Kullanım:
    python src/01_build_master.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dizin sabitleri
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "api_cache"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Kaggle game.csv yükle ve filtrele
# ---------------------------------------------------------------------------
def load_kaggle_games() -> pd.DataFrame:
    print("[1/5] Kaggle game.csv yükleniyor...")
    game = pd.read_csv(RAW_DIR / "game.csv", parse_dates=["game_date"])

    # Tarih filtresi: 2000+ modern dönem
    game = game[game["game_date"] >= "2000-01-01"]

    # Season type: sadece Regular Season
    game = game[game["season_type"] == "Regular Season"]

    # Geçerli NBA takım ID'leri (team.csv referans)
    team = pd.read_csv(RAW_DIR / "team.csv")
    valid_ids = set(team["id"].astype(str))
    game = game[
        game["team_id_home"].astype(str).isin(valid_ids)
        & game["team_id_away"].astype(str).isin(valid_ids)
    ]

    # Hedef değişken eksikse çıkar (çok nadir)
    game = game.dropna(subset=["wl_home"])

    # Binary hedef
    game["target"] = (game["wl_home"] == "W").astype(int)

    # Sütun temizliği: sadece ihtiyaç duyduklarımız
    keep_cols = [
        "game_id", "game_date", "season_id", "season_type",
        "team_id_home", "team_abbreviation_home", "team_name_home",
        "team_id_away", "team_abbreviation_away", "team_name_away",
        # Box score - home
        "fgm_home", "fga_home", "fg_pct_home",
        "fg3m_home", "fg3a_home", "fg3_pct_home",
        "ftm_home", "fta_home", "ft_pct_home",
        "oreb_home", "dreb_home", "reb_home",
        "ast_home", "stl_home", "blk_home", "tov_home", "pf_home",
        "pts_home", "plus_minus_home",
        # Box score - away
        "fgm_away", "fga_away", "fg_pct_away",
        "fg3m_away", "fg3a_away", "fg3_pct_away",
        "ftm_away", "fta_away", "ft_pct_away",
        "oreb_away", "dreb_away", "reb_away",
        "ast_away", "stl_away", "blk_away", "tov_away", "pf_away",
        "pts_away", "plus_minus_away",
        # Hedef
        "target",
    ]
    # Sadece mevcut sütunları al (bazı sütunlar CSV versiyonuna göre değişebilir)
    keep_cols = [c for c in keep_cols if c in game.columns]
    game = game[keep_cols].copy()

    game = game.sort_values("game_date").reset_index(drop=True)
    print(f"  → {len(game)} maç ({game['game_date'].min().date()} – {game['game_date'].max().date()})")
    return game


# ---------------------------------------------------------------------------
# 2. nba_api önbelleğini yükle (api_cache/all_api_seasons.parquet)
# ---------------------------------------------------------------------------
def load_api_cache() -> pd.DataFrame:
    combined_path = CACHE_DIR / "all_api_seasons.parquet"

    if not combined_path.exists():
        # Tek tek parquet dosyalarını dene
        individual = sorted(CACHE_DIR.glob("season_*.parquet"))
        if not individual:
            print("[2/5] API önbelleği bulunamadı. Önce 06_update_data.py çalıştırın.")
            return pd.DataFrame()

        frames = [pd.read_parquet(p) for p in individual]
        api_df = pd.concat(frames, ignore_index=True)
    else:
        api_df = pd.read_parquet(combined_path)

    print(f"[2/5] API önbelleği yüklendi: {len(api_df)} maç")

    # game_date sütununun datetime olduğundan emin ol
    api_df["game_date"] = pd.to_datetime(api_df["game_date"])

    # API verisindeki sütun adlarını Kaggle formatına hizala
    col_map = {
        "team_id_home": "team_id_home",
        "team_abbreviation_home": "team_abbreviation_home",
        "team_name_home": "team_name_home",
        "team_id_away": "team_id_away",
        "team_abbreviation_away": "team_abbreviation_away",
        "team_name_away": "team_name_away",
        # API küçük harf, Kaggle de küçük harf — tutarlı
    }
    # pts_home / pts_away Kaggle'da var, API'de de var
    if "pts_home" not in api_df.columns and "pts_home" in api_df.columns:
        pass  # zaten doğru

    return api_df


# ---------------------------------------------------------------------------
# 3. İki kaynağı birleştir, çakışmaları gider
# ---------------------------------------------------------------------------
def merge_sources(kaggle_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Kaynaklar birleştiriliyor...")

    if api_df.empty:
        print("  → Sadece Kaggle verisi kullanılıyor.")
        return kaggle_df

    # API verisindeki maçlardan Kaggle'da olmayanları al
    if "game_id" in api_df.columns and "game_id" in kaggle_df.columns:
        existing_ids = set(kaggle_df["game_id"].astype(str))
        new_games = api_df[~api_df["game_id"].astype(str).isin(existing_ids)].copy()
    else:
        # game_id yoksa tarih bazlı kesi kesişim analizi
        kaggle_last = kaggle_df["game_date"].max()
        new_games = api_df[api_df["game_date"] > kaggle_last].copy()

    print(f"  → Kaggle: {len(kaggle_df)} maç")
    print(f"  → API (yeni): {len(new_games)} maç")

    # game_id'yi her iki kaynakta da string'e zorla (mixed type → pyarrow hatası)
    kaggle_df["game_id"] = kaggle_df["game_id"].astype(str)
    new_games["game_id"] = new_games["game_id"].astype(str)

    # Ortak sütunları bul, eksik olanları NaN ekle
    all_cols = list(dict.fromkeys(list(kaggle_df.columns) + list(new_games.columns)))

    for col in all_cols:
        if col not in kaggle_df.columns:
            kaggle_df[col] = np.nan
        if col not in new_games.columns:
            new_games[col] = np.nan

    combined = pd.concat([kaggle_df[all_cols], new_games[all_cols]], ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)

    print(f"  → Birleşik toplam: {len(combined)} maç")
    return combined


# ---------------------------------------------------------------------------
# 4. other_stats.csv merge
# ---------------------------------------------------------------------------
def merge_other_stats(master: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] other_stats.csv birleştiriliyor (LEFT JOIN)...")
    other = pd.read_csv(RAW_DIR / "other_stats.csv")

    keep = [
        "game_id",
        "pts_paint_home", "pts_2nd_chance_home", "pts_fb_home",
        "largest_lead_home", "lead_changes", "times_tied",
        "team_turnovers_home", "total_turnovers_home",
        "team_rebounds_home", "pts_off_to_home",
        "pts_paint_away", "pts_2nd_chance_away", "pts_fb_away",
        "largest_lead_away",
        "team_turnovers_away", "total_turnovers_away",
        "team_rebounds_away", "pts_off_to_away",
    ]
    keep = [c for c in keep if c in other.columns]
    other = other[keep].copy()
    other["game_id"] = other["game_id"].astype(str)  # tip tutarlılığı

    # Birleştirilmiş tabloda bu sütunlar zaten NaN olarak varsa kaldır
    # (API verisi için boş bırakıldılar), other_stats'tan doldur
    existing_other_cols = [c for c in keep if c != "game_id" and c in master.columns]
    master_no_other = master.drop(columns=existing_other_cols, errors="ignore")

    master_merged = master_no_other.merge(other, on="game_id", how="left")

    coverage = other["game_id"].nunique()
    total = master["game_id"].nunique()
    print(f"  → other_stats kapsama: {coverage}/{total} maç ({100*coverage/total:.1f}%)")
    return master_merged


# ---------------------------------------------------------------------------
# 5. line_score.csv merge (periyot skorları)
# ---------------------------------------------------------------------------
def merge_line_score(master: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] line_score.csv birleştiriliyor (LEFT JOIN)...")
    ls = pd.read_csv(RAW_DIR / "line_score.csv")

    keep = [
        "game_id",
        "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home",
        "pts_ot1_home",
        "pts_qtr1_away", "pts_qtr2_away", "pts_qtr3_away", "pts_qtr4_away",
        "pts_ot1_away",
    ]
    keep = [c for c in keep if c in ls.columns]
    ls = ls[keep].drop_duplicates(subset=["game_id"]).copy()
    ls["game_id"] = ls["game_id"].astype(str)  # tip tutarlılığı

    # Mevcut line_score sütunlarını düşür (API verisi boş bıraktı)
    existing_ls_cols = [c for c in keep if c != "game_id" and c in master.columns]
    master_no_ls = master.drop(columns=existing_ls_cols, errors="ignore")

    master_merged = master_no_ls.merge(ls, on="game_id", how="left")

    # OT ve çeyrek fark türetmeleri
    master_merged["went_to_ot"] = (
        master_merged.get("pts_ot1_home", pd.Series(dtype=float)).fillna(0) > 0
    ).astype(float)

    for q in ["qtr1", "qtr2", "qtr3", "qtr4"]:
        h_col = f"pts_{q}_home"
        a_col = f"pts_{q}_away"
        if h_col in master_merged.columns and a_col in master_merged.columns:
            master_merged[f"qdiff_{q}"] = (
                master_merged[h_col] - master_merged[a_col]
            )
        else:
            master_merged[f"qdiff_{q}"] = np.nan

    coverage = ls["game_id"].nunique()
    total = master["game_id"].nunique()
    print(f"  → line_score kapsama: {coverage}/{total} maç ({100*coverage/total:.1f}%)")
    return master_merged


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print("NBA Master Tablo Oluşturuluyor")
    print(f"{'='*60}\n")

    kaggle = load_kaggle_games()
    api = load_api_cache()
    master = merge_sources(kaggle, api)
    master = merge_other_stats(master)
    master = merge_line_score(master)

    # Son kontrol
    master = master.sort_values("game_date").reset_index(drop=True)
    master["game_date"] = pd.to_datetime(master["game_date"])

    # Tip tutarlılığı — pyarrow mixed-type sütunlarda patlar
    # game_id: her zaman string (Kaggle'da '0022300061', API'de int olabilir)
    master["game_id"] = master["game_id"].astype(str)
    # team_id'ler: string'e normalize et
    for col in ["team_id_home", "team_id_away"]:
        if col in master.columns:
            master[col] = master[col].astype(str)
    # season_id: string
    if "season_id" in master.columns:
        master["season_id"] = master["season_id"].astype(str)

    out_path = PROC_DIR / "master_games.parquet"
    master.to_parquet(out_path, index=False)

    print(f"\n{'='*60}")
    print("ÖZET")
    print(f"  Toplam maç     : {len(master)}")
    print(f"  Tarih aralığı  : {master['game_date'].min().date()} → {master['game_date'].max().date()}")
    print(f"  Ev sahibi W oranı: {master['target'].mean():.3f}")
    print(f"  Toplam sütun   : {master.shape[1]}")
    print(f"  Kayıt          : {out_path}")
    print(f"{'='*60}\n")

    # Basit veri kalitesi raporu
    null_pcts = (master.isnull().sum() / len(master) * 100).sort_values(ascending=False)
    null_pcts = null_pcts[null_pcts > 0]
    if not null_pcts.empty:
        print("Eksik veri oranları (> 0%):")
        print(null_pcts.to_string())
        print()


if __name__ == "__main__":
    main()
