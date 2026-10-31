"""
02_build_features.py
--------------------
Calculates rolling features from master table and produces
a feature matrix ready for training.

Input : data/processed/master_games.parquet
Output :
  data/processed/team_game_log.parquet   (tall format, including rolling features)
  data/processed/features.parquet        (game-level, for model training)

Usage:
    python src/02_build_features.py
"""

from pathlib import Path

import pandas as pd

from feature_engineering import build_feature_matrix

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"


def main():
    print(f"\n{'='*60}")
    print("Feature Engineering Pipeline")
    print(f"{'='*60}\n")

    # ---- 1. Master tabloyu yükle ----
    print("[1/3] loading master_games.parquet...")
    master = pd.read_parquet(PROC_DIR / "master_games.parquet")
    master["game_date"] = pd.to_datetime(master["game_date"])
    print(f"  → {len(master)} maç, {master.shape[1]} sütun")

    player_path = PROC_DIR / "player_games.parquet"
    player_games = pd.read_parquet(player_path) if player_path.exists() else None
    if player_games is None:
        print("  → Oyuncu verisi yok: oyuncu eksikliği feature'ları üretilmeyecek")
    else:
        print(f"  → Oyuncu box score: {len(player_games)} satır")

    # ---- 2. Ortak pipeline (05_predict_today.py ile aynı kod) ----
    print("[2/3] Feature matrix oluşturuluyor (rolling, Elo, SRS, H2H, oyuncu eksikliği)...")
    features, team_log_feats = build_feature_matrix(master, player_games=player_games)

    team_log_path = PROC_DIR / "team_game_log.parquet"
    team_log_feats.to_parquet(team_log_path, index=False)
    print(f"  → Team log: {len(team_log_feats)} satır → {team_log_path}")

    # ---- 3. İlk sezonu çıkar ----
    # Veri setinin ilk sezonunda rolling pencerelerin geçmişi yok.
    # Diğer sezonların başı korunur: rolling pencereler sezonlar arası
    # devam ettiği için sezon başı tahminleri de eğitimde temsil edilmeli.
    print("[3/3] İlk sezon filtreleniyor...")
    first_season = features["season_id"].astype(str).min()
    before_drop = len(features)
    features = features[features["season_id"].astype(str) != first_season].reset_index(drop=True)
    print(f"  → Sezon {first_season} çıkarıldı: {before_drop} → {len(features)}")

    # Son sıralama
    features = features.sort_values("game_date").reset_index(drop=True)

    feat_path = PROC_DIR / "features.parquet"
    features.to_parquet(feat_path, index=False)

    # ---- Rapor ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Feature matrix shape : {features.shape}")
    print(f"  Tarih aralığı        : {features['game_date'].min().date()} → {features['game_date'].max().date()}")
    print(f"  Ev sahibi win oranı  : {features['target'].mean():.3f}")
    print(f"  Kayıt                : {feat_path}")

    # Feature grupları
    feat_cols = [c for c in features.columns if c not in [
        "game_id", "game_date", "season_id", "season_type",
        "team_id_home", "team_abbreviation_home",
        "team_id_away", "team_abbreviation_away",
        "target",
    ]]
    home_feat  = [c for c in feat_cols if c.startswith("home_")]
    away_feat  = [c for c in feat_cols if c.startswith("away_")]
    diff_feat  = [c for c in feat_cols if c.startswith("diff_") or c.startswith("fatigue_")]
    h2h_feat   = [c for c in feat_cols if c.startswith("h2h_")]
    other_feat = [c for c in feat_cols if c not in home_feat + away_feat + diff_feat + h2h_feat]

    print(f"\n  Feature dağılımı:")
    print(f"    Home features  : {len(home_feat)}")
    print(f"    Away features  : {len(away_feat)}")
    print(f"    Diff features  : {len(diff_feat)}")
    print(f"    H2H features   : {len(h2h_feat)}")
    print(f"    Diğer          : {len(other_feat)}")
    print(f"    TOPLAM         : {len(feat_cols)}")
    print(f"{'='*60}\n")

    # Eksik veri oranı (model feature'ları için)
    null_pcts = (features[feat_cols].isnull().sum() / len(features) * 100).sort_values(ascending=False)
    high_null = null_pcts[null_pcts > 20]
    if not high_null.empty:
        print("Yüksek eksik veri oranı (>20%) sütunlar:")
        print(high_null.to_string())
        print()


if __name__ == "__main__":
    main()
