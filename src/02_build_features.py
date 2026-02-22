"""
02_build_features.py
--------------------
Master tablodan rolling feature'ları hesaplar ve eğitime hazır
feature matrisini üretir.

Girdi : data/processed/master_games.parquet
Çıktı :
  data/processed/team_game_log.parquet   (tall format, rolling feature'lar dahil)
  data/processed/features.parquet        (game-level, model eğitimi için)

Kullanım:
    python src/02_build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_engineering import (
    build_team_game_log,
    compute_team_rolling_features,
    compute_h2h_features,
    pivot_to_game_level,
    TEAM_FEATURE_COLS,
    ROLLING_STAT_COLS,
    ROLLING_WINDOWS,
)
from elo_srs import compute_elo, compute_opponent_quality, compute_srs

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"


def main():
    print(f"\n{'='*60}")
    print("Feature Engineering Pipeline")
    print(f"{'='*60}\n")

    # ---- 1. Master tabloyu yükle ----
    print("[1/5] master_games.parquet yükleniyor...")
    master = pd.read_parquet(PROC_DIR / "master_games.parquet")
    master["game_date"] = pd.to_datetime(master["game_date"])
    print(f"  → {len(master)} maç, {master.shape[1]} sütun")

    # ---- 2. Tall format (team game log) oluştur ----
    print("[2/5] Team game log (tall format) oluşturuluyor...")
    print("  Bu adım biraz zaman alabilir...")
    team_log = build_team_game_log(master)
    team_log_path = PROC_DIR / "team_game_log.parquet"
    team_log.to_parquet(team_log_path, index=False)
    print(f"  → {len(team_log)} satır ({team_log['team_id'].nunique()} takım)")
    print(f"  → Kaydedildi: {team_log_path}")

    # ---- 3. Rolling feature'ları hesapla ----
    print("[3/6] Rolling feature'lar hesaplanıyor (groupby team)...")
    team_log_feats = compute_team_rolling_features(team_log)
    print(f"  → Rolling hesaplandı. Sütun sayısı: {team_log_feats.shape[1]}")

    # ---- 3b. Elo rating hesapla ----
    print("  [3b] Elo rating hesaplanıyor...")
    elo_df = compute_elo(master)
    print(f"  → Elo: {len(elo_df)} maç, örnek: home_elo={elo_df['elo_home'].mean():.0f}")

    # ---- 3c. Rakip kalitesi + SRS ----
    print("  [3c] Rakip kalitesi ve SRS hesaplanıyor...")
    team_log_feats = compute_opponent_quality(team_log_feats)
    team_log_feats = compute_srs(team_log_feats)
    print(f"  → opp_quality ve srs_L10 eklendi.")

    # ---- 4. H2H feature'ları hesapla ----
    print("[4/6] Head-to-Head feature'lar hesaplanıyor (dict cache)...")
    master_with_h2h = compute_h2h_features(master)
    # Elo feature'larını master_with_h2h'a ekle (game_id üzerinden)
    master_with_h2h = master_with_h2h.merge(elo_df, on="game_id", how="left")
    print(f"  → H2H + Elo merge tamamlandı.")

    # ---- 5. Game-level feature matrix oluştur ----
    print("[5/6] Game-level feature matrix oluşturuluyor...")
    features = pivot_to_game_level(team_log_feats, master_with_h2h)

    # ---- 6. Sezon başı satırları çıkar ----
    print("[6/6] Sezon başı filtreleme...")
    # Eğitimden çıkarılacak satırlar: her takımın sezon başındaki ilk 10 maçı
    # (rolling pencereler çok gürültülü, model kalitesini düşürür)
    before_drop = len(features)
    features = features[
        (features["home_season_game_num"] >= 10) &
        (features["away_season_game_num"] >= 10)
    ].reset_index(drop=True)
    after_drop = len(features)
    print(f"  → Sezon başı satırları çıkarıldı: {before_drop} → {after_drop} ({before_drop - after_drop} satır)")

    # Son sıralama
    features = features.sort_values("game_date").reset_index(drop=True)

    feat_path = PROC_DIR / "features.parquet"
    features.to_parquet(feat_path, index=False)

    # ---- Rapor ----
    print(f"\n{'='*60}")
    print("ÖZET")
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
