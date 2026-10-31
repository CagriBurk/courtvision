"""
06_update_data.py
-----------------
Fetches NBA seasons with the nba_api LeagueGameLog endpoint,
converts them to wide format (home+away on same row, game.csv format)
and saves each season as parquet under data/api_cache/.

Completed seasons are cached; the current season is always re-fetched.

Usage:
    python src/06_update_data.py --since 2000            # 2000-01 → current season
    python src/06_update_data.py --seasons 2024-25 2025-26
    python src/06_update_data.py                         # current season only
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nba_data_client import NBA_API_AVAILABLE, fetch_season_wide
from season import current_season, season_start_year, season_str

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def seasons_since(start_year: int) -> list[str]:
    """Season strings from start_year up to the current season."""
    return [season_str(y) for y in range(start_year, season_start_year() + 1)]


def season_cache_path(season: str) -> Path:
    return CACHE_DIR / f"season_{season}.parquet"


# ---------------------------------------------------------------------------
# Önbellekten yükle veya çekip kaydet
# ---------------------------------------------------------------------------
def get_season(season: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = season_cache_path(season)

    # The current season is still being played — never trust its cache
    if cache_path.exists() and not force_refresh and season != current_season():
        print(f"  [ÖNBELLEKTEN] {season}: {cache_path.name}")
        return pd.read_parquet(cache_path)

    wide = fetch_season_wide(season)

    if wide.empty:
        print(f"  [UYARI] {season} için veri gelmedi, atlanıyor.")
        return pd.DataFrame()

    wide.to_parquet(cache_path, index=False)
    print(f"  [KAYDEDİLDİ] {season}: {len(wide)} maç → {cache_path.name}")
    return wide


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main(seasons: list[str], force_refresh: bool = False):
    if not NBA_API_AVAILABLE:
        raise SystemExit("nba_api yüklü değil. Önce: pip install nba_api")

    print(f"\n{'='*60}")
    print(f"NBA Sezon Veri Güncelleme")
    print(f"Hedef sezonlar: {seasons}")
    print(f"Kayıt dizini: {CACHE_DIR}")
    print(f"{'='*60}\n")

    all_frames = []
    for s in tqdm(seasons, desc="Sezonlar", unit="sezon"):
        df = get_season(s, force_refresh=force_refresh)
        if not df.empty:
            all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"  Toplam maç: {len(combined)}")
        print(f"  Tarih aralığı: {combined['game_date'].min().date()} → {combined['game_date'].max().date()}")
        print(f"{'='*60}\n")
    else:
        print("\n[UYARI] Hiçbir sezon için veri çekilemedi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA sezon verilerini nba_api'den çeker.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[current_season()],
        help="Çekilecek sezonlar (örn: 2023-24 2024-25 2025-26)",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=None,
        help="Bu yıldan güncel sezona kadar tüm sezonlar (örn: 2000 → 2000-01 ... ); --seasons'ı ezer",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Önbellekte olsa bile yeniden çek",
    )
    args = parser.parse_args()
    seasons = seasons_since(args.since) if args.since else args.seasons
    main(seasons=seasons, force_refresh=args.force)
