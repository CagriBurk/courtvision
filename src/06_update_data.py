"""
06_update_data.py
-----------------
Kaggle dataset ends ~June 2023. This script;
  - Fetches missing seasons using nba_api LeagueGameLog endpoint
  - Converts to Wide format (home+away on same row, game.csv format)
  - Saves as parquet under data/api_cache/

Usage:
    python src/06_update_data.py
    python src/06_update_data.py --seasons 2023-24 2024-25 2025-26
    python src/06_update_data.py --seasons 2025-26  # update only current season
"""

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from nba_api.stats.endpoints import leaguegamelog
    from nba_api.stats.static import teams as nba_teams_static
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("[WARNING] nba_api not installed. First: pip install nba_api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default missing seasons (Kaggle dataset comes up to ~2022-23 end)
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]

# nba_api rate limit: ~600ms wait is enough
SLEEP_SEC = 0.7


# ---------------------------------------------------------------------------
# Helper: season strings ("2000-01", ...) up to the current season
# ---------------------------------------------------------------------------
def seasons_since(start_year: int) -> list[str]:
    today = date.today()
    # NBA regular season starts in October
    current_start = today.year if today.month >= 10 else today.year - 1
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, current_start + 1)]


# ---------------------------------------------------------------------------
# Helper: Valid NBA team IDs (30 teams)
# ---------------------------------------------------------------------------
def get_valid_team_ids() -> set:
    teams = nba_teams_static.get_teams()
    return {str(t["id"]) for t in teams}


# ---------------------------------------------------------------------------
# Step 1: Fetch LeagueGameLog
# ---------------------------------------------------------------------------
def fetch_season_raw(season_str: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetches specified season from nba_api.
    Returned DataFrame is in tall format (2 rows per game: home + away).

    Columns: SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID,
              GAME_DATE, MATCHUP, WL, PTS, FGM, FGA, FG_PCT, FG3M, FG3A,
              FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK,
              TOV, PF, PLUS_MINUS
    """
    print(f"  Fetching: {season_str} ({season_type})...", end=" ", flush=True)
    try:
        gl = leaguegamelog.LeagueGameLog(
            season=season_str,
            season_type_all_star=season_type,
            league_id="00",
            timeout=60,
        )
        time.sleep(SLEEP_SEC)
        df = gl.get_data_frames()[0]
        print(f"{len(df)} rows")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 2: Tall -> Wide conversion (similar to game.csv format)
# ---------------------------------------------------------------------------
def convert_to_wide(df: pd.DataFrame, valid_team_ids: set) -> pd.DataFrame:
    """
    Converts LeagueGameLog tall format to wide:
    Her maç için 1 rows, home ve away sütunları yan yana.

    Output columns are lowercased to match game.csv.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].apply(lambda m: 1 if "vs." in m else 0)

    # Filter to valid NBA teams (exclude All-Star, international games, etc.)
    df = df[df["TEAM_ID"].astype(str).isin(valid_team_ids)]

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    # Sütun adlarını eşleştir
    stat_cols = [
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "WL", "PTS", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
    ]

    home_renamed = home[["GAME_ID", "SEASON_ID", "GAME_DATE"] + stat_cols].copy()
    home_renamed.columns = (
        ["game_id", "season_id", "game_date"]
        + [c.lower() + "_home" for c in stat_cols]
    )

    away_renamed = away[["GAME_ID"] + stat_cols].copy()
    away_renamed.columns = ["game_id"] + [c.lower() + "_away" for c in stat_cols]

    wide = home_renamed.merge(away_renamed, on="game_id", how="inner")

    # Hedef değişken
    wide["target"] = (wide["wl_home"] == "W").astype(int)

    # game.csv ile tutarlı sütun adları
    wide = wide.rename(columns={
        "team_id_home": "team_id_home",
        "team_abbreviation_home": "team_abbreviation_home",
        "team_name_home": "team_name_home",
        "team_id_away": "team_id_away",
        "team_abbreviation_away": "team_abbreviation_away",
        "team_name_away": "team_name_away",
        "ft_pct_home": "ft_pct_home",
        "ft_pct_away": "ft_pct_away",
        "fg_pct_home": "fg_pct_home",
        "fg_pct_away": "fg_pct_away",
        "fg3_pct_home": "fg3_pct_home",
        "fg3_pct_away": "fg3_pct_away",
    })

    # Eksik other_stats ve line_score kolonları — NaN olarak bırakılır,
    # XGBoost native NaN handling ile tolere eder
    other_stats_cols = [
        "pts_paint_home", "pts_2nd_chance_home", "pts_fb_home",
        "largest_lead_home", "lead_changes", "times_tied",
        "team_turnovers_home", "total_turnovers_home", "team_rebounds_home",
        "pts_off_to_home",
        "pts_paint_away", "pts_2nd_chance_away", "pts_fb_away",
        "largest_lead_away", "team_turnovers_away", "total_turnovers_away",
        "team_rebounds_away", "pts_off_to_away",
    ]
    line_score_cols = [
        "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home",
        "pts_qtr1_away", "pts_qtr2_away", "pts_qtr3_away", "pts_qtr4_away",
        "pts_ot1_home", "pts_ot1_away",
    ]
    for col in other_stats_cols + line_score_cols:
        wide[col] = float("nan")

    wide["went_to_ot"] = float("nan")  # periyot verisi olmadan belirlenemiyor
    wide["season_type"] = "Regular Season"

    wide = wide.sort_values("game_date").reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
# Adım 3: Önbellekten yükle veya çekip kaydet
# ---------------------------------------------------------------------------
def get_season(season_str: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"season_{season_str}.parquet"

    if cache_path.exists() and not force_refresh:
        print(f"  [ÖNBELLEKTEN] {season_str}: {cache_path.name}")
        return pd.read_parquet(cache_path)

    valid_ids = get_valid_team_ids()
    raw = fetch_season_raw(season_str)

    if raw.empty:
        print(f"  [UYARI] {season_str} için veri gelmedi, atlanıyor.")
        return pd.DataFrame()

    wide = convert_to_wide(raw, valid_ids)

    if wide.empty:
        print(f"  [UYARI] {season_str} dönüşüm sonrası boş, atlanıyor.")
        return pd.DataFrame()

    wide.to_parquet(cache_path, index=False)
    print(f"  [KAYDEDİLDİ] {season_str}: {len(wide)} maç → {cache_path.name}")
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
        summary_path = CACHE_DIR / "all_api_seasons.parquet"
        combined.to_parquet(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"  Toplam maç: {len(combined)}")
        print(f"  Tarih aralığı: {combined['game_date'].min().date()} → {combined['game_date'].max().date()}")
        print(f"  Birleşik dosya: {summary_path}")
        print(f"{'='*60}\n")
    else:
        print("\n[UYARI] Hiçbir sezon için veri çekilemedi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA sezon verilerini nba_api'den çeker.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
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
