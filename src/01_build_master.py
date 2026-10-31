"""
01_build_master.py
------------------
Merges Kaggle CSVs + nba_api cache into a single
master_games.parquet file.

Output: data/processed/master_games.parquet
  - One row = one Regular Season game
  - ~2000–bugün all seasons between
  - home + away columns side by side (wide format)
  - target: 1 = home team won, 0 = away team won

Usage:
    python src/01_build_master.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "api_cache"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load and filter Kaggle game.csv
# ---------------------------------------------------------------------------
def load_kaggle_games() -> pd.DataFrame:
    game_path = RAW_DIR / "game.csv"
    if not game_path.exists():
        print("[1/5] Kaggle game.csv not found — using nba_api cache only.")
        return pd.DataFrame()

    print("[1/5] Loading Kaggle game.csv...")
    game = pd.read_csv(game_path, parse_dates=["game_date"])

    # Date filter: 2000+ modern era
    game = game[game["game_date"] >= "2000-01-01"]

    # Season type: Regular Season only
    game = game[game["season_type"] == "Regular Season"]

    # Valid NBA team IDs (team.csv referans)
    team = pd.read_csv(RAW_DIR / "team.csv")
    valid_ids = set(team["id"].astype(str))
    game = game[
        game["team_id_home"].astype(str).isin(valid_ids)
        & game["team_id_away"].astype(str).isin(valid_ids)
    ]

    # Drop if target variable is missing (very rare)
    game = game.dropna(subset=["wl_home"])

    # Binary target
    game["target"] = (game["wl_home"] == "W").astype(int)

    # Column cleanup: only what we need
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
    # Only take existing columns (some columns may vary by CSV version)
    keep_cols = [c for c in keep_cols if c in game.columns]
    game = game[keep_cols].copy()

    game = game.sort_values("game_date").reset_index(drop=True)
    print(f"  → {len(game)} games ({game['game_date'].min().date()} – {game['game_date'].max().date()})")
    return game


# ---------------------------------------------------------------------------
# 2. Load nba_api cache (api_cache/all_api_seasons.parquet)
# ---------------------------------------------------------------------------
def load_api_cache() -> pd.DataFrame:
    # Per-season files are the source of truth (06 refreshes the current one)
    individual = sorted(CACHE_DIR.glob("season_*.parquet"))
    if not individual:
        print("[2/5] API cache not found. Run 06_update_data.py first.")
        return pd.DataFrame()

    frames = [pd.read_parquet(p) for p in individual]
    api_df = pd.concat(frames, ignore_index=True)

    print(f"[2/5] API cache loaded: {len(api_df)} games")

    # ensure game_date column is datetime
    api_df["game_date"] = pd.to_datetime(api_df["game_date"])

    # Align API data column names to Kaggle format
    col_map = {
        "team_id_home": "team_id_home",
        "team_abbreviation_home": "team_abbreviation_home",
        "team_name_home": "team_name_home",
        "team_id_away": "team_id_away",
        "team_abbreviation_away": "team_abbreviation_away",
        "team_name_away": "team_name_away",
        # API lowercase, Kaggle also lowercase — consistent
    }
    # pts_home / pts_away Exists in Kaggle, exists in API too
    if "pts_home" not in api_df.columns and "pts_home" in api_df.columns:
        pass  # already correct

    return api_df


# ---------------------------------------------------------------------------
# 3. Merge two sources, resolve conflicts
# ---------------------------------------------------------------------------
def merge_sources(kaggle_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Merging sources...")

    if api_df.empty:
        print("  → Only Kaggle data is used.")
        return kaggle_df

    if kaggle_df.empty:
        print(f"  → Only API data is used: {len(api_df)} games")
        return api_df.sort_values("game_date").reset_index(drop=True)

    # API verisindeki gameslardan Kaggle'da olmayanları al
    if "game_id" in api_df.columns and "game_id" in kaggle_df.columns:
        existing_ids = set(kaggle_df["game_id"].astype(str))
        new_games = api_df[~api_df["game_id"].astype(str).isin(existing_ids)].copy()
    else:
        # if no game_id, date-based intersection analysis
        kaggle_last = kaggle_df["game_date"].max()
        new_games = api_df[api_df["game_date"] > kaggle_last].copy()

    print(f"  → Kaggle: {len(kaggle_df)} games")
    print(f"  → API (new): {len(new_games)} games")

    # force game_id to string in both sources (mixed type → pyarrow hatası)
    kaggle_df["game_id"] = kaggle_df["game_id"].astype(str)
    new_games["game_id"] = new_games["game_id"].astype(str)

    # Find common columns, add NaN for missing ones
    all_cols = list(dict.fromkeys(list(kaggle_df.columns) + list(new_games.columns)))

    for col in all_cols:
        if col not in kaggle_df.columns:
            kaggle_df[col] = np.nan
        if col not in new_games.columns:
            new_games[col] = np.nan

    combined = pd.concat([kaggle_df[all_cols], new_games[all_cols]], ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)

    print(f"  → Combined total: {len(combined)} games")
    return combined


# ---------------------------------------------------------------------------
# 4. merge other_stats.csv
# ---------------------------------------------------------------------------
def merge_other_stats(master: pd.DataFrame) -> pd.DataFrame:
    other_path = RAW_DIR / "other_stats.csv"
    if not other_path.exists():
        print("[4/5] other_stats.csv not found — skipped (columns stay NaN).")
        return master

    print("[4/5] other_stats.csv merging (LEFT JOIN)...")
    other = pd.read_csv(other_path)

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
    other["game_id"] = other["game_id"].astype(str)  # type consistency

    # Drop these columns from combined table if they already exist as NaN
    # (they were left empty for API data), fill from other_stats
    existing_other_cols = [c for c in keep if c != "game_id" and c in master.columns]
    master_no_other = master.drop(columns=existing_other_cols, errors="ignore")

    master_merged = master_no_other.merge(other, on="game_id", how="left")

    coverage = other["game_id"].nunique()
    total = master["game_id"].nunique()
    print(f"  → other_stats coverage: {coverage}/{total} games ({100*coverage/total:.1f}%)")
    return master_merged


# ---------------------------------------------------------------------------
# 5. line_score.csv merge (period scores)
# ---------------------------------------------------------------------------
def merge_line_score(master: pd.DataFrame) -> pd.DataFrame:
    ls_path = RAW_DIR / "line_score.csv"
    ls = None
    if ls_path.exists():
        print("[5/5] line_score.csv merging (LEFT JOIN)...")
        ls = pd.read_csv(ls_path)

        keep = [
            "game_id",
            "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home",
            "pts_ot1_home",
            "pts_qtr1_away", "pts_qtr2_away", "pts_qtr3_away", "pts_qtr4_away",
            "pts_ot1_away",
        ]
        keep = [c for c in keep if c in ls.columns]
        ls = ls[keep].drop_duplicates(subset=["game_id"]).copy()
        ls["game_id"] = ls["game_id"].astype(str)  # type consistency

        # Drop existing line_score columns (API data left empty)
        existing_ls_cols = [c for c in keep if c != "game_id" and c in master.columns]
        master_no_ls = master.drop(columns=existing_ls_cols, errors="ignore")

        master_merged = master_no_ls.merge(ls, on="game_id", how="left")
    else:
        # Derived columns below are still created (as NaN) so downstream
        # rolling features get numeric dtypes
        print("[5/5] line_score.csv not found — skipped (columns stay NaN).")
        master_merged = master.copy()

    # OT and quarter difference derivations
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

    if ls is not None:
        coverage = ls["game_id"].nunique()
        total = master["game_id"].nunique()
        print(f"  → line_score coverage: {coverage}/{total} games ({100*coverage/total:.1f}%)")
    return master_merged


# ---------------------------------------------------------------------------
# 6. Player box scores (oyuncu eksikliği feature'ları için)
# ---------------------------------------------------------------------------
def build_player_games() -> None:
    files = sorted(CACHE_DIR.glob("players_*.parquet"))
    if not files:
        print("[6/6] Oyuncu verisi yok (06_update_data.py ile çekilir) — atlandı.")
        return
    players = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    players = players.drop_duplicates(subset=["game_id", "player_id"])
    out_path = PROC_DIR / "player_games.parquet"
    players.to_parquet(out_path, index=False)
    print(f"[6/6] Oyuncu box score'ları: {len(players)} satır, {len(files)} sezon → {out_path}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print("Building NBA Master Table")
    print(f"{'='*60}\n")

    kaggle = load_kaggle_games()
    api = load_api_cache()
    master = merge_sources(kaggle, api)
    master = merge_other_stats(master)
    master = merge_line_score(master)

    # Final check
    master = master.sort_values("game_date").reset_index(drop=True)
    master["game_date"] = pd.to_datetime(master["game_date"])

    # Type consistency - pyarrow crashes on mixed-type columns
    # game_id: always string (Kaggle'da '0022300061', API'de int can be)
    master["game_id"] = master["game_id"].astype(str)
    # team_id'ler: string'e normalize
    for col in ["team_id_home", "team_id_away"]:
        if col in master.columns:
            master[col] = master[col].astype(str)
    # season_id: string
    if "season_id" in master.columns:
        master["season_id"] = master["season_id"].astype(str)

    out_path = PROC_DIR / "master_games.parquet"
    master.to_parquet(out_path, index=False)
    build_player_games()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Toplam games     : {len(master)}")
    print(f"  Date range  : {master['game_date'].min().date()} → {master['game_date'].max().date()}")
    print(f"  Home W ratio: {master['target'].mean():.3f}")
    print(f"  Total columns   : {master.shape[1]}")
    print(f"  Save path          : {out_path}")
    print(f"{'='*60}\n")

    # Simple data quality report
    null_pcts = (master.isnull().sum() / len(master) * 100).sort_values(ascending=False)
    null_pcts = null_pcts[null_pcts > 0]
    if not null_pcts.empty:
        print("Missing data ratios (> 0%):")
        print(null_pcts.to_string())
        print()


if __name__ == "__main__":
    main()
