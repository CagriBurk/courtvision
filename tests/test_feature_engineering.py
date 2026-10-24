import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import (
    build_team_game_log,
    _streak,
    compute_team_rolling_features,
    compute_h2h_features,
    pivot_to_game_level
)

def test_build_team_game_log():
    # Setup dummy data for a single game
    data = {
        "game_id": ["1"],
        "game_date": ["2023-01-01"],
        "team_id_home": ["1610612737"],
        "team_abbreviation_home": ["ATL"],
        "team_name_home": ["Atlanta Hawks"],
        "team_id_away": ["1610612738"],
        "team_abbreviation_away": ["BOS"],
        "team_name_away": ["Boston Celtics"],
        "target": [1],
        "pts_home": [110],
        "pts_away": [105],
        "ast_home": [25],
        "ast_away": [22],
        "fgm_home": [40],
        "fgm_away": [38]
    }
    df = pd.DataFrame(data)

    # Convert to tall format
    tall_df = build_team_game_log(df)

    # Check that we have 2 rows
    assert len(tall_df) == 2

    # Check the home team row
    atl = tall_df[tall_df["team_abbr"] == "ATL"].iloc[0]
    assert atl["is_home"] == 1
    assert atl["pts_scored"] == 110
    assert atl["pts_allowed"] == 105
    assert atl["won"] == 1

    # Check the away team row
    bos = tall_df[tall_df["team_abbr"] == "BOS"].iloc[0]
    assert bos["is_home"] == 0
    assert bos["pts_scored"] == 105
    assert bos["pts_allowed"] == 110
    assert bos["won"] == 0

def test_streak_shift():
    # 0th element should be 0 because it's shifted(1) and fillna(0)
    # The last element should only consider previous wins
    won_series = pd.Series([1, 1, 0, 1, 0, 0])
    # Expected:
    # idx 0: prev is 0 -> 0 -> but streak logic sets 0? No, current is 0, w is 0 -> -1
    # Actually let's trace:
    # shift(1) fillna(0) -> [0, 1, 1, 0, 1, 0]
    # w=0 -> current = -1
    # w=1 -> current = 1
    # w=1 -> current = 2
    # w=0 -> current = -1
    # w=1 -> current = 1
    # w=0 -> current = -1
    streaks = _streak(won_series)
    assert streaks == [-1, 1, 2, -1, 1, -1]

def test_compute_team_rolling_features_shift():
    # Create sample team log
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    data = {
        "team_id": ["T1"] * 4,
        "team_abbr": ["T1"] * 4,
        "game_id": ["G1", "G2", "G3", "G4"],
        "game_date": dates,
        "season_id": ["22023"] * 4,
        "is_home": [1, 0, 1, 0],
        "won": [1, 0, 1, 1],
        "pts_scored": [100, 90, 110, 120],
        "pts_allowed": [90, 100, 105, 110],
        "plus_minus": [10, -10, 5, 10],
        "fg_pct": [0.45, 0.40, 0.50, 0.55],
        # Add required columns
        "fg3_pct": [0.3, 0.3, 0.3, 0.3],
        "ft_pct": [0.8, 0.8, 0.8, 0.8],
        "ast": [20, 20, 20, 20],
        "tov": [10, 10, 10, 10],
        "reb": [40, 40, 40, 40],
        "oreb": [10, 10, 10, 10],
        "dreb": [30, 30, 30, 30],
        "stl": [5, 5, 5, 5],
        "blk": [5, 5, 5, 5],
        "total_turnovers": [10, 10, 10, 10],
        "pts_paint": [40, 40, 40, 40],
        "pts_2nd_chance": [10, 10, 10, 10],
        "pts_fb": [10, 10, 10, 10],
        "largest_lead": [10, 10, 10, 10],
        "ast_tov_ratio": [2.0, 2.0, 2.0, 2.0],
        "ts_pct": [0.55, 0.55, 0.55, 0.55],
        "efg_pct": [0.5, 0.5, 0.5, 0.5],
        "ftr": [0.2, 0.2, 0.2, 0.2],
        "tov_pct": [12.0, 12.0, 12.0, 12.0],
        "orb_pct": [25.0, 25.0, 25.0, 25.0]
    }
    team_log = pd.DataFrame(data)

    df_features = compute_team_rolling_features(team_log)

    # Check that game 1 has NaN for L5 metrics or expected shift
    # G1: shifted(1) is NaN, so rolling means might be NaN
    assert pd.isna(df_features.iloc[0]["pts_scored_L5"])

    # Check G2: should only use G1
    assert df_features.iloc[1]["pts_scored_L5"] == 100.0

    # Check G3: should use G1 and G2
    assert df_features.iloc[2]["pts_scored_L5"] == 95.0 # (100 + 90) / 2

    # Ensure current game stats aren't leaking
    assert df_features.iloc[3]["pts_scored_L5"] == 100.0 # (100+90+110)/3

def test_compute_h2h_features():
    # Setup master df with multiple games between two teams
    data = {
        "game_id": ["G1", "G2", "G3"],
        "game_date": pd.to_datetime(["2023-01-01", "2023-01-10", "2023-01-20"]),
        "team_id_home": ["A", "B", "A"],
        "team_id_away": ["B", "A", "B"],
        "target": [1, 0, 1], # home won
        "pts_home": [100, 90, 110],
        "pts_away": [90, 100, 100],
    }
    master = pd.DataFrame(data)

    h2h_df = compute_h2h_features(master)

    # G1: No previous history
    assert h2h_df.iloc[0]["h2h_n_games"] == 0
    assert h2h_df.iloc[0]["h2h_home_win_rate"] == 0.5
    assert h2h_df.iloc[0]["h2h_pts_diff_mean"] == 0.0

    # G2: A won against B as home. Diff was +10 for A.
    # Current home is B. Previous game A was home and won.
    # The home team for G2 is B. B lost previous game.
    # The history loop evaluates points based on the exact matching 'home' and 'away'. Wait, no.
    # The history loop uses dict mapping home team's perspective in previous matchups.
    assert h2h_df.iloc[1]["h2h_n_games"] == 1

    # G3: A is home again. Previous games:
    # G1: A home, B away, A won (+10)
    # G2: B home, A away, B lost (-10) -> for B vs A
    assert h2h_df.iloc[2]["h2h_n_games"] == 2

def test_pivot_to_game_level():
    # Setup simple tall_features
    data = {
        "game_id": ["G1", "G1"],
        "team_id": ["A", "B"],
        "is_home": [1, 0],
        "pts_scored_L5": [100.0, 90.0]
    }
    tall_features = pd.DataFrame(data)

    master_data = {
        "game_id": ["G1"],
        "game_date": pd.to_datetime(["2023-01-01"]),
        "team_id_home": ["A"],
        "team_id_away": ["B"],
        "target": [1],
        "h2h_n_games": [0],
        "h2h_home_win_rate": [0.5],
        "h2h_pts_diff_mean": [0.0],
        "season_type": ["Regular Season"]
    }
    master_with_h2h = pd.DataFrame(master_data)

    wide_features = pivot_to_game_level(tall_features, master_with_h2h)

    # Check structure
    assert len(wide_features) == 1
    assert "home_pts_scored_L5" in wide_features.columns
    assert "away_pts_scored_L5" in wide_features.columns

    assert wide_features.iloc[0]["home_pts_scored_L5"] == 100.0
    assert wide_features.iloc[0]["away_pts_scored_L5"] == 90.0

def test_compute_live_team_features():
    from feature_engineering import compute_live_team_features
    data = {
        "GAME_DATE": ["2023-01-01", "2023-01-03", "2023-01-05"],
        "WL": ["W", "L", "W"],
        "MATCHUP": ["ATL vs. BOS", "ATL @ NYK", "ATL vs. MIA"],
        "PTS": [110, 100, 115],
        "PLUS_MINUS": [5, -10, 10],
        "FG_PCT": [0.45, 0.40, 0.50],
        "FG3_PCT": [0.3, 0.3, 0.3],
        "FT_PCT": [0.8, 0.8, 0.8],
        "FGA": [90, 90, 90],
        "FTA": [20, 20, 20],
        "REB": [40, 40, 40],
        "AST": [20, 20, 20],
        "TOV": [10, 10, 10],
        "STL": [5, 5, 5],
        "BLK": [5, 5, 5],
        "OREB": [10, 10, 10],
        "DREB": [30, 30, 30]
    }
    df = pd.DataFrame(data)

    # Compute live features for next game (which is index 3 technically if today is later)
    today = pd.Timestamp("2023-01-07")
    feats = compute_live_team_features(df, today=today)

    import pytest
    assert "pace_L5" in feats
    assert "point_diff_L5" in feats
    assert feats["pace_L5"] > 0
