import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import build_team_game_log

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
