import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import prediction_tracker

@pytest.fixture
def temp_log_dir(tmp_path, monkeypatch):
    """Fixture to override the output directory and log file path for tests."""
    temp_dir = tmp_path / "predictions"
    temp_dir.mkdir()
    temp_log = temp_dir / "predictions_log.parquet"

    monkeypatch.setattr(prediction_tracker, "PRED_DIR", temp_dir)
    monkeypatch.setattr(prediction_tracker, "PRED_LOG", temp_log)

    return temp_log

def test_save_predictions_new_file(temp_log_dir):
    predictions = [
        {
            "game_id": "1",
            "date": "2023-11-01",
            "home_team_abbr": "BOS",
            "home_team_id": "100",
            "away_team_abbr": "LAL",
            "away_team_id": "200",
            "prob_home": 0.6,
            "confidence": "YÜKSEK"
        },
        {
            "game_id": "2",
            "date": "2023-11-01",
            "home_team_abbr": "MIA",
            "home_team_id": "300",
            "away_team_abbr": "CHI",
            "away_team_id": "400",
            "prob_home": 0.4,
            "confidence": "DÜŞÜK"
        }
    ]

    df = prediction_tracker.save_predictions(predictions)

    assert len(df) == 2
    assert temp_log_dir.exists()

    saved_df = pd.read_parquet(temp_log_dir)
    assert len(saved_df) == 2

    # Check mappings and predicted winner
    bos_lal = saved_df[saved_df["game_id"] == "1"].iloc[0]
    assert bos_lal["confidence"] == "HIGH"
    assert bos_lal["predicted_winner"] == "HOME"

    mia_chi = saved_df[saved_df["game_id"] == "2"].iloc[0]
    assert mia_chi["confidence"] == "LOW"
    assert mia_chi["predicted_winner"] == "AWAY"
    assert pd.isna(mia_chi["actual_winner"])
    assert pd.isna(mia_chi["correct"])

def test_save_predictions_append(temp_log_dir):
    # First save
    pred1 = [{"game_id": "1", "prob_home": 0.8}]
    prediction_tracker.save_predictions(pred1)

    # Second save with one duplicate and one new
    pred2 = [
        {"game_id": "1", "prob_home": 0.2}, # Should be ignored because game_id=1 exists
        {"game_id": "2", "prob_home": 0.7}  # Should be added
    ]
    df = prediction_tracker.save_predictions(pred2)

    assert len(df) == 2

    # The existing code updates the row if it exists
    game1 = df[df["game_id"] == "1"].iloc[0]
    assert game1["prob_home"] == 0.2
    assert game1["prob_away"] == 0.8

def test_fill_results_no_log(temp_log_dir):
    # If no log exists, should return empty df
    df = prediction_tracker.fill_results(pd.DataFrame())
    assert df.empty

def test_fill_results_with_master(temp_log_dir):
    # Setup log with pending results
    predictions = [
        {"game_id": "1", "prob_home": 0.8}, # HOME wins predicted
        {"game_id": "2", "prob_home": 0.2}, # AWAY wins predicted
        {"game_id": "3", "prob_home": 0.6}, # pending
    ]
    prediction_tracker.save_predictions(predictions)

    # Create mock master df
    master_data = {
        "game_id": ["1", "2"],
        "team_id_home": ["A", "C"],
        "team_id_away": ["B", "D"],
        "target": [1, 1] # 1=home won for both
    }
    master_df = pd.DataFrame(master_data)

    # Fill results
    df = prediction_tracker.fill_results(master_df)

    # Assertions
    game1 = df[df["game_id"] == "1"].iloc[0]
    assert game1["actual_winner"] == "HOME"
    assert game1["correct"] == True

    game2 = df[df["game_id"] == "2"].iloc[0]
    assert game2["actual_winner"] == "HOME"
    assert game2["correct"] == False

    game3 = df[df["game_id"] == "3"].iloc[0]
    assert pd.isna(game3["actual_winner"])
    assert pd.isna(game3["correct"])

def test_get_stats_empty(temp_log_dir):
    stats = prediction_tracker.get_stats()
    assert stats["total"] == 0
    assert stats["raw_accuracy"] is None

def test_get_stats_with_data(temp_log_dir):
    # Create raw dataframe and save directly to parquet
    today = pd.Timestamp.today().date()
    yesterday = today - pd.Timedelta(days=1)
    old = today - pd.Timedelta(days=10)

    data = {
        "game_id": ["1", "2", "3", "4"],
        "date": [str(yesterday), str(yesterday), str(today), str(old)],
        "confidence": ["HIGH", "HIGH", "MEDIUM", "LOW"],
        "correct": [True, False, True, True]
    }
    df = pd.DataFrame(data)
    # add other columns just in case
    df["actual_winner"] = "HOME"
    df["predicted_winner"] = "HOME"
    df.to_parquet(temp_log_dir)

    stats = prediction_tracker.get_stats()

    assert stats["total"] == 4
    assert stats["correct"] == 3
    assert stats["raw_accuracy"] == 75.0

    # Weighted accuracy
    # HIGH=3, MEDIUM=2, LOW=1
    # total weights = 3+3+2+1 = 9
    # correct weights = 3(true) + 0(false) + 2(true) + 1(true) = 6
    # weighted = 6/9 = 66.7%
    assert stats["weighted_accuracy"] == round((6/9) * 100, 1)

    # By tier
    assert stats["by_tier"]["HIGH"]["total"] == 2
    assert stats["by_tier"]["HIGH"]["correct"] == 1
    assert stats["by_tier"]["HIGH"]["accuracy"] == 50.0

    # Last 7 days
    # game 1, 2, 3 are within 7 days. game 4 is old.
    assert stats["last_7_days"]["total"] == 3
    assert stats["last_7_days"]["correct"] == 2

@patch('prediction_tracker.pd.read_parquet')
def test_fill_results_live_no_api(mock_read_parquet, temp_log_dir):
    with patch.dict('sys.modules', {'nba_api.stats.endpoints': None}):
        # Mocking nba_api to be missing
        df = prediction_tracker.fill_results_live()
        assert df.empty

def test_fill_results_live_with_api(temp_log_dir):
    # Setup pending logs
    predictions = [
        {"game_id": "0022300001", "prob_home": 0.8}, # HOME wins predicted
    ]
    prediction_tracker.save_predictions(predictions)

    # We will mock the LeagueGameLog API call
    mock_game_log_data = pd.DataFrame({
        "GAME_ID": ["0022300001"],
        "GAME_DATE": [pd.Timestamp.today().strftime("%Y-%m-%d")],
        "MATCHUP": ["BOS vs. LAL"], # HOME MATCHUP
        "WL": ["W"] # HOME won
    })

    class MockLeagueGameLog:
        def __init__(self, **kwargs):
            pass
        def get_data_frames(self):
            return [mock_game_log_data]

    with patch.dict('sys.modules', {'nba_api.stats.endpoints': MagicMock(leaguegamelog=MagicMock(LeagueGameLog=MockLeagueGameLog))}):
        # Run live update
        df = prediction_tracker.fill_results_live()

        # Verify
        game = df.iloc[0]
        assert game["actual_winner"] == "HOME"
        assert game["correct"] == True
