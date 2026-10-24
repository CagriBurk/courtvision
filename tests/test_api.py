import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api_server import app, _run_predictions_if_stale

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_get_teams():
    response = client.get("/api/teams")
    assert response.status_code == 200
    data = response.json()
    assert "teams" in data
    teams = data["teams"]
    assert isinstance(teams, list)
    # Check that ATL is in the list
    assert any(team["abbreviation"] == "ATL" for team in teams)
    # Check proper format
    atl = next(team for team in teams if team["abbreviation"] == "ATL")
    assert atl["full_name"] == "Atlanta Hawks"

def test_today_games(mocker):
    # Mock _run_predictions_if_stale since it runs modeling code
    mocker.patch('api_server._run_predictions_if_stale', return_value=None)

    # Also mock _load_model and prediction logic if they get hit, or just check the schema
    # The actual endpoint calls _run_predictions_if_stale() and then opens predictions_{today}.json
    # It might return empty or error if file not found. Let's mock the file read too.

    mock_data = [
        {
            "game_id": "0022300001",
            "home_team": "LAL",
            "away_team": "DEN",
            "home_prob": 0.45,
            "away_prob": 0.55,
            "prediction": "DEN",
            "confidence": "Medium",
            "why_wins": ["Better form"],
            "home_form": [1, 0, 1, 1, 0],
            "away_form": [1, 1, 1, 0, 1],
            "home_color": "#552583",
            "away_color": "#0E2240",
            "home_logo": "...",
            "away_logo": "..."
        }
    ]

    import json
    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(mock_data)))
    mocker.patch('os.path.exists', return_value=True)

    # Since api_server does not call _run_predictions_if_stale anymore and reads from parquet directly:
    # let's mock pd.read_parquet and Path.exists instead

    import pandas as pd
    from unittest.mock import MagicMock

    df_mock = pd.DataFrame([
        {
            "game_id": "0022300001",
            "home_team_abbr": "LAL",
            "away_team_abbr": "DEN",
            "home_prob": 0.45,
            "away_prob": 0.55,
            "home_team_id": "1610612747",
            "away_team_id": "1610612743",
        }
    ])

    log_mock = pd.DataFrame([
        {"team_id": "1610612747", "game_date": pd.to_datetime("2023-01-01"), "won": 1},
        {"team_id": "1610612743", "game_date": pd.to_datetime("2023-01-01"), "won": 0},
    ])

    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('player_stats.get_injuries', return_value={})
    mocker.patch('player_stats.get_team_top_players', return_value=[])

    def fake_read_parquet(path, columns=None, **kwargs):
        if "team_game_log" in str(path):
            return log_mock
        return df_mock
    mocker.patch('pandas.read_parquet', side_effect=fake_read_parquet)

    response = client.get("/api/today-games")
    assert response.status_code == 200
    data = response.json()
    assert "games" in data
    assert isinstance(data["games"], list)
    assert len(data["games"]) == 1
    assert data["games"][0]["id"] == "0022300001"
    assert data["games"][0]["home"]["abbreviation"] == "LAL"
    assert data["games"][0]["away"]["abbreviation"] == "DEN"
