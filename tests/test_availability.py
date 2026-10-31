import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import compute_availability_features
from player_stats import match_player_ids, normalize_player_name


def box(player_id, team_id, game_id, game_date, minutes, pts=10):
    return {
        "player_id": player_id, "player_name": player_id, "team_id": team_id,
        "game_id": game_id, "game_date": pd.Timestamp(game_date), "min": minutes,
        "pts": pts, "fgm": 4, "fga": 9, "ftm": 2, "fta": 2, "oreb": 1, "dreb": 3,
        "ast": 2, "stl": 1, "blk": 0, "tov": 1, "pf": 2,
    }


@pytest.fixture
def league():
    """
    Team A plays 13 games (one per day). Regulars: STAR 30 min, MID 24 min,
    BENCH 10 min. TRADED plays for A in games 1-6, then for team B.
    STAR misses game 12 (index 11) and returns for game 13;
    TRADED is gone from A after game 6.
    """
    dates = pd.date_range("2025-01-01", periods=13, freq="D")
    team_games = pd.DataFrame({
        "game_id": [f"A{i}" for i in range(13)], "game_date": dates, "team_id": "A",
    })
    rows = []
    for i, d in enumerate(dates):
        gid = f"A{i}"
        if i != 11:
            rows.append(box("STAR", "A", gid, d, 30, pts=25))
        rows.append(box("MID", "A", gid, d, 24))
        rows.append(box("BENCH", "A", gid, d, 10))
        if i < 6:
            rows.append(box("TRADED", "A", gid, d, 20))
        else:
            rows.append(box("TRADED", "B", f"B{i}", d, 20))
    return team_games, pd.DataFrame(rows)


def feats_for(result, game_id, team_id="A"):
    return result[(result["game_id"] == game_id) & (result["team_id"] == team_id)].iloc[0]


def test_absent_star_counts_as_missing(league):
    team_games, players = league
    result = compute_availability_features(team_games, players)

    last = feats_for(result, "A11")
    assert last["missing_min_share"] == pytest.approx(30 / 240)
    assert last["n_missing_rotation"] == 1
    assert 0 < last["missing_value_share"] < 1

    # A normal game with everyone available
    assert feats_for(result, "A9")["missing_min_share"] == 0


def test_long_absence_does_not_dilute_expected_minutes(league):
    # STAR also misses games 8-10: still a 30-minute player when he misses game 12
    team_games, players = league
    players = players[~((players["player_id"] == "STAR") & players["game_id"].isin(["A8", "A9", "A10"]))]
    result = compute_availability_features(team_games, players)

    last = feats_for(result, "A11")
    assert last["missing_min_share"] == pytest.approx(30 / 240)
    assert last["n_missing_rotation"] == 1


def test_traded_player_is_not_missing(league):
    team_games, players = league
    result = compute_availability_features(team_games, players)
    # Games 7-10: TRADED still in A's 10-game window but now plays for B
    for i in range(7, 11):
        assert feats_for(result, f"A{i}")["missing_min_share"] == 0


def test_first_game_without_history_is_nan(league):
    team_games, players = league
    result = compute_availability_features(team_games, players)
    assert np.isnan(feats_for(result, "A0")["missing_min_share"])


def test_player_who_never_returns_is_not_missing(league):
    # Retired / left the league: no later appearance for the team → not on the roster
    team_games, players = league
    players = players[~((players["player_id"] == "STAR") & (players["game_id"] == "A12"))]
    result = compute_availability_features(team_games, players)
    assert feats_for(result, "A11")["missing_min_share"] == 0


def test_upcoming_game_uses_injury_report(league):
    team_games, players = league
    upcoming = pd.DataFrame({
        "game_id": ["A13"], "game_date": [pd.Timestamp("2025-01-14")], "team_id": ["A"],
    })
    games = pd.concat([team_games, upcoming], ignore_index=True)

    # MID listed as Out
    result = compute_availability_features(games, players, {("A13", "A"): {"MID"}})
    row = feats_for(result, "A13")
    assert row["missing_min_share"] == pytest.approx(24 / 240)

    # Injury report unavailable → unknown, not "everyone plays"
    result = compute_availability_features(games, players)
    assert np.isnan(feats_for(result, "A13")["missing_min_share"])


def test_name_matching():
    assert normalize_player_name("Nikola Jokić") == "nikola jokic"
    assert normalize_player_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_player_name("De'Aaron Fox") == "de aaron fox"

    roster = {"1": "Nikola Jokic", "2": "Jamal Murray", "3": "Michael Porter Jr."}
    ids, missed = match_player_ids(["Nikola Jokić", "Michael Porter Jr.", "Unknown Guy"], roster)
    assert ids == {"1", "3"}
    assert missed == ["Unknown Guy"]
    # Unique last name fallback (ESPN short names)
    ids, _ = match_player_ids(["J. Murray"], roster)
    assert ids == {"2"}
