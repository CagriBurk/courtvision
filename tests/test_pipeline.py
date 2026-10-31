import importlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import build_feature_matrix, build_team_game_log, compute_team_rolling_features
from season import current_season, season_id, season_start_year, season_str

predict_today = importlib.import_module("05_predict_today")

META_COLS = [
    "game_id", "game_date", "season_id", "season_type",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "team_id_away", "team_abbreviation_away", "team_name_away",
    "target",
]

TEAMS = ["101", "102", "103", "104"]


def make_master(seasons=(2023, 2024, 2025), games_per_season=60, seed=0) -> pd.DataFrame:
    """Synthetic league: one game per day, rotating matchups, random box scores."""
    rng = np.random.default_rng(seed)
    rows = []
    n = 0
    for start_year in seasons:
        day0 = pd.Timestamp(f"{start_year}-10-20")
        for i in range(games_per_season):
            home, away = rng.choice(TEAMS, size=2, replace=False)
            pts_h, pts_a = rng.integers(90, 130, size=2)
            if pts_h == pts_a:
                pts_h += 1
            row = {
                "game_id": f"G{n:05d}",
                "game_date": day0 + pd.Timedelta(days=i),
                "season_id": season_id(start_year),
                "season_type": "Regular Season",
                "team_id_home": home, "team_abbreviation_home": f"T{home}", "team_name_home": f"Team {home}",
                "team_id_away": away, "team_abbreviation_away": f"T{away}", "team_name_away": f"Team {away}",
                "target": int(pts_h > pts_a),
                "pts_home": pts_h, "pts_away": pts_a,
                "plus_minus_home": pts_h - pts_a, "plus_minus_away": pts_a - pts_h,
            }
            for side in ("home", "away"):
                fga, fta = rng.integers(80, 95), rng.integers(15, 30)
                fgm, ftm = rng.integers(35, 48), rng.integers(10, 15)
                fg3a, fg3m = rng.integers(25, 40), rng.integers(8, 16)
                oreb, dreb = rng.integers(5, 15), rng.integers(28, 40)
                row.update({
                    f"fgm_{side}": fgm, f"fga_{side}": fga, f"fg_pct_{side}": fgm / fga,
                    f"fg3m_{side}": fg3m, f"fg3a_{side}": fg3a, f"fg3_pct_{side}": fg3m / fg3a,
                    f"ftm_{side}": ftm, f"fta_{side}": fta, f"ft_pct_{side}": ftm / fta,
                    f"oreb_{side}": oreb, f"dreb_{side}": dreb, f"reb_{side}": oreb + dreb,
                    f"ast_{side}": rng.integers(18, 30), f"stl_{side}": rng.integers(4, 12),
                    f"blk_{side}": rng.integers(2, 8), f"tov_{side}": rng.integers(8, 18),
                    f"pf_{side}": rng.integers(15, 25),
                })
            rows.append(row)
            n += 1
    return pd.DataFrame(rows)


def test_live_features_match_training_features():
    """
    The feature vector built at prediction time (game appended without a
    result) must equal the one the model was trained on for the same game.
    """
    master = make_master()
    target_game = master.iloc[-1]
    target_date = target_game["game_date"].date()

    # Training path: full table, the game's own result is known
    train_feats, _ = build_feature_matrix(master)
    feat_cols = [c for c in train_feats.columns if c not in META_COLS]
    expected = train_feats.set_index("game_id").loc[target_game["game_id"], feat_cols]

    # Live path: only games before that day + the game as an upcoming fixture
    games = [{
        "game_id": target_game["game_id"],
        "home_team_id": int(target_game["team_id_home"]),
        "away_team_id": int(target_game["team_id_away"]),
        "home_team_abbr": target_game["team_abbreviation_home"],
        "away_team_abbr": target_game["team_abbreviation_away"],
        "home_team_name": target_game["team_name_home"],
        "away_team_name": target_game["team_name_away"],
    }]
    live = predict_today.compute_live_features(master, games, target_date)
    actual = live.loc[target_game["game_id"], feat_cols]

    # rolling_since must have dropped the first season for this test to mean anything
    assert pd.Timestamp(f"{season_start_year(target_date) - 1}-08-01") > master["game_date"].min()

    mismatched = [
        c for c in feat_cols
        if not np.isclose(float(expected[c]), float(actual[c]), equal_nan=True)
    ]
    assert mismatched == [], f"Live features differ from training: {mismatched}"


def test_live_features_ignore_games_on_or_after_target_date():
    """--date in the past must not see that day's (or later) results."""
    master = make_master()
    cutoff_game = master.iloc[100]
    target_date = cutoff_game["game_date"].date()
    games = [{
        "game_id": "LIVE1",
        "home_team_id": int(cutoff_game["team_id_home"]),
        "away_team_id": int(cutoff_game["team_id_away"]),
        "home_team_abbr": "H", "away_team_abbr": "A",
        "home_team_name": "Home", "away_team_name": "Away",
    }]

    live_full = predict_today.compute_live_features(master, games, target_date)
    truncated = master[master["game_date"] < pd.Timestamp(target_date)]
    live_trunc = predict_today.compute_live_features(truncated, games, target_date)

    pd.testing.assert_frame_equal(live_full, live_trunc)


def test_games_in_last_n_days_matches_brute_force():
    dates = pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-06",
        "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-20",
    ])
    master = pd.DataFrame({
        "game_id": [f"G{i}" for i in range(len(dates))],
        "game_date": dates,
        "season_id": "22023",
        "team_id_home": "A",
        "team_id_away": "B",
        "target": 1,
        "pts_home": 100, "pts_away": 90,
        "plus_minus_home": 10, "plus_minus_away": -10,
    })
    feats = compute_team_rolling_features(build_team_game_log(master))
    feats = feats[feats["team_id"] == "A"].sort_values("game_date")

    for n in (5, 7):
        expected = [
            int(((dates < d) & (dates >= d - pd.Timedelta(days=n))).sum()) for d in dates
        ]
        assert feats[f"games_in_last_{n}_days"].tolist() == expected


def test_season_helpers():
    assert current_season(date(2026, 10, 31)) == "2026-27"
    assert current_season(date(2027, 4, 15)) == "2026-27"
    assert current_season(date(2026, 7, 1)) == "2025-26"
    assert season_str(1999) == "1999-00"
    assert season_id(2026) == "22026"


def make_players(master: pd.DataFrame, seed=1) -> pd.DataFrame:
    """9-man rosters per team; each game 0-2 random players sit out."""
    rng = np.random.default_rng(seed)
    weights = np.array([36, 34, 32, 30, 26, 24, 22, 20, 16], dtype=float)
    rows = []
    for _, g in master.iterrows():
        for team in (g["team_id_home"], g["team_id_away"]):
            playing = rng.random(9) > 0.12
            mins = weights * playing
            mins = mins / mins.sum() * 240
            for k in np.flatnonzero(playing):
                rows.append({
                    "player_id": f"{team}_{k}", "player_name": f"Player {team} {k}",
                    "team_id": team, "game_id": g["game_id"], "game_date": g["game_date"],
                    "min": mins[k], "pts": rng.integers(0, 30), "fgm": rng.integers(0, 12),
                    "fga": rng.integers(5, 20), "ftm": rng.integers(0, 6), "fta": rng.integers(0, 8),
                    "oreb": rng.integers(0, 4), "dreb": rng.integers(0, 8), "ast": rng.integers(0, 8),
                    "stl": rng.integers(0, 3), "blk": rng.integers(0, 3), "tov": rng.integers(0, 5),
                    "pf": rng.integers(0, 5),
                })
    return pd.DataFrame(rows)


def test_live_features_match_training_with_players():
    """Same parity check with availability features: the live path gets the
    absent players from the 'injury report', training sees them in box scores."""
    master = make_master()
    players = make_players(master)
    # A game late in the last season (absent players return in later games)
    target_game = master.iloc[-20]
    target_date = target_game["game_date"].date()
    gid = target_game["game_id"]

    train_feats, _ = build_feature_matrix(master, player_games=players)
    feat_cols = [c for c in train_feats.columns if c not in META_COLS]
    assert "home_missing_min_share" in feat_cols
    expected = train_feats.set_index("game_id").loc[gid, feat_cols]

    # Injury report = roster players who did not play in that game
    override = {}
    for team in (target_game["team_id_home"], target_game["team_id_away"]):
        roster = {f"{team}_{k}" for k in range(9)}
        played = set(players[(players["game_id"] == gid) & (players["team_id"] == team)]["player_id"])
        override[(gid, team)] = roster - played

    games = [{
        "game_id": gid,
        "home_team_id": int(target_game["team_id_home"]),
        "away_team_id": int(target_game["team_id_away"]),
        "home_team_abbr": target_game["team_abbreviation_home"],
        "away_team_abbr": target_game["team_abbreviation_away"],
        "home_team_name": target_game["team_name_home"],
        "away_team_name": target_game["team_name_away"],
    }]
    live_players = players[players["game_date"] < pd.Timestamp(target_date)]
    live = predict_today.compute_live_features(master, games, target_date, live_players, override)
    actual = live.loc[gid, feat_cols]

    mismatched = [
        c for c in feat_cols
        if not np.isclose(float(expected[c]), float(actual[c]), equal_nan=True)
    ]
    assert mismatched == [], f"Live features differ from training: {mismatched}"


def test_factor_groups():
    from modeling import factor_group

    assert factor_group("diff_missing_value_share") == "Eksik oyuncular"
    assert factor_group("elo_diff") == "Elo gücü"
    assert factor_group("h2h_pts_diff_L5") == "Karşılaşma geçmişi"
    assert factor_group("home_days_since_last_game") == "Dinlenme / yorgunluk"
    assert factor_group("diff_season_win_pct") == "Sezon performansı"
    assert factor_group("away_away_win_rate_L10") == "İç saha / deplasman performansı"
    assert factor_group("diff_pts_allowed_L5") == "Savunma (yenilen sayı)"
    assert factor_group("diff_win_rate_L10") == "Son form"
    assert factor_group("is_playoff") == "Diğer"


def test_explain_prediction_matches_model():
    import xgboost as xgb
    from modeling import explain_prediction

    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "elo_diff": rng.normal(0, 100, 400),
        "diff_win_rate_L10": rng.normal(0, 0.2, 400),
        "is_playoff": 0.0,
    })
    y = (X["elo_diff"] / 100 + rng.normal(0, 0.5, 400) > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=2).fit(X, y)

    row = X.loc[[X["elo_diff"].abs().idxmax()]]  # a game where Elo clearly matters
    factors = explain_prediction(model, row)
    labels = {f["label"] for f in factors}
    assert labels <= {"Elo gücü", "Son form", "Diğer"}
    # Elo drives the target, so it should be the biggest factor with the right sign
    assert factors[0]["label"] == "Elo gücü"
    assert np.sign(factors[0]["effect"]) == np.sign(row["elo_diff"].iloc[0])
