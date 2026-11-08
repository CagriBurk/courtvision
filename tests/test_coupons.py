import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import coupons
from coupons import (
    TIERS,
    assess_game,
    build_coupons,
    handicap_options,
    leg_won,
    min_odds,
    pick_line,
)
from score_dist import game_lines

DIST = {"task": "margin", "shape": "normal", "sigma": {"kind": "const", "value": 13.6}, "no_tie": True}


def pred(gid, margin, p_clf=None, played=20, status=1, home="HOM", away="AWY"):
    lines = game_lines({"margin": DIST}, margin, None)
    p_mar = lines["margin"]["p_home_win"]
    return {
        "game_id": gid, "home_team_abbr": home, "away_team_abbr": away,
        "pred_margin": margin, "lines": json.dumps(lines),
        "prob_home": p_mar if p_clf is None else p_clf,
        "prob_home_classifier": p_mar if p_clf is None else p_clf,
        "home_games_played": played, "away_games_played": played, "game_status": status,
    }


def test_handicap_lines_and_settlement():
    # Away favoured by 9: legs back the away side with a negative handicap
    opts = handicap_options(DIST, -9.0, "away")
    probs = [o["p"] for o in sorted(opts, key=lambda o: o["handicap"])]
    assert all(np.diff(probs) >= -1e-12)  # more points received → likelier
    leg = pick_line(opts, 0.62)
    assert leg["handicap"] < 0 and abs(leg["p"] - 0.62) < 0.03
    # away -4.5 ⇔ line -4.5 on the home margin: away wins by 5+
    away_leg = {"side": "away", "line": -4.5}
    assert leg_won(away_leg, 100, 106) and not leg_won(away_leg, 100, 104)
    home_leg = {"side": "home", "line": 6.5}  # home -6.5
    assert leg_won(home_leg, 110, 103) and not leg_won(home_leg, 110, 104)
    assert min_odds(0.5) == 2.1 and min_odds(0.72) >= 1.05 / 0.72


def test_game_filters():
    assert assess_game(pred("G1", 8.0), DIST)["strong"]
    assert not assess_game(pred("G2", 2.0), DIST)["strong"]            # no clear favourite
    assert assess_game(pred("G2", 2.0), DIST)["eligible"]
    assert not assess_game(pred("G3", 8.0, p_clf=0.45), DIST)["eligible"]  # opposite sides
    far = assess_game(pred("G4", 8.0, p_clf=0.95), DIST)
    assert not far["eligible"] and "fark" in far["reasons"][0]
    assert not assess_game(pred("G5", 8.0, played=3), DIST)["eligible"]  # early season
    assert not assess_game(pred("G6", 8.0, status=2), DIST)["eligible"]   # already started
    assert assess_game({**pred("G7", 8.0), "lines": None}, DIST) is None


def test_build_coupons_uses_reliable_games_only():
    preds = [pred(f"S{i}", m) for i, m in enumerate([9.0, -7.5, 6.0])]  # 3 strong
    preds += [pred("W1", 1.5), pred("X1", 8.0, p_clf=0.4)]               # weak, contradicting
    out = build_coupons(preds, DIST)
    assert out["n_legs"] == 3
    assert [c["tier"] for c in out["coupons"]] == [t["key"] for t in TIERS]
    for c in out["coupons"]:
        ids = [l["game_id"] for l in c["legs"]]
        assert sorted(ids) == ["S0", "S1", "S2"] and len(set(ids)) == len(ids)
        assert c["p"] == pytest.approx(np.prod([l["p"] for l in c["legs"]]), abs=1e-3)
        assert c["min_odds"] >= c["fair_odds"] * 1.05 ** len(c["legs"]) - 0.02
        for l in c["legs"]:
            assert abs(l["p"] - c["target"]) < 0.03
    safe, risky = out["coupons"][0], out["coupons"][-1]
    assert safe["p"] > risky["p"] and safe["fair_odds"] < risky["fair_odds"]
    # The away favourite's leg is labelled from the away side
    s1 = next(l for l in safe["legs"] if l["game_id"] == "S1")
    assert s1["team"] == "AWY" and s1["side"] == "away"


def test_leg_count_follows_strong_games():
    many_weak = [pred(f"W{i}", 2.0 + 0.1 * i) for i in range(6)]
    assert build_coupons(many_weak, DIST)["n_legs"] == 2      # busy night, no strong games
    many_strong = [pred(f"S{i}", 8.0 + i) for i in range(6)]
    assert build_coupons(many_strong, DIST)["n_legs"] == 4    # capped
    lonely = build_coupons([pred("S0", 9.0), pred("X", 9.0, p_clf=0.3)], DIST)
    assert lonely["coupons"] == [] and "en az 2" in lonely["note"]


def test_legs_win_as_often_as_promised():
    """Simulated games drawn from the same distribution: hit rate ≈ promised p."""
    rng = np.random.default_rng(0)
    for target in (0.72, 0.54):
        hits, promised = [], []
        for mu in rng.normal(0, 7, 400):
            side = coupons.favourite_side(DIST, mu)
            leg = pick_line(handicap_options(DIST, mu, side), target)
            for m in np.round(mu + rng.normal(0, 13.6, 25)):
                if m == 0:
                    continue
                hits.append(leg_won(leg, m, 0))
                promised.append(leg["p"])
        assert np.mean(hits) == pytest.approx(np.mean(promised), abs=0.015)
