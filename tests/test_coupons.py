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
    legs_per_coupon,
    min_odds,
    pick_line,
)
from market_blend import build_lines
from score_dist import game_lines

DIST = {"task": "margin", "shape": "normal", "sigma": {"kind": "const", "value": 13.6}, "no_tie": True}
MARKET = {"task": "margin", "shape": "normal", "sigma": {"kind": "const", "value": 13.4}, "no_tie": True}
MARKET_DIST = {"margin": MARKET, "blend": {"margin": {"w": 0.0}, "total": {"w": 0.0}}}


def pred(gid, margin, p_clf=None, played=20, status=1, home="HOM", away="AWY"):
    """A game priced by the model only (no market line)."""
    lines = game_lines({"margin": DIST}, margin, None)
    p_mar = lines["margin"]["p_home_win"]
    return {
        "game_id": gid, "home_team_abbr": home, "away_team_abbr": away,
        "pred_margin": margin, "lines": json.dumps(lines),
        "prob_home": p_mar if p_clf is None else p_clf,
        "prob_home_classifier": p_mar if p_clf is None else p_clf,
        "home_games_played": played, "away_games_played": played, "game_status": status,
    }


def mpred(gid, market_margin, model_margin=None, status=1, home="HOM", away="AWY"):
    """A game with a market line, built the way 05 builds it."""
    model_margin = market_margin + 5 if model_margin is None else model_margin
    market = {"spread_home": -market_margin, "market_margin": market_margin, "total": None,
              "p_home_novig": None, "provider": "DraftKings"}
    lines, model_lines, info = build_lines({"margin": DIST}, {"margin": model_margin}, market, MARKET_DIST)
    return {
        "game_id": gid, "home_team_abbr": home, "away_team_abbr": away,
        "pred_margin": model_margin, "game_status": status,
        "lines": json.dumps(lines), "model_lines": json.dumps(model_lines), "market": json.dumps(info),
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


def test_assess_game_prices_with_the_market():
    g = assess_game(mpred("M", 7.5, model_margin=12.0, home="DAL", away="BKN"), DIST, MARKET)
    assert g["source"] == "market" and g["eligible"]
    assert g["center"] == pytest.approx(7.5)              # the market, not the model's 12
    assert g["market_label"] == "DAL -7.5"
    assert g["model_label"].startswith("DAL -")
    assert any("piyasa: DAL -7.5" in r for r in g["reasons"])

    m = assess_game(pred("P", 8.0), DIST, MARKET)          # no market line → model
    assert m["source"] == "model" and m["market_label"] is None
    assert not assess_game(mpred("S", 7.5, status=2), DIST, MARKET)["eligible"]   # started
    assert assess_game({**pred("N", 8.0), "lines": None, "pred_margin": None}, DIST, MARKET) is None


def test_one_game_one_coupon():
    margins = [14.0, -13.0, 9.0, -8.0, 4.5, -4.0]
    preds = [mpred(f"G{i}", m) for i, m in enumerate(margins)]
    out = build_coupons(preds, DIST, MARKET)
    assert out["source"] == "market" and out["n_legs"] == 2
    assert [c["tier"] for c in out["coupons"]] == [t["key"] for t in TIERS]
    ids = [l["game_id"] for c in out["coupons"] for l in c["legs"]]
    assert len(ids) == len(set(ids)) == 6                 # every game used once
    for c in out["coupons"]:
        tier = next(t for t in TIERS if t["key"] == c["tier"])
        assert c["p"] == pytest.approx(np.prod([l["p"] for l in c["legs"]]), abs=1e-3)
        for l in c["legs"]:
            assert tier["lo"] <= l["p"] < tier["hi"]
            assert l["handicap"] <= -1.5 and l["market"] == "spread" and l["source"] == "market"
            assert l["fair"].startswith(l["team"])
    # The safe coupon takes the biggest favourites
    safe = {l["game_id"] for l in out["coupons"][0]["legs"]}
    assert safe == {"G0", "G1"}
    away = next(l for l in out["coupons"][0]["legs"] if l["game_id"] == "G1")
    assert away["team"] == "AWY" and away["side"] == "away" and away["label"].startswith("AWY -")
    assert "ust" not in [c["tier"] for c in out["coupons"]]


def test_market_games_take_precedence_over_model_only_games():
    preds = [mpred(f"M{i}", m) for i, m in enumerate([12.0, -11.0, 6.0, -5.0])] + [pred("P", 15.0)]
    out = build_coupons(preds, DIST, MARKET)
    p = next(g for g in out["games"] if g["game_id"] == "P")
    assert not p["eligible"] and "kupon dışı" in p["reasons"][-1]
    assert "P" not in [l["game_id"] for c in out["coupons"] for l in c["legs"]]
    assert out["note"] is None


def test_model_lines_only_on_days_without_market():
    preds = [pred(f"S{i}", m) for i, m in enumerate([12.0, -11.0, 9.5, -9.0, 5.0, -4.5])]
    out = build_coupons(preds, DIST, MARKET)
    assert out["source"] == "model" and "piyasa çizgisi yok" in out["note"]
    assert out["coupons"] and all(l["source"] == "model" for c in out["coupons"] for l in c["legs"])


def test_slight_favourite_never_receives_points():
    """A 3-4 point favourite would need +points to reach ~72%: it stays out of the safe coupon."""
    preds = [mpred("MIA", 4.0, home="MIA", away="CHI"), mpred("BIG", 12.0), mpred("BIG2", -11.5),
             mpred("MID", 7.0)]
    out = build_coupons(preds, DIST, MARKET)
    tiers = {c["tier"]: c for c in out["coupons"]}
    assert "MIA" not in [l["game_id"] for l in tiers["guvenli"]["legs"]]
    assert all(l["handicap"] <= -1.5 for c in out["coupons"] for l in c["legs"])
    # Two slight favourites only: no safe coupon at all, with the reason
    out = build_coupons([mpred("A", 4.0), mpred("B", -4.5)], DIST, MARKET)
    assert "guvenli" not in [c["tier"] for c in out["coupons"]]
    assert out["skipped"][0]["tier"] == "guvenli" and "aralığına" in out["skipped"][0]["reason"]


def test_legs_per_coupon_and_too_few_games():
    assert [legs_per_coupon(n) for n in (2, 3, 6, 8, 9, 20)] == [2, 2, 2, 2, 3, 3]
    lonely = build_coupons([mpred("S0", 9.0), mpred("S1", 9.0, status=2)], DIST, MARKET)
    assert lonely["coupons"] == [] and "en az 2" in lonely["note"]


def test_legacy_over_legs_still_settle():
    assert leg_won({"market": "total", "side": "over", "line": 220.5}, 112, 110)
    assert not leg_won({"market": "total", "side": "over", "line": 220.5}, 110, 110)
    assert coupons.TIER_ORDER["ust"] == 3


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
