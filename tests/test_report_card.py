import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from report_card import build_report_card, calibration, coupon_report, line_pairs, wilson
from score_dist import game_lines

MARGIN = {"task": "margin", "shape": "normal", "sigma": {"kind": "const", "value": 13.6}, "no_tie": True}
TOTAL = {"task": "total", "shape": "normal", "sigma": {"kind": "const", "value": 18.8}, "no_tie": False}


def simulated_log(n=600, seed=0):
    """Prediction log whose results really follow the model's distribution."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mu, tot = rng.normal(0, 7), rng.normal(225, 8)
        lines = game_lines({"margin": MARGIN, "total": TOTAL}, mu, tot)
        m = 0
        while m == 0:
            m = int(round(mu + rng.normal(0, 13.6)))
        t = int(round(tot + rng.normal(0, 18.8)))
        if (t + m) % 2:  # integer team scores
            t += 1
        h, a = (t + m) / 2, (t - m) / 2
        p_home = lines["margin"]["p_home_win"]
        rows.append({
            "game_id": f"G{i}", "prob_home": p_home,
            "correct": (m > 0) == (p_home >= 0.5), "actual_winner": "HOME" if m > 0 else "AWAY",
            "pred_margin": mu, "pred_total": tot, "lines_json": json.dumps(lines),
            "actual_home_pts": h, "actual_away_pts": a,
        })
    return pd.DataFrame(rows)


def test_wilson_and_buckets():
    lo, hi = wilson(7, 10)
    assert lo < 0.7 < hi and hi - lo > 0.4  # 10 picks say little
    lo, hi = wilson(700, 1000)
    assert hi - lo < 0.06
    cal = calibration([(0.52, True), (0.58, False), (0.72, True), (0.74, True), (0.9, False)])
    assert [r["n"] for r in cal["rows"]] == [1, 1, 2, 1]
    assert cal["rows"][2]["hit_rate"] == 1.0 and cal["hit_rate"] == pytest.approx(0.6)


def test_lines_are_read_from_the_favoured_side():
    lines = {"margin": {"ladder": [{"line": 4.5, "p_over": 0.30}, {"line": -4.5, "p_over": 0.70}]}}
    log = pd.DataFrame([{"lines_json": json.dumps(lines), "actual_home_pts": 100, "actual_away_pts": 98}])
    # margin +2: under 4.5 came in (promised 70%), over -4.5 came in (promised 70%)
    assert line_pairs(log, "margin") == [(pytest.approx(0.7), True), (0.7, True)]


def test_a_calibrated_model_scores_as_promised():
    card = build_report_card(simulated_log(), coupons=[])
    for name in ("handicap", "total"):
        cal = card[name]["calibration"]
        assert cal["n"] == 600 * 9
        for r in cal["rows"]:
            if r["n"] >= 200:
                assert r["ci_lo"] - 0.02 <= r["mean_pred"] <= r["ci_hi"] + 0.02
    side = card["side"]
    assert side["n"] == 600 and abs(side["hit_rate"] - side["mean_pred"]) < 0.05
    bands = card["handicap"]["bands"]
    assert len(bands) == 12 and sum(b["obs"] for b in bands) == pytest.approx(1)
    assert card["handicap"]["error"]["n"] == 600


def test_coupon_tiers_and_profit():
    def coupon(tier, result, p, odds=None):
        return {"tier": tier, "result": result, "p": p, "played_odds": odds,
                "legs": [{"p": 0.72, "won": result == "won"}, {"p": 0.71, "won": True}]}
    rep = coupon_report([
        coupon("guvenli", "won", 0.51, 2.10), coupon("guvenli", "lost", 0.50, 2.00),
        coupon("guvenli", "won", 0.52), coupon("riskli", "lost", 0.29, 3.60),
        {"tier": "dengeli", "result": None, "p": 0.38, "legs": []},
    ])
    safe = rep["tiers"][0]
    assert (safe["n"], safe["won"], safe["n_played"]) == (3, 2, 2)
    assert safe["profit"] == pytest.approx(1.10 - 1) and safe["roi"] == pytest.approx(0.05)
    assert safe["legs"]["n"] == 6
    assert rep["pending"] == 1 and rep["played"]["n"] == 3
    assert rep["played"]["profit"] == pytest.approx(0.10 - 1)
    assert rep["tiers"][1]["n"] == 0 and rep["tiers"][1]["hit_rate"] is None


def test_empty_inputs():
    card = build_report_card(None, [])
    assert card["side"]["n"] == 0 and card["coupons"]["n"] == 0
    old = pd.DataFrame([{"game_id": "G1", "prob_home": 0.6, "correct": True}])  # pre-Faz 5 log
    card = build_report_card(old, [])
    assert card["side"]["n"] == 1 and card["handicap"]["calibration"]["n"] == 0


def test_market_vs_model_live():
    from report_card import market_vs_model

    rng = np.random.default_rng(1)
    n = 200
    truth = rng.normal(0, 7, n)
    market = truth + rng.normal(0, 1.5, n)            # the market is closer to the truth
    model = truth + rng.normal(0, 4, n)
    margin = np.round(truth + rng.normal(0, 12, n))
    margin[margin == 0] = 1
    total = np.round(225 + rng.normal(0, 18, n))
    log = pd.DataFrame({
        "game_id": [f"G{i}" for i in range(n)],
        "prob_home": 1 / (1 + np.exp(-model / 6)), "prob_home_market": 1 / (1 + np.exp(-market / 6)),
        "model_margin_center": model, "market_margin": market,
        "model_total_center": 225 + rng.normal(0, 6, n), "market_total": np.full(n, 225.0),
        "actual_home_pts": (total + margin) / 2, "actual_away_pts": (total - margin) / 2,
    })
    log.loc[:9, "market_margin"] = np.nan                # games without a line are skipped
    r = market_vs_model(log)
    assert r["margin"]["n"] == n - 10 and r["total"]["n"] == n
    assert r["margin"]["market_mae"] < r["margin"]["model_mae"]
    assert 0 <= r["margin"]["model_closer"] <= 1 and "w" in r["margin"]
    assert r["winner"]["n"] == n and r["winner"]["market_logloss"] < 0.7

    few = market_vs_model(log.iloc[:20])
    assert few["margin"]["n"] == 10 and "w" not in few["margin"]   # too few games for w
    assert market_vs_model(pd.DataFrame({"game_id": ["X"]}))["margin"] == {"n": 0}
    assert build_report_card(log, [])["market"]["margin"]["n"] == n - 10
