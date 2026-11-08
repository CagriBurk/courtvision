import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_eval import (
    BREAK_EVEN_110,
    apply_prob_blend,
    ats_table,
    blend_weight,
    blend_weight_for,
    early_season_mask,
    fit_market_distribution,
    fit_prob_blend,
    line_section,
    log_loss,
)


def synthetic(w_true, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 7, n)                    # what both try to estimate
    market = truth + rng.normal(0, 1.5, n)
    model = truth + rng.normal(0, 4, n)
    # actual follows the chosen mix of market and model, plus game noise
    y = market + w_true * (model - market) + rng.normal(0, 11, n)
    return y, market, model


@pytest.mark.parametrize("w_true", [0.0, 0.3])
def test_blend_weight_recovers_w(w_true):
    y, m, p = synthetic(w_true)
    r = blend_weight(y, m, p)
    assert r["w"] == pytest.approx(w_true, abs=3 * r["se"])
    assert 0 < r["se"] < 0.05
    assert r["ci95"][0] < r["w"] < r["ci95"][1]


def test_blend_weight_degenerate():
    assert np.isnan(blend_weight([1, 2], [0, 0], [0, 0])["w"])
    assert np.isnan(blend_weight([1, 2, 3], [1, 1, 1], [1, 1, 1])["w"])   # model == market


def test_ats_table_counts_wins_losses_pushes():
    market = np.array([-5.5, -5.5, 3.0, 3.0, 0.0])
    model = np.array([-1.0, -9.0, 7.0, 4.0, 0.0])        # last: no side
    y = np.array([-2.0, -3.0, 3.0, 10.0, 4.0])
    # game 1: model above line, result above → win
    # game 2: model below, result above → loss
    # game 3: lands on 3 → push; game 4: above, model above → win
    t0 = ats_table(y, market, model, thresholds=(0, 4))
    assert t0[0]["bets"] == 3 and t0[0]["pushes"] == 1
    assert t0[0]["win_rate"] == pytest.approx(2 / 3)
    assert t0[0]["roi_at_110"] == pytest.approx((2 * 100 / 110 - 1) / 3)
    # |gap| ≥ 4: games 1 (4.5), 2 (3.5 → out), 3 (4.0, push)
    assert t0[1]["bets"] == 1 and t0[1]["pushes"] == 1
    assert t0[1]["win_rate"] == 1.0


def test_ats_break_even():
    assert BREAK_EVEN_110 == pytest.approx(0.5238, abs=1e-4)
    empty = ats_table([1.0], [1.0], [1.0], thresholds=(0,))[0]
    assert empty["bets"] == 0 and np.isnan(empty["win_rate"])


def test_log_loss_and_prob_blend():
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    rng = np.random.default_rng(1)
    n = 20000
    z = rng.normal(0, 1.2, n)
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p_true).astype(float)
    p_market = p_true                                   # market is right
    p_model = 1 / (1 + np.exp(-(z + rng.normal(0, 1, n))))  # model is noisier
    fit = fit_prob_blend(y, p_market, p_model)
    assert fit["a_market"] == pytest.approx(1.0, abs=0.1)
    assert abs(fit["b_model"]) < 3 * fit["se_b_model"] + 0.02
    blended = apply_prob_blend(fit, p_market, p_model)
    assert log_loss(y, blended) <= log_loss(y, p_model)


def test_line_section_out_of_sample_blend():
    y, m, p = synthetic(0.3, n=5000)
    sec = line_section(y, m, p, w_train=0.3)
    assert sec["n"] == 5000
    assert sec["blend_out_of_sample"]["mae"] <= sec["market"]["mae"] + 0.02
    assert "blend_out_of_sample" not in line_section(y, m, p)


def test_blend_weight_needs_oof_and_holdout_evidence():
    def sec(oof_w, oof_se, h_w, h_se):
        return {"oof": {"weight": {"w": oof_w, "se": oof_se}},
                "holdout": {"weight": {"w": h_w, "se": h_se}}}

    assert blend_weight_for(sec(0.35, 0.03, 0.32, 0.11))["w"] == pytest.approx(0.35)
    assert blend_weight_for(sec(0.35, 0.03, 0.15, 0.11))["w"] == 0      # holdout not significant
    assert blend_weight_for(sec(0.05, 0.03, 0.40, 0.10))["w"] == 0      # OOF not significant
    assert blend_weight_for(sec(0.90, 0.05, 0.80, 0.10))["w"] == 0.5    # capped
    assert blend_weight_for({})["w"] == 0


def test_fit_market_distribution_learns_line_error():
    rng = np.random.default_rng(3)
    rows = []
    for season in ("22021", "22022", "22023", "22024", "22025"):
        m = rng.normal(0, 7, 1500)
        t = rng.normal(225, 8, 1500)
        rows.append(pd.DataFrame({
            "season_id": season, "market_margin": m, "market_total": t,
            "y_margin": m + rng.normal(0, 13, 1500), "y_total": t + rng.normal(0, 18, 1500),
        }))
    dist = fit_market_distribution(pd.concat(rows, ignore_index=True), "22025")
    assert dist["fit_seasons"] == ["22022", "22023", "22024"]
    assert dist["margin"]["sigma"]["value"] == pytest.approx(13, rel=0.05)
    assert dist["total"]["sigma"]["value"] == pytest.approx(18, rel=0.05)
    assert dist["margin"]["shape"] in ("normal", "t", "kde") and dist["margin"]["no_tie"]
    assert dist["margin"]["mean_map"] is None
    assert dist["margin"]["selection"]["n_holdout"] == 1500


def test_early_season_mask():
    d = pd.Series(pd.to_datetime(["2025-10-22", "2025-11-30", "2025-12-01", "2026-04-10"]))
    assert early_season_mask(d).tolist() == [True, True, False, False]
