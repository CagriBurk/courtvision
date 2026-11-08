import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import modeling
from score_dist import (
    MARGIN_BANDS,
    center,
    crps,
    fair_line,
    fit_mean_map,
    fit_distribution,
    ladder_eval,
    prob_between,
    prob_over,
    score_pmf,
)
from test_training import make_feature_table

evaluate_module = importlib.import_module("04_evaluate_model")


def const_dist(task, sigma, shape="normal"):
    return {"task": task, "shape": shape, "sigma": {"kind": "const", "value": sigma},
            "no_tie": task == "margin"}


def test_pmf_sums_to_one_and_margin_has_no_ties():
    k, p = score_pmf(const_dist("margin", 13.0), [2.0, -7.5, 0.0])
    assert np.allclose(p.sum(axis=1), 1)
    assert (p[k == 0] == 0).all()


def test_total_probabilities_match_the_normal_distribution():
    d = const_dist("total", 18.0)
    # Line exactly at the prediction: 50/50
    assert prob_over(d, 230.5, 230.5)[0] == pytest.approx(0.5, abs=1e-9)
    # Half-point lines on integer scores equal the continuous normal tail
    assert prob_over(d, 226.0, 240.5)[0] == pytest.approx(1 - ndtr(14.5 / 18), abs=1e-6)
    lines = np.arange(200.5, 260.5, 5)
    p = np.array([prob_over(d, 230.0, line)[0] for line in lines])
    assert (np.diff(p) < 0).all()
    assert fair_line(d, [225.3])[0] in (224.5, 225.5)


def test_margin_symmetry_and_bands():
    d = const_dist("margin", 13.0)
    # Even game: each side 50%, and -0.5 / +0.5 are the same bet (no ties)
    assert prob_over(d, 0.0, 0.5)[0] == pytest.approx(0.5)
    assert prob_over(d, 0.0, -0.5)[0] == pytest.approx(prob_over(d, 0.0, 0.5)[0])

    mu = -9.0  # away team favored by 9
    home = [prob_between(d, mu, lo, hi)[0] for lo, hi in MARGIN_BANDS]
    away = [prob_between(d, mu, None if hi is None else -hi, -lo)[0] for lo, hi in MARGIN_BANDS]
    assert sum(home) + sum(away) == pytest.approx(1)
    assert sum(away) == pytest.approx(1 - prob_over(d, mu, 0.5)[0])
    # Away by 10-15 is likelier than home by 10-15 when away is favored
    assert prob_between(d, mu, -15, -10)[0] > prob_between(d, mu, 10, 15)[0]
    # "away -12.5" = margin below -12.5
    assert 1 - prob_over(d, mu, -12.5)[0] == pytest.approx(prob_between(d, mu, None, -13)[0])


def test_fit_constant_sigma_and_empirical_shape():
    rng = np.random.default_rng(1)
    pred = rng.normal(225, 8, 20000)
    resid = rng.normal(0, 18, 20000)
    d = fit_distribution("total", resid, pred)
    assert d["sigma"]["kind"] == "const"
    assert d["sigma"]["value"] == pytest.approx(18, rel=0.03)
    # Empirical shape of normal errors ≈ normal
    kde = {**d, "shape": "kde"}
    for line in (215.5, 225.5, 240.5):
        assert prob_over(kde, 225.0, line)[0] == pytest.approx(prob_over(d, 225.0, line)[0], abs=0.015)


def test_noisy_small_sample_keeps_constant_sigma():
    """One noisy bin must not make sigma depend on the prediction."""
    rng = np.random.default_rng(4)
    pred = rng.normal(0, 7, 300)
    resid = rng.normal(0, 13, 300)
    resid[np.argsort(np.abs(pred))[:60]] *= 1.2  # the closest games look noisier
    resid[np.argsort(np.abs(pred))[-60:]] *= 1.2  # ... and so do the blowouts
    assert fit_distribution("margin", resid, pred)["sigma"]["kind"] == "const"


def test_fit_sigma_that_grows_with_the_prediction():
    rng = np.random.default_rng(2)
    pred = rng.uniform(200, 250, 20000)
    resid = rng.normal(0, 1, 20000) * (10 + 0.2 * (pred - 200))  # 10 → 20
    d = fit_distribution("total", resid, pred)
    assert d["sigma"]["kind"] == "linear" and d["sigma"]["on"] == "pred"
    from score_dist import sigma_for
    lo, hi = sigma_for(d, [205.0, 245.0])
    assert lo == pytest.approx(11, rel=0.1) and hi == pytest.approx(19, rel=0.1)


def test_t_shape_fits_heavy_tails_and_keeps_variance():
    rng = np.random.default_rng(5)
    resid = 12 * rng.standard_t(4, 30000) / np.sqrt(2)  # unit-variance t(4) scaled to sd 12
    d = fit_distribution("margin", resid, np.zeros_like(resid))
    assert d["t_df"] <= 6
    t = {**d, "shape": "t", "no_tie": False}
    k, p = score_pmf(t, [0.0])
    sd = np.sqrt((p * k ** 2).sum())
    assert sd == pytest.approx(d["sigma"]["value"], rel=0.03)
    # Heavier tails than the normal with the same spread, fewer games near the center
    n = {**d, "shape": "normal", "no_tie": False}
    assert prob_over(t, 0.0, 30.5)[0] > prob_over(n, 0.0, 30.5)[0]
    assert prob_between(t, 0.0, -3, 3)[0] > prob_between(n, 0.0, -3, 3)[0]  # peakier center
    # ...and they describe this data better
    y = np.round(resid[:3000])
    assert crps(t, y, np.zeros(3000)) <= crps(n, y, np.zeros(3000)) + 1e-3


def test_mean_map_stretches_compressed_predictions():
    rng = np.random.default_rng(6)
    truth = rng.normal(0, 9, 20000)
    y = np.round(truth + rng.normal(0, 12, 20000))
    pred = 0.7 * truth  # model shrinks toward 0
    mm = fit_mean_map(y, pred)
    assert mm["b"] == pytest.approx(1 / 0.7, rel=0.08)
    raw = fit_distribution("margin", y - pred, pred)
    c = center({"mean_map": mm}, pred)
    fixed = fit_distribution("margin", y - c, c, mm)
    assert crps(fixed, y, pred) < crps(raw, y, pred)
    # The stored correction is applied at use time: a strong favorite gets a bigger margin
    assert fair_line(fixed, [10.0])[0] > fair_line(raw, [10.0])[0]


def test_ladder_is_calibrated_when_the_distribution_is_right():
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 7, 4000)
    y = np.round(mu + rng.normal(0, 13, 4000))
    y[y == 0] = 1
    ev = ladder_eval(const_dist("margin", 13.0), y, mu)
    assert ev["ece"] < 0.02
    wrong = ladder_eval(const_dist("margin", 6.0), y, mu)  # far too confident
    assert wrong["brier"] > ev["brier"]


def test_evaluate_regression_and_win_source(tmp_path, monkeypatch):
    monkeypatch.setitem(modeling.COMMON_PARAMS, "device", "cpu")
    df = make_feature_table(n_per_season=200, seasons=("22021", "22022", "22023", "22024"))
    feat_cols = ["x", "noise"]
    monkeypatch.setattr(evaluate_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(evaluate_module, "DIST_PATH", tmp_path / "distribution.json")
    monkeypatch.setattr(evaluate_module, "load_feature_table", lambda: (df.copy(), feat_cols))
    with open(tmp_path / "feature_columns.pkl", "wb") as f:
        pickle.dump(feat_cols, f)
    for task in ("margin", "total"):
        meta = {"holdout_season": "22024", "best_params": {"n_estimators": 50, "max_depth": 2}}
        (tmp_path / modeling.TASKS[task]["meta_file"]).write_text(json.dumps(meta))

    out = evaluate_module.evaluate_regression("margin")
    evaluate_module.evaluate_regression("total")

    dists = json.loads((tmp_path / "distribution.json").read_text())
    assert set(dists) == {"margin", "total"}
    assert dists["margin"]["no_tie"] and not dists["total"]["no_tie"]
    m = json.loads((tmp_path / "metrics_margin.json").read_text())
    assert m["headline"] == "holdout" and m["n_holdout_games"] == 200
    assert len(m["candidates"]) == 12  # 2 windows x 2 mean options x 3 shapes
    best = min(m["candidates"], key=lambda c: c["crps"])
    assert m["ladder"]["chosen"] == best["name"] == dists["margin"]["selected"]
    assert dists["margin"]["shape"] == best["shape"]
    assert len(m["bands"]) == 12 and "bands" in json.loads((tmp_path / "metrics_total.json").read_text())
    assert "description" in m["baseline"]

    # Fake classifier that knows nothing: the margin model must take over
    holdout = df[df["season_id"] == "22024"]
    win = {"game_ids": holdout["game_id"].values, "y_true": holdout["target"].values,
           "y_prob": np.full(len(holdout), 0.5)}
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))
    evaluate_module.compare_win_sources(win, out)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["win_source"] == "margin"
    assert metrics["accuracy"] == 0.5  # existing win metrics are kept


def test_team_score_ranges():
    from score_dist import game_lines, team_score_dist

    s = team_score_dist(110.4, 11.6)
    assert sum(b["p"] for b in s["bins"]) > 0.98  # ±2.5 sd covered
    assert max(b["p"] for b in s["bins"]) < 0.18  # no 5-point bin is likely
    best = s["best10"]
    assert best["hi"] - best["lo"] == 9 and best["lo"] <= 110.4 <= best["hi"]
    assert 0.3 < best["p"] < 0.35

    d = {"margin": const_dist("margin", 13.6), "total": const_dist("total", 18.85)}
    g = game_lines(d, -12.0, 228.0)
    assert g["scores"]["away"]["mean"] == pytest.approx(120.0)
    assert g["scores"]["home"]["sd"] == pytest.approx(np.hypot(13.6, 18.85) / 2, abs=0.01)
