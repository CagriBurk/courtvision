import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_blend import build_lines, load_market_distribution

MODEL_DISTS = {
    "margin": {"task": "margin", "shape": "normal", "sigma": {"kind": "const", "value": 13.6},
               "no_tie": True, "mean_map": {"a": 0.0, "b": 1.2}},
    "total": {"task": "total", "shape": "normal", "sigma": {"kind": "const", "value": 18.8}, "no_tie": False},
}
MARKET_DIST = {
    "margin": {"task": "margin", "shape": "normal", "sigma": {"kind": "const", "value": 13.4}, "no_tie": True},
    "total": {"task": "total", "shape": "normal", "sigma": {"kind": "const", "value": 18.0}, "no_tie": False},
    "blend": {"margin": {"w": 0.0}, "total": {"w": 0.0}},
}
REG = {"margin": 10.0, "total": 226.0}            # model: home by 12 after the ×1.2 map
DK = {"spread_home": -7.5, "market_margin": 7.5, "total": 222.5, "p_home_novig": 0.735,
      "provider": "DraftKings"}


def test_market_centres_the_lines_and_keeps_the_model():
    lines, model, info = build_lines(MODEL_DISTS, REG, DK, MARKET_DIST)
    assert lines["margin"]["pred"] == pytest.approx(7.5)
    assert lines["margin"]["sigma"] == pytest.approx(13.4)
    assert lines["total"]["pred"] == pytest.approx(222.5)
    assert model["margin"]["pred"] == pytest.approx(12.0)
    assert info["model_margin"] == pytest.approx(12.0) and info["market_margin"] == 7.5
    assert info["p_home"] == 0.735 and info["provider"] == "DraftKings"
    assert info["source"] == {"margin": "market", "total": "market"}


def test_blend_weight_nudges_towards_the_model():
    dist = {**MARKET_DIST, "blend": {"margin": {"w": 0.2}, "total": {"w": 0.0}}}
    lines, _, info = build_lines(MODEL_DISTS, REG, DK, dist)
    assert lines["margin"]["pred"] == pytest.approx(7.5 + 0.2 * (12.0 - 7.5))
    assert lines["total"]["pred"] == pytest.approx(222.5)
    assert info["w"] == {"margin": 0.2, "total": 0.0}


def test_without_market_distribution_the_model_prices():
    lines, model, info = build_lines(MODEL_DISTS, REG, DK, None)
    assert lines == model
    assert info["source"] == {"margin": "model", "total": "model"}
    assert info["market_margin"] == 7.5          # the market is still shown


def test_no_market_line():
    lines, model, info = build_lines(MODEL_DISTS, REG, None, MARKET_DIST)
    assert lines == model and info is None
    lines, model, info = build_lines(MODEL_DISTS, REG, {}, MARKET_DIST)
    assert lines == model and info is None


def test_missing_parts_and_nan_values():
    # Only a total (NaN spread from the parquet table), no moneyline
    market = {"spread_home": float("nan"), "market_margin": float("nan"), "total": 230.5,
              "p_home_novig": float("nan"), "provider": "DraftKings"}
    lines, _, info = build_lines(MODEL_DISTS, REG, market, MARKET_DIST)
    assert info["source"] == {"margin": "model", "total": "market"}
    assert lines["margin"]["pred"] == pytest.approx(12.0)
    assert lines["total"]["pred"] == pytest.approx(230.5)
    assert info["market_margin"] is None and info["p_home"] is None

    # Spread but no moneyline: the home win probability comes from the market distribution
    market = {**DK, "p_home_novig": None}
    lines, _, info = build_lines(MODEL_DISTS, REG, market, MARKET_DIST)
    assert info["p_home"] == pytest.approx(lines["margin"]["p_home_win"]) and info["p_home"] > 0.6


def test_market_line_without_model_regressors():
    lines, model, info = build_lines(MODEL_DISTS, {}, DK, MARKET_DIST)
    assert model is None
    assert lines["margin"]["pred"] == pytest.approx(7.5)
    assert info["model_margin"] is None


def test_load_market_distribution(tmp_path):
    assert load_market_distribution(tmp_path / "none.json") is None
    p = tmp_path / "m.json"
    p.write_text('{"blend": {}}', encoding="utf-8")
    assert load_market_distribution(p) is None
    p.write_text('{"margin": {"task": "margin"}, "blend": {}}', encoding="utf-8")
    assert load_market_distribution(p)["margin"]["task"] == "margin"
