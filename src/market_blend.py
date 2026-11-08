"""
market_blend.py
---------------
The handicap / total lines the app shows for one game.

When the market has a line (data/processed/market_lines.parquet, written by
05 through espn_odds), the score distribution is centred on it and uses the
closing line's own error distribution (models/market_distribution.json,
from 09_market_eval.py):

    centre = market + w · (model − market)

w is the model's blend weight measured against closing lines; it is 0
unless the model clearly added information there, in which case the model
only nudges the market. Without a market line (or before 09 has been run)
the model's own lines are used, as before.

Both stay visible: `lines` is what the app prices with, `model` keeps the
model's own centre and fair line for comparison.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from score_dist import game_lines

ROOT = Path(__file__).parent.parent
MARKET_DIST_PATH = ROOT / "models" / "market_distribution.json"

_MARKET_COLS = {"margin": "market_margin", "total": "total"}


def load_market_distribution(path: Path = MARKET_DIST_PATH) -> dict | None:
    if not path.exists():
        return None
    dist = json.loads(path.read_text(encoding="utf-8"))
    return dist if "margin" in dist or "total" in dist else None


def _value(v) -> float | None:
    """None for missing values (None or NaN from the parquet table)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def blend_weight(market_dist: dict | None, task: str) -> float:
    return float(((market_dist or {}).get("blend") or {}).get(task, {}).get("w", 0.0))


def build_lines(model_dists: dict, reg_pred: dict, market: dict | None,
                market_dist: dict | None) -> tuple[dict | None, dict | None, dict | None]:
    """
    Returns (lines, model_lines, market_info):
      lines        game_lines() output the app prices with (market-centred
                   where possible, per market: margin and total separately)
      model_lines  the model's own game_lines() (None without regressors)
      market_info  what the market said and how the lines were built, None
                   when there is no market line for the game
    """
    model_lines = game_lines(model_dists or {}, reg_pred.get("margin"), reg_pred.get("total")) \
        if reg_pred else None

    dists, centres, source = {}, {}, {}
    for task, col in _MARKET_COLS.items():
        line = _value((market or {}).get(col))
        model_part = (model_lines or {}).get(task)
        if line is not None and market_dist and task in market_dist:
            w = blend_weight(market_dist, task)
            model_c = model_part["pred"] if model_part else None
            dists[task] = market_dist[task]
            centres[task] = line + w * (model_c - line) if model_c is not None else line
            source[task] = "market"
        elif model_part and task in (model_dists or {}):
            dists[task] = model_dists[task]
            centres[task] = reg_pred[task]
            source[task] = "model"

    lines = game_lines(dists, centres.get("margin"), centres.get("total")) if dists else None

    info = None
    if market and (_value(market.get("market_margin")) is not None or _value(market.get("total")) is not None):
        m_margin = (model_lines or {}).get("margin") or {}
        m_total = (model_lines or {}).get("total") or {}
        p_home = _value(market.get("p_home_novig"))
        if p_home is None and source.get("margin") == "market":
            p_home = lines["margin"]["p_home_win"]
        info = {
            "provider":      market.get("provider") or "",
            "spread_home":   _value(market.get("spread_home")),
            "market_margin": _value(market.get("market_margin")),
            "market_total":  _value(market.get("total")),
            "p_home":        p_home,
            "source":        source,
            "w":             {t: blend_weight(market_dist, t) for t in _MARKET_COLS},
            "model_margin":  m_margin.get("pred"),
            "model_total":   m_total.get("pred"),
            "model_fair_margin": m_margin.get("fair_line"),
            "model_fair_total":  m_total.get("fair_line"),
        }
    return lines, model_lines, info
