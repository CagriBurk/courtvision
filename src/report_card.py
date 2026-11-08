"""
report_card.py
--------------
The "karne" (Faz 6): how often what the model promised actually happened.

  side      winner picks (pre-tip predictions)
  handicap  every priced handicap line of every game (9 per game) + the
            winning-margin bands
  total     every priced over/under line (tracked only; not used in coupons)
  coupons   by tier: hit rate vs promised probability, legs, optional profit
  market    live model vs market (closing line) on the same games (6b)

Calibration rows answer "of the picks we gave ~70%, how many came in?".
Every line is read from the side the model favours (p >= 50%), so a 30%
over is counted as a 70% under. Each row carries a 95% Wilson interval: with
a few dozen picks the true rate can easily be ±10 points away.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.0001]


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def calibration(pairs: list[tuple[float, bool]]) -> dict:
    """(promised p ≥ 0.5, came in) pairs → bucket table + overall summary."""
    if not pairs:
        return {"n": 0, "rows": [], "mean_pred": None, "hit_rate": None}
    p = np.array([x[0] for x in pairs], dtype=float)
    hit = np.array([x[1] for x in pairs], dtype=float)
    rows = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            continue
        k = int(hit[m].sum())
        ci = wilson(k, n)
        rows.append({
            "lo": lo, "hi": min(hi, 1.0), "n": n,
            "mean_pred": float(p[m].mean()), "hit_rate": k / n,
            "ci_lo": ci[0], "ci_hi": ci[1],
        })
    return {"n": int(len(p)), "rows": rows, "mean_pred": float(p.mean()), "hit_rate": float(hit.mean())}


def _favoured(p_over: float, over: bool) -> tuple[float, bool]:
    """Read a line from the side with p >= 0.5."""
    return (p_over, over) if p_over >= 0.5 else (1 - p_over, not over)


# ---------------------------------------------------------------------------
def _settled(log: pd.DataFrame) -> pd.DataFrame:
    if "actual_home_pts" not in log.columns:
        return log.iloc[0:0]
    return log[log["actual_home_pts"].notna() & log["actual_away_pts"].notna()]


def side_report(log: pd.DataFrame) -> dict:
    done = log[log["correct"].notna()]
    pairs = [(max(p, 1 - p), bool(c)) for p, c in zip(done["prob_home"].astype(float), done["correct"])]
    return calibration(pairs)


def line_pairs(log: pd.DataFrame, task: str) -> list[tuple[float, bool]]:
    pairs = []
    done = _settled(log)
    if "lines_json" not in done.columns:
        return pairs
    for js, h, a in zip(done["lines_json"], done["actual_home_pts"], done["actual_away_pts"]):
        if not isinstance(js, str):
            continue
        part = json.loads(js).get(task)
        if not part:
            continue
        value = h - a if task == "margin" else h + a
        for rung in part.get("ladder", []):
            pairs.append(_favoured(rung["p_over"], value > rung["line"]))
    return pairs


def band_report(log: pd.DataFrame) -> list[dict]:
    """Promised vs observed share of each winning-margin band (both sides)."""
    done = _settled(log)
    acc: dict[tuple, list] = {}
    if "lines_json" not in done.columns:
        return []
    for js, h, a in zip(done["lines_json"], done["actual_home_pts"], done["actual_away_pts"]):
        if not isinstance(js, str):
            continue
        margin = json.loads(js).get("margin")
        if not margin:
            continue
        m = h - a
        for b in margin.get("bands", []):
            lo, hi = b["lo"], b["hi"]
            win = m if b["side"] == "home" else -m
            came = win >= lo and (hi is None or win <= hi)
            acc.setdefault((b["side"], lo, hi), []).append((b["p"], came))
    out = []
    for (side, lo, hi), xs in acc.items():
        n = len(xs)
        out.append({"side": side, "lo": lo, "hi": hi, "n": n,
                    "pred": float(np.mean([x[0] for x in xs])),
                    "obs": float(np.mean([x[1] for x in xs]))})
    out.sort(key=lambda r: (r["side"] != "home", r["lo"]))
    return out


def coupon_report(coupons: list[dict]) -> dict:
    settled = [c for c in coupons if c.get("result")]
    tiers = []
    for key, label in (("guvenli", "Güvenli"), ("dengeli", "Dengeli"), ("riskli", "Riskli"),
                       ("ust", "Üst (deneysel)")):
        cs = [c for c in settled if c["tier"] == key]
        won = sum(c["result"] == "won" for c in cs)
        legs = [l for c in cs for l in c["legs"] if l.get("won") is not None]
        played = [c for c in cs if c.get("played_odds")]
        profit = sum((c["played_odds"] - 1) if c["result"] == "won" else -1 for c in played)
        tiers.append({
            "tier": key, "label": label, "n": len(cs), "won": won,
            "hit_rate": won / len(cs) if cs else None,
            "expected": float(np.mean([c["p"] for c in cs])) if cs else None,
            "legs": calibration([(l["p"], bool(l["won"])) for l in legs]),
            "n_played": len(played),
            "profit": round(profit, 2) if played else None,
            "roi": round(profit / len(played), 4) if played else None,
        })
    played = [c for c in settled if c.get("played_odds")]
    total_profit = sum((c["played_odds"] - 1) if c["result"] == "won" else -1 for c in played)
    return {
        "n": len(settled),
        "pending": sum(1 for c in coupons if not c.get("result")),
        "tiers": tiers,
        "played": {
            "n": len(played),
            "won": sum(c["result"] == "won" for c in played),
            "profit": round(total_profit, 2) if played else None,
            "roi": round(total_profit / len(played), 4) if played else None,
        },
    }


MIN_GAMES_FOR_W = 30   # canlı w ancak bu kadar maçtan sonra anlamlı


def market_vs_model(log: pd.DataFrame) -> dict:
    """
    Live head-to-head on finished games that had a market line (6b): the
    model's own centre vs the market's line (margin, total) and the model's
    win probability vs the no-vig market one. `w` is the live blend weight
    (actual ≈ market + w·(model − market)), the number 09 measured on the
    past; it only means something after a few hundred games.
    """
    from market_eval import blend_weight, log_loss

    done = _settled(log)
    out = {}
    for task, model_col, market_col in (("margin", "model_margin_center", "market_margin"),
                                         ("total", "model_total_center", "market_total")):
        if model_col not in done.columns or market_col not in done.columns:
            out[task] = {"n": 0}
            continue
        g = done[done[model_col].notna() & done[market_col].notna()]
        y = (g["actual_home_pts"] - g["actual_away_pts"] if task == "margin"
             else g["actual_home_pts"] + g["actual_away_pts"]).astype(float)
        model, market = g[model_col].astype(float), g[market_col].astype(float)
        row = {"n": int(len(g))}
        if len(g):
            row["model_mae"] = float((y - model).abs().mean())
            row["market_mae"] = float((y - market).abs().mean())
            row["model_closer"] = float(((y - model).abs() < (y - market).abs()).mean())
        if len(g) >= MIN_GAMES_FOR_W:
            w = blend_weight(y, market, model)
            row["w"], row["w_se"] = w["w"], w["se"]
        out[task] = row

    win = {"n": 0}
    if "prob_home_market" in done.columns:
        g = done[done["prob_home_market"].notna() & done["prob_home"].notna()]
        if len(g):
            y = (g["actual_home_pts"] > g["actual_away_pts"]).astype(float)
            pm, pmod = g["prob_home_market"].astype(float), g["prob_home"].astype(float)
            win = {
                "n": int(len(g)),
                "model_logloss": log_loss(y, pmod), "market_logloss": log_loss(y, pm),
                "model_acc": float(((pmod > 0.5) == (y > 0.5)).mean()),
                "market_acc": float(((pm > 0.5) == (y > 0.5)).mean()),
            }
    out["winner"] = win
    return out


def build_report_card(pred_log: pd.DataFrame | None, coupons: list[dict]) -> dict:
    """Everything the Karne page shows."""
    from prediction_tracker import line_report

    log = pred_log if pred_log is not None else pd.DataFrame()
    has = not log.empty
    errors = line_report(log) if has else {}
    return {
        "side": side_report(log) if has and "correct" in log.columns else calibration([]),
        "handicap": {
            "calibration": calibration(line_pairs(log, "margin")) if has else calibration([]),
            "bands": band_report(log) if has else [],
            "error": errors.get("margin"),
        },
        "total": {
            "calibration": calibration(line_pairs(log, "total")) if has else calibration([]),
            "error": errors.get("total"),
        },
        "coupons": coupon_report(coupons),
        "market": market_vs_model(log) if has else market_vs_model(pd.DataFrame()),
    }
