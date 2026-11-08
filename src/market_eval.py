"""
market_eval.py
--------------
Model vs market (closing lines) on games the model did not train on.

The question: does the model add information beyond the market? With
x = model − market and d = actual − market, the least-squares weight

    actual ≈ market + w · (model − market)

says how far to move from the market towards the model: w ≈ 0 → the model
adds nothing the market does not already know, w ≈ 1 → ignore the market.

Used by 09_market_eval.py; pure functions so they can be tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BREAK_EVEN_110 = 110 / 210          # win rate needed at -110 (≈ 52.38%)
EDGE_THRESHOLDS = (0, 1, 2, 3, 4, 6, 8)


def blend_weight(y, market, model) -> dict:
    """
    w (no intercept) with its standard error, plus the two-variable fit
    actual = α + β_market·market + β_model·model as a cross-check.
    """
    y, m, p = (np.asarray(v, dtype=float) for v in (y, market, model))
    x, d = p - m, y - m
    sxx = float(np.sum(x * x))
    if len(y) < 3 or sxx <= 0:
        return {"n": int(len(y)), "w": float("nan"), "se": float("nan")}
    w = float(np.sum(x * d) / sxx)
    resid = d - w * x
    se = float(np.sqrt(np.sum(resid ** 2) / (len(y) - 1) / sxx))
    A = np.column_stack([np.ones_like(m), m, p])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {
        "n": int(len(y)), "w": w, "se": se,
        "ci95": [w - 1.96 * se, w + 1.96 * se],
        "alpha": float(coef[0]), "beta_market": float(coef[1]), "beta_model": float(coef[2]),
        "gap_mean_abs": float(np.mean(np.abs(x))),
    }


def error_summary(y, pred) -> dict:
    e = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(-e))}


def ats_table(y, market, model, thresholds=EDGE_THRESHOLDS) -> list[dict]:
    """
    Bet the model's side against the market line whenever |model − market| ≥ t.
    A bet wins when the actual result lands on the model's side of the line;
    exact landings (integer lines) are pushes and excluded from the rate.
    """
    y, m, p = (np.asarray(v, dtype=float) for v in (y, market, model))
    side = np.sign(p - m)
    outcome = np.sign(y - m) * side           # +1 win, -1 loss, 0 push
    rows = []
    for t in thresholds:
        sel = (np.abs(p - m) >= t) & (side != 0)
        wins, losses = int(np.sum(outcome[sel] > 0)), int(np.sum(outcome[sel] < 0))
        n = wins + losses
        rate = wins / n if n else float("nan")
        se = np.sqrt(BREAK_EVEN_110 * (1 - BREAK_EVEN_110) / n) if n else float("nan")
        rows.append({
            "threshold": t, "bets": n, "pushes": int(np.sum(sel) - n),
            "win_rate": rate,
            "z_vs_break_even": (rate - BREAK_EVEN_110) / se if n else float("nan"),
            "roi_at_110": (wins * (100 / 110) - losses) / n if n else float("nan"),
        })
    return rows


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_prob_blend(y, p_market, p_model, iters: int = 50) -> dict:
    """
    logit(p) = c + a·logit(p_market) + b·logit(p_model), fitted by Newton's
    method (plain logistic regression, no regularisation).
    """
    X = np.column_stack([np.ones(len(y)), _logit(p_market), _logit(p_model)])
    y = np.asarray(y, dtype=float)
    beta = np.array([0.0, 1.0, 0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(hess + 1e-9 * np.eye(3), grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(hess)
    return {"c": float(beta[0]), "a_market": float(beta[1]), "b_model": float(beta[2]),
            "se_b_model": float(np.sqrt(cov[2, 2]))}


def apply_prob_blend(fit: dict, p_market, p_model) -> np.ndarray:
    z = fit["c"] + fit["a_market"] * _logit(p_market) + fit["b_model"] * _logit(p_model)
    return 1 / (1 + np.exp(-z))


def line_section(y, market, model, w_train: float | None = None) -> dict:
    """Errors, blend weight and ATS record of one prediction set."""
    y, m, p = (np.asarray(v, dtype=float) for v in (y, market, model))
    out = {
        "n": int(len(y)),
        "market": error_summary(y, m),
        "model": error_summary(y, p),
        "weight": blend_weight(y, m, p),
        "ats": ats_table(y, m, p),
    }
    if w_train is not None and np.isfinite(w_train):
        out["blend_out_of_sample"] = {"w": w_train, **error_summary(y, m + w_train * (p - m))}
    return out


def prob_section(y, p_market, p_model, blend_fit: dict | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    out = {
        "n": int(len(y)),
        "market_logloss": log_loss(y, p_market),
        "model_logloss": log_loss(y, p_model),
        "market_acc": float(np.mean((np.asarray(p_market) > 0.5) == (y > 0.5))),
        "model_acc": float(np.mean((np.asarray(p_model) > 0.5) == (y > 0.5))),
        "fit_in_sample": fit_prob_blend(y, p_market, p_model) if len(y) >= 50 else None,
    }
    if blend_fit is not None:
        out["blend_out_of_sample_logloss"] = log_loss(y, apply_prob_blend(blend_fit, p_market, p_model))
    return out


def early_season_mask(dates: pd.Series) -> np.ndarray:
    """October-November games: teams have played < ~15 games."""
    month = pd.to_datetime(dates).dt.month
    return month.isin([10, 11]).to_numpy()
