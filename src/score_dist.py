"""
score_dist.py
-------------
Turns a point prediction of the margin (home - away) or the total
(home + away) into a probability distribution over integer final scores,
so any betting line can be priced from one model:

    P(margin > 12.5), P(away wins by 11-15), P(total > 240.5), ...

The error distribution is learned from the walk-forward out-of-fold
residuals of 04_evaluate_model.py and saved in models/distribution.json:

  sigma  spread of the error: constant, or a linear function of the
         prediction when the residual spread clearly depends on it
  shape  "normal", or "kde" (smoothed empirical shape of the standardized
         residuals); 04 picks whichever is better calibrated on the holdout
  no_tie NBA games cannot end level, so a margin of 0 gets no probability

A "line" is a threshold on the target itself: over = target > line. For the
margin, line 12.5 means "home wins by 13+" and line -12.5 means "away wins
by 12 or less, or home wins".
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

# Kazanma farkı bantları (her iki taraf için), son bant üstü açık
MARGIN_BANDS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, None)]
# Barem merdiveni: adil çizginin etrafındaki sayı farkları
LADDER_OFFSETS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)

HETERO_MIN_RATIO = 1.10   # grup sapmalarına oturan doğru uçtan uca bu kadar değişmiyorsa sabit sigma
HETERO_MIN_T = 3.0        # ... ya da eğimi standart hatasının 3 katı değilse (gürültü)
N_SIGMA_BINS = 5
N_QUANTILES = 201
_SUPPORT_SIGMAS = 8       # tahminin ±8 sigma'sı kadar tam sayı skor değerlendirilir
_CDF_GRID = np.linspace(-12, 12, 4801)


# ---------------------------------------------------------------------------
# Fitting (04_evaluate_model.py)
# ---------------------------------------------------------------------------
def _scale_input(task_or_on: str, pred):
    pred = np.asarray(pred, dtype=float)
    return np.abs(pred) if task_or_on in ("margin", "abs_pred") else pred


def fit_distribution(task: str, resid, pred) -> dict:
    """
    Error distribution from out-of-fold residuals (actual - predicted).
    Margin spread is checked against |prediction| (blowout vs close game),
    total spread against the prediction itself (high vs low scoring game).
    """
    resid = np.asarray(resid, dtype=float)
    pred = np.asarray(pred, dtype=float)
    on = "abs_pred" if task == "margin" else "pred"
    x = _scale_input(on, pred)

    # Grup bazında hata büyüklüğü (RMS: sistematik sapma da belirsizliğe dahil)
    edges = np.quantile(x, np.linspace(0, 1, N_SIGMA_BINS + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_SIGMA_BINS - 1)
    bins = []
    for b in range(N_SIGMA_BINS):
        m = idx == b
        bins.append({
            "x_mean": float(x[m].mean()),
            "rms": float(np.sqrt(np.mean(resid[m] ** 2))),
            "n": int(m.sum()),
        })
    rms = np.array([b["rms"] for b in bins])
    xs = np.array([b["x_mean"] for b in bins])
    w = np.array([b["n"] for b in bins], dtype=float)

    # Gruplara ağırlıklı doğru oturt (RMS'in standart hatası ≈ rms / sqrt(2n)).
    # Sigma ancak doğru uçtan uca belirgin değişiyorsa VE eğim gürültüden
    # açıkça büyükse tahmine bağlı olur; küçük örnekte rastgele dalgalanma
    # tek başına oranı 1.2'ye çıkarabilir.
    se = rms / np.sqrt(2 * w)
    slope, intercept = np.polyfit(xs, rms, 1, w=1 / se)
    wt = 1 / se ** 2
    x_bar = np.sum(wt * xs) / wt.sum()
    slope_t = float(abs(slope) * np.sqrt(np.sum(wt * (xs - x_bar) ** 2)))
    fitted = intercept + slope * xs
    ratio = float(fitted.max() / fitted.min()) if fitted.min() > 0 else np.inf

    overall = float(np.sqrt(np.mean(resid ** 2)))
    if ratio >= HETERO_MIN_RATIO and slope_t >= HETERO_MIN_T:
        sigma = {
            "kind": "linear", "on": on,
            "a": float(intercept), "b": float(slope),
            "min": float(rms.min() * 0.9), "max": float(rms.max() * 1.1),
        }
    else:
        sigma = {"kind": "const", "value": overall}

    params = {
        "task":   task,
        "shape":  "normal",
        "sigma":  sigma,
        "no_tie": task == "margin",
        "diagnostics": {
            "n": int(len(resid)),
            "bias": float(resid.mean()),        # > 0: model az tahmin ediyor
            "rms": overall,
            "spread_ratio": ratio,
            "slope_t": slope_t,
            "bins": bins,
        },
    }

    # Ampirik şekil: standartlaştırılmış hataların kantilleri + Gauss çekirdek.
    # Çekirdek varyansı ekler; kantiller küçültülerek toplam varyans korunur.
    z = resid / sigma_for(params, pred)
    z_std = float(z.std())
    h = 1.06 * z_std * len(z) ** -0.2
    shrink = 1 / np.sqrt(1 + (h / z_std) ** 2)
    zq = np.quantile(z, np.linspace(0, 1, N_QUANTILES)) * shrink
    params["z"] = [round(float(v), 4) for v in zq]
    params["h"] = float(h * shrink)
    return params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def sigma_for(params: dict, mu) -> np.ndarray:
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    s = params["sigma"]
    if s["kind"] == "const":
        return np.full(mu.shape, s["value"])
    x = _scale_input(s["on"], mu)
    return np.clip(s["a"] + s["b"] * x, s["min"], s["max"])


def _std_cdf(params: dict, z: np.ndarray) -> np.ndarray:
    """CDF of the standardized error."""
    if params["shape"] == "normal":
        return ndtr(z)
    zq = np.asarray(params["z"])
    table = ndtr((_CDF_GRID[:, None] - zq[None, :]) / params["h"]).mean(axis=1)
    return np.interp(z, _CDF_GRID, table, left=0.0, right=1.0)


def score_pmf(params: dict, mu) -> tuple[np.ndarray, np.ndarray]:
    """
    (k, p): for each prediction (rows) the integer outcomes k and their
    probabilities p, rows summing to 1.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sig = sigma_for(params, mu)
    r = int(np.ceil(_SUPPORT_SIGMAS * sig.max()))
    k = np.round(mu)[:, None] + np.arange(-r, r + 1)[None, :]
    upper = _std_cdf(params, (k + 0.5 - mu[:, None]) / sig[:, None])
    lower = _std_cdf(params, (k - 0.5 - mu[:, None]) / sig[:, None])
    p = np.clip(upper - lower, 0.0, None)
    if params.get("no_tie"):
        p[k == 0] = 0.0
    p /= p.sum(axis=1, keepdims=True)
    return k, p


def prob_over(params: dict, mu, line) -> np.ndarray:
    """P(target > line) for each prediction; line is a scalar or one per row."""
    k, p = score_pmf(params, mu)
    line = np.broadcast_to(np.asarray(line, dtype=float), (k.shape[0],))
    return (p * (k > line[:, None])).sum(axis=1)


def prob_between(params: dict, mu, lo: float | None, hi: float | None) -> np.ndarray:
    """P(lo <= target <= hi) for integer bounds, either side open when None."""
    k, p = score_pmf(params, mu)
    mask = np.ones_like(k, dtype=bool)
    if lo is not None:
        mask &= k >= lo
    if hi is not None:
        mask &= k <= hi
    return (p * mask).sum(axis=1)


def fair_line(params: dict, mu) -> np.ndarray:
    """Half-point line where over and under are closest to 50/50."""
    k, p = score_pmf(params, mu)
    p_over = 1 - np.cumsum(p, axis=1)           # P(target > k + 0.5)
    i = np.abs(p_over - 0.5).argmin(axis=1)
    return k[np.arange(len(k)), i] + 0.5


def ladder_lines(mu) -> np.ndarray:
    """Half-point lines around each prediction (rows) at LADDER_OFFSETS."""
    base = np.floor(np.atleast_1d(np.asarray(mu, dtype=float))) + 0.5
    return base[:, None] + np.asarray(LADDER_OFFSETS, dtype=float)[None, :]


def ladder_eval(params: dict, y, mu) -> dict:
    """
    How well the line probabilities match reality: for every game, the lines
    around its prediction are priced and compared with the actual result.
    """
    from modeling import calibration_table

    y = np.asarray(y, dtype=float)
    lines = ladder_lines(mu)
    k, p = score_pmf(params, mu)
    probs = np.stack(
        [(p * (k > lines[:, j:j + 1])).sum(axis=1) for j in range(lines.shape[1])], axis=1
    )
    hits = (y[:, None] > lines).astype(float)
    ece, rows = calibration_table(hits.ravel(), probs.ravel())
    by_offset = [
        {"offset": int(o), "mean_pred": float(probs[:, j].mean()), "frac_over": float(hits[:, j].mean())}
        for j, o in enumerate(LADDER_OFFSETS)
    ]
    return {
        "brier": float(np.mean((probs - hits) ** 2)),
        "ece": float(ece),
        "calibration": rows,
        "by_offset": by_offset,
        "n_games": int(len(y)),
    }
