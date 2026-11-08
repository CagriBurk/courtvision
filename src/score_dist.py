"""
score_dist.py
-------------
Turns a point prediction of the margin (home - away) or the total
(home + away) into a probability distribution over integer final scores,
so any betting line can be priced from one model:

    P(margin > 12.5), P(away wins by 11-15), P(total > 240.5), ...

The error distribution is learned from the walk-forward out-of-fold
residuals of 04_evaluate_model.py and saved in models/distribution.json:

  mean_map  optional straight-line correction of the prediction
            (a + b * pred): regularized trees shrink predictions toward the
            average, which thins the tails (blowouts)
  sigma     spread of the error: constant, or a linear function of the
            prediction when the residual spread clearly depends on it
  shape     "normal", "t" (Student-t, heavier tails) or "kde" (smoothed
            empirical shape of the standardized residuals)
  no_tie    NBA games cannot end level, so a margin of 0 gets no probability

04 builds several candidates (residual window x mean correction x shape)
and keeps the one with the best CRPS on the holdout season.

A "line" is a threshold on the target itself: over = target > line. For the
margin, line 12.5 means "home wins by 13+" and line -12.5 means "away wins
by 12 or less, or home wins".
"""

from __future__ import annotations

import numpy as np
from scipy import stats
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
_T_DF_GRID = (3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100)


# ---------------------------------------------------------------------------
# Fitting (04_evaluate_model.py)
# ---------------------------------------------------------------------------
def _scale_input(task_or_on: str, pred):
    pred = np.asarray(pred, dtype=float)
    return np.abs(pred) if task_or_on in ("margin", "abs_pred") else pred


def fit_mean_map(y, pred) -> dict:
    """actual ≈ a + b * pred on out-of-fold data; b > 1 = predictions too compressed."""
    b, a = np.polyfit(np.asarray(pred, dtype=float), np.asarray(y, dtype=float), 1)
    return {"a": float(a), "b": float(b)}


def center(params: dict, mu) -> np.ndarray:
    """Prediction after the optional mean correction."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    m = params.get("mean_map")
    return mu if not m else m["a"] + m["b"] * mu


def fit_distribution(task: str, resid, pred, mean_map: dict | None = None) -> dict:
    """
    Error distribution from out-of-fold residuals (actual - center), where
    `pred` is the (already corrected) center and `mean_map` the correction
    applied to raw predictions at use time.
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
        if m.sum() < 2:
            continue  # aynı tahminler kantilleri çakıştırabilir
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
    slope = intercept = 0.0
    slope_t, ratio = 0.0, 1.0
    if len(bins) >= 3 and np.ptp(xs) > 0 and (rms > 0).all():
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
        "task":     task,
        "shape":    "normal",
        "mean_map": mean_map,
        "sigma":    sigma,
        "no_tie":   task == "margin",
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

    # Student-t: serbestlik derecesi en yüksek olabilirlikle (birim varyansa ölçekli)
    def loglik(df):
        s = np.sqrt((df - 2) / df)
        return float(np.sum(stats.t.logpdf(z / s, df)) - len(z) * np.log(s))
    params["t_df"] = float(max(_T_DF_GRID, key=loglik))
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
    if params["shape"] == "t":
        df = params["t_df"]
        return stats.t.cdf(z * np.sqrt(df / (df - 2)), df)  # birim varyanslı t
    zq = np.asarray(params["z"])
    table = ndtr((_CDF_GRID[:, None] - zq[None, :]) / params["h"]).mean(axis=1)
    return np.interp(z, _CDF_GRID, table, left=0.0, right=1.0)


def score_pmf(params: dict, mu) -> tuple[np.ndarray, np.ndarray]:
    """
    (k, p): for each prediction (rows) the integer outcomes k and their
    probabilities p, rows summing to 1.
    """
    mu = center(params, mu)
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


def crps(params: dict, y, mu) -> float:
    """
    Mean continuous ranked probability score over integer outcomes (points;
    lower is better). Unlike the ±8 ladder it scores the whole distribution,
    tails included, so it decides between candidate distributions.
    """
    y = np.asarray(y, dtype=float)
    k, p = score_pmf(params, mu)
    cdf = np.cumsum(p, axis=1)
    return float(np.mean(np.sum((cdf - (k >= y[:, None])) ** 2, axis=1)))


def band_table(params: dict, y, mu) -> list[dict]:
    """Predicted vs observed share of each winning-margin band (both sides)."""
    y = np.asarray(y, dtype=float)
    rows = []
    for side, sign in (("home", 1), ("away", -1)):
        for lo, hi in MARGIN_BANDS:
            a, b = (lo, hi) if sign == 1 else ((None if hi is None else -hi), -lo)
            obs = np.ones(len(y), dtype=bool)
            if a is not None:
                obs &= y >= a
            if b is not None:
                obs &= y <= b
            rows.append({"side": side, "lo": lo, "hi": hi, "obs": float(obs.mean()),
                         "pred": float(prob_between(params, mu, a, b).mean())})
    return rows


# ---------------------------------------------------------------------------
# Daily prediction output (05_predict_today.py)
# ---------------------------------------------------------------------------
def _round(v, nd=4):
    return round(float(v), nd)


def game_lines(dists: dict, pred_margin: float | None, pred_total: float | None) -> dict | None:
    """
    Everything the app shows for one game's handicap and total markets.

    margin.ladder[i]: line on the home margin, p_over = P(home margin > line).
      line  6.5 → "Ev -6.5" wins with p_over, "Dep +6.5" with 1 - p_over
      line -4.5 → "Ev +4.5" wins with p_over, "Dep -4.5" with 1 - p_over
    margin.bands: winning-margin bands for each side (MARGIN_BANDS).
    total.ladder[i]: p_over = P(total > line).
    Fair odds are 1 / p; the app computes them.
    """
    out = {}
    if pred_margin is not None and "margin" in dists:
        d = dists["margin"]
        line0 = float(fair_line(d, pred_margin)[0])
        lines = line0 + np.asarray(LADDER_OFFSETS, dtype=float)
        p_over = prob_over(d, np.full(len(lines), pred_margin), lines)
        bands = []
        for side, sign in (("home", 1), ("away", -1)):
            for lo, hi in MARGIN_BANDS:
                if sign == 1:
                    p = prob_between(d, pred_margin, lo, hi)[0]
                else:
                    p = prob_between(d, pred_margin, None if hi is None else -hi, -lo)[0]
                bands.append({"side": side, "lo": lo, "hi": hi, "p": _round(p)})
        out["margin"] = {
            "pred": _round(center(d, pred_margin)[0], 2),
            "sigma": _round(sigma_for(d, center(d, pred_margin))[0], 2),
            "fair_line": line0,
            "p_home_win": _round(prob_over(d, pred_margin, 0.0)[0]),
            "ladder": [{"line": float(l), "p_over": _round(p)} for l, p in zip(lines, p_over)],
            "bands": bands,
        }
    if pred_total is not None and "total" in dists:
        d = dists["total"]
        line0 = float(fair_line(d, pred_total)[0])
        lines = line0 + np.asarray(LADDER_OFFSETS, dtype=float)
        p_over = prob_over(d, np.full(len(lines), pred_total), lines)
        out["total"] = {
            "pred": _round(center(d, pred_total)[0], 2),
            "sigma": _round(sigma_for(d, center(d, pred_total))[0], 2),
            "fair_line": line0,
            "ladder": [{"line": float(l), "p_over": _round(p)} for l, p in zip(lines, p_over)],
        }
    if "margin" in out and "total" in out:
        m, t = out["margin"]["pred"], out["total"]["pred"]
        out["projected"] = {"home": round((t + m) / 2, 1), "away": round((t - m) / 2, 1)}
        # Takım skoru = (toplam ± fark) / 2; hatalar bağımsız kabul edilir
        sd = float(np.hypot(out["margin"]["sigma"], out["total"]["sigma"]) / 2)
        out["scores"] = {
            "home": team_score_dist((t + m) / 2, sd),
            "away": team_score_dist((t - m) / 2, sd),
        }
    return out or None


def team_score_dist(mean: float, sd: float, width: int = 5) -> dict:
    """
    One team's points: 5-point bins around the mean (±2.5 sd) and the most
    likely 10-point window. With sd ≈ 11-12 no 5-point bin gets much more
    than ~17%, a 10-point window ~33%.
    """
    def p_range(lo, hi):  # P(lo <= points <= hi), integer points
        return float(ndtr((hi + 0.5 - mean) / sd) - ndtr((lo - 0.5 - mean) / sd))

    start = int(np.floor((mean - 2.5 * sd) / width) * width)
    stop = int(np.ceil((mean + 2.5 * sd) / width) * width)
    bins = [{"lo": lo, "hi": lo + width - 1, "p": _round(p_range(lo, lo + width - 1))}
            for lo in range(start, stop, width)]
    best_lo = max(range(int(mean) - 15, int(mean) + 6), key=lambda lo: p_range(lo, lo + 9))
    return {
        "mean": round(float(mean), 1),
        "sd": round(sd, 2),
        "bins": bins,
        "best10": {"lo": best_lo, "hi": best_lo + 9, "p": _round(p_range(best_lo, best_lo + 9))},
    }
