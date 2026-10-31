"""
modeling.py
-----------
Shared pieces of the training (03), evaluation (04) and experiment (07)
scripts: feature table loading, holdout split, fixed XGBoost params, metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from season import nba_today, season_id, season_start_year

ROOT       = Path(__file__).parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Meta columns not used for modeling
META_COLS = [
    "game_id", "game_date", "season_id", "season_type",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "team_id_away", "team_abbreviation_away", "team_name_away",
    "target",
]

# Fixed XGBoost params (Optuna tunes the rest)
BASE_PARAMS = {
    "objective":    "binary:logistic",
    "eval_metric":  "logloss",
    "tree_method":  "hist",
    "device":       "cuda",
    "random_state": 42,
    "n_jobs":       -1,
}


def load_feature_table() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    df = df.sort_values("game_date").reset_index(drop=True)
    df["season_id"] = df["season_id"].astype(str)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def default_holdout_season() -> str:
    """Last completed season, e.g. '22025' while 2026-27 is being played."""
    return season_id(season_start_year(nba_today()) - 1)


def split_holdout(df: pd.DataFrame, holdout_season: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    dev     = seasons before the holdout (hyper-parameter search + CV)
    holdout = the holdout season, never seen during tuning
    Later seasons (the one in progress) belong to neither; they are only
    used when the production model is refit on all data.
    """
    if holdout_season is None:
        return df, df.iloc[0:0]
    sid = df["season_id"].astype(str)
    return df[sid < holdout_season], df[sid == holdout_season]


def classification_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "accuracy":     float(accuracy_score(y_true, (y_prob > 0.5).astype(int))),
        "auc":          float(roc_auc_score(y_true, y_prob)),
        "logloss":      float(log_loss(y_true, y_prob)),
        "brier":        float(brier_score_loss(y_true, y_prob)),
        "baseline_acc": float(y_true.mean()),  # "always pick the home team"
    }


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(y_true, y_prob) -> dict:
    """
    Platt scaling: p' = sigmoid(a * logit(p) + b). a < 1 shrinks
    over-confident probabilities toward 50%.
    """
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6).fit(_logit(y_prob).reshape(-1, 1), np.asarray(y_true))
    return {"method": "platt", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def apply_calibration(cal: dict | None, y_prob):
    if not cal:
        return np.asarray(y_prob)
    z = cal["a"] * _logit(y_prob) + cal["b"]
    return 1 / (1 + np.exp(-z))


def calibration_table(y_true, y_prob, n_bins: int = 10) -> tuple[float, list[dict]]:
    """
    Expected calibration error + per-bin table. ECE = average gap between
    predicted probability and observed win rate, weighted by bin size.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += n / len(y_true) * abs(mean_pred - frac_pos)
        rows.append({"bin": b, "n": n, "mean_pred": mean_pred, "frac_pos": frac_pos})
    return float(ece), rows
