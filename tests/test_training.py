import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import modeling
from modeling import TASKS, cv_loss, make_model, predict_task

train_module = importlib.import_module("03_train_model")


def make_feature_table(n_per_season=120, seasons=("22022", "22023", "22024"), seed=0):
    """Feature table shaped like features.parquet, with targets that depend on x."""
    rng = np.random.default_rng(seed)
    n = n_per_season * len(seasons)
    x = rng.normal(0, 1, n)
    margin = np.round(6 * x + rng.normal(0, 10, n))
    margin[margin == 0] = 1
    total = np.round(220 + 8 * x + rng.normal(0, 15, n))
    return pd.DataFrame({
        "game_id": [f"G{i:04d}" for i in range(n)],
        "game_date": pd.date_range("2022-10-20", periods=n, freq="D"),
        "season_id": np.repeat(seasons, n_per_season),
        "season_type": "Regular Season",
        "target": (margin > 0).astype(int),
        "target_margin": margin,
        "target_total": total,
        "x": x,
        "noise": rng.normal(0, 1, n),
    })


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setitem(modeling.COMMON_PARAMS, "device", "cpu")


def test_task_specs_are_consistent():
    files = [TASKS[t][k] for t in TASKS for k in ("model_file", "study_file", "meta_file", "metrics_file")]
    assert len(files) == len(set(files)), "two tasks would overwrite each other's files"
    # The win model keeps its original file names (05 and the API load them)
    assert TASKS["win"]["model_file"] == "xgb_best.pkl"
    assert TASKS["win"]["meta_file"] == "train_meta.json"
    assert all(TASKS[t]["target"] in modeling.META_COLS for t in TASKS)


def test_model_kinds_and_losses(cpu):
    df = make_feature_table()
    X = df[["x", "noise"]]
    params = {"n_estimators": 20, "max_depth": 2}

    clf = make_model("win", params).fit(X, df["target"])
    p = predict_task("win", clf, X)
    assert ((p > 0) & (p < 1)).all()

    reg = make_model("margin", params).fit(X, df["target_margin"])
    pred = predict_task("margin", reg, X)
    # Regressor predicts points, not probabilities, and beats a constant guess
    assert pred.min() < -1 and pred.max() > 1
    const = np.full(len(df), df["target_margin"].mean())
    assert cv_loss("margin", df["target_margin"], pred) < cv_loss("margin", df["target_margin"], const)

    assert cv_loss("total", [200, 210], [203, 206]) == pytest.approx(np.sqrt((9 + 16) / 2))


@pytest.mark.parametrize("task", ["margin", "total"])
def test_train_task_writes_its_own_files(task, cpu, tmp_path, monkeypatch):
    df = make_feature_table()
    feat_cols = ["x", "noise"]
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "load_feature_table", lambda: (df.copy(), feat_cols))

    train_module.train_task(task, n_trials=2, holdout_season="22024")

    spec = TASKS[task]
    meta = json.loads((tmp_path / spec["meta_file"]).read_text(encoding="utf-8"))
    assert meta["task"] == task and meta["target"] == spec["target"]
    assert meta["n_dev_games"] == 240 and meta["holdout_season"] == "22024"
    assert meta["cv_metric"] == "RMSE"
    assert not (tmp_path / "xgb_best.pkl").exists()  # the win model is untouched

    with open(tmp_path / spec["model_file"], "rb") as f:
        model = pickle.load(f)
    pred = model.predict(df[feat_cols])
    assert abs(pred.mean() - df[spec["target"]].mean()) < 5
