"""
03_train_model.py
-----------------
Trains the XGBoost models with Optuna + TimeSeriesSplit.

Tasks (modeling.TASKS):
  win     home win probability (classifier, log-loss)
  margin  home points - away points (regressor, RMSE)
  total   home points + away points (regressor, RMSE)

The last completed season is held out: Optuna and cross-validation only see
earlier seasons, so 04_evaluate_model.py can measure each model on games it
was never tuned on. The production model is then refit on ALL data
(holdout and the season in progress included).

Input : data/processed/features.parquet
Output (per task, file names in modeling.TASKS):
  models/xgb_best.pkl | xgb_margin.pkl | xgb_total.pkl        (trained model)
  models/optuna_study*.pkl                                     (Optuna study)
  models/train_meta.json | train_meta_margin.json | ..._total  (info for API/UI and 04)
  models/feature_columns.pkl   (feature order, shared by all tasks)

Usage:
    python src/03_train_model.py                      # win, 100 trials
    python src/03_train_model.py --task margin
    python src/03_train_model.py --task all           # win + margin + total
    python src/03_train_model.py --trials 20          # quick test
    python src/03_train_model.py --holdout-season 22024
    python src/03_train_model.py --no-holdout
"""

import argparse
import json
import pickle
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from modeling import (
    MODELS_DIR,
    TASKS,
    cv_loss,
    default_holdout_season,
    load_feature_table,
    make_model,
    predict_task,
    split_holdout,
    task_params,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRIALS = 100


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def suggest_params(trial: optuna.Trial, task: str) -> dict:
    # min_child_weight = minimum hessian sum per leaf. Squared error has
    # hessian 1 per game, log-loss about p(1-p) ≈ 0.25, so the same number of
    # games per leaf needs a ~4x larger value for the regressors.
    mcw_max = 20 if TASKS[task]["kind"] == "classifier" else 100
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, mcw_max),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
    }


def make_objective(task: str, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, task)

        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = make_model(task, params)
            model.fit(X_tr, y_tr, verbose=False)
            cv_scores.append(cv_loss(task, y_val, predict_task(task, model, X_val)))

            # Optuna pruning: ilk fold'dan sonra çok kötüyse kes
            trial.report(np.mean(cv_scores), step=len(cv_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    return objective


# ---------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------
def train_task(task: str, n_trials: int, holdout_season: str | None):
    spec = TASKS[task]
    metric = spec["cv_metric"]

    print(f"\n{'='*60}")
    print(f"XGBoost Eğitimi — {spec['label']} ({task}) — Optuna ({n_trials} trial) + TimeSeriesSplit")
    print(f"{'='*60}\n")

    print("[1/4] Features yükleniyor...")
    df, feat_cols = load_feature_table()
    target = spec["target"]
    if target not in df.columns:
        raise SystemExit(
            f"features.parquet içinde '{target}' yok. Önce çalıştır: python src/02_build_features.py"
        )
    df = df[df[target].notna()].reset_index(drop=True)
    dev, holdout = split_holdout(df, holdout_season)
    print(f"  → Toplam maç    : {len(df)} ({df['game_date'].min().date()} → {df['game_date'].max().date()})")
    print(f"  → Feature sayısı: {len(feat_cols)}")
    if holdout_season:
        print(f"  → Holdout sezon : {holdout_season} ({len(holdout)} maç) — aramaya dahil değil")
    print(f"  → Arama verisi  : {len(dev)} maç")

    # ---- Optuna (sadece holdout öncesi sezonlar) ----
    # 5 fold, sıralı split. gap=0 çünkü shift(1) zaten koruma sağlıyor.
    tscv = TimeSeriesSplit(n_splits=5, gap=0)

    print("\n[2/4] Optuna hiper-parametre araması başlıyor...")
    print(f"  Trials: {n_trials} | Folds: 5 | Objective: {metric}")
    print(f"  (trial başına ~20 sn, toplam ~{n_trials * 21 / 60:.0f} dk; ilerleme gösterilir)\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    t0 = time.time()
    study.optimize(
        make_objective(task, dev[feat_cols], dev[target], tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    search_minutes = (time.time() - t0) / 60

    print(f"\nOptuna tamamlandı ({search_minutes:.1f} dk).")
    print(f"  En iyi CV {metric}: {study.best_value:.5f}")
    print(f"  En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ---- Final model: tüm veri ile eğit (canlı tahmin için) ----
    print("\n[3/4] Final model tüm veri ile eğitiliyor...")
    final_model = make_model(task, study.best_params)
    final_model.fit(df[feat_cols], df[target])

    # ---- Kaydet ----
    print("[4/4] Modeller kaydediliyor...")

    model_path = MODELS_DIR / spec["model_file"]
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  → Model: {model_path}")

    study_path = MODELS_DIR / spec["study_file"]
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"  → Optuna study: {study_path}")

    # Feature listesi bütün görevler için aynı (aynı features.parquet)
    feat_path = MODELS_DIR / "feature_columns.pkl"
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)
    print(f"  → Feature sütunları: {feat_path}")

    # API/UI ve 04 bu dosyadan okur
    meta = {
        "task":           task,
        "target":         target,
        "trained_at":     pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_trials":       n_trials,
        "search_minutes": round(search_minutes, 1),
        "n_games":        int(len(df)),
        "n_dev_games":    int(len(dev)),
        "n_features":     len(feat_cols),
        "date_from":      str(df["game_date"].min().date()),
        "date_to":        str(df["game_date"].max().date()),
        "holdout_season": holdout_season,
        "cv_metric":      metric,
        "cv_score":       float(study.best_value),
        "best_params":    study.best_params,
        "fixed_params":   task_params(task),
    }
    meta_path = MODELS_DIR / spec["meta_file"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  → Eğitim bilgisi: {meta_path}")

    # Naive tahmin: hep aynı değer (dev ortalaması)
    y_dev = dev[target]
    baseline = cv_loss(task, y_dev, np.full(len(y_dev), y_dev.mean()))
    print(f"\n  Baseline {metric} (hep ortalama): {baseline:.5f}")
    print(f"  Model CV {metric}            : {study.best_value:.5f}")
    improvement = (baseline - study.best_value) / baseline * 100
    print(f"  İyileşme                      : %{improvement:.1f}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main(tasks: list[str], n_trials: int = DEFAULT_TRIALS, holdout_season: str | None = None):
    for task in tasks:
        train_task(task, n_trials, holdout_season)

    print(f"\n{'='*60}")
    print(f"Eğitim tamamlandı ({', '.join(tasks)}).")
    print("Sonraki adım: python src/04_evaluate_model.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=[*TASKS, "all"], default="win",
        help="Eğitilecek model: win (kazanan), margin (fark), total (toplam sayı) ya da all",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Optuna trial sayısı (varsayılan: {DEFAULT_TRIALS}, hızlı test için: 20-30)",
    )
    parser.add_argument(
        "--holdout-season", type=str, default=None,
        help="Aramadan ayrılacak sezon (SEASON_ID, örn: 22025). Varsayılan: son tamamlanmış sezon",
    )
    parser.add_argument(
        "--no-holdout", action="store_true",
        help="Holdout ayırma (tüm veride arama yap)",
    )
    args = parser.parse_args()
    holdout = None if args.no_holdout else (args.holdout_season or default_holdout_season())
    tasks = list(TASKS) if args.task == "all" else [args.task]
    main(tasks, n_trials=args.trials, holdout_season=holdout)
