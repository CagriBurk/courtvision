"""
03_train_model.py
-----------------
Trains XGBoost model with Optuna + TimeSeriesSplit.

The last completed season is held out: Optuna and cross-validation only see
earlier seasons, so 04_evaluate_model.py can measure the model on games it
was never tuned on. The production model is then refit on ALL data
(holdout and the season in progress included).

Input : data/processed/features.parquet
Output :
  models/xgb_best.pkl          (trained model)
  models/optuna_study.pkl      (Optuna study, for inspection)
  models/feature_columns.pkl   (feature order, critical for inference)
  models/train_meta.json       (training info for API/UI and 04)

Usage:
    python src/03_train_model.py
    python src/03_train_model.py --trials 50           # for fast testing
    python src/03_train_model.py --holdout-season 22024
    python src/03_train_model.py --no-holdout
"""

import argparse
import json
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from modeling import (
    BASE_PARAMS,
    MODELS_DIR,
    default_holdout_season,
    load_feature_table,
    split_holdout,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            **BASE_PARAMS,
        }

        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]
            cv_scores.append(log_loss(y_val, y_prob))

            # Optuna pruning: ilk fold'dan sonra çok kötüyse kes
            trial.report(np.mean(cv_scores), step=len(cv_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    return objective


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main(n_trials: int = 150, holdout_season: str | None = None):
    print(f"\n{'='*60}")
    print(f"XGBoost Eğitimi — Optuna ({n_trials} trial) + TimeSeriesSplit")
    print(f"{'='*60}\n")

    print("[1/4] Features yükleniyor...")
    df, feat_cols = load_feature_table()
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
    print(f"  Trials: {n_trials} | Folds: 5 | Objective: log_loss")
    print("  (Bu ~30-90 dakika sürebilir; ilerleme gösterilir)\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(
        make_objective(dev[feat_cols], dev["target"], tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nOptuna tamamlandı.")
    print(f"  En iyi CV log-loss: {study.best_value:.5f}")
    print(f"  En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ---- Final model: tüm veri ile eğit (canlı tahmin için) ----
    print("\n[3/4] Final model tüm veri ile eğitiliyor...")
    final_model = xgb.XGBClassifier(**study.best_params, **BASE_PARAMS)
    final_model.fit(df[feat_cols], df["target"])

    # ---- Kaydet ----
    print("[4/4] Modeller kaydediliyor...")

    model_path = MODELS_DIR / "xgb_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  → Model: {model_path}")

    study_path = MODELS_DIR / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"  → Optuna study: {study_path}")

    feat_path = MODELS_DIR / "feature_columns.pkl"
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)
    print(f"  → Feature sütunları: {feat_path}")

    # API/UI ve 04 bu dosyadan okur
    meta = {
        "trained_at":     pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_trials":       n_trials,
        "n_games":        int(len(df)),
        "n_dev_games":    int(len(dev)),
        "n_features":     len(feat_cols),
        "date_from":      str(df["game_date"].min().date()),
        "date_to":        str(df["game_date"].max().date()),
        "holdout_season": holdout_season,
        "cv_logloss":     float(study.best_value),
        "best_params":    study.best_params,
    }
    meta_path = MODELS_DIR / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  → Eğitim bilgisi: {meta_path}")

    y_dev = dev["target"]
    baseline_logloss = log_loss(y_dev, np.full(len(y_dev), y_dev.mean()))
    print(f"\n  Baseline log-loss (naive): {baseline_logloss:.5f}")
    print(f"  Model CV log-loss        : {study.best_value:.5f}")
    improvement = (baseline_logloss - study.best_value) / baseline_logloss * 100
    print(f"  İyileşme                 : %{improvement:.1f}")

    print(f"\n{'='*60}")
    print("Eğitim tamamlandı. Sonraki adım: python src/04_evaluate_model.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=150,
        help="Optuna trial sayısı (varsayılan: 150, hızlı test için: 20-30)"
    )
    parser.add_argument(
        "--holdout-season", type=str, default=None,
        help="Aramadan ayrılacak sezon (SEASON_ID, örn: 22025). Varsayılan: son tamamlanmış sezon"
    )
    parser.add_argument(
        "--no-holdout", action="store_true",
        help="Holdout ayırma (tüm veride arama yap)"
    )
    args = parser.parse_args()
    holdout = None if args.no_holdout else (args.holdout_season or default_holdout_season())
    main(n_trials=args.trials, holdout_season=holdout)
