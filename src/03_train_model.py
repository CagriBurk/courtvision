"""
03_train_model.py
-----------------
XGBoost modelini Optuna + TimeSeriesSplit ile eğitir.

Girdi : data/processed/features.parquet
Çıktı :
  models/xgb_best.pkl          (eğitilmiş model)
  models/optuna_study.pkl      (Optuna study, inceleme için)
  models/feature_columns.pkl   (feature sırası, inference için kritik)

Kullanım:
    python src/03_train_model.py
    python src/03_train_model.py --trials 50   # hızlı test için
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model için kullanılmayacak meta sütunlar
META_COLS = [
    "game_id", "game_date", "season_id", "season_type",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "team_id_away", "team_abbreviation_away", "team_name_away",
    "target",
]


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
def load_features() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    print("[1/4] Features yükleniyor...")
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    df = df.sort_values("game_date").reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols]
    y = df["target"]

    print(f"  → Toplam maç   : {len(df)}")
    print(f"  → Feature sayısı: {len(feat_cols)}")
    print(f"  → Ev sahibi W  : {y.mean():.3f}")
    print(f"  → Tarih aralığı: {df['game_date'].min().date()} → {df['game_date'].max().date()}")
    return X, y, feat_cols


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
            # Sabit parametreler
            "objective":         "binary:logistic",
            "eval_metric":       "logloss",
            "tree_method":       "hist",
            "device":            "cuda",
            "random_state":      42,
            "n_jobs":            -1,
        }

        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            y_prob = model.predict_proba(X_val)[:, 1]
            cv_scores.append(log_loss(y_val, y_prob))

            # Optuna pruning: ilk fold'dan sonra çok kötüyse kes
            trial.report(np.mean(cv_scores), step=len(cv_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    return objective


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main(n_trials: int = 150):
    print(f"\n{'='*60}")
    print(f"XGBoost Eğitimi — Optuna ({n_trials} trial) + TimeSeriesSplit")
    print(f"{'='*60}\n")

    X, y, feat_cols = load_features()

    # ---- TimeSeriesSplit ----
    # 5 fold, sıralı split. gap=0 çünkü shift(1) zaten koruma sağlıyor.
    tscv = TimeSeriesSplit(n_splits=5, gap=0)

    print("[2/4] Optuna hiper-parametre araması başlıyor...")
    print(f"  Trials: {n_trials} | Folds: 5 | Objective: log_loss")
    print("  (Bu ~30-90 dakika sürebilir; ilerleme gösterilir)\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )

    study.optimize(
        make_objective(X, y, tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nOptuna tamamlandı.")
    print(f"  En iyi CV log-loss: {study.best_value:.5f}")
    print(f"  En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ---- Final model: tüm veri ile eğit ----
    print("\n[3/4] Final model tüm veri ile eğitiliyor...")
    best_params = {**study.best_params,
                   "objective":   "binary:logistic",
                   "eval_metric": "logloss",
                   "tree_method": "hist",
                   "device":      "cuda",
                   "random_state": 42,
                   "n_jobs": -1}

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y)

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

    # ---- Baseline karşılaştırma ----
    baseline_logloss = log_loss(y, np.full(len(y), y.mean()))
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
    args = parser.parse_args()
    main(n_trials=args.trials)
