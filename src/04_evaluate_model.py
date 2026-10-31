"""
04_evaluate_model.py
--------------------
Measures model performance with walk-forward (out-of-fold) evaluation.
Reruns 5-fold with best parameters, collecting validation data for each fold.

Input :
  data/processed/features.parquet
  models/xgb_best.pkl
  models/optuna_study.pkl
  models/feature_columns.pkl

Output :
  reports/evaluation_report.html  (interactive HTML report)
  Console: fold-by-fold and total metrics

Usage:
    python src/04_evaluate_model.py
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = [
    "game_id", "game_date", "season_id", "season_type",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "team_id_away", "team_abbreviation_away", "team_name_away",
    "target",
]


# ---------------------------------------------------------------------------
def load_artifacts():
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    df = df.sort_values("game_date").reset_index(drop=True)

    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feat_cols = pickle.load(f)

    with open(MODELS_DIR / "optuna_study.pkl", "rb") as f:
        study = pickle.load(f)

    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols]
    y = df["target"]
    return df, X, y, feat_cols, study


# ---------------------------------------------------------------------------
def walk_forward_eval(X, y, best_params, tscv):
    """5-fold walk-forward değerlendirme, OOF tahminleri döner."""
    oof_probs  = np.full(len(y), np.nan)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        fold_metrics.append({
            "fold":     fold + 1,
            "n_train":  len(train_idx),
            "n_val":    len(val_idx),
            "accuracy": accuracy_score(y_val, (probs > 0.5).astype(int)),
            "auc":      roc_auc_score(y_val, probs),
            "logloss":  log_loss(y_val, probs),
            "brier":    brier_score_loss(y_val, probs),
        })
        print(
            f"  Fold {fold+1}: "
            f"Acc={fold_metrics[-1]['accuracy']:.3f}  "
            f"AUC={fold_metrics[-1]['auc']:.3f}  "
            f"LogLoss={fold_metrics[-1]['logloss']:.4f}"
        )

    return oof_probs, fold_metrics


# ---------------------------------------------------------------------------
def fig_to_base64(fig) -> str:
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_plots(y_true, y_prob, X, model):
    plots = {}

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Eğrisi"); ax.legend()
    plots["roc"] = fig_to_base64(fig); plt.close(fig)

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Mükemmel kalibrasyon")
    ax.plot(mean_pred, fraction_pos, "o-", lw=2, label="Model")
    ax.set_xlabel("Ortalama Tahmin Edilen Olasılık")
    ax.set_ylabel("Gerçek Pozitif Oranı")
    ax.set_title("Kalibrasyon Eğrisi"); ax.legend()
    plots["calibration"] = fig_to_base64(fig); plt.close(fig)

    # Feature importance (top 30)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top30 = imp.nlargest(30).sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    top30.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 30 Feature (Gain)"); ax.set_xlabel("Importance")
    plt.tight_layout()
    plots["importance"] = fig_to_base64(fig); plt.close(fig)

    # Confusion matrix
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Dep.", "Pred: Ev"]); ax.set_yticklabels(["Gerçek: Dep.", "Gerçek: Ev"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix"); plt.tight_layout()
    plots["cm"] = fig_to_base64(fig); plt.close(fig)

    return plots


# ---------------------------------------------------------------------------
def generate_html_report(
    metrics: dict, fold_metrics: list, plots: dict,
    feature_cols: list, out_path: Path
):
    fold_rows = "\n".join(
        f"<tr><td>{m['fold']}</td><td>{m['n_train']}</td><td>{m['n_val']}</td>"
        f"<td>{m['accuracy']:.3f}</td><td>{m['auc']:.3f}</td>"
        f"<td>{m['logloss']:.4f}</td><td>{m['brier']:.4f}</td></tr>"
        for m in fold_metrics
    )

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>NBA Model Değerlendirme Raporu</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; }} h2 {{ color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: center; }}
  th {{ background: #16213e; color: white; }}
  tr:nth-child(even) {{ background: #eef2f7; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .metric-card {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.1); }}
  .metric-val {{ font-size: 2em; font-weight: bold; color: #e94560; }}
  .metric-label {{ color: #666; font-size: 0.9em; margin-top: 6px; }}
  .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .plot-card {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,.1); text-align: center; }}
  img {{ max-width: 100%; border-radius: 6px; }}
</style>
</head>
<body>
<h1>🏀 NBA Maç Tahmin Modeli — Değerlendirme Raporu</h1>
<p><b>Tarih:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
   <b>Feature sayısı:</b> {len(feature_cols)} &nbsp;|&nbsp;
   <b>Algoritma:</b> XGBoost + Optuna + TimeSeriesSplit (5-fold)</p>

<h2>Genel Metrikler (Out-of-Fold)</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="metric-val">{metrics['accuracy']:.1%}</div><div class="metric-label">Accuracy</div></div>
  <div class="metric-card"><div class="metric-val">{metrics['auc']:.3f}</div><div class="metric-label">AUC-ROC</div></div>
  <div class="metric-card"><div class="metric-val">{metrics['logloss']:.4f}</div><div class="metric-label">Log-Loss</div></div>
  <div class="metric-card"><div class="metric-val">{metrics['brier']:.4f}</div><div class="metric-label">Brier Score</div></div>
</div>
<p><b>Baseline (her zaman ev sahibi):</b> %{metrics['baseline_acc']:.1%} accuracy &nbsp;|&nbsp;
   <b>İyileşme:</b> +{(metrics['accuracy'] - metrics['baseline_acc'])*100:.1f} pp</p>

<h2>Fold-by-Fold Metrikler</h2>
<table>
  <tr><th>Fold</th><th>Train</th><th>Val</th><th>Accuracy</th><th>AUC</th><th>Log-Loss</th><th>Brier</th></tr>
  {fold_rows}
</table>

<h2>Grafikler</h2>
<div class="plot-grid">
  <div class="plot-card"><img src="data:image/png;base64,{plots['roc']}" alt="ROC"><p>ROC Eğrisi</p></div>
  <div class="plot-card"><img src="data:image/png;base64,{plots['calibration']}" alt="Kalibrasyon"><p>Kalibrasyon Eğrisi</p></div>
  <div class="plot-card"><img src="data:image/png;base64,{plots['cm']}" alt="Confusion Matrix"><p>Confusion Matrix</p></div>
  <div class="plot-card"><img src="data:image/png;base64,{plots['importance']}" alt="Feature Importance"><p>Top 30 Feature Importance</p></div>
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  → HTML rapor: {out_path}")


# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print("Model Değerlendirme — Walk-Forward OOF")
    print(f"{'='*60}\n")

    df, X, y, feat_cols, study = load_artifacts()

    with open(MODELS_DIR / "xgb_best.pkl", "rb") as f:
        final_model = pickle.load(f)

    best_params = {**study.best_params,
                   "objective": "binary:logistic", "eval_metric": "logloss",
                   "tree_method": "hist", "device": "cuda",
                   "random_state": 42, "n_jobs": -1}

    tscv = TimeSeriesSplit(n_splits=5, gap=0)

    print("[1/3] Walk-forward değerlendirme çalışıyor...")
    oof_probs, fold_metrics = walk_forward_eval(X, y, best_params, tscv)

    # OOF tahmini olan satırlar
    valid_mask = ~np.isnan(oof_probs)
    y_true  = y.values[valid_mask]
    y_prob  = oof_probs[valid_mask]
    y_pred  = (y_prob > 0.5).astype(int)

    metrics = {
        "accuracy":     accuracy_score(y_true, y_pred),
        "auc":          roc_auc_score(y_true, y_prob),
        "logloss":      log_loss(y_true, y_prob),
        "brier":        brier_score_loss(y_true, y_prob),
        "baseline_acc": y_true.mean(),  # "hep ev sahibi" baseline
    }

    print(f"\n  OOF Toplam Metrikler:")
    print(f"    Accuracy  : {metrics['accuracy']:.3f}  (baseline: {metrics['baseline_acc']:.3f})")
    print(f"    AUC-ROC   : {metrics['auc']:.3f}")
    print(f"    Log-Loss  : {metrics['logloss']:.4f}")
    print(f"    Brier     : {metrics['brier']:.4f}")

    # Sezon bazlı doğruluk
    df_val = df[valid_mask].copy()
    df_val["pred_prob"] = y_prob
    df_val["correct"]   = (y_pred == y_true)
    print(f"\n  Sezon-by-Sezon Accuracy:")
    for sid in sorted(df_val["season_id"].unique()):
        sub = df_val[df_val["season_id"] == sid]
        print(f"    Sezon {sid}: {sub['correct'].mean():.3f}  ({len(sub)} maç)")

    print("\n[2/3] Grafikler oluşturuluyor...")
    plots = build_plots(y_true, y_prob, X.iloc[valid_mask], final_model)

    # API/UI bu dosyadan okur (sabit değer yerine)
    metrics_out = {
        "evaluated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_oof_games":  int(valid_mask.sum()),
        **{k: float(v) for k, v in metrics.items()},
        "folds": [
            {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in m.items()}
            for m in fold_metrics
        ],
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"  → Metrikler: {metrics_path}")

    print("[3/3] HTML raporu yazılıyor...")
    generate_html_report(
        metrics, fold_metrics, plots, feat_cols,
        REPORTS_DIR / "evaluation_report.html"
    )

    print(f"\n{'='*60}")
    print("Değerlendirme tamamlandı.")
    print("Sonraki adım: python src/05_predict_today.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
