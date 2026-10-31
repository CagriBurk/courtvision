"""
04_evaluate_model.py
--------------------
Measures model performance two ways:

  1. Walk-forward OOF on the seasons before the holdout (5-fold
     TimeSeriesSplit with the tuned parameters).
  2. Holdout: a model trained on all pre-holdout seasons predicts the
     held-out season, which Optuna never saw. This is the honest headline
     number; calibration is measured here too.

Input :
  data/processed/features.parquet
  models/xgb_best.pkl
  models/feature_columns.pkl
  models/train_meta.json      (best params + holdout season, from 03)

Output :
  models/metrics.json             (read by the API/UI)
  reports/evaluation_report.html
  Console: fold-by-fold, holdout and per-season metrics

Usage:
    python src/04_evaluate_model.py
"""

import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

from modeling import (
    BASE_PARAMS,
    MODELS_DIR,
    ROOT,
    apply_calibration,
    calibration_table,
    classification_metrics,
    fit_platt,
    load_feature_table,
    split_holdout,
)

warnings.filterwarnings("ignore")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_MIN_GAIN = 0.001  # holdout log-loss iyileşmesi bundan küçükse kalibrasyon kapalı


# ---------------------------------------------------------------------------
def load_artifacts():
    df, _ = load_feature_table()

    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    meta = json.loads((MODELS_DIR / "train_meta.json").read_text(encoding="utf-8"))

    feat_cols = [c for c in feat_cols if c in df.columns]
    return df, feat_cols, meta


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
def _metric_cards(m: dict) -> str:
    return f"""<div class="metric-grid">
  <div class="metric-card"><div class="metric-val">{m['accuracy']:.1%}</div><div class="metric-label">Accuracy</div></div>
  <div class="metric-card"><div class="metric-val">{m['auc']:.3f}</div><div class="metric-label">AUC-ROC</div></div>
  <div class="metric-card"><div class="metric-val">{m['logloss']:.4f}</div><div class="metric-label">Log-Loss</div></div>
  <div class="metric-card"><div class="metric-val">{m['brier']:.4f}</div><div class="metric-label">Brier Score</div></div>
</div>
<p><b>Baseline (her zaman ev sahibi):</b> {m['baseline_acc']:.1%} accuracy &nbsp;|&nbsp;
   <b>İyileşme:</b> +{(m['accuracy'] - m['baseline_acc'])*100:.1f} pp</p>"""


def generate_html_report(
    oof: dict, holdout: dict | None, holdout_season: str | None, ece: float | None,
    fold_metrics: list, plots: dict, feature_cols: list, out_path
):
    holdout_html = (
        f"""<h2>Holdout — Sezon {holdout_season} (aramada hiç görülmedi)</h2>
{_metric_cards(holdout)}
<p><b>Kalibrasyon hatası (ECE):</b> {ece:.4f}</p>"""
        if holdout else ""
    )
    plots_source = f"holdout sezon {holdout_season}" if holdout else "OOF"
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

{holdout_html}

<h2>Walk-Forward Out-of-Fold (holdout öncesi sezonlar)</h2>
{_metric_cards(oof)}

<h2>Fold-by-Fold Metrikler</h2>
<table>
  <tr><th>Fold</th><th>Train</th><th>Val</th><th>Accuracy</th><th>AUC</th><th>Log-Loss</th><th>Brier</th></tr>
  {fold_rows}
</table>

<h2>Grafikler ({plots_source})</h2>
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
    print("Model Değerlendirme — Walk-Forward OOF + Holdout")
    print(f"{'='*60}\n")

    df, feat_cols, meta = load_artifacts()
    holdout_season = meta.get("holdout_season")
    dev, holdout = split_holdout(df, holdout_season)

    with open(MODELS_DIR / "xgb_best.pkl", "rb") as f:
        final_model = pickle.load(f)

    best_params = {**meta["best_params"], **BASE_PARAMS}
    tscv = TimeSeriesSplit(n_splits=5, gap=0)

    # ---- 1. Walk-forward OOF (holdout öncesi sezonlar) ----
    print("[1/4] Walk-forward değerlendirme çalışıyor (holdout öncesi sezonlar)...")
    X_dev, y_dev = dev[feat_cols], dev["target"]
    oof_probs, fold_metrics = walk_forward_eval(X_dev, y_dev, best_params, tscv)

    valid_mask = ~np.isnan(oof_probs)
    oof = classification_metrics(y_dev.values[valid_mask], oof_probs[valid_mask])
    print(f"\n  OOF Toplam Metrikler:")
    print(f"    Accuracy  : {oof['accuracy']:.3f}  (baseline: {oof['baseline_acc']:.3f})")
    print(f"    AUC-ROC   : {oof['auc']:.3f}")
    print(f"    Log-Loss  : {oof['logloss']:.4f}")
    print(f"    Brier     : {oof['brier']:.4f}")

    df_val = dev[valid_mask].copy()
    df_val["correct"] = (oof_probs[valid_mask] > 0.5).astype(int) == y_dev.values[valid_mask]
    print(f"\n  Sezon-by-Sezon Accuracy (OOF):")
    for sid in sorted(df_val["season_id"].unique()):
        sub = df_val[df_val["season_id"] == sid]
        print(f"    Sezon {sid}: {sub['correct'].mean():.3f}  ({len(sub)} maç)")

    # Kalibratör (Platt) holdout öncesi OOF tahminlerinden öğrenilir
    platt = fit_platt(y_dev.values[valid_mask], oof_probs[valid_mask])

    # ---- 2. Holdout (aramada hiç görülmemiş sezon) ----
    holdout_metrics, ece, calib_rows, raw_holdout = None, None, [], None
    calibration = None
    cal_path = MODELS_DIR / "calibration.json"
    if len(holdout):
        print(f"\n[2/4] Holdout değerlendirmesi: sezon {holdout_season} ({len(holdout)} maç)...")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_dev, y_dev)
        y_true = holdout["target"].values
        y_raw = model.predict_proba(holdout[feat_cols])[:, 1]
        raw_holdout = classification_metrics(y_true, y_raw)
        raw_ece, _ = calibration_table(y_true, y_raw)

        # Kalibrasyon sadece holdout'ta log-loss'u anlamlı ölçüde iyileştiriyorsa
        # kullanılır (a≈1, b≈0 gibi küçük düzeltmeler gürültüdür)
        y_cal = apply_calibration(platt, y_raw)
        cal_holdout = classification_metrics(y_true, y_cal)
        cal_ece, _ = calibration_table(y_true, y_cal)
        print(f"  Kalibrasyon (Platt a={platt['a']:.3f}, b={platt['b']:.3f}):")
        print(f"    Ham         : log-loss {raw_holdout['logloss']:.4f}  ECE {raw_ece:.4f}")
        print(f"    Kalibre     : log-loss {cal_holdout['logloss']:.4f}  ECE {cal_ece:.4f}")
        gain = raw_holdout["logloss"] - cal_holdout["logloss"]
        if gain >= CALIBRATION_MIN_GAIN:
            calibration = platt
            y_prob = y_cal
            print(f"    → Kalibrasyon kullanılacak (log-loss -{gain:.5f}, models/calibration.json)\n")
        else:
            y_prob = y_raw
            print(f"    → Kalibrasyon kapalı (log-loss iyileşmesi {gain:.5f} < {CALIBRATION_MIN_GAIN})\n")

        holdout_metrics = classification_metrics(y_true, y_prob)
        ece, calib_rows = calibration_table(y_true, y_prob)
        print(f"    Accuracy  : {holdout_metrics['accuracy']:.3f}  (baseline: {holdout_metrics['baseline_acc']:.3f})")
        print(f"    AUC-ROC   : {holdout_metrics['auc']:.3f}")
        print(f"    Log-Loss  : {holdout_metrics['logloss']:.4f}")
        print(f"    Brier     : {holdout_metrics['brier']:.4f}")
        print(f"    ECE       : {ece:.4f}  (kalibrasyon hatası; 0 = mükemmel)")
        print(f"\n  Kalibrasyon (holdout):  tahmin → gerçekleşen  (maç)")
        for r in calib_rows:
            print(f"    %{r['mean_pred']*100:5.1f} → %{r['frac_pos']*100:5.1f}  ({r['n']})")
        plot_true, plot_prob, plot_X = y_true, y_prob, holdout[feat_cols]
    else:
        print("\n[2/4] Holdout yok (03 --no-holdout ile eğitilmiş) — grafikler OOF'tan.")
        plot_true, plot_prob, plot_X = y_dev.values[valid_mask], oof_probs[valid_mask], X_dev[valid_mask]

    # Başlık metrik: varsa holdout (dürüst), yoksa OOF
    headline = holdout_metrics or oof

    print("\n[3/4] Grafikler oluşturuluyor...")
    plots = build_plots(plot_true, plot_prob, plot_X, final_model)

    # API/UI bu dosyadan okur (sabit değer yerine)
    metrics_out = {
        "evaluated_at":    pd.Timestamp.now().isoformat(timespec="seconds"),
        "headline":        "holdout" if holdout_metrics else "oof",
        "holdout_season":  holdout_season if holdout_metrics else None,
        "n_holdout_games": int(len(holdout)),
        "n_oof_games":     int(valid_mask.sum()),
        **headline,
        "ece":             ece,
        "calibration":     calib_rows,
        "calibrated":      calibration is not None,
        "holdout_raw":     raw_holdout,
        "oof":             oof,
        "folds": [
            {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in m.items()}
            for m in fold_metrics
        ],
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"  → Metrikler: {metrics_path}")

    # 05 canlı tahminde bu dosya varsa olasılıkları kalibre eder
    if calibration:
        cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"  → Kalibrasyon: {cal_path}")
    elif cal_path.exists():
        cal_path.unlink()

    print("[4/4] HTML raporu yazılıyor...")
    generate_html_report(
        oof, holdout_metrics, holdout_season, ece, fold_metrics, plots, feat_cols,
        REPORTS_DIR / "evaluation_report.html"
    )

    print(f"\n{'='*60}")
    print("Değerlendirme tamamlandı.")
    print("Sonraki adım: python src/05_predict_today.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
