"""
04_evaluate_model.py
--------------------
Measures each model two ways:

  1. Walk-forward OOF on the seasons before the holdout (5-fold
     TimeSeriesSplit with the tuned parameters).
  2. Holdout: a model trained on all pre-holdout seasons predicts the
     held-out season, which Optuna never saw. This is the honest headline
     number; calibration is measured here too.

Win model: accuracy / AUC / log-loss, Platt calibration decision.
Margin and total models: MAE / RMSE / bias against a simple baseline, plus
the error distribution (score_dist.py) that prices betting lines. Its
spread comes from the most recent OOF folds, its shape (normal or
empirical) is chosen by line calibration on the holdout.
When both win and margin are evaluated, the home win probability shown in
the app comes from whichever has the lower holdout log-loss (win_source).

Input :
  data/processed/features.parquet
  models/feature_columns.pkl
  models/xgb_best.pkl, models/train_meta*.json   (from 03)

Output :
  models/metrics.json             (win model, read by the API/UI)
  models/metrics_margin.json, models/metrics_total.json
  models/distribution.json        (error distributions, read by 05)
  reports/evaluation_report.html  (win model)

Usage:
    python src/04_evaluate_model.py                 # win + margin + total
    python src/04_evaluate_model.py --task margin
"""

import argparse
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
    TASKS,
    apply_calibration,
    calibration_table,
    classification_metrics,
    fit_platt,
    load_feature_table,
    make_model,
    regression_metrics,
    split_holdout,
)
from score_dist import (
    band_table,
    center,
    crps,
    fit_distribution,
    fit_mean_map,
    ladder_eval,
    prob_between,
    prob_over,
)

warnings.filterwarnings("ignore")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_MIN_GAIN = 0.001  # holdout log-loss iyileşmesi bundan küçükse kalibrasyon kapalı
WIN_SOURCE_MIN_GAIN = 0.001   # fark modeli kazananı ancak bu kadar daha iyi log-loss ile devralır
RECENT_FOLDS = 3              # hata dağılımı son 3 OOF fold'undan (güncel lig, daha çok eğitim verisi)
DIST_PATH = MODELS_DIR / "distribution.json"


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
def evaluate_win() -> dict | None:
    """Win model. Returns its holdout probabilities (for win_source) or None."""
    print(f"\n{'='*60}")
    print("Model Değerlendirme — Kazanan (win) — Walk-Forward OOF + Holdout")
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

    if not holdout_metrics:
        return None
    return {"game_ids": holdout["game_id"].astype(str).values, "y_true": y_true, "y_prob": y_prob}


# ---------------------------------------------------------------------------
# Margin / total models
# ---------------------------------------------------------------------------
def baseline_prediction(task: str, dev: pd.DataFrame, holdout: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    Simple reference the model has to beat:
      margin: home margin from the Elo difference (straight line fit on dev)
      total:  average game total of both teams' last 10 games
    """
    target = TASKS[task]["target"]
    if task == "margin" and "elo_diff" in dev.columns:
        ok = dev["elo_diff"].notna()
        slope, intercept = np.polyfit(dev.loc[ok, "elo_diff"], dev.loc[ok, target], 1)
        pred = intercept + slope * holdout["elo_diff"].fillna(0).values
        return pred, f"Elo farkından doğrusal: {intercept:+.2f} + {slope:.4f} × elo_diff"
    cols = ["home_pace_L10", "away_pace_L10"]
    if task == "total" and all(c in dev.columns for c in cols):
        fallback = float(dev[target].tail(1000).mean())
        pred = holdout[cols].mean(axis=1).fillna(fallback).values
        return pred, "İki takımın son 10 maç toplam ortalaması"
    return np.full(len(holdout), float(dev[target].mean())), "Arama verisi ortalaması"


def _print_reg(name: str, m: dict) -> None:
    print(f"    {name:<16}: MAE {m['mae']:6.2f}  RMSE {m['rmse']:6.2f}  sapma {m['bias']:+6.2f}  "
          f"±5 içinde %{m['within_5'] * 100:4.1f}  ±10 içinde %{m['within_10'] * 100:4.1f}")


def _save_distribution(task: str, params: dict) -> None:
    dists = json.loads(DIST_PATH.read_text(encoding="utf-8")) if DIST_PATH.exists() else {}
    dists[task] = params
    DIST_PATH.write_text(json.dumps(dists, indent=2), encoding="utf-8")


def _build_dist(task: str, y, pred, mean_corr: bool) -> dict:
    """Error distribution of `pred` vs `y`, optionally around a corrected mean."""
    mm = fit_mean_map(y, pred) if mean_corr else None
    c = center({"mean_map": mm}, pred)
    return fit_distribution(task, np.asarray(y, dtype=float) - c, c, mm)


def _choose_distribution(task: str, candidates: list[dict], y_h, pred_h) -> dict:
    """Scores every candidate on the holdout, prints the table, returns the best CRPS."""
    is_margin = task == "margin"
    for c in candidates:
        c["crps"] = crps(c["params"], y_h, pred_h)
        c["ladder"] = ladder_eval(c["params"], y_h, pred_h)
        c["brier"], c["ece"] = c["ladder"]["brier"], c["ladder"]["ece"]
        if is_margin:
            c["tail_26"] = float(1 - prob_between(c["params"], pred_h, -25, 25).mean())
            c["close_1_5"] = float(prob_between(c["params"], pred_h, -5, 5).mean())
    chosen = min(candidates, key=lambda c: c["crps"])

    print("\n  Dağılım adayları (CRPS = tüm dağılımın skoru, düşük iyi; seçim buna göre):")
    print(f"    {'aday':36}{'CRPS':>8}{'Brier':>8}{'ECE':>8}" + (f"{'26+':>8}{'1-5':>8}" if is_margin else ""))
    for c in candidates:
        line = f"    {c['name']:36}{c['crps']:>8.4f}{c['brier']:>8.4f}{c['ece']:>8.4f}"
        if is_margin:
            line += f"{c['tail_26'] * 100:>7.1f}%{c['close_1_5'] * 100:>7.1f}%"
        print(line + ("  ← seçildi" if c is chosen else ""))
    if is_margin:
        y_h = np.asarray(y_h)
        print(f"    {'gerçekleşen':36}{'':24}{np.mean(np.abs(y_h) >= 26) * 100:>7.1f}%"
              f"{np.mean(np.abs(y_h) <= 5) * 100:>7.1f}%")

    print(f"\n  Seçilen: {chosen['name']}")
    print("    Barem farkı → tahmini üst olasılığı → gerçekleşen üst oranı:")
    for r in chosen["ladder"]["by_offset"]:
        print(f"      {r['offset']:+3d}: %{r['mean_pred'] * 100:5.1f} → %{r['frac_over'] * 100:5.1f}")
    return chosen


def evaluate_regression(task: str) -> dict | None:
    """Margin or total model. Returns its holdout predictions or None."""
    spec = TASKS[task]
    print(f"\n{'='*60}")
    print(f"Model Değerlendirme — {spec['label']} ({task}) — Walk-Forward OOF + Holdout")
    print(f"{'='*60}\n")

    meta_path = MODELS_DIR / spec["meta_file"]
    if not meta_path.exists():
        print(f"  [ATLANDI] {meta_path.name} yok. Önce: python src/03_train_model.py --task {task}")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df, _ = load_feature_table()
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
        feat_cols = [c for c in pickle.load(f) if c in df.columns]
    target = spec["target"]
    df = df[df[target].notna()].reset_index(drop=True)
    holdout_season = meta.get("holdout_season")
    dev, holdout = split_holdout(df, holdout_season)
    params = meta["best_params"]

    # ---- 1. Walk-forward OOF ----
    print("[1/3] Walk-forward değerlendirme (holdout öncesi sezonlar)...")
    tscv = TimeSeriesSplit(n_splits=5, gap=0)
    X_dev, y_dev = dev[feat_cols], dev[target].values.astype(float)
    oof = np.full(len(dev), np.nan)
    fold_of = np.full(len(dev), -1)
    folds = []
    for fold, (tr, va) in enumerate(tscv.split(X_dev)):
        model = make_model(task, params)
        model.fit(X_dev.iloc[tr], y_dev[tr], verbose=False)
        pred = model.predict(X_dev.iloc[va])
        oof[va], fold_of[va] = pred, fold
        fm = regression_metrics(y_dev[va], pred)
        folds.append({"fold": fold + 1, "n_train": len(tr), "n_val": len(va), **fm})
        print(f"  Fold {fold + 1}: MAE={fm['mae']:.2f}  RMSE={fm['rmse']:.2f}  sapma={fm['bias']:+.2f}")

    valid = ~np.isnan(oof)
    oof_m = regression_metrics(y_dev[valid], oof[valid])
    _print_reg("OOF toplam", oof_m)

    print("\n  Sezon bazında sapma (OOF; + = fazla tahmin):")
    seasons = dev["season_id"].values
    for sid in sorted(set(seasons[valid])):
        m = valid & (seasons == sid)
        print(f"    Sezon {sid}: sapma {np.mean(oof[m] - y_dev[m]):+5.2f}  MAE {np.mean(np.abs(oof[m] - y_dev[m])):5.2f}  ({m.sum()} maç)")

    # ---- 2. Hata dağılımı (son fold'lar) ----
    # Pencere: hangi OOF hataları (lig son sezonlarda daha dalgalı); ortalama
    # düzeltmesi: tahminler ortalamaya sıkışmış mı; şekil: normal / t / ampirik.
    windows = {
        f"son {RECENT_FOLDS} fold": fold_of >= 5 - RECENT_FOLDS,
        "son fold": fold_of == 4,
    }
    print("\n[2/3] Hata dağılımı adayları (OOF hatalarından)...")
    candidates = []
    for wname, mask in windows.items():
        mm = fit_mean_map(y_dev[mask], oof[mask])
        rms = np.sqrt(np.mean((y_dev[mask] - oof[mask]) ** 2))
        print(f"  {wname}: {mask.sum()} maç | RMS hata {rms:.2f} | gerçek ≈ {mm['a']:+.2f} + {mm['b']:.3f} × tahmin")
        for mean_corr in (False, True):
            base = _build_dist(task, y_dev[mask], oof[mask], mean_corr)
            for shape in ("normal", "t", "kde"):
                candidates.append({
                    "name": f"{wname} | {'düzeltilmiş' if mean_corr else 'ham'} | {shape}",
                    "window": wname, "mean_corr": mean_corr, "shape": shape,
                    "params": {**base, "shape": shape},
                })
    default = candidates[0]  # son 3 fold | ham | normal (ilk sürüm)
    diag = default["params"]["diagnostics"]
    print(f"  Tahmin büyüklüğüne göre hata (son {RECENT_FOLDS} fold, "
          + ("|tahmin|" if task == "margin" else "tahmin") + "):")
    for b in diag["bins"]:
        print(f"    ~{b['x_mean']:6.1f}: RMS {b['rms']:.2f}  ({b['n']} maç)")
    sig = default["params"]["sigma"]
    if sig["kind"] == "linear":
        print(f"  → Sapma tahmine bağlı (eğilim ×{diag['spread_ratio']:.2f}, t={diag['slope_t']:.1f}): "
              f"sigma = {sig['a']:.2f} {sig['b']:+.4f} × x  [{sig['min']:.2f}, {sig['max']:.2f}]")
    else:
        print(f"  → Sabit sigma {sig['value']:.2f} (eğilim ×{diag['spread_ratio']:.2f}, t={diag['slope_t']:.1f})")

    # ---- 3. Holdout ----
    result = None
    holdout_out = None
    chosen, dist = default, default["params"]
    if len(holdout):
        print(f"\n[3/3] Holdout: sezon {holdout_season} ({len(holdout)} maç)...")
        model = make_model(task, params)
        model.fit(X_dev, y_dev, verbose=False)
        y_h = holdout[target].values.astype(float)
        pred_h = model.predict(holdout[feat_cols])
        hold_m = regression_metrics(y_h, pred_h)
        base_pred, base_desc = baseline_prediction(task, dev, holdout)
        base_m = regression_metrics(y_h, base_pred)
        _print_reg("Model", hold_m)
        _print_reg("Taban", base_m)
        print(f"    (Taban: {base_desc})")
        mm_h = fit_mean_map(y_h, pred_h)
        print(f"    Holdout'ta gerçek ≈ {mm_h['a']:+.2f} + {mm_h['b']:.3f} × tahmin  (b > 1: tahminler sıkışık)")

        chosen = _choose_distribution(task, candidates, y_h, pred_h)
        bands = None
        if task == "margin":
            bands = band_table(chosen["params"], y_h, pred_h)
            first = band_table(default["params"], y_h, pred_h)
            print(f"\n    {'Fark bandı':12}{'gerçek':>8}{'seçilen':>9}{'ilk sürüm':>11}")
            for b, f0 in zip(bands, first):
                side = "Ev" if b["side"] == "home" else "Dep"
                label = f"{side} {b['lo']}-{b['hi']}" if b["hi"] else f"{side} {b['lo']}+"
                print(f"    {label:12}{b['obs'] * 100:>7.1f}%{b['pred'] * 100:>8.1f}%{f0['pred'] * 100:>10.1f}%")

        # Üretim dağılımı: seçilen yapılandırma, pencere + holdout hatalarıyla
        # yeniden öğrenilir (canlı model de holdout dahil tüm veriyle eğitildi)
        mask = windows[chosen["window"]]
        dist = _build_dist(task, np.concatenate([y_dev[mask], y_h]),
                           np.concatenate([oof[mask], pred_h]), chosen["mean_corr"])
        dist["shape"] = chosen["shape"]
        dist["selected"] = chosen["name"]
        s = dist["sigma"]
        mm = dist.get("mean_map")
        print(f"\n  Üretim dağılımı (pencere + holdout, {dist['diagnostics']['n']} maç): "
              f"sigma {s.get('value', float('nan')):.2f} ({s['kind']}) | şekil {dist['shape']}"
              + (f" (t df={dist['t_df']:.0f})" if dist["shape"] == "t" else "")
              + (f" | düzeltme {mm['a']:+.2f} + {mm['b']:.3f} × tahmin" if mm else ""))

        result = {
            "headline": "holdout",
            "holdout": hold_m,
            "baseline": {**base_m, "description": base_desc},
            "holdout_mean_map": mm_h,
            "ladder": {"chosen": chosen["name"], **chosen["ladder"]},
            "candidates": [
                {k: c[k] for k in ("name", "window", "mean_corr", "shape", "crps", "brier", "ece",
                                   "tail_26", "close_1_5") if k in c}
                for c in candidates
            ],
            "bands": bands,
        }
        holdout_out = {"game_ids": holdout["game_id"].astype(str).values, "pred": pred_h,
                       "dist": chosen["params"]}
    else:
        print("\n[3/3] Holdout yok (03 --no-holdout) — ilk aday kullanıldı.")

    metrics_out = {
        "task":            task,
        "evaluated_at":    pd.Timestamp.now().isoformat(timespec="seconds"),
        "holdout_season":  holdout_season if len(holdout) else None,
        "n_holdout_games": int(len(holdout)),
        "n_oof_games":     int(valid.sum()),
        "oof":             oof_m,
        "folds":           folds,
        **(result or {"headline": "oof"}),
        "distribution":    {k: v for k, v in dist.items() if k not in ("z",)},
    }
    metrics_path = MODELS_DIR / spec["metrics_file"]
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    _save_distribution(task, dist)
    print(f"\n  → Metrikler: {metrics_path}")
    print(f"  → Dağılım  : {DIST_PATH} ({task})")
    return holdout_out


# ---------------------------------------------------------------------------
# Which model supplies the home win probability
# ---------------------------------------------------------------------------
def compare_win_sources(win: dict, margin: dict) -> None:
    """
    P(home wins) from the classifier vs P(margin > 0) from the margin model,
    on the same holdout games. The margin model takes over only with a clearly
    lower log-loss; the result is stored in metrics.json (win_source).
    """
    print(f"\n{'='*60}")
    print("Kazanan olasılığı: sınıflandırıcı mı, fark modeli mi?")
    print(f"{'='*60}")
    p_margin = dict(zip(margin["game_ids"], prob_over(margin["dist"], margin["pred"], 0.0)))
    keep = np.array([g in p_margin for g in win["game_ids"]])
    y = np.asarray(win["y_true"])[keep]
    p_clf = np.asarray(win["y_prob"])[keep]
    p_mar = np.array([p_margin[g] for g in np.asarray(win["game_ids"])[keep]])

    clf = classification_metrics(y, p_clf)
    mar = classification_metrics(y, np.clip(p_mar, 1e-6, 1 - 1e-6))
    for name, m in (("Sınıflandırıcı", clf), ("Fark modeli", mar)):
        print(f"  {name:<15}: log-loss {m['logloss']:.4f}  doğruluk %{m['accuracy'] * 100:.1f}  "
              f"AUC {m['auc']:.3f}  Brier {m['brier']:.4f}")
    source = "margin" if mar["logloss"] <= clf["logloss"] - WIN_SOURCE_MIN_GAIN else "classifier"
    print(f"  → Uygulamada kullanılacak: {'fark modeli' if source == 'margin' else 'sınıflandırıcı'} "
          f"(eşik {WIN_SOURCE_MIN_GAIN})")

    path = MODELS_DIR / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metrics["win_source"] = source
    metrics["win_source_compare"] = {"n_games": int(keep.sum()), "classifier": clf, "margin": mar}
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
def main(tasks: list[str]) -> None:
    results = {}
    for task in tasks:
        results[task] = evaluate_win() if task == "win" else evaluate_regression(task)

    if results.get("win") and results.get("margin"):
        compare_win_sources(results["win"], results["margin"])
    elif "margin" in tasks or "win" in tasks:
        print("\n  (Kazanan kaynağı karşılaştırması için win ve margin aynı çalıştırmada değerlendirilmeli.)")

    print(f"\n{'='*60}")
    print(f"Değerlendirme tamamlandı ({', '.join(tasks)}).")
    print("Sonraki adım: python src/05_predict_today.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=[*TASKS, "all"], default="all",
        help="Değerlendirilecek model: win, margin, total ya da all (varsayılan)",
    )
    args = parser.parse_args()
    main(list(TASKS) if args.task == "all" else [args.task])
