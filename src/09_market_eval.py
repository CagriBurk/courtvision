"""
09_market_eval.py
-----------------
Model vs market (ESPN closing lines) on games the model never trained on:

  OOF       walk-forward predictions on the seasons before the holdout
            (same 5-fold TimeSeriesSplit as 04), where market lines exist
  Holdout   the holdout season (2025-26), model trained on the seasons before
  Sonrası   seasons after the holdout (2026-27 so far: the early-season
            regime), model trained on everything up to and incl. the holdout

For margin and total: MAE of market vs model, the blend weight w in
actual ≈ market + w·(model − market) with its standard error, w fitted on
OOF and applied to the later sets, and the record of betting the model's
side against the line. The margin model is evaluated raw and with the
mean correction (a + b·prediction) that 05 applies, fitted on the last OOF
fold as 04 does. For the winner: log-loss of the no-vig moneyline vs the
classifier and a logistic blend of both.

Predictions are cached (training takes a few minutes); the analysis reruns
from the cache in seconds.

Input : data/processed/features.parquet, data/processed/market_lines.parquet,
        models/train_meta*.json (parameters only; no model file is read or written)
Output: data/processed/market_eval_preds.parquet (prediction cache)
        reports/market_eval.json

Usage:
    python src/09_market_eval.py             # uses the cache when present
    python src/09_market_eval.py --refresh   # retrain and rebuild the cache
"""

import argparse
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from espn_odds import load_market_lines
from market_eval import (
    BREAK_EVEN_110,
    early_season_mask,
    line_section,
    prob_section,
)
from modeling import MODELS_DIR, ROOT, TASKS, load_feature_table, make_model, predict_task, split_holdout
from score_dist import fit_mean_map

warnings.filterwarnings("ignore")

PROC_DIR    = ROOT / "data" / "processed"
PREDS_PATH  = PROC_DIR / "market_eval_preds.parquet"
REPORT_PATH = ROOT / "reports" / "market_eval.json"
N_SPLITS    = 5
SETS        = [("oof", "OOF (holdout öncesi)"), ("holdout", "Holdout"), ("later", "Sonrası (sezon başı)")]


# ---------------------------------------------------------------------------
# 1. Predictions (cached)
# ---------------------------------------------------------------------------
def task_predictions(task: str, df: pd.DataFrame) -> pd.DataFrame:
    spec = TASKS[task]
    meta = json.loads((MODELS_DIR / spec["meta_file"]).read_text(encoding="utf-8"))
    params, holdout_season, target = meta["best_params"], meta["holdout_season"], spec["target"]
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:   # same columns as 03 / 04
        feat_cols = [c for c in pickle.load(f) if c in df.columns]

    df = df[df[target].notna()].reset_index(drop=True)
    dev, holdout = split_holdout(df, holdout_season)
    later = df[df["season_id"].astype(str) > holdout_season]
    parts = []

    def frame(rows: pd.DataFrame, pred, set_name: str, fold) -> pd.DataFrame:
        return pd.DataFrame({
            "task": task, "set": set_name, "fold": fold,
            "game_id": rows["game_id"].astype(str).values,
            "season_id": rows["season_id"].astype(str).values,
            "game_date": rows["game_date"].values,
            "y": rows[target].astype(float).values, "pred": np.asarray(pred, dtype=float),
        })

    X_dev, y_dev = dev[feat_cols], dev[target].astype(float).values
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X_dev)):
        model = make_model(task, params)
        model.fit(X_dev.iloc[tr], y_dev[tr], verbose=False)
        parts.append(frame(dev.iloc[va], predict_task(task, model, X_dev.iloc[va]), "oof", fold))
        print(f"  {task}: fold {fold + 1}/{N_SPLITS}")

    model = make_model(task, params)
    model.fit(X_dev, y_dev, verbose=False)
    parts.append(frame(holdout, predict_task(task, model, holdout[feat_cols]), "holdout", -1))
    print(f"  {task}: holdout {holdout_season} ({len(holdout)} maç)")

    if len(later):
        both = pd.concat([dev, holdout])
        model = make_model(task, params)
        model.fit(both[feat_cols], both[target].astype(float).values, verbose=False)
        parts.append(frame(later, predict_task(task, model, later[feat_cols]), "later", -1))
        print(f"  {task}: sonrası ({len(later)} maç)")
    return pd.concat(parts, ignore_index=True)


def build_predictions() -> pd.DataFrame:
    df, _ = load_feature_table()
    preds = pd.concat([task_predictions(t, df) for t in ("win", "margin", "total")], ignore_index=True)
    preds.to_parquet(PREDS_PATH, index=False)
    print(f"Tahminler kaydedildi → {PREDS_PATH} ({len(preds)} satır)")
    return preds


# ---------------------------------------------------------------------------
# 2. Analysis
# ---------------------------------------------------------------------------
def _with_market(preds: pd.DataFrame, lines: pd.DataFrame, task: str, col: str) -> pd.DataFrame:
    sub = preds[preds["task"] == task].merge(
        lines[["game_id", col, "provider"]].dropna(subset=[col]), on="game_id")
    return sub.rename(columns={col: "market"})


def analyse(preds: pd.DataFrame, lines: pd.DataFrame) -> dict:
    report = {"break_even_110": BREAK_EVEN_110}

    # Margin mean correction as 04 fits it (last OOF fold, before the holdout)
    m_all = preds[preds["task"] == "margin"]
    last = m_all[(m_all["set"] == "oof") & (m_all["fold"] == m_all["fold"].max())]
    mean_map = fit_mean_map(last["y"], last["pred"])
    report["mean_map_last_fold"] = mean_map

    variants = {
        "margin_raw":   ("margin", "market_margin", lambda p: p),
        "margin_mapped": ("margin", "market_margin", lambda p: mean_map["a"] + mean_map["b"] * p),
        "total":        ("total", "total", lambda p: p),
    }
    for key, (task, col, transform) in variants.items():
        data = _with_market(preds, lines, task, col)
        data["model"] = transform(data["pred"].to_numpy())
        oof = data[data["set"] == "oof"]
        w_oof = line_section(oof["y"], oof["market"], oof["model"])["weight"]["w"] if len(oof) else None
        sec = {}
        for set_name, _ in SETS:
            s = data[data["set"] == set_name]
            if len(s):
                sec[set_name] = line_section(s["y"], s["market"], s["model"],
                                             None if set_name == "oof" else w_oof)
        # Early season vs the rest, every out-of-sample game pooled
        early = early_season_mask(data["game_date"])
        sec["early_season"] = line_section(data["y"][early], data["market"][early], data["model"][early])
        sec["rest_of_season"] = line_section(data["y"][~early], data["market"][~early], data["model"][~early])
        sec["by_season"] = {
            sid: line_section(g["y"], g["market"], g["model"])["weight"]
            for sid, g in data.groupby("season_id") if len(g) >= 200
        }
        report[key] = sec

    # Winner: no-vig moneyline vs classifier
    win = _with_market(preds, lines, "win", "p_home_novig")
    oof = win[win["set"] == "oof"]
    fit = prob_section(oof["y"], oof["market"], oof["pred"])["fit_in_sample"] if len(oof) >= 50 else None
    report["win"] = {
        set_name: prob_section(s["y"], s["market"], s["pred"], None if set_name == "oof" else fit)
        for set_name, _ in SETS if len(s := win[win["set"] == set_name])
    }
    margin_preds = preds[preds["task"] == "margin"]
    margin_lines = _with_market(preds, lines, "margin", "market_margin")
    report["coverage"] = {
        set_name: {"games": int((margin_preds["set"] == set_name).sum()),
                   "with_line": int((margin_lines["set"] == set_name).sum())}
        for set_name, _ in SETS
    }
    return report


# ---------------------------------------------------------------------------
# 3. Printing
# ---------------------------------------------------------------------------
def _f(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def print_report(r: dict) -> None:
    mm = r["mean_map_last_fold"]
    print(f"\nOrtalama düzeltmesi (son OOF fold): gerçek ≈ {mm['a']:+.2f} + {mm['b']:.3f} × tahmin")
    print("Kapsam (handikap çizgisi olan maç / tüm maç): "
          + ", ".join(f"{lbl} {r['coverage'][k]['with_line']}/{r['coverage'][k]['games']}"
                      for k, lbl in SETS))

    titles = {"margin_raw": "FARK — ham model", "margin_mapped": "FARK — düzeltilmiş model (05'teki)",
              "total": "TOPLAM"}
    for key, title in titles.items():
        sec = r[key]
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(f"{'Küme':<24}{'maç':>6}{'MAE piy.':>10}{'MAE model':>11}{'harman*':>9}"
              f"{'w':>7}{'± se':>7}{'|fark|':>8}")
        rows = [(lbl, sec.get(k)) for k, lbl in SETS] + [
            ("Sezon başı (Eki-Kas)", sec["early_season"]), ("Sezonun kalanı", sec["rest_of_season"])]
        for lbl, s in rows:
            if not s:
                continue
            b = s.get("blend_out_of_sample")
            print(f"{lbl:<24}{s['n']:>6}{_f(s['market']['mae']):>10}{_f(s['model']['mae']):>11}"
                  f"{_f(b['mae'] if b else None):>9}{_f(s['weight']['w']):>7}{_f(s['weight']['se']):>7}"
                  f"{_f(s['weight']['gap_mean_abs'], 1):>8}")
        print("  * harman: piyasa + w_OOF × (model − piyasa), w OOF'ta öğrenildi")

        for k, lbl in SETS[1:]:
            if k not in sec:
                continue
            print(f"\n  Piyasaya karşı modelin tarafı ({lbl}; başabaş %{100 * r['break_even_110']:.1f}):")
            print(f"  {'|model-piy.| ≥':>16}{'bahis':>7}{'kazanma':>9}{'z':>7}{'ROI':>8}")
            for a in sec[k]["ats"]:
                print(f"  {a['threshold']:>16}{a['bets']:>7}{_f(100 * a['win_rate'], 1):>8}%"
                      f"{_f(a['z_vs_break_even'], 1):>7}{_f(100 * a['roi_at_110'], 1):>7}%")

        print("\n  Sezona göre w (± se):  " + "  ".join(
            f"{sid[1:]}: {_f(v['w'])}±{_f(v['se'])}" for sid, v in sec["by_season"].items()))

    print(f"\n{'=' * 78}\nKAZANAN — marjsız moneyline vs sınıflandırıcı (log-loss, düşük iyi)\n{'=' * 78}")
    print(f"{'Küme':<24}{'maç':>6}{'piyasa':>9}{'model':>9}{'harman*':>9}{'b_model':>9}{'± se':>7}")
    for k, lbl in SETS:
        s = r["win"].get(k)
        if not s:
            continue
        fit = s["fit_in_sample"] or {}
        print(f"{lbl:<24}{s['n']:>6}{_f(s['market_logloss'], 4):>9}{_f(s['model_logloss'], 4):>9}"
              f"{_f(s.get('blend_out_of_sample_logloss'), 4):>9}{_f(fit.get('b_model')):>9}"
              f"{_f(fit.get('se_b_model')):>7}")
    print("  * harman: logit(p) = c + a·logit(piyasa) + b·logit(model), OOF'ta öğrenildi")


def main(refresh: bool) -> None:
    if refresh or not PREDS_PATH.exists():
        print("Tahminler üretiliyor (OOF + holdout + sonrası, 3 görev)...")
        preds = build_predictions()
    else:
        preds = pd.read_parquet(PREDS_PATH)
        print(f"Önbellekteki tahminler kullanılıyor: {PREDS_PATH} (yeniden üretmek için --refresh)")

    lines = load_market_lines()
    if lines.empty:
        raise SystemExit("Piyasa çizgisi yok. Önce: python src/espn_odds.py --backfill --season ...")

    report = analyse(preds, lines)
    print_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nRapor → {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model ile piyasa kapanış çizgilerinin kıyası")
    parser.add_argument("--refresh", action="store_true", help="tahminleri yeniden üret (eğitim yapar)")
    main(parser.parse_args().refresh)
