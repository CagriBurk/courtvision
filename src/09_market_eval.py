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
        models/train_meta*.json (parameters only; no model file is read)
Output: data/processed/market_eval_preds.parquet (prediction cache)
        reports/market_eval.json
        models/market_distribution.json  (closing-line error distribution and
            the model's blend weight, read by 05; not with --no-availability)

Caveat: in training, player-absence features come from the game's own box
score (who got no minutes), while live they come from the injury report.
The model therefore knows, in hindsight, whether questionable players
played, so w measured here is an upper bound for live use.
--no-availability drops those features, which gives the lower bound.

Usage:
    python src/09_market_eval.py                     # uses the cache when present
    python src/09_market_eval.py --refresh           # retrain and rebuild the cache
    python src/09_market_eval.py --no-availability   # without player-absence features
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
    blend_weight_for,
    early_season_mask,
    fit_market_distribution,
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
def is_availability_feature(col: str) -> bool:
    """Player-absence features: in training they come from the game's own box score."""
    return "missing_" in col


def task_predictions(task: str, df: pd.DataFrame, no_availability: bool = False) -> pd.DataFrame:
    spec = TASKS[task]
    meta = json.loads((MODELS_DIR / spec["meta_file"]).read_text(encoding="utf-8"))
    params, holdout_season, target = meta["best_params"], meta["holdout_season"], spec["target"]
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:   # same columns as 03 / 04
        feat_cols = [c for c in pickle.load(f) if c in df.columns]
    if no_availability:
        feat_cols = [c for c in feat_cols if not is_availability_feature(c)]

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


def build_predictions(path, no_availability: bool) -> pd.DataFrame:
    df, _ = load_feature_table()
    preds = pd.concat([task_predictions(t, df, no_availability) for t in ("win", "margin", "total")],
                      ignore_index=True)
    preds.to_parquet(path, index=False)
    print(f"Tahminler kaydedildi → {path} ({len(preds)} satır)")
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


def main(refresh: bool, no_availability: bool) -> None:
    preds_path, report_path = PREDS_PATH, REPORT_PATH
    if no_availability:
        preds_path = preds_path.with_name(preds_path.stem + "_noavail.parquet")
        report_path = report_path.with_name(report_path.stem + "_noavail.json")
        print("Oyuncu eksikliği özellikleri ÇIKARILDI (canlıdaki bilginin alt sınırı)")

    if refresh or not preds_path.exists():
        print("Tahminler üretiliyor (OOF + holdout + sonrası, 3 görev)...")
        preds = build_predictions(preds_path, no_availability)
    else:
        preds = pd.read_parquet(preds_path)
        print(f"Önbellekteki tahminler kullanılıyor: {preds_path} (yeniden üretmek için --refresh)")

    lines = load_market_lines()
    if lines.empty:
        raise SystemExit("Piyasa çizgisi yok. Önce: python src/espn_odds.py --backfill --season ...")

    report = analyse(preds, lines)
    report["no_availability"] = no_availability
    print_report(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nRapor → {report_path}")

    if not no_availability:   # the app's distribution comes from the real model only
        write_market_distribution(preds, lines, report)


def write_market_distribution(preds: pd.DataFrame, lines: pd.DataFrame, report: dict) -> None:
    """models/market_distribution.json: closing-line error distribution + blend weights (read by 05)."""
    master = pd.read_parquet(PROC_DIR / "master_games.parquet",
                             columns=["game_id", "season_id", "pts_home", "pts_away"])
    master["game_id"] = master["game_id"].astype(str)
    games = master.merge(lines[["game_id", "market_margin", "total"]], on="game_id")
    games = games.rename(columns={"total": "market_total"})
    games["y_margin"] = games["pts_home"] - games["pts_away"]
    games["y_total"] = games["pts_home"] + games["pts_away"]

    holdout_season = str(preds.loc[preds["set"] == "holdout", "season_id"].iloc[0])
    dist = fit_market_distribution(games, holdout_season)
    dist["blend"] = {"margin": blend_weight_for(report["margin_mapped"]),
                     "total": blend_weight_for(report["total"])}

    path = MODELS_DIR / "market_distribution.json"
    path.write_text(json.dumps(dist, indent=2, default=float), encoding="utf-8")
    print(f"\nPiyasa dağılımı → {path}")
    for task in ("margin", "total"):
        p, b = dist[task], dist["blend"][task]
        sig = p["sigma"]
        sig_txt = f"{sig['value']:.2f}" if sig["kind"] == "const" else f"{sig['a']:.2f}{sig['b']:+.3f}·x"
        crps_txt = ", ".join(f"{k} {v:.3f}" for k, v in p["selection"]["crps"].items())
        print(f"  {task}: şekil {p['shape']} (CRPS {crps_txt}), sigma {sig_txt}, "
              f"model payı w = {b['w']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model ile piyasa kapanış çizgilerinin kıyası")
    parser.add_argument("--refresh", action="store_true", help="tahminleri yeniden üret (eğitim yapar)")
    parser.add_argument("--no-availability", action="store_true",
                        help="oyuncu eksikliği özellikleri olmadan (eğitimde maçın kendi kutu skorundan "
                             "geldikleri için canlıdan iyimser; bu, bilginin alt sınırını verir)")
    args = parser.parse_args()
    main(args.refresh, args.no_availability)
