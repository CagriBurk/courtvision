"""
08_line_experiment.py
---------------------
Faz 5.4 experiment for the margin / total regressors, on the holdout season
(nothing here is saved; the winner gets ported into the real pipeline).

Variants (same tuned params from models/train_meta_<task>.json, averaged
over --seeds runs):
  current      all features, raw target (what 03 trains today)
  residual     target minus a simple baseline; prediction = baseline + model
               (trees approximate additive structure poorly, the baseline
               carries it)
  new          + possession-based features (pace, offensive / defensive
               rating per 100 possessions, structural expected score) and
               the league scoring trend
  new+residual new features, residual on the structural baseline

Every variant is compared with "current" game by game: mean |error|
difference with its standard error (negative = better than current).

--task margin also prints winning-margin band calibration (normal vs
empirical error shape) for the bands shown in the app.

Usage:
    python src/08_line_experiment.py --task total
    python src/08_line_experiment.py --task margin
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from modeling import (
    MODELS_DIR,
    PROC_DIR,
    TASKS,
    default_holdout_season,
    load_feature_table,
    make_model,
    regression_metrics,
    split_holdout,
)
from score_dist import MARGIN_BANDS, prob_between

warnings.filterwarnings("ignore", category=UserWarning)

LEAGUE_TREND_DAYS = 30


# ---------------------------------------------------------------------------
# Candidate features (computed here; ported to feature_engineering if they win)
# ---------------------------------------------------------------------------
def possession_features() -> pd.DataFrame:
    """
    Per game and side: rolling (previous games only) possessions, offensive
    and defensive rating per 100 possessions. poss = FGA + 0.44 FTA - OREB + TOV.
    """
    log = pd.read_parquet(PROC_DIR / "team_game_log.parquet",
                          columns=["game_id", "game_date", "season_id", "team_id", "opponent_id",
                                   "is_home", "pts_scored", "pts_allowed", "fga", "fta", "oreb", "tov"])
    log["game_id"] = log["game_id"].astype(str)
    log["team_id"] = log["team_id"].astype(str)
    log["opponent_id"] = log["opponent_id"].astype(str)
    for c in ("fga", "fta", "oreb", "tov", "pts_scored", "pts_allowed"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    log["poss"] = log["fga"] + 0.44 * log["fta"] - log["oreb"] + log["tov"]

    # Rakibin aynı maçtaki hücum sayısı → savunma verimi
    opp = log[["game_id", "team_id", "poss"]].rename(columns={"team_id": "opponent_id", "poss": "opp_poss"})
    log = log.merge(opp, on=["game_id", "opponent_id"], how="left")
    game_poss = (log["poss"] + log["opp_poss"]) / 2
    log["pace_poss"] = game_poss
    log["ortg"] = 100 * log["pts_scored"] / game_poss
    log["drtg"] = 100 * log["pts_allowed"] / game_poss

    log = log.sort_values(["team_id", "game_date"])
    g = log.groupby("team_id", group_keys=False)
    out = log[["game_id", "is_home"]].copy()
    for col in ("pace_poss", "ortg", "drtg"):
        prev = g[col].shift(1)
        out[f"{col}_L10"] = prev.groupby(log["team_id"]).transform(lambda s: s.rolling(10, min_periods=3).mean())
        out[f"{col}_season"] = prev.groupby([log["team_id"], log["season_id"]]).transform(
            lambda s: s.expanding(min_periods=3).mean())

    cols = [c for c in out.columns if c not in ("game_id", "is_home")]
    home = out[out["is_home"] == 1].set_index("game_id")[cols].add_prefix("home_")
    away = out[out["is_home"] == 0].set_index("game_id")[cols].add_prefix("away_")
    f = home.join(away, how="inner")

    # Yapısal beklenti: ortak tempo × (hücum ile rakip savunmanın ortalaması)
    pace = (f["home_pace_poss_L10"] + f["away_pace_poss_L10"]) / 2
    exp_home = pace * (f["home_ortg_L10"] + f["away_drtg_L10"]) / 200
    exp_away = pace * (f["away_ortg_L10"] + f["home_drtg_L10"]) / 200
    f["exp_pace"] = pace
    f["exp_total_struct"] = exp_home + exp_away
    f["exp_margin_struct"] = exp_home - exp_away
    return f.reset_index()


def league_trend(df: pd.DataFrame) -> pd.Series:
    """League average game total over the previous LEAGUE_TREND_DAYS days (games before that day)."""
    daily = df.groupby("game_date")["target_total"].agg(["sum", "count"]).sort_index()
    roll = daily.rolling(f"{LEAGUE_TREND_DAYS}D").sum().shift(1)  # gün dahil değil
    avg = (roll["sum"] / roll["count"]).rename("league_total_30d")
    return df["game_date"].map(avg)


# ---------------------------------------------------------------------------
def baseline(task: str, frame: pd.DataFrame, dev: pd.DataFrame, structural: bool) -> np.ndarray:
    if task == "total":
        simple = frame[["home_pace_L10", "away_pace_L10"]].mean(axis=1)
        base = frame["exp_total_struct"].fillna(simple) if structural else simple
        return base.fillna(dev["target_total"].tail(1000).mean()).values
    ok = dev["elo_diff"].notna()
    slope, intercept = np.polyfit(dev.loc[ok, "elo_diff"], dev.loc[ok, "target_margin"], 1)
    elo = intercept + slope * frame["elo_diff"].fillna(0)
    if structural:
        return (frame["exp_margin_struct"] + 2.5).fillna(elo).values  # + ev sahibi avantajı (yaklaşık)
    return elo.values


def run_variant(task, dev, holdout, cols, params, seeds, residual, structural):
    target = TASKS[task]["target"]
    y = dev[target].values.astype(float)
    b_dev = baseline(task, dev, dev, structural) if residual else 0.0
    b_hold = baseline(task, holdout, dev, structural) if residual else 0.0
    preds = []
    for seed in range(seeds):
        model = make_model(task, params).set_params(random_state=seed)
        model.fit(dev[cols], y - b_dev, verbose=False)
        preds.append(model.predict(holdout[cols]) + b_hold)
    return np.mean(preds, axis=0)


def main(task: str, holdout_season: str, seeds: int):
    df, feat_cols = load_feature_table()
    target = TASKS[task]["target"]
    df = df[df[target].notna()].reset_index(drop=True)

    poss = possession_features()
    df["game_id"] = df["game_id"].astype(str)
    df = df.merge(poss, on="game_id", how="left")
    df["league_total_30d"] = league_trend(df)
    new_cols = [c for c in poss.columns if c != "game_id"] + ["league_total_30d"]

    dev, holdout = split_holdout(df, holdout_season)
    params = json.loads((MODELS_DIR / TASKS[task]["meta_file"]).read_text(encoding="utf-8"))["best_params"]
    y_h = holdout[target].values.astype(float)

    print(f"\n{TASKS[task]['label']} — holdout {holdout_season}: {len(holdout)} maç | arama verisi {len(dev)} maç")
    print(f"Yeni özellik doluluğu (holdout): %{holdout['exp_total_struct'].notna().mean() * 100:.0f}"
          f" | {seeds} tekrar ortalaması\n")

    variants = {
        "mevcut":         (feat_cols, False, False),
        "artık":          (feat_cols, True, False),
        "yeni":           (feat_cols + new_cols, False, False),
        "yeni+artık":     (feat_cols + new_cols, True, True),
    }
    preds = {}
    for name, (cols, residual, structural) in variants.items():
        preds[name] = run_variant(task, dev, holdout, cols, params, seeds, residual, structural)
        print(f"  {name} bitti")

    # Referans tabanlar (model yok)
    refs = {
        "taban: son 10" if task == "total" else "taban: Elo": baseline(task, holdout, dev, False),
        "taban: yapısal": baseline(task, holdout, dev, True),
    }
    if task == "total":
        # Lig trendi kadar düzeltilmiş son-10 tabanı
        refs["taban: son 10 + lig"] = (refs["taban: son 10"] * 0.5
                                       + holdout["league_total_30d"].fillna(refs["taban: son 10"].mean()).values * 0.5)

    err_cur = np.abs(preds["mevcut"] - y_h)
    print(f"\n{'':22}{'MAE':>7}{'RMSE':>8}{'sapma':>8}   fark (mevcuda göre, ± std hata)")
    for name, p in {**preds, **refs}.items():
        m = regression_metrics(y_h, p)
        d = np.abs(p - y_h) - err_cur
        diff = "" if name == "mevcut" else f"   {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}"
        print(f"  {name:20}{m['mae']:>7.2f}{m['rmse']:>8.2f}{m['bias']:>+8.2f}{diff}")
    print("\n(fark < 0 = mevcuttan iyi; |fark| std hatanın 2 katından büyükse anlamlı)")

    if task == "margin":
        band_calibration(preds["mevcut"], y_h)


def band_calibration(pred: np.ndarray, y: np.ndarray) -> None:
    """Predicted vs observed frequency of each winning-margin band, normal vs empirical shape."""
    dists = json.loads((MODELS_DIR / "distribution.json").read_text(encoding="utf-8"))
    d = dists["margin"]
    print(f"\nFark bantları — holdout ({len(y)} maç), sigma {d['sigma'].get('value', 0):.2f}:")
    print(f"  {'bant':14}{'gerçek':>8}{'normal':>9}{'ampirik':>9}")
    for side, sign in (("Ev", 1), ("Dep", -1)):
        for lo, hi in MARGIN_BANDS:
            if sign == 1:
                a, b = lo, hi
            else:
                a, b = (None if hi is None else -hi), -lo
            obs = np.mean((y >= (a if a is not None else -999)) & (y <= (b if b is not None else 999)))
            pn = prob_between({**d, "shape": "normal"}, pred, a, b).mean()
            pk = prob_between({**d, "shape": "kde"}, pred, a, b).mean()
            label = f"{side} {lo}-{hi}" if hi else f"{side} {lo}+"
            print(f"  {label:14}{obs * 100:>7.1f}%{pn * 100:>8.1f}%{pk * 100:>8.1f}%")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["margin", "total"], default="total")
    parser.add_argument("--holdout-season", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    main(args.task, args.holdout_season or default_holdout_season(), args.seeds)
