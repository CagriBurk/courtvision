"""
07_experiment.py
----------------
Quick feature-group ablation on the holdout season: trains the same XGBoost
(params from models/train_meta.json, or defaults) with and without a group
of features and compares holdout metrics. Much faster than a full Optuna
run, so it answers "does this feature group help?" before retraining.

Usage:
    python src/07_experiment.py                      # availability features
    python src/07_experiment.py --group h2h_ elo_    # any column prefixes/substrings
"""

import argparse
import json
import warnings

import numpy as np
import xgboost as xgb

from modeling import (
    BASE_PARAMS,
    MODELS_DIR,
    classification_metrics,
    default_holdout_season,
    load_feature_table,
    split_holdout,
)

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_PARAMS = {
    "n_estimators": 600, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
    "colsample_bytree": 0.6, "min_child_weight": 10,
}


def main(group: list[str], holdout_season: str, seeds: int):
    df, feat_cols = load_feature_table()
    dev, holdout = split_holdout(df, holdout_season)

    meta_path = MODELS_DIR / "train_meta.json"
    params = (json.loads(meta_path.read_text(encoding="utf-8"))["best_params"]
              if meta_path.exists() else DEFAULT_PARAMS)

    group_cols = [c for c in feat_cols if any(g in c for g in group)]
    if not group_cols:
        raise SystemExit(f"'{group}' ile eşleşen feature yok. (Oyuncu verisi çekilip 01/02 çalıştırıldı mı?)")
    without = [c for c in feat_cols if c not in group_cols]

    print(f"\nHoldout sezon {holdout_season}: {len(holdout)} maç | arama verisi: {len(dev)} maç")
    print(f"Grup ({len(group_cols)} feature): {', '.join(group_cols)}")
    coverage = dev[group_cols[0]].notna().mean()
    print(f"Doluluk (arama verisi): %{coverage * 100:.0f}\n")

    results = {}
    for name, cols in (("grup olmadan", without), ("grup ile", feat_cols)):
        runs = []
        for seed in range(seeds):
            model = xgb.XGBClassifier(**{**params, **BASE_PARAMS, "random_state": seed})
            model.fit(dev[cols], dev["target"])
            runs.append(classification_metrics(
                holdout["target"], model.predict_proba(holdout[cols])[:, 1]
            ))
        results[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        if name == "grup ile":
            gain = model.get_booster().get_score(importance_type="gain")
            total = sum(gain.values()) or 1
            ranked = sorted(gain, key=gain.get, reverse=True)

    print(f"{'':14}{'Accuracy':>10}{'Log-loss':>10}{'Brier':>9}{'AUC':>8}")
    for name, m in results.items():
        print(f"{name:14}{m['accuracy']:>10.4f}{m['logloss']:>10.4f}{m['brier']:>9.4f}{m['auc']:>8.4f}")
    d = {k: results["grup ile"][k] - results["grup olmadan"][k] for k in results["grup ile"]}
    print(f"{'fark':14}{d['accuracy']:>+10.4f}{d['logloss']:>+10.4f}{d['brier']:>+9.4f}{d['auc']:>+8.4f}")
    print("\n(Log-loss ve Brier'da negatif fark = iyileşme)")

    print("\nGrubun önem sırası (gain, tüm feature'lar içinde):")
    for c in group_cols:
        if c in gain:
            print(f"  {ranked.index(c) + 1:>3}. {c:32} %{gain[c] / total * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", nargs="+", default=["missing_"],
                        help="Feature adında geçen parçalar (varsayılan: missing_)")
    parser.add_argument("--holdout-season", default=None)
    parser.add_argument("--seeds", type=int, default=3, help="Ortalaması alınacak tekrar sayısı")
    args = parser.parse_args()
    main(args.group, args.holdout_season or default_holdout_season(), args.seeds)
