"""
prediction_tracker.py
----------------------
Saves daily predictions and calculates accuracy after match results are announced.

Functions:
  save_predictions(predictions)   → append to predictions_log.parquet
  fill_results(master)            → fill actual_winner and correct
  get_stats()                     → weighted accuracy and tier based statistics

Prediction log: data/predictions/predictions_log.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
PRED_DIR = ROOT / "data" / "predictions"
PRED_LOG = PRED_DIR / "predictions_log.parquet"
PRED_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_WEIGHTS = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
SCORE_COLS = ["actual_home_pts", "actual_away_pts"]


def _opt_float(v):
    return None if v is None or pd.isna(v) else float(v)


def _ensure_score_cols(log: pd.DataFrame) -> pd.DataFrame:
    """Logs written before the handicap/total models have no score columns."""
    for c in SCORE_COLS:
        if c not in log.columns:
            log[c] = np.nan
        log[c] = log[c].astype(float)
    return log


# ---------------------------------------------------------------------------
# 1. Tahminleri kaydet
# ---------------------------------------------------------------------------
def save_predictions(predictions: list[dict]) -> pd.DataFrame:
    """
    05_predict_today.py'nin ürettiği predictions listesini alır,
    append to predictions_log.parquetr.

    Her kayıt:
      game_id, date, home_team, home_team_id, away_team, away_team_id,
      prob_home, prob_away, confidence, predicted_winner,
      actual_winner (None), correct (None)
    """
    rows = []
    for p in predictions:
        # confidence: YÜKSEK/ORTA/DÜŞÜK (TR) → HIGH/MEDIUM/LOW (EN)
        conf_tr_to_en = {"YÜKSEK": "HIGH", "ORTA": "MEDIUM", "DÜŞÜK": "LOW"}
        conf = conf_tr_to_en.get(p.get("confidence", ""), p.get("confidence", "LOW"))

        prob_home = float(p.get("prob_home", 0.5))
        lines = json.loads(p["lines"]) if p.get("lines") else {}

        rows.append(
            {
                "game_id": str(p["game_id"]),
                "date": str(p.get("date", pd.Timestamp.today().date())),
                "home_team": str(p.get("home_team_abbr", "")),
                "home_team_id": str(p.get("home_team_id", "")),
                "away_team": str(p.get("away_team_abbr", "")),
                "away_team_id": str(p.get("away_team_id", "")),
                "prob_home": round(prob_home, 4),
                "prob_away": round(1.0 - prob_home, 4),
                "confidence": conf,
                "predicted_winner": "HOME" if prob_home >= 0.5 else "AWAY",
                "actual_winner": None,
                "correct": None,
                # Handikap / toplam (modeller yoksa boş)
                "pred_margin": _opt_float(p.get("pred_margin")),
                "pred_total": _opt_float(p.get("pred_total")),
                "fair_margin_line": _opt_float(lines.get("margin", {}).get("fair_line")),
                "fair_total_line": _opt_float(lines.get("total", {}).get("fair_line")),
                "lines_json": p.get("lines"),
                "actual_home_pts": np.nan,
                "actual_away_pts": np.nan,
            }
        )

    new_df = pd.DataFrame(rows)

    if PRED_LOG.exists():
        existing = pd.read_parquet(PRED_LOG)
        # Aynı game_id zaten varsa güncelleme yap, yoksa ekle
        existing = existing[~existing["game_id"].isin(new_df["game_id"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(PRED_LOG, index=False)
    print(f"[tracker] {len(new_df)} tahmin kaydedildi → {PRED_LOG}")
    return combined


# ---------------------------------------------------------------------------
# 2. Sonuçları doldur (ertesi gün çağrılır)
# ---------------------------------------------------------------------------
def fill_results(master: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    master_games.parquet'ten gerçek maç sonuçlarını okur,
    predictions_log'daki actual_winner ve correct sütunlarını doldurur.

    master: game_id, team_id_home, team_id_away, target (1=ev kazandı) içermeli.
    """
    if not PRED_LOG.exists():
        print("[tracker] Kayıt dosyası bulunamadı.")
        return pd.DataFrame()

    log = _ensure_score_cols(pd.read_parquet(PRED_LOG))

    # Kazananı ya da skoru henüz doldurulmamış satırlar
    unfilled = log[log["actual_winner"].isna() | log["actual_home_pts"].isna()].copy()
    if unfilled.empty:
        print("[tracker] Doldurulacak kayıt yok.")
        return log

    # Master'ı yükle
    if master is None:
        master_path = ROOT / "data" / "processed" / "master_games.parquet"
        if not master_path.exists():
            print("[tracker] master_games.parquet bulunamadı.")
            return log
        master = pd.read_parquet(master_path)

    master = master.copy()
    master["game_id"] = master["game_id"].astype(str)

    filled_count = 0
    for idx, row in unfilled.iterrows():
        match = master[master["game_id"] == row["game_id"]]
        if match.empty or pd.isna(match.iloc[0].get("target")):
            continue  # Sonuç henüz yok (maç oynanmamış)
        m = match.iloc[0]

        if pd.isna(row["actual_winner"]):
            actual = "HOME" if int(m["target"]) == 1 else "AWAY"  # 1=ev kazandı
            log.at[idx, "actual_winner"] = actual
            log.at[idx, "correct"] = actual == row["predicted_winner"]
            filled_count += 1
        if pd.notna(m.get("pts_home")) and pd.notna(m.get("pts_away")):
            log.at[idx, "actual_home_pts"] = float(m["pts_home"])
            log.at[idx, "actual_away_pts"] = float(m["pts_away"])

    log.to_parquet(PRED_LOG, index=False)
    print(f"[tracker] {filled_count} maçın sonucu dolduruldu.")
    _settle_coupons(log)
    return log


# ---------------------------------------------------------------------------
# 3. İstatistikler
# ---------------------------------------------------------------------------
def get_stats() -> dict:
    """
    Tüm kayıtlardan weighted accuracy and tier based statistics döner.

    Döner:
      total, correct, raw_accuracy,
      weighted_accuracy,
      by_tier: {HIGH: {correct, total, accuracy}, ...},
      last_7_days_accuracy
    """
    if not PRED_LOG.exists():
        return _empty_stats()

    log = pd.read_parquet(PRED_LOG)
    completed = log[log["correct"].notna()].copy()

    if completed.empty:
        return _empty_stats()

    # Ham accuracy
    total = len(completed)
    correct = int(completed["correct"].sum())

    # Ağırlıklı accuracy
    weights = completed["confidence"].map(CONFIDENCE_WEIGHTS).fillna(1)
    weighted_correct = (completed["correct"].astype(float) * weights).sum()
    weighted_total = weights.sum()
    weighted_acc = (
        float(weighted_correct / weighted_total) if weighted_total > 0 else 0.0
    )

    # Tier bazlı
    by_tier = {}
    for tier in ("HIGH", "MEDIUM", "LOW"):
        tier_rows = completed[completed["confidence"] == tier]
        t = len(tier_rows)
        c = int(tier_rows["correct"].sum()) if t > 0 else 0
        by_tier[tier] = {
            "correct": c,
            "total": t,
            "accuracy": round(c / t * 100, 1) if t > 0 else None,
        }

    # Son 7 gün
    completed["date"] = pd.to_datetime(completed["date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=7)
    recent = completed[completed["date"] >= cutoff]
    r_total = len(recent)
    r_correct = int(recent["correct"].sum()) if r_total > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "raw_accuracy": round(correct / total * 100, 1),
        "weighted_accuracy": round(weighted_acc * 100, 1),
        "by_tier": by_tier,
        "last_7_days": {
            "total": r_total,
            "correct": r_correct,
            "accuracy": round(r_correct / r_total * 100, 1) if r_total > 0 else None,
        },
    }


# ---------------------------------------------------------------------------
# 2b. Canlı sonuç doldurma (nba_api — 2024+ sezonlar için)
# ---------------------------------------------------------------------------
def fill_results_live(days_back: int = 7) -> pd.DataFrame:
    """
    Son `days_back` günde oynanan maçların sonuçlarını nba_api'den çeker
    ve predictions_log'daki boş actual_winner satırlarını doldurur.

    nba_api yoksa sessizce geçer.
    """
    if not PRED_LOG.exists():
        return pd.DataFrame()

    try:
        from nba_api.stats.endpoints import leaguegamelog
        import time as _time
    except ImportError:
        print("[tracker] nba_api yüklü değil, fill_results_live atlandı.")
        return pd.DataFrame()

    log = _ensure_score_cols(pd.read_parquet(PRED_LOG))
    unfilled = log[log["actual_winner"].isna() | log["actual_home_pts"].isna()].copy()
    if unfilled.empty:
        return log

    # Son days_back gün için güncel sezonun LeagueGameLog'unu çek
    from datetime import date, timedelta
    from season import current_season

    cutoff = date.today() - timedelta(days=days_back)

    try:
        gl = leaguegamelog.LeagueGameLog(
            season=current_season(),
            season_type_all_star="Regular Season",
            league_id="00",
        )
        _time.sleep(0.6)
        df = gl.get_data_frames()[0]
    except Exception as e:
        print(f"[tracker] LeagueGameLog hatası: {e}")
        return log

    if df.empty or "WL" not in df.columns:
        return log

    # Tamamlanmış maçlar (WL dolu)
    df = df[df["WL"].notna()].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    df = df[df["GAME_DATE"] >= cutoff]
    df["MATCHUP_IS_HOME"] = df["MATCHUP"].apply(lambda m: "@" not in str(m))

    # GAME_ID → home_team (kazandı mı?) eşlemesi oluştur
    home_rows = df[df["MATCHUP_IS_HOME"]].copy()
    home_rows["actual_winner_flag"] = (home_rows["WL"] == "W").astype(int)
    game_results: dict[str, int] = dict(
        zip(home_rows["GAME_ID"].astype(str), home_rows["actual_winner_flag"])
    )

    # GAME_ID → (ev sayısı, deplasman sayısı)
    scores: dict[str, tuple[float, float]] = {}
    if "PTS" in df.columns:
        away_pts = dict(zip(df.loc[~df["MATCHUP_IS_HOME"], "GAME_ID"].astype(str),
                            df.loc[~df["MATCHUP_IS_HOME"], "PTS"]))
        for gid, pts in zip(home_rows["GAME_ID"].astype(str), home_rows["PTS"]):
            if gid in away_pts and pd.notna(pts) and pd.notna(away_pts[gid]):
                scores[gid] = (float(pts), float(away_pts[gid]))

    filled_count = 0
    changed = False
    for idx, row in unfilled.iterrows():
        gid = str(row["game_id"])
        if gid not in game_results:
            continue
        if pd.isna(row["actual_winner"]):
            target = game_results[gid]  # 1=ev kazandı, 0=dep kazandı
            actual = "HOME" if target == 1 else "AWAY"
            log.at[idx, "actual_winner"] = actual
            log.at[idx, "correct"] = actual == row["predicted_winner"]
            filled_count += 1
            changed = True
        if gid in scores:
            log.at[idx, "actual_home_pts"], log.at[idx, "actual_away_pts"] = scores[gid]
            changed = True

    if changed:
        log.to_parquet(PRED_LOG, index=False)
        print(f"[tracker] fill_results_live: {filled_count} maç dolduruldu.")
    else:
        print("[tracker] fill_results_live: yeni sonuç bulunamadı.")

    _settle_coupons(log)
    return log


def _settle_coupons(log: pd.DataFrame) -> None:
    """Grade the coupon log from the scores just filled (Faz 6)."""
    try:
        from coupon_tracker import settle_coupons
        n = settle_coupons(log, PRED_DIR / "coupons_log.json")
        if n:
            print(f"[tracker] {n} kupon sonuçlandı.")
    except Exception as e:  # kupon kaydı tahmin kaydını asla bozmasın
        print(f"[tracker] kupon sonuçlandırma atlandı: {e!r}")


def print_live_report() -> None:
    """
    Honest summary of live (pre-tip) predictions: accuracy with a 95% interval,
    the "always pick the home team" baseline on the same games, and
    probability quality (log-loss, Brier), to compare with models/metrics.json.
    """
    if not PRED_LOG.exists():
        print("  Kayıt dosyası yok.")
        return
    log = pd.read_parquet(PRED_LOG)
    done = log[log["correct"].notna()].copy()
    pending = len(log) - len(done)
    if done.empty:
        print(f"  Henüz sonuçlanmış tahmin yok ({pending} tahmin sonuç bekliyor).")
        return

    import numpy as np

    n = len(done)
    correct = done["correct"].astype(float)
    home_won = (done["actual_winner"] == "HOME").astype(float)
    p = done["prob_home"].astype(float).clip(1e-6, 1 - 1e-6)
    acc = correct.mean()
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)
    logloss = float(-(home_won * np.log(p) + (1 - home_won) * np.log(1 - p)).mean())
    brier = float(((p - home_won) ** 2).mean())

    print(f"  Sonuçlanan maç      : {n}  (sonuç bekleyen: {pending})")
    print(f"  Tarih aralığı       : {done['date'].min()} → {done['date'].max()}")
    lo, hi = max(acc - half, 0.0), min(acc + half, 1.0)
    print(f"  Doğruluk            : %{acc * 100:.1f}  (%95 aralık: %{lo * 100:.1f} – %{hi * 100:.1f})")
    print(f"  Hep ev sahibi dese  : %{home_won.mean() * 100:.1f}")
    print(f"  Log-loss / Brier    : {logloss:.4f} / {brier:.4f}")

    print("\n  Güven seviyesine göre:")
    for tier in ("HIGH", "MEDIUM", "LOW"):
        t = done[done["confidence"] == tier]
        if len(t):
            print(f"    {tier:6s}: {int(t['correct'].sum())}/{len(t)}  (%{t['correct'].mean() * 100:.1f})")

    print("\n  Günlük:")
    for d, g in done.groupby("date"):
        print(f"    {d}: {int(g['correct'].sum())}/{len(g)}")

    _print_line_report(_ensure_score_cols(log))


def line_report(log: pd.DataFrame) -> dict:
    """
    Handicap / total accuracy of settled games: point error of the predicted
    margin and total, and how often the priced ladder lines came in compared
    with their probability (calibration in 10-point probability buckets).
    """
    log = _ensure_score_cols(log.copy())
    has_score = log["actual_home_pts"].notna() & log["actual_away_pts"].notna()
    actual = {
        "margin": log["actual_home_pts"] - log["actual_away_pts"],
        "total": log["actual_home_pts"] + log["actual_away_pts"],
    }
    out = {}
    for task, col in (("margin", "pred_margin"), ("total", "pred_total")):
        if col not in log.columns:
            continue
        rows = log[has_score & log[col].notna()]
        if rows.empty:
            continue
        err = rows[col].astype(float) - actual[task][rows.index]
        # Kayıtlı barem olasılıkları ↔ gerçekleşen
        pairs = []
        for i, js in rows["lines_json"].items():
            lines = json.loads(js).get(task) if isinstance(js, str) else None
            for rung in (lines or {}).get("ladder", []):
                pairs.append((rung["p_over"], float(actual[task][i] > rung["line"])))
        buckets = []
        if pairs:
            p, hit = np.array(pairs).T
            b = np.minimum((p * 10).astype(int), 9)
            for k in range(10):
                m = b == k
                if m.any():
                    buckets.append({"bucket": k, "n": int(m.sum()),
                                    "mean_pred": float(p[m].mean()), "frac_over": float(hit[m].mean())})
        out[task] = {
            "n": int(len(rows)),
            "mae": float(err.abs().mean()),
            "bias": float(err.mean()),
            "within_5": float((err.abs() <= 5).mean()),
            "within_10": float((err.abs() <= 10).mean()),
            "ladder_calibration": buckets,
        }
    return out


def _print_line_report(log: pd.DataFrame) -> None:
    rep = line_report(log)
    if not rep:
        return
    labels = {"margin": "Fark (handikap)", "total": "Toplam sayı"}
    for task, r in rep.items():
        print(f"\n  {labels[task]} — {r['n']} maç:")
        print(f"    MAE {r['mae']:.2f} sayı | sapma {r['bias']:+.2f} (+ = fazla tahmin) | "
              f"±5 içinde %{r['within_5'] * 100:.0f} | ±10 içinde %{r['within_10'] * 100:.0f}")
        if r["ladder_calibration"]:
            print("    Barem olasılığı → gerçekleşen üst oranı (barem sayısı):")
            for b in r["ladder_calibration"]:
                print(f"      %{b['mean_pred'] * 100:4.0f} → %{b['frac_over'] * 100:4.0f}  ({b['n']})")


def _empty_stats() -> dict:
    return {
        "total": 0,
        "correct": 0,
        "raw_accuracy": None,
        "weighted_accuracy": None,
        "by_tier": {
            t: {"correct": 0, "total": 0, "accuracy": None}
            for t in ("HIGH", "MEDIUM", "LOW")
        },
        "last_7_days": {"total": 0, "correct": 0, "accuracy": None},
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sonuçları doldur ve istatistikleri yazdır
    import sys
    sys.stdout.reconfigure(errors="replace")

    print("Sonuçlar dolduruluyor...")
    fill_results()
    fill_results_live(days_back=60)

    print("\nCanlı tahmin raporu:")
    print_live_report()
