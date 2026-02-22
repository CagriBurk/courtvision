"""
prediction_tracker.py
----------------------
Günlük tahminleri kaydeder ve maç sonuçları açıklandıktan sonra
doğruluğu hesaplar.

Fonksiyonlar:
  save_predictions(predictions)   → predictions_log.parquet'e ekle
  fill_results(master)            → actual_winner ve correct doldur
  get_stats()                     → ağırlıklı accuracy ve tier bazlı istatistikler

Tahmin kaydı: data/predictions/predictions_log.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
PRED_DIR     = ROOT / "data" / "predictions"
PRED_LOG     = PRED_DIR / "predictions_log.parquet"
PRED_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_WEIGHTS = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ---------------------------------------------------------------------------
# 1. Tahminleri kaydet
# ---------------------------------------------------------------------------
def save_predictions(predictions: list[dict]) -> pd.DataFrame:
    """
    05_predict_today.py'nin ürettiği predictions listesini alır,
    predictions_log.parquet'e ekler.

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

        rows.append({
            "game_id":          str(p["game_id"]),
            "date":             str(p.get("date", pd.Timestamp.today().date())),
            "home_team":        str(p.get("home_team_abbr", "")),
            "home_team_id":     str(p.get("home_team_id", "")),
            "away_team":        str(p.get("away_team_abbr", "")),
            "away_team_id":     str(p.get("away_team_id", "")),
            "prob_home":        round(prob_home, 4),
            "prob_away":        round(1.0 - prob_home, 4),
            "confidence":       conf,
            "predicted_winner": "HOME" if prob_home >= 0.5 else "AWAY",
            "actual_winner":    None,
            "correct":          None,
        })

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

    log = pd.read_parquet(PRED_LOG)

    # Henüz doldurulmamış satırlar
    unfilled = log[log["actual_winner"].isna()].copy()
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
        if match.empty:
            continue  # Sonuç henüz yok (maç oynanmamış)

        target = int(match.iloc[0]["target"])   # 1=ev kazandı, 0=deplasman
        actual = "HOME" if target == 1 else "AWAY"
        correct = actual == row["predicted_winner"]

        log.at[idx, "actual_winner"] = actual
        log.at[idx, "correct"]       = correct
        filled_count += 1

    log.to_parquet(PRED_LOG, index=False)
    print(f"[tracker] {filled_count} maçın sonucu dolduruldu.")
    return log


# ---------------------------------------------------------------------------
# 3. İstatistikler
# ---------------------------------------------------------------------------
def get_stats() -> dict:
    """
    Tüm kayıtlardan ağırlıklı accuracy ve tier bazlı istatistikler döner.

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
    total   = len(completed)
    correct = int(completed["correct"].sum())

    # Ağırlıklı accuracy
    weights = completed["confidence"].map(CONFIDENCE_WEIGHTS).fillna(1)
    weighted_correct = (completed["correct"].astype(float) * weights).sum()
    weighted_total   = weights.sum()
    weighted_acc = float(weighted_correct / weighted_total) if weighted_total > 0 else 0.0

    # Tier bazlı
    by_tier = {}
    for tier in ("HIGH", "MEDIUM", "LOW"):
        tier_rows = completed[completed["confidence"] == tier]
        t = len(tier_rows)
        c = int(tier_rows["correct"].sum()) if t > 0 else 0
        by_tier[tier] = {
            "correct":  c,
            "total":    t,
            "accuracy": round(c / t * 100, 1) if t > 0 else None,
        }

    # Son 7 gün
    completed["date"] = pd.to_datetime(completed["date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=7)
    recent = completed[completed["date"] >= cutoff]
    r_total   = len(recent)
    r_correct = int(recent["correct"].sum()) if r_total > 0 else 0

    return {
        "total":             total,
        "correct":           correct,
        "raw_accuracy":      round(correct / total * 100, 1),
        "weighted_accuracy": round(weighted_acc * 100, 1),
        "by_tier":           by_tier,
        "last_7_days": {
            "total":    r_total,
            "correct":  r_correct,
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

    log = pd.read_parquet(PRED_LOG)
    unfilled = log[log["actual_winner"].isna()].copy()
    if unfilled.empty:
        return log

    # Son days_back gün için LeagueGameLog çek (2025-26 current season)
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=days_back)

    try:
        gl = leaguegamelog.LeagueGameLog(
            season="2025-26",
            season_type_all_star="Regular Season",
            league_id="00",
        )
        _time.sleep(0.6)
        df = gl.get_data_frames()[0]
    except Exception as e:
        print(f"[tracker] LeagueGameLog hatası: {e}")
        return log

    if df.empty:
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

    filled_count = 0
    for idx, row in unfilled.iterrows():
        gid = str(row["game_id"])
        if gid not in game_results:
            continue
        target = game_results[gid]   # 1=ev kazandı, 0=dep kazandı
        actual  = "HOME" if target == 1 else "AWAY"
        correct = actual == row["predicted_winner"]
        log.at[idx, "actual_winner"] = actual
        log.at[idx, "correct"]       = correct
        filled_count += 1

    if filled_count > 0:
        log.to_parquet(PRED_LOG, index=False)
        print(f"[tracker] fill_results_live: {filled_count} maç dolduruldu.")
    else:
        print("[tracker] fill_results_live: yeni sonuç bulunamadı.")

    return log


def _empty_stats() -> dict:
    return {
        "total": 0, "correct": 0,
        "raw_accuracy": None, "weighted_accuracy": None,
        "by_tier": {t: {"correct": 0, "total": 0, "accuracy": None} for t in ("HIGH", "MEDIUM", "LOW")},
        "last_7_days": {"total": 0, "correct": 0, "accuracy": None},
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sonuçları doldur ve istatistikleri yazdır
    print("Sonuçlar dolduruluyor...")
    fill_results()

    print("\nİstatistikler:")
    stats = get_stats()
    if stats["total"] == 0:
        print("  Henüz tamamlanmış tahmin yok.")
    else:
        print(f"  Toplam       : {stats['correct']}/{stats['total']} doğru")
        print(f"  Ham Accuracy : {stats['raw_accuracy']}%")
        print(f"  Ağırlıklı    : {stats['weighted_accuracy']}%")
        print(f"  Son 7 gün    : {stats['last_7_days']['accuracy']}%")
        print(f"\n  Tier bazlı:")
        for tier, d in stats["by_tier"].items():
            acc = f"{d['accuracy']}%" if d["accuracy"] is not None else "—"
            print(f"    {tier:6s}: {d['correct']}/{d['total']} ({acc})")
