"""
coupon_tracker.py
-----------------
Daily coupon log (Faz 6): data/predictions/coupons_log.json

  save_coupons(date, day, started_ids)  store today's coupons; a coupon is
                                        locked once one of its games starts or
                                        the user records the odds they played
  settle_coupons(pred_log)              grade legs from the final scores in the
                                        prediction log
  set_played_odds(coupon_id, odds)      optional: the total odds actually played
  write_live_view(date, day)            data/live/coupons.json for the API

Coupon ids are "<date>:<tier>" (e.g. "2026-11-08:guvenli").
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from coupons import leg_won

ROOT = Path(__file__).parent.parent
PRED_DIR = ROOT / "data" / "predictions"
LIVE_DIR = ROOT / "data" / "live"
COUPON_LOG = PRED_DIR / "coupons_log.json"


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def load_coupons(path: Path | None = None) -> list[dict]:
    path = path or COUPON_LOG
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save(coupons: list[dict], path: Path | None = None) -> None:
    path = path or COUPON_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(coupons, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _is_locked(c: dict, started_ids: set[str]) -> bool:
    return bool(c.get("locked") or c.get("played_odds")
                or any(l["game_id"] in started_ids for l in c["legs"]))


def save_coupons(date: str, day: dict, started_ids: set[str]) -> list[dict]:
    """
    Replace the date's unlocked coupons with today's build; locked ones are kept
    as they were (they are what gets graded). Returns the date's coupons.
    """
    log = load_coupons()
    old = {c["tier"]: c for c in log if c["date"] == date}
    keep = [c for c in log if c["date"] != date]
    new = {c["tier"]: c for c in day.get("coupons", [])}

    today = []
    for tier in [*old, *[t for t in new if t not in old]]:
        prev = old.get(tier)
        if prev and _is_locked(prev, started_ids):
            prev["locked"] = True
            today.append(prev)
        elif tier in new:
            c = new[tier]
            today.append({
                **c,
                "id": f"{date}:{tier}",
                "date": date,
                "created_at": _now(),
                "locked": False,
                "played_odds": None,
                "result": None,
                "settled_at": None,
                "legs": [{**l, "won": None, "score": None} for l in c["legs"]],
            })
    _save(keep + today)
    return today


def settle_coupons(pred_log: pd.DataFrame, path: Path | None = None) -> int:
    """Grades legs whose final score is in the prediction log. Returns coupons settled."""
    log = load_coupons(path)
    if not log or "actual_home_pts" not in pred_log.columns:
        return 0
    done = pred_log[pred_log["actual_home_pts"].notna() & pred_log["actual_away_pts"].notna()]
    scores = {str(g): (float(h), float(a)) for g, h, a in
              zip(done["game_id"], done["actual_home_pts"], done["actual_away_pts"])}

    settled, changed = 0, False
    for c in log:
        if c.get("result"):
            continue
        for leg in c["legs"]:
            if leg.get("won") is None and leg["game_id"] in scores:
                h, a = scores[leg["game_id"]]
                leg["won"] = leg_won(leg, h, a)
                leg["score"] = [h, a]
                changed = True
        results = [l.get("won") for l in c["legs"]]
        if False in results:
            c["result"] = "lost"
        elif all(r is True for r in results):
            c["result"] = "won"
        if c.get("result"):
            c["locked"] = True
            c["settled_at"] = _now()
            settled += 1
    if changed or settled:
        _save(log, path)
    return settled


def set_played_odds(coupon_id: str, odds: float | None, path: Path | None = None) -> dict:
    """Records (or clears) the total odds the user actually played; locks the coupon."""
    if odds is not None and not odds > 1:
        raise ValueError("oran 1'den büyük olmalı")
    log = load_coupons(path)
    for c in log:
        if c["id"] == coupon_id:
            c["played_odds"] = None if odds is None else round(float(odds), 2)
            # Oran kaydı kilitler; silmek kilidi açar (maç başladıysa 05 yeniden kilitler)
            c["locked"] = odds is not None or bool(c.get("result"))
            _save(log, path)
            return c
    raise KeyError(coupon_id)


def write_live_view(date: str, day: dict, coupons_today: list[dict]) -> None:
    """What the coupon page shows: the coupons that will be graded + why games were picked."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    view = {
        "date": date,
        "generated_at": _now(),
        "n_legs": day.get("n_legs", 0),
        "note": day.get("note"),
        "skipped": day.get("skipped", []),
        "games": day.get("games", []),
        "coupons": coupons_today,
    }
    (LIVE_DIR / "coupons.json").write_text(json.dumps(view, indent=2, ensure_ascii=False), encoding="utf-8")
