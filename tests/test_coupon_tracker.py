import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import coupon_tracker
import prediction_tracker
from coupons import build_coupons
from test_coupons import DIST, pred

predict_today = importlib.import_module("05_predict_today")


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(coupon_tracker, "COUPON_LOG", tmp_path / "coupons_log.json")
    monkeypatch.setattr(coupon_tracker, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(prediction_tracker, "PRED_DIR", tmp_path)
    monkeypatch.setattr(prediction_tracker, "PRED_LOG", tmp_path / "predictions_log.parquet")
    return tmp_path


def day_of(*margins):
    preds = [pred(f"G{i}", m) for i, m in enumerate(margins)]
    return preds, build_coupons(preds, DIST)


def test_unlocked_coupons_are_rebuilt_locked_ones_kept(paths):
    _, day = day_of(9.0, -8.0, 7.0)
    first = coupon_tracker.save_coupons("2026-11-08", day, started_ids=set())
    assert [c["id"] for c in first] == ["2026-11-08:guvenli", "2026-11-08:dengeli", "2026-11-08:riskli"]
    assert not any(c["locked"] for c in first)

    # Later refresh: injury news moved the lines, nothing has started → replaced
    _, later = day_of(12.0, -8.0, 7.0)
    second = coupon_tracker.save_coupons("2026-11-08", later, started_ids=set())
    assert second[0]["legs"][0]["label"] != first[0]["legs"][0]["label"]

    # The user played the safe coupon → it no longer changes
    coupon_tracker.set_played_odds("2026-11-08:guvenli", 3.40)
    _, again = day_of(15.0, -8.0, 7.0)
    third = {c["tier"]: c for c in coupon_tracker.save_coupons("2026-11-08", again, set())}
    assert third["guvenli"]["legs"] == second[0]["legs"] and third["guvenli"]["played_odds"] == 3.4
    assert third["dengeli"]["legs"] != second[1]["legs"]

    # A game started → every coupon holding it is locked
    fourth = coupon_tracker.save_coupons("2026-11-08", {"coupons": []}, started_ids={"G1"})
    assert all(c["locked"] for c in fourth) and len(fourth) == 3

    # Other days are untouched
    coupon_tracker.save_coupons("2026-11-09", again, set())
    assert {c["date"] for c in coupon_tracker.load_coupons()} == {"2026-11-08", "2026-11-09"}

    with pytest.raises(ValueError):
        coupon_tracker.set_played_odds("2026-11-08:riskli", 0.9)


def test_coupons_settle_when_scores_arrive(paths):
    preds, day = day_of(9.0, -8.0, 7.0)
    coupon_tracker.save_coupons("2026-11-08", day, set())
    prediction_tracker.save_predictions(preds)

    safe = next(c for c in coupon_tracker.load_coupons() if c["tier"] == "guvenli")
    legs = {l["game_id"]: l for l in safe["legs"]}
    # Scores that beat every safe-coupon line except G2's in the risky one
    master = pd.DataFrame({
        "game_id": ["G0", "G1", "G2"], "target": [1, 0, 1],
        "pts_home": [130, 95, 110], "pts_away": [100, 125, 104],
    })
    prediction_tracker.fill_results(master)

    log = {c["tier"]: c for c in coupon_tracker.load_coupons()}
    for leg in log["guvenli"]["legs"]:
        h, a = leg["score"]
        expected = (h - a > leg["line"]) if leg["side"] == "home" else (h - a < leg["line"])
        assert leg["won"] is expected
    assert legs["G1"]["side"] == "away"
    for c in log.values():
        assert c["result"] == ("won" if all(l["won"] for l in c["legs"]) else "lost")
        assert c["locked"] and c["settled_at"]


def test_a_lost_leg_settles_the_coupon_early(paths):
    preds, day = day_of(9.0, -8.0)
    coupon_tracker.save_coupons("2026-11-08", day, set())
    prediction_tracker.save_predictions(preds)
    # Only G0 finished, and the favourite lost
    prediction_tracker.fill_results(pd.DataFrame(
        {"game_id": ["G0"], "target": [0], "pts_home": [99], "pts_away": [104]}))
    for c in coupon_tracker.load_coupons():
        assert c["result"] == "lost"
        assert next(l for l in c["legs"] if l["game_id"] == "G1")["won"] is None


def test_predict_today_writes_the_live_view(paths):
    preds = [pred(f"G{i}", m) for i, m in enumerate([9.0, -8.0, 7.0, 1.0])]
    predict_today.update_coupons(preds, {"margin": DIST}, pd.Timestamp("2026-11-08").date())
    view = json.loads((paths / "live" / "coupons.json").read_text(encoding="utf-8"))
    assert view["date"] == "2026-11-08" and view["n_legs"] == 3
    assert len(view["coupons"]) == 3 and view["coupons"][0]["id"] == "2026-11-08:guvenli"
    assert len(view["games"]) == 4  # every game with its reasons, chosen or not

    predict_today.update_coupons(preds, {}, pd.Timestamp("2026-11-09").date())
    view = json.loads((paths / "live" / "coupons.json").read_text(encoding="utf-8"))
    assert view["coupons"] == [] and "Handikap modeli yok" in view["note"]
