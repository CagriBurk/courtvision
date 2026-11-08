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


SIX = (14.0, -13.0, 9.0, -8.0, 4.5, -4.0)   # two big, two medium, two slight favourites
ALL = {f"G{i}" for i in range(6)}


def day_of(*margins, locked=None):
    preds = [pred(f"G{i}", m) for i, m in enumerate(margins)]
    return preds, build_coupons(preds, DIST, locked=locked)


def test_unlocked_coupons_are_rebuilt_locked_ones_kept(paths):
    _, day = day_of(*SIX)
    first = coupon_tracker.save_coupons("2026-11-08", day, started_ids=set())
    assert [c["id"] for c in first] == ["2026-11-08:guvenli", "2026-11-08:dengeli", "2026-11-08:riskli"]
    assert not any(c["locked"] for c in first)

    # Later refresh: injury news moved the lines, nothing has started → replaced
    _, later = day_of(17.0, *SIX[1:])
    second = coupon_tracker.save_coupons("2026-11-08", later, started_ids=set())
    assert second[0]["legs"][0]["label"] != first[0]["legs"][0]["label"]

    # The user played the safe coupon → it no longer changes, and its games stay out of the rest
    coupon_tracker.set_played_odds("2026-11-08:guvenli", 3.40)
    locked = coupon_tracker.locked_games("2026-11-08", set())
    assert locked == {"guvenli": {"G0", "G1"}}
    _, again = day_of(19.0, -13.0, 10.0, -8.0, 4.5, -4.0, locked=locked)
    assert "guvenli" not in [c["tier"] for c in again["coupons"]]
    assert not {"G0", "G1"} & {l["game_id"] for c in again["coupons"] for l in c["legs"]}
    third = {c["tier"]: c for c in coupon_tracker.save_coupons("2026-11-08", again, set())}
    assert third["guvenli"]["legs"] == second[0]["legs"] and third["guvenli"]["played_odds"] == 3.4
    assert third["dengeli"]["legs"] != second[1]["legs"]

    # Games started → the coupons holding them are locked
    fourth = coupon_tracker.save_coupons("2026-11-08", {"coupons": []}, started_ids=ALL)
    assert all(c["locked"] for c in fourth) and len(fourth) == 3

    # Other days are untouched
    coupon_tracker.save_coupons("2026-11-09", again, set())
    assert {c["date"] for c in coupon_tracker.load_coupons()} == {"2026-11-08", "2026-11-09"}

    with pytest.raises(ValueError):
        coupon_tracker.set_played_odds("2026-11-08:riskli", 0.9)


def test_clearing_played_odds_unlocks_until_a_game_starts(paths):
    _, day = day_of(*SIX)
    coupon_tracker.save_coupons("2026-11-08", day, set())
    coupon_tracker.set_played_odds("2026-11-08:guvenli", 3.1)
    c = coupon_tracker.set_played_odds("2026-11-08:guvenli", None)
    assert c["played_odds"] is None and c["locked"] is False
    _, later = day_of(17.0, *SIX[1:])
    rebuilt = {c["tier"]: c for c in coupon_tracker.save_coupons("2026-11-08", later, set())}
    assert rebuilt["guvenli"]["legs"] != c["legs"]          # unlocked → rebuilt
    relocked = {x["tier"]: x for x in coupon_tracker.save_coupons("2026-11-08", later, {"G0"})}
    assert relocked["guvenli"]["locked"]                    # its game started → locked again
    assert not relocked["dengeli"]["locked"]                # G0 is not in the balanced coupon


def test_coupons_settle_when_scores_arrive(paths):
    preds, day = day_of(*SIX)
    coupon_tracker.save_coupons("2026-11-08", day, set())
    prediction_tracker.save_predictions(preds)

    safe = next(c for c in coupon_tracker.load_coupons() if c["tier"] == "guvenli")
    legs = {l["game_id"]: l for l in safe["legs"]}
    master = pd.DataFrame({
        "game_id": [f"G{i}" for i in range(6)], "target": [1, 0, 1, 0, 0, 1],
        "pts_home": [130, 95, 110, 100, 101, 99], "pts_away": [100, 125, 104, 112, 103, 104],
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
    preds, day = day_of(14.0, -13.0)
    assert day["coupons"]
    coupon_tracker.save_coupons("2026-11-08", day, set())
    prediction_tracker.save_predictions(preds)
    # Only G0 finished, and the favourite lost
    prediction_tracker.fill_results(pd.DataFrame(
        {"game_id": ["G0"], "target": [0], "pts_home": [99], "pts_away": [104]}))
    for c in coupon_tracker.load_coupons():
        assert c["result"] == "lost"
        assert next(l for l in c["legs"] if l["game_id"] == "G1")["won"] is None


def test_predict_today_writes_the_live_view(paths):
    preds = [pred(f"G{i}", m) for i, m in enumerate([*SIX, 1.0])]
    predict_today.update_coupons(preds, {"margin": DIST}, pd.Timestamp("2026-11-08").date())
    view = json.loads((paths / "live" / "coupons.json").read_text(encoding="utf-8"))
    assert view["date"] == "2026-11-08" and view["n_legs"] == 2
    assert len(view["coupons"]) == 3 and view["coupons"][0]["id"] == "2026-11-08:guvenli"
    assert len(view["games"]) == 7  # every game with its reasons, chosen or not
    assert "piyasa çizgisi yok" in view["note"]   # no market line: model-priced, and it says so

    predict_today.update_coupons(preds, {}, pd.Timestamp("2026-11-09").date())
    view = json.loads((paths / "live" / "coupons.json").read_text(encoding="utf-8"))
    assert view["coupons"] == [] and "Handikap dağılımı yok" in view["note"]
