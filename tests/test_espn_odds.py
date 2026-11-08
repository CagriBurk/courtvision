import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import espn_odds
from espn_odds import (
    backfill,
    canon_abbr,
    load_market_lines,
    nba_abbr,
    no_vig_home_prob,
    parse_american,
    parse_odds_entry,
    parse_scoreboard,
    pick_entry,
    update_today_lines,
    upsert_lines,
)

# All 30 abbreviations from ESPN's /teams endpoint (8 Nov 2026)
ESPN_ABBRS = ["ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND",
              "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX",
              "POR", "SA", "SAC", "TOR", "UTAH", "WSH"]
NBA_ABBRS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND",
             "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
             "POR", "SAS", "SAC", "TOR", "UTA", "WAS"}


def scoreboard_entry(spread, total, ml_home, ml_away, details=""):
    """Scoreboard `odds[0]` shape (8 Nov 2026 PHI@CHI, trimmed)."""
    home_line = f"{spread:+.1f}"
    away_line = f"{-spread:+.1f}"
    return {
        "provider": {"id": "100", "name": "DraftKings", "priority": 1},
        "details": details, "overUnder": total, "spread": spread,
        "moneyline": {"home": {"close": {"odds": ml_home}, "open": {"odds": "+251"}},
                      "away": {"close": {"odds": ml_away}, "open": {"odds": "-311"}}},
        "pointSpread": {"home": {"close": {"line": home_line, "odds": "-115"},
                                 "open": {"line": home_line, "odds": "-110"}},
                        "away": {"close": {"line": away_line, "odds": "-105"},
                                 "open": {"line": away_line, "odds": "-110"}}},
        "total": {"over": {"close": {"line": f"o{total}", "odds": "-110"},
                           "open": {"line": "o238.5", "odds": "-107"}},
                  "under": {"close": {"line": f"u{total}", "odds": "-110"},
                            "open": {"line": "u238.5", "odds": "-113"}}},
    }


# Summary `pickcenter[0]` of event 401810433 (MEM @ ORL, 15 Jan 2026), trimmed
PICKCENTER_ORL = {
    "provider": {"id": "100", "name": "DraftKings", "priority": 1},
    "details": "ORL -5.5", "overUnder": 227.5, "spread": -5.5,
    "overOdds": -108.0, "underOdds": -112.0,
    "awayTeamOdds": {"favorite": False, "moneyLine": 190, "spreadOdds": -112.0, "teamId": "29"},
    "homeTeamOdds": {"favorite": True, "moneyLine": -230, "spreadOdds": -108.0, "teamId": "19"},
}


# Core API odds items of event 401809234 (CLE @ NYK, Oct 2025), trimmed:
# the pre-game ESPN BET line plus an in-game feed that must be ignored.
CORE_ESPN_BET = {
    "provider": {"id": "58", "name": "ESPN BET", "priority": 0},
    "details": "CLE -1.5", "overUnder": 228.5, "spread": 1.5,
    "overOdds": -115.0, "underOdds": -105.0,
    "homeTeamOdds": {"moneyLine": 100, "spreadOdds": -110.0,
                     "open": {"pointSpread": {"american": "-3.5"}}},
    "awayTeamOdds": {"moneyLine": -120, "spreadOdds": -110.0,
                     "open": {"pointSpread": {"american": "+3.5"}}},
    "open": {"total": {"american": "230.5"}},
}
CORE_LIVE = {
    "provider": {"id": "59", "name": "ESPN Bet - Live Odds", "priority": 0},
    "details": "NY -6.5", "overUnder": 229.5, "spread": -6.5,
}


def event(eid, home, away, odds=None, status="STATUS_SCHEDULED"):
    return {"id": eid, "competitions": [{
        "status": {"type": {"name": status}},
        "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                        {"homeAway": "away", "team": {"abbreviation": away}}],
        "odds": [odds] if odds else [],
    }]}


# ---------------------------------------------------------------------------
def test_abbreviation_mapping_covers_all_teams():
    assert {nba_abbr(a) for a in ESPN_ABBRS} == NBA_ABBRS
    assert nba_abbr("UTAH") == "UTA" and nba_abbr("NO") == "NOP" and nba_abbr("GS") == "GSW"
    assert nba_abbr("BOS") == "BOS"
    # Older tricodes in master_games fold into today's franchise
    assert canon_abbr("NJN") == "BKN" and canon_abbr("NOH") == "NOP" and canon_abbr("LAL") == "LAL"


def test_american_odds_and_no_vig():
    assert parse_american("+329") == 329 and parse_american("-420") == -420
    assert parse_american(190) == 190 and parse_american("EVEN") == 100
    assert parse_american("OFF") is None and parse_american(None) is None and parse_american("") is None
    p = no_vig_home_prob(-230, 190)
    raw_home, raw_away = 230 / 330, 100 / 290
    assert p == pytest.approx(raw_home / (raw_home + raw_away))
    assert no_vig_home_prob(-110, -110) == pytest.approx(0.5)
    assert no_vig_home_prob(None, 190) is None


def test_sign_convention_away_favourite():
    # PHI @ CHI, "PHI -9.5": ESPN spread is the home handicap (+9.5)
    o = parse_odds_entry(scoreboard_entry(9.5, 237.5, "+329", "-420", "PHI -9.5"))
    assert o["spread_home"] == 9.5
    assert o["market_margin"] == -9.5          # home − away expected: CHI loses by 9.5
    assert o["total"] == 237.5
    assert o["ml_home"] == 329 and o["ml_away"] == -420
    assert o["p_home_novig"] < 0.25
    assert o["spread_odds_home"] == -115 and o["over_odds"] == -110
    assert o["spread_home_open"] == 9.5 and o["total_open"] == 238.5
    assert o["provider"] == "DraftKings"


def test_sign_convention_home_favourite():
    # BKN @ DAL, "DAL -7.5"
    o = parse_odds_entry(scoreboard_entry(-7.5, 222.5, "-310", "+250", "DAL -7.5"))
    assert o["spread_home"] == -7.5 and o["market_margin"] == 7.5
    assert o["p_home_novig"] > 0.7


def test_pickcenter_shape():
    o = parse_odds_entry(PICKCENTER_ORL)
    assert o["spread_home"] == -5.5 and o["market_margin"] == 5.5 and o["total"] == 227.5
    assert o["ml_home"] == -230 and o["ml_away"] == 190
    assert o["spread_odds_home"] == -108 and o["spread_odds_away"] == -112
    assert o["over_odds"] == -108 and o["under_odds"] == -112
    assert o["spread_home_open"] is None


def test_core_item_shape_and_live_feed_skipped():
    chosen = pick_entry([CORE_LIVE, CORE_ESPN_BET])
    assert chosen["provider"]["name"] == "ESPN BET"
    o = parse_odds_entry(chosen)
    assert o["spread_home"] == 1.5 and o["market_margin"] == -1.5 and o["total"] == 228.5
    assert o["ml_home"] == 100 and o["ml_away"] == -120
    assert o["over_odds"] == -115 and o["spread_odds_home"] == -110
    assert o["spread_home_open"] == -3.5 and o["total_open"] == 230.5
    assert pick_entry([CORE_LIVE]) is None


def test_missing_odds():
    assert parse_odds_entry(None) is None
    assert parse_odds_entry({}) is None
    assert parse_odds_entry({"provider": {"name": "DraftKings"}, "details": "OFF"}) is None
    # A total without a spread is still a line
    o = parse_odds_entry({"overUnder": 221.5})
    assert o["total"] == 221.5 and o["spread_home"] is None and o["market_margin"] is None
    assert o["p_home_novig"] is None
    # Pick'em: zero spread, not missing
    assert parse_odds_entry({"spread": 0.0, "overUnder": 230})["market_margin"] == 0
    assert pick_entry([]) is None and pick_entry([{"provider": {"name": "X"}}]) is None


def test_pick_entry_prefers_draftkings():
    other = dict(PICKCENTER_ORL, provider={"name": "Other", "priority": 0}, spread=-9.0)
    assert pick_entry([other, PICKCENTER_ORL])["spread"] == -5.5
    assert pick_entry([other])["spread"] == -9.0


def test_parse_scoreboard():
    data = {"events": [
        event("1", "MIN", "UTAH", scoreboard_entry(-11.5, 223.5, "-700", "+500")),
        event("2", "LAC", "NO"),                              # no odds yet
        {"id": "3", "competitions": [{"competitors": []}]},   # malformed → skipped
    ]}
    evs = parse_scoreboard(data)
    assert [e["event_id"] for e in evs] == ["1", "2"]
    assert evs[0]["home_team_abbr"] == "MIN" and evs[0]["away_team_abbr"] == "UTA"
    assert evs[0]["odds"]["market_margin"] == 11.5
    assert evs[1]["away_team_abbr"] == "NOP" and evs[1]["odds"] is None


def test_upsert_replaces_by_game_id(tmp_path):
    path = tmp_path / "lines.parquet"
    base = {"game_date": pd.Timestamp("2026-11-08"), "home_team_abbr": "CHI", "away_team_abbr": "PHI"}
    upsert_lines([dict(base, game_id="A", spread_home=9.5), dict(base, game_id="B", spread_home=1.0)], path)
    table = upsert_lines([dict(base, game_id="A", spread_home=8.5)], path)
    assert len(table) == 2
    assert table.set_index("game_id").loc["A", "spread_home"] == 8.5
    assert list(load_market_lines(path).columns) == espn_odds.COLUMNS
    assert upsert_lines([], path).shape[0] == 2


def test_update_today_lines_keeps_pregame_line(tmp_path, monkeypatch):
    path = tmp_path / "lines.parquet"
    day = date(2026, 11, 8)
    games = [
        {"game_id": "G1", "home_team_abbr": "CHI", "away_team_abbr": "PHI"},
        {"game_id": "G2", "home_team_abbr": "SAC", "away_team_abbr": "MIA"},
        {"game_id": "G3", "home_team_abbr": "LAC", "away_team_abbr": "NOP"},
    ]
    boards = iter([
        {"events": [event("1", "CHI", "PHI", scoreboard_entry(9.5, 237.5, "+329", "-420")),
                    event("2", "SAC", "MIA", scoreboard_entry(10.5, 238.5, "+400", "-550")),
                    event("3", "LAC", "NO")]},
        # Later run: CHI-PHI has tipped off and ESPN shows a live line
        {"events": [event("1", "CHI", "PHI", scoreboard_entry(2.5, 210.5, "+120", "-140"),
                          status="STATUS_IN_PROGRESS"),
                    event("2", "SAC", "MIA", scoreboard_entry(11.5, 238.5, "+420", "-600")),
                    event("3", "LAC", "NO")]},
    ])
    monkeypatch.setattr(espn_odds, "fetch_scoreboard", lambda d, cache, cache_dir=None: next(boards))

    first = update_today_lines(day, games, path)
    assert set(first) == {"G1", "G2"}                 # G3 has no line
    second = update_today_lines(day, games, path)
    assert second["G1"]["spread_home"] == 9.5         # pre-game line kept
    assert second["G2"]["spread_home"] == 11.5        # moved line updated
    assert second["G1"]["source"] == "scoreboard"


def test_backfill_matches_master_and_resumes(tmp_path, monkeypatch):
    path, cache = tmp_path / "lines.parquet", tmp_path / "cache"
    master = pd.DataFrame({
        "game_id": ["0022500001", "0022500002", "0022500003"],
        "season_id": ["22025"] * 3,
        "game_date": pd.to_datetime(["2026-01-15", "2026-01-15", "2026-01-16"]),
        "team_abbreviation_home": ["ORL", "UTA", "GSW"],
        "team_abbreviation_away": ["MEM", "NOP", "LAL"],
    })
    boards = {
        "20260115": {"events": [event("401810433", "ORL", "MEM", status="STATUS_FINAL"),
                                event("401810434", "UTAH", "NO", status="STATUS_FINAL"),
                                event("999", "WORLD", "USA", status="STATUS_FINAL")]},  # not in master
        "20260116": {"events": [event("401810440", "GS", "LAL", status="STATUS_FINAL")]},
    }
    core = {
        "401810433": {"items": [CORE_LIVE, PICKCENTER_ORL]},
        "401810434": {"items": []},
        "401810440": {"items": [CORE_LIVE]},             # only an in-game feed
    }
    summaries = {
        "401810434": {"pickcenter": []},                 # ESPN has no line for this game
        "401810440": {"pickcenter": [dict(PICKCENTER_ORL, spread=-3.0, overUnder=231.0)]},
    }
    calls = []

    def fake_get(url):
        calls.append(url)
        if "scoreboard" in url:
            return boards[url.rsplit("=", 1)[1]]
        if "sports.core" in url:
            return core[url.split("/events/")[1].split("/")[0]]
        return summaries[url.rsplit("=", 1)[1]]

    monkeypatch.setattr(espn_odds, "_get_json", fake_get)
    monkeypatch.setattr(espn_odds, "REQUEST_PAUSE", 0)

    table = backfill(["2025-26"], master, today=date(2026, 11, 8), path=path, cache_dir=cache)
    lines = table.set_index("game_id")
    assert set(lines.index) == {"0022500001", "0022500003"}
    assert lines.loc["0022500001", "market_margin"] == 5.5
    assert lines.loc["0022500001", "source"] == "core"
    assert lines.loc["0022500003", "spread_home"] == -3.0
    assert lines.loc["0022500003", "source"] == "summary"
    assert len(calls) == 7                    # 2 scoreboards + 3 core + 2 summaries

    # Second run is served entirely from the cache
    calls.clear()
    backfill(["2025-26"], master, today=date(2026, 11, 8), path=path, cache_dir=cache)
    assert calls == []


def test_backfill_skips_today(tmp_path, monkeypatch):
    master = pd.DataFrame({
        "game_id": ["0022600200"], "season_id": ["22026"],
        "game_date": pd.to_datetime(["2026-11-08"]),
        "team_abbreviation_home": ["CHI"], "team_abbreviation_away": ["PHI"],
    })
    monkeypatch.setattr(espn_odds, "_get_json", lambda url: pytest.fail("no request expected"))
    table = backfill(["2026-27"], master, today=date(2026, 11, 8),
                     path=tmp_path / "l.parquet", cache_dir=tmp_path / "c")
    assert table.empty
