"""
espn_odds.py
------------
Market lines (DraftKings via ESPN's unofficial API) for NBA games.

Sign convention: ESPN's `spread` is the HOME team's handicap ("CHI +9.5" →
9.5). Our margin is home − away, so the market's expected home margin is
`market_margin = -spread_home`.

Sources:
  - today / upcoming games: the scoreboard (`competitions[0].odds[0]`)
  - past games: the scoreboard no longer carries odds, so the core API's
    per-event odds are used (DraftKings, or ESPN BET before ~Dec 2025), with
    the event summary's `pickcenter` as a fallback (cached, resumable)

Output:
  data/processed/market_lines.parquet  (one row per game_id, upserted)

Usage:
    python src/espn_odds.py --backfill --season 2025-26          # past lines
    python src/espn_odds.py --backfill --season 2025-26 2026-27
    python src/espn_odds.py --date 2026-11-08                    # print a day's lines
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent
PROC_DIR   = ROOT / "data" / "processed"
LINES_PATH = PROC_DIR / "market_lines.parquet"
ESPN_CACHE = ROOT / "data" / "api_cache" / "espn"

BASE_URL       = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SCOREBOARD_URL = BASE_URL + "/scoreboard?dates={ymd}"
SUMMARY_URL    = BASE_URL + "/summary?event={event_id}"
# Per-provider odds of an event. Until ~Dec 2025 ESPN's provider was ESPN BET,
# whose lines only live here (the summary's pickcenter is empty for them).
CORE_ODDS_URL  = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
                  "/events/{event_id}/competitions/{event_id}/odds")

REQUEST_PAUSE     = 0.35   # seconds between uncached requests (per worker)
RETRIES           = 3
PARALLEL_REQUESTS = 6      # backfill: concurrent requests within one day

# ESPN abbreviations that differ from NBA tricodes; the rest are identical.
ESPN_TO_NBA: dict[str, str] = {
    "GS":   "GSW",
    "NO":   "NOP",
    "NY":   "NYK",
    "SA":   "SAS",
    "UTAH": "UTA",
    "WSH":  "WAS",
}

# Older NBA tricodes in master_games → today's franchise tricode, so that a
# game can be matched whatever era it comes from.
LEGACY_TO_CURRENT: dict[str, str] = {
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
}

FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FINAL_OT"}

COLUMNS = [
    "game_id", "game_date", "home_team_abbr", "away_team_abbr", "espn_event_id",
    "spread_home", "market_margin", "total",
    "spread_odds_home", "spread_odds_away", "over_odds", "under_odds",
    "ml_home", "ml_away", "p_home_novig",
    "spread_home_open", "total_open",
    "provider", "source", "fetched_at",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def nba_abbr(espn: str) -> str:
    """ESPN team abbreviation → NBA tricode."""
    espn = (espn or "").upper()
    return ESPN_TO_NBA.get(espn, espn)


def canon_abbr(abbr: str) -> str:
    """NBA tricode (any era) → current franchise tricode."""
    abbr = (abbr or "").upper()
    return LEGACY_TO_CURRENT.get(abbr, abbr)


def parse_american(value) -> float | None:
    """American odds ("+329", "-420", 190, "EVEN") → float; None if missing/off."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value != 0 else None
    s = str(value).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100.0
    try:
        v = float(s.replace("+", ""))
    except ValueError:
        return None   # "OFF", "", "-"
    return v if v != 0 else None


def implied_prob(american: float | None) -> float | None:
    """Break-even probability of American odds (includes the bookmaker's margin)."""
    if american is None:
        return None
    if american < 0:
        return -american / (-american + 100.0)
    return 100.0 / (american + 100.0)


def no_vig_home_prob(ml_home: float | None, ml_away: float | None) -> float | None:
    """Home win probability from both moneylines with the margin removed."""
    ph, pa = implied_prob(ml_home), implied_prob(ml_away)
    if ph is None or pa is None or ph + pa <= 0:
        return None
    return ph / (ph + pa)


def _num(value) -> float | None:
    """Line value ("+9.5", "o237.5", -5.5) → float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def _first_num(*values) -> float | None:
    for v in values:
        n = _num(v)
        if n is not None:
            return n
    return None


def _get(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse_odds_entry(entry: dict | None) -> dict | None:
    """
    One ESPN odds entry (scoreboard `odds[i]` or summary `pickcenter[i]`;
    both share this shape) → market line dict. None when neither a spread
    nor a total is offered.
    """
    if not entry:
        return None

    spread = _num(_get(entry, "pointSpread", "home", "close", "line"))
    if spread is None:
        spread = _num(entry.get("spread"))
    total = _num(_get(entry, "total", "over", "close", "line"))
    if total is None:
        total = _num(entry.get("overUnder"))
    if spread is None and total is None:
        return None

    ml_home = parse_american(_get(entry, "moneyline", "home", "close", "odds"))
    if ml_home is None:
        ml_home = parse_american(_get(entry, "homeTeamOdds", "moneyLine"))
    ml_away = parse_american(_get(entry, "moneyline", "away", "close", "odds"))
    if ml_away is None:
        ml_away = parse_american(_get(entry, "awayTeamOdds", "moneyLine"))

    def odds(path: tuple, fallback):
        v = parse_american(_get(entry, *path))
        return v if v is not None else parse_american(fallback)

    return {
        "spread_home":      spread,
        "market_margin":    -spread if spread is not None else None,
        "total":            total,
        "spread_odds_home": odds(("pointSpread", "home", "close", "odds"),
                                 _get(entry, "homeTeamOdds", "spreadOdds")),
        "spread_odds_away": odds(("pointSpread", "away", "close", "odds"),
                                 _get(entry, "awayTeamOdds", "spreadOdds")),
        "over_odds":        odds(("total", "over", "close", "odds"), entry.get("overOdds")),
        "under_odds":       odds(("total", "under", "close", "odds"), entry.get("underOdds")),
        "ml_home":          ml_home,
        "ml_away":          ml_away,
        "p_home_novig":     no_vig_home_prob(ml_home, ml_away),
        "spread_home_open": _first_num(_get(entry, "pointSpread", "home", "open", "line"),
                                       _get(entry, "homeTeamOdds", "open", "pointSpread", "american")),
        "total_open":       _first_num(_get(entry, "total", "over", "open", "line"),
                                       _get(entry, "open", "total", "american")),
        "provider":         _get(entry, "provider", "name") or "",
    }


def pick_entry(entries: list | None) -> dict | None:
    """
    Prefer DraftKings, otherwise the highest-priority provider with a line.
    In-game feeds ("ESPN Bet - Live Odds") are not pre-game lines and are skipped.
    """
    entries = [e for e in (entries or [])
               if parse_odds_entry(e) and "live" not in (_get(e, "provider", "name") or "").lower()]
    if not entries:
        return None
    for e in entries:
        if (_get(e, "provider", "name") or "").lower() == "draftkings":
            return e
    return min(entries, key=lambda e: _get(e, "provider", "priority") or 99)


def parse_scoreboard(data: dict) -> list[dict]:
    """
    Scoreboard response → one dict per event:
    {event_id, status, home_team_abbr, away_team_abbr, odds (dict | None)}.
    """
    events = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): nba_abbr(_get(c, "team", "abbreviation"))
                 for c in comp.get("competitors", [])}
        if not teams.get("home") or not teams.get("away"):
            continue
        events.append({
            "event_id":       str(ev.get("id")),
            "status":         _get(comp, "status", "type", "name") or _get(ev, "status", "type", "name"),
            "home_team_abbr": teams["home"],
            "away_team_abbr": teams["away"],
            "odds":           parse_odds_entry(pick_entry(comp.get("odds"))),
        })
    return events


# ---------------------------------------------------------------------------
# HTTP + cache
# ---------------------------------------------------------------------------
def _get_json(url: str) -> dict:
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:   # network hiccup / rate limit → back off and retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN isteği başarısız: {url} ({last})")


def _slim_scoreboard(data: dict) -> dict:
    """Keep only what parse_scoreboard needs (full responses are ~100 KB/day)."""
    events = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        events.append({
            "id": ev.get("id"),
            "competitions": [{
                "status": {"type": {"name": _get(comp, "status", "type", "name")}},
                "competitors": [{"homeAway": c.get("homeAway"),
                                 "team": {"abbreviation": _get(c, "team", "abbreviation")}}
                                for c in comp.get("competitors", [])],
                "odds": [_strip_links(o) for o in comp.get("odds") or []],
            }],
        })
    return {"events": events}


def _strip_links(x):
    if isinstance(x, dict):
        return {k: _strip_links(v) for k, v in x.items()
                if k not in ("link", "links", "logos", "logo", "header", "footer")}
    if isinstance(x, list):
        return [_strip_links(v) for v in x]
    return x


def fetch_scoreboard(day: date, cache: bool, cache_dir: Path = ESPN_CACHE) -> dict:
    """Scoreboard for a US Eastern date. Cached only when `cache` (finished days)."""
    path = cache_dir / "scoreboard" / f"{day:%Y%m%d}.json"
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _slim_scoreboard(_get_json(SCOREBOARD_URL.format(ymd=f"{day:%Y%m%d}")))
    time.sleep(REQUEST_PAUSE)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_pickcenter(event_id: str, cache_dir: Path = ESPN_CACHE) -> list:
    """Summary `pickcenter` of a finished game (cached; the summary itself is large)."""
    path = cache_dir / "summary" / f"{event_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("pickcenter", [])
    data = _get_json(SUMMARY_URL.format(event_id=event_id))
    time.sleep(REQUEST_PAUSE)
    pickcenter = _strip_links(data.get("pickcenter") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pickcenter": pickcenter}), encoding="utf-8")
    return pickcenter


def fetch_core_odds(event_id: str, cache_dir: Path = ESPN_CACHE) -> list:
    """Core API odds items of a finished game (small response, cached)."""
    path = cache_dir / "core" / f"{event_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("items", [])
    data = _get_json(CORE_ODDS_URL.format(event_id=event_id))
    time.sleep(REQUEST_PAUSE)
    items = [{k: v for k, v in _strip_links(i).items() if k not in ("propBets", "$ref")}
             for i in data.get("items") or []]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    return items


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def load_market_lines(path: Path = LINES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(path)
    df["game_id"] = df["game_id"].astype(str)
    return df


@contextmanager
def _file_lock(path: Path, timeout: float = 120.0, stale: float = 300.0):
    """
    Cross-process lock next to `path`, so several backfill terminals (and 05)
    can update the same table without one overwriting the other's rows.
    """
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:   # holder crashed
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Kilit alınamadı: {lock}")
            time.sleep(0.05 + random.random() * 0.1)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _replace(src: Path, dst: Path) -> None:
    """os.replace with retries: on Windows a reader holding `dst` blocks it briefly."""
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def upsert_lines(rows: list[dict], path: Path = LINES_PATH) -> pd.DataFrame:
    """Insert/replace rows by game_id; returns the stored table."""
    if not rows:
        return load_market_lines(path)
    new = pd.DataFrame(rows).reindex(columns=COLUMNS)
    new["game_id"] = new["game_id"].astype(str)
    with _file_lock(path):
        existing = load_market_lines(path)
        keep = existing[~existing["game_id"].isin(new["game_id"])]
        out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
        out["game_date"] = pd.to_datetime(out["game_date"])
        out = out.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        out.to_parquet(tmp, index=False)
        _replace(tmp, path)
    return out


def _row(game_id: str, day: date, ev: dict, odds: dict, source: str) -> dict:
    return {
        "game_id": str(game_id), "game_date": pd.Timestamp(day),
        "home_team_abbr": ev["home_team_abbr"], "away_team_abbr": ev["away_team_abbr"],
        "espn_event_id": ev["event_id"], **odds,
        "source": source, "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Today's lines (called from 05_predict_today.py, one request per run)
# ---------------------------------------------------------------------------
def update_today_lines(target_date: date, games: list[dict],
                       path: Path = LINES_PATH) -> dict[str, dict]:
    """
    Fetch the scoreboard for `target_date`, match its events to `games`
    (05's schedule: game_id + home/away tricodes) and store the lines of
    games that have NOT started — once a game tips off, ESPN's line is no
    longer the pre-game one, so the stored row is kept as it was.

    Returns {game_id: line dict} for every game with a stored line.
    """
    events = parse_scoreboard(fetch_scoreboard(target_date, cache=False))
    by_teams = {(canon_abbr(e["home_team_abbr"]), canon_abbr(e["away_team_abbr"])): e
                for e in events}

    rows = []
    for g in games:
        ev = by_teams.get((canon_abbr(g.get("home_team_abbr")), canon_abbr(g.get("away_team_abbr"))))
        if ev is None or ev["odds"] is None or ev["status"] != "STATUS_SCHEDULED":
            continue
        rows.append(_row(g["game_id"], target_date, ev, ev["odds"], "scoreboard"))

    table = upsert_lines(rows, path)
    ids = {str(g["game_id"]) for g in games}
    stored = table[table["game_id"].isin(ids)]
    return {r["game_id"]: r for r in stored.to_dict("records")}


# ---------------------------------------------------------------------------
# Backfill (past seasons; run by the user, resumable through the cache)
# ---------------------------------------------------------------------------
def master_index(master: pd.DataFrame) -> dict[tuple, str]:
    """(date, home, away) with current franchise tricodes → NBA game_id."""
    idx = {}
    for gid, d, h, a in zip(master["game_id"].astype(str), pd.to_datetime(master["game_date"]),
                            master["team_abbreviation_home"], master["team_abbreviation_away"]):
        idx[(d.date(), canon_abbr(h), canon_abbr(a))] = gid
    return idx


def backfill_day(day: date, index: dict[tuple, str],
                 cache_dir: Path = ESPN_CACHE) -> tuple[list[dict], int, int]:
    """Lines of one finished day. Returns (rows, matched games, games without a line)."""
    games = []
    for ev in parse_scoreboard(fetch_scoreboard(day, cache=True, cache_dir=cache_dir)):
        gid = index.get((day, canon_abbr(ev["home_team_abbr"]), canon_abbr(ev["away_team_abbr"])))
        if gid is not None and ev["status"] in FINAL_STATUSES:
            games.append((gid, ev))   # others: All-Star / exhibition / postponed

    # A request takes ~1 s, so the day's games are fetched in parallel
    need_core = [ev["event_id"] for _, ev in games if ev["odds"] is None]
    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as pool:
        core = dict(zip(need_core, pool.map(lambda eid: fetch_core_odds(eid, cache_dir), need_core)))

    rows, matched, missing = [], len(games), 0
    for gid, ev in games:
        odds, source = ev["odds"], "scoreboard"
        if odds is None:
            odds = parse_odds_entry(pick_entry(core[ev["event_id"]]))
            source = "core"
        if odds is None:   # the large summary only as a last resort
            odds = parse_odds_entry(pick_entry(fetch_pickcenter(ev["event_id"], cache_dir)))
            source = "summary"
        if odds is None:
            missing += 1
            continue
        rows.append(_row(gid, day, ev, odds, source))
    return rows, matched, missing


def backfill(seasons: list[str], master: pd.DataFrame, today: date,
             path: Path = LINES_PATH, cache_dir: Path = ESPN_CACHE) -> pd.DataFrame:
    """Download past lines for the given seasons ('2025-26') and store them."""
    table = load_market_lines(path)
    for season in seasons:
        sid = f"2{int(season[:4])}"
        games = master[master["season_id"].astype(str) == sid]
        days = sorted(d.date() for d in pd.to_datetime(games["game_date"]).unique())
        days = [d for d in days if d < today]   # today's games are not final yet
        index = master_index(games)
        print(f"\n{season}: {len(games)} maç, {len(days)} gün")

        n_rows = n_missing = n_matched = 0
        for i, day in enumerate(days, 1):
            try:
                rows, matched, missing = backfill_day(day, index, cache_dir)
            except RuntimeError as e:
                print(f"  [ATLANDI] {day}: {e} — tekrar çalıştırınca kaldığı yerden devam eder")
                continue
            table = upsert_lines(rows, path)
            n_rows += len(rows)
            n_missing += missing
            n_matched += matched
            print(f"  [{i:>3}/{len(days)}] {day}: {matched} maç eşleşti, {len(rows)} oran"
                  + (f", {missing} oransız" if missing else ""))

        print(f"{season} bitti: {n_rows} oran / {len(games)} maç "
              f"(eşleşmeyen {len(games) - n_matched}, oransız {n_missing})")
    print(f"\nKaydedildi → {path} ({len(table)} satır)")
    return table


# ---------------------------------------------------------------------------
def _print_day(day: date) -> None:
    events = parse_scoreboard(fetch_scoreboard(day, cache=False))
    print(f"{day}: {len(events)} maç")
    for e in events:
        o = e["odds"]
        if o is None:
            print(f"  {e['away_team_abbr']:>3} @ {e['home_team_abbr']:<3}  oran yok ({e['status']})")
            continue
        p = o["p_home_novig"]
        print(f"  {e['away_team_abbr']:>3} @ {e['home_team_abbr']:<3}  ev handikap {o['spread_home']:+5.1f}"
              f"  toplam {o['total']}  ML {o['ml_home']}/{o['ml_away']}"
              f"  ev kazanır {'-' if p is None else f'%{100 * p:.1f}'}  [{o['provider']}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="ESPN (DraftKings) piyasa çizgileri")
    parser.add_argument("--backfill", action="store_true", help="geçmiş sezon(lar)ın çizgilerini indir")
    parser.add_argument("--season", nargs="+", default=[], help="ör. 2025-26 2026-27")
    parser.add_argument("--date", help="bir günün çizgilerini yazdır (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.backfill:
        if not args.season:
            parser.error("--backfill için --season gerekli (ör. --season 2025-26)")
        from season import nba_today
        master = pd.read_parquet(PROC_DIR / "master_games.parquet")
        backfill(args.season, master, nba_today())
    elif args.date:
        _print_day(date.fromisoformat(args.date))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
