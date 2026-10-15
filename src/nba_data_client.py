"""
nba_data_client.py
------------------
A dedicated module to handle data fetching from the nba_api.
"""

import time
from datetime import date

try:
    from nba_api.stats.endpoints import scoreboardv3
    from nba_api.stats.static import teams as nba_teams_static
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

SLEEP_SEC = 0.7

def fetch_today_schedule(target_date: date) -> list[dict]:
    """
    Fetches the schedule for a given date from the NBA API.
    """
    if not NBA_API_AVAILABLE:
        print("  nba_api is not installed. Returning empty schedule.")
        return []

    date_str = target_date.strftime("%m/%d/%Y")
    print(f"  Fetching schedule for: {date_str}...")

    sb = scoreboardv3.ScoreboardV3(game_date=date_str, league_id="00")
    time.sleep(SLEEP_SEC)

    dfs = sb.get_data_frames()
    if len(dfs) < 3:
        return []

    game_df = dfs[1]   # gameId, gameStatus, gameCode, gameStatusText ...
    team_df = dfs[2]   # gameId, teamId, teamTricode ... (home+away same table)

    # gameStatus: 1=not started, 2=in progress, 3=finished
    upcoming = game_df[game_df["gameStatus"] == 1]

    games = []
    seen_ids = set()
    for _, row in upcoming.iterrows():
        gid = str(row["gameId"])
        if gid in seen_ids:
            continue
        seen_ids.add(gid)

        # gameCode: '20260222/CLEOKC' -> away=CLE, home=OKC (always 3+3 chars)
        try:
            code_part = row["gameCode"].split("/")[1]
            away_tricode = code_part[:3]
            home_tricode = code_part[3:]
        except Exception:
            continue

        game_teams = team_df[team_df["gameId"] == row["gameId"]]
        home_row = game_teams[game_teams["teamTricode"] == home_tricode]
        away_row = game_teams[game_teams["teamTricode"] == away_tricode]
        if home_row.empty or away_row.empty:
            continue

        games.append({
            "game_id": gid,
            "home_team_id": int(home_row.iloc[0]["teamId"]),
            "away_team_id": int(away_row.iloc[0]["teamId"]),
            "status": str(row.get("gameStatusText", "")),
        })

    # Add team names
    all_teams = {t["id"]: t for t in nba_teams_static.get_teams()}
    for g in games:
        h = all_teams.get(g["home_team_id"], {})
        a = all_teams.get(g["away_team_id"], {})
        g["home_team_name"] = h.get("full_name", str(g["home_team_id"]))
        g["home_team_abbr"] = h.get("abbreviation", "???")
        g["away_team_name"] = a.get("full_name", str(g["away_team_id"]))
        g["away_team_abbr"] = a.get("abbreviation", "???")

    print(f"  -> Found {len(games)} upcoming games")
    return games
