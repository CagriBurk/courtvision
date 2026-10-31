"""
nba_data_client.py
------------------
A dedicated module to handle data fetching from the nba_api.
"""

import time
from datetime import date

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguegamelog, scoreboardv3
    from nba_api.stats.static import teams as nba_teams_static
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

SLEEP_SEC = 0.7


def get_valid_team_ids() -> set:
    """Current 30 NBA franchise IDs (as strings)."""
    return {str(t["id"]) for t in nba_teams_static.get_teams()}


# ---------------------------------------------------------------------------
# Season game logs (LeagueGameLog)
# ---------------------------------------------------------------------------
def fetch_season_raw(season_str: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetches specified season from nba_api.
    Returned DataFrame is in tall format (2 rows per game: home + away).

    Columns: SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID,
              GAME_DATE, MATCHUP, WL, PTS, FGM, FGA, FG_PCT, FG3M, FG3A,
              FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK,
              TOV, PF, PLUS_MINUS
    """
    print(f"  Fetching: {season_str} ({season_type})...", end=" ", flush=True)
    try:
        gl = leaguegamelog.LeagueGameLog(
            season=season_str,
            season_type_all_star=season_type,
            league_id="00",
            timeout=60,
        )
        time.sleep(SLEEP_SEC)
        df = gl.get_data_frames()[0]
        print(f"{len(df)} rows")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()


def convert_to_wide(df: pd.DataFrame, valid_team_ids: set) -> pd.DataFrame:
    """
    Converts LeagueGameLog tall format to wide (game.csv format):
    one row per game, home and away columns side by side, lowercase names.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].apply(lambda m: 1 if "vs." in m else 0)

    # Filter to valid NBA teams (exclude All-Star, international games, etc.)
    df = df[df["TEAM_ID"].astype(str).isin(valid_team_ids)]

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    stat_cols = [
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "WL", "PTS", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
    ]

    home_renamed = home[["GAME_ID", "SEASON_ID", "GAME_DATE"] + stat_cols].copy()
    home_renamed.columns = (
        ["game_id", "season_id", "game_date"]
        + [c.lower() + "_home" for c in stat_cols]
    )

    away_renamed = away[["GAME_ID"] + stat_cols].copy()
    away_renamed.columns = ["game_id"] + [c.lower() + "_away" for c in stat_cols]

    wide = home_renamed.merge(away_renamed, on="game_id", how="inner")

    # Target variable
    wide["target"] = (wide["wl_home"] == "W").astype(int)

    # Missing other_stats and line_score columns — left as NaN,
    # XGBoost tolerates them with native NaN handling
    other_stats_cols = [
        "pts_paint_home", "pts_2nd_chance_home", "pts_fb_home",
        "largest_lead_home", "lead_changes", "times_tied",
        "team_turnovers_home", "total_turnovers_home", "team_rebounds_home",
        "pts_off_to_home",
        "pts_paint_away", "pts_2nd_chance_away", "pts_fb_away",
        "largest_lead_away", "team_turnovers_away", "total_turnovers_away",
        "team_rebounds_away", "pts_off_to_away",
    ]
    line_score_cols = [
        "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home",
        "pts_qtr1_away", "pts_qtr2_away", "pts_qtr3_away", "pts_qtr4_away",
        "pts_ot1_home", "pts_ot1_away",
    ]
    for col in other_stats_cols + line_score_cols:
        wide[col] = float("nan")

    wide["went_to_ot"] = float("nan")  # cannot be determined without period data
    wide["season_type"] = "Regular Season"

    wide = wide.sort_values("game_date").reset_index(drop=True)
    return wide


def fetch_season_wide(season_str: str) -> pd.DataFrame:
    """Fetch a season and return it in wide (one row per game) format."""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    raw = fetch_season_raw(season_str)
    if raw.empty:
        return pd.DataFrame()
    return convert_to_wide(raw, get_valid_team_ids())


# ---------------------------------------------------------------------------
# Daily schedule (ScoreboardV3)
# ---------------------------------------------------------------------------
def fetch_today_schedule(target_date: date) -> list[dict]:
    """
    Fetches all games scheduled on a given date (US Eastern) from the NBA API.
    game_status: 1=not started, 2=in progress, 3=finished
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

    games = []
    seen_ids = set()
    for _, row in game_df.iterrows():
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

        def score(team_row):
            s = team_row.iloc[0].get("score")
            return int(s) if pd.notna(s) else None

        games.append({
            "game_id": gid,
            "home_team_id": int(home_row.iloc[0]["teamId"]),
            "away_team_id": int(away_row.iloc[0]["teamId"]),
            "status": str(row.get("gameStatusText", "")),
            "game_status": int(row["gameStatus"]),
            "game_time_utc": str(row.get("gameTimeUTC", "")),
            # Scores are 0 before tip-off; only meaningful once the game started
            "home_score": score(home_row) if int(row["gameStatus"]) > 1 else None,
            "away_score": score(away_row) if int(row["gameStatus"]) > 1 else None,
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

    n_upcoming = sum(g["game_status"] == 1 for g in games)
    print(f"  -> Found {len(games)} games ({n_upcoming} not started)")
    return games
