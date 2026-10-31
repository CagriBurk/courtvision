"""
season.py
---------
NBA season helpers. The regular season starts in October, so dates from
October onwards belong to the season starting that year.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

# NBA game dates follow US Eastern time. From Turkey, tonight's games tip off
# after midnight local time, but they still belong to the previous US date.
NBA_TZ = ZoneInfo("America/New_York")


def nba_today() -> date:
    """Today's date on the NBA schedule (US Eastern)."""
    return datetime.now(NBA_TZ).date()


def season_start_year(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def season_str(start_year: int) -> str:
    """2026 → '2026-27' (nba_api season format)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season(today: date | None = None) -> str:
    return season_str(season_start_year(today))


def season_id(start_year: int) -> str:
    """2026 → '22026' (SEASON_ID format for Regular Season games)."""
    return f"2{start_year}"
