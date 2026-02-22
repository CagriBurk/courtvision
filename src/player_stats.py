"""
player_stats.py
---------------
Takım oyuncu istatistikleri ve sakatlık raporu.

Kullanım:
  from player_stats import get_team_top_players, get_injuries, refresh_cache
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

try:
    from nba_api.stats.endpoints import LeagueDashPlayerStats
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

ROOT       = Path(__file__).parent.parent
CACHE_DIR  = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_CACHE = CACHE_DIR / "player_stats.parquet"
INJURY_CACHE = CACHE_DIR / "injuries.json"

CACHE_TTL_HOURS = 6   # 6 saat geçerliliği
CURRENT_SEASON  = "2025-26"

# ESPN tam takım adı → NBA kısaltma eşlemesi
TEAM_NAME_TO_ABR: dict[str, str] = {
    "Atlanta Hawks":           "ATL",
    "Boston Celtics":          "BOS",
    "Brooklyn Nets":           "BKN",
    "Charlotte Hornets":       "CHA",
    "Chicago Bulls":           "CHI",
    "Cleveland Cavaliers":     "CLE",
    "Dallas Mavericks":        "DAL",
    "Denver Nuggets":          "DEN",
    "Detroit Pistons":         "DET",
    "Golden State Warriors":   "GSW",
    "Houston Rockets":         "HOU",
    "Indiana Pacers":          "IND",
    "LA Clippers":             "LAC",
    "Los Angeles Clippers":    "LAC",
    "Los Angeles Lakers":      "LAL",
    "Memphis Grizzlies":       "MEM",
    "Miami Heat":              "MIA",
    "Milwaukee Bucks":         "MIL",
    "Minnesota Timberwolves":  "MIN",
    "New Orleans Pelicans":    "NOP",
    "New York Knicks":         "NYK",
    "Oklahoma City Thunder":   "OKC",
    "Orlando Magic":           "ORL",
    "Philadelphia 76ers":      "PHI",
    "Phoenix Suns":            "PHX",
    "Portland Trail Blazers":  "POR",
    "Sacramento Kings":        "SAC",
    "San Antonio Spurs":       "SAS",
    "Toronto Raptors":         "TOR",
    "Utah Jazz":               "UTA",
    "Washington Wizards":      "WAS",
}


# ---------------------------------------------------------------------------
# Oyuncu istatistikleri
# ---------------------------------------------------------------------------
def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours < CACHE_TTL_HOURS


def get_player_stats(force_refresh: bool = False) -> pd.DataFrame:
    """Mevcut sezon oyuncu istatistiklerini döndürür (cache'li)."""
    if not force_refresh and _cache_fresh(PLAYER_CACHE):
        return pd.read_parquet(PLAYER_CACHE)

    if not NBA_API_AVAILABLE:
        return pd.DataFrame()

    try:
        print("[player_stats] LeagueDashPlayerStats çekiliyor...")
        stats = LeagueDashPlayerStats(
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        time.sleep(0.7)
        df = stats.get_data_frames()[0]
        df.to_parquet(PLAYER_CACHE, index=False)
        print(f"[player_stats] {len(df)} oyuncu kaydedildi → {PLAYER_CACHE}")
        return df
    except Exception as e:
        print(f"[player_stats] LeagueDashPlayerStats hatası: {e}")
        if PLAYER_CACHE.exists():
            return pd.read_parquet(PLAYER_CACHE)   # eski cache ile devam
        return pd.DataFrame()


def get_team_top_players(team_id: int, n: int = 5) -> list[dict]:
    """Takımın impact score'a göre top n oyuncusunu döndürür."""
    df = get_player_stats()
    if df.empty:
        return []

    team_df = df[df["TEAM_ID"] == team_id].copy()
    if team_df.empty:
        return []

    # Basit impact score: PTS + 1.2*REB + 1.5*AST + STL + BLK
    for col in ["PTS", "REB", "AST", "STL", "BLK"]:
        team_df[col] = pd.to_numeric(team_df.get(col, 0), errors="coerce").fillna(0)

    team_df["impact"] = (
        team_df["PTS"] +
        team_df["REB"] * 1.2 +
        team_df["AST"] * 1.5 +
        team_df["STL"] +
        team_df["BLK"]
    )

    top = team_df.nlargest(n, "impact")[
        ["PLAYER_NAME", "PLAYER_ID", "PTS", "REB", "AST", "impact"]
    ]

    return [
        {
            "name":   str(row["PLAYER_NAME"]),
            "id":     int(row["PLAYER_ID"]),
            "pts":    round(float(row["PTS"]), 1),
            "reb":    round(float(row["REB"]), 1),
            "ast":    round(float(row["AST"]), 1),
            "impact": round(float(row["impact"]), 1),
        }
        for _, row in top.iterrows()
    ]


# ---------------------------------------------------------------------------
# Sakatlık raporu — ESPN unofficial API
# ---------------------------------------------------------------------------
ESPN_INJURY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)


def get_injuries(force_refresh: bool = False) -> dict[str, list[dict]]:
    """
    ESPN'den sakatlık raporunu çeker, NBA takım kısaltmasına göre gruplar.

    Döner: { "OKC": [...], "LAL": [...], ... }
    Her oyuncu: { name, id, status, detail }
    status: "Out" | "Questionable" | "Doubtful" | "Day-To-Day"
    """
    if not force_refresh and _cache_fresh(INJURY_CACHE):
        with open(INJURY_CACHE, encoding="utf-8") as f:
            return json.load(f)

    try:
        req = urllib.request.Request(
            ESPN_INJURY_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[player_stats] ESPN injuries hatası: {e}")
        if INJURY_CACHE.exists():
            with open(INJURY_CACHE, encoding="utf-8") as f:
                return json.load(f)
        return {}

    by_team: dict[str, list[dict]] = {}

    for team_entry in data.get("injuries", []):
        # ESPN tam takım adından kısaltmaya çevir
        display_name = team_entry.get("displayName", "")
        abbr = TEAM_NAME_TO_ABR.get(display_name, "")
        if not abbr:
            continue  # tanınmayan takım

        players = []
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            status  = inj.get("status", "")
            name    = athlete.get("displayName", "")
            pos     = athlete.get("position", {})
            pos_str = pos.get("abbreviation", "") if isinstance(pos, dict) else str(pos)
            if not name:
                continue
            players.append({
                "name":   name,
                "id":     str(athlete.get("id", inj.get("id", ""))),
                "status": status,
                "detail": pos_str,   # pozisyon bilgisi (G, F, C)
            })

        if players:
            by_team[abbr] = players

    with open(INJURY_CACHE, "w", encoding="utf-8") as f:
        json.dump(by_team, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in by_team.values())
    print(f"[player_stats] {total} sakatlık kaydedildi ({len(by_team)} takım) → {INJURY_CACHE}")
    return by_team


# ---------------------------------------------------------------------------
# Toplu yenileme (API startup veya "Veriyi Güncelle")
# ---------------------------------------------------------------------------
def refresh_cache() -> None:
    """Hem oyuncu stats hem sakatlık cache'ini yeniler."""
    get_player_stats(force_refresh=True)
    get_injuries(force_refresh=True)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Sakatlık Raporu ===")
    injuries = get_injuries(force_refresh=True)
    for team, players in list(injuries.items())[:5]:
        print(f"\n{team}:")
        for p in players:
            print(f"  {p['name']} — {p['status']} ({p['detail']})")

    print("\n=== OKC Top 5 Oyuncu ===")
    # OKC team_id = 1610612760
    players = get_team_top_players(1610612760)
    for p in players:
        print(f"  {p['name']}: {p['pts']}pts {p['reb']}reb {p['ast']}ast (impact={p['impact']})")
