"""
coupons.py
----------
Daily handicap coupons without bookmaker odds (Faz 6).

Every leg backs the model's favourite side of one game with the half-point
handicap whose probability is closest to the coupon's target (safe ≈ 72%,
balanced ≈ 62%, risky ≈ 54%). Because the odds are unknown, each leg and
each coupon carries a minimum price: fair odds (1 / p) times a safety
margin for what the bookmaker may know and the model does not. A leg is
only worth playing at or above that price.

Which games: only ones the model is sure about, not as many as possible.
  eligible  not started, both teams have played MIN_GAMES_PLAYED games,
            the win classifier and the margin model pick the same side and
            their win probabilities differ by at most MAX_DISAGREEMENT
  strong    eligible, near-identical probabilities (STRONG_DISAGREEMENT)
            and a clear favourite (STRONG_MIN_MARGIN points)
The number of legs follows the number of strong games (MIN_LEGS..MAX_LEGS);
a busy night does not force four legs. One leg per game per coupon.

Line convention (as everywhere): `line` is on the home margin.
  home leg wins when home margin > line   (home handicap = -line)
  away leg wins when home margin < line   (away handicap = +line)
"""

from __future__ import annotations

import json
import math

import numpy as np

from score_dist import center, score_pmf

TIERS = [
    {"key": "guvenli", "label": "Güvenli", "target": 0.72},
    {"key": "dengeli", "label": "Dengeli", "target": 0.62},
    {"key": "riskli", "label": "Riskli", "target": 0.54},
]
SAFETY = 1.05              # her ayakta adil oranın en az %5 üstü
MIN_LEGS, MAX_LEGS = 2, 4
MAX_DISAGREEMENT = 0.10    # iki modelin ev kazanma olasılığı farkı (uygunluk)
STRONG_DISAGREEMENT = 0.05
STRONG_MIN_MARGIN = 4.0    # güçlü maç: favori en az 4 sayı önde bekleniyor
MIN_GAMES_PLAYED = 5       # sezon başı: son-10 ve sezon istatistikleri oturmamış
MAX_HANDICAP = 30.0


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
def favourite_side(dist: dict, pred_margin: float) -> str:
    return "home" if center(dist, pred_margin)[0] >= 0 else "away"


def handicap_options(dist: dict, pred_margin: float, side: str) -> list[dict]:
    """Every half-point handicap for `side` with its probability."""
    k, p = score_pmf(dist, pred_margin)
    k, p = k[0], p[0]
    cdf = np.cumsum(p)
    out = []
    for kk, c in zip(k, cdf):
        line = float(kk) + 0.5
        prob = float(1 - c) if side == "home" else float(c)  # P(M > line) / P(M < line)
        handicap = -line if side == "home" else line
        if abs(handicap) <= MAX_HANDICAP and 0.02 < prob < 0.98:
            out.append({"side": side, "line": line, "handicap": handicap, "p": prob})
    return out


def pick_line(options: list[dict], target: float) -> dict:
    """Option closest to the target probability; ties → smaller handicap."""
    return min(options, key=lambda o: (round(abs(o["p"] - target), 6), abs(o["handicap"])))


def handicap_label(abbr: str, handicap: float) -> str:
    return f"{abbr} {'+' if handicap > 0 else ''}{handicap:.1f}"


def min_odds(p: float, safety: float = SAFETY) -> float:
    """Lowest price worth taking, rounded up to 2 decimals."""
    return math.ceil(safety / p * 100) / 100


def leg_won(leg: dict, home_pts: float, away_pts: float) -> bool:
    margin = home_pts - away_pts
    return margin > leg["line"] if leg["side"] == "home" else margin < leg["line"]


# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------
def assess_game(pred: dict, dist: dict) -> dict | None:
    """Reliability of one game for coupons, with readable reasons. None = no handicap data."""
    if pred.get("pred_margin") is None or not pred.get("lines"):
        return None
    lines = json.loads(pred["lines"]) if isinstance(pred["lines"], str) else pred["lines"]
    if "margin" not in lines:
        return None

    home, away = pred["home_team_abbr"], pred["away_team_abbr"]
    c = float(center(dist, pred["pred_margin"])[0])
    side = favourite_side(dist, pred["pred_margin"])
    p_margin = float(lines["margin"]["p_home_win"])
    p_clf = float(pred.get("prob_home_classifier", pred.get("prob_home", 0.5)))
    gap = abs(p_clf - p_margin)

    reasons, ok = [], True
    if pred.get("game_status", 1) != 1:
        ok = False
        reasons.append("maç başladı")
    if (p_clf - 0.5) * (p_margin - 0.5) <= 0:
        ok = False
        reasons.append("modeller farklı tarafı gösteriyor")
    elif gap > MAX_DISAGREEMENT:
        ok = False
        reasons.append(f"modeller arasında %{gap * 100:.0f} fark")
    played = [pred.get("home_games_played"), pred.get("away_games_played")]
    if any(g is not None and g < MIN_GAMES_PLAYED for g in played):
        ok = False
        reasons.append(f"takımlardan biri {MIN_GAMES_PLAYED} maçtan az oynadı")

    strong = ok and gap <= STRONG_DISAGREEMENT and abs(c) >= STRONG_MIN_MARGIN
    if ok:
        reasons.append(f"modeller uyumlu (fark %{gap * 100:.1f})")
        reasons.append(f"beklenen fark {abs(c):.1f} sayı")
    score = (1 - min(gap / MAX_DISAGREEMENT, 1)) + 0.5 * min(abs(c) / 10, 1)

    return {
        "game_id": str(pred["game_id"]),
        "home": home,
        "away": away,
        "favourite": home if side == "home" else away,
        "side": side,
        "pred_margin": float(pred["pred_margin"]),
        "center": round(c, 2),
        "p_classifier": round(p_clf, 4),
        "p_margin": round(p_margin, 4),
        "gap": round(gap, 4),
        "score": round(score, 4),
        "eligible": ok,
        "strong": strong,
        "reasons": reasons,
        "game_time_utc": pred.get("game_time_utc"),
    }


def n_legs_for(assessed: list[dict]) -> int:
    eligible = [g for g in assessed if g["eligible"]]
    n_strong = sum(g["strong"] for g in eligible)
    return min(max(n_strong, MIN_LEGS), MAX_LEGS, len(eligible))


# ---------------------------------------------------------------------------
# Coupons
# ---------------------------------------------------------------------------
def build_leg(game: dict, dist: dict, target: float) -> dict:
    opt = pick_line(handicap_options(dist, game["pred_margin"], game["side"]), target)
    p = opt["p"]
    return {
        "game_id": game["game_id"],
        "home": game["home"],
        "away": game["away"],
        "team": game["favourite"],
        "side": opt["side"],
        "line": opt["line"],
        "handicap": opt["handicap"],
        "label": handicap_label(game["favourite"], opt["handicap"]),
        "p": round(p, 4),
        "fair_odds": round(1 / p, 2),
        "min_odds": min_odds(p),
        "game_time_utc": game.get("game_time_utc"),
    }


def build_coupons(predictions: list[dict], dist: dict) -> dict:
    """
    Today's coupons from 05's prediction dicts and the margin distribution.
    Returns {"games": [assessment...], "coupons": [...], "n_legs": int, "note": str|None}.
    """
    assessed = [a for a in (assess_game(p, dist) for p in predictions) if a]
    assessed.sort(key=lambda g: (g["eligible"], g["strong"], g["score"]), reverse=True)
    n_legs = n_legs_for(assessed)
    out = {"games": assessed, "coupons": [], "n_legs": n_legs, "note": None}
    if n_legs < MIN_LEGS:
        out["note"] = (f"Kupon için en az {MIN_LEGS} güvenilir maç gerekli; "
                       f"bugün {sum(g['eligible'] for g in assessed)} uygun maç var.")
        return out

    chosen = [g for g in assessed if g["eligible"]][:n_legs]
    for tier in TIERS:
        legs = [build_leg(g, dist, tier["target"]) for g in chosen]
        p = float(np.prod([l["p"] for l in legs]))
        out["coupons"].append({
            "tier": tier["key"],
            "label": tier["label"],
            "target": tier["target"],
            "legs": legs,
            "p": round(p, 4),
            "fair_odds": round(float(np.prod([1 / l["p"] for l in legs])), 2),
            "min_odds": round(float(np.prod([l["min_odds"] for l in legs])), 2),
        })
    return out
