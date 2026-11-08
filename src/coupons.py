"""
coupons.py
----------
Daily coupons without bookmaker odds (Faz 6).

Handicap coupons (safe / balanced / risky): every leg backs the model's
favourite *giving points* (handicap -1.5 or more; -0.5 is just the money
line, and a favourite receiving points is a line bookmakers rarely offer).
A tier is a probability band; a game joins a tier only if one of its real
handicaps falls in that band, and the line closest to the tier's target is
taken. A slight favourite therefore cannot enter the safe coupon: reaching
~72% would need it to *receive* points.

Over coupon (experimental): the over line closest to ~62% on the same
reliable games. The total model does not beat its simple baseline, so this
coupon is tracked separately in the report card.

Because the odds are unknown, each leg carries its fair line (where the
model sees 50/50) and a minimum price: fair odds (1 / p) times a safety
margin for what the bookmaker may know and the model does not. If the
bookmaker's lines sit several points away from the fair line, the market
disagrees with the model (injury, rest...) and the leg is best dropped.

Which games: only ones the model is sure about, not as many as possible.
  eligible  not started, both teams have played MIN_GAMES_PLAYED games,
            the win classifier and the margin model pick the same side and
            their win probabilities differ by at most MAX_DISAGREEMENT
  strong    eligible, near-identical probabilities (STRONG_DISAGREEMENT)
            and a clear favourite (STRONG_MIN_MARGIN points)
The number of legs follows the number of strong games (MIN_LEGS..MAX_LEGS);
a busy night does not force four legs. One leg per game per coupon.

Line convention (as everywhere): handicap `line` is on the home margin.
  home leg wins when home margin > line   (home handicap = -line)
  away leg wins when home margin < line   (away handicap = +line)
  over leg wins when home + away > line
"""

from __future__ import annotations

import json
import math

import numpy as np

from score_dist import center, fair_line, score_pmf

TIERS = [
    {"key": "guvenli", "label": "Güvenli", "target": 0.72, "lo": 0.68, "hi": 0.80},
    {"key": "dengeli", "label": "Dengeli", "target": 0.62, "lo": 0.58, "hi": 0.68},
    {"key": "riskli", "label": "Riskli", "target": 0.54, "lo": 0.50, "hi": 0.58},
]
OVER_TIER = {"key": "ust", "label": "Üst (deneysel)", "target": 0.62, "lo": 0.58, "hi": 0.68}
TIER_ORDER = {t["key"]: i for i, t in enumerate([*TIERS, OVER_TIER])}

SAFETY = 1.05              # her ayakta adil oranın en az %5 üstü
MIN_LEGS, MAX_LEGS = 2, 4
MAX_DISAGREEMENT = 0.10    # iki modelin ev kazanma olasılığı farkı (uygunluk)
STRONG_DISAGREEMENT = 0.05
STRONG_MIN_MARGIN = 4.0    # güçlü maç: favori en az 4 sayı önde bekleniyor
MIN_GAMES_PLAYED = 5       # sezon başı: son-10 ve sezon istatistikleri oturmamış
MIN_GIVEN = 1.5            # favori en az 1.5 sayı handikap verir
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


def favourite_options(dist: dict, pred_margin: float) -> list[dict]:
    """The favourite's handicaps where it gives at least MIN_GIVEN points."""
    side = favourite_side(dist, pred_margin)
    return [o for o in handicap_options(dist, pred_margin, side) if o["handicap"] <= -MIN_GIVEN]


def over_options(dist: dict, pred_total: float) -> list[dict]:
    k, p = score_pmf(dist, pred_total)
    cdf = np.cumsum(p[0])
    return [{"side": "over", "line": float(kk) + 0.5, "p": float(1 - c)}
            for kk, c in zip(k[0], cdf) if 0.02 < 1 - c < 0.98]


def pick_line(options: list[dict], target: float, lo: float = 0.0, hi: float = 1.0) -> dict | None:
    """Option closest to the target inside [lo, hi); ties → smaller handicap. None if none fits."""
    fits = [o for o in options if lo <= o["p"] < hi]
    if not fits:
        return None
    return min(fits, key=lambda o: (round(abs(o["p"] - target), 6), abs(o.get("handicap", 0))))


def handicap_label(abbr: str, handicap: float) -> str:
    return f"{abbr} {'+' if handicap > 0 else ''}{handicap:.1f}"


def min_odds(p: float, safety: float = SAFETY) -> float:
    """Lowest price worth taking, rounded up to 2 decimals."""
    return math.ceil(safety / p * 100) / 100


def leg_won(leg: dict, home_pts: float, away_pts: float) -> bool:
    if leg.get("market") == "total":
        total = home_pts + away_pts
        return total > leg["line"] if leg["side"] == "over" else total < leg["line"]
    margin = home_pts - away_pts
    return margin > leg["line"] if leg["side"] == "home" else margin < leg["line"]


# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------
def _lines(pred: dict) -> dict:
    raw = pred.get("lines")
    return (json.loads(raw) if isinstance(raw, str) else raw) or {}


def assess_game(pred: dict, dist: dict) -> dict | None:
    """Reliability of one game for coupons, with readable reasons. None = no handicap data."""
    lines = _lines(pred)
    if pred.get("pred_margin") is None or "margin" not in lines:
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

    fav = home if side == "home" else away
    fl = float(fair_line(dist, pred["pred_margin"])[0])
    total = lines.get("total")
    return {
        "game_id": str(pred["game_id"]),
        "home": home,
        "away": away,
        "favourite": fav,
        "side": side,
        "pred_margin": float(pred["pred_margin"]),
        "pred_total": pred.get("pred_total"),
        "center": round(c, 2),
        "fair_label": handicap_label(fav, -abs(fl)),
        "fair_total": total["fair_line"] if total else None,
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
def _leg(game: dict, opt: dict, market: str) -> dict:
    p = opt["p"]
    if market == "spread":
        label, fair = handicap_label(game["favourite"], opt["handicap"]), game["fair_label"]
        team = game["favourite"]
    else:
        label, fair, team = f"Üst {opt['line']:.1f}", f"{game['fair_total']:.1f}", None
    return {
        "game_id": game["game_id"],
        "home": game["home"],
        "away": game["away"],
        "market": market,
        "team": team,
        "side": opt["side"],
        "line": opt["line"],
        "handicap": opt.get("handicap"),
        "label": label,
        "fair": fair,
        "p": round(p, 4),
        "fair_odds": round(1 / p, 2),
        "min_odds": min_odds(p),
        "game_time_utc": game.get("game_time_utc"),
    }


def _coupon(tier: dict, legs: list[dict]) -> dict:
    return {
        "tier": tier["key"],
        "label": tier["label"],
        "target": tier["target"],
        "experimental": tier is OVER_TIER,
        "legs": legs,
        "p": round(float(np.prod([l["p"] for l in legs])), 4),
        "fair_odds": round(float(np.prod([1 / l["p"] for l in legs])), 2),
        "min_odds": round(float(np.prod([l["min_odds"] for l in legs])), 2),
    }


def build_coupons(predictions: list[dict], dist: dict, total_dist: dict | None = None) -> dict:
    """
    Today's coupons from 05's prediction dicts and the margin (and total)
    distribution. Returns {"games", "coupons", "n_legs", "note", "skipped"}.
    """
    assessed = [a for a in (assess_game(p, dist) for p in predictions) if a]
    assessed.sort(key=lambda g: (g["eligible"], g["strong"], g["score"]), reverse=True)
    n_legs = n_legs_for(assessed)
    out = {"games": assessed, "coupons": [], "n_legs": n_legs, "note": None, "skipped": []}
    eligible = [g for g in assessed if g["eligible"]]
    if n_legs < MIN_LEGS:
        out["note"] = (f"Kupon için en az {MIN_LEGS} güvenilir maç gerekli; "
                       f"bugün {len(eligible)} uygun maç var.")
        return out

    for tier in TIERS:
        legs = []
        for g in eligible:
            opt = pick_line(favourite_options(dist, g["pred_margin"]), tier["target"], tier["lo"], tier["hi"])
            if opt:
                legs.append(_leg(g, opt, "spread"))
            if len(legs) == n_legs:
                break
        if len(legs) >= MIN_LEGS:
            out["coupons"].append(_coupon(tier, legs))
        else:
            out["skipped"].append({"tier": tier["key"], "label": tier["label"],
                                   "reason": f"handikapla %{tier['lo'] * 100:.0f}-{tier['hi'] * 100:.0f} "
                                             f"aralığına giren {MIN_LEGS} güvenilir maç yok"})

    if total_dist:
        legs = []
        for g in eligible:
            if g["pred_total"] is None or g["fair_total"] is None:
                continue
            opt = pick_line(over_options(total_dist, g["pred_total"]),
                            OVER_TIER["target"], OVER_TIER["lo"], OVER_TIER["hi"])
            if opt:
                legs.append(_leg(g, opt, "total"))
            if len(legs) == n_legs:
                break
        if len(legs) >= MIN_LEGS:
            out["coupons"].append(_coupon(OVER_TIER, legs))
    return out
