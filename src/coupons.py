"""
coupons.py
----------
Daily handicap coupons (Faz 6, reworked in 6b).

Every leg backs a favourite *giving points* (handicap -1.5 or more). A tier
is a probability band (safe / balanced / risky); a game joins a tier only
if one of its handicaps falls in that band, and the line closest to the
tier's target is taken.

Pricing: each game is priced with the distribution the app shows for it.
With a market line (6b) that is the closing line's error distribution
centred on the market (see market_blend.py), which measured better than
the model against real results; the model's own distribution is used only
on days without any market line, and the coupon says so.

Which games: every game with a market line that has not started. The old
filters (the two models agreeing, a strong favourite, enough games played)
are gone: the market already prices those, and two models trained on the
same features agreeing was not evidence of reliability.

One game, one coupon: a game appears in at most one coupon, so one result
cannot sink every coupon at once. Tiers are filled in order (safe takes
the big favourites that reach its band, balanced the next, risky the
rest); legs per coupon follow the number of eligible games
(MIN_LEGS..MAX_LEGS).

Each leg carries its fair odds (1 / p). The old over coupon is retired:
against closing lines the total model's side won 47% of the time. Old
over legs in the log still settle (leg_won).

Line convention (as everywhere): handicap `line` is on the home margin.
  home leg wins when home margin > line   (home handicap = -line)
  away leg wins when home margin < line   (away handicap = +line)
  over leg wins when home + away > line   (legacy over coupons)
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
# Retired, kept so coupons logged before 6b still sort and display
OVER_TIER = {"key": "ust", "label": "Üst (deneysel)", "target": 0.62, "lo": 0.58, "hi": 0.68}
TIER_ORDER = {t["key"]: i for i, t in enumerate([*TIERS, OVER_TIER])}

SAFETY = 1.05              # legacy "minimum odds" = fair odds × 1.05 (shown, not a bet advice)
MIN_LEGS, MAX_LEGS = 2, 3
MIN_GIVEN = 1.5            # favori en az 1.5 sayı handikap verir
MAX_HANDICAP = 30.0


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
def favourite_side(dist: dict, mu: float) -> str:
    return "home" if center(dist, mu)[0] >= 0 else "away"


def handicap_options(dist: dict, mu: float, side: str) -> list[dict]:
    """Every half-point handicap for `side` with its probability."""
    k, p = score_pmf(dist, mu)
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


def favourite_options(dist: dict, mu: float) -> list[dict]:
    """The favourite's handicaps where it gives at least MIN_GIVEN points."""
    side = favourite_side(dist, mu)
    return [o for o in handicap_options(dist, mu, side) if o["handicap"] <= -MIN_GIVEN]


def pick_line(options: list[dict], target: float, lo: float = 0.0, hi: float = 1.0) -> dict | None:
    """Option closest to the target inside [lo, hi); ties → smaller handicap. None if none fits."""
    fits = [o for o in options if lo <= o["p"] < hi]
    if not fits:
        return None
    return min(fits, key=lambda o: (round(abs(o["p"] - target), 6), abs(o.get("handicap", 0))))


def handicap_label(abbr: str, handicap: float) -> str:
    return f"{abbr} {'+' if handicap > 0 else ''}{handicap:.1f}"


def margin_label(home: str, away: str, margin: float | None) -> str | None:
    """Expected home margin → the favourite's handicap label ('DAL -7.5')."""
    if margin is None:
        return None
    return handicap_label(home if margin >= 0 else away, -abs(margin))


def min_odds(p: float, safety: float = SAFETY) -> float:
    """Fair odds with a 5% margin, rounded up to 2 decimals."""
    return math.ceil(safety / p * 100) / 100


def leg_won(leg: dict, home_pts: float, away_pts: float) -> bool:
    if leg.get("market") == "total":
        total = home_pts + away_pts
        return total > leg["line"] if leg["side"] == "over" else total < leg["line"]
    margin = home_pts - away_pts
    return margin > leg["line"] if leg["side"] == "home" else margin < leg["line"]


# ---------------------------------------------------------------------------
# Pricing and game selection
# ---------------------------------------------------------------------------
def _json(pred: dict, key: str) -> dict:
    raw = pred.get(key)
    return (json.loads(raw) if isinstance(raw, str) else raw) or {}


def pricing(pred: dict, model_dist: dict | None, market_dist: dict | None) -> tuple[dict, float, str] | None:
    """(distribution, centre argument, 'market' | 'model') of the game's margin, None if unpriced."""
    lines, market = _json(pred, "lines"), _json(pred, "market")
    if (market.get("source") or {}).get("margin") == "market" and market_dist and "margin" in lines:
        return market_dist, float(lines["margin"]["pred"]), "market"
    if model_dist and pred.get("pred_margin") is not None:
        return model_dist, float(pred["pred_margin"]), "model"
    return None


def assess_game(pred: dict, model_dist: dict | None, market_dist: dict | None = None) -> dict | None:
    """What the coupon page shows about one game. None = no handicap pricing at all."""
    priced = pricing(pred, model_dist, market_dist)
    if priced is None:
        return None
    dist, mu, source = priced
    home, away = pred["home_team_abbr"], pred["away_team_abbr"]
    c = float(center(dist, mu)[0])
    side = favourite_side(dist, mu)
    market, lines = _json(pred, "market"), _json(pred, "lines")
    model_lines = _json(pred, "model_lines") or (lines if source == "model" else {})

    reasons = []
    started = pred.get("game_status", 1) != 1
    if started:
        reasons.append("maç başladı")
    if source == "market":
        reasons.append(f"piyasa: {margin_label(home, away, market.get('market_margin'))} "
                       f"({market.get('provider') or 'piyasa'})")
    else:
        reasons.append("piyasa çizgisi yok: model çizgisi")
    model_fair = (model_lines.get("margin") or {}).get("fair_line")
    if model_fair is not None:
        reasons.append(f"model: {margin_label(home, away, model_fair)}")

    total = lines.get("total")
    return {
        "game_id": str(pred["game_id"]),
        "home": home,
        "away": away,
        "favourite": home if side == "home" else away,
        "side": side,
        "center": round(c, 2),
        "fair_label": margin_label(home, away, float(fair_line(dist, mu)[0])),
        "fair_total": total["fair_line"] if total else None,
        "source": source,
        "market_label": margin_label(home, away, market.get("market_margin")),
        "model_label": margin_label(home, away, model_fair),
        "eligible": not started,
        "reasons": reasons,
        "game_time_utc": pred.get("game_time_utc"),
    }


def legs_per_coupon(n_eligible: int) -> int:
    """Enough legs to fill the tiers with different games: 2-3."""
    return int(min(MAX_LEGS, max(MIN_LEGS, n_eligible // len(TIERS))))


# ---------------------------------------------------------------------------
# Coupons
# ---------------------------------------------------------------------------
def _leg(game: dict, opt: dict) -> dict:
    p = opt["p"]
    return {
        "game_id": game["game_id"],
        "home": game["home"],
        "away": game["away"],
        "market": "spread",
        "team": game["favourite"],
        "side": opt["side"],
        "line": opt["line"],
        "handicap": opt["handicap"],
        "label": handicap_label(game["favourite"], opt["handicap"]),
        "fair": game["fair_label"],
        "source": game["source"],
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
        "experimental": False,
        "legs": legs,
        "p": round(float(np.prod([l["p"] for l in legs])), 4),
        "fair_odds": round(float(np.prod([1 / l["p"] for l in legs])), 2),
        "min_odds": round(float(np.prod([l["min_odds"] for l in legs])), 2),
    }


def build_coupons(predictions: list[dict], model_dist: dict | None,
                  market_dist: dict | None = None, locked: dict[str, set[str]] | None = None) -> dict:
    """
    Today's coupons from 05's prediction dicts. `model_dist` / `market_dist`
    are the margin distributions (distribution.json / market_distribution.json).
    `locked` = {tier: game_ids} of the day's locked coupons (coupon_tracker):
    those tiers are not rebuilt and their games are not reused.
    Returns {"games", "coupons", "n_legs", "note", "skipped", "source"}.
    """
    locked = locked or {}
    priced = {}
    assessed = []
    for p in predictions:
        a = assess_game(p, model_dist, market_dist)
        if a:
            assessed.append(a)
            priced[a["game_id"]] = pricing(p, model_dist, market_dist)[:2]

    # Market-priced games when there are any; the model only on days without a line
    has_market = any(a["source"] == "market" for a in assessed)
    source = "market" if has_market else "model"
    for a in assessed:
        if a["source"] != source and a["eligible"]:
            a["eligible"] = False
            a["reasons"].append("piyasa çizgisi olmadığı için kupon dışı")
    assessed.sort(key=lambda g: (g["eligible"], abs(g["center"])), reverse=True)
    eligible = [g for g in assessed if g["eligible"]]

    reserved = set().union(*locked.values()) if locked else set()
    n_legs = legs_per_coupon(len([g for g in eligible if g["game_id"] not in reserved]))
    out = {"games": assessed, "coupons": [], "n_legs": n_legs, "note": None, "skipped": [],
           "source": source}
    if not has_market and assessed:
        out["note"] = "Bugün piyasa çizgisi yok: kuponlar modelin çizgisiyle kuruldu (piyasadan zayıf)."
    if len(eligible) < MIN_LEGS:
        out["note"] = (f"Kupon için en az {MIN_LEGS} uygun maç gerekli; bugün {len(eligible)} var.")
        return out

    used = set(reserved)
    for tier in TIERS:
        if tier["key"] in locked:
            continue
        candidates = []
        for g in eligible:
            if g["game_id"] in used:
                continue
            dist, mu = priced[g["game_id"]]
            opt = pick_line(favourite_options(dist, mu), tier["target"], tier["lo"], tier["hi"])
            if opt:
                candidates.append((round(abs(opt["p"] - tier["target"]), 6), -abs(g["center"]), g, opt))
        candidates.sort(key=lambda t: t[:2])
        chosen = candidates[:n_legs]
        if len(chosen) >= MIN_LEGS:
            out["coupons"].append(_coupon(tier, [_leg(g, opt) for *_, g, opt in chosen]))
            used.update(g["game_id"] for *_, g, _ in chosen)
        else:
            out["skipped"].append({"tier": tier["key"], "label": tier["label"],
                                   "reason": f"handikapla %{tier['lo'] * 100:.0f}-{tier['hi'] * 100:.0f} "
                                             f"aralığına giren, başka kupona girmemiş {MIN_LEGS} maç yok"})
    return out
