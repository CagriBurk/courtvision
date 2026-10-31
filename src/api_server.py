"""
api_server.py
-------------
Electron uygulamasının bağlandığı FastAPI local REST server.
Start: uvicorn src.api_server:app --host 127.0.0.1 --port 8765
or directly: python src/api_server.py
"""

import json
import pickle
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# src/ if inside src, add parent to sys.path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import build_team_game_log
from season import nba_today, season_str


# ---------------------------------------------------------------------------
# Team colors
# ---------------------------------------------------------------------------
TEAM_COLORS: dict[str, str] = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BKN': '#000000', 'CHA': '#1D1160',
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240',
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62',
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E',
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#E56020',
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#8A8D8F', 'TOR': '#CE1141',
    'UTA': '#002B5C', 'WAS': '#002B5C',
}

CONF_TR_EN = {'YÜKSEK': 'HIGH', 'ORTA': 'MEDIUM', 'DÜŞÜK': 'LOW'}

# ---------------------------------------------------------------------------
app = FastAPI(
    title="NBA Predictor API",
    description="Electron shell için local REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8765",
        "http://127.0.0.1:8765",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR  = ROOT / "models"
PROC_DIR    = ROOT / "data" / "processed"
LIVE_DIR    = ROOT / "data" / "live"
PRED_DIR    = ROOT / "data" / "predictions"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Uygulama başlarken yükle
# ---------------------------------------------------------------------------
_model = None
_feature_columns = None


def _load_model():
    global _model, _feature_columns
    model_path = MODELS_DIR / "xgb_best.pkl"
    feat_path  = MODELS_DIR / "feature_columns.pkl"

    if not model_path.exists():
        return False

    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    with open(feat_path, "rb") as f:
        _feature_columns = pickle.load(f)
    _model.set_params(device="cpu")  # inference CPU'da
    return True


# Prediction runs (startup, scheduler, "Veriyi Güncelle") share one lock so
# 05_predict_today.py never runs twice at the same time.
_predictions_lock = threading.Lock()
# Set while a run is pending or in progress; /api/today-games reports it as
# "updating" and the frontend polls until it clears.
_predictions_running = threading.Event()

SCHEDULER_INTERVAL_SEC = 15 * 60


def _live_date() -> str | None:
    """Slate date of the current data/live files (written by 05_predict_today.py)."""
    try:
        return json.loads((LIVE_DIR / "meta.json").read_text(encoding="utf-8")).get("date")
    except Exception:
        return None


def _predictions_fresh() -> bool:
    """True if 05_predict_today.py already ran for today's slate."""
    return _live_date() == str(nba_today())


def _run_predictions() -> bool:
    """Run 05_predict_today.py (blocking). False if a run is already in progress."""
    if not _predictions_lock.acquire(blocking=False):
        return False
    _predictions_running.set()
    try:
        script = ROOT / "src" / "05_predict_today.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[API] predict_today error code: {result.returncode}")
        else:
            print("[API] Predictions updated.")
        return True
    finally:
        _predictions_running.clear()
        _predictions_lock.release()


def _run_predictions_if_stale() -> None:
    """If today's predictions are missing, regenerate them (blocking)."""
    if _predictions_fresh():
        print("[API] Today predictions are already up to date, skipping.")
        return

    print("[API] Predictions are old/missing — running 05_predict_today.py...")
    _run_predictions()


def _startup_background_jobs() -> None:
    """Slow startup work (network + pipeline), run off the event loop."""
    try:
        _run_predictions_if_stale()
    except Exception as e:
        print(f"[API] auto-predict skipped: {e}")

    # Önceki günlerin sonuçlarını otomatik doldur (yerel master_games.parquet'ten)
    try:
        from prediction_tracker import fill_results
        filled = fill_results()
        if filled is not None and not filled.empty:
            completed = filled["correct"].notna().sum()
            print(f"[API] fill_results: {completed} completed predictions.")
    except Exception as e:
        print(f"[API] fill_results skipped: {e}")

    # Canlı sezon sonuçlarını doldur (nba_api)
    try:
        from prediction_tracker import fill_results_live
        fill_results_live()
    except Exception as e:
        print(f"[API] fill_results_live skipped: {e!r}")

    # Sakatlık raporunu yenile (cache yoksa veya eskimişse)
    try:
        from player_stats import get_injuries
        get_injuries()   # cache varsa hızlı döner
    except Exception as e:
        print(f"[API] injuries cache skipped: {e}")

    _scheduler_loop()


def _scheduler_loop() -> None:
    """Keeps predictions current while the app stays open (new slate each day)."""
    while True:
        time.sleep(SCHEDULER_INTERVAL_SEC)
        try:
            if not _predictions_fresh():
                print("[API] New day on the NBA schedule — refreshing predictions.")
                _run_predictions()
        except Exception as e:
            print(f"[API] scheduler error: {e!r}")


@app.on_event("startup")
async def startup_event():
    ok = _load_model()
    if ok:
        print(f"[API] Model loaded. {len(_feature_columns)} feature.")
    else:
        print("[API] Model not found. Run 03_train_model.py first.")

    # Don't block startup: the window opens right away and the frontend
    # polls /api/today-games while "updating" is true.
    if not _predictions_fresh():
        _predictions_running.set()  # visible to the first request already
    threading.Thread(target=_startup_background_jobs, daemon=True).start()


# ---------------------------------------------------------------------------
# Yardımcı: live_features satırını frontend Game formatına dönüştür
# ---------------------------------------------------------------------------
def _logo_url(team_id: str) -> str:
    # NBA's own CDN (the ESPN logo CDN is blocked on some networks)
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


def _streak_str(s: int) -> str:
    return f"{s}G" if s > 0 else f"{-s}M" if s < 0 else "-"


def _build_game_response(
    row: dict,
    injuries: dict | None = None,
    top_players: dict | None = None,
) -> dict:
    """live_features satırını frontend Game arayüzüne çevirir."""
    h_abbr = str(row.get('home_team_abbr') or '')
    a_abbr = str(row.get('away_team_abbr') or '')

    # team_id: int veya str olabilir
    raw_hid = row.get('home_team_id')
    raw_aid = row.get('away_team_id')
    h_tid = str(int(raw_hid)) if raw_hid is not None else ''
    a_tid = str(int(raw_aid)) if raw_aid is not None else ''

    # Written by 05_predict_today.py from the freshly updated master table
    def get_form(side: str) -> list[int]:
        seq = row.get(f'{side}_form_seq')
        return [int(w) for w in seq] if seq is not None else []

    home_form = get_form('home')
    away_form = get_form('away')

    # Kazanma faktörleri (WhyWinsPanel için).
    # positive = ev sahibi lehine; value = çubuk uzunluğu için göreli güç (0–20);
    # detail = kullanıcıya gösterilen gerçek değer.
    factors = []

    elo_diff = float(row.get('elo_diff') or 0)
    if abs(elo_diff) > 5:
        factors.append({
            'label':    'Elo Rating Farkı',
            'detail':   f"{abs(elo_diff):.0f} puan",
            'value':    min(round(abs(elo_diff) / 10), 20),
            'positive': elo_diff > 0,
        })

    pm_diff = float(row.get('diff_plus_minus_L10') or 0)
    if abs(pm_diff) > 0.5:
        factors.append({
            'label':    'Sayı Farkı (son 10 maç)',
            'detail':   f"{abs(pm_diff):.1f} sayı/maç",
            'value':    min(round(abs(pm_diff) * 2), 15),
            'positive': pm_diff > 0,
        })

    # Form farkı L5 — plus/minus farkı yoksa göster
    h_f5 = float(row.get('home_form_L5') or 0)
    a_f5 = float(row.get('away_form_L5') or 0)
    form_diff = h_f5 - a_f5
    if abs(form_diff) > 0.15 and abs(pm_diff) <= 0.5:
        h_w, a_w = round(h_f5 * 5), round(a_f5 * 5)
        factors.append({
            'label':    'Son 5 Maç Formu',
            'detail':   f"{h_w}G-{5 - h_w}M / {a_w}G-{5 - a_w}M",
            'value':    min(round(abs(form_diff) * 20), 12),
            'positive': form_diff > 0,
        })

    h_streak = int(row.get('home_streak') or 0)
    a_streak = int(row.get('away_streak') or 0)
    streak_diff = h_streak - a_streak
    if abs(streak_diff) >= 3:
        factors.append({
            'label':    'Galibiyet / Mağlubiyet Serisi',
            'detail':   f"{h_abbr} {_streak_str(h_streak)} · {a_abbr} {_streak_str(a_streak)}",
            'value':    min(abs(streak_diff) * 2, 10),
            'positive': streak_diff > 0,
        })

    # Arka arkaya ikinci maç (yorgunluk)
    diff_b2b = int(row.get('home_b2b') or 0) - int(row.get('away_b2b') or 0)
    if diff_b2b != 0:
        tired = h_abbr if diff_b2b > 0 else a_abbr
        factors.append({
            'label':    'Yorgunluk (arka arkaya 2. maç)',
            'detail':   f"{tired} dün de oynadı",
            'value':    7,
            'positive': diff_b2b < 0,
        })

    h_rest = row.get('home_rest')
    a_rest = row.get('away_rest')
    if h_rest is not None and a_rest is not None:
        rest_diff = float(h_rest) - float(a_rest)
        if abs(rest_diff) >= 2 and len(factors) < 5:
            factors.append({
                'label':    'Dinlenme Avantajı',
                'detail':   f"{abs(rest_diff):.0f} gün fazla",
                'value':    min(round(abs(rest_diff) * 2), 8),
                'positive': rest_diff > 0,
            })

    # H2H: ev sahibinin bu rakibe karşı geçmişi
    h2h_n = int(row.get('h2h_games') or 0)
    h2h_wr = float(row.get('h2h_home_wr') or 0.5)
    if h2h_n > 0 and abs(h2h_wr - 0.5) > 0.05:
        factors.append({
            'label':    f"Karşılaşma Geçmişi ({h2h_n} maç)",
            'detail':   f"{h_abbr} %{h2h_wr * 100:.0f} kazandı",
            'value':    round(abs(h2h_wr - 0.5) * 20),
            'positive': h2h_wr > 0.5,
        })

    wct_diff = float(row.get('diff_season_win_pct') or 0)
    if abs(wct_diff) > 0.02 and len(factors) < 5:
        factors.append({
            'label':    'Sezon Galibiyet Oranı',
            'detail':   f"{abs(wct_diff) * 100:.0f} puan fark",
            'value':    min(round(abs(wct_diff) * 30), 12),
            'positive': wct_diff > 0,
        })

    srs_diff = float(row.get('diff_srs_L10') or 0)
    if abs(srs_diff) > 0.5 and len(factors) < 5:
        factors.append({
            'label':    'Rakip Gücüne Göre Sayı Farkı (SRS)',
            'detail':   f"{abs(srs_diff):.1f}",
            'value':    min(round(abs(srs_diff) * 2), 10),
            'positive': srs_diff > 0,
        })

    if not factors:
        prob = float(row.get('prob_home') or 0.5)
        factors.append({
            'label':    'Model Tahmini',
            'detail':   f"%{max(prob, 1 - prob) * 100:.0f}",
            'value':    round(abs(prob - 0.5) * 20),
            'positive': prob > 0.5,
        })

    conf_raw = str(row.get('confidence') or 'DÜŞÜK')
    confidence = CONF_TR_EN.get(conf_raw, conf_raw)  # zaten EN ise geç

    def _opt_int(v):
        return int(v) if v is not None else None

    return {
        'id': str(row.get('game_id') or ''),
        'home': {
            'id':           h_tid,
            'name':         str(row.get('home_team_name') or ''),
            'abbreviation': h_abbr,
            'color':        TEAM_COLORS.get(h_abbr, '#1a1a2e'),
            'logoUrl':      _logo_url(h_tid),
        },
        'away': {
            'id':           a_tid,
            'name':         str(row.get('away_team_name') or ''),
            'abbreviation': a_abbr,
            'color':        TEAM_COLORS.get(a_abbr, '#1a1a2e'),
            'logoUrl':      _logo_url(a_tid),
        },
        'probHome':       float(row.get('prob_home') or 0.5),
        'confidence':     confidence,
        'gameTime':       str(row.get('status') or ''),
        'gameTimeUtc':    row.get('game_time_utc') or None,
        'gameStatus':     _opt_int(row.get('game_status')) or 1,
        'homeScore':      _opt_int(row.get('home_score')),
        'awayScore':      _opt_int(row.get('away_score')),
        'venue':          '',
        'factors':        factors,
        'homeForm':       home_form,
        'awayForm':       away_form,
        'homeInjuries':   (injuries or {}).get(h_abbr, []),
        'awayInjuries':   (injuries or {}).get(a_abbr, []),
        'homeTopPlayers': (top_players or {}).get(h_tid, []),
        'awayTopPlayers': (top_players or {}).get(a_tid, []),
    }


# ---------------------------------------------------------------------------
# Endpoint: Sağlık kontrolü
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    try:
        meta = json.loads((LIVE_DIR / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    return {
        "status":           "ok",
        "model_loaded":     _model is not None,
        "feature_count":    len(_feature_columns) if _feature_columns else 0,
        "timestamp":        datetime.now().isoformat(),
        # Last 05_predict_today.py run (shown on the Settings page)
        "live_date":        meta.get("date"),
        "live_generated_at": meta.get("generated_at"),
        "updating":         _predictions_running.is_set(),
        "scheduler_minutes": SCHEDULER_INTERVAL_SEC // 60,
    }


# ---------------------------------------------------------------------------
# Endpoint: Today predictions (frontend Game format)
# ---------------------------------------------------------------------------
@app.get("/api/today-games")
async def today_games():
    """Returns today prediction report — compatible with frontend Game interface."""
    live_path = LIVE_DIR / "live_features.parquet"

    updating = _predictions_running.is_set()

    if not live_path.exists():
        return {
            "date":     str(nba_today()),
            "games":    [],
            "count":    0,
            "updating": updating,
            "message":  "Predictions not run yet. Run 05_predict_today.py.",
        }

    df = pd.read_parquet(live_path)
    df = df.where(pd.notna(df), None)

    # Sakatlık ve oyuncu verisini yükle (cache'den hızlı)
    injuries   = {}
    top_players: dict[str, list] = {}
    try:
        from player_stats import get_injuries, get_player_stats, get_team_top_players
        injuries = get_injuries()

        # Her maç için her iki takımın top oyuncularını çek
        team_ids = set()
        for _, row in df.iterrows():
            if row.get('home_team_id') is not None:
                team_ids.add(str(int(row['home_team_id'])))
            if row.get('away_team_id') is not None:
                team_ids.add(str(int(row['away_team_id'])))

        cached_player_stats_df = get_player_stats()

        for tid in team_ids:
            players = get_team_top_players(int(tid), df=cached_player_stats_df)
            if players:
                top_players[tid] = players
    except Exception as e:
        print(f"[API] failed to load player/injury: {e}")

    games = [
        _build_game_response(dict(row), injuries, top_players)
        for _, row in df.iterrows()
    ]

    return {
        "date":     _live_date() or str(nba_today()),
        "games":    games,
        "count":    len(games),
        "updating": updating,
    }


# ---------------------------------------------------------------------------
# Endpoint: Sakatlık raporu
# ---------------------------------------------------------------------------
@app.get("/api/injuries")
async def get_injuries_endpoint(refresh: bool = False):
    """ESPN'den sakatlık raporunu döndürür (takım bazlı, cache'li)."""
    try:
        from player_stats import get_injuries
        data = get_injuries(force_refresh=refresh)
        total = sum(len(v) for v in data.values())
        return {"injuries": data, "total": total, "teams": len(data)}
    except Exception as e:
        return {"injuries": {}, "total": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Endpoint: Takım top oyuncuları
# ---------------------------------------------------------------------------
@app.get("/api/players/{team_id}")
async def get_top_players(team_id: int, n: int = 5):
    """Takımın mevcut sezon top n oyuncusunu döndürür."""
    try:
        from player_stats import get_team_top_players
        players = get_team_top_players(team_id, n=n)
        return {"team_id": team_id, "players": players}
    except Exception as e:
        return {"team_id": team_id, "players": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Endpoint: Refresh predictions (calls 05_predict_today)
# ---------------------------------------------------------------------------
@app.post("/api/refresh-predictions")
async def refresh_predictions():
    """
    Starts 05_predict_today.py in the background (fresh nba_api data) and
    returns immediately; /api/today-games reports "updating" until it ends.
    """
    if _predictions_running.is_set():
        return {"started": False, "message": "Tahminler zaten güncelleniyor."}

    _predictions_running.set()  # visible to the very next poll

    def job():
        if not _run_predictions():
            return
        try:
            from prediction_tracker import fill_results
            fill_results()
        except Exception as e:
            print(f"[API] fill_results skipped: {e!r}")

    threading.Thread(target=job, daemon=True).start()
    return {"started": True}


# ---------------------------------------------------------------------------
# Endpoint: Past predictions (predictions_log.parquet)
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history():
    """Returns saved prediction history."""
    pred_path = PRED_DIR / "predictions_log.parquet"
    if not pred_path.exists():
        return {"predictions": [], "stats": None}

    df = pd.read_parquet(pred_path)
    df = df.sort_values("date", ascending=False)
    df = df.where(pd.notna(df), None)

    records = []
    for _, row in df.iterrows():
        conf = str(row.get('confidence') or 'LOW')
        records.append({
            'gameId':           str(row.get('game_id') or ''),
            'date':             str(row.get('date') or ''),
            'homeTeam':         str(row.get('home_team') or ''),
            'awayTeam':         str(row.get('away_team') or ''),
            'probHome':         float(row.get('prob_home') or 0.5),
            'confidence':       conf,
            'predictedWinner':  str(row.get('predicted_winner') or ''),
            'actualWinner':     row.get('actual_winner'),
            'correct':          row.get('correct'),
        })

    # İstatistikler
    completed = [r for r in records if r['correct'] is not None]
    weights   = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

    if completed:
        total_w   = sum(weights.get(r['confidence'], 1) for r in completed)
        correct_w = sum(weights.get(r['confidence'], 1) for r in completed if r['correct'])
        weighted_acc = round(correct_w / total_w * 100, 1) if total_w > 0 else 0

        by_tier = {}
        for tier in ('HIGH', 'MEDIUM', 'LOW'):
            t_games = [r for r in completed if r['confidence'] == tier]
            by_tier[tier] = {
                'correct': sum(1 for r in t_games if r['correct']),
                'total':   len(t_games),
            }

        stats = {
            'totalGames':       len(records),
            'completedGames':   len(completed),
            'weightedAccuracy': weighted_acc,
            'byTier':           by_tier,
        }
    else:
        stats = {
            'totalGames':       len(records),
            'completedGames':   0,
            'weightedAccuracy': 0,
            'byTier':           {t: {'correct': 0, 'total': 0} for t in ('HIGH', 'MEDIUM', 'LOW')},
        }

    return {"predictions": records, "stats": stats}


# ---------------------------------------------------------------------------
# Endpoint: Model metrikleri (fold tablosu + feature importance)
# ---------------------------------------------------------------------------
@app.get("/api/model/metrics")
async def model_metrics():
    """Model değerlendirme metriklerini döndürür."""
    feat_imp: list[dict] = []
    if _model is not None:
        try:
            scores = _model.get_booster().get_fscore(importance_type='gain')
            total  = sum(scores.values()) or 1
            feat_imp = sorted(
                [{'feature': k, 'importance': round(v / total * 100, 2)} for k, v in scores.items()],
                key=lambda x: x['importance'], reverse=True,
            )[:20]
        except Exception:
            pass

    def _read_json(name: str) -> dict:
        try:
            return json.loads((MODELS_DIR / name).read_text(encoding="utf-8"))
        except Exception:
            return {}

    # Written by 03_train_model.py / 04_evaluate_model.py
    train = _read_json("train_meta.json")
    ev    = _read_json("metrics.json")

    def _season_label(sid):
        # '22025' -> '2025-26'
        return season_str(int(str(sid)[1:])) if sid else None

    def _pct(v):
        return round(v * 100, 1) if v is not None else None

    return {
        # Headline = holdout season (never seen while tuning) when available
        'headline':         ev.get('headline'),
        'holdoutSeason':    _season_label(ev.get('holdout_season')),
        'nHoldoutGames':    ev.get('n_holdout_games'),
        'ece':              ev.get('ece'),
        'oofAccuracy':      _pct((ev.get('oof') or {}).get('accuracy')),
        'accuracy':         _pct(ev.get('accuracy')),
        'baselineAccuracy': _pct(ev.get('baseline_acc')),
        'aucRoc':           ev.get('auc'),
        'logLoss':          ev.get('logloss'),
        'brierScore':       ev.get('brier'),
        'featureCount':     len(_feature_columns) if _feature_columns else train.get('n_features'),
        'dataPoints':       train.get('n_games'),
        'nTrials':          train.get('n_trials'),
        'dateFrom':         train.get('date_from'),
        'dateTo':           train.get('date_to'),
        'trainedAt':        train.get('trained_at'),
        'evaluatedAt':      ev.get('evaluated_at'),
        'foldMetrics': [
            {
                'fold':     f['fold'],
                'accuracy': _pct(f['accuracy']),
                'auc':      round(f['auc'], 3),
                'logLoss':  round(f['logloss'], 4),
            }
            for f in ev.get('folds', [])
        ],
        'featureImportance': feat_imp,
    }


# ---------------------------------------------------------------------------
# Endpoint: Takım istatistikleri
# ---------------------------------------------------------------------------
@app.get("/api/team/{team_id}/stats")
async def team_stats(team_id: int, n_games: int = 20):
    """Takımın son n_games maçı (master_games — 05 her gün günceller)."""
    master_path = PROC_DIR / "master_games.parquet"
    if not master_path.exists():
        raise HTTPException(status_code=404, detail="master_games.parquet bulunamadı.")

    tid = str(team_id)
    master = pd.read_parquet(master_path)
    games = master[
        ((master["team_id_home"].astype(str) == tid) | (master["team_id_away"].astype(str) == tid))
        & master["target"].notna()
    ].sort_values("game_date").tail(n_games)

    if games.empty:
        raise HTTPException(status_code=404, detail=f"Takım bulunamadı: {team_id}")

    team_log = build_team_game_log(games)
    team_log = team_log[team_log["team_id"].astype(str) == tid].sort_values("game_date")

    recent = team_log.tail(10)
    summary = {
        "team_id":          team_id,
        "n_games":          len(team_log),
        "win_rate_L5":      float(team_log.tail(5)["won"].mean()),
        "win_rate_L10":     float(recent["won"].mean()),
        "pts_avg_L10":      float(recent["pts_scored"].mean()),
        "pts_allowed_L10":  float(recent["pts_allowed"].mean()),
        "point_diff_L10":   float(recent["plus_minus"].mean()),
        "last_game_date":   str(team_log["game_date"].max().date()),
    }

    cols = ["game_date", "won", "pts_scored", "pts_allowed", "plus_minus",
            "fg_pct", "fg3_pct", "ast", "reb", "tov", "is_home"]
    records = team_log[cols].assign(game_date=team_log["game_date"].dt.strftime("%Y-%m-%d"))
    records = records.astype(object).where(records.notna(), None).to_dict(orient="records")

    return {"summary": summary, "recent_games": records}


# ---------------------------------------------------------------------------
# Endpoint: Tüm takımlar listesi
# ---------------------------------------------------------------------------
@app.get("/api/teams")
async def get_teams():
    try:
        from nba_api.stats.static import teams as nba_teams_static
        teams = nba_teams_static.get_teams()
        return {"teams": teams}
    except ImportError:
        team_path = ROOT / "data" / "raw" / "team.csv"
        if team_path.exists():
            df = pd.read_csv(team_path)
            return {"teams": df.to_dict(orient="records")}
        return {"teams": []}


# ---------------------------------------------------------------------------
# Endpoint: Sonuç doldurma (manuel tetikleme)
# ---------------------------------------------------------------------------
@app.post("/api/fill-results")
async def fill_results_endpoint():
    """Geçmiş maçların sonuçlarını hem offline hem nba_api'den doldurur."""
    from prediction_tracker import fill_results, fill_results_live
    filled_offline = fill_results()
    filled_live    = fill_results_live()

    n_offline = int(filled_offline["correct"].notna().sum()) if filled_offline is not None and not filled_offline.empty else 0
    n_live    = int(filled_live["correct"].notna().sum())    if filled_live    is not None and not filled_live.empty    else 0
    return {"status": "ok", "completedOffline": n_offline, "completedLive": n_live}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )
