"""
api_server.py
-------------
Electron uygulamasının bağlandığı FastAPI local REST sunucusu.
Başlatma: uvicorn src.api_server:app --host 127.0.0.1 --port 8765
veya doğrudan: python src/api_server.py
"""

import json
import pickle
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# src/ içindeyse parent'ı sys.path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import compute_live_team_features

# ---------------------------------------------------------------------------
# Takım renk ve logo eşlemeleri
# ---------------------------------------------------------------------------
ESPN_ID: dict[str, str] = {
    'ATL': 'atl', 'BOS': 'bos', 'BKN': 'bkn', 'CHA': 'cha', 'CHI': 'chi',
    'CLE': 'cle', 'DAL': 'dal', 'DEN': 'den', 'DET': 'det', 'GSW': 'gs',
    'HOU': 'hou', 'IND': 'ind', 'LAC': 'lac', 'LAL': 'lal', 'MEM': 'mem',
    'MIA': 'mia', 'MIL': 'mil', 'MIN': 'min', 'NOP': 'no',  'NYK': 'ny',
    'OKC': 'okc', 'ORL': 'orl', 'PHI': 'phi', 'PHX': 'phx', 'POR': 'por',
    'SAC': 'sac', 'SAS': 'sa',  'TOR': 'tor', 'UTA': 'utah','WAS': 'wsh',
}

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
    allow_origins=["*"],
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


def _run_predictions_if_stale() -> None:
    """Bugünkü live_features.parquet yoksa veya dünden kalma ise tahminleri yeniden üret."""
    live_path = LIVE_DIR / "live_features.parquet"
    try:
        fresh = (
            live_path.exists()
            and pd.to_datetime(
                pd.read_parquet(live_path, columns=["game_date"])["game_date"].max()
            ).date() == date.today()
        )
    except Exception:
        fresh = False

    if fresh:
        print("[API] Bugünkü tahminler zaten güncel, atlanıyor.")
        return

    print("[API] Tahminler eski/eksik — 05_predict_today.py çalıştırılıyor...")
    script = ROOT / "src" / "05_predict_today.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[API] predict_today hata kodu: {result.returncode}")
    else:
        print("[API] Tahminler güncellendi.")


@app.on_event("startup")
async def startup_event():
    ok = _load_model()
    if ok:
        print(f"[API] Model yüklendi. {len(_feature_columns)} feature.")
    else:
        print("[API] Model bulunamadı. Önce 03_train_model.py çalıştırın.")

    # Tahminler güncel değilse otomatik üret
    try:
        _run_predictions_if_stale()
    except Exception as e:
        print(f"[API] auto-predict atlandı: {e}")

    # Önceki günlerin sonuçlarını otomatik doldur (yerel master_games.parquet'ten)
    try:
        from prediction_tracker import fill_results
        filled = fill_results()
        if filled is not None and not filled.empty:
            completed = filled["correct"].notna().sum()
            print(f"[API] fill_results: {completed} tamamlanmış tahmin.")
    except Exception as e:
        print(f"[API] fill_results atlandı: {e}")

    # Canlı sezon sonuçlarını doldur (nba_api — 2024+ maçlar)
    try:
        from prediction_tracker import fill_results_live
        fill_results_live()
    except Exception as e:
        print(f"[API] fill_results_live atlandı: {e}")

    # Sakatlık raporunu arka planda yenile (cache yoksa veya eskimişse)
    try:
        from player_stats import get_injuries
        get_injuries()   # cache varsa hızlı döner
    except Exception as e:
        print(f"[API] injuries cache atlandı: {e}")


# ---------------------------------------------------------------------------
# Yardımcı: live_features satırını frontend Game formatına dönüştür
# ---------------------------------------------------------------------------
def _logo_url(abbr: str) -> str:
    espn_id = ESPN_ID.get(abbr, abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{espn_id}.png"


def _build_game_response(
    row: dict,
    log: pd.DataFrame | None,
    injuries: dict | None = None,
    top_players: dict | None = None,
) -> dict:
    """live_features satırını + team_game_log'u frontend Game arayüzüne çevirir."""
    h_abbr = str(row.get('home_team_abbr') or '')
    a_abbr = str(row.get('away_team_abbr') or '')

    # team_id: int veya str olabilir
    raw_hid = row.get('home_team_id')
    raw_aid = row.get('away_team_id')
    h_tid = str(int(raw_hid)) if raw_hid is not None else ''
    a_tid = str(int(raw_aid)) if raw_aid is not None else ''

    # Form dizileri — son 10 maç (1=G, 0=M), kronolojik sıralı
    def get_form(team_id: str) -> list[int]:
        if log is None or not team_id:
            return []
        tlog = log[log['team_id'] == team_id].sort_values('game_date').tail(10)
        return [int(w) for w in tlog['won'].fillna(0).astype(int).tolist()]

    home_form = get_form(h_tid)
    away_form = get_form(a_tid)

    # Kazanma faktörleri (WhyWinsPanel için)
    # Hem direkt kaydedilmiş alanları hem türetilmiş değerleri kullanır.
    factors = []

    # Elo farkı — predict_games v2'de kaydediliyorsa kullan
    elo_diff = float(row.get('elo_diff') or 0)
    if abs(elo_diff) > 5:
        factors.append({
            'label': 'Elo Rating Farkı',
            'value': min(round(abs(elo_diff) / 10), 20),
            'positive': elo_diff > 0,
        })

    # Plus/minus farkı — kaydedilmişse veya form farkından türet
    pm_diff = float(row.get('diff_plus_minus_L10') or 0)
    if abs(pm_diff) > 0.5:
        factors.append({
            'label': 'Plus/Minus Farkı (L10)',
            'value': min(round(abs(pm_diff) * 2), 15),
            'positive': pm_diff > 0,
        })

    # Form farkı L5 — her zaman mevcut
    h_f5 = float(row.get('home_form_L5') or 0)
    a_f5 = float(row.get('away_form_L5') or 0)
    form_diff = h_f5 - a_f5
    if abs(form_diff) > 0.15 and abs(pm_diff) <= 0.5:
        factors.append({
            'label': f"Son 5 Maç Formu ({int(h_f5 * 5)}G-{5 - int(h_f5 * 5)}M vs {int(a_f5 * 5)}G-{5 - int(a_f5 * 5)}M)",
            'value': min(round(abs(form_diff) * 20), 12),
            'positive': form_diff > 0,
        })

    # Seri farkı — her zaman mevcut
    h_streak = int(row.get('home_streak') or 0)
    a_streak = int(row.get('away_streak') or 0)
    streak_diff = h_streak - a_streak
    if abs(streak_diff) >= 3:
        label = f"Ev Serisi G{h_streak}" if h_streak > 0 else f"Dep. Serisi G{a_streak}"
        factors.append({
            'label': label,
            'value': min(abs(streak_diff) * 2, 10),
            'positive': streak_diff > 0,
        })

    # B2B yorgunluk — her zaman mevcut
    h_b2b = int(row.get('home_b2b') or 0)
    a_b2b = int(row.get('away_b2b') or 0)
    b2b_raw = float(row.get('fatigue_diff_b2b') or (h_b2b - a_b2b))
    if b2b_raw != 0 or h_b2b != a_b2b:
        diff_b2b = h_b2b - a_b2b
        if diff_b2b != 0:
            factors.append({
                'label': 'B2B Yorgunluk Dezavantajı' if diff_b2b > 0 else 'Deplasman B2B Dezavantajı',
                'value': 7,
                'positive': diff_b2b < 0,
            })

    # Dinlenme farkı
    h_rest = row.get('home_rest')
    a_rest = row.get('away_rest')
    if h_rest is not None and a_rest is not None:
        rest_diff = float(h_rest) - float(a_rest)
        if abs(rest_diff) >= 2 and len(factors) < 5:
            factors.append({
                'label': 'Dinlenme Avantajı',
                'value': min(round(abs(rest_diff) * 2), 8),
                'positive': rest_diff > 0,
            })

    # H2H
    h2h_wr = float(row.get('h2h_home_wr') or 0.5)
    if abs(h2h_wr - 0.5) > 0.05:
        factors.append({
            'label': 'H2H İç Saha Oranı',
            'value': round(abs(h2h_wr - 0.5) * 20),
            'positive': h2h_wr > 0.5,
        })

    # Sezon galibiyet oranı farkı
    wct_diff = float(row.get('diff_season_win_pct') or 0)
    if abs(wct_diff) > 0.02 and len(factors) < 5:
        factors.append({
            'label': 'Sezon Galibiyet Oranı',
            'value': min(round(abs(wct_diff) * 30), 12),
            'positive': wct_diff > 0,
        })

    # SRS farkı
    srs_diff = float(row.get('diff_srs_L10') or 0)
    if abs(srs_diff) > 0.5 and len(factors) < 5:
        factors.append({
            'label': 'SRS Farkı (L10)',
            'value': min(round(abs(srs_diff) * 2), 10),
            'positive': srs_diff > 0,
        })

    if not factors:
        prob = float(row.get('prob_home') or 0.5)
        factors.append({
            'label': 'Model Tahmini',
            'value': round(abs(prob - 0.5) * 20),
            'positive': prob > 0.5,
        })

    conf_raw = str(row.get('confidence') or 'DÜŞÜK')
    confidence = CONF_TR_EN.get(conf_raw, conf_raw)  # zaten EN ise geç

    return {
        'id': str(row.get('game_id') or ''),
        'home': {
            'id':           h_tid,
            'name':         str(row.get('home_team_name') or ''),
            'abbreviation': h_abbr,
            'color':        TEAM_COLORS.get(h_abbr, '#1a1a2e'),
            'logoUrl':      _logo_url(h_abbr),
        },
        'away': {
            'id':           a_tid,
            'name':         str(row.get('away_team_name') or ''),
            'abbreviation': a_abbr,
            'color':        TEAM_COLORS.get(a_abbr, '#1a1a2e'),
            'logoUrl':      _logo_url(a_abbr),
        },
        'probHome':       float(row.get('prob_home') or 0.5),
        'confidence':     confidence,
        'gameTime':       str(row.get('status') or ''),
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
    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "feature_count": len(_feature_columns) if _feature_columns else 0,
        "timestamp":     datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoint: Bugünün tahminleri (frontend Game formatında)
# ---------------------------------------------------------------------------
@app.get("/api/today-games")
async def today_games():
    """Bugünün tahmin raporunu döndürür — frontend Game arayüzüne uygun."""
    live_path = LIVE_DIR / "live_features.parquet"

    if not live_path.exists():
        return {
            "date":    str(date.today()),
            "games":   [],
            "count":   0,
            "message": "Henüz tahmin çalıştırılmadı. 05_predict_today.py çalıştırın.",
        }

    df = pd.read_parquet(live_path)
    df = df.where(pd.notna(df), None)

    # team_game_log — form dizileri için (sadece gerekli kolonlar)
    log = None
    log_path = PROC_DIR / "team_game_log.parquet"
    if log_path.exists():
        log = pd.read_parquet(log_path, columns=["team_id", "game_date", "won"])
        log["team_id"] = log["team_id"].astype(str)

    # Sakatlık ve oyuncu verisini yükle (cache'den hızlı)
    injuries   = {}
    top_players: dict[str, list] = {}
    try:
        from player_stats import get_injuries, get_team_top_players
        injuries = get_injuries()

        # Her maç için her iki takımın top oyuncularını çek
        team_ids = set()
        for _, row in df.iterrows():
            if row.get('home_team_id') is not None:
                team_ids.add(str(int(row['home_team_id'])))
            if row.get('away_team_id') is not None:
                team_ids.add(str(int(row['away_team_id'])))

        for tid in team_ids:
            players = get_team_top_players(int(tid))
            if players:
                top_players[tid] = players
    except Exception as e:
        print(f"[API] oyuncu/sakatlık yüklenemedi: {e}")

    games = [
        _build_game_response(dict(row), log, injuries, top_players)
        for _, row in df.iterrows()
    ]

    return {
        "date":  str(date.today()),
        "games": games,
        "count": len(games),
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
# Endpoint: Tahminleri yenile (05_predict_today'i çağırır)
# ---------------------------------------------------------------------------
@app.post("/api/refresh-predictions")
async def refresh_predictions():
    """Tahminleri nba_api'den taze veri çekerek günceller."""
    import subprocess
    script = ROOT / "src" / "05_predict_today.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            return {"success": True, "message": "Tahminler güncellendi."}
        else:
            return {"success": False, "error": result.stderr[-1000:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoint: Geçmiş tahminler (predictions_log.parquet)
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history():
    """Kaydedilmiş tahmin geçmişini döndürür."""
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

    return {
        'accuracy':    66.7,
        'aucRoc':      0.715,
        'logLoss':     0.6085,
        'brierScore':  0.2105,
        'featureCount': len(_feature_columns) if _feature_columns else 236,
        'dataPoints':  26332,
        'foldMetrics': [
            {'fold': 1, 'accuracy': 66.9, 'auc': 0.705, 'logLoss': 0.6092},
            {'fold': 2, 'accuracy': 69.0, 'auc': 0.737, 'logLoss': 0.5868},
            {'fold': 3, 'accuracy': 67.1, 'auc': 0.724, 'logLoss': 0.6023},
            {'fold': 4, 'accuracy': 65.3, 'auc': 0.703, 'logLoss': 0.6219},
            {'fold': 5, 'accuracy': 64.9, 'auc': 0.705, 'logLoss': 0.6224},
        ],
        'featureImportance': feat_imp,
    }


# ---------------------------------------------------------------------------
# Endpoint: Takım istatistikleri
# ---------------------------------------------------------------------------
@app.get("/api/team/{team_id}/stats")
async def team_stats(team_id: int, n_games: int = 20):
    """Takımın son n_games maçının rolling istatistiklerini döndürür."""
    log_path = PROC_DIR / "team_game_log.parquet"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="team_game_log.parquet bulunamadı.")

    log = pd.read_parquet(log_path)
    log["team_id"] = log["team_id"].astype(str)
    team_log = log[log["team_id"] == str(team_id)].sort_values("game_date").tail(n_games)

    if team_log.empty:
        raise HTTPException(status_code=404, detail=f"Takım bulunamadı: {team_id}")

    team_log = team_log.where(pd.notna(team_log), None)
    records = team_log[["game_date", "won", "pts_scored", "pts_allowed",
                         "plus_minus", "fg_pct", "fg3_pct", "ast", "reb",
                         "tov", "is_home"]].to_dict(orient="records")

    recent = team_log.tail(10)
    summary = {
        "team_id":          team_id,
        "n_games":          len(team_log),
        "win_rate_L5":      float(team_log.tail(5)["won"].mean()),
        "win_rate_L10":     float(recent["won"].mean()),
        "pts_avg_L10":      float(recent["pts_scored"].mean()),
        "pts_allowed_L10":  float(recent["pts_allowed"].mean()),
        "point_diff_L10":   float(recent["plus_minus"].mean()),
        "last_game_date":   str(team_log["game_date"].max()),
    }

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
