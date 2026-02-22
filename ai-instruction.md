# NBA Maç Tahmin Sistemi — AI Talimat Dosyası

## Proje Amacı
NBA Regular Season maçlarında **ev sahibi kazanır mı?** sorusunu cevaplamak.
XGBoost + rolling pre-game features. Hedef: makul bir insan kullanıcıya günlük tahmin + bağlam sağlamak.

---

## Teknik Stack
- **ML:** XGBoost (Optuna ile tune edildi, TimeSeriesSplit 5-fold)
- **Veri:** Kaggle Wyatt Walsh NBA DB (CSV) + nba_api (2023–günümüz boşluk doldurma)
- **Backend:** FastAPI (port 8765) — Electron app için local REST API
- **Frontend:** Electron + React 19 + Tailwind CSS v4 + TypeScript ✅ TAMAMLANDI
- **Python 3.13**, pandas, pyarrow, scikit-learn, optuna, shap (planlı)

---

## Proje Yapısı

```
Pred/
├── data/
│   ├── raw/           → game.csv, other_stats.csv, line_score.csv, team.csv (Kaggle)
│   ├── api_cache/     → nba_api çekilen sezonlar (parquet)
│   ├── processed/     → master_games.parquet, team_game_log.parquet, features.parquet
│   ├── live/          → today_schedule.json, live_features.parquet
│   └── predictions/   → predictions_log.parquet (prediction tracker — yapılacak)
├── src/
│   ├── 01_build_master.py        → Kaggle CSV + API cache birleştirme
│   ├── 02_build_features.py      → Feature matrix üretimi (Elo/SRS entegre)
│   ├── 03_train_model.py         → XGBoost + Optuna eğitimi
│   ├── 04_evaluate_model.py      → Walk-forward OOF değerlendirme
│   ├── 05_predict_today.py       → Günlük canlı tahmin (Elo fix gerekli — TODO)
│   ├── 06_update_data.py         → nba_api'den eksik sezonları çek
│   ├── feature_engineering.py   → Paylaşımlı feature modülü (02, 05, api_server import eder)
│   ├── elo_srs.py                → Elo rating, SRS, rakip kalitesi (YENİ ✅)
│   ├── prediction_tracker.py     → Tahmin kayıt + sonuç doldurma (YAPILACAK)
│   └── api_server.py             → FastAPI sunucu
├── electron-app/                 → Electron + React + Tailwind UI (✅ TAMAMLANDI)
│   ├── src/renderer/src/
│   │   ├── App.tsx               → 3 panelli ana layout
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx     → Günlük tahminler (mock → API bağlanacak)
│   │   │   ├── History.tsx       → Geçmiş tahminler + ağırlıklı accuracy
│   │   │   ├── ModelMetrics.tsx  → OOF metrikleri + feature importance chart
│   │   │   └── Settings.tsx      → API endpoint, auto-update, hakkında
│   │   ├── components/
│   │   │   ├── Sidebar.tsx       → Sol navigasyon + NBA logosu
│   │   │   ├── PredictionCard.tsx → Maç kartı (ESPN logolu)
│   │   │   ├── WinProbBar.tsx    → Kazanma olasılığı bar
│   │   │   ├── WhyWinsPanel.tsx  → "Neden kazanır?" detay paneli
│   │   │   ├── FormChart.tsx     → Son 10 maç Recharts line chart
│   │   │   └── ModelHealthPanel.tsx → Model sağlık metrikleri
│   │   ├── data/
│   │   │   ├── mockData.ts       → 6 maç mock verisi (ESPN logo URL'leri)
│   │   │   └── historyData.ts    → Mock geçmiş + feature importance data
│   │   └── types/index.ts        → TypeScript interfaces
├── models/
│   ├── xgb_best.pkl, optuna_study.pkl, feature_columns.pkl
├── reports/
│   ├── evaluation_report.html, predictions_YYYY-MM-DD.html
└── ai-instruction.md
```

---

## Pipeline Çalıştırma Sırası

```bash
# Tek seferlik kurulum
python src/06_update_data.py --seasons 2023-24 2024-25 2025-26
python src/01_build_master.py
python src/02_build_features.py
python src/03_train_model.py       # --trials 150, GPU: device="cuda"
python src/04_evaluate_model.py

# Günlük kullanım (sabah 10-11 TR)
python src/05_predict_today.py
# veya FastAPI üzerinden:
uvicorn src.api_server:app --host 127.0.0.1 --port 8765

# Electron uygulaması
cd electron-app && npm run dev
```

---

## Mevcut Model Performansı

### Eski model (172 feature, 2026-02-22 sabahı)
| Metrik | Değer | Baseline |
|--------|-------|----------|
| Accuracy (OOF) | **66.4%** | 57.9% |
| AUC-ROC | **0.714** | — |
| Log-Loss | 0.610 | 0.679 |
| Brier Score | 0.211 | — |

### Yeni model (236 feature, 2026-02-22) ✅ TAMAMLANDI
| Metrik | Değer | Baseline |
|--------|-------|----------|
| Accuracy (OOF) | **66.7%** | 57.9% |
| AUC-ROC | **0.715** | — |
| Log-Loss | **0.6085** | 0.679 |
| Brier Score | **0.2105** | — |

**Top feature'lar (XGBoost gain):** elo_diff (3.49%), diff_season_pd (3.41%), diff_season_win_pct (3.06%), elo_win_prob (3.06%), diff_plus_minus_L10 (2.26%), diff_srs_L10 (1.87%)

Veri: 26.332 maç × **236 feature** (2000–2026)

---

## Feature Engineering — Kritik Kurallar

1. **shift(1) zorunlu:** Tüm rolling hesaplamalar mevcut maçın verisini görmemeli
2. **H2H strict `<` date filter:** Mevcut maç dahil edilmez
3. **Season reset:** Cumulative stats `groupby('season_id')` ile sıfırlanır
4. **Hedef:** `target` = 1 (ev sahibi kazandı), 0 (deplasman kazandı)
5. **`pts_home/away`, `wl_home/away` ASLA feature olmaz** — data leakage

---

## Feature Kategorileri (236 feature)

| Kategori | Örnekler |
|----------|---------|
| Yorgunluk | days_since_last_game, is_back_to_back, games_in_L5/7_days |
| Rolling stats | won/pts/pm/fg_pct/reb/ast... L3, L5, L10 (home_ ve away_ prefix) |
| Verimlilik | ts_pct_L5/10, off_eff_L5/10 |
| Form trend | won_trend, plus_minus_trend (L3-L10 delta) |
| Sezon | season_game_num, season_win_pct, season_pct_complete |
| Ev/Deplasman | home_win_rate_L10, away_win_rate_L10 |
| H2H | h2h_n_games, h2h_home_win_rate, h2h_pts_diff_mean |
| Diff features | diff_win_rate_L5/10, diff_plus_minus, diff_streak, vb. |
| **Elo** ✅ YENİ | elo_home, elo_away, elo_diff, elo_win_prob |
| **SRS** ✅ YENİ | home_srs_L10, away_srs_L10, srs_L5, diff_srs_L10 |
| **Opp Quality** ✅ YENİ | opp_quality_L10/L5, opp_pdiff_L10, diff_opp_quality |

---

## Faz 1 — TAMAMLANDI ✅

### Yapılanlar:
- `src/elo_srs.py` oluşturuldu: `compute_elo()`, `get_current_elo()`, `compute_opponent_quality()`, `compute_srs()`
- `src/02_build_features.py` güncellendi: Elo/SRS pipeline entegrasyonu
- `src/feature_engineering.py` güncellendi: 236 feature, yeni diff pairs, Elo sütunları pivot'a eklendi
- `features.parquet` yeniden üretildi: 26.332 × 245 sütun
- Model yeniden eğitiliyor: 150 trial, log-loss ~0.608

### Bekleyen küçük düzeltme:
- `src/05_predict_today.py`: live tahmin için `get_current_elo(master)` çağrısı eksik
  - Model Elo feature bekliyor ama live feature vektörü bunları içermiyor → YAPILACAK

---

## Frontend (Electron App) — TAMAMLANDI ✅

### Teknoloji:
- Electron + React 19 + TypeScript + Tailwind CSS v4 + Recharts + lucide-react
- Pencere: 1440×900

### Sayfalar:
- **Dashboard:** 2-sütun maç kartı grid, ESPN logoları, kazanma olasılığı bar, form dots, sağda "Neden Kazanır?" paneli + Recharts form chart + model health
- **History:** Günlük gruplu geçmiş tahminler, doğru/yanlış ikonlar, ağırlıklı accuracy (HIGH×3, MED×2, LOW×1)
- **Model Metrikleri:** 4 ana metrik kartı, fold bazlı tablo, Top 15 feature importance yatay bar chart
- **Ayarlar:** FastAPI endpoint test, auto-update toggle + saat, hakkında

### Şu an mock data kullanıyor → API entegrasyonu sonraki adım

### ESPN CDN logo sistemi:
```typescript
// https://a.espncdn.com/i/teamlogos/nba/500/{espnId}.png
// GSW→gs, NYK→ny, SAS→sa, BOS→bos, LAL→lal, MIA→mia...
// NBA logosu: https://a.espncdn.com/i/teamlogos/leagues/500/nba.png
// img onError → fallback renkli daire badge
```

---

## Aktif Geliştirme — Şu An Yapılacaklar

### 1. Elo Fix — 05_predict_today.py
```python
# Eksik olan kısım:
from elo_srs import get_current_elo
current_elo = get_current_elo(master)  # {team_id: elo}
# Feature vektörüne ekle: elo_home, elo_away, elo_diff, elo_win_prob
```

### 2. Prediction Tracker — src/prediction_tracker.py
```
data/predictions/predictions_log.parquet
  game_id, date, home_team, away_team, prob_home, confidence,
  predicted_winner, actual_winner (null→doldurulacak), correct
```
- `save_predictions(predictions_df)` → append to parquet
- `fill_results(master)` → actual_winner, correct doldur
- `get_stats()` → weighted accuracy, by tier breakdown

---

## Sonraki Faz (Faz 2) — Planned

**Oyuncu Önem + Injury Overlay (modele dahil edilmeyecek, UI katmanı):**
- nba_api `LeagueDashPlayerStats` → sezon verimliliğine göre top-5 oyuncu (PTS + 1.2×REB + 1.5×AST - TOV)
- nba_api `CommonTeamRoster` + injury report → eksik oyuncular
- Eksik önemli oyuncuları UI'da vurgula: `⚠️ LeBron (1. verimli) OYNAMAYACAK`
- Model tahminine dokunmaz, kullanıcıya bağlam sağlar
- Prediction card'a küçük uyarı badge'i

---

## Önemli Teknik Notlar

- `feature_engineering.py` hem offline (02) hem live (05) tarafından import edilir — değiştirilirse her iki path'i test et
- model `device="cuda"` ile eğitildi; inference için `.set_params(device="cpu")` çağrılıyor (05_predict_today.py ve api_server.py'de mevcut)
- `ScoreboardV3` kullanılıyor (V2 deprecated, 2025-26 sezonu için bug var)
- `GAME_DATE` format: `pd.to_datetime(..., format="mixed")` kullan
- `game_id`, `team_id_*`, `season_id` her yerde `str`'e cast edilmeli (pyarrow mixed-type hatası)
- nba_api rate limit: çağrılar arası `time.sleep(0.7)` zorunlu
- ScoreboardV3 gameCode parsing: `"20260222/CLEOKC"` → away=`CLE`, home=`OKC`
- Tailwind v4: `@import "tailwindcss"` in CSS, `@tailwindcss/vite` plugin in vite config — tailwind.config.js gerekmez
- CSP (index.html): `img-src 'self' data: https://a.espncdn.com https://cdn.nba.com`
