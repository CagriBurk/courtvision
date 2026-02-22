# CourtVision — NBA AI Prediction System

A machine learning-powered desktop app for predicting NBA game outcomes. The XGBoost model is trained on 26,000+ games spanning the 2000–2026 seasons.

## Features

- **Daily predictions** — Win/loss probabilities automatically computed each morning for today's games
- **Why wins analysis** — Key factors based on recent form, fatigue, head-to-head history, and stat differentials
- **Team detail modal** — Last 15 games trend chart, rolling Pts / Opp / FG% stats
- **Injury tracking** — Out and Doubtful players surfaced automatically per game
- **Game notifications** — Desktop alert 30 minutes before tip-off for watched games
- **Prediction history** — Accuracy tracking broken down by confidence tier
- **Model metrics** — OOF Accuracy 66.7%, AUC-ROC 0.715, Brier Score 0.210

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy (OOF) | **66.7%** |
| AUC-ROC | **0.715** |
| Log-Loss | **0.608** |
| Brier Score | **0.210** |
| Baseline (always pick home) | 57.9% |
| Features | 236 |
| Training data | 26,332 games (2000–2026) |
| Method | XGBoost + Optuna (150 trials) + 5-Fold TimeSeriesSplit |

## Tech Stack

**Backend:** Python · FastAPI · XGBoost · Optuna · nba_api · pandas
**Frontend:** Electron · React 19 · TypeScript · Tailwind CSS · Recharts

## Setup

### Requirements
- Python 3.10+
- Node.js 18+

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the dataset (first-time setup)

```bash
python src/06_update_data.py --seasons 2023-24 2024-25 2025-26
python src/01_build_master.py
python src/02_build_features.py
python src/03_train_model.py
```

### 3. Run in development

```bash
cd electron-app
npm install
npm run dev
```

The app automatically starts the FastAPI server on launch. On first run, you will be prompted to select the project root folder.

### 4. Build installer (Windows)

```bash
cd electron-app
npm run build:win
```

Produces `dist/CourtVision-1.0.0-setup.exe`.

## Project Structure

```
├── src/
│   ├── 01_build_master.py       # Merge raw data sources
│   ├── 02_build_features.py     # Rolling feature engineering
│   ├── 03_train_model.py        # XGBoost + Optuna training
│   ├── 04_evaluate_model.py     # OOF metrics & reports
│   ├── 05_predict_today.py      # Daily prediction pipeline
│   ├── 06_update_data.py        # Fetch missing seasons via nba_api
│   ├── feature_engineering.py   # Shared rolling feature module
│   ├── prediction_tracker.py    # Prediction log & result filling
│   └── api_server.py            # FastAPI server
├── electron-app/                # Electron + React frontend
├── data/                        # Data files (gitignored)
├── models/                      # Trained model artifacts (gitignored)
└── requirements.txt
```

## License

Personal use.
