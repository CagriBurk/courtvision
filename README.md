# CourtVision — NBA Game Prediction & Market Analysis

A personal desktop app (Electron + FastAPI) that predicts NBA games with XGBoost models
trained on 30,000+ regular-season games (2001–2026), turns those predictions into full score
distributions, and measures itself honestly against the betting market's closing lines.

It is a hobby and learning project, not a betting tool. Its most interesting result is a
negative one: a model built from public box-score data beats simple baselines and lands
within ~0.4 points of the closing line, but adds no information the market does not already
have. The apparent edge it first showed came from a hindsight leak, found and fixed with 16,900
historical closing lines.
See [EVALUATION.md](EVALUATION.md).

## Features

- **Daily predictions** — win probability, expected margin and total for every game, refreshed
  through the day with the injury report.
- **Score distributions** — each prediction becomes a calibrated distribution over integer
  scores, so any handicap or over/under line, winning-margin band and team score range gets a
  probability.
- **Market lines** — DraftKings handicap, total and moneyline via ESPN on every run, plus
  16,900 historical closing lines (2012-13 onward) for evaluation.
- **Market-centred pricing** — when a line exists, probabilities are built around it with the
  market's own error distribution; the model's view is shown next to it.
- **Coupons and report card** — probability-tier handicap coupons (one game per coupon) and a
  calibration report card ("when we said 70%, how often did it happen?", Wilson intervals),
  including a live model-vs-market comparison.
- **Explanations** — SHAP contributions grouped into readable factors (form, Elo, rest,
  missing players…), team pages, prediction history.

## Results (2025-26 holdout, 1,225 games never seen in tuning)

| | Model | Simple baseline | Market (closing line) |
|---|---|---|---|
| Winner accuracy / log-loss | 69.0% / 0.592 | 55.4% (always home) | 69.2% / **0.568** |
| Margin MAE (points) | 11.31 | 11.61 (Elo) | **10.89** |
| Total MAE (points) | 15.41 | 15.35 (last-10 average) | **14.60** |

Does the model add anything to the market? In `actual ≈ market + w·(model − market)`, the
weight was **w = 0.32 ± 0.11** while player absences were read from each game's own box score,
and **w = 0.00 ± 0.11** once only absences known before tip-off were counted — the model's
whole edge was hindsight (it cost ~0.25 points of margin error). The app therefore prices
around the market and shows the model as a second opinion. Details and data-quality lessons:
[EVALUATION.md](EVALUATION.md).

## Tech Stack

**Backend:** Python · FastAPI · XGBoost · Optuna · nba_api · pandas · SciPy
**Frontend:** Electron · React 19 · TypeScript · Tailwind CSS · TanStack Query · Recharts
**Data:** nba_api (games, box scores) · ESPN (injury report, betting lines)

## Setup

### Requirements
- Python 3.11 (virtual environment in `.venv`)
- Node.js 18+
- An NVIDIA GPU is used for training (`device: cuda`); inference runs on the CPU.

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd electron-app && npm install
```

### 2. Build the data and models (first time)

```bash
python src/06_update_data.py --seasons 2024-25 2025-26 2026-27   # missing seasons via nba_api
python src/01_build_master.py
python src/02_build_features.py
python src/03_train_model.py --task all --trials 100            # win, margin, total (~75 min)
python src/04_evaluate_model.py                                 # holdout, calibration, distributions
python src/espn_odds.py --backfill --season 2012-13 ... 2026-27  # closing lines (resumable)
python src/09_market_eval.py --refresh                          # model vs market, market distribution
```

On Windows `MODEL_EGIT.bat` runs steps 02–04 in one go.

### 3. Run

```bash
cd electron-app
npm run dev        # or double-click CourtVision.bat
```

The app starts the FastAPI server (port 8765) itself and runs the daily prediction script.

### 4. Tests

```bash
.venv\Scripts\python -m pytest tests -q
cd electron-app && npm run typecheck
```

## Project Structure

```text
├── src/
│   ├── 01_build_master.py       # Merge game logs into one row per game
│   ├── 02_build_features.py     # Leak-free rolling features (shift(1) everywhere)
│   ├── 03_train_model.py        # XGBoost + Optuna, per task (win / margin / total)
│   ├── 04_evaluate_model.py     # Walk-forward OOF + holdout, error distributions
│   ├── 05_predict_today.py      # Daily pipeline: schedule, injuries, market lines, predictions, coupons
│   ├── 06_update_data.py        # Fetch seasons via nba_api
│   ├── 07_experiment.py         # Feature-group ablation on the holdout
│   ├── 08_line_experiment.py    # Margin / total model experiments
│   ├── 09_market_eval.py        # Model vs closing lines; writes the market distribution
│   ├── feature_engineering.py   # Shared feature pipeline (training and live)
│   ├── score_dist.py            # Prediction → integer score distribution → line probabilities
│   ├── espn_odds.py             # ESPN / DraftKings lines: today + resumable backfill
│   ├── market_eval.py           # Blend weight, ATS record, market error distribution
│   ├── market_blend.py          # Market-centred lines for the app
│   ├── coupons.py, coupon_tracker.py, report_card.py
│   ├── prediction_tracker.py    # Prediction log, results, live error stats
│   └── api_server.py            # FastAPI server
├── tests/                       # pytest suite
├── electron-app/                # Electron + React frontend
├── data/                        # Data files (gitignored)
├── models/                      # Trained model artifacts (gitignored)
├── EVALUATION.md                # Model vs market evaluation report
└── GUNLUK_KULLANIM.md           # Daily usage guide (Turkish)
```

## License

Personal use.
