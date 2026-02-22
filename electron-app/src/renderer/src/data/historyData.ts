import { PredictionRecord } from '../types'

export const mockHistory: PredictionRecord[] = [
  { gameId: 'h1', date: '2026-02-21', homeTeam: 'BOS', awayTeam: 'NYK', probHome: 0.681, confidence: 'HIGH', predictedWinner: 'BOS', actualWinner: 'BOS', correct: true },
  { gameId: 'h2', date: '2026-02-21', homeTeam: 'LAL', awayTeam: 'DEN', probHome: 0.423, confidence: 'MEDIUM', predictedWinner: 'DEN', actualWinner: 'DEN', correct: true },
  { gameId: 'h3', date: '2026-02-21', homeTeam: 'MIL', awayTeam: 'CHI', probHome: 0.714, confidence: 'HIGH', predictedWinner: 'MIL', actualWinner: 'CHI', correct: false },
  { gameId: 'h4', date: '2026-02-21', homeTeam: 'PHX', awayTeam: 'SAC', probHome: 0.552, confidence: 'LOW', predictedWinner: 'PHX', actualWinner: 'PHX', correct: true },
  { gameId: 'h5', date: '2026-02-20', homeTeam: 'GSW', awayTeam: 'POR', probHome: 0.731, confidence: 'HIGH', predictedWinner: 'GSW', actualWinner: 'GSW', correct: true },
  { gameId: 'h6', date: '2026-02-20', homeTeam: 'MIA', awayTeam: 'ATL', probHome: 0.608, confidence: 'MEDIUM', predictedWinner: 'MIA', actualWinner: 'ATL', correct: false },
  { gameId: 'h7', date: '2026-02-20', homeTeam: 'OKC', awayTeam: 'UTA', probHome: 0.689, confidence: 'HIGH', predictedWinner: 'OKC', actualWinner: 'OKC', correct: true },
  { gameId: 'h8', date: '2026-02-20', homeTeam: 'TOR', awayTeam: 'DET', probHome: 0.571, confidence: 'LOW', predictedWinner: 'TOR', actualWinner: 'DET', correct: false },
  { gameId: 'h9', date: '2026-02-19', homeTeam: 'MIN', awayTeam: 'NOP', probHome: 0.663, confidence: 'HIGH', predictedWinner: 'MIN', actualWinner: 'MIN', correct: true },
  { gameId: 'h10', date: '2026-02-19', homeTeam: 'PHI', awayTeam: 'IND', probHome: 0.534, confidence: 'LOW', predictedWinner: 'PHI', actualWinner: 'IND', correct: false },
  { gameId: 'h11', date: '2026-02-19', homeTeam: 'CLE', awayTeam: 'WAS', probHome: 0.742, confidence: 'HIGH', predictedWinner: 'CLE', actualWinner: 'CLE', correct: true },
  { gameId: 'h12', date: '2026-02-18', homeTeam: 'DEN', awayTeam: 'LAC', probHome: 0.625, confidence: 'MEDIUM', predictedWinner: 'DEN', actualWinner: 'DEN', correct: true },
  { gameId: 'h13', date: '2026-02-18', homeTeam: 'SAS', awayTeam: 'HOU', probHome: 0.389, confidence: 'MEDIUM', predictedWinner: 'HOU', actualWinner: 'HOU', correct: true },
  { gameId: 'h14', date: '2026-02-18', homeTeam: 'DAL', awayTeam: 'MEM', probHome: 0.657, confidence: 'HIGH', predictedWinner: 'DAL', actualWinner: 'MEM', correct: false },
  { gameId: 'h15', date: '2026-02-17', homeTeam: 'NYK', awayTeam: 'BKN', probHome: 0.701, confidence: 'HIGH', predictedWinner: 'NYK', actualWinner: 'NYK', correct: true },
  { gameId: 'h16', date: '2026-02-17', homeTeam: 'ORL', awayTeam: 'CHA', probHome: 0.618, confidence: 'MEDIUM', predictedWinner: 'ORL', actualWinner: 'ORL', correct: true },
  { gameId: 'h17', date: '2026-02-17', homeTeam: 'CHI', awayTeam: 'MIL', probHome: 0.421, confidence: 'MEDIUM', predictedWinner: 'MIL', actualWinner: 'CHI', correct: false },
  { gameId: 'h18', date: '2026-02-17', homeTeam: 'UTA', awayTeam: 'POR', probHome: 0.543, confidence: 'LOW', predictedWinner: 'UTA', actualWinner: 'UTA', correct: true },
]

// Gerçek feature importance — XGBoost gain, normalize edilmiş (%)
// Kaynak: models/xgb_best.pkl, 236 feature, 2026-02-22 eğitimi
export const mockFeatureImportance = [
  { feature: 'elo_diff', importance: 3.49 },
  { feature: 'diff_season_pd', importance: 3.41 },
  { feature: 'diff_season_win_pct', importance: 3.06 },
  { feature: 'elo_win_prob', importance: 3.06 },
  { feature: 'diff_plus_minus_L10', importance: 2.26 },
  { feature: 'diff_srs_L10', importance: 1.87 },
  { feature: 'diff_off_eff_L10', importance: 1.81 },
  { feature: 'elo_home', importance: 1.03 },
  { feature: 'home_season_pd_before', importance: 0.77 },
  { feature: 'home_point_diff_L10', importance: 0.71 },
  { feature: 'home_plus_minus_L10', importance: 0.70 },
  { feature: 'home_off_eff_L10', importance: 0.64 },
  { feature: 'elo_away', importance: 0.62 },
  { feature: 'diff_plus_minus_L5', importance: 0.59 },
  { feature: 'home_season_win_pct', importance: 0.58 },
  { feature: 'home_srs_L10', importance: 0.54 },
  { feature: 'home_won_L10', importance: 0.52 },
  { feature: 'away_games_in_last_5_days', importance: 0.52 },
  { feature: 'away_season_pd_before', importance: 0.52 },
  { feature: 'fatigue_diff_b2b', importance: 0.51 },
]

// Gerçek fold metrikleri — 04_evaluate_model.py çıktısı, 2026-02-22
export const mockFoldMetrics = [
  { fold: 1, accuracy: 66.9, auc: 0.705, logLoss: 0.6092 },
  { fold: 2, accuracy: 69.0, auc: 0.737, logLoss: 0.5868 },
  { fold: 3, accuracy: 67.1, auc: 0.724, logLoss: 0.6023 },
  { fold: 4, accuracy: 65.3, auc: 0.703, logLoss: 0.6219 },
  { fold: 5, accuracy: 64.9, auc: 0.705, logLoss: 0.6224 },
]
