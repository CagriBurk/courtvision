export interface Team {
  id: string
  name: string
  abbreviation: string
  color: string
  logoUrl: string // ESPN CDN logo
}

export interface WinFactor {
  label: string
  detail?: string // gösterilen gerçek değer, ör. "201 puan"
  value: number // çubuk uzunluğu için göreli güç (0–20)
  positive: boolean // true = ev sahibi lehine
}

export interface InjuredPlayer {
  name: string
  id: string
  status: 'Out' | 'Questionable' | 'Doubtful' | 'Day-To-Day' | string
  detail: string
}

export interface TopPlayer {
  name: string
  id: number
  pts: number
  reb: number
  ast: number
  impact: number
}

export interface Game {
  id: string
  home: Team
  away: Team
  probHome: number // 0-1
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  gameTime: string // NBA durum metni: "8:00 pm ET", "Q3 5:21", "Final"
  gameTimeUtc?: string | null // "2026-10-31T23:00:00Z"
  gameStatus?: number // 1=başlamadı, 2=oynanıyor, 3=bitti
  homeScore?: number | null
  awayScore?: number | null
  venue: string
  factors: WinFactor[]
  homeForm: number[] // son 10 maç: 1=G, 0=M
  awayForm: number[]
  homeInjuries?: InjuredPlayer[]
  awayInjuries?: InjuredPlayer[]
  homeTopPlayers?: TopPlayer[]
  awayTopPlayers?: TopPlayer[]
}

export type NavPage = 'dashboard' | 'history' | 'metrics' | 'settings'

export interface PredictionRecord {
  gameId: string
  date: string
  homeTeam: string
  awayTeam: string
  probHome: number
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  predictedWinner: string
  actualWinner: string | null
  correct: boolean | null
}

export interface ModelHealth {
  oofAccuracy: number
  brierScore: number
  lastTuning: string
  recentPredictions: { correct: number; total: number }
}
