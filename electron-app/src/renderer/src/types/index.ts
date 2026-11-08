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
  factorsSource?: 'model' | 'rules' // model = XGBoost SHAP katkıları, rules = elle yazılmış kurallar
  homeForm: number[] // son 10 maç: 1=G, 0=M
  awayForm: number[]
  homeInjuries?: InjuredPlayer[]
  awayInjuries?: InjuredPlayer[]
  homeTopPlayers?: TopPlayer[]
  awayTopPlayers?: TopPlayer[]
  lines?: GameLines | null // handikap / toplam (modeller yoksa null)
}

// Barem merdiveni basamağı. Handikapta line ev sahibi farkı üzerindedir:
// pOver = P(ev farkı > line) → "Ev -line" kazanır; 1 - pOver → "Dep +line".
export interface LadderRung {
  line: number
  pOver: number
}

export interface MarginBand {
  side: 'home' | 'away'
  lo: number
  hi: number | null // null = üstü açık (26+)
  p: number
}

export interface TeamScoreDist {
  mean: number
  sd: number
  bins: { lo: number; hi: number; p: number }[]
  best10: { lo: number; hi: number; p: number }
}

export interface GameLines {
  projected: { home: number; away: number } | null
  scores: { home: TeamScoreDist; away: TeamScoreDist } | null
  spread: {
    pred: number // beklenen ev farkı (+ = ev önde)
    sigma: number
    fairLine: number
    pHomeWin: number
    ladder: LadderRung[]
    bands: MarginBand[]
    lowSignal: boolean
  } | null
  total: {
    pred: number
    sigma: number
    fairLine: number
    ladder: LadderRung[]
    lowSignal: boolean // model basit tabanı geçemiyor
    baseline?: string | null
  } | null
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
  // Handikap / toplam (Faz 5 öncesi kayıtlarda null)
  predMargin?: number | null
  predTotal?: number | null
  fairMarginLine?: number | null
  fairTotalLine?: number | null
  homeScore?: number | null
  awayScore?: number | null
}

export interface LineStats {
  n: number
  mae: number
  bias: number
  within5: number
  within10: number
  ladderCalibration: { meanPred: number; fracOver: number; n: number }[]
}

export interface ModelHealth {
  oofAccuracy: number
  brierScore: number
  lastTuning: string
  recentPredictions: { correct: number; total: number }
}
