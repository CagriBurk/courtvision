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
  probHomeMarket?: number | null // piyasanın marjsız ev kazanma olasılığı (çizgi yoksa null)
}

// Piyasanın (ESPN / DraftKings kapanış çizgisi) söyledikleri ve modelin kendi görüşü.
// marketMargin / modelMargin ev sahibi farkıdır (+ = ev önde).
export interface MarketView {
  provider: string
  spreadHome: number | null // ev handikapı ("DAL -7.5" → -7.5)
  marketMargin: number | null
  marketTotal: number | null
  pHome: number | null
  source: { margin?: 'market' | 'model'; total?: 'market' | 'model' }
  w: { margin?: number; total?: number } // model payı: merkez = piyasa + w·(model − piyasa)
  modelMargin: number | null
  modelTotal: number | null
  modelFairMargin: number | null
  modelFairTotal: number | null
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
  market?: MarketView | null // piyasa çizgisi yoksa null
  spread: {
    pred: number // beklenen ev farkı (+ = ev önde)
    sigma: number
    fairLine: number
    pHomeWin: number
    ladder: LadderRung[]
    bands: MarginBand[]
    source?: 'market' | 'model' // olasılıklar piyasa çizgisi etrafında mı, modelden mi
    lowSignal: boolean
  } | null
  total: {
    pred: number
    sigma: number
    fairLine: number
    ladder: LadderRung[]
    source?: 'market' | 'model'
    lowSignal: boolean // model basit tabanı geçemiyor (yalnız model kaynaklıysa)
    baseline?: string | null
  } | null
}

export type NavPage =
  | 'dashboard'
  | 'coupons'
  | 'reportCard'
  | 'history'
  | 'metrics'
  | 'guide'
  | 'settings'

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

// ---------------------------------------------------------------------------
// Kuponlar ve karne (Faz 6)
// ---------------------------------------------------------------------------
export interface CouponLeg {
  gameId: string
  home: string
  away: string
  market: 'spread' | 'total'
  team: string | null // handikapta seçilen (favori) takım
  side: 'home' | 'away' | 'over'
  line: number // handikapta ev farkı, toplamda toplam sayı çizgisi
  handicap: number | null
  label: string // "LAL -4.5" / "Üst 222.5"
  fair: string // adil çizgi: "LAL -3.5" / "228.5"
  source?: 'market' | 'model' // olasılığı veren dağılım (eski kayıtlarda yok = model)
  p: number
  fairOdds: number
  minOdds: number
  gameTimeUtc?: string | null
  won?: boolean | null
  score?: [number, number] | null
}

export interface Coupon {
  id: string
  date: string
  tier: 'guvenli' | 'dengeli' | 'riskli' | 'ust'
  label: string
  target: number
  experimental?: boolean // üst kuponu: toplam modeli tabanı geçemiyor
  legs: CouponLeg[]
  p: number
  fairOdds: number
  minOdds: number
  locked: boolean
  playedOdds: number | null
  result: 'won' | 'lost' | null
}

export interface CouponGame {
  gameId: string
  home: string
  away: string
  favourite: string
  fairLabel: string // fiyatlamanın merkezi: piyasa çizgisi (ya da piyasa yoksa model)
  fairTotal: number | null
  center: number
  source: 'market' | 'model'
  marketLabel: string | null // "DAL -7.5"
  modelLabel: string | null // modelin kendi adil çizgisi
  eligible: boolean
  reasons: string[]
  gameTimeUtc?: string | null
}

export interface CouponsToday {
  date: string | null
  nLegs: number
  source?: 'market' | 'model'
  note: string | null
  skipped?: { tier: string; label: string; reason: string }[]
  games: CouponGame[]
  coupons: Coupon[]
}

export interface CalibrationRow {
  lo: number
  hi: number
  n: number
  meanPred: number
  hitRate: number
  ciLo: number
  ciHi: number
}

export interface Calibration {
  n: number
  rows: CalibrationRow[]
  meanPred: number | null
  hitRate: number | null
}

export interface CouponTierStats {
  tier: Coupon['tier']
  label: string
  n: number
  won: number
  hitRate: number | null
  expected: number | null
  legs: Calibration
  nPlayed: number
  profit: number | null
  roi: number | null
}

export interface ReportCard {
  side: Calibration
  handicap: {
    calibration: Calibration
    bands: {
      side: 'home' | 'away'
      lo: number
      hi: number | null
      n: number
      pred: number
      obs: number
    }[]
    error: LineStats | null
  }
  total: { calibration: Calibration; error: LineStats | null }
  coupons: {
    n: number
    pending: number
    tiers: CouponTierStats[]
    played: { n: number; won: number; profit: number | null; roi: number | null }
  }
  market?: MarketReport
  recentCoupons: Coupon[]
}

// Canlı model ↔ piyasa kıyası (yalnız piyasa çizgisi olan, bitmiş maçlar)
export interface MarketLineReport {
  n: number
  modelMae?: number
  marketMae?: number
  modelCloser?: number // modelin gerçeğe piyasadan yakın olduğu maç oranı
  w?: number // canlı model payı (30+ maçtan sonra)
  wSe?: number
}

export interface MarketReport {
  margin: MarketLineReport
  total: MarketLineReport
  winner: {
    n: number
    modelLogloss?: number
    marketLogloss?: number
    modelAcc?: number
    marketAcc?: number
  }
}
