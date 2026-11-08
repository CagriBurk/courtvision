import { GameLines, LadderRung } from '../types'

/** Adil oran = 1 / olasılık. Çok düşük olasılıkta anlamsız büyür. */
export function fairOdds(p: number): string {
  return p >= 0.02 ? (1 / p).toFixed(2) : '—'
}

export function pct(p: number, digits = 0): string {
  return `%${(p * 100).toFixed(digits)}`
}

/** Handikap değerini işaretli yazar: -12.5, +4.5 */
export function signed(h: number): string {
  return `${h > 0 ? '+' : ''}${h.toFixed(1)}`
}

export interface SpreadOption {
  label: string // "LAL -12.5"
  p: number // seçeneğin tutma olasılığı
  line: number // ev farkı çizgisi
  isFair: boolean
}

/**
 * Merdivendeki her basamağı favori takım açısından bir seçeneğe çevirir.
 * line, ev sahibi farkı üzerindedir:
 *   Ev handikapı -line   → ev farkı > line   → olasılık pOver
 *   Dep handikapı +line  → ev farkı < line   → olasılık 1 - pOver
 * Sıralama: en güvenliden (yüksek olasılık) en riskliye.
 */
export function spreadOptions(
  spread: NonNullable<GameLines['spread']>,
  homeAbbr: string,
  awayAbbr: string,
  side: 'favorite' | 'underdog' = 'favorite'
): SpreadOption[] {
  const homeFavored = spread.fairLine > 0
  const useHome = side === 'favorite' ? homeFavored : !homeFavored
  return spread.ladder
    .map((r: LadderRung) => ({
      label: useHome ? `${homeAbbr} ${signed(-r.line)}` : `${awayAbbr} ${signed(r.line)}`,
      p: useHome ? r.pOver : 1 - r.pOver,
      line: r.line,
      isFair: r.line === spread.fairLine
    }))
    .sort((a, b) => b.p - a.p)
}

/** Favori takımın adil handikap etiketi: "LAL -12.5" */
export function fairHandicapLabel(
  spread: NonNullable<GameLines['spread']>,
  homeAbbr: string,
  awayAbbr: string
): string {
  const fl = spread.fairLine
  return fl > 0 ? `${homeAbbr} ${signed(-fl)}` : `${awayAbbr} ${signed(fl)}`
}
