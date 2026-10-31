import { Game } from '../types'

/** Tip-off time in the viewer's local time zone (Turkey → "02:00"). */
export function formatTipoff(utc?: string | null): string | null {
  if (!utc) return null
  const d = new Date(utc)
  if (isNaN(d.getTime())) return null
  return d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })
}

export function tipoffMs(game: Game): number | null {
  if (!game.gameTimeUtc) return null
  const ms = Date.parse(game.gameTimeUtc)
  return isNaN(ms) ? null : ms
}

export const isLive = (game: Game): boolean => game.gameStatus === 2
export const isFinal = (game: Game): boolean => game.gameStatus === 3

/** Card footer text: local tip-off time, live clock, or final score. */
export function gameStatusLabel(game: Game): string {
  const score =
    game.homeScore != null && game.awayScore != null ? `${game.homeScore}–${game.awayScore}` : ''
  if (isFinal(game)) return score ? `Bitti · ${score}` : 'Bitti'
  if (isLive(game)) return score ? `${game.gameTime} · ${score}` : game.gameTime
  return formatTipoff(game.gameTimeUtc) ?? game.gameTime
}

/** Was the prediction right? null until the game is final with a known score. */
export function predictionCorrect(game: Game): boolean | null {
  if (!isFinal(game) || game.homeScore == null || game.awayScore == null) return null
  return game.probHome >= 0.5 === game.homeScore > game.awayScore
}

const NAME_SUFFIXES = new Set(['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv'])

/** "Jaren Jackson Jr." → "Jackson" (skips Jr./Sr./II/III suffixes). */
export function lastName(fullName: string): string {
  const parts = fullName.trim().split(/\s+/)
  while (parts.length > 1 && NAME_SUFFIXES.has(parts[parts.length - 1].toLowerCase())) {
    parts.pop()
  }
  return parts[parts.length - 1]
}
