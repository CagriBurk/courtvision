import { TrendingUp, TrendingDown, MousePointerClick, AlertTriangle, Users } from 'lucide-react'
import { useState } from 'react'
import { Game, InjuredPlayer, Team, TopPlayer } from '../types'
import FormChart from './FormChart'

function TeamLogoSmall({ team }: { team: Team }) {
  const [err, setErr] = useState(false)
  if (!err) {
    return (
      <img
        src={team.logoUrl}
        alt={team.abbreviation}
        className="w-9 h-9 object-contain shrink-0"
        onError={() => setErr(true)}
      />
    )
  }
  return (
    <div
      className="w-9 h-9 rounded-full flex items-center justify-center text-white font-bold text-xs shrink-0"
      style={{ backgroundColor: team.color }}
    >
      {team.abbreviation}
    </div>
  )
}

const statusColor: Record<string, string> = {
  'Out':          'text-red-400',
  'Doubtful':     'text-orange-400',
  'Questionable': 'text-amber-400',
  'Day-To-Day':   'text-amber-400',
}

const statusDot: Record<string, string> = {
  'Out':          'bg-red-500',
  'Doubtful':     'bg-orange-500',
  'Questionable': 'bg-amber-500',
  'Day-To-Day':   'bg-amber-500',
}

function InjuryList({ injuries, label }: { injuries: InjuredPlayer[]; label: string }) {
  const shown = injuries.filter(
    (p) => p.status === 'Out' || p.status === 'Doubtful' || p.status === 'Questionable' || p.status === 'Day-To-Day'
  ).slice(0, 4)

  if (shown.length === 0) return null

  return (
    <div>
      {label ? <p className="text-slate-500 text-[10px] font-medium mb-1">{label}</p> : null}
      <div className="space-y-1">
        {shown.map((p) => (
          <div key={p.id} className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${statusDot[p.status] ?? 'bg-slate-500'}`} />
            <span className="text-slate-300 text-xs flex-1 truncate">{p.name}</span>
            <span className={`text-[10px] font-medium shrink-0 ${statusColor[p.status] ?? 'text-slate-400'}`}>
              {p.status === 'Day-To-Day' ? 'Günlük' : p.status === 'Questionable' ? 'Şüpheli' : p.status === 'Doubtful' ? 'Şüpheli' : 'Yok'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PlayerList({ players, label, color }: { players: TopPlayer[]; label: string; color: string }) {
  if (players.length === 0) return null

  return (
    <div>
      <p className="text-slate-500 text-[10px] font-medium mb-1.5">{label}</p>
      <div className="space-y-1">
        {players.slice(0, 4).map((p) => (
          <div key={p.id} className="flex items-center gap-1.5">
            <div className="w-1 h-1 rounded-full shrink-0" style={{ backgroundColor: color }} />
            <span className="text-slate-300 text-xs flex-1 truncate">{p.name}</span>
            <span className="text-slate-500 text-[10px] shrink-0">
              {p.pts}p {p.reb}r {p.ast}a
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

interface WhyWinsPanelProps {
  game: Game | null
  onOpenTeam?: (teamId: string, team: Team) => void
}

export default function WhyWinsPanel({ game, onOpenTeam }: WhyWinsPanelProps) {
  if (!game) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center p-6">
        <MousePointerClick size={28} className="text-slate-700" />
        <p className="text-slate-600 text-sm">Detay için bir maç seçin</p>
      </div>
    )
  }

  const predictedHome = game.probHome >= 0.5
  const winner = predictedHome ? game.home : game.away
  const prob = predictedHome ? game.probHome : 1 - game.probHome

  const homeInjuries = game.homeInjuries ?? []
  const awayInjuries = game.awayInjuries ?? []
  const homeTopPlayers = game.homeTopPlayers ?? []
  const awayTopPlayers = game.awayTopPlayers ?? []

  const hasInjuries = homeInjuries.length > 0 || awayInjuries.length > 0
  const hasPlayers  = homeTopPlayers.length > 0 || awayTopPlayers.length > 0

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* Predicted winner header */}
      <div className="bg-slate-800/60 rounded-xl p-3">
        <p className="text-slate-500 text-xs mb-2 font-medium tracking-wide">TAHMİN EDİLEN GALİP</p>
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => onOpenTeam?.(winner.id, winner)}
            className="shrink-0 hover:opacity-80 transition-opacity"
            title={`${winner.name} detayı`}
          >
            <TeamLogoSmall team={winner} />
          </button>
          <div className="flex-1 min-w-0">
            <button
              onClick={() => onOpenTeam?.(winner.id, winner)}
              className="text-white font-bold text-sm leading-none truncate hover:text-blue-300 transition-colors text-left"
            >
              {winner.name}
            </button>
            <p className="text-slate-400 text-xs mt-0.5">
              {predictedHome ? 'Ev Sahibi' : 'Deplasman'}
            </p>
          </div>
          <div className="text-right">
            <p className="text-emerald-400 font-bold text-lg leading-none">
              {Math.round(prob * 100)}%
            </p>
            <p className="text-slate-500 text-xs">olasılık</p>
          </div>
        </div>
      </div>

      {/* Why wins section */}
      <div>
        <p className="text-slate-400 text-xs font-medium mb-2 tracking-wide">
          NEDEN {winner.abbreviation} KAZANIR?
        </p>
        <div className="space-y-2">
          {game.factors.map((factor, i) => {
            const barWidth = Math.min(Math.abs(factor.value) * 5, 100)
            return (
              <div key={i} className="bg-slate-800/40 rounded-lg p-2.5">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-slate-300 text-xs leading-tight flex-1 mr-2">
                    {factor.label}
                  </span>
                  <div className="flex items-center gap-1 shrink-0">
                    {factor.positive ? (
                      <TrendingUp size={11} className="text-emerald-400" />
                    ) : (
                      <TrendingDown size={11} className="text-red-400" />
                    )}
                    <span
                      className={`text-xs font-bold ${
                        factor.positive ? 'text-emerald-400' : 'text-red-400'
                      }`}
                    >
                      {factor.positive ? '+' : ''}{factor.value}%
                    </span>
                  </div>
                </div>
                <div className="h-1 bg-slate-700/80 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      factor.positive ? 'bg-emerald-500' : 'bg-red-500/80'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Top players */}
      {hasPlayers && (
        <div className="bg-slate-800/40 rounded-xl p-3 space-y-3">
          <p className="text-slate-400 text-xs font-medium tracking-wide flex items-center gap-1.5">
            <Users size={11} />
            KADRO (Sezon Ort.)
          </p>
          <PlayerList players={homeTopPlayers} label={`${game.home.abbreviation} (Ev)`} color={game.home.color} />
          <PlayerList players={awayTopPlayers} label={`${game.away.abbreviation} (Dep)`} color={game.away.color} />
        </div>
      )}

      {/* Injuries */}
      {hasInjuries && (
        <div className="bg-red-950/20 border border-red-500/20 rounded-xl p-3 space-y-3">
          <p className="text-red-400 text-xs font-medium tracking-wide flex items-center gap-1.5">
            <AlertTriangle size={11} />
            SAKATLИК / ŞÜPHELI
          </p>
          <InjuryList injuries={homeInjuries} label={`${game.home.abbreviation} (Ev)`} />
          <InjuryList injuries={awayInjuries} label={`${game.away.abbreviation} (Dep)`} />
        </div>
      )}

      {/* Form chart */}
      <div>
        <p className="text-slate-400 text-xs font-medium mb-2 tracking-wide">SON 10 MAÇ FORMU</p>
        <div className="bg-slate-800/40 rounded-xl p-3">
          <FormChart
            homeForm={game.homeForm}
            awayForm={game.awayForm}
            homeColor={game.home.color}
            awayColor={game.away.color}
            homeAbbr={game.home.abbreviation}
            awayAbbr={game.away.abbreviation}
          />
          <div className="flex gap-4 mt-2">
            <button
              onClick={() => onOpenTeam?.(game.home.id, game.home)}
              className="flex items-center gap-1.5 hover:opacity-75 transition-opacity"
            >
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: game.home.color }} />
              <span className="text-slate-400 text-xs">{game.home.abbreviation} (Ev)</span>
            </button>
            <button
              onClick={() => onOpenTeam?.(game.away.id, game.away)}
              className="flex items-center gap-1.5 hover:opacity-75 transition-opacity"
            >
              <div className="w-2 h-0.5 rounded-full" style={{ backgroundColor: game.away.color }} />
              <div className="w-1 h-0.5 rounded-full" style={{ backgroundColor: game.away.color }} />
              <span className="text-slate-400 text-xs">{game.away.abbreviation} (Dep)</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
