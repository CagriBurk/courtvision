import { Clock, MapPin, ChevronRight, AlertTriangle, Bell, BellOff } from 'lucide-react'
import { useState } from 'react'
import { Game, InjuredPlayer, Team, TopPlayer } from '../types'
import WinProbBar from './WinProbBar'

interface PredictionCardProps {
  game: Game
  isSelected: boolean
  onSelect: () => void
  isWatched?: boolean
  onToggleWatch?: () => void
}

const confidenceStyle: Record<string, string> = {
  HIGH: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  MEDIUM: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  LOW: 'bg-slate-600/20 text-slate-400 border-slate-600/40',
}

const confidenceLabel: Record<string, string> = {
  HIGH: 'Yüksek Güven',
  MEDIUM: 'Orta Güven',
  LOW: 'Düşük Güven',
}

// Takımın en kritik Out oyuncusunu bul — top player ise öncelikli
function getCriticalOut(
  injuries: InjuredPlayer[],
  topPlayers: TopPlayer[],
): { name: string; extra: number } | null {
  const outList = injuries.filter((p) => p.status === 'Out' || p.status === 'Doubtful')
  if (outList.length === 0) return null
  const topNames = new Set(topPlayers.map((p) => p.name))
  const star = outList.find((p) => topNames.has(p.name)) ?? outList[0]
  return { name: star.name.split(' ').slice(-1)[0], extra: outList.length - 1 }
}

function FormDots({ form }: { form: number[] }) {
  return (
    <div className="flex gap-0.5 justify-center">
      {form.map((result, i) => (
        <div
          key={i}
          className={`w-1.5 h-1.5 rounded-full ${result ? 'bg-emerald-500' : 'bg-red-500/70'}`}
        />
      ))}
    </div>
  )
}

function TeamLogo({ team }: { team: Team }) {
  const [imgError, setImgError] = useState(false)

  if (!imgError) {
    return (
      <img
        src={team.logoUrl}
        alt={team.abbreviation}
        className="w-12 h-12 object-contain"
        onError={() => setImgError(true)}
      />
    )
  }

  return (
    <div
      className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-sm shadow-lg"
      style={{ backgroundColor: team.color }}
    >
      {team.abbreviation}
    </div>
  )
}

export default function PredictionCard({
  game,
  isSelected,
  onSelect,
  isWatched = false,
  onToggleWatch,
}: PredictionCardProps) {
  const predictedHome = game.probHome >= 0.5
  const winner = predictedHome ? game.home : game.away

  const homeOut = getCriticalOut(game.homeInjuries ?? [], game.homeTopPlayers ?? [])
  const awayOut = getCriticalOut(game.awayInjuries ?? [], game.awayTopPlayers ?? [])
  const hasInjuryAlert = homeOut !== null || awayOut !== null

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-4 rounded-xl border transition-all duration-200 ${
        isSelected
          ? 'bg-slate-700/60 border-blue-500/50 shadow-lg shadow-blue-500/10 ring-1 ring-blue-500/20'
          : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-700/30 hover:border-slate-600/70'
      }`}
    >
      {/* Confidence badge */}
      <div className="flex items-center justify-between mb-3">
        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${confidenceStyle[game.confidence]}`}>
          {confidenceLabel[game.confidence]}
        </span>
        <ChevronRight
          size={15}
          className={`transition-colors ${isSelected ? 'text-blue-400' : 'text-slate-600'}`}
        />
      </div>

      {/* Teams */}
      <div className="flex items-center gap-2 mb-3">
        {/* Home */}
        <div className="flex-1 flex flex-col items-center gap-1.5">
          <TeamLogo team={game.home} />
          <p className="text-white font-semibold text-sm leading-none text-center">
            {game.home.name.split(' ').slice(-1)[0]}
          </p>
          <p className="text-slate-500 text-xs">Ev</p>
          <FormDots form={game.homeForm.slice(-5)} />
        </div>

        <span className="text-slate-600 font-bold text-base px-1">VS</span>

        {/* Away */}
        <div className="flex-1 flex flex-col items-center gap-1.5">
          <TeamLogo team={game.away} />
          <p className="text-white font-semibold text-sm leading-none text-center">
            {game.away.name.split(' ').slice(-1)[0]}
          </p>
          <p className="text-slate-500 text-xs">Dep</p>
          <FormDots form={game.awayForm.slice(-5)} />
        </div>
      </div>

      {/* Win probability bar */}
      <WinProbBar
        probHome={game.probHome}
        homeColor={game.home.color}
        awayColor={game.away.color}
      />
      <div className="flex justify-between text-xs mt-1 font-semibold">
        <span className="text-slate-300">{Math.round(game.probHome * 100)}%</span>
        <span className="text-slate-300">{Math.round((1 - game.probHome) * 100)}%</span>
      </div>

      {/* Footer */}
      <div className="flex items-center gap-3 mt-3 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <Clock size={10} />
          {game.gameTime}
        </span>
        {game.venue && (
          <span className="flex items-center gap-1 min-w-0">
            <MapPin size={10} className="shrink-0" />
            <span className="truncate">{game.venue.split(',')[0]}</span>
          </span>
        )}
      </div>

      {/* Injury alert row — sadece Out/Doubtful varsa görünür */}
      {hasInjuryAlert && (
        <div className="flex items-center justify-between px-0.5 mt-2 gap-2">
          <div className="flex-1 flex justify-start">
            {homeOut ? (
              <span className="flex items-center gap-1 text-[10px] text-red-400 font-medium">
                <AlertTriangle size={9} />
                {homeOut.name} oynamayacak{homeOut.extra > 0 ? ` +${homeOut.extra}` : ''}
              </span>
            ) : (
              <span />
            )}
          </div>
          <div className="flex-1 flex justify-end">
            {awayOut ? (
              <span className="flex items-center gap-1 text-[10px] text-red-400 font-medium">
                {awayOut.name} oynamayacak{awayOut.extra > 0 ? ` +${awayOut.extra}` : ''}
                <AlertTriangle size={9} />
              </span>
            ) : (
              <span />
            )}
          </div>
        </div>
      )}

      {/* Predicted winner + watch button */}
      <div className="mt-2 flex items-center justify-between pt-2.5 border-t border-slate-700/50">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
          <span className="text-xs text-blue-400 font-medium">
            Tahmin: {winner.abbreviation} kazanır
          </span>
        </div>
        {onToggleWatch && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onToggleWatch()
            }}
            className={`p-1 rounded-md transition-colors ${
              isWatched
                ? 'text-amber-400 hover:text-amber-300'
                : 'text-slate-600 hover:text-slate-400'
            }`}
            title={isWatched ? 'Bildirimi kapat' : '30 dk önce bildir'}
          >
            {isWatched ? <Bell size={13} fill="currentColor" /> : <BellOff size={13} />}
          </button>
        )}
      </div>
    </button>
  )
}
