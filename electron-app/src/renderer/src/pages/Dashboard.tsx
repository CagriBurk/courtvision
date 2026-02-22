import { RefreshCw, Zap, CheckCircle, Wifi, WifiOff, Terminal } from 'lucide-react'
import { useState } from 'react'
import { Game } from '../types'
import PredictionCard from '../components/PredictionCard'

type ConfFilter = 'ALL' | 'HIGH' | 'MEDIUM' | 'LOW'

interface DashboardProps {
  games: Game[]
  selectedGame: Game | null
  onSelectGame: (game: Game) => void
  isLoading?: boolean
  apiConnected?: boolean
  apiError?: string | null
  onRefresh?: () => void
  watchedGames?: Set<string>
  onToggleWatch?: (gameId: string) => void
}

const today = new Date().toLocaleDateString('tr-TR', {
  weekday: 'long',
  day: 'numeric',
  month: 'long',
  year: 'numeric',
})

const filterBtnClass = (active: boolean) =>
  `px-3 py-1 rounded-full text-xs font-semibold transition-colors ${
    active
      ? 'bg-blue-600 text-white'
      : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-300'
  }`

export default function Dashboard({
  games,
  selectedGame,
  onSelectGame,
  isLoading = false,
  apiConnected = false,
  apiError = null,
  onRefresh,
  watchedGames = new Set(),
  onToggleWatch,
}: DashboardProps) {
  const [filterConf, setFilterConf] = useState<ConfFilter>('ALL')
  const [filterStrong, setFilterStrong] = useState(false)

  const filteredGames = games.filter((g) => {
    if (filterConf !== 'ALL' && g.confidence !== filterConf) return false
    if (filterStrong && Math.max(g.probHome, 1 - g.probHome) <= 0.6) return false
    return true
  })

  const highConfidence = games.filter((g) => g.confidence === 'HIGH').length

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-slate-950/80 backdrop-blur-sm border-b border-slate-800 px-6 py-4 shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-white font-bold text-xl">Günlük Tahminler</h1>
              <div className="flex items-center gap-1.5 bg-red-500/15 border border-red-500/30 rounded-full px-2.5 py-0.5">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                <span className="text-red-400 text-xs font-semibold">CANLI</span>
              </div>
            </div>
            <p className="text-slate-500 text-sm mt-0.5 capitalize">{today}</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              {apiConnected ? (
                <Wifi size={13} className="text-emerald-400" />
              ) : (
                <WifiOff size={13} className="text-slate-600" />
              )}
              <span className={`text-xs ${apiConnected ? 'text-emerald-400' : 'text-slate-600'}`}>
                {apiConnected ? 'API Bağlı' : 'Bağlantı Yok'}
              </span>
            </div>

            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white text-sm font-semibold px-4 py-2 rounded-lg transition-colors"
            >
              <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
              {isLoading ? 'Yükleniyor...' : 'Veriyi Güncelle'}
            </button>
          </div>
        </div>
      </div>

      {/* Stats + filter bar */}
      {apiConnected && (
        <div className="px-6 py-2.5 flex items-center justify-between border-b border-slate-800/60 bg-slate-900/30 shrink-0">
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-1.5">
              <Zap size={13} className="text-amber-400" />
              <span className="text-slate-400 text-sm">
                <span className="text-white font-semibold">{filteredGames.length}</span>
                {filteredGames.length !== games.length && (
                  <span className="text-slate-600"> / {games.length}</span>
                )}{' '}
                maç
              </span>
            </div>
            <div className="w-px h-4 bg-slate-700" />
            <div className="flex items-center gap-1.5">
              <CheckCircle size={13} className="text-emerald-400" />
              <span className="text-slate-400 text-sm">
                <span className="text-white font-semibold">{highConfidence}</span> yüksek güven
              </span>
            </div>
          </div>

          {/* Filter buttons */}
          <div className="flex items-center gap-1.5">
            {(['ALL', 'HIGH', 'MEDIUM', 'LOW'] as ConfFilter[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilterConf(f)}
                className={filterBtnClass(filterConf === f)}
              >
                {f === 'ALL' ? 'Tümü' : f === 'HIGH' ? 'Yüksek' : f === 'MEDIUM' ? 'Orta' : 'Düşük'}
              </button>
            ))}
            <div className="w-px h-4 bg-slate-700 mx-1" />
            <button
              onClick={() => setFilterStrong(!filterStrong)}
              className={filterBtnClass(filterStrong)}
            >
              60%+
            </button>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5">
        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center h-64 text-slate-500 text-sm gap-2">
            <RefreshCw size={16} className="animate-spin" />
            <span>Tahminler yükleniyor...</span>
          </div>
        )}

        {/* API error */}
        {!isLoading && apiError && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <div className="bg-slate-800/60 border border-red-500/30 rounded-2xl p-6 max-w-md w-full">
              <div className="flex items-center gap-2 mb-3">
                <WifiOff size={18} className="text-red-400" />
                <p className="text-red-400 font-semibold text-sm">API Sunucusuna Bağlanılamadı</p>
              </div>
              <p className="text-slate-400 text-xs leading-relaxed mb-4">
                FastAPI sunucusunu başlatmadan tahminler görüntülenemez.
              </p>
              <div className="bg-slate-900 rounded-lg p-3 flex items-start gap-2">
                <Terminal size={13} className="text-slate-500 mt-0.5 shrink-0" />
                <code className="text-emerald-400 text-xs break-all">
                  python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8765
                </code>
              </div>
              <p className="text-slate-600 text-xs mt-3">
                Sunucu başladıktan sonra "Veriyi Güncelle" butonuna tıklayın.
              </p>
            </div>
          </div>
        )}

        {/* Games grid */}
        {!isLoading && !apiError && (
          <>
            {filteredGames.length > 0 ? (
              <div className="grid grid-cols-2 gap-3">
                {filteredGames.map((game) => (
                  <PredictionCard
                    key={game.id}
                    game={game}
                    isSelected={selectedGame?.id === game.id}
                    onSelect={() => onSelectGame(game)}
                    isWatched={watchedGames.has(game.id)}
                    onToggleWatch={onToggleWatch ? () => onToggleWatch(game.id) : undefined}
                  />
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-slate-600 text-sm">
                {games.length > 0 ? 'Filtreye uyan maç yok' : 'Bugün maç bulunamadı'}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
