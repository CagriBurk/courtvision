import { useState, useEffect, useCallback, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import ModelMetrics from './pages/ModelMetrics'
import Settings from './pages/Settings'
import WhyWinsPanel from './components/WhyWinsPanel'
import ModelHealthPanel from './components/ModelHealthPanel'
import TeamDetailModal from './components/TeamDetailModal'
import { Game, NavPage, Team } from './types'

const API_BASE = 'http://127.0.0.1:8765'
const WATCHED_KEY = 'nba_watched_games'
const NOTIF_BEFORE_MS = 30 * 60 * 1000 // 30 dakika

interface TodayGamesResponse {
  date: string
  games: Game[]
  updating?: boolean
}

async function fetchGamesFromApi(): Promise<TodayGamesResponse> {
  const res = await fetch(`${API_BASE}/api/today-games`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

/** "8:30 PM" veya "20:30" gibi saat string'ini bugünün ms timestamp'ine çevirir */
function parseGameTimeMs(timeStr: string): number | null {
  try {
    const now = new Date()
    const base = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
    const dt = new Date(`${base} ${timeStr}`)
    if (isNaN(dt.getTime())) return null
    return dt.getTime()
  } catch {
    return null
  }
}

export default function App(): React.JSX.Element {
  const [page, setPage] = useState<NavPage>('dashboard')
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data, isLoading, isError } = useQuery({
    queryKey: ['todayGames'],
    queryFn: fetchGamesFromApi,
    retry: 1,
    // While the API generates predictions poll fast; otherwise check every few
    // minutes so a new day's slate (refreshed by the API scheduler) shows up.
    refetchInterval: (query) => (query.state.data?.updating ? 3000 : 5 * 60 * 1000)
  })

  // "Veriyi Güncelle": API re-runs 05_predict_today.py in the background
  const refresh = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/api/refresh-predictions`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['todayGames'] })
  })

  const games = data?.games || []
  const updating = !!data?.updating || refresh.isPending
  const preparing = updating && games.length === 0

  // When an update finishes, results of earlier days may have been filled in
  const wasUpdating = useRef(false)
  useEffect(() => {
    if (wasUpdating.current && !updating) {
      queryClient.invalidateQueries({ queryKey: ['history'] })
    }
    wasUpdating.current = updating
  }, [updating, queryClient])
  const apiConnected = !isError && !isLoading
  const apiError = isError
    ? 'FastAPI sunucusuna bağlanılamadı. Sunucuyu başlatın:\npython -m uvicorn src.api_server:app --host 127.0.0.1 --port 8765'
    : null

  // Selection is kept by id so refreshed data replaces the shown game
  const selectedGame = games.find((g) => g.id === selectedGameId) ?? games[0] ?? null

  // Takip edilen maçlar (bildirim için)
  const [watchedGames, setWatchedGames] = useState<Set<string>>(() => {
    try {
      const stored = localStorage.getItem(WATCHED_KEY)
      return stored ? new Set(JSON.parse(stored)) : new Set()
    } catch {
      return new Set()
    }
  })
  const notifTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  // Takım detay modali
  const [teamDetailId, setTeamDetailId] = useState<string | null>(null)
  const [teamDetailData, setTeamDetailData] = useState<Team | null>(null)

  // Bildirimleri izlenen maçlara göre zamanla
  const scheduleNotif = useCallback((game: Game) => {
    const gameMs = parseGameTimeMs(game.gameTime)
    if (!gameMs) return
    const delay = gameMs - Date.now() - NOTIF_BEFORE_MS
    if (delay <= 0) return

    const id = setTimeout(() => {
      if (Notification.permission === 'granted') {
        new Notification(`${game.home.abbreviation} vs ${game.away.abbreviation} — 30 dk kaldı`, {
          body: `Tahmin: ${game.probHome >= 0.5 ? game.home.name : game.away.name} kazanır (${Math.round(Math.max(game.probHome, 1 - game.probHome) * 100)}%)`,
          icon: game.home.logoUrl
        })
      }
      notifTimers.current.delete(game.id)
    }, delay)

    notifTimers.current.set(game.id, id)
  }, [])

  // games değişince izlenen maçları yeniden zamanla
  useEffect(() => {
    // Eski timer'ları temizle
    notifTimers.current.forEach((t) => clearTimeout(t))
    notifTimers.current.clear()

    for (const game of games) {
      if (watchedGames.has(game.id)) {
        scheduleNotif(game)
      }
    }
  }, [games, watchedGames, scheduleNotif])

  const toggleWatch = useCallback(
    (gameId: string) => {
      // Bildirim izni iste
      if (Notification.permission === 'default') {
        Notification.requestPermission()
      }

      setWatchedGames((prev) => {
        const next = new Set(prev)
        if (next.has(gameId)) {
          next.delete(gameId)
          const t = notifTimers.current.get(gameId)
          if (t) {
            clearTimeout(t)
            notifTimers.current.delete(gameId)
          }
        } else {
          next.add(gameId)
          const game = games.find((g) => g.id === gameId)
          if (game) scheduleNotif(game)
        }
        localStorage.setItem(WATCHED_KEY, JSON.stringify([...next]))
        return next
      })
    },
    [games, scheduleNotif]
  )

  const openTeamDetail = useCallback((teamId: string, team: Team) => {
    setTeamDetailId(teamId)
    setTeamDetailData(team)
  }, [])

  const showRightPanel = page === 'dashboard'

  return (
    <div className="flex h-screen bg-slate-950 text-white overflow-hidden">
      <Sidebar activePage={page} onNavigate={setPage} />

      <main className="flex-1 overflow-hidden">
        {page === 'dashboard' && (
          <Dashboard
            games={games}
            selectedGame={selectedGame}
            onSelectGame={(game) => setSelectedGameId(game.id)}
            slateDate={data?.date}
            isLoading={isLoading || preparing}
            isUpdating={updating}
            apiConnected={apiConnected}
            apiError={apiError}
            onRefresh={() => refresh.mutate()}
            watchedGames={watchedGames}
            onToggleWatch={toggleWatch}
          />
        )}
        {page === 'history' && <History apiBase={API_BASE} />}
        {page === 'metrics' && <ModelMetrics apiBase={API_BASE} />}
        {page === 'settings' && <Settings />}
      </main>

      {showRightPanel && (
        <aside className="w-72 bg-slate-900 border-l border-slate-800 flex flex-col overflow-hidden shrink-0">
          <div className="px-4 py-3 border-b border-slate-800 shrink-0">
            <p className="text-slate-400 text-xs font-medium tracking-wide">MAÇ DETAYI</p>
          </div>
          <WhyWinsPanel game={selectedGame} onOpenTeam={openTeamDetail} />
          <ModelHealthPanel apiBase={API_BASE} />
        </aside>
      )}

      {teamDetailId && teamDetailData && (
        <TeamDetailModal
          apiBase={API_BASE}
          teamId={teamDetailId}
          team={teamDetailData}
          onClose={() => {
            setTeamDetailId(null)
            setTeamDetailData(null)
          }}
        />
      )}
    </div>
  )
}
