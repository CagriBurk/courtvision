import { useEffect, useState } from 'react'
import { X } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { Team } from '../types'

interface GameRecord {
  game_date: string
  won: number
  pts_scored: number
  pts_allowed: number
  plus_minus: number
  fg_pct: number | null
  fg3_pct: number | null
  ast: number | null
  reb: number | null
  tov: number | null
  is_home: number
}

interface TeamStats {
  summary: {
    win_rate_L5: number
    win_rate_L10: number
    pts_avg_L10: number
    pts_allowed_L10: number
    point_diff_L10: number
    last_game_date: string
    n_games: number
  }
  recent_games: GameRecord[]
}

interface TeamDetailModalProps {
  apiBase: string
  teamId: string
  team: Team
  onClose: () => void
}

function avg(arr: (number | null)[], decimals = 1): string {
  const clean = arr.filter((v): v is number => v != null)
  if (clean.length === 0) return '—'
  return (clean.reduce((a, b) => a + b, 0) / clean.length).toFixed(decimals)
}

export default function TeamDetailModal({ apiBase, teamId, team, onClose }: TeamDetailModalProps) {
  const [data, setData] = useState<TeamStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetch(`${apiBase}/api/team/${teamId}/stats?n_games=15`)
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [apiBase, teamId])

  const games = data?.recent_games ?? []
  const chartData = games.map((g, i) => ({
    idx: i + 1,
    pts: g.pts_scored,
    opp: g.pts_allowed,
    pm: g.plus_minus,
    label: new Date(g.game_date).toLocaleDateString('tr-TR', { month: 'short', day: 'numeric' }),
  }))

  return (
    <div
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-slate-700/60 rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-hidden flex flex-col shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-800 shrink-0">
          <img
            src={team.logoUrl}
            alt={team.abbreviation}
            className="w-10 h-10 object-contain"
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
          />
          <div className="flex-1">
            <p className="text-white font-bold text-base leading-none">{team.name}</p>
            {data && (
              <p className="text-slate-400 text-xs mt-0.5">
                Son {data.summary.n_games} maç · L5:{' '}
                <span className="text-white font-semibold">
                  {Math.round(data.summary.win_rate_L5 * 100)}%
                </span>{' '}
                · L10:{' '}
                <span className="text-white font-semibold">
                  {Math.round(data.summary.win_rate_L10 * 100)}%
                </span>
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-white transition-colors p-1"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {loading && (
            <div className="flex items-center justify-center h-40 text-slate-500 text-sm">
              Yükleniyor...
            </div>
          )}

          {!loading && data && (
            <>
              {/* Pts / Opp Line Chart */}
              <div>
                <p className="text-slate-400 text-xs font-medium mb-2 tracking-wide">
                  SON {games.length} MAÇ — PTS / OPP
                </p>
                <div className="bg-slate-800/40 rounded-xl p-3">
                  <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                      <XAxis
                        dataKey="label"
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        tickLine={false}
                        axisLine={false}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                        labelStyle={{ color: '#94a3b8' }}
                        formatter={(val, name) => [
                          val ?? 0,
                          (name as string) === 'pts' ? 'Pts' : 'Opp',
                        ]}
                      />
                      <ReferenceLine y={110} stroke="#334155" strokeDasharray="3 3" />
                      <Line
                        type="monotone"
                        dataKey="pts"
                        stroke={team.color}
                        strokeWidth={2}
                        dot={{ r: 3, fill: team.color }}
                        activeDot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="opp"
                        stroke="#64748b"
                        strokeWidth={1.5}
                        strokeDasharray="4 2"
                        dot={{ r: 2, fill: '#64748b' }}
                        activeDot={{ r: 3 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="flex gap-4 mt-1">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-0.5 rounded-full" style={{ backgroundColor: team.color }} />
                      <span className="text-slate-400 text-[10px]">Pts</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-0.5 rounded-full bg-slate-500" />
                      <span className="text-slate-400 text-[10px]">Opp</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sezon ortalamaları */}
              <div className="grid grid-cols-5 gap-2">
                {[
                  { label: 'Pts', val: data.summary.pts_avg_L10.toFixed(1) },
                  { label: 'Opp', val: data.summary.pts_allowed_L10.toFixed(1) },
                  { label: 'PM', val: (data.summary.point_diff_L10 >= 0 ? '+' : '') + data.summary.point_diff_L10.toFixed(1) },
                  { label: 'FG%', val: avg(games.map((g) => g.fg_pct != null ? g.fg_pct * 100 : null), 1) + '%' },
                  { label: '3P%', val: avg(games.map((g) => g.fg3_pct != null ? g.fg3_pct * 100 : null), 1) + '%' },
                ].map(({ label, val }) => (
                  <div key={label} className="bg-slate-800/50 rounded-lg p-2.5 text-center">
                    <p className="text-slate-500 text-[10px] mb-1">{label}</p>
                    <p className="text-white font-bold text-sm">{val}</p>
                    <p className="text-slate-600 text-[9px]">L10 ort</p>
                  </div>
                ))}
              </div>

              {/* Son 15 maç listesi */}
              <div>
                <p className="text-slate-400 text-xs font-medium mb-2 tracking-wide">MAÇ LOG</p>
                <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 overflow-hidden">
                  {[...games].reverse().map((g, i) => {
                    const dateStr = new Date(g.game_date).toLocaleDateString('tr-TR', {
                      month: 'short', day: 'numeric',
                    })
                    return (
                      <div
                        key={i}
                        className={`flex items-center gap-3 px-4 py-2 text-xs ${
                          i < games.length - 1 ? 'border-b border-slate-700/30' : ''
                        }`}
                      >
                        <span className="text-slate-500 w-14 shrink-0">{dateStr}</span>
                        <span
                          className={`w-5 font-bold shrink-0 ${g.won ? 'text-emerald-400' : 'text-red-400'}`}
                        >
                          {g.won ? 'G' : 'M'}
                        </span>
                        <span className="text-slate-500 shrink-0 text-[10px]">
                          {g.is_home ? 'Ev' : 'Dep'}
                        </span>
                        <span className="text-white font-semibold flex-1">
                          {g.pts_scored} – {g.pts_allowed}
                        </span>
                        <span
                          className={`font-semibold w-10 text-right shrink-0 ${
                            g.plus_minus >= 0 ? 'text-emerald-400' : 'text-red-400'
                          }`}
                        >
                          {g.plus_minus >= 0 ? '+' : ''}{g.plus_minus}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          )}

          {!loading && !data && (
            <div className="flex items-center justify-center h-40 text-slate-600 text-sm">
              Takım verisi bulunamadı
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
