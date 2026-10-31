import { Activity, Target, Clock, TrendingUp } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { formatDateTr, useModelMetrics } from '../hooks/useModelMetrics'

interface ModelHealthPanelProps {
  apiBase?: string
}

export default function ModelHealthPanel({
  apiBase = 'http://127.0.0.1:8765'
}: ModelHealthPanelProps) {
  const { data: metrics } = useModelMetrics(apiBase)
  const { data } = useQuery({
    queryKey: ['history', apiBase],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/api/history`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    retry: 1
  })

  let recent = { correct: 0, total: 0 }
  if (data?.stats) {
    const s = data.stats
    const tier = s.byTier || {}
    const totalCompleted = s.completedGames ?? 0
    const totalCorrect =
      (tier.HIGH?.correct ?? 0) + (tier.MEDIUM?.correct ?? 0) + (tier.LOW?.correct ?? 0)
    recent = { correct: totalCorrect, total: totalCompleted }
  }

  const hasRecent = recent.total > 0
  const recentAcc = hasRecent ? Math.round((recent.correct / recent.total) * 100) : null

  return (
    <div className="border-t border-slate-800 p-4 shrink-0">
      <p className="text-slate-400 text-xs font-medium mb-3 flex items-center gap-1.5 tracking-wide">
        <Activity size={11} />
        MODEL SAĞLIĞI
      </p>

      <div className="grid grid-cols-2 gap-2">
        {/* OOF Accuracy */}
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 mb-0.5">
            <Target size={10} className="text-slate-500" />
            <p className="text-slate-500 text-xs">OOF Accuracy</p>
          </div>
          <p className="text-emerald-400 font-bold text-lg leading-none">{metrics?.accuracy != null ? `${metrics.accuracy}%` : '—'}</p>
        </div>

        {/* Brier Score */}
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 mb-0.5">
            <TrendingUp size={10} className="text-slate-500" />
            <p className="text-slate-500 text-xs">Brier Score</p>
          </div>
          <p className="text-blue-400 font-bold text-lg leading-none">{metrics?.brierScore != null ? metrics.brierScore.toFixed(4) : '—'}</p>
        </div>

        {/* Recent accuracy */}
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <p className="text-slate-500 text-xs mb-0.5">Son Tahminler</p>
          {hasRecent ? (
            <>
              <p className="text-white font-bold text-lg leading-none">{recentAcc}%</p>
              <p className="text-slate-600 text-xs">
                {recent.correct}/{recent.total} doğru
              </p>
            </>
          ) : (
            <p className="text-slate-600 text-xs mt-1">Veri bekleniyor...</p>
          )}
        </div>

        {/* Last tuning */}
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 mb-0.5">
            <Clock size={10} className="text-slate-500" />
            <p className="text-slate-500 text-xs">Son Tuning</p>
          </div>
          <p className="text-white text-xs font-medium mt-1">{formatDateTr(metrics?.trainedAt)}</p>
          <p className="text-slate-600 text-xs">{metrics?.nTrials ?? '—'} trial</p>
        </div>
      </div>
    </div>
  )
}
