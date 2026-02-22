import { useEffect, useState } from 'react'
import { Activity, Target, Clock, TrendingUp } from 'lucide-react'

interface ModelHealthPanelProps {
  apiBase?: string
}

// Gerçek OOF değerleri — 04_evaluate_model.py, 2026-02-22
const OOF_ACCURACY = 66.7
const BRIER_SCORE  = 0.2105
const LAST_TUNING  = '22 Şub 2026'

export default function ModelHealthPanel({ apiBase = 'http://127.0.0.1:8765' }: ModelHealthPanelProps) {
  const [recent, setRecent] = useState<{ correct: number; total: number }>({ correct: 0, total: 0 })

  useEffect(() => {
    fetch(`${apiBase}/api/history`)
      .then((r) => r.json())
      .then((data) => {
        if (data.stats) {
          const s = data.stats
          const tier = s.byTier || {}
          const totalCompleted = s.completedGames ?? 0
          const totalCorrect =
            (tier.HIGH?.correct ?? 0) +
            (tier.MEDIUM?.correct ?? 0) +
            (tier.LOW?.correct ?? 0)
          setRecent({ correct: totalCorrect, total: totalCompleted })
        }
      })
      .catch(() => {
        // API erişilemez — varsayılan 0/0 kalır
      })
  }, [apiBase])

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
          <p className="text-emerald-400 font-bold text-lg leading-none">{OOF_ACCURACY}%</p>
        </div>

        {/* Brier Score */}
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 mb-0.5">
            <TrendingUp size={10} className="text-slate-500" />
            <p className="text-slate-500 text-xs">Brier Score</p>
          </div>
          <p className="text-blue-400 font-bold text-lg leading-none">{BRIER_SCORE}</p>
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
          <p className="text-white text-xs font-medium mt-1">{LAST_TUNING}</p>
          <p className="text-slate-600 text-xs">150 trial</p>
        </div>
      </div>
    </div>
  )
}
