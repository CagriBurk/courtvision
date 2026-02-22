import { useEffect, useState } from 'react'
import { CheckCircle, XCircle, Minus } from 'lucide-react'
import { PredictionRecord } from '../types'

interface HistoryProps {
  apiBase?: string
}

const confidenceLabel: Record<string, string> = {
  HIGH: 'Yüksek',
  MEDIUM: 'Orta',
  LOW: 'Düşük',
}

const confidenceStyle: Record<string, string> = {
  HIGH: 'text-emerald-400',
  MEDIUM: 'text-amber-400',
  LOW: 'text-slate-400',
}

function computeStats(records: PredictionRecord[]) {
  const weights = { HIGH: 3, MEDIUM: 2, LOW: 1 }
  let totalWeight = 0
  let correctWeight = 0
  const byTier = {
    HIGH: { correct: 0, total: 0 },
    MEDIUM: { correct: 0, total: 0 },
    LOW: { correct: 0, total: 0 },
  }

  for (const r of records) {
    if (r.correct === null) continue
    const w = weights[r.confidence] ?? 1
    totalWeight += w
    if (r.correct) correctWeight += w
    byTier[r.confidence].total++
    if (r.correct) byTier[r.confidence].correct++
  }

  return {
    weightedAcc: totalWeight > 0 ? Math.round((correctWeight / totalWeight) * 100) : null,
    total: records.filter((r) => r.correct !== null).length,
    correct: records.filter((r) => r.correct === true).length,
    byTier,
  }
}

function groupByDate(records: PredictionRecord[]) {
  const groups: Record<string, PredictionRecord[]> = {}
  for (const r of records) {
    if (!groups[r.date]) groups[r.date] = []
    groups[r.date].push(r)
  }
  return Object.entries(groups).sort(([a], [b]) => b.localeCompare(a))
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('tr-TR', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  })
}

export default function History({ apiBase = 'http://127.0.0.1:8765' }: HistoryProps) {
  const [predictions, setPredictions] = useState<PredictionRecord[]>([])
  const [apiError, setApiError] = useState(false)

  useEffect(() => {
    setApiError(false)
    fetch(`${apiBase}/api/history`)
      .then((r) => r.json())
      .then((data) => {
        if (data.predictions) {
          setPredictions(data.predictions)
        }
      })
      .catch(() => {
        setApiError(true)
      })
  }, [apiBase])

  const stats = computeStats(predictions)
  const grouped = groupByDate(predictions)

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4 shrink-0">
        <h1 className="text-white font-bold text-xl">Geçmiş Tahminler</h1>
        <p className="text-slate-500 text-sm mt-0.5">Prediction tracking & doğruluk analizi</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Summary stats */}
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
            <p className="text-slate-500 text-xs mb-1">Ağırlıklı Accuracy</p>
            <p className="text-emerald-400 font-bold text-2xl">
              {stats.weightedAcc !== null ? `${stats.weightedAcc}%` : '—'}
            </p>
            <p className="text-slate-600 text-xs mt-0.5">HIGH×3, MED×2, LOW×1</p>
          </div>
          <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
            <p className="text-slate-500 text-xs mb-1">Toplam Doğru</p>
            <p className="text-white font-bold text-2xl">
              {stats.correct}/{stats.total}
            </p>
            <p className="text-slate-600 text-xs mt-0.5">
              {stats.total > 0 ? `${Math.round((stats.correct / stats.total) * 100)}% ham accuracy` : '—'}
            </p>
          </div>
          {(['HIGH', 'MEDIUM', 'LOW'] as const).map((tier) => {
            const t = stats.byTier[tier]
            const acc = t.total > 0 ? Math.round((t.correct / t.total) * 100) : null
            return (
              <div key={tier} className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
                <p className={`text-xs mb-1 font-medium ${confidenceStyle[tier]}`}>
                  {confidenceLabel[tier]} Güven
                </p>
                <p className="text-white font-bold text-2xl">
                  {acc !== null ? `${acc}%` : '—'}
                </p>
                <p className="text-slate-600 text-xs mt-0.5">{t.correct}/{t.total} doğru</p>
              </div>
            )
          })}
        </div>

        {/* Prediction list by date */}
        <div className="space-y-5">
          {grouped.map(([date, records]) => {
            const dayCorrect = records.filter((r) => r.correct === true).length
            return (
              <div key={date}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-slate-400 text-sm font-medium capitalize">{formatDate(date)}</p>
                  <span className="text-slate-500 text-xs">
                    {dayCorrect}/{records.length} doğru
                  </span>
                </div>
                <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 overflow-hidden">
                  {records.map((r, i) => (
                    <div
                      key={r.gameId}
                      className={`flex items-center gap-4 px-4 py-3 ${
                        i < records.length - 1 ? 'border-b border-slate-700/30' : ''
                      }`}
                    >
                      {/* Result icon */}
                      <div className="shrink-0">
                        {r.correct === true && <CheckCircle size={16} className="text-emerald-500" />}
                        {r.correct === false && <XCircle size={16} className="text-red-500" />}
                        {r.correct === null && <Minus size={16} className="text-slate-600" />}
                      </div>

                      {/* Matchup */}
                      <div className="flex-1 flex items-center gap-2">
                        <span className="text-white font-semibold text-sm">{r.homeTeam}</span>
                        <span className="text-slate-600 text-xs">vs</span>
                        <span className="text-slate-300 text-sm">{r.awayTeam}</span>
                      </div>

                      {/* Prediction */}
                      <div className="text-right">
                        <p className="text-slate-400 text-xs">Tahmin</p>
                        <p className={`text-sm font-semibold ${r.correct === null ? 'text-slate-400' : r.correct ? 'text-emerald-400' : 'text-red-400'}`}>
                          {r.predictedWinner}
                        </p>
                      </div>

                      {/* Actual */}
                      <div className="text-right w-16">
                        <p className="text-slate-400 text-xs">Gerçek</p>
                        <p className="text-white text-sm font-semibold">
                          {r.actualWinner ?? '—'}
                        </p>
                      </div>

                      {/* Probability */}
                      <div className="text-right w-12">
                        <p className="text-slate-400 text-xs">Olasılık</p>
                        <p className="text-slate-300 text-sm font-semibold">
                          {Math.round(Math.max(r.probHome, 1 - r.probHome) * 100)}%
                        </p>
                      </div>

                      {/* Confidence */}
                      <div className="w-16 text-right">
                        <span className={`text-xs font-medium ${confidenceStyle[r.confidence]}`}>
                          {confidenceLabel[r.confidence]}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}

          {grouped.length === 0 && (
            <div className="flex flex-col items-center justify-center h-40 gap-2 text-center">
              {apiError ? (
                <>
                  <p className="text-slate-500 text-sm">API bağlantısı yok</p>
                  <p className="text-slate-600 text-xs">Sunucuyu başlatın ve sayfayı yenileyin</p>
                </>
              ) : (
                <p className="text-slate-600 text-sm">Henüz tahmin kaydı yok</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
