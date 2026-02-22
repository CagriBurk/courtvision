import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from 'recharts'
import { useEffect, useState } from 'react'
import { mockFeatureImportance, mockFoldMetrics } from '../data/historyData'

interface FeatureItem { feature: string; importance: number }
interface FoldItem    { fold: number; accuracy: number; auc: number; logLoss: number }

interface ModelMetricsProps {
  apiBase?: string
}

// Gerçek değerler — 04_evaluate_model.py, 236 feature, 2026-02-22
const DEFAULT_METRICS = [
  { label: 'Accuracy (OOF)', value: '66.7%', sub: 'Baseline: 57.9% (+8.8pp)', color: 'text-emerald-400' },
  { label: 'AUC-ROC',        value: '0.715', sub: 'İyi ayrım gücü',           color: 'text-blue-400'    },
  { label: 'Log-Loss',       value: '0.608', sub: 'Baseline: 0.679',           color: 'text-amber-400'   },
  { label: 'Brier Score',    value: '0.210', sub: 'Kalibrasyon kalitesi',      color: 'text-purple-400'  },
]

const IMPORTANCE_COLORS = [
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#c084fc',
  '#e879f9', '#f472b6', '#fb7185', '#f87171', '#fca5a5',
]

export default function ModelMetrics({ apiBase = 'http://127.0.0.1:8765' }: ModelMetricsProps) {
  const [features, setFeatures] = useState<FeatureItem[]>(mockFeatureImportance)
  const [folds,    setFolds]    = useState<FoldItem[]>(mockFoldMetrics)

  useEffect(() => {
    fetch(`${apiBase}/api/model/metrics`)
      .then((r) => r.json())
      .then((data) => {
        if (data.featureImportance && data.featureImportance.length > 0) {
          setFeatures(data.featureImportance)
        }
        if (data.foldMetrics && data.foldMetrics.length > 0) {
          setFolds(data.foldMetrics)
        }
      })
      .catch(() => {
        // API erişilemez — mock data kalır
      })
  }, [apiBase])

  const sortedFeatures = [...features]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15)

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-white font-bold text-xl">Model Metrikleri</h1>
        <p className="text-slate-500 text-sm mt-0.5">
          XGBoost · 236 feature · 26,332 maç · 5-Fold TimeSeriesSplit · En önemli:{' '}
          <span className="text-blue-400">elo_diff</span>
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Main metrics */}
        <div className="grid grid-cols-4 gap-3">
          {DEFAULT_METRICS.map((m) => (
            <div key={m.label} className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
              <p className="text-slate-500 text-xs mb-1">{m.label}</p>
              <p className={`font-bold text-2xl ${m.color}`}>{m.value}</p>
              <p className="text-slate-600 text-xs mt-0.5">{m.sub}</p>
            </div>
          ))}
        </div>

        {/* Fold performance */}
        <div>
          <h2 className="text-slate-300 font-semibold text-sm mb-3">Fold Bazlı Performans</h2>
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 overflow-hidden">
            <div className="grid grid-cols-4 text-xs text-slate-500 font-medium px-4 py-2.5 border-b border-slate-700/40 uppercase tracking-wide">
              <span>Fold</span>
              <span className="text-center">Accuracy</span>
              <span className="text-center">AUC-ROC</span>
              <span className="text-center">Log-Loss</span>
            </div>
            {folds.map((f, i) => (
              <div
                key={f.fold}
                className={`grid grid-cols-4 px-4 py-3 text-sm ${
                  i < folds.length - 1 ? 'border-b border-slate-700/30' : ''
                }`}
              >
                <span className="text-slate-400 font-medium">Fold {f.fold}</span>
                <span className="text-center text-emerald-400 font-semibold">{f.accuracy}%</span>
                <span className="text-center text-blue-400 font-semibold">{f.auc}</span>
                <span className="text-center text-amber-400 font-semibold">{f.logLoss}</span>
              </div>
            ))}
            {/* Average row */}
            <div className="grid grid-cols-4 px-4 py-3 text-sm bg-slate-700/20 border-t border-slate-700/50">
              <span className="text-white font-bold">Ortalama</span>
              <span className="text-center text-emerald-400 font-bold">
                {(folds.reduce((s, f) => s + f.accuracy, 0) / folds.length).toFixed(1)}%
              </span>
              <span className="text-center text-blue-400 font-bold">
                {(folds.reduce((s, f) => s + f.auc, 0) / folds.length).toFixed(3)}
              </span>
              <span className="text-center text-amber-400 font-bold">
                {(folds.reduce((s, f) => s + f.logLoss, 0) / folds.length).toFixed(3)}
              </span>
            </div>
          </div>
        </div>

        {/* Feature importance */}
        <div>
          <h2 className="text-slate-300 font-semibold text-sm mb-3">
            Top 15 Feature Importance <span className="text-slate-500 font-normal">(XGBoost gain)</span>
          </h2>
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-4">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={sortedFeatures}
                  layout="vertical"
                  margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
                >
                  <XAxis
                    type="number"
                    tick={{ fontSize: 10, fill: '#64748b' }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={145}
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    formatter={(v) => [`${(v ?? 0).toFixed(2)}%`, 'Önem']}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {sortedFeatures.map((_, i) => (
                      <Cell key={i} fill={IMPORTANCE_COLORS[i % IMPORTANCE_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Training info */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Veri',          value: '26,332 maç',   sub: '2000–2026 Regular Season'   },
            { label: 'Eğitim Süresi', value: '~45 dk',       sub: 'RTX 4050 6GB, 236 feature'  },
            { label: 'Optuna Sampler',value: 'TPESampler',   sub: 'MedianPruner, n_trials=150' },
          ].map((item) => (
            <div key={item.label} className="bg-slate-800/40 rounded-xl p-4 border border-slate-700/40">
              <p className="text-slate-500 text-xs">{item.label}</p>
              <p className="text-white font-semibold mt-1">{item.value}</p>
              <p className="text-slate-600 text-xs mt-0.5">{item.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
