import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'
import { formatDateTr, useModelMetrics } from '../hooks/useModelMetrics'

interface ModelMetricsProps {
  apiBase?: string
}

const fmt = (v: number | null | undefined, digits: number, suffix = '') =>
  v == null ? '—' : `${v.toFixed(digits)}${suffix}`

const IMPORTANCE_COLORS = [
  '#3b82f6',
  '#6366f1',
  '#8b5cf6',
  '#a855f7',
  '#c084fc',
  '#e879f9',
  '#f472b6',
  '#fb7185',
  '#f87171',
  '#fca5a5'
]

export default function ModelMetrics({ apiBase }: ModelMetricsProps) {
  const { data } = useModelMetrics(apiBase)

  const features = data?.featureImportance ?? []
  const folds = data?.foldMetrics ?? []
  const sortedFeatures = [...features].sort((a, b) => b.importance - a.importance).slice(0, 15)
  const avg = (key: 'accuracy' | 'auc' | 'logLoss') =>
    folds.length ? folds.reduce((s, f) => s + f[key], 0) / folds.length : null

  const lift =
    data?.accuracy != null && data?.baselineAccuracy != null
      ? data.accuracy - data.baselineAccuracy
      : null
  const metrics = [
    {
      label: 'Accuracy (OOF)',
      value: fmt(data?.accuracy, 1, '%'),
      sub:
        lift != null
          ? `Baseline: ${data?.baselineAccuracy}% (${lift >= 0 ? '+' : ''}${lift.toFixed(1)}pp)`
          : 'Henüz değerlendirilmedi',
      color: 'text-emerald-400'
    },
    { label: 'AUC-ROC', value: fmt(data?.aucRoc, 3), sub: 'Ayrım gücü', color: 'text-blue-400' },
    { label: 'Log-Loss', value: fmt(data?.logLoss, 3), sub: 'Düşük = iyi', color: 'text-amber-400' },
    {
      label: 'Brier Score',
      value: fmt(data?.brierScore, 3),
      sub: 'Kalibrasyon kalitesi',
      color: 'text-purple-400'
    }
  ]

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-white font-bold text-xl">Model Metrikleri</h1>
        <p className="text-slate-500 text-sm mt-0.5">
          XGBoost · {data?.featureCount ?? '—'} feature ·{' '}
          {data?.dataPoints?.toLocaleString('tr-TR') ?? '—'} maç · 5-Fold TimeSeriesSplit
          {sortedFeatures.length > 0 && (
            <>
              {' '}
              · En önemli: <span className="text-blue-400">{sortedFeatures[0].feature}</span>
            </>
          )}
        </p>
      </div>

      <div className="p-6 space-y-6">
        {!data?.evaluatedAt && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3 text-amber-300 text-xs">
            Değerlendirme metrikleri yok. Çalıştırın:{' '}
            <code className="text-amber-200">python src/04_evaluate_model.py</code>
          </div>
        )}

        {/* Main metrics */}
        <div className="grid grid-cols-4 gap-3">
          {metrics.map((m) => (
            <div
              key={m.label}
              className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50"
            >
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
                {fmt(avg('accuracy'), 1, '%')}
              </span>
              <span className="text-center text-blue-400 font-bold">{fmt(avg('auc'), 3)}</span>
              <span className="text-center text-amber-400 font-bold">
                {fmt(avg('logLoss'), 3)}
              </span>
            </div>
          </div>
        </div>

        {/* Feature importance */}
        <div>
          <h2 className="text-slate-300 font-semibold text-sm mb-3">
            Top 15 Feature Importance{' '}
            <span className="text-slate-500 font-normal">(XGBoost gain)</span>
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
                      fontSize: 11
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
            {
              label: 'Veri',
              value: `${data?.dataPoints?.toLocaleString('tr-TR') ?? '—'} maç`,
              sub: `${data?.dateFrom?.slice(0, 4) ?? '—'}–${data?.dateTo?.slice(0, 4) ?? '—'} Regular Season`
            },
            {
              label: 'Son Eğitim',
              value: formatDateTr(data?.trainedAt),
              sub: `${data?.featureCount ?? '—'} feature`
            },
            {
              label: 'Optuna Sampler',
              value: 'TPESampler',
              sub: `MedianPruner, n_trials=${data?.nTrials ?? '—'}`
            }
          ].map((item) => (
            <div
              key={item.label}
              className="bg-slate-800/40 rounded-xl p-4 border border-slate-700/40"
            >
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
