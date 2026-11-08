import type { ReactElement } from 'react'
import { LineModelMetrics } from '../hooks/useModelMetrics'

const pct = (v: number, d = 1): string => `%${(v * 100).toFixed(d)}`

function ModelCard({ title, m }: { title: string; m: LineModelMetrics }): ReactElement {
  const gain = m.baselineMae != null ? m.baselineMae - m.mae : null
  return (
    <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-slate-300 text-sm font-semibold">{title}</p>
        {m.lowSignal ? (
          <span className="text-[10px] px-1.5 py-0.5 rounded border border-amber-500/30 bg-amber-500/10 text-amber-400">
            tahmin gücü düşük
          </span>
        ) : (
          <span className="text-[10px] px-1.5 py-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
            tabanı geçiyor
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <p className="text-white font-bold text-xl tabular-nums">{m.mae.toFixed(2)}</p>
          <p className="text-slate-500 text-[10px]">MAE (sayı)</p>
        </div>
        <div>
          <p className="text-slate-300 font-bold text-xl tabular-nums">
            {m.baselineMae != null ? m.baselineMae.toFixed(2) : '—'}
          </p>
          <p className="text-slate-500 text-[10px]">taban MAE</p>
        </div>
        <div>
          <p
            className={`font-bold text-xl tabular-nums ${gain != null && gain > 0.1 ? 'text-emerald-400' : 'text-slate-400'}`}
          >
            {gain != null ? `${gain > 0 ? '-' : '+'}${Math.abs(gain).toFixed(2)}` : '—'}
          </p>
          <p className="text-slate-500 text-[10px]">fark</p>
        </div>
      </div>
      <p className="text-slate-600 text-[10px]">
        Taban: {m.baseline ?? '—'} · sapma {m.bias > 0 ? '+' : ''}
        {m.bias.toFixed(2)} · ±10 içinde %{m.within10.toFixed(0)}
      </p>

      {/* Barem kalibrasyonu */}
      <div>
        <p className="text-slate-400 text-[11px] font-medium mb-1">
          Barem kalibrasyonu{m.ladderEce != null ? ` (ECE ${pct(m.ladderEce)})` : ''}
        </p>
        <div className="grid grid-cols-9 gap-1 text-center text-[10px]">
          {m.ladderByOffset.map((r) => (
            <div key={r.offset} className="bg-slate-900/40 rounded px-0.5 py-1">
              <p className="text-slate-500">{r.offset > 0 ? `+${r.offset}` : r.offset}</p>
              <p className="text-slate-300 tabular-nums">{Math.round(r.meanPred * 100)}</p>
              <p className="text-slate-500 tabular-nums">{Math.round(r.fracOver * 100)}</p>
            </div>
          ))}
        </div>
        <p className="text-slate-600 text-[10px] mt-1">
          Tahminden barem farkı · modelin üst olasılığı · gerçekleşen üst oranı (%)
        </p>
      </div>

      {m.bands.length > 0 && (
        <div>
          <p className="text-slate-400 text-[11px] font-medium mb-1">Kazanma farkı bantları</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10px]">
            {m.bands.map((b) => {
              const label = `${b.side === 'home' ? 'Ev' : 'Dep'} ${b.hi === null ? `${b.lo}+` : `${b.lo}-${b.hi}`}`
              return (
                <div key={label} className="flex justify-between tabular-nums">
                  <span className="text-slate-400">{label}</span>
                  <span>
                    <span className="text-slate-300">{pct(b.pred)}</span>
                    <span className="text-slate-600"> / </span>
                    <span className="text-slate-500">{pct(b.obs)}</span>
                  </span>
                </div>
              )
            })}
          </div>
          <p className="text-slate-600 text-[10px] mt-1">model / gerçekleşen</p>
        </div>
      )}

      {m.distribution && (
        <p className="text-slate-600 text-[10px]">Hata dağılımı: {m.distribution}</p>
      )}
    </div>
  )
}

export default function LineModelsSection({
  lines,
  holdoutSeason
}: {
  lines?: Partial<Record<'margin' | 'total', LineModelMetrics>>
  holdoutSeason?: string | null
}): ReactElement | null {
  if (!lines || (!lines.margin && !lines.total)) return null
  return (
    <div>
      <h2 className="text-slate-300 font-semibold text-sm mb-3">
        Handikap & Toplam Modelleri{holdoutSeason ? ` — Holdout ${holdoutSeason}` : ''}
      </h2>
      <div className="grid grid-cols-2 gap-3">
        {lines.margin && <ModelCard title="Fark (handikap)" m={lines.margin} />}
        {lines.total && <ModelCard title="Toplam sayı" m={lines.total} />}
      </div>
    </div>
  )
}
