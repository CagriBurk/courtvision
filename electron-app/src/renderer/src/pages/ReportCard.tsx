import type { ReactElement, ReactNode } from 'react'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ClipboardCheck } from 'lucide-react'
import CouponCard from '../components/CouponCard'
import { Calibration, ReportCard as ReportCardData } from '../types'

const pct = (v: number | null | undefined, d = 0): string =>
  v === null || v === undefined ? '—' : `%${(v * 100).toFixed(d)}`

// 50%..100% ölçeğinde konum
const pos = (p: number): string => `${Math.min(Math.max((p - 0.5) / 0.5, 0), 1) * 100}%`

function CalibrationTable({ cal, unit }: { cal: Calibration; unit: string }): ReactElement {
  if (cal.n === 0) {
    return (
      <p className="text-slate-500 text-xs">
        Henüz sonuçlanan {unit} yok; maçlar bittikçe dolacak.
      </p>
    )
  }
  return (
    <div>
      <div className="grid grid-cols-[80px_50px_60px_70px_1fr] gap-2 text-[10px] text-slate-500 font-medium px-2 pb-1">
        <span>Model dedi</span>
        <span className="text-right">adet</span>
        <span className="text-right">vaat</span>
        <span className="text-right">gerçekleşen</span>
        <span className="pl-2">%95 aralık (50 → 100)</span>
      </div>
      {cal.rows.map((r) => {
        const honest = r.ciLo <= r.meanPred && r.meanPred <= r.ciHi
        return (
          <div
            key={r.lo}
            className="grid grid-cols-[80px_50px_60px_70px_1fr] gap-2 items-center text-xs px-2 py-1 border-t border-slate-700/30"
          >
            <span className="text-slate-300">
              %{Math.round(r.lo * 100)}-{Math.round(r.hi * 100)}
            </span>
            <span className="text-right text-slate-500 tabular-nums">{r.n}</span>
            <span className="text-right text-slate-300 tabular-nums">{pct(r.meanPred, 1)}</span>
            <span
              className={`text-right font-semibold tabular-nums ${honest ? 'text-slate-100' : r.hitRate < r.meanPred ? 'text-red-400' : 'text-emerald-400'}`}
            >
              {pct(r.hitRate, 1)}
            </span>
            <div className="relative h-3 ml-2 bg-slate-900/60 rounded">
              <div
                className="absolute top-0.5 bottom-0.5 bg-slate-600/60 rounded"
                style={{ left: pos(r.ciLo), right: `calc(100% - ${pos(r.ciHi)})` }}
              />
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-blue-400"
                style={{ left: pos(r.meanPred) }}
                title={`vaat ${pct(r.meanPred, 1)}`}
              />
              <div
                className="absolute top-0.5 w-2 h-2 -ml-1 rounded-full bg-white"
                style={{ left: pos(r.hitRate) }}
                title={`gerçekleşen ${pct(r.hitRate, 1)}`}
              />
            </div>
          </div>
        )
      })}
      <p className="text-slate-600 text-[10px] mt-2 px-2">
        Mavi çizgi vaat edilen, beyaz nokta gerçekleşen oran, gri şerit %95 aralık. Mavi çizgi
        şeridin içindeyse model o seviyede dürüst; sürekli sağında kalıyorsa model fazla iyimser.
      </p>
    </div>
  )
}

function Section({
  title,
  sub,
  children
}: {
  title: string
  sub?: string
  children: ReactNode
}): ReactElement {
  return (
    <section className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-4 space-y-3">
      <div>
        <h2 className="text-slate-200 font-semibold text-sm">{title}</h2>
        {sub && <p className="text-slate-500 text-[11px] mt-0.5">{sub}</p>}
      </div>
      {children}
    </section>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }): ReactElement {
  return (
    <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
      <p className="text-slate-500 text-xs mb-1">{label}</p>
      <p className="text-white font-bold text-2xl tabular-nums">{value}</p>
      {sub && <p className="text-slate-600 text-xs mt-0.5">{sub}</p>}
    </div>
  )
}

type Tab = 'coupons' | 'handicap' | 'side' | 'total'

export default function ReportCard({ apiBase }: { apiBase: string }): ReactElement {
  const [tab, setTab] = useState<Tab>('coupons')
  const { data, isLoading, isError } = useQuery<ReportCardData>({
    queryKey: ['reportCard', apiBase],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/api/report-card`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    }
  })

  const c = data?.coupons
  const tabs: { key: Tab; label: string }[] = [
    { key: 'coupons', label: 'Kuponlar' },
    { key: 'handicap', label: 'Handikap' },
    { key: 'side', label: 'Taraf' },
    { key: 'total', label: 'Alt / Üst' }
  ]

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="border-b border-slate-800 px-6 py-4 shrink-0">
        <h1 className="text-white font-bold text-xl flex items-center gap-2">
          <ClipboardCheck size={20} className="text-blue-400" />
          Karne
        </h1>
        <p className="text-slate-500 text-sm mt-0.5">Model ne vaat etti, ne gerçekleşti</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        {isLoading && <p className="text-slate-500 text-sm">Yükleniyor…</p>}
        {isError && <p className="text-slate-500 text-sm">API bağlantısı yok.</p>}

        {data && (
          <>
            <div className="grid grid-cols-5 gap-3">
              <Stat
                label="Taraf (kazanan)"
                value={pct(data.side.hitRate, 1)}
                sub={`${data.side.n} maç · vaat ${pct(data.side.meanPred, 1)}`}
              />
              <Stat
                label="Handikap baremleri"
                value={pct(data.handicap.calibration.hitRate, 1)}
                sub={`${data.handicap.calibration.n} barem · vaat ${pct(data.handicap.calibration.meanPred, 1)}`}
              />
              <Stat
                label="Alt / üst baremleri"
                value={pct(data.total.calibration.hitRate, 1)}
                sub={`${data.total.calibration.n} barem · vaat ${pct(data.total.calibration.meanPred, 1)}`}
              />
              <Stat
                label="Kuponlar"
                value={c ? `${c.tiers.reduce((s, t) => s + t.won, 0)}/${c.n}` : '—'}
                sub={c ? `tuttu · ${c.pending} sonuç bekliyor` : undefined}
              />
              <Stat
                label="Kâr (oynananlar)"
                value={
                  c?.played.profit != null
                    ? `${c.played.profit > 0 ? '+' : ''}${c.played.profit.toFixed(2)}`
                    : '—'
                }
                sub={
                  c?.played.n
                    ? `${c.played.n} kupon · birim başı ROI ${pct(c.played.roi, 1)}`
                    : 'kupon sayfasından oran girersen hesaplanır'
                }
              />
            </div>

            <div className="flex gap-1">
              {tabs.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium ${
                    tab === t.key ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {tab === 'coupons' && c && (
              <>
                <Section
                  title="Kupon seviyeleri"
                  sub="Tutma oranı, kuponun kurulduğu andaki olasılıkla karşılaştırılır."
                >
                  <div className="grid grid-cols-4 gap-3">
                    {c.tiers.map((t) => (
                      <div
                        key={t.tier}
                        className="bg-slate-900/40 rounded-lg p-3 space-y-1 text-xs"
                      >
                        <p className="text-slate-200 font-semibold">{t.label}</p>
                        <p className="text-slate-400">
                          {t.n
                            ? `${t.won}/${t.n} tuttu (${pct(t.hitRate)})`
                            : 'henüz sonuçlanan yok'}
                          {t.expected != null && (
                            <span className="text-slate-500"> · beklenen {pct(t.expected)}</span>
                          )}
                        </p>
                        <p className="text-slate-500">
                          ayaklar:{' '}
                          {t.legs.n
                            ? `${pct(t.legs.hitRate)} tuttu, vaat ${pct(t.legs.meanPred)}`
                            : '—'}
                        </p>
                        <p className="text-slate-500">
                          {t.nPlayed
                            ? `oynanan ${t.nPlayed} · kâr ${t.profit! > 0 ? '+' : ''}${t.profit!.toFixed(2)} · ROI ${pct(t.roi, 1)}`
                            : 'oynanan oran girilmedi'}
                        </p>
                      </div>
                    ))}
                  </div>
                  <p className="text-slate-600 text-[10px]">
                    Kupon tutma oranı küçük örneklerde çok oynar: %37’lik 10 kuponun 1’i de 6’sı da
                    tutabilir. Asıl göstergeler ayak ve barem kalibrasyonu.
                  </p>
                </Section>

                {data.recentCoupons.length > 0 && (
                  <Section title="Son kuponlar" sub="Son 30 gün">
                    <div className="grid grid-cols-2 2xl:grid-cols-4 gap-3">
                      {data.recentCoupons.map((cp) => (
                        <CouponCard key={cp.id} coupon={cp} compact />
                      ))}
                    </div>
                  </Section>
                )}
              </>
            )}

            {tab === 'handicap' && (
              <>
                <Section
                  title="Handikap baremleri: model %X dediğinde ne oldu?"
                  sub="Her maçın 9 handikap baremi, favori tarafından okunur (%30 üst = %70 alt)."
                >
                  <CalibrationTable cal={data.handicap.calibration} unit="barem" />
                </Section>
                <Section
                  title="Kazanma farkı bantları"
                  sub="Bandın ortalama olasılığı ve gerçekleşme oranı"
                >
                  {data.handicap.bands.length === 0 ? (
                    <p className="text-slate-500 text-xs">Henüz sonuçlanan maç yok.</p>
                  ) : (
                    <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                      {data.handicap.bands.map((b) => (
                        <div key={`${b.side}${b.lo}`} className="flex justify-between tabular-nums">
                          <span className="text-slate-400">
                            {b.side === 'home' ? 'Ev' : 'Dep'}{' '}
                            {b.hi === null ? `${b.lo}+` : `${b.lo}-${b.hi}`}
                          </span>
                          <span>
                            <span className="text-slate-300">{pct(b.pred, 1)}</span>
                            <span className="text-slate-600"> → </span>
                            <span className="text-white font-semibold">{pct(b.obs, 1)}</span>
                            <span className="text-slate-600"> ({b.n})</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </Section>
                {data.handicap.error && (
                  <Section title="Fark tahmini hatası">
                    <p className="text-slate-300 text-xs">
                      {data.handicap.error.n} maçta ortalama hata{' '}
                      <span className="text-white font-semibold">
                        {data.handicap.error.mae.toFixed(1)} sayı
                      </span>{' '}
                      · ±10 içinde %{(data.handicap.error.within10 * 100).toFixed(0)} · sapma{' '}
                      {data.handicap.error.bias > 0 ? '+' : ''}
                      {data.handicap.error.bias.toFixed(1)} (holdout referansı: 11.1)
                    </p>
                  </Section>
                )}
              </>
            )}

            {tab === 'side' && (
              <Section
                title="Taraf: model %X kazanır dediğinde ne oldu?"
                sub="Maç başlamadan kaydedilen kazanan tahminleri"
              >
                <CalibrationTable cal={data.side} unit="maç" />
              </Section>
            )}

            {tab === 'total' && (
              <>
                <Section
                  title="Alt / üst baremleri"
                  sub="Sadece takip: toplam modeli basit ortalamayı geçemediği için kuponlara girmiyor."
                >
                  <CalibrationTable cal={data.total.calibration} unit="barem" />
                </Section>
                {data.total.error && (
                  <Section title="Toplam tahmini hatası">
                    <p className="text-slate-300 text-xs">
                      {data.total.error.n} maçta ortalama hata{' '}
                      <span className="text-white font-semibold">
                        {data.total.error.mae.toFixed(1)} sayı
                      </span>{' '}
                      · ±10 içinde %{(data.total.error.within10 * 100).toFixed(0)} (holdout
                      referansı: 15.4)
                    </p>
                  </Section>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
