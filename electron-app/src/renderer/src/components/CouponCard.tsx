import type { ReactElement } from 'react'
import { useState } from 'react'
import { CheckCircle, Clock, Lock, XCircle } from 'lucide-react'
import { Coupon, CouponLeg } from '../types'
import { formatTipoff } from '../utils/game'

const tierStyle: Record<Coupon['tier'], { ring: string; text: string; chip: string }> = {
  guvenli: {
    ring: 'border-emerald-500/30',
    text: 'text-emerald-400',
    chip: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
  },
  dengeli: {
    ring: 'border-sky-500/30',
    text: 'text-sky-400',
    chip: 'bg-sky-500/10 text-sky-300 border-sky-500/30'
  },
  riskli: {
    ring: 'border-orange-500/30',
    text: 'text-orange-400',
    chip: 'bg-orange-500/10 text-orange-300 border-orange-500/30'
  },
  ust: {
    ring: 'border-violet-500/30',
    text: 'text-violet-400',
    chip: 'bg-violet-500/10 text-violet-300 border-violet-500/30'
  }
}

const pct = (p: number): string => `%${Math.round(p * 100)}`

function LegRow({ leg }: { leg: CouponLeg }): ReactElement {
  const matchup =
    leg.market === 'total' || !leg.team
      ? `${leg.away} @ ${leg.home}`
      : `${leg.team === leg.home ? leg.away : leg.home} karşısında ${leg.team === leg.home ? 'evinde' : 'deplasmanda'}`
  const tip = formatTipoff(leg.gameTimeUtc)
  return (
    <div className="flex items-center gap-2 py-1.5 border-b border-slate-700/40 last:border-0">
      <div className="w-4 shrink-0">
        {leg.won === true && <CheckCircle size={14} className="text-emerald-400" />}
        {leg.won === false && <XCircle size={14} className="text-red-400" />}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white text-sm font-semibold">
          {leg.label}
          {leg.fair && (
            <span
              className="text-slate-500 text-[10px] font-normal ml-1.5"
              title={
                leg.source === 'market'
                  ? 'Piyasa çizgisi: olasılıklar onun etrafında kuruldu'
                  : 'Modelin %50 gördüğü çizgi (piyasa çizgisi yoktu)'
              }
            >
              {leg.source === 'market' ? 'piyasa' : 'adil'} {leg.fair}
            </span>
          )}
        </p>
        <p className="text-slate-500 text-[10px] truncate">
          {matchup}
          {tip && ` · ${tip}`}
          {leg.score && ` · ${leg.score[0]}-${leg.score[1]}`}
        </p>
      </div>
      <div className="text-right w-10">
        <p className="text-slate-200 text-xs font-semibold tabular-nums">{pct(leg.p)}</p>
        <p className="text-slate-600 text-[9px]">olasılık</p>
      </div>
      <div className="text-right w-10">
        <p className="text-slate-300 text-xs tabular-nums">{leg.fairOdds.toFixed(2)}</p>
        <p className="text-slate-600 text-[9px]">adil</p>
      </div>
      <div className="text-right w-12">
        <p className="text-white text-xs font-bold tabular-nums">≥ {leg.minOdds.toFixed(2)}</p>
        <p className="text-slate-600 text-[9px]">en az</p>
      </div>
    </div>
  )
}

function PlayedOddsForm({
  coupon,
  onSave
}: {
  coupon: Coupon
  onSave: (odds: number | null) => Promise<void>
}): ReactElement {
  const [value, setValue] = useState(coupon.playedOdds?.toString() ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (odds: number | null): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      await onSave(odds)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Kaydedilemedi')
    } finally {
      setBusy(false)
    }
  }

  if (coupon.playedOdds) {
    return (
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-slate-400">
          Oynandı: <span className="text-white font-semibold">{coupon.playedOdds.toFixed(2)}</span>
          {coupon.playedOdds < coupon.minOdds && (
            <span className="text-amber-400 ml-1">(en az oranın altında)</span>
          )}
        </span>
        {!coupon.result && (
          <button
            onClick={() => submit(null)}
            disabled={busy}
            className="text-slate-500 hover:text-slate-300"
          >
            temizle
          </button>
        )}
      </div>
    )
  }
  if (coupon.result) return <></>

  const odds = Number(value.replace(',', '.'))
  return (
    <div>
      <div className="flex items-center gap-1.5">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Oynadığın toplam oran"
          inputMode="decimal"
          className="flex-1 min-w-0 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white placeholder:text-slate-600 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={() => submit(odds)}
          disabled={busy || !(odds > 1)}
          className="text-xs px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium"
        >
          Kaydet
        </button>
      </div>
      <p className="text-slate-600 text-[10px] mt-1">
        İsteğe bağlı: karnede kâr/zarar için. Kaydedince kupon kilitlenir.
      </p>
      {error && <p className="text-red-400 text-[10px] mt-1">{error}</p>}
    </div>
  )
}

export default function CouponCard({
  coupon,
  onSaveOdds,
  compact = false
}: {
  coupon: Coupon
  onSaveOdds?: (odds: number | null) => Promise<void>
  compact?: boolean
}): ReactElement {
  const style = tierStyle[coupon.tier]
  return (
    <div className={`bg-slate-800/50 rounded-xl border ${style.ring} p-4 flex flex-col gap-3`}>
      <div className="flex items-start justify-between">
        <div>
          <span
            className={`text-[10px] px-2 py-0.5 rounded-full border font-semibold ${style.chip}`}
          >
            {coupon.label} · ayak başı ≈ {pct(coupon.target)}
          </span>
          {coupon.experimental && (
            <span
              className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded border border-amber-500/30 bg-amber-500/10 text-amber-400"
              title="Eski deneysel üst kuponu: toplam modeli piyasaya karşı %47 tuttuğu için artık kurulmuyor"
            >
              kaldırıldı
            </span>
          )}
          {compact && <p className="text-slate-500 text-[10px] mt-1">{coupon.date}</p>}
        </div>
        <div className="flex items-center gap-1.5 text-[10px]">
          {coupon.result === 'won' && (
            <span className="text-emerald-400 font-semibold flex items-center gap-1">
              <CheckCircle size={12} /> Tuttu
            </span>
          )}
          {coupon.result === 'lost' && (
            <span className="text-red-400 font-semibold flex items-center gap-1">
              <XCircle size={12} /> Tutmadı
            </span>
          )}
          {!coupon.result && coupon.locked && (
            <span
              className="text-slate-400 flex items-center gap-1"
              title="Maç başladı ya da oran kaydedildi; kupon artık değişmez"
            >
              <Lock size={11} /> kilitli
            </span>
          )}
          {!coupon.result && !coupon.locked && (
            <span
              className="text-slate-500 flex items-center gap-1"
              title="Maçlar başlayana kadar her güncellemede yeniden kurulur"
            >
              <Clock size={11} /> güncellenebilir
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <p className={`font-bold text-xl tabular-nums ${style.text}`}>{pct(coupon.p)}</p>
          <p className="text-slate-500 text-[10px]">tutma olasılığı</p>
        </div>
        <div>
          <p className="text-slate-200 font-bold text-xl tabular-nums">
            {coupon.fairOdds.toFixed(2)}
          </p>
          <p className="text-slate-500 text-[10px]">adil toplam oran</p>
        </div>
        <div>
          <p className="text-white font-bold text-xl tabular-nums">≥ {coupon.minOdds.toFixed(2)}</p>
          <p className="text-slate-500 text-[10px]">en az toplam oran</p>
        </div>
      </div>

      <div>
        {coupon.legs.map((leg) => (
          <LegRow key={leg.gameId} leg={leg} />
        ))}
      </div>

      {onSaveOdds && <PlayedOddsForm coupon={coupon} onSave={onSaveOdds} />}
    </div>
  )
}
