import type { ReactElement } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, CircleSlash, Info, Ticket } from 'lucide-react'
import CouponCard from '../components/CouponCard'
import { CouponGame, CouponsToday } from '../types'

function GameRow({ g }: { g: CouponGame }): ReactElement {
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-700/30 last:border-0 text-xs">
      <div className="w-4 shrink-0">
        {g.eligible ? (
          <CheckCircle2 size={14} className={g.strong ? 'text-emerald-400' : 'text-slate-400'} />
        ) : (
          <CircleSlash size={14} className="text-slate-600" />
        )}
      </div>
      <span className="w-28 text-slate-200 font-medium">
        {g.away} @ {g.home}
      </span>
      <span className="w-24 text-slate-400">
        favori <span className="text-white font-semibold">{g.favourite}</span>
      </span>
      <span className="w-20 text-slate-400 tabular-nums">{Math.abs(g.center).toFixed(1)} sayı</span>
      <span
        className="w-28 text-slate-500 tabular-nums"
        title="Kazanan modeli / fark modeli: ev sahibinin kazanma olasılığı"
      >
        %{Math.round(g.pClassifier * 100)} / %{Math.round(g.pMargin * 100)}
      </span>
      <span className={`flex-1 ${g.eligible ? 'text-slate-400' : 'text-slate-600'}`}>
        {g.strong ? 'güçlü · ' : ''}
        {g.reasons.join(' · ')}
      </span>
    </div>
  )
}

export default function Coupons({ apiBase }: { apiBase: string }): ReactElement {
  const queryClient = useQueryClient()
  const { data, isLoading, isError } = useQuery<CouponsToday>({
    queryKey: ['coupons', apiBase],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/api/coupons/today`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    refetchInterval: 60_000
  })

  const saveOdds =
    (id: string) =>
    async (odds: number | null): Promise<void> => {
      const res = await fetch(`${apiBase}/api/coupons/${encodeURIComponent(id)}/played-odds`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ odds })
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      await queryClient.invalidateQueries({ queryKey: ['coupons'] })
      await queryClient.invalidateQueries({ queryKey: ['reportCard'] })
    }

  const coupons = data?.coupons ?? []
  const games = data?.games ?? []

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="border-b border-slate-800 px-6 py-4 shrink-0">
        <h1 className="text-white font-bold text-xl flex items-center gap-2">
          <Ticket size={20} className="text-blue-400" />
          Günün Kuponları
        </h1>
        <p className="text-slate-500 text-sm mt-0.5">
          {data?.date ?? '—'} · yalnızca handikap · maç başına tek seçim
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="flex gap-2 bg-blue-500/5 border border-blue-500/20 rounded-lg px-3 py-2 text-[12px] text-blue-100/80">
          <Info size={14} className="shrink-0 mt-0.5 text-blue-400" />
          <p>
            Kuponlar iddaa oranlarını bilmeden kuruldu. Her ayağın yanındaki{' '}
            <span className="text-white font-semibold">en az</span> oran, modelin olasılığına %5
            güvenlik payı eklenmiş eşik: iddaa bu ayağa daha düşük oran veriyorsa o ayak değerli
            değil. Kupon, maçlardan biri başlayınca ya da oynadığın oranı kaydedince kilitlenir ve
            karneye o haliyle geçer.
          </p>
        </div>

        {isLoading && <p className="text-slate-500 text-sm">Yükleniyor…</p>}
        {isError && <p className="text-slate-500 text-sm">API bağlantısı yok.</p>}
        {!isLoading && !isError && coupons.length === 0 && (
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-6 text-center">
            <p className="text-slate-400 text-sm">Bugün kupon yok.</p>
            {data?.note && <p className="text-slate-500 text-xs mt-1">{data.note}</p>}
          </div>
        )}

        {coupons.length > 0 && (
          <div className="grid grid-cols-3 gap-4">
            {coupons.map((c) => (
              <CouponCard key={c.id} coupon={c} onSaveOdds={saveOdds(c.id)} />
            ))}
          </div>
        )}

        {games.length > 0 && (
          <div>
            <h2 className="text-slate-300 font-semibold text-sm mb-1">Maç seçimi</h2>
            <p className="text-slate-600 text-[11px] mb-2">
              Kupona yalnızca iki modelin aynı tarafı benzer olasılıkla gösterdiği, sezonda en az 5
              maç oynamış takımların başlamamış maçları girer. Ayak sayısı güçlü maç sayısına göre
              (2-4); bugün {data?.nLegs ?? 0} ayak.
            </p>
            <div className="bg-slate-800/40 rounded-xl border border-slate-700/40">
              {games.map((g) => (
                <GameRow key={g.gameId} g={g} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
