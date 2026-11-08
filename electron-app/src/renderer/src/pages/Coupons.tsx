import type { ReactElement } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, CheckCircle2, CircleSlash, Info, Ticket } from 'lucide-react'
import CouponCard from '../components/CouponCard'
import { CouponGame, CouponsToday } from '../types'

function GameRow({ g }: { g: CouponGame }): ReactElement {
  // Piyasa / model çizgileri ayrı sütunlarda; gerekçelerde tekrar etme
  const notes = g.reasons.filter((r) => !r.startsWith('piyasa:') && !r.startsWith('model:'))
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-700/30 last:border-0 text-xs">
      <div className="w-4 shrink-0">
        {g.eligible ? (
          <CheckCircle2
            size={14}
            className={g.source === 'market' ? 'text-emerald-400' : 'text-slate-400'}
          />
        ) : (
          <CircleSlash size={14} className="text-slate-600" />
        )}
      </div>
      <span className="w-28 text-slate-200 font-medium">
        {g.away} @ {g.home}
      </span>
      <span className="w-32 text-slate-400" title="Piyasanın kapanış çizgisi (DraftKings)">
        piyasa <span className="text-white font-semibold">{g.marketLabel ?? '—'}</span>
      </span>
      <span className="w-32 text-slate-500" title="Modelin kendi adil çizgisi (yalnızca görüş)">
        model <span className="text-slate-300">{g.modelLabel ?? '—'}</span>
      </span>
      <span className="w-20 text-slate-400 tabular-nums" title="Adil toplam">
        {g.fairTotal != null ? `toplam ${g.fairTotal.toFixed(1)}` : ''}
      </span>
      <span className={`flex-1 ${g.eligible ? 'text-slate-400' : 'text-slate-600'}`}>
        {notes.join(' · ')}
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
          {data?.date ?? '—'} · favori handikap verir · her maç tek kupon
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="flex gap-2 bg-blue-500/5 border border-blue-500/20 rounded-lg px-3 py-2 text-[12px] text-blue-100/80">
          <Info size={14} className="shrink-0 mt-0.5 text-blue-400" />
          <p>
            Olasılıklar piyasanın kapanış çizgisi etrafında, piyasanın geçmiş hata dağılımıyla
            kurulur; her maç en fazla bir kupona girer, böylece tek bir maç bütün kuponları yakmaz.{' '}
            <span className="text-white font-semibold">Adil</span> oran = 1 / olasılık. Kupon,
            maçlardan biri başlayınca kilitlenir ve karneye o haliyle geçer.
          </p>
        </div>
        <div className="flex gap-2 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2 text-[12px] text-amber-100/80">
          <AlertTriangle size={14} className="shrink-0 mt-0.5 text-amber-400" />
          <p>
            <span className="text-white font-semibold">Neden piyasa?</span> 2012’den bu yana 16 bin
            maçta model, piyasadan ayrıldığı yerlerde haklı çıkmadı: ayrılıklar değer değil,
            çoğunlukla modelin hatasıydı. Bu yüzden kuponlar piyasa çizgisiyle kurulur; modelin
            çizgisi aşağıda yalnızca görüş olarak durur.
            {data?.source === 'model' && ' Bugün piyasa çizgisi yok: kuponlar modelden kuruldu.'}
          </p>
        </div>

        {(data?.skipped ?? []).length > 0 && (
          <div className="text-[11px] text-slate-500 space-y-0.5">
            {data!.skipped!.map((s) => (
              <p key={s.tier}>
                <span className="text-slate-400 font-medium">{s.label} kupon bugün yok:</span>{' '}
                {s.reason}.
              </p>
            ))}
          </div>
        )}

        {isLoading && <p className="text-slate-500 text-sm">Yükleniyor…</p>}
        {isError && <p className="text-slate-500 text-sm">API bağlantısı yok.</p>}
        {!isLoading && !isError && coupons.length === 0 && (
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-6 text-center">
            <p className="text-slate-400 text-sm">Bugün kupon yok.</p>
            {data?.note && <p className="text-slate-500 text-xs mt-1">{data.note}</p>}
          </div>
        )}

        {coupons.length > 0 && (
          <div className="grid grid-cols-2 2xl:grid-cols-4 gap-4">
            {coupons.map((c) => (
              <CouponCard key={c.id} coupon={c} onSaveOdds={saveOdds(c.id)} />
            ))}
          </div>
        )}

        {games.length > 0 && (
          <div>
            <h2 className="text-slate-300 font-semibold text-sm mb-1">Maç seçimi</h2>
            <p className="text-slate-600 text-[11px] mb-2">
              Kupona piyasa çizgisi olan, başlamamış maçlar girer; her maç tek kupona. Güvenli kupon
              büyük favorileri, dengeli ortadakileri, riskli kalanları alır. Ayak sayısı uygun maç
              sayısına göre (2-3); bugün {data?.nLegs ?? 0} ayak.
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
