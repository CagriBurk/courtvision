import { Info, Scale, Sigma, Target } from 'lucide-react'
import { useState, type ReactElement, type ReactNode } from 'react'
import { Game, MarginBand, MarketView, TeamScoreDist } from '../types'
import { fairHandicapLabel, fairOdds, marginLabel, pct, spreadOptions } from '../utils/lines'

function SectionTitle({ icon, children }: { icon: ReactNode; children: ReactNode }): ReactElement {
  return (
    <p className="text-slate-400 text-xs font-medium tracking-wide flex items-center gap-1.5 mb-2">
      {icon}
      {children}
    </p>
  )
}

function LowSignalBadge({ title }: { title: string }): ReactElement {
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded border border-amber-500/30 bg-amber-500/10 text-amber-400 font-medium cursor-help"
      title={title}
    >
      tahmin gücü düşük
    </span>
  )
}

// Olasılığa göre satır rengi: güvenli yeşil, riskli kırmızımsı
function probTone(p: number): string {
  if (p >= 0.65) return 'text-emerald-400'
  if (p >= 0.55) return 'text-sky-300'
  if (p >= 0.45) return 'text-slate-200'
  return 'text-orange-300'
}

function OddsTable({
  rows
}: {
  rows: { label: string; p: number; isFair: boolean }[]
}): ReactElement {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-slate-500 text-[10px]">
          <th className="text-left font-medium pb-1">Seçenek</th>
          <th className="text-right font-medium pb-1">Olasılık</th>
          <th className="text-right font-medium pb-1">Adil oran</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr
            key={r.label}
            className={r.isFair ? 'bg-blue-500/10' : ''}
            title={r.isFair ? 'Adil çizgi: iki tarafa eşit şans' : undefined}
          >
            <td className="py-0.5 pl-1 text-slate-300 font-medium">
              {r.label}
              {r.isFair && <span className="text-blue-400 text-[10px] ml-1">adil</span>}
            </td>
            <td className={`py-0.5 text-right font-semibold ${probTone(r.p)}`}>{pct(r.p)}</td>
            <td className="py-0.5 pr-1 text-right text-slate-200 tabular-nums">{fairOdds(r.p)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ScoreRange({
  abbr,
  color,
  dist
}: {
  abbr: string
  color: string
  dist: TeamScoreDist
}): ReactElement {
  const maxP = Math.max(...dist.bins.map((b) => b.p))
  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-slate-300 text-xs font-semibold">{abbr}</span>
        <span className="text-white text-sm font-bold tabular-nums">{Math.round(dist.mean)}</span>
      </div>
      <p className="text-slate-500 text-[10px] mb-1.5">
        {pct(dist.best10.p)} ihtimalle {dist.best10.lo}-{dist.best10.hi}
      </p>
      <div className="flex items-end gap-px h-8" title="5 sayılık dilimlerin olasılığı">
        {dist.bins.map((b) => (
          <div
            key={b.lo}
            className="flex-1 rounded-sm"
            style={{
              height: `${Math.max((b.p / maxP) * 100, 4)}%`,
              backgroundColor: color,
              opacity: b.lo >= dist.best10.lo && b.hi <= dist.best10.hi ? 0.95 : 0.35
            }}
            title={`${b.lo}-${b.hi}: ${pct(b.p, 1)}`}
          />
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
        <span>{dist.bins[0]?.lo}</span>
        <span>{dist.bins[dist.bins.length - 1]?.hi}</span>
      </div>
    </div>
  )
}

function TotalTable({
  ladder,
  fairLine
}: {
  ladder: { line: number; pOver: number }[]
  fairLine: number
}): ReactElement {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-slate-500 text-[10px]">
          <th className="text-left font-medium pb-1">Barem</th>
          <th className="text-right font-medium pb-1">Üst</th>
          <th className="text-right font-medium pb-1">oran</th>
          <th className="text-right font-medium pb-1">Alt</th>
          <th className="text-right font-medium pb-1">oran</th>
        </tr>
      </thead>
      <tbody>
        {ladder.map((r) => {
          const pUnder = 1 - r.pOver
          return (
            <tr key={r.line} className={r.line === fairLine ? 'bg-blue-500/10' : ''}>
              <td className="py-0.5 pl-1 text-slate-300 font-medium tabular-nums">
                {r.line.toFixed(1)}
                {r.line === fairLine && (
                  <span className="text-blue-400 text-[10px] ml-1">adil</span>
                )}
              </td>
              <td className={`py-0.5 text-right font-semibold ${probTone(r.pOver)}`}>
                {pct(r.pOver)}
              </td>
              <td className="py-0.5 text-right text-slate-200 tabular-nums">{fairOdds(r.pOver)}</td>
              <td className={`py-0.5 text-right font-semibold ${probTone(pUnder)}`}>
                {pct(pUnder)}
              </td>
              <td className="py-0.5 pr-1 text-right text-slate-200 tabular-nums">
                {fairOdds(pUnder)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function BandBars({
  bands,
  game,
  favorite
}: {
  bands: MarginBand[]
  game: Game
  favorite: 'home' | 'away'
}): ReactElement {
  const maxP = Math.max(...bands.map((b) => b.p))
  const label = (b: MarginBand): string => (b.hi === null ? `${b.lo}+` : `${b.lo}-${b.hi}`)
  const sideRows = (side: 'home' | 'away'): ReactElement => {
    const team = side === 'home' ? game.home : game.away
    const rows = bands.filter((b) => b.side === side)
    const sum = rows.reduce((s, b) => s + b.p, 0)
    return (
      <div>
        <p className="text-[10px] text-slate-500 mb-1">
          {team.abbreviation} kazanır ({pct(sum)}) — fark
        </p>
        <div className="space-y-0.5">
          {rows.map((b) => (
            <div key={label(b)} className="flex items-center gap-1.5 text-[10px]">
              <span className="w-9 text-slate-400 tabular-nums">{label(b)}</span>
              <div className="flex-1 h-2.5 bg-slate-700/50 rounded-sm overflow-hidden">
                <div
                  className="h-full rounded-sm"
                  style={{ width: `${(b.p / maxP) * 100}%`, backgroundColor: team.color }}
                />
              </div>
              <span className="w-9 text-right text-slate-300 tabular-nums">{pct(b.p, 1)}</span>
              <span className="w-9 text-right text-slate-500 tabular-nums">{fairOdds(b.p)}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }
  return (
    <div className="space-y-2.5">
      {sideRows(favorite)}
      {sideRows(favorite === 'home' ? 'away' : 'home')}
    </div>
  )
}

function MarketCompare({ game, market }: { game: Game; market: MarketView }): ReactElement {
  const home = game.home.abbreviation
  const away = game.away.abbreviation
  const gap =
    market.marketMargin != null && market.modelMargin != null
      ? Math.abs(market.modelMargin - market.marketMargin)
      : null
  const wMargin = market.w.margin ?? 0
  return (
    <div className="bg-slate-900/50 rounded-lg px-3 py-2 text-xs space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-slate-400">
          Piyasa <span className="text-slate-600">({market.provider || 'kapanış çizgisi'})</span>
        </span>
        <span className="text-white font-semibold tabular-nums">
          {marginLabel(market.marketMargin, home, away)}
          {market.marketTotal != null && ` · ${market.marketTotal.toFixed(1)}`}
          {market.pHome != null && (
            <span className="text-slate-400 font-normal">
              {' '}
              · {home} {pct(market.pHome)}
            </span>
          )}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-slate-400">Model</span>
        <span className="text-slate-300 tabular-nums">
          {marginLabel(market.modelFairMargin, home, away)}
          {market.modelFairTotal != null && ` · ${market.modelFairTotal.toFixed(1)}`}
          {gap != null && gap >= 4 && (
            <span
              className="text-amber-400/80 ml-1 cursor-help"
              title="Model piyasadan 4+ sayı ayrılıyor. Geçmiş 16 bin maçta bu ayrılıklar değer değil, çoğunlukla modelin hatasıydı."
            >
              ({gap.toFixed(1)} sayı ayrı)
            </span>
          )}
        </span>
      </div>
      <p className="text-slate-600 text-[10px]">
        Aşağıdaki olasılıklar piyasa çizgisi etrafında kuruldu
        {wMargin > 0 ? ` (model payı %${Math.round(wMargin * 100)})` : '; model yalnızca görüş'}.
      </p>
    </div>
  )
}

export default function LinesPanel({ game }: { game: Game }): ReactElement | null {
  const [showUnderdog, setShowUnderdog] = useState(false)
  const lines = game.lines
  if (!lines) return null
  const { spread, total, scores, market } = lines
  const home = game.home.abbreviation
  const away = game.away.abbreviation
  const marketPriced = spread?.source === 'market' || total?.source === 'market'

  return (
    <div className="bg-slate-800/40 rounded-xl p-3 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-slate-300 text-xs font-semibold tracking-wide flex items-center gap-1.5">
          <Scale size={12} />
          HANDİKAP & TOPLAM
        </p>
        <span
          className="text-slate-600 cursor-help"
          title={
            marketPriced
              ? 'Olasılıklar piyasanın kapanış çizgisi etrafında, piyasanın geçmiş hata dağılımıyla kurulur. Adil oran = 1 / olasılık.'
              : 'Piyasa çizgisi yok: olasılıklar modelin tahmininden. Model, geçmişte piyasadan daha isabetsizdi. Adil oran = 1 / olasılık.'
          }
        >
          <Info size={12} />
        </span>
      </div>

      {market && <MarketCompare game={game} market={market} />}

      {/* Tahmini skor */}
      {scores && (
        <div>
          <SectionTitle icon={<Target size={11} />}>
            TAHMİNİ SKOR
            {total?.lowSignal && (
              <LowSignalBadge title="Takım skorunun belirsizliği büyük ölçüde toplam sayıdan gelir; toplam modeli basit tabandan daha iyi değil." />
            )}
          </SectionTitle>
          <div className="flex gap-4">
            <ScoreRange abbr={home} color={game.home.color} dist={scores.home} />
            <ScoreRange abbr={away} color={game.away.color} dist={scores.away} />
          </div>
        </div>
      )}

      {/* Handikap merdiveni */}
      {spread && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-slate-400 text-xs font-medium tracking-wide">
              HANDİKAP · adil{' '}
              <span className="text-white font-bold">{fairHandicapLabel(spread, home, away)}</span>
            </p>
            <button
              onClick={() => setShowUnderdog((v) => !v)}
              className="text-[10px] text-blue-400 hover:text-blue-300"
            >
              {showUnderdog ? 'favori tarafı göster' : 'diğer tarafı göster'}
            </button>
          </div>
          <OddsTable
            rows={spreadOptions(spread, home, away, showUnderdog ? 'underdog' : 'favorite')}
          />
          <p className="text-slate-600 text-[10px] mt-1">
            Üstten alta: en güvenliden en riskliye. Handikap arttıkça olasılık düşer, adil oran
            yükselir.
          </p>
        </div>
      )}

      {/* Fark bantları */}
      {spread && spread.bands.length > 0 && (
        <div>
          <SectionTitle icon={<Sigma size={11} />}>KAZANMA FARKI</SectionTitle>
          <BandBars
            bands={spread.bands}
            game={game}
            favorite={spread.fairLine > 0 ? 'home' : 'away'}
          />
          <p className="text-slate-600 text-[10px] mt-1">Sütunlar: olasılık · adil oran</p>
        </div>
      )}

      {/* Toplam */}
      {total && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <p className="text-slate-400 text-xs font-medium tracking-wide">
              TOPLAM SAYI · adil{' '}
              <span className="text-white font-bold">{total.fairLine.toFixed(1)}</span>
            </p>
            {total.lowSignal && (
              <LowSignalBadge
                title={`Model ${total.baseline ?? 'basit tabandan'} daha iyi tahmin etmiyor; bu maç için piyasa çizgisi de yok.`}
              />
            )}
          </div>
          <TotalTable ladder={total.ladder} fairLine={total.fairLine} />
        </div>
      )}
    </div>
  )
}
