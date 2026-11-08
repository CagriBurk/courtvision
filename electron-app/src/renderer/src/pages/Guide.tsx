import type { ReactElement, ReactNode } from 'react'
import { useState } from 'react'
import {
  AlertTriangle,
  BookOpen,
  Calculator,
  ClipboardCheck,
  Clock,
  Lightbulb,
  RefreshCw,
  Scale,
  Target,
  Ticket,
  TrendingUp
} from 'lucide-react'
import { useModelMetrics } from '../hooks/useModelMetrics'

function Card({
  icon,
  title,
  children
}: {
  icon: ReactNode
  title: string
  children: ReactNode
}): ReactElement {
  return (
    <section className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-4">
      <h2 className="text-white font-semibold text-sm flex items-center gap-2 mb-3">
        <span className="text-blue-400">{icon}</span>
        {title}
      </h2>
      <div className="text-slate-300 text-[13px] leading-relaxed space-y-2">{children}</div>
    </section>
  )
}

function Tip({
  tone = 'tip',
  children
}: {
  tone?: 'tip' | 'warn'
  children: ReactNode
}): ReactElement {
  const style =
    tone === 'warn'
      ? 'border-amber-500/30 bg-amber-500/5 text-amber-200/90'
      : 'border-blue-500/30 bg-blue-500/5 text-blue-100/90'
  return <div className={`border rounded-lg px-3 py-2 text-[12px] ${style}`}>{children}</div>
}

function B({ children }: { children: ReactNode }): ReactElement {
  return <span className="text-white font-semibold">{children}</span>
}

// Model olasılığı + bahis oranı → adil oran, beklenen değer
function ValueCalculator(): ReactElement {
  const [prob, setProb] = useState('62')
  const [odds, setOdds] = useState('1.75')
  const p = Number(prob.replace(',', '.')) / 100
  const o = Number(odds.replace(',', '.'))
  const valid = p > 0 && p < 1 && o > 1
  const fair = valid ? 1 / p : null
  const ev = valid ? p * o - 1 : null
  const verdict =
    ev === null ? null : o >= (fair as number) * 1.05 ? 'value' : ev > 0 ? 'thin' : 'no'

  return (
    <div className="bg-slate-900/50 rounded-lg p-3 space-y-3">
      <div className="flex gap-3">
        <label className="flex-1 text-[11px] text-slate-400">
          Model olasılığı (%)
          <input
            value={prob}
            onChange={(e) => setProb(e.target.value)}
            inputMode="decimal"
            className="mt-1 w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-white text-sm tabular-nums focus:outline-none focus:border-blue-500"
          />
        </label>
        <label className="flex-1 text-[11px] text-slate-400">
          iddaa oranı
          <input
            value={odds}
            onChange={(e) => setOdds(e.target.value)}
            inputMode="decimal"
            className="mt-1 w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-white text-sm tabular-nums focus:outline-none focus:border-blue-500"
          />
        </label>
      </div>
      {valid ? (
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-white font-bold tabular-nums">{fair?.toFixed(2)}</p>
            <p className="text-slate-500 text-[10px]">adil oran</p>
          </div>
          <div>
            <p
              className={`font-bold tabular-nums ${(ev as number) > 0 ? 'text-emerald-400' : 'text-red-400'}`}
            >
              {(ev as number) > 0 ? '+' : ''}
              {((ev as number) * 100).toFixed(1)}%
            </p>
            <p className="text-slate-500 text-[10px]">beklenen değer</p>
          </div>
          <div>
            <p
              className={`font-bold text-sm ${
                verdict === 'value'
                  ? 'text-emerald-400'
                  : verdict === 'thin'
                    ? 'text-amber-400'
                    : 'text-red-400'
              }`}
            >
              {verdict === 'value' ? 'Değerli' : verdict === 'thin' ? 'Sınırda' : 'Değersiz'}
            </p>
            <p className="text-slate-500 text-[10px]">%5 güvenlik payıyla</p>
          </div>
        </div>
      ) : (
        <p className="text-slate-500 text-[11px]">Olasılık 0-100 arası, oran 1’den büyük olmalı.</p>
      )}
    </div>
  )
}

export default function Guide(): ReactElement {
  const { data } = useModelMetrics()
  const margin = data?.lines?.margin
  const total = data?.lines?.total

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="border-b border-slate-800 px-6 py-4 shrink-0">
        <h1 className="text-white font-bold text-xl flex items-center gap-2">
          <BookOpen size={20} className="text-blue-400" />
          Nasıl Kullanılır
        </h1>
        <p className="text-slate-500 text-sm mt-0.5">Günlük rutin, ekranları okumak, ipuçları</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-2 gap-4 max-w-6xl">
          <Card icon={<Clock size={15} />} title="Günlük rutin (Türkiye saati)">
            <p>
              NBA günü yaklaşık <B>08:00</B>’de değişir (Mart ortasından sonra 07:00). Akşam maçları
              gece <B>03:00 – 06:30</B> arası oynanır.
            </p>
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Sabah</B>: uygulamayı aç, 1-3 dakika bekle. Tahminler ve dünün sonuçları
                kendiliğinden gelir.
              </li>
              <li>
                <B>Öğlen</B>: <B>Veriyi Güncelle</B> (sakatlık raporu gün içinde değişir).
              </li>
              <li>
                <B>Akşam, maçlardan önce</B>: tekrar <B>Veriyi Güncelle</B>. Karneye geçen tahmin,
                maç başlamadan önceki son güncellemedir.
              </li>
            </ul>
            <Tip>
              Kupon yapmadan hemen önce bir kez güncelle: son dakika sakatlıkları en çok handikabı
              ve toplamı oynatır.
            </Tip>
          </Card>

          <Card icon={<Target size={15} />} title="Maç kartı">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Çubuk ve yüzdeler</B>: iki takımın kazanma olasılığı. Güven rozeti: %65+ yüksek,
                %55-65 orta, altı düşük.
              </li>
              <li>
                <B>Handikap · Toplam · Skor</B> satırı: modelin adil handikabı, adil toplamı ve
                tahmini skoru. Toplamın yanındaki <span className="text-amber-500">*</span> “tahmin
                gücü düşük” demek.
              </li>
              <li>
                Karta tıklayınca sağda detay paneli açılır: handikap merdiveni, fark bantları, skor
                aralıkları, “neden kazanır”.
              </li>
            </ul>
          </Card>

          <Card icon={<Scale size={15} />} title="Adil çizgi ve adil oran">
            <p>
              <B>Adil çizgi</B>, modelin iki tarafa da %50 şans verdiği barem. Örneğin adil handikap{' '}
              <B>LAL -12.5</B> ise model “LAL 13+ farkla kazanır” ile “kazanamaz” arasında kararsız.
            </p>
            <p>
              <B>Adil oran = 1 / olasılık</B>. %62’lik bir seçeneğin adil oranı 1.61. iddaa bu
              seçeneğe <B>1.61’den yüksek</B> oran veriyorsa model bahsi değerli görür.
            </p>
            <p>
              <B>Merdiven</B> üstten alta en güvenliden en riskliye sıralı: handikap büyüdükçe
              olasılık düşer, adil oran yükselir. Aynı maç için iddaa’nın açtığı farklı baremlerden,
              adil orana göre en iyi fiyatlananı seçebilirsin.
            </p>
          </Card>

          <Card icon={<Calculator size={15} />} title="Değer hesaplayıcı">
            <p className="text-slate-400 text-[12px]">
              Panelden bir seçeneğin olasılığını, iddaa’dan oranını yaz. Beklenen değer = olasılık ×
              oran − 1.
            </p>
            <ValueCalculator />
            <Tip>
              <B>%5 güvenlik payı</B>: model bahis şirketinden az şey bilebilir (son dakika kadro,
              hakem). Oran adil oranı ancak belirgin şekilde geçiyorsa değerli say.
            </Tip>
          </Card>

          <Card icon={<TrendingUp size={15} />} title="Modeller ne kadar iyi?">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Kazanan</B>: {data?.holdoutSeason ?? 'son'} sezonunda doğruluk{' '}
                <B>{data?.accuracy != null ? `%${data.accuracy.toFixed(1)}` : '—'}</B> (hep ev
                sahibi deseydi{' '}
                {data?.baselineAccuracy != null ? `%${data.baselineAccuracy.toFixed(1)}` : '—'}).
              </li>
              <li>
                <B>Handikap</B>: ortalama hata{' '}
                <B>{margin ? `${margin.mae.toFixed(1)} sayı` : '—'}</B>
                {margin?.baselineMae != null &&
                  ` (basit Elo tahmini ${margin.baselineMae.toFixed(1)})`}
                . Barem olasılıkları gerçekleşen oranlarla uyumlu.
              </li>
              <li>
                <B>Toplam</B>: ortalama hata <B>{total ? `${total.mae.toFixed(1)} sayı` : '—'}</B>
                {total?.baselineMae != null &&
                  ` (son 10 maç ortalaması ${total.baselineMae.toFixed(1)})`}
                .
                {total?.lowSignal &&
                  ' Basit ortalamadan iyi değil; olasılıklar dürüst ama avantaj sağlamaz.'}
              </li>
            </ul>
            <p className="text-slate-500 text-[11px]">Ayrıntı: Model Metrikleri sayfası.</p>
          </Card>

          <Card icon={<Lightbulb size={15} />} title="İpuçları">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Handikap toplamdan daha güvenilir.</B> Takımlar arasındaki kalite farkı
                kalıcıdır; toplam ise üçlük şansı ve uzatmaya bağlı.
              </li>
              <li>
                <B>Toplamda yarım-bir sayıya güvenme.</B> iddaa çizgisi adil çizgiden birkaç sayı
                uzaksa bakmaya değer.
              </li>
              <li>
                <B>Kombinede marj çarpılır.</B> 3 maçlık kupon ancak her ayağı ayrı ayrı değerliyse
                değerlidir; %70’lik üç seçimin birlikte tutma ihtimali %34.
              </li>
              <li>
                <B>Aynı maçtan iki seçim koyma.</B> Kazanan ve handikap birbirine bağlıdır,
                olasılıkları çarpılamaz.
              </li>
              <li>
                <B>Skor aralıkları geniştir.</B> Bir takımın skoru ±11-12 sayı oynar; 10 sayılık en
                olası aralık bile ~%33.
              </li>
            </ul>
          </Card>

          <Card icon={<AlertTriangle size={15} />} title="Dikkat">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Sezonun ilk haftaları</B>: son-10 ve sezon istatistikleri az maça dayanır,
                tahminler daha belirsiz.
              </li>
              <li>
                <B>Playoff</B>: model normal sezon için kurulu, playoff tahminleri güvenilir
                olmayabilir.
              </li>
              <li>
                <B>Yıldız oyuncu maç saatine yakın oynamayacak çıkarsa</B> tahmin eskimiştir: Veriyi
                Güncelle.
              </li>
              <li>
                Model bir referanstır, garanti değil. Kaybetmeyi göze alamayacağın parayla oynama.
              </li>
            </ul>
          </Card>

          <Card icon={<Ticket size={15} />} title="Kuponlar">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                Üç handikap kuponu: <B>Güvenli</B> (ayak başı %68-80), <B>Dengeli</B> (%58-68),{' '}
                <B>Riskli</B> (%50-58). Favori her zaman handikap verir (-1.5 ve üstü); hafif
                favoriler güvenli kupona giremez. Risk arttıkça handikap büyür.
              </li>
              <li>
                <B>Üst (deneysel)</B> kuponu: aynı maçlarda ≈%62&apos;lik üst baremleri. Toplam
                modeli zayıf olduğu için karnede ayrı izlenir; karma kupon yapacaksan aynı maçın
                handikabı ile üstünü birlikte koyma.
              </li>
              <li>
                <B>Piyasa kontrolü:</B> ayağın yanındaki <B>adil</B> çizgiye bak. iddaa&apos;nın
                baremleri bundan 4+ sayı uzaksa (adil MIA -3.5, iddaa en düşük MIA -7.5) piyasa bir
                şey biliyor olabilir; o ayağı çıkar.
              </li>
              <li>
                Sadece iki modelin aynı tarafı benzer olasılıkla gösterdiği maçlar girer; ayak
                sayısı (2-4) güçlü maç sayısına göre. Kalabalık gün diye 4 ayak zorlanmaz.
              </li>
              <li>
                Her ayağın <B>en az</B> oranı: iddaa o seçeneğe bunun altında oran veriyorsa o ayak
                değerli değil. Kuponun “en az toplam oranı” da aynı mantık.
              </li>
              <li>
                Kupon, maçlarından biri başlayınca kilitlenir. Oynadığın kuponun toplam oranını
                kaydedersen hemen kilitlenir ve karnede kâr/zarar hesaplanır (isteğe bağlı).
              </li>
            </ul>
          </Card>

          <Card icon={<ClipboardCheck size={15} />} title="Karne">
            <p>
              Asıl soru: <B>model %70 dediğinde gerçekten ~%70 tutuyor mu?</B> Karne bunu taraf,
              handikap baremleri ve alt/üst için ayrı ayrı gösterir. Her maç 9 handikap baremi
              eklediği için handikap tablosu birkaç haftada anlamlı hale gelir.
            </p>
            <Tip>
              Kupon tutma oranı küçük örneklerde çok oynar. Modelin dürüstlüğünü barem ve ayak
              kalibrasyonundan oku; mavi çizgi gri şeridin içindeyse o seviyede vaat tutuyor.
            </Tip>
          </Card>

          <Card icon={<RefreshCw size={15} />} title="Geçmiş ve yeniden eğitim">
            <p>
              <B>Geçmiş</B> sayfası her maçın sonucunu, fark ve toplam için “tahmin → gerçek”
              değerini gösterir (yeşil: ±5 içinde, kırmızı: 10+ sayı hata). Üstteki kartlar canlı
              ortalama hatayı özetler.
            </p>
            <p>
              <B>Modeli yeniden eğitmek</B> için uygulamayı kapat ve proje klasöründeki{' '}
              <B>MODEL_EGIT.bat</B>’a çift tıkla (yaklaşık 2 saat). Bitince uygulamayı yeniden aç.
              Sezonda ayda bir yeterli.
            </p>
          </Card>
        </div>
      </div>
    </div>
  )
}
