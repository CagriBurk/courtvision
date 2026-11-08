import type { ReactElement, ReactNode } from 'react'
import {
  AlertTriangle,
  BookOpen,
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
                <B>Sabah</B>: uygulamayı aç, 1-3 dakika bekle. Tahminler, piyasa çizgileri ve dünün
                sonuçları kendiliğinden gelir.
              </li>
              <li>
                <B>Öğlen</B>: <B>Veriyi Güncelle</B> (sakatlık raporu ve çizgiler gün içinde
                değişir).
              </li>
              <li>
                <B>Akşam, maçlardan önce</B>: tekrar <B>Veriyi Güncelle</B>. Karneye geçen tahmin,
                maç başlamadan önceki son güncellemedir.
              </li>
            </ul>
            <Tip>
              Piyasa çizgisi yalnızca başlamamış maçlar için güncellenir; maç başladıktan sonra son
              maç öncesi çizgi korunur.
            </Tip>
          </Card>

          <Card icon={<Target size={15} />} title="Maç kartı">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Çubuk ve yüzdeler</B>: modelin iki takıma verdiği kazanma olasılığı. Ortadaki gri{' '}
                <B>piyasa</B> yüzdesi, kapanış oranlarından marjı çıkarılmış olasılık. Güven rozeti:
                %65+ yüksek, %55-65 orta, altı düşük.
              </li>
              <li>
                <B>Handikap · Toplam</B> satırı: piyasa çizgisi varsa onun etrafında kurulan adil
                handikap ve toplam; yoksa modelinki. Toplamın yanındaki{' '}
                <span className="text-amber-500">*</span> “tahmin gücü düşük” (yalnız model
                kaynaklıysa) demek.
              </li>
              <li>
                Karta tıklayınca sağda detay paneli açılır: piyasa ve model çizgileri yan yana,
                handikap merdiveni, fark bantları, skor aralıkları, “neden kazanır”.
              </li>
            </ul>
          </Card>

          <Card icon={<Scale size={15} />} title="Piyasa, model ve adil çizgi">
            <p>
              <B>Piyasa çizgisi</B>, DraftKings’in maç öncesi handikap ve toplamı (ESPN’den, günde
              birkaç kez). Panelin olasılıkları bu çizgi etrafında, piyasanın geçmiş 3 sezondaki
              hata dağılımıyla kurulur.
            </p>
            <p>
              <B>Model</B> satırı modelin kendi adil çizgisi. Piyasadan 4+ sayı ayrıldığında sarı
              uyarı çıkar: geçmişte bu ayrılıklar çoğunlukla modelin hatasıydı.
            </p>
            <p>
              <B>Adil oran = 1 / olasılık</B>. %62’lik bir seçeneğin adil oranı 1.61.{' '}
              <B>Merdiven</B> üstten alta en güvenliden en riskliye sıralı: handikap büyüdükçe
              olasılık düşer, adil oran yükselir.
            </p>
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
                . Piyasanın kapanış çizgisi aynı sezonda 10.9 sayı.
              </li>
              <li>
                <B>Toplam</B>: ortalama hata <B>{total ? `${total.mae.toFixed(1)} sayı` : '—'}</B>
                {total?.baselineMae != null &&
                  ` (son 10 maç ortalaması ${total.baselineMae.toFixed(1)})`}
                ; piyasa 14.6 sayı.
              </li>
            </ul>
            <Tip>
              Model, halka açık verilerle piyasaya çok yaklaşıyor ama 2012’den bu yana 16 bin maçta
              piyasanın bilmediği bir şey bilmediği görüldü: ayrıldığı yerlerde haklı çıkmadı. Bu
              yüzden olasılıklar piyasa merkezli, model ikinci görüş. Ayrıntı: EVALUATION.md.
            </Tip>
          </Card>

          <Card icon={<Lightbulb size={15} />} title="İpuçları">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <B>Handikap toplamdan daha öngörülebilir.</B> Takımlar arasındaki kalite farkı
                kalıcıdır; toplam ise üçlük şansı ve uzatmaya bağlı.
              </li>
              <li>
                <B>Birden çok seçimin birlikte tutması zordur.</B> %70’lik üç seçimin birlikte tutma
                ihtimali %34.
              </li>
              <li>
                <B>Aynı maçtan iki seçim bağımsız değildir.</B> Kazanan ve handikap birbirine
                bağlıdır, olasılıkları çarpılamaz.
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
                <B>Sezonun ilk haftaları</B>: son-10 ve sezon istatistikleri az maça dayanır;
                modelin çizgisi piyasadan daha uç olabilir.
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
                <B>ESPN çizgi vermezse</B> (nadiren) o gün olasılıklar modelden kurulur ve ekranda
                “piyasa çizgisi yok” yazar.
              </li>
            </ul>
          </Card>

          <Card icon={<Ticket size={15} />} title="Kuponlar">
            <ul className="list-disc pl-5 space-y-1">
              <li>
                Üç handikap kuponu: <B>Güvenli</B> (ayak başı %68-80), <B>Dengeli</B> (%58-68),{' '}
                <B>Riskli</B> (%50-58). Favori her zaman handikap verir (-1.5 ve üstü); hafif
                favoriler güvenli kupona giremez.
              </li>
              <li>
                Olasılıklar piyasa çizgisi etrafında kurulur. Kupona piyasa çizgisi olan, başlamamış
                maçlar girer.
              </li>
              <li>
                <B>Her maç en fazla bir kupona girer</B>: güvenli kupon büyük favorileri, dengeli
                ortadakileri, riskli kalanları alır. Ayak sayısı uygun maç sayısına göre 2-3.
              </li>
              <li>
                Eski <B>Üst (deneysel)</B> kuponu kaldırıldı: toplam modelinin tarafı piyasaya karşı
                %47 tuttu.
              </li>
              <li>Kupon, maçlarından biri başlayınca kilitlenir ve karneye o haliyle geçer.</li>
            </ul>
          </Card>

          <Card icon={<ClipboardCheck size={15} />} title="Karne">
            <p>
              Asıl soru: <B>%70 dendiğinde gerçekten ~%70 tutuyor mu?</B> Karne bunu taraf, handikap
              baremleri ve alt/üst için ayrı ayrı gösterir. Her maç 9 handikap baremi eklediği için
              handikap tablosu birkaç haftada anlamlı hale gelir.
            </p>
            <p>
              <B>Model ↔ Piyasa</B> sekmesi aynı maçlarda modelin ve piyasanın hatasını yan yana
              koyar; canlı <B>w</B> değeri modelin piyasaya bir şey katıp katmadığını ölçer (birkaç
              yüz maç sonra anlamlı).
            </p>
            <Tip>
              Kupon tutma oranı küçük örneklerde çok oynar. Dürüstlüğü barem ve ayak
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
              <B>MODEL_EGIT.bat</B>’a çift tıkla (yaklaşık 2 saat), ardından piyasa kıyasını
              güncelle: <B>src/09_market_eval.py --refresh</B>. Bitince uygulamayı yeniden aç.
              Sezonda ayda bir yeterli.
            </p>
          </Card>
        </div>
      </div>
    </div>
  )
}
