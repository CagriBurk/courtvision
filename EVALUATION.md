# CourtVision — Değerlendirme: Model ve Piyasa

Kasım 2026. Bu rapor modelin ne kadar iyi olduğunu ve bahis piyasasının kapanış çizgisine
karşı ne kattığını, sızıntısız bir düzenekle ölçer. Kısa özet:

- Model, halka açık kutu skoru verileriyle **piyasanın kapanış çizgisine 0.1-0.2 sayı
  yaklaşıyor** (fark hatası 10.96-11.07, piyasa 10.89), ama piyasanın bilmediği bir şey bilmiyor.
- İlk ölçümde görülen avantaj (w ≈ 0.3) **geriye dönük bilgi sızıntısıydı**: eğitimde "eksik
  oyuncu", maçın kendi kutu skorundan okunuyordu. Oyuncu bilgisi çıkarılınca w ≈ 0.
- Toplam modeli piyasaya hiçbir şey katmıyor (w ≈ 0); tarafını oynamak %47 tuttu.
- Uygulama bu yüzden olasılıkları **piyasa çizgisi etrafında** kurar; model ikinci görüş olarak
  kalır ve karne canlıda ikisini kıyaslamaya devam eder.

---

## 1. Veri ve düzenek

| | |
|---|---|
| Maçlar | 30.194 normal sezon maçı, 2001-02 → 2026-27 (nba_api) |
| Holdout | 2025-26 (1.225 maç); Optuna aramasında hiç görülmedi |
| Walk-forward OOF | holdout öncesi sezonlarda 5 katlı `TimeSeriesSplit` (04 ile aynı) |
| Sonrası | 2026-27'nin ilk 138 maçı; holdout dahil eğitilmiş modelle (sezon başı) |
| Piyasa çizgileri | 16.890 maç, 2012-13 → 2026-27, ESPN (aşağıda kaynaklar) |

Özellikler `shift(1)` ile yalnızca önceki maçlardan üretilir; canlı yol eğitimle aynı
pipeline'dan geçer (test: `test_live_features_match_training*`).

## 2. Modelin kendi başına başarısı (2025-26 holdout)

| Model | Ölçüt | Model | Basit taban |
|---|---|---|---|
| Kazanan | doğruluk / log-loss | %69.3 / 0.586 | hep ev sahibi %55.4 |
| Fark (ev − dep.) | ortalama hata (MAE) | 11.07 sayı (×1.22 düzeltmeyle 10.96) | Elo farkı 11.61 |
| Toplam | ortalama hata | 15.38 sayı | son-10 ortalaması 15.35 |

Fark modeli tahminleri ortalamaya sıkıştırıyor (gerçek ≈ 1.2 × tahmin); hata dağılımı bu yüzden
düzeltilmiş ortalama + ampirik şekille kuruldu, 12 aday arasından holdout CRPS'e göre seçildi.
Toplam modeli tabanı geçemiyor: sezon içi toplam dalgalanmasının yalnızca ~%6'sı maç öncesi
veriden açıklanabiliyor (`08_line_experiment.py`'de tempo, verim, lig trendi denendi).

*Bu tablo sızıntılı özelliklerle eğitilmiş modelin; sızıntısız sonuçlar 6. bölümde.*

## 3. Piyasa ile kıyas

Kapanış çizgisi (2025-26 holdout, aynı 1.224 maç):

| | Piyasa | Model |
|---|---|---|
| Fark MAE | **10.89** | 10.96 – 11.07 |
| Toplam MAE | **14.60** | 15.38 |
| Kazanan log-loss / doğruluk | **0.568** / %69.2 | 0.586 / %69.3 |

Piyasanın daha isabetli olması beklenen bir şey. Asıl soru modelin **ek bilgi** taşıyıp
taşımadığı. Bunu her maç için şu regresyonla ölçtük:

```
gerçek ≈ piyasa + w · (model − piyasa)
```

- **w ≈ 0:** model, piyasanın zaten bildiğini söylüyor.
- **w > 0:** model piyasadan ayrıldığı yönde haklı; ayrıldığı kadar değil, w kadar.

w OOF'ta öğrenildi ve holdout'ta sınandı (`09_market_eval.py`). Kazananda aynı soru lojistik
harmanla soruldu: `logit(p) = c + a·logit(piyasa) + b·logit(model)`.

## 4. Bulgu: avantaj geriye dönük bilgiydi

| Fark modeli | w (OOF, 15.522 maç) | w (holdout) | Modelin tarafı, holdout |
|---|---|---|---|
| Oyuncu eksikliği kutu skorundan (ilk ölçüm) | 0.35 ± 0.03 | 0.32 ± 0.11 | %52.9 (başabaş %52.4) |
| Oyuncu eksikliği özellikleri yok | 0.05 ± 0.03 | −0.13 ± 0.10 | %47.6 |

Kazananda harman katsayısı da aynı biçimde 0.11 ± 0.05'ten −0.01 ± 0.05'e indi.

**Neden?** Eğitimde bir oyuncu "o maçta süre almadıysa" eksik sayılıyordu. Bu bilgi maçın kendi
kutu skorundan geliyor: son dakika kadro dışı kalmaları, dinlendirme kararlarını ve en önemlisi
**şüpheli (questionable) oyuncuların sonunda oynayıp oynamadığını** kapsıyor. Piyasa bu son
bilgiyi bilemez; olasılığını fiyatlar. Model ise testte cevabı biliyordu.

**Kanıt:** sızıntı varsa w en çok "kimsenin eksik olmadığı" maçlarda görünmeli, çünkü bunlar
tam da "şüpheliler sonunda oynadı" maçları. Öyle çıktı:

| Rotasyondan eksik oyuncu (geriye dönük) | maç | w (OOF) |
|---|---|---|
| hiç | 2.820 | **0.66 ± 0.07** |
| 1 | 4.449 | 0.50 ± 0.06 |
| 2+ | 8.253 | 0.17 ± 0.04 |

**Düzeltme** (`f0440be`): eğitimde bir oyuncu ancak **bu maçta ve takımın bir önceki maçında**
süre almadıysa eksik sayılıyor, yani maç öncesi bilinen bir yokluk. Canlıda sakatlık raporunun
"Out/Doubtful" listesi kullanılmaya devam ediyor (gün içinde açıklanan yeni yoklukları da
içerdiği için canlı model eğitimdeki tanımdan biraz fazlasını bilir; ölçüm temkinli kalır).

**Toplam** modelinde sızıntı olsun olmasın w ≈ 0 (OOF 0.01 ± 0.03, holdout −0.15 ± 0.10); modelin
tarafını oynamak holdout'ta %47.0 tuttu (z = −3.8). Eski "Üst (deneysel)" kuponu bu yüzden kaldırıldı.

**Sezon başı:** 2026-27'nin ilk maçlarında model piyasadan ortalama 4.9 sayı ayrılıyor (sezonun
kalanında 2.7). 8 Kasım'da görülen "model favorileri piyasadan 5-10 sayı uç görüyor" gözlemi
bunun sonucu: az maçlık son-10 istatistikleri ve ×1.2 düzeltme farkı büyütüyor; bu farklar değer
değil, hata.

## 5. Veri kalitesi: piyasa verisinden öğrenilenler

Kapanış çizgileri ESPN'in resmi olmayan API'sinden (`espn_odds.py`):

- **Kaynak sezona göre değişiyor.** Maç özetindeki `pickcenter` yalnız DraftKings dönemini
  içeriyor; Aralık 2025'e kadarki ESPN BET çizgileri ve eski sezonlar core API'de. Sezonlar:
  2012-13 consensus (yalnız handikap), 2013-17 5Dimes, 2017-19 Caesars, 2019-24 DraftKings,
  2024-25 ESPN BET, sonrası DraftKings.
- **Her maçta onlarca kaynak var ve hepsinin önceliği aynı.** İlk gelen çoğu zaman bir tahmin
  sitesiydi (numberfire, teamrankings, accuscore) ya da açılış çizgisiydi. Kaynak artık sabit bir
  sırayla seçiliyor; yalnız tahmin sitesi olan 132 maç "çizgisiz" sayıldı.
- **Bazı "kapanış" çizgileri maç sırasında alınmış.** Eski maçlarda her şirketin yalnız son
  çizgisi kalmış; ~70 maçta bu canlı bir çizgi (ör. ~230'luk maçta toplam 263). Bu çizgilerde
  sonuç çizgiye ±2 sayı yakın %22 oranında düştü (normali ~%8). Üç ve daha fazla şirket varsa,
  medyandan 3 (handikap) / 6 (toplam) sayı sapan çizgi atlanıyor.
- **Giriş hataları:** bir maçın toplamı 2001.0, bir diğerininki −1.0. Kareler toplamıyla çalışan
  w hesabını tek başlarına domine ediyorlardı (toplamın OOF w'si 0.86 çıkmıştı). Aralık
  kontrolüyle elendi.

Temizlik sonrası her sezonda piyasa hatası makul aralıkta: fark 9.1-10.9, toplam 13.4-15.0 sayı
(lig skoru arttıkça yükseliyor).

## 6. Sızıntısız model

⏳ *`MODEL_EGIT.bat` ve `09_market_eval.py --refresh` sonrası doldurulacak.*

| | Sızıntılı | Sızıntısız |
|---|---|---|
| Kazanan doğruluk / log-loss (holdout) | %69.3 / 0.586 | ⏳ |
| Fark MAE (holdout) | 11.07 | ⏳ |
| Toplam MAE (holdout) | 15.38 | ⏳ |
| Fark w (OOF / holdout) | 0.35 / 0.32 | ⏳ |
| Uygulamadaki model payı (kural aşağıda) | — | ⏳ |

## 7. Uygulamaya etkisi

- **Piyasa merkezli olasılıklar** (`market_blend.py`): piyasa çizgisi varsa handikap merdiveni,
  fark bantları ve toplam baremleri çizgi etrafında, piyasanın kendi hata dağılımıyla kurulur
  (2022-25 sezonlarından; fark σ ≈ 13.4, kde şekli; toplam σ ≈ 18.0, normal; şekil holdout
  CRPS'e göre seçilir). Merkez = piyasa + w · (model − piyasa); w yalnızca OOF ve holdout'ta
  ikisi de 2 standart hatadan büyükse kullanılır, aksi halde 0.
- **Model görünür kalır:** panelde piyasa ve model çizgisi yan yana; 4+ sayı ayrılıkta uyarı.
  Kazanma çubuğu modelin, yanındaki "piyasa" yüzdesi marjsız moneyline.
- **Kuponlar:** piyasa olasılığıyla; her maç en fazla bir kupona; iki modelin uyumu ve "güçlü
  maç" filtreleri kalktı (aynı özelliklerle eğitilmiş iki modelin uyumu güvenilirlik kanıtı değil).
- **Karne:** "Model ↔ Piyasa" sekmesi canlıda iki tarafın hatasını ve canlı w'yi biriktirir.

## 8. Sınırlar

- Canlıda 05 çizgiyi çalıştığı saatte alır; bu kapanıştan erken ve biraz daha ham. Geçmişte
  ölçülen piyasa isabeti canlıdakinin üst sınırıdır.
- Tarihsel sakatlık raporu yok; "maç öncesi bilinen yokluk" tanımı bir yaklaşıklık.
- Hiperparametreler aynı katlarda arandığı için OOF sonuçları hafif iyimser; holdout temiz.
- ESPN API'si resmi değil; biçimi değişebilir (sakatlık raporu için de aynı risk).

## 9. Yeniden üretmek

```powershell
python src/espn_odds.py --backfill --season 2012-13 2013-14 ... 2026-27   # önbellekli, devam edebilir
.\MODEL_EGIT.bat                                   # 02 → 03 (100 deneme) → 04
python src/09_market_eval.py --refresh             # model ↔ piyasa, models/market_distribution.json
python src/09_market_eval.py --no-availability     # oyuncu özellikleri olmadan (alt sınır)
```

Raporlar: `reports/market_eval.json`, `reports/market_eval_noavail.json`; sızıntılı ilk ölçüm
`reports/market_eval_leaky.json` olarak saklandı.
