# CourtVision — Değerlendirme: Model ve Piyasa

Kasım 2026. Bu rapor modelin ne kadar iyi olduğunu ve bahis piyasasının kapanış çizgisine
karşı ne kattığını, sızıntısız bir düzenekle ölçer. Kısa özet:

- Sızıntısız model, halka açık kutu skoru verileriyle **Elo tabanını 0.3 sayı geçiyor, piyasanın
  kapanış çizgisinin 0.4 sayı gerisinde kalıyor** (fark hatası 11.27-11.31; Elo 11.61, piyasa 10.89)
  ve piyasanın bilmediği bir şey bilmiyor (holdout w = 0.00 ± 0.11).
- İlk ölçümde görülen avantaj (w ≈ 0.3) **geriye dönük bilgi sızıntısıydı**: eğitimde "eksik
  oyuncu", maçın kendi kutu skorundan okunuyordu. Düzeltilince w sıfıra indi.
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

Sızıntısız özelliklerle yeniden eğitilmiş modeller (100 Optuna denemesi):

| Model | Ölçüt | Model | Basit taban |
|---|---|---|---|
| Kazanan | doğruluk / log-loss | %69.0 / 0.592 (fark modelinden)¹ | hep ev sahibi %55.4 |
| Fark (ev − dep.) | ortalama hata (MAE) | 11.31 sayı (düzeltmeyle 11.27) | Elo farkı 11.61 |
| Toplam | ortalama hata | 15.41 sayı | son-10 ortalaması 15.35 |

¹ Sınıflandırıcı %68.6 / 0.596; fark modelinin P(fark > 0) olasılığı 0.001'den fazla daha iyi
olduğu için uygulama kazananı artık fark modelinden alıyor.

Fark modeli tahminleri ortalamaya sıkıştırıyor (gerçek ≈ 1.17 × tahmin); hata dağılımı bu yüzden
düzeltilmiş ortalama + ampirik şekille kuruldu, 12 aday arasından holdout CRPS'e göre seçildi
(±8 sayılık barem merdiveninde tahmin ile gerçekleşen arası en fazla 2.4 puan). Toplam modeli
tabanı geçemiyor: sezon içi toplam dalgalanmasının yalnızca ~%6'sı maç öncesi veriden
açıklanabiliyor (`08_line_experiment.py`'de tempo, verim, lig trendi denendi).

## 3. Piyasa ile kıyas

Kapanış çizgisi (2025-26 holdout, aynı 1.224 maç):

| | Piyasa | Model (sızıntısız) |
|---|---|---|
| Fark MAE | **10.89** | 11.27 – 11.32 |
| Toplam MAE | **14.60** | 15.41 |
| Kazanan log-loss / doğruluk | **0.568** / %69.2 | 0.592 / %69.0 |

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
| Sızıntısız (önceden bilinen yokluk, 6. bölüm) | 0.12 ± 0.03 | 0.00 ± 0.11 | %50.5 |

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

Düzeltmeden sonra yeniden eğitim (`MODEL_EGIT.bat`, 9 Kasım 2026) ve `09_market_eval.py --refresh`:

| | Sızıntılı | Sızıntısız |
|---|---|---|
| Kazanan doğruluk / log-loss (holdout) | %69.3 / 0.586 | %69.0 / 0.592 (sınıflandırıcı %68.6 / 0.596) |
| Fark MAE (holdout) | 11.07 | 11.31 (düzeltmeyle 11.27) |
| Toplam MAE (holdout) | 15.38 | 15.41 |
| Fark w, OOF | 0.35 ± 0.03 | 0.12 ± 0.03 |
| Fark w, holdout | 0.32 ± 0.11 | **0.00 ± 0.11** |
| Modelin tarafı, holdout (başabaş %52.4) | %52.9 | %50.5 |
| Kazanan harman katsayısı b, OOF | 0.11 ± 0.05 | 0.02 ± 0.05 |
| Uygulamadaki model payı | — | **0** (holdout w anlamlı değil) |

- **Sızıntının bedeli:** fark hatasında ~0.25 sayı, kazanan log-loss'unda ~0.006. İlk rakamlar
  bu kadar iyimserdi; dürüst model hâlâ Elo tabanını geçiyor (11.31 / 11.61).
- **OOF'taki 0.12 eski sezonlardan geliyor:** sezona göre w 2012-13'te 0.52 (consensus çizgisi,
  zayıf), 2013-20 arası 0.1-0.2, 2020 sonrası ≈ 0 (DraftKings). Güncel kapanış çizgileri modelin
  söyleyebileceği her şeyi zaten içeriyor.
- **Uygulamanın piyasa dağılımı** (`models/market_distribution.json`): fark σ 13.39 (kde),
  toplam σ 17.98 (normal); üç şeklin CRPS'i neredeyse eşit. İkisinde de model payı 0.

## 7. Uygulamaya etkisi

- **Piyasa merkezli olasılıklar** (`market_blend.py`): piyasa çizgisi varsa handikap merdiveni,
  fark bantları ve toplam baremleri çizgi etrafında, piyasanın kendi hata dağılımıyla kurulur
  (2022-25 sezonlarından, holdout dahil yeniden öğrenilir; şekil holdout CRPS'e göre seçilir).
  Merkez = piyasa + w · (model − piyasa); w yalnızca OOF ve holdout'ta ikisi de 2 standart
  hatadan büyükse kullanılır, aksi halde 0 (şu an 0).
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
