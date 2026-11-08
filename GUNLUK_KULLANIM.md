# CourtVision — Günlük Kullanım ve Veri Toplama Planı

Hazırlanma: 31 Ekim 2026 · Güncelleme: 8 Kasım 2026 (Faz 6b: piyasa çizgileri, piyasa merkezli olasılıklar)

Amaç: Modeli değiştirmeden birkaç hafta boyunca her gün tahmin üretmek, sonuçların
otomatik dolduğunu doğrulamak ve canlı isabet oranını ölçmek.

**Kısaca:** Günde üç kez `CourtVision.bat`'a çift tıkla. Sabah açılışta her şey
otomatik; öğlen ve akşam **"Veriyi Güncelle"**ye bas. Dünün tahminleri ertesi gün
kendiliğinden kontrol edilir; sonuçlar **Geçmiş** sayfasında "Doğru/Yanlış" olarak görünür.
Handikap, toplam ve skor aralıkları maç detay panelinde; okuma rehberi uygulamadaki
**Nasıl Kullanılır** sayfasında ve aşağıda 11. bölümde.

---

## 1. Uygulamayı açmak

**Kolay yol:** Proje klasöründeki `CourtVision.bat` dosyasına çift tıkla. İstersen
sağ tık → *Gönder → Masaüstü (kısayol oluştur)* ile masaüstüne kısayol koy.

**Terminalden:**

```powershell
cd C:\Users\cbura\Desktop\courtvision\electron-app
npm run dev
```

- Venv'i ayrıca etkinleştirmen gerekmez: Electron, API'yi projedeki `.venv` Python'u ile kendisi başlatır.
- Açılan **siyah pencereyi kapatma.** API ve tahmin logları orada `[uvicorn]` önekiyle görünür.
- Uygulama penceresini kapatınca API de kapanır. Zamanlayıcı yalnızca uygulama açıkken çalışır.
- İlk açılışta proje klasörü sorulursa `C:\Users\cbura\Desktop\courtvision` klasörünü seç.

---

## 2. Saatler (Türkiye saati)

NBA takvimi ABD Doğu saatine göre işler. Uygulama da "bugün"ü ona göre belirler.

| Olay | 1 Kasım 2026 – 14 Mart 2027 | 14 Mart 2027'den sonra |
|---|---|---|
| NBA günü değişir (yeni maç listesi) | **08:00** | **07:00** |
| Akşam maçları başlar (ABD 19:00–22:30) | 03:00 – 06:30 | 02:00 – 05:30 |
| Hafta sonu / tatil gündüz maçları (ABD 12:00–15:30) | 20:00 – 23:30 | 19:00 – 22:30 |
| En geç biten maçlar | ~09:00 | ~08:00 |

Yani Türkiye'de "bugünün maçları" gece yarısından sonra oynanır. Sabah 08:00'den sonra
uygulama, o akşam (ABD saatiyle) oynanacak yeni maçları gösterir.

---

## 3. Günlük rutin

| Ne zaman | Ne yap | Ne olur |
|---|---|---|
| **Sabah** (08:00'den sonra, ideali 10:00+) | Uygulamayı aç. Bir şeye basma, 1–3 dakika bekle. | Yeni günün tahminleri otomatik üretilir; dünün sonuçları doldurulur. |
| **Öğlen** | Aç (ya da açıksa olduğu gibi bırak) → **Veriyi Güncelle** | Güncel sakatlık raporuyla tahminler yenilenir; eksik sonuçlar doldurulur. |
| **Akşam** (22:00 – 02:30, maçlardan önce) | **Veriyi Güncelle** → bitince kapatabilirsin | **En önemli güncelleme.** Kayda geçen tahmin, maç başlamadan önceki son güncellemedir. |

Ek olarak:
- **Sabahtan beri açık bırakılan uygulama** aynı gün tahminleri kendiliğinden yenilemez. Öğlen ve akşam butona basmak bu yüzden gerekli: sakatlık raporu gün içinde değişiyor.
- **Uygulama gece açık kalırsa** NBA günü değiştiğinde (08:00) 15 dakika içinde yeni tahminleri kendisi üretir.
- **Bir gün hiç açmazsan** o günün maçları için tahmin kaydı oluşmaz, geriye dönük de üretilemez. Önceki günlerin sonuç kontrolü etkilenmez.

### Tahmin kaydı nasıl oluşur?
- Her "Veriyi Güncelle"de **henüz başlamamış** maçların tahmini yenisiyle değiştirilir.
- **Başlamış veya bitmiş maçların** tahmini kilitlenir, bir daha değişmez. Ölçüm dürüst kalır: maç sırasında öğrenilen bilgi tahmine giremez.
- Örnek: Akşam 22:00'de güncellediğinde 20:00'de başlamış bir hafta sonu maçı varsa, onun kayıtlı tahmini öğlenki güncellemede yapılmış olandır.

---

## 4. Ertesi gün tahminler otomatik kontrol edilir mi?

**Evet.** Elle bir şey yapman gerekmez.

- **Her tahmin çalışması** (yeni günün ilk açılışı, zamanlayıcı ve her "Veriyi Güncelle")
  sezonun maç sonuçlarını NBA'den yeniden çeker. Ardından kayıttaki sonuçsuz tahminleri
  bu sonuçlarla eşleştirip **Doğru/Yanlış** olarak işaretler.
- **Her açılışta** ayrıca son 7 günün sonuçlarına doğrudan NBA'den bakılır. Bu, aynı
  gün içinde uygulamayı yeniden açtığında da sonuçların dolmasını sağlar.

Bilmen gerekenler:
- **Sabah çok erken açarsan** (09:00'dan önce) gecenin son maçları henüz bitmemiş ya da NBA verisine düşmemiş olabilir. Bunlar öğlen güncellemesinde dolar.
- **Birkaç gün ara verirsen** sonuçlar kaybolmaz; bir sonraki açılışta hepsi doldurulur.
- **Sonuçları görmek için:** Sol menü → **Geçmiş**. Her maç kartında Doğru/Yanlış yazar. Gün başlıklarında "x/y doğru" görünür.
- **Dikkat:** Geçmiş sayfasındaki genel yüzde *ağırlıklı* doğruluktur: YÜKSEK güvenli tahminler 3 kat, ORTA 2 kat sayılır. Gerçek isabet oranı için 6. bölümdeki rapor komutunu kullan.

---

## 5. Siyah pencerede neye bakmalı (sağlık kontrolü)

Her tahmin çalışmasında şunlara benzer satırlar görmelisin:

```
NBA Günlük Tahmin — 2026-11-02
Handikap/toplam: margin, total | Kazanan olasılığı: sınıflandırıcı   ← "model yok" yazmamalı
  → Master güncellendi: ... maç (son: 2026-11-01)      ← dünün tarihi olmalı
  -> Found 9 games (9 not started)
  Sakatlık raporu: 18 oyuncu eksik olarak eşleşti
  Eşleşmeyen isimler (...): ...                        ← normal, bkz. aşağı
[tracker] 9 tahmin kaydedildi
[tracker] 7 maçın sonucu dolduruldu.
[API] Predictions updated.
```

- **"Eşleşmeyen isimler" normaldir.** Bunlar ya NBA'de son 400 günde hiç maçı olmayan oyuncular ya da yeni takımında henüz oynamamış oyuncular. Tahmini etkilemezler.
- **`[UYARI] Sakatlık raporu alınamadı`** çıkarsa ESPN'e ulaşılamamış demektir. Tahminler oyuncu eksikliği bilinmeden üretilir. Bir süre sonra tekrar güncelle.
- **`[UYARI] ... çekilemedi`** ya da zaman aşımı hatası çıkarsa stats.nba.com yavaştır veya geçici olarak engelliyordur. Birkaç dakika sonra tekrar "Veriyi Güncelle"ye bas.
- **`[API] predict_today error code: 1`** çıkarsa tahmin çalışması çöktü demektir. Pencerenin üstündeki Traceback'i kopyala ve sakla.

---

## 6. Haftalık kontrol (isteğe bağlı, 1 dakika)

Haftada bir, uygulama kapalıyken ya da açıkken:

```powershell
cd C:\Users\cbura\Desktop\courtvision
.\.venv\Scripts\Activate.ps1
python src/prediction_tracker.py
```

Çıktı örneği:

```
  Sonuçlanan maç      : 84  (sonuç bekleyen: 9)
  Tarih aralığı       : 2026-10-31 → 2026-11-12
  Doğruluk            : %67.9  (%95 aralık: %57.9 – %77.9)
  Hep ev sahibi dese  : %56.0
  Log-loss / Brier    : 0.6012 / 0.2080
  Güven seviyesine göre: ...
  Günlük: ...
```

Kontrol listesi:
- **Her gün için bir satır olmalı.** Açtığın her günün sonuçlanmış olması gerekir.
- **"Sonuç bekleyen" sayısı birkaç günde birikmemeli.** Normalde yalnızca bugünün maçları bekler.

---

## 7. Birkaç hafta sonra değerlendirme

**Referans (2025-26 holdout sezonu, 1225 maç; 8 Kasım 2026 eğitimi, 100 deneme):**

| Ölçüt | Değer |
|---|---|
| Doğruluk | %69.3 |
| Log-loss | 0.586 |
| Brier | 0.200 |
| Hep ev sahibi | %55.4 |
| Yalnız Elo ile | %66.5 doğruluk, 0.614 log-loss |
| Fark (handikap) ortalama hatası | 11.1 sayı (Elo tabanı 11.6) |
| Toplam sayı ortalama hatası | 15.4 sayı (son 10 maç tabanı 15.35, tahmin gücü düşük) |

Handikap ve toplam için canlı karşılaştırma: Geçmiş sayfasındaki ortalama hata kartları ya da
`python src/prediction_tracker.py` çıktısının sonundaki "Fark (handikap)" ve "Toplam sayı" bölümleri.
Canlıda handikap ortalama hatası 11 civarında, toplamınki 15 civarında kalıyorsa model beklendiği gibi.

**Belirsizlik:** Birkaç haftalık veri sadece kaba bir resim verir. %69 civarı bir
doğruluk için %95 aralığın genişliği:

| Sonuçlanan maç | Yaklaşık süre | Aralık |
|---|---|---|
| 100 | ~2 hafta | ±9 puan |
| 200 | ~4 hafta | ±6.5 puan |
| 300 | ~6 hafta | ±5 puan |
| 500 | ~10 hafta | ±4 puan |

**Nasıl yorumlanır:**
- **Program düzgün çalışıyor mu?** Her gün kayıt oluşmuş, sonuçlar ertesi gün dolmuş ve siyah pencerede hata yoksa evet. Bu birkaç haftada kesin olarak cevaplanabilir.
- **Model beklendiği gibi mi?** Canlı doğruluk aralığı %69.5'i kapsıyorsa model holdout'taki performansıyla tutarlıdır. Log-loss 0.60–0.61 civarındaysa iyi sayılır.
- **Canlıda biraz daha düşük sonuç beklenir.** Eğitimde kimin oynamadığı kesin biliniyor; canlıda yalnızca sabah/akşam sakatlık raporu var.
- **Ciddi sorun işareti:** Doğruluk 300+ maçta %62'nin altında, ya da "Hep ev sahibi dese" değerine yakın çıkarsa. Ayrıca log-loss 0.65'in üstündeyse.
- **Sezonun ilk haftalarında model biraz daha zayıf olabilir.** Sezon istatistikleri az maça dayanıyor.

---

## 8. Veri toplama süresince YAPMA

Ölçümün temiz kalması için modeli ve kodu sabit tut:

- `03_train_model.py` / `04_evaluate_model.py` / `MODEL_EGIT.bat` çalıştırma (model değişir, ölçüm karışır).
  Yeniden eğitim için 12. bölüme bak; ölçüm döneminin sonunda yap.
- `06_update_data.py` veya `01`/`02`'yi çalıştırma. Günlük veri güncellemesini uygulama kendisi yapar.
- Branch değiştirme, `git pull` yapma.
- `data/predictions/predictions_log.parquet` dosyasını silme veya taşıma. Bütün canlı kayıt bu dosyada.
  İstersen haftada bir başka bir yere kopyala.

---

## 9. Sorun giderme

**"API zaten çalışıyor" yazıyor ama ekran boş, ya da port meşgul.** Eski bir API
süreci açık kalmış olabilir:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen | Select-Object OwningProcess
Stop-Process -Id <yukarıdaki numara>
```

Sonra uygulamayı yeniden aç.

**Geçmiş sayfasında dünün maçları hâlâ sonuçsuz.** Öğlen "Veriyi Güncelle"ye bas.
Olmazsa 6. bölümdeki komutu çalıştır; o da son 60 günü NBA'den kontrol eder.

**Ekranda "Tahminler güncelleniyor" uzun süre kalıyor.** Siyah pencereye bak. nba_api
bazen 1–2 dakika bekletir. 5 dakikayı geçerse uygulamayı kapatıp yeniden aç.

**Playoff dönemi (Nisan 2027 ortasından itibaren).** Model ve veri çekme yalnızca
normal sezon için kurulu. Playoff tahminleri ve sonuç kontrolü güvenilir olmayabilir.
Veri toplamayı normal sezon sonunda bitirmek en temizi.

---

## 10. Ara bittiğinde

Önce 6. bölümdeki raporu çalıştırıp sonuçlara bak. Ardından 31 Ekim değerlendirmesindeki
plana dön:

- **A — Toparlama:** eski branch'leri ve şablon dosyalarını temizle, README'yi yeniden
  yaz, playoff uyarısı ekle, ilk maçtan önce otomatik güncelleme ekle.
- **B — Model:** Birden çok sezonda güven aralıklı değerlendirme; daha sade model
  (lojistik regresyon, 215 yerine 20–30 feature ile holdout'ta XGBoost kadar iyi);
  son sezonlara ağırlık; bahis oranlarıyla kıyas.

Canlı sonuçlar bu kararlar için ilk gerçek veri olacak.

---

## 11. Handikap, toplam ve skor aralıkları nasıl okunur

Bir maça tıkla; sağ paneldeki **Handikap & Toplam** bölümü:

- **Piyasa / Model kutusu:** üstte DraftKings'in maç öncesi çizgisi (ESPN'den: handikap,
  toplam, marjsız ev kazanma olasılığı), altında modelin kendi adil çizgisi. Model piyasadan
  4+ sayı ayrılırsa sarı uyarı çıkar; geçmişte bu ayrılıklar çoğunlukla modelin hatasıydı.
- **Olasılıklar piyasa merkezli:** merdiven, bantlar ve toplam baremleri piyasa çizgisi
  etrafında, piyasanın son sezonlardaki hata dağılımıyla kurulur (`models/market_distribution.json`).
  Piyasa çizgisi yoksa (ESPN vermediyse) modelin dağılımı kullanılır ve kutu görünmez.
- **Adil çizgi:** iki tarafa da %50 şans veren barem. **Adil oran = 1 / olasılık**
  (%62'lik seçeneğin adil oranı 1.61).
- **Handikap merdiveni:** favori tarafın seçenekleri en güvenliden en riskliye sıralı.
  "Diğer tarafı göster" ile karşı takımın seçenekleri.
- **Kazanma farkı bantları:** 1-5, 6-10, 11-15, 16-20, 21-25, 26+ için olasılık ve adil oran.
- **Toplam sayı:** her barem için üst / alt olasılığı. "Tahmin gücü düşük" etiketi yalnız
  piyasa çizgisi olmayan maçlarda çıkar (toplam modeli basit ortalamayı geçemiyor).
- **Tahmini skor:** her takım için en olası 10 sayılık aralık. Bir takımın skoru ±11-12 sayı
  oynadığı için bu aralık bile en fazla ~%33 ihtimal taşır.

Maç kartındaki çubuk **modelin** kazanma olasılığı; yanındaki gri "piyasa" yüzdesi kapanış
oranlarından marjı çıkarılmış olasılık. Neden ikisi ayrı: bkz. `EVALUATION.md`.

---

## 12. Modeli yeniden eğitmek

Ölçüm dönemlerinin arasında, sezonda ayda bir yeterli.

1. Uygulamayı kapat.
2. Proje klasöründeki **`MODEL_EGIT.bat`**'a çift tıkla (yaklaşık 2 saat). Sırasıyla özellik
   tablosunu kurar, kazanan + fark + toplam modellerini 100 denemeyle eğitir ve değerlendirir.
3. Bitince çıktının sonundaki holdout sonuçlarına bak (kazanan doğruluğu, fark/toplam ortalama
   hatası, "Kazanan olasılığı: sınıflandırıcı mı, fark modeli mi?").
4. Piyasa kıyasını ve piyasa dağılımını yenile (~10 dk): `python src/09_market_eval.py --refresh`.
   Çıktının sonunda "model payı w" yazar; 0 ise uygulama tamamen piyasa çizgisiyle fiyatlar.
5. Uygulamayı yeniden aç.

Elle çalıştırmak istersen aynı adımlar:

```powershell
cd C:\Users\cbura\Desktop\courtvision
.\.venv\Scripts\Activate.ps1
python src/02_build_features.py
python src/03_train_model.py --task all --trials 100
python src/04_evaluate_model.py
python src/09_market_eval.py --refresh
```

**Geçmiş piyasa çizgileri** (`data/processed/market_lines.parquet`) bir kez indirildi
(2012-13'ten bu yana ~16.900 maç). Günlük çizgileri 05 kendisi ekler. Eksik bir sezonu tamamlamak
gerekirse (yarıda kesilirse kaldığı yerden devam eder, birkaç terminalde paralel çalışabilir):

```powershell
python src/espn_odds.py --backfill --season 2025-26 2026-27
```

---

## 13. Kuponlar ve karne

**Kuponlar sayfası** her tahmin güncellemesinde yeniden kurulur:

- Üç handikap kuponu: **Güvenli** (ayak başı %68-80), **Dengeli** (%58-68), **Riskli** (%50-58).
  Favori her zaman **handikap verir** (-1.5 ve üstü; artı handikap yok). Hafif favoriler bu yüzden
  güvenli kupona giremez; o gün güvenli seviyeye giren 2 maç yoksa güvenli kupon kurulmaz.
- **Olasılıklar piyasa merkezli** (11. bölüm). Kupona piyasa çizgisi olan, başlamamış maçlar girer.
  O gün hiç piyasa çizgisi yoksa kuponlar modelden kurulur ve sayfada bunu söyleyen bir not çıkar.
- **Her maç en fazla bir kupona girer**, böylece tek bir sürpriz bütün kuponları yakmaz. Güvenli kupon
  büyük favorileri, dengeli ortadakileri, riskli kalanları alır; ayak sayısı uygun maç sayısına göre
  2-3. Kilitli bir kuponun maçları yeniden kurulan kuponlara girmez.
- Eski **Üst (deneysel)** kuponu kaldırıldı: toplam modelinin tarafı piyasaya karşı %47 tuttu. Eski
  üst kuponları kayıtta kalır ve sonuçlanmaya devam eder.
- Sayfanın altındaki "Maç seçimi" tablosu her maç için piyasa ve model çizgisini yan yana yazar.
- **Kilit:** kupon, içindeki bir maç başlayınca (ya da bir oran kaydedilince) kilitlenir; karneye o
  haliyle geçer. Kilitlenmemiş kuponlar sonraki güncellemede değişebilir (sakatlık haberi).

**Karne sayfası** "model ne vaat etti, ne gerçekleşti" sorusunu cevaplar:

- **Kuponlar:** seviye başına tutma oranı (beklenenle), ayakların tutma oranı, oynananların kârı.
- **Handikap:** her maçın 9 baremi favori tarafından okunur; "model %70-75 dediğinde ne kadar tuttu?"
  tablosu ve kazanma farkı bantları. Mavi çizgi (vaat) gri şeridin (%95 aralık) içindeyse model
  o seviyede dürüst.
- **Taraf** ve **Alt / Üst:** aynı tablo; alt/üst sadece takip içindir.
- **Model ↔ Piyasa:** piyasa çizgisi olan bitmiş maçlarda modelin ve piyasanın ortalama hatası,
  modelin kaç maçta gerçeğe daha yakın olduğu, kazananda log-loss ve isabet. 30 maçtan sonra canlı
  **w** (gerçek ≈ piyasa + w × (model − piyasa)) görünür; birkaç yüz maç olmadan ±0.1'den dar olmaz.
- Kupon tutma oranı küçük örneklerde çok oynar; modelin dürüstlüğünü barem ve ayak kalibrasyonundan oku.

Kayıtlar: `data/predictions/coupons_log.json` (silme; tahmin kaydı gibi haftada bir yedekle).
