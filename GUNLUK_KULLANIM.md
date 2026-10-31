# CourtVision — Günlük Kullanım ve Veri Toplama Planı

Hazırlanma: 31 Ekim 2026 · Branch: `faz4-model`

Amaç: Modeli değiştirmeden birkaç hafta boyunca her gün tahmin üretmek, sonuçların
otomatik dolduğunu doğrulamak ve canlı isabet oranını ölçmek.

**Kısaca:** Günde üç kez `CourtVision.bat`'a çift tıkla. Sabah açılışta her şey
otomatik; öğlen ve akşam **"Veriyi Güncelle"**ye bas. Dünün tahminleri ertesi gün
kendiliğinden kontrol edilir; sonuçlar **Geçmiş** sayfasında "Doğru/Yanlış" olarak görünür.

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

**Referans (2025-26 holdout sezonu, 1225 maç):**

| Ölçüt | Değer |
|---|---|
| Doğruluk | %69.5 |
| Log-loss | 0.587 |
| Brier | 0.201 |
| Hep ev sahibi | %55.4 |
| Yalnız Elo ile | %66.5 doğruluk, 0.614 log-loss |

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

- `03_train_model.py` / `04_evaluate_model.py` çalıştırma (model değişir, ölçüm karışır).
- `06_update_data.py` veya `01`/`02`'yi çalıştırma. Günlük veri güncellemesini uygulama kendisi yapar.
- Branch değiştirme, `git pull` yapma. `faz4-model`'de kal.
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

- **A — Toparlama:** `faz4-model`'i main'e merge et, eski branch'leri ve şablon
  dosyalarını temizle, README'yi yeniden yaz, playoff uyarısı ekle, ilk maçtan önce
  otomatik güncelleme ekle.
- **B — Model:** Birden çok sezonda güven aralıklı değerlendirme; daha sade model
  (lojistik regresyon, 215 yerine 20–30 feature ile holdout'ta XGBoost kadar iyi);
  son sezonlara ağırlık; bahis oranlarıyla kıyas.

Canlı sonuçlar bu kararlar için ilk gerçek veri olacak.
