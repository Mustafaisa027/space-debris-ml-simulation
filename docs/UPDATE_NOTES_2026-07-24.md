# Güncelleme Notları — 2026-07-24

**Dal:** `claude/space-debris-code-analysis-a67c66`
**Bağlam:** IAC 2026 bildirisi 114764 için kod tabanının bilimsel makaleye
hazırlık analizi ve "çalışır hale getirme" turu. Kabul edilen abstract
(`IAC-26,A6,IP,6,x114764`) sözleşme olarak alındı.

---

## 1. Genel durum tespiti (analiz)

Bu tur başlamadan önce sistemin **zaten çalışır durumda** olduğu doğrulandı.
Bu önemli bir bulgu: "sistemi ayağa kaldır" talebine karşılık, kırık bir sistem
değil, **olgun ve titiz** bir sistem bulundu. Somut kanıtlar:

- **Test paketi:** tur öncesi `288 passed`. Testler yüzeysel "çökmedi"
  testleri değil; RIC ayrışımı, kapalı-form CPA, sızıntı-önleme `assert`'leri
  gibi analitik referanslarla karşılaştırıyor.
- **Demo pipeline uçtan uca çalışıyor:** `run_pipeline.py` sentetik
  `demo_tles.txt` üzerinde 60 obje / 77 conjunction üretti, dört sınıflandırıcı
  + sabit-eşik baseline karşılaştırmasını ve tüm bilimsel/fiziksel grafikleri
  hatasız üretti.
- **Dondurulmuş kanıt zinciri kurulu ve test edilmiş:**
  `import_collection_archive → resimulate_snapshots → rebuild_history →
  train_from_history → evidence → plots` zinciri
  `scripts/finalize_iac_experiment.ps1` ile orkestre ediliyor; her aşama
  parmak-izi (fingerprint) checkpoint'iyle bağlı ve ayrı ayrı birim-test
  ediliyor.
- **Bilimsel dürüstlük mimarisi sağlam:** `snapshot_only` (birincil iddia) ile
  `full_rule_recovery` (kontrol) ayrımı, çift-gruplu kronolojik sızıntı-önleme
  bölmesi, fail-closed yayın kapıları hepsi yerinde.

**Sonuç:** Sistemi "çalışır hale getirmek" için yeni bir mimari inşa etmeye
gerek yoktu. Yapılacak iş, (a) makaleye/açık-kaynağa uygunluk için depo
hijyeni, (b) inceleme raporunun işaret ettiği belirli test/dayanıklılık
boşluklarını kapatmak ve (c) 10 günlük veri toplama penceresine hizalı somut
bir yol haritası üretmekti.

**Tek gerçek darboğaz kod değil, zamandır:** birincil (claim-eligible) sonuç,
`2026-07-24T00:17:00Z → 2026-08-03T00:17:00Z` penceresinde 2 saatlik cadence ile
toplanacak **gerçek TLE verisine** bağlı. Bu pencere **bugün** açıldı ve
kısaltılamaz. Kod tarafı bu veriyi işlemeye hazır.

---

## 2. Bu turda yapılan değişiklikler

Hepsi yalnızca yerelde/dalda; hiçbir kişisel bilgi paylaşılmadı, hiçbir uzak
(remote) işlem yapılmadı.

### 2.1 Depo hijyeni — `src/` kökündeki step1..step10 kopyaları silindi
İnceleme raporu bulgusu **3.2**. `src/step1_iss_position.py` …
`src/step10_live_pipeline.py` dosyalarının, `src/legacy/` altındaki belgeli
"yalnızca referans" kopyalarla **bayt-bayt aynı** olduğu doğrulandı ve hiçbir
yerde import edilmedikleri teyit edildi (`grep` ile). Kök kopyalar silindi;
`src/legacy/` altındaki tek, belgeli kopya korundu. README'nin "Project layout"
bölümü zaten yalnızca `legacy/`'yi listeliyor, dokümantasyon tutarlı kaldı.

**Neden:** Açık-kaynak bir makale deposunda, bakımı yapılan `fetch_tles.py`
ile aynı klasörde duran, bilinen hatalı (boş-CSV `IndexError`, binlerce-km
eşikler) belgesiz prototiplerin bulunması, bir hakem/okuyucu için kafa
karıştırıcı ve özensiz görünür.

### 2.2 Pandas 3 `verify_integrity` deprecation'ları giderildi
`evidence.py` içinde 6 yerde kullanılan, pandas 3'te kaldırılması planlanan
`set_index(..., verify_integrity=True)` deseni, aynı fail-closed sözleşmeyi
(çift `source_row_id` = raise) koruyan paylaşılan bir yardımcıya
(`_index_by_unique_row_id`) taşındı.

**Neden:** Test çıktısındaki 39 deprecation uyarısı, yeniden-üretilebilirlik
iddiası olan bir depoyu çalıştıran bir hakem için gürültü ve kırılganlık
sinyalidir. Davranış değişmedi (uniqueness ihlali hâlâ hata fırlatıyor).

### 2.3 `tle_validation.py` için doğrudan birim testleri eklendi
İnceleme raporu bulgusu **4.x**. Fail-closed veri-bütünlüğü hikâyesinin güven
kökü olan bu modülün kendi testi yoktu (yalnızca dolaylı test ediliyordu). Yeni
`tests/test_tle_validation.py` (21 test) şunları doğrudan sınıyor: checksum
doğrulama (geçerli/yanlış-uzunluk/bozuk-hane/eksi-işareti kuralı), yapısal TLE
bloğu doğrulama (üçlü-satır, prefix, 69-karakter, catalog-ID uyumu, çift-ID,
checksum) ve `partition_tle_hash_quality` (hash çeşitliliği, kronolojik run
uzunluğu, bozuk/çelişkili/eksik kayıt reddi, özel `error_cls`).

**Yan bulgu:** `data/sample_tles.txt` içindeki 10 gerçek TLE satırından 9'unun
checksum'ı geçersiz (dosya elle düzenlenmiş). Bu, kritik olmayan, bayat bir
illüstrasyon dosyası — demo yolu `demo_tles.txt` (checksum'ları doğru sentetik
veri), gerçek-veri yolu ise taze `fetch_tles.py` çıktısı kullanır. Yol
haritasında düşük-öncelikli bir madde olarak not edildi.

### 2.4 `run_pipeline` için uçtan uca smoke testi eklendi
İnceleme raporu bulgusu **4.x**. README'nin ilk gösterdiği giriş noktası olan
`run_pipeline.main()` hiçbir testte çalıştırılmıyordu. Yeni
`tests/test_run_pipeline_smoke.py`, gerçek giriş noktasını küçük deterministik
sentetik katalog üzerinde (hem `snapshot_only` hem `full_rule_recovery` ile)
çalıştırıp belgelenen CSV/PNG çıktılarının üretildiğini ve dört zorunlu
sınıflandırıcı + baseline'ın rapora girdiğini doğruluyor.

---

## 3. Doğrulama

```
python -m pytest -q
# 311 passed, 1 warning   (öncesi: 288 passed, 39 warning)
```

- +23 yeni test (21 tle_validation + 2 smoke).
- Kalan 1 uyarı meşru: bir `collect_observations` testinin bilinçli olarak
  sınadığı "bayat TLE epoch'u" `UserWarning`'i.
- Davranış-değiştiren bir düzenleme yok; tüm mevcut testler değişmeden geçti.

---

## 4. Bilinçli olarak yapılmayanlar (kapsam kararı)

- **`evidence.py`'nin alt modüllere bölünmesi** (rapor madde 7): 2000+ satırlık
  "god-file" refactor'ü yüksek blast-radius / düşük aciliyet. Bildirinin
  istatistiksel çekirdeğine dokunur ve makaleden 10 gün önce risk/fayda oranı
  kötü. Ertelendi.
- **`ml.py` kapı-mantığı tekrarının birleştirilmesi** (rapor madde 7): aynı
  gerekçe.
- **Herhangi bir uzak/remote işlem:** GitHub Actions, push, PR — hiçbiri
  yapılmadı. Bunlar kullanıcının kontrolünde (bkz. yol haritası).

Bu iki refactor sistem çalışmasını engellemiyor; yalnızca bakım
ergonomisiyle ilgili ve makale sonrası temiz bir PR'a bırakmak daha doğru.
