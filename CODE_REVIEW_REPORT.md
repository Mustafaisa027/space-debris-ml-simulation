# Space Debris Projesi — Kod Tabanı İnceleme Raporu

**Depo:** space-debris-project · **Dal:** codex/complete-iac-evidence · **Tarih:** 21 Temmuz 2026

**Kapsam:** ~48 Python dosyası, ~17.000 satır kod (paket + scriptler + testler). Üç paralel odaklı analize dayanır: (1) çekirdek bilimsel paket, (2) veri toplama/pipeline scriptleri ve CI, (3) test paketi ve proje yapısı.

---

## 1. Genel Değerlendirme

Bu proje, bir konferans bildirisine (IAC 2026, bildiri 114764) eşlik eden araştırma kodu için **alışılmadık derecede titiz** bir mühendislik disiplinine sahip: SHA-256 tabanlı sağlama toplamları, çift-bazlı kronolojik (leakage-proof) veri bölme, dondurulmuş değerlendirme kohortları, bootstrap güven aralıkları ve "fail-closed" doğrulama kapıları hepsi gerçekten var ve büyük ölçüde iddia edildiği gibi çalışıyor. Test paketi (~5.600 satır) sığ "çökmedi" testleri değil, analitik olarak türetilmiş referans değerlerle (RIC ayrışımı, kapalı-form CPA) karşılaştırma yapıyor.

Buna karşın üç paralel incelemede **gerçek ve somut sorunlar** bulundu — en önemlisi, README'nin "gerçek veriyle çalıştır" olarak belgelediği hızlı-başlangıç komutunun, projenin kendi iddia ettiği sızıntı-önleme garantisini fiilen ihlal etmesi. Ayrıca "atomik yazma" ve "%90 kapsam" gibi genel-geçer ifade edilen bazı garantiler kod tabanında tutarsız biçimde uygulanmış durumda.

---

## 2. Güçlü Yönler

- **Sızıntı-önleme mimarisi sağlam:** `ml.py:555-618` içindeki çift-gruplu kronolojik bölme, eğitim/test çiftleri ve zaman damgaları arasında kesişim olmadığını açıkça `assert` ile doğruluyor.
- **İstatistiksel çekirdek matematiksel olarak doğru:** `evidence.py` içindeki bootstrap, kalibre edilmiş mesafe karşılaştırması ve kapalı-form CPA (en yakın yaklaşım) formülü elle türetilen referans değerlerle örtüşüyor.
- **Dondurulmuş protokol gerçekten dondurulmuş:** `experiment.py`'deki varsayılan eşikler (destek minimumları, blok sayıları, bootstrap çekilim sayısı, sabit tohum 114764) hem README hem `docs/iac_gap_analysis.md` ile birebir örtüşüyor.
- **Test paketi iddiaları gerçekten sınıyor:** Örn. `test_experiment.py:495-618`, sızıntı kontrolü veya permütasyon kontrolü şüpheli göründüğünde yayın kapısının bilinçli olarak *başarısız* olduğunu doğruluyor.
- **Provenance/hash zinciri kayıtlara güvenmiyor, yeniden hesaplıyor:** `train_from_history.py:767-944` arşiv→resimülasyon→geçmiş bağını her seferinde yeniden özetleyerek doğruluyor.

---

## 3. Yüksek Öncelikli Bulgular

### 3.1 Belgelenen "gerçek veriyle çalıştır" komutu etiket sızıntısı üretiyor — *Bilimsel Bütünlük*

`run_pipeline.py:127`, `compare_models()`'ı `feature_columns` vermeden çağırıyor; bu da varsayılan olarak `ml.py:45-59`'daki tam 13 özellikli sete düşüyor — ki bu set `min_distance_km` ve `relative_velocity_km_s`'yi de içeriyor. Bu iki değer `core.py:209-237`'deki `risk_label`'ın tam olarak girdisidir.

README'nin "Scientific runs (real data)" bölümü (satır 57-69) kullanıcıyı doğrudan bu komuta yönlendiriyor; yani belgelenen yol, projenin "sızıntı yok" garantisinin (README satır 242) tersini üretiyor ve hiçbir uyarı vermiyor.

**Neden önemli:** Bir kullanıcı README'yi harfiyen izleyip gerçek Starlink verisiyle çalıştırırsa, modeller etiketi trivially yeniden inşa ettiği için yapay olarak kusursuz görünür — asıl titiz protokol (`train_from_history.py`) ayrı bir giriş noktasında yaşıyor ve bu fark hiçbir yerde açıklanmıyor.

### 3.2 Eski prototip betiklerinin belgesiz, birebir kopyası src/ kökünde duruyor — *Proje Yapısı*

`src/step1_iss_position.py` … `step10_live_pipeline.py`, `src/legacy/` altındaki "yalnızca referans, bakımı yapılmıyor" diye işaretlenmiş dosyalarla bayt-bayt aynı. Git geçmişi, `legacy/` dizini oluşturulurken kök dizindeki kopyaların silinmediğini gösteriyor. README'nin "Project layout" bölümü (satır 260-276) yalnızca `legacy/`'yi listeliyor.

**Neden önemli:** `src/*.py`'yi tarayan bir katkıcı, bakımı yapılan `fetch_tles.py` ile aynı klasördeki `step7_add_risk_score.py`'yi ayırt edemez ve `legacy/README.md`'nin açıkça uyardığı bilinen hatalara (boş CSV'de `IndexError`, yalnızca-Windows yollar, binlerce km eşikler) sessizce maruz kalır.

### 3.3 "Atomik yazma" iddiası kod tabanında tutarsız uygulanmış — *Veri Bütünlüğü*

README (satır 161) ve `docs/GITHUB_DATA_COLLECTION.md` (satır 64) yazmaların atomik olduğunu genel-geçer ifade ediyor, ve bu örüntü (`tempfile` + `fsync` + `os.replace`) gerçekten `fetch_tles.py:377-398`, `collect_observations.py:148-228`, `import_collection_archive.py:13-27` ve `train_from_history.py:138-152`'de özenle uygulanmış. Ama şu dosyalar düz `Path.write_text()` kullanıyor:

| Dosya | Neden kritik |
|---|---|
| `build_collection_manifest.py:75` | Arşiv bütünlüğünün güven kökü olan `manifest.json`'un kendisi |
| `resimulate_snapshots.py:329` | `train_from_history.py`'nin sonradan hash'ine güvendiği resimülasyon raporu |
| `rebuild_history.py:295` | Geçmiş-yeniden-inşa raporu |
| `provenance.py:93-106` | Tüm `model_comparison*.csv` sonuç tabloları |

**Neden önemli:** Kod tabanı doğru örüntüyü zaten biliyor (`resimulate_snapshots.py:208-211` geçici dizin + `os.replace` ile doğru yapıyor) — sorun bilgisizlik değil, tutarsız uygulama.

### 3.4 Fail-closed doğrulama mantığı iki yerde bağımsız olarak kopyalanmış — *Bakım Riski*

TLE sağlama toplamı doğrulaması hem `fetch_tles.py:235-256`'da hem `space_debris/tle_validation.py:6-43`'te neredeyse birebir yeniden yazılmış. Benzer şekilde `_partition_tle_hash_quality`, `ml.py:453-500` ve `evidence.py:70-121`'de ayrı ayrı tanımlı.

**Neden önemli:** İleride bir uç durum düzeltmesi (örn. satır uzunluğu kontrolü) yalnızca bir kopyaya uygulanırsa, "fail-closed" garantisi kullanılan koda göre farklılaşır.

### 3.5 %90 kapsam eşiği yalnızca --group ile çağrılan alımlarda uygulanmıyor — *Veri Bütünlüğü*

`fetch_tles.py:329-374`'te kapsam kontrolü `if requested_unique:` koşuluyla sarılı; yalnızca `--group` (CATNR olmadan) kullanıldığında `requested_unique == 0` olur ve tüm kapsam-oranı bloğu (352-361) atlanır. Dondurulmuş IAC protokolü her zaman `leo_mixed` CATNR ön ayarını kullandığı için asıl bildiri sonucunu etkilemiyor, ama README'nin koşulsuz "%90 altı reddedilir" ifadesi (satır 161), belgelenen ve tamamen desteklenen `--group` kullanımı için doğru değil.

### 3.6 "Açık kaynak" olarak tanımlanan projede LICENSE dosyası yok — *Yasal / Yayın*

README ve `pyproject.toml` projeyi tekrar tekrar "open-source" olarak tanımlıyor ama depo kökünde `LICENSE` yok. Projenin kendi `docs/iac_gap_analysis.md:101-102` dosyası bunu zaten bilinen, henüz kapatılmamış bir eksiklik olarak listeliyor.

---

## 4. Orta Öncelikli Bulgular

- **`evidence.py` 2.230 satırlık bir "god-file":** Tek bir fonksiyon (`paired_pair_cluster_bootstrap`, satır 392-853) ~460 satır; `_generate_evaluation_evidence_files` (satır 1463-2230) ~770 satır ve veri doğrulama, bölme, eşik kalibrasyonu, 5 farklı kontrol kolu ve bootstrap'ı tek fonksiyonda birleştiriyor.
- **`ml.py`'de `compare_models` / `time_series_cv_report` arasında ~300 satırlık kapı mantığı tekrarı:** İki fonksiyon (`ml.py:977-1274`, `ml.py:1276-1563`) pencere-kalite, TLE-yaş ve destek kapılarını neredeyse satır satır tekrarlıyor.
- **CPA kontrol kolu, mesafe-karşılaştırma kolunun aksine zorunlu değil:** `evidence.py:465-468`, `claim_eligible=True` iken kalibre-edilmiş mesafe karşılaştırmasını zorunlu kılıp yoksa hata fırlatıyor, ama `compare_to_cpa` (satır 580-582) için eşdeğer bir zorunluluk yok.
- **60 günlük gözetimsiz toplayıcıda hiçbir yerde `logging` kullanılmıyor:** Tüm durum/uyarı çıktıları (örn. `fetch_tles.py:302,322-325,349`) düz `print()` ile veriliyor.
- **GitHub Actions: data-collection dalına push için yeniden deneme yok:** `collect_observations.yml:74-113`'te push başarısız olursa iş doğrudan başarısız oluyor. Ayrıca dal-varlığı kontrolü (satır 83-94) `ls-remote`'un her türlü hatasını yutuyor.
- **`tle_validation.py`'nin kendine ait hiçbir birim testi yok:** Fail-closed veri bütünlüğü hikayesinin merkezindeki bu modül yalnızca dolaylı olarak test ediliyor.
- **`run_pipeline.py` ve `generate_plots.py` uçtan uca test edilmiyor:** README'nin ilk gösterdiği giriş noktası olmasına rağmen `run_pipeline.main()`'ı gerçekten çalıştıran bir test yok.
- **Depo hijyeni:** Kök dizinde `.gitignore`'a eklenmiş ama silinmemiş `cd`/`dir` adlı sıfır-bayt dosyalar; `.gitignore` tarafından kapsanmayan bir `tmp/` dizini; kalıcı olarak izlenen bir `COMMIT_MESSAGE.txt` taslağı.

---

## 5. Öncelikli Aksiyon Listesi

1. **`run_pipeline.py`'nin varsayılan özellik setini düzelt** — ya sızıntı içeren tam seti opt-in yap, ya da anlık (snapshot-only) güvenli seti varsayılan yap ve tam set kullanıldığında ekrana uyarı bas.
2. **`src/` kökündeki step1..step10 kopyalarını sil** — `legacy/` altındaki tek kopya yeterli.
3. **manifest.json, resimülasyon/geçmiş raporları ve CSV provenance yazımlarını** mevcut `tempfile + os.replace` yardımcısına taşı.
4. **TLE doğrulama ve `_partition_tle_hash_quality` mantığını tek bir paylaşılan yardımcıya birleştir.**
5. **%90 kapsam kontrolünü `--group` alımları için de zorunlu kıl**, ya da README'deki iddiayı yalnızca CATNR akışıyla sınırlandıracak şekilde netleştir.
6. **LICENSE dosyası ekle.**
7. **`evidence.py`'yi veri-doğrulama / bölme / kol-üretimi / manifest-yazımı olarak ayrı, test edilebilir modüllere böl.**
8. **Depo hijyenini temizle:** `tmp/`'yi `.gitignore`'a ekle veya sil, `cd`/`dir`/`COMMIT_MESSAGE.txt` dosyalarını kaldır.

---

*Bu rapor, kod tabanının salt-okunur statik incelemesine dayanır. Satır numaraları inceleme anındaki çalışma kopyasına (commit 1d7e272 + kayıtsız değişiklikler) aittir ve dal ilerledikçe kayabilir.*

---

## 6. Bu Dalda Çözülen Bulgular (21 Temmuz 2026)

Kapsam bilinçli olarak sınırlandırıldı: aşağıdaki maddeler bu turda kapatıldı;
`evidence.py`'nin alt modüllere bölünmesi ve `ml.py`'deki
`compare_models`/`time_series_cv_report` kapı-mantığı tekrarının
birleştirilmesi (madde 7) **bilinçli olarak ertelendi** — davranış
değiştirmeyen ama bildirinin istatistiksel çekirdeğine dokunan bu iki refactor,
en yüksek risk/en düşük acil ihtiyaç oranına sahip; ayrı, daha temkinli bir
PR'a bırakıldı.

- **3.1 (etiket sızıntısı) — Çözüldü.** `run_pipeline.py`'ye `--feature-set`
  argümanı eklendi (`space_debris.ml.FEATURE_SETS`'e bağlı,
  `default="snapshot_only"`). `compare_models()` artık her zaman açık bir
  `feature_columns` alıyor; `full_rule_recovery` seçilirse konsola bunun bir
  sızıntı olmadığını ama "kural geri kazanım" kontrolü olduğunu açıklayan bir
  not basılıyor. README'nin "What is (and isn't) claimed" ve "Scientific runs"
  bölümleri bu ayrımı `train_from_history.py`'nin zaten yaptığı
  birincil/kontrol ayrımıyla tutarlı şekilde belgeliyor. Doğrulama: sentetik
  demo verisiyle çalıştırıldığında `snapshot_only` altında
  random_forest/xgboost PR-AUC'si artık 1.000 değil, gerçekçi 0.93-0.96
  aralığında — sızıntının kaldırıldığının doğrudan kanıtı.
  **Ek olarak keşfedilen ve düzeltilen regresyon:** `run_pipeline.py`'nin bu
  turdan önce zaten çağırdığı `create_publication_plots()`, ayrı bir
  (bu daldaki önceki, taahhüt edilmemiş çalışmadan gelen) değişiklikle artık
  yalnızca `train_from_history.py`'nin ürettiği dondurulmuş kanıt dosyalarını
  kabul ediyor ve argümansız çağrıldığında koşulsuz `ValueError` fırlatıyordu
  — yani README'nin ilk "Quick start" komutu bu daldaki mevcut haliyle
  tamamen çöküyordu. `run_pipeline.py`'den bu çağrı kaldırıldı (yayın-kalite
  grafikleri artık yalnızca dondurulmuş kanıt zincirinden üretiliyor, tek-atım
  demo/gerçek-veri yolu kendi bilimsel/pipeline grafiklerini üretmeye devam
  ediyor).
- **3.3 (tutarsız atomik yazma) — Kök nedenden çözüldü.**
  `provenance.py`'ye paylaşılan `write_text_atomic`/`write_json_atomic`
  yardımcıları eklendi (tempfile.mkstemp + fsync + os.replace,
  `evidence.py`'nin daha önce özel/yinelenen `_write_json_atomic`'iyle aynı
  desen). `write_csv_text_with_provenance` artık bu deseni kullanıyor — tek
  değişiklik `core.py` ve `ml.py`'nin ürettiği TÜM CSV çıktılarını otomatik
  atomikleştirdi. `build_collection_manifest.py`, `rebuild_history.py`,
  `resimulate_snapshots.py`'deki düz `write_text()` çağrıları
  `write_json_atomic`'e taşındı. `evidence.py`'nin özel kopyası silinip
  paylaşılana yönlendirildi.
- **3.4 (TLE doğrulama tekrarı) — Çözüldü.** `fetch_tles.py`'deki
  `tle_checksum_is_valid` kopyası silindi, `space_debris.tle_validation`'dan
  import ediliyor. `_partition_tle_hash_quality`'nin `ml.py` ve `evidence.py`
  kopyaları `tle_validation.partition_tle_hash_quality`'ye taşındı
  (`error_cls` parametresiyle her çağıranın kendi istisna tipini korumasına
  izin veriliyor).
- **3.5 (%90 kapsam iddiası) — README netleştirildi (kod davranışı
  değiştirilmedi).** `--group` alımları için yapay bir kapsam oranı icat
  etmek (`--max-objects`'e göre) küçük-ama-meşru gruplarda sahte
  reddedilmelere yol açardı; bunun yerine README'deki koşulsuz "%90 altı
  reddedilir" ifadesi CATNR akışıyla sınırlandırıldı ve `--group` akışının
  `write_tle_file` üzerinden zaten sahip olduğu "0 obje alındıysa reddet"
  garantisi belgelendi. Dondurulmuş IAC protokolü zaten her zaman CATNR
  `leo_mixed` presetini kullanıyor.
- **3.6 (LICENSE yok) — Çözüldü.** Kök dizine MIT `LICENSE` eklendi,
  `pyproject.toml`'a `license = {file = "LICENSE"}` eklendi,
  `docs/iac_gap_analysis.md`'deki ilgili madde "done" işaretlendi.
- **Depo hijyeni — Çözüldü.** Untracked `cd`/`dir` sıfır-bayt dosyaları
  silindi (`.gitignore` onları zaten kapsıyordu, sadece dosyalar diskte
  kalmıştı). `.gitignore`'a `/tmp/` eklendi (içeriğine dokunulmadı — yerel
  benchmark/pdf verisi). `COMMIT_MESSAGE.txt`'e kasıtlı olarak dokunulmadı:
  tracked ve geçmiş bir düzeltmenin gerçek tarihsel kaydı; silmek
  dokümantasyon kaybı olurdu, hijyen kazancı marjinal.
- **3.2 (step1..step10 kopyaları) ve madde 7 (`evidence.py` bölünmesi,
  `ml.py` kapı-mantığı birleştirme) — Bu turda ertelendi**, kullanıcı
  onayıyla: ilki düşük risk ama bu PR'ın kapsamı dışında bırakıldı, ikincisi
  yüksek blast-radius/düşük aciliyet nedeniyle ayrı bir PR'a bırakıldı.

Doğrulama: `python -m pytest tests/ -q` → 251/251 geçti (davranış
değişmeden önce ve sonra). Sentetik demo pipeline'ı hem `snapshot_only`
(varsayılan) hem `full_rule_recovery` (opt-in) feature-set'leriyle uçtan uca
çalıştırıldı.
