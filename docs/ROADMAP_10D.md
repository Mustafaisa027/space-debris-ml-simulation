# 10 Günlük Yol Haritası — IAC 114764 Bildirisine Giden Yol

> **Arşiv notu (2026-08-07):** Bu belge tamamlanmış v2/v3 operasyon planını
> korur; aktif talimat değildir. Aktif bağımsız doğrulama deneyi
> `iac26-15d-v4` ve bağlayıcı protokol `docs/EXPERIMENT_15D_V4.md` içindedir.
> v2/v3 verileri v4 ile havuzlanmaz.

**Toplama penceresi:** `2026-07-26T00:17:00Z → 2026-08-05T00:17:00Z`
**Deney:** `iac26-10d-v3`, 2 saatlik cadence, 120 slot, 75 NORAD ID (`iac26-leo-mixed-75-v2`)
**Kritik yol:** Gerçek TLE toplama = tek darboğaz. Kod hazır; kısaltılamaz.

> **v3 re-anchor notu:** İlk pencere (v2, 24 Tem açıldı) bir TLE-feed cold-start'ına
> denk geldi (9 ardışık bin aynı katalog → frozen `≤6` kapısını kalıcı ihlal).
> Gün 1'de, sonuç görülmeden, pencere 26 Tem'e çıpalandı; v2 arşivi pilot kalır.
> Detay: `docs/EXPERIMENT_10D_V3.md`. **v3 workflow'u `main`'e 26 Tem 00:17Z'den
> önce push edilmeli** ki toplama zamanında başlasın.

Bu yol haritası, bağlamdan kopmadan "veri topla → kanıt üret → makale yaz"
akışını üç paralel şeride ayırır: **(A) Toplama & izleme**, **(B) Kod & kanıt
hazırlığı**, **(C) Makale yazımı**. A şeridi otomatiktir (GitHub Actions); asıl
insan emeği C'ye ayrılmalı.

---

## Şerit A — Veri toplama ve izleme (otomatik + günlük 5 dk kontrol)

### Gün 0 (BUGÜN, 24 Tem) — Toplama gerçekten çalışıyor mu? ⚠️ EN KRİTİK ADIM
Pencere bugün açıldı. Kaybedilen her slot geri gelmez. **Şunları doğrula:**

1. **GitHub Actions etkin mi?** GitHub deposunda Actions sekmesi açık olmalı ve
   "Collect TLE observations" workflow'u `main` dalında bulunmalı
   (`.github/workflows/collect_observations.yml`).
2. **Cron tetikleniyor mu?** Actions sekmesinde `:17` ve `:47` işlerinin
   çalıştığını gör. İlk birkaç saat içinde en az bir yeşil çalıştırma olmalı.
3. **`data-collection` dalı oluştu mu?** İlk başarılı toplama, `data-collection`
   dalı altına `experiments/iac26-10d-v3/collections/github-run-*` bundle'ı
   push etmeli.
4. **(Opsiyonel) Space-Track fallback:** `SPACETRACK_IDENTITY` /
   `SPACETRACK_PASSWORD` repo secret'ları tanımlıysa CelesTrak kesintilerinde
   yedek devreye girer. Yoksa CelesTrak tek kaynak — çoğu zaman yeterli.

> Not: Bu adımlar depo/GitHub tarafındadır; kod tarafı hazır. Actions
> çalışmıyorsa 10 gün boşa gider, o yüzden Gün 0 doğrulaması pazarlık konusu
> değildir.

### Gün 1–9 — Günlük sağlık kontrolü (yerelde, ~5 dk)
Her gün `data-collection` dalını çekip pencere/cadence sağlığına bak:

```powershell
# data-collection dalını yerel bir worktree'ye al (ilk sefer):
git worktree add ..\sd-archive data-collection
# sonraki günler sadece güncelle:
git -C ..\sd-archive pull

# Pencere doluluk durumu (kaç slot doldu, boşluk var mı):
.venv\Scripts\python.exe src\collection_window_status.py `
    --config config\experiment_10_days_v3.json

# Cadence sağlığı (6 saatten büyük boşluk, bayat-hash run'ı var mı):
.venv\Scripts\python.exe src\collection_cadence_health.py `
    ..\sd-archive --config config\experiment_10_days_v3.json
```

**Neye bak:**
- Doluluk oranı trendi ≥ %90'a gidiyor mu? (gereken: 108/120 slot)
- 6 saatten büyük boşluk (`max_snapshot_gap_hours`) oluştu mu?
- TLE hash çeşitliliği ≥ %30 mü? (feed bayatlamıyor mu)

**Kırmızı bayrak görürsen:** Actions loglarına bak. Erken müdahale (ör. secret
düzeltme, CelesTrak preset teyidi) bir sonraki slotu kurtarır; geç müdahale
slotları kalıcı kaybettirir.

---

## Şerit B — Kod & kanıt hazırlığı (Gün 0–3, tek seferlik)

Toplama arka planda dönerken kanıt zincirini **kuru prova** ve hazır tut.

### B.1 Ortam sağlığı
```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pytest -q      # 311 passed beklenir
```

### B.2 Finalize zincirini tanı (henüz veri yok, sadece anla)
Toplama bitince çalıştırılacak tek komut:

```powershell
scripts\finalize_iac_experiment.ps1 -ArchiveRoot ..\sd-archive
```

Bu script sırayla şunları yapar (her aşama checkpoint'li, tekrar çalıştırınca
doğrulanmış aşamaları atlar):
1. `collection_window_status.py` — pencere/doluluk kapısı
2. `collection_cadence_health.py` — cadence kapısı
3. `import_collection_archive.py` — bundle'ları hash-doğrulayıp içe al
4. `resimulate_snapshots.py` — düzeltilmiş-TCA ile yeniden simüle et
5. `train_from_history.py` — LR/RF/SVM/XGBoost + kanıt + yayın doğrulaması

`train_from_history.py` **fail-closed**: sınıf/zaman desteği yetersizse
`not_enough_data` döner ve eşikleri gevşetmez. Bu bir hata değil, sözleşmedir.

### B.3 (Opsiyonel ama önerilir) Küçük kuru prova
Gerçek veri gelmeden zincirin bütününü prova etmek istersen, mevcut v1
pilot/audit arşivi (varsa) veya birkaç `collect_observations.py --once`
çalıştırmasıyla oluşturulmuş küçük bir arşiv üzerinde
`import → resimulate → train` adımlarını elle çalıştırıp entegrasyon
hatalarını **şimdi** yakala (Gün 10'da değil). Not: bu prova sonucu makaleye
girmez; yalnızca boru hattını doğrular.

### B.4 Düşük öncelikli temizlik (boş vakit olursa)
- `data/sample_tles.txt` checksum'larını düzelt veya "illüstratif, checksum
  doğrulanmamış" notu ekle (bkz. güncelleme notları 2.3).

---

## Şerit C — Makale yazımı (Gün 0'dan itibaren PARALEL, asıl emek burada)

Sayısal sonuçlar Gün 10'da gelecek; ama makalenin **%70'i sonuçlardan
bağımsız** yazılabilir. Krediyi/vakti burada kullan.

### Gün 0–4: Sonuçtan bağımsız bölümler
- **Giriş & literatür:** LEO yoğunluğu, CRA, Pc ve covariance verisi kısıtı
  (abstract'taki çerçeve). Açık-veri/açık-kaynak motivasyonu.
- **Metodoloji:** `docs/METHODOLOGY.md` ve `docs/iac_gap_analysis.md`'yi temel
  al. TLE→SGP4 propagasyon, coarse screening + exact TCA refinement, risk
  metrikleri (min mesafe, TCA, bağıl hız), `risk_label` proxy tanımı ve
  **"Pc değildir"** sınırının açıkça yazılması.
- **Deney tasarımı:** dondurulmuş 10-günlük v2 protokolü, çift-gruplu
  kronolojik sızıntı-önleme bölmesi, `snapshot_only` vs `full_rule_recovery`
  ayrımı, sabit-eşik + kalibre-mesafe baseline'ları. `docs/EXPERIMENT_10D_V2.md`
  birebir kaynak.
- **Şekil iskeletleri:** demo çalıştırmasının ürettiği grafik türlerini
  yer-tutucu olarak yerleştir (feature space, TCA-distance scatter, model
  metrics). Gerçek veriyle aynı kod bunları yeniden üretecek.

### Gün 5–7: Yarı-yolda ara analiz
- Toplanan kısmi veriyle (henüz kapı geçmese de) boru hattını çalıştırıp
  şekil/tablo formatlarını sonlandır. **Uyarı:** bunlar ön-bakış; makaleye
  yalnızca Gün 10'daki dondurulmuş sonuç girer.

### Gün 8–10: Gerçek sonuç entegrasyonu
1. Pencere kapandıktan (03 Ağu 00:17Z) sonra `data-collection`'ı çek.
2. `scripts\finalize_iac_experiment.ps1 -ArchiveRoot ..\sd-archive` çalıştır.
3. Çıktılar:
   - `outputs/history/model_comparison_iac26_75_v2_pair_grouped_time_split.csv`
   - `train_from_history.py`'nin ürettiği kanıt dosyaları + yayın grafikleri
4. Precision / recall / F1 / PR-AUC / confusion matrix / false-alarm rate
   tablolarını ve şekilleri makaleye yerleştir.
5. **Dürüstlük kapısı:** Sonuçlar abstract'taki "significantly higher
   adaptability and reduced false alarm rate" cümlesini destekliyorsa yaz.
   **Desteklemiyorsa** (`not_enough_data` veya kapı düşerse), analizi cümleye
   uydurmaya çalışma — abstract/metin ifadesini gerçeğe göre revize et. Bu,
   deponun kendi sözleşmesidir (`iac_gap_analysis.md`).

---

## Beklenmedik durum planı: `not_enough_data`

10 gün sonunda pozitif olay/sınıf/pair/snapshot desteği kapıyı geçmezse,
geçerli ve dürüst sonuç `not_enough_data`'dır. Seçenekler:
1. **Kapsamı daralt:** Birincil iddiayı, geçen kapılarla sınırlı sun (ör.
   yalnızca tanımlayıcı 5-fold CV + sabit-eşik karşılaştırması), adaptability
   iddiasını "gelecek çalışma" olarak işaretle.
2. **Metni revize et:** Abstract'ın sonuç cümlesini "framework + karşılaştırma
   altyapısı" katkısına indir (ki bu zaten başlıca çıktı olarak yazılı),
   sayısal üstünlük iddiasını hipotez olarak bırak.
3. **Eşikleri ASLA gevşetme:** Sonuç görüldükten sonra kapı düşürmek
   outcome-dependent olur ve bilimsel geçerliliği bozar.

---

## Tek bakışta kontrol listesi

- [ ] **Gün 0:** GitHub Actions çalışıyor, `data-collection` dalı bundle alıyor
- [ ] **Gün 1–9:** Günlük `collection_window_status` + `collection_cadence_health`
- [ ] **Gün 0–3:** `pytest` yeşil, finalize zinciri anlaşıldı, (ops.) kuru prova
- [ ] **Gün 0–7:** Makalenin metodoloji/giriş/tasarım bölümleri yazıldı
- [ ] **Gün 8:** Pencere kapandı, arşiv çekildi
- [ ] **Gün 8–9:** `finalize_iac_experiment.ps1` çalıştı, kanıt üretildi
- [ ] **Gün 9–10:** Sonuç tabloları/şekilleri makaleye girdi, dürüstlük kapısı
      uygulandı
