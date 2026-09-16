# CODEX PROMPT — "Orbital Sentinel" Masaüstü GUI (PySide6 / VTK)
### LEO Uzay Enkazı Çarpışma Riski Simülasyonu — video-hazır, tamamen Python masaüstü arayüzü

> Bu dosyanın tamamını Codex'e (terminal ajanına) yapıştır. Codex önce mevcut `gui/` prototipini ve `src/export_gui_data.py` sözleşmesini analiz etmeli, sonra bu spesifikasyona göre yeni bir masaüstü uygulaması inşa etmelidir. Web tabanlı bir çözüm **istenmiyor** — tamamen Python/Qt tabanlı bir masaüstü uygulaması hedefleniyor.

---

## 0. BAĞLAM — MEVCUT DURUM (zaten doğrulandı, tekrar keşfetme)

Önceki bir analizde şu tespit edildi, bunu ön bilgi olarak kabul et:

- `gui/index.html`, `gui/app.js`, `gui/styles.css`: statik, Canvas tabanlı, framework'süz bir web prototipi. Bu **referans** olarak kullanılacak (UX akışı, senaryo listesi, renk fikirleri için) ama **yeni uygulama bunun üzerine yazılmayacak** — yeni uygulama saf Python/Qt olacak.
- `gui/data/simulation.json`: 8 senaryo, 7 model, 3 kalite kapısı içeren, halihazırda üretilmiş ve doğrulanmış veri sözleşmesi.
- `src/export_gui_data.py`: analiz/ML çıktısını yukarıdaki JSON'a dönüştüren mevcut ihracat katmanı.
- `tests/test_export_gui_data.py`: bu ihracat katmanının testi.
- `gui/start_gui.ps1`, `gui/README.md`: eski web prototipinin çalıştırma talimatları (artık ikincil).
- `gui/`, exporter ve testleri **henüz git'e commit edilmemiş** (untracked).
- Depo `.venv`'i bozuk/eski bir Python yoluna bağlı; alternatif runtime Python'da `pytest` kurulu değil.

## 1. GÖREV

1. Önce `git status` / `git add` ile mevcut çalışan `gui/`, `src/export_gui_data.py`, `tests/test_export_gui_data.py` dosyalarını **ayrı, açıklayıcı bir commit** ile kaydet (checkpoint) — bunları kaybetmeyelim.
2. Bozuk `.venv`'i teşhis et ve düzelt ya da temiz bir sanal ortam kur; `pytest` dahil gerekli paketleri kur; mevcut test paketini (`pytest`) çalıştırıp yeşil olduğunu doğrula. Bunu yapmadan yeni bağımlılık eklemeye geçme.
3. Yeni bir **PySide6 tabanlı masaüstü GUI** inşa et (aşağıdaki mimariye göre), mevcut `gui/data/simulation.json` **veri sözleşmesini birebir tekrar kullanarak** (şemayı bozma; ihtiyaç olursa yalnızca geriye-uyumlu şekilde genişlet ve `export_gui_data.py`'yi buna göre güncelle, testi de güncelle).
4. Simülasyon/ML çekirdek koduna (SGP4 yayılımı, etiketleme, model eğitimi, bölme mantığı) **dokunma**.
5. Bitirince bana: hangi dosyaları oluşturdun/değiştirdin, hangi varsayımları yaptın, `simulation.json` şemasında genişletme yaptıysan neyi neden değiştirdiğini özetle.

## 2. NEDEN PYSIDE6 (kütüphane seçimi ve gerekçesi)

| İhtiyaç | Seçim | Gerekçe |
|---|---|---|
| Pencere/uygulama iskeleti | **PySide6** (Qt for Python, LGPL) | PyQt6'ya göre lisanslama daha basit (LGPL, ticari kısıtlama yok); API neredeyse birebir aynı. Codex bunun yerine PyQt6 kullanmayı daha uygun görürse gerekçesini açıklayıp öyle uygulayabilir. |
| 3B Dünya + yörünge görselleştirme | **PyVista + pyvistaqt** (VTK tabanlı) | Ham OpenGL/VisPy'ye göre çok daha az kod ile gerçekçi ışıklandırma, doku (Dünya haritası), kamera fly-to animasyonu, glow/çizgi efektleri sağlar; Qt widget'ı olarak doğrudan gömülür. |
| 2B grafikler (model karşılaştırma, zaman çizelgesi yoğunluğu) | **pyqtgraph** | Matplotlib'e göre çok daha performanslı, gerçek zamanlı/etkileşimli güncellemede takılmaz; Qt ile doğal entegre. Statik, yayın-kalitesi bir grafik gerekiyorsa (ör. dışa aktarılan tekil bir rapor görseli) o noktada matplotlib de kullanılabilir. |
| Stil/tema | **Qt Style Sheets (QSS)** | CSS benzeri sözdizimiyle koyu "mission control" temasını tanımlamak için yeterli, ek bağımlılık gerektirmez. |
| Video üretimi | **Kare-kare render + OpenCV (`cv2.VideoWriter`)** | Ekran kaydına (OBS vb.) güvenmek yerine, uygulamanın kendisi her kareyi `widget.grab()` ile yakalayıp doğrudan video dosyasına yazar. Böylece video, bilgisayarın o anki performansından bağımsız, **tamamen deterministik ve pürüzsüz** (sabit 60 fps) çıkar — her çalıştırmada birebir aynı sonucu üretir. |

Gerekli yeni bağımlılıklar (`requirements.txt`'e ekle, sürüm pinle): `PySide6`, `pyvista`, `pyvistaqt`, `pyqtgraph`, `opencv-python`, `numpy` (muhtemelen zaten var).

## 3. DİZİN YAPISI

```
space-debris-project/
├── (mevcut simülasyon/ML kodu — dokunulmaz)
├── gui/                        # eski web prototipi — referans/yedek olarak kalır, silinmez
├── src/export_gui_data.py      # mevcut ihracat katmanı — yeniden kullanılır
├── desktop_gui/                # YENİ — PySide6 uygulaması
│   ├── main.py                  # uygulama giriş noktası
│   ├── app_window.py            # ana pencere, sahne geçişleri (QStackedWidget)
│   ├── data_loader.py           # gui/data/simulation.json okuma + doğrulama (pydantic modeliyle)
│   ├── theme/
│   │   ├── mission_control.qss  # QSS tema dosyası
│   │   └── tokens.py             # renk/tipografi sabitleri (tek kaynak)
│   ├── scenes/
│   │   ├── briefing_scene.py     # Sahne A - açılış
│   │   ├── orbit_scene.py        # Sahne B - PyVista 3B ana görünüm
│   │   ├── conjunction_panel.py  # Sahne C - seçili çift detay paneli
│   │   ├── model_dashboard.py    # Sahne D - pyqtgraph model karşılaştırma
│   │   ├── quality_gates_panel.py# Sahne E - veri kalitesi/şeffaflık
│   │   └── summary_scene.py      # Sahne F - kapanış özeti
│   ├── playback/
│   │   ├── timeline_controller.py# oynatma/hız/seek mantığı, QTimer tabanlı, delta-time
│   │   └── camera_sequences.py   # video için önceden tanımlı kamera yörüngesi tanımları
│   └── recording/
│       └── frame_exporter.py     # widget.grab() → cv2.VideoWriter kare-kare video üretimi
└── tests/
    └── test_desktop_gui_smoke.py # uygulamanın hatasız açılıp simulation.json'u yüklediğini doğrulayan duman testi
```

## 4. VERİ SÖZLEŞMESİ — MEVCUT `simulation.json` KULLANILACAK

Yeni exporter yazma; `gui/data/simulation.json`'daki mevcut yapıyı (8 senaryo, 7 model, 3 kalite kapısı) `desktop_gui/data_loader.py` içinde bir `pydantic` (veya `dataclasses`) modeline yükle. Şemayı önce oku ve gerçek alan adlarını tespit et; bu promptta alan adlarını tahmin etmiyorum — **gerçek JSON'u incele ve ona sadık kal.** Sadece masaüstü uygulamasının ihtiyaç duyup da eksik olan bir alan varsa (ör. kamera fly-to için önerilen 3B koordinatlar), `export_gui_data.py`'yi geriye-uyumlu şekilde genişlet ve ilgili testi güncelle.

Her sahnede gösterilecek sayılar (Tablo 1/Tablo 2/Şekil 2 eşdeğerleri) **yalnızca bu JSON'dan** gelecek; hiçbir sayı GUI kodunda hardcode/uydurulmayacak.

## 5. SAHNELER (aynı içerik hedefi, Qt widget'ı olarak)

### Sahne A — Açılış (Briefing)
Koyu ekran, ortalanmış başlık + alt yazı fade-in (QGraphicsOpacityEffect + QPropertyAnimation). 3-4 saniye sonra otomatik Sahne B'ye geçiş (video akışı kesintisiz olsun).

### Sahne B — Ana 3B Simülasyon (PyVista sahnesi)
- `pyvistaqt.QtInteractor` widget'ı: dönen, dokulu bir Dünya küresi (serbestçe bulunabilen bir Dünya doku haritası kullan; lisans/atıf gerekiyorsa `desktop_gui/README.md`'de belirt).
- 75 LEO nesnesi, gerçek yörünge eğimlerine göre parametrik olarak yerleştirilmiş; ince, hafif parlayan iz çizgileri (orbit trail = `pyvista.Spline` veya polyline actor).
- Alt kısımda özel bir Qt widget'ı ile zaman çizelgesi: 120 dilimlik ızgara, dolu/boş dilimler renk kodlu (kapsama görselleştirmesi).
- Play/Pause/Hız kontrolü (`QSlider` + `QPushButton`); `timeline_controller.py` her karede `QTimer` ile obje pozisyonlarını enterpole edip `actor.SetPosition(...)` çağırarak akıcı hareket üretir (ani sıçrama yok).
- Konjonksiyon anına yaklaşınca: iki nesne arasına çizgi eklenir, mesafe HUD etiketinde (`pyvista.Text2D` veya overlay `QLabel`) canlı güncellenir; eşik aşılırsa çizgi kırmızıya döner ve hafif pulsing opaklık animasyonu uygulanır.
- `camera_sequences.py`'de tanımlı, PyVista'nın `camera.SetPosition/SetFocalPoint` ile kare kare interpolasyonla oynatılan bir "fly-to" sekansı; bir tuşla (`F`) tetiklenir.

### Sahne C — Konjonksiyon Detay Paneli
Bir nesne çiftine tıklanınca (PyVista picking API'si ile) veya otomatik sekansta, yan panelde (`QDockWidget` veya sabit sidebar) çift kimliği, TCA, min. mesafe, göreli hız, vekil etiket sonucu ve varsa model tahminleri gösterilir. Yarı saydam koyu panel (QSS `background-color: rgba(...)`).

### Sahne D — Model Karşılaştırma Panosu
`pyqtgraph.BarGraphItem` / `pyqtgraph.PlotWidget` ile Tablo 2'nin tüm satırları (Mesafe karşılaştırıcısı dahil 7 satır × PR-AUC/ROC-AUC/Kesinlik/Duyarlılık/F1/FAR) interaktif barlarla gösterilir. Mesafe karşılaştırıcısı görsel olarak (farklı renk/desen + tooltip açıklaması: "hedef-hizalı kontrol, bağımsız çarpışma gerçeği değil") diğerlerinden ayrıştırılır — makalenin 4.1 bölümündeki uyarı burada da net olmalı.

### Sahne E — Veri Kalitesi / Şeffaflık Paneli
3 kalite kapısının (kapsama, maks. boşluk, tekrarlanan TLE dizisi) FAIL durumu büyük, net simgelerle (kırmızı X) gösterilir; alt yazı: "Sonuçlar keşifseldir, operasyonel çarpışma riski performansı olarak yorumlanmamalıdır." **Bu sahne atlanmaz, zorunludur** — makalenin bilimsel dürüstlüğünün GUI'ye yansıması budur.

### Sahne F — Kapanış Özeti
Anahtar sayıların büyük tipografik gösterimi (count-up animasyonlu `QLabel`) + repo linki QR/metin olarak.

Sahneler arası geçiş: `QStackedWidget` + her geçişte kısa crossfade (`QGraphicsOpacityEffect`).

## 6. TASARIM SİSTEMİ ("Mission Control / Deep Space")

`desktop_gui/theme/tokens.py` içinde tek kaynak olarak tanımla, `mission_control.qss` bunları kullansın:

- Zemin: `#05070D`, panel yüzeyi: `#0E1420` (yarı saydam), birincil vurgu camgöbeği `#37E6E0`, uyarı/amber `#F5A623`, risk/kırmızı `#FF4D4F`, birincil metin `#C9D6E3`, ikincil metin `#6B7A8F`.
- Tipografi: HUD/telemetri için mono font (`JetBrains Mono`/`Space Mono`, sistemde yoksa Qt'nin bulacağı bir mono fallback), panel başlıkları için `Inter` (yoksa `Segoe UI`/sistem sans fallback).
- Hareket dili: easing eğrileri (`QEasingCurve.OutCubic` vb.), ani kesme yok.
- Abartılı neon/oyunlaştırma yok; NASA JPL "Eyes on the Solar System" / ESA MOC ekranlarına yakın, ölçülü ve güvenilir bir his.

## 7. VİDEO ÜRETİMİ (`desktop_gui/recording/frame_exporter.py`)

Ekran kaydı (OBS vb.) yerine **uygulama içi deterministik render** iste:

1. "Kayıt Modu" tetiklendiğinde, `timeline_controller` gerçek zamanlı `QTimer` yerine sabit bir adım boyu (ör. 1/60 sn) ile ilerler; her adımda `widget.grab()` (`QPixmap` → `QImage` → `numpy` dizisi) alınır.
2. Kareler `cv2.VideoWriter` ile doğrudan `.mp4` (H.264, 1920×1080, 60fps) dosyasına yazılır — ara PNG dizisi biriktirmeye gerek yok, ama istenirse debug için opsiyonel PNG dökümü de eklenebilir.
3. `camera_sequences.py`'deki tanımlı sekans(lar) baştan sona otomatik oynatılır; kullanıcı etkileşimi gerekmez (tam otomatik "demo modu").
4. Çıktı: `outputs/orbital_sentinel_demo.mp4`.
5. `desktop_gui/README.md`'ye şu komutu ekle: `python -m desktop_gui.main --record` (veya benzeri CLI bayrağı) → kayıt modunu tetikler.

## 8. KISITLAR

- Simülasyon/ML çekirdek mantığına dokunma.
- Hiçbir sayı uydurulmayacak; her değer `gui/data/simulation.json`'dan gelecek.
- Kod tanımlayıcıları İngilizce, arayüz metinleri Türkçe; ileride çeviri için basit bir `strings_tr.py`/`strings_en.py` ayrımı bırak.
- Eski `gui/` web prototipini silme; `desktop_gui/README.md`'de "bu, eski web prototipinin yerini alan birincil GUI'dir" notunu ekle.
- Büyük ölçekli dosya oluşturmadan önce bana kısa bir plan özeti sun (özellikle `.venv` onarımı ve simulation.json şema incelemesi bittikten sonra).

## 9. TESLİM VE DOĞRULAMA

1. `tests/test_desktop_gui_smoke.py`: uygulamanın `simulation.json`'u hatasız yüklediğini ve ana pencerenin oluşturulabildiğini doğrulayan (headless/offscreen Qt platform eklentisiyle, `QT_QPA_PLATFORM=offscreen`) bir duman testi.
2. `pytest` tüm paketle (eski testler dahil) yeşil olmalı.
3. `desktop_gui/README.md`: kurulum (`pip install -r requirements.txt`), çalıştırma (`python -m desktop_gui.main`) ve kayıt (`python -m desktop_gui.main --record`) komutları.
4. Kısa özet: hangi dosyalar eklendi/değişti, hangi varsayımlar yapıldı, `simulation.json` şemasında bir genişletme yapıldıysa ne/neden.

---

**Şimdi önce 1. adımdaki checkpoint commit'i ve `.venv`/test onarımını yap, ardından `gui/data/simulation.json`'un gerçek şemasını incele ve bana kısa bir plan özeti sun — onay almadan büyük ölçekli dosya oluşturmaya geçme.**
