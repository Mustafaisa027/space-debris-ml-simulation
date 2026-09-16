# Gelidonya İDA — Test Checklist

Sıralı ilerle: her aşama bir öncekine güvenir. Bir madde FAIL ise devam etme,
yanındaki "çözüm" satırına bak. Tüm komutlar Jetson Orin Nano'da (aksi yazmadıkça).

Ön koşul: `colcon build` başarılı + `source install/setup.bash` yapılmış.

---

## Aşama A — Derleme ve statik kontroller (masa, donanımsız)

- [ ] **A1. Workspace derlemesi**
  ```bash
  cd ~/obstacle_map_ws
  colcon build --packages-select mission_interfaces && source install/setup.bash
  colcon build && source install/setup.bash
  ```
  Geçti sayılır: hata yok. FAIL → hata mesajını kaydet, eksik bağımlılık için
  `rosdep install --from-paths src --ignore-src -r -y`.

- [ ] **A2. Arayüzler üretildi mi**
  ```bash
  ros2 interface show mission_interfaces/msg/MissionState
  ros2 interface show mission_interfaces/srv/SetParkur
  ```
  Geçti: alanlar listeleniyor.

---

## Aşama B — Masa testi (FC ve kamera OLMADAN görev katmanı)

Sadece görev düğümlerini başlat (ZED/MAVROS/YOLO kapalı):
```bash
ros2 launch mission_control mission_bringup.launch.py use_yolo:=false use_avoidance:=true
```

- [ ] **B1. Düğümler ayakta**
  ```bash
  ros2 node list
  ```
  Beklenen: `mission_manager_node`, `iha_color_receiver_node`, `kamikaze_node`,
  `mission_logger_node`, `obstacle_bridge_node`.

- [ ] **B2. Mission state yayında (BOOT/READY)**
  ```bash
  ros2 topic echo /mission/state --once
  ```
  Beklenen: `state: 0` (BOOT — renk yok) veya `1` (READY).

- [ ] **B3. Renk zinciri (elle)**
  ```bash
  ros2 service call /mission/set_target_color mission_interfaces/srv/SetTargetColor "{color: 1}"
  ros2 topic echo /mission/state --once
  ```
  Beklenen: `success: true`; state artık READY (mavros yoksa BOOT'ta kalabilir —
  bu durumda B4'e geç). `info` içinde `renk=red`.

- [ ] **B4. Parkur geçişleri (debug servisi)**
  ```bash
  ros2 service call /mission/set_parkur mission_interfaces/srv/SetParkur "{parkur: 2}"
  ros2 topic echo /mission/state --once
  ```
  Beklenen: `parkur: 2`, `avoidance_active: true`. Manager loglarında
  "Engel kacinma -> ACIK".
  ```bash
  ros2 service call /mission/set_parkur mission_interfaces/srv/SetParkur "{parkur: 3}"
  ```
  Beklenen: `parkur: 3`, `kamikaze_active: true`, `avoidance_active: false`.

- [ ] **B5. Kamikaze sahte tespitle karar üretiyor**
  P3'teyken (B4 sonrası) ayrı terminalde sahte YOLO tespiti bas:
  ```bash
  ros2 topic pub -r 5 /yolo/detections std_msgs/msg/String "{data: 'class=red, conf=0.90, bbox=(500,200,780,520), center=(640,360)'}"
  ```
  ```bash
  ros2 topic echo /kamikaze/status
  ```
  Beklenen: `target_locked: true`, `phase` APPROACH/TERMINAL, `bearing_deg` ~0
  (merkez 640 = görüntü ortası). `center=(200,360)` ile tekrarla → bearing pozitif
  (sol). FAIL → hedef renk atanmış mı (B3), sınıf adı `red` mi kontrol et.

- [ ] **B6. Angajman tespiti**
  B5 komutunu büyük kutuyla ver: `bbox=(100,50,1180,700), center=(640,375)`.
  Beklenen: birkaç saniyede `engaged: true`, mission state → `DONE` (state 5).

- [ ] **B7. Log düğümü dosya üretiyor**
  ```bash
  ls -lt ~/gelidonya_logs/ | head
  tail -5 ~/gelidonya_logs/events_*.log
  ```
  Beklenen: `mission_*.jsonl`, `events_*.log`, `track_*.csv` var; events içinde
  PARKUR GECISI satırları.

- [ ] **B8. Acil durdurma**
  ```bash
  ros2 service call /mission/emergency_stop std_srvs/srv/Trigger
  ros2 topic echo /mission/state --once
  ```
  Beklenen: `state: 6` (EMERGENCY), avoidance+kamikaze kapalı.

> Launch'ı yeniden başlatmadan sonraki aşamaya geçme (state makinesi DONE/EMERGENCY'de kaldı).

---

## Aşama B2 — Simülatörle uçtan uca (donanımsız, EN DEĞERLİ TEST)

`sim_mavros_node` sahte uçuş kontrolcüsü + kinematik tekne, `sim_target_node`
sahte YOLO tespiti üretir. ZED, CUAV X7+ ve YOLO modeli **gerekmez**.

- [ ] **B2-1. Parkur-1 (waypoint takibi)**
  ```bash
  ros2 launch mission_control test_parkur.launch.py parkur:=1
  ```
  Ayrı terminalde başlat:
  ```bash
  ros2 service call /sim/arm std_srvs/srv/SetBool "{data: true}"
  ```
  ```bash
  ros2 topic echo /mission/state
  ```
  Beklenen: state READY→PARKUR1, `wp_seq` 0→3 sırayla ulaşılıyor, WP4'te
  PARKUR2'ye geçiyor. Loglarda `[sim] WP1..WP4 ulasildi`.

- [ ] **B2-2. Parkur-2 (GUIDED + WP5)**
  ```bash
  ros2 launch mission_control test_parkur.launch.py parkur:=2
  ```
  Beklenen: PARKUR2'ye geçince mod GUIDED oluyor (`[sim] mod -> GUIDED`),
  manager `/mavros/setpoint_position/global`'a WP5 yayınlıyor, sim tekne
  oraya gidiyor, `/mission/state` `info` alanında `d=..m` azalıyor,
  3 m altına inince PARKUR3.
  ```bash
  ros2 topic echo /mavros/setpoint_position/global --once
  ```

- [ ] **B2-3. Parkur-3 (kamikaze angajman)**
  ```bash
  ros2 launch mission_control test_parkur.launch.py parkur:=3 target_color:=red
  ```
  ```bash
  ros2 topic echo /kamikaze/status
  ```
  Beklenen: `target_locked: true`, faz SEARCH→APPROACH→TERMINAL→IMPACT,
  ~12 sn'de `engaged: true`, mission state DONE.
  **Ayrım testi:** sahte üretici yanlış renkte (yeşil) bir çapan hedef de
  yayınlıyor — kamikaze onu seçmemeli, `bearing_deg` kırmızı hedefe göre olmalı.

- [ ] **B2-4. Hedef kaybı emniyeti**
  P3 koşarken hedefi gizle:
  ```bash
  ros2 service call /sim_target/visible std_srvs/srv/SetBool "{data: false}"
  ```
  Beklenen: `target_locked: false`, faz SEARCH, hız komutu sıfırlanıyor.

- [ ] **B2-5. Tam görev zinciri**
  ```bash
  ros2 launch mission_control test_parkur.launch.py parkur:=all
  ```
  Beklenen: P1→P2→P3→DONE kesintisiz. `events_*.log` dosyasında tüm geçişler.

- [ ] **B2-6. Teslim dosyaları üretiliyor mu**
  ```bash
  ls -lht ~/gelidonya_logs/ | head
  ```
  Beklenen: `telemetri_*.csv` var ve satırları doluyor.
  ```bash
  head -3 ~/gelidonya_logs/telemetri_*.csv
  ```
  Header + lat/lon/hız/heading dolu olmalı. (Video dosyaları için gerçek kamera
  gerekir — Aşama C'de doğrulanır.)

---

## Aşama C — Sensör testi (ZED takılı, FC hâlâ opsiyonel)

```bash
ros2 launch mission_control system_bringup.launch.py use_mavros:=false
```

- [ ] **C1. ZED topic'leri var ve isimler config ile aynı**
  ```bash
  ros2 topic list | grep zed
  ```
  Beklenen: `/zed/zed_node/point_cloud/cloud_registered` ve `.../rgb/color/rect/image`.
  FAIL (ör. `/zed2i/...` ise) → `mission_params.yaml` içindeki `cloud_topic` /
  `image_topic` değerlerini gerçek isimlere güncelle, tekrar dene.

- [ ] **C2. QoS uyumu (sessiz kopukluk kontrolü)**
  ```bash
  ros2 topic info /zed/zed_node/point_cloud/cloud_registered --verbose
  ```
  Beklenen: publisher BEST_EFFORT; subscriber sayısı ≥1 ve BEST_EFFORT.

- [ ] **C3. obstacle_bridge kapalıyken sessiz**
  ```bash
  ros2 topic hz /mavros/obstacle/send
  ```
  Beklenen: mesaj YOK (P2 dışında bridge kapalı — start_enabled:false).

- [ ] **C4. P2'de bridge veri akıtıyor**
  ```bash
  ros2 service call /mission/set_parkur mission_interfaces/srv/SetParkur "{parkur: 2}"
  ros2 topic hz /mavros/obstacle/send
  ros2 topic echo /mavros/obstacle/send --once
  ```
  Beklenen: ~5-15 Hz LaserScan; öne cisim tutunca ilgili sektörlerde ~mesafe (m),
  boşken `inf`. Bridge loglarında `ROI/FOV/OCC/NEAREST` değerleri.
  FAIL (0 Hz) → C2'yi tekrar kontrol et; ZED cloud gerçekten akıyor mu:
  `ros2 topic hz /zed/zed_node/point_cloud/cloud_registered`.

- [ ] **C5. Yön doğrulaması (kritik — yanlışsa BendyRuler ters kaçar)**
  Cismi kameranın SOLUNA tut → `echo`'da yüksek indeksli (angle_max tarafı)
  sektörler dolmalı (`ranges` sol=pozitif açı=dizinin sonu). Sağda tut → düşük
  indeksler. FAIL → ZED montaj yönü / `coordinate_mode` kontrolü.

- [ ] **C6. YOLO gerçek görüntüyle**
  ```bash
  ros2 topic echo /yolo/detections
  ```
  Telefonda kırmızı/yeşil/siyah görsel gösterip tespit metnini gör. FPS düşükse
  `imgsz: 416` + `process_every_n_frames` artır.
  Model henüz hazır değilse bu maddeyi atla — `sim_target_node` yerine geçer.

- [ ] **C7. Teslim videoları (Dosya 1 ve 3)**
  ZED açıkken 30 sn kayıt alıp kapat (`Ctrl+C` ile düzgün kapat!):
  ```bash
  ls -lh ~/gelidonya_logs/*.mp4
  ```
  Her iki mp4 de 0 byte'tan büyük olmalı. Oynatıp kontrol et: kamera videosunda
  bbox+sınıf+zaman etiketi, harita videosunda kırmızı engel hücreleri görünmeli.
  FAIL (0 byte) → launch'ı `Ctrl+C` yerine `kill -9` ile kapatmışsındır; mp4
  sonlandırılamaz. Detay: VERI_TESLIMI.md

---

## Aşama D — FC bağlı (HIL, motorlar TAKILI DEĞİL veya pervanesiz!)

```bash
ros2 launch mission_control system_bringup.launch.py fcu_url:=serial:///dev/ttyACM0:115200
```

- [ ] **D1. MAVROS bağlantısı**
  ```bash
  ros2 topic echo /mavros/state --once
  ```
  Beklenen: `connected: true`. FAIL → port/baud (`ls /dev/ttyACM* /dev/ttyTHS*`),
  `dialout` grubu, FC'de SERIALx_PROTOCOL=2.

- [ ] **D2. Renk zinciri uçtan uca (RFD900x üzerinden)**
  Yer laptop'ta MAVProxy + `send_color.py` (bkz. ida_gcs/renk_relay_README.md):
  ```bash
  ros2 topic echo /mavros/statustext/recv
  ros2 topic echo /iha/target_color
  ```
  Beklenen: STATUSTEXT'te `COLOR:RED`, ardından target_color `color: 1`.
  FAIL → Mission Planner mesaj ekranında görünüyor mu? Görünüyorsa sorun
  SERIALx_OPTIONS forward biti; görünmüyorsa MAVProxy enjeksiyonu.

- [ ] **D3. Otomatik başlama**
  Mission Planner'dan görev yükle (≥5 WP), ARM + AUTO yap (pervanesiz!).
  Beklenen: `/mission/state` READY→PARKUR1. Loglarda "DURUM GECISI".

- [ ] **D4. WP geçişiyle otomatik parkur değişimi**
  Mission Planner'da WP'leri simüle et ya da:
  ```bash
  ros2 topic pub --once /mavros/mission/reached mavros_msgs/msg/WaypointReached "{wp_seq: 3}"
  ```
  Beklenen: PARKUR1→PARKUR2, bridge AÇIK, **FC modu GUIDED'e geçiyor**
  (Mission Planner HUD'da doğrula), manager WP5'i hedef olarak yayınlamaya
  başlıyor. Doğrula:
  ```bash
  ros2 topic echo /mavros/setpoint_position/global --once
  ```
  Araç WP5'e 3 m yaklaşınca PARKUR3'e geçmeli. Mod dönmüyorsa manager 1 sn'de
  bir yeniden dener (loglarda `Ucus modu istendi -> GUIDED`).

  ⚠️ WP5 konumu yüklü görevden okunur. Görev yüklü değilse logda
  "WP5 konumu bilinmiyor" uyarısı çıkar ve GUIDED hedefi gönderilmez.
  Kontrol:
  ```bash
  ros2 topic echo /mavros/mission/waypoints --once
  ```

- [ ] **D5. BendyRuler OA parametreleri FC'de ayarlı**
  Mission Planner → Full Parameter List: `OA_TYPE=1`, `PRX1_TYPE=2`,
  `AVOID_ENABLE` açık, `OA_LOOKAHEAD`≈3, `OA_MARGIN_MAX`≈1.5.
  Proximity ekranında (Ctrl+F → Proximity) bridge'in sektörleri görünmeli.
  **Bu madde geçmeden suya çıkma — P2 kaçınması çalışmaz.**

---

## Aşama E — Su testi

- [ ] **E1. P1 nokta takip:** 4 WP'lik görev, AUTO+ARM → araç sırayla gezip
  WP4'te PARKUR2'ye geçiyor (`/mission/state` + events log).
- [ ] **E2. P2 kaçınma tuning:** rotaya şişme duba koy. Araç ~2 m'den kaçmalı.
  Erken/geç kaçıyorsa FC'de `OA_LOOKAHEAD`/`OA_MARGIN_MAX` ve WP hızını ayarla;
  algı tarafında `obstacle_bridge max_range/fov` oynanabilir.
- [ ] **E3. P3 kuru koşu:** `send_velocity: false` iken hedefe yaklaş, 
  `/kamikaze/status` kerteriz/faz doğru mu izle (araç sürmez, sadece karar).
- [ ] **E4. P3 canlı:** `send_velocity: true` yap, yeniden başlat. GUIDED'de araç
  hedefi ortalayıp hızlanmalı; temas sonrası `engaged` + DONE + motor komutu sıfır.
- [ ] **E5. Tam prova:** boot'tan (systemd) tek seferde P1→P2→P3, süre ölç
  (yarışma limiti 20 dk), logları USB'ye kopyala (veri teslimi provası).

---

## Hızlı hata tablosu

| Belirti | İlk bakılacak |
|---|---|
| Topic var, veri yok | QoS mismatch → `ros2 topic info <t> --verbose` |
| Bridge 0 Hz | ZED cloud akışı + topic ismi (C1) |
| Parkur geçmiyor | `/mavros/mission/reached` geliyor mu, WP indeksleri (0-tabanlı) |
| Renk gelmiyor | D2 ikiye bölme testi |
| P3'te araç sürmüyor | mod GUIDED mi + `send_velocity` true mu |
| BendyRuler kaçmıyor | D5 parametreleri + Proximity ekranı |
