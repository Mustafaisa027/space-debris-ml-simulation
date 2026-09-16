# Gelidonya İDA — Donanım ↔ Düğüm Uyumu

Kullanılan bileşenlerin ROS2 düğümlerine nasıl bağlandığı.

## Bileşen → Düğüm/Topic haritası

| Bileşen | Bağlandığı yer | İlgili düğüm / topic |
|---------|----------------|----------------------|
| **ZED2i** stereo kamera | Jetson Orin Nano (USB3) | zed-ros2-wrapper → `/zed/zed_node/point_cloud/cloud_registered`, `/zed/zed_node/rgb/color/rect/image` |
| **NVIDIA Jetson Orin Nano** (İDA) | companion computer | tüm algılama + mission_control düğümleri burada koşar |
| **CUAV X7+** (ArduPilot) | Jetson ↔ FC (USB/UART) | MAVROS → `/mavros/*`; obstacle_bridge → `/mavros/obstacle/send` |
| **Here4+ GPS** (DroneCAN) | CUAV X7+ CAN portu | ArduPilot → MAVROS `/mavros/global_position/global` (logger + manager) |
| **RFD900x @915 MHz** (İDA) | CUAV X7+ TELEM | İDA ↔ Yer İstasyonu (Mission Planner) MAVLink telemetri |
| **RFD868 @868 MHz + Raspberry Pi** (İHA) | İHA üzerinde | İHA renk tespiti + kendi telemetri linki |

## İHA→İDA renk aktarımı (onaylanan mimari — sadece RFD900x)

Modemler ayrı; İDA'nın Jetson'ına IP erişimi YOK. Tek link RFD900x MAVLink.
Renk, İDA yer laptop'unda RFD900x akışına enjekte edilir ve CUAV X7+ üzerinden
companion (Jetson) porta forward edilir:

```
İHA kamera → RFD868 → RPi (renk tespit)
   → [RPi → İDA laptop, yerel ağ/USB]
   → İDA laptop: MAVProxy (RFD900x'i paylaştırır: Mission Planner + send_color.py)
   → RFD900x → CUAV X7+ (STATUSTEXT broadcast, ArduPilot companion porta forward)
   → MAVROS → /mavros/statustext/recv → iha_color_receiver → /iha/target_color
```

- İDA tarafı: `iha_color_receiver_node`, `source_mode: mavlink` (config'de ayarlı).
  STATUSTEXT metninde renk kelimesi arar ("COLOR:RED" veya "red").
- Enjektör: `iha_rpi/send_color.py --mode mavlink --device udpout:<laptop_ip>:14551`.
- Multiplexer + ArduPilot forward ayarı + doğrulama: **ida_gcs/renk_relay_README.md**.
- ArduPilot: companion `SERIALx_PROTOCOL=2`, `SERIALx_OPTIONS` "don't forward" biti 0.

> **Yarışma kuralı uyarısı (5.5.3.1):** "İDA harekete başladıktan sonra YKİ/RC
> üzerinden komut verilemez (acil durdurma hariç)." Renk görev başlamadan
> (STATE_READY) ulaşmalı — mission_manager hareket başlayınca (PARKUR1) yeni rengi
> reddeder. Belirsizse **hakeme sor**; yedek olarak `default_color` ile elle atama.

## CUAV X7+ / ArduPilot ayarları (Rover — obstacle avoidance)

obstacle_bridge `/mavros/obstacle/send`'e `OBSTACLE_DISTANCE` MAVLink mesajı
gönderir. Gömülü BendyRuler'ın bunu kullanması için (Mission Planner → Full
Parameter List):

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `OA_TYPE` | 1 | BendyRuler nesne kaçınma |
| `AVOID_ENABLE` | 3 (veya 7) | Proximity + …'ya göre kaçınma aktif |
| `OA_LOOKAHEAD` | ~3-5 m | İleri bakış mesafesi (Parkur-2 ~2 m algıya uygun ayarla) |
| `OA_MARGIN_MAX` | ~1-2 m | Engelden bırakılacak pay |
| `PRX1_TYPE` | 2 | MAVLink proximity kaynağı (OBSTACLE_DISTANCE) |

> Değerler başlangıç önerisidir; su testinde BendyRuler'ın ~2 m'den tepki
> vermesi için `OA_LOOKAHEAD`/`OA_MARGIN_MAX` ayarlanmalı. Görev noktaları
> Mission Planner'dan yüklenir; mission_manager `/mavros/mission/reached` ile
> WP sıralarını takip edip parkur geçişini otomatik yapar.

## ZED2i topic doğrulaması

zed-ros2-wrapper topic ön-eki launch'a göre `/zed/zed_node` **veya**
`/zed2i/zed_node` olabilir. İlk açılışta doğrula:
```bash
ros2 topic list | grep zed
ros2 topic info /zed/zed_node/point_cloud/cloud_registered --verbose
```
Farklıysa `mission_params.yaml` içinde `cloud_topic` / `image_topic` güncelle.
Tüm cloud düğümleri sensor QoS kullanıyor (ZED BEST_EFFORT yayınlar).

## Jetson Orin Nano performans notları

Orin Nano giriş seviyesi (Orin NX/AGX'e göre zayıf); üç ağır iş (ZED derinlik +
point cloud + YOLO) aynı anda koşacağı için optimizasyon önemli:
- Cloud düğümleri her noktayı Python'da işliyor → `point_stride` ≥ 2 tutuldu.
  Gerekirse 3-4'e çıkar (daha az nokta, daha hızlı ama daha kaba engel).
- YOLO modelini **TensorRT (.engine)** olarak export et; `imgsz`'i 416/512'ye
  düşürmek FPS'i belirgin artırır. `process_every_n_frames` ile kare atla.
- ZED çözünürlüğünü HD720 yerine gerekirse VGA'ya düşür; derinlik modunu
  PERFORMANCE seç.
- `nvpmodel -m 0` + `jetson_clocks` ile tam güç modunu aç (start_mission.sh'te
  açılabilir).
