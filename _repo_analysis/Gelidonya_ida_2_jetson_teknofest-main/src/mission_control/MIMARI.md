# Gelidonya İDA — Görev Mimarisi

> Test için: kök dizindeki **TEST_CHECKLIST.md** (masa → sensör → FC → su, aşamalı).

3 parkuru **tek seferde**, operatör girişi olmadan **otomatik geçişle** yöneten
ROS2 katmanı. Şartname 5.5.2.2: *"Parkurlar arası geçiş kullanıcı girişi olmadan
otomatik olarak algılanacak ve yapılacaktır."*

## Durum makinesi (mission_manager_node)

```
BOOT ──► READY ──► PARKUR1 ──► PARKUR2 ──► PARKUR3 ──► DONE
  │        │          │           │           │
  │        │          │           │           └─ kamikaze.engaged=True
  │        │          │           └─ WP5'e ulaşıldı (parkur2_last_wp)
  │        │          └─ WP4'e ulaşıldı (parkur1_last_wp)
  │        └─ araç ARM + AUTO mod (START = YKİ/RC)
  └─ MAVROS bağlı + hedef renk atandı
                                          herhangi an ──► EMERGENCY (kill switch)
```

| Parkur | Uçuş modu | Ne olur | Kaçınma | Kamikaze |
|--------|-----------|---------|---------|----------|
| BAŞLA | AUTO+ARM | operatör güç verip AUTO'ya alıp arm eder → görev başlar | — | — |
| PARKUR1 | **AUTO** | WP1→WP4 nokta takip (ArduPilot mission) | KAPALI | KAPALI |
| PARKUR2 | **GUIDED** | Jetson WP5'i hedef verir, ArduPilot manevra yapar | **AÇIK** (BendyRuler) | KAPALI |
| PARKUR3 | **GUIDED** | İHA renk + YOLO ile hedefe angajman (companion sürer) | KAPALI | **AÇIK** |
| EMERGENCY | **HOLD** | acil durdurma | KAPALI | KAPALI |

Uçuş modunu `mission_manager` her parkur geçişinde `/mavros/set_mode` ile kendisi
ayarlar (`manage_flight_mode: true`) ve `/mavros/state` ile doğrular; tutmazsa
saniyede bir yeniden dener. Başlangıç modu (AUTO) operatörden gelir.

### Parkur-2 GUIDED navigasyonu — görev bölüşümü

```
Jetson (ne yapılacak)              CUAV X7+ / ArduPilot (nasıl yapılacak)
─────────────────────              ──────────────────────────────────────
ZED2i → point cloud                BendyRuler: OBSTACLE_DISTANCE'a göre
  → filtre (0.30–3.00 m)             rotayı büker, engelden kaçar
  → 40 sektör → LaserScan          Motor/dümen kontrolü, hız profili
  → /mavros/obstacle/send          Hedefe seyir
WP5 konumu → GUIDED hedefi
  → /mavros/setpoint_position/global
```

Jetson **manevra yapmaz** — sadece "nereye gidilecek" (WP5) ve "nerede engel var"
(sektör mesafeleri) bilgisini verir. Direksiyon ArduPilot'ta.

WP5 konumu yüklü görevden okunur (`/mavros/mission/waypoints`). GUIDED'de
`mission/reached` tetiklenmediği için varış, WP5'e olan haversine mesafesi
`p2_arrival_radius_m` (3 m) altına düşünce algılanır.

> **GUIDED bağımlılığı:** Parkur-3'te aracın hedefe sürmesi için `kamikaze_node`
> `send_velocity: true` olmalı. GUIDED'de ArduPilot sürekli setpoint bekler;
> kamikaze bunu `setpoint_rate_hz` (varsayılan 10 Hz) ile sabit akıtır. Kuru
> testte `false` bırak, suda GUIDED denemesinde bilinçli `true` yap.

Geçiş tetikleyici: `/mavros/mission/reached` (wp_seq). WP indeksleri 0-tabanlı:
WP1..WP4 = seq 0..3, WP5 = seq 4.

## Düğümler

| Düğüm | Paket | Rol |
|-------|-------|-----|
| **mission_manager_node** | mission_control | Beyin. Durum makinesi, otomatik geçiş, alt-sistem aç/kapa |
| **iha_color_receiver_node** | mission_control | İHA'dan (UDP) renk alır → `/iha/target_color` |
| **kamikaze_node** | mission_control | YOLO + renk → hedefe yönelme kararı / hız komutu |
| **mission_logger_node** | mission_control | Tüm düğümleri dinler, JSONL+CSV kayıt (veri teslimi) |
| obstacle_bridge_node | zed_obstacle_map | Cloud→LaserScan→BendyRuler (P2). enable servisi eklendi |
| sector_avoidance_node | zed_obstacle_map | Companion-side kaçış (yedek/analiz) |
| pointcloud_filter_node | zed_obstacle_map | Filtreli cloud (RViz) |
| obstacle_map_node | zed_obstacle_map | OccupancyGrid (RViz) |
| zed_yolo_node | yolo_jetson | YOLO tespit → `/yolo/detections` (P3) |

## Topic'ler

| Topic | Tip | Yön | QoS |
|-------|-----|-----|-----|
| `/mission/state` | MissionState | manager→herkes | STATE (latch) |
| `/mission/emergency` | std_msgs/Bool | kill→manager | COMMAND |
| `/iha/target_color` | TargetColor | iha→manager,kamikaze | STATE (latch) |
| `/kamikaze/status` | KamikazeStatus | kamikaze→manager,logger | COMMAND |
| `/yolo/detections` | std_msgs/String | yolo→kamikaze | COMMAND |
| `/mavros/obstacle/send` | LaserScan | bridge→BendyRuler | default |
| `/obstacle_avoidance/decision` | std_msgs/String | sector→logger | COMMAND |
| `/mavros/state` | mavros_msgs/State | mavros→manager | SENSOR |
| `/mavros/mission/reached` | WaypointReached | mavros→manager | COMMAND |

## Servisler

| Servis | Tip | Amaç |
|--------|-----|------|
| `/mission/set_target_color` | SetTargetColor | Elle hedef renk (İHA'sız) |
| `/mission/set_parkur` | SetParkur | **Sadece test** — parkur zorla |
| `/mission/emergency_stop` | std_srvs/Trigger | Acil durdur |
| `/obstacle_bridge/enable` | std_srvs/SetBool | manager P2'de açar |
| `/kamikaze/enable` | std_srvs/SetBool | manager P3'te açar |

## Güvenlik / şartname notları

- **Hedef renk kilidi** (5.5.3.1): Görev başladıktan (PARKUR1) sonra gelen renk
  reddedilir. Hareket başlamadan atanmalı.
- **kamikaze_node.send_velocity** varsayılan **false**. Gerçek suda hız komutu
  göndermeden önce bilinçli olarak `true` yapın ve GUIDED mod ayarını doğrulayın.
- **QoS**: ZED BEST_EFFORT yayınlar; TÜM cloud abonelikleri artık
  `qos_profile_sensor_data` kullanıyor (obstacle_bridge + diğer 3 düğüm
  düzeltildi). Doğrula: `ros2 topic info <cloud_topic> --verbose`.
- **İHA renk taşıma = MAVLink STATUSTEXT**: İHA rengi MAVLink STATUSTEXT ile
  yollar → mavros `/mavros/statustext/recv` → iha_color_receiver metinde renk
  anahtar kelimesi arar ("COLOR:RED" veya sadece "red"). `source_mode` ile
  udp/both'a da alınabilir.

## Açık işler (öneri)

1. `zed_yolo_node` çıktısını `String` yerine `vision_msgs/Detection2DArray` yap
   (kamikaze parse'ı kırılgan).
2. Angajman "temas" tespiti şu an görüntü-alanı sezgiseli; IMU/bumper ile
   doğrulanması daha güvenilir olur.
3. İHA'nın STATUSTEXT'e tam olarak ne yazdığını netleştir (ör. "COLOR:RED").
4. `mission_logger` disk yazımı: flush timer'da `os.fsync` var; tek-thread'li
   executor'da bu kısa süreli bloklayabilir (ros2-robotics: blocking-callback).
   Düşük öncelik (yalnızca log düğümünü etkiler); maksimum sağlamlık için
   arka-plan yazıcı thread'i + bellek kuyruğu önerilir.

## ros2-robotics denetiminde düzeltilenler

- Tüm cloud abonelikleri (4 düğüm) + `zed_yolo` görüntü aboneliği → sensor QoS
  (qos-mismatch / default-qos-for-sensors).
- `mission_manager` uçuş modu artık `/mavros/state` ile doğrulanıp tutmazsa
  yeniden gönderiliyor (`call_async` gönder-unut riski kapatıldı).
- Tüm yeni düğümler topic sabitleri (`common.Topics`) + adlandırılmış QoS
  profilleri kullanıyor; `main()`'lerde try/finally + shutdown var.
