# Şartname Veri Teslimi — 3 Dosya

Şartname, İDA karaya alındıktan sonra **20 dakika içinde** USB ile teslim
edilmesini istiyor. Teslim edilmeyen her dosya için **5 ceza puanı** (5.5.4.3.5).

Tüm kayıtlar `~/gelidonya_logs/` altına, çalıştırma zamanı damgalı olarak yazılır.

## Üretilen dosyalar

| Şartname | Dosya | Üreten düğüm | Frekans |
|---|---|---|---|
| **Dosya 1** — işlenmiş kamera | `dosya1_kamera_<ts>.mp4` | `video_recorder_node` | 10 Hz (min 1) |
| **Dosya 2** — araç telemetrisi | `telemetri_<ts>.csv` | `telemetry_logger_node` | 5 Hz (min 1) |
| **Dosya 3** — engel haritası | `dosya3_engel_haritasi_<ts>.mp4` | `video_recorder_node` | 10 Hz (min 1) |

Ek olarak (teslim zorunlu değil, analiz için): `mission_<ts>.jsonl`,
`events_<ts>.log`, `track_<ts>.csv` — `mission_logger_node` üretir.

## Dosya 1 — işlenmiş kamera verisi

- Kaynak: `/yolo/annotated_image` (YOLO bounding box + sınıf adı zaten çizili)
- Her kareye sol üste `IDA YYYY-MM-DD HH:MM:SS.mmm` zaman etiketi basılır
- mp4 (mp4v codec)

Şartname "tespit ve takip işlemleri sonucunda obje çerçeve çizimleri ve tespit
sınıf bilgileri görünecek şekilde" diyor — YOLO düğümü `result.plot()` ile bunu
çiziyor, kaydedici üstüne zaman damgası ekliyor.

## Dosya 2 — araç telemetri verisi (CSV)

İlk satır header. Kolonlar şartnamenin istediği alanları karşılar:

| Kolon | Şartname karşılığı | Kaynak |
|---|---|---|
| `timestamp_iso`, `unix_time` | zaman | sistem saati |
| `lat`, `lon` | konum | `/mavros/global_position/global` |
| `ground_speed_mps` | hız (yer hızı) | `/mavros/local_position/velocity_body` |
| `roll_deg`, `pitch_deg`, `heading_deg` | yönelim açıları | `/mavros/imu/data`, `compass_hdg` |
| `speed_setpoint_mps` | hız set pointi | `/mavros/setpoint_velocity/cmd_vel_unstamped` |
| `yaw_setpoint_degps` | yön set pointi | aynı topic (`angular.z`) |

Veri gelmeyen alan boş bırakılır, satır yine yazılır (frekans garantisi).

> Not: set point kolonları Parkur-3'te (GUIDED, kamikaze sürerken) dolar.
> Parkur-1/2'de manevrayı ArduPilot yaptığı için companion set point üretmez;
> bu satırlarda alanlar boş kalır. Hakem sorarsa açıklaması budur.

## Dosya 3 — lokal harita / engel haritası

- Kaynak: `/zed_obstacle_map/grid` (`nav_msgs/OccupancyGrid`, 8×8 m, 10 cm çözünürlük)
- Renk: dolu hücre kırmızı, boş koyu, bilinmeyen gri
- Araç alt-ortada sarı nokta, altta çözünürlük bilgisi, üstte zaman etiketi
- mp4

## Kayıt kontrolü

Kayıt düğümleri `mission_bringup` ile otomatik başlar. Kapatmak için:

```bash
ros2 launch mission_control mission_bringup.launch.py use_recording:=false
```

Görev sonrası dosyaları kontrol et:

```bash
ls -lht ~/gelidonya_logs/ | head
```

Video sürelerini ve kare sayısını doğrula:

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 ~/gelidonya_logs/dosya1_kamera_*.mp4
```

CSV satır sayısı (≈ süre × 5 olmalı):

```bash
wc -l ~/gelidonya_logs/telemetri_*.csv
```

## USB'ye kopyalama (yarışma sonrası)

```bash
cp ~/gelidonya_logs/dosya1_kamera_*.mp4 ~/gelidonya_logs/telemetri_*.csv ~/gelidonya_logs/dosya3_engel_haritasi_*.mp4 /media/$USER/USB/
```

⚠️ **Kayıt düğümü kapanırken mp4 dosyasını sonlandırır.** Jetson'ın fişini
çekmeden önce `Ctrl+C` ile launch'ı düzgün kapat, yoksa video dosyası bozuk
kalabilir. systemd kullanıyorsan:

```bash
sudo systemctl stop gelidonya-mission
```

## Bilinen sınırlama

Şartname "diğer otonomi sensörleri" için de mp4 istiyor. ZED2i'nin point cloud
verisi Dosya 3'teki engel haritasına dönüştürülerek sunuluyor; ham nokta bulutu
ayrıca video olarak kaydedilmiyor. Hakem ayrı bir sensör videosu isterse
`pointcloud_filter_node` çıktısı RViz üzerinden kaydedilebilir — bu durumda
ekran kaydı gerekir.
