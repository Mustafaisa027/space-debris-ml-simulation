# İDA Yer İstasyonu — Renk Relay (RFD900x üzerinden)

İHA'nın RPi'de işlenen renk bilgisini, İDA'nın **tek linki olan RFD900x**
telemetrisi üzerinden İDA'ya (Jetson) ulaştırmak için kurulum.

## Neden MAVProxy?

Mission Planner, RFD900x'in COM portunu tek başına açar. Aynı porttan ikinci bir
program (renk enjektörü) veri gönderemez. **MAVProxy** portu çoğullar: RFD900x'e
tek bağlanır, birden fazla UDP çıkışı açar. Mission Planner ve `send_color.py`
bu UDP uçlarına bağlanır. MAVProxy Windows + Linux'ta çalışır.

```
İDA laptop (Windows/Linux)
  MAVProxy ── COM/USB ── RFD900x ── (hava) ── İDA CUAV X7+
     ├─ udp:14550  ← Mission Planner buraya bağlanır
     └─ udpin:14551 ← send_color.py (RPi veya laptop) buraya enjekte eder
```

## 1) MAVProxy kur ve çalıştır (İDA laptop)

```bash
pip install MAVProxy
```

RFD900x portunu bul (Windows: Aygıt Yöneticisi → COMx; Linux: /dev/ttyUSB0) ve:

```bash
# Windows örnek (COM5, 57600 baud):
mavproxy.py --master=COM5,57600 --out=udp:127.0.0.1:14550 --out=udpin:0.0.0.0:14551

# Linux örnek:
mavproxy.py --master=/dev/ttyUSB0,57600 --out=udp:127.0.0.1:14550 --out=udpin:0.0.0.0:14551
```

- `--out=udp:127.0.0.1:14550` → **Mission Planner** bu porta bağlanır.
- `--out=udpin:0.0.0.0:14551` → **renk enjektörü** buraya gönderir (dışarıdan da erişilebilir).

## 2) Mission Planner'ı MAVProxy'ye bağla

Mission Planner üstte bağlantı türü **UDP** seç, port **14550**, Connect.
(Artık RFD900x'e doğrudan değil, MAVProxy üzerinden bağlısın.)

## 3) Rengi enjekte et

**RPi'den** (RPi ile laptop yerel ağda; laptop IP = 192.168.1.10 varsayalım):
```bash
python3 send_color.py --mode mavlink --device udpout:192.168.1.10:14551 --color red
```

**Ya da laptop'un kendisinden** (RPi rengi laptop'a iletmişse):
```bash
python3 send_color.py --mode mavlink --device udpout:127.0.0.1:14551 --color red
```

`send_color.py` STATUSTEXT `"COLOR:RED"` yollar; MAVProxy → RFD900x → CUAV X7+.

## 4) ArduPilot: STATUSTEXT companion'a forward edilsin

CUAV X7+, Jetson'a bağlı seri portta (mavros) MAVLink forward AÇIK olmalı
(varsayılan açık). Mission Planner → Full Parameter List:
- Jetson'ın bağlı olduğu `SERIALx_PROTOCOL = 2` (MAVLink2)
- `SERIALx_OPTIONS` içindeki **"Don't forward mavlink"** biti KAPALI olmalı (0)

Böylece RFD900x'ten (TELEM) gelen broadcast STATUSTEXT, Jetson portuna iletilir.

## 5) Doğrula (İDA Jetson)

```bash
ros2 topic echo /mavros/statustext/recv
```
RPi/laptop'tan `send_color.py` çalıştırınca burada `COLOR:RED` görmelisin.
Ardından:
```bash
ros2 topic echo /iha/target_color
```
`color: 1` (kırmızı) görürsen zincir tamam. mission_manager Parkur-3'te bu rengi
kullanır.

## Sorun giderme

| Belirti | Olası neden |
|---------|-------------|
| `/mavros/statustext/recv`'de görünmüyor | ArduPilot forward kapalı (SERIALx_OPTIONS) ya da MAVProxy enjektör portu yanlış |
| MAVProxy'ye enjekte oluyor ama araca gitmiyor | `--out=udpin:0.0.0.0:14551` yerine yanlış yön; send_color `udpout:` kullanmalı |
| `/iha/target_color` boş | iha_color_receiver `source_mode: mavlink` mi? metinde renk kelimesi var mı? |
| Renk reddedildi (log: "gorev basladiktan sonra") | İDA harekete başladı; renk görev başlamadan gönderilmeli (kural 5.5.3.1) |
