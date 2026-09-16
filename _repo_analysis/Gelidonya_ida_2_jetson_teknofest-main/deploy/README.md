# Boot'ta otomatik başlatma (Jetson Orin Nano)

Güç verilince tüm düğümlerin (ZED2i + MAVROS + görev düğümleri) otomatik
başlaması için systemd servisi.

## Önce: manuel test et

Servisi kurmadan önce launch'ın elle çalıştığını doğrula:
```bash
cd ~/obstacle_map_ws
colcon build --packages-select mission_interfaces && source install/setup.bash
colcon build && source install/setup.bash
ros2 launch mission_control system_bringup.launch.py
```
Düğümler ayağa kalkıyor mu, `ros2 node list` ile bak. Sorun yoksa devam.

## Kurulum

1. Başlatma script'ini çalıştırılabilir yap ve yolunu doğrula
   (`start_mission.sh` içindeki `WORKSPACE` ve `FCU_URL`'yi kendine göre ayarla):
   ```bash
   cp ~/obstacle_map_ws/src/.../deploy/start_mission.sh ~/obstacle_map_ws/start_mission.sh
   chmod +x ~/obstacle_map_ws/start_mission.sh
   ```

2. Servisi kur ve etkinleştir:
   ```bash
   sudo cp deploy/gelidonya-mission.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable gelidonya-mission.service
   sudo systemctl start gelidonya-mission.service
   ```

3. Artık her boot'ta otomatik başlar. Yeniden başlatıp test et:
   ```bash
   sudo reboot
   ```

## İzleme / kontrol

```bash
systemctl status gelidonya-mission      # durum
journalctl -u gelidonya-mission -f       # canlı log
sudo systemctl restart gelidonya-mission # yeniden başlat
sudo systemctl stop gelidonya-mission    # durdur
sudo systemctl disable gelidonya-mission # otomatik başlatmayı kapat
```

## Notlar

- **Güvenli:** Düğümler boot'ta başlar ama görev, sen AUTO'ya alıp ARM edene
  kadar STATE_READY'de bekler. Motor hareket etmez.
- Seri port izni gerekebilir: `sudo usermod -a -G dialout gelidonya_2` (bir kez).
- ZED USB'nin hazır olması için `ExecStartPre=/bin/sleep 12` var; gerekiyorsa artır.
- `nvpmodel -m 0` + `jetson_clocks` için start_mission.sh içindeki satırları aç
  (sudoers'a şifresiz izin gerekebilir).
- Kablo/port değişirse `FCU_URL` (start_mission.sh) ve gerekiyorsa launch
  argümanlarını güncelle.
