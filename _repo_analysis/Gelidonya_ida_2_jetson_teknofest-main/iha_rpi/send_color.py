#!/usr/bin/env python3
"""
send_color.py  —  İHA Raspberry Pi tarafında çalışır.

İHA'nın tespit ettiği plaka rengini İDA'ya gönderir. İki mod:

  --mode mavlink : İDA telemetri linkine (RFD900x) MAVLink STATUSTEXT enjekte eder.
                   İDA'daki MAVROS bunu /mavros/statustext/recv olarak yayınlar,
                   iha_color_receiver_node oradan okur.
  --mode udp     : İDA Jetson'ı IP üzerinden erişilebilirse (kıyıda ortak ağ),
                   doğrudan UDP paketi yollar (iha_color_receiver udp/both modu).

Renk string'i: "red" | "green" | "black"  (İDA tarafı "COLOR:RED" gibi ekleri de tolere eder)

KURULUM (RPi):
    pip install pymavlink          # mavlink modu için

ÖRNEK — MAVLink (İDA telemetrisi RPi'ye /dev/ttyUSB0'da, RFD900x router'a bağlı):
    python3 send_color.py --mode mavlink --device /dev/ttyUSB0 --baud 57600 --color red

ÖRNEK — MAVLink (yerde mavlink-router UDP çıkışı 14550'de):
    python3 send_color.py --mode mavlink --device udpout:127.0.0.1:14550 --color green

ÖRNEK — UDP (İDA Jetson IP=192.168.1.50):
    python3 send_color.py --mode udp --ida-ip 192.168.1.50 --port 5010 --color black

NOT: Rengi görev boyunca tek sefer değil, İDA alana kadar birkaç kez tekrar
göndermek güvenlidir (--repeat ile). Yarışma kuralı 5.5.3.1 gereği rengin
İDA harekete başlamadan önce ulaşması hedeflenmeli — hakeme danış.
"""

import argparse
import socket
import sys
import time

VALID = {'red', 'kirmizi', 'green', 'yesil', 'black', 'siyah'}


def send_mavlink(device, baud, text, repeat, interval):
    try:
        from pymavlink import mavutil
    except ImportError:
        sys.exit('pymavlink yok. Kur: pip install pymavlink')

    # device 'udpout:ip:port' ya da '/dev/ttyUSB0' olabilir.
    if device.startswith(('udp', 'tcp')):
        master = mavutil.mavlink_connection(device)
    else:
        master = mavutil.mavlink_connection(device, baud=baud)

    # STATUSTEXT max 50 karakter. severity INFO=6.
    payload = text.encode('utf-8')[:50]
    for i in range(repeat):
        master.mav.statustext_send(
            mavutil.mavlink.MAV_SEVERITY_INFO,
            payload
        )
        print(f'[mavlink] gonderildi ({i + 1}/{repeat}): {text}')
        if i < repeat - 1:
            time.sleep(interval)


def send_udp(ida_ip, port, text, repeat, interval):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(repeat):
        sock.sendto(text.encode('utf-8'), (ida_ip, port))
        print(f'[udp] gonderildi ({i + 1}/{repeat}): {text} -> {ida_ip}:{port}')
        if i < repeat - 1:
            time.sleep(interval)
    sock.close()


def main():
    p = argparse.ArgumentParser(description='İHA RPi -> İDA renk gonderici')
    p.add_argument('--mode', choices=['mavlink', 'udp'], default='mavlink')
    p.add_argument('--color', required=True, help='red | green | black')
    p.add_argument('--prefix', default='COLOR:', help='STATUSTEXT on-eki (mavlink)')
    # mavlink
    p.add_argument('--device', default='/dev/ttyUSB0',
                   help='seri port ya da udpout:ip:port')
    p.add_argument('--baud', type=int, default=57600)
    # udp
    p.add_argument('--ida-ip', default='192.168.1.50')
    p.add_argument('--port', type=int, default=5010)
    # ortak
    p.add_argument('--repeat', type=int, default=5, help='kac kez gonderilsin')
    p.add_argument('--interval', type=float, default=1.0, help='tekrar araligi (s)')
    args = p.parse_args()

    color = args.color.strip().lower()
    if color not in VALID:
        sys.exit(f'Gecersiz renk: {color}. Secenek: red/green/black')

    if args.mode == 'mavlink':
        text = f'{args.prefix}{color.upper()}'
        send_mavlink(args.device, args.baud, text, args.repeat, args.interval)
    else:
        send_udp(args.ida_ip, args.port, color, args.repeat, args.interval)


if __name__ == '__main__':
    main()
