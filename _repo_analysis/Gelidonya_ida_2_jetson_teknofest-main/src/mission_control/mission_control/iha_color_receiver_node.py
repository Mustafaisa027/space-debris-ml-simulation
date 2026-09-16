#!/usr/bin/env python3
"""
iha_color_receiver_node
=======================

IHA (drone) tarafindan tespit edilen plaka rengini alip Parkur-3 icin
/iha/target_color topic'ine TargetColor olarak yayinlar.

Sartname 5.5.3.1:
  - Kiyiya birakilan DSB renkli plakanin rengi IHA tarafindan otomatik algilanir.
  - Angajman bolgesinde 3 farkli renkte hedef vardir (RAL 9005/3026/6037).
  - IDA, IHA'nin saptadigi renkteki hedefe angaje olur.
  - IHA'siz yarisilirsa hedef rengi Yarisma Alanina giris sonrasi ELLE verilir.

Giris yollari (source_mode ile secilir):
  1) MAVLINK (varsayilan): IHA rengi MAVLink STATUSTEXT ile yollar; mavros bunu
     /mavros/statustext/recv (mavros_msgs/StatusText) olarak yayinlar. Metin
     icinde renk anahtar kelimesi aranir (or. "COLOR:RED" veya sadece "red").
  2) UDP: drone/yer istasyonu bir UDP paketiyle renk yollar.
     Payload ornekleri:  "red" | "green" | "black" | "1" | "2" | "3"  (opsiyonel: "red,0.92")
  3) Elle: 'default_color' parametresi ya da /mission/set_target_color servisi.

source_mode: "mavlink" | "udp" | "both"
UDP dinleme ayri bir thread'de yapilir; ROS2 executor'unu bloklamaz
(ros2-robotics skill: blocking-callback).
"""

import socket
import threading

import rclpy
from rclpy.node import Node

from mission_interfaces.msg import TargetColor

from .common import Topics, SENSOR_QOS, STATE_QOS

# mavros_msgs Jetson'da mevcuttur; gelistirme makinesinde olmayabilir.
try:
    from mavros_msgs.msg import StatusText
    MAVROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    StatusText = None
    MAVROS_AVAILABLE = False


NAME_TO_COLOR = {
    'red': TargetColor.RED, 'kirmizi': TargetColor.RED, '1': TargetColor.RED,
    'ral3026': TargetColor.RED,
    'green': TargetColor.GREEN, 'yesil': TargetColor.GREEN, '2': TargetColor.GREEN,
    'ral6037': TargetColor.GREEN,
    'black': TargetColor.BLACK, 'siyah': TargetColor.BLACK, '3': TargetColor.BLACK,
    'ral9005': TargetColor.BLACK,
}

RAL_CODES = {
    TargetColor.RED: 'RAL 3026',
    TargetColor.GREEN: 'RAL 6037',
    TargetColor.BLACK: 'RAL 9005',
}
COLOR_NAMES = {
    TargetColor.RED: 'red',
    TargetColor.GREEN: 'green',
    TargetColor.BLACK: 'black',
}


class IhaColorReceiverNode(Node):

    def __init__(self):
        super().__init__('iha_color_receiver_node')

        self.declare_parameter('source_mode', 'mavlink')  # mavlink | udp | both
        self.declare_parameter('statustext_topic', Topics.MAVROS_STATUSTEXT)
        self.declare_parameter('udp_ip', '0.0.0.0')
        self.declare_parameter('udp_port', 5010)
        self.declare_parameter('default_color', '')  # 'red'/'green'/'black' veya bos
        self.declare_parameter('republish_period_s', 1.0)  # latch pekistirme

        self.source_mode = str(self.get_parameter('source_mode').value).strip().lower()
        self.statustext_topic = self.get_parameter('statustext_topic').value
        self.udp_ip = self.get_parameter('udp_ip').value
        self.udp_port = int(self.get_parameter('udp_port').value)
        default_color = str(self.get_parameter('default_color').value).strip().lower()
        self.republish_period = float(self.get_parameter('republish_period_s').value)

        self.udp_enabled = self.source_mode in ('udp', 'both')
        self.mavlink_enabled = self.source_mode in ('mavlink', 'both')

        # TRANSIENT_LOCAL: gec baglanan mission_manager da son rengi alir.
        self.pub = self.create_publisher(TargetColor, Topics.IHA_TARGET_COLOR, STATE_QOS)

        self._lock = threading.Lock()
        self._current = None   # (color, confidence, source)

        if default_color:
            color = NAME_TO_COLOR.get(default_color)
            if color:
                self._current = (color, 1.0, 'manual')
                self.get_logger().info(f'Varsayilan hedef renk: {default_color}')
            else:
                self.get_logger().warn(f'Gecersiz default_color: {default_color}')

        # Latch'i periyodik pekistir (yeni abone gelirse veri kaybolmasin).
        self.create_timer(self.republish_period, self._republish)

        # MAVLINK: mavros STATUSTEXT'ini dinle
        if self.mavlink_enabled:
            if MAVROS_AVAILABLE:
                # MAVROS STATUSTEXT'i BEST_EFFORT (sensor) QoS ile yayinlar.
                # RELIABLE abone olursak baglanti kurulmaz ve IHA rengi HIC
                # gelmez (ros2-robotics: qos-mismatch). MAVROS bunu acikca
                # uyariyor: "requesting incompatible QoS. No messages sent."
                self.create_subscription(
                    StatusText, self.statustext_topic,
                    self.on_statustext, SENSOR_QOS
                )
                self.get_logger().info(
                    f'MAVLink renk dinleniyor: {self.statustext_topic}'
                )
            else:
                self.get_logger().warn(
                    'source_mode=mavlink ama mavros_msgs yok. MAVLink girisi PASIF.'
                )

        self._sock = None
        self._udp_thread = None
        self._running = True
        if self.udp_enabled:
            self._start_udp()

        self.get_logger().info(
            f'iha_color_receiver basladi. mode={self.source_mode} '
            f'UDP={"acik" if self.udp_enabled else "kapali"} {self.udp_ip}:{self.udp_port}'
        )

    def on_statustext(self, msg):
        # STATUSTEXT serbest metin: icinde renk anahtar kelimesi ara.
        color = self._parse_color_from_text(msg.text)
        if color is None:
            return
        with self._lock:
            self._current = (color, 1.0, 'iha')
        self.get_logger().info(
            f'IHA renk alindi (MAVLink): {COLOR_NAMES[color]} <- "{msg.text}"'
        )
        self._republish()

    @staticmethod
    def _parse_color_from_text(text: str):
        """Serbest metinde (STATUSTEXT) bilinen renk anahtar kelimesini bul.

        Sadece harf-tabanli anahtarlar aranir; '1'/'2'/'3' gibi rakam anahtarlari
        serbest metinde yanlis eslesir (or. koordinat/sayi), o yuzden atlanir.
        """
        if not text:
            return None
        low = text.lower()
        for key, color in NAME_TO_COLOR.items():
            if key.isalpha() and key in low:   # red, kirmizi, green, yesil, black, siyah
                return color
        return None

    def _start_udp(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.udp_ip, self.udp_port))
            self._sock.settimeout(0.5)
        except OSError as e:
            self.get_logger().error(f'UDP baglanamadi: {e}')
            self._sock = None
            return
        self._udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._udp_thread.start()

    def _udp_loop(self):
        while self._running and self._sock is not None:
            try:
                data, _addr = self._sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_payload(data.decode('utf-8', errors='ignore').strip())

    def _handle_payload(self, payload: str):
        if not payload:
            return
        parts = payload.replace(' ', '').lower().split(',')
        key = parts[0]
        confidence = 1.0
        if len(parts) > 1:
            try:
                confidence = float(parts[1])
            except ValueError:
                confidence = 1.0
        color = NAME_TO_COLOR.get(key)
        if color is None:
            self.get_logger().warn(f'Taninmayan renk payload: "{payload}"')
            return
        with self._lock:
            self._current = (color, confidence, 'iha')
        self.get_logger().info(
            f'IHA renk alindi: {COLOR_NAMES[color]} (conf={confidence:.2f})'
        )
        self._republish()

    def _republish(self):
        with self._lock:
            if self._current is None:
                return
            color, confidence, source = self._current
        msg = TargetColor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'iha'
        msg.color = color
        msg.color_name = COLOR_NAMES.get(color, 'unknown')
        msg.ral_code = RAL_CODES.get(color, '')
        msg.confidence = float(confidence)
        msg.source = source
        self.pub.publish(msg)

    def destroy_node(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IhaColorReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
