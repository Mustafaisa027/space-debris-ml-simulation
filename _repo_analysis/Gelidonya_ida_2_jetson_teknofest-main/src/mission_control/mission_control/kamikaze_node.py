#!/usr/bin/env python3
"""
kamikaze_node — Parkur-3 angajman dugumu (ROS sarmalayicisi)
=============================================================

Hedefleme mantigi bu dosyada DEGIL: kamikaze_color_controller.py icindeki
KamikazeColorController sinifinda. O sinif ROS'tan bagimsizdir (saf Python),
boylece birim testi yazilabilir ve simulasyonda ayrica kosturulabilir.

Bu dugumun isi:
  1. /yolo/detections metnini Detection nesnelerine cevirmek
  2. /iha/target_color'dan gelen rengi kontrolcuye vermek (set_color_code)
  3. Her karede controller.update() cagirip cikan soyut komutlari
     (forward_command 0..1, turn_command -1..+1) gercek hiz komutuna cevirmek
  4. GUIDED icin setpoint'i SABIT HIZDA akitmak (ArduPilot surekli komut ister)
  5. Temas/angajman tespit edip /kamikaze/status yayinlamak
     (mission_manager bunu gorunce STATE_DONE'a gecer)

ISARET DONUSUMU — DIKKAT
  Kontrolcu:  turn_command  pozitif = SAGA don
  ROS/REP-103: angular.z    pozitif = SOLA don (saat yonu tersi)
  Bu yuzden angular.z = -turn_command * yaw_scale  (isaret ters cevrilir).
  Yanlis isaret araci hedeften uzaga cevirir; en sinsi hatalardan biridir.

GUVENLIK
  send_velocity varsayilan False. Kuru testte karar/durum uretilir ama motora
  komut gitmez. Suda GUIDED denemesinde bilincli olarak True yapilir.
"""

import re

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist

from mission_interfaces.msg import TargetColor, KamikazeStatus, MissionState

from .common import Topics, Services, COMMAND_QOS, STATE_QOS
from .kamikaze_color_controller import (
    KamikazeColorController,
    Detection,
    COLOR_CODE_TO_CLASS,
)

# zed_yolo_node ciktisi:
# "class=red, conf=0.92, bbox=(x1,y1,x2,y2), center=(cx,cy)"
DET_RE = re.compile(
    r'class=(?P<cls>[^,]+),\s*conf=(?P<conf>[0-9.]+),\s*'
    r'bbox=\((?P<x1>-?\d+),(?P<y1>-?\d+),(?P<x2>-?\d+),(?P<y2>-?\d+)\)'
)

# TargetColor sabitleri ile kontrolcunun renk kodlari birebir ayni
# (1=RED, 2=GREEN, 3=BLACK) — COLOR_CODE_TO_CLASS bunu dogrular.

# Kontrolcu durum metni -> KamikazeStatus faz sabiti
STATE_TO_PHASE = {
    'RENK_KODU_YOK': KamikazeStatus.PHASE_SEARCH,
    'HEDEF_DUBA_BULUNAMADI': KamikazeStatus.PHASE_SEARCH,
    'HEDEF_ORTALANIYOR': KamikazeStatus.PHASE_APPROACH,
    'HEDEF_MERKEZDE': KamikazeStatus.PHASE_APPROACH,
}


class KamikazeNode(Node):

    def __init__(self):
        super().__init__('kamikaze_node')

        # --- Kontrolcu parametreleri (kamikaze_color_controller'a gecer) ---
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('min_conf', 0.55)
        self.declare_parameter('center_tolerance_frac', 0.06)
        self.declare_parameter('turn_gain', 1.30)
        self.declare_parameter('max_turn', 0.75)

        # --- ROS tarafi: soyut komut -> gercek hiz olcekleme ---
        self.declare_parameter('max_speed_mps', 2.0)      # forward_command=1.0 karsiligi
        self.declare_parameter('max_yaw_rate_radps', 1.0)  # turn_command=1.0 karsiligi
        self.declare_parameter('setpoint_rate_hz', 10.0)   # GUIDED surekli akis
        self.declare_parameter('send_velocity', False)     # GUVENLIK: varsayilan kapali
        self.declare_parameter('require_parkur3', True)

        # --- Temas/angajman tespiti ---
        self.declare_parameter('impact_area_frac', 0.45)
        self.declare_parameter('impact_frames', 3)
        self.declare_parameter('terminal_area_frac', 0.25)  # TERMINAL faz esigi
        self.declare_parameter('lost_target_timeout_s', 2.0)

        # --- YOLO sinif adi -> kontrolcu sinif adi eslemesi ---
        # Model farkli isimler kullaniyorsa (ornegin 'kirmizi_duba') burada
        # eslenir. Bos birakilirsa dogrudan sinif adi kullanilir.
        self.declare_parameter('class_aliases', [''])

        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.max_speed = float(self.get_parameter('max_speed_mps').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate_radps').value)
        self.setpoint_rate = float(self.get_parameter('setpoint_rate_hz').value)
        self.send_velocity = bool(self.get_parameter('send_velocity').value)
        self.require_parkur3 = bool(self.get_parameter('require_parkur3').value)
        self.impact_area_frac = float(self.get_parameter('impact_area_frac').value)
        self.impact_frames = int(self.get_parameter('impact_frames').value)
        self.terminal_area_frac = float(self.get_parameter('terminal_area_frac').value)
        self.lost_timeout = float(self.get_parameter('lost_target_timeout_s').value)

        self.class_aliases = self._parse_aliases(
            self.get_parameter('class_aliases').value)

        # --- Hedefleme kontrolcusu (ROS'tan bagimsiz cekirdek) ---
        self.controller = KamikazeColorController(
            image_width=self.image_width,
            image_height=self.image_height,
            minimum_confidence=float(self.get_parameter('min_conf').value),
            center_tolerance_ratio=float(
                self.get_parameter('center_tolerance_frac').value),
            turn_gain=float(self.get_parameter('turn_gain').value),
            maximum_turn=float(self.get_parameter('max_turn').value),
        )

        # --- Durum ---
        self.enabled = False
        self.in_parkur3 = False
        self.target_color = TargetColor.UNKNOWN
        self.engaged = False
        self._impact_counter = 0
        self._last_seen_time = None
        self._last_twist = Twist()

        cb = MutuallyExclusiveCallbackGroup()

        self.status_pub = self.create_publisher(
            KamikazeStatus, Topics.KAMIKAZE_STATUS, COMMAND_QOS)
        self.vel_pub = self.create_publisher(
            Twist, Topics.MAVROS_SETPOINT_VEL, COMMAND_QOS)

        self.create_subscription(
            String, Topics.YOLO_DETECTIONS, self.on_detections,
            COMMAND_QOS, callback_group=cb)
        self.create_subscription(
            TargetColor, Topics.IHA_TARGET_COLOR, self.on_target_color,
            STATE_QOS, callback_group=cb)
        self.create_subscription(
            MissionState, Topics.MISSION_STATE, self.on_mission_state,
            STATE_QOS, callback_group=cb)

        self.create_service(
            SetBool, Services.KAMIKAZE_ENABLE, self.srv_enable,
            callback_group=cb)

        self.create_timer(
            1.0 / max(1.0, self.setpoint_rate), self.setpoint_loop,
            callback_group=cb)

        self.get_logger().info(
            f'kamikaze_node basladi. send_velocity={self.send_velocity} '
            f'max_speed={self.max_speed:.1f} m/s'
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_aliases(raw):
        """['kirmizi_duba:red', 'yesil_duba:green'] -> {'kirmizi_duba': 'red'}"""
        aliases = {}
        for item in raw or []:
            if ':' in item:
                src, dst = item.split(':', 1)
                aliases[src.strip().lower()] = dst.strip().lower()
        return aliases

    # ------------------------------------------------------------------
    def srv_enable(self, request, response):
        self.enabled = bool(request.data)
        if not self.enabled:
            self._publish_stop()
        self.get_logger().info(f'Kamikaze {"ACIK" if self.enabled else "KAPALI"}')
        response.success = True
        response.message = f'kamikaze enabled={self.enabled}'
        return response

    def on_target_color(self, msg: TargetColor):
        if msg.color == self.target_color:
            return
        self.target_color = msg.color
        if self.controller.set_color_code(int(msg.color)):
            self.get_logger().info(
                f'Angajman rengi: {COLOR_CODE_TO_CLASS.get(int(msg.color))}')
        else:
            self.get_logger().warn(f'Gecersiz renk kodu: {msg.color}')

    def on_mission_state(self, msg: MissionState):
        self.in_parkur3 = (msg.state == MissionState.STATE_PARKUR3)

    def _active(self) -> bool:
        if not self.enabled:
            return False
        if self.require_parkur3 and not self.in_parkur3:
            return False
        return self.controller.target_class is not None

    # ------------------------------------------------------------------
    def _parse_detections(self, text: str):
        """YOLO metin ciktisini Detection listesine cevirir."""
        detections = []
        for m in DET_RE.finditer(text):
            cls = m.group('cls').strip().lower()
            cls = self.class_aliases.get(cls, cls)
            detections.append(Detection(
                class_name=cls,
                confidence=float(m.group('conf')),
                x1=float(m.group('x1')),
                y1=float(m.group('y1')),
                x2=float(m.group('x2')),
                y2=float(m.group('y2')),
            ))
        return detections

    def on_detections(self, msg: String):
        if not self._active() or self.engaged:
            return

        detections = self._parse_detections(msg.data)
        out = self.controller.update(detections)

        if not out.target_found:
            self._publish_status(out, phase=KamikazeStatus.PHASE_SEARCH)
            return

        self._last_seen_time = self.get_clock().now()
        area = out.target_area_ratio

        # --- Temas tespiti: hedef kutusu goruntuyu doldurdu ve merkezde ---
        centered = abs(out.horizontal_error) <= self.controller.center_tolerance_ratio
        if area >= self.impact_area_frac and centered:
            self._impact_counter += 1
        else:
            self._impact_counter = 0

        if self._impact_counter >= self.impact_frames:
            self.engaged = True
            self._last_twist = Twist()     # motoru kes
            self._publish_stop()
            self._publish_status(out, phase=KamikazeStatus.PHASE_IMPACT, engaged=True)
            self.get_logger().warn(
                f'ANGAJMAN: hedefe temas (alan={area:.2f}). Motor kesildi.')
            return

        # --- Soyut komutlari gercek hiz komutuna cevir ---
        twist = Twist()
        twist.linear.x = out.forward_command * self.max_speed
        # ISARET TERS: kontrolcude + = saga, ROS'ta + = sola
        twist.angular.z = -out.turn_command * self.max_yaw_rate
        self._last_twist = twist

        phase = STATE_TO_PHASE.get(out.state, KamikazeStatus.PHASE_APPROACH)
        if phase == KamikazeStatus.PHASE_APPROACH and area >= self.terminal_area_frac \
                and centered:
            phase = KamikazeStatus.PHASE_TERMINAL

        self._publish_status(out, phase=phase)

    # ------------------------------------------------------------------
    def setpoint_loop(self):
        """GUIDED icin sabit hizda setpoint akisi + hedef kayip emniyeti.

        ArduPilot GUIDED modda surekli komut bekler; tespit geldiginde tek
        seferlik yayin yeterli degildir. Hedef 'lost_timeout' suresince
        gorunmezse sifir hiz akitilir (guvenli dur, akis surer).
        """
        if not self._active() or self.engaged or not self.send_velocity:
            return

        if self._last_seen_time is None:
            self.vel_pub.publish(Twist())
            return

        elapsed = (self.get_clock().now() - self._last_seen_time).nanoseconds / 1e9
        if elapsed <= self.lost_timeout:
            self.vel_pub.publish(self._last_twist)
        else:
            self.vel_pub.publish(Twist())

    def _publish_stop(self):
        if self.send_velocity:
            self.vel_pub.publish(Twist())

    def _publish_status(self, out, phase, engaged=False):
        msg = KamikazeStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'kamikaze'
        msg.target_locked = out.target_found
        msg.target_color = self.target_color
        # Goreli kerteriz: yatay hata -1..+1 -> derece (+ sol, - sag)
        msg.bearing_deg = float(-out.horizontal_error * 45.0)
        msg.image_center_error = float(
            (out.target_center_x - self.image_width / 2.0)
            if out.target_center_x is not None else 0.0
        )
        msg.distance_m = -1.0
        msg.phase = phase
        msg.engaged = engaged
        msg.info = (
            f'{out.state} alan={out.target_area_ratio:.3f} '
            f'ileri={out.forward_command:.2f} donus={out.turn_command:+.2f}'
        )
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KamikazeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
