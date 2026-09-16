#!/usr/bin/env python3
"""
sim_target_node — Parkur-3 icin sahte YOLO tespiti uretir
==========================================================

Gercek YOLO modeli hazir olmadan kamikaze_node'un angajman mantigini
(hedef secimi, kerteriz hesabi, faz gecisleri, temas tespiti) test etmek icin.

zed_yolo_node ile AYNI metin formatinda /yolo/detections yayinlar:
    class=<ad>, conf=<0-1>, bbox=(x1,y1,x2,y2), center=(cx,cy)

Senaryo: hedef baslangicta goruntunun kenarinda ve kucuktur; zamanla merkeze
kayar ve buyur — yani araci hedefe yaklasiyormus gibi gosterir. Kamikaze'nin
SEARCH -> APPROACH -> TERMINAL -> IMPACT gecislerini tetikler.

Ek olarak yanlis renkte bir "capan" hedef de yayinlanabilir (distractor):
kamikaze'nin dogru rengi secip digerini yok saydigi dogrulanir.

Kullanim:
  ros2 run mission_control sim_target_node --ros-args -p target_class:=red
  ros2 service call /sim_target/start std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool

from mission_interfaces.msg import MissionState

from .common import Topics, COMMAND_QOS, STATE_QOS


class SimTargetNode(Node):

    def __init__(self):
        super().__init__('sim_target_node')

        self.declare_parameter('target_class', 'red')     # angaje olunacak renk
        self.declare_parameter('distractor_class', 'green')  # yanlis hedef
        self.declare_parameter('publish_distractor', True)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('approach_time_s', 12.0)   # temasa kadar sure
        self.declare_parameter('start_offset_frac', 0.30) # baslangic yanal sapma
        # Senaryo ne zaman baslasin?
        #   start_on_parkur3=True (varsayilan): /mission/state PARKUR3 olunca.
        #     Aksi halde senaryo kamikaze acilmadan bitip test bosa gider.
        #   auto_start=True: dugum acilir acilmaz basla (izole hedef testi).
        self.declare_parameter('start_on_parkur3', True)
        self.declare_parameter('auto_start', False)

        self.target_class = str(self.get_parameter('target_class').value)
        self.distractor_class = str(self.get_parameter('distractor_class').value)
        self.publish_distractor = bool(self.get_parameter('publish_distractor').value)
        self.w = int(self.get_parameter('image_width').value)
        self.h = int(self.get_parameter('image_height').value)
        rate = max(1.0, float(self.get_parameter('rate_hz').value))
        self.approach_time = max(1.0, float(self.get_parameter('approach_time_s').value))
        self.start_offset = float(self.get_parameter('start_offset_frac').value)
        self.running = bool(self.get_parameter('auto_start').value)

        self.start_on_parkur3 = bool(self.get_parameter('start_on_parkur3').value)

        self.pub = self.create_publisher(String, Topics.YOLO_DETECTIONS, COMMAND_QOS)

        if self.start_on_parkur3:
            self.create_subscription(
                MissionState, Topics.MISSION_STATE, self.on_mission_state, STATE_QOS)

        self.create_service(Trigger, '/sim_target/start', self.srv_start)
        self.create_service(Trigger, '/sim_target/reset', self.srv_reset)
        self.create_service(SetBool, '/sim_target/visible', self.srv_visible)

        self.visible = True
        self.t = 0.0
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self.tick)

        self.get_logger().info(
            f'sim_target basladi. hedef={self.target_class} '
            f'capan={self.distractor_class if self.publish_distractor else "yok"} '
            f'yaklasma={self.approach_time:.0f}s'
        )

    # ------------------------------------------------------------------
    def on_mission_state(self, msg: MissionState):
        """PARKUR3'e girilince senaryoyu basa alip baslat."""
        if msg.state == MissionState.STATE_PARKUR3 and not self.running:
            self.t = 0.0
            self.visible = True
            self.running = True
            self.get_logger().info('PARKUR3 algilandi -> sahte hedef akisi basladi')

    def srv_start(self, request, response):
        self.running = True
        self.t = 0.0
        response.success = True
        response.message = 'sahte hedef akisi basladi'
        return response

    def srv_reset(self, request, response):
        self.t = 0.0
        response.success = True
        response.message = 'senaryo basa alindi'
        return response

    def srv_visible(self, request, response):
        """Hedefi gizle/goster — kamikaze'nin 'hedef kayboldu' davranisi testi."""
        self.visible = bool(request.data)
        response.success = True
        response.message = f'hedef gorunur={self.visible}'
        return response

    # ------------------------------------------------------------------
    def _make_detection(self, class_name, cx, cy, box_w, box_h, conf):
        x1 = int(max(0, cx - box_w / 2))
        y1 = int(max(0, cy - box_h / 2))
        x2 = int(min(self.w, cx + box_w / 2))
        y2 = int(min(self.h, cy + box_h / 2))
        return (f'class={class_name}, conf={conf:.2f}, '
                f'bbox=({x1},{y1},{x2},{y2}), center=({int(cx)},{int(cy)})')

    def tick(self):
        if not self.running:
            return

        self.t += self.dt
        # p: 0 -> 1 arasi yaklasma orani (1 = temas)
        p = min(1.0, self.t / self.approach_time)

        detections = []

        if self.visible:
            # Yanal sapma zamanla kapanir (arac hedefi ortaliyor)
            offset = self.start_offset * (1.0 - p) * self.w
            cx = self.w / 2.0 + offset
            cy = self.h / 2.0

            # Kutu alani buyur: goruntunun %2'sinden %55'ine
            area_frac = 0.02 + p * p * 0.53
            box_area = area_frac * self.w * self.h
            box_w = (box_area * (self.w / self.h)) ** 0.5
            box_h = box_area / box_w

            detections.append(self._make_detection(
                self.target_class, cx, cy, box_w, box_h, 0.70 + 0.25 * p))

        if self.publish_distractor:
            # Capan hedef: sabit, kenarda, kucuk — kamikaze bunu SECMEMELI
            detections.append(self._make_detection(
                self.distractor_class, self.w * 0.15, self.h * 0.55,
                self.w * 0.08, self.h * 0.12, 0.85))

        msg = String()
        msg.data = ' | '.join(detections) if detections else 'no_detection'
        self.pub.publish(msg)

        if p >= 1.0 and self.running:
            self.running = False
            self.get_logger().info(
                'Senaryo tamamlandi (hedef goruntuyu doldurdu). '
                '/sim_target/start ile tekrar calistir.'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimTargetNode()
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
