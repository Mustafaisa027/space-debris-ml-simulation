#!/usr/bin/env python3
"""
mission_logger_node
===================

Tum kritik dugumlere baglanip yarisma boyunca yapisal (JSONL) kayit tutar.
Sartname 5.5.4.3.5 geregi IDA karaya alindiktan sonra veri teslimi gerekiyor;
bu dugum o teslim dosyalarini uretir.

Kaydedilenler (her satir bir olay, ISO zaman damgali):
  - /mission/state           gorev durumu / parkur gecisleri
  - /kamikaze/status         angajman
  - /iha/target_color        hedef renk
  - /obstacle_avoidance/decision  kacis kararlari
  - /obstacle_avoidance/sectors   sektor acikliklari (ozet)
  - /yolo/detections         tespitler
  - /mavros/state            arm/mod
  - /mavros/global_position/global  GPS izi

Ciktilar (out_dir icine, calistirma zamani damgali):
  mission_<ts>.jsonl   -> tum olaylar
  track_<ts>.csv       -> GPS izi (harita/analiz icin)
  events_<ts>.log      -> insan-okur ozet (parkur gecisleri, angajman)

NOT: Yuksek-hacimli ham veri (point cloud, goruntu) icin bu dugum yerine
'ros2 bag record' kullanilmasi onerilir; bu dugum karar/durum/iz katmanini tutar.
Dosyalar her N saniyede flush edilir; ani guc kesintisinde veri kaybi minimum olur.
"""

import json
import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import String, Float32MultiArray

from mission_interfaces.msg import MissionState, TargetColor, KamikazeStatus

from .common import Topics, SENSOR_QOS, COMMAND_QOS, STATE_QOS

try:
    from mavros_msgs.msg import State as MavrosState
    from sensor_msgs.msg import NavSatFix
    MAVROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    MavrosState = None
    NavSatFix = None
    MAVROS_AVAILABLE = False


class MissionLoggerNode(Node):

    def __init__(self):
        super().__init__('mission_logger_node')

        default_dir = os.path.expanduser('~/gelidonya_logs')
        self.declare_parameter('out_dir', default_dir)
        self.declare_parameter('flush_period_s', 2.0)
        self.declare_parameter('sectors_log_every', 10)  # sektor mesaji seyreklestir

        self.out_dir = os.path.expanduser(str(self.get_parameter('out_dir').value))
        flush_period = float(self.get_parameter('flush_period_s').value)
        self.sectors_every = int(self.get_parameter('sectors_log_every').value)

        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.jsonl_path = os.path.join(self.out_dir, f'mission_{ts}.jsonl')
        self.track_path = os.path.join(self.out_dir, f'track_{ts}.csv')
        self.events_path = os.path.join(self.out_dir, f'events_{ts}.log')

        self._jsonl = open(self.jsonl_path, 'a', encoding='utf-8')
        self._events = open(self.events_path, 'a', encoding='utf-8')
        self._track = open(self.track_path, 'a', newline='', encoding='utf-8')
        self._track_csv = csv.writer(self._track)
        self._track_csv.writerow(['iso_time', 'lat', 'lon', 'alt'])

        self._last_state = None
        self._sectors_counter = 0

        cb = MutuallyExclusiveCallbackGroup()

        self.create_subscription(MissionState, Topics.MISSION_STATE,
                                 self.on_state, STATE_QOS, callback_group=cb)
        self.create_subscription(KamikazeStatus, Topics.KAMIKAZE_STATUS,
                                 self.on_kamikaze, COMMAND_QOS, callback_group=cb)
        self.create_subscription(TargetColor, Topics.IHA_TARGET_COLOR,
                                 self.on_color, STATE_QOS, callback_group=cb)
        self.create_subscription(String, Topics.AVOID_DECISION,
                                 self.on_decision, COMMAND_QOS, callback_group=cb)
        self.create_subscription(Float32MultiArray, Topics.AVOID_SECTORS,
                                 self.on_sectors, COMMAND_QOS, callback_group=cb)
        self.create_subscription(String, Topics.YOLO_DETECTIONS,
                                 self.on_yolo, COMMAND_QOS, callback_group=cb)

        if MAVROS_AVAILABLE:
            self.create_subscription(MavrosState, Topics.MAVROS_STATE,
                                     self.on_mavros, SENSOR_QOS, callback_group=cb)
            self.create_subscription(NavSatFix, Topics.MAVROS_GLOBAL_POS,
                                     self.on_gps, SENSOR_QOS, callback_group=cb)

        self.create_timer(flush_period, self._flush, callback_group=cb)

        self.get_logger().info(f'mission_logger basladi. Kayit klasoru: {self.out_dir}')
        self._write('logger_start', {'jsonl': self.jsonl_path})

    # ------------------------------------------------------------------
    def _now_iso(self):
        return datetime.now().isoformat(timespec='milliseconds')

    def _write(self, kind, payload):
        rec = {'t': self._now_iso(), 'kind': kind, **payload}
        try:
            self._jsonl.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except (OSError, ValueError) as e:
            self.get_logger().error(f'JSONL yazilamadi: {e}')

    def _event(self, text):
        line = f'{self._now_iso()}  {text}'
        self._events.write(line + '\n')
        self.get_logger().info(text)

    # ------------------------------------------------------------------
    def on_state(self, msg: MissionState):
        self._write('mission_state', {
            'state': int(msg.state), 'parkur': int(msg.parkur),
            'current_wp': int(msg.current_wp), 'reached_wp': int(msg.last_reached_wp),
            'avoid': bool(msg.avoidance_active), 'kamikaze': bool(msg.kamikaze_active),
            'armed': bool(msg.armed), 'mode': msg.flight_mode, 'info': msg.info,
        })
        if self._last_state != msg.state:
            self._event(f'PARKUR GECISI -> {msg.info}')
            self._last_state = msg.state

    def on_kamikaze(self, msg: KamikazeStatus):
        self._write('kamikaze', {
            'locked': bool(msg.target_locked), 'color': int(msg.target_color),
            'bearing_deg': float(msg.bearing_deg), 'phase': int(msg.phase),
            'engaged': bool(msg.engaged), 'info': msg.info,
        })
        if msg.engaged:
            self._event(f'ANGAJMAN TAMAM (renk={msg.target_color})')

    def on_color(self, msg: TargetColor):
        self._write('target_color', {
            'color': int(msg.color), 'name': msg.color_name,
            'ral': msg.ral_code, 'conf': float(msg.confidence), 'source': msg.source,
        })
        self._event(f'HEDEF RENK: {msg.color_name} ({msg.source})')

    def on_decision(self, msg: String):
        self._write('avoid_decision', {'decision': msg.data})

    def on_sectors(self, msg: Float32MultiArray):
        self._sectors_counter += 1
        if self._sectors_counter % self.sectors_every != 0:
            return
        d = list(msg.data)
        self._write('avoid_sectors', {
            'left': d[0] if len(d) > 0 else None,
            'front': d[1] if len(d) > 1 else None,
            'right': d[2] if len(d) > 2 else None,
            'corridor': d[3] if len(d) > 3 else None,
        })

    def on_yolo(self, msg: String):
        self._write('yolo', {'detections': msg.data})

    def on_mavros(self, msg):
        self._write('mavros_state', {
            'connected': bool(msg.connected), 'armed': bool(msg.armed),
            'mode': msg.mode,
        })

    def on_gps(self, msg):
        self._track_csv.writerow([self._now_iso(), msg.latitude,
                                  msg.longitude, msg.altitude])

    # ------------------------------------------------------------------
    def _flush(self):
        for f in (self._jsonl, self._events, self._track):
            try:
                f.flush()
                os.fsync(f.fileno())
            except (OSError, ValueError):
                pass

    def destroy_node(self):
        self._write('logger_stop', {})
        for f in (self._jsonl, self._events, self._track):
            try:
                f.flush()
                f.close()
            except (OSError, ValueError):
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionLoggerNode()
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
