#!/usr/bin/env python3
"""
telemetry_logger_node — Sartname "Dosya 2: Arac telemetri verisi"
==================================================================

Sartname gereksinimi (5.5 veri teslimi):
  - En az 1 Hz
  - Konum (lat, lon)
  - Hiz (yer hizi)
  - Yonelim acilari (roll, pitch, heading)
  - Hiz set pointi
  - Yon set pointi
  - csv formati, ILK SATIR header

Cikti: <out_dir>/telemetri_<ts>.csv

Kaynaklar (hepsi opsiyonel; yoksa hucre bos kalir, satir yine yazilir):
  /mavros/global_position/global      NavSatFix     -> lat, lon
  /mavros/global_position/raw/gps_vel TwistStamped  -> yer hizi (yedek)
  /mavros/local_position/velocity_body TwistStamped -> yer hizi (tercih)
  /mavros/imu/data                    Imu           -> roll, pitch, yaw
  /mavros/global_position/compass_hdg Float64       -> heading (derece)
  /mavros/setpoint_velocity/cmd_vel_unstamped Twist -> hiz/yon set pointi (P3)
  /mavros/setpoint_raw/target_global  GlobalPositionTarget -> set point (P2 GUIDED)

Tasarim notu: yazma islemi timer'da yapilir ve her satirdan sonra flush edilir;
ani guc kesintisinde en fazla son satir kaybolur. fsync periyodik (her N satir)
yapilir ki tek-thread'li executor uzun sure bloklanmasin
(ros2-robotics: blocking-callback).
"""

import csv
import math
import os
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import NavSatFix, Imu

from .common import SENSOR_QOS, COMMAND_QOS

CSV_HEADER = [
    'timestamp_iso',
    'unix_time',
    'lat',
    'lon',
    'ground_speed_mps',
    'roll_deg',
    'pitch_deg',
    'heading_deg',
    'speed_setpoint_mps',
    'yaw_setpoint_degps',
]


def quaternion_to_euler_deg(x, y, z, w):
    """Quaternion -> (roll, pitch, yaw) derece. tf_transformations bagimliligi yok."""
    # roll (x ekseni)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y ekseni) — sinir disi degerlerde kelepceleme
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z ekseni)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class TelemetryLoggerNode(Node):

    def __init__(self):
        super().__init__('telemetry_logger_node')

        self.declare_parameter('out_dir', os.path.expanduser('~/gelidonya_logs'))
        self.declare_parameter('rate_hz', 5.0)      # sartname min 1 Hz
        self.declare_parameter('fsync_every', 25)   # her N satirda diske zorla

        self.out_dir = os.path.expanduser(str(self.get_parameter('out_dir').value))
        rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        self.fsync_every = int(self.get_parameter('fsync_every').value)

        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.out_dir, f'telemetri_{ts}.csv')

        self._fh = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self._csv = csv.writer(self._fh)
        self._csv.writerow(CSV_HEADER)
        self._fh.flush()
        self._row_count = 0

        # Son bilinen degerler (None = henuz veri yok -> hucre bos yazilir)
        self.lat = None
        self.lon = None
        self.ground_speed = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.heading = None
        self.speed_sp = None
        self.yaw_sp = None

        cb = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            NavSatFix, '/mavros/global_position/global',
            self.on_gps, SENSOR_QOS, callback_group=cb)
        self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_body',
            self.on_vel, SENSOR_QOS, callback_group=cb)
        self.create_subscription(
            TwistStamped, '/mavros/global_position/raw/gps_vel',
            self.on_gps_vel, SENSOR_QOS, callback_group=cb)
        self.create_subscription(
            Imu, '/mavros/imu/data',
            self.on_imu, SENSOR_QOS, callback_group=cb)
        self.create_subscription(
            Float64, '/mavros/global_position/compass_hdg',
            self.on_hdg, SENSOR_QOS, callback_group=cb)
        self.create_subscription(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped',
            self.on_setpoint, COMMAND_QOS, callback_group=cb)

        self.create_timer(1.0 / rate_hz, self.write_row, callback_group=cb)

        self.get_logger().info(
            f'telemetry_logger basladi @ {rate_hz:.1f} Hz -> {self.csv_path}'
        )

    # ------------------------------------------------------------------
    def on_gps(self, msg: NavSatFix):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def on_vel(self, msg: TwistStamped):
        v = msg.twist.linear
        self.ground_speed = math.sqrt(v.x * v.x + v.y * v.y)

    def on_gps_vel(self, msg: TwistStamped):
        # local_position yoksa yedek kaynak
        if self.ground_speed is None:
            v = msg.twist.linear
            self.ground_speed = math.sqrt(v.x * v.x + v.y * v.y)

    def on_imu(self, msg: Imu):
        q = msg.orientation
        self.roll, self.pitch, self.yaw = quaternion_to_euler_deg(q.x, q.y, q.z, q.w)

    def on_hdg(self, msg: Float64):
        self.heading = msg.data

    def on_setpoint(self, msg: Twist):
        self.speed_sp = msg.linear.x
        self.yaw_sp = math.degrees(msg.angular.z)

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(value, digits=6):
        if value is None:
            return ''
        return f'{value:.{digits}f}'

    def write_row(self):
        now = datetime.now(timezone.utc).astimezone()
        # heading yoksa IMU yaw'undan turet (0-360 normalize)
        heading = self.heading
        if heading is None and self.yaw is not None:
            heading = self.yaw % 360.0

        self._csv.writerow([
            now.isoformat(timespec='milliseconds'),
            f'{now.timestamp():.3f}',
            self._fmt(self.lat, 7),
            self._fmt(self.lon, 7),
            self._fmt(self.ground_speed, 3),
            self._fmt(self.roll, 2),
            self._fmt(self.pitch, 2),
            self._fmt(heading, 2),
            self._fmt(self.speed_sp, 3),
            self._fmt(self.yaw_sp, 2),
        ])
        self._fh.flush()

        self._row_count += 1
        if self.fsync_every > 0 and self._row_count % self.fsync_every == 0:
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    def destroy_node(self):
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except (OSError, ValueError):
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLoggerNode()
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
