#!/usr/bin/env python3
"""
sim_mavros_node — Donanimsiz test icin sahte MAVROS + basit tekne kinematigi
============================================================================

CUAV X7+ ve GPS olmadan tum gorev zincirini (mission_manager, kamikaze,
logger'lar) ucdan uca calistirmak icin. MAVROS'un mission_control'un
kullandigi arayuzlerini taklit eder:

  Yayinlar:
    /mavros/state                       arm + ucus modu
    /mavros/mission/waypoints           5 noktali sahte gorev
    /mavros/mission/reached             AUTO'da WP'ye varinca
    /mavros/global_position/global      simule konum
    /mavros/local_position/velocity_body  yer hizi (telemetri CSV testi)
    /mavros/imu/data                    yonelim (telemetri CSV testi)
    /mavros/global_position/compass_hdg  heading

  Servisler:
    /mavros/set_mode                    mod degistirme (manager bunu cagirir)
    /sim/arm                            (std_srvs/SetBool) arm/disarm
    /sim/reset                          (std_srvs/Trigger) basa al

  Dinler:
    /mavros/setpoint_position/global    GUIDED hedefi (Parkur-2)
    /mavros/setpoint_velocity/cmd_vel_unstamped  GUIDED hiz (Parkur-3)

Kinematik: duz bir duzlemde sabit hizla hedefe yonelen nokta-tekne.
Gercek dinamik (atalet, akinti, surukleme) modellenmez — amac gorev
mantigini ve durum gecislerini dogrulamaktir, tekne performansini degil.

Kullanim:
  ros2 run mission_control sim_mavros_node
  ros2 service call /sim/arm std_srvs/srv/SetBool "{data: true}"
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist, TwistStamped

from mavros_msgs.msg import State as MavrosState
from mavros_msgs.msg import Waypoint, WaypointList, WaypointReached
from mavros_msgs.srv import SetMode
from geographic_msgs.msg import GeoPoseStamped

from .common import Topics, SENSOR_QOS, COMMAND_QOS, STATE_QOS

EARTH_R = 6371000.0


def meters_to_latlon(lat0, lon0, north_m, east_m):
    """Yerel duzlem ofsetini cografi koordinata cevirir (kucuk mesafe yaklasimi)."""
    dlat = math.degrees(north_m / EARTH_R)
    dlon = math.degrees(east_m / (EARTH_R * math.cos(math.radians(lat0))))
    return lat0 + dlat, lon0 + dlon


def latlon_to_meters(lat0, lon0, lat, lon):
    """Iki nokta arasi (kuzey, dogu) metre farki."""
    north = math.radians(lat - lat0) * EARTH_R
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return north, east


class SimMavrosNode(Node):

    def __init__(self):
        super().__init__('sim_mavros_node')

        # Baslangic konumu (varsayilan: Antalya kiyisi civari — anlamsiz degil,
        # sadece gecerli bir cografi nokta olmasi yeterli).
        self.declare_parameter('origin_lat', 36.8500000)
        self.declare_parameter('origin_lon', 30.6300000)
        self.declare_parameter('speed_mps', 2.0)          # simule seyir hizi
        self.declare_parameter('wp_spacing_m', 25.0)      # noktalar arasi mesafe
        self.declare_parameter('wp_reach_radius_m', 3.0)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('auto_arm', False)         # test kolayligi

        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)
        self.speed = float(self.get_parameter('speed_mps').value)
        spacing = float(self.get_parameter('wp_spacing_m').value)
        self.reach_radius = float(self.get_parameter('wp_reach_radius_m').value)
        rate = max(1.0, float(self.get_parameter('rate_hz').value))

        # --- Tekne durumu (yerel duzlem, metre) ---
        self.north = 0.0
        self.east = 0.0
        self.heading = 0.0          # derece, 0 = kuzey
        self.speed_now = 0.0
        self.armed = bool(self.get_parameter('auto_arm').value)
        self.mode = 'MANUAL'

        # --- Sahte gorev: 5 nokta, duz bir hat uzerinde ---
        # WP0..WP3 = Parkur-1, WP4 = Parkur-2 sonu
        self.waypoints = [(spacing * (i + 1), 0.0) for i in range(5)]
        self.auto_wp_index = 0
        self.reached_seqs = set()

        # --- GUIDED hedefleri ---
        self.guided_target = None   # (north, east)
        self.guided_twist = None    # Twist (Parkur-3)

        cb = MutuallyExclusiveCallbackGroup()

        # --- Yayincilar ---
        self.pub_state = self.create_publisher(MavrosState, Topics.MAVROS_STATE, SENSOR_QOS)
        self.pub_wps = self.create_publisher(WaypointList, Topics.MAVROS_WAYPOINTS, STATE_QOS)
        self.pub_reached = self.create_publisher(
            WaypointReached, Topics.MAVROS_WP_REACHED, COMMAND_QOS)
        self.pub_gps = self.create_publisher(NavSatFix, Topics.MAVROS_GLOBAL_POS, SENSOR_QOS)
        self.pub_vel = self.create_publisher(
            TwistStamped, '/mavros/local_position/velocity_body', SENSOR_QOS)
        self.pub_imu = self.create_publisher(Imu, '/mavros/imu/data', SENSOR_QOS)
        self.pub_hdg = self.create_publisher(
            Float64, '/mavros/global_position/compass_hdg', SENSOR_QOS)

        # --- Aboneler ---
        self.create_subscription(
            GeoPoseStamped, Topics.MAVROS_SETPOINT_GLOBAL,
            self.on_guided_target, COMMAND_QOS, callback_group=cb)
        self.create_subscription(
            Twist, Topics.MAVROS_SETPOINT_VEL,
            self.on_guided_twist, COMMAND_QOS, callback_group=cb)

        # --- Servisler ---
        self.create_service(SetMode, '/mavros/set_mode', self.srv_set_mode, callback_group=cb)
        self.create_service(SetBool, '/sim/arm', self.srv_arm, callback_group=cb)
        self.create_service(Trigger, '/sim/reset', self.srv_reset, callback_group=cb)

        self.dt = 1.0 / rate
        self.create_timer(self.dt, self.step, callback_group=cb)
        self.create_timer(1.0, self.publish_waypoints, callback_group=cb)

        self.get_logger().info(
            f'sim_mavros basladi. {len(self.waypoints)} WP, {spacing:.0f} m aralikli, '
            f'{self.speed:.1f} m/s. Baslatmak icin: '
            f'ros2 service call /sim/arm std_srvs/srv/SetBool "{{data: true}}"'
        )

    # ==================================================================
    # SERVISLER
    # ==================================================================
    def srv_set_mode(self, request, response):
        self.mode = request.custom_mode.upper()
        response.mode_sent = True
        self.get_logger().info(f'[sim] mod -> {self.mode}')
        return response

    def srv_arm(self, request, response):
        self.armed = bool(request.data)
        if self.armed and self.mode == 'MANUAL':
            self.mode = 'AUTO'      # gercek akista operator AUTO'ya alir
            self.get_logger().info('[sim] arm + AUTO (gorev basliyor)')
        response.success = True
        response.message = f'armed={self.armed} mode={self.mode}'
        return response

    def srv_reset(self, request, response):
        self.north = self.east = 0.0
        self.heading = 0.0
        self.auto_wp_index = 0
        self.reached_seqs.clear()
        self.guided_target = None
        self.guided_twist = None
        self.armed = False
        self.mode = 'MANUAL'
        response.success = True
        response.message = 'sim sifirlandi'
        self.get_logger().info('[sim] sifirlandi')
        return response

    # ==================================================================
    # ABONE CALLBACK'LERI
    # ==================================================================
    def on_guided_target(self, msg: GeoPoseStamped):
        n, e = latlon_to_meters(
            self.origin_lat, self.origin_lon,
            msg.pose.position.latitude, msg.pose.position.longitude)
        self.guided_target = (n, e)

    def on_guided_twist(self, msg: Twist):
        self.guided_twist = msg

    # ==================================================================
    # KINEMATIK ADIM
    # ==================================================================
    def step(self):
        if self.armed:
            if self.mode == 'AUTO':
                self._step_auto()
            elif self.mode == 'GUIDED':
                self._step_guided()
            else:
                self.speed_now = 0.0
        else:
            self.speed_now = 0.0

        self.publish_telemetry()

    def _move_toward(self, target_n, target_e):
        dn = target_n - self.north
        de = target_e - self.east
        dist = math.hypot(dn, de)
        if dist < 1e-3:
            self.speed_now = 0.0
            return 0.0
        step = min(self.speed * self.dt, dist)
        self.north += dn / dist * step
        self.east += de / dist * step
        self.heading = math.degrees(math.atan2(de, dn)) % 360.0
        self.speed_now = self.speed
        return dist

    def _step_auto(self):
        """AUTO: yuklu gorevi sirayla takip et, varista reached yayinla."""
        if self.auto_wp_index >= len(self.waypoints):
            self.speed_now = 0.0
            return
        tn, te = self.waypoints[self.auto_wp_index]
        dist = self._move_toward(tn, te)
        if dist <= self.reach_radius:
            # ArduPilot'ta item 0 HOME oldugu icin gercek WP'ler 1'den baslar.
            seq = self.auto_wp_index + 1
            if seq not in self.reached_seqs:
                self.reached_seqs.add(seq)
                msg = WaypointReached()
                msg.wp_seq = seq
                self.pub_reached.publish(msg)
                self.get_logger().info(f'[sim] WP{seq} ulasildi (seq={seq})')
            self.auto_wp_index += 1

    def _step_guided(self):
        """GUIDED: once konum hedefi (P2), yoksa hiz komutu (P3)."""
        if self.guided_target is not None:
            self._move_toward(*self.guided_target)
            return
        if self.guided_twist is not None:
            v = self.guided_twist.linear.x
            yaw_rate = self.guided_twist.angular.z
            self.heading = (self.heading + math.degrees(yaw_rate) * self.dt) % 360.0
            rad = math.radians(self.heading)
            self.north += v * math.cos(rad) * self.dt
            self.east += v * math.sin(rad) * self.dt
            self.speed_now = v
            return
        self.speed_now = 0.0

    # ==================================================================
    # YAYINLAR
    # ==================================================================
    def publish_waypoints(self):
        """ArduPilot ile AYNI duzeni yayinla: item 0 = HOME, gercek noktalar 1..N.

        Gercek FC'den okunan gorev boyle geliyor (WP: item #0 x:0 y:0 = HOME).
        Sim de ayni olsun ki parkur1_last_wp/parkur2_last_wp degerleri hem
        simulasyonda hem sahada ayni calissin.
        """
        msg = WaypointList()
        msg.current_seq = min(self.auto_wp_index + 1, len(self.waypoints))

        home = Waypoint()
        home.frame = 0              # GLOBAL
        home.command = 16           # NAV_WAYPOINT
        home.autocontinue = True
        home.x_lat = 0.0            # ArduPilot HOME'u boyle bildiriyor
        home.y_long = 0.0
        home.z_alt = 0.0
        msg.waypoints.append(home)

        for n, e in self.waypoints:
            lat, lon = meters_to_latlon(self.origin_lat, self.origin_lon, n, e)
            wp = Waypoint()
            wp.frame = 3            # GLOBAL_RELATIVE_ALT
            wp.command = 16         # NAV_WAYPOINT
            wp.is_current = False
            wp.autocontinue = True
            wp.x_lat = lat
            wp.y_long = lon
            wp.z_alt = 0.0
            msg.waypoints.append(wp)
        self.pub_wps.publish(msg)

    def publish_telemetry(self):
        now = self.get_clock().now().to_msg()

        state = MavrosState()
        state.header.stamp = now
        state.connected = True
        state.armed = self.armed
        state.guided = (self.mode == 'GUIDED')
        state.mode = self.mode
        self.pub_state.publish(state)

        lat, lon = meters_to_latlon(
            self.origin_lat, self.origin_lon, self.north, self.east)
        fix = NavSatFix()
        fix.header.stamp = now
        fix.header.frame_id = 'base_link'
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = 0.0
        self.pub_gps.publish(fix)

        vel = TwistStamped()
        vel.header.stamp = now
        rad = math.radians(self.heading)
        vel.twist.linear.x = self.speed_now * math.cos(rad)
        vel.twist.linear.y = self.speed_now * math.sin(rad)
        self.pub_vel.publish(vel)

        # Yalnizca yaw'i olan quaternion (roll=pitch=0)
        half = math.radians(self.heading) / 2.0
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'base_link'
        imu.orientation.z = math.sin(half)
        imu.orientation.w = math.cos(half)
        self.pub_imu.publish(imu)

        hdg = Float64()
        hdg.data = self.heading
        self.pub_hdg.publish(hdg)


def main(args=None):
    rclpy.init(args=args)
    node = SimMavrosNode()
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
