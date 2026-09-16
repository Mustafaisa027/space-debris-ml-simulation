#!/usr/bin/env python3
"""
mission_manager_node
=====================

Gorevin BEYNI. 3 parkuru tek seferde, operator girisi olmadan otomatik
yoneten durum makinesi (sartname 5.5.2.2: "Parkurlar arasi gecis kullanici
girisi olmadan otomatik olarak algilanacak ve yapilacaktir").

AKIS
----
  BOOT      : MAVROS baglantisi + GPS + sensor bekle
  READY     : gorev yuklu, hedef renk atanmis, START (arm+AUTO) bekle
  PARKUR1   : AUTO. WP1..WP4 nokta takip. Engel kacinma KAPALI.
              -> WP4'e (parkur1_last_wp) ulasilinca PARKUR2
  PARKUR2   : GUIDED. Jetson WP5'i hedef olarak yayinlar; ZED->obstacle_bridge
              CUAV X7+'daki BendyRuler'i besler ve manevrayi ArduPilot yapar.
              Arac 0.30-3.00 m araligindaki engellerden kacar.
              -> WP5'e varilinca (mesafe esigi) PARKUR3
  PARKUR3   : kamikaze aktif. IHA renk + YOLO ile hedefe angajman.
              -> /kamikaze/status.engaged True olunca DONE
  DONE      : gorev tamam (baslangica donus manuel yapilir)
  EMERGENCY : kill switch; her seyi durdur

Bu dugum motoru DOGRUDAN surmez. Kacinmayi ArduPilot'un gomulu BendyRuler'ina
birakir; sadece hangi alt-sistemin aktif oldugunu kontrol eder ve /mission/state
yayinlar. Diger dugumler bu state'i dinleyip kendini aktif/pasif eder.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from mission_interfaces.msg import MissionState, TargetColor, KamikazeStatus
from mission_interfaces.srv import SetTargetColor, SetParkur

from .common import Topics, Services, SENSOR_QOS, COMMAND_QOS, STATE_QOS

# MAVROS mesajlari (Jetson uzerinde mevcut). Gelistirme makinesinde
# bulunmayabilir; import basarisiz olursa dugum yine calisir, sadece
# MAVROS'a bagli otomatik gecis pasif kalir (test/debug icin SetParkur kullan).
try:
    from mavros_msgs.msg import State as MavrosState
    from mavros_msgs.msg import WaypointReached, WaypointList
    from mavros_msgs.srv import SetMode
    from sensor_msgs.msg import NavSatFix
    from geographic_msgs.msg import GeoPoseStamped
    MAVROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    MavrosState = None
    WaypointReached = None
    WaypointList = None
    SetMode = None
    NavSatFix = None
    GeoPoseStamped = None
    MAVROS_AVAILABLE = False


def haversine_m(lat1, lon1, lat2, lon2):
    """Iki cografi nokta arasi mesafe (metre). GUIDED'de WP5'e varisi olcer."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2)
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


COLOR_NAMES = {
    TargetColor.UNKNOWN: 'unknown',
    TargetColor.RED: 'red',
    TargetColor.GREEN: 'green',
    TargetColor.BLACK: 'black',
}


class MissionManagerNode(Node):

    def __init__(self):
        super().__init__('mission_manager_node')

        # ------------------------------------------------------------------
        # PARAMETRELER
        # ------------------------------------------------------------------
        # MAVROS wp_seq ArduPilot mission indeksidir (HOME dahil).
        # ArduPilot item 0 = HOME. Gercek noktalar 1'den baslar:
        self.declare_parameter('parkur1_last_wp', 4)   # WP4 (item0=HOME)
        self.declare_parameter('parkur2_last_wp', 5)   # WP5 (item0=HOME)
        self.declare_parameter('require_target_color', True)  # P3 icin renk sart mi
        self.declare_parameter('control_rate_hz', 5.0)
        self.declare_parameter('auto_start_on_auto_mode', True)
        self.declare_parameter('kamikaze_timeout_s', 90.0)

        # Ucus modu yonetimi. Baslangic: operator AUTO+ARM yapar -> gorev baslar.
        # Sonra mission_manager her parkur icin modu kendisi ayarlar:
        #   Parkur-1/2: AUTO  -> ArduPilot yuklu waypoint gorevini izler
        #               (Parkur-2'de gomulu BendyRuler engelden kacar)
        #   Parkur-3:   GUIDED -> hedef bir waypoint degil (IHA'nin sectigi gorsel
        #               hedef); companion/kamikaze setpoint gonderip ustune surer
        # EMERGENCY: HOLD (araci durdur)
        self.declare_parameter('parkur1_mode', 'AUTO')
        self.declare_parameter('parkur2_mode', 'GUIDED')
        self.declare_parameter('parkur3_mode', 'GUIDED')
        self.declare_parameter('emergency_mode', 'HOLD')
        self.declare_parameter('manage_flight_mode', True)  # False: sadece izle
        self.declare_parameter('mode_retry_period_s', 1.0)  # mod dogrulanmazsa tekrar

        # Parkur-2 GUIDED navigasyonu:
        # Jetson WP5'i hedef olarak surekli gonderir; CUAV X7+ uzerindeki
        # BendyRuler (OA_TYPE=1) bu hedefe giderken ZED'den beslenen engellerden
        # kacar. Yani manevra/motor kontrolu ArduPilot'ta kalir, biz sadece
        # "nereye gidilecek" bilgisini veririz.
        self.declare_parameter('p2_send_guided_target', True)
        self.declare_parameter('p2_arrival_radius_m', 2.5)   # WP5'e varis yaricapi
        self.declare_parameter('p2_target_alt', 0.0)         # deniz araci: 0

        self.parkur1_last_wp = int(self.get_parameter('parkur1_last_wp').value)
        self.parkur2_last_wp = int(self.get_parameter('parkur2_last_wp').value)
        self.require_target_color = bool(self.get_parameter('require_target_color').value)
        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.auto_start = bool(self.get_parameter('auto_start_on_auto_mode').value)
        self.kamikaze_timeout_s = float(self.get_parameter('kamikaze_timeout_s').value)

        self.parkur1_mode = str(self.get_parameter('parkur1_mode').value)
        self.parkur2_mode = str(self.get_parameter('parkur2_mode').value)
        self.parkur3_mode = str(self.get_parameter('parkur3_mode').value)
        self.emergency_mode = str(self.get_parameter('emergency_mode').value)
        self.manage_flight_mode = bool(self.get_parameter('manage_flight_mode').value)
        self.mode_retry_period = float(self.get_parameter('mode_retry_period_s').value)

        self.p2_send_target = bool(self.get_parameter('p2_send_guided_target').value)
        self.p2_arrival_radius = float(self.get_parameter('p2_arrival_radius_m').value)
        self.p2_target_alt = float(self.get_parameter('p2_target_alt').value)

        # ------------------------------------------------------------------
        # DURUM DEGISKENLERI
        # ------------------------------------------------------------------
        self.state = MissionState.STATE_BOOT
        self.parkur = 0
        self.last_reached_wp = -1
        self.current_wp = 0

        self.mavros_connected = False
        self.armed = False
        self.flight_mode = ''

        self.target_color = TargetColor.UNKNOWN
        self.target_color_locked = False   # gorev basladiktan sonra kilitlenir

        self.kamikaze_engaged = False
        self.kamikaze_phase = 0
        self.parkur3_enter_time = None

        self.emergency = False

        # Parkur-2 GUIDED navigasyonu icin konum bilgisi
        self.wp_targets = {}        # {seq: (lat, lon)} — yuklu gorevden
        self.current_lat = None
        self.current_lon = None
        self.dist_to_wp5 = None

        # Callback gruplari: agir/uzun servis cagrilari kontrol dongusunu
        # bloklamasin (ros2-robotics skill: blocking-callback).
        self.cb_sensors = MutuallyExclusiveCallbackGroup()
        self.cb_clients = ReentrantCallbackGroup()

        # ------------------------------------------------------------------
        # PUBLISHER
        # ------------------------------------------------------------------
        self.state_pub = self.create_publisher(
            MissionState, Topics.MISSION_STATE, STATE_QOS
        )

        # Parkur-2 GUIDED hedefi (WP5). ArduPilot bu hedefe giderken
        # BendyRuler ile engellerden kacar.
        self.guided_target_pub = None
        if MAVROS_AVAILABLE:
            self.guided_target_pub = self.create_publisher(
                GeoPoseStamped, Topics.MAVROS_SETPOINT_GLOBAL, COMMAND_QOS
            )

        # ------------------------------------------------------------------
        # SUBSCRIBER
        # ------------------------------------------------------------------
        self.create_subscription(
            Bool, Topics.EMERGENCY, self.on_emergency, COMMAND_QOS,
            callback_group=self.cb_sensors
        )
        self.create_subscription(
            TargetColor, Topics.IHA_TARGET_COLOR, self.on_target_color, COMMAND_QOS,
            callback_group=self.cb_sensors
        )
        self.create_subscription(
            KamikazeStatus, Topics.KAMIKAZE_STATUS, self.on_kamikaze_status, COMMAND_QOS,
            callback_group=self.cb_sensors
        )

        if MAVROS_AVAILABLE:
            self.create_subscription(
                MavrosState, Topics.MAVROS_STATE, self.on_mavros_state, SENSOR_QOS,
                callback_group=self.cb_sensors
            )
            self.create_subscription(
                WaypointReached, Topics.MAVROS_WP_REACHED, self.on_wp_reached, COMMAND_QOS,
                callback_group=self.cb_sensors
            )
            # Yuklu gorev: WP5'in cografi konumunu buradan ogreniyoruz.
            self.create_subscription(
                WaypointList, Topics.MAVROS_WAYPOINTS, self.on_waypoints, STATE_QOS,
                callback_group=self.cb_sensors
            )
            # Anlik konum: GUIDED'de WP5'e varisi mesafeyle tespit ediyoruz
            # (mission/reached sadece AUTO gorevinde tetiklenir).
            self.create_subscription(
                NavSatFix, Topics.MAVROS_GLOBAL_POS, self.on_global_pos, SENSOR_QOS,
                callback_group=self.cb_sensors
            )
        else:
            self.get_logger().warn(
                'mavros_msgs bulunamadi. Otomatik parkur gecisi PASIF. '
                'Test icin /mission/set_parkur servisini kullan.'
            )

        # ------------------------------------------------------------------
        # SERVIS SUNUCULARI
        # ------------------------------------------------------------------
        self.create_service(
            SetTargetColor, Services.SET_TARGET_COLOR, self.srv_set_target_color,
            callback_group=self.cb_clients
        )
        self.create_service(
            SetParkur, Services.SET_PARKUR, self.srv_set_parkur,
            callback_group=self.cb_clients
        )
        self.create_service(
            Trigger, Services.EMERGENCY_STOP, self.srv_emergency_stop,
            callback_group=self.cb_clients
        )

        # ------------------------------------------------------------------
        # SERVIS ISTEMCILERI (alt sistemleri ac/kapa)
        # ------------------------------------------------------------------
        self.cli_avoid = self.create_client(
            SetBool, Services.OBSTACLE_BRIDGE_ENABLE, callback_group=self.cb_clients
        )
        self.cli_kamikaze = self.create_client(
            SetBool, Services.KAMIKAZE_ENABLE, callback_group=self.cb_clients
        )

        # MAVROS ucus modu istemcisi (AUTO/GUIDED/HOLD ayarlamak icin)
        self.cli_set_mode = None
        if MAVROS_AVAILABLE:
            self.cli_set_mode = self.create_client(
                SetMode, '/mavros/set_mode', callback_group=self.cb_clients
            )

        self._avoid_desired = None      # son gonderilen istek (idempotent)
        self._kamikaze_desired = None
        self._mode_desired = None       # istenen ucus modu (AUTO/GUIDED/HOLD)
        self._mode_last_cmd_time = None # son set_mode cagrisi zamani (retry icin)

        # ------------------------------------------------------------------
        # KONTROL DONGUSU
        # ------------------------------------------------------------------
        period = 1.0 / max(0.5, self.control_rate_hz)
        self.create_timer(period, self.control_loop, callback_group=self.cb_clients)

        self.get_logger().info('mission_manager_node basladi (STATE_BOOT)')
        self.get_logger().info(
            f'parkur1_last_wp={self.parkur1_last_wp} '
            f'parkur2_last_wp={self.parkur2_last_wp}'
        )

    # ======================================================================
    # CALLBACK'LER
    # ======================================================================
    def on_emergency(self, msg: Bool):
        if msg.data and not self.emergency:
            self.get_logger().error('ACIL DURDURMA alindi (/mission/emergency)')
        self.emergency = bool(msg.data)

    def on_mavros_state(self, msg):
        self.mavros_connected = msg.connected
        self.armed = msg.armed
        self.flight_mode = msg.mode

    def on_wp_reached(self, msg):
        self.last_reached_wp = int(msg.wp_seq)
        self.get_logger().info(f'Waypoint ulasildi: seq={self.last_reached_wp}')

    def on_waypoints(self, msg):
        """Yuklu gorevden WP konumlarini al (Parkur-2 GUIDED hedefi icin)."""
        targets = {}
        for i, wp in enumerate(msg.waypoints):
            # Sadece konum tasiyan komutlar (NAV_WAYPOINT vb.) anlamli
            if wp.x_lat != 0.0 or wp.y_long != 0.0:
                targets[i] = (wp.x_lat, wp.y_long)
        if targets != self.wp_targets:
            self.wp_targets = targets
            self.get_logger().info(
                f'Gorev alindi: {len(targets)} konumlu waypoint '
                f'(WP{self.parkur2_last_wp} = seq {self.parkur2_last_wp})'
            )

    def on_global_pos(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    def on_target_color(self, msg: TargetColor):
        # Sartname 5.5.3.1: hedef bilgisi arac harekete BASLADIKTAN sonra
        # aktarilamaz. Gorev basladiysa (renk kilitli) yeni renk reddedilir.
        if self.target_color_locked:
            self.get_logger().warn(
                'Hedef renk gorev basladiktan sonra degistirilemez. Yok sayildi.'
            )
            return
        self.target_color = msg.color
        self.get_logger().info(
            f'Hedef renk atandi: {COLOR_NAMES.get(msg.color, "?")} '
            f'(kaynak={msg.source})'
        )

    def on_kamikaze_status(self, msg: KamikazeStatus):
        self.kamikaze_engaged = msg.engaged
        self.kamikaze_phase = msg.phase

    # ======================================================================
    # SERVISLER
    # ======================================================================
    def srv_set_target_color(self, request, response):
        if self.target_color_locked:
            response.success = False
            response.message = 'Renk kilitli (gorev basladi).'
            return response
        if request.color not in COLOR_NAMES or request.color == TargetColor.UNKNOWN:
            response.success = False
            response.message = f'Gecersiz renk: {request.color}'
            return response
        self.target_color = request.color
        response.success = True
        response.message = f'Hedef renk = {COLOR_NAMES[request.color]}'
        self.get_logger().info(response.message)
        return response

    def srv_set_parkur(self, request, response):
        # SADECE test/debug. Otomatik gecisi bypass eder.
        mapping = {
            1: MissionState.STATE_PARKUR1,
            2: MissionState.STATE_PARKUR2,
            3: MissionState.STATE_PARKUR3,
        }
        if request.parkur not in mapping:
            response.success = False
            response.message = 'parkur 1,2,3 olmali'
            response.current_parkur = self.parkur
            return response
        self._transition_to(mapping[request.parkur])
        response.success = True
        response.message = f'(DEBUG) parkur -> {request.parkur}'
        response.current_parkur = self.parkur
        return response

    def srv_emergency_stop(self, request, response):
        self.emergency = True
        self._transition_to(MissionState.STATE_EMERGENCY)
        response.success = True
        response.message = 'Acil durdurma etkinlestirildi'
        return response

    # ======================================================================
    # ALT SISTEM KONTROLU
    # ======================================================================
    def _set_avoidance(self, enable: bool):
        """obstacle_bridge yayinini ac/kapa (idempotent, non-blocking)."""
        if self._avoid_desired == enable:
            return
        self._avoid_desired = enable
        if not self.cli_avoid.service_is_ready():
            self.get_logger().warn('obstacle_bridge/enable servisi hazir degil')
            return
        req = SetBool.Request()
        req.data = enable
        self.cli_avoid.call_async(req)  # sonucu bloklamadan bekleme
        self.get_logger().info(f'Engel kacinma -> {"ACIK" if enable else "KAPALI"}')

    def _set_kamikaze(self, enable: bool):
        if self._kamikaze_desired == enable:
            return
        self._kamikaze_desired = enable
        if not self.cli_kamikaze.service_is_ready():
            self.get_logger().warn('kamikaze/enable servisi hazir degil')
            return
        req = SetBool.Request()
        req.data = enable
        self.cli_kamikaze.call_async(req)
        self.get_logger().info(f'Kamikaze -> {"ACIK" if enable else "KAPALI"}')

    def _set_mode(self, mode: str):
        """Istenen ucus modunu kaydet ve hemen komut gonder.

        Gercek gonderim/tekrar _ensure_flight_mode'da yapilir; boylece komut
        dusarse (servis mesgul, ArduPilot reddederse) mod dogrulanana kadar
        tekrar denenir. call_async 'gonder-unut' oldugu icin tek atis riskli.
        """
        if not self.manage_flight_mode or not mode:
            return
        self._mode_desired = mode
        self._mode_last_cmd_time = None   # bir sonraki ensure hemen gondersin
        self._ensure_flight_mode()

    def _ensure_flight_mode(self):
        """Arac istenen modda degilse set_mode'u periyodik yeniden gonder."""
        if not self.manage_flight_mode or self._mode_desired is None:
            return
        if not MAVROS_AVAILABLE or self.cli_set_mode is None:
            return
        # Zaten istenen moddaysak islem yok.
        if self.flight_mode.upper() == self._mode_desired.upper():
            return
        # Degilse retry periyoduna gore yeniden dene.
        now = self.get_clock().now()
        if self._mode_last_cmd_time is not None:
            elapsed = (now - self._mode_last_cmd_time).nanoseconds / 1e9
            if elapsed < self.mode_retry_period:
                return
        if not self.cli_set_mode.service_is_ready():
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = self._mode_desired   # 'AUTO' / 'GUIDED' / 'HOLD'
        self.cli_set_mode.call_async(req)
        self._mode_last_cmd_time = now
        self.get_logger().warn(
            f'Ucus modu istendi -> {self._mode_desired} '
            f'(su anki: {self.flight_mode or "?"})'
        )

    def _transition_to(self, new_state: int):
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state

        if new_state == MissionState.STATE_PARKUR1:
            self.parkur = 1
            self.target_color_locked = True   # gorev basladi, renk kilitle
            self._set_mode(self.parkur1_mode)   # AUTO: yuklu waypoint gorevi
            self._set_avoidance(False)
            self._set_kamikaze(False)
        elif new_state == MissionState.STATE_PARKUR2:
            self.parkur = 2
            self._set_mode(self.parkur2_mode)   # AUTO + BendyRuler
            self._set_avoidance(True)         # BendyRuler beslemesi acilir
            self._set_kamikaze(False)
        elif new_state == MissionState.STATE_PARKUR3:
            self.parkur = 3
            self._set_mode(self.parkur3_mode)   # GUIDED: companion hedefe surer
            self._set_avoidance(False)
            self._set_kamikaze(True)
            self.parkur3_enter_time = self.get_clock().now()
        elif new_state == MissionState.STATE_DONE:
            self._set_avoidance(False)
            self._set_kamikaze(False)
        elif new_state == MissionState.STATE_EMERGENCY:
            self._set_mode(self.emergency_mode)  # HOLD: araci durdur
            self._set_avoidance(False)
            self._set_kamikaze(False)

        self.get_logger().info(
            f'DURUM GECISI: {self._state_name(old)} -> {self._state_name(new_state)}'
        )

    @staticmethod
    def _state_name(s):
        return {
            MissionState.STATE_BOOT: 'BOOT',
            MissionState.STATE_READY: 'READY',
            MissionState.STATE_PARKUR1: 'PARKUR1',
            MissionState.STATE_PARKUR2: 'PARKUR2',
            MissionState.STATE_PARKUR3: 'PARKUR3',
            MissionState.STATE_DONE: 'DONE',
            MissionState.STATE_EMERGENCY: 'EMERGENCY',
        }.get(s, '?')

    # ======================================================================
    # ANA DURUM MAKINESI
    # ======================================================================
    def control_loop(self):
        if self.emergency and self.state != MissionState.STATE_EMERGENCY:
            self._transition_to(MissionState.STATE_EMERGENCY)

        # Istenen ucus modu tutmadiysa yeniden dene (call_async gonder-unut).
        self._ensure_flight_mode()

        if self.state == MissionState.STATE_BOOT:
            ready = (self.mavros_connected or not MAVROS_AVAILABLE)
            if self.require_target_color:
                ready = ready and (self.target_color != TargetColor.UNKNOWN)
            if ready:
                self._transition_to(MissionState.STATE_READY)

        elif self.state == MissionState.STATE_READY:
            # START = arac arm + AUTO mod (sartname: baslat komutu YKI/RC'den).
            started = self.armed and (self.flight_mode.upper() == 'AUTO')
            if not MAVROS_AVAILABLE:
                started = False  # MAVROS yoksa otomatik baslama; SetParkur ile test et
            if self.auto_start and started:
                self._transition_to(MissionState.STATE_PARKUR1)

        elif self.state == MissionState.STATE_PARKUR1:
            self.current_wp = min(self.last_reached_wp + 1, self.parkur1_last_wp)
            if self.last_reached_wp >= self.parkur1_last_wp:
                self._transition_to(MissionState.STATE_PARKUR2)

        elif self.state == MissionState.STATE_PARKUR2:
            self.current_wp = self.parkur2_last_wp
            self._parkur2_navigate()
            if self._parkur2_arrived():
                self._transition_to(MissionState.STATE_PARKUR3)

        elif self.state == MissionState.STATE_PARKUR3:
            if self.kamikaze_engaged:
                self._transition_to(MissionState.STATE_DONE)
            elif self.parkur3_enter_time is not None:
                elapsed = (self.get_clock().now() - self.parkur3_enter_time).nanoseconds / 1e9
                if elapsed > self.kamikaze_timeout_s:
                    self.get_logger().warn(
                        f'Kamikaze zaman asimi ({elapsed:.0f}s). DONE.'
                    )
                    self._transition_to(MissionState.STATE_DONE)

        self._publish_state()

    # ======================================================================
    # PARKUR-2 GUIDED NAVIGASYONU
    # ======================================================================
    def _parkur2_navigate(self):
        """WP5'i GUIDED hedefi olarak yayinla.

        Manevra ve motor kontrolu ArduPilot'ta kalir: BendyRuler bu hedefe
        giderken ZED'den beslenen engellerden kacar. Biz sadece hedefi
        tazeleriz (GUIDED surekli hedef ister).
        """
        if not self.p2_send_target or self.guided_target_pub is None:
            return
        target = self.wp_targets.get(self.parkur2_last_wp)
        if target is None:
            self.get_logger().warn(
                f'WP{self.parkur2_last_wp} konumu bilinmiyor '
                f'(gorev yuklu mu?). GUIDED hedefi gonderilemiyor.',
                throttle_duration_sec=5.0
            )
            return

        lat, lon = target
        msg = GeoPoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.latitude = float(lat)
        msg.pose.position.longitude = float(lon)
        msg.pose.position.altitude = float(self.p2_target_alt)
        msg.pose.orientation.w = 1.0
        self.guided_target_pub.publish(msg)

    def _parkur2_arrived(self) -> bool:
        """WP5'e varildi mi? GUIDED'de mission/reached tetiklenmez, mesafe olceriz."""
        # AUTO'da kalmayi secen kurulumlar icin wp_reached de kabul edilir.
        if self.last_reached_wp >= self.parkur2_last_wp:
            return True

        target = self.wp_targets.get(self.parkur2_last_wp)
        if target is None or self.current_lat is None:
            return False

        self.dist_to_wp5 = haversine_m(
            self.current_lat, self.current_lon, target[0], target[1]
        )
        if self.dist_to_wp5 <= self.p2_arrival_radius:
            self.get_logger().info(
                f'WP{self.parkur2_last_wp} varis: {self.dist_to_wp5:.1f} m '
                f'(esik {self.p2_arrival_radius:.1f} m)'
            )
            return True
        return False

    def _publish_state(self):
        msg = MissionState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mission'
        msg.state = self.state
        msg.parkur = self.parkur
        msg.current_wp = int(self.current_wp)
        msg.last_reached_wp = int(self.last_reached_wp)
        msg.avoidance_active = bool(self._avoid_desired)
        msg.kamikaze_active = bool(self._kamikaze_desired)
        msg.armed = self.armed
        msg.flight_mode = self.flight_mode
        dist_text = (
            f' d={self.dist_to_wp5:.1f}m'
            if (self.state == MissionState.STATE_PARKUR2 and
                self.dist_to_wp5 is not None)
            else ''
        )
        msg.info = (
            f'{self._state_name(self.state)} '
            f'wp={self.current_wp}/{self.last_reached_wp} '
            f'renk={COLOR_NAMES.get(self.target_color, "?")}{dist_text}'
        )
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
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
