"""
Tum gorev dugumlerinin paylastigi sabitler ve QoS profilleri.

Neden tek dosya?  ROS2'de topic ismi derleme aninda dogrulanmaz. Bir yerde
'/mission/state', baska yerde 'mission/state' yazarsan iki AYRI topic olusur
ve hicbir hata almadan veri akmaz (ros2-robotics skill: topic-name-typo).
Isimleri tek yerde tutup her yerde buradan import ederek bu sinifi engelliyoruz.
"""

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


# ---------------------------------------------------------------------------
# TOPIC ISIMLERI
# ---------------------------------------------------------------------------
class Topics:
    # Gorev yonetimi
    MISSION_STATE = '/mission/state'
    EMERGENCY = '/mission/emergency'          # std_msgs/Bool  (kill switch)

    # Algilama (mevcut dugumler)
    ZED_CLOUD = '/zed/zed_node/point_cloud/cloud_registered'
    ZED_IMAGE = '/zed/zed_node/rgb/color/rect/image'
    OBSTACLE_SEND = '/mavros/obstacle/send'   # BendyRuler beslemesi (LaserScan)
    OBSTACLE_GRID = '/zed_obstacle_map/grid'  # OccupancyGrid (Dosya 3 kaynagi)
    AVOID_SECTORS = '/obstacle_avoidance/sectors'
    AVOID_DECISION = '/obstacle_avoidance/decision'

    # YOLO / kamikaze
    YOLO_DETECTIONS = '/yolo/detections'
    YOLO_ANNOTATED = '/yolo/annotated_image'
    IHA_TARGET_COLOR = '/iha/target_color'
    KAMIKAZE_STATUS = '/kamikaze/status'

    # MAVROS
    MAVROS_STATE = '/mavros/state'
    MAVROS_WP_REACHED = '/mavros/mission/reached'
    MAVROS_GLOBAL_POS = '/mavros/global_position/global'
    MAVROS_SETPOINT_VEL = '/mavros/setpoint_velocity/cmd_vel_unstamped'
    MAVROS_SETPOINT_GLOBAL = '/mavros/setpoint_position/global'  # GUIDED hedef (P2)
    MAVROS_WAYPOINTS = '/mavros/mission/waypoints'  # yuklu gorev (WP5 konumu icin)
    MAVROS_STATUSTEXT = '/mavros/statustext/recv'   # IHA renk (MAVLink STATUSTEXT)


# ---------------------------------------------------------------------------
# SERVIS ISIMLERI
# ---------------------------------------------------------------------------
class Services:
    SET_TARGET_COLOR = '/mission/set_target_color'
    SET_PARKUR = '/mission/set_parkur'            # sadece test/debug
    EMERGENCY_STOP = '/mission/emergency_stop'     # std_srvs/Trigger

    OBSTACLE_BRIDGE_ENABLE = '/obstacle_bridge/enable'   # std_srvs/SetBool
    KAMIKAZE_ENABLE = '/kamikaze/enable'                 # std_srvs/SetBool


# ---------------------------------------------------------------------------
# QoS PROFILLERI  (ros2-robotics skill: qos-mismatch = sessiz baglanti hatasi)
# ---------------------------------------------------------------------------

# Sensor verisi: ZED gibi kaynaklar BEST_EFFORT yayinlar. Aboneligi de
# BEST_EFFORT yapmazsan HICBIR veri gelmez ama hata da almazsin.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# Komut/karar verisi: mutlaka ulassin.
COMMAND_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# Durum/latch: gec baglanan dugum de son degeri alsin (mission state gibi).
STATE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
