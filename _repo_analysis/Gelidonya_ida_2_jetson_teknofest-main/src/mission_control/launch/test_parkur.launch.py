"""
test_parkur.launch.py — Parkurlari TEK TEK, donanimsiz test et
===============================================================

Sahte MAVROS (sim_mavros) + gorev dugumleri + istege bagli sahte hedef
(sim_target) ile tum zinciri ucdan uca kosturur. ZED, CUAV X7+ ve YOLO
modeli GEREKMEZ.

Kullanim:

  # Parkur-1: sadece waypoint takibi (WP1..WP4)
  ros2 launch mission_control test_parkur.launch.py parkur:=1

  # Parkur-2: GUIDED'de WP5'e gidis (engel kacinma zinciri acik)
  ros2 launch mission_control test_parkur.launch.py parkur:=2

  # Parkur-3: sahte hedefle kamikaze angajmani
  ros2 launch mission_control test_parkur.launch.py parkur:=3

  # Tam gorev: P1 -> P2 -> P3 arka arkaya
  ros2 launch mission_control test_parkur.launch.py parkur:=all

Baslatmak icin (launch ayaga kalktiktan sonra, ayri terminalde):
  ros2 service call /sim/arm std_srvs/srv/SetBool "{data: true}"

Izlemek icin:
  ros2 topic echo /mission/state

parkur:=1 secildiginde sim WP1..WP4'u gecip PARKUR2'ye gecer ve orada durur
(GUIDED hedefi gonderilir ama sen sadece P1 davranisini gozlersin). Izole test
istiyorsan parkur:=2 ile baslangic parkurunu dogrudan 2 yapabilirsin.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    parkur_arg = DeclareLaunchArgument(
        'parkur', default_value='all',
        description='Test edilecek parkur: 1 | 2 | 3 | all'
    )
    color_arg = DeclareLaunchArgument(
        'target_color', default_value='red',
        description='Parkur-3 hedef rengi: red | green | black'
    )
    auto_arm_arg = DeclareLaunchArgument(
        'auto_arm', default_value='false',
        description='Sim otomatik arm olsun mu (false: elle /sim/arm cagir)'
    )

    parkur = LaunchConfiguration('parkur')
    target_color = LaunchConfiguration('target_color')

    # Parkur-3 (veya all) secildiyse sahte hedef uretilir
    need_target = PythonExpression(
        ["'", parkur, "' in ('3', 'all')"]
    )
    # Parkur-3 izole test ediliyorsa sim dogrudan P3'e alinmali
    start_at_p3 = PythonExpression(["'", parkur, "' == '3'"])

    sim_mavros = Node(
        package='mission_control', executable='sim_mavros_node',
        name='sim_mavros_node', output='screen',
        parameters=[{
            'auto_arm': LaunchConfiguration('auto_arm'),
            'speed_mps': 3.0,
            'wp_spacing_m': 20.0,
        }],
    )

    mission_manager = Node(
        package='mission_control', executable='mission_manager_node',
        name='mission_manager_node', output='screen',
        parameters=[{
            'require_target_color': True,
            'kamikaze_timeout_s': 60.0,
            'p2_arrival_radius_m': 2.5,
        }],
    )

    kamikaze = Node(
        package='mission_control', executable='kamikaze_node',
        name='kamikaze_node', output='screen',
        parameters=[{
            # Testte hiz komutu ACIK: sim tekne hareket etsin ki
            # angajman zinciri ucdan uca dogrulansin. Gercek arac icin
            # mission_params.yaml'da false kalir.
            'send_velocity': True,
            'image_width': 1280,
            'image_height': 720,
        }],
    )

    logger = Node(
        package='mission_control', executable='mission_logger_node',
        name='mission_logger_node', output='screen',
    )
    telemetry = Node(
        package='mission_control', executable='telemetry_logger_node',
        name='telemetry_logger_node', output='screen',
    )

    sim_target = Node(
        package='mission_control', executable='sim_target_node',
        name='sim_target_node', output='screen',
        parameters=[{
            'target_class': target_color,
            'approach_time_s': 12.0,
        }],
        condition=IfCondition(need_target),
    )

    # Hedef rengini gorev baslamadan once ata (sartname: harekete
    # baslamadan aktarilmali).
    set_color = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/mission/set_target_color',
                 'mission_interfaces/srv/SetTargetColor', '{color: 1}'],
            output='screen',
        )],
    )

    # Parkur-3 izole: sim'i dogrudan P3'e al
    jump_p3 = TimerAction(
        period=6.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/mission/set_parkur',
                 'mission_interfaces/srv/SetParkur', '{parkur: 3}'],
            output='screen',
        )],
        condition=IfCondition(start_at_p3),
    )

    return LaunchDescription([
        parkur_arg, color_arg, auto_arm_arg,
        sim_mavros, mission_manager, kamikaze, logger, telemetry,
        sim_target, set_color, jump_p3,
    ])
