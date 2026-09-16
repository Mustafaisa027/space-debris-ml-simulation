"""
mission_bringup.launch.py
=========================

Tum gorev sistemini tek komutla ayaga kaldirir:

  ros2 launch mission_control mission_bringup.launch.py

Baslattigi dugumler:
  mission_control:  mission_manager, iha_color_receiver, kamikaze, mission_logger
  zed_obstacle_map: obstacle_bridge (Parkur-2 BendyRuler beslemesi)
  yolo_jetson:      zed_yolo  (Parkur-3 tespit)  [use_yolo:=true ise]

Argumanlar:
  params_file   : parametre yaml yolu
  use_yolo      : YOLO dugumunu baslat (default true)
  use_avoidance : obstacle_bridge dugumunu baslat (default true)

NOT: ZED wrapper ve MAVROS ayri launch'lardir; onlar zaten calisiyor
varsayilir. Bu launch sadece gorev katmanini baslatir.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('mission_control')
    default_params = os.path.join(pkg_share, 'config', 'mission_params.yaml')

    params_file = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Gorev parametre yaml dosyasi'
    )
    use_yolo = DeclareLaunchArgument(
        'use_yolo', default_value='true',
        description='YOLO tespit dugumunu baslat'
    )
    use_avoidance = DeclareLaunchArgument(
        'use_avoidance', default_value='true',
        description='obstacle_bridge (BendyRuler beslemesi) dugumunu baslat'
    )
    use_recording = DeclareLaunchArgument(
        'use_recording', default_value='true',
        description='Sartname veri teslimi (Dosya 1/2/3) kayit dugumleri'
    )

    params = LaunchConfiguration('params_file')

    mission_manager = Node(
        package='mission_control', executable='mission_manager_node',
        name='mission_manager_node', output='screen', parameters=[params],
    )
    iha_color = Node(
        package='mission_control', executable='iha_color_receiver_node',
        name='iha_color_receiver_node', output='screen', parameters=[params],
    )
    kamikaze = Node(
        package='mission_control', executable='kamikaze_node',
        name='kamikaze_node', output='screen', parameters=[params],
    )
    blackbox = Node(   # kara kutu: /rosout + FC mesajlari + saglik + ozet
        package='mission_control', executable='blackbox_node',
        name='blackbox_node', output='screen', parameters=[params],
    )
    logger = Node(
        package='mission_control', executable='mission_logger_node',
        name='mission_logger_node', output='screen', parameters=[params],
    )
    obstacle_bridge = Node(
        package='zed_obstacle_map', executable='obstacle_bridge_node',
        name='obstacle_bridge_node', output='screen', parameters=[params],
        condition=IfCondition(LaunchConfiguration('use_avoidance')),
    )
    yolo = Node(
        package='yolo_jetson', executable='zed_yolo_node',
        name='zed_yolo_node', output='screen', parameters=[params],
        condition=IfCondition(LaunchConfiguration('use_yolo')),
    )

    # --- Sartname veri teslimi (Dosya 1/2/3) ---
    telemetry_logger = Node(   # Dosya 2: telemetri CSV
        package='mission_control', executable='telemetry_logger_node',
        name='telemetry_logger_node', output='screen', parameters=[params],
        condition=IfCondition(LaunchConfiguration('use_recording')),
    )
    video_recorder = Node(     # Dosya 1: kamera mp4, Dosya 3: harita mp4
        package='mission_control', executable='video_recorder_node',
        name='video_recorder_node', output='screen', parameters=[params],
        condition=IfCondition(LaunchConfiguration('use_recording')),
    )
    obstacle_map = Node(       # Dosya 3'un kaynagi (OccupancyGrid)
        package='zed_obstacle_map', executable='obstacle_map_node',
        name='obstacle_map_node', output='screen', parameters=[params],
        condition=IfCondition(LaunchConfiguration('use_recording')),
    )

    return LaunchDescription([
        params_file, use_yolo, use_avoidance, use_recording,
        mission_manager, iha_color, kamikaze, logger, blackbox,
        obstacle_bridge, yolo,
        telemetry_logger, video_recorder, obstacle_map,
    ])
