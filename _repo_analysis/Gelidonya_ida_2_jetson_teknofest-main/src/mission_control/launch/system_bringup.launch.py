"""
system_bringup.launch.py  —  TAM SİSTEM (Jetson Orin Nano'ya güç verince her şey)
=============================================================================

Tek komutla İDA'nın tüm otonomi yığınını ayağa kaldırır:
  1. ZED2i sürücüsü (zed_wrapper)
  2. MAVROS (CUAV X7+ ↔ ROS2 köprüsü)
  3. Görev düğümleri (mission_bringup: manager, kamikaze, iha_color, logger,
     obstacle_bridge, zed_yolo)

Kullanım:
  ros2 launch mission_control system_bringup.launch.py

Boot'ta otomatik başlatma için: deploy/gelidonya-mission.service (systemd).

Argümanlar:
  fcu_url       : MAVROS'un CUAV X7+'a bağlantısı (varsayilan USB)
  use_zed       : ZED2i sürücüsünü başlat (default true)
  use_mavros    : MAVROS'u başlat (default true)
  params_file   : görev parametre yaml'ı
  mission_delay : ZED/MAVROS otursun diye görev düğümlerini geciktirme (s)

NOT: ZED wrapper ve MAVROS harici paketlerdir; launch dosyası isimleri/argümanları
kurulu sürüme göre değişebilir. Aşağıdaki include'lar zed-ros2-wrapper ve
mavros ROS2 (Humble) varsayar. Farklıysa ilgili bloğu düzenle ya da o bileşeni
use_zed:=false / use_mavros:=false ile kapatıp ayrı başlat.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('mission_control')
    default_params = os.path.join(pkg_share, 'config', 'mission_params.yaml')

    # ------------------------------------------------------------------
    # Argümanlar
    # ------------------------------------------------------------------
    args = [
        DeclareLaunchArgument(
            'fcu_url',
            # Jetson CUAV X7+ TELEM2 (SERIAL2, 921600) portuna bağlı.
            # Cihaz adı bağlantı şekline göre değişir:
            #   USB-TTL adaptör      -> /dev/ttyUSB0
            #   Jetson 40-pin UART   -> /dev/ttyTHS1
            #   FC'nin USB portu     -> /dev/ttyACM0  (TELEM2 değil, SERIAL0)
            # Doğrula: ls -l /dev/ttyUSB* /dev/ttyTHS* /dev/serial/by-id/
            default_value='serial:///dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0:115200',
            description='MAVROS FCU bağlantısı (CUAV X7+)'
        ),
        DeclareLaunchArgument('use_zed', default_value='true'),
        DeclareLaunchArgument('use_mavros', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument(
            'mission_delay', default_value='6.0',
            description='ZED/MAVROS otursun diye görev düğümü gecikmesi (s)'
        ),
    ]

    params = LaunchConfiguration('params_file')

    # ------------------------------------------------------------------
    # 1) ZED2i sürücüsü
    # ------------------------------------------------------------------
    zed = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('zed_wrapper'), 'launch', 'zed_camera.launch.py'
            ])
        ]),
        launch_arguments={'camera_model': 'zed2i'}.items(),
        condition=IfCondition(LaunchConfiguration('use_zed')),
    )

    # ------------------------------------------------------------------
    # 2) MAVROS (CUAV X7+ / ArduPilot köprüsü)
    #    Node olarak çalıştırılıyor; plugin allowlist gerekiyorsa mavros
    #    apm_pluginlists.yaml / apm_config.yaml parametre olarak eklenebilir.
    # ------------------------------------------------------------------
    # DIKKAT: mavros_node bir CONTAINER'dir; icinde 'mavros_router' ve 'mavros'
    # (UAS) dugumlerini kendisi olusturur. Disaridan name= / namespace= vermek
    # bu isimlendirmeyi bozar ve dugum hic ayaga kalkmaz (topic'ler gorunur
    # ama yayinci yoktur -> 'ros2 topic echo' sonsuz bekler). Varsayilanlari
    # birak: topic'ler zaten /mavros/... altinda olusur.
    mavros = Node(
        package='mavros', executable='mavros_node', output='screen',
        parameters=[{
            'fcu_url': LaunchConfiguration('fcu_url'),
            'gcs_url': '',
            'tgt_system': 1,
            'tgt_component': 1,
            'fcu_protocol': 'v2.0',
            # ArduPilot proximity verisini DISTANCE_SENSOR olarak geri yayinliyor;
            # kullanmiyoruz ve plugin saniyede 3 hata basip gercek hatalari
            # gizliyor ('DS: no mapping for sensor id'). Kapat.
            'plugin_denylist': ['distance_sensor'],
        }],
        condition=IfCondition(LaunchConfiguration('use_mavros')),
    )

    # ------------------------------------------------------------------
    # 3) Görev düğümleri — ZED/MAVROS otursun diye gecikmeli başlat
    # ------------------------------------------------------------------
    mission = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('mission_control'),
                'launch', 'mission_bringup.launch.py'
            ])
        ]),
        launch_arguments={'params_file': params}.items(),
    )
    mission_delayed = TimerAction(
        period=LaunchConfiguration('mission_delay'),
        actions=[LogInfo(msg='Gorev dugumleri baslatiliyor...'), mission],
    )

    return LaunchDescription(args + [zed, mavros, mission_delayed])
