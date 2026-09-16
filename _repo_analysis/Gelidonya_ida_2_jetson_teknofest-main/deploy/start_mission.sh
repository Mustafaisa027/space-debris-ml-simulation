#!/usr/bin/env bash
#
# start_mission.sh — Jetson Orin Nano'te tüm otonomi yığınını başlatır.
# systemd (gelidonya-mission.service) tarafından boot'ta çağrılır; elle de
# çalıştırılabilir. ROS ortamını source edip tam-sistem launch'ını başlatır.
#
set -e

# --- AYARLA: kendi kurulumuna göre ---
ROS_DISTRO_SETUP="/opt/ros/humble/setup.bash"
WORKSPACE="/home/gelidonya_2/obstacle_map_ws"
FCU_URL="serial:///dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0:115200"   # CUAV X7+ bağlantısı (USB/UART)
export ROS_DOMAIN_ID=0
# --------------------------------------

source "${ROS_DISTRO_SETUP}"
source "${WORKSPACE}/install/setup.bash"

# Jetson tam güç modu (opsiyonel ama Orin Nano'te önerilir)
# sudo nvpmodel -m 0 >/dev/null 2>&1 || true
# sudo jetson_clocks >/dev/null 2>&1 || true

exec ros2 launch mission_control system_bringup.launch.py \
    fcu_url:="${FCU_URL}"
