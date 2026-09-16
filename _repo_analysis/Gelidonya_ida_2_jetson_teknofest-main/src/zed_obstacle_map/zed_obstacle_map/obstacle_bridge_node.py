import math
import struct

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, PointCloud2
from std_srvs.srv import SetBool
from rclpy.qos import qos_profile_sensor_data


class ObstacleBridgeNode(Node):
    """
    ZED2i PointCloud2 verisini ArduPilot BendyRuler icin LaserScan'e cevirir.

    Bu node yalnizca algilama koprusudur; karar veya kacis mantigi icermez.
    """

    def __init__(self):
        super().__init__('obstacle_bridge_node')

        self.declare_parameter(
            'cloud_topic',
            '/zed/zed_node/point_cloud/cloud_registered'
        )
        self.declare_parameter('scan_topic', '/mavros/obstacle/send')

        # coordinate_mode:
        #
        # ros:
        #   forward = x
        #   lateral = y   -> sol pozitif
        #   height  = z
        #
        # optical:
        #   forward = z
        #   lateral = -x
        #     optical frame'de x sag oldugu icin sol pozitif yapmak adina -x
        #   height  = -y
        self.declare_parameter('coordinate_mode', 'ros')

        # Algilama araligi: 0.30 - 3.00 m.
        # Alt sinir ZED2i'nin guvenilir stereo derinlik baslangicina ve tekne
        # burnunun goruse girmemesine gore; ust sinir Parkur-2'de dubalar arasi
        # manevra mesafesine gore secildi. Daha uzagi BendyRuler'a bildirmek
        # gereksiz erken kacisa ve parkur disina cikmaya yol acar.
        self.declare_parameter('min_forward', 0.30)
        self.declare_parameter('max_range', 3.00)
        # Su yuzeyi filtresi: dalga/yansima noktalari alt sinirin altinda kalir.
        self.declare_parameter('min_height', -0.20)
        self.declare_parameter('max_height', 2.00)

        self.declare_parameter('fov_deg', 100.0)
        self.declare_parameter('sector_count', 40)
        self.declare_parameter('percentile_ratio', 0.10)

        self.declare_parameter('point_stride', 1)
        self.declare_parameter('scan_time', 0.10)
        self.declare_parameter('frame_id', '')
        self.declare_parameter('print_debug', True)

        # start_enabled: tek basina calistirildiginda True kalir (davranis
        # degismez). mission_manager, Parkur-1'de bunu /obstacle_bridge/enable
        # servisiyle False yapar; Parkur-2'de True yapar. Boylece BendyRuler
        # sadece engelli parkurda beslenir.
        self.declare_parameter('start_enabled', True)

        self.cloud_topic = self.get_parameter('cloud_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.coordinate_mode = self.get_parameter('coordinate_mode').value

        self.min_forward = float(self.get_parameter('min_forward').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)

        self.fov_deg = float(self.get_parameter('fov_deg').value)
        self.sector_count = max(
            1,
            int(self.get_parameter('sector_count').value)
        )
        self.percentile_ratio = float(
            self.get_parameter('percentile_ratio').value
        )

        self.point_stride = max(
            1,
            int(self.get_parameter('point_stride').value)
        )
        self.scan_time = float(self.get_parameter('scan_time').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.print_debug = bool(self.get_parameter('print_debug').value)
        self.enabled = bool(self.get_parameter('start_enabled').value)

        # mission_manager bu servisle Parkur-2 disinda BendyRuler beslemesini keser.
        self.enable_srv = self.create_service(
            SetBool,
            'obstacle_bridge/enable',
            self.enable_callback
        )

        self.fov_rad = math.radians(self.fov_deg)
        self.angle_min_edge = -0.5 * self.fov_rad
        self.angle_max_edge = 0.5 * self.fov_rad
        self.sector_width = self.fov_rad / float(self.sector_count)

        # ZED point cloud'u BEST_EFFORT (sensor) QoS ile yayinlar. Aboneligi
        # de sensor QoS yapmazsak baglanti sessizce kurulmaz ve BendyRuler'a
        # veri gitmez (ros2-robotics skill: qos-mismatch). Dogrula:
        #   ros2 topic info /zed/zed_node/point_cloud/cloud_registered --verbose
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data
        )

        self.scan_pub = self.create_publisher(
            LaserScan,
            self.scan_topic,
            10
        )

        self.debug_counter = 0

        self.get_logger().info(f'Listening PointCloud2: {self.cloud_topic}')
        self.get_logger().info(f'Publishing LaserScan: {self.scan_topic}')
        self.get_logger().info(f'Coordinate mode: {self.coordinate_mode}')
        self.get_logger().info(
            'ROS mode expected: forward=x, lateral=y, height=z'
        )
        self.get_logger().info(
            'Optical mode expected: forward=z, lateral=-x, height=-y'
        )
        self.get_logger().info(
            f'FOV: {self.fov_deg:.1f} deg, sectors: {self.sector_count}, '
            f'range: {self.min_forward:.2f}-{self.max_range:.2f} m'
        )

    def read_points_xyz(self, cloud_msg):
        fields = {field.name: field.offset for field in cloud_msg.fields}

        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            self.get_logger().warn(
                'PointCloud2 does not contain x/y/z fields.'
            )
            return

        x_offset = fields['x']
        y_offset = fields['y']
        z_offset = fields['z']

        fmt = '>f' if cloud_msg.is_bigendian else '<f'
        point_step = cloud_msg.point_step
        data = cloud_msg.data
        step = point_step * self.point_stride

        for i in range(0, len(data), step):
            try:
                x = struct.unpack_from(fmt, data, i + x_offset)[0]
                y = struct.unpack_from(fmt, data, i + y_offset)[0]
                z = struct.unpack_from(fmt, data, i + z_offset)[0]
            except struct.error:
                continue

            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                yield x, y, z

    def convert_point(self, x, y, z):
        if self.coordinate_mode == 'optical':
            forward = z
            lateral = -x
            height = -y
        else:
            forward = x
            lateral = y
            height = z

        return forward, lateral, height

    @staticmethod
    def percentile(values, ratio):
        if not values:
            return math.inf

        values = sorted(values)
        ratio = max(0.0, min(1.0, ratio))
        index = int((len(values) - 1) * ratio)
        return values[index]

    def enable_callback(self, request, response):
        self.enabled = bool(request.data)
        response.success = True
        response.message = f'obstacle_bridge enabled={self.enabled}'
        self.get_logger().info(response.message)
        return response

    def cloud_callback(self, msg):
        # Parkur-2 disinda BendyRuler'a veri gonderme (mission_manager kontrolu).
        if not self.enabled:
            return

        sector_distances = [[] for _ in range(self.sector_count)]
        roi_total = 0
        fov_total = 0

        for x, y, z in self.read_points_xyz(msg):
            forward, lateral, height = self.convert_point(x, y, z)

            if forward < self.min_forward:
                continue

            if height < self.min_height or height > self.max_height:
                continue

            distance = math.sqrt(forward * forward + lateral * lateral)

            if distance > self.max_range:
                continue

            roi_total += 1
            angle = math.atan2(lateral, forward)

            if angle < self.angle_min_edge or angle >= self.angle_max_edge:
                continue

            sector_index = int(
                (angle - self.angle_min_edge) / self.sector_width
            )
            sector_index = max(0, min(sector_index, self.sector_count - 1))
            sector_distances[sector_index].append(distance)
            fov_total += 1

        scan_msg = LaserScan()
        scan_msg.header = msg.header

        if self.frame_id:
            scan_msg.header.frame_id = self.frame_id

        # Ranges sektor merkezlerini temsil eder.
        scan_msg.angle_increment = self.sector_width
        scan_msg.angle_min = self.angle_min_edge + 0.5 * self.sector_width
        scan_msg.angle_max = (
            scan_msg.angle_min +
            scan_msg.angle_increment * float(self.sector_count - 1)
        )
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = self.scan_time
        scan_msg.range_min = self.min_forward
        scan_msg.range_max = self.max_range
        scan_msg.ranges = [
            float(self.percentile(distances, self.percentile_ratio))
            for distances in sector_distances
        ]
        scan_msg.intensities = []

        self.scan_pub.publish(scan_msg)

        self.debug_counter += 1

        if self.print_debug and self.debug_counter % 10 == 0:
            occupied = sum(
                1 for value in scan_msg.ranges if math.isfinite(value)
            )
            nearest = min(scan_msg.ranges) if occupied else math.inf
            nearest_text = (
                f'{nearest:.2f}' if math.isfinite(nearest) else 'inf'
            )

            self.get_logger().info(
                f'ROI:{roi_total} FOV:{fov_total} '
                f'OCC:{occupied}/{self.sector_count} NEAREST:{nearest_text}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleBridgeNode()
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
