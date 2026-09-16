import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Float32MultiArray


class PointCloudFilterNode(Node):
    """
    ZED2i ham PointCloud2 verisini filtreleyerek sadece engel olabilecek
    noktaları /perception/filtered_obstacle_cloud topic'i olarak yayınlar.

    Girdi:
        /zed/zed_node/point_cloud/cloud_registered

    Çıktı:
        /perception/filtered_obstacle_cloud
        /perception/filter_stats
    """

    def __init__(self):
        super().__init__('pointcloud_filter_node')

        self.declare_parameter(
            'cloud_topic',
            '/zed/zed_node/point_cloud/cloud_registered'
        )

        self.declare_parameter(
            'output_topic',
            '/perception/filtered_obstacle_cloud'
        )

        # coordinate_mode:
        # ros:
        #   forward = x
        #   lateral = y
        #   height  = z
        #
        # optical:
        #   forward = z
        #   lateral = -x
        #   height  = -y
        self.declare_parameter('coordinate_mode', 'ros')

        # Parkur-2 için değerlendirme mesafesi
        self.declare_parameter('min_forward', 0.50)
        self.declare_parameter('max_forward', 4.00)

        # Sağ-sol filtreleme genişliği
        self.declare_parameter('max_lateral', 2.50)

        # Su yüzeyi / zemin / gökyüzü filtresi
        # İlk testte su yüzeyi noktalarını elemek için min_height pozitif başlatıyoruz.
        self.declare_parameter('min_height', 0.05)
        self.declare_parameter('max_height', 1.20)

        # Yoğun point cloud Jetson'u yormasın diye örnekleme
        self.declare_parameter('point_stride', 2)

        # Voxel downsample: aynı küçük hücre içindeki noktaları azaltır.
        # 0.05 m = 5 cm voxel
        self.declare_parameter('voxel_size', 0.05)

        # Çok fazla nokta yayınlanırsa RViz ve Jetson yorulur.
        self.declare_parameter('max_output_points', 20000)

        self.declare_parameter('print_debug', True)

        self.cloud_topic = self.get_parameter('cloud_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.coordinate_mode = self.get_parameter('coordinate_mode').value

        self.min_forward = float(self.get_parameter('min_forward').value)
        self.max_forward = float(self.get_parameter('max_forward').value)
        self.max_lateral = float(self.get_parameter('max_lateral').value)

        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)

        self.point_stride = int(self.get_parameter('point_stride').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.max_output_points = int(self.get_parameter('max_output_points').value)

        self.print_debug = bool(self.get_parameter('print_debug').value)
        self.debug_counter = 0

        # ZED BEST_EFFORT yayinlar -> sensor QoS (qos-mismatch onlenir)
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data
        )

        self.filtered_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )

        self.stats_pub = self.create_publisher(
            Float32MultiArray,
            '/perception/filter_stats',
            10
        )

        self.get_logger().info(f'Listening: {self.cloud_topic}')
        self.get_logger().info(f'Publishing filtered cloud: {self.output_topic}')
        self.get_logger().info(f'Coordinate mode: {self.coordinate_mode}')

    def read_points_xyz(self, cloud_msg):
        fields = {field.name: field.offset for field in cloud_msg.fields}

        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            self.get_logger().warn('PointCloud2 does not contain x/y/z fields.')
            return

        x_offset = fields['x']
        y_offset = fields['y']
        z_offset = fields['z']

        point_step = cloud_msg.point_step
        data = cloud_msg.data

        stride = max(1, self.point_stride)
        step = point_step * stride

        for i in range(0, len(data), step):
            try:
                x = struct.unpack_from('f', data, i + x_offset)[0]
                y = struct.unpack_from('f', data, i + y_offset)[0]
                z = struct.unpack_from('f', data, i + z_offset)[0]
            except struct.error:
                continue

            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                yield x, y, z

    def convert_point(self, x, y, z):
        """
        Filtreleme için noktayı ortak eksene çevirir.

        forward: ileri
        lateral: sol pozitif
        height : yukarı pozitif
        """

        if self.coordinate_mode == 'optical':
            forward = z
            lateral = -x
            height = -y
        else:
            forward = x
            lateral = y
            height = z

        return forward, lateral, height

    def pass_filter(self, forward, lateral, height):
        if forward < self.min_forward or forward > self.max_forward:
            return False

        if abs(lateral) > self.max_lateral:
            return False

        if height < self.min_height or height > self.max_height:
            return False

        return True

    def voxel_key(self, forward, lateral, height):
        v = self.voxel_size

        if v <= 0.0:
            return None

        return (
            int(forward / v),
            int(lateral / v),
            int(height / v)
        )

    def create_xyz_cloud(self, header, points):
        """
        Sadece x, y, z alanlarına sahip PointCloud2 üretir.

        Not:
        Burada yayınlanan x,y,z, orijinal ZED frame'indeki x,y,z değerleridir.
        Yani RViz'de ham point cloud ile aynı frame'de doğru görünür.
        """

        msg = PointCloud2()
        msg.header = header

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        buffer = bytearray()

        for x, y, z in points:
            buffer.extend(struct.pack('fff', x, y, z))

        msg.data = bytes(buffer)

        return msg

    def cloud_callback(self, msg):
        input_count = 0
        roi_count = 0
        output_points = []

        used_voxels = set()

        min_forward_seen = 999.0
        max_forward_seen = -999.0
        min_lateral_seen = 999.0
        max_lateral_seen = -999.0
        min_height_seen = 999.0
        max_height_seen = -999.0

        for x, y, z in self.read_points_xyz(msg):
            input_count += 1

            forward, lateral, height = self.convert_point(x, y, z)

            min_forward_seen = min(min_forward_seen, forward)
            max_forward_seen = max(max_forward_seen, forward)

            min_lateral_seen = min(min_lateral_seen, lateral)
            max_lateral_seen = max(max_lateral_seen, lateral)

            min_height_seen = min(min_height_seen, height)
            max_height_seen = max(max_height_seen, height)

            if not self.pass_filter(forward, lateral, height):
                continue

            roi_count += 1

            key = self.voxel_key(forward, lateral, height)

            if key is not None:
                if key in used_voxels:
                    continue
                used_voxels.add(key)

            output_points.append((x, y, z))

            if len(output_points) >= self.max_output_points:
                break

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id

        filtered_cloud = self.create_xyz_cloud(header, output_points)
        self.filtered_pub.publish(filtered_cloud)

        stats = Float32MultiArray()
        stats.data = [
            float(input_count),
            float(roi_count),
            float(len(output_points)),
            float(min_forward_seen if min_forward_seen != 999.0 else 0.0),
            float(max_forward_seen if max_forward_seen != -999.0 else 0.0),
            float(min_lateral_seen if min_lateral_seen != 999.0 else 0.0),
            float(max_lateral_seen if max_lateral_seen != -999.0 else 0.0),
            float(min_height_seen if min_height_seen != 999.0 else 0.0),
            float(max_height_seen if max_height_seen != -999.0 else 0.0),
        ]
        self.stats_pub.publish(stats)

        self.debug_counter += 1

        if self.print_debug and self.debug_counter % 10 == 0:
            self.get_logger().info(
                f'INPUT:{input_count} '
                f'ROI:{roi_count} '
                f'OUT:{len(output_points)} '
                f'FWD[{min_forward_seen:.2f},{max_forward_seen:.2f}] '
                f'LAT[{min_lateral_seen:.2f},{max_lateral_seen:.2f}] '
                f'H[{min_height_seen:.2f},{max_height_seen:.2f}]'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilterNode()
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
