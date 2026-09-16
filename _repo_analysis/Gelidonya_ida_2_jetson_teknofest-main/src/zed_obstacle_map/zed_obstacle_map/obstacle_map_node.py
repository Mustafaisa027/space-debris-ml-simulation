import math
import struct
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header


class ZedObstacleMapNode(Node):
    def __init__(self):
        super().__init__('zed_obstacle_map_node')

        self.declare_parameter('cloud_topic', '/zed/zed_node/point_cloud/cloud_registered')
        self.declare_parameter('map_frame', 'zed_left_camera_frame')

        self.declare_parameter('map_width_m', 8.0)
        self.declare_parameter('map_height_m', 8.0)
        self.declare_parameter('resolution', 0.10)

        self.declare_parameter('min_range', 0.4)
        self.declare_parameter('max_range', 6.0)

        self.declare_parameter('min_z', -0.5)
        self.declare_parameter('max_z', 1.5)

        self.cloud_topic = self.get_parameter('cloud_topic').value
        self.map_frame = self.get_parameter('map_frame').value

        self.map_width_m = float(self.get_parameter('map_width_m').value)
        self.map_height_m = float(self.get_parameter('map_height_m').value)
        self.resolution = float(self.get_parameter('resolution').value)

        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)

        self.width_cells = int(self.map_width_m / self.resolution)
        self.height_cells = int(self.map_height_m / self.resolution)

        # ZED BEST_EFFORT yayinlar -> sensor QoS (qos-mismatch onlenir)
        self.sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data
        )

        self.pub = self.create_publisher(
            OccupancyGrid,
            '/zed_obstacle_map/grid',
            10
        )

        self.get_logger().info(f'Listening cloud: {self.cloud_topic}')
        self.get_logger().info('Publishing map: /zed_obstacle_map/grid')

    def read_points_xyz(self, cloud_msg):
        """
        PointCloud2 içinden x, y, z noktalarını okur.
        Basit ve anlaşılır olması için struct ile okunuyor.
        """
        field_names = [field.name for field in cloud_msg.fields]

        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            self.get_logger().warn('PointCloud2 x/y/z fields not found.')
            return

        x_offset = next(f.offset for f in cloud_msg.fields if f.name == 'x')
        y_offset = next(f.offset for f in cloud_msg.fields if f.name == 'y')
        z_offset = next(f.offset for f in cloud_msg.fields if f.name == 'z')

        point_step = cloud_msg.point_step
        data = cloud_msg.data

        for i in range(0, len(data), point_step):
            try:
                x = struct.unpack_from('f', data, i + x_offset)[0]
                y = struct.unpack_from('f', data, i + y_offset)[0]
                z = struct.unpack_from('f', data, i + z_offset)[0]
            except struct.error:
                continue

            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                yield x, y, z

    def cloud_callback(self, msg):
        grid = [-1] * (self.width_cells * self.height_cells)

        """
        Harita mantığı:
        - Kamera önü pozitif X kabul edildi.
        - Y sağ-sol ekseni.
        - Harita 8m x 8m.
        - Kamera haritanın alt-orta kısmında kabul ediliyor.
        """

        for x, y, z in self.read_points_xyz(msg):
            distance = math.sqrt(x * x + y * y)

            if distance < self.min_range or distance > self.max_range:
                continue

            if z < self.min_z or z > self.max_z:
                continue

            # Sadece kameranın ön tarafını al
            if x < 0.0:
                continue

            # Harita koordinatı:
            # x ileri yön, y sağ-sol
            map_x = int(x / self.resolution)
            map_y = int((y + self.map_width_m / 2.0) / self.resolution)

            if 0 <= map_x < self.height_cells and 0 <= map_y < self.width_cells:
                index = map_x * self.width_cells + map_y
                grid[index] = 100

        occ = OccupancyGrid()
        occ.header = Header()
        occ.header.stamp = self.get_clock().now().to_msg()
        occ.header.frame_id = self.map_frame

        occ.info.resolution = self.resolution
        occ.info.width = self.width_cells
        occ.info.height = self.height_cells

        occ.info.origin.position.x = 0.0
        occ.info.origin.position.y = -self.map_width_m / 2.0
        occ.info.origin.position.z = 0.0
        occ.info.origin.orientation.w = 1.0

        occ.data = grid

        self.pub.publish(occ)


def main(args=None):
    rclpy.init(args=args)
    node = ZedObstacleMapNode()
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
