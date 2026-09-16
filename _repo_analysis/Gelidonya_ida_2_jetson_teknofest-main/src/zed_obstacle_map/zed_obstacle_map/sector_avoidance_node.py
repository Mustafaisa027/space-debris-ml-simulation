import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String


class SectorAvoidanceNode(Node):
    """
    ZED2i PointCloud2 verisinden Parkur-2 için engel kaçış kararı üretir.

    Kullanım amacı:
    - ZED2i canlı point cloud verisini okur.
    - 0.3 - 4.0 m aralığında engel analizi yapar.
    - Su yüzeyi / gürültü / uzak nokta filtrelemesi uygular.
    - Ön çarpışma koridorunu ayrıca kontrol eder.
    - Sol / ön / sağ açıklık bilgisini çıkarır.
    - Bendy Ruler benzeri aday yön taramasıyla kaçış yönü üretir.

    Çıktılar:
    /obstacle_avoidance/sectors
    /obstacle_avoidance/decision
    """

    def __init__(self):
        super().__init__('sector_avoidance_node')

        # ============================================================
        # PARAMETRELER
        # ============================================================

        self.declare_parameter(
            'cloud_topic',
            '/zed/zed_node/point_cloud/cloud_registered'
        )

        # coordinate_mode:
        #
        # ros:
        #   forward = x
        #   lateral = y   -> sol pozitif
        #   height  = z
        #
        # optical:
        #   forward = z
        #   lateral = -x  -> optical frame'de x sağ olduğu için sol pozitif yapmak adına -x
        #   height  = -y
        self.declare_parameter('coordinate_mode', 'ros')

        # İDA için çok uzağa bakmaya gerek yok.
        # Parkur-2 için 3-4 metre yeterli.
        self.declare_parameter('min_forward', 0.30)
        self.declare_parameter('max_forward', 4.00)

        # Sağ-sol algılama genişliği
        self.declare_parameter('max_lateral', 2.50)

        # İlk testte geniş yükseklik filtresi kullanıyoruz.
        # Daha sonra su yüzeyi gürültüsüne göre daraltılabilir.
        self.declare_parameter('min_height', -1.50)
        self.declare_parameter('max_height', 2.00)

        # Güvenlik eşikleri
        self.declare_parameter('safe_distance', 1.50)
        self.declare_parameter('emergency_distance', 0.80)

        # Aracın ön çarpışma koridoru yarı genişliği.
        # 0.85 m demek toplam 1.70 m genişlikte ön koridor izlenir.
        self.declare_parameter('front_corridor_half_width', 0.85)

        # Gürültü bastırma eşikleri.
        # İlk testte düşük tutuldu; engel kaçırmamak için.
        self.declare_parameter('min_close_points', 1)
        self.declare_parameter('min_sector_points', 1)

        # PointCloud çok yoğunsa örnekleme yapılır.
        # İlk testte 1 kullanıyoruz, yani her noktayı oku.
        self.declare_parameter('point_stride', 1)

        # Aday yönlerin açısal genişliği
        self.declare_parameter('candidate_width_deg', 20.0)

        # Hız değerleri
        self.declare_parameter('forward_speed', 0.70)
        self.declare_parameter('avoid_speed', 0.40)
        self.declare_parameter('slow_speed', 0.20)

        self.declare_parameter('print_debug', True)

        # ============================================================
        # PARAMETRELERİ OKU
        # ============================================================

        self.cloud_topic = self.get_parameter('cloud_topic').value
        self.coordinate_mode = self.get_parameter('coordinate_mode').value

        self.min_forward = float(self.get_parameter('min_forward').value)
        self.max_forward = float(self.get_parameter('max_forward').value)
        self.max_lateral = float(self.get_parameter('max_lateral').value)

        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)

        self.safe_distance = float(self.get_parameter('safe_distance').value)
        self.emergency_distance = float(self.get_parameter('emergency_distance').value)

        self.front_corridor_half_width = float(
            self.get_parameter('front_corridor_half_width').value
        )

        self.min_close_points = int(self.get_parameter('min_close_points').value)
        self.min_sector_points = int(self.get_parameter('min_sector_points').value)
        self.point_stride = int(self.get_parameter('point_stride').value)

        self.candidate_width_deg = float(
            self.get_parameter('candidate_width_deg').value
        )

        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.avoid_speed = float(self.get_parameter('avoid_speed').value)
        self.slow_speed = float(self.get_parameter('slow_speed').value)

        self.print_debug = bool(self.get_parameter('print_debug').value)

        self.debug_counter = 0

        # ============================================================
        # ROS2 SUBSCRIBER / PUBLISHER
        # ============================================================

        # ZED BEST_EFFORT yayinlar -> sensor QoS (qos-mismatch onlenir)
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data
        )

        self.sector_pub = self.create_publisher(
            Float32MultiArray,
            '/obstacle_avoidance/sectors',
            10
        )

        self.decision_pub = self.create_publisher(
            String,
            '/obstacle_avoidance/decision',
            10
        )

        self.get_logger().info(f'Listening PointCloud2: {self.cloud_topic}')
        self.get_logger().info(f'Coordinate mode: {self.coordinate_mode}')
        self.get_logger().info('ROS mode expected: forward=x, lateral=y, height=z')
        self.get_logger().info('Optical mode expected: forward=z, lateral=-x, height=-y')
        self.get_logger().info('Publishing: /obstacle_avoidance/sectors')
        self.get_logger().info('Publishing: /obstacle_avoidance/decision')

    # ============================================================
    # POINTCLOUD OKUMA
    # ============================================================

    def read_points_xyz(self, cloud_msg):
        """
        PointCloud2 içinden x, y, z alanlarını okur.
        """

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

    # ============================================================
    # KOORDİNAT DÖNÜŞÜMÜ
    # ============================================================

    def convert_point(self, x, y, z):
        """
        ZED point cloud koordinatlarını kaçış algoritmasının kullandığı
        ortak eksen sistemine çevirir.

        Ortak eksen:
        forward: ileri yön, metre
        lateral: sol pozitif, sağ negatif, metre
        height : yukarı pozitif, metre
        """

        if self.coordinate_mode == 'optical':
            # Optical frame:
            # x: sağ
            # y: aşağı
            # z: ileri
            forward = z
            lateral = -x
            height = -y
        else:
            # ROS frame:
            # x: ileri
            # y: sol
            # z: yukarı
            forward = x
            lateral = y
            height = z

        return forward, lateral, height

    # ============================================================
    # YARDIMCI FONKSİYONLAR
    # ============================================================

    @staticmethod
    def percentile(values, ratio):
        if not values:
            return None

        values = sorted(values)
        index = int((len(values) - 1) * ratio)
        index = max(0, min(index, len(values) - 1))
        return values[index]

    @staticmethod
    def angle_diff(a, b):
        d = a - b

        while d > 180.0:
            d -= 360.0

        while d < -180.0:
            d += 360.0

        return d

    def clearance(self, distances):
        """
        Mesafe listesi içinden güvenilir açıklık değeri üretir.

        Minimum almak yerine 10. yüzdelik kullanıyoruz.
        Böylece tek bir gürültü noktası tüm kararı bozmaz.
        """

        if len(distances) < self.min_sector_points:
            return self.max_forward

        value = self.percentile(distances, 0.10)

        if value is None:
            return self.max_forward

        return value

    # ============================================================
    # ANA POINTCLOUD CALLBACK
    # ============================================================

    def cloud_callback(self, msg):
        valid_points = []

        left_distances = []
        front_distances = []
        right_distances = []
        corridor_distances = []

        close_count = 0
        emergency_count = 0

        roi_total = 0

        # Debug için min/max eksen değerleri
        min_forward = 999.0
        max_forward = -999.0
        min_lateral = 999.0
        max_lateral = -999.0
        min_height = 999.0
        max_height = -999.0

        for x, y, z in self.read_points_xyz(msg):
            forward, lateral, height = self.convert_point(x, y, z)

            # Debug eksen aralıkları
            min_forward = min(min_forward, forward)
            max_forward = max(max_forward, forward)

            min_lateral = min(min_lateral, lateral)
            max_lateral = max(max_lateral, lateral)

            min_height = min(min_height, height)
            max_height = max(max_height, height)

            # ----------------------------------------------------
            # ROI FİLTRESİ
            # ----------------------------------------------------

            if forward < self.min_forward or forward > self.max_forward:
                continue

            if abs(lateral) > self.max_lateral:
                continue

            if height < self.min_height or height > self.max_height:
                continue

            roi_total += 1

            distance = math.sqrt(forward * forward + lateral * lateral)
            angle_deg = math.degrees(math.atan2(lateral, forward))

            valid_points.append(
                (distance, angle_deg, forward, lateral, height)
            )

            # ----------------------------------------------------
            # ÖN ÇARPIŞMA KORİDORU
            # ----------------------------------------------------
            # Bu alan aracın doğrudan çarpabileceği ön bölgedir.
            # COR değeri buradan hesaplanır.
            # Eğer COR safe_distance altındaysa asla CLEAR_GO_FORWARD denmez.
            # ----------------------------------------------------

            if abs(lateral) <= self.front_corridor_half_width:
                corridor_distances.append(distance)

                if distance < self.safe_distance:
                    close_count += 1

                if distance < self.emergency_distance:
                    emergency_count += 1

            # ----------------------------------------------------
            # SOL / ÖN / SAĞ SEKTÖR ÖZETİ
            # ----------------------------------------------------

            if angle_deg > 20.0:
                left_distances.append(distance)
            elif angle_deg < -20.0:
                right_distances.append(distance)
            else:
                front_distances.append(distance)

        left_clearance = self.clearance(left_distances)
        front_clearance = self.clearance(front_distances)
        right_clearance = self.clearance(right_distances)
        corridor_clearance = self.clearance(corridor_distances)

        front_blocked = (
            corridor_clearance < self.safe_distance or
            close_count >= self.min_close_points or
            emergency_count >= max(1, int(self.min_close_points / 2))
        )

        # --------------------------------------------------------
        # BENDY RULER BENZERİ ADAY YÖN TARAMASI
        # --------------------------------------------------------

        candidate_angles = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]

        best_angle = 0.0
        best_clearance = 0.0
        best_cost = 1e9
        best_blocked = True

        for candidate_angle in candidate_angles:
            candidate_distances = []
            candidate_close = 0
            candidate_emergency = 0

            for distance, angle_deg, forward, lateral, height in valid_points:
                diff = abs(self.angle_diff(angle_deg, candidate_angle))

                if diff <= self.candidate_width_deg:
                    candidate_distances.append(distance)

                    if distance < self.safe_distance:
                        candidate_close += 1

                    if distance < self.emergency_distance:
                        candidate_emergency += 1

            candidate_clearance = self.clearance(candidate_distances)

            candidate_blocked = (
                candidate_clearance < self.safe_distance or
                candidate_close >= self.min_close_points or
                candidate_emergency >= max(1, int(self.min_close_points / 2))
            )

            # Maliyet hesabı:
            # - Hedeften sapma az olsun.
            # - Engel uzaklığı fazla olsun.
            # - Bloke yönler ağır cezalandırılsın.
            target_cost = abs(candidate_angle) * 0.04
            obstacle_cost = max(0.0, self.max_forward - candidate_clearance) * 0.70
            turn_cost = abs(candidate_angle) * 0.015
            blocked_cost = 100.0 if candidate_blocked else 0.0

            cost = target_cost + obstacle_cost + turn_cost + blocked_cost

            if cost < best_cost:
                best_cost = cost
                best_angle = candidate_angle
                best_clearance = candidate_clearance
                best_blocked = candidate_blocked

        decision, yaw_deg, speed = self.make_decision(
            front_blocked=front_blocked,
            best_angle=best_angle,
            best_blocked=best_blocked,
            corridor_clearance=corridor_clearance,
            left_clearance=left_clearance,
            right_clearance=right_clearance
        )

        # --------------------------------------------------------
        # TOPIC YAYINI
        # --------------------------------------------------------

        sectors_msg = Float32MultiArray()

        sectors_msg.data = [
            float(left_clearance),
            float(front_clearance),
            float(right_clearance),
            float(corridor_clearance),
            float(yaw_deg),
            float(speed),
            float(roi_total),
            float(close_count),
            float(emergency_count),
            float(best_clearance),
            float(1.0 if front_blocked else 0.0),
            float(1.0 if best_blocked else 0.0),
            float(min_forward if min_forward != 999.0 else 0.0),
            float(max_forward if max_forward != -999.0 else 0.0),
            float(min_lateral if min_lateral != 999.0 else 0.0),
            float(max_lateral if max_lateral != -999.0 else 0.0),
            float(min_height if min_height != 999.0 else 0.0),
            float(max_height if max_height != -999.0 else 0.0),
        ]

        decision_msg = String()
        decision_msg.data = decision

        self.sector_pub.publish(sectors_msg)
        self.decision_pub.publish(decision_msg)

        # --------------------------------------------------------
        # DEBUG LOG
        # --------------------------------------------------------

        self.debug_counter += 1

        if self.print_debug and self.debug_counter % 5 == 0:
            self.get_logger().info(
                f'L:{left_clearance:.2f} '
                f'F:{front_clearance:.2f} '
                f'R:{right_clearance:.2f} '
                f'COR:{corridor_clearance:.2f} '
                f'CLOSE:{close_count} '
                f'EMG:{emergency_count} '
                f'ROI:{roi_total} '
                f'BEST:{best_angle:.1f}/{best_clearance:.2f} '
                f'DECISION:{decision} '
                f'YAW:{yaw_deg:.1f} '
                f'SPEED:{speed:.2f} '
                f'FWD[{min_forward:.2f},{max_forward:.2f}] '
                f'LAT[{min_lateral:.2f},{max_lateral:.2f}] '
                f'H[{min_height:.2f},{max_height:.2f}]'
            )

    # ============================================================
    # KARAR FONKSİYONU
    # ============================================================

    def make_decision(
        self,
        front_blocked,
        best_angle,
        best_blocked,
        corridor_clearance,
        left_clearance,
        right_clearance
    ):
        """
        Ana karar mantığı.

        En önemli kural:
        Ön koridor safe_distance altındaysa asla CLEAR_GO_FORWARD denmez.
        """

        # --------------------------------------------------------
        # 1. Ön koridor tehlikeli ise
        # --------------------------------------------------------

        if corridor_clearance < self.safe_distance or front_blocked:
            left_safe = left_clearance > self.safe_distance
            right_safe = right_clearance > self.safe_distance

            if left_safe and right_safe:
                if right_clearance > left_clearance:
                    return 'FRONT_BLOCKED_GO_RIGHT', -30.0, self.avoid_speed
                else:
                    return 'FRONT_BLOCKED_GO_LEFT', 30.0, self.avoid_speed

            if right_safe and not left_safe:
                return 'FRONT_BLOCKED_GO_RIGHT', -30.0, self.avoid_speed

            if left_safe and not right_safe:
                return 'FRONT_BLOCKED_GO_LEFT', 30.0, self.avoid_speed

            return 'STOP_OR_SLOW_SCAN', 0.0, 0.0

        # --------------------------------------------------------
        # 2. Ön açık ama aday yönler riskli ise
        # --------------------------------------------------------

        if best_blocked:
            return 'SLOW_FORWARD_SCAN', 0.0, self.slow_speed

        # --------------------------------------------------------
        # 3. Ön açık, en iyi yön düz ise
        # --------------------------------------------------------

        if abs(best_angle) <= 15.0:
            return 'CLEAR_GO_FORWARD', 0.0, self.forward_speed

        # --------------------------------------------------------
        # 4. Ön açık ama daha güvenli koridor solda ise
        # --------------------------------------------------------

        if best_angle > 15.0:
            return 'GO_LEFT_CORRIDOR', best_angle, self.avoid_speed

        # --------------------------------------------------------
        # 5. Ön açık ama daha güvenli koridor sağda ise
        # --------------------------------------------------------

        if best_angle < -15.0:
            return 'GO_RIGHT_CORRIDOR', best_angle, self.avoid_speed

        return 'CLEAR_GO_FORWARD', 0.0, self.forward_speed


def main(args=None):
    rclpy.init(args=args)
    node = SectorAvoidanceNode()
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
