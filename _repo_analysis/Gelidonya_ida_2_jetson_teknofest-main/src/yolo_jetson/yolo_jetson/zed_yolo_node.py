import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class ZedYoloNode(Node):
    def __init__(self):
        super().__init__('zed_yolo_node')

        self.declare_parameter(
            'image_topic',
            '/zed/zed_node/rgb/color/rect/image'
        )
        self.declare_parameter(
            'model_path',
            '/home/gelidonya_2/yolo_jetson/best.pt'
        )
        self.declare_parameter('device', 0)
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('show', False)
        self.declare_parameter('process_every_n_frames', 2)

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        self.show = bool(self.get_parameter('show').value)
        self.process_every_n_frames = int(
            self.get_parameter('process_every_n_frames').value
        )

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.frame_count = 0
        self.processed_count = 0
        self.last_time = time.time()

        # ZED image topic'i BEST_EFFORT (sensor) QoS ile yayinlar. Aboneligi de
        # sensor QoS yapmazsak baglanti sessizce kurulmaz ve YOLO hic kare almaz
        # -> Parkur-3 tespiti calismaz (ros2-robotics: qos-mismatch /
        # default-qos-for-sensors). Dogrula:
        #   ros2 topic info /zed/zed_node/rgb/color/rect/image --verbose
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.detection_pub = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )

        self.annotated_pub = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )

        self.get_logger().info(f'Listening image topic: {self.image_topic}')
        self.get_logger().info(f'YOLO model loaded: {self.model_path}')
        self.get_logger().info('Publishing detections: /yolo/detections')
        self.get_logger().info('Publishing annotated image: /yolo/annotated_image')

    def image_callback(self, msg):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion error: {e}')
            return

        try:
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference error: {e}')
            return

        result = results[0]
        detections_text = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = self.model.names.get(cls_id, str(cls_id))

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections_text.append(
                    f'class={class_name}, conf={conf:.2f}, '
                    f'bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), '
                    f'center=({cx},{cy})'
                )

        detection_msg = String()

        if detections_text:
            detection_msg.data = ' | '.join(detections_text)
        else:
            detection_msg.data = 'no_detection'

        self.detection_pub.publish(detection_msg)

        if self.publish_annotated or self.show:
            annotated = result.plot()

            if self.publish_annotated:
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated,
                    encoding='bgr8'
                )
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)

            if self.show:
                cv2.imshow('ZED YOLO', annotated)
                cv2.waitKey(1)

        # FPS: gercekten islenen kare sayisini say.
        # Onceki hesap 'process_every_n_frames / dt' idi; dt ~1 sn oldugu icin
        # sonuc her zaman ~2.0 cikiyordu ve gercek hizi olcmuyordu (TensorRT'ye
        # gecince bile deger degismedi, hata boylece ortaya cikti).
        self.processed_count += 1
        now = time.time()
        dt = now - self.last_time
        if dt > 1.0:
            fps = self.processed_count / dt
            self.get_logger().info(
                f'YOLO | tespit: {len(detections_text)} | '
                f'islenen FPS: {fps:.1f} (kamera karesi ~{fps * self.process_every_n_frames:.1f})'
            )
            self.processed_count = 0
            self.last_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ZedYoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.ok() kontrolu olmadan cift shutdown RCLError firlatir
        # (ros2-robotics: no-shutdown-handler). Ctrl+C'de launch zaten
        # context'i kapatiyor; guard olmadan dugum hata ile oluyordu.
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

