#!/usr/bin/env python3
"""
video_recorder_node — Sartname "Dosya 1" ve "Dosya 3"
======================================================

Dosya 1: Islenmis kamera verisi
  - En az 1 Hz, mp4
  - Her frame zaman etiketli
  - Tespit/takip sonucu obje cerceveleri + sinif bilgileri gorunur
  Kaynak: /yolo/annotated_image (zed_yolo_node zaten bbox+sinif cizer)
  Cikti:  <out_dir>/dosya1_kamera_<ts>.mp4

Dosya 3: Lokal harita / cost map / engel haritasi
  - En az 1 Hz, mp4, zaman etiketli
  Kaynak: /zed_obstacle_map/grid (OccupancyGrid)
  Cikti:  <out_dir>/dosya3_engel_haritasi_<ts>.mp4

Tasarim: mp4 kodlama pahalidir; ROS callback'inde yapilirsa executor bloklanir
ve sensor callback'leri gecikir (ros2-robotics: blocking-callback). Bu yuzden
her kayit icin sinirli boyutlu bir kuyruk + ayri yazici thread kullanilir.
Kuyruk dolarsa en eski kare atilir (kayit surekliligi > her karenin garantisi).
"""

import os
import queue
import threading
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

from .common import Topics, SENSOR_QOS, COMMAND_QOS


class VideoWriterWorker:
    """Kuyruktan kare alip mp4'e yazan arka plan isci."""

    def __init__(self, path, fps, logger, queue_size=30):
        self.path = path
        self.fps = fps
        self.logger = logger
        self.q = queue.Queue(maxsize=queue_size)
        self.writer = None
        self.size = None
        self.frames_written = 0
        self.frames_dropped = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Callback'ten cagrilir; asla bloklamaz."""
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            # En eskiyi at, yenisini koy (canli kayit onceligi)
            try:
                self.q.get_nowait()
                self.q.put_nowait(frame)
            except queue.Empty:
                pass
            self.frames_dropped += 1

    def _loop(self):
        while self._running:
            try:
                frame = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            self._write(frame)

    def _write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
            if not self.writer.isOpened():
                self.logger.error(f'VideoWriter acilamadi: {self.path}')
                self.writer = None
                return
            self.logger.info(f'Kayit basladi: {self.path} ({w}x{h} @ {self.fps} fps)')

        # Boyut degisirse yeniden olceklendir (writer tek boyut kabul eder)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)

        self.writer.write(frame)
        self.frames_written += 1

    def close(self):
        self._running = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)
        if self.writer is not None:
            self.writer.release()
            self.logger.info(
                f'Kayit kapandi: {self.path} '
                f'({self.frames_written} kare, {self.frames_dropped} atlandi)'
            )


def stamp_frame(frame, text):
    """Sol ust koseye okunakli zaman etiketi basar (siyah zemin + beyaz yazi)."""
    cv2.rectangle(frame, (0, 0), (max(320, 9 * len(text)), 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class VideoRecorderNode(Node):

    def __init__(self):
        super().__init__('video_recorder_node')

        self.declare_parameter('out_dir', os.path.expanduser('~/gelidonya_logs'))
        self.declare_parameter('fps', 10.0)          # sartname min 1 Hz
        self.declare_parameter('record_camera', True)
        self.declare_parameter('record_map', True)
        self.declare_parameter('map_pixels', 480)    # harita videosu kenar uzunlugu
        self.declare_parameter('queue_size', 30)

        self.out_dir = os.path.expanduser(str(self.get_parameter('out_dir').value))
        self.fps = max(1.0, float(self.get_parameter('fps').value))
        record_camera = bool(self.get_parameter('record_camera').value)
        record_map = bool(self.get_parameter('record_map').value)
        self.map_pixels = int(self.get_parameter('map_pixels').value)
        qsize = int(self.get_parameter('queue_size').value)

        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.bridge = CvBridge()
        cb = MutuallyExclusiveCallbackGroup()

        self.cam_worker = None
        self.map_worker = None

        if record_camera:
            self.cam_worker = VideoWriterWorker(
                os.path.join(self.out_dir, f'dosya1_kamera_{ts}.mp4'),
                self.fps, self.get_logger(), qsize)
            self.create_subscription(
                Image, Topics.YOLO_ANNOTATED, self.on_image,
                SENSOR_QOS, callback_group=cb)

        if record_map:
            self.map_worker = VideoWriterWorker(
                os.path.join(self.out_dir, f'dosya3_engel_haritasi_{ts}.mp4'),
                self.fps, self.get_logger(), qsize)
            self.create_subscription(
                OccupancyGrid, Topics.OBSTACLE_GRID, self.on_grid,
                COMMAND_QOS, callback_group=cb)

        self.get_logger().info(
            f'video_recorder basladi. kamera={record_camera} harita={record_map} '
            f'fps={self.fps:.0f} -> {self.out_dir}'
        )

    # ------------------------------------------------------------------
    def _now_text(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def on_image(self, msg: Image):
        """Dosya 1 — YOLO'nun bbox+sinif cizdigi kare, uzerine zaman etiketi."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # cv_bridge cesitli hata tipleri firlatir
            self.get_logger().error(f'cv_bridge hatasi: {e}')
            return
        frame = stamp_frame(frame.copy(), f'IDA {self._now_text()}')
        self.cam_worker.submit(frame)

    def on_grid(self, msg: OccupancyGrid):
        """Dosya 3 — OccupancyGrid'i gorsellestirip zaman etiketiyle yaz."""
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return

        data = np.array(msg.data, dtype=np.int8).reshape(h, w)

        # Renklendirme: bilinmiyor(-1)=gri, bos(0)=koyu, dolu(100)=kirmizi
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data < 0] = (60, 60, 60)
        img[data == 0] = (20, 20, 20)
        img[data > 50] = (0, 0, 220)

        # Harita alt-ortada arac konumu: goruntuyu dik cevir (x ileri = yukari)
        img = cv2.flip(img, 0)
        img = cv2.resize(img, (self.map_pixels, self.map_pixels),
                         interpolation=cv2.INTER_NEAREST)

        # Arac isareti (alt orta) + olcek bilgisi
        cx, cy = self.map_pixels // 2, self.map_pixels - 6
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
        res_text = f'{msg.info.resolution:.2f} m/hucre  {w}x{h}'
        cv2.putText(img, res_text, (6, self.map_pixels - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        img = stamp_frame(img, f'ENGEL HARITASI {self._now_text()}')
        self.map_worker.submit(img)

    # ------------------------------------------------------------------
    def destroy_node(self):
        for worker in (self.cam_worker, self.map_worker):
            if worker is not None:
                worker.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
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
