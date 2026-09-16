#!/usr/bin/env python3
"""
blackbox_node — Gorev sonrasi teshis icin KARA KUTU
====================================================

Gorev sirasinda Jetson'a bakamayacagiz. Terminalde goreceklerimizin TAMAMINI
diske yazar ki gorev sonunda ne olduugunu anlayabilelim.

Mevcut loglayicilardan farki:
  mission_logger      -> topic verisi (JSONL)
  telemetry_logger    -> sartname Dosya 2 (CSV)
  video_recorder      -> sartname Dosya 1/3 (mp4)
  blackbox (bu dugum) -> HER SEYIN sozel kaydi + saglik + ozet

Kaydettikleri:
  1. /rosout          — TUM dugumlerin butun log mesajlari (INFO/WARN/ERROR)
                        Terminalde akan cikti aynen buraya duser.
  2. /mavros/statustext/recv — ucus kontrolcunun kendi mesajlari
                        (failsafe, EKF hatasi, arm reddi, pre-arm uyarilari)
  3. Dugum canliligi  — beklenen dugumlerden biri olurse ANINDA kaydedilir.
                        Gorev ortasinda zed_yolo cokerse boyle ogreniriz.
  4. Jetson sagligi   — sicaklik ve throttle. Orin Nano yuk altinda kisilir;
                        FPS dususu bundansa loglardan anlasilir.
  5. Gorev durumu     — parkur gecisleri zaman damgali

Ciktilar (<out_dir> altina):
  blackbox_<ts>.log   — her sey, zaman sirali, insan-okur
  ozet_<ts>.txt       — gorev sonu ozeti (kapanista yazilir)

ozet_ dosyasi once okunur: kac hata oldu, hangi dugum coktu, parkurlar ne
zaman gecildi, sicaklik ne oldu. Detay gerekirse blackbox_ dosyasina bakilir.
"""

import os
from collections import Counter
from datetime import datetime
from glob import glob

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy,
)

from rcl_interfaces.msg import Log

from mission_interfaces.msg import MissionState

from .common import Topics, SENSOR_QOS, STATE_QOS

try:
    from mavros_msgs.msg import StatusText
    MAVROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    StatusText = None
    MAVROS_AVAILABLE = False


# ROS log seviyesi -> okunakli etiket
LEVELS = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}

STATE_NAMES = {
    0: 'BOOT', 1: 'READY', 2: 'PARKUR1', 3: 'PARKUR2',
    4: 'PARKUR3', 5: 'DONE', 6: 'EMERGENCY',
}

# /rosout RELIABLE + TRANSIENT_LOCAL yayinlanir. VOLATILE abone olmak
# uyumludur (abone daha azini ister) ve gecmis mesaj selini onler.
ROSOUT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1000,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class BlackboxNode(Node):

    def __init__(self):
        super().__init__('blackbox_node')

        self.declare_parameter('out_dir', os.path.expanduser('~/gelidonya_logs'))
        self.declare_parameter('flush_period_s', 2.0)
        self.declare_parameter('health_period_s', 5.0)
        self.declare_parameter('node_check_period_s', 3.0)
        self.declare_parameter('min_level', 20)      # 20=INFO, 30=WARN
        # Gorev icin kritik dugumler. Biri kaybolursa kaydedilir.
        self.declare_parameter('watch_nodes', [
            'mission_manager_node',
            'kamikaze_node',
            'obstacle_bridge_node',
            'zed_yolo_node',
            'telemetry_logger_node',
            'video_recorder_node',
        ])

        self.out_dir = os.path.expanduser(str(self.get_parameter('out_dir').value))
        flush_period = float(self.get_parameter('flush_period_s').value)
        health_period = float(self.get_parameter('health_period_s').value)
        node_period = float(self.get_parameter('node_check_period_s').value)
        self.min_level = int(self.get_parameter('min_level').value)
        self.watch_nodes = set(self.get_parameter('watch_nodes').value or [])

        os.makedirs(self.out_dir, exist_ok=True)
        self.ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.out_dir, f'blackbox_{self.ts}.log')
        self.summary_path = os.path.join(self.out_dir, f'ozet_{self.ts}.txt')
        self._fh = open(self.log_path, 'w', encoding='utf-8')

        # --- Ozet icin biriktirilenler ---
        self.start_time = datetime.now()
        self.level_counts = Counter()
        self.errors = []            # (zaman, kaynak, mesaj) — ilk N tanesi
        self.node_deaths = []       # (zaman, dugum)
        self.state_timeline = []    # (zaman, durum adi, info)
        self.fc_messages = []       # (zaman, severity, metin)
        self.max_temp = 0.0
        self.throttle_seen = False
        self._seen_nodes = set()
        self._last_state = None

        cb = MutuallyExclusiveCallbackGroup()

        self.create_subscription(Log, '/rosout', self.on_rosout, ROSOUT_QOS,
                                 callback_group=cb)
        self.create_subscription(MissionState, Topics.MISSION_STATE,
                                 self.on_state, STATE_QOS, callback_group=cb)
        if MAVROS_AVAILABLE:
            # MAVROS STATUSTEXT'i BEST_EFFORT yayinlar (qos-mismatch tuzagi).
            self.create_subscription(StatusText, Topics.MAVROS_STATUSTEXT,
                                     self.on_statustext, SENSOR_QOS,
                                     callback_group=cb)

        self.create_timer(flush_period, self._flush, callback_group=cb)
        self.create_timer(health_period, self.check_health, callback_group=cb)
        self.create_timer(node_period, self.check_nodes, callback_group=cb)

        self._write('BLACKBOX', f'kayit basladi -> {self.log_path}')
        self.get_logger().info(f'blackbox basladi -> {self.log_path}')

    # ------------------------------------------------------------------
    @staticmethod
    def _now():
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]

    def _write(self, tag, text):
        try:
            self._fh.write(f'{self._now()}  [{tag}] {text}\n')
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------
    def on_rosout(self, msg: Log):
        """Tum dugumlerin log mesajlari — terminalde gordugumuz her sey."""
        if msg.level < self.min_level:
            return
        # Kendi mesajlarimizi tekrar yazma (sonsuz dongu olmaz ama gurultu yapar)
        if msg.name == self.get_name():
            return

        level = LEVELS.get(msg.level, str(msg.level))
        self.level_counts[level] += 1
        self._write(f'{level:<5} {msg.name}', msg.msg)

        if msg.level >= 40 and len(self.errors) < 200:
            self.errors.append((self._now(), msg.name, msg.msg))

    def on_statustext(self, msg):
        """Ucus kontrolcunun kendi mesajlari — failsafe, EKF, arm reddi."""
        text = msg.text
        self._write('FCU', text)
        if len(self.fc_messages) < 200:
            self.fc_messages.append((self._now(), int(msg.severity), text))

    def on_state(self, msg: MissionState):
        if msg.state == self._last_state:
            return
        self._last_state = msg.state
        name = STATE_NAMES.get(msg.state, str(msg.state))
        entry = (self._now(), name, msg.info)
        self.state_timeline.append(entry)
        self._write('GOREV', f'{name}  ({msg.info})')

    # ------------------------------------------------------------------
    def check_nodes(self):
        """Beklenen dugumlerden biri kayboldu mu? Cokme boyle yakalanir."""
        try:
            alive = set(self.get_node_names())
        except Exception:   # rclpy cesitli hata tipleri firlatabilir
            return

        for name in self.watch_nodes:
            if name in alive:
                self._seen_nodes.add(name)
            elif name in self._seen_nodes:
                # Bir kere gorulmus ama artik yok -> oldu
                self._seen_nodes.discard(name)
                self.node_deaths.append((self._now(), name))
                self._write('COKME', f'{name} ARTIK YOK (surec oldu)')
                self.get_logger().error(f'DUGUM OLDU: {name}')

    def check_health(self):
        """Jetson sicakligi ve throttle. Orin Nano yuk altinda kisilir."""
        temps = []
        for path in glob('/sys/devices/virtual/thermal/thermal_zone*/temp'):
            try:
                with open(path) as f:
                    temps.append(int(f.read().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
        if not temps:
            return

        t = max(temps)
        self.max_temp = max(self.max_temp, t)
        note = ''
        if t >= 85.0:
            self.throttle_seen = True
            note = '  <-- KRITIK, throttle bekleniyor'
        elif t >= 75.0:
            note = '  <-- yuksek'
        self._write('SAGLIK', f'max sicaklik {t:.1f} C{note}')

    # ------------------------------------------------------------------
    def _flush(self):
        try:
            self._fh.flush()
        except (OSError, ValueError):
            pass

    def write_summary(self):
        """Gorev sonu ozeti — once bu dosya okunur."""
        dur = (datetime.now() - self.start_time).total_seconds()
        lines = [
            '=' * 62,
            ' GELIDONYA IDA — GOREV OZETI',
            '=' * 62,
            f'Baslangic : {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'Sure      : {dur:.0f} saniye ({dur / 60:.1f} dk)',
            f'Detay log : {os.path.basename(self.log_path)}',
            '',
            '--- GOREV AKISI ---',
        ]
        if self.state_timeline:
            for t, name, info in self.state_timeline:
                lines.append(f'  {t}  {name:<10} {info}')
        else:
            lines.append('  (durum degisimi kaydedilmedi — gorev hic baslamadi mi?)')

        lines += ['', '--- LOG SAYIMI ---']
        for lvl in ('FATAL', 'ERROR', 'WARN', 'INFO'):
            if self.level_counts.get(lvl):
                lines.append(f'  {lvl:<6}: {self.level_counts[lvl]}')

        lines += ['', '--- DUGUM COKMELERI ---']
        if self.node_deaths:
            for t, name in self.node_deaths:
                lines.append(f'  {t}  {name}')
        else:
            lines.append('  yok (tum dugumler ayakta kaldi)')

        lines += ['', '--- JETSON SAGLIGI ---',
                  f'  En yuksek sicaklik: {self.max_temp:.1f} C']
        if self.throttle_seen:
            lines.append('  UYARI: 85 C asildi, performans kisilmis olabilir')

        lines += ['', '--- UCUS KONTROLCU MESAJLARI (son 20) ---']
        if self.fc_messages:
            for t, sev, text in self.fc_messages[-20:]:
                lines.append(f'  {t}  [sev {sev}] {text}')
        else:
            lines.append('  yok')

        lines += ['', '--- HATALAR (ilk 30) ---']
        if self.errors:
            for t, src, text in self.errors[:30]:
                lines.append(f'  {t}  {src}: {text}')
        else:
            lines.append('  yok')

        lines.append('=' * 62)

        try:
            with open(self.summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except OSError:
            pass
        return '\n'.join(lines)

    def destroy_node(self):
        self._write('BLACKBOX', 'kayit kapaniyor')
        summary = self.write_summary()
        print('\n' + summary, flush=True)   # terminale de bas
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except (OSError, ValueError):
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BlackboxNode()
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
