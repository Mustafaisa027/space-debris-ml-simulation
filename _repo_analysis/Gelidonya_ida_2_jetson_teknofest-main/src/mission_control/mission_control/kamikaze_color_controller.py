from dataclasses import dataclass
from typing import Optional, Sequence


# İHA'dan gelen renk kodları
COLOR_CODE_TO_CLASS = {
    1: "red",
    2: "green",
    3: "black",
}


@dataclass
class Detection:
    """
    YOLO veya başka algılama sisteminden gelecek duba bilgisi.

    class_name : red, green veya black
    confidence : 0.0 - 1.0
    x1, y1     : kutunun sol üst köşesi
    x2, y2     : kutunun sağ alt köşesi
    """
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class KamikazeOutput:
    target_found: bool
    target_color: str
    target_center_x: Optional[float]
    target_center_y: Optional[float]

    # -1.0 sola, 0 merkez, +1.0 sağa
    horizontal_error: float

    # ROS tarafının kullanacağı soyut komutlar
    forward_command: float
    turn_command: float

    state: str

    # Hedef kutusunun görüntüye oranı (0.0 - 1.0).
    # ROS sarmalayıcısı temas/angajman tespitinde kullanır: kutu görüntüyü
    # doldurduysa araç hedefe değmiş kabul edilir. Kontrol mantığını
    # etkilemez, sadece dışarıya bilgi verir.
    target_area_ratio: float = 0.0


class KamikazeColorController:
    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        minimum_confidence: float = 0.55,
        center_tolerance_ratio: float = 0.06,
        turn_gain: float = 1.30,
        maximum_turn: float = 0.75,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.minimum_confidence = minimum_confidence

        # Görüntü genişliğinin yüzde 6'sı kadar merkez toleransı
        self.center_tolerance_ratio = center_tolerance_ratio

        self.turn_gain = turn_gain
        self.maximum_turn = maximum_turn

        self.target_color_code: Optional[int] = None
        self.target_class: Optional[str] = None

    def set_color_code(self, color_code: int) -> bool:
        """
        YKİ'den İDA'ya gelen renk kodu burada ayarlanır.
        """

        target_class = COLOR_CODE_TO_CLASS.get(color_code)

        if target_class is None:
            self.target_color_code = None
            self.target_class = None
            return False

        self.target_color_code = color_code
        self.target_class = target_class
        return True

    def update(
        self,
        detections: Sequence[Detection],
    ) -> KamikazeOutput:
        """
        Bir kamera karesi için çalıştırılır.

        Çıktılar:
        forward_command:
            0.0 = dur
            1.0 = tam ileri

        turn_command:
            negatif = sola dön
            pozitif = sağa dön
        """

        if self.target_class is None:
            return self._empty_output(
                state="RENK_KODU_YOK"
            )

        target = self._select_target(detections)

        if target is None:
            return self._empty_output(
                state="HEDEF_DUBA_BULUNAMADI"
            )

        image_center_x = self.image_width / 2.0

        # Hata görüntü genişliğine göre -1 ile +1 arasına getirilir.
        horizontal_error = (
            target.center_x - image_center_x
        ) / image_center_x

        horizontal_error = self._clamp(
            horizontal_error,
            -1.0,
            1.0,
        )

        center_tolerance = self.center_tolerance_ratio

        if abs(horizontal_error) <= center_tolerance:
            # Duba yeterince merkezdeyse düz git.
            turn_command = 0.0
            state = "HEDEF_MERKEZDE"
        else:
            turn_command = self._clamp(
                horizontal_error * self.turn_gain,
                -self.maximum_turn,
                self.maximum_turn,
            )
            state = "HEDEF_ORTALANIYOR"

        forward_command = self._calculate_forward_speed(
            target=target,
            horizontal_error=horizontal_error,
        )

        return KamikazeOutput(
            target_found=True,
            target_color=self.target_class,
            target_center_x=target.center_x,
            target_center_y=target.center_y,
            horizontal_error=horizontal_error,
            forward_command=forward_command,
            turn_command=turn_command,
            state=state,
            target_area_ratio=target.area / (
                self.image_width * self.image_height
            ),
        )

    def _select_target(
        self,
        detections: Sequence[Detection],
    ) -> Optional[Detection]:
        """
        Gelen renk koduyla aynı sınıftaki dubaları seçer.

        Birden fazla aynı renk duba görülürse:
        - Güveni yüksek,
        - Büyük görünen,
        - Görüntü merkezine yakın olan

        duba tercih edilir.
        """

        valid_targets = [
            detection
            for detection in detections
            if detection.class_name.lower() == self.target_class
            and detection.confidence >= self.minimum_confidence
            and detection.width > 5
            and detection.height > 5
        ]

        if not valid_targets:
            return None

        image_center_x = self.image_width / 2.0

        def target_score(detection: Detection) -> float:
            normalized_area = detection.area / (
                self.image_width * self.image_height
            )

            center_distance = abs(
                detection.center_x - image_center_x
            ) / image_center_x

            return (
                detection.confidence * 2.0
                + normalized_area * 10.0
                - center_distance * 0.35
            )

        return max(valid_targets, key=target_score)

    def _calculate_forward_speed(
        self,
        target: Detection,
        horizontal_error: float,
    ) -> float:
        """
        Duba merkezden uzaksa hız düşürülür.

        Duba büyüdükçe araç dubaya yaklaşmış kabul edilir.
        Ancak kamikaze görevi olduğu için hedefe yaklaşınca tamamen durmaz.
        """

        area_ratio = target.area / (
            self.image_width * self.image_height
        )

        absolute_error = abs(horizontal_error)

        if absolute_error > 0.45:
            # Duba çok yandaysa önce dön.
            return 0.18

        if absolute_error > 0.22:
            return 0.30

        if absolute_error > self.center_tolerance_ratio:
            return 0.45

        # Hedef merkezde
        if area_ratio < 0.03:
            return 0.75

        if area_ratio < 0.08:
            return 0.90

        # Hedef çok yakın ve merkezde: kamikaze hücumu
        return 1.00

    def _empty_output(
        self,
        state: str,
    ) -> KamikazeOutput:
        return KamikazeOutput(
            target_found=False,
            target_color=self.target_class or "unknown",
            target_center_x=None,
            target_center_y=None,
            horizontal_error=0.0,
            forward_command=0.0,
            turn_command=0.0,
            state=state,
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(minimum, min(value, maximum))
