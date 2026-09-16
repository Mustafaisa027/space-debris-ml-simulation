from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraKeyframe:
    t: float
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0)


MISSION_FLY_TO = (
    CameraKeyframe(0.0, (14500.0, -17000.0, 9200.0), (0.0, 0.0, 0.0)),
    CameraKeyframe(0.35, (9000.0, -10500.0, 5200.0), (1100.0, 400.0, 250.0)),
    CameraKeyframe(0.7, (4300.0, -6200.0, 3100.0), (1800.0, 1200.0, 650.0)),
    CameraKeyframe(1.0, (2600.0, -3800.0, 2100.0), (2100.0, 1450.0, 720.0)),
)


def interpolate_camera(t: float, keyframes=MISSION_FLY_TO) -> CameraKeyframe:
    clamped = min(max(float(t), 0.0), 1.0)
    previous = keyframes[0]
    for current in keyframes[1:]:
        if clamped <= current.t:
            local = (clamped - previous.t) / max(current.t - previous.t, 0.0001)
            eased = 1.0 - (1.0 - local) ** 3
            return CameraKeyframe(
                t=clamped,
                position=_lerp3(previous.position, current.position, eased),
                focal_point=_lerp3(previous.focal_point, current.focal_point, eased),
                view_up=_lerp3(previous.view_up, current.view_up, eased),
            )
        previous = current
    return keyframes[-1]


def _lerp3(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))  # type: ignore[return-value]


def targeted_camera(
    t: float,
    start_position: tuple[float, float, float],
    target: tuple[float, float, float],
    start_focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraKeyframe:
    clamped = min(max(float(t), 0.0), 1.0)
    eased = 1.0 - (1.0 - clamped) ** 3
    magnitude = max(sum(value * value for value in target) ** 0.5, 1.0)
    radial = tuple(value / magnitude for value in target)
    end_position = tuple(target[index] + radial[index] * 1800.0 for index in range(3))
    return CameraKeyframe(
        t=clamped,
        position=_lerp3(start_position, end_position, eased),
        focal_point=_lerp3(start_focal_point, target, eased),
    )
