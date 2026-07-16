from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


EARTH_MU_KM3_S2 = 398600.4418
EARTH_EQUATORIAL_RADIUS_KM = 6378.137
# Escape speed at the Earth's equatorial surface bounds the speed of a bound
# Earth-orbiting object outside that radius. Two objects can approach in
# opposite directions, hence the factor of two for relative speed.
MAX_BOUND_ORBIT_SPEED_KM_S = math.sqrt(
    2.0 * EARTH_MU_KM3_S2 / EARTH_EQUATORIAL_RADIUS_KM
)
MAX_BOUND_RELATIVE_SPEED_KM_S = 2.0 * MAX_BOUND_ORBIT_SPEED_KM_S
SCREENING_NUMERICAL_GUARD_KM = 1.0


@dataclass(frozen=True)
class ScreeningDecision:
    is_candidate: bool
    min_coarse_distance_km: float
    max_interval_margin_km: float


def conservative_candidate_screen(
    offsets_s: Sequence[float],
    distances_km: Sequence[float],
    threshold_km: float,
    *,
    relative_speed_bound_km_s: float = MAX_BOUND_RELATIVE_SPEED_KM_S,
    numerical_guard_km: float = SCREENING_NUMERICAL_GUARD_KM,
) -> ScreeningDecision:
    """Conservatively decide whether an exact TCA search can be skipped.

    Distance is Lipschitz-continuous with constant no greater than relative
    speed. If a true encounter inside an interval reaches ``threshold_km``,
    its nearest endpoint (at most half an interval away) must be within
    ``threshold + speed_bound * interval_duration / 2``. Therefore a pair is
    rejected only when both endpoints of every interval exceed that bound.

    The recall guarantee depends on ``relative_speed_bound_km_s`` actually
    bounding the trajectory. The default uses twice Earth-surface escape speed
    and is conservative for bound Earth orbits represented by this LEO study.
    """
    if not math.isfinite(threshold_km) or threshold_km < 0:
        raise ValueError("threshold_km must be finite and non-negative")
    if not math.isfinite(relative_speed_bound_km_s) or relative_speed_bound_km_s <= 0:
        raise ValueError("relative_speed_bound_km_s must be finite and positive")
    if not math.isfinite(numerical_guard_km) or numerical_guard_km < 0:
        raise ValueError("numerical_guard_km must be finite and non-negative")
    if len(offsets_s) != len(distances_km) or not offsets_s:
        raise ValueError("offsets_s and distances_km must be non-empty and equal length")
    offsets = [float(value) for value in offsets_s]
    distances = [float(value) for value in distances_km]
    if any(not math.isfinite(value) or value < 0 for value in distances):
        raise ValueError("distances_km must be finite and non-negative")
    if any(not math.isfinite(value) for value in offsets):
        raise ValueError("offsets_s must be finite")
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("offsets_s must be strictly increasing")

    min_coarse = min(distances)
    if len(offsets) == 1:
        return ScreeningDecision(
            min_coarse <= threshold_km + numerical_guard_km, min_coarse, 0.0
        )

    max_margin = 0.0
    for index, (left_s, right_s) in enumerate(zip(offsets, offsets[1:])):
        margin = relative_speed_bound_km_s * (right_s - left_s) / 2.0
        max_margin = max(max_margin, margin)
        if (
            min(distances[index], distances[index + 1])
            <= threshold_km + margin + numerical_guard_km
        ):
            return ScreeningDecision(True, min_coarse, max_margin)
    return ScreeningDecision(False, min_coarse, max_margin)
