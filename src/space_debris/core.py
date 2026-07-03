from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

from skyfield.api import EarthSatellite, load

EARTH_RADIUS_KM = 6378.137


@dataclass(frozen=True)
class TleObject:
    name: str
    line1: str
    line2: str


@dataclass(frozen=True)
class PairResult:
    snapshot_utc: str
    object_1: str
    object_2: str
    tle_epoch_1_utc: str
    tle_epoch_2_utc: str
    max_tle_age_hours: float
    tca_utc: str
    time_to_tca_min: float
    current_distance_km: float
    min_distance_km: float
    relative_velocity_km_s: float
    altitude_1_km: float
    altitude_2_km: float
    altitude_difference_km: float
    risk_score: float
    fixed_threshold_alarm: int
    risk_label: int


def read_tles(path: Path) -> list[TleObject]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"TLE file must contain name/line1/line2 groups: {path}")
    return [TleObject(lines[i], lines[i + 1], lines[i + 2]) for i in range(0, len(lines), 3)]


def build_satellites(objects: Iterable[TleObject]) -> list[EarthSatellite]:
    return [EarthSatellite(obj.line1, obj.line2, obj.name) for obj in objects]


def _distance_km(a, b) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _norm(vec) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _altitude_km(position_km) -> float:
    return _norm(position_km) - EARTH_RADIUS_KM


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_skyfield_time(ts, dt: datetime):
    dt = dt.astimezone(timezone.utc)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def _sat_epoch_iso(sat: EarthSatellite) -> str:
    return _iso_z(sat.epoch.utc_datetime())


def _tle_age_hours(sat: EarthSatellite, reference_utc: datetime) -> float:
    epoch = sat.epoch.utc_datetime().astimezone(timezone.utc)
    return abs((reference_utc - epoch).total_seconds()) / 3600.0


def _risk_label(
    min_distance_km: float,
    relative_velocity_km_s: float,
    label_distance_km: float,
    label_relative_velocity_km_s: float,
) -> int:
    """Physics-based ground-truth label for a conjunction event.

    A pair is labelled "risky" (1) when BOTH conditions hold:
      * the minimum approach distance is inside the physical screening
        radius (``label_distance_km``), and
      * the relative velocity at TCA is high enough that a hypothetical
        impact would be energetic (``label_relative_velocity_km_s``).

    This label deliberately couples geometry (distance) with kinematics
    (relative velocity). A classical fixed-distance threshold only sees
    distance, so it necessarily misclassifies:
      * slow, close passes  -> fixed threshold raises a false alarm,
      * fast, slightly-more-distant passes -> fixed threshold misses them.

    Because the label is NOT derived from ``risk_score`` (which is kept
    out of the model feature set), there is no target leakage: a learner
    must actually recover the distance/velocity decision boundary rather
    than echo a pre-computed score.
    """
    return int(
        min_distance_km <= label_distance_km
        and relative_velocity_km_s >= label_relative_velocity_km_s
    )


def simulate_pairs(
    satellites: list[EarthSatellite],
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
    leo_min_altitude_km: float,
    leo_max_altitude_km: float,
    fixed_threshold_km: float,
    label_threshold_km: float,
    label_relative_velocity_km_s: float = 10.0,
    max_tle_age_hours: float = 336.0,
) -> list[PairResult]:
    """Propagate every LEO pair and score its closest approach.

    ``max_tle_age_hours`` (default 14 days) only controls a data-quality
    warning: SGP4 accuracy degrades quickly, so stale TLEs make the TCA
    and minimum-distance figures physically unreliable. The simulation
    still runs, but a warning is emitted so results are not over-trusted.
    """
    ts = load.timescale()
    start_utc = start_utc.astimezone(timezone.utc).replace(microsecond=0)
    offsets = list(range(0, horizon_minutes + 1, step_minutes))
    results: list[PairResult] = []

    stale = [
        f"{sat.name} ({_tle_age_hours(sat, start_utc) / 24.0:.1f} d)"
        for sat in satellites
        if _tle_age_hours(sat, start_utc) > max_tle_age_hours
    ]
    if stale:
        warnings.warn(
            "TLE epoch older than "
            f"{max_tle_age_hours / 24.0:.0f} days for: {', '.join(stale)}. "
            "SGP4 propagation error grows with age; treat TCA and minimum "
            "distance as illustrative, not operational.",
            stacklevel=2,
        )

    for sat1, sat2 in combinations(satellites, 2):
        start_t = _to_skyfield_time(ts, start_utc)
        p1_start = sat1.at(start_t).position.km
        p2_start = sat2.at(start_t).position.km
        alt1 = _altitude_km(p1_start)
        alt2 = _altitude_km(p2_start)
        if not (leo_min_altitude_km <= alt1 <= leo_max_altitude_km):
            continue
        if not (leo_min_altitude_km <= alt2 <= leo_max_altitude_km):
            continue

        current_distance = _distance_km(p1_start, p2_start)
        best_offset = 0
        best_distance = current_distance

        for offset in offsets:
            dt = start_utc + timedelta(minutes=offset)
            t = _to_skyfield_time(ts, dt)
            distance = _distance_km(sat1.at(t).position.km, sat2.at(t).position.km)
            if distance < best_distance:
                best_distance = distance
                best_offset = offset

        tca_dt = start_utc + timedelta(minutes=best_offset)
        tca_t = _to_skyfield_time(ts, tca_dt)
        v1 = sat1.at(tca_t).velocity.km_per_s
        v2 = sat2.at(tca_t).velocity.km_per_s
        relative_velocity = _distance_km(v1, v2)
        altitude_difference = abs(alt1 - alt2)
        # risk_score is a human-readable ranking aid only. It is written to
        # the CSV for triage but is deliberately EXCLUDED from the ML feature
        # set (see space_debris.ml.FEATURES) to avoid target leakage.
        urgency = 1.0 + ((horizon_minutes - best_offset) / max(horizon_minutes, 1))
        risk_score = (relative_velocity / max(best_distance, 1.0)) * urgency
        risk_label = _risk_label(
            min_distance_km=best_distance,
            relative_velocity_km_s=relative_velocity,
            label_distance_km=label_threshold_km,
            label_relative_velocity_km_s=label_relative_velocity_km_s,
        )

        results.append(
            PairResult(
                snapshot_utc=_iso_z(start_utc),
                object_1=sat1.name,
                object_2=sat2.name,
                tle_epoch_1_utc=_sat_epoch_iso(sat1),
                tle_epoch_2_utc=_sat_epoch_iso(sat2),
                max_tle_age_hours=max(_tle_age_hours(sat1, start_utc), _tle_age_hours(sat2, start_utc)),
                tca_utc=_iso_z(tca_dt),
                time_to_tca_min=float(best_offset),
                current_distance_km=current_distance,
                min_distance_km=best_distance,
                relative_velocity_km_s=relative_velocity,
                altitude_1_km=alt1,
                altitude_2_km=alt2,
                altitude_difference_km=altitude_difference,
                risk_score=risk_score,
                fixed_threshold_alarm=int(best_distance <= fixed_threshold_km),
                risk_label=risk_label,
            )
        )

    return sorted(results, key=lambda row: row.risk_score, reverse=True)


def write_pair_results(path: Path, rows: list[PairResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PairResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "object_1": row.object_1,
                    "object_2": row.object_2,
                    "snapshot_utc": row.snapshot_utc,
                    "tle_epoch_1_utc": row.tle_epoch_1_utc,
                    "tle_epoch_2_utc": row.tle_epoch_2_utc,
                    "max_tle_age_hours": f"{row.max_tle_age_hours:.3f}",
                    "tca_utc": row.tca_utc,
                    "time_to_tca_min": f"{row.time_to_tca_min:.0f}",
                    "current_distance_km": f"{row.current_distance_km:.3f}",
                    "min_distance_km": f"{row.min_distance_km:.3f}",
                    "relative_velocity_km_s": f"{row.relative_velocity_km_s:.6f}",
                    "altitude_1_km": f"{row.altitude_1_km:.3f}",
                    "altitude_2_km": f"{row.altitude_2_km:.3f}",
                    "altitude_difference_km": f"{row.altitude_difference_km:.3f}",
                    "risk_score": f"{row.risk_score:.10f}",
                    "fixed_threshold_alarm": row.fixed_threshold_alarm,
                    "risk_label": row.risk_label,
                }
            )


def filter_conjunctions(rows: list[PairResult], candidate_threshold_km: float) -> list[PairResult]:
    return [row for row in rows if row.min_distance_km <= candidate_threshold_km]


def write_distance_timeseries(
    csv_path: Path,
    sat1: EarthSatellite,
    sat2: EarthSatellite,
    start_utc: datetime,
    horizon_minutes: int,
    step_minutes: int,
) -> None:
    ts = load.timescale()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["minute", "utc_iso", "distance_km"])
        writer.writeheader()
        for offset in range(0, horizon_minutes + 1, step_minutes):
            dt = start_utc + timedelta(minutes=offset)
            t = _to_skyfield_time(ts, dt)
            distance = _distance_km(sat1.at(t).position.km, sat2.at(t).position.km)
            writer.writerow(
                {
                    "minute": offset,
                    "utc_iso": _iso_z(dt),
                    "distance_km": f"{distance:.3f}",
                }
            )
