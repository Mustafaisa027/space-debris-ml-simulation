from __future__ import annotations

import csv
import io
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from skyfield.api import EarthSatellite, load

from space_debris.provenance import write_csv_text_with_provenance

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
    # --- Derived geometry features (ROADMAP_YOL1.md GOREV 5) ---
    # RIC decomposition of the relative position AT TCA, in satellite 1's
    # local orbital frame: describes HOW the close approach is oriented
    # (radial/in-track/cross-track), not just how close it is. The three
    # combine in quadrature to min_distance_km, but individually each one is
    # a distinct directional signal, not a duplicate/monotone copy of it.
    relative_radial_km: float
    relative_intrack_km: float
    relative_crosstrack_km: float
    # Angle between the two orbits' angular-momentum directions at TCA: 0 =
    # co-planar/co-rotating, 90 = perpendicular planes, 180 = co-planar
    # counter-rotating. Pure orbit-geometry information, independent of how
    # close/fast this particular encounter is.
    relative_inclination_deg: float
    # Radial (closing/opening rate) vs tangential split of the CURRENT
    # relative velocity (evaluated at snapshot time, not at TCA -- at TCA the
    # closing rate is ~0 by definition of a distance minimum, so it carries
    # no signal there). Negative radial velocity means the pair is currently
    # approaching each other.
    radial_velocity_km_s: float
    tangential_velocity_km_s: float
    # Angle between the current relative-position and relative-velocity
    # vectors: ~180 deg = head-on approach along the line of sight, ~90 deg =
    # tangential/grazing pass, ~0 deg = directly receding.
    approach_angle_deg: float
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


def _unit(vec) -> np.ndarray:
    array = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(array)
    if norm < 1e-9:
        return np.zeros(3)
    return array / norm


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


def _encounter_geometry_features(
    p1_start, p2_start, v1_start, v2_start,
    p1_tca, p2_tca, v1_tca, v2_tca,
) -> dict[str, float]:
    """Derived geometry features for one conjunction pair (ROADMAP_YOL1.md
    GOREV 5): relative-position/velocity components, radial/tangential
    velocity split, approach angle, and encounter (orbital-plane) geometry.

    A pure function of position/velocity vectors -- computed here so it can
    be unit tested directly with hand-built vectors, independent of SGP4
    propagation. None of the outputs is a duplicate or monotone transform of
    min_distance_km / relative_velocity_km_s: the RIC components describe
    ORIENTATION at TCA (not magnitude), radial/tangential velocity and
    approach_angle are evaluated at snapshot time rather than at TCA, and
    relative_inclination_deg is pure orbital-plane geometry.
    """
    # RIC decomposition of the relative position AT TCA, in satellite 1's
    # local orbital frame (same convention as
    # space_debris.plots.plot_relative_ric_components).
    rel_r_tca = np.asarray(p2_tca) - np.asarray(p1_tca)
    r_hat = _unit(p1_tca)
    h1_hat = _unit(np.cross(np.asarray(p1_tca), np.asarray(v1_tca)))
    i_hat = _unit(np.cross(h1_hat, r_hat))

    # Angle between the two orbital planes' angular-momentum directions: 0 =
    # co-planar/co-rotating, 90 = perpendicular planes, 180 = co-planar
    # counter-rotating.
    h2_hat = _unit(np.cross(np.asarray(p2_tca), np.asarray(v2_tca)))
    relative_inclination_deg = float(np.degrees(np.arccos(np.clip(np.dot(h1_hat, h2_hat), -1.0, 1.0))))

    # Radial (closing/opening rate) vs tangential split of the CURRENT
    # relative velocity, evaluated at snapshot time -- at TCA the closing
    # rate is ~0 by definition of a distance minimum, so it carries no
    # signal there. Negative radial velocity means the pair is approaching.
    rel_v_start = np.asarray(v2_start) - np.asarray(v1_start)
    rel_r_start = np.asarray(p2_start) - np.asarray(p1_start)
    separation_hat_start = _unit(rel_r_start)
    radial_velocity_km_s = float(np.dot(rel_v_start, separation_hat_start))
    current_relative_speed = float(np.linalg.norm(rel_v_start))
    tangential_velocity_km_s = float(
        math.sqrt(max(current_relative_speed**2 - radial_velocity_km_s**2, 0.0))
    )
    if current_relative_speed > 1e-9:
        approach_angle_deg = float(
            np.degrees(np.arccos(np.clip(radial_velocity_km_s / current_relative_speed, -1.0, 1.0)))
        )
    else:
        approach_angle_deg = 90.0

    return {
        "relative_radial_km": float(np.dot(rel_r_tca, r_hat)),
        "relative_intrack_km": float(np.dot(rel_r_tca, i_hat)),
        "relative_crosstrack_km": float(np.dot(rel_r_tca, h1_hat)),
        "relative_inclination_deg": relative_inclination_deg,
        "radial_velocity_km_s": radial_velocity_km_s,
        "tangential_velocity_km_s": tangential_velocity_km_s,
        "approach_angle_deg": approach_angle_deg,
    }


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
        geocentric_1_start = sat1.at(start_t)
        geocentric_2_start = sat2.at(start_t)
        p1_start = geocentric_1_start.position.km
        p2_start = geocentric_2_start.position.km
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
        geocentric_1_tca = sat1.at(tca_t)
        geocentric_2_tca = sat2.at(tca_t)
        p1_tca = geocentric_1_tca.position.km
        p2_tca = geocentric_2_tca.position.km
        v1 = geocentric_1_tca.velocity.km_per_s
        v2 = geocentric_2_tca.velocity.km_per_s
        relative_velocity = _distance_km(v1, v2)
        altitude_difference = abs(alt1 - alt2)

        v1_start = geocentric_1_start.velocity.km_per_s
        v2_start = geocentric_2_start.velocity.km_per_s
        geometry = _encounter_geometry_features(
            p1_start, p2_start, v1_start, v2_start,
            p1_tca, p2_tca, v1, v2,
        )

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
                relative_radial_km=geometry["relative_radial_km"],
                relative_intrack_km=geometry["relative_intrack_km"],
                relative_crosstrack_km=geometry["relative_crosstrack_km"],
                relative_inclination_deg=geometry["relative_inclination_deg"],
                radial_velocity_km_s=geometry["radial_velocity_km_s"],
                tangential_velocity_km_s=geometry["tangential_velocity_km_s"],
                approach_angle_deg=geometry["approach_angle_deg"],
                risk_score=risk_score,
                fixed_threshold_alarm=int(best_distance <= fixed_threshold_km),
                risk_label=risk_label,
            )
        )

    return sorted(results, key=lambda row: row.risk_score, reverse=True)


def write_pair_results(
    path: Path,
    rows: list[PairResult],
    *,
    source: str = "",
    config_summary: str = "",
) -> None:
    fieldnames = list(PairResult.__dataclass_fields__.keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
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
                "relative_radial_km": f"{row.relative_radial_km:.3f}",
                "relative_intrack_km": f"{row.relative_intrack_km:.3f}",
                "relative_crosstrack_km": f"{row.relative_crosstrack_km:.3f}",
                "relative_inclination_deg": f"{row.relative_inclination_deg:.3f}",
                "radial_velocity_km_s": f"{row.radial_velocity_km_s:.6f}",
                "tangential_velocity_km_s": f"{row.tangential_velocity_km_s:.6f}",
                "approach_angle_deg": f"{row.approach_angle_deg:.3f}",
                "risk_score": f"{row.risk_score:.10f}",
                "fixed_threshold_alarm": row.fixed_threshold_alarm,
                "risk_label": row.risk_label,
            }
        )

    write_csv_text_with_provenance(path, buffer.getvalue(), source=source, config_summary=config_summary)


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
