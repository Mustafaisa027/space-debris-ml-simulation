from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday
from sgp4.propagation import gstime


@dataclass(frozen=True)
class TleObject:
    name: str
    catalog_id: str
    satellite: Satrec


def load_tle_catalog(path: Path) -> tuple[TleObject, ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"TLE catalog must contain name/line1/line2 triples: {path}")
    objects: list[TleObject] = []
    for offset in range(0, len(lines), 3):
        name, line1, line2 = lines[offset : offset + 3]
        objects.append(
            TleObject(
                name=name,
                catalog_id=line1[2:7].strip(),
                satellite=Satrec.twoline2rv(line1, line2),
            )
        )
    return tuple(objects)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def propagate(item: TleObject, when: datetime) -> np.ndarray | None:
    state = propagate_state(item, when)
    return state[0] if state is not None else None


def propagate_state(item: TleObject, when: datetime) -> tuple[np.ndarray, np.ndarray] | None:
    """Return TEME position (km) and velocity (km/s) from SGP4."""
    utc = when.astimezone(timezone.utc)
    jd, fraction = jday(
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        utc.second + utc.microsecond / 1_000_000.0,
    )
    error, position, velocity = item.satellite.sgp4(jd, fraction)
    if error:
        return None
    return np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)


def greenwich_angle_deg(when: datetime) -> float:
    utc = when.astimezone(timezone.utc)
    jd, fraction = jday(
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        utc.second + utc.microsecond / 1_000_000.0,
    )
    return float(np.degrees(gstime(jd + fraction)))


def orbit_track(item: TleObject, center: datetime, minutes: float = 110.0) -> np.ndarray:
    samples: list[np.ndarray] = []
    for offset in np.linspace(-minutes / 2.0, minutes / 2.0, 120):
        position = propagate(item, center + timedelta(minutes=float(offset)))
        if position is not None:
            samples.append(position)
    if len(samples) < 2:
        return np.empty((0, 3), dtype=float)
    return np.vstack(samples)
