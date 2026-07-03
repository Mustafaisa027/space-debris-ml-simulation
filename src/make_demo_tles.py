"""Generate a deterministic *synthetic* LEO catalogue as a 3-line TLE file.

Why this exists
---------------
The bundled ``data/sample_tles.txt`` holds five real objects whose epochs are
now years old, and five objects almost never produce a genuine close approach
inside a physical screening radius (tens of km). That makes the machine-learning
stage in the paper impossible to demonstrate offline: there are simply no
"risky" events to classify.

This script builds a reproducible synthetic constellation whose members share
similar shells so that real conjunctions (small minimum distance, high relative
velocity) occur when propagated. The output is a normal 3-line TLE file that the
existing pipeline reads unchanged.

Important honesty note
----------------------
These are NOT observed objects. They are clearly named ``DEMO-SAT-xx`` so no one
mistakes them for a real catalogue. For scientifically meaningful results, fetch
current elements with ``src/fetch_tles.py`` (e.g. a Starlink/OneWeb group) and
run the pipeline against those. This generator only exists so the code path,
plots, and model comparison can be exercised without network access.

Usage
-----
    python src/make_demo_tles.py --count 40 --output data/demo_tles.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle

# SGP4 epoch is "days since 1949-12-31 00:00 UT". 27941.0 corresponds to
# 2026-07-01, kept fixed so the generated file is reproducible run to run and
# recent enough that the pipeline's TLE-age warning stays quiet for the demo.
_EPOCH_DAYS = 27941.0

MU = 398600.4418  # km^3/s^2, Earth gravitational parameter
EARTH_RADIUS_KM = 6378.137


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    """Circular-orbit mean motion (rev/day) for a given altitude."""
    a = EARTH_RADIUS_KM + altitude_km
    n_rad_s = (MU / a**3) ** 0.5
    return n_rad_s * 86400.0 / (2.0 * 3.141592653589793)


def build_satellite(index: int, altitude_km: float, inclination_deg: float,
                    raan_deg: float, mean_anomaly_deg: float) -> Satrec:
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        index + 1,          # satnum
        _EPOCH_DAYS,        # epoch
        0.0,                # bstar
        0.0,                # ndot
        0.0,                # nddot
        0.0001,             # ecco (near-circular)
        0.0,                # argpo
        inclination_deg * 3.141592653589793 / 180.0,
        mean_anomaly_deg * 3.141592653589793 / 180.0,
        _mean_motion_rev_per_day(altitude_km) * 2.0 * 3.141592653589793 / 1440.0,
        raan_deg * 3.141592653589793 / 180.0,
    )
    return sat


def generate(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    blocks: list[str] = []
    # Two families of orbital planes that CROSS each other:
    #   * a prograde 53 deg group (Starlink-like), and
    #   * a near-polar 97.6 deg sun-synchronous group.
    # Members of crossing planes periodically pass close at high relative
    # velocity, which is exactly the "risky" regime the classifier must learn.
    planes = [
        # (altitude_km, inclination_deg, raan_deg)
        # Prograde 53 deg shell (two crossing RAANs) ...
        (550.0, 53.0, 0.0),
        (550.0, 53.0, 60.0),
        # ... near-polar sun-synchronous shell (crosses the prograde one) ...
        (555.0, 97.6, 20.0),
        (548.0, 97.6, 80.0),
        # ... and a high-inclination shell that produces head-on geometries,
        # giving a spread of relative velocities so the distance/velocity
        # label boundary is genuinely two-dimensional.
        (552.0, 120.0, 40.0),
        (546.0, 120.0, 100.0),
    ]
    for i in range(count):
        altitude, inclination, raan = planes[i % len(planes)]
        # Small spread so members share a plane but occupy different phases.
        altitude += rng.uniform(-3.0, 3.0)
        inclination += rng.uniform(-0.15, 0.15)
        raan = (raan + rng.uniform(-2.0, 2.0)) % 360.0
        mean_anomaly = rng.uniform(0.0, 360.0)
        sat = build_satellite(i, altitude, inclination, raan, mean_anomaly)
        line1, line2 = export_tle(sat)
        name = f"DEMO-SAT-{i:02d}"
        blocks.append("\n".join([name, line1, line2]))
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic LEO demo TLEs")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--output", default="data/demo_tles.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = generate(args.count, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"OK -> {out} ({len(blocks)} synthetic DEMO objects)")


if __name__ == "__main__":
    main()
