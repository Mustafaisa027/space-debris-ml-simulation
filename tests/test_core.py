"""Unit tests for the space-debris conjunction pipeline.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from space_debris.core import (
    _altitude_km,
    _distance_km,
    _risk_label,
    build_satellites,
    read_tles,
    simulate_pairs,
)
from space_debris.ml import FEATURES


DEMO_TLE = Path(__file__).resolve().parents[1] / "data" / "demo_tles.txt"


def test_distance_and_altitude_math():
    assert _distance_km((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0
    # A point one Earth radius + 100 km out has altitude ~100 km.
    alt = _altitude_km((6478.137, 0.0, 0.0))
    assert abs(alt - 100.0) < 1e-6


def test_risk_label_needs_both_distance_and_velocity():
    # Close but slow -> not risky (fixed-distance methods would false-alarm).
    assert _risk_label(min_distance_km=5.0, relative_velocity_km_s=1.0,
                       label_distance_km=20.0, label_relative_velocity_km_s=10.0) == 0
    # Slightly farther but fast -> risky (fixed-distance methods would miss it).
    assert _risk_label(min_distance_km=18.0, relative_velocity_km_s=12.0,
                       label_distance_km=20.0, label_relative_velocity_km_s=10.0) == 1
    # Close and fast -> risky.
    assert _risk_label(min_distance_km=5.0, relative_velocity_km_s=12.0,
                       label_distance_km=20.0, label_relative_velocity_km_s=10.0) == 1


def test_no_target_leakage_in_feature_set():
    # risk_score is a monotone transform of the label-defining quantities and
    # must never be handed to the models as a predictor.
    assert "risk_score" not in FEATURES
    assert "risk_label" not in FEATURES
    assert "fixed_threshold_alarm" not in FEATURES


def test_read_tles_roundtrip():
    objs = read_tles(DEMO_TLE)
    assert len(objs) >= 10
    assert all(o.line1.startswith("1 ") and o.line2.startswith("2 ") for o in objs)


def test_simulate_pairs_smoke():
    objs = read_tles(DEMO_TLE)[:8]
    sats = build_satellites(objs)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = simulate_pairs(
        sats, start, horizon_minutes=30, step_minutes=5,
        leo_min_altitude_km=160, leo_max_altitude_km=2000,
        fixed_threshold_km=50, label_threshold_km=200,
        label_relative_velocity_km_s=5,
    )
    assert rows, "expected at least one LEO pair"
    r = rows[0]
    # Minimum distance never exceeds the initial distance.
    assert r.min_distance_km <= r.current_distance_km + 1e-6
    assert r.relative_velocity_km_s >= 0.0
    assert r.risk_label in (0, 1)
