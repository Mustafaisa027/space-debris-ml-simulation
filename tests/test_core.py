"""Unit tests for the space-debris conjunction pipeline.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from space_debris.core import (
    _altitude_km,
    _distance_km,
    _encounter_geometry_features,
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


def test_feature_set_includes_encounter_geometry_features():
    # GOREV 5: new geometry features must be exposed to the models ...
    for feature in (
        "relative_radial_km",
        "relative_intrack_km",
        "relative_crosstrack_km",
        "relative_inclination_deg",
        "radial_velocity_km_s",
        "tangential_velocity_km_s",
        "approach_angle_deg",
    ):
        assert feature in FEATURES
    # ... without reintroducing leakage of the label-defining raw scalars.
    assert "risk_score" not in FEATURES
    assert "risk_label" not in FEATURES


def test_encounter_geometry_ric_decomposition_is_exact():
    # Sat 1 at TCA on the +x axis moving along +y -> local frame is exactly
    # radial=+x, in-track=+y, cross-track=+z. A relative position of
    # (3, 4, 5) in that frame must decompose back to exactly (3, 4, 5).
    p1_tca = (7000.0, 0.0, 0.0)
    v1_tca = (0.0, 7.5, 0.0)
    p2_tca = (7003.0, 4.0, 5.0)
    v2_tca = (0.0, 7.5, 0.0)  # arbitrary, only RIC components are checked here

    geometry = _encounter_geometry_features(
        p1_start=p1_tca, p2_start=p2_tca, v1_start=v1_tca, v2_start=v2_tca,
        p1_tca=p1_tca, p2_tca=p2_tca, v1_tca=v1_tca, v2_tca=v2_tca,
    )

    assert geometry["relative_radial_km"] == pytest.approx(3.0)
    assert geometry["relative_intrack_km"] == pytest.approx(4.0)
    assert geometry["relative_crosstrack_km"] == pytest.approx(5.0)


def test_encounter_geometry_relative_inclination_perpendicular_planes():
    # Orbit 1 in the xy-plane (angular momentum along +z), orbit 2 in the
    # yz-plane (angular momentum along +x) -> perpendicular planes = 90 deg.
    p1 = (7000.0, 0.0, 0.0)
    v1 = (0.0, 7.5, 0.0)
    p2 = (0.0, 7000.0, 0.0)
    v2 = (0.0, 0.0, 7.5)

    geometry = _encounter_geometry_features(
        p1_start=p1, p2_start=p2, v1_start=v1, v2_start=v2,
        p1_tca=p1, p2_tca=p2, v1_tca=v1, v2_tca=v2,
    )

    assert geometry["relative_inclination_deg"] == pytest.approx(90.0)


def test_encounter_geometry_radial_velocity_sign_convention():
    # Separation vector points in +x. Relative velocity purely along -x
    # (closing) must give a NEGATIVE radial velocity and a ~180 deg
    # (head-on) approach angle with zero tangential component.
    approaching = _encounter_geometry_features(
        p1_start=(0.0, 0.0, 0.0), p2_start=(100.0, 0.0, 0.0),
        v1_start=(0.0, 0.0, 0.0), v2_start=(-1.0, 0.0, 0.0),
        p1_tca=(0.0, 0.0, 0.0), p2_tca=(100.0, 0.0, 0.0),
        v1_tca=(0.0, 7.5, 0.0), v2_tca=(0.0, 7.5, 0.0),
    )
    assert approaching["radial_velocity_km_s"] == pytest.approx(-1.0)
    assert approaching["tangential_velocity_km_s"] == pytest.approx(0.0, abs=1e-9)
    assert approaching["approach_angle_deg"] == pytest.approx(180.0)

    # Relative velocity purely along +x (same direction as separation) means
    # the pair is receding: POSITIVE radial velocity, ~0 deg approach angle.
    receding = _encounter_geometry_features(
        p1_start=(0.0, 0.0, 0.0), p2_start=(100.0, 0.0, 0.0),
        v1_start=(0.0, 0.0, 0.0), v2_start=(2.0, 0.0, 0.0),
        p1_tca=(0.0, 0.0, 0.0), p2_tca=(100.0, 0.0, 0.0),
        v1_tca=(0.0, 7.5, 0.0), v2_tca=(0.0, 7.5, 0.0),
    )
    assert receding["radial_velocity_km_s"] == pytest.approx(2.0)
    assert receding["approach_angle_deg"] == pytest.approx(0.0)

    # Relative velocity purely perpendicular to the separation vector is a
    # tangential/grazing pass: zero radial velocity, ~90 deg approach angle.
    grazing = _encounter_geometry_features(
        p1_start=(0.0, 0.0, 0.0), p2_start=(100.0, 0.0, 0.0),
        v1_start=(0.0, 0.0, 0.0), v2_start=(0.0, 5.0, 0.0),
        p1_tca=(0.0, 0.0, 0.0), p2_tca=(100.0, 0.0, 0.0),
        v1_tca=(0.0, 7.5, 0.0), v2_tca=(0.0, 7.5, 0.0),
    )
    assert grazing["radial_velocity_km_s"] == pytest.approx(0.0, abs=1e-9)
    assert grazing["tangential_velocity_km_s"] == pytest.approx(5.0)
    assert grazing["approach_angle_deg"] == pytest.approx(90.0)


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

    # GOREV 5 geometry features: sane ranges and internal consistency.
    assert 0.0 <= r.relative_inclination_deg <= 180.0
    assert 0.0 <= r.approach_angle_deg <= 180.0
    assert r.tangential_velocity_km_s >= 0.0
    ric_magnitude = (r.relative_radial_km**2 + r.relative_intrack_km**2 + r.relative_crosstrack_km**2) ** 0.5
    assert ric_magnitude == pytest.approx(r.min_distance_km, abs=1e-6)
