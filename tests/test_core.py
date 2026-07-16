"""Unit tests for the space-debris conjunction pipeline.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skyfield.api import load

from space_debris.core import (
    PairResult,
    _altitude_km,
    _bounded_min_search,
    _coarse_offsets_seconds,
    _distance_km,
    _encounter_geometry_features,
    _global_interval_min_search,
    _iso_z,
    _refine_tca,
    _risk_label,
    _to_skyfield_time,
    _to_skyfield_time_precise,
    build_satellites,
    read_tles,
    simulate_pairs,
    write_pair_results,
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


def test_bounded_min_search_matches_analytic_linear_encounter():
    # Standard short-duration conjunction approximation: near TCA, relative
    # motion is well approximated as a straight line r(t) = r0 + v*t, which
    # makes d(t) an exact quadratic with a closed-form minimum. Here the
    # relative velocity is purely along -y, so the y-separation is nulled at
    # t* = 40/15 s, leaving the x-separation (30 km) as the true minimum
    # distance -- both known exactly by construction, not just numerically.
    r0 = np.array([30.0, 40.0, 0.0])
    v = np.array([0.0, -15.0, 0.0])
    t_star = float(-np.dot(r0, v) / np.dot(v, v))
    d_star = float(np.linalg.norm(r0 + v * t_star))
    assert d_star == pytest.approx(30.0)

    def distance_fn(offset_s: float) -> float:
        return float(np.linalg.norm(r0 + v * offset_s))

    offset_s, distance_km, boundary_flag = _bounded_min_search(
        distance_fn, lower_s=t_star - 300.0, upper_s=t_star + 300.0,
    )

    assert abs(offset_s - t_star) <= 1.0
    assert abs(distance_km - d_star) <= 0.1
    assert boundary_flag is False


def test_bounded_min_search_flags_boundary_when_true_minimum_outside_window():
    # Same analytic encounter as above, but the search window is narrowed to
    # [0, 1] s, well short of the true t* (~2.67 s). Distance is strictly
    # decreasing across the whole window, so the optimizer must land on the
    # upper edge -- exactly the "true minimum may be outside the window"
    # case the boundary flag exists to catch.
    r0 = np.array([30.0, 40.0, 0.0])
    v = np.array([0.0, -15.0, 0.0])

    def distance_fn(offset_s: float) -> float:
        return float(np.linalg.norm(r0 + v * offset_s))

    offset_s, _distance_km, boundary_flag = _bounded_min_search(
        distance_fn, lower_s=0.0, upper_s=1.0,
    )

    assert offset_s == pytest.approx(1.0, abs=0.1)
    assert boundary_flag is True


@pytest.mark.parametrize(
    ("distance_fn", "expected_offset"),
    [
        (lambda offset_s: offset_s, 0.0),
        (lambda offset_s: 60.0 - offset_s, 60.0),
    ],
)
def test_bounded_min_search_keeps_exact_start_and_end_minima(distance_fn, expected_offset):
    offset_s, distance_km, boundary_flag = _bounded_min_search(distance_fn, 0.0, 60.0)

    assert offset_s == expected_offset
    assert distance_km == 0.0
    assert 0.0 <= offset_s <= 60.0
    assert boundary_flag is True


def test_precise_skyfield_conversion_and_iso_preserve_microseconds():
    class RecordingTimescale:
        def utc(self, year, month, day, hour, minute, second):
            return year, month, day, hour, minute, second

    dt = datetime(2026, 7, 1, 12, 34, 56, 789123, tzinfo=timezone.utc)

    converted = _to_skyfield_time_precise(RecordingTimescale(), dt)

    assert converted[-1] == pytest.approx(56.789123, abs=1e-9)
    assert _iso_z(dt, preserve_microseconds=True) == "2026-07-01T12:34:56.789123Z"


def test_refine_tca_delegates_full_simulation_horizon(monkeypatch):
    captured = {}

    def fake_search(_distance_fn, horizon_s, step_s, xatol_s=0.1):
        captured.update(horizon_s=horizon_s, step_s=step_s, xatol_s=xatol_s)
        return horizon_s, 5.0, True

    monkeypatch.setattr("space_debris.core._global_interval_min_search", fake_search)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)

    tca, distance_km, boundary_flag = _refine_tca(
        object(), object(), object(), start,
        step_minutes=5.0,
        horizon_minutes=12.0,
    )

    assert captured["horizon_s"] == 12.0 * 60.0
    assert captured["step_s"] == 5.0 * 60.0
    assert tca == start + timedelta(minutes=12)
    assert distance_km == 5.0
    assert boundary_flag is True


def test_global_interval_search_finds_narrow_minimum_missed_by_single_window():
    # Coarse samples at 0, 10, 20 and 30 seconds see the broad minimum at
    # t=10 (distance 2). A genuinely closer but narrow minimum at t=25 lies
    # between two coarse samples whose distances are both worse. The former
    # single-window strategy searched only [0, 20] around the coarse winner.
    def distance_fn(offset_s: float) -> float:
        broad_minimum = 2.0 + ((offset_s - 10.0) / 10.0) ** 2
        narrow_minimum = 0.5 + ((offset_s - 25.0) / 1.0) ** 2
        return min(broad_minimum, narrow_minimum)

    coarse_offsets = _coarse_offsets_seconds(horizon_s=30.0, step_s=10.0)
    coarse_winner = min(coarse_offsets, key=distance_fn)
    old_offset, old_distance, _ = _bounded_min_search(
        distance_fn,
        max(0.0, coarse_winner - 10.0),
        min(30.0, coarse_winner + 10.0),
    )

    new_offset, new_distance, boundary_flag = _global_interval_min_search(
        distance_fn,
        horizon_s=30.0,
        step_s=10.0,
    )

    assert old_offset == pytest.approx(10.0, abs=0.1)
    assert old_distance == pytest.approx(2.0, abs=0.1)
    assert new_offset == pytest.approx(25.0, abs=0.1)
    assert new_distance == pytest.approx(0.5, abs=0.1)
    assert new_distance < old_distance
    assert boundary_flag is False


def test_global_interval_search_refines_final_short_interval(monkeypatch):
    searched_intervals = []
    real_search = _bounded_min_search

    def recording_search(distance_fn, lower_s, upper_s, xatol_s=0.1):
        searched_intervals.append((lower_s, upper_s))
        return real_search(distance_fn, lower_s, upper_s, xatol_s)

    monkeypatch.setattr("space_debris.core._bounded_min_search", recording_search)

    offset_s, distance_km, boundary_flag = _global_interval_min_search(
        lambda value: (value - 10.5) ** 2,
        horizon_s=11.0,
        step_s=5.0,
    )

    assert searched_intervals == [(0.0, 5.0), (5.0, 10.0), (10.0, 11.0)]
    assert offset_s == pytest.approx(10.5, abs=0.1)
    assert distance_km == pytest.approx(0.0, abs=0.01)
    assert boundary_flag is True  # within one second of the real horizon


@pytest.mark.parametrize(
    ("distance_fn", "expected_offset"),
    [
        (lambda offset_s: offset_s, 0.0),
        (lambda offset_s: 20.0 - offset_s, 20.0),
    ],
)
def test_global_interval_search_flags_only_real_simulation_boundaries(distance_fn, expected_offset):
    offset_s, distance_km, boundary_flag = _global_interval_min_search(
        distance_fn,
        horizon_s=20.0,
        step_s=10.0,
    )

    assert offset_s == expected_offset
    assert distance_km == 0.0
    assert boundary_flag is True


def test_global_interval_search_does_not_flag_internal_interval_boundary():
    offset_s, distance_km, boundary_flag = _global_interval_min_search(
        lambda value: abs(value - 10.0),
        horizon_s=20.0,
        step_s=10.0,
    )

    assert offset_s == 10.0
    assert distance_km == 0.0
    assert boundary_flag is False


def test_global_interval_search_never_worse_than_its_coarse_grid():
    distance_fn = lambda value: 4.0 + math.sin(value / 3.0) + ((value - 17.0) / 20.0) ** 2
    coarse_offsets = _coarse_offsets_seconds(horizon_s=23.0, step_s=5.0)
    coarse_best = min(distance_fn(offset_s) for offset_s in coarse_offsets)

    offset_s, refined_distance, _ = _global_interval_min_search(
        distance_fn,
        horizon_s=23.0,
        step_s=5.0,
    )

    assert 0.0 <= offset_s <= 23.0
    assert refined_distance <= coarse_best


@pytest.mark.parametrize(
    ("horizon_s", "step_s", "message"),
    [(-1.0, 5.0, "horizon must be non-negative"), (10.0, 0.0, "step must be positive")],
)
def test_global_interval_search_rejects_invalid_grid(horizon_s, step_s, message):
    with pytest.raises(ValueError, match=message):
        _global_interval_min_search(lambda value: value, horizon_s, step_s)


def test_refined_tca_never_worse_than_coarse_grid_only():
    # Regression test for the missed-close-approach bug: rebuild the OLD
    # coarse-grid-only minimum (sampling every step_minutes and nothing
    # else) independently, and confirm the pipeline's refined min_distance_km
    # is never larger than it -- refinement must only ever tighten the
    # estimate, never loosen it.
    objs = read_tles(DEMO_TLE)[:8]
    sats = build_satellites(objs)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    horizon_minutes, step_minutes = 30, 5

    rows = simulate_pairs(
        sats, start, horizon_minutes=horizon_minutes, step_minutes=step_minutes,
        leo_min_altitude_km=160, leo_max_altitude_km=2000,
        fixed_threshold_km=50, label_threshold_km=200,
        label_relative_velocity_km_s=5,
    )
    assert rows
    by_pair = {(r.object_1, r.object_2): r for r in rows}

    ts = load.timescale()
    checked = 0
    for sat1, sat2 in combinations(sats, 2):
        key = (sat1.name, sat2.name)
        if key not in by_pair:
            continue
        checked += 1

        coarse_best = _distance_km(
            sat1.at(_to_skyfield_time(ts, start)).position.km,
            sat2.at(_to_skyfield_time(ts, start)).position.km,
        )
        for offset in range(0, horizon_minutes + 1, step_minutes):
            t = _to_skyfield_time(ts, start + timedelta(minutes=offset))
            coarse_best = min(coarse_best, _distance_km(sat1.at(t).position.km, sat2.at(t).position.km))

        assert by_pair[key].min_distance_km <= coarse_best + 1e-6

    assert checked > 0


def _make_pair_result(**overrides) -> PairResult:
    defaults = dict(
        snapshot_utc="2026-07-01T00:00:00Z",
        object_1="SAT-A",
        object_2="SAT-B",
        tle_epoch_1_utc="2026-06-30T00:00:00Z",
        tle_epoch_2_utc="2026-06-30T00:00:00Z",
        max_tle_age_hours=24.0,
        tca_utc="2026-07-01T00:10:00Z",
        time_to_tca_min=10.0,
        current_distance_km=50.0,
        min_distance_km=15.0,
        relative_velocity_km_s=12.0,
        altitude_1_km=550.0,
        altitude_2_km=555.0,
        altitude_difference_km=5.0,
        relative_radial_km=3.0,
        relative_intrack_km=4.0,
        relative_crosstrack_km=0.0,
        relative_inclination_deg=45.0,
        radial_velocity_km_s=-1.0,
        tangential_velocity_km_s=0.5,
        approach_angle_deg=170.0,
        risk_score=0.8,
        fixed_threshold_alarm=1,
        risk_label=1,
    )
    defaults.update(overrides)
    return PairResult(**defaults)


def test_write_pair_results_embeds_provenance_header(tmp_path):
    rows = [
        _make_pair_result(
            tca_utc="2026-07-01T00:10:00.123456Z",
            time_to_tca_min=10.0020576,
            tca_boundary_flag=1,
        )
    ]
    path = tmp_path / "conjunction_dataset.csv"

    write_pair_results(path, rows, source="leo_mixed preset", config_summary="horizon=720min")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert raw_lines[0].startswith("# generated_utc:")
    assert any(line.startswith("# git_commit:") for line in raw_lines)
    assert any("leo_mixed preset" in line for line in raw_lines)
    assert any("horizon=720min" in line for line in raw_lines)

    # Downstream readers must be able to skip the header and recover the data.
    df = pd.read_csv(path, comment="#")
    assert len(df) == 1
    assert df.iloc[0]["object_1"] == "SAT-A"
    assert df.iloc[0]["risk_label"] == 1
    assert df.iloc[0]["tca_utc"] == "2026-07-01T00:10:00.123456Z"
    assert df.iloc[0]["time_to_tca_min"] == pytest.approx(10.0020576, abs=1e-9)
    assert df.iloc[0]["tca_boundary_flag"] == 1


def test_write_pair_results_without_provenance_args_still_readable(tmp_path):
    # Backwards-compatible default: existing callers that don't pass
    # source/config_summary still produce a valid, readable CSV.
    rows = [_make_pair_result()]
    path = tmp_path / "conjunction_dataset.csv"

    write_pair_results(path, rows)

    df = pd.read_csv(path, comment="#")
    assert len(df) == 1
    assert df.iloc[0]["tca_boundary_flag"] == 0
