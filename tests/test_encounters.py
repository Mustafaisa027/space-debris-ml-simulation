from __future__ import annotations

import math

import pytest

from space_debris.encounters import (
    MAX_BOUND_RELATIVE_SPEED_KM_S,
    conservative_candidate_screen,
)


def test_default_speed_bound_is_twice_earth_surface_escape_speed():
    assert 22.0 < MAX_BOUND_RELATIVE_SPEED_KM_S < 23.0


def test_conservative_screen_keeps_between_sample_threshold_crossing():
    offsets = [0.0, 300.0]
    relative_speed = 20.0
    miss_distance = 10.0
    distances = [
        math.hypot(miss_distance, relative_speed * (offset - 150.0))
        for offset in offsets
    ]

    decision = conservative_candidate_screen(offsets, distances, threshold_km=200.0)

    assert min(distances) > 200.0  # coarse-only screening would reject it
    assert decision.is_candidate is True
    assert decision.max_interval_margin_km == pytest.approx(
        MAX_BOUND_RELATIVE_SPEED_KM_S * 150.0
    )


@pytest.mark.parametrize("tca_s", [0.0, 1.0, 75.0, 149.0, 150.0, 225.0, 299.0, 300.0])
def test_screen_has_no_false_negative_for_bounded_linear_encounters(tca_s):
    offsets = [0.0, 300.0]
    relative_speed = MAX_BOUND_RELATIVE_SPEED_KM_S * 0.99
    miss_distance = 50.0
    distances = [
        math.hypot(miss_distance, relative_speed * (offset - tca_s))
        for offset in offsets
    ]

    decision = conservative_candidate_screen(offsets, distances, threshold_km=50.0)

    assert decision.is_candidate is True


def test_screen_rejects_pair_beyond_conservative_margin():
    offsets = [0.0, 300.0]
    margin = MAX_BOUND_RELATIVE_SPEED_KM_S * 150.0

    decision = conservative_candidate_screen(
        offsets,
        [200.0 + margin + 2.0, 200.0 + margin + 2.0],
        threshold_km=200.0,
    )

    assert decision.is_candidate is False


def test_screen_retains_equality_at_margin_plus_guard():
    margin = MAX_BOUND_RELATIVE_SPEED_KM_S * 150.0
    endpoint_distance = 200.0 + margin + 1.0

    decision = conservative_candidate_screen(
        [0.0, 300.0], [endpoint_distance, endpoint_distance], threshold_km=200.0
    )

    assert decision.is_candidate is True


def test_final_short_interval_uses_its_actual_duration():
    offsets = [0.0, 300.0, 420.0]
    long_margin = MAX_BOUND_RELATIVE_SPEED_KM_S * 150.0
    short_margin = MAX_BOUND_RELATIVE_SPEED_KM_S * 60.0
    distances = [200.0 + long_margin + 10.0, 200.0 + short_margin, 200.0 + short_margin]

    decision = conservative_candidate_screen(offsets, distances, threshold_km=200.0)

    assert decision.is_candidate is True


@pytest.mark.parametrize(
    ("offsets", "distances", "message"),
    [([], [], "non-empty"), ([0, 0], [1, 1], "strictly increasing"), ([0], [-1], "non-negative")],
)
def test_screen_rejects_invalid_inputs(offsets, distances, message):
    with pytest.raises(ValueError, match=message):
        conservative_candidate_screen(offsets, distances, threshold_km=200.0)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_screen_rejects_non_finite_threshold(threshold):
    with pytest.raises(ValueError, match="finite"):
        conservative_candidate_screen([0.0, 30.0], [100.0, 100.0], threshold)
