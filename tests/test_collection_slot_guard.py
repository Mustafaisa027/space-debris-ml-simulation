from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from collection_slot_guard import archived_snapshot_times, slot_guard
from space_debris.experiment import load_experiment_config


def _config():
    return load_experiment_config("config/experiment_10_days_v3.json")


def test_slot_guard_collects_once_per_half_open_slot():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    first = start + timedelta(minutes=5)

    assert slot_guard(config, [], first)["collect"] is True
    duplicate = slot_guard(config, [first], start + timedelta(minutes=35))
    assert duplicate["collect"] is False
    assert duplicate["status"] == "slot_already_collected"
    assert slot_guard(config, [first], start + timedelta(hours=2, minutes=5))[
        "collect"
    ] is True


def test_slot_guard_does_not_refetch_just_across_a_slot_boundary():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    late_previous_slot_fetch = start + timedelta(hours=1, minutes=59)

    report = slot_guard(
        config,
        [late_previous_slot_fetch],
        start + timedelta(hours=2),
    )

    assert report["current_slot"] == 1
    assert report["collect"] is False
    assert report["status"] == "poll_interval_not_elapsed"
    assert report["latest_snapshot_utc"] == "2026-07-26T02:16:00Z"
    assert report["minimum_poll_interval_hours"] == 2.0
    assert report["seconds_until_next_poll"] == pytest.approx(7140.0)


def test_slot_guard_allows_empty_slot_at_exact_minimum_poll_interval():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    late_previous_slot_fetch = start + timedelta(hours=1, minutes=59)

    report = slot_guard(
        config,
        [late_previous_slot_fetch],
        late_previous_slot_fetch + timedelta(hours=2),
    )

    assert report["current_slot"] == 1
    assert report["collect"] is True
    assert report["status"] == "collect"
    assert report["seconds_until_next_poll"] == 0.0


def test_v4_separates_scientific_slot_width_from_provider_poll_floor():
    config = load_experiment_config("config/experiment_15_days_v4.json")
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    late_previous_slot_fetch = start + timedelta(hours=2, minutes=50)

    report = slot_guard(
        config,
        [late_previous_slot_fetch],
        late_previous_slot_fetch + timedelta(hours=2),
    )

    assert report["current_slot"] == 1
    assert report["scientific_slot_interval_hours"] == 3.0
    assert report["minimum_poll_interval_hours"] == 2.0
    assert report["collect"] is True
    assert report["status"] == "collect"


def test_slot_guard_is_closed_outside_window():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))

    assert slot_guard(config, [], start - timedelta(seconds=1))["status"] == "before_window"
    assert slot_guard(config, [], end)["status"] == "window_complete"


def test_archived_snapshot_times_reads_only_scoped_v3_sidecars(tmp_path):
    config = _config()
    bundle = (
        tmp_path
        / config.archive_collections
        / "github-run-42-attempt-1"
        / "tle"
    )
    bundle.mkdir(parents=True)
    sidecar = {
        "fetched_utc": "2026-07-26T00:22:00Z",
        "catalog_version": config.catalog_version,
        "catalog_sha256": config.catalog_sha256,
    }
    (bundle / "tles_20260726_002200.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    assert archived_snapshot_times(tmp_path, config) == [
        datetime.fromisoformat("2026-07-26T00:22:00+00:00")
    ]


def test_archived_snapshot_times_rejects_cross_experiment_content(tmp_path):
    config = _config()
    bundle = (
        tmp_path
        / config.archive_collections
        / "github-run-42-attempt-1"
        / "tle"
    )
    bundle.mkdir(parents=True)
    (bundle / "tles_20260726_002200.json").write_text(
        json.dumps(
            {
                "fetched_utc": "2026-07-26T00:22:00Z",
                "catalog_version": "wrong",
                "catalog_sha256": config.catalog_sha256,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="another cohort"):
        archived_snapshot_times(tmp_path, config)
