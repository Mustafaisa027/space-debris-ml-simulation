from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from collection_slot_guard import archived_snapshot_times, slot_guard
from space_debris.experiment import load_experiment_config


def _config():
    return load_experiment_config("config/experiment_10_days_v2.json")


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


def test_slot_guard_is_closed_outside_window():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))

    assert slot_guard(config, [], start - timedelta(seconds=1))["status"] == "before_window"
    assert slot_guard(config, [], end)["status"] == "window_complete"


def test_archived_snapshot_times_reads_only_scoped_v2_sidecars(tmp_path):
    config = _config()
    bundle = (
        tmp_path
        / config.archive_collections
        / "github-run-42-attempt-1"
        / "tle"
    )
    bundle.mkdir(parents=True)
    sidecar = {
        "fetched_utc": "2026-07-24T00:22:00Z",
        "catalog_version": config.catalog_version,
        "catalog_sha256": config.catalog_sha256,
    }
    (bundle / "tles_20260724_002200.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    assert archived_snapshot_times(tmp_path, config) == [
        datetime.fromisoformat("2026-07-24T00:22:00+00:00")
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
    (bundle / "tles_20260724_002200.json").write_text(
        json.dumps(
            {
                "fetched_utc": "2026-07-24T00:22:00Z",
                "catalog_version": "wrong",
                "catalog_sha256": config.catalog_sha256,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="another cohort"):
        archived_snapshot_times(tmp_path, config)
