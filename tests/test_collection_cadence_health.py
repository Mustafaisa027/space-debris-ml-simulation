from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from collection_cadence_health import cadence_health, verified_bundle_times
from space_debris.experiment import load_experiment_config


def _config():
    return load_experiment_config("config/experiment_60_days.json")


def test_cadence_health_fails_after_frozen_gap_without_bundle():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    report = cadence_health(config, [], start + timedelta(hours=6, seconds=1))

    assert report["status"] == "stale"
    assert report["healthy"] is False
    assert report["metric_role"] == "operational_liveness_only"


def test_cadence_health_uses_latest_schema_two_manifest(tmp_path):
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    bundle = tmp_path / "collections" / "github-run-42-attempt-1"
    bundle.mkdir(parents=True)
    generated = start + timedelta(hours=8)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "github_run_id": "42",
                "generated_utc": generated.isoformat().replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )

    times = verified_bundle_times(tmp_path)
    report = cadence_health(config, times, generated + timedelta(hours=4))

    assert times == [generated]
    assert report["status"] == "healthy"
    assert report["schema_2_bundle_count"] == 1


def test_cadence_health_ignores_legacy_manifests_and_closes_after_window(tmp_path):
    config = _config()
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
    bundle = tmp_path / "collections" / "github-run-42-attempt-1"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "generated_utc": end.isoformat()}),
        encoding="utf-8",
    )

    assert verified_bundle_times(tmp_path) == []
    assert cadence_health(config, [], end)["status"] == "window_complete"


def test_cadence_health_rejects_naive_now():
    with pytest.raises(ValueError, match="timezone-aware"):
        cadence_health(_config(), [], datetime(2026, 7, 20))
