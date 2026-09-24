from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from collection_cadence_health import (
    cadence_report_passes,
    VerifiedObservation,
    cadence_health,
    scientific_collection_progress,
    verified_bundle_times,
)
from space_debris.experiment import load_experiment_config
from space_debris.ml import _apply_collection_window


def _config():
    return load_experiment_config("config/experiment_60_days.json")


def test_cadence_health_fails_after_frozen_gap_without_bundle():
    config = _config()
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    report = cadence_health(config, [], start + timedelta(hours=6, seconds=1))

    assert report["status"] == "stale"
    assert report["healthy"] is False
    assert report["metric_role"] == "operational_liveness_only"


def test_cadence_report_blocks_finalization_when_scientific_gate_fails():
    active = {
        "healthy": True,
        "scientific_progress": {
            "final_gate_evaluated": False,
            "final_gate_pass": False,
        },
    }
    failed_final = {
        "healthy": True,
        "scientific_progress": {
            "final_gate_evaluated": True,
            "final_gate_pass": False,
        },
    }
    passed_final = {
        "healthy": True,
        "scientific_progress": {
            "final_gate_evaluated": True,
            "final_gate_pass": True,
        },
    }

    assert cadence_report_passes(active) is True
    assert cadence_report_passes(failed_final) is False
    assert cadence_report_passes(passed_final) is True


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
    assert report["verified_bundle_count"] == 1


def test_cadence_health_ignores_legacy_manifests_and_fails_stale_after_window(tmp_path):
    config = _config()
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
    bundle = tmp_path / "collections" / "github-run-42-attempt-1"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "generated_utc": end.isoformat()}),
        encoding="utf-8",
    )

    assert verified_bundle_times(tmp_path) == []
    report = cadence_health(config, [], end)
    assert report["status"] == "window_complete_stale"
    assert report["healthy"] is False


def test_cadence_health_closes_healthy_with_recent_final_bundle():
    config = _config()
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))

    report = cadence_health(config, [end - timedelta(hours=2)], end)

    assert report["status"] == "window_complete"
    assert report["healthy"] is True


def test_cadence_health_rejects_naive_now():
    with pytest.raises(ValueError, match="timezone-aware"):
        cadence_health(_config(), [], datetime(2026, 7, 20))


def _observation(slot: int, tle_hash: str = "a" * 64) -> VerifiedObservation:
    config = load_experiment_config("config/experiment_10_days_v2.json")
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    timestamp = start + timedelta(hours=2 * slot, minutes=5)
    return VerifiedObservation(
        bundle_name=f"github-run-{slot + 1}-attempt-1",
        manifest_utc=timestamp + timedelta(minutes=5),
        snapshot_utc=timestamp,
        tle_sha256=tle_hash,
    )


def test_scientific_progress_reports_due_and_total_coverage():
    config = load_experiment_config("config/experiment_10_days_v2.json")
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    report = scientific_collection_progress(
        config,
        [_observation(0), _observation(1, "b" * 64)],
        start + timedelta(hours=4, minutes=10),
    )

    assert report["status"] == "collecting"
    assert report["expected_snapshot_slots"] == 120
    assert report["due_snapshot_slots"] == 3
    assert report["occupied_due_slots"] == 2
    assert report["coverage_of_due_slots"] == pytest.approx(2 / 3)
    assert report["final_window_coverage_fraction"] == pytest.approx(2 / 120)
    assert report["tle_hash_diversity_fraction"] == 1.0
    assert report["max_identical_tle_hash_run_bins"] == 1
    assert report["final_gate_evaluated"] is False


def test_scientific_progress_rejects_duplicate_physical_slot():
    config = load_experiment_config("config/experiment_10_days_v2.json")
    first = _observation(0)
    duplicate = VerifiedObservation(
        bundle_name="github-run-99-attempt-1",
        manifest_utc=first.manifest_utc + timedelta(minutes=1),
        snapshot_utc=first.snapshot_utc + timedelta(minutes=10),
        tle_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="Duplicate scientific slot 0"):
        scientific_collection_progress(
            config, [first, duplicate], first.snapshot_utc + timedelta(minutes=20)
        )


def test_scientific_progress_detects_unreachable_final_coverage():
    config = load_experiment_config("config/experiment_10_days_v2.json")
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
    report = scientific_collection_progress(config, [_observation(0)], end)

    assert report["status"] == "coverage_mathematically_unreachable"
    assert report["final_gate_evaluated"] is True
    assert report["final_gate_pass"] is False
    assert report["maximum_reachable_snapshot_slots"] == 1


def test_final_progress_enforces_frozen_tle_quality_gates():
    config = load_experiment_config("config/experiment_10_days_v2.json")
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
    observations = [_observation(slot, "a" * 64) for slot in range(120)]

    report = scientific_collection_progress(config, observations, end)

    assert report["final_window_coverage_fraction"] == 1.0
    assert report["observed_to_now_max_snapshot_gap_hours"] < 6.0
    assert report["tle_hash_diversity_fraction"] == pytest.approx(1 / 120)
    assert report["max_identical_tle_hash_run_bins"] == 120
    assert report["final_gate_pass"] is False


def test_final_progress_matches_publication_window_quality_metrics():
    config = load_experiment_config("config/experiment_10_days_v2.json")
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
    observations = [_observation(0), _observation(1, "b" * 64)]
    frame = pd.DataFrame(
        {
            "snapshot_utc": [item.snapshot_utc for item in observations],
            "collection_id": [f"collection-{index}" for index in range(2)],
        }
    )
    records = [
        {
            "snapshot_utc": item.snapshot_utc.isoformat(),
            "collection_id": f"collection-{index}",
            "input_sha256": item.tle_sha256,
        }
        for index, item in enumerate(observations)
    ]
    _, publication = _apply_collection_window(
        frame,
        "snapshot_utc",
        start_utc=config.collection_start_utc,
        end_utc=config.collection_end_utc,
        poll_interval_hours=config.poll_interval_hours,
        snapshot_records=records,
    )
    progress = scientific_collection_progress(config, observations, end)

    assert progress["expected_snapshot_slots"] == publication["expected_snapshot_slots"]
    assert (
        progress["occupied_snapshot_slots"]
        == publication["occupied_snapshot_slots"]
    )
    assert (
        progress["final_window_coverage_fraction"]
        == publication["snapshot_coverage_fraction"]
    )
    assert (
        progress["observed_to_now_max_snapshot_gap_hours"]
        == publication["max_snapshot_gap_hours"]
    )
    assert (
        progress["tle_hash_diversity_fraction"]
        == publication["tle_input_hash_diversity_fraction"]
    )
    assert (
        progress["max_identical_tle_hash_run_bins"]
        == publication["max_identical_tle_hash_run_bins"]
    )
