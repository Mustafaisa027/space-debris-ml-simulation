from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from space_debris.archive import verify_collection_bundle
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config


@dataclass(frozen=True)
class VerifiedObservation:
    bundle_name: str
    manifest_utc: datetime
    snapshot_utc: datetime
    tle_sha256: str


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def verified_bundle_times(
    archive_root: Path,
    collections_relative_root: str = "collections",
    config=None,
) -> list[datetime]:
    """Read immutable bundle publication times from schema-2/3 manifests.

    This is an operational liveness signal, not the scientific snapshot-slot
    coverage calculation used by the publication gate.
    """
    collections = archive_root / collections_relative_root
    if not collections.is_dir():
        return []
    timestamps: list[datetime] = []
    for bundle in sorted(collections.glob("github-run-*-attempt-*")):
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid collection manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") not in {2, 3}:
            continue
        if not str(manifest.get("github_run_id", "")).strip():
            raise ValueError(
                f"Schema-{manifest.get('schema_version')} manifest lacks "
                f"github_run_id: {manifest_path}"
            )
        if config is not None:
            verify_collection_bundle(bundle, config)
        timestamps.append(_utc(manifest.get("generated_utc"), str(manifest_path)))
    return sorted(timestamps)


def verified_observations(
    archive_root: Path, collections_relative_root: str, config
) -> list[VerifiedObservation]:
    collections = archive_root / collections_relative_root
    if not collections.is_dir():
        return []
    observations: list[VerifiedObservation] = []
    for bundle in sorted(collections.glob("github-run-*-attempt-*")):
        verified = verify_collection_bundle(bundle, config)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        tle_files = [
            item
            for item in verified.files
            if re.fullmatch(r"tle/tles_[0-9]{8}_[0-9]{6}\.txt", item.relative_path)
        ]
        if len(tle_files) != 1:
            raise ValueError(f"Verified bundle lacks one unambiguous TLE file: {bundle}")
        observations.append(
            VerifiedObservation(
                bundle_name=bundle.name,
                manifest_utc=_utc(manifest.get("generated_utc"), str(bundle / "manifest.json")),
                snapshot_utc=verified.snapshot_utc,
                tle_sha256=tle_files[0].sha256,
            )
        )
    return sorted(observations, key=lambda item: (item.snapshot_utc, item.bundle_name))


def scientific_collection_progress(
    config, observations: list[VerifiedObservation], now_utc: datetime
) -> dict:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    start = _utc(config.collection_start_utc, "collection_start_utc")
    end = _utc(config.collection_end_utc, "collection_end_utc")
    now = now_utc.astimezone(timezone.utc)
    interval_hours = float(config.poll_interval_hours)
    interval_seconds = interval_hours * 3600.0
    expected_float = (end - start).total_seconds() / interval_seconds
    expected_slots = int(round(expected_float))
    if expected_slots <= 0 or not math.isclose(expected_float, expected_slots):
        raise ValueError("Frozen window must contain an integer number of cadence slots")

    by_slot: dict[int, VerifiedObservation] = {}
    for observation in observations:
        if not start <= observation.snapshot_utc < end:
            raise ValueError(
                f"Verified observation is outside the frozen window: "
                f"{observation.bundle_name}"
            )
        slot = int((observation.snapshot_utc - start).total_seconds() // interval_seconds)
        if slot in by_slot:
            raise ValueError(
                f"Duplicate scientific slot {slot}: "
                f"{by_slot[slot].bundle_name}, {observation.bundle_name}"
            )
        by_slot[slot] = observation

    if now < start:
        due_slots = expired_slots = 0
    elif now >= end:
        due_slots = expired_slots = expected_slots
    else:
        elapsed_slots = int((now - start).total_seconds() // interval_seconds)
        expired_slots = min(expected_slots, elapsed_slots)
        due_slots = min(expected_slots, elapsed_slots + 1)

    occupied_slots = sorted(by_slot)
    occupied_due = sum(slot < due_slots for slot in occupied_slots)
    expired_occupied = sum(slot < expired_slots for slot in occupied_slots)
    maximum_reachable_slots = expired_occupied + (expected_slots - expired_slots)
    required_slots = math.ceil(
        expected_slots * float(config.min_snapshot_coverage_fraction)
    )
    ordered = [by_slot[slot] for slot in occupied_slots]
    hashes = [item.tle_sha256 for item in ordered]
    max_hash_run = 0
    current_hash_run = 0
    previous_hash = None
    for value in hashes:
        current_hash_run = current_hash_run + 1 if value == previous_hash else 1
        max_hash_run = max(max_hash_run, current_hash_run)
        previous_hash = value

    endpoint = min(max(now, start), end)
    snapshot_times = [item.snapshot_utc for item in ordered if item.snapshot_utc <= endpoint]
    if endpoint == start:
        observed_max_gap_hours = 0.0
    elif snapshot_times:
        gaps = [(snapshot_times[0] - start).total_seconds() / 3600.0]
        gaps.extend(
            (right - left).total_seconds() / 3600.0
            for left, right in zip(snapshot_times, snapshot_times[1:])
        )
        gaps.append((endpoint - snapshot_times[-1]).total_seconds() / 3600.0)
        observed_max_gap_hours = max(gaps)
    else:
        observed_max_gap_hours = (endpoint - start).total_seconds() / 3600.0

    final_window = now >= end
    # These four values are immutable publication gates in the experiment
    # config.  They remain fail-closed even when the cause of a failure is an
    # upstream feed cadence or CI outage; observed outcomes cannot relax a
    # frozen protocol.
    final_gate_pass = (
        final_window
        and len(occupied_slots) >= required_slots
        and observed_max_gap_hours <= float(config.max_snapshot_gap_hours)
        and bool(hashes)
        and len(set(hashes)) / len(hashes)
        >= float(config.min_tle_hash_diversity_fraction)
        and max_hash_run <= int(config.max_identical_tle_hash_run_bins)
    )
    if now < start:
        status = "before_window"
    elif maximum_reachable_slots < required_slots:
        status = "coverage_mathematically_unreachable"
    elif final_window:
        status = "final_gate_pass" if final_gate_pass else "final_gate_fail"
    else:
        status = "collecting"

    return {
        "schema_version": 1,
        "status": status,
        "final_gate_evaluated": final_window,
        "final_gate_pass": final_gate_pass,
        "expected_snapshot_slots": expected_slots,
        "required_snapshot_slots": required_slots,
        "due_snapshot_slots": due_slots,
        "occupied_due_slots": occupied_due,
        "occupied_snapshot_slots": len(occupied_slots),
        "coverage_of_due_slots": (
            occupied_due / due_slots if due_slots else None
        ),
        "final_window_coverage_fraction": len(occupied_slots) / expected_slots,
        "maximum_reachable_snapshot_slots": maximum_reachable_slots,
        "final_coverage_still_reachable": maximum_reachable_slots >= required_slots,
        "observed_to_now_max_snapshot_gap_hours": observed_max_gap_hours,
        "allowed_max_snapshot_gap_hours": float(config.max_snapshot_gap_hours),
        "unique_tle_hashes": len(set(hashes)),
        "tle_hash_diversity_fraction": len(set(hashes)) / len(hashes) if hashes else None,
        "required_tle_hash_diversity_fraction": float(
            config.min_tle_hash_diversity_fraction
        ),
        "max_identical_tle_hash_run_bins": max_hash_run,
        "allowed_max_identical_tle_hash_run_bins": int(
            config.max_identical_tle_hash_run_bins
        ),
    }


def cadence_health(config, bundle_times: list[datetime], now_utc: datetime) -> dict:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    if config.collection_start_utc is None or config.collection_end_utc is None:
        raise ValueError("cadence health requires a frozen collection window")
    if config.max_snapshot_gap_hours is None:
        raise ValueError("cadence health requires max_snapshot_gap_hours")

    now = now_utc.astimezone(timezone.utc)
    start = _utc(config.collection_start_utc, "collection_start_utc")
    end = _utc(config.collection_end_utc, "collection_end_utc")
    normalized = sorted(
        value.astimezone(timezone.utc)
        for value in bundle_times
        if value.tzinfo is not None and start <= value < end
    )
    latest = normalized[-1] if normalized else None
    reference = latest or start
    observed_gap_hours = max(0.0, (min(now, end) - reference).total_seconds() / 3600.0)
    allowed_gap_hours = float(config.max_snapshot_gap_hours)

    if now < start:
        status, healthy = "before_window", True
    elif now >= end:
        healthy = latest is not None and observed_gap_hours <= allowed_gap_hours
        status = "window_complete" if healthy else "window_complete_stale"
    elif observed_gap_hours > allowed_gap_hours:
        status, healthy = "stale", False
    else:
        status, healthy = "healthy", True

    return {
        "schema_version": 1,
        "status": status,
        "healthy": healthy,
        "now_utc": now.isoformat().replace("+00:00", "Z"),
        "collection_start_utc": config.collection_start_utc,
        "collection_end_utc": config.collection_end_utc,
        "verified_bundle_count": len(normalized),
        "latest_bundle_manifest_utc": (
            latest.isoformat().replace("+00:00", "Z") if latest else None
        ),
        "observed_manifest_gap_hours": observed_gap_hours,
        "allowed_gap_hours": allowed_gap_hours,
        "metric_role": "operational_liveness_only",
    }


def cadence_report_passes(report: dict) -> bool:
    """Gate finalization on both operational liveness and scientific quality."""
    if report.get("healthy") is not True:
        return False
    progress = report.get("scientific_progress")
    if not isinstance(progress, dict):
        return False
    if progress.get("final_gate_evaluated") is True:
        return progress.get("final_gate_pass") is True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify archive liveness and report scientific collection progress"
    )
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    parser.add_argument("--now-utc", help="test/diagnostic override; ISO-8601 UTC")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    now = _utc(args.now_utc, "--now-utc") if args.now_utc else datetime.now(timezone.utc)
    observations = verified_observations(
        args.archive_root, config.archive_collections, config
    )
    report = cadence_health(
        config, [item.manifest_utc for item in observations], now
    )
    report["scientific_progress"] = scientific_collection_progress(
        config, observations, now
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not cadence_report_passes(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
