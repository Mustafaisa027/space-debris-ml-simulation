from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from space_debris.archive import verify_collection_bundle
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config


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
    """Read immutable bundle publication times from schema-2 manifests.

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
            raise ValueError(f"Schema-2 manifest lacks github_run_id: {manifest_path}")
        if config is not None:
            verify_collection_bundle(bundle, config)
        timestamps.append(_utc(manifest.get("generated_utc"), str(manifest_path)))
    return sorted(timestamps)


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
        "schema_2_bundle_count": len(normalized),
        "latest_bundle_manifest_utc": (
            latest.isoformat().replace("+00:00", "Z") if latest else None
        ),
        "observed_manifest_gap_hours": observed_gap_hours,
        "allowed_gap_hours": allowed_gap_hours,
        "metric_role": "operational_liveness_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when the active frozen collection has no recent schema-2 bundle"
    )
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    parser.add_argument("--now-utc", help="test/diagnostic override; ISO-8601 UTC")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    now = _utc(args.now_utc, "--now-utc") if args.now_utc else datetime.now(timezone.utc)
    report = cadence_health(
        config,
        verified_bundle_times(
            args.archive_root, config.archive_collections, config
        ),
        now,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
