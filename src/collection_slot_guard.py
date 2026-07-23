from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

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


def archived_snapshot_times(archive_root: Path, config) -> list[datetime]:
    collections = archive_root / config.archive_collections
    if not collections.is_dir():
        return []
    timestamps: list[datetime] = []
    for bundle in sorted(collections.glob("github-run-*-attempt-*")):
        sidecars = sorted((bundle / "tle").glob("tles_*.json"))
        if len(sidecars) != 1:
            raise ValueError(f"Expected exactly one TLE sidecar in {bundle}")
        try:
            metadata = json.loads(sidecars[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid TLE sidecar {sidecars[0]}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"TLE sidecar must contain a JSON object: {sidecars[0]}")
        if (
            metadata.get("catalog_version") != config.catalog_version
            or metadata.get("catalog_sha256") != config.catalog_sha256
        ):
            raise ValueError(f"Experiment-scoped archive contains another cohort: {sidecars[0]}")
        timestamps.append(_utc(metadata.get("fetched_utc"), str(sidecars[0])))
    return sorted(timestamps)


def slot_guard(config, snapshot_times: list[datetime], now_utc: datetime) -> dict:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    if config.collection_start_utc is None or config.collection_end_utc is None:
        raise ValueError("slot guard requires a frozen collection window")
    now = now_utc.astimezone(timezone.utc)
    start = _utc(config.collection_start_utc, "collection_start_utc")
    end = _utc(config.collection_end_utc, "collection_end_utc")
    interval_seconds = float(config.poll_interval_hours) * 3600.0

    if now < start:
        status, collect, current_slot = "before_window", False, None
    elif now >= end:
        status, collect, current_slot = "window_complete", False, None
    else:
        current_slot = int((now - start).total_seconds() // interval_seconds)
        occupied = {
            int((timestamp.astimezone(timezone.utc) - start).total_seconds() // interval_seconds)
            for timestamp in snapshot_times
            if timestamp.tzinfo is not None and start <= timestamp.astimezone(timezone.utc) < end
        }
        collect = current_slot not in occupied
        status = "collect" if collect else "slot_already_collected"

    return {
        "schema_version": 1,
        "status": status,
        "collect": collect,
        "now_utc": now.isoformat().replace("+00:00", "Z"),
        "collection_start_utc": config.collection_start_utc,
        "collection_end_utc": config.collection_end_utc,
        "current_slot": current_slot,
        "archived_snapshot_count": len(snapshot_times),
        "archive_collections": config.archive_collections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect only when the current frozen cadence slot is absent"
    )
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    parser.add_argument("--now-utc", help="test/diagnostic override; ISO-8601 UTC")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    now = _utc(args.now_utc, "--now-utc") if args.now_utc else datetime.now(timezone.utc)
    report = slot_guard(config, archived_snapshot_times(args.archive_root, config), now)
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as stream:
            stream.write(f"collect={'true' if report['collect'] else 'false'}\n")
            stream.write(f"status={report['status']}\n")
            stream.write(
                f"current_slot={'' if report['current_slot'] is None else report['current_slot']}\n"
            )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
