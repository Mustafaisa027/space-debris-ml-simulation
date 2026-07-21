from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from capture_runtime_environment import runtime_environment
from rebuild_history import rebuild_latest_schema_history
from space_debris import core as core_module
from space_debris import encounters as encounters_module
from space_debris.core import (
    build_satellites,
    filter_conjunctions,
    read_tles,
    simulate_pairs,
    write_pair_results,
)
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.archive import catalog_id_set_sha256
from space_debris.provenance import generated_utc, git_commit_hash, write_json_atomic


RESIMULATION_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_metadata(snapshot_path: Path) -> dict:
    sidecar = snapshot_path.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"Canonical resimulation requires a TLE sidecar: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid TLE sidecar {sidecar}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"TLE sidecar must contain a JSON object: {sidecar}")
    return metadata


def _snapshot_utc(snapshot_path: Path, metadata: dict) -> datetime:
    if metadata.get("fetched_utc"):
        parsed = datetime.fromisoformat(str(metadata["fetched_utc"]).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    stamp = snapshot_path.stem.removeprefix("tles_")
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def _run_fingerprint(snapshot_path: Path, metadata: dict, config) -> tuple[str, dict]:
    sidecar = snapshot_path.with_suffix(".json")
    algorithm_files = [Path(__file__), Path(core_module.__file__), Path(encounters_module.__file__)]
    algorithm_hashes = {path.name: _sha256(path) for path in algorithm_files}
    payload = {
        "schema_version": RESIMULATION_SCHEMA_VERSION,
        "input_sha256": _sha256(snapshot_path),
        "sidecar_sha256": _sha256(sidecar) if sidecar.exists() else None,
        "snapshot_utc": _snapshot_utc(snapshot_path, metadata).isoformat(),
        "algorithm_files": algorithm_hashes,
        "runtime_environment": runtime_environment(),
        "simulation": {
            "max_objects": config.max_objects,
            "horizon_minutes": config.horizon_minutes,
            "step_minutes": config.step_minutes,
            "screening_step_seconds": config.screening_step_seconds,
            "leo_min_altitude_km": config.leo_min_altitude_km,
            "leo_max_altitude_km": config.leo_max_altitude_km,
            "candidate_threshold_km": config.candidate_threshold_km,
            "fixed_threshold_km": config.fixed_threshold_km,
            "label_threshold_km": config.label_threshold_km,
            "label_relative_velocity_km_s": config.label_relative_velocity_km_s,
            "max_tle_age_hours": config.max_tle_age_hours,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), payload


def resimulate_snapshot(
    snapshot_path: Path,
    output_root: Path,
    config,
) -> dict:
    """Re-simulate one immutable TLE snapshot and atomically publish its run."""
    metadata = _snapshot_metadata(snapshot_path)
    collection_id = snapshot_path.stem.removeprefix("tles_")
    final_dir = output_root / collection_id
    completion_file = final_dir / "resimulation.json"
    input_sha256 = _sha256(snapshot_path)
    run_fingerprint, fingerprint_inputs = _run_fingerprint(snapshot_path, metadata, config)
    if completion_file.exists():
        existing = json.loads(completion_file.read_text(encoding="utf-8"))
        required_outputs = [final_dir / "conjunction_dataset.csv", final_dir / "identified_conjunctions.csv"]
        outputs_valid = all(path.exists() for path in required_outputs) and all(
            existing.get("output_sha256", {}).get(path.name) == _sha256(path)
            for path in required_outputs
        )
        if existing.get("run_fingerprint") == run_fingerprint and outputs_valid:
            return {**existing, "status": "skipped_complete"}
        raise ValueError(f"Existing resimulation fingerprint/output integrity differs: {final_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{collection_id}.", dir=output_root))
    try:
        all_objects = read_tles(snapshot_path)
        actual_catalog_ids = [obj.line1[2:7].strip() for obj in all_objects]
        expected_metadata = {
            "catalog_version": config.catalog_version,
            "catalog_sha256": config.catalog_sha256,
            "object_count": config.max_objects,
            "requested_object_count": config.max_objects,
            "preset": config.preset,
        }
        mismatches = {
            key: {"expected": value, "observed": metadata.get(key)}
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if (
            len(actual_catalog_ids) != config.max_objects
            or len(set(actual_catalog_ids)) != config.max_objects
            or catalog_id_set_sha256(actual_catalog_ids) != config.catalog_sha256
        ):
            mismatches["tle_catalog_ids"] = {
                "expected_count": config.max_objects,
                "expected_sha256": config.catalog_sha256,
                "observed_count": len(actual_catalog_ids),
                "observed_sha256": catalog_id_set_sha256(actual_catalog_ids),
            }
        if mismatches:
            raise ValueError(
                "Canonical resimulation frozen catalogue mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        objects = all_objects
        satellites = build_satellites(objects)
        start_utc = _snapshot_utc(snapshot_path, metadata)
        screening: dict = {}
        rows = simulate_pairs(
            satellites=satellites,
            start_utc=start_utc,
            horizon_minutes=config.horizon_minutes,
            step_minutes=config.step_minutes,
            leo_min_altitude_km=config.leo_min_altitude_km,
            leo_max_altitude_km=config.leo_max_altitude_km,
            fixed_threshold_km=config.fixed_threshold_km,
            label_threshold_km=config.label_threshold_km,
            label_relative_velocity_km_s=config.label_relative_velocity_km_s,
            max_tle_age_hours=config.max_tle_age_hours,
            candidate_screening_threshold_km=config.candidate_threshold_km,
            screening_step_seconds=config.screening_step_seconds,
            screening_report=screening,
        )
        conjunctions = filter_conjunctions(rows, config.candidate_threshold_km)
        screening["exact_candidates"] = len(conjunctions)
        config_summary = (
            f"experiment_config={config.path}, horizon={config.horizon_minutes}min, "
            f"step={config.step_minutes}min, candidate_threshold_km={config.candidate_threshold_km}, "
            f"screening_step={config.screening_step_seconds}s, "
            f"fixed_threshold_km={config.fixed_threshold_km}, "
            f"label_threshold_km={config.label_threshold_km}, "
            f"label_relative_velocity_km_s={config.label_relative_velocity_km_s}"
        )
        source = f"historical_tle_snapshot={snapshot_path} sha256={input_sha256}"
        write_pair_results(
            temporary_dir / "conjunction_dataset.csv", rows, source=source, config_summary=config_summary
        )
        write_pair_results(
            temporary_dir / "identified_conjunctions.csv",
            conjunctions,
            source=source,
            config_summary=config_summary,
        )
        report = {
            "status": "complete",
            "collection_id": collection_id,
            "input_snapshot": str(snapshot_path),
            "input_sha256": input_sha256,
            "run_fingerprint": run_fingerprint,
            "fingerprint_inputs": fingerprint_inputs,
            "input_provenance": metadata,
            "snapshot_utc": start_utc.isoformat().replace("+00:00", "Z"),
            "generated_utc": generated_utc(),
            "git_commit": git_commit_hash(),
            "experiment_config": str(config.path),
            "object_count": len(objects),
            "input_object_count": len(all_objects),
            "max_objects": config.max_objects,
            "screening": screening,
            "refined_rows": len(rows),
            "exact_candidates": len(conjunctions),
            "output_sha256": {
                "conjunction_dataset.csv": _sha256(temporary_dir / "conjunction_dataset.csv"),
                "identified_conjunctions.csv": _sha256(temporary_dir / "identified_conjunctions.csv"),
            },
        }
        (temporary_dir / "resimulation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_dir, final_dir)
        return report
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(description="Re-simulate immutable historical TLE snapshots")
    parser.add_argument("--config", default=str(config.path))
    parser.add_argument("--snapshot-dir", type=Path, default=Path(config.snapshot_dir))
    parser.add_argument("--output-run-root", type=Path, default=Path(config.resimulated_runs))
    parser.add_argument(
        "--history", type=Path, default=Path(config.resimulated_history)
    )
    parser.add_argument(
        "--report", type=Path, default=Path(config.resimulation_report)
    )
    parser.add_argument("--pattern", default="tles_*.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "independent snapshot workers; completed fingerprinted runs are reused "
            "on restart (default: 1)"
        ),
    )
    return parser.parse_args()


def _resimulate_worker(snapshot: Path, output_root: Path, config) -> dict:
    """Pickle-safe process worker for one immutable snapshot."""
    return resimulate_snapshot(snapshot, output_root, config)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    snapshots = sorted(args.snapshot_dir.glob(args.pattern))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        snapshots = snapshots[: args.limit]
    if not snapshots:
        raise ValueError(f"No snapshots match {args.snapshot_dir / args.pattern}")
    workers = int(getattr(args, "workers", 1))
    if workers <= 0:
        raise ValueError("--workers must be positive")
    subset_requested = args.limit is not None or args.pattern != "tles_*.txt"
    if subset_requested and args.history == Path(config.resimulated_history):
        raise ValueError("Subset resimulation requires an explicit non-canonical --history path")

    completed: list[dict] = []
    failures: list[dict] = []
    if workers == 1:
        for snapshot in snapshots:
            try:
                result = resimulate_snapshot(snapshot, args.output_run_root, config)
                completed.append(result)
                print(f"{result['status']} -> {snapshot.name}")
            except Exception as exc:
                failures.append({"snapshot": str(snapshot), "error": str(exc)})
                print(f"FAILED -> {snapshot.name}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_snapshot = {
                executor.submit(
                    _resimulate_worker, snapshot, args.output_run_root, config
                ): snapshot
                for snapshot in snapshots
            }
            for future in as_completed(future_to_snapshot):
                snapshot = future_to_snapshot[future]
                try:
                    result = future.result()
                    completed.append(result)
                    print(f"{result['status']} -> {snapshot.name}", flush=True)
                except Exception as exc:
                    failures.append({"snapshot": str(snapshot), "error": str(exc)})
                    print(f"FAILED -> {snapshot.name}: {exc}", flush=True)
    completed.sort(key=lambda item: str(item.get("collection_id", "")))
    failures.sort(key=lambda item: str(item.get("snapshot", "")))

    history_report = None
    if not failures and completed:
        requested_ids = {snapshot.stem.removeprefix("tles_") for snapshot in snapshots}
        history_report = rebuild_latest_schema_history(
            args.output_run_root,
            args.snapshot_dir,
            args.history,
            candidate_threshold_km=config.candidate_threshold_km,
            fixed_threshold_km=config.fixed_threshold_km,
            label_threshold_km=config.label_threshold_km,
            label_relative_velocity_km_s=config.label_relative_velocity_km_s,
            included_collection_ids=requested_ids,
            catalog_version=config.catalog_version,
            catalog_sha256=config.catalog_sha256 or None,
            require_resimulation_integrity=True,
        )
    report = {
        "schema_version": 2,
        "generated_utc": generated_utc(),
        "git_commit": git_commit_hash(),
        "experiment_config": str(config.path),
        "snapshots_requested": len(snapshots),
        "workers": workers,
        "completed": completed,
        "failures": failures,
        "history": history_report,
    }
    write_json_atomic(args.report, report)
    print(json.dumps({"completed": len(completed), "failures": len(failures)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
