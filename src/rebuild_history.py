from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.provenance import write_json_atomic


METADATA_COLUMNS = [
    "collection_id",
    "source",
    "preset",
    "catalog_version",
    "catalog_sha256",
    "fetched_utc",
    "tle_file",
    "object_count",
]

# These historical catalogue entries were attached to, docked with, or part of
# ISS/CSS during collection. Treating their metre-scale separation from the
# parent station as an independent conjunction creates false training labels.
NON_INDEPENDENT_OBJECT_NAMES = {
    "CSS (MENGTIAN)",
    "CSS (WENTIAN)",
    "CREW DRAGON 12",
    "CYGNUS NG-24",
    "ISS (NAUKA)",
    "POISK",
    "PROGRESS-MS 33",
}

ADDITIVE_SCHEMA_DEFAULTS = {
    "tca_boundary_flag": "",
    "object_1_catalog_id": "",
    "object_2_catalog_id": "",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(path: Path) -> tuple[list[str], list[dict], dict[str, str]]:
    comments: dict[str, str] = {}
    data_lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                key, separator, value = line[1:].partition(":")
                if separator:
                    comments[key.strip()] = value.strip()
            else:
                data_lines.append(line)
    reader = csv.DictReader(data_lines)
    rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"Dataset has no header: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"Dataset has malformed rows wider than its header: {path}")
    return list(reader.fieldnames), rows, comments


def rebuild_latest_schema_history(
    run_root: Path,
    snapshot_dir: Path,
    output: Path,
    *,
    candidate_threshold_km: float | None = None,
    fixed_threshold_km: float | None = None,
    label_threshold_km: float | None = None,
    label_relative_velocity_km_s: float | None = None,
    included_collection_ids: set[str] | None = None,
    catalog_version: str | None = None,
    catalog_sha256: str | None = None,
    require_resimulation_integrity: bool = False,
) -> dict:
    datasets = sorted(run_root.glob("*/conjunction_dataset.csv"))
    if included_collection_ids is not None:
        datasets = [path for path in datasets if path.parent.name in included_collection_ids]
    if not datasets:
        raise ValueError(f"No conjunction_dataset.csv files found below {run_root}")

    cohort_skipped: list[str] = []
    if catalog_version is not None:
        cohort_datasets: list[Path] = []
        for path in datasets:
            collection_id = path.parent.name
            sidecar = snapshot_dir / f"tles_{collection_id}.json"
            provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            if (
                provenance.get("catalog_version") == catalog_version
                and (catalog_sha256 is None or provenance.get("catalog_sha256") == catalog_sha256)
            ):
                cohort_datasets.append(path)
            else:
                cohort_skipped.append(collection_id)
        datasets = cohort_datasets
    if not datasets:
        raise ValueError(
            "No conjunction datasets match frozen catalogue cohort "
            f"version={catalog_version!r} sha256={catalog_sha256!r} below {run_root}"
        )

    if require_resimulation_integrity:
        for dataset in datasets:
            resimulation_path = dataset.parent / "resimulation.json"
            if not resimulation_path.is_file():
                raise ValueError(f"Missing resimulation integrity record: {resimulation_path}")
            try:
                resimulation = json.loads(resimulation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid resimulation integrity record: {resimulation_path}: {exc}"
                ) from exc
            expected_hash = resimulation.get("output_sha256", {}).get(
                "conjunction_dataset.csv"
            )
            if expected_hash != _sha256(dataset):
                raise ValueError(
                    f"Resimulation dataset hash mismatch before history rebuild: {dataset}"
                )

    inspected = [(path, *read_dataset(path)) for path in datasets]
    latest_columns = max((columns for _, columns, _, _ in inspected), key=len)
    def compatible_with_latest(columns: list[str]) -> bool:
        missing = [column for column in latest_columns if column not in columns]
        return (
            not any(column not in latest_columns for column in columns)
            and columns == [column for column in latest_columns if column in columns]
            and all(column in ADDITIVE_SCHEMA_DEFAULTS for column in missing)
        )

    compatible = [item for item in inspected if compatible_with_latest(item[1])]
    skipped = [
        path.parent.name
        for path, columns, _, _ in inspected
        if not compatible_with_latest(columns)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    additive_defaults = {
        column: default
        for column, default in ADDITIVE_SCHEMA_DEFAULTS.items()
        if column in latest_columns
    }
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    written = 0
    filtered_non_independent = 0
    filtered_non_candidates = 0
    positive_labels = 0
    fixed_alarms = 0
    resimulation_fingerprints: dict[str, str] = {}
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METADATA_COLUMNS + latest_columns)
            writer.writeheader()
            for path, _columns, rows, comments in compatible:
                collection_id = path.parent.name
                sidecar = snapshot_dir / f"tles_{collection_id}.json"
                provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                resimulation_path = path.parent / "resimulation.json"
                resimulation = (
                    json.loads(resimulation_path.read_text(encoding="utf-8"))
                    if resimulation_path.exists()
                    else {}
                )
                if resimulation.get("run_fingerprint"):
                    resimulation_fingerprints[collection_id] = resimulation["run_fingerprint"]
                object_names = {row["object_1"] for row in rows} | {row["object_2"] for row in rows}
                metadata = {
                    "collection_id": collection_id,
                    "source": (
                        comments.get("source", "unknown")
                        if resimulation
                        else provenance.get("source", comments.get("source", "unknown"))
                    ),
                    "preset": provenance.get("preset", ""),
                    "catalog_version": provenance.get("catalog_version", ""),
                    "catalog_sha256": provenance.get("catalog_sha256", ""),
                    "fetched_utc": resimulation.get(
                        "snapshot_utc",
                        provenance.get("fetched_utc", rows[0].get("snapshot_utc", "") if rows else ""),
                    ),
                    "tle_file": resimulation.get(
                        "input_snapshot",
                        provenance.get("tle_file", str(snapshot_dir / f"tles_{collection_id}.txt")),
                    ),
                    "object_count": provenance.get("object_count", len(object_names)),
                }
                for row in rows:
                    if {row.get("object_1", ""), row.get("object_2", "")} & NON_INDEPENDENT_OBJECT_NAMES:
                        filtered_non_independent += 1
                        continue
                    if (
                        candidate_threshold_km is not None
                        and float(row["min_distance_km"]) > candidate_threshold_km
                    ):
                        filtered_non_candidates += 1
                        continue
                    if fixed_threshold_km is not None:
                        row["fixed_threshold_alarm"] = str(
                            int(float(row["min_distance_km"]) <= fixed_threshold_km)
                        )
                    if label_threshold_km is not None and label_relative_velocity_km_s is not None:
                        row["risk_label"] = str(
                            int(
                                float(row["min_distance_km"]) <= label_threshold_km
                                and float(row["relative_velocity_km_s"])
                                >= label_relative_velocity_km_s
                            )
                        )
                    fixed_alarms += int(row.get("fixed_threshold_alarm", "0") or 0)
                    positive_labels += int(row.get("risk_label", "0") or 0)
                    writer.writerow({**additive_defaults, **metadata, **row})
                    written += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "output": str(output),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "schema_columns": latest_columns,
        "included_runs": [path.parent.name for path, *_ in compatible],
        "skipped_incompatible_runs": skipped,
        "skipped_catalog_version_runs": cohort_skipped,
        "catalog_version": catalog_version,
        "catalog_sha256": catalog_sha256,
        "filtered_non_independent_rows": filtered_non_independent,
        "filtered_non_candidate_rows": filtered_non_candidates,
        "rows_written": written,
        "candidate_threshold_km": candidate_threshold_km,
        "fixed_threshold_km": fixed_threshold_km,
        "label_threshold_km": label_threshold_km,
        "label_relative_velocity_km_s": label_relative_velocity_km_s,
        "fixed_alarms": fixed_alarms,
        "positive_labels": positive_labels,
        "resimulation_fingerprints": resimulation_fingerprints,
        "resimulation_integrity_required": require_resimulation_integrity,
        "resimulation_fingerprint_set_sha256": hashlib.sha256(
            json.dumps(
                resimulation_fingerprints, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(description="Rebuild a clean history from per-run datasets")
    parser.add_argument("--config", default=str(config.path))
    parser.add_argument("--run-root", type=Path, default=Path(config.run_root))
    parser.add_argument("--snapshot-dir", type=Path, default=Path(config.snapshot_dir))
    parser.add_argument("--output", type=Path, default=Path(config.history))
    parser.add_argument("--report", type=Path, default=Path(config.history_rebuild_report))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    report = rebuild_latest_schema_history(
        args.run_root,
        args.snapshot_dir,
        args.output,
        candidate_threshold_km=config.candidate_threshold_km,
        fixed_threshold_km=config.fixed_threshold_km,
        label_threshold_km=config.label_threshold_km,
        label_relative_velocity_km_s=config.label_relative_velocity_km_s,
        catalog_version=config.catalog_version,
        catalog_sha256=config.catalog_sha256 or None,
    )
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
