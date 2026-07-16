from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


METADATA_COLUMNS = [
    "collection_id",
    "source",
    "preset",
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
}


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
    if not reader.fieldnames or not rows:
        raise ValueError(f"Dataset is empty or has no header: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"Dataset has malformed rows wider than its header: {path}")
    return list(reader.fieldnames), rows, comments


def rebuild_latest_schema_history(run_root: Path, snapshot_dir: Path, output: Path) -> dict:
    datasets = sorted(run_root.glob("*/conjunction_dataset.csv"))
    if not datasets:
        raise ValueError(f"No conjunction_dataset.csv files found below {run_root}")

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
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METADATA_COLUMNS + latest_columns)
            writer.writeheader()
            for path, _columns, rows, comments in compatible:
                collection_id = path.parent.name
                sidecar = snapshot_dir / f"tles_{collection_id}.json"
                provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                object_names = {row["object_1"] for row in rows} | {row["object_2"] for row in rows}
                metadata = {
                    "collection_id": collection_id,
                    "source": provenance.get("source", comments.get("source", "unknown")),
                    "preset": provenance.get("preset", ""),
                    "fetched_utc": provenance.get("fetched_utc", rows[0].get("snapshot_utc", "")),
                    "tle_file": provenance.get("tle_file", str(snapshot_dir / f"tles_{collection_id}.txt")),
                    "object_count": provenance.get("object_count", len(object_names)),
                }
                for row in rows:
                    if {row.get("object_1", ""), row.get("object_2", "")} & NON_INDEPENDENT_OBJECT_NAMES:
                        filtered_non_independent += 1
                        continue
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
        "schema_columns": latest_columns,
        "included_runs": [path.parent.name for path, *_ in compatible],
        "skipped_incompatible_runs": skipped,
        "filtered_non_independent_rows": filtered_non_independent,
        "rows_written": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a clean history from per-run datasets")
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data/tle_snapshots"))
    parser.add_argument("--output", type=Path, default=Path("outputs/history/conjunction_observations_v2.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/history/history_rebuild_report.json"))
    args = parser.parse_args()
    report = rebuild_latest_schema_history(args.run_root, args.snapshot_dir, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
