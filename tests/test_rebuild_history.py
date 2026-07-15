from __future__ import annotations

import csv

from rebuild_history import rebuild_latest_schema_history


def _write_dataset(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# source: unit-test\n" + ",".join(header) + "\n" + ",".join(row) + "\n",
        encoding="utf-8",
    )


def test_rebuild_uses_latest_schema_and_reports_legacy_runs(tmp_path):
    run_root = tmp_path / "runs"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    base = ["snapshot_utc", "object_1", "object_2", "min_distance_km"]
    latest = base + ["relative_velocity_km_s"]
    _write_dataset(run_root / "old" / "conjunction_dataset.csv", base, ["2026-01-01Z", "A", "B", "10"])
    _write_dataset(
        run_root / "new" / "conjunction_dataset.csv",
        latest,
        ["2026-01-02Z", "A", "C", "20", "12"],
    )
    _write_dataset(
        run_root / "attached" / "conjunction_dataset.csv",
        latest,
        ["2026-01-03Z", "CSS (TIANHE)", "CSS (WENTIAN)", "1", "0.001"],
    )
    output = tmp_path / "history.csv"

    report = rebuild_latest_schema_history(run_root, snapshots, output)

    assert report["included_runs"] == ["attached", "new"]
    assert report["skipped_incompatible_runs"] == ["old"]
    assert report["filtered_non_independent_rows"] == 1
    assert report["rows_written"] == 1
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["collection_id"] == "new"
    assert rows[0]["relative_velocity_km_s"] == "12"
