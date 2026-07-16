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


def test_rebuild_treats_boundary_flag_as_additive_schema_column(tmp_path):
    run_root = tmp_path / "runs"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    base = ["snapshot_utc", "object_1", "object_2", "min_distance_km"]
    refined = base + ["tca_boundary_flag"]
    _write_dataset(run_root / "old" / "conjunction_dataset.csv", base, ["2026-01-01Z", "A", "B", "10"])
    _write_dataset(
        run_root / "new" / "conjunction_dataset.csv",
        refined,
        ["2026-01-02Z", "A", "C", "20", "1"],
    )
    output = tmp_path / "history.csv"

    report = rebuild_latest_schema_history(run_root, snapshots, output)

    assert report["included_runs"] == ["new", "old"]
    assert report["skipped_incompatible_runs"] == []
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["tca_boundary_flag"] == "1"
    assert rows[1]["tca_boundary_flag"] == ""


def test_rebuild_relabels_all_runs_with_one_experiment_threshold_set(tmp_path):
    run_root = tmp_path / "runs"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    columns = [
        "snapshot_utc", "object_1", "object_2", "min_distance_km",
        "relative_velocity_km_s", "fixed_threshold_alarm", "risk_label",
    ]
    _write_dataset(
        run_root / "old" / "conjunction_dataset.csv",
        columns,
        ["2026-01-01Z", "A", "B", "40", "12", "0", "0"],
    )
    output = tmp_path / "history.csv"

    report = rebuild_latest_schema_history(
        run_root,
        snapshots,
        output,
        candidate_threshold_km=50,
        fixed_threshold_km=25,
        label_threshold_km=50,
        label_relative_velocity_km_s=10,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["fixed_threshold_alarm"] == "0"
    assert row["risk_label"] == "1"
    assert report["fixed_alarms"] == 0
    assert report["positive_labels"] == 1
    assert report["filtered_non_candidate_rows"] == 0


def test_rebuild_history_excludes_pairs_outside_candidate_screen(tmp_path):
    run_root = tmp_path / "runs"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    columns = ["snapshot_utc", "object_1", "object_2", "min_distance_km"]
    _write_dataset(
        run_root / "run" / "conjunction_dataset.csv",
        columns,
        ["2026-01-01Z", "A", "B", "500"],
    )
    output = tmp_path / "history.csv"

    report = rebuild_latest_schema_history(
        run_root, snapshots, output, candidate_threshold_km=200
    )

    with output.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == []
    assert report["rows_written"] == 0
    assert report["filtered_non_candidate_rows"] == 1


def test_rebuild_accepts_header_only_zero_candidate_run(tmp_path):
    run_root = tmp_path / "runs"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    dataset = run_root / "empty" / "conjunction_dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        "snapshot_utc,object_1,object_2,min_distance_km\n", encoding="utf-8"
    )
    output = tmp_path / "history.csv"

    report = rebuild_latest_schema_history(run_root, snapshots, output)

    assert report["included_runs"] == ["empty"]
    assert report["rows_written"] == 0
    with output.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == []
