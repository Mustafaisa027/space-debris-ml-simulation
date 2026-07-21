from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import resimulate_snapshots
from space_debris.experiment import load_experiment_config
from space_debris.archive import catalog_id_set_sha256


def _resimulation_config_and_sidecar(snapshot):
    ids = ["10001", "10002"]
    config = replace(
        load_experiment_config("config/experiment_60_days.json"),
        catalog_version="test-frozen-2-v1",
        catalog_sha256=catalog_id_set_sha256(ids),
        max_objects=2,
        preset="test_preset",
    )
    sidecar = {
        "fetched_utc": "2026-07-02T15:04:45Z",
        "catalog_version": config.catalog_version,
        "catalog_sha256": config.catalog_sha256,
        "catalog_ids": ids,
        "object_count": config.max_objects,
        "requested_object_count": config.max_objects,
        "preset": config.preset,
    }
    snapshot.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    objects = [
        SimpleNamespace(line1=f"1 {catalog_id:>5}") for catalog_id in ids
    ]
    return config, objects


def test_snapshot_utc_prefers_provenance_and_falls_back_to_filename(tmp_path):
    snapshot = tmp_path / "tles_20260702_150445.txt"
    snapshot.write_text("x", encoding="utf-8")

    assert resimulate_snapshots._snapshot_utc(snapshot, {}) == datetime(
        2026, 7, 2, 15, 4, 45, tzinfo=timezone.utc
    )
    assert resimulate_snapshots._snapshot_utc(
        snapshot, {"fetched_utc": "2026-07-02T16:00:00Z"}
    ) == datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc)


def test_resimulation_is_atomic_and_resumable(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / "tles_20260702_150445.txt"
    snapshot.parent.mkdir()
    snapshot.write_text("valid-placeholder", encoding="utf-8")
    config, objects = _resimulation_config_and_sidecar(snapshot)
    monkeypatch.setattr(resimulate_snapshots, "read_tles", lambda _path: objects)
    monkeypatch.setattr(resimulate_snapshots, "build_satellites", lambda objects: objects)
    monkeypatch.setattr(resimulate_snapshots, "simulate_pairs", lambda **kwargs: [])
    monkeypatch.setattr(resimulate_snapshots, "filter_conjunctions", lambda rows, threshold: [])
    output_root = tmp_path / "resimulated"

    first = resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)
    second = resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)

    final_dir = output_root / "20260702_150445"
    assert first["status"] == "complete"
    assert second["status"] == "skipped_complete"
    assert (final_dir / "conjunction_dataset.csv").exists()
    assert (final_dir / "identified_conjunctions.csv").exists()
    assert not list(output_root.glob(".20260702_150445.*"))


def test_resimulation_refuses_changed_input_for_completed_run(monkeypatch, tmp_path):
    snapshot = tmp_path / "tles_20260702_150445.txt"
    snapshot.write_text("first", encoding="utf-8")
    output_root = tmp_path / "runs"
    final_dir = output_root / "20260702_150445"
    final_dir.mkdir(parents=True)
    (final_dir / "resimulation.json").write_text(
        json.dumps({"input_sha256": "different"}), encoding="utf-8"
    )
    config, _objects = _resimulation_config_and_sidecar(snapshot)

    with pytest.raises(ValueError, match="fingerprint/output integrity differs"):
        resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)


def test_resimulation_fingerprint_detects_sidecar_epoch_change(monkeypatch, tmp_path):
    snapshot = tmp_path / "tles_20260702_150445.txt"
    snapshot.write_text("valid-placeholder", encoding="utf-8")
    config, objects = _resimulation_config_and_sidecar(snapshot)
    sidecar = snapshot.with_suffix(".json")
    monkeypatch.setattr(resimulate_snapshots, "read_tles", lambda _path: objects)
    monkeypatch.setattr(resimulate_snapshots, "build_satellites", lambda objects: objects)
    monkeypatch.setattr(resimulate_snapshots, "simulate_pairs", lambda **kwargs: [])
    monkeypatch.setattr(resimulate_snapshots, "filter_conjunctions", lambda rows, threshold: [])
    output_root = tmp_path / "runs"
    resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)
    changed = json.loads(sidecar.read_text(encoding="utf-8"))
    changed["fetched_utc"] = "2026-07-02T16:04:45Z"
    sidecar.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint/output integrity differs"):
        resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)


def test_resimulation_rejects_actual_tle_ids_outside_frozen_cohort(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "tles_20260702_150445.txt"
    snapshot.write_text("valid-placeholder", encoding="utf-8")
    config, _objects = _resimulation_config_and_sidecar(snapshot)
    wrong_objects = [
        SimpleNamespace(line1="1 10001"),
        SimpleNamespace(line1="1 99999"),
    ]
    monkeypatch.setattr(resimulate_snapshots, "read_tles", lambda _path: wrong_objects)

    with pytest.raises(ValueError, match="frozen catalogue mismatch"):
        resimulate_snapshots.resimulate_snapshot(snapshot, tmp_path / "runs", config)


def test_resimulation_fingerprint_records_numerical_runtime(monkeypatch, tmp_path):
    snapshot = tmp_path / "tles_20260702_150445.txt"
    snapshot.write_text("valid-placeholder", encoding="utf-8")
    config, objects = _resimulation_config_and_sidecar(snapshot)
    monkeypatch.setattr(resimulate_snapshots, "read_tles", lambda _path: objects)
    monkeypatch.setattr(resimulate_snapshots, "build_satellites", lambda values: values)
    monkeypatch.setattr(resimulate_snapshots, "simulate_pairs", lambda **kwargs: [])
    monkeypatch.setattr(resimulate_snapshots, "filter_conjunctions", lambda rows, threshold: [])

    result = resimulate_snapshots.resimulate_snapshot(
        snapshot, tmp_path / "runs", config
    )

    runtime = result["fingerprint_inputs"]["runtime_environment"]
    assert runtime["schema_version"] == 1
    assert runtime["python"]["version"]
    assert isinstance(runtime["packages"], list)


def test_failed_batch_does_not_publish_canonical_history(monkeypatch, tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "tles_20260702_150445.txt").write_text("x", encoding="utf-8")
    history = tmp_path / "history.csv"
    report = tmp_path / "report.json"
    args = Namespace(
        config="config/experiment_60_days.json",
        snapshot_dir=snapshot_dir,
        output_run_root=tmp_path / "runs",
        history=history,
        report=report,
        pattern="tles_*.txt",
        limit=None,
    )
    monkeypatch.setattr(resimulate_snapshots, "parse_args", lambda: args)
    monkeypatch.setattr(
        resimulate_snapshots,
        "resimulate_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    rebuild_called = {"value": False}
    monkeypatch.setattr(
        resimulate_snapshots,
        "rebuild_latest_schema_history",
        lambda *a, **k: rebuild_called.update(value=True),
    )

    with pytest.raises(SystemExit):
        resimulate_snapshots.main()

    assert rebuild_called["value"] is False
    assert not history.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["failures"]
