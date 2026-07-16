from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone

import pytest

import resimulate_snapshots
from space_debris.experiment import load_experiment_config


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
    snapshot.with_suffix(".json").write_text(
        json.dumps({"fetched_utc": "2026-07-02T15:04:45Z"}), encoding="utf-8"
    )
    config = load_experiment_config("config/experiment_60_days.json")
    monkeypatch.setattr(resimulate_snapshots, "read_tles", lambda _path: [object(), object()])
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
    config = load_experiment_config("config/experiment_60_days.json")

    with pytest.raises(ValueError, match="fingerprint/output integrity differs"):
        resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)


def test_resimulation_fingerprint_detects_sidecar_epoch_change(monkeypatch, tmp_path):
    snapshot = tmp_path / "tles_20260702_150445.txt"
    snapshot.write_text("valid-placeholder", encoding="utf-8")
    sidecar = snapshot.with_suffix(".json")
    sidecar.write_text(json.dumps({"fetched_utc": "2026-07-02T15:04:45Z"}), encoding="utf-8")
    config = load_experiment_config("config/experiment_60_days.json")
    monkeypatch.setattr(resimulate_snapshots, "read_tles", lambda _path: [object(), object()])
    monkeypatch.setattr(resimulate_snapshots, "build_satellites", lambda objects: objects)
    monkeypatch.setattr(resimulate_snapshots, "simulate_pairs", lambda **kwargs: [])
    monkeypatch.setattr(resimulate_snapshots, "filter_conjunctions", lambda rows, threshold: [])
    output_root = tmp_path / "runs"
    resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)
    sidecar.write_text(json.dumps({"fetched_utc": "2026-07-02T16:04:45Z"}), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint/output integrity differs"):
        resimulate_snapshots.resimulate_snapshot(snapshot, output_root, config)


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
