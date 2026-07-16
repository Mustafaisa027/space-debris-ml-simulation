"""Unit tests for collect_observations.py: resilience and provenance.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

import collect_observations
import fetch_tles


def _args(**overrides) -> argparse.Namespace:
    base = dict(catnr=None, group=None, preset="leo_mixed")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_source_defaults_to_leo_mixed_preset():
    catnrs, groups, source = collect_observations.resolve_source(_args())
    assert source == "preset:leo_mixed"
    assert groups == []
    assert set(catnrs) == set(fetch_tles.PRESETS["leo_mixed"])


def test_resolve_source_explicit_catnr_overrides_preset():
    catnrs, groups, source = collect_observations.resolve_source(_args(catnr=["12345"]))
    assert source == "explicit"
    assert catnrs == ["12345"]
    assert groups == []


def test_resolve_source_explicit_group_overrides_preset():
    catnrs, groups, source = collect_observations.resolve_source(_args(group=["WEATHER"]))
    assert source == "explicit"
    assert catnrs == []
    assert groups == ["WEATHER"]


def test_main_reports_single_collect_once_failure_to_scheduler(monkeypatch):
    calls = {"n": 0}

    def fake_collect_once(args):
        calls["n"] += 1
        raise RuntimeError("simulated fetch failure")

    monkeypatch.setattr(collect_observations, "collect_once", fake_collect_once)
    monkeypatch.setattr(
        collect_observations,
        "parse_args",
        lambda: _args(days=60.0, interval_hours=2.0, once=True),
    )

    with pytest.raises(RuntimeError, match="simulated fetch failure"):
        collect_observations.main()

    assert calls["n"] == 1


def test_main_continues_after_repeated_failures_until_deadline(monkeypatch):
    calls = {"n": 0}

    def fake_collect_once(args):
        calls["n"] += 1
        raise RuntimeError(f"simulated failure #{calls['n']}")

    monkeypatch.setattr(collect_observations, "collect_once", fake_collect_once)
    monkeypatch.setattr(
        collect_observations,
        "parse_args",
        lambda: _args(days=0.001, interval_hours=2.0, once=False),
    )
    monkeypatch.setattr(collect_observations.time, "sleep", lambda _s: None)

    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 50.0
        return clock["t"]

    monkeypatch.setattr(collect_observations.time, "time", fake_time)

    collect_observations.main()  # must not raise despite every cycle failing

    assert calls["n"] >= 2


def test_collect_once_writes_provenance_sidecar(monkeypatch, tmp_path):
    from make_demo_tles import generate

    demo_blocks = generate(count=6, seed=1)
    demo_tle_text = "\n".join(demo_blocks) + "\n"

    def fake_fetch_to_file(
        output, catnrs, groups, max_objects=None, request_delay=1.5, *, provider="auto", report=None
    ):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(demo_tle_text, encoding="utf-8")
        if report is not None:
            report.update(provider="test", catalog_ids=[str(i) for i in range(6)], requested_count=6)
        return output

    monkeypatch.setattr(collect_observations, "fetch_to_file", fake_fetch_to_file)

    args = _args(
        days=60.0,
        interval_hours=2.0,
        once=True,
        max_objects=None,
        provider="auto",
        snapshot_dir=str(tmp_path / "snapshots"),
        run_root=str(tmp_path / "runs"),
        history=str(tmp_path / "history" / "conjunction_observations.csv"),
        horizon_minutes=30,
        step_minutes=10,
        leo_min_altitude_km=160.0,
        leo_max_altitude_km=2000.0,
        candidate_threshold_km=5000.0,
        fixed_threshold_km=25.0,
        label_threshold_km=200.0,
        label_relative_velocity_km_s=5.0,
        max_tle_age_hours=336.0,
    )

    collect_observations.collect_once(args)

    snapshot_dir = tmp_path / "snapshots"
    json_files = list(snapshot_dir.glob("*.json"))
    assert len(json_files) == 1

    provenance = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert provenance["source"] == "preset:leo_mixed"
    assert provenance["preset"] == "leo_mixed"
    assert provenance["object_count"] == 6
    assert "fetched_utc" in provenance and provenance["fetched_utc"].endswith("Z")
    assert "git_commit" in provenance
    assert provenance["tle_provider"] == "test"
    assert provenance["requested_object_count"] == 6

    history_path = tmp_path / "history" / "conjunction_observations.csv"
    assert history_path.exists()
    header = history_path.read_text(encoding="utf-8").splitlines()[0]
    assert "source" in header and "preset" in header and "fetched_utc" in header
    # write_pair_results() embeds a '#' provenance header in dataset_path
    # (ROADMAP_YOL1.md GOREV 6); append_csv must not have leaked a literal
    # '#' comment line into the accumulated history CSV.
    assert not header.startswith("#")


def test_append_csv_skips_provenance_header_lines_in_source(tmp_path):
    source_path = tmp_path / "conjunction_dataset.csv"
    source_path.write_text(
        "# generated_utc: 2026-07-09T00:00:00Z\n"
        "# git_commit: abc1234\n"
        "# source: leo_mixed preset\n"
        "# config: horizon=720min\n"
        "object_1,object_2,min_distance_km\n"
        "SAT-A,SAT-B,12.5\n"
        "SAT-C,SAT-D,88.0\n",
        encoding="utf-8",
    )
    target_path = tmp_path / "history.csv"

    count = collect_observations.append_csv(target_path, source_path, {"collection_id": "run1"})

    assert count == 2
    lines = target_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "collection_id,object_1,object_2,min_distance_km"
    assert not any(line.startswith("#") for line in lines)
    assert lines[1] == "run1,SAT-A,SAT-B,12.5"
    assert lines[2] == "run1,SAT-C,SAT-D,88.0"


def test_append_csv_rejects_schema_drift_without_modifying_history(tmp_path):
    target = tmp_path / "history.csv"
    target.write_text("collection_id,old_column\nold,1\n", encoding="utf-8")
    before = target.read_bytes()
    source = tmp_path / "dataset.csv"
    source.write_text("new_column\n2\n", encoding="utf-8")

    with pytest.raises(collect_observations.HistorySchemaError, match="schema mismatch"):
        collect_observations.append_csv(target, source, {"collection_id": "new"})

    assert target.read_bytes() == before


def test_append_csv_adds_refined_tca_flag_to_existing_history(tmp_path):
    target = tmp_path / "history.csv"
    target.write_text(
        "collection_id,object_1,object_2,min_distance_km\n"
        "old,SAT-A,SAT-B,12.5\n",
        encoding="utf-8",
    )
    source = tmp_path / "dataset.csv"
    source.write_text(
        "object_1,object_2,min_distance_km,tca_boundary_flag\n"
        "SAT-C,SAT-D,8.0,1\n",
        encoding="utf-8",
    )

    count = collect_observations.append_csv(target, source, {"collection_id": "new"})

    assert count == 1
    rows = list(csv.DictReader(target.open(newline="", encoding="utf-8")))
    assert rows[0]["tca_boundary_flag"] == ""
    assert rows[1]["tca_boundary_flag"] == "1"


def test_append_csv_rejects_malformed_existing_rows_without_modification(tmp_path):
    target = tmp_path / "history.csv"
    target.write_text("collection_id,value\nold,1,unexpected\n", encoding="utf-8")
    before = target.read_bytes()
    source = tmp_path / "dataset.csv"
    source.write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(collect_observations.HistorySchemaError, match="malformed rows"):
        collect_observations.append_csv(target, source, {"collection_id": "new"})

    assert target.read_bytes() == before
