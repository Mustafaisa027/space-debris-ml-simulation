"""Unit tests for collect_observations.py: resilience and provenance.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def test_main_tolerates_a_single_collect_once_failure_with_once_flag(monkeypatch):
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

    collect_observations.main()  # must not raise

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

    def fake_fetch_to_file(output, catnrs, groups, max_objects=None, request_delay=1.5):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(demo_tle_text, encoding="utf-8")
        return output

    monkeypatch.setattr(collect_observations, "fetch_to_file", fake_fetch_to_file)

    args = _args(
        days=60.0,
        interval_hours=2.0,
        once=True,
        max_objects=None,
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

    history_path = tmp_path / "history" / "conjunction_observations.csv"
    assert history_path.exists()
    header = history_path.read_text(encoding="utf-8").splitlines()[0]
    assert "source" in header and "preset" in header and "fetched_utc" in header
