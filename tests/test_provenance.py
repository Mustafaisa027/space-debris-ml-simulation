"""Unit tests for space_debris.provenance (ROADMAP_YOL1.md GOREV 6).

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from space_debris import provenance


def test_git_commit_hash_returns_nonempty_string():
    result = provenance.git_commit_hash()
    assert isinstance(result, str)
    assert result != ""


def test_git_commit_hash_falls_back_to_unknown_on_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert provenance.git_commit_hash() == "unknown"


def test_git_commit_full_hash_uses_unabbreviated_revision(monkeypatch):
    full = "a" * 40
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, full + "\n", ""),
    )

    assert provenance.git_commit_full_hash() == full


def test_git_worktree_state_records_dirty_flag_and_status_hash(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, " M src/example.py\n", ""
        ),
    )

    state = provenance.git_worktree_state()

    assert state["dirty"] is True
    assert len(state["status_sha256"]) == 64


def test_generated_utc_is_iso_with_z_suffix():
    stamp = provenance.generated_utc()
    assert stamp.endswith("Z")
    # Must round-trip through pandas' timestamp parser without error.
    pd.Timestamp(stamp)


def test_provenance_header_lines_contains_expected_keys():
    lines = provenance.provenance_header_lines(source="leo_mixed preset", config_summary="horizon=720min")
    assert len(lines) == 4
    assert all(line.startswith("#") for line in lines)
    joined = "\n".join(lines)
    assert "generated_utc:" in joined
    assert "git_commit:" in joined
    assert "source: leo_mixed preset" in joined
    assert "config: horizon=720min" in joined


def test_provenance_header_lines_defaults_when_unspecified():
    lines = provenance.provenance_header_lines()
    joined = "\n".join(lines)
    assert "source: unspecified" in joined
    assert "config: unspecified" in joined


def test_write_csv_text_with_provenance_is_readable_with_comment_flag(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "out.csv"

    provenance.write_csv_text_with_provenance(
        path, df.to_csv(index=False), source="unit-test", config_summary="n=3"
    )

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert raw_lines[0].startswith("#")
    assert raw_lines[1].startswith("#")

    round_tripped = pd.read_csv(path, comment="#")
    pd.testing.assert_frame_equal(round_tripped, df)


def test_png_provenance_metadata_embeds_source_and_config():
    metadata = provenance.png_provenance_metadata(source="demo.tle", config_summary="step=5min")
    assert metadata["Software"] == "space-debris-ml-simulation"
    assert "demo.tle" in metadata["Description"]
    assert "step=5min" in metadata["Description"]
    assert "git_commit=" in metadata["Description"]
