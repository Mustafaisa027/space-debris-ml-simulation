from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from space_debris.experiment import load_experiment_config
from space_debris.finalization import (
    FinalizationCheckpointError,
    prepare_checkpoint,
    record_stage,
    stage_is_reusable,
)


def _archive(tmp_path: Path, config) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    subprocess.run(["git", "init", "-q", str(archive)], check=True)
    subprocess.run(["git", "-C", str(archive), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(archive), "config", "user.name", "Test"], check=True)
    bundle = archive / config.archive_collections / "github-run-1-attempt-1"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"schema_version":3}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(archive), "add", "."], check=True)
    subprocess.run(["git", "-C", str(archive), "commit", "-qm", "archive"], check=True)
    return archive


def test_checkpoint_reuses_only_unchanged_outputs(tmp_path):
    config = load_experiment_config("config/experiment_10_days_v2.json")
    archive = _archive(tmp_path, config)
    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "history.csv"
    output.write_text("a,b\n1,2\n", encoding="utf-8")

    prepare_checkpoint(checkpoint, archive, config)
    record_stage(checkpoint, "import", [output])
    assert stage_is_reusable(checkpoint, "import", [output])

    output.write_text("a,b\n3,4\n", encoding="utf-8")
    assert not stage_is_reusable(checkpoint, "import", [output])


def test_checkpoint_rejects_changed_archive_inputs(tmp_path):
    config = load_experiment_config("config/experiment_10_days_v2.json")
    archive = _archive(tmp_path, config)
    checkpoint = tmp_path / "checkpoint.json"
    prepare_checkpoint(checkpoint, archive, config)

    manifest = next(archive.rglob("manifest.json"))
    manifest.write_text('{"schema_version":3,"changed":true}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(archive), "add", "."], check=True)
    subprocess.run(["git", "-C", str(archive), "commit", "-qm", "change"], check=True)

    with pytest.raises(FinalizationCheckpointError, match="different inputs"):
        prepare_checkpoint(checkpoint, archive, config)


def test_checkpoint_rejects_dirty_archive(tmp_path):
    config = load_experiment_config("config/experiment_10_days_v2.json")
    archive = _archive(tmp_path, config)
    next(archive.rglob("manifest.json")).write_text("dirty\n", encoding="utf-8")

    with pytest.raises(FinalizationCheckpointError, match="clean committed"):
        prepare_checkpoint(tmp_path / "checkpoint.json", archive, config)
