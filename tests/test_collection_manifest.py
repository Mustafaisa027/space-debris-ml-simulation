from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build_collection_manifest import build_manifest, main


def _write_runtime(root):
    path = root / "environment" / "runtime.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": {"version": "3.12.0"},
                "platform": {"system": "test"},
                "packages": [],
            }
        ),
        encoding="utf-8",
    )


def test_build_manifest_records_relative_paths_sizes_and_hashes(tmp_path, monkeypatch):
    _write_runtime(tmp_path)
    (tmp_path / "nested").mkdir()
    payload = b"tle-data\n"
    (tmp_path / "nested" / "snapshot.txt").write_bytes(payload)
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")

    manifest = build_manifest(tmp_path)

    assert manifest["git_commit"] == "abc123"
    assert manifest["github_run_id"] == "42"
    assert manifest["schema_version"] == 2
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] >= len(payload)
    assert next(item for item in manifest["files"] if item["path"] == "nested/snapshot.txt") == {
        "path": "nested/snapshot.txt",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_main_writes_manifest_without_hashing_an_old_manifest(tmp_path, monkeypatch):
    _write_runtime(tmp_path)
    (tmp_path / "result.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("old", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["build_collection_manifest.py", str(tmp_path)])

    main()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["file_count"] == 2
    assert {item["path"] for item in manifest["files"]} == {
        "environment/runtime.json",
        "result.csv",
    }


def test_build_manifest_fails_closed_without_runtime_environment(tmp_path):
    (tmp_path / "result.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema-2 manifests require"):
        build_manifest(tmp_path)


def test_schema_three_manifest_binds_embedded_experiment_config(tmp_path):
    _write_runtime(tmp_path)
    source = Path("config/experiment_10_days_v2.json")
    embedded = tmp_path / "experiment" / "config.json"
    embedded.parent.mkdir()
    embedded.write_bytes(source.read_bytes())

    manifest = build_manifest(tmp_path, source)

    assert manifest["schema_version"] == 3
    assert manifest["experiment_id"] == "iac26-10d-v2"
    assert manifest["experiment_config_path"] == "experiment/config.json"
    assert len(manifest["experiment_config_sha256"]) == 64
    assert "experiment/config.json" in {
        item["path"] for item in manifest["files"]
    }


def test_collection_workflow_creates_archive_parent_before_first_bundle():
    workflow = Path(".github/workflows/collect_observations.yml").read_text(
        encoding="utf-8"
    )

    parent_creation = 'mkdir -p "$(dirname "$destination")"'
    immutable_destination = 'mkdir "$destination"'
    assert parent_creation in workflow
    assert immutable_destination in workflow
    assert workflow.index(parent_creation) < workflow.index(immutable_destination)
    assert workflow.index("actions/setup-python@v5") < workflow.index(
        "Check frozen collection slot"
    )
    assert 'cron: "17 */2 * * *"' in workflow
    assert 'cron: "47 */2 * * *"' in workflow
    assert "src/collection_slot_guard.py archive-check" in workflow
    assert "experiments/iac26-10d-v2/collections" in workflow
    assert "'experiments/** -text'" in workflow
    assert (
        "github.event_name != 'workflow_dispatch' || github.ref == 'refs/heads/main'"
        in workflow
    )


def test_cadence_workflow_checks_separate_data_archive():
    workflow = Path(".github/workflows/collection_cadence_health.yml").read_text(
        encoding="utf-8"
    )

    assert "ref: data-collection" in workflow
    assert "path: archive" in workflow
    assert "src/collection_cadence_health.py ../archive" in workflow
    assert "config/experiment_10_days_v2.json" in workflow
    assert 'cron: "07 1-23/2 * * *"' in workflow
    assert "Verify recent schema-3 bundle" in workflow


def test_finalizer_is_window_locked_and_orders_the_evidence_chain():
    script = Path("scripts/finalize_iac_experiment.ps1").read_text(encoding="utf-8")
    window_check = 'src\\collection_window_status.py'
    archive_import = 'src\\import_collection_archive.py'
    resimulation = 'src\\resimulate_snapshots.py'
    training = 'src\\train_from_history.py'

    assert '$window.status -ne "window_complete"' in script
    assert "--workers `$Workers" not in script
    assert "--workers $Workers" in script
    assert "config\\experiment_10_days_v2.json" in script
    assert "src\\finalization_checkpoint.py" in script
    assert "Test-CheckpointStage \"import\"" in script
    assert "Test-CheckpointStage \"resimulation\"" in script
    assert script.index(window_check) < script.index(archive_import)
    assert script.index(archive_import) < script.index(resimulation)
    assert script.index(resimulation) < script.index(training)
