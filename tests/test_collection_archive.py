from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fetch_tles import PRESETS
from space_debris.archive import (
    ArchiveVerificationError,
    catalog_id_set_sha256,
    file_sha256,
    import_collection_archive,
    verify_clean_archive_checkout,
    verify_collection_bundle,
)
from space_debris.experiment import load_experiment_config


def _with_checksum(line: str) -> str:
    body = line[:68].ljust(68)
    checksum = sum(int(character) for character in body if character.isdigit())
    checksum += body.count("-")
    return body + str(checksum % 10)


def _tle_payload(ids: list[str]) -> bytes:
    blocks = []
    for catalog_id in ids:
        blocks.extend(
            [
                f"TEST {catalog_id}",
                _with_checksum(
                    f"1 {catalog_id:>5}U 20001A   26001.00000000  .00000000  00000-0  00000-0 0  9990"
                ),
                _with_checksum(
                    f"2 {catalog_id:>5}  51.6400 000.0000 0001000   0.0000   0.0000 15.50000000  1000"
                ),
            ]
        )
    return ("\n".join(blocks) + "\n").encode("utf-8")


def _test_config(tmp_path: Path):
    ids = ["10001", "10002"]
    config = replace(
        load_experiment_config(),
        catalog_version="test-frozen-2-v1",
        catalog_sha256=catalog_id_set_sha256(ids),
        max_objects=2,
        preset="test_preset",
        history=str(tmp_path / "canonical_history.csv"),
    )
    return config, ids


def _write_bundle(
    archive_root: Path,
    config,
    ids: list[str],
    *,
    run_id: str = "42",
    attempt: str = "1",
    collection_id: str = "20260101_000000",
    schema_version: int = 1,
) -> Path:
    bundle = (
        archive_root
        / config.archive_collections
        / f"github-run-{run_id}-attempt-{attempt}"
    )
    files = {
        f"history/{Path(config.history).name}": b"history\n",
        f"runs/{collection_id}/conjunction_dataset.csv": b"dataset\n",
        f"runs/{collection_id}/identified_conjunctions.csv": b"identified\n",
        f"tle/tles_{collection_id}.txt": _tle_payload(ids),
    }
    sidecar = {
        "catalog_ids": ids,
        "catalog_sha256": config.catalog_sha256,
        "catalog_version": config.catalog_version,
        "collection_id": collection_id,
        "object_count": config.max_objects,
        "requested_object_count": config.max_objects,
        "preset": config.preset,
        "git_commit": "a" * 7,
        "fetched_utc": datetime.strptime(
            collection_id, "%Y%m%d_%H%M%S"
        ).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if schema_version == 3:
        sidecar.update(
            {
                "experiment_id": config.experiment_id,
                "experiment_config_sha256": config.config_sha256,
                "simulation": {
                    "horizon_minutes": config.horizon_minutes,
                    "step_minutes": config.step_minutes,
                    "screening_step_seconds": config.screening_step_seconds,
                    "candidate_threshold_km": config.candidate_threshold_km,
                    "fixed_threshold_km": config.fixed_threshold_km,
                    "label_threshold_km": config.label_threshold_km,
                    "label_relative_velocity_km_s": config.label_relative_velocity_km_s,
                    "max_tle_age_hours": config.max_tle_age_hours,
                    "leo_min_altitude_km": config.leo_min_altitude_km,
                    "leo_max_altitude_km": config.leo_max_altitude_km,
                },
            }
        )
        files["experiment/config.json"] = config.path.read_bytes()
    files[f"tle/tles_{collection_id}.json"] = (
        json.dumps(sidecar, sort_keys=True) + "\n"
    ).encode("utf-8")
    if schema_version in {2, 3}:
        runtime = {
            "schema_version": 1,
            "python": {"version": "3.12.0", "implementation": "CPython"},
            "platform": {"system": "Linux", "machine": "x86_64"},
            "packages": [{"name": "skyfield", "version": "1.54"}],
        }
        files["environment/runtime.json"] = (
            json.dumps(runtime, sort_keys=True) + "\n"
        ).encode("utf-8")
    for relative, content in files.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    entries = [
        {
            "path": relative,
            "bytes": (bundle / relative).stat().st_size,
            "sha256": file_sha256(bundle / relative),
        }
        for relative in sorted(files)
    ]
    manifest = {
        "schema_version": schema_version,
        "generated_utc": "2026-01-01T00:00:00Z",
        "git_commit": "a" * 40,
        "github_run_id": run_id,
        "github_run_attempt": attempt,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    if schema_version == 3:
        manifest.update(
            {
                "experiment_id": config.experiment_id,
                "experiment_config_path": "experiment/config.json",
                "experiment_config_sha256": config.config_sha256,
            }
        )
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return bundle


def _commit_archive(archive: Path) -> str:
    subprocess.run(["git", "init", str(archive)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(archive), "config", "user.name", "Archive Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(archive), "config", "user.email", "archive@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(archive), "add", "collections"], check=True)
    subprocess.run(
        ["git", "-C", str(archive), "commit", "-m", "test archive"],
        check=True,
        capture_output=True,
    )
    return verify_clean_archive_checkout(archive)


def test_archive_import_verifies_and_is_idempotent(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids)
    snapshots = tmp_path / "snapshots"
    runs = tmp_path / "runs"

    verified = verify_collection_bundle(bundle, config)
    first = import_collection_archive(archive, snapshots, runs, config)
    second = import_collection_archive(archive, snapshots, runs, config)

    assert verified.collection_id == "20260101_000000"
    assert verified.runtime_provenance_status == "legacy_runtime_missing"
    assert verified.manifest_schema_version == 1
    assert first["bundles_verified"] == 1
    assert first["clean_checkout_verified"] is False
    assert first["bundles"][0]["input_snapshot_sha256"] == file_sha256(
        bundle / "tle" / "tles_20260101_000000.txt"
    )
    assert first["bundles"][0]["input_sidecar_sha256"] == file_sha256(
        bundle / "tle" / "tles_20260101_000000.json"
    )
    assert first["files_planned"] == 4
    assert first["files_imported"] == 4
    assert first["files_already_identical"] == 0
    assert second["files_imported"] == 0
    assert second["files_already_identical"] == 4
    assert (snapshots / "tles_20260101_000000.txt").read_bytes() == (
        bundle / "tle" / "tles_20260101_000000.txt"
    ).read_bytes()
    assert (snapshots / "tles_20260101_000000.json").is_file()
    assert (runs / "20260101_000000" / "conjunction_dataset.csv").is_file()
    assert not (snapshots / Path(config.history).name).exists()


def test_archive_import_verify_only_does_not_create_destinations(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    _write_bundle(archive, config, ids)
    snapshots = tmp_path / "snapshots"
    runs = tmp_path / "runs"

    report = import_collection_archive(
        archive, snapshots, runs, config, verify_only=True
    )

    assert report["files_planned"] == 4
    assert report["files_imported"] == 0
    assert not snapshots.exists()
    assert not runs.exists()


def test_archive_rejects_tampering_before_copy(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids)
    (bundle / "tle" / "tles_20260101_000000.txt").write_text("tampered", encoding="utf-8")
    snapshots = tmp_path / "snapshots"

    with pytest.raises(ArchiveVerificationError, match="Manifest mismatch"):
        import_collection_archive(archive, snapshots, tmp_path / "runs", config)

    assert not snapshots.exists()


def test_archive_rejects_conflicting_destination_before_any_copy(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    _write_bundle(archive, config, ids)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "tles_20260101_000000.json").write_text("conflict", encoding="utf-8")
    runs = tmp_path / "runs"

    with pytest.raises(ArchiveVerificationError, match="Refusing to overwrite"):
        import_collection_archive(archive, snapshots, runs, config)

    assert not (snapshots / "tles_20260101_000000.txt").exists()
    assert not runs.exists()


def test_archive_rejects_manifest_path_traversal(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../outside.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveVerificationError, match="Unsafe manifest path"):
        verify_collection_bundle(bundle, config)


def test_archive_rejects_duplicate_collection_ids_across_bundles(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    _write_bundle(archive, config, ids, run_id="42")
    _write_bundle(archive, config, ids, run_id="43")

    with pytest.raises(ArchiveVerificationError, match="Duplicate collection_id"):
        import_collection_archive(archive, tmp_path / "snapshots", tmp_path / "runs", config)


def test_archive_rejects_wrong_frozen_catalog_hash(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids)
    sidecar_path = bundle / "tle" / "tles_20260101_000000.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["catalog_ids"] = ["10001", "99999"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"].endswith(".json"):
            entry["bytes"] = sidecar_path.stat().st_size
            entry["sha256"] = file_sha256(sidecar_path)
    manifest["total_bytes"] = sum(entry["bytes"] for entry in manifest["files"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveVerificationError, match="Frozen cohort mismatch"):
        verify_collection_bundle(bundle, config)


def test_archive_rejects_tle_bytes_that_disagree_with_sidecar_catalog(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids)
    tle_path = bundle / "tle" / "tles_20260101_000000.txt"
    tle_path.write_bytes(_tle_payload([ids[0], "99999"]))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"].endswith(".txt"):
            entry["bytes"] = tle_path.stat().st_size
            entry["sha256"] = file_sha256(tle_path)
    manifest["total_bytes"] = sum(entry["bytes"] for entry in manifest["files"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveVerificationError, match="frozen catalogue ID set"):
        verify_collection_bundle(bundle, config)


def test_archive_schema_two_requires_runtime_environment(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveVerificationError, match="requires environment/runtime.json"):
        verify_collection_bundle(bundle, config)


def test_schema_three_binds_exact_experiment_and_simulation(tmp_path):
    config = load_experiment_config("config/experiment_10_days_v2.json")
    ids = list(PRESETS[config.preset])
    bundle = _write_bundle(
        tmp_path,
        config,
        ids,
        collection_id="20260724_002200",
        schema_version=3,
    )

    verified = verify_collection_bundle(bundle, config)
    assert verified.manifest_schema_version == 3
    assert verified.runtime_provenance_status == "verified"

    changed = replace(config, fixed_threshold_km=26.0)
    with pytest.raises(ArchiveVerificationError, match="simulation"):
        verify_collection_bundle(bundle, changed)


def test_v2_archive_scope_ignores_v1_root_collections(tmp_path):
    v1, ids = _test_config(tmp_path)
    _write_bundle(tmp_path, v1, ids)
    v2 = load_experiment_config("config/experiment_10_days_v2.json")

    with pytest.raises(ArchiveVerificationError, match="No GitHub collection bundles"):
        import_collection_archive(
            tmp_path,
            tmp_path / "snapshots",
            tmp_path / "runs",
            v2,
        )


def test_schema_three_rejects_two_bundles_in_one_frozen_slot(tmp_path):
    config = load_experiment_config("config/experiment_10_days_v2.json")
    ids = list(PRESETS[config.preset])
    _write_bundle(
        tmp_path,
        config,
        ids,
        run_id="42",
        collection_id="20260724_002200",
        schema_version=3,
    )
    _write_bundle(
        tmp_path,
        config,
        ids,
        run_id="43",
        collection_id="20260724_004000",
        schema_version=3,
    )

    with pytest.raises(ArchiveVerificationError, match="Duplicate frozen slot=0"):
        import_collection_archive(
            tmp_path,
            tmp_path / "snapshots",
            tmp_path / "runs",
            config,
        )


def test_archive_schema_two_runtime_happy_path(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids, schema_version=2)

    verified = verify_collection_bundle(bundle, config)
    report = import_collection_archive(
        archive,
        tmp_path / "snapshots",
        tmp_path / "runs",
        config,
        verify_only=True,
    )

    assert verified.runtime_provenance_status == "verified"
    assert report["bundles"][0]["runtime_provenance_status"] == "verified"
    assert report["bundles"][0]["manifest_schema_version"] == 2
    assert report["files_planned"] == 4


def test_archive_import_rejects_dirty_checkout(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    bundle = _write_bundle(archive, config, ids, schema_version=2)
    revision = _commit_archive(archive)
    (bundle / "manifest.json").write_text("locally modified\n", encoding="utf-8")

    with pytest.raises(ArchiveVerificationError, match="archive_root is not clean"):
        import_collection_archive(
            archive,
            tmp_path / "snapshots",
            tmp_path / "runs",
            config,
            verify_only=True,
            archive_revision=revision,
            require_clean_checkout=True,
        )


def test_archive_import_records_clean_checkout_revision(tmp_path):
    config, ids = _test_config(tmp_path)
    archive = tmp_path / "archive"
    _write_bundle(archive, config, ids, schema_version=2)
    revision = _commit_archive(archive)

    report = import_collection_archive(
        archive,
        tmp_path / "snapshots",
        tmp_path / "runs",
        config,
        verify_only=True,
        require_clean_checkout=True,
    )

    assert report["archive_revision"] == revision
    assert report["clean_checkout_verified"] is True
