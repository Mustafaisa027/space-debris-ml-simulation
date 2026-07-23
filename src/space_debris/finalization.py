from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .experiment import ExperimentConfig


class FinalizationCheckpointError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_archive_state(archive_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(archive_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(archive_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalizationCheckpointError(
            f"Cannot inspect archive Git state: {archive_root}"
        ) from exc
    if dirty:
        raise FinalizationCheckpointError(
            "Finalization checkpoint requires a clean committed archive"
        )
    return revision, dirty


def build_input_binding(archive_root: Path, config: ExperimentConfig) -> dict:
    archive_root = archive_root.resolve()
    collections = archive_root / config.archive_collections
    manifests = sorted(collections.glob("github-run-*-attempt-*/manifest.json"))
    if not manifests:
        raise FinalizationCheckpointError(
            f"No experiment manifests below {collections}"
        )
    revision, _ = _git_archive_state(archive_root)
    manifest_hashes = {
        path.relative_to(archive_root).as_posix(): _sha256(path) for path in manifests
    }
    fingerprint_inputs = {
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.config_sha256,
        "archive_revision": revision,
        "manifest_count": len(manifest_hashes),
        "manifest_sha256": manifest_hashes,
    }
    binding = {"archive_root": str(archive_root), **fingerprint_inputs}
    binding["input_fingerprint_sha256"] = _canonical_sha256(fingerprint_inputs)
    return binding


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_checkpoint(
    checkpoint_path: Path, archive_root: Path, config: ExperimentConfig
) -> dict:
    binding = build_input_binding(archive_root, config)
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FinalizationCheckpointError(
                f"Cannot read finalization checkpoint {checkpoint_path}: {exc}"
            ) from exc
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("experiment_id") != config.experiment_id
            or checkpoint.get("input_fingerprint_sha256")
            != binding["input_fingerprint_sha256"]
        ):
            raise FinalizationCheckpointError(
                "Existing finalization checkpoint belongs to different inputs; "
                "use a new checkpoint path instead of reusing stale stages"
            )
        return checkpoint
    checkpoint = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "input_fingerprint_sha256": binding["input_fingerprint_sha256"],
        "inputs": binding,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stages": {},
    }
    _atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def _path_binding(path: Path) -> dict:
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "size": resolved.stat().st_size,
        }
    if resolved.is_dir():
        files = {
            child.relative_to(resolved).as_posix(): _sha256(child)
            for child in sorted(item for item in resolved.rglob("*") if item.is_file())
        }
        return {
            "kind": "directory",
            "path": str(resolved),
            "file_count": len(files),
            "tree_sha256": _canonical_sha256(files),
        }
    raise FinalizationCheckpointError(f"Stage output does not exist: {path}")


def output_bindings(paths: Iterable[Path]) -> list[dict]:
    return [_path_binding(Path(path)) for path in paths]


def stage_is_reusable(
    checkpoint_path: Path, stage: str, paths: Iterable[Path]
) -> bool:
    if not checkpoint_path.is_file():
        return False
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    recorded = checkpoint.get("stages", {}).get(stage)
    if not isinstance(recorded, dict):
        return False
    try:
        current = output_bindings(paths)
    except FinalizationCheckpointError:
        return False
    return recorded.get("outputs") == current


def record_stage(
    checkpoint_path: Path, stage: str, paths: Iterable[Path]
) -> dict:
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationCheckpointError(
            f"Cannot read finalization checkpoint {checkpoint_path}: {exc}"
        ) from exc
    stages = checkpoint.setdefault("stages", {})
    stages[stage] = {
        "completed_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "outputs": output_bindings(paths),
    }
    _atomic_json(checkpoint_path, checkpoint)
    return checkpoint
