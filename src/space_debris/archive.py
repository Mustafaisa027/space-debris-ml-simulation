"""Verification and fail-closed import of GitHub collection bundles.

The scheduled collector stores one immutable bundle per workflow run on the
``data-collection`` branch.  This module is the trust boundary between that
nested archive and the flat local snapshot/run layout consumed by historical
re-simulation.  Nothing is copied until every selected bundle and manifest has
been verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

from space_debris.experiment import ExperimentConfig
from space_debris.tle_validation import validated_tle_catalog_ids
from space_debris.provenance import generated_utc, git_commit_full_hash, git_worktree_state


BUNDLE_NAME = re.compile(r"^github-run-(?P<run_id>[0-9]+)-attempt-(?P<attempt>[0-9]+)$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class ArchiveVerificationError(RuntimeError):
    """Raised when an archive cannot be trusted or imported unambiguously."""


def _run_archive_git(archive_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(archive_root), *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArchiveVerificationError(
            f"archive_root must be a readable Git checkout/worktree: {archive_root}"
        ) from exc
    return result.stdout.strip()


def verify_clean_archive_checkout(
    archive_root: Path,
    *,
    expected_revision: str | None = None,
) -> str:
    """Return HEAD only when ``archive_root`` is its clean Git worktree root.

    The clean-tree requirement binds all verified bundle bytes and manifests to
    the revision recorded in the import report.  Both tracked changes and
    untracked files are rejected.
    """
    archive_root = Path(archive_root).resolve(strict=True)
    top_level = Path(_run_archive_git(archive_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != archive_root:
        raise ArchiveVerificationError(
            f"archive_root must be the Git worktree root: root={archive_root}, top={top_level}"
        )
    revision = _run_archive_git(archive_root, "rev-parse", "--verify", "HEAD^{commit}")
    if not HEX_SHA1.fullmatch(revision):
        raise ArchiveVerificationError(f"archive_root HEAD is not a full Git commit hash: {revision!r}")
    if expected_revision is not None and revision != expected_revision:
        raise ArchiveVerificationError(
            f"archive_root HEAD changed during verification: {expected_revision} -> {revision}"
        )
    status = _run_archive_git(
        archive_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ArchiveVerificationError(
            "archive_root is not clean; import only an unmodified committed checkout: "
            + status.splitlines()[0]
        )
    return revision


@dataclass(frozen=True)
class VerifiedFile:
    source: Path
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    name: str
    collection_id: str
    manifest_sha256: str
    source_git_commit: str
    runtime_provenance_status: str
    manifest_schema_version: int
    snapshot_utc: datetime
    files: tuple[VerifiedFile, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_id_set_sha256(catalog_ids: Iterable[object]) -> str:
    normalized = sorted({str(value).strip() for value in catalog_ids if str(value).strip()})
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _safe_manifest_path(bundle_root: Path, raw_path: object) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ArchiveVerificationError(f"Unsafe manifest path: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ArchiveVerificationError(f"Unsafe manifest path: {raw_path!r}")
    candidate = bundle_root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=False).relative_to(bundle_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ArchiveVerificationError(f"Manifest path escapes bundle: {raw_path!r}") from exc
    return relative.as_posix(), candidate


def _load_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(f"Invalid {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveVerificationError(f"{description} must be a JSON object: {path}")
    return value


def _verify_sidecar(
    sidecar_path: Path,
    collection_id: str,
    config: ExperimentConfig,
    source_commit: str,
    schema_version: int,
) -> dict:
    sidecar = _load_json(sidecar_path, "TLE provenance sidecar")
    ids = sidecar.get("catalog_ids")
    if not isinstance(ids, list):
        raise ArchiveVerificationError(f"catalog_ids must be a list: {sidecar_path}")
    observed_ids = [str(value).strip() for value in ids]
    if len(observed_ids) != config.max_objects or len(set(observed_ids)) != config.max_objects:
        raise ArchiveVerificationError(
            f"Frozen cohort must contain {config.max_objects} unique IDs: {sidecar_path}"
        )
    observed_hash = catalog_id_set_sha256(observed_ids)
    expected = {
        "collection_id": collection_id,
        "catalog_version": config.catalog_version,
        "catalog_sha256": config.catalog_sha256,
        "object_count": config.max_objects,
        "requested_object_count": config.max_objects,
        "preset": config.preset,
    }
    if schema_version == 3:
        expected.update(
            {
                "experiment_id": config.experiment_id,
                "experiment_config_sha256": config.config_sha256,
            }
        )
    mismatches = {
        key: {"expected": value, "observed": sidecar.get(key)}
        for key, value in expected.items()
        if sidecar.get(key) != value
    }
    if observed_hash != config.catalog_sha256:
        mismatches["catalog_ids_sha256"] = {
            "expected": config.catalog_sha256,
            "observed": observed_hash,
        }
    if schema_version == 3:
        expected_simulation = {
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
        }
        if sidecar.get("simulation") != expected_simulation:
            mismatches["simulation"] = {
                "expected": expected_simulation,
                "observed": sidecar.get("simulation"),
            }
    sidecar_commit = str(sidecar.get("git_commit", "")).strip().lower()
    if (
        not re.fullmatch(r"[0-9a-f]{7,40}", sidecar_commit)
        or not source_commit.startswith(sidecar_commit)
    ):
        mismatches["git_commit"] = {
            "expected_prefix_of": source_commit,
            "observed": sidecar_commit,
        }
    if mismatches:
        raise ArchiveVerificationError(
            f"Frozen cohort mismatch in {sidecar_path}: {json.dumps(mismatches, sort_keys=True)}"
        )
    return sidecar


def verify_collection_bundle(bundle_root: Path, config: ExperimentConfig) -> VerifiedBundle:
    """Verify one immutable collection bundle without modifying local data."""
    bundle_root = Path(bundle_root)
    match = BUNDLE_NAME.fullmatch(bundle_root.name)
    if not match or not bundle_root.is_dir():
        raise ArchiveVerificationError(f"Invalid collection bundle directory: {bundle_root}")
    manifest_path = bundle_root / "manifest.json"
    manifest = _load_json(manifest_path, "collection manifest")
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2, 3}:
        raise ArchiveVerificationError(f"Unsupported manifest schema in {manifest_path}")
    if str(manifest.get("github_run_id", "")) != match.group("run_id"):
        raise ArchiveVerificationError(f"Bundle/run ID mismatch in {manifest_path}")
    if str(manifest.get("github_run_attempt", "")) != match.group("attempt"):
        raise ArchiveVerificationError(f"Bundle/run attempt mismatch in {manifest_path}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ArchiveVerificationError(f"Manifest has no files: {manifest_path}")

    verified: list[VerifiedFile] = []
    listed_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ArchiveVerificationError(f"Malformed manifest file entry: {entry!r}")
        relative_path, path = _safe_manifest_path(bundle_root, entry.get("path"))
        if relative_path == "manifest.json" or relative_path in listed_paths:
            raise ArchiveVerificationError(f"Duplicate/reserved manifest path: {relative_path}")
        listed_paths.add(relative_path)
        expected_bytes = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or not HEX_SHA256.fullmatch(expected_hash)
        ):
            raise ArchiveVerificationError(f"Invalid size/hash for {relative_path}")
        if not path.is_file() or path.is_symlink():
            raise ArchiveVerificationError(f"Manifest file is missing or not regular: {path}")
        actual_bytes = path.stat().st_size
        actual_hash = file_sha256(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise ArchiveVerificationError(
                f"Manifest mismatch for {relative_path}: "
                f"bytes={actual_bytes}/{expected_bytes}, sha256={actual_hash}/{expected_hash}"
            )
        verified.append(VerifiedFile(path, relative_path, actual_bytes, actual_hash))

    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ArchiveVerificationError(f"Symlinks are not allowed in bundles: {path}")
        if path.is_file() and path != manifest_path:
            actual_paths.add(path.relative_to(bundle_root).as_posix())
    if actual_paths != listed_paths:
        raise ArchiveVerificationError(
            f"Manifest file set mismatch in {bundle_root}: "
            f"unlisted={sorted(actual_paths - listed_paths)}, missing={sorted(listed_paths - actual_paths)}"
        )
    if manifest.get("file_count") != len(verified):
        raise ArchiveVerificationError(f"file_count mismatch in {manifest_path}")
    if manifest.get("total_bytes") != sum(item.bytes for item in verified):
        raise ArchiveVerificationError(f"total_bytes mismatch in {manifest_path}")

    by_path = {item.relative_path: item for item in verified}
    runtime_path = "environment/runtime.json"
    if schema_version in {2, 3}:
        if runtime_path not in by_path:
            raise ArchiveVerificationError(
                f"Manifest schema {schema_version} requires {runtime_path}: {manifest_path}"
            )
        runtime = _load_json(by_path[runtime_path].source, "runtime environment")
        if (
            runtime.get("schema_version") != 1
            or not isinstance(runtime.get("packages"), list)
            or not runtime.get("python")
            or not runtime.get("platform")
        ):
            raise ArchiveVerificationError(
                f"Runtime environment is incomplete: {by_path[runtime_path].source}"
            )
        runtime_status = "verified"
    else:
        runtime_status = "verified_legacy" if runtime_path in by_path else "legacy_runtime_missing"
    if schema_version == 3:
        embedded_path = "experiment/config.json"
        if (
            manifest.get("experiment_id") != config.experiment_id
            or manifest.get("experiment_config_path") != embedded_path
            or manifest.get("experiment_config_sha256") != config.config_sha256
            or embedded_path not in by_path
        ):
            raise ArchiveVerificationError(
                f"Schema-3 experiment binding mismatch in {manifest_path}"
            )
        embedded = _load_json(by_path[embedded_path].source, "embedded experiment config")
        canonical = hashlib.sha256(
            json.dumps(
                embedded, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        if canonical != config.config_sha256:
            raise ArchiveVerificationError(
                f"Embedded experiment config digest mismatch in {manifest_path}"
            )
    expected_history = f"history/{Path(config.history).name}"
    tle_texts = sorted(path for path in by_path if re.fullmatch(r"tle/tles_[0-9]{8}_[0-9]{6}\.txt", path))
    tle_sidecars = sorted(path for path in by_path if re.fullmatch(r"tle/tles_[0-9]{8}_[0-9]{6}\.json", path))
    run_datasets = sorted(
        path for path in by_path
        if re.fullmatch(r"runs/[0-9]{8}_[0-9]{6}/conjunction_dataset\.csv", path)
    )
    run_identified = sorted(
        path for path in by_path
        if re.fullmatch(r"runs/[0-9]{8}_[0-9]{6}/identified_conjunctions\.csv", path)
    )
    if (
        expected_history not in by_path
        or len(tle_texts) != 1
        or len(tle_sidecars) != 1
        or len(run_datasets) != 1
        or len(run_identified) != 1
    ):
        raise ArchiveVerificationError(f"Bundle layout is incomplete or ambiguous: {bundle_root}")
    collection_ids = {
        Path(tle_texts[0]).stem.removeprefix("tles_"),
        Path(tle_sidecars[0]).stem.removeprefix("tles_"),
        PurePosixPath(run_datasets[0]).parts[1],
        PurePosixPath(run_identified[0]).parts[1],
    }
    if len(collection_ids) != 1:
        raise ArchiveVerificationError(f"Collection ID mismatch in {bundle_root}: {collection_ids}")
    collection_id = collection_ids.pop()
    source_commit = str(manifest.get("git_commit", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ArchiveVerificationError(f"Manifest git_commit must be a full SHA-1: {manifest_path}")
    sidecar = _verify_sidecar(
        by_path[tle_sidecars[0]].source,
        collection_id,
        config,
        source_commit,
        schema_version,
    )
    try:
        snapshot_utc = datetime.fromisoformat(
            str(sidecar["fetched_utc"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveVerificationError(
            f"Invalid fetched_utc in {by_path[tle_sidecars[0]].source}"
        ) from exc
    try:
        actual_catalog_ids = validated_tle_catalog_ids(by_path[tle_texts[0]].source)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveVerificationError(
            f"Invalid TLE content in {by_path[tle_texts[0]].source}: {exc}"
        ) from exc
    if (
        len(actual_catalog_ids) != config.max_objects
        or catalog_id_set_sha256(actual_catalog_ids) != config.catalog_sha256
    ):
        raise ArchiveVerificationError(
            "TLE bytes do not contain the frozen catalogue ID set: "
            f"{by_path[tle_texts[0]].source}"
        )

    return VerifiedBundle(
        root=bundle_root,
        name=bundle_root.name,
        collection_id=collection_id,
        manifest_sha256=file_sha256(manifest_path),
        source_git_commit=source_commit,
        runtime_provenance_status=runtime_status,
        manifest_schema_version=schema_version,
        snapshot_utc=snapshot_utc,
        files=tuple(verified),
    )


def _destination_for(
    item: VerifiedFile,
    snapshot_dir: Path,
    run_root: Path,
) -> Path | None:
    relative = PurePosixPath(item.relative_path)
    if relative.parts[0] == "tle":
        return snapshot_dir / relative.name
    if relative.parts[0] == "runs":
        return run_root.joinpath(*relative.parts[1:])
    return None


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as src, os.fdopen(handle, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def import_collection_archive(
    archive_root: Path,
    snapshot_dir: Path,
    run_root: Path,
    config: ExperimentConfig,
    *,
    verify_only: bool = False,
    archive_revision: str | None = None,
    require_clean_checkout: bool = False,
    trusted_source_ref: str | None = None,
) -> dict:
    """Verify every GitHub bundle, then atomically import TLE/run artifacts.

    Existing byte-identical destinations are idempotently retained.  Any
    destination collision with different content aborts before new files are
    copied.  Per-bundle history CSVs are intentionally not imported: the
    canonical history is rebuilt from corrected-TCA re-simulation outputs.
    """
    archive_root = Path(archive_root)
    if require_clean_checkout:
        verified_revision = verify_clean_archive_checkout(
            archive_root,
            expected_revision=archive_revision,
        )
        archive_revision = verified_revision
    collections_root = archive_root / config.archive_collections
    bundle_paths = sorted(collections_root.glob("github-run-*-attempt-*"))
    if not bundle_paths:
        raise ArchiveVerificationError(f"No GitHub collection bundles below {collections_root}")
    unexpected = sorted(
        path.name for path in collections_root.iterdir()
        if path.is_dir() and not BUNDLE_NAME.fullmatch(path.name)
    )
    if unexpected:
        raise ArchiveVerificationError(f"Unexpected collection directories: {unexpected}")

    bundles = [verify_collection_bundle(path, config) for path in bundle_paths]
    trusted_source_ref_commit = None
    if trusted_source_ref is not None:
        if not require_clean_checkout:
            raise ArchiveVerificationError(
                "trusted_source_ref verification requires a clean archive checkout"
            )
        trusted_source_ref_commit = _run_archive_git(
            archive_root, "rev-parse", "--verify", f"{trusted_source_ref}^{{commit}}"
        )
        if not HEX_SHA1.fullmatch(trusted_source_ref_commit):
            raise ArchiveVerificationError(
                f"Trusted source ref is not a full commit: {trusted_source_ref}"
            )
        for bundle in bundles:
            try:
                _run_archive_git(
                    archive_root,
                    "merge-base",
                    "--is-ancestor",
                    bundle.source_git_commit,
                    trusted_source_ref_commit,
                )
            except ArchiveVerificationError as exc:
                raise ArchiveVerificationError(
                    f"Bundle source commit is not an ancestor of {trusted_source_ref}: "
                    f"{bundle.name} {bundle.source_git_commit}"
                ) from exc
    operations: dict[Path, tuple[VerifiedFile, str]] = {}
    collection_owners: dict[str, str] = {}
    slot_owners: dict[int, str] = {}
    for bundle in bundles:
        previous_owner = collection_owners.setdefault(bundle.collection_id, bundle.name)
        if previous_owner != bundle.name:
            raise ArchiveVerificationError(
                f"Duplicate collection_id={bundle.collection_id} in {previous_owner} and {bundle.name}"
            )
        if config.collection_start_utc is not None and bundle.manifest_schema_version == 3:
            start = datetime.fromisoformat(
                config.collection_start_utc.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            end = datetime.fromisoformat(
                config.collection_end_utc.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            if not start <= bundle.snapshot_utc < end:
                raise ArchiveVerificationError(
                    f"Bundle is outside the frozen collection window: {bundle.name}"
                )
            slot = int(
                (bundle.snapshot_utc - start).total_seconds()
                // (config.poll_interval_hours * 3600.0)
            )
            previous_slot_owner = slot_owners.setdefault(slot, bundle.name)
            if previous_slot_owner != bundle.name:
                raise ArchiveVerificationError(
                    f"Duplicate frozen slot={slot} in {previous_slot_owner} and {bundle.name}"
                )
        for item in bundle.files:
            destination = _destination_for(item, Path(snapshot_dir), Path(run_root))
            if destination is None:
                continue
            previous = operations.get(destination)
            if previous and previous[0].sha256 != item.sha256:
                raise ArchiveVerificationError(
                    f"Archive sources disagree for destination {destination}: "
                    f"{previous[1]} vs {bundle.name}"
                )
            operations[destination] = (item, bundle.name)

    existing_identical: list[str] = []
    pending: list[tuple[Path, VerifiedFile, str]] = []
    for destination, (item, bundle_name) in sorted(operations.items(), key=lambda pair: str(pair[0])):
        if destination.exists():
            if not destination.is_file() or destination.is_symlink():
                raise ArchiveVerificationError(f"Import destination is not a regular file: {destination}")
            observed_hash = file_sha256(destination)
            if observed_hash != item.sha256:
                raise ArchiveVerificationError(
                    f"Refusing to overwrite conflicting destination {destination}: "
                    f"existing={observed_hash}, archive={item.sha256}"
                )
            existing_identical.append(str(destination))
        else:
            pending.append((destination, item, bundle_name))

    imported: list[str] = []
    if require_clean_checkout:
        verify_clean_archive_checkout(archive_root, expected_revision=archive_revision)
    if not verify_only:
        for destination, item, _bundle_name in pending:
            _atomic_copy(item.source, destination)
            if destination.stat().st_size != item.bytes or file_sha256(destination) != item.sha256:
                destination.unlink(missing_ok=True)
                raise ArchiveVerificationError(f"Post-copy verification failed: {destination}")
            imported.append(str(destination))

    if require_clean_checkout:
        verify_clean_archive_checkout(archive_root, expected_revision=archive_revision)

    bundle_index = []
    for bundle in bundles:
        tle_inputs = [
            item
            for item in bundle.files
            if re.fullmatch(r"tle/tles_[0-9]{8}_[0-9]{6}\.txt", item.relative_path)
        ]
        if len(tle_inputs) != 1:
            raise ArchiveVerificationError(
                f"Verified bundle has an ambiguous TLE input: {bundle.root}"
            )
        sidecar_inputs = [
            item
            for item in bundle.files
            if re.fullmatch(r"tle/tles_[0-9]{8}_[0-9]{6}\.json", item.relative_path)
        ]
        if len(sidecar_inputs) != 1:
            raise ArchiveVerificationError(
                f"Verified bundle has an ambiguous TLE provenance sidecar: {bundle.root}"
            )
        bundle_index.append(
            {
                "name": bundle.name,
                "collection_id": bundle.collection_id,
                "manifest_sha256": bundle.manifest_sha256,
                "input_snapshot_sha256": tle_inputs[0].sha256,
                "input_sidecar_sha256": sidecar_inputs[0].sha256,
                "source_git_commit": bundle.source_git_commit,
                "runtime_provenance_status": bundle.runtime_provenance_status,
                "manifest_schema_version": bundle.manifest_schema_version,
                "snapshot_utc": bundle.snapshot_utc.isoformat().replace("+00:00", "Z"),
            }
        )
    manifest_set_sha256 = hashlib.sha256(
        json.dumps(bundle_index, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "generated_utc": generated_utc(),
        "git_commit": git_commit_full_hash(),
        "source_worktree": git_worktree_state(),
        "archive_root": str(archive_root),
        "archive_revision": archive_revision or "unspecified",
        "clean_checkout_verified": require_clean_checkout,
        "trusted_source_ref": trusted_source_ref,
        "trusted_source_ref_commit": trusted_source_ref_commit,
        "source_ancestry_verified": trusted_source_ref is not None,
        "manifest_set_sha256": manifest_set_sha256,
        "catalog_version": config.catalog_version,
        "catalog_sha256": config.catalog_sha256,
        "verify_only": verify_only,
        "bundles_verified": len(bundles),
        "collections_verified": len(collection_owners),
        "files_planned": len(operations),
        "files_imported": len(imported),
        "files_already_identical": len(existing_identical),
        "bundles": bundle_index,
        "imported_files": imported,
        "already_identical_files": existing_identical,
    }
