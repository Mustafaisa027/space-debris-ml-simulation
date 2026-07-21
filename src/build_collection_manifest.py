from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from space_debris.provenance import write_json_atomic


RUNTIME_PATH = "environment/runtime.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime_environment(root: Path) -> None:
    runtime_path = root / RUNTIME_PATH
    if not runtime_path.is_file() or runtime_path.is_symlink():
        raise ValueError(
            f"Authoritative schema-2 manifests require a regular {RUNTIME_PATH} file"
        )
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid runtime environment record: {runtime_path}: {exc}") from exc
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema_version") != 1
        or not isinstance(runtime.get("packages"), list)
        or not runtime.get("python")
        or not runtime.get("platform")
    ):
        raise ValueError(f"Incomplete runtime environment record: {runtime_path}")


def build_manifest(root: Path) -> dict:
    _validate_runtime_environment(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": os.getenv("GITHUB_SHA", "unknown"),
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create hashes and provenance for one collection bundle")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Collection root does not exist: {args.root}")
    output = args.output or args.root / "manifest.json"
    write_json_atomic(output, build_manifest(args.root))


if __name__ == "__main__":
    main()
