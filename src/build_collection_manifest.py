from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
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
        "schema_version": 1,
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
    output.write_text(json.dumps(build_manifest(args.root), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
