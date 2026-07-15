from __future__ import annotations

import hashlib
import json

from build_collection_manifest import build_manifest, main


def test_build_manifest_records_relative_paths_sizes_and_hashes(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    payload = b"tle-data\n"
    (tmp_path / "nested" / "snapshot.txt").write_bytes(payload)
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")

    manifest = build_manifest(tmp_path)

    assert manifest["git_commit"] == "abc123"
    assert manifest["github_run_id"] == "42"
    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] == len(payload)
    assert manifest["files"][0] == {
        "path": "nested/snapshot.txt",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_main_writes_manifest_without_hashing_an_old_manifest(tmp_path, monkeypatch):
    (tmp_path / "result.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("old", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["build_collection_manifest.py", str(tmp_path)])

    main()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"] == "result.csv"
