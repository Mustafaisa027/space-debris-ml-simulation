"""Shared provenance metadata for output artifacts (ROADMAP_YOL1.md GOREV 6).

Every CSV/figure this pipeline writes should be traceable to WHERE the data
came from, WHEN it was generated, and WHICH exact code version (git commit)
produced it -- without that, a reviewer can't tell whether a table or figure
in the paper is reproducible or which run it belongs to.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

CSV_COMMENT_PREFIX = "#"


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so readers never observe a partial file.

    Writes to a sibling temp file, fsyncs it, then ``os.replace``s it over
    the destination -- a crash or concurrent read mid-write cannot produce a
    truncated/corrupt artifact, which matters for anything downstream that
    hashes or re-parses these files as evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically write ``value`` as indented, sorted-key JSON to ``path``."""
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_commit_hash() -> str:
    """Best-effort short git commit hash.

    Returns "unknown" outside a git checkout (e.g. a source tarball) or if
    git isn't on PATH -- provenance must never crash the pipeline it exists
    to describe.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_commit_full_hash() -> str:
    """Best-effort full commit hash for machine-verifiable manifests."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        value = result.stdout.strip()
        return value if len(value) == 40 else "unknown"
    except Exception:
        return "unknown"


def git_worktree_state() -> dict[str, object]:
    """Record whether generated evidence came from committed source only."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        porcelain = result.stdout.replace("\r\n", "\n")
        return {
            "dirty": bool(porcelain.strip()),
            "status_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
        }
    except Exception:
        return {"dirty": None, "status_sha256": "unknown"}


def generated_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def provenance_header_lines(source: str = "", config_summary: str = "") -> list[str]:
    """'#'-prefixed lines describing an output's provenance.

    Any reader of the resulting CSV must pass ``comment="#"`` to
    ``pandas.read_csv`` (or filter these lines before ``csv.DictReader``) to
    skip them.
    """
    return [
        f"{CSV_COMMENT_PREFIX} generated_utc: {generated_utc()}",
        f"{CSV_COMMENT_PREFIX} git_commit: {git_commit_hash()}",
        f"{CSV_COMMENT_PREFIX} source: {source or 'unspecified'}",
        f"{CSV_COMMENT_PREFIX} config: {config_summary or 'unspecified'}",
    ]


def write_csv_text_with_provenance(
    path: Path,
    csv_text: str,
    source: str = "",
    config_summary: str = "",
) -> None:
    """Write ``csv_text`` (as produced by e.g. ``DataFrame.to_csv()`` or a
    hand-built ``csv.DictWriter`` buffer) to ``path``, prefixed with a
    provenance comment header.
    """
    header = "\n".join(provenance_header_lines(source, config_summary))
    write_text_atomic(path, header + "\n" + csv_text)


def png_provenance_metadata(source: str = "", config_summary: str = "") -> dict[str, str]:
    """Metadata dict for ``matplotlib.pyplot.savefig(..., metadata=...)``.

    Embedded as PNG tEXt chunks -- invisible in the rendered image but
    inspectable in the file itself (e.g. via Pillow's ``Image.info``),
    so a figure alone still carries its generation date/commit/source/config.
    """
    return {
        "Software": "space-debris-ml-simulation",
        "Creation Time": generated_utc(),
        "Description": (
            f"git_commit={git_commit_hash()}; source={source or 'unspecified'}; "
            f"config={config_summary or 'unspecified'}"
        ),
    }
