from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import pandas as pd


def tle_checksum_is_valid(line: str) -> bool:
    if len(line) != 69 or not line[-1].isdigit():
        return False
    checksum = sum(int(character) for character in line[:68] if character.isdigit())
    checksum += line[:68].count("-")
    return checksum % 10 == int(line[-1])


def validated_tle_blocks(path: Path) -> list[tuple[str, str, str]]:
    """Read strict name/line-1/line-2 TLE blocks with matching checksums/IDs."""
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or len(lines) % 3 != 0:
        raise ValueError(f"TLE file must contain name/line1/line2 groups: {path}")
    blocks: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for index in range(0, len(lines), 3):
        name, line1, line2 = lines[index : index + 3]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise ValueError(f"TLE lines must start with '1 ' and '2 ': {path}")
        if len(line1) != 69 or len(line2) != 69:
            raise ValueError(f"TLE lines must each be exactly 69 characters: {path}")
        catalog_1 = line1[2:7].strip()
        catalog_2 = line2[2:7].strip()
        if not catalog_1 or catalog_1 != catalog_2:
            raise ValueError(
                f"TLE catalogue identifiers disagree: {catalog_1!r} vs {catalog_2!r}"
            )
        if catalog_1 in seen:
            raise ValueError(f"Duplicate TLE catalogue identifier {catalog_1}: {path}")
        if not tle_checksum_is_valid(line1) or not tle_checksum_is_valid(line2):
            raise ValueError(f"TLE checksum validation failed for CATNR={catalog_1}")
        seen.add(catalog_1)
        blocks.append((name, line1, line2))
    return blocks


def validated_tle_catalog_ids(path: Path) -> list[str]:
    return [line1[2:7].strip() for _name, line1, _line2 in validated_tle_blocks(path)]


def partition_tle_hash_quality(
    frame: "pd.DataFrame",
    snapshot_records: Sequence[dict[str, object]],
    error_cls: type[Exception] = ValueError,
) -> dict[str, object]:
    """Measure independent TLE updates represented inside one model partition.

    Shared by ``ml.py`` and ``evidence.py``, which previously carried
    functionally identical copies of this check (differing only in which
    exception type they raised on malformed input) -- ``error_cls`` lets each
    caller keep raising its own domain-specific error.
    """
    import pandas as pd

    if "collection_id" not in frame.columns:
        raise error_cls("Partition TLE quality requires collection_id")
    by_collection: dict[str, tuple[pd.Timestamp, str]] = {}
    for record in snapshot_records:
        collection_id = str(record.get("collection_id", "")).strip()
        timestamp = pd.Timestamp(record.get("snapshot_utc"))
        input_sha256 = str(record.get("input_sha256", "")).strip().lower()
        if (
            not collection_id
            or timestamp.tzinfo is None
            or len(input_sha256) != 64
            or any(character not in "0123456789abcdef" for character in input_sha256)
        ):
            raise error_cls("Malformed snapshot record for partition TLE quality")
        normalized = (timestamp.tz_convert("UTC"), input_sha256)
        if collection_id in by_collection and by_collection[collection_id] != normalized:
            raise error_cls(f"Conflicting snapshot record for collection_id={collection_id}")
        by_collection[collection_id] = normalized
    collection_ids = sorted(
        set(frame["collection_id"].dropna().astype(str).str.strip()) - {""}
    )
    missing = sorted(set(collection_ids) - set(by_collection))
    if not collection_ids or missing:
        raise error_cls(f"Partition snapshot records are missing or empty: missing={missing}")
    ordered = sorted(
        (by_collection[collection_id][0], by_collection[collection_id][1])
        for collection_id in collection_ids
    )
    hashes = [value for _timestamp, value in ordered]
    max_run = 0
    current_run = 0
    previous = None
    for value in hashes:
        current_run = current_run + 1 if value == previous else 1
        max_run = max(max_run, current_run)
        previous = value
    unique_hashes = len(set(hashes))
    return {
        "snapshots": len(hashes),
        "unique_tle_input_hashes": unique_hashes,
        "tle_input_hash_diversity_fraction": unique_hashes / len(hashes),
        "max_identical_tle_hash_run_bins": max_run,
    }
