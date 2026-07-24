"""Direct unit tests for the fail-closed TLE validation module.

`space_debris.tle_validation` is the trust root for the project's
data-integrity story (checksum, structure, catalogue-ID and hash-diversity
gates). It was previously exercised only indirectly through the fetch/ML/
evidence code paths; these tests pin its behaviour directly so a regression in
one caller cannot hide a weakening of the shared guarantee.
"""

from __future__ import annotations

import pandas as pd
import pytest

from space_debris.tle_validation import (
    partition_tle_hash_quality,
    tle_checksum_is_valid,
    validated_tle_blocks,
    validated_tle_catalog_ids,
)

def _with_valid_checksum(line69: str) -> str:
    """Return the 69-char line with its final check digit made spec-correct.

    Standard TLE checksum: sum of digits in columns 1-68 plus one per minus
    sign, taken mod 10. Building fixtures this way keeps them valid regardless
    of transcription typos in any source file.
    """
    prefix = line69[:68]
    total = sum(int(c) for c in prefix if c.isdigit()) + prefix.count("-")
    return prefix + str(total % 10)


# Real ISS/NOAA element sets (from data/sample_tles.txt), re-stamped with a
# spec-correct check digit so the fixtures are guaranteed checksum-valid.
ISS_L1 = _with_valid_checksum(
    "1 25544U 98067A   24019.54791435  .00016717  00000+0  10270-3 0  9991"
)
ISS_L2 = _with_valid_checksum(
    "2 25544  51.6415  67.7316 0005447  83.1490  28.3976 15.50012378431589"
)
NOAA_L1 = _with_valid_checksum(
    "1 25338U 98030A   24019.54262076  .00000073  00000+0  69177-4 0  9993"
)
NOAA_L2 = _with_valid_checksum(
    "2 25338  98.7314  58.7588 0011420 193.1801 166.9396 14.25933904353136"
)


def _write(path, *blocks: tuple[str, str, str]) -> None:
    lines = []
    for name, l1, l2 in blocks:
        lines += [name, l1, l2]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestChecksum:
    def test_accepts_valid_lines(self):
        assert tle_checksum_is_valid(ISS_L1)
        assert tle_checksum_is_valid(ISS_L2)

    def test_rejects_wrong_length(self):
        assert not tle_checksum_is_valid(ISS_L1[:-1])
        assert not tle_checksum_is_valid(ISS_L1 + "0")

    def test_rejects_non_digit_check_char(self):
        assert not tle_checksum_is_valid(ISS_L1[:-1] + "X")

    def test_rejects_corrupted_digit(self):
        # Flip the checksum digit: 1 -> 2.
        assert not tle_checksum_is_valid(ISS_L1[:-1] + "2")

    def test_minus_sign_contributes_to_checksum(self):
        # NOAA line 1 contains a '-' in the drag term; validity proves the
        # minus-sign-counts-as-1 rule is applied, not just digit summation.
        assert "-" in NOAA_L1
        assert tle_checksum_is_valid(NOAA_L1)


class TestValidatedBlocks:
    def test_reads_valid_multi_object_file(self, tmp_path):
        path = tmp_path / "tles.txt"
        _write(path, ("ISS", ISS_L1, ISS_L2), ("NOAA 15", NOAA_L1, NOAA_L2))
        blocks = validated_tle_blocks(path)
        assert [b[0] for b in blocks] == ["ISS", "NOAA 15"]
        assert validated_tle_catalog_ids(path) == ["25544", "25338"]

    def test_rejects_non_triple_line_count(self, tmp_path):
        path = tmp_path / "tles.txt"
        path.write_text("\n".join(["ISS", ISS_L1]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="name/line1/line2"):
            validated_tle_blocks(path)

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "tles.txt"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="name/line1/line2"):
            validated_tle_blocks(path)

    def test_rejects_wrong_line_prefix(self, tmp_path):
        path = tmp_path / "tles.txt"
        # Swap line order so line1 no longer starts with '1 '.
        _write(path, ("ISS", ISS_L2, ISS_L1))
        with pytest.raises(ValueError, match="must start with"):
            validated_tle_blocks(path)

    def test_rejects_wrong_line_length(self, tmp_path):
        path = tmp_path / "tles.txt"
        # Insert an interior digit (not stripped) so the line is 70 chars while
        # the '1 ' prefix and catalogue field (cols 2-7) stay intact.
        too_long = ISS_L1[:30] + "0" + ISS_L1[30:]
        assert len(too_long) == 70
        _write(path, ("ISS", too_long, ISS_L2))
        with pytest.raises(ValueError, match="exactly 69 characters"):
            validated_tle_blocks(path)

    def test_rejects_mismatched_catalog_ids(self, tmp_path):
        path = tmp_path / "tles.txt"
        # Line 2 from a different object -> catalogue IDs disagree.
        _write(path, ("MIX", ISS_L1, NOAA_L2))
        with pytest.raises(ValueError, match="identifiers disagree"):
            validated_tle_blocks(path)

    def test_rejects_duplicate_catalog_id(self, tmp_path):
        path = tmp_path / "tles.txt"
        _write(path, ("ISS", ISS_L1, ISS_L2), ("ISS again", ISS_L1, ISS_L2))
        with pytest.raises(ValueError, match="Duplicate TLE catalogue"):
            validated_tle_blocks(path)

    def test_rejects_bad_checksum(self, tmp_path):
        path = tmp_path / "tles.txt"
        corrupted = ISS_L1[:-1] + ("2" if ISS_L1[-1] != "2" else "3")
        _write(path, ("ISS", corrupted, ISS_L2))
        with pytest.raises(ValueError, match="checksum validation failed"):
            validated_tle_blocks(path)


def _record(collection_id, ts, sha):
    return {"collection_id": collection_id, "snapshot_utc": ts, "input_sha256": sha}


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class TestPartitionHashQuality:
    def test_happy_path_diversity_and_runs(self):
        frame = pd.DataFrame({"collection_id": ["c1", "c2", "c3", "c1"]})
        records = [
            _record("c1", "2026-07-24T00:17:00Z", HASH_A),
            _record("c2", "2026-07-24T02:17:00Z", HASH_A),
            _record("c3", "2026-07-24T04:17:00Z", HASH_B),
        ]
        result = partition_tle_hash_quality(frame, records)
        assert result["snapshots"] == 3
        assert result["unique_tle_input_hashes"] == 2
        assert result["tle_input_hash_diversity_fraction"] == pytest.approx(2 / 3)
        # HASH_A repeats in the first two chronological slots -> run of 2.
        assert result["max_identical_tle_hash_run_bins"] == 2

    def test_run_length_uses_chronological_order(self):
        frame = pd.DataFrame({"collection_id": ["c1", "c2", "c3"]})
        # Provide records out of chronological order; A,B,A by time means no run.
        records = [
            _record("c2", "2026-07-24T02:17:00Z", HASH_B),
            _record("c1", "2026-07-24T00:17:00Z", HASH_A),
            _record("c3", "2026-07-24T04:17:00Z", HASH_A),
        ]
        result = partition_tle_hash_quality(frame, records)
        assert result["max_identical_tle_hash_run_bins"] == 1
        assert result["unique_tle_input_hashes"] == 2

    def test_missing_collection_id_column(self):
        with pytest.raises(ValueError, match="requires collection_id"):
            partition_tle_hash_quality(pd.DataFrame({"x": [1]}), [])

    def test_malformed_record_rejected(self):
        frame = pd.DataFrame({"collection_id": ["c1"]})
        bad = [_record("c1", "2026-07-24T00:17:00Z", "tooshort")]
        with pytest.raises(ValueError, match="Malformed snapshot record"):
            partition_tle_hash_quality(frame, bad)

    def test_naive_timestamp_rejected(self):
        frame = pd.DataFrame({"collection_id": ["c1"]})
        naive = [_record("c1", "2026-07-24T00:17:00", HASH_A)]  # no tz
        with pytest.raises(ValueError, match="Malformed snapshot record"):
            partition_tle_hash_quality(frame, naive)

    def test_conflicting_record_rejected(self):
        frame = pd.DataFrame({"collection_id": ["c1"]})
        conflict = [
            _record("c1", "2026-07-24T00:17:00Z", HASH_A),
            _record("c1", "2026-07-24T00:17:00Z", HASH_B),
        ]
        with pytest.raises(ValueError, match="Conflicting snapshot record"):
            partition_tle_hash_quality(frame, conflict)

    def test_missing_snapshot_records_rejected(self):
        frame = pd.DataFrame({"collection_id": ["c1", "c2"]})
        records = [_record("c1", "2026-07-24T00:17:00Z", HASH_A)]
        with pytest.raises(ValueError, match="missing or empty"):
            partition_tle_hash_quality(frame, records)

    def test_custom_error_class_is_used(self):
        class Boom(Exception):
            pass

        with pytest.raises(Boom):
            partition_tle_hash_quality(
                pd.DataFrame({"x": [1]}), [], error_cls=Boom
            )
