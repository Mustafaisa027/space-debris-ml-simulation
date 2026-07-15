"""Unit tests for fetch_tles.py: GROUP->CATNR fallback and retry/backoff.

CelesTrak's GROUP endpoint has been observed to return HTTP 403 for real
requests, so these tests mock fetch_gp instead of hitting the network.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import fetch_tles


def _fake_tle_text(catnr: str) -> str:
    def with_checksum(line: str) -> str:
        body = line[:68].ljust(68)
        checksum = sum(int(char) for char in body if char.isdigit()) + body.count("-")
        return body + str(checksum % 10)

    return "\n".join(
        [
            with_checksum(f"1 {catnr}U 20001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"),
            with_checksum(f"2 {catnr}  51.6400 000.0000 0001000   0.0000   0.0000 15.50000000  1000"),
        ]
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Every test disables real delays by default; tests that care about the
    # number/order of sleeps override this with their own recording stub.
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda _seconds: None)


def test_fetch_gp_with_retry_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_fetch_gp(query, value):
        calls["n"] += 1
        if calls["n"] < 3:
            raise fetch_tles.CelesTrakHTTPError(429, "Too Many Requests")
        return _fake_tle_text(value)

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda s: sleeps.append(s))

    result = fetch_tles.fetch_gp_with_retry("CATNR", "25544")

    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential backoff
    assert "25544" in result


def test_fetch_gp_with_retry_raises_after_exhausting_retries(monkeypatch):
    def fake_fetch_gp(query, value):
        raise fetch_tles.CelesTrakHTTPError(403, "Forbidden")

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)

    with pytest.raises(fetch_tles.CelesTrakHTTPError):
        fetch_tles.fetch_gp_with_retry("GROUP", "STARLINK", retries=3)


def test_fetch_gp_with_retry_does_not_retry_non_retryable_errors(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_fetch_gp(query, value):
        calls["n"] += 1
        raise fetch_tles.CelesTrakHTTPError(400, "Bad Request")

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(fetch_tles.CelesTrakHTTPError):
        fetch_tles.fetch_gp_with_retry("CATNR", "99999")

    assert calls["n"] == 1
    assert sleeps == []


def test_normalize_rejects_bad_checksum():
    text = _fake_tle_text("25544")
    bad = text[:-1] + ("0" if text[-1] != "0" else "1")

    with pytest.raises(ValueError, match="checksum"):
        fetch_tles.normalize_tle_blocks(bad)


def test_fetch_tle_rejects_wrong_catalog_number(monkeypatch):
    monkeypatch.setattr(fetch_tles, "fetch_gp", lambda _query, _value: _fake_tle_text("12345"))

    with pytest.raises(ValueError, match="Expected CATNR=25544"):
        fetch_tles.fetch_tle("25544")


def test_write_tle_file_is_atomic_on_validation_failure(tmp_path):
    output = tmp_path / "snapshot.txt"
    output.write_text("previous-good-data\n", encoding="utf-8")

    with pytest.raises(ValueError):
        fetch_tles.write_tle_file(output, ["BAD\n1 invalid\n2 invalid"])

    assert output.read_text(encoding="utf-8") == "previous-good-data\n"


def test_fetch_tle_blocks_rejects_low_catalog_coverage(monkeypatch):
    monkeypatch.setattr(
        fetch_tles,
        "fetch_catnr_blocks",
        lambda _catnrs, request_delay=0: (["SAT\n" + _fake_tle_text("25544")], ["25338"]),
    )

    with pytest.raises(RuntimeError, match="coverage"):
        fetch_tles.fetch_tle_blocks(["25544", "25338"], [], min_success_ratio=0.9)


def test_explicit_space_track_does_not_call_celestrak(monkeypatch, tmp_path):
    block = "ISS\n" + _fake_tle_text("25544")
    monkeypatch.setattr(
        fetch_tles,
        "fetch_tle_blocks",
        lambda *_args, **_kwargs: pytest.fail("CelesTrak must not be called"),
    )
    monkeypatch.setattr(fetch_tles, "fetch_spacetrack_blocks", lambda catnrs: [block])
    report = {}

    output = fetch_tles.fetch_to_file(
        tmp_path / "snapshot.txt", ["25544"], [], provider="space-track", report=report
    )

    assert output.exists()
    assert report["provider"] == "space-track"
    assert report["catalog_ids"] == ("25544",)


def test_auto_falls_back_to_space_track_only_with_credentials(monkeypatch, tmp_path):
    block = "ISS\n" + _fake_tle_text("25544")
    monkeypatch.setenv("SPACETRACK_IDENTITY", "researcher@example.test")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "secret")
    monkeypatch.setattr(
        fetch_tles,
        "fetch_tle_blocks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(fetch_tles.TLETransportError("offline")),
    )
    monkeypatch.setattr(fetch_tles, "fetch_spacetrack_blocks", lambda catnrs: [block])

    fetch_tles.fetch_to_file(tmp_path / "snapshot.txt", ["25544"], [], provider="auto")

    assert (tmp_path / "snapshot.txt").exists()


def test_fetch_group_blocks_succeeds_without_fallback(monkeypatch):
    def fake_fetch_gp(query, value):
        assert query == "GROUP"
        return "TEST SAT\n" + _fake_tle_text("12345")

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)

    blocks, failures = fetch_tles.fetch_group_blocks("WEATHER")

    assert failures == []
    assert len(blocks) == 1


def test_fetch_group_blocks_falls_back_to_catnr_catalog_on_403(monkeypatch):
    def fake_fetch_gp(query, value):
        if query == "GROUP":
            raise fetch_tles.CelesTrakHTTPError(403, "Forbidden")
        return _fake_tle_text(value)

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)

    blocks, failures = fetch_tles.fetch_group_blocks("STARLINK")

    assert failures == []
    assert len(blocks) == len(fetch_tles.LEO_MIXED_CATALOG)


def test_fetch_group_blocks_raises_when_no_fallback_registered(monkeypatch):
    def fake_fetch_gp(query, value):
        raise fetch_tles.CelesTrakHTTPError(403, "Forbidden")

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)

    with pytest.raises(fetch_tles.CelesTrakHTTPError, match="no CATNR fallback"):
        fetch_tles.fetch_group_blocks("WEATHER")


def test_fetch_catnr_blocks_preserves_partial_results_on_failure(monkeypatch):
    def fake_fetch_gp(query, value):
        if value == "40000":
            raise fetch_tles.CelesTrakHTTPError(404, "Not Found")
        return _fake_tle_text(value)

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)

    blocks, failures = fetch_tles.fetch_catnr_blocks(["25544", "40000"])

    assert len(blocks) == 1
    assert failures == ["40000"]


def test_leo_mixed_preset_registered_and_used_as_starlink_fallback():
    assert "leo_mixed" in fetch_tles.PRESETS
    catalog = fetch_tles.PRESETS["leo_mixed"]
    assert catalog == fetch_tles.LEO_MIXED_CATALOG
    assert len(catalog) >= 5
    assert fetch_tles.GROUP_FALLBACK_CATALOGS["STARLINK"] == catalog


def test_parse_args_accepts_preset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_tles.py", "--preset", "leo_mixed"])
    args = fetch_tles.parse_args()
    assert args.preset == "leo_mixed"


def test_main_preset_uses_leo_mixed_catalog_and_default_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch_tles.py", "--preset", "leo_mixed"])

    captured: dict = {}

    def fake_fetch_to_file(
        output, catnrs, groups, max_objects=None, request_delay=1.5, *, provider="auto", report=None
    ):
        captured["output"] = output
        captured["catnrs"] = catnrs
        captured["groups"] = groups
        captured["provider"] = provider
        return output

    monkeypatch.setattr(fetch_tles, "fetch_to_file", fake_fetch_to_file)

    fetch_tles.main()

    assert captured["output"] == Path("data/catalog_leo_mixed.txt")
    assert captured["groups"] == []
    assert captured["provider"] == "auto"
    assert set(captured["catnrs"]) == set(fetch_tles.LEO_MIXED_CATALOG)
