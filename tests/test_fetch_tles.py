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
    return (
        f"1 {catnr}U 20001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
        f"2 {catnr}  51.6400 000.0000 0001000   0.0000   0.0000 15.50000000  1000"
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
        raise fetch_tles.CelesTrakHTTPError(500, "Server Error")

    monkeypatch.setattr(fetch_tles, "fetch_gp", fake_fetch_gp)
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(fetch_tles.CelesTrakHTTPError):
        fetch_tles.fetch_gp_with_retry("CATNR", "99999")

    assert calls["n"] == 1
    assert sleeps == []


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

    def fake_fetch_to_file(output, catnrs, groups, max_objects=None, request_delay=1.5):
        captured["output"] = output
        captured["catnrs"] = catnrs
        captured["groups"] = groups
        return output

    monkeypatch.setattr(fetch_tles, "fetch_to_file", fake_fetch_to_file)

    fetch_tles.main()

    assert captured["output"] == Path("data/catalog_leo_mixed.txt")
    assert captured["groups"] == []
    assert set(captured["catnrs"]) == set(fetch_tles.LEO_MIXED_CATALOG)
