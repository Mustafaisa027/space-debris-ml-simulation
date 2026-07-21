from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from dataclasses import dataclass
from http.client import HTTPResponse
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from space_debris.tle_validation import tle_checksum_is_valid


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php?{query}={value}&FORMAT=TLE"
SPACETRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_GP_URL = (
    "https://www.space-track.org/basicspacedata/query/class/gp/"
    "norad_cat_id/{catnrs}/orderby/norad_cat_id/format/tle"
)
DEFAULT_OBJECTS = {
    "25544": "ISS (ZARYA)",
    "25338": "NOAA 15",
    "20580": "HUBBLE SPACE TELESCOPE",
    "25994": "TERRA",
    "27424": "AQUA",
}

# CelesTrak's GROUP endpoint (e.g. GROUP=STARLINK) has been observed to return
# HTTP 403 even for small requests, while the per-object CATNR endpoint stays
# reachable. LEO_MIXED_CATALOG is a curated fallback: 75 long-lived, publicly
# documented objects spanning inclinations from ~28 deg to ~98 deg (sun-
# synchronous weather/earth-observation satellites, ISS/CSS and cargo
# traffic, ocean-altimetry and polar-orbiting science missions, and the
# debris objects from both sides of the 2009 Iridium 33 / Cosmos 2251
# collision)
# so their orbital planes actually cross, unlike a single constellation whose
# satellites share near-identical planes and altitudes. Docked/same-platform
# modules and crew/cargo vehicles (POISK, ISS NAUKA, Crew Dragon, Progress-MS,
# Cygnus, CSS Wentian, CSS Mengtian) are excluded: they share the station's
# orbital position, so their separation is ~0 km and would register as false
# "close approach" positives despite there being no independent collision
# risk. Verify/refresh against CelesTrak before a final scientific run.
LEO_MIXED_CATALOG: dict[str, str] = {
    "25338": "NOAA 15",              # ~98.5 deg, sun-synchronous
    "43013": "NOAA 20",              # ~98.7 deg, sun-synchronous
    "54234": "NOAA 21",              # ~98.7 deg, sun-synchronous
    "25994": "TERRA",                # ~98.0 deg, sun-synchronous
    "27424": "AQUA",                 # ~98.4 deg, sun-synchronous
    "28376": "AURA",                 # ~98.3 deg, sun-synchronous
    "39084": "LANDSAT 8",            # ~98.2 deg, sun-synchronous
    "49260": "LANDSAT 9",            # ~98.2 deg, sun-synchronous
    "39634": "SENTINEL-1A",          # ~98.2 deg, sun-synchronous
    "40697": "SENTINEL-2A",          # ~98.6 deg, sun-synchronous
    "41335": "SENTINEL-3A",          # ~98.6 deg, sun-synchronous
    "42969": "SENTINEL-5P",          # ~98.8 deg, sun-synchronous
    "25544": "ISS (ZARYA)",          # ~51.6 deg
    "48274": "CSS (TIANHE)",         # ~41.5 deg, Chinese Space Station
    "20580": "HUBBLE SPACE TELESCOPE",  # ~28.5 deg
    "41884": "CYGFM05",              # ~35.0 deg, CYGNSS
    "41885": "CYGFM04",              # ~34.9 deg, CYGNSS
    "41886": "CYGFM02",              # ~34.9 deg, CYGNSS
    "60452": "LEGION 3",             # ~45.0 deg
    "41240": "JASON-3",              # ~66.0 deg, ocean altimetry
    "46984": "SENTINEL-6A",          # ~66.0 deg, ocean altimetry
    "54754": "SWOT",                 # ~77.6 deg, ocean altimetry
    "39451": "SWARM B",              # ~87.7 deg, polar
    "39452": "SWARM A",              # ~87.3 deg, polar
    "26998": "TIMED",                # ~74.1 deg
    "36508": "CRYOSAT 2",            # ~92.0 deg, polar
    "29228": "RESURS-DK 1",          # ~70.0 deg
    "24946": "IRIDIUM 33",           # ~86.4 deg (destroyed 2009-02-10)
    "33773": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33775": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33776": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33777": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33850": "IRIDIUM 33 DEB",       # ~86.3 deg, Iridium 33 collision debris
    "33860": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33862": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33866": "IRIDIUM 33 DEB",       # ~86.3 deg, Iridium 33 collision debris
    "33953": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "33960": "IRIDIUM 33 DEB",       # ~86.3 deg, Iridium 33 collision debris
    "34071": "IRIDIUM 33 DEB",       # ~86.3 deg, Iridium 33 collision debris
    "34077": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "34079": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "34088": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    "34521": "IRIDIUM 33 DEB",       # ~86.4 deg, Iridium 33 collision debris
    # Cosmos 2251 fragments: ~74 deg. Selected from the official CelesTrak
    # COSMOS 2251 debris table on 2026-07-16; each had a current GP solution,
    # perigee above 400 km, and a five-digit catalogue ID.
    "33757": "COSMOS 2251 DEB",
    "33758": "COSMOS 2251 DEB",
    "33760": "COSMOS 2251 DEB",
    "33761": "COSMOS 2251 DEB",
    "33762": "COSMOS 2251 DEB",
    "33764": "COSMOS 2251 DEB",
    "33765": "COSMOS 2251 DEB",
    "33766": "COSMOS 2251 DEB",
    "33768": "COSMOS 2251 DEB",
    "33779": "COSMOS 2251 DEB",
    "33819": "COSMOS 2251 DEB",
    "33821": "COSMOS 2251 DEB",
    "33822": "COSMOS 2251 DEB",
    "33823": "COSMOS 2251 DEB",
    "33825": "COSMOS 2251 DEB",
    "33826": "COSMOS 2251 DEB",
    "33828": "COSMOS 2251 DEB",
    "33830": "COSMOS 2251 DEB",
    "33832": "COSMOS 2251 DEB",
    "33835": "COSMOS 2251 DEB",
    "33836": "COSMOS 2251 DEB",
    "33837": "COSMOS 2251 DEB",
    "33838": "COSMOS 2251 DEB",
    "33839": "COSMOS 2251 DEB",
    "33911": "COSMOS 2251 DEB",
    "33912": "COSMOS 2251 DEB",
    "33913": "COSMOS 2251 DEB",
    "33914": "COSMOS 2251 DEB",
    "33915": "COSMOS 2251 DEB",
    "33916": "COSMOS 2251 DEB",
    "33917": "COSMOS 2251 DEB",
    "33920": "COSMOS 2251 DEB",
}

# Name lookup used when labelling any fetched TLE block.
KNOWN_OBJECT_NAMES: dict[str, str] = {**DEFAULT_OBJECTS, **LEO_MIXED_CATALOG}

PRESETS: dict[str, dict[str, str]] = {
    "leo_mixed": LEO_MIXED_CATALOG,
}

# Curated CATNR catalog to fall back to when a given GROUP query is blocked.
GROUP_FALLBACK_CATALOGS: dict[str, dict[str, str]] = {
    "STARLINK": LEO_MIXED_CATALOG,
}

RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class FetchReport:
    provider: str
    catalog_ids: tuple[str, ...]
    requested_count: int


class CelesTrakHTTPError(RuntimeError):
    def __init__(self, status_code: int, reason: str):
        self.status_code = status_code
        super().__init__(f"CelesTrak HTTP {status_code}: {reason}")


class TLETransportError(RuntimeError):
    """A transient network/transport failure suitable for retry."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current 3-line TLEs from CelesTrak")
    parser.add_argument("--catnr", nargs="*", default=list(DEFAULT_OBJECTS), help="NORAD catalog numbers")
    parser.add_argument("--group", nargs="*", default=[], help="CelesTrak GP groups, e.g. STATIONS WEATHER ACTIVE")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="use a curated multi-orbit CATNR catalog instead of --catnr/--group",
    )
    parser.add_argument("--max-objects", type=int, default=None, help="limit objects written to the output file")
    parser.add_argument(
        "--provider",
        choices=["auto", "celestrak", "space-track"],
        default="auto",
        help="auto uses CelesTrak first and Space-Track only when credentials are configured",
    )
    parser.add_argument("--output", default=None, help="output TLE file")
    return parser.parse_args()


def _read_response(response: HTTPResponse) -> str:
    if response.status != 200:
        raise CelesTrakHTTPError(response.status, response.reason)
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise ValueError("Provider returned HTML instead of orbital elements")
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError(f"Provider response exceeded {MAX_RESPONSE_BYTES} bytes")
    return payload.decode("utf-8-sig").strip()


def fetch_gp(query: str, value: str) -> str:
    query = query.upper()
    url = CELESTRAK_GP_URL.format(query=query, value=quote(value.upper() if query == "GROUP" else value))
    request = Request(url, headers={"User-Agent": "space-debris-iac-research/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            text = _read_response(response)
    except HTTPError as exc:
        raise CelesTrakHTTPError(exc.code, str(exc.reason)) from exc
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise TLETransportError(f"CelesTrak transport failure: {exc}") from exc
    if "DOCTYPE html" in text[:200] or "<html" in text[:200].lower():
        raise RuntimeError(f"CelesTrak returned HTML instead of TLE data for {query}={value}")
    return text


def fetch_gp_with_retry(query: str, value: str, retries: int = 4, base_delay: float = 1.5) -> str:
    """Call fetch_gp, retrying with exponential backoff on 403/429 responses.

    Non-retryable errors (e.g. a genuinely invalid CATNR) propagate immediately.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fetch_gp(query, value)
        except CelesTrakHTTPError as exc:
            if exc.status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_error = exc
            if attempt < retries:
                time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25))
        except TLETransportError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25))
    assert last_error is not None
    message = f"{last_error} (gave up after {retries} attempts for {query}={value})"
    if isinstance(last_error, CelesTrakHTTPError):
        raise CelesTrakHTTPError(last_error.status_code, message) from last_error
    raise TLETransportError(message) from last_error


def validate_tle_pair(line1: str, line2: str, expected_catnr: str | None = None) -> str:
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("TLE lines must start with '1 ' and '2 '")
    if len(line1) != 69 or len(line2) != 69:
        raise ValueError(f"TLE lines must each be exactly 69 characters, got {len(line1)} and {len(line2)}")
    catalog_1 = line1[2:7].strip()
    catalog_2 = line2[2:7].strip()
    if not catalog_1 or catalog_1 != catalog_2:
        raise ValueError(f"TLE catalog identifiers disagree: {catalog_1!r} vs {catalog_2!r}")
    if expected_catnr and catalog_1 != expected_catnr:
        raise ValueError(f"Expected CATNR={expected_catnr}, provider returned CATNR={catalog_1}")
    if not tle_checksum_is_valid(line1) or not tle_checksum_is_valid(line2):
        raise ValueError(f"TLE checksum validation failed for CATNR={catalog_1}")
    return catalog_1


def normalize_tle_blocks(text: str, fallback_name: str | None = None) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            catalog_id = validate_tle_pair(lines[i], lines[i + 1])
            name = fallback_name or KNOWN_OBJECT_NAMES.get(catalog_id, f"CATNR {catalog_id}")
            blocks.append("\n".join([name, lines[i], lines[i + 1]]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            validate_tle_pair(lines[i + 1], lines[i + 2])
            blocks.append("\n".join(lines[i : i + 3]))
            i += 3
        else:
            raise ValueError(f"Invalid TLE response around line {i + 1}: {lines[i][:120]}")
    return blocks


def fetch_tle(catnr: str, retries: int = 3) -> str:
    text = fetch_gp_with_retry("CATNR", catnr, retries=retries)
    blocks = normalize_tle_blocks(text, KNOWN_OBJECT_NAMES.get(catnr, f"CATNR {catnr}"))
    if len(blocks) != 1:
        raise ValueError(f"Expected one TLE for CATNR={catnr}, got {len(blocks)}")
    validate_tle_pair(*blocks[0].splitlines()[1:], expected_catnr=catnr)
    return blocks[0]


def fetch_catnr_blocks(catnrs: list[str], request_delay: float = 1.5) -> tuple[list[str], list[str]]:
    """Fetch each catalog number individually, tolerating per-object failures.

    Returns (blocks, failed_catnrs) so callers keep whatever partial result was
    obtained instead of losing an entire batch to one bad object.
    """
    blocks: list[str] = []
    failures: list[str] = []
    for i, catnr in enumerate(catnrs):
        if i > 0:
            time.sleep(request_delay)
        try:
            blocks.append(fetch_tle(catnr))
        except (CelesTrakHTTPError, TLETransportError, RuntimeError, ValueError) as exc:
            failures.append(catnr)
            print(f"WARN: failed to fetch CATNR={catnr}: {exc}")
    return blocks, failures


def fetch_group_blocks(group: str, request_delay: float = 1.5) -> tuple[list[str], list[str]]:
    """Fetch a CelesTrak GROUP, falling back to a curated CATNR catalog when the
    GROUP endpoint itself is blocked (403/429), e.g. GROUP_FALLBACK_CATALOGS.
    """
    try:
        text = fetch_gp_with_retry("GROUP", group)
        return normalize_tle_blocks(text), []
    except CelesTrakHTTPError as exc:
        if exc.status_code not in RETRYABLE_STATUS_CODES:
            raise
        fallback_catalog = GROUP_FALLBACK_CATALOGS.get(group.upper())
        if not fallback_catalog:
            raise CelesTrakHTTPError(
                exc.status_code,
                f"{exc}; no CATNR fallback catalog registered for GROUP={group}",
            ) from exc
        print(
            f"WARN: GROUP={group} returned HTTP {exc.status_code}; "
            f"falling back to {len(fallback_catalog)} curated CATNR objects."
        )
        return fetch_catnr_blocks(list(fallback_catalog), request_delay=request_delay)


def fetch_tle_blocks(
    catnrs: list[str],
    groups: list[str],
    max_objects: int | None = None,
    request_delay: float = 1.5,
    min_success_ratio: float = 0.90,
) -> list[str]:
    blocks: list[str] = []
    failures: list[str] = []

    catnr_blocks, catnr_failures = fetch_catnr_blocks(catnrs, request_delay=request_delay)
    blocks.extend(catnr_blocks)
    failures.extend(catnr_failures)

    for group in groups:
        group_blocks, group_failures = fetch_group_blocks(group, request_delay=request_delay)
        blocks.extend(group_blocks)
        failures.extend(group_failures)

    if failures:
        print(f"WARN: {len(failures)} object(s) could not be fetched: {failures}")

    requested_unique = len(set(catnrs))
    if requested_unique:
        received_for_requested = len(
            {block.splitlines()[1][2:7].strip() for block in blocks} & set(catnrs)
        )
        success_ratio = received_for_requested / requested_unique
        if success_ratio < min_success_ratio:
            raise RuntimeError(
                f"CATNR coverage {received_for_requested}/{requested_unique} ({success_ratio:.1%}) "
                f"is below the required {min_success_ratio:.0%}; refusing partial snapshot"
            )

    seen: set[str] = set()
    unique_blocks: list[str] = []
    for block in blocks:
        lines = block.splitlines()
        catalog_id = lines[1][2:7].strip()
        if catalog_id in seen:
            continue
        seen.add(catalog_id)
        unique_blocks.append(block)
        if max_objects and len(unique_blocks) >= max_objects:
            break
    return unique_blocks


def write_tle_file(output: Path, blocks: list[str]) -> None:
    if not blocks:
        raise ValueError("No TLE blocks fetched.")
    output.parent.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        lines = block.splitlines()
        if len(lines) != 3:
            raise ValueError("Every output block must contain name, line 1, and line 2")
        validate_tle_pair(lines[1], lines[2])
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(blocks) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def fetch_spacetrack_blocks(catnrs: list[str]) -> list[str]:
    identity = os.getenv("SPACETRACK_IDENTITY")
    password = os.getenv("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise RuntimeError("Space-Track fallback requires SPACETRACK_IDENTITY and SPACETRACK_PASSWORD")
    opener = build_opener(HTTPCookieProcessor())
    login = Request(
        SPACETRACK_LOGIN_URL,
        data=urlencode({"identity": identity, "password": password}).encode("utf-8"),
        headers={"User-Agent": "space-debris-iac-research/0.2"},
    )
    try:
        with opener.open(login, timeout=30) as response:
            response.read(MAX_RESPONSE_BYTES + 1)
        query = SPACETRACK_GP_URL.format(catnrs=",".join(dict.fromkeys(catnrs)))
        with opener.open(Request(query, headers={"User-Agent": "space-debris-iac-research/0.2"}), timeout=60) as response:
            text = _read_response(response)
    except HTTPError as exc:
        raise RuntimeError(f"Space-Track HTTP {exc.code}: {exc.reason}") from exc
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise TLETransportError(f"Space-Track transport failure: {exc}") from exc
    blocks = normalize_tle_blocks(text)
    returned = {block.splitlines()[1][2:7].strip() for block in blocks}
    requested = set(catnrs)
    ratio = len(returned & requested) / len(requested) if requested else 1.0
    if ratio < 0.90:
        raise RuntimeError(f"Space-Track CATNR coverage {ratio:.1%} is below required 90%")
    return blocks


def fetch_to_file(
    output: Path,
    catnrs: list[str],
    groups: list[str],
    max_objects: int | None = None,
    request_delay: float = 1.5,
    *,
    provider: str = "auto",
    report: dict | None = None,
) -> Path:
    actual_provider = "celestrak"
    if provider == "space-track":
        if groups and not catnrs:
            raise RuntimeError("Space-Track requires an explicit/preset CATNR catalog, not GROUP-only input")
        blocks = fetch_spacetrack_blocks(catnrs)
        if max_objects:
            blocks = blocks[:max_objects]
        actual_provider = "space-track"
    else:
        try:
            blocks = fetch_tle_blocks(catnrs, groups, max_objects, request_delay=request_delay)
        except (CelesTrakHTTPError, TLETransportError, RuntimeError, ValueError) as primary_error:
            can_fallback = provider == "auto" and bool(
                os.getenv("SPACETRACK_IDENTITY") and os.getenv("SPACETRACK_PASSWORD")
            )
            if not can_fallback:
                raise
            print(f"WARN: CelesTrak unavailable or invalid ({primary_error}); trying Space-Track GP API")
            if groups and not catnrs:
                raise RuntimeError("Space-Track fallback requires an explicit/preset CATNR catalog, not GROUP-only input")
            blocks = fetch_spacetrack_blocks(catnrs)
            if max_objects:
                blocks = blocks[:max_objects]
            actual_provider = "space-track"
    write_tle_file(output, blocks)
    if report is not None:
        report.update(
            FetchReport(
                provider=actual_provider,
                catalog_ids=tuple(block.splitlines()[1][2:7].strip() for block in blocks),
                requested_count=len(set(catnrs)),
            ).__dict__
        )
    return output


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.preset:
        catnrs = list(PRESETS[args.preset])
        groups: list[str] = []
        default_output = f"data/catalog_{args.preset}.txt"
    else:
        catnrs = args.catnr
        groups = args.group
        default_output = f"data/live_tles_{timestamp}.txt"

    output = Path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fetch_to_file(output, catnrs, groups, args.max_objects, provider=args.provider)
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()
