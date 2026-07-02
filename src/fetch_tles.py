from __future__ import annotations

import argparse
from http.client import HTTPResponse
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php?{query}={value}&FORMAT=TLE"
DEFAULT_OBJECTS = {
    "25544": "ISS (ZARYA)",
    "25338": "NOAA 15",
    "20580": "HUBBLE SPACE TELESCOPE",
    "25994": "TERRA",
    "27424": "AQUA",
}


class CelesTrakHTTPError(RuntimeError):
    def __init__(self, status_code: int, reason: str):
        self.status_code = status_code
        super().__init__(f"CelesTrak HTTP {status_code}: {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current 3-line TLEs from CelesTrak")
    parser.add_argument("--catnr", nargs="*", default=list(DEFAULT_OBJECTS), help="NORAD catalog numbers")
    parser.add_argument("--group", nargs="*", default=[], help="CelesTrak GP groups, e.g. STATIONS WEATHER ACTIVE")
    parser.add_argument("--max-objects", type=int, default=None, help="limit objects written to the output file")
    parser.add_argument("--output", default=None, help="output TLE file")
    return parser.parse_args()


def _read_response(response: HTTPResponse) -> str:
    if response.status != 200:
        raise CelesTrakHTTPError(response.status, response.reason)
    return response.read().decode("utf-8").strip()


def fetch_gp(query: str, value: str) -> str:
    query = query.upper()
    url = CELESTRAK_GP_URL.format(query=query, value=quote(value.upper() if query == "GROUP" else value))
    request = Request(url, headers={"User-Agent": "space-debris-iac-research/0.1"})
    with urlopen(request, timeout=30) as response:
        text = _read_response(response)
    if "DOCTYPE html" in text[:200] or "<html" in text[:200].lower():
        raise RuntimeError(f"CelesTrak returned HTML instead of TLE data for {query}={value}")
    return text


def normalize_tle_blocks(text: str, fallback_name: str | None = None) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = fallback_name or f"CATNR {lines[i][2:7].strip()}"
            blocks.append("\n".join([name, lines[i], lines[i + 1]]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            blocks.append("\n".join(lines[i : i + 3]))
            i += 3
        else:
            raise ValueError(f"Invalid TLE response around line {i + 1}: {lines[i][:120]}")
    return blocks


def fetch_tle(catnr: str) -> str:
    text = fetch_gp("CATNR", catnr)
    blocks = normalize_tle_blocks(text, DEFAULT_OBJECTS.get(catnr, f"CATNR {catnr}"))
    if len(blocks) != 1:
        raise ValueError(f"Expected one TLE for CATNR={catnr}, got {len(blocks)}")
    return blocks[0]


def fetch_tle_blocks(catnrs: list[str], groups: list[str], max_objects: int | None = None) -> list[str]:
    blocks: list[str] = []
    for catnr in catnrs:
        blocks.append(fetch_tle(catnr))
    for group in groups:
        blocks.extend(normalize_tle_blocks(fetch_gp("GROUP", group)))

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
    output.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def fetch_to_file(output: Path, catnrs: list[str], groups: list[str], max_objects: int | None = None) -> Path:
    blocks = fetch_tle_blocks(catnrs, groups, max_objects)
    write_tle_file(output, blocks)
    return output


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or f"data/live_tles_{timestamp}.txt")
    output.parent.mkdir(parents=True, exist_ok=True)

    fetch_to_file(output, args.catnr, args.group, args.max_objects)
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()
