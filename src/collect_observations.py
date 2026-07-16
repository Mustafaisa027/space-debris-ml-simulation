from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fetch_tles import PRESETS, fetch_to_file
from space_debris.core import build_satellites, filter_conjunctions, read_tles, simulate_pairs, write_pair_results
from space_debris.provenance import git_commit_hash


ADDITIVE_HISTORY_DEFAULTS = {
    # Historical rows predate the refined-TCA boundary diagnostic. Blank
    # means "not recorded"; treating them as 0 would incorrectly assert that
    # the old coarse-only result was checked and found away from a boundary.
    "tca_boundary_flag": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect repeated TLE snapshots and conjunction observations")
    parser.add_argument("--days", type=float, default=60.0, help="collection duration")
    parser.add_argument("--interval-hours", type=float, default=2.0, help="CelesTrak recommends not polling more often")
    parser.add_argument("--once", action="store_true", help="run one fetch/simulate cycle and exit")
    parser.add_argument("--catnr", nargs="*", default=None, help="NORAD catalog numbers (overrides --preset)")
    parser.add_argument("--group", nargs="*", default=None, help="CelesTrak groups, e.g. STATIONS WEATHER (overrides --preset)")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="leo_mixed",
        help="curated multi-orbit CATNR catalog used when --catnr/--group are not given",
    )
    parser.add_argument("--max-objects", type=int, default=75, help="cap objects to control O(n^2) pair growth")
    parser.add_argument(
        "--provider",
        choices=["auto", "celestrak", "space-track"],
        default="auto",
        help="TLE provider; auto permits credentialed Space-Track fallback",
    )
    parser.add_argument("--snapshot-dir", default="data/tle_snapshots")
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--history", default="outputs/history/conjunction_observations_v2.csv")
    parser.add_argument("--horizon-minutes", type=int, default=720)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--leo-min-altitude-km", type=float, default=160.0)
    parser.add_argument("--leo-max-altitude-km", type=float, default=2000.0)
    parser.add_argument("--candidate-threshold-km", type=float, default=50.0)
    parser.add_argument("--fixed-threshold-km", type=float, default=25.0)
    parser.add_argument("--label-threshold-km", type=float, default=20.0)
    parser.add_argument("--label-relative-velocity-km-s", type=float, default=10.0)
    parser.add_argument("--max-tle-age-hours", type=float, default=336.0)
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_source(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    """Return (catnrs, groups, source_label) for a collection cycle.

    Explicit --catnr/--group override the default. Otherwise the curated
    --preset catalog (leo_mixed by default) is used, since a single
    constellation's near-identical orbital planes rarely produce genuine
    close approaches (see ROADMAP_YOL1.md observations G3-G4).
    """
    if args.catnr or args.group:
        return args.catnr or [], args.group or [], "explicit"
    return list(PRESETS[args.preset]), [], f"preset:{args.preset}"


def write_provenance(path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class HistorySchemaError(RuntimeError):
    """Raised before mutation when accumulated-history columns are incompatible."""


def append_csv(target: Path, source: Path, extra: dict[str, str]) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as src:
        # `source` may carry a '#'-prefixed provenance header (ROADMAP_YOL1.md
        # GOREV 6, see space_debris.provenance) that csv.DictReader must not
        # mistake for the fieldnames row.
        non_comment_lines = (line for line in src if not line.startswith("#"))
        reader = csv.DictReader(non_comment_lines)
        fieldnames = list(extra.keys()) + list(reader.fieldnames or [])
        source_rows = list(reader)
    if not source_rows:
        raise HistorySchemaError(f"Source dataset has no rows: {source}")
    if any(None in row for row in source_rows):
        raise HistorySchemaError(f"Source dataset contains rows wider than its header: {source}")

    existing_rows: list[dict] = []
    if target.exists() and target.stat().st_size:
        with target.open(newline="", encoding="utf-8") as current:
            current_reader = csv.DictReader(current)
            existing_fieldnames = list(current_reader.fieldnames or [])
            added_columns = [name for name in fieldnames if name not in existing_fieldnames]
            additive_upgrade = (
                bool(added_columns)
                and all(name in ADDITIVE_HISTORY_DEFAULTS for name in added_columns)
                and existing_fieldnames == [name for name in fieldnames if name not in added_columns]
            )
            if existing_fieldnames != fieldnames and not additive_upgrade:
                raise HistorySchemaError(
                    "History schema mismatch; refusing to append. "
                    f"expected={fieldnames}, existing={current_reader.fieldnames}"
                )
            existing_rows = list(current_reader)
            if additive_upgrade:
                for row in existing_rows:
                    for name in added_columns:
                        row[name] = ADDITIVE_HISTORY_DEFAULTS[name]
        if any(None in row for row in existing_rows):
            raise HistorySchemaError(
                f"Existing history contains malformed rows wider than its header: {target}"
            )

    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    count = len(source_rows)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            for row in source_rows:
                writer.writerow({**extra, **row})
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count


def collect_once(args: argparse.Namespace) -> int:
    stamp = utc_stamp()
    snapshot_utc = datetime.now(timezone.utc).replace(microsecond=0)
    snapshot_path = Path(args.snapshot_dir) / f"tles_{stamp}.txt"
    run_dir = Path(args.run_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    catnrs, groups, source = resolve_source(args)
    fetch_report: dict = {}
    fetch_to_file(
        snapshot_path,
        catnrs,
        groups,
        args.max_objects,
        provider=args.provider,
        report=fetch_report,
    )
    objects = read_tles(snapshot_path)

    provenance = {
        "collection_id": stamp,
        "source": source,
        "preset": args.preset if source.startswith("preset:") else "",
        "fetched_utc": snapshot_utc.isoformat().replace("+00:00", "Z"),
        "object_count": len(objects),
        "tle_file": str(snapshot_path),
        "git_commit": git_commit_hash(),
        "tle_provider": fetch_report.get("provider", "unknown"),
        "catalog_ids": fetch_report.get("catalog_ids", []),
        "requested_object_count": fetch_report.get("requested_count", len(catnrs)),
    }
    write_provenance(snapshot_path.with_suffix(".json"), provenance)

    satellites = build_satellites(objects)
    rows = simulate_pairs(
        satellites=satellites,
        start_utc=snapshot_utc,
        horizon_minutes=args.horizon_minutes,
        step_minutes=args.step_minutes,
        leo_min_altitude_km=args.leo_min_altitude_km,
        leo_max_altitude_km=args.leo_max_altitude_km,
        fixed_threshold_km=args.fixed_threshold_km,
        label_threshold_km=args.label_threshold_km,
        label_relative_velocity_km_s=args.label_relative_velocity_km_s,
        max_tle_age_hours=args.max_tle_age_hours,
    )
    conjunctions = filter_conjunctions(rows, args.candidate_threshold_km)

    config_summary = (
        f"horizon={args.horizon_minutes}min step={args.step_minutes}min "
        f"label_threshold_km={args.label_threshold_km} "
        f"label_relative_velocity_km_s={args.label_relative_velocity_km_s} "
        f"candidate_threshold_km={args.candidate_threshold_km}"
    )
    dataset_path = run_dir / "conjunction_dataset.csv"
    conjunction_path = run_dir / "identified_conjunctions.csv"
    write_pair_results(dataset_path, rows, source=source, config_summary=config_summary)
    write_pair_results(conjunction_path, conjunctions, source=source, config_summary=config_summary)

    count = append_csv(
        Path(args.history),
        dataset_path,
        {
            "collection_id": stamp,
            "source": provenance["source"],
            "preset": provenance["preset"],
            "fetched_utc": provenance["fetched_utc"],
            "tle_file": str(snapshot_path),
            "object_count": str(len(objects)),
        },
    )
    print(
        f"{stamp}: objects={len(objects)} pairs={len(rows)} "
        f"conjunctions={len(conjunctions)} appended={count}"
    )
    return count


def main() -> None:
    args = parse_args()
    deadline = time.time() + args.days * 24 * 3600
    interval_seconds = max(args.interval_hours, 2.0) * 3600

    while True:
        try:
            collect_once(args)
        except Exception as exc:
            # A single bad cycle (fetch failure, transient CelesTrak block,
            # etc.) must not take down a 60-day scheduled task: log it and
            # pick back up on the next interval instead of crashing.
            print(f"{utc_stamp()}: collector error (continuing next cycle): {exc}")
            if args.once:
                raise
        if args.once or time.time() >= deadline:
            break
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
