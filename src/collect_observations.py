from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from fetch_tles import CelesTrakHTTPError, DEFAULT_OBJECTS, fetch_to_file
from space_debris.core import build_satellites, filter_conjunctions, read_tles, simulate_pairs, write_pair_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect repeated TLE snapshots and conjunction observations")
    parser.add_argument("--days", type=float, default=60.0, help="collection duration")
    parser.add_argument("--interval-hours", type=float, default=2.0, help="CelesTrak recommends not polling more often")
    parser.add_argument("--once", action="store_true", help="run one fetch/simulate cycle and exit")
    parser.add_argument("--catnr", nargs="*", default=list(DEFAULT_OBJECTS), help="NORAD catalog numbers")
    parser.add_argument("--group", nargs="*", default=[], help="CelesTrak groups, e.g. STATIONS WEATHER")
    parser.add_argument("--max-objects", type=int, default=75, help="cap objects to control O(n^2) pair growth")
    parser.add_argument("--snapshot-dir", default="data/tle_snapshots")
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--history", default="outputs/history/conjunction_observations.csv")
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


def append_csv(target: Path, source: Path, extra: dict[str, str]) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = list(extra.keys()) + list(reader.fieldnames or [])
        write_header = not target.exists() or target.stat().st_size == 0
        count = 0
        with target.open("a", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in reader:
                writer.writerow({**extra, **row})
                count += 1
    return count


def collect_once(args: argparse.Namespace) -> int:
    stamp = utc_stamp()
    snapshot_utc = datetime.now(timezone.utc).replace(microsecond=0)
    snapshot_path = Path(args.snapshot_dir) / f"tles_{stamp}.txt"
    run_dir = Path(args.run_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    fetch_to_file(snapshot_path, args.catnr, args.group, args.max_objects)
    objects = read_tles(snapshot_path)
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

    dataset_path = run_dir / "conjunction_dataset.csv"
    conjunction_path = run_dir / "identified_conjunctions.csv"
    write_pair_results(dataset_path, rows)
    write_pair_results(conjunction_path, conjunctions)

    count = append_csv(
        Path(args.history),
        dataset_path,
        {
            "collection_id": stamp,
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
        except CelesTrakHTTPError as exc:
            print(f"collector fatal: {exc}")
            if exc.status_code in {403, 404}:
                print("Stopping to avoid repeated blocked or invalid CelesTrak requests.")
                break
        except Exception as exc:
            print(f"collector error: {exc}")
        if args.once or time.time() >= deadline:
            break
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
