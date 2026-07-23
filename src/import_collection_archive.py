from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from space_debris.archive import import_collection_archive
from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Verify data-collection bundles and import canonical TLE/run artifacts"
    )
    parser.add_argument("archive_root", type=Path, help="checkout/worktree root of data-collection")
    parser.add_argument("--config", default=str(config.path))
    parser.add_argument("--snapshot-dir", type=Path, default=Path(config.snapshot_dir))
    parser.add_argument("--run-root", type=Path, default=Path(config.run_root))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(config.archive_import_report),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--trusted-source-ref",
        default="origin/main",
        help="Git ref that must contain every bundle source commit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    report = import_collection_archive(
        args.archive_root,
        args.snapshot_dir,
        args.run_root,
        config,
        verify_only=args.verify_only,
        require_clean_checkout=True,
        trusted_source_ref=args.trusted_source_ref,
    )
    _write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
