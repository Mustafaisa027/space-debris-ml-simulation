from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_debris.experiment import ACTIVE_EXPERIMENT_CONFIG, load_experiment_config
from space_debris.finalization import (
    prepare_checkpoint,
    record_stage,
    stage_is_reusable,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fingerprint-bound checkpoint journal for IAC finalization"
    )
    parser.add_argument("command", choices=("prepare", "check", "mark"))
    parser.add_argument("--config", default=str(ACTIVE_EXPERIMENT_CONFIG))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--output", action="append", type=Path, default=[])
    args = parser.parse_args()
    config = load_experiment_config(args.config)

    if args.command == "prepare":
        if args.archive_root is None:
            parser.error("prepare requires --archive-root")
        report = prepare_checkpoint(args.checkpoint, args.archive_root, config)
        result = {
            "status": "prepared",
            "checkpoint": str(args.checkpoint.resolve()),
            "input_fingerprint_sha256": report["input_fingerprint_sha256"],
        }
    elif args.command == "check":
        if not args.stage or not args.output:
            parser.error("check requires --stage and at least one --output")
        reusable = stage_is_reusable(args.checkpoint, args.stage, args.output)
        result = {"status": "reusable" if reusable else "run", "reusable": reusable}
    else:
        if not args.stage or not args.output:
            parser.error("mark requires --stage and at least one --output")
        record_stage(args.checkpoint, args.stage, args.output)
        result = {"status": "recorded", "stage": args.stage}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
