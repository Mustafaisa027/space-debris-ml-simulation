from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from space_debris.experiment import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config


def window_status(config, now_utc: datetime) -> dict:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    now_utc = now_utc.astimezone(timezone.utc)
    if config.collection_start_utc is None:
        return {"status": "unbounded", "collect": True, "now_utc": now_utc.isoformat()}
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))
    if now_utc < start:
        status, collect = "before_window", False
    elif now_utc >= end:
        status, collect = "window_complete", False
    else:
        status, collect = "active", True
    return {
        "status": status,
        "collect": collect,
        "now_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "start_utc": config.collection_start_utc,
        "end_utc": config.collection_end_utc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the frozen collection window")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    report = window_status(config, datetime.now(timezone.utc))
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as stream:
            stream.write(f"collect={'true' if report['collect'] else 'false'}\n")
            stream.write(f"status={report['status']}\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

