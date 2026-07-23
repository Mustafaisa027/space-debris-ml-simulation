from __future__ import annotations

from datetime import datetime

import pytest

from collection_window_status import window_status
from space_debris.experiment import load_experiment_config


def test_frozen_collection_window_is_half_open():
    config = load_experiment_config("config/experiment_60_days.json")
    start = datetime.fromisoformat(config.collection_start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(config.collection_end_utc.replace("Z", "+00:00"))

    assert window_status(config, start)["collect"] is True
    assert window_status(config, end)["collect"] is False
    assert window_status(config, start.replace(year=2025))["status"] == "before_window"
    assert window_status(config, end)["status"] == "window_complete"


def test_window_status_rejects_naive_now():
    config = load_experiment_config("config/experiment_60_days.json")

    with pytest.raises(ValueError, match="timezone-aware"):
        window_status(config, datetime(2026, 7, 16))

