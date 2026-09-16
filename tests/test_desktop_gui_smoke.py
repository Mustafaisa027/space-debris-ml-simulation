from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from desktop_gui.app_window import OrbitalSentinelWindow, load_stylesheet
from desktop_gui.data_loader import PROJECT_ROOT, load_simulation_data
from desktop_gui.playback.timeline_controller import TimelineController
from desktop_gui.tle_catalog import load_tle_catalog


def test_simulation_payload_contract() -> None:
    data = load_simulation_data()

    assert len(data.scenarios) == 8
    assert len(data.models) == 7
    assert len(data.quality_gates) == 3
    assert all(scenario.distance_series for scenario in data.scenarios)
    assert data.replay["tle_source"]


def test_replay_tle_catalog_contains_all_scenario_objects() -> None:
    data = load_simulation_data()
    catalog = load_tle_catalog(PROJECT_ROOT / data.replay["tle_source"])
    catalog_ids = {item.catalog_id for item in catalog}

    assert len(catalog) == 75
    for scenario in data.scenarios:
        assert str(scenario.value("catalog_id_1")) in catalog_ids
        assert str(scenario.value("catalog_id_2")) in catalog_ids


def test_stylesheet_resolves_theme_tokens() -> None:
    stylesheet = load_stylesheet()

    assert "@BACKGROUND@" not in stylesheet
    assert "#05070D" in stylesheet


def test_window_builds_all_six_scenes_offscreen() -> None:
    app = QApplication.instance() or QApplication([])
    window = OrbitalSentinelWindow(load_simulation_data(), enable_3d=False)

    assert window.stack.count() == 6
    assert len(window.nav_buttons) == 6
    assert not window.orbit_scene.has_3d_backend

    window.show_scene(5, animate=False)
    assert window.stack.currentIndex() == 5
    window.close()
    app.processEvents()


def test_timeline_controller_seek_and_step() -> None:
    app = QApplication.instance() or QApplication([])
    controller = TimelineController(720.0)

    controller.seek_fraction(0.5)
    assert controller.current_minute == 360.0
    assert controller.step_fixed(1.0) == 378.0
    controller.pause()
    app.processEvents()
