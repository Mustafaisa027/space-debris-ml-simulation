from __future__ import annotations

import json
import os
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from desktop_gui.app_window import OrbitalSentinelWindow
from desktop_gui.data_loader import DistancePoint, Scenario, load_simulation_data
from desktop_gui.main import build_parser
from desktop_gui.renderer_probe import (
    RendererProbeResult,
    run_renderer_probe,
    select_3d_mode,
)
from desktop_gui.scenes.conjunction_panel import ConjunctionPanel, engineering_value
from desktop_gui.scenes.orbit_scene import (
    OrbitScene,
    approach_sample_minutes,
    selected_pair_ids,
    tca_label_text,
)
from desktop_gui.scenes.timeline_widget import normalized_series_points


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_selected_encounter_maps_primary_secondary_and_real_tca() -> None:
    scenario = load_simulation_data().scenarios[0]
    assert selected_pair_ids(scenario) == (
        str(scenario.value("catalog_id_1")),
        str(scenario.value("catalog_id_2")),
    )
    assert scenario.tca_minute == float(scenario.value("time_to_tca_min"))


def test_tca_label_uses_encounter_minimum_and_velocity() -> None:
    scenario = load_simulation_data().scenarios[0]
    label = tca_label_text(scenario)
    assert f"MISS DISTANCE: {scenario.minimum_distance_km:.3f} km" in label
    assert f"RELATIVE VELOCITY: {float(scenario.value('relative_velocity_km_s')):.3f} km/s" in label


def test_graph_x_coordinates_use_physical_time_and_keep_irregular_spacing() -> None:
    scenario = Scenario(
        raw={"id": "irregular", "label": "Irregular", "time_to_tca_min": 2.0, "min_distance_km": 1.0},
        distance_series=(
            DistancePoint(0.0, "", 5.0),
            DistancePoint(1.0, "", 3.0),
            DistancePoint(2.0, "", 1.0),
            DistancePoint(10.0, "", 8.0),
        ),
    )
    x_values = [point[0] for point in normalized_series_points(scenario)]
    assert x_values == pytest.approx([0.0, 0.1, 0.2, 1.0])
    assert x_values[2] - x_values[1] != pytest.approx(x_values[3] - x_values[2])


def test_tca_marker_and_current_cursor_share_canonical_time(app: QApplication) -> None:
    data = load_simulation_data()
    scene = OrbitScene(data, enable_3d=False)
    scenario = data.scenarios[0]
    scene.controller.seek_fraction(scenario.tca_minute / scene.controller.duration_minutes)
    assert scene.timeline.current_fraction == pytest.approx(scene.timeline.tca_fraction)
    scene.close()


def test_approach_window_uses_covered_samples_and_real_tca() -> None:
    scenario = load_simulation_data().scenarios[0]
    samples = approach_sample_minutes(scenario)
    assert scenario.tca_minute in samples
    assert min(samples) >= scenario.distance_series[0].minute
    assert max(samples) <= scenario.distance_series[-1].minute


def test_selected_pair_redraw_does_not_remove_catalog_actors(monkeypatch, app: QApplication) -> None:
    class FakePlotter:
        def __init__(self) -> None:
            self.actors = {"catalog-points", "catalog-orbits"}

        def remove_actor(self, name: str, reset_camera: bool = False) -> None:
            self.actors.discard(name)

    scene = OrbitScene(load_simulation_data(), enable_3d=False)
    scene.has_3d_backend = True
    scene._plotter = FakePlotter()
    scene._catalog = ()
    monkeypatch.setattr(scene, "_draw_approach_geometry", lambda *args: None)
    scene._draw_selected_orbits()
    assert scene._plotter.actors == {"catalog-points", "catalog-orbits"}
    scene._plotter = None
    scene.close()


def test_no_3d_flag_skips_renderer_initialization(monkeypatch, app: QApplication) -> None:
    assert build_parser().parse_args(["--no-3d"]).no_3d
    monkeypatch.setattr(OrbitScene, "_build_3d", lambda *args: pytest.fail("3D initialized"))
    window = OrbitalSentinelWindow(load_simulation_data(), enable_3d=False)
    assert not window.orbit_scene.has_3d_backend
    window.close()


def test_failed_probe_selects_safe_mode() -> None:
    failed = RendererProbeResult(False, "failed", {}, "OpenGL unavailable")
    enabled, result = select_3d_mode(False, probe=lambda: failed)
    assert not enabled
    assert result.error == "OpenGL unavailable"


def test_successful_probe_enables_3d() -> None:
    success = RendererProbeResult(True, "success", {"renderer": "test GPU"})
    enabled, result = select_3d_mode(False, probe=lambda: success)
    assert enabled
    assert result.status == "success"


def test_renderer_probe_timeout_fails_safely(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 1.0))

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_renderer_probe(0.1)
    assert not result.success
    assert result.status == "timeout"


def test_renderer_probe_success_payload(monkeypatch) -> None:
    payload = {"success": True, "diagnostics": {"opengl_version": "4.6"}}
    completed = subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    result = run_renderer_probe()
    assert result.success
    assert result.diagnostics["opengl_version"] == "4.6"


def test_missing_optional_numerical_data_is_na(app: QApplication) -> None:
    assert engineering_value(None, "km/s") == "N/A"
    panel = ConjunctionPanel(load_simulation_data())
    panel.set_playback_state(0, 0.0, None, None, None)
    assert panel.live_labels["primary_velocity"].text() == "N/A"
    panel.close()


def test_existing_pages_still_instantiate(app: QApplication) -> None:
    window = OrbitalSentinelWindow(load_simulation_data(), enable_3d=False)
    assert window.stack.count() == 6
    assert all(window.stack.widget(index) is not None for index in range(6))
    window.close()
