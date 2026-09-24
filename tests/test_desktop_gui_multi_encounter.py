from __future__ import annotations

from datetime import timedelta
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from desktop_gui.app_window import OrbitalSentinelWindow
from desktop_gui.data_loader import PROJECT_ROOT, load_simulation_data
from desktop_gui.renderer_probe import RendererProbeResult, select_3d_mode
from desktop_gui.scenes.orbit_scene import (
    DISTANCE_CONSISTENCY_TOLERANCE_KM,
    SELECTED_REBUILT_ACTOR_NAMES,
    SELECTED_ORBIT_LINE_WIDTH,
    SELECTED_ORBIT_OPACITY,
    TCA_DIRECTION_ARROW_LENGTH_KM,
    OrbitScene,
    build_all_encounter_geometries,
    compute_tca_camera_frame,
    encounter_key,
    overview_visual_style,
    selected_pair_ids,
)
from desktop_gui.tle_catalog import load_tle_catalog, parse_utc, propagate


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def encounter_data():
    data = load_simulation_data()
    catalog = load_tle_catalog(PROJECT_ROOT / data.replay["tle_source"])
    geometries = build_all_encounter_geometries(data, catalog)
    return data, catalog, geometries


def test_exactly_eight_overlays_map_deterministically(encounter_data) -> None:
    data, _catalog, geometries = encounter_data
    assert len(geometries) == len(data.scenarios) == 8
    assert [geometry.key for geometry in geometries] == [f"E{i:02d}" for i in range(1, 9)]
    assert [geometry.scenario_id for geometry in geometries] == [item.id for item in data.scenarios]


def test_each_overlay_uses_its_own_canonical_tca(encounter_data) -> None:
    data, _catalog, geometries = encounter_data
    assert [geometry.tca_minute for geometry in geometries] == [
        scenario.tca_minute for scenario in data.scenarios
    ]


def test_overview_midpoints_derive_from_actual_sgp4_tca_positions(encounter_data) -> None:
    data, catalog, geometries = encounter_data
    lookup = {item.catalog_id: item for item in catalog}
    for scenario, geometry in zip(data.scenarios, geometries):
        when = parse_utc(str(scenario.value("snapshot_utc"))) + timedelta(
            minutes=scenario.tca_minute
        )
        primary = propagate(lookup[selected_pair_ids(scenario)[0]], when)
        secondary = propagate(lookup[selected_pair_ids(scenario)[1]], when)
        assert primary is not None and secondary is not None
        assert geometry.primary_tca == pytest.approx(primary)
        assert geometry.secondary_tca == pytest.approx(secondary)
        assert geometry.midpoint == pytest.approx((primary + secondary) / 2.0)


def test_selected_encounter_is_visually_distinct_in_all_mode() -> None:
    selected = overview_visual_style("ALL", True)
    background = overview_visual_style("ALL", False)
    assert selected.marker_opacity > background.marker_opacity
    assert selected.path_opacity > background.path_opacity
    assert selected.path_line_width > background.path_line_width
    assert selected.marker_size > background.marker_size


def test_current_separation_visibility_is_mode_specific(app: QApplication) -> None:
    class FakeProperty:
        def SetOpacity(self, _value: float) -> None:
            pass

        def SetLineWidth(self, _value: float) -> None:
            pass

        def SetPointSize(self, _value: float) -> None:
            pass

    class FakeActor:
        def __init__(self) -> None:
            self.visible = False
            self.property = FakeProperty()

        def SetVisibility(self, visible: bool) -> None:
            self.visible = bool(visible)

        def GetProperty(self) -> FakeProperty:
            return self.property

    class FakePlotter:
        def __init__(self, actor: FakeActor) -> None:
            self.actors = {"current-separation": actor}

        def render(self) -> None:
            pass

    scene = OrbitScene(load_simulation_data(), enable_3d=False)
    actor = FakeActor()
    scene.has_3d_backend = True
    scene._plotter = FakePlotter(actor)
    label_points = FakeActor()
    scene._plotter.actors["selected-pair-labels-points"] = label_points
    scene._encounter_actor = actor
    for mode, expected in (("ALL", False), ("SELECTED", True), ("TCA", False)):
        scene.view_mode = mode
        scene._apply_visual_hierarchy()
        assert actor.visible is expected
        assert label_points.visible is expected
    scene._plotter = None
    scene.has_3d_backend = False
    scene.close()


def test_all_and_selected_modes_preserve_background_encounters() -> None:
    for mode in ("ALL", "SELECTED"):
        styles = [overview_visual_style(mode, selected=index == 2) for index in range(8)]
        assert len(styles) == 8
        assert all(style.marker_opacity > 0.0 for style in styles)
        assert all(style.path_opacity > 0.0 for style in styles)
        assert all(style.label_visible for style in styles)


def test_tca_mode_keeps_only_selected_detail_dominant() -> None:
    selected = overview_visual_style("TCA", True)
    background = overview_visual_style("TCA", False)
    assert selected.marker_opacity > background.marker_opacity
    assert selected.path_opacity > background.path_opacity
    assert not selected.label_visible and not background.label_visible


def test_scenario_change_clears_all_stale_detailed_actors(
    encounter_data, app: QApplication, monkeypatch
) -> None:
    data, catalog, geometries = encounter_data

    class FakePlotter:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def remove_actor(self, name: str, reset_camera: bool = False) -> None:
            self.removed.append(name)

    scene = OrbitScene(data, enable_3d=False)
    scene.has_3d_backend = True
    scene._plotter = FakePlotter()
    scene._catalog = catalog
    scene._encounter_geometries = geometries
    monkeypatch.setattr("desktop_gui.scenes.orbit_scene.orbit_track", lambda *args: np.empty((0, 3)))
    monkeypatch.setattr(scene, "_draw_approach_geometry", lambda *args: None)
    monkeypatch.setattr(scene, "_update_overview_status", lambda: None)
    monkeypatch.setattr(scene, "_apply_visual_hierarchy", lambda: None)
    scene.set_scenario(1)
    assert scene._plotter.removed == list(SELECTED_REBUILT_ACTOR_NAMES)
    assert scene.scenario_index == 1
    scene._plotter = None
    scene.has_3d_backend = False
    scene.close()


def test_primary_secondary_ids_remain_source_mapped(encounter_data) -> None:
    data, _catalog, _geometries = encounter_data
    for scenario in data.scenarios:
        assert selected_pair_ids(scenario) == (
            str(scenario.value("catalog_id_1")),
            str(scenario.value("catalog_id_2")),
        )


def test_declared_miss_matches_sgp4_tca_with_explicit_tolerance(encounter_data) -> None:
    data, _catalog, geometries = encounter_data
    assert DISTANCE_CONSISTENCY_TOLERANCE_KM == 0.001
    for scenario, geometry in zip(data.scenarios, geometries):
        separation = float(np.linalg.norm(geometry.primary_tca - geometry.secondary_tca))
        assert abs(separation - scenario.minimum_distance_km) <= DISTANCE_CONSISTENCY_TOLERANCE_KM


def test_distance_series_minimum_and_time_match_all_scenarios(encounter_data) -> None:
    data, _catalog, _geometries = encounter_data
    for scenario in data.scenarios:
        minimum = min(scenario.distance_series, key=lambda point: point.distance_km)
        assert minimum.distance_km == pytest.approx(scenario.minimum_distance_km, abs=1e-12)
        assert minimum.minute == pytest.approx(scenario.tca_minute, abs=1e-6)


def test_series_interpolation_matches_sgp4_at_playback_samples(encounter_data) -> None:
    data, catalog, _geometries = encounter_data
    lookup = {item.catalog_id: item for item in catalog}
    for scenario in data.scenarios:
        center = parse_utc(str(scenario.value("snapshot_utc")))
        primary = lookup[selected_pair_ids(scenario)[0]]
        secondary = lookup[selected_pair_ids(scenario)[1]]
        sample_minutes = (0.0, 180.0, 360.0, 540.0, scenario.tca_minute, 720.0)
        for minute in sample_minutes:
            when = center + timedelta(minutes=minute)
            primary_position = propagate(primary, when)
            secondary_position = propagate(secondary, when)
            assert primary_position is not None and secondary_position is not None
            geometric = float(np.linalg.norm(primary_position - secondary_position))
            assert abs(scenario.distance_at(minute) - geometric) <= DISTANCE_CONSISTENCY_TOLERANCE_KM


def test_current_distance_display_uses_authoritative_series(app: QApplication) -> None:
    data = load_simulation_data()
    scene = OrbitScene(data, enable_3d=False)
    minute = 123.456
    scene.controller.seek_fraction(minute / scene.controller.duration_minutes)
    expected = data.scenarios[0].distance_at(minute)
    assert scene.distance_label.text() == f"CURRENT DISTANCE [SERIES] {expected:.3f} km"
    scene.close()


def test_go_to_tca_and_tca_mode_stay_exactly_synchronized(app: QApplication) -> None:
    data = load_simulation_data()
    scene = OrbitScene(data, enable_3d=False)
    scene.set_scenario(3)
    scene.go_to_tca()
    assert scene.controller.current_minute == pytest.approx(data.scenarios[3].tca_minute)
    assert scene.timeline.current_fraction == pytest.approx(scene.timeline.tca_fraction)
    scene.controller.play()
    scene.set_visualization_mode("TCA")
    assert not scene.controller.is_playing
    assert scene.camera_selector.currentText() == "CAMERA: TCA"
    assert scene.controller.current_minute == pytest.approx(data.scenarios[3].tca_minute)
    assert scene.timeline.current_fraction == pytest.approx(scene.timeline.tca_fraction)
    assert scene.tca_time_label.text() == "TCA"
    assert scene.distance_label.text() == (
        f"CURRENT DISTANCE [SERIES] {data.scenarios[3].minimum_distance_km:.3f} km"
    )
    scene.set_scenario(5)
    assert scene.controller.current_minute == pytest.approx(data.scenarios[5].tca_minute)
    scene.close()


def test_tca_direction_arrows_remain_secondary_scale() -> None:
    assert TCA_DIRECTION_ARROW_LENGTH_KM == 25.0


def test_tca_role_offsets_only_move_label_anchors(app: QApplication, encounter_data) -> None:
    from types import SimpleNamespace

    data, _, geometries = encounter_data
    scene = OrbitScene(data, enable_3d=False)
    scene._encounter_geometries = geometries
    physical = np.vstack((geometries[0].primary_tca, geometries[0].secondary_tca))
    scene._encounter_points = SimpleNamespace(points=physical.copy())
    scene._encounter_labels = SimpleNamespace(points=physical.copy())
    scene.view_mode = "TCA"
    scene._update_pair_labels()
    offsets = scene._encounter_labels.points - physical
    assert np.allclose(np.linalg.norm(offsets, axis=1), 18.0)
    assert np.allclose(offsets[0], -offsets[1])
    assert np.array_equal(scene._encounter_points.points, physical)
    scene.view_mode = "SELECTED"
    scene._update_pair_labels()
    assert np.array_equal(scene._encounter_labels.points, physical)
    scene.close()


def test_paused_scenario_switch_synchronizes_detail(app: QApplication) -> None:
    window = OrbitalSentinelWindow(load_simulation_data(), enable_3d=False)
    scene = window.orbit_scene
    detail = window.conjunction_scene
    scene.set_visualization_mode("TCA")
    for index in (0, 3, 5, 7):
        scene.set_scenario(index)
        scenario = scene.data.scenarios[index]
        assert detail.selector.currentIndex() == index
        assert detail.live_labels["simulation_time"].text() == f"{scenario.tca_minute:.3f} min"
        assert detail.live_labels["tca_offset"].text() == "TCA"
        assert detail.live_labels["current_distance"].text() == f"{scenario.distance_at(scenario.tca_minute):.3f} km"
    scene.set_visualization_mode("SELECTED")
    detail.selector.setCurrentIndex(0)
    assert detail.live_labels["current_distance"].text() == f"{scene.data.scenarios[0].distance_at(0):.3f} km"
    window.close()


def test_tca_snapshot_locks_time_controls_and_restores_playback(app: QApplication) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    scene = OrbitScene(load_simulation_data(), enable_3d=False)
    scene.controller.play()
    scene.set_visualization_mode("TCA")
    minute = scene.controller.current_minute
    assert minute == pytest.approx(scene.data.scenarios[0].tca_minute, abs=1e-9)
    assert scene.camera_selector.currentIndex() == 4
    assert scene.play_button.isEnabled()
    for control in (scene.slider, scene.timeline, scene.speed):
        assert not control.isEnabled()
        QTest.mouseClick(control, Qt.MouseButton.LeftButton)
    assert not scene.controller.is_playing
    assert scene.controller.current_minute == minute
    assert scene.tca_time_label.text() == "TCA"
    started = []
    scene.controller.state_changed.connect(
        lambda playing: started.append((scene.view_mode, scene.controller.current_minute)) if playing else None
    )
    scene.play_button.click()
    assert started == [("SELECTED", minute)]
    assert scene.view_mode == "SELECTED"
    assert scene.controller.is_playing
    assert scene.controller.current_minute == minute
    assert scene.camera_selector.currentIndex() == 4
    for control in (scene.slider, scene.timeline, scene.speed):
        assert control.isEnabled()
    scene.controller.step_fixed(0.01)
    assert scene.controller.current_minute > minute
    scene.play_button.click()
    assert not scene.controller.is_playing
    for mode in ("SELECTED", "ALL"):
        scene.set_visualization_mode(mode)
        for control in (scene.play_button, scene.slider, scene.timeline, scene.speed):
            assert control.isEnabled()
        scene.play_button.click()
        assert scene.controller.is_playing
        scene.controller.pause()
    scene.close()


def test_tca_camera_target_and_frame_derive_from_selected_geometry(encounter_data) -> None:
    _data, _catalog, geometries = encounter_data
    distances = []
    for geometry in geometries:
        frame = compute_tca_camera_frame(geometry)
        distances.append(frame.distance)
        assert frame.focal_point == pytest.approx(geometry.midpoint)
        assert np.linalg.norm(geometry.primary_tca - np.asarray(frame.focal_point)) <= frame.framing_radius
        assert np.linalg.norm(geometry.secondary_tca - np.asarray(frame.focal_point)) <= frame.framing_radius
        assert np.linalg.norm(np.asarray(frame.position)) >= 6371.0 + 180.0
        assert frame.distance > frame.framing_radius
        assert 220.0 <= frame.distance <= 360.0
    assert len({round(value, 3) for value in distances}) > 1


def test_selected_full_orbits_remain_subordinate_to_approach_paths() -> None:
    assert SELECTED_ORBIT_LINE_WIDTH == pytest.approx(1.0)
    assert SELECTED_ORBIT_OPACITY == pytest.approx(0.14)
    assert SELECTED_ORBIT_LINE_WIDTH < 3.0
    assert SELECTED_ORBIT_OPACITY < 0.30


def test_existing_pages_safe_mode_and_renderer_selection_remain_intact(app: QApplication) -> None:
    window = OrbitalSentinelWindow(load_simulation_data(), enable_3d=False)
    assert window.stack.count() == 6
    assert not window.orbit_scene.has_3d_backend
    window.close()
    success = RendererProbeResult(True, "success", {"renderer": "test"})
    assert select_3d_mode(False, probe=lambda: success)[0]
    assert not select_3d_mode(True, probe=lambda: success)[0]
