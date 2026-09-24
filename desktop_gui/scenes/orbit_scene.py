from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QElapsedTimer, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from desktop_gui.data_loader import Scenario, SimulationData
from desktop_gui.playback.camera_sequences import CameraKeyframe, targeted_camera
from desktop_gui.playback.timeline_controller import TimelineController
from desktop_gui.renderer_probe import append_renderer_startup_error
from desktop_gui.scenes.timeline_widget import TimelineWidget
from desktop_gui.theme.tokens import AMBER, CYAN, RED
from desktop_gui.tle_catalog import (
    TleObject,
    greenwich_angle_deg,
    load_tle_catalog,
    orbit_track,
    parse_utc,
    propagate,
    propagate_state,
)


def selected_pair_ids(scenario: Scenario) -> tuple[str, str]:
    return str(scenario.value("catalog_id_1")), str(scenario.value("catalog_id_2"))


def approach_sample_minutes(scenario: Scenario, half_window_minutes: float = 15.0) -> tuple[float, ...]:
    """Use only the encounter's covered time interval plus its canonical TCA."""
    start = max(scenario.distance_series[0].minute, scenario.tca_minute - half_window_minutes)
    end = min(scenario.distance_series[-1].minute, scenario.tca_minute + half_window_minutes)
    covered = [point.minute for point in scenario.distance_series if start <= point.minute <= end]
    covered.append(scenario.tca_minute)
    return tuple(sorted(set(covered)))


def overview_sample_minutes(
    scenario: Scenario,
    half_window_minutes: float = 2.0,
    sample_count: int = 7,
) -> tuple[float, ...]:
    """Bound a lightweight rendered path to the scenario's real coverage."""
    start = max(scenario.distance_series[0].minute, scenario.tca_minute - half_window_minutes)
    end = min(scenario.distance_series[-1].minute, scenario.tca_minute + half_window_minutes)
    return tuple(float(value) for value in np.linspace(start, end, max(sample_count, 2)))


def tca_clock_text(current_minute: float, tca_minute: float) -> str:
    delta_seconds = round((float(current_minute) - float(tca_minute)) * 60.0)
    if delta_seconds == 0:
        return "TCA"
    sign = "+" if delta_seconds > 0 else "-"
    hours, remainder = divmod(abs(delta_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"TCA {sign} {hours:02d}:{minutes:02d}:{seconds:02d}"


def tca_label_text(scenario: Scenario) -> str:
    relative_velocity = scenario.value("relative_velocity_km_s")
    velocity_text = (
        f"{float(relative_velocity):.3f} km/s" if relative_velocity is not None else "N/A"
    )
    return (
        f"TCA\nMISS DISTANCE: {scenario.minimum_distance_km:.3f} km"
        f"\nRELATIVE VELOCITY: {velocity_text}"
    )


@dataclass(frozen=True)
class EncounterGeometry:
    index: int
    key: str
    scenario_id: str
    tca_minute: float
    primary_tca: np.ndarray
    secondary_tca: np.ndarray
    midpoint: np.ndarray
    primary_path: np.ndarray
    secondary_path: np.ndarray
    primary_direction: np.ndarray
    secondary_direction: np.ndarray


@dataclass(frozen=True)
class TcaCameraFrame:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    distance: float
    framing_radius: float


@dataclass(frozen=True)
class OverviewVisualStyle:
    marker_opacity: float
    path_opacity: float
    label_visible: bool
    marker_size: float
    path_line_width: float


SELECTED_REBUILT_ACTOR_NAMES = (
    "orbit-primary",
    "orbit-secondary",
    "approach-primary-past",
    "approach-primary-future",
    "approach-secondary-past",
    "approach-secondary-future",
    "direction-primary",
    "direction-secondary",
    "tca-points",
    "tca-separation",
    "tca-label",
)
DISTANCE_CONSISTENCY_TOLERANCE_KM = 0.001
SELECTED_ORBIT_LINE_WIDTH = 1.0
SELECTED_ORBIT_OPACITY = 0.14
TCA_DIRECTION_ARROW_LENGTH_KM = 25.0
TCA_LABEL_OFFSET_KM = 45.0
TCA_MISS_LINE_WIDTH = 8.0


def overview_visual_style(mode: str, selected: bool) -> OverviewVisualStyle:
    normalized = mode.upper()
    if normalized == "ALL":
        return OverviewVisualStyle(
            *(1.0, 0.90, True, 13.0, 4.0)
            if selected
            else (0.62, 0.58, True, 9.0, 3.0)
        )
    if normalized == "SELECTED":
        return OverviewVisualStyle(
            *(0.52, 0.18, True, 10.0, 2.0)
            if selected
            else (0.22, 0.07, True, 9.0, 2.0)
        )
    if normalized == "TCA":
        return OverviewVisualStyle(
            *(0.30, 0.06, False, 9.0, 2.0)
            if selected
            else (0.08, 0.02, False, 9.0, 2.0)
        )
    raise ValueError(f"Unknown visualization mode: {mode}")


def encounter_key(index: int) -> str:
    return f"E{int(index) + 1:02d}"


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude > 1e-9:
        return vector / magnitude
    if fallback is not None:
        return _unit(fallback)
    return np.array([0.0, 0.0, 1.0], dtype=float)


def build_encounter_geometry(
    scenario: Scenario,
    index: int,
    catalog_by_id: dict[str, TleObject],
) -> EncounterGeometry:
    primary = catalog_by_id[selected_pair_ids(scenario)[0]]
    secondary = catalog_by_id[selected_pair_ids(scenario)[1]]
    center = parse_utc(str(scenario.value("snapshot_utc")))
    minutes = overview_sample_minutes(scenario)

    def path_for(item: TleObject) -> np.ndarray:
        points = [
            propagate(item, center + timedelta(minutes=minute)) for minute in minutes
        ]
        valid = [point for point in points if point is not None]
        if not valid:
            raise ValueError(f"No valid SGP4 path for {scenario.id}/{item.catalog_id}")
        return np.vstack(valid)

    tca_when = center + timedelta(minutes=scenario.tca_minute)
    primary_state = propagate_state(primary, tca_when)
    secondary_state = propagate_state(secondary, tca_when)
    if primary_state is None or secondary_state is None:
        raise ValueError(f"No valid SGP4 TCA state for {scenario.id}")
    primary_tca, primary_velocity = primary_state
    secondary_tca, secondary_velocity = secondary_state
    return EncounterGeometry(
        index=index,
        key=encounter_key(index),
        scenario_id=scenario.id,
        tca_minute=scenario.tca_minute,
        primary_tca=primary_tca,
        secondary_tca=secondary_tca,
        midpoint=(primary_tca + secondary_tca) / 2.0,
        primary_path=path_for(primary),
        secondary_path=path_for(secondary),
        primary_direction=_unit(primary_velocity),
        secondary_direction=_unit(secondary_velocity),
    )


def build_all_encounter_geometries(
    data: SimulationData,
    catalog: tuple[TleObject, ...],
) -> tuple[EncounterGeometry, ...]:
    lookup = {item.catalog_id: item for item in catalog}
    return tuple(
        build_encounter_geometry(scenario, index, lookup)
        for index, scenario in enumerate(data.scenarios)
    )


def compute_tca_camera_frame(geometry: EncounterGeometry) -> TcaCameraFrame:
    """Create an outward, geometry-derived close view without moving physical points."""
    separation = geometry.secondary_tca - geometry.primary_tca
    radial = _unit(geometry.midpoint)
    track = _unit(
        geometry.primary_direction + geometry.secondary_direction,
        geometry.primary_direction,
    )
    view = _unit(np.cross(track, separation), np.cross(track, radial))
    if float(np.dot(view, radial)) < 0.0:
        view = -view
    view = _unit(view + radial * 0.35, radial)
    up = _unit(radial - np.dot(radial, view) * view, track)
    miss_distance = float(np.linalg.norm(separation))
    relative_direction = float(
        np.linalg.norm(geometry.primary_direction - geometry.secondary_direction)
    )
    distance = float(
        np.clip(120.0 + miss_distance * 3.0 + relative_direction * 30.0, 220.0, 360.0)
    )
    position = geometry.midpoint + view * distance
    minimum_radius = 6371.0 + 180.0
    if float(np.linalg.norm(position)) < minimum_radius:
        position = position + radial * (minimum_radius - float(np.linalg.norm(position)))
    framing_radius = max(
        float(np.linalg.norm(geometry.primary_tca - geometry.midpoint)),
        float(np.linalg.norm(geometry.secondary_tca - geometry.midpoint)),
    )
    return TcaCameraFrame(
        position=tuple(float(value) for value in position),
        focal_point=tuple(float(value) for value in geometry.midpoint),
        view_up=tuple(float(value) for value in up),
        distance=distance,
        framing_radius=framing_radius,
    )
class OrbitScene(QWidget):
    scenario_changed = Signal(int)
    details_requested = Signal()
    playback_updated = Signal(int, float, object, object, object)

    def __init__(
        self,
        data: SimulationData,
        enable_3d: bool = True,
        safe_mode_message: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.scenario_index = 0
        self.view_mode = "SELECTED"
        self.has_3d_backend = False
        self._plotter = None
        self._catalog: tuple[TleObject, ...] = ()
        self._catalog_points = None
        self._catalog_actor = None
        self._catalog_orbit_actor = None
        self._encounter_geometries: tuple[EncounterGeometry, ...] = ()
        self._overview_actors: dict[int, dict[str, list[object]]] = {}
        self._encounter_points = None
        self._encounter_labels = None
        self._encounter_line = None
        self._encounter_actor = None
        self._earth_actor = None
        self._tca_positions: np.ndarray | None = None
        self._primary_velocity_km_s: float | None = None
        self._secondary_velocity_km_s: float | None = None
        self._last_rendered_minute = -999.0
        self._camera_clock = QElapsedTimer()
        self._camera_start = (14500.0, -17000.0, 9200.0)
        self._camera_focal_start = (0.0, 0.0, 0.0)
        self._camera_up_start = (0.0, 0.0, 1.0)
        self._camera_target = (0.0, 0.0, 0.0)
        self._camera_end_position: tuple[float, float, float] | None = None
        self._camera_end_up = (0.0, 0.0, 1.0)
        self._last_tca_camera_frame: TcaCameraFrame | None = None
        self._camera_timer = QTimer(self)
        self._camera_timer.setInterval(16)
        self._camera_timer.timeout.connect(self._advance_camera)
        self._fly_shortcut = QShortcut(QKeySequence("F"), self)
        self._fly_shortcut.activated.connect(self.fly_to_encounter)

        duration = max(point.minute for item in data.scenarios for point in item.distance_series)
        self.controller = TimelineController(duration, self)
        self.controller.ticked.connect(self._on_tick)
        self.controller.state_changed.connect(self._on_play_state)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("orbitToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 10, 14, 10)
        title = QLabel("LEO STATUS VIEW")
        title.setProperty("heading", True)
        toolbar_layout.addWidget(title)
        self.selector = QComboBox()
        self.selector.addItems([item.label for item in data.scenarios])
        self.selector.currentIndexChanged.connect(self.set_scenario)
        toolbar_layout.addWidget(self.selector, 1)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(("MODE: ALL", "MODE: SELECTED", "MODE: TCA"))
        self.mode_selector.setCurrentIndex(1)
        self.mode_selector.activated.connect(self._set_visualization_mode_by_index)
        toolbar_layout.addWidget(self.mode_selector)
        fly = QPushButton("Focus on Pair")
        fly.clicked.connect(self.fly_to_encounter)
        toolbar_layout.addWidget(fly)
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(
            ("CAMERA: EARTH", "CAMERA: PRIMARY", "CAMERA: SECONDARY", "CAMERA: PAIR", "CAMERA: TCA")
        )
        self.camera_selector.activated.connect(self._apply_camera_mode)
        toolbar_layout.addWidget(self.camera_selector)
        details = QPushButton("Conjunction Detail")
        details.clicked.connect(self.details_requested)
        toolbar_layout.addWidget(details)
        root.addWidget(toolbar)

        self.encounter_identity = QLabel()
        self.encounter_identity.setObjectName("encounterIdentity")
        self.encounter_identity.setProperty("mono", True)
        self.encounter_identity.setContentsMargins(14, 5, 14, 5)
        root.addWidget(self.encounter_identity)

        self.viewport = QFrame()
        viewport_layout = QVBoxLayout(self.viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        if enable_3d:
            self._build_3d(viewport_layout)
        else:
            placeholder = QLabel(
                safe_mode_message or "3D RENDERER UNAVAILABLE\nSAFE MODE ACTIVE"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setProperty("muted", True)
            viewport_layout.addWidget(placeholder)
        root.addWidget(self.viewport, 1)

        controls = QFrame()
        controls.setObjectName("playbackControls")
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(14, 9, 14, 12)
        control_layout.setSpacing(7)
        row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        row.addWidget(self.play_button)
        self.time_label = QLabel("T+0.000 min")
        self.time_label.setProperty("mono", True)
        row.addWidget(self.time_label)
        self.tca_time_label = QLabel("TCA")
        self.tca_time_label.setProperty("accent", True)
        row.addWidget(self.tca_time_label)
        go_tca = QPushButton("GO TO TCA")
        go_tca.clicked.connect(self.go_to_tca)
        row.addWidget(go_tca)
        row.addStretch()
        speed_label = QLabel("SPEED")
        speed_label.setProperty("muted", True)
        row.addWidget(speed_label)
        self.speed = QComboBox()
        for value in (1, 6, 18, 60):
            self.speed.addItem(f"{value}x", value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(
            lambda index: self.controller.set_speed(float(self.speed.itemData(index)))
        )
        row.addWidget(self.speed)
        self.distance_label = QLabel()
        self.distance_label.setProperty("metricSmall", True)
        row.addWidget(self.distance_label)
        control_layout.addLayout(row)
        coverage = next(gate for gate in data.quality_gates if gate["key"] == "coverage")
        self.timeline = TimelineWidget(float(coverage["value"]))
        self.timeline.seek_requested.connect(self.controller.seek_fraction)
        control_layout.addWidget(self.timeline)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 10000)
        self.slider.sliderMoved.connect(lambda value: self.controller.seek_fraction(value / 10000.0))
        control_layout.addWidget(self.slider)
        note = QLabel("TLE/SGP4 replay - 3D view; object sizes are not to scale")
        note.setProperty("muted", True)
        control_layout.addWidget(note)
        legend = QLabel(
            "PRIMARY: CYAN  |  SECONDARY: AMBER  |  TCA- PATH: STRONG  |  "
            "TCA+ PATH: FADED  |  TCA / MISS: RED"
        )
        legend.setProperty("mono", True)
        legend.setProperty("muted", True)
        control_layout.addWidget(legend)
        root.addWidget(controls)
        self.timeline.set_scenario(data.scenarios[0])
        self._update_encounter_identity()
        self._on_tick(0.0)

    def _build_3d(self, layout: QVBoxLayout) -> None:
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            self._plotter = QtInteractor(self.viewport)
            layout.addWidget(self._plotter.interactor)
            self._plotter.set_background("#02040A")
            earth = pv.Sphere(radius=6371.0, theta_resolution=128, phi_resolution=96)
            earth.texture_map_to_sphere(inplace=True)
            self._earth_actor = self._plotter.add_mesh(
                earth,
                texture=self._earth_texture(pv),
                smooth_shading=True,
                ambient=0.18,
                diffuse=0.82,
                specular=0.12,
            )
            atmosphere = pv.Sphere(radius=6455.0, theta_resolution=96, phi_resolution=72)
            self._plotter.add_mesh(
                atmosphere,
                color="#2B8CC4",
                opacity=0.075,
                smooth_shading=True,
            )
            star_rng = np.random.default_rng(20260716)
            star_vectors = star_rng.normal(size=(900, 3))
            star_vectors /= np.linalg.norm(star_vectors, axis=1)[:, None]
            stars = pv.PolyData(star_vectors * 42000.0)
            self._plotter.add_mesh(
                stars,
                color="#DDE9F4",
                point_size=1.4,
                render_points_as_spheres=False,
                ambient=1.0,
            )
            tle_path = self._resolve_tle_path()
            self._catalog = load_tle_catalog(tle_path)
            self._encounter_geometries = build_all_encounter_geometries(
                self.data, self._catalog
            )
            self._draw_catalog_orbits(pv)
            initial = self._propagated_catalog(0.0)
            self._catalog_points = pv.PolyData(initial)
            self._catalog_actor = self._plotter.add_mesh(
                self._catalog_points,
                color="#B8C7D9",
                point_size=5,
                render_points_as_spheres=True,
                opacity=0.65,
                name="catalog-points",
            )
            self._encounter_points = pv.PolyData(np.zeros((2, 3), dtype=float))
            self._encounter_points["role_color"] = np.array(
                [self._rgb255(CYAN), self._rgb255(AMBER)], dtype=np.uint8
            )
            self._encounter_points["role"] = np.array(["PRIMARY", "SECONDARY"])
            self._plotter.add_mesh(
                self._encounter_points,
                scalars="role_color",
                rgb=True,
                point_size=17,
                render_points_as_spheres=True,
                name="selected-pair-points",
            )
            self._encounter_labels = self._encounter_points.copy()
            self._plotter.add_point_labels(
                self._encounter_labels,
                "role",
                font_size=13,
                text_color="#F4FBFF",
                point_size=0,
                shape_opacity=0.35,
                always_visible=True,
                name="selected-pair-labels",
            )
            self._encounter_line = pv.PolyData(np.zeros((2, 3), dtype=float))
            self._encounter_line.lines = np.array([2, 0, 1])

            self._encounter_actor = self._plotter.add_mesh(
                self._encounter_line,
                color=CYAN,
                line_width=3,
                opacity=0.45,
                name="current-separation",
            )
            self._plotter.add_axes(
                line_width=2,
                color="#5E7088",
                xlabel="TEME X",
                ylabel="TEME Y",
                zlabel="TEME Z",
                interactive=False,
            )
            self._plotter.add_text(
                "FRAME: TEME | POSITION: km",
                position="upper_left",
                font_size=9,
                color="#6B7A8F",
                name="frame-label",
            )
            self._draw_all_encounter_overlays(pv)
            self._plotter.camera_position = [
                (18125.0, -21250.0, 11500.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
            self.has_3d_backend = True
            self._draw_selected_orbits()
            self._apply_visual_hierarchy()
        except Exception as exc:
            append_renderer_startup_error(str(exc))
            if self._plotter is not None:
                try:
                    self._plotter.close()
                except Exception:
                    pass
                self._plotter = None
            fallback = QLabel("3D RENDERER UNAVAILABLE\nSAFE MODE ACTIVE")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setWordWrap(True)
            fallback.setProperty("warning", True)
            layout.addWidget(fallback)
            self.has_3d_backend = False

    @staticmethod
    def _earth_texture(pv):
        asset_path = Path(__file__).resolve().parents[1] / "assets" / "earth_daymap.png"
        if asset_path.exists():
            return pv.read_texture(asset_path)
        height, width = 256, 512
        longitude = np.linspace(-np.pi, np.pi, width)[None, :]
        latitude = np.linspace(np.pi / 2, -np.pi / 2, height)[:, None]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        ocean = 25 + (20 * np.cos(latitude) ** 2)
        image[:, :, 0] = ocean.astype(np.uint8)
        image[:, :, 1] = (65 + 35 * np.cos(latitude) ** 2).astype(np.uint8)
        image[:, :, 2] = (105 + 55 * np.cos(latitude) ** 2).astype(np.uint8)
        terrain = (
            np.sin(longitude * 2.1 + np.sin(latitude * 3.0))
            + 0.65 * np.cos(longitude * 3.7 - latitude * 2.4)
            + 0.35 * np.sin(longitude * 8.2 + latitude * 5.1)
        )
        land = (terrain > 0.72) & (np.abs(latitude) < 1.25)
        image[land] = np.array([68, 104, 72], dtype=np.uint8)
        ice = np.abs(latitude) > 1.30
        image[ice.repeat(width, axis=1)] = np.array([202, 220, 225], dtype=np.uint8)
        return pv.numpy_to_texture(image)

    def _resolve_tle_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        raw = str(self.data.replay["tle_source"]).replace("\\", "/")
        return project_root / Path(raw)

    def _catalog_by_id(self) -> dict[str, TleObject]:
        return {item.catalog_id: item for item in self._catalog}

    def _draw_catalog_orbits(self, pv) -> None:
        if self._plotter is None:
            return
        center = parse_utc(str(self.data.scenarios[0].value("snapshot_utc")))
        trails = []
        for item in self._catalog:
            track = orbit_track(item, center)
            if len(track) >= 2:
                trails.append(pv.lines_from_points(track))
        if trails:
            self._catalog_orbit_actor = self._plotter.add_mesh(
                pv.merge(trails),
                color="#5F7894",
                line_width=1,
                opacity=0.13,
                name="catalog-orbits",
            )

    def _draw_all_encounter_overlays(self, pv) -> None:
        if self._plotter is None:
            return
        self._overview_actors.clear()
        for geometry in self._encounter_geometries:
            actors: dict[str, list[object]] = {"marker": [], "path": [], "label": []}
            marker = self._plotter.add_mesh(
                pv.PolyData(np.asarray([geometry.midpoint])),
                color="#7890AA",
                point_size=10,
                render_points_as_spheres=True,
                name=f"overview-{geometry.key.lower()}-marker",
            )
            actors["marker"].append(marker)
            for suffix, path in (
                ("primary", geometry.primary_path),
                ("secondary", geometry.secondary_path),
            ):
                if len(path) >= 2:
                    actor = self._plotter.add_mesh(
                        pv.lines_from_points(path),
                        color="#66809A",
                        line_width=2,
                        opacity=0.22,
                        name=f"overview-{geometry.key.lower()}-{suffix}",
                    )
                    actors["path"].append(actor)
            label_point = pv.PolyData(np.asarray([geometry.midpoint]))
            label_point["label"] = np.array([geometry.key])
            label_actor = self._plotter.add_point_labels(
                label_point,
                "label",
                font_size=12,
                text_color="#B8C7D9",
                point_size=0,
                shape_opacity=0.22,
                always_visible=True,
                name=f"overview-{geometry.key.lower()}-label",
            )
            actors["label"].append(label_actor)
            self._overview_actors[geometry.index] = actors
        self._update_overview_status()

    def _propagated_catalog(self, minute: float) -> np.ndarray:
        scenario = self.data.scenarios[self.scenario_index]
        when = parse_utc(str(scenario.value("snapshot_utc"))) + timedelta(minutes=minute)
        positions = [propagate(item, when) for item in self._catalog]
        valid = [position for position in positions if position is not None]
        return np.vstack(valid) if valid else np.empty((0, 3), dtype=float)

    def _draw_selected_orbits(self) -> None:
        if not self.has_3d_backend or self._plotter is None:
            return
        self._clear_selected_geometry_actors()
        scenario = self.data.scenarios[self.scenario_index]
        lookup = self._catalog_by_id()
        center = parse_utc(str(scenario.value("snapshot_utc")))
        for catalog_id, color, name in (
            (selected_pair_ids(scenario)[0], CYAN, "orbit-primary"),
            (selected_pair_ids(scenario)[1], AMBER, "orbit-secondary"),
        ):
            item = lookup.get(catalog_id)
            if item is None:
                continue
            track = orbit_track(item, center)
            if len(track) >= 2:
                import pyvista as pv

                line = pv.lines_from_points(track)
                self._plotter.add_mesh(
                    line,
                    color=color,
                    line_width=SELECTED_ORBIT_LINE_WIDTH,
                    opacity=SELECTED_ORBIT_OPACITY,
                    name=name,
                )
        self._draw_approach_geometry(scenario, lookup, center)

    def _clear_selected_geometry_actors(self) -> None:
        if self._plotter is None:
            return
        for name in SELECTED_REBUILT_ACTOR_NAMES:
            self._plotter.remove_actor(name, reset_camera=False)

    def _draw_approach_geometry(
        self,
        scenario: Scenario,
        lookup: dict[str, TleObject],
        center,
    ) -> None:
        import pyvista as pv

        minutes = approach_sample_minutes(scenario)
        tca = scenario.tca_minute
        tca_positions: list[np.ndarray] = []
        for role, catalog_id, color in (
            ("primary", selected_pair_ids(scenario)[0], CYAN),
            ("secondary", selected_pair_ids(scenario)[1], AMBER),
        ):
            item = lookup.get(catalog_id)
            if item is None:
                return
            samples = {
                minute: propagate(item, center + timedelta(minutes=minute)) for minute in minutes
            }
            past = [samples[minute] for minute in minutes if minute <= tca and samples[minute] is not None]
            future = [samples[minute] for minute in minutes if minute >= tca and samples[minute] is not None]
            if len(past) >= 2:
                self._plotter.add_mesh(
                    pv.lines_from_points(np.vstack(past)),
                    color=color,
                    line_width=5,
                    opacity=0.92,
                    name=f"approach-{role}-past",
                )
            if len(future) >= 2:
                self._plotter.add_mesh(
                    pv.lines_from_points(np.vstack(future)),
                    color=color,
                    line_width=3,
                    opacity=0.30,
                    name=f"approach-{role}-future",
                )
            tca_position = samples.get(tca)
            if tca_position is not None:
                tca_positions.append(tca_position)
        if len(tca_positions) != 2:
            self._tca_positions = None
            return
        self._tca_positions = np.vstack(tca_positions)
        geometry = self._encounter_geometries[self.scenario_index]
        self._plotter.add_mesh(
            pv.PolyData(self._tca_positions),
            color=RED,
            point_size=17 if self.view_mode == "TCA" else 13,
            render_points_as_spheres=True,
            name="tca-points",
        )
        separation = pv.lines_from_points(self._tca_positions)
        self._plotter.add_mesh(
            separation,
            color=RED,
            line_width=TCA_MISS_LINE_WIDTH,
            opacity=0.95,
            name="tca-separation",
        )
        midpoint = np.mean(self._tca_positions, axis=0)
        label_up = np.asarray(compute_tca_camera_frame(geometry).view_up)
        label_point = pv.PolyData(
            np.asarray([midpoint + label_up * TCA_LABEL_OFFSET_KM])
        )
        label_point["label"] = np.array([tca_label_text(scenario)])
        self._plotter.add_point_labels(
            label_point,
            "label",
            font_size=12,
            text_color="#F4FBFF",
            point_size=0,
            shape_color="#2A1015",
            shape_opacity=0.78,
            always_visible=True,
            name="tca-label",
        )
        arrow_length = TCA_DIRECTION_ARROW_LENGTH_KM
        for role, position, direction, color in (
            ("primary", geometry.primary_tca, geometry.primary_direction, CYAN),
            ("secondary", geometry.secondary_tca, geometry.secondary_direction, AMBER),
        ):
            self._plotter.add_arrows(
                np.asarray([position]),
                np.asarray([direction]),
                mag=arrow_length,
                color=color,
                name=f"direction-{role}",
            )

    def _set_visualization_mode_by_index(self, index: int) -> None:
        self.set_visualization_mode(("ALL", "SELECTED", "TCA")[min(max(int(index), 0), 2)])

    def set_visualization_mode(self, mode: str) -> None:
        normalized = str(mode).upper()
        if normalized not in {"ALL", "SELECTED", "TCA"}:
            raise ValueError(f"Unknown visualization mode: {mode}")
        self.view_mode = normalized
        for control in (self.slider, self.timeline, self.speed):
            control.setEnabled(normalized != "TCA")
        target_index = {"ALL": 0, "SELECTED": 1, "TCA": 2}[normalized]
        if self.mode_selector.currentIndex() != target_index:
            self.mode_selector.blockSignals(True)
            self.mode_selector.setCurrentIndex(target_index)
            self.mode_selector.blockSignals(False)
        self._apply_visual_hierarchy()
        self._update_encounter_identity()
        self._update_overview_status()
        if normalized == "ALL":
            self.focus_earth()
        elif normalized == "TCA":
            self.controller.pause()
            self.camera_selector.blockSignals(True)
            self.camera_selector.setCurrentIndex(4)
            self.camera_selector.blockSignals(False)
            self.go_to_tca()
            self.focus_tca()

    @staticmethod
    def _actor_style(
        actor: object,
        *,
        visible: bool,
        opacity: float | None = None,
        line_width: float | None = None,
        point_size: float | None = None,
    ) -> None:
        actor.SetVisibility(bool(visible))
        try:
            prop = actor.GetProperty()
            if opacity is not None:
                prop.SetOpacity(float(opacity))
            if line_width is not None and hasattr(prop, "SetLineWidth"):
                prop.SetLineWidth(float(line_width))
            if point_size is not None and hasattr(prop, "SetPointSize"):
                prop.SetPointSize(float(point_size))
        except Exception:
            pass

    def _set_named_detail_visibility(self, visible: bool) -> None:
        if self._plotter is None:
            return
        prefixes = (
            "orbit-primary",
            "orbit-secondary",
            "approach-",
            "direction-",
            "tca-",
            "selected-pair-",
            "current-separation",
        )
        for name, actor in self._plotter.actors.items():
            if str(name).startswith(prefixes):
                self._actor_style(actor, visible=visible)

    def _apply_visual_hierarchy(self) -> None:
        if not self.has_3d_backend or self._plotter is None:
            return
        detail_visible = self.view_mode in {"SELECTED", "TCA"}
        self._set_named_detail_visibility(detail_visible)
        # TCA-only presentation; preserve the existing SELECTED styling.
        tca = self.view_mode == "TCA"
        label_points = self._plotter.actors.get("selected-pair-labels-points")
        if label_points is not None:
            label_points.SetVisibility(detail_visible and not tca)
        for name, size, width in (("tca-points", 17 if tca else 13, None),
                                  ("tca-separation", None, TCA_MISS_LINE_WIDTH if tca else 6.0)):
            actor = self._plotter.actors.get(name)
            if actor is not None:
                self._actor_style(actor, visible=detail_visible, point_size=size, line_width=width)
        if self._tca_positions is not None:
            for role, position in zip(("primary", "secondary"), self._tca_positions):
                actor = self._plotter.actors.get(f"direction-{role}")
                if actor is not None:
                    actor.SetOrigin(*position)
                    actor.SetScale(1.0 if tca else 55.0 / TCA_DIRECTION_ARROW_LENGTH_KM)
        self._update_pair_labels()

        catalog_opacity = {"ALL": 0.65, "SELECTED": 0.52, "TCA": 0.12}[self.view_mode]
        orbit_opacity = {"ALL": 0.07, "SELECTED": 0.10, "TCA": 0.03}[self.view_mode]
        if self._catalog_actor is not None:
            self._actor_style(self._catalog_actor, visible=True, opacity=catalog_opacity)
        if self._catalog_orbit_actor is not None:
            self._actor_style(self._catalog_orbit_actor, visible=True, opacity=orbit_opacity)

        for index, groups in self._overview_actors.items():
            selected = index == self.scenario_index
            style = overview_visual_style(self.view_mode, selected)
            for actor in groups["marker"]:
                self._actor_style(
                    actor,
                    visible=True,
                    opacity=style.marker_opacity,
                    point_size=style.marker_size,
                )
            for actor in groups["path"]:
                self._actor_style(
                    actor,
                    visible=True,
                    opacity=style.path_opacity,
                    line_width=style.path_line_width,
                )
            for actor in groups["label"]:
                self._actor_style(
                    actor,
                    visible=style.label_visible,
                    opacity=1.0 if selected else 0.62,
                )
        if self._encounter_actor is not None:
            self._actor_style(
                self._encounter_actor,
                visible=self.view_mode == "SELECTED",
            )
        self._plotter.render()

    def _update_overview_status(self) -> None:
        if self._plotter is None:
            return
        status = (
            f"8 ENCOUNTERS  |  SELECTED: {encounter_key(self.scenario_index)}  |  "
            f"MODE: {self.view_mode}"
        )
        if self.view_mode == "ALL":
            status += "\nEVENT LOCATIONS SHOWN AT EACH ENCOUNTER'S OWN TCA"
        self._plotter.add_text(
            status,
            position="lower_left",
            font_size=9,
            color="#B8C7D9",
            name="encounter-status",
        )

    def _update_encounter_identity(self) -> None:
        scenario = self.data.scenarios[self.scenario_index]
        self.encounter_identity.setText(
            f"{encounter_key(self.scenario_index)}  |  "
            f"PRIMARY: {scenario.value('object_1')} [{scenario.value('catalog_id_1')}]  |  "
            f"SECONDARY: {scenario.value('object_2')} [{scenario.value('catalog_id_2')}]  |  "
            f"TCA: {scenario.value('tca_utc')}  |  "
            f"MISS: {scenario.minimum_distance_km:.3f} km  |  "
            f"VREL: {float(scenario.value('relative_velocity_km_s')):.3f} km/s"
        )

    def set_scenario(self, index: int) -> None:
        self.scenario_index = min(max(int(index), 0), len(self.data.scenarios) - 1)
        if self.selector.currentIndex() != self.scenario_index:
            self.selector.blockSignals(True)
            self.selector.setCurrentIndex(self.scenario_index)
            self.selector.blockSignals(False)
        self.timeline.set_scenario(self.data.scenarios[self.scenario_index])
        self._draw_selected_orbits()
        self._last_rendered_minute = -999.0
        self.scenario_changed.emit(self.scenario_index)
        if self.view_mode == "TCA":
            self.go_to_tca()
            self.focus_tca()
        else:
            self.controller.seek_fraction(0.0)
        self._update_encounter_identity()
        self._update_overview_status()
        self._apply_visual_hierarchy()

    def _on_tick(self, minute: float) -> None:
        fraction = minute / self.controller.duration_minutes
        self.slider.blockSignals(True)
        self.slider.setValue(round(fraction * self.slider.maximum()))
        self.slider.blockSignals(False)
        self.timeline.set_current_minute(minute)
        self.time_label.setText(f"T+{minute:.3f} min")
        scenario = self.data.scenarios[self.scenario_index]
        current_distance = scenario.distance_at(minute)
        self.distance_label.setText(f"CURRENT DISTANCE [SERIES] {current_distance:.3f} km")
        self.tca_time_label.setText(tca_clock_text(minute, scenario.tca_minute))
        if self.has_3d_backend and abs(minute - self._last_rendered_minute) >= 0.25:
            self._update_3d(minute)
            self._last_rendered_minute = minute
        self.playback_updated.emit(
            self.scenario_index,
            minute,
            current_distance,
            self._primary_velocity_km_s,
            self._secondary_velocity_km_s,
        )

    def _update_pair_labels(self) -> None:
        if self._encounter_labels is None or self._encounter_points is None:
            return
        anchors = self._encounter_points.points.copy()
        if self.view_mode == "TCA" and self._encounter_geometries:
            frame = compute_tca_camera_frame(self._encounter_geometries[self.scenario_index])
            right = _unit(np.cross(frame.view_up, np.asarray(frame.position) - frame.focal_point))
            anchors += np.asarray([-18.0, 18.0])[:, None] * right
        self._encounter_labels.points = anchors

    def _update_3d(self, minute: float) -> None:
        if self._plotter is None or self._catalog_points is None or self._encounter_points is None:
            return
        catalog_positions = self._propagated_catalog(minute)
        if len(catalog_positions) == len(self._catalog_points.points):
            self._catalog_points.points = catalog_positions
        scenario = self.data.scenarios[self.scenario_index]
        lookup = self._catalog_by_id()
        when = parse_utc(str(scenario.value("snapshot_utc"))) + timedelta(minutes=minute)
        encounter = []
        velocities: list[float | None] = []
        for key in ("catalog_id_1", "catalog_id_2"):
            item = lookup.get(str(scenario.value(key)))
            state = propagate_state(item, when) if item else None
            position = state[0] if state is not None else None
            velocities.append(float(np.linalg.norm(state[1])) if state is not None else None)
            encounter.append(position if position is not None else np.zeros(3))
        self._primary_velocity_km_s, self._secondary_velocity_km_s = velocities
        self._encounter_points.points = np.vstack(encounter)
        self._update_pair_labels()
        if self._encounter_line is not None:
            self._encounter_line.points = np.vstack(encounter)
        if self._earth_actor is not None:
            self._earth_actor.orientation = (0.0, 0.0, greenwich_angle_deg(when))
        self._plotter.render()

    def _fixed_threshold_km(self) -> float | None:
        config = str(self.data.replay["source_metadata"].get("config", ""))
        match = re.search(r"fixed_threshold_km=([0-9.]+)", config)
        return float(match.group(1)) if match else None

    @staticmethod
    def _hex_rgb(value: str) -> tuple[float, float, float]:
        cleaned = value.lstrip("#")
        return tuple(int(cleaned[offset : offset + 2], 16) / 255.0 for offset in (0, 2, 4))

    @staticmethod
    def _rgb255(value: str) -> tuple[int, int, int]:
        cleaned = value.lstrip("#")
        return tuple(int(cleaned[offset : offset + 2], 16) for offset in (0, 2, 4))

    def _toggle_playback(self) -> None:
        if self.view_mode == "TCA":
            self.set_visualization_mode("SELECTED")
            self.controller.play()
        else:
            self.controller.toggle()

    def _on_play_state(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")

    def fly_to_encounter(self) -> None:
        if not self.has_3d_backend or self._plotter is None or self._encounter_points is None:
            return
        self._camera_start = tuple(float(value) for value in self._plotter.camera.position)
        self._camera_focal_start = tuple(float(value) for value in self._plotter.camera.focal_point)
        self._camera_up_start = tuple(float(value) for value in self._plotter.camera.up)
        midpoint = np.mean(self._encounter_points.points, axis=0)
        self._camera_target = tuple(float(value) for value in midpoint)
        self._camera_end_position = None
        self._camera_clock.restart()
        self._camera_timer.start()

    def go_to_tca(self) -> None:
        scenario = self.data.scenarios[self.scenario_index]
        self.controller.seek_fraction(scenario.tca_minute / self.controller.duration_minutes)

    def _apply_camera_mode(self, index: int) -> None:
        actions = (
            self.focus_earth,
            lambda: self._focus_current_object(0),
            lambda: self._focus_current_object(1),
            self.fly_to_encounter,
            self.focus_tca,
        )
        actions[min(max(int(index), 0), len(actions) - 1)]()

    def focus_earth(self) -> None:
        if self._plotter is None:
            return
        self._camera_timer.stop()
        self.camera_selector.setCurrentIndex(0)
        self._plotter.camera_position = [
            (18125.0, -21250.0, 11500.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        self._plotter.render()

    def _focus_current_object(self, index: int) -> None:
        if self._encounter_points is None:
            return
        self._start_focus(tuple(float(value) for value in self._encounter_points.points[index]))

    def focus_tca(self) -> None:
        if not self._encounter_geometries:
            return
        frame = compute_tca_camera_frame(self._encounter_geometries[self.scenario_index])
        self._last_tca_camera_frame = frame
        self._start_camera_frame(frame)

    def _start_focus(self, target: tuple[float, float, float]) -> None:
        if not self.has_3d_backend or self._plotter is None:
            return
        self._camera_start = tuple(float(value) for value in self._plotter.camera.position)
        self._camera_focal_start = tuple(float(value) for value in self._plotter.camera.focal_point)
        self._camera_up_start = tuple(float(value) for value in self._plotter.camera.up)
        self._camera_target = target
        self._camera_end_position = None
        self._camera_clock.restart()
        self._camera_timer.start()

    def _start_camera_frame(self, frame: TcaCameraFrame) -> None:
        if not self.has_3d_backend or self._plotter is None:
            return
        self._camera_start = tuple(float(value) for value in self._plotter.camera.position)
        self._camera_focal_start = tuple(float(value) for value in self._plotter.camera.focal_point)
        self._camera_up_start = tuple(float(value) for value in self._plotter.camera.up)
        self._camera_target = frame.focal_point
        self._camera_end_position = frame.position
        self._camera_end_up = frame.view_up
        self._camera_clock.restart()
        self._camera_timer.start()

    def _advance_camera(self) -> None:
        if self._plotter is None:
            self._camera_timer.stop()
            return
        progress = min(self._camera_clock.elapsed() / 1800.0, 1.0)
        if self._camera_end_position is None:
            frame = targeted_camera(
                progress,
                self._camera_start,
                self._camera_target,
                self._camera_focal_start,
            )
        else:
            eased = 1.0 - (1.0 - progress) ** 3
            position = tuple(
                self._camera_start[i]
                + (self._camera_end_position[i] - self._camera_start[i]) * eased
                for i in range(3)
            )
            focal_point = tuple(
                self._camera_focal_start[i]
                + (self._camera_target[i] - self._camera_focal_start[i]) * eased
                for i in range(3)
            )
            up = _unit(
                np.asarray(self._camera_up_start)
                + (np.asarray(self._camera_end_up) - np.asarray(self._camera_up_start)) * eased
            )
            frame = CameraKeyframe(
                t=progress,
                position=position,
                focal_point=focal_point,
                view_up=tuple(float(value) for value in up),
            )
        self._plotter.camera_position = [frame.position, frame.focal_point, frame.view_up]
        self._plotter.render()
        if progress >= 1.0:
            self._camera_timer.stop()

    def shutdown(self) -> None:
        self.controller.pause()
        self._camera_timer.stop()
        if self._plotter is not None:
            self._plotter.close()
