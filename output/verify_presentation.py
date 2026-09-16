"""Native renderer widget verification without launching another main window."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from desktop_gui.data_loader import load_simulation_data
from desktop_gui.scenes.orbit_scene import OrbitScene
from desktop_gui.scenes.conjunction_panel import ConjunctionPanel

app = QApplication([])
data = load_simulation_data()
scene = OrbitScene(data, enable_3d=True)
detail = ConjunctionPanel(data)
scene.scenario_changed.connect(detail.set_scenario)
scene.playback_updated.connect(detail.set_playback_state)
assert scene.has_3d_backend
counts, geometry = [], []
for repeat in range(2):
    for i in (0, 3, 5, 7):
        scene.set_visualization_mode("TCA")
        scene.set_scenario(i)
        scenario = data.scenarios[i]
        raw = float(np.linalg.norm(scene._tca_positions[0] - scene._tca_positions[1]))
        series = scenario.distance_at(scenario.tca_minute)
        assert abs(raw - scenario.minimum_distance_km) <= 0.001
        assert abs(series - scenario.minimum_distance_km) <= 0.001
        assert np.allclose(scene._encounter_points.points, scene._tca_positions, atol=1e-7, rtol=0)
        assert abs(scene.controller.current_minute - scenario.tca_minute) <= 1e-9
        assert not scene.controller.is_playing
        assert not scene._encounter_actor.GetVisibility()
        assert not scene._plotter.actors["selected-pair-labels-points"].GetVisibility()
        assert detail.live_labels["tca_offset"].text() == "TCA"
        assert detail.live_labels["current_distance"].text() == f"{series:.3f} km"
        assert detail.live_labels["primary_velocity"].text() != "N/A"
        counts.append(len(scene._plotter.renderer.actors))
        if repeat == 0:
            geometry.append({"event": f"E{i+1:02}", "source_km": scenario.minimum_distance_km, "raw_km": raw, "series_km": series})
scene.set_scenario(0)
for _ in range(3):
    scene.set_visualization_mode("TCA")
    minute = scene.controller.current_minute
    camera = tuple(scene._plotter.camera_position)
    assert scene.play_button.isEnabled()
    scene.play_button.click()
    assert scene.view_mode == "SELECTED" and scene.controller.is_playing
    assert scene.controller.current_minute == minute
    assert tuple(scene._plotter.camera_position) == camera
    assert all(c.isEnabled() for c in (scene.slider, scene.timeline, scene.speed))
    counts.append(len(scene._plotter.renderer.actors))
    scene.controller.pause()
scene._plotter.ren_win.SetSize(1248, 600)
scene.set_visualization_mode("TCA")
QTest.qWait(2100)
scene._plotter.screenshot("output/audit_verified_tca.png")
scene.set_visualization_mode("SELECTED")
scene.controller.seek_fraction((data.scenarios[0].tca_minute - 0.03) / scene.controller.duration_minutes)
scene._plotter.screenshot("output/audit_verified_selected.png")
scene.set_visualization_mode("ALL")
scene._plotter.screenshot("output/audit_verified_all.png")
counts.append(len(scene._plotter.renderer.actors))
assert len(set(counts)) == 1
result = {"success": True, "actor_counts": counts, "geometry": geometry, "detail_sync": True}
Path("output/presentation_verified.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
scene.shutdown()
scene.close()
detail.close()
