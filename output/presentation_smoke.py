"""One live main() launch; capture settled screens and check presentation contracts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import desktop_gui.main as entry

out = Path(__file__).resolve().parent
original_window = entry.OrbitalSentinelWindow
results = {"geometry": [], "actor_counts": [], "errors": []}


def instrument(*args, **kwargs):
    window = original_window(*args, **kwargs)
    scene = window.orbit_scene

    def capture(name):
        QApplication.processEvents()
        window.grab().save(str(out / f"audit_final_{name}.png"))
        results["actor_counts"].append(len(scene._plotter.renderer.actors))

    def overview():
        assert scene.has_3d_backend
        window.show_scene(1, animate=False)
        scene.set_scenario(0)
        scene.set_visualization_mode("ALL")
        assert len(scene._overview_actors) == 8
        assert sum(len(g["path"]) for g in scene._overview_actors.values()) == 16
        capture("all")
        scene.set_visualization_mode("TCA")

    def tca():
        scenario = scene.data.scenarios[0]
        assert not scene.controller.is_playing
        assert abs(scene.controller.current_minute - scenario.tca_minute) <= 1e-9
        assert scene.tca_time_label.text() == "TCA"
        assert scene.camera_selector.currentIndex() == 4
        assert not scene._encounter_actor.GetVisibility()
        scene.play_button.click()
        assert not scene.controller.is_playing
        capture("tca")
        scene.set_visualization_mode("SELECTED")
        scene.controller.seek_fraction((scenario.tca_minute - 0.03) / scene.controller.duration_minutes)

    def selected():
        assert scene._encounter_actor.GetVisibility()
        capture("selected")
        scene.play_button.click()
        assert scene.controller.is_playing

    def detail():
        scene.controller.pause()
        assert scene.controller.current_minute > scene.data.scenarios[0].tca_minute
        scene.set_visualization_mode("TCA")
        window.show_scene(2, animate=False)
        capture("detail")

    def geometry():
        window.show_scene(1, animate=False)
        for _ in range(2):
            for index in (0, 3, 5, 7):
                scene.set_scenario(index)
                scenario = scene.data.scenarios[index]
                physical = float(np.linalg.norm(scene._tca_positions[0] - scene._tca_positions[1]))
                assert abs(physical - scenario.minimum_distance_km) <= 0.001
                assert abs(scenario.distance_at(scenario.tca_minute) - scenario.minimum_distance_km) <= 0.001
                assert abs(scene.controller.current_minute - scenario.tca_minute) <= 1e-9
                assert scene.tca_time_label.text() == "TCA"
                assert not scene.controller.is_playing
                assert np.allclose(scene._encounter_points.points, scene._tca_positions, atol=1e-7, rtol=0)
                results["geometry"].append({"event": f"E{index+1:02}", "source_km": scenario.minimum_distance_km, "raw_tca_km": physical})
                results["actor_counts"].append(len(scene._plotter.renderer.actors))
        scene.set_scenario(3)
        window.show_scene(2, animate=False)
        capture("detail_switched")
        scene.set_scenario(0)
        window.show_scene(1, animate=False)
        scene.set_visualization_mode("TCA")

    def finish():
        capture("tca_end")
        assert len(set(results["actor_counts"])) == 1, results["actor_counts"]
        log = json.loads((out / "desktop_gui_renderer.log").read_text())
        assert log["success"] and not log["safe_mode"] and not log.get("startup_error")
        results["renderer"] = log
        results["success"] = not results["errors"]
        (out / "presentation_smoke.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results), flush=True)

    def run(step):
        try:
            step()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results["success"] = False
            results["errors"].append(f"{step.__name__}: {type(exc).__name__}: {exc}")
            (out / "presentation_smoke.json").write_text(json.dumps(results, indent=2))

    for delay, step in ((4000, overview), (7000, tca), (8000, selected), (8500, detail), (10000, geometry), (14000, finish)):
        QTimer.singleShot(delay, lambda step=step: run(step))
    return window


entry.OrbitalSentinelWindow = instrument
raise SystemExit(entry.main())
