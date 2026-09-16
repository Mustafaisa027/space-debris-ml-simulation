"""Single live E01 TCA smoke plus in-process geometry/style checks."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtCore import QTimer
import desktop_gui.main as entry

original = entry.OrbitalSentinelWindow
result = {"success": False}

def instrument(*args, **kwargs):
    w = original(*args, **kwargs)
    s = w.orbit_scene
    def enter():
        w.show_scene(1, animate=False)
        s.set_visualization_mode("TCA")
    def verify():
        try:
            assert s.has_3d_backend
            s._plotter.screenshot("output/tca_hierarchy_e01.png")
            w.screen().grabWindow(int(w.winId())).save("output/tca_hierarchy_window.png")
            counts, checks = [], []
            for i in (0, 3, 5, 7, 0):
                s.set_scenario(i)
                x = s.data.scenarios[i]
                physical = s._tca_positions.copy()
                raw = float(np.linalg.norm(physical[0] - physical[1]))
                assert abs(raw - x.minimum_distance_km) <= 0.001
                assert abs(x.distance_at(x.tca_minute) - x.minimum_distance_km) <= 0.001
                assert np.allclose(s._encounter_points.points, physical, atol=1e-7, rtol=0)
                assert abs(s.controller.current_minute - x.tca_minute) < 1e-9
                assert s.tca_time_label.text() == "TCA" and s.camera_selector.currentIndex() == 4
                assert all(not c.isEnabled() for c in (s.play_button, s.timeline, s.slider, s.speed))
                s.play_button.click()
                assert not s.controller.is_playing
                actors = s._plotter.actors
                assert actors["tca-points"].GetProperty().GetPointSize() == 17
                assert actors["tca-separation"].GetProperty().GetLineWidth() == 8
                assert np.allclose(actors["tca-separation"].mapper.dataset.points, physical)
                offsets = s._encounter_labels.points - physical
                assert np.allclose(np.linalg.norm(offsets, axis=1), 18)
                assert np.allclose(offsets[0], -offsets[1])
                for role in ("primary", "secondary"):
                    assert actors[f"direction-{role}"].GetScale() == (1., 1., 1.)
                counts.append(len(actors))
                checks.append({"event": f"E{i+1:02}", "raw_km": raw})
            s.set_visualization_mode("SELECTED")
            assert np.array_equal(s._encounter_labels.points, s._encounter_points.points)
            assert actors["tca-points"].GetProperty().GetPointSize() == 13
            assert actors["tca-separation"].GetProperty().GetLineWidth() == 6
            assert actors["direction-primary"].GetScale() == (2.2, 2.2, 2.2)
            counts.append(len(actors))
            assert set(counts) == {64}
            log = json.loads(Path("output/desktop_gui_renderer.log").read_text())
            assert log["success"] and not log["safe_mode"] and not log.get("startup_error")
            result.update(success=True, actor_counts=counts, geometry=checks, renderer=log)
        except Exception:
            import traceback
            result["error"] = traceback.format_exc()
        finally:
            Path("output/tca_hierarchy_result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            w.close()
    QTimer.singleShot(4000, enter)
    QTimer.singleShot(7500, verify)
    return w

entry.OrbitalSentinelWindow = instrument
entry.main()
raise SystemExit(0 if result["success"] else 1)
