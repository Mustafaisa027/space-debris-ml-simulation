from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable


@dataclass(frozen=True)
class RendererProbeResult:
    success: bool
    status: str
    diagnostics: dict[str, Any]
    error: str = ""


def run_renderer_probe(timeout_seconds: float = 20.0) -> RendererProbeResult:
    """Exercise Qt/VTK/OpenGL in a disposable process before the GUI imports it."""
    command = [sys.executable, "-m", "desktop_gui.renderer_probe", "--child"]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    base_diagnostics = _base_diagnostics()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 0.1),
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        return RendererProbeResult(False, "timeout", base_diagnostics, "renderer probe timed out")
    except Exception as exc:
        return RendererProbeResult(False, "launch-failed", base_diagnostics, str(exc))

    payload = _last_json_object(completed.stdout)
    if completed.returncode == 0 and payload and payload.get("success"):
        return RendererProbeResult(
            True,
            "success",
            dict(payload.get("diagnostics", {})),
            "",
        )
    error = ""
    if payload:
        error = str(payload.get("error", ""))
    if not error:
        error = _compact_error(completed.stderr) or f"probe exited with code {completed.returncode}"
    failed_diagnostics = (
        dict(payload.get("diagnostics", {})) if payload else base_diagnostics
    )
    return RendererProbeResult(False, "failed", failed_diagnostics, error)


def disabled_probe_result() -> RendererProbeResult:
    diagnostics = _base_diagnostics()
    diagnostics["safe_mode"] = True
    return RendererProbeResult(False, "disabled-by-user", diagnostics, "--no-3d requested")


def select_3d_mode(
    no_3d: bool,
    probe: Callable[[], RendererProbeResult] = run_renderer_probe,
) -> tuple[bool, RendererProbeResult]:
    result = disabled_probe_result() if no_3d else probe()
    return bool(result.success and not no_3d), result


def write_renderer_log(result: RendererProbeResult) -> None:
    """Best-effort diagnostics; logging must never block startup."""
    try:
        root = Path(__file__).resolve().parents[1]
        log_dir = root / "output"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        payload["safe_mode"] = not result.success
        (log_dir / "desktop_gui_renderer.log").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def append_renderer_startup_error(error: str) -> None:
    try:
        path = Path(__file__).resolve().parents[1] / "output" / "desktop_gui_renderer.log"
        payload: dict[str, Any] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload["safe_mode"] = True
        payload["startup_error"] = str(error)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    except Exception:
        pass


def _base_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "python": platform.python_version(),
    }
    try:
        diagnostics["platform"] = platform.platform()
    except Exception as exc:
        diagnostics["platform_error"] = str(exc)
    try:
        import PySide6

        diagnostics["pyside6"] = PySide6.__version__
    except Exception as exc:
        diagnostics["pyside6_error"] = str(exc)
    return diagnostics


def _child_probe() -> int:
    diagnostics = _base_diagnostics()
    plotter = None
    try:
        from PySide6.QtWidgets import QApplication
        import pyvista as pv
        import vtk
        from pyvistaqt import QtInteractor

        diagnostics.update(
            {
                "pyvista": pv.__version__,
                "vtk": vtk.vtkVersion.GetVTKVersion(),
            }
        )
        app = QApplication.instance() or QApplication(["renderer-probe"])
        plotter = QtInteractor(off_screen=True)
        plotter.add_mesh(pv.Sphere(theta_resolution=12, phi_resolution=12))
        plotter.render()
        app.processEvents()
        capabilities = str(plotter.render_window.ReportCapabilities())
        diagnostics.update(_parse_capabilities(capabilities))
        print(json.dumps({"success": True, "diagnostics": diagnostics}), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps({"success": False, "diagnostics": diagnostics, "error": str(exc)}),
            flush=True,
        )
        return 2
    finally:
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


def _parse_capabilities(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        cleaned = line.strip()
        lower = cleaned.lower()
        if "opengl version" in lower:
            result["opengl_version"] = cleaned.split(":", 1)[-1].strip()
        elif "renderer" in lower and "string" in lower:
            result["renderer"] = cleaned.split(":", 1)[-1].strip()
        elif "vendor" in lower and "string" in lower:
            result["gpu_vendor"] = cleaned.split(":", 1)[-1].strip()
    if not result and text.strip():
        result["capabilities"] = _compact_error(text, limit=600)
    return result


def _last_json_object(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _compact_error(text: str, limit: int = 300) -> str:
    return " ".join(text.strip().split())[-limit:]


if __name__ == "__main__" and "--child" in sys.argv:
    raise SystemExit(_child_probe())
