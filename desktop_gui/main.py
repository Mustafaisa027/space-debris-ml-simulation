from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from desktop_gui.app_window import OrbitalSentinelWindow, load_stylesheet
from desktop_gui.data_loader import DEFAULT_DATA_PATH, load_simulation_data
from desktop_gui.renderer_probe import select_3d_mode, write_renderer_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbital Sentinel Mission Control GUI")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="simulation.json path",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="skip PyVista/VTK initialization and run the remaining GUI in safe mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_simulation_data(args.data)
    enable_3d, probe_result = select_3d_mode(args.no_3d)
    write_renderer_log(probe_result)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Orbital Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet())
    safe_mode_message = None
    if not enable_3d:
        safe_mode_message = "3D RENDERER UNAVAILABLE\nSAFE MODE ACTIVE"
    window = OrbitalSentinelWindow(
        data,
        enable_3d=enable_3d,
        safe_mode_message=safe_mode_message,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
