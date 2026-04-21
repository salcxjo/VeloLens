"""
capture/calibration.py
──────────────────────
Bottom bracket calibration. Three modes:

  1. Headless (SSH / no display) — saves a frame as JPEG, you measure
     the BB pixel coordinates with any image viewer and pass them via CLI.

  2. Interactive GUI — click-to-calibrate when a display is available.

  3. Config fallback — if CALIB_BB_X / CALIB_BB_Y are set in config.py,
     use those directly with no interaction at all.

Usage:
  # Headless: capture frame, then set coordinates manually
  python -m capture.calibration --headless
  python -m capture.calibration --set 320 240

  # GUI (requires display)
  python -m capture.calibration --gui

  # Use config.py fallback values permanently
  Set CALIB_BB_X and CALIB_BB_Y in config.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

CALIB_PATH   = Path(config.SESSION_DIR) / "calibration.json"
CALIB_FRAME  = Path(config.SESSION_DIR) / "calibration_frame.jpg"


@dataclass
class CalibrationData:
    bb_x: int
    bb_y: int
    frame_width: int
    frame_height: int
    timestamp: float

    def save(self):
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[calibration] Saved → {CALIB_PATH}")

    @classmethod
    def load(cls) -> Optional["CalibrationData"]:
        if not CALIB_PATH.exists():
            return None
        with open(CALIB_PATH) as f:
            d = json.load(f)
        return cls(**d)


# ── mode 1: headless ─────────────────────────────────────────────────────────

def run_headless(camera) -> None:
    """
    Capture a still frame and save it as a JPEG so the user can measure
    the BB centre with any image viewer (Preview, GIMP, even a browser).
    """
    CALIB_FRAME.parent.mkdir(parents=True, exist_ok=True)
    frame = camera.read()
    # draw crosshair at frame centre as a reference point
    img = frame.image.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 100), 1)
    cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 100), 1)
    cv2.imwrite(str(CALIB_FRAME), img)
    print(f"""
[calibration] Frame saved → {CALIB_FRAME}

Open it in any image viewer and find the pixel coordinates of the
bottom bracket axle (the centre of the crank spindle).

Most viewers show cursor coordinates in the status bar.
  - macOS Preview:  Tools > Show Inspector > second tab
  - GIMP:           coordinates shown in bottom toolbar
  - Python:         python3 -c "from PIL import Image; img=Image.open('{CALIB_FRAME}'); print(img.size)"
  - scp to laptop:  scp pi@<ip>:{CALIB_FRAME} ~/Desktop/

Then run:
  python -m capture.calibration --set <x> <y>

Frame size: {w}x{h}  —  centre crosshair is at ({cx}, {cy})
""")


def run_set(camera, x: int, y: int) -> CalibrationData:
    """Save manually measured BB coordinates."""
    frame = camera.read()
    h, w = frame.image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(
            f"Coordinates ({x}, {y}) are outside the frame ({w}x{h}). "
            "Check your values."
        )
    calib = CalibrationData(
        bb_x=x, bb_y=y,
        frame_width=w, frame_height=h,
        timestamp=time.time(),
    )
    calib.save()
    print(f"[calibration] BB centre set to ({x}, {y})  frame {w}x{h}")
    return calib


# ── mode 2: interactive GUI ───────────────────────────────────────────────────

_click_point: Optional[tuple[int, int]] = None


def _on_mouse(event, x, y, flags, param):
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)


def _display_available() -> bool:
    """Return True if an X/Wayland display is reachable."""
    # Only trust environment variables — namedWindow succeeds even on
    # headless OpenCV builds, the error comes later on imshow/waitKey.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_gui(camera) -> CalibrationData:
    global _click_point
    _click_point = None

    if not _display_available():
        raise RuntimeError(
            "No display available. Use headless mode instead:\n"
            "  python -m capture.calibration --headless\n"
            "  python -m capture.calibration --set <x> <y>"
        )

    window = "VeloLens — Click the bottom bracket centre, then press ENTER"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse)

    print("\n[calibration] Click the BOTTOM BRACKET AXLE in the window.")
    print("              ENTER to confirm  |  R to re-pick  |  Q to quit\n")

    frame = camera.read()
    base  = frame.image.copy()

    while True:
        show = base.copy()
        if _click_point:
            cx, cy = _click_point
            cv2.drawMarker(show, (cx, cy), (0, 255, 100), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(show, (cx, cy), 6, (0, 255, 100), -1)
            cv2.putText(show, f"BB: ({cx}, {cy}) — ENTER to confirm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
        else:
            cv2.putText(show, "Click the bottom bracket axle",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2)

        cv2.imshow(window, show)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and _click_point:
            break
        elif key == ord('r'):
            _click_point = None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled.")

    cv2.destroyAllWindows()
    h, w = base.shape[:2]
    calib = CalibrationData(
        bb_x=_click_point[0], bb_y=_click_point[1],
        frame_width=w, frame_height=h,
        timestamp=time.time(),
    )
    calib.save()
    return calib


# ── mode 3: config fallback ───────────────────────────────────────────────────

def _from_config(camera) -> Optional[CalibrationData]:
    """Use CALIB_BB_X / CALIB_BB_Y from config.py if set."""
    x = getattr(config, "CALIB_BB_X", None)
    y = getattr(config, "CALIB_BB_Y", None)
    if x is None or y is None:
        return None
    frame = camera.read()
    h, w  = frame.image.shape[:2]
    print(f"[calibration] Using config.py values: BB=({x}, {y})")
    return CalibrationData(bb_x=x, bb_y=y, frame_width=w, frame_height=h,
                           timestamp=time.time())


# ── main entry point ──────────────────────────────────────────────────────────

def load_or_run(camera) -> CalibrationData:
    """
    Load saved calibration if it exists, otherwise:
      1. Use config.py CALIB_BB_X/Y if set
      2. Try GUI if a display is available
      3. Fall back to headless (save frame + prompt for --set)
    """
    calib = CalibrationData.load()
    if calib:
        print(f"[calibration] Loaded: BB=({calib.bb_x}, {calib.bb_y})")
        return calib

    # config.py hardcoded values
    calib = _from_config(camera)
    if calib:
        return calib

    # try GUI
    if _display_available():
        print("[calibration] No saved calibration — launching GUI.")
        return run_gui(camera)

    # headless fallback
    print("[calibration] No display detected — switching to headless mode.")
    run_headless(camera)
    raise RuntimeError(
        "\nCalibration required before first session.\n"
        "Open the saved frame, find the BB pixel coordinates, then run:\n"
        f"  python -m capture.calibration --set <x> <y>\n"
        f"Frame saved to: {CALIB_FRAME}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from capture.camera import Camera

    parser = argparse.ArgumentParser(description="VeloLens BB calibration")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--headless", action="store_true",
                       help="Capture a frame and save it as JPEG for manual measurement")
    group.add_argument("--set", nargs=2, type=int, metavar=("X", "Y"),
                       help="Save BB coordinates directly (after measuring from frame)")
    group.add_argument("--gui",  action="store_true",
                       help="Interactive click-to-calibrate (requires display)")
    group.add_argument("--show", action="store_true",
                       help="Print current saved calibration")
    args = parser.parse_args()

    if args.show:
        c = CalibrationData.load()
        if c:
            print(f"BB centre: ({c.bb_x}, {c.bb_y})  frame: {c.frame_width}x{c.frame_height}")
        else:
            print("No calibration saved yet.")
        sys.exit(0)

    cam = Camera().start()
    try:
        if args.headless:
            run_headless(cam)
        elif args.set:
            run_set(cam, args.set[0], args.set[1])
        elif args.gui:
            run_gui(cam)
    finally:
        cam.stop()
