"""
tools/awb_tune.py
─────────────────
White balance tuning for the IMX219 (Pi Camera v2).

The camera often skews blue indoors. This tool captures frames at
different ColourGains settings and saves them as a contact sheet so
you can pick the one that looks most neutral.

Usage:
    python -m tools.awb_tune

Outputs:
    sessions/awb_sheet.jpg  — contact sheet of all gain combinations
    sessions/awb_auto.jpg   — what auto-AWB gives you (for comparison)

Then set your preferred values in config.py:
    CAMERA_COLOUR_GAINS = (red_gain, blue_gain)
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

os.makedirs("sessions", exist_ok=True)


def capture_with_gains(picam, red: float, blue: float) -> np.ndarray:
    """Set gains, wait for sensor to settle, capture one frame."""
    picam.set_controls({"ColourGains": (red, blue)})
    time.sleep(0.4)   # settle time
    frame = picam.capture_array("main")
    if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return frame


def run():
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not available — ensure --system-site-packages venv")
        sys.exit(1)

    print("[AWB tune] Starting camera ...")
    picam = Picamera2()
    cfg   = picam.create_video_configuration(
        main={"size": (320, 240), "format": "BGR888"},   # small for speed
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1.0)

    # 1. Capture auto-AWB reference
    print("[AWB tune] Capturing auto-AWB reference ...")
    picam.set_controls({"AwbEnable": True})
    time.sleep(0.8)
    auto_frame = picam.capture_array("main")
    if auto_frame is not None and auto_frame.ndim == 3 and auto_frame.shape[2] == 4:
        auto_frame = auto_frame[:, :, :3]
    cv2.imwrite("sessions/awb_auto.jpg", auto_frame)

    # 2. Switch to manual and sweep gains
    picam.set_controls({"AwbEnable": False})
    time.sleep(0.3)

    red_values  = [1.2, 1.5, 1.8, 2.1]
    blue_values = [1.0, 1.2, 1.4, 1.6]

    print(f"[AWB tune] Capturing {len(red_values) * len(blue_values)} gain "
          f"combinations ...")

    rows = []
    for r in red_values:
        row_frames = []
        for b in blue_values:
            frame = capture_with_gains(picam, r, b)
            # add label
            label = f"R:{r:.1f} B:{b:.1f}"
            cv2.putText(frame, label, (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, label, (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        lineType=cv2.LINE_AA)
            row_frames.append(frame)
            print(f"  R={r:.1f} B={b:.1f}", flush=True)
        rows.append(np.hstack(row_frames))

    # Add auto row at top
    auto_label = auto_frame.copy()
    auto_wide  = np.hstack([auto_label] + [np.zeros_like(auto_label)] * (len(blue_values) - 1))
    cv2.putText(auto_wide, "AUTO AWB (reference)", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)

    sheet = np.vstack([auto_wide] + rows)
    cv2.imwrite("sessions/awb_sheet.jpg", sheet)

    picam.stop()
    picam.close()

    print(f"""
[AWB tune] Done.

  sessions/awb_auto.jpg  — auto white balance (probably blue)
  sessions/awb_sheet.jpg — grid of manual gain combinations

Copy to laptop:
  scp pi@<ip>:~/VeloLens/sessions/awb_sheet.jpg ~/Desktop/

Find the panel that looks most neutral, then set in config.py:
  CAMERA_COLOUR_GAINS = (red_gain, blue_gain)

Typical indoor values: (1.5, 1.2) to (2.1, 1.4)
For the CV pipeline, colour accuracy matters less than contrast —
the HSV marker tracking cares about hue, not absolute white balance.
""")


if __name__ == "__main__":
    run()
