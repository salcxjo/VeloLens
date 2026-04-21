"""
tools/record_session.py
───────────────────────
Records raw frames from the camera to a .avi file so you can:
  - Review what the camera actually sees
  - Play back through the CV pipeline offline
  - Share clips for debugging

Usage:
    python -m tools.record_session --duration 30 --output sessions/test_clip.avi
    python -m tools.record_session --duration 10   # saves to sessions/rec_<timestamp>.avi
"""

import argparse
import datetime
import time
from pathlib import Path

import cv2

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from capture.camera import Camera


def record(duration_secs: float, output_path: Path):
    cam = Camera().start()
    w, h = cam.resolution
    fps_target = config.TARGET_FPS

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps_target,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer at {output_path}")

    print(f"[recorder] Recording {duration_secs:.0f}s → {output_path}")
    print(f"           Resolution: {w}x{h}  Press Ctrl-C to stop early.")

    start    = time.monotonic()
    frames   = 0
    last_ts  = None

    try:
        while time.monotonic() - start < duration_secs:
            frame = cam.read()
            # deduplicate — only write when a new frame arrives
            if last_ts == frame.timestamp:
                time.sleep(0.005)
                continue
            last_ts = frame.timestamp
            writer.write(frame.image)
            frames += 1
            elapsed = time.monotonic() - start
            print(f"\r  {elapsed:.1f}s / {duration_secs:.0f}s   "
                  f"{frames} frames   {cam.fps:.1f} fps", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[recorder] Stopped early.")

    print(f"\n[recorder] Saved {frames} frames → {output_path}")
    writer.release()
    cam.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeloLens frame recorder")
    parser.add_argument("--duration", type=float, default=30,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .avi path (default: sessions/rec_<timestamp>.avi)")
    args = parser.parse_args()

    out = Path(args.output) if args.output else (
        Path(config.SESSION_DIR) /
        f"rec_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    record(args.duration, out)
