"""
tools/hsv_tuner.py
──────────────────
HSV range calibrator for the crank marker sticker.

Two modes:

  1. Headless (SSH) — samples HSV values from the centre region of
     a saved frame. Point the camera at the sticker, run this, and
     it prints suggested HSV bounds you can paste into config.py.

  2. GUI — interactive sliders (requires a local display, not SSH).

Usage:
    # Headless: capture frame and sample HSV automatically
    python -m tools.hsv_tuner --headless

    # Headless: sample from a specific saved frame
    python -m tools.hsv_tuner --headless --image sessions/calibration_frame.jpg

    # GUI (local display only)
    python -m tools.hsv_tuner --gui
"""

import argparse
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from capture.camera import Camera


# ── headless mode ─────────────────────────────────────────────────────────────

def run_headless(image_path: str = None):
    """
    Sample HSV from a live frame or saved image.
    Captures a still, analyses the HSV histogram, and suggests bounds.
    Also saves an annotated debug image for manual inspection.
    """
    if image_path:
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise RuntimeError(f"Cannot load image: {image_path}")
        print(f"[HSV tuner] Loaded {image_path}")
    else:
        print("[HSV tuner] Capturing live frame ...")
        cam = Camera().start()
        # warm up — skip first few frames
        for _ in range(10):
            f = cam.read()
        frame_bgr = cam.read().image.copy()
        cam.stop()

    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Sample a central 40x40 region — user should aim camera at sticker
    cx, cy = w // 2, h // 2
    margin = 20
    roi = hsv[cy - margin:cy + margin, cx - margin:cx + margin]

    h_vals = roi[:, :, 0].flatten()
    s_vals = roi[:, :, 1].flatten()
    v_vals = roi[:, :, 2].flatten()

    # Suggest bounds with ±15 tolerance on H, ±40 on S/V
    h_med = int(np.median(h_vals))
    s_med = int(np.median(s_vals))
    v_med = int(np.median(v_vals))

    h_lo = max(0,   h_med - 15)
    h_hi = min(179, h_med + 15)
    s_lo = max(0,   s_med - 40)
    s_hi = min(255, s_med + 40)
    v_lo = max(0,   v_med - 40)
    v_hi = min(255, v_med + 40)

    print(f"""
[HSV tuner] Centre region ({cx-margin},{cy-margin}) → ({cx+margin},{cy+margin}):
  H median: {h_med}   S median: {s_med}   V median: {v_med}

Suggested config.py values:
  MARKER_HSV_LOWER = ({h_lo}, {s_lo}, {v_lo})
  MARKER_HSV_UPPER = ({h_hi}, {s_hi}, {v_hi})

To use these, edit config.py or run:
  python -m tools.hsv_tuner --headless --apply

Notes:
  - This samples the dead centre of the frame.
  - For best results, position the sticker in the centre of the camera view
    before running, or use --image with a saved frame where the sticker
    is clearly visible and dominant in the centre region.
  - Common sticker colours:
      Bright orange: H≈10  S≈220  V≈220
      Neon yellow:   H≈30  S≈255  V≈220
      Bright red:    H≈0   S≈220  V≈200  (also check H≈170 for red wraparound)
      Neon green:    H≈60  S≈255  V≈200
""")

    # Save annotated debug image
    debug = frame_bgr.copy()
    cv2.rectangle(debug, (cx - margin, cy - margin),
                  (cx + margin, cy + margin), (0, 255, 100), 2)
    cv2.putText(debug, f"H:{h_med} S:{s_med} V:{v_med}",
                (cx - margin, cy - margin - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)

    # Apply mask with suggested values so you can see what's detected
    mask = cv2.inRange(hsv,
                       np.array([h_lo, s_lo, v_lo]),
                       np.array([h_hi, s_hi, v_hi]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    # Stack original + mask + masked result side by side
    mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined  = np.hstack([debug, mask_bgr, masked])
    # Scale down if too wide
    if combined.shape[1] > 1280:
        combined = cv2.resize(combined, (1280, combined.shape[0] * 1280 // combined.shape[1]))

    out_path = "sessions/hsv_debug.jpg"
    os.makedirs("sessions", exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"[HSV tuner] Debug image saved → {out_path}")
    print( "            Left: original with sample box")
    print( "            Centre: mask with suggested values")
    print( "            Right: masked result (white pixels = detected)")
    print( "            scp pi@<ip>:~/VeloLens/sessions/hsv_debug.jpg ~/Desktop/\n")

    return (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)


def apply_to_config(lower: tuple, upper: tuple):
    import re
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
    src = open(config_path).read()
    src = re.sub(r"MARKER_HSV_LOWER\s*=\s*\(.*?\)",
                 f"MARKER_HSV_LOWER    = {lower}", src)
    src = re.sub(r"MARKER_HSV_UPPER\s*=\s*\(.*?\)",
                 f"MARKER_HSV_UPPER    = {upper}", src)
    open(config_path, "w").write(src)
    print(f"[HSV tuner] config.py updated: lower={lower}  upper={upper}")


# ── GUI mode ──────────────────────────────────────────────────────────────────

def _display_available() -> bool:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    try:
        cv2.namedWindow("_t", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_t")
        return True
    except Exception:
        return False


def run_gui():
    if not _display_available():
        print("[HSV tuner] No display detected. Use --headless instead.")
        sys.exit(1)

    def _nothing(_): pass

    cam = Camera().start()
    win = "HSV Tuner — S to save, Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", win, config.MARKER_HSV_LOWER[0], 179, _nothing)
    cv2.createTrackbar("H max", win, config.MARKER_HSV_UPPER[0], 179, _nothing)
    cv2.createTrackbar("S min", win, config.MARKER_HSV_LOWER[1], 255, _nothing)
    cv2.createTrackbar("S max", win, config.MARKER_HSV_UPPER[1], 255, _nothing)
    cv2.createTrackbar("V min", win, config.MARKER_HSV_LOWER[2], 255, _nothing)
    cv2.createTrackbar("V max", win, config.MARKER_HSV_UPPER[2], 255, _nothing)

    while True:
        frame = cam.read().image
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([cv2.getTrackbarPos("H min", win),
                       cv2.getTrackbarPos("S min", win),
                       cv2.getTrackbarPos("V min", win)])
        hi = np.array([cv2.getTrackbarPos("H max", win),
                       cv2.getTrackbarPos("S max", win),
                       cv2.getTrackbarPos("V max", win)])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow(win, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result]))
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            apply_to_config(tuple(lo.tolist()), tuple(hi.tolist()))
            break

    cv2.destroyAllWindows()
    cam.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeloLens HSV marker tuner")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--headless", action="store_true",
                       help="Sample HSV from camera/image (SSH-friendly)")
    group.add_argument("--gui",      action="store_true",
                       help="Interactive sliders (requires local display)")
    parser.add_argument("--image",  type=str, default=None,
                        help="Use a saved image instead of live capture")
    parser.add_argument("--apply",  action="store_true",
                        help="Auto-write suggested values to config.py")
    args = parser.parse_args()

    # default to headless if no flag given
    if not args.gui:
        lower, upper = run_headless(args.image)
        if args.apply:
            apply_to_config(lower, upper)
        else:
            print("Run with --apply to write these values to config.py automatically.")
    else:
        run_gui()
