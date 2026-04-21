"""
tools/render_annotated.py
─────────────────────────
Renders an annotated MP4 from a recorded session clip.

Overlays:
  - Skeleton (shoulder → hip → knee → ankle)
  - Joint angle arcs at knee and hip
  - Crank circle + marker dot + angle line
  - Live metrics HUD (cadence, knee angle, torso, quality score)
  - Colour-coded quality bar
  - Rolling cadence graph (bottom strip)

Output is a standard H.264 MP4 readable on any device / uploadable
directly to LinkedIn.

Usage:
    python -m tools.render_annotated --video sessions/rec_20260411.avi
    python -m tools.render_annotated --video sessions/rec_20260411.avi --output demo.mp4
    python -m tools.render_annotated --video sessions/rec_20260411.avi --slow 0.5
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from cv.crank           import CrankTracker
from cv.pose            import PoseEstimator
from processing.metrics import MetricsEngine


# ── colour palette (BGR) ──────────────────────────────────────────────────────
C_GREEN   = (60,  220, 100)
C_AMBER   = (30,  180, 255)
C_RED     = (60,   60, 220)
C_BLUE    = (220, 160,  40)
C_WHITE   = (240, 240, 240)
C_DARK    = ( 20,  20,  20)
C_TEAL    = (180, 200,  60)
C_PANEL   = ( 18,  18,  18)   # HUD background


def _quality_colour(score):
    if score is None:
        return C_TEAL
    if score >= 75:
        return C_GREEN
    if score >= 50:
        return C_AMBER
    return C_RED


def _put(img, text, pos, scale=0.55, colour=C_WHITE, thickness=1, bold=False):
    if bold:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                    scale, C_DARK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def _draw_angle_arc(img, vertex, p_a, p_b, angle_deg, colour, radius=30):
    """Draw an arc showing the angle at vertex between p_a and p_b."""
    if vertex is None or p_a is None or p_b is None:
        return
    vx, vy = int(vertex[0]), int(vertex[1])
    ax = math.degrees(math.atan2(p_a[1] - vy, p_a[0] - vx))
    bx = math.degrees(math.atan2(p_b[1] - vy, p_b[0] - vx))
    start_a = int(min(ax, bx))
    end_a   = int(max(ax, bx))
    if end_a - start_a > 180:
        start_a, end_a = end_a, start_a + 360
    cv2.ellipse(img, (vx, vy), (radius, radius), 0,
                start_a, end_a, colour, 1, cv2.LINE_AA)
    # label
    mid_a = math.radians((start_a + end_a) / 2)
    lx = int(vx + (radius + 10) * math.cos(mid_a))
    ly = int(vy + (radius + 10) * math.sin(mid_a))
    _put(img, f"{angle_deg:.0f}", (lx - 12, ly + 4), scale=0.42, colour=colour)


def _draw_skeleton(img, pose, crank_state):
    """Draw joint dots, limb lines, and angle arcs."""
    if pose is None:
        return

    # limb connections
    connections = [
        (config.MP_RIGHT_SHOULDER, config.MP_RIGHT_HIP,   C_TEAL),
        (config.MP_RIGHT_HIP,      config.MP_RIGHT_KNEE,  C_GREEN),
        (config.MP_RIGHT_KNEE,     config.MP_RIGHT_ANKLE, C_GREEN),
    ]
    for a_idx, b_idx, col in connections:
        a = pose.xy(a_idx)
        b = pose.xy(b_idx)
        if a and b:
            cv2.line(img, (int(a[0]), int(a[1])),
                     (int(b[0]), int(b[1])), col, 2, cv2.LINE_AA)

    # joint dots
    for idx in [config.MP_RIGHT_SHOULDER, config.MP_RIGHT_HIP,
                config.MP_RIGHT_KNEE, config.MP_RIGHT_ANKLE]:
        pt = pose.xy(idx)
        if pt:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, C_WHITE,  -1, cv2.LINE_AA)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, C_GREEN,  -1, cv2.LINE_AA)

    # knee angle arc
    knee_angle = pose.joint_angle(
        config.MP_RIGHT_HIP, config.MP_RIGHT_KNEE, config.MP_RIGHT_ANKLE)
    if knee_angle is not None:
        _draw_angle_arc(
            img,
            pose.xy(config.MP_RIGHT_KNEE),
            pose.xy(config.MP_RIGHT_HIP),
            pose.xy(config.MP_RIGHT_ANKLE),
            knee_angle, C_AMBER, radius=32,
        )

    # torso angle label at hip
    torso = pose.vector_angle_from_horizontal(
        config.MP_RIGHT_HIP, config.MP_RIGHT_SHOULDER)
    hip = pose.xy(config.MP_RIGHT_HIP)
    if torso is not None and hip:
        _put(img, f"torso {torso:.0f}", (int(hip[0]) + 8, int(hip[1]) - 8),
             scale=0.42, colour=C_TEAL)


def _draw_crank(img, crank_state, bb_x, bb_y):
    """Draw BB centre, crank arm line, and marker dot."""
    cv2.circle(img, (bb_x, bb_y), 80, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.drawMarker(img, (bb_x, bb_y), C_AMBER, cv2.MARKER_CROSS, 14, 1)

    if crank_state.marker_px and crank_state.angle_deg is not None:
        mx, my = crank_state.marker_px
        dot_col = C_GREEN if crank_state.tracking == "detected" else C_BLUE
        cv2.line(img, (bb_x, bb_y), (mx, my), (80, 80, 180), 1, cv2.LINE_AA)
        cv2.circle(img, (mx, my), 9, dot_col, -1, cv2.LINE_AA)
        cv2.circle(img, (mx, my), 9, C_WHITE,  1, cv2.LINE_AA)


def _draw_hud(img, fm, sm_live, cadence_history, frame_n, total_frames, fps):
    """
    Semi-transparent HUD panel on the right side.
    sm_live: dict of running averages updated each frame.
    """
    h, w = img.shape[:2]
    panel_x = w - 200
    panel_w = 198

    # semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    x  = panel_x + 10
    y  = 28
    dy = 26

    # title
    _put(img, "VeloLens", (x, y), scale=0.65, colour=C_GREEN, bold=True)
    y += dy

    # quality score bar
    qs = sm_live.get("quality")
    qs_col = _quality_colour(qs)
    qs_str = f"{qs:.0f}" if qs is not None else "--"
    _put(img, f"Quality  {qs_str}/100", (x, y), scale=0.52, colour=qs_col)
    y += 18
    bar_w = panel_w - 20
    cv2.rectangle(img, (x, y), (x + bar_w, y + 6), (50, 50, 50), -1)
    if qs:
        fill = int(bar_w * qs / 100)
        cv2.rectangle(img, (x, y), (x + fill, y + 6), qs_col, -1)
    y += 18

    # divider
    cv2.line(img, (x, y), (x + bar_w, y), (60, 60, 60), 1)
    y += 10

    def row(label, val, unit="", colour=C_WHITE, target=None):
        nonlocal y
        val_str = f"{val:.0f}{unit}" if val is not None else "--"
        _put(img, label, (x, y), scale=0.42, colour=(160, 160, 160))
        _put(img, val_str, (x + 80, y), scale=0.52, colour=colour)
        if target:
            _put(img, target, (x + 130, y), scale=0.38, colour=(100, 100, 100))
        y += dy - 4

    rpm   = fm.cadence_rpm
    knee  = fm.knee_angle_deg
    torso = fm.torso_angle_deg

    rpm_col   = C_GREEN if rpm and 80 <= rpm <= 95 else C_AMBER
    knee_col  = C_GREEN if knee and config.KNEE_EXTENSION_OPTIMAL_MIN <= knee <= config.KNEE_EXTENSION_OPTIMAL_MAX else C_AMBER
    torso_col = C_GREEN if torso and 40 <= torso <= 60 else C_AMBER

    row("Cadence",  rpm,   " rpm", rpm_col,   "80-95")
    row("Knee@BDC", sm_live.get("knee_bdc"), "°", knee_col, "140-150")
    row("Knee now", knee,  "°",    knee_col)
    row("Torso",    torso, "°",    torso_col,  "45-55")
    row("Avg RPM",  sm_live.get("avg_rpm"), " rpm", C_WHITE)

    # tracking status dot
    y += 4
    track = getattr(fm, '_tracking', 'unknown')
    cv2.line(img, (x, y), (x + bar_w, y), (60, 60, 60), 1)
    y += 10

    # progress bar at very bottom of panel
    progress = frame_n / max(total_frames, 1)
    cv2.rectangle(img, (panel_x, h - 8), (w, h), (40, 40, 40), -1)
    cv2.rectangle(img, (panel_x, h - 8),
                  (panel_x + int(panel_w * progress), h), C_TEAL, -1)

    # cadence sparkline (bottom strip across full width)
    _draw_sparkline(img, cadence_history, panel_x)


def _draw_sparkline(img, history, panel_x):
    """Small cadence graph along the bottom of the main video area."""
    h = img.shape[0]
    strip_h = 40
    strip_y = h - strip_h
    vals = list(history)
    if len(vals) < 2:
        return

    cv2.rectangle(img, (0, strip_y), (panel_x, h), (10, 10, 10), -1)
    _put(img, "cadence", (4, strip_y + 14), scale=0.38, colour=(100, 100, 100))

    lo, hi = 60.0, 120.0
    pts = []
    for i, v in enumerate(vals):
        x = int(i / (len(vals) - 1) * (panel_x - 2))
        y = strip_y + strip_h - 4 - int((v - lo) / (hi - lo) * (strip_h - 8))
        y = max(strip_y + 2, min(h - 2, y))
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], C_TEAL, 1, cv2.LINE_AA)

    # target zone (80–95 RPM) shaded
    y_top = strip_y + strip_h - 4 - int((95 - lo) / (hi - lo) * (strip_h - 8))
    y_bot = strip_y + strip_h - 4 - int((80 - lo) / (hi - lo) * (strip_h - 8))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y_top), (panel_x, y_bot), (40, 80, 40), -1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


# ── main renderer ─────────────────────────────────────────────────────────────

def render(video_path: Path, output_path: Path, slow_factor: float = 1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps      = src_fps * slow_factor
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # H.264 MP4 — widest compatibility for LinkedIn / web
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_path}")

    bb_x = getattr(config, "CALIB_BB_X", None) or w // 2
    bb_y = getattr(config, "CALIB_BB_Y", None) or int(h * 0.7)

    crank   = CrankTracker(bb_x=bb_x, bb_y=bb_y)
    pose    = PoseEstimator()
    engine  = MetricsEngine(session_id="render")
    engine.start()

    cadence_history = deque(maxlen=120)   # ~4 s at 30 fps
    pose_result     = None
    frame_n         = 0

    # running averages for HUD
    sm_live = {"quality": None, "avg_rpm": None, "knee_bdc": None}
    rpm_acc  = []
    knee_acc = []

    print(f"[render] {video_path.name}  {total_frames} frames @ {src_fps:.0f}fps")
    print(f"         Output: {output_path}  (slow={slow_factor}x)")
    t0 = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1
        ts = frame_n / src_fps

        cs = crank.update(frame, ts)

        if frame_n % config.POSE_SKIP_FRAMES == 0:
            pose_result = pose.update(frame)

        fm = engine.update(cs, pose_result)

        if fm.cadence_rpm:
            cadence_history.append(fm.cadence_rpm)
            rpm_acc.append(fm.cadence_rpm)
            sm_live["avg_rpm"] = sum(rpm_acc) / len(rpm_acc)

        ca = cs.angle_deg
        if fm.knee_angle_deg and ca and 160 <= ca <= 200:
            knee_acc.append(fm.knee_angle_deg)
            sm_live["knee_bdc"] = sum(knee_acc) / len(knee_acc)

        if frame_n % 30 == 0:
            sm = engine.finish()
            engine = MetricsEngine(session_id="render")
            engine.start()
            sm_live["quality"] = sm.quality_score

        # ── draw overlays ──────────────────────────────────────────────────
        out = frame.copy()
        _draw_skeleton(out, pose_result, cs)
        _draw_crank(out, cs, bb_x, bb_y)
        _draw_hud(out, fm, sm_live, cadence_history, frame_n, total_frames, src_fps)

        writer.write(out)

        if frame_n % 30 == 0:
            elapsed  = time.monotonic() - t0
            pct      = frame_n / total_frames * 100
            eta      = elapsed / frame_n * (total_frames - frame_n)
            print(f"\r  {pct:.0f}%  frame {frame_n}/{total_frames}  "
                  f"rpm={fm.cadence_rpm or 0:.0f}  "
                  f"knee={fm.knee_angle_deg or 0:.0f}°  "
                  f"ETA {eta:.0f}s     ", end="", flush=True)

    print(f"\n[render] Done → {output_path}  ({frame_n} frames)")
    cap.release()
    writer.release()
    pose.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeloLens annotated video renderer")
    parser.add_argument("--video",  required=True,
                        help="Input .avi recorded clip")
    parser.add_argument("--output", default=None,
                        help="Output .mp4 path (default: same name as input)")
    parser.add_argument("--slow",   type=float, default=1.0,
                        help="Slow motion factor e.g. 0.5 = half speed (default: 1.0)")
    args = parser.parse_args()

    inp = Path(args.video)
    out = Path(args.output) if args.output else inp.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    render(inp, out, slow_factor=args.slow)
