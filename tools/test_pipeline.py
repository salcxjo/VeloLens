"""
tools/test_pipeline.py
──────────────────────
Synthetic pipeline test — runs the full CV + metrics stack without
a camera or bike, using generated fake data. Lets you verify that
every module is wired correctly before you get on the bike.

Also supports playing back a recorded .avi file through the pipeline.

Usage:
    # Synthetic data (no camera needed)
    python -m tools.test_pipeline

    # Replay a recorded clip
    python -m tools.test_pipeline --video sessions/rec_20260411_002643.avi

    # Verbose: print every frame's metrics
    python -m tools.test_pipeline --verbose
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from cv.filters     import KalmanFilter1D, RollingStats, AngleUnwrapper
from cv.crank       import CrankTracker, CrankState
from cv.pose        import PoseResult, Landmark
from processing.metrics import MetricsEngine


# ── synthetic data generators ─────────────────────────────────────────────────

def _fake_crank_state(t: float, cadence_rpm: float = 88.0) -> CrankState:
    """Simulate a crank rotating at cadence_rpm with slight noise."""
    angle = (t * cadence_rpm / 60.0 * 360.0) % 360.0
    noisy_rpm = cadence_rpm + np.random.normal(0, 2.0)
    return CrankState(
        angle_deg   = angle,
        cadence_rpm = max(0, noisy_rpm),
        marker_px   = (320 + int(80 * math.cos(math.radians(angle))),
                       380 + int(80 * math.sin(math.radians(angle)))),
        tracking    = "detected",
        total_revs  = int(t * cadence_rpm / 60.0),
    )


def _fake_pose_result(t: float, frame: np.ndarray) -> PoseResult:
    """
    Simulate a side-profile rider pose.
    Knee angle oscillates between 75° (TDC) and 148° (BDC) with cadence.
    Torso angle drifts slightly over time to simulate fatigue.
    """
    # crank phase → knee angle (simplified sinusoidal model)
    crank_phase = (t * 88.0 / 60.0 * 2 * math.pi)
    knee_angle_deg = 110 + 38 * (0.5 + 0.5 * math.cos(crank_phase))  # 72–148°
    knee_angle_deg += np.random.normal(0, 1.5)

    # fatigue: torso rises (less aero) 3° over 60 seconds
    torso_angle_deg = 48.0 + (t / 60.0) * 3.0 + np.random.normal(0, 0.5)

    # derive pixel positions from angles (approximate, for a 640x480 frame)
    # hip is anchor point
    hip_x, hip_y = 320, 200
    # shoulder above hip at torso_angle
    sho_x = hip_x - int(100 * math.cos(math.radians(torso_angle_deg)))
    sho_y = hip_y - int(100 * math.sin(math.radians(torso_angle_deg)))
    # knee below hip
    thigh_len = 110
    knee_x = hip_x + int(thigh_len * math.sin(math.radians(20)))
    knee_y = hip_y + int(thigh_len * math.cos(math.radians(20)))
    # ankle below knee — distance determined by knee_angle
    shin_len = 105
    shin_angle_rad = math.radians(180 - knee_angle_deg + 20)
    ank_x = knee_x + int(shin_len * math.sin(shin_angle_rad))
    ank_y = knee_y + int(shin_len * math.cos(shin_angle_rad))

    def lm(x, y, score=0.9):
        return Landmark(x=float(x), y=float(y), visibility=score)

    landmarks = {
        config.MP_RIGHT_HIP:      lm(hip_x, hip_y),
        config.MP_RIGHT_KNEE:     lm(knee_x, knee_y),
        config.MP_RIGHT_ANKLE:    lm(ank_x, ank_y),
        config.MP_RIGHT_SHOULDER: lm(sho_x, sho_y),
        config.MP_RIGHT_HEEL:     lm(ank_x, ank_y),
        config.MP_RIGHT_TOE:      lm(ank_x, ank_y),
        config.MP_LEFT_HIP:       lm(hip_x - 15, hip_y, 0.4),
        config.MP_LEFT_KNEE:      lm(knee_x - 15, knee_y, 0.4),
        config.MP_LEFT_ANKLE:     lm(ank_x - 15, ank_y, 0.4),
        config.MP_LEFT_SHOULDER:  lm(sho_x - 15, sho_y, 0.4),
    }
    return PoseResult(landmarks=landmarks, raw_image=frame)


def _draw_synthetic(frame: np.ndarray, crank: CrankState,
                    pose: PoseResult) -> np.ndarray:
    """Draw skeleton and crank overlay on a blank frame."""
    # crank circle
    bb = (320, 380)
    cv2.circle(frame, bb, 80, (60, 60, 60), 1)
    cv2.drawMarker(frame, bb, (255, 200, 0), cv2.MARKER_CROSS, 16, 1)
    if crank.marker_px:
        cv2.circle(frame, crank.marker_px, 8, (0, 200, 255), -1)
        cv2.line(frame, bb, crank.marker_px, (80, 80, 200), 1)

    # skeleton
    connections = [
        (config.MP_RIGHT_SHOULDER, config.MP_RIGHT_HIP),
        (config.MP_RIGHT_HIP,      config.MP_RIGHT_KNEE),
        (config.MP_RIGHT_KNEE,     config.MP_RIGHT_ANKLE),
    ]
    for a_idx, b_idx in connections:
        a = pose.xy(a_idx)
        b = pose.xy(b_idx)
        if a and b:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (0, 220, 120), 2)
    for idx in [config.MP_RIGHT_HIP, config.MP_RIGHT_KNEE,
                config.MP_RIGHT_ANKLE, config.MP_RIGHT_SHOULDER]:
        pt = pose.xy(idx)
        if pt:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 220, 120), -1)
    return frame


# ── video replay ──────────────────────────────────────────────────────────────

def run_replay(video_path: Path, verbose: bool):
    """Run the CV pipeline on a recorded video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[test] Replaying {video_path.name}  "
          f"({total_frames} frames @ {fps:.0f} fps)")

    # Use frame centre as BB — user can refine
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bb_x = getattr(config, "CALIB_BB_X", None) or w // 2
    bb_y = getattr(config, "CALIB_BB_Y", None) or int(h * 0.7)
    print(f"[test] Using BB=({bb_x},{bb_y})  "
          f"(set CALIB_BB_X/Y in config.py to override)")

    from cv.crank import CrankTracker
    from cv.pose  import PoseEstimator
    crank   = CrankTracker(bb_x=bb_x, bb_y=bb_y)
    pose    = PoseEstimator()
    engine  = MetricsEngine(session_id="replay_test")
    engine.start()

    frame_n = 0
    t_start = time.monotonic()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_n += 1
        ts = frame_n / fps

        cs = crank.update(frame_bgr, ts)
        pr = pose.update(frame_bgr) if frame_n % config.POSE_SKIP_FRAMES == 0 else None
        fm = engine.update(cs, pr)

        if verbose:
            print(f"  f{frame_n:04d}  crank={cs.angle_deg or 0:6.1f}°  "
                  f"rpm={fm.cadence_rpm or 0:5.1f}  "
                  f"knee={fm.knee_angle_deg or 0:5.1f}°  "
                  f"torso={fm.torso_angle_deg or 0:4.1f}°  "
                  f"track={cs.tracking}")
        elif frame_n % 30 == 0:
            print(f"  {frame_n}/{total_frames} frames  "
                  f"rpm={fm.cadence_rpm or 0:.0f}  "
                  f"knee={fm.knee_angle_deg or 0:.0f}°", flush=True)

    cap.release()
    pose.close()
    sm = engine.finish()
    _print_summary(sm)


# ── synthetic run ─────────────────────────────────────────────────────────────

def run_synthetic(duration: float, verbose: bool):
    print(f"[test] Running synthetic pipeline for {duration:.0f}s")
    print( "       Simulating: 88 RPM cadence, 140–148° knee at BDC, "
           "mild fatigue drift\n")

    engine = MetricsEngine(session_id="synthetic_test")
    engine.start()
    blank  = np.zeros((480, 640, 3), dtype=np.uint8)

    t_start = time.monotonic()
    frame_n = 0
    dt      = 1.0 / config.TARGET_FPS

    while time.monotonic() - t_start < duration:
        t  = time.monotonic() - t_start
        cs = _fake_crank_state(t)
        pr = _fake_pose_result(t, blank.copy())
        fm = engine.update(cs, pr)
        frame_n += 1

        if verbose:
            print(f"  t={t:5.1f}s  rpm={fm.cadence_rpm or 0:5.1f}  "
                  f"knee={fm.knee_angle_deg or 0:5.1f}°  "
                  f"torso={fm.torso_angle_deg or 0:4.1f}°")
        elif frame_n % 150 == 0:
            print(f"  {t:.0f}s  rpm={fm.cadence_rpm or 0:.0f}  "
                  f"knee={fm.knee_angle_deg or 0:.0f}°  "
                  f"torso={fm.torso_angle_deg or 0:.1f}°", flush=True)

        time.sleep(dt)

    sm = engine.finish()
    _print_summary(sm)


# ── shared summary printer ────────────────────────────────────────────────────

def _print_summary(sm):
    print("\n── Pipeline test results ───────────────────────────────────")
    print(f"  Frames processed   : {sm.total_frames}")
    print(f"  Pose detections    : {sm.pose_frames}")
    print(f"  Crank detections   : {sm.crank_frames}")
    print(f"  Avg cadence        : {sm.avg_cadence_rpm or 0:.1f} RPM")
    print(f"  Cadence CV         : {sm.cadence_cv or 0:.4f}  (< 0.05 = smooth)")
    print(f"  Knee @ BDC         : {sm.avg_knee_angle_bdc or 0:.1f}°  (target 140–150°)")
    print(f"  Torso angle        : {sm.avg_torso_angle or 0:.1f}°")
    print(f"  Torso drift        : {sm.torso_drift_deg or 0:+.1f}°  (fatigue proxy)")
    print(f"  Quality score      : {sm.quality_score or 0:.0f} / 100")
    if sm.quality_breakdown:
        for k, v in sm.quality_breakdown.items():
            bar = "█" * int(v / 5) + "░" * (20 - int(v / 5))
            print(f"    {k:<24} {bar} {v:.0f}")
    print("────────────────────────────────────────────────────────────")

    issues = []
    if sm.cadence_cv and sm.cadence_cv > 0.08:
        issues.append("High cadence variability — check marker tracking stability")
    if sm.avg_knee_angle_bdc and not (135 <= sm.avg_knee_angle_bdc <= 155):
        issues.append(f"Knee angle {sm.avg_knee_angle_bdc:.0f}° outside 135–155° range — check BB calibration or pose detection")
    if sm.pose_frames == 0:
        issues.append("No pose detected — check camera angle and lighting")
    if sm.crank_frames == 0:
        issues.append("No crank tracking — check HSV marker colour tuning")
    if issues:
        print("\n  ⚠ Diagnostics:")
        for i in issues:
            print(f"    • {i}")
    else:
        print("\n  ✓ All checks passed")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeloLens pipeline tester")
    parser.add_argument("--video",   type=str,   default=None,
                        help="Path to a recorded .avi file to replay")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Synthetic test duration in seconds (default: 15)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print metrics for every frame")
    args = parser.parse_args()

    if args.video:
        run_replay(Path(args.video), args.verbose)
    else:
        run_synthetic(args.duration, args.verbose)
