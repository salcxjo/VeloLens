"""
VeloLens — Phase 1 entry point
───────────────────────────────
Wires capture → CV → metrics → storage, then prints a session
summary to the terminal.

Usage:
    python main.py [--no-preview] [--duration 120]

Options:
    --no-preview    Skip the OpenCV display window (useful over SSH)
    --duration N    Auto-stop after N seconds (default: press Q to stop)
    --recalibrate   Force a new bottom-bracket calibration before starting
"""

import argparse
import sys
import time
import datetime
from pathlib import Path

import cv2

# ── local imports ─────────────────────────────────────────────────────────
import config
from capture.camera        import Camera
from capture.calibration   import load_or_run
from cv.pose               import PoseEstimator
from cv.crank              import CrankTracker
from processing.metrics    import MetricsEngine
from storage.db            import get_connection, save_session, save_frame_events
from ble.cadence_server    import CadenceBLEServer
from strava.client         import StravaClient, StravaNotConfiguredError, StravaNotAuthorisedError

# rich for the pretty terminal report
try:
    from rich.console import Console
    from rich.table   import Table
    from rich         import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False


# ── helpers ───────────────────────────────────────────────────────────────

def _session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt(val, fmt=".1f", suffix="", fallback="--"):
    if val is None:
        return fallback
    return f"{val:{fmt}}{suffix}"


# ── terminal report ───────────────────────────────────────────────────────

def print_session_report(sm):
    duration = sm.end_time - sm.start_time

    if _RICH:
        console = Console()
        console.print()
        console.rule("[bold cyan]VeloLens — Session Report[/]")
        console.print(f"  Session ID : [bold]{sm.session_id}[/]")
        console.print(f"  Duration   : {duration:.0f} s  "
                      f"({sm.total_frames} frames, "
                      f"{sm.pose_frames} with pose, "
                      f"{sm.crank_frames} with crank)")
        console.print()

        # quality score
        qs = sm.quality_score
        if qs is not None:
            colour = "green" if qs >= 75 else "yellow" if qs >= 50 else "red"
            console.print(
                f"  Quality Score : [bold {colour}]{qs:.0f} / 100[/]"
            )
            t = Table(box=rich_box.SIMPLE, show_header=True, header_style="dim")
            t.add_column("Component",   style="dim")
            t.add_column("Score",       justify="right")
            t.add_column("Weight",      justify="right", style="dim")
            weights = {"cadence_smoothness": "35%", "cadence_target": "20%",
                       "knee_extension": "25%", "posture_stability": "20%"}
            for k, v in sm.quality_breakdown.items():
                colour2 = "green" if v >= 75 else "yellow" if v >= 50 else "red"
                t.add_row(k.replace("_", " ").title(),
                          f"[{colour2}]{v:.0f}[/]",
                          weights.get(k, ""))
            console.print(t)

        # metrics table
        console.print()
        mt = Table(box=rich_box.SIMPLE, show_header=True, header_style="dim")
        mt.add_column("Metric",  style="dim")
        mt.add_column("Value",   justify="right")
        mt.add_column("Target",  justify="right", style="dim")

        mt.add_row("Avg cadence",
                   _fmt(sm.avg_cadence_rpm, suffix=" RPM"),
                   "80–95 RPM")
        mt.add_row("Cadence CV (smoothness)",
                   _fmt(sm.cadence_cv, fmt=".3f"),
                   "< 0.05")
        mt.add_row("Knee angle at BDC",
                   _fmt(sm.avg_knee_angle_bdc, suffix="°"),
                   "140–150°")
        mt.add_row("Knee range",
                   f"{_fmt(sm.min_knee_angle, suffix='°')} – {_fmt(sm.max_knee_angle, suffix='°')}",
                   "")
        mt.add_row("Avg torso angle",
                   _fmt(sm.avg_torso_angle, suffix="°"),
                   "45–55°")
        mt.add_row("Torso drift (fatigue proxy)",
                   _fmt(sm.torso_drift_deg, fmt="+.1f", suffix="°"),
                   "< 2°")
        mt.add_row("Low cadence events",  str(sm.low_cadence_events),  "0")
        mt.add_row("Knee flag events",    str(sm.knee_flag_events),    "0")
        console.print(mt)
        console.rule()

    else:
        # plain fallback
        print("\n── VeloLens Session Report ──────────────────────────")
        print(f"  Session      : {sm.session_id}")
        print(f"  Duration     : {duration:.0f} s")
        print(f"  Quality      : {_fmt(sm.quality_score, fmt='.0f')} / 100")
        print(f"  Avg cadence  : {_fmt(sm.avg_cadence_rpm, suffix=' RPM')}")
        print(f"  Cadence CV   : {_fmt(sm.cadence_cv, fmt='.3f')}")
        print(f"  Knee @ BDC   : {_fmt(sm.avg_knee_angle_bdc, suffix='°')}")
        print(f"  Torso angle  : {_fmt(sm.avg_torso_angle, suffix='°')}")
        print(f"  Torso drift  : {_fmt(sm.torso_drift_deg, fmt='+.1f', suffix='°')}")
        print("─────────────────────────────────────────────────────\n")


# ── main loop ──────────────────────────────────────────────────────────────

def run(show_preview: bool = True,
        duration_secs: float = None,
        recalibrate: bool = False):

    Path(config.SESSION_DIR).mkdir(parents=True, exist_ok=True)
    session_id = _session_id()
    print(f"[VeloLens] Starting session {session_id}")

    # ── hardware init ─────────────────────────────────────────────────────
    cam = Camera().start()
    print(f"[VeloLens] Camera: {cam.resolution[0]}×{cam.resolution[1]}")

    if recalibrate:
        from capture.calibration import run_calibration
        calib = run_calibration(cam)
    else:
        calib = load_or_run(cam)

    # ── cv init ───────────────────────────────────────────────────────────
    pose_estimator = PoseEstimator()
    crank_tracker  = CrankTracker(bb_x=calib.bb_x, bb_y=calib.bb_y)
    engine         = MetricsEngine(session_id=session_id)
    engine.start()

    # ── BLE cadence broadcast ─────────────────────────────────────────────
    ble_server = CadenceBLEServer(device_name=config.BLE_DEVICE_NAME)
    if config.BLE_ENABLED:
        ble_server.start()

    session_start_unix = time.time()

    # downsampled frame event buffer
    frame_events: list[dict] = []
    SAMPLE_INTERVAL = 1.0 / config.METRICS_SAMPLE_HZ

    frame_count      = 0
    pose_tick        = 0
    last_sample_time = time.monotonic()
    stop_time        = (time.monotonic() + duration_secs) if duration_secs else None
    window_name      = "VeloLens — Q to stop"

    print("[VeloLens] Running … press Q in the preview window (or Ctrl-C) to stop.")

    try:
        while True:
            # ── check stop conditions ──────────────────────────────────
            if stop_time and time.monotonic() >= stop_time:
                break

            # ── grab frame ────────────────────────────────────────────
            frame_obj   = cam.read()
            frame_bgr   = frame_obj.image.copy()
            ts          = frame_obj.timestamp
            frame_count += 1

            # ── crank tracking (every frame) ──────────────────────────
            crank_state = crank_tracker.update(frame_bgr, ts)

            # ── pose estimation (every POSE_SKIP_FRAMES) ──────────────
            pose_tick += 1
            pose_result = None
            if pose_tick >= config.POSE_SKIP_FRAMES:
                pose_result = pose_estimator.update(frame_bgr)
                pose_tick   = 0

            # ── metrics update ────────────────────────────────────────
            fm = engine.update(crank_state, pose_result)

            # ── BLE cadence push ──────────────────────────────────────────
            if config.BLE_ENABLED and fm.cadence_rpm is not None:
                ble_server.push_cadence(fm.cadence_rpm)

            # ── live status (headless) ───────────────────────────────────
            if not show_preview and frame_count % 150 == 0:
                elapsed = time.monotonic() - session_start_unix
                print(f"  {elapsed:.0f}s  cadence={_fmt(fm.cadence_rpm, suffix=' RPM')}  "
                      f"knee={_fmt(fm.knee_angle_deg, suffix=chr(176))}  "
                      f"torso={_fmt(fm.torso_angle_deg, suffix=chr(176))}  "
                      f"track={crank_state.tracking}  fps={cam.fps:.0f}",
                      flush=True)

            # ── downsample to storage ──────────────────────────────────
            if ts - last_sample_time >= SAMPLE_INTERVAL:
                frame_events.append({
                    "timestamp":       ts,
                    "cadence_rpm":     fm.cadence_rpm,
                    "knee_angle_deg":  fm.knee_angle_deg,
                    "torso_angle_deg": fm.torso_angle_deg,
                    "crank_angle_deg": fm.crank_angle_deg,
                })
                last_sample_time = ts

            # ── preview window ────────────────────────────────────────
            if show_preview:
                try:
                    display = frame_bgr.copy()

                    if pose_result:
                        pose_estimator.draw_skeleton(display, pose_result)

                    crank_tracker.draw_overlay(display, crank_state)

                    hud_lines = [
                        f"Cadence: {_fmt(fm.cadence_rpm, suffix=' RPM')}",
                        f"Knee:    {_fmt(fm.knee_angle_deg, suffix=chr(176))}",
                        f"Torso:   {_fmt(fm.torso_angle_deg, suffix=chr(176))}",
                        f"Track:   {crank_state.tracking}",
                        f"FPS:     {cam.fps:.0f}",
                    ]
                    for i, line in enumerate(hud_lines):
                        cv2.putText(display, line, (10, 100 + i * 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (255, 255, 200), 2)

                    cv2.imshow(window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except cv2.error:
                    show_preview = False   # disable silently if no display

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless OpenCV has no window manager
        cam.stop()
        pose_estimator.close()
        if config.BLE_ENABLED:
            ble_server.stop()

    # ── session summary ────────────────────────────────────────────────────
    session_duration = time.monotonic() - engine._start_time
    if session_duration < config.MIN_SESSION_SECS:
        print(f"[VeloLens] Session too short ({session_duration:.0f}s < "
              f"{config.MIN_SESSION_SECS}s) — not saved.")
        return

    sm = engine.finish()

    # persist
    conn = get_connection()
    save_session(sm, conn)
    save_frame_events(session_id, frame_events, conn)
    conn.commit()
    conn.close()

    # also write JSON export
    json_path = Path(config.SESSION_DIR) / f"{session_id}.json"
    json_path.write_text(sm.to_json())
    print(f"[VeloLens] Session saved → {json_path}")

    print_session_report(sm)

    # ── Strava annotation ─────────────────────────────────────────────────
    if getattr(config, 'STRAVA_AUTO_ANNOTATE', False) and config.STRAVA_CLIENT_ID:
        try:
            strava = StravaClient()
            if strava.is_authorised:
                session_end_unix = session_start_unix + (sm.end_time - sm.start_time)
                activity = strava.match_activity(session_start_unix, session_end_unix)
                if activity:
                    strava.annotate_activity(activity["id"], sm)
                    print(f"[Strava] Annotated activity: {activity['name']}")
                else:
                    print("[Strava] No matching activity found — start Strava before riding.")
            else:
                print("[Strava] Not authorised. Run: python -m veloLens.strava.client --auth")
        except StravaNotConfiguredError:
            pass
        except Exception as e:
            print(f"[Strava] Annotation failed: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeloLens Phase 1")
    parser.add_argument("--no-preview",   action="store_true",
                        help="Disable OpenCV display window (SSH mode)")
    parser.add_argument("--duration",     type=float, default=None,
                        help="Auto-stop after N seconds")
    parser.add_argument("--recalibrate",  action="store_true",
                        help="Force new BB calibration before starting")
    args = parser.parse_args()

    run(
        show_preview  = not args.no_preview,
        duration_secs = args.duration,
        recalibrate   = args.recalibrate,
    )
