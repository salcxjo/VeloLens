"""
tools/monitor_server.py
───────────────────────
Live monitoring web UI. Run alongside main.py --no-preview.

Shows:
  - Live camera feed with skeleton + crank overlay
  - Real-time metrics: cadence, knee angle, torso, quality score
  - Crank tracking status (detected / tracked / lost)
  - Pose landmark confidence

Usage:
    # Terminal 1
    python main.py --no-preview --duration 3600

    # Terminal 2
    python -m tools.monitor_server

    Then open http://<pi-ip>:5001 on your laptop.
"""

import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from capture.camera     import Camera
from cv.crank           import CrankTracker
from cv.pose            import PoseEstimator
from processing.metrics import MetricsEngine

# ── shared live state ─────────────────────────────────────────────────────────
_metrics = {
    "cadence_rpm":     None,
    "knee_angle":      None,
    "torso_angle":     None,
    "tracking":        "lost",
    "quality":         None,
    "pose_confidence": 0.0,
    "elapsed_secs":    0,
    "fps":             0.0,
}
_latest_jpeg: bytes = b""
_lock = threading.Lock()


# ── drawing helpers ───────────────────────────────────────────────────────────

def _colour(val, lo_good, hi_good):
    """BGR colour: green if in range, amber if close, red if far."""
    if val is None:
        return (120, 120, 120)
    if lo_good <= val <= hi_good:
        return (60, 220, 80)
    margin = (hi_good - lo_good) * 0.3
    if lo_good - margin <= val <= hi_good + margin:
        return (30, 180, 255)
    return (60, 60, 220)


def _put(img, text, pos, scale=0.55, colour=(220,220,220), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, (10,10,10), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def _draw_skeleton(frame, pose_result):
    if pose_result is None:
        return
    connections = [
        (config.MP_RIGHT_SHOULDER, config.MP_RIGHT_HIP),
        (config.MP_RIGHT_HIP,      config.MP_RIGHT_KNEE),
        (config.MP_RIGHT_KNEE,     config.MP_RIGHT_ANKLE),
    ]
    for a_idx, b_idx in connections:
        a = pose_result.xy(a_idx)
        b = pose_result.xy(b_idx)
        if a and b:
            cv2.line(frame, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])),
                     (60,220,80), 2, cv2.LINE_AA)
    for idx in [config.MP_RIGHT_SHOULDER, config.MP_RIGHT_HIP,
                config.MP_RIGHT_KNEE, config.MP_RIGHT_ANKLE]:
        pt = pose_result.xy(idx)
        if pt:
            cv2.circle(frame, (int(pt[0]),int(pt[1])), 7, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(pt[0]),int(pt[1])), 5, (60,220,80),  -1, cv2.LINE_AA)
    # knee angle arc label
    knee = pose_result.joint_angle(
        config.MP_RIGHT_HIP, config.MP_RIGHT_KNEE, config.MP_RIGHT_ANKLE)
    kpt = pose_result.xy(config.MP_RIGHT_KNEE)
    if knee and kpt:
        col = _colour(knee, config.KNEE_EXTENSION_OPTIMAL_MIN,
                      config.KNEE_EXTENSION_OPTIMAL_MAX)
        _put(frame, f"{knee:.0f}deg", (int(kpt[0])+10, int(kpt[1])-10),
             scale=0.48, colour=col)


def _draw_crank(frame, crank_state, bb_x, bb_y):
    cv2.circle(frame, (bb_x, bb_y), 85, (50,50,50), 1, cv2.LINE_AA)
    cv2.drawMarker(frame, (bb_x, bb_y), (30,180,255), cv2.MARKER_CROSS, 16, 1)
    if crank_state.marker_px:
        mx, my = crank_state.marker_px
        col = (60,220,80) if crank_state.tracking=="detected" else (180,140,40)
        cv2.line(frame, (bb_x,bb_y), (mx,my), (100,100,200), 1, cv2.LINE_AA)
        cv2.circle(frame, (mx,my), 10, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (mx,my), 10, (255,255,255), 1, cv2.LINE_AA)


def _draw_hud(frame, m):
    h, w = frame.shape[:2]
    px = w - 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (w, h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    x, y, dy = px+10, 30, 28

    _put(frame, "VeloLens", (x, y), scale=0.7, colour=(60,220,80))
    y += 8
    cv2.line(frame, (x, y), (w-10, y), (50,50,50), 1)
    y += dy - 8

    def row(label, val, unit, lo, hi):
        nonlocal y
        col = _colour(val, lo, hi)
        val_str = f"{val:.0f}{unit}" if val is not None else "--"
        _put(frame, label, (x, y), scale=0.40, colour=(140,140,140))
        _put(frame, val_str, (x+90, y), scale=0.55, colour=col)
        y += dy - 2

    row("Cadence",  m["cadence_rpm"], " rpm", 80, 95)
    row("Knee",     m["knee_angle"],  "deg",  config.KNEE_EXTENSION_OPTIMAL_MIN,
                                              config.KNEE_EXTENSION_OPTIMAL_MAX)
    row("Torso",    m["torso_angle"], "deg",  40, 60)

    y += 4
    cv2.line(frame, (x, y), (w-10, y), (50,50,50), 1)
    y += 10

    qs = m["quality"]
    qs_col = _colour(qs, 75, 100)
    _put(frame, "Quality", (x, y), scale=0.40, colour=(140,140,140))
    _put(frame, f"{qs:.0f}/100" if qs else "--", (x+90, y), scale=0.55, colour=qs_col)
    if qs:
        bar_w = w - px - 20
        cv2.rectangle(frame, (x, y+8), (x+bar_w, y+16), (40,40,40), -1)
        cv2.rectangle(frame, (x, y+8), (x+int(bar_w*qs/100), y+16), qs_col, -1)
    y += dy + 10

    # tracking pill
    track = m["tracking"]
    tcol  = (60,220,80) if track=="detected" else \
            (30,180,255) if track=="tracked" else (60,60,200)
    cv2.rectangle(frame, (x, y-14), (x+130, y+4), tcol, -1, cv2.LINE_AA)
    _put(frame, track, (x+6, y), scale=0.40, colour=(10,10,10))
    y += dy

    # fps + elapsed
    elapsed = m["elapsed_secs"]
    mm, ss  = divmod(int(elapsed), 60)
    _put(frame, f"{mm:02d}:{ss:02d}  {m['fps']:.0f}fps",
         (x, y), scale=0.38, colour=(100,100,100))


# ── pipeline thread ───────────────────────────────────────────────────────────

def _pipeline_loop(cam, crank, pose, engine, start_time):
    global _latest_jpeg
    pose_tick  = 0
    pose_result = None
    frame_count = 0
    fps_buf = []

    while True:
        t0 = time.monotonic()
        try:
            frame_obj  = cam.read()
            frame      = frame_obj.image.copy()
            ts         = frame_obj.timestamp
            frame_count += 1

            crank_state = crank.update(frame, ts)

            pose_tick += 1
            if pose_tick >= config.POSE_SKIP_FRAMES:
                pose_result = pose.update(frame)
                pose_tick   = 0

            fm = engine.update(crank_state, pose_result)

            # update shared metrics
            knee  = None
            torso = None
            conf  = 0.0
            if pose_result:
                knee  = pose_result.joint_angle(
                    config.MP_RIGHT_HIP,
                    config.MP_RIGHT_KNEE,
                    config.MP_RIGHT_ANKLE)
                torso = pose_result.vector_angle_from_horizontal(
                    config.MP_RIGHT_HIP, config.MP_RIGHT_SHOULDER)
                hip_lm = pose_result.get(config.MP_RIGHT_HIP)
                conf   = hip_lm.visibility if hip_lm else 0.0

            # rolling quality (recompute every 30 frames)
            if frame_count % 30 == 0:
                sm = engine.finish()
                _metrics["quality"] = sm.quality_score
                # restart engine keeping history
                engine.__init__(session_id="monitor")
                engine.start()

            elapsed = time.monotonic() - start_time
            fps_buf.append(time.monotonic())
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            fps = (len(fps_buf)-1)/(fps_buf[-1]-fps_buf[0]) if len(fps_buf)>1 else 0

            _metrics.update({
                "cadence_rpm":     fm.cadence_rpm,
                "knee_angle":      knee,
                "torso_angle":     torso,
                "tracking":        crank_state.tracking,
                "pose_confidence": conf,
                "elapsed_secs":    elapsed,
                "fps":             fps,
            })

            # draw and encode
            _draw_skeleton(frame, pose_result)
            _draw_crank(frame, crank_state,
                        getattr(config, "CALIB_BB_X", None) or frame.shape[1]//2,
                        getattr(config, "CALIB_BB_Y", None) or int(frame.shape[0]*0.7))
            _draw_hud(frame, _metrics)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with _lock:
                _latest_jpeg = buf.tobytes()

        except Exception as e:
            print(f"[monitor] pipeline error: {e}")
            time.sleep(0.1)

        # pace to ~TARGET_FPS
        elapsed_frame = time.monotonic() - t0
        sleep = max(0, 1.0/config.TARGET_FPS - elapsed_frame)
        time.sleep(sleep)


# ── HTTP handler ──────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html>
<head>
<title>VeloLens Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#0e0e0e;color:#ddd;font-family:monospace;margin:0;padding:8px}
h2{color:#4ecf80;margin:0 0 6px;font-size:18px}
#stream{display:block;max-width:100%;border:1px solid #222}
#stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;
      padding:10px;text-align:center}
.label{font-size:11px;color:#666;margin-bottom:4px}
.value{font-size:24px;font-weight:500}
.green{color:#4ecf80}.amber{color:#ffb830}.red{color:#e05050}.gray{color:#666}
</style>
</head>
<body>
<h2>VeloLens — Live Monitor</h2>
<img id="stream" src="/stream" width="100%">
<div id="stats">
  <div class="card"><div class="label">Cadence</div>
    <div class="value" id="rpm">--</div><div class="label">rpm</div></div>
  <div class="card"><div class="label">Knee angle</div>
    <div class="value" id="knee">--</div><div class="label">deg (target 140-150)</div></div>
  <div class="card"><div class="label">Torso</div>
    <div class="value" id="torso">--</div><div class="label">deg (target 45-55)</div></div>
  <div class="card"><div class="label">Quality</div>
    <div class="value" id="quality">--</div><div class="label">/ 100</div></div>
</div>
<script>
function col(v,lo,hi){
  if(v===null)return'gray';
  if(v>=lo&&v<=hi)return'green';
  var m=(hi-lo)*0.3;
  if(v>=lo-m&&v<=hi+m)return'amber';
  return'red';
}
function fmt(v){return v===null?'--':Math.round(v)}
function update(){
  fetch('/metrics').then(r=>r.json()).then(d=>{
    document.getElementById('rpm').textContent=fmt(d.cadence_rpm);
    document.getElementById('rpm').className='value '+col(d.cadence_rpm,80,95);
    document.getElementById('knee').textContent=fmt(d.knee_angle);
    document.getElementById('knee').className='value '+col(d.knee_angle,140,150);
    document.getElementById('torso').textContent=fmt(d.torso_angle);
    document.getElementById('torso').className='value '+col(d.torso_angle,40,60);
    document.getElementById('quality').textContent=fmt(d.quality);
    document.getElementById('quality').className='value '+col(d.quality,75,100);
  });
}
setInterval(update,500);
update();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            data = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)

        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpg = _latest_jpeg
                    if jpg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpg + b"\r\n")
                    time.sleep(0.08)
            except Exception:
                pass

        elif path == "/metrics":
            data = json.dumps(_metrics).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    from capture.calibration import load_or_run
    print("[monitor] Starting camera ...")
    cam   = Camera().start()
    calib = load_or_run(cam)

    crank  = CrankTracker(bb_x=calib.bb_x, bb_y=calib.bb_y)
    pose   = PoseEstimator()
    engine = MetricsEngine(session_id="monitor")
    engine.start()

    start_time = time.monotonic()

    pipeline_thread = threading.Thread(
        target=_pipeline_loop,
        args=(cam, crank, pose, engine, start_time),
        daemon=True
    )
    pipeline_thread.start()

    ip = os.popen("hostname -I 2>/dev/null").read().strip().split()[0]
    print(f"[monitor] Open:  http://{ip}:{args.port}")
    print( "[monitor] Ctrl-C to stop\n")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        pose.close()
        print("[monitor] Stopped.")
