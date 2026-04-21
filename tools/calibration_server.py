"""
tools/calibration_server.py
────────────────────────────
Web-based calibration UI served from the Pi.
Open http://<pi-ip>:5000 in your laptop browser.

  Step 1 — click 'Set BB centre' then click the crank spindle
  Step 2 — click 'Sample marker colour' then click the sticker
  Step 3 — toggle mask overlay to verify detection
  Step 4 — Ctrl-C when done

Saves immediately to calibration.json and config.py on every click.
"""

import json
import os
import re
import sys
import time
import threading

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from capture.camera import Camera
from capture.calibration import CalibrationData

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")

# ── mutable server state (dict so handlers can mutate freely) ─────────────────
_state = {
    "bb_x":       None,
    "bb_y":       None,
    "hsv_lower":  [5,  150, 150],
    "hsv_upper":  [25, 255, 255],
    "show_mask":  False,
    "mode":       "bright",   # 'hsv' | 'bright' | 'dark'
    "bright_thresh": 210,
}
_cam: Camera = None
_lock = threading.Lock()


# ── config.py helpers ─────────────────────────────────────────────────────────

def _read_config_value(key: str):
    """Read a single value from config.py at runtime (always fresh)."""
    src = open(CONFIG_PATH).read()
    m   = re.search(r"^" + re.escape(key) + r"\s*=\s*([^#\n]+)", src, re.MULTILINE)
    return m.group(1).strip() if m else None


def _patch_config(key: str, value: str):
    """Write a value to config.py, preserving inline comments."""
    with _lock:
        src = open(CONFIG_PATH).read()
        new_src = re.sub(
            r"^" + re.escape(key) + r"[ \t]*=[ \t]*[^#\n]*?[ \t]*(?=#|$)",
            key + " = " + value + "   ",
            src, flags=re.MULTILINE
        )
        if new_src == src:
            # key doesn't exist — append it
            new_src = src.rstrip() + f"\n{key} = {value}\n"
        open(CONFIG_PATH, "w").write(new_src)
    print(f"[cal-server] config.py: {key} = {value}")


def _reload_state():
    """Sync _state from current config.py values."""
    mode = _read_config_value("MARKER_MODE")
    if mode:
        _state["mode"] = mode.strip("'\"")
    thresh = _read_config_value("MARKER_BRIGHT_THRESH")
    if thresh:
        try:
            _state["bright_thresh"] = int(thresh)
        except ValueError:
            pass
    bbx = _read_config_value("CALIB_BB_X")
    bby = _read_config_value("CALIB_BB_Y")
    if bbx and bbx != "None":
        try:
            _state["bb_x"] = int(bbx)
            _state["bb_y"] = int(bby)
        except (ValueError, TypeError):
            pass


# ── frame builder ─────────────────────────────────────────────────────────────

def _build_frame() -> np.ndarray:
    frame = _cam.read().image.copy()
    h, w  = frame.shape[:2]

    if _state["show_mask"]:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mode = _state["mode"]
        if mode == "bright":
            t    = _state["bright_thresh"]
            mask = cv2.threshold(hsv[:, :, 2], t, 255, cv2.THRESH_BINARY)[1]
        elif mode == "dark":
            mask = cv2.threshold(hsv[:, :, 2], 40, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            lo   = np.array(_state["hsv_lower"])
            hi   = np.array(_state["hsv_upper"])
            mask = cv2.inRange(hsv, lo, hi)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # apply area filter — same bounds as the tracker uses
        import importlib, sys
        cfg = importlib.reload(sys.modules['config'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid   = [c for c in contours
                   if cfg.MARKER_MIN_AREA_PX < cv2.contourArea(c) < cfg.MARKER_MAX_AREA_PX]
        too_small = [c for c in contours if cv2.contourArea(c) <= cfg.MARKER_MIN_AREA_PX]
        too_large = [c for c in contours if cv2.contourArea(c) >= cfg.MARKER_MAX_AREA_PX]

        # rebuild mask from valid contours only
        mask_valid = np.zeros_like(mask)
        cv2.drawContours(mask_valid, valid, -1, 255, -1)

        overlay = frame.copy()
        overlay[mask_valid > 0] = [0, 220, 80]
        frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

        # valid blobs — green outline + area label
        for c in valid:
            cv2.drawContours(frame, [c], -1, (0, 255, 80), 2)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, f"{int(cv2.contourArea(c))}px",
                            (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80), 1)
        # too-small blobs — red outline
        cv2.drawContours(frame, too_small, -1, (60, 60, 220), 1)
        # too-large blobs — yellow outline
        cv2.drawContours(frame, too_large, -1, (0, 220, 255), 1)

        # legend
        cv2.putText(frame, f"min:{cfg.MARKER_MIN_AREA_PX} max:{cfg.MARKER_MAX_AREA_PX}",
                    (6, frame.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (180, 180, 180), 1)

    # BB crosshair
    if _state["bb_x"] is not None:
        cx, cy = int(_state["bb_x"]), int(_state["bb_y"])
        r = int(min(w, h) * 0.14)
        cv2.circle(frame, (cx, cy), r,  (0, 200, 80), 1)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 100), cv2.MARKER_CROSS, 28, 2)

    # status bar
    mode_str   = _state["mode"]
    bb_str     = f"BB:({_state['bb_x']},{_state['bb_y']})" if _state["bb_x"] else "BB:not set"
    mask_str   = "MASK:ON" if _state["show_mask"] else "MASK:OFF"
    status     = f"  {bb_str}   mode:{mode_str}   {mask_str}"
    cv2.rectangle(frame, (0, 0), (w, 22), (15, 15, 15), -1)
    cv2.putText(frame, status, (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 255, 180), 1)
    return frame


def _jpeg() -> bytes:
    _, buf = cv2.imencode(".jpg", _build_frame(),
                          [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes()


# ── actions ───────────────────────────────────────────────────────────────────

def _action_set_bb(x: int, y: int) -> str:
    frame = _cam.read().image
    h, w  = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return f"Out of bounds ({x},{y}) for {w}x{h} frame"
    _state["bb_x"] = x
    _state["bb_y"] = y
    calib = CalibrationData(bb_x=x, bb_y=y, frame_width=w,
                            frame_height=h, timestamp=time.time())
    calib.save()
    _patch_config("CALIB_BB_X", str(x))
    _patch_config("CALIB_BB_Y", str(y))
    return f"BB saved: ({x}, {y})"


def _action_sample_hsv(x: int, y: int, radius: int = 15) -> str:
    frame = _cam.read().image
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x - radius);  y1 = max(0, y - radius)
    x2 = min(w_img, x + radius);  y2 = min(h_img, y + radius)
    roi   = hsv[y1:y2, x1:x2]
    h_med = int(np.median(roi[:, :, 0]))
    s_med = int(np.median(roi[:, :, 1]))
    v_med = int(np.median(roi[:, :, 2]))
    lo = (max(0,   h_med - 15), max(0,   s_med - 50), max(0,   v_med - 50))
    hi = (min(179, h_med + 15), min(255, s_med + 50), min(255, v_med + 50))
    _state["hsv_lower"] = list(lo)
    _state["hsv_upper"] = list(hi)
    _patch_config("MARKER_HSV_LOWER", f"({lo[0]}, {lo[1]}, {lo[2]})")
    _patch_config("MARKER_HSV_UPPER", f"({hi[0]}, {hi[1]}, {hi[2]})")
    _patch_config("MARKER_MODE", "'hsv'")
    _state["mode"] = "hsv"
    return f"HSV saved: lo={lo} hi={hi}  (mode set to hsv)"


def _action_set_mode(mode: str) -> str:
    _state["mode"] = mode
    _patch_config("MARKER_MODE", f"'{mode}'")
    return f"Mode set to: {mode}"


def _action_set_thresh(t: int) -> str:
    _state["bright_thresh"] = t
    _patch_config("MARKER_BRIGHT_THRESH", str(t))
    return f"Brightness threshold set to: {t}"


# ── HTTP handler ──────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html>
<head>
<title>VeloLens Calibration</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#111;color:#ddd;font-family:monospace;margin:0;padding:10px}
h2{color:#4ecf80;margin:0 0 8px}
#stream{display:block;max-width:100%;cursor:crosshair;border:1px solid #333;margin-bottom:8px}
.row{display:flex;gap:6px;margin:4px 0;flex-wrap:wrap;align-items:center}
button{background:#1e1e1e;color:#4ecf80;border:1px solid #4ecf80;padding:7px 14px;
       cursor:pointer;border-radius:4px;font-size:13px}
button:hover{background:#4ecf80;color:#111}
button.active{background:#4ecf80;color:#111}
select,input[type=range]{background:#1e1e1e;color:#ddd;border:1px solid #444;
       padding:4px;border-radius:4px}
#mode-hint{color:#ff9;font-size:12px;padding:4px 0}
#status{color:#8f8;font-size:13px;min-height:18px;padding:2px 0}
label{font-size:12px;color:#aaa}
</style>
</head>
<body>
<h2>VeloLens Calibration</h2>

<div class="row">
  <button id="btn-bb"  onclick="setMode('bb')">1. Set BB centre</button>
  <button id="btn-hsv" onclick="setMode('hsv')">2. Sample HSV colour</button>
  <button id="btn-mask" onclick="toggleMask()">Toggle mask</button>
</div>

<div class="row">
  <label>Marker mode:</label>
  <select id="marker-mode" onchange="setMarkerMode(this.value)">
    <option value="bright">bright (white/silver tape)</option>
    <option value="hsv">hsv (coloured sticker)</option>
    <option value="dark">dark (black marker)</option>
  </select>
</div>

<div class="row" id="thresh-row">
  <label>Brightness threshold: <span id="thresh-val">210</span></label>
  <input type="range" id="thresh" min="100" max="254" value="210"
         oninput="document.getElementById('thresh-val').textContent=this.value"
         onchange="setThresh(this.value)">
</div>

<div id="mode-hint">Click a button then click the image</div>
<img id="stream" src="/stream" width="800">
<div id="status">Ready.</div>

<script>
var clickMode = null;

function setMode(m) {
  clickMode = m;
  document.getElementById('mode-hint').textContent =
    m === 'bb'  ? '▶ Click the bottom bracket axle centre on the image' :
                  '▶ Click the marker sticker on the image';
  ['btn-bb','btn-hsv'].forEach(id =>
    document.getElementById(id).classList.remove('active'));
  document.getElementById('btn-' + m).classList.add('active');
}

function toggleMask() {
  fetch('/toggle_mask').then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = 'Mask: ' + (d.mask ? 'ON' : 'OFF');
    document.getElementById('btn-mask').classList.toggle('active', d.mask);
  });
}

function setMarkerMode(v) {
  fetch('/set_marker_mode?mode=' + v).then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = d.msg;
    document.getElementById('thresh-row').style.display =
      v === 'bright' ? '' : 'none';
  });
}

function setThresh(v) {
  fetch('/set_thresh?t=' + v).then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = d.msg;
  });
}

document.getElementById('stream').addEventListener('click', function(e) {
  if (!clickMode) { document.getElementById('status').textContent = 'Select step 1 or 2 first.'; return; }
  var rect  = this.getBoundingClientRect();
  var scaleX = this.naturalWidth  / this.clientWidth;
  var scaleY = this.naturalHeight / this.clientHeight;
  var x = Math.round((e.clientX - rect.left) * scaleX);
  var y = Math.round((e.clientY - rect.top)  * scaleY);
  document.getElementById('status').textContent = 'Saving...';
  var url = clickMode === 'bb' ? '/set_bb?x='+x+'&y='+y : '/set_hsv?x='+x+'&y='+y;
  fetch(url).then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = d.msg;
    clickMode = null;
    ['btn-bb','btn-hsv'].forEach(id =>
      document.getElementById(id).classList.remove('active'));
    document.getElementById('mode-hint').textContent = 'Done! Click again for another step.';
  });
});

// init select from server
fetch('/state').then(r=>r.json()).then(d=>{
  document.getElementById('marker-mode').value = d.mode;
  document.getElementById('thresh').value = d.bright_thresh;
  document.getElementById('thresh-val').textContent = d.bright_thresh;
  if (d.mode !== 'bright')
    document.getElementById('thresh-row').style.display = 'none';
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)

        if p.path == "/":
            self._html(_HTML)

        elif p.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = _jpeg()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpg + b"\r\n"
                    )
                    time.sleep(0.1)
            except Exception:
                pass

        elif p.path == "/state":
            self._json({
                "mode":          _state["mode"],
                "bright_thresh": _state["bright_thresh"],
                "bb_x":          _state["bb_x"],
                "bb_y":          _state["bb_y"],
            })

        elif p.path == "/set_bb":
            msg = _action_set_bb(int(q["x"][0]), int(q["y"][0]))
            self._json({"msg": msg})

        elif p.path == "/set_hsv":
            msg = _action_sample_hsv(int(q["x"][0]), int(q["y"][0]))
            self._json({"msg": msg})

        elif p.path == "/toggle_mask":
            _state["show_mask"] = not _state["show_mask"]
            self._json({"mask": _state["show_mask"]})

        elif p.path == "/set_marker_mode":
            msg = _action_set_mode(q["mode"][0])
            self._json({"msg": msg})

        elif p.path == "/set_thresh":
            msg = _action_set_thresh(int(q["t"][0]))
            self._json({"msg": msg})

        else:
            self.send_response(404)
            self.end_headers()

    def _html(self, body: str):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    _reload_state()
    print("[cal-server] Starting camera ...")
    _cam = Camera().start()

    ip = os.popen("hostname -I 2>/dev/null").read().strip().split()[0]
    print(f"\n[cal-server] Open in your browser:  http://{ip}:{args.port}\n")
    print( "  1. Set BB centre  — click button, then click the crank axle bolt")
    print( "  2. Sample colour  — click button, then click the marker sticker")
    print( "     (or use 'bright' mode for white tape — no colour sampling needed)")
    print( "  3. Toggle mask    — verify the marker is highlighted green")
    print( "  4. Ctrl-C to stop — config.py is saved after every click\n")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _cam.stop()
        print("[cal-server] Stopped.")
