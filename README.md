# VeloLens

Real-time indoor cycling analysis on a Raspberry Pi 4. Uses computer vision to measure cadence, knee extension angle, and torso posture — no power meter, no extra sensors.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

- **Cadence tracking** — HSV colour masking + Lucas-Kanade optical flow on a pedal marker. Broadcasts over BLE as a standard CSC sensor (pairs with MyWhoosh, Zwift, Wahoo)
- **Pose estimation** — MoveNet Lightning via TFLite at ~15fps on-device. Tracks hip, knee, ankle, shoulder landmarks
- **Knee extension angle** — computed at bottom dead centre from joint vectors; target 140–150°
- **Torso angle** — shoulder-to-hip vector vs horizontal; proxy for aero position and fatigue
- **Quality score** — composite 0–100 across cadence smoothness (35%), cadence target (20%), knee extension (25%), posture stability (20%)
- **Session storage** — SQLite + JSON export per session
- **Strava integration** — auto-annotates the matching Strava activity after each session
- **Web calibration UI** — browser-based BB centre and marker colour calibration (no SSH file editing)
- **Live monitor** — browser-based live feed with skeleton overlay, joint angles, and metrics dashboard
- **Annotated video export** — burn overlays onto recorded clips for review or sharing

---

## Hardware

| Item | Notes |
|---|---|
| Raspberry Pi 4 (4GB+) | Main compute |
| Pi Camera v2 (IMX219) | Side-profile view of rider |
| Smart trainer | Any — VeloLens is sensor-agnostic |
| Marker sticker | White tape / retroreflective strip on pedal. See [Marker setup](#marker-setup) |

**Camera placement:** 2–3 m from bike, lens at crank height, perpendicular to the drivetrain side. Mount left-side (non-drivetrain) for clearest view of pedal. Full body must be in frame: shoulder through ankle.

---

## Installation

```bash
# Clone
git clone https://github.com/<you>/VeloLens.git && cd VeloLens

# Create venv with system-site-packages (required for picamera2 on Pi OS Bookworm)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Download MoveNet Lightning model (~3 MB, run once)
python -m cv.pose --download
```

---

## First-time setup

### 1. Calibrate bottom bracket centre and marker colour

Start the web calibration UI:

```bash
python -m tools.calibration_server
# Open http://<pi-ip>:5000 in your laptop browser
```

- Click **Set BB centre** → click the crank spindle bolt on the live feed
- Click **Sample marker colour** → click the marker sticker (or use `bright` mode for white tape)
- Click **Toggle mask** → verify the sticker highlights green
- Ctrl-C when done — `config.py` updates automatically

### 2. Strava (optional)

Create an API app at [strava.com/settings/api](https://www.strava.com/settings/api). Set callback domain to `localhost`. Add credentials to `config.py`:

```python
STRAVA_CLIENT_ID     = "12345"
STRAVA_CLIENT_SECRET = "abc..."
```

Authorise once:

```bash
python -m strava.client --auth
```

---

## Running a session

```bash
# Headless (SSH)
python main.py --no-preview --duration 3600

# With live browser monitor (open http://<pi-ip>:5001)
python -m tools.monitor_server

# Timed with preview
python main.py --duration 1800
```

Session output:
- Terminal quality report on completion
- `sessions/<id>.json` — full metrics
- `sessions/veloLens.db` — SQLite (queryable for trends)
- Strava activity annotated (if configured)
- BLE cadence broadcast throughout (pairs as `VeloLens`)

---

## Marker setup

The tracker detects a small high-contrast marker attached to the pedal platform.

**Best options (in order):**
1. White electrical tape — use `MARKER_MODE = 'bright'` in `config.py`
2. Retroreflective strip (bike helmet tape) — use `bright` mode
3. Neon orange/yellow sticker — use `hsv` mode, calibrate with the web UI

**Avoid:** colours that match the bike frame, wall, or floor. The calibration server's mask overlay shows exactly what the tracker sees.

---

## Tools

| Tool | Command | Purpose |
|---|---|---|
| Calibration server | `python -m tools.calibration_server` | Web UI for BB + marker colour |
| Live monitor | `python -m tools.monitor_server` | Browser feed with skeleton overlay |
| Record session | `python -m tools.record_session --duration 30` | Save raw footage |
| Render annotated | `python -m tools.render_annotated --video sessions/clip.avi --slow 0.5` | Export annotated MP4 |
| Test pipeline | `python -m tools.test_pipeline` | Synthetic end-to-end test (no camera) |
| HSV tuner | `python -m tools.hsv_tuner --headless --apply` | Sample marker colour headlessly |
| AWB tuner | `python -m tools.awb_tune` | White balance contact sheet |

---

## Configuration

All tunable values live in `config.py`. Key settings:

```python
# Camera
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
CAMERA_FLIP         = 0        # 0=vertical flip, 1=horizontal, -1=both, None=off

# Marker detection
MARKER_MODE         = 'bright' # 'bright' | 'hsv' | 'dark'
MARKER_BRIGHT_THRESH = 210     # V-channel threshold for bright/white markers

# Calibration (set these to skip the GUI entirely)
CALIB_BB_X          = None     # e.g. 640
CALIB_BB_Y          = None     # e.g. 540

# BLE
BLE_ENABLED         = True
BLE_DEVICE_NAME     = "VeloLens"

# Thresholds
KNEE_EXTENSION_OPTIMAL_MIN = 140.0
KNEE_EXTENSION_OPTIMAL_MAX = 150.0
```

---

## Quality score

| Component | Weight | What's measured | Target |
|---|---|---|---|
| Cadence smoothness | 35% | Coefficient of variation of RPM | CV < 0.05 |
| Cadence target | 20% | How close avg cadence is to 80–95 RPM | 80–95 RPM |
| Knee extension | 25% | Average knee angle at BDC | 140–150° |
| Posture stability | 20% | Torso angle drift start vs end | < 2° |

---

## Project structure

```
VeloLens/
├── main.py              # Session entry point
├── config.py            # All tunable constants
├── requirements.txt
├── capture/
│   ├── camera.py        # picamera2 + OpenCV fallback, auto-detection
│   └── calibration.py   # BB calibration (headless / GUI / config)
├── cv/
│   ├── pose.py          # MoveNet Lightning via TFLite
│   ├── crank.py         # HSV masking + Lucas-Kanade optical flow
│   └── filters.py       # Kalman filter, rolling stats, angle unwrapper
├── processing/
│   └── metrics.py       # MetricsEngine + quality score
├── storage/
│   └── db.py            # SQLite schema + session persistence
├── ble/
│   └── cadence_server.py # GATT peripheral (CSC 0x1816 + DevInfo)
├── strava/
│   └── client.py        # OAuth2 + activity matching + annotation
├── tools/
│   ├── calibration_server.py  # Web UI: BB centre + marker colour
│   ├── monitor_server.py      # Web UI: live skeleton feed + metrics
│   ├── record_session.py      # Raw footage recorder
│   ├── render_annotated.py    # Annotated MP4 exporter
│   ├── test_pipeline.py       # Synthetic + replay pipeline tester
│   ├── hsv_tuner.py           # HSV range calibrator (headless)
│   └── awb_tune.py            # White balance sweep tool
├── models/
│   └── movenet_lightning.tflite  # Downloaded on first run
└── tests/
    └── test_metrics.py
```

---

## Roadmap

- [ ] LLM coaching (Gemini API — session metrics → plain-English feedback)
- [ ] Fatigue modelling across multiple sessions
- [ ] Dead-spot detection via FFT on cadence signal
- [ ] L/R balance estimation from bilateral knee curves
- [ ] Flask REST API
- [ ] Android companion app
- [ ] Multi-session trend dashboard

---

## Tech stack

| Layer | Technology |
|---|---|
| Pose estimation | MoveNet Lightning (TFLite INT8) |
| Camera | picamera2 + libcamera |
| CV | OpenCV 4.x (headless) |
| BLE | bless (BlueZ D-Bus GATT peripheral) |
| Storage | SQLite via Python stdlib |
| Strava | OAuth2 + REST API |
| Web UI tools | Python stdlib HTTPServer (ThreadingHTTPServer) |
| Tests | pytest |

---

## Known limitations

- Requires full right/left-side body profile in frame — shoulder through ankle must be visible
- Marker tracking degrades in low light or when the sticker is occluded by the foot for >5 frames
- L/R power balance is estimated from knee curve symmetry, not measured — treat as qualitative
- BLE GATT peripheral requires bless + BlueZ ≥ 5.50 (standard on Pi OS Bookworm)

---

## Running tests

```bash
python -m pytest tests/ -v
```

14 unit tests covering Kalman filter, rolling stats, angle unwrapper, quality score, and joint angle math.

---

## License

MIT
