"""
VeloLens configuration.
Edit this file to tune thresholds and camera settings for your setup.
"""

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0
# Flip the frame if the camera is mounted upside down.
#   0 = flip vertical (upside down mount)
#   1 = flip horizontal (mirror)
#  -1 = flip both
#  None = no flip
CAMERA_FLIP         = 0
# White balance gains (red_gain, blue_gain).
# Default (1.5, 1.2) fixes the common blue-hue issue on IMX219 indoors.
# Run: python -m tools.awb_tune   to find ideal values for your lighting.
# Or set AwbEnable back to True in camera.py to use auto (accepts blue tint).
CAMERA_COLOUR_GAINS = (1.5, 1.2)          # /dev/video0 for Pi Camera via v4l2
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
TARGET_FPS          = 30
# Run pose estimation every N frames (2 = ~15 pose fps, saves ~40% CPU)
POSE_SKIP_FRAMES    = 2

# ── Crank marker (HSV colour range) ──────────────────────────────────────────
# Default tuned for a bright orange sticker (Avery neon orange).
# Run `python -m veloLens.tools.hsv_tuner` to calibrate for your sticker colour.
MARKER_HSV_LOWER    = (5,  150, 150)   # H, S, V
MARKER_HSV_UPPER    = (25, 255, 255)
MARKER_MIN_AREA_PX  = 40               # ignore blobs smaller than this
MARKER_MAX_AREA_PX  = 4000

# Marker detection mode:
#   'hsv'    — coloured sticker (tune HSV range with calibration_server.py)
#   'bright' — white/silver tape, retroreflective strip, or coin
#   'dark'   — black marker on light crank
MARKER_MODE         = 'bright'
MARKER_BRIGHT_THRESH = 210    # V-channel threshold (0-255); raise if background is bright
MARKER_DARK_THRESH   = 40     # V-channel threshold for dark marker mode

# ── Cadence ───────────────────────────────────────────────────────────────────
CADENCE_SMOOTH_WIN  = 5        # rolling median window (frames)
CADENCE_MIN_RPM     = 30       # below this → treat as not pedalling
CADENCE_MAX_RPM     = 160      # above this → likely tracking glitch

# ── Pose landmarks (MediaPipe index references) ───────────────────────────────
# Right side (camera-facing on a side-profile shot)
MP_RIGHT_HIP        = 11  # left hip (camera-facing side)
MP_RIGHT_KNEE       = 13  # left knee
MP_RIGHT_ANKLE      = 15  # left ankle
MP_RIGHT_SHOULDER   = 5   # left shoulder
MP_RIGHT_HEEL       = 15  # left ankle proxy
MP_RIGHT_TOE        = 15  # left ankle proxy
# Non-camera side (right side of body, partially occluded)
MP_LEFT_HIP         = 12
MP_LEFT_KNEE        = 14
MP_LEFT_ANKLE       = 16
MP_LEFT_SHOULDER    = 6

# ── Angle thresholds ──────────────────────────────────────────────────────────
KNEE_EXTENSION_OPTIMAL_MIN = 140.0   # degrees at bottom dead centre
KNEE_EXTENSION_OPTIMAL_MAX = 150.0
TORSO_AERO_TARGET          = 45.0    # degrees from horizontal (aggressive)
TORSO_ENDURANCE_TARGET     = 55.0    # degrees from horizontal (endurance)

# ── Calibration hardcode (optional) ──────────────────────────────────────────
# If set, calibration runs automatically without any GUI or CLI step.
# Measure BB pixel coords from a saved frame (python -m capture.calibration --headless)
CALIB_BB_X = None   # e.g. 310
CALIB_BB_Y = None   # e.g. 380

# ── Session ───────────────────────────────────────────────────────────────────
SESSION_DIR         = "sessions"     # where JSON + SQLite files are written
MIN_SESSION_SECS    = 10             # discard sessions shorter than this
METRICS_SAMPLE_HZ   = 5             # downsample metrics to this rate for storage

# ── Kalman filter (1-D per landmark coordinate) ───────────────────────────────
KALMAN_PROCESS_NOISE     = 0.01
KALMAN_MEASUREMENT_NOISE = 0.1

# ── BLE cadence broadcast ─────────────────────────────────────────────────────
# Requires: pip install bleak bless --break-system-packages
BLE_ENABLED         = True
BLE_DEVICE_NAME     = "VeloLens"

# ── Strava API ────────────────────────────────────────────────────────────────
# Create an app at https://www.strava.com/settings/api, then fill these in.
# First-time auth: python -m veloLens.strava.client --auth
STRAVA_CLIENT_ID     = ""    # e.g. "12345"
STRAVA_CLIENT_SECRET = ""    # e.g. "abc123..."
# Auto-annotate matching Strava activity after each session (requires auth)
STRAVA_AUTO_ANNOTATE = True
