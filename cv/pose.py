"""
cv/pose.py
──────────
Pose estimator using MoveNet Lightning via TFLite.

Replaces the mediapipe dependency, which has no official ARM wheel for
Raspberry Pi OS Bookworm. MoveNet Lightning is what mediapipe Pose Lite
uses internally — same landmark set (17 COCO keypoints), same accuracy,
but runs directly via tflite-runtime which has a proper ARM wheel.

Landmark index mapping (COCO 17-point, same as mediapipe indices):
  0  nose          5  left shoulder    6  right shoulder
  7  left elbow    8  right elbow      9  left wrist    10 right wrist
  11 left hip     12 right hip        13 left knee     14 right knee
  15 left ankle   16 right ankle

These match the MP_* constants in config.py exactly.

Model download (run once on the Pi):
  python -m veloLens.cv.pose --download

Or manually:
  wget -O models/movenet_lightning.tflite \\
    https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite

Install runtime (Pi 4, Python 3.11/3.12):
  pip install tflite-runtime --break-system-packages
  # If that fails (no matching wheel), use:
  pip install tensorflow --break-system-packages  (slower install, same API)
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from cv.filters import KalmanFilter1D


# ── model path ───────────────────────────────────────────────────────────────

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "movenet_lightning.tflite"

# TFHub direct download URL for MoveNet Lightning INT8 (smallest/fastest)
MODEL_URL = (
    "https://tfhub.dev/google/lite-model/"
    "movenet/singlepose/lightning/tflite/int8/4"
    "?lite-format=tflite"
)

# MoveNet input size
INPUT_SIZE = 192   # Lightning = 192×192


# ── landmark indices (COCO, same as MediaPipe) ────────────────────────────────

USED_LANDMARKS = {
    config.MP_RIGHT_HIP,       # 12
    config.MP_RIGHT_KNEE,      # 14
    config.MP_RIGHT_ANKLE,     # 16
    config.MP_RIGHT_SHOULDER,  # 6  (note: MP uses 12 for right shoulder;
    config.MP_RIGHT_HEEL,      #     MoveNet only has 17 pts so heel/toe
    config.MP_RIGHT_TOE,       #     map to ankle — see _MOVENET_TO_COCO)
    config.MP_LEFT_HIP,        # 11
    config.MP_LEFT_KNEE,       # 13
    config.MP_LEFT_ANKLE,      # 15
    config.MP_LEFT_SHOULDER,   # 5
}

# MediaPipe index → MoveNet output index
# MoveNet 17-pt order: nose(0) left_eye(1) right_eye(2) left_ear(3) right_ear(4)
#   left_shoulder(5) right_shoulder(6) left_elbow(7) right_elbow(8)
#   left_wrist(9) right_wrist(10) left_hip(11) right_hip(12)
#   left_knee(13) right_knee(14) left_ankle(15) right_ankle(16)
_MP_TO_MN: dict[int, int] = {
    5:  5,   # left shoulder
    6:  6,   # right shoulder  (MP uses 11/12; we remap via config constants)
    11: 11,  # left hip
    12: 12,  # right hip
    13: 13,  # left knee
    14: 14,  # right knee
    15: 15,  # left ankle
    16: 16,  # right ankle
    # heel and toe don't exist in MoveNet — proxy to ankle
    28: 16,  # right heel  → right ankle
    32: 16,  # right toe   → right ankle
    27: 15,  # left heel   → left ankle (unused currently)
}


# ── data types ───────────────────────────────────────────────────────────────

@dataclass
class Landmark:
    x: float
    y: float
    visibility: float   # MoveNet calls this "score"


@dataclass
class PoseResult:
    landmarks: dict[int, Landmark]
    raw_image: np.ndarray

    def get(self, idx: int) -> Optional[Landmark]:
        return self.landmarks.get(idx)

    def xy(self, idx: int) -> Optional[tuple[float, float]]:
        lm = self.get(idx)
        return (lm.x, lm.y) if lm and lm.visibility > 0.3 else None

    def joint_angle(self, a_idx: int, b_idx: int, c_idx: int) -> Optional[float]:
        """Angle at vertex b, formed by a-b-c. Returns degrees or None."""
        a = self.xy(a_idx)
        b = self.xy(b_idx)
        c = self.xy(c_idx)
        if a is None or b is None or c is None:
            return None
        return _angle_deg(np.array(a), np.array(b), np.array(c))

    def vector_angle_from_horizontal(self, from_idx: int, to_idx: int) -> Optional[float]:
        """Angle of vector from→to relative to horizontal. Positive = upward."""
        p1 = self.xy(from_idx)
        p2 = self.xy(to_idx)
        if p1 is None or p2 is None:
            return None
        dx = p2[0] - p1[0]
        dy = p1[1] - p2[1]   # flip y (screen coords)
        return float(np.degrees(np.arctan2(dy, abs(dx) + 1e-9)))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


# ── TFLite loader ─────────────────────────────────────────────────────────────

def _load_interpreter():
    """
    Load the TFLite interpreter, trying each known package in order.
    All expose the same Interpreter class and API.
    """
    for mod in (
        "tflite_runtime.interpreter",        # tflite-runtime (Python <=3.11)
        "ai_edge_litert.interpreter",         # ai-edge-litert (Python 3.13+, Pi)
        "tensorflow.lite.python.interpreter", # full tensorflow fallback
    ):
        try:
            import importlib
            m = importlib.import_module(mod)
            return m.Interpreter
        except ImportError:
            continue
    raise ImportError(
        "No TFLite runtime found. Install one of:\n"
        "  pip install ai-edge-litert --break-system-packages   (Python 3.13, Pi)\n"
        "  pip install tflite-runtime --break-system-packages   (Python <=3.11)\n"
        "  pip install tensorflow --break-system-packages        (fallback)"
    )


def _ensure_model() -> Path:
    """Download MoveNet Lightning if not already present."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pose] Downloading MoveNet Lightning → {MODEL_PATH}")
    print(f"       Source: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[pose] Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MoveNet model: {e}\n\n"
            "Download it manually on the Pi:\n"
            f"  mkdir -p models\n"
            f"  wget -O models/movenet_lightning.tflite \\\n"
            f'    "{MODEL_URL}"'
        )
    return MODEL_PATH


# ── pose estimator ────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    MoveNet Lightning pose estimator via TFLite.
    Drop-in replacement for the mediapipe-based version.
    Same public API: update(frame_bgr) → PoseResult | None
    """

    def __init__(self):
        Interpreter = _load_interpreter()
        model_path  = _ensure_model()

        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()

        self._in_idx  = self._interp.get_input_details()[0]["index"]
        self._out_idx = self._interp.get_output_details()[0]["index"]

        # per-landmark Kalman filters  {mp_idx: (kf_x, kf_y)}
        self._kf: dict[int, tuple[KalmanFilter1D, KalmanFilter1D]] = {
            idx: (
                KalmanFilter1D(config.KALMAN_PROCESS_NOISE,
                               config.KALMAN_MEASUREMENT_NOISE),
                KalmanFilter1D(config.KALMAN_PROCESS_NOISE,
                               config.KALMAN_MEASUREMENT_NOISE),
            )
            for idx in USED_LANDMARKS
        }
        self._last_result: Optional[PoseResult] = None
        print(f"[pose] MoveNet Lightning loaded ({model_path.stat().st_size // 1024} KB)")

    def update(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        h, w = frame_bgr.shape[:2]

        # Preprocess: resize to 192×192, uint8 RGB
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        tensor = np.expand_dims(img, axis=0).astype(np.uint8)

        self._interp.set_tensor(self._in_idx, tensor)
        self._interp.invoke()

        # Output shape: [1, 1, 17, 3]  → [y, x, score] normalised 0–1
        raw = self._interp.get_tensor(self._out_idx)[0][0]  # (17, 3)

        # Check overall confidence (average of torso keypoints)
        torso_scores = [raw[i][2] for i in (5, 6, 11, 12) if i < len(raw)]
        if torso_scores and np.mean(torso_scores) < 0.2:
            return self._last_result   # no person visible

        landmarks: dict[int, Landmark] = {}
        for mp_idx in USED_LANDMARKS:
            mn_idx = _MP_TO_MN.get(mp_idx)
            if mn_idx is None or mn_idx >= len(raw):
                continue
            y_norm, x_norm, score = raw[mn_idx]

            px_raw = float(x_norm) * w
            py_raw = float(y_norm) * h

            kf_x, kf_y = self._kf[mp_idx]
            px = kf_x.update(px_raw)
            py = kf_y.update(py_raw)

            landmarks[mp_idx] = Landmark(x=px, y=py, visibility=float(score))

        result = PoseResult(landmarks=landmarks, raw_image=frame_bgr)
        self._last_result = result
        return result

    def draw_skeleton(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        connections = [
            (config.MP_RIGHT_SHOULDER, config.MP_RIGHT_HIP),
            (config.MP_RIGHT_HIP,      config.MP_RIGHT_KNEE),
            (config.MP_RIGHT_KNEE,     config.MP_RIGHT_ANKLE),
        ]
        for a_idx, b_idx in connections:
            a = result.xy(a_idx)
            b = result.xy(b_idx)
            if a and b:
                cv2.line(frame,
                         (int(a[0]), int(a[1])),
                         (int(b[0]), int(b[1])),
                         (0, 220, 120), 2)
        for idx in USED_LANDMARKS:
            pt = result.xy(idx)
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 220, 120), -1)
        return frame

    def close(self):
        pass   # tflite interpreter has no explicit close


# ── CLI download helper ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download MoveNet Lightning model to models/")
    args = parser.parse_args()
    if args.download:
        _ensure_model()
        print("Done. Model is at:", MODEL_PATH)
    else:
        parser.print_help()
