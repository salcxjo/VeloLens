"""
cv/crank.py
───────────
Crank marker tracker.

Strategy:
  1. Detect the marker each frame via HSV colour masking → centroid
  2. Track with Lucas-Kanade sparse optical flow between detections
     (handles motion blur and brief occlusion gracefully)
  3. Compute raw angle from the bottom bracket centre (set at calibration)
  4. Compute cadence (RPM) from angular velocity with rolling-median smoothing

CrankState is emitted each frame:
  - angle_deg    : crank position 0–360° (0 = 3-o'clock / horizontal right)
  - cadence_rpm  : smoothed cadence
  - marker_px    : detected (x, y) in pixel space, or None
  - tracking     : 'detected' | 'tracked' | 'lost'
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from cv.filters import AngleUnwrapper, RollingStats


# ── data type ────────────────────────────────────────────────────────────────

@dataclass
class CrankState:
    angle_deg:   Optional[float]          # 0–360°, None if lost
    cadence_rpm: Optional[float]          # smoothed
    marker_px:   Optional[tuple[int,int]] # pixel location of sticker centroid
    tracking:    str                      # 'detected' | 'tracked' | 'lost'
    total_revs:  int                      # full pedal revolutions this session


# ── tracker ──────────────────────────────────────────────────────────────────

# LK optical flow parameters
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class CrankTracker:
    """
    Tracks the crank marker and computes cadence.

    Args:
        bb_x, bb_y: bottom bracket centre in pixels (from calibration)
    """

    def __init__(self, bb_x: int, bb_y: int):
        self._bb   = np.array([bb_x, bb_y], dtype=float)
        self._prev_gray: Optional[np.ndarray] = None
        self._track_pt:  Optional[np.ndarray] = None  # (1,1,2) float32 for LK

        self._last_ts:   Optional[float] = None
        self._unwrap     = AngleUnwrapper()
        self._cadence_rs = RollingStats(window=config.CADENCE_SMOOTH_WIN)

        # for cadence from angle derivative
        self._prev_unwrapped: Optional[float] = None
        self._prev_cadence_ts: Optional[float] = None

        # re-detect every N frames even when tracking succeeds
        # (prevents drift; set to 0 to always use detection)
        self._redetect_interval  = 15
        self._frames_since_detect = 0

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray, timestamp: float) -> CrankState:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        marker_px = None
        tracking  = "lost"

        # 1. Try optical flow tracking on existing point
        if self._track_pt is not None and self._prev_gray is not None:
            new_pt, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._track_pt, None, **_LK_PARAMS
            )
            if st[0][0] == 1:
                nx, ny = int(new_pt[0][0][0]), int(new_pt[0][0][1])
                if self._point_in_frame(nx, ny, frame_bgr.shape):
                    marker_px = (nx, ny)
                    self._track_pt = new_pt
                    tracking = "tracked"

        # 2. Force re-detection periodically or when tracking lost
        self._frames_since_detect += 1
        if (tracking == "lost"
                or self._frames_since_detect >= self._redetect_interval):
            detected = self._detect_marker(hsv)
            if detected:
                marker_px  = detected
                self._track_pt = np.array(
                    [[[detected[0], detected[1]]]], dtype=np.float32
                )
                tracking = "detected"
                self._frames_since_detect = 0
            elif tracking != "tracked":
                self._track_pt = None

        self._prev_gray = gray

        # 3. Compute angle and cadence
        angle_deg   = None
        cadence_rpm = None

        if marker_px:
            dx = marker_px[0] - self._bb[0]
            dy = marker_px[1] - self._bb[1]
            # atan2 returns (-180, 180]; map to (0, 360]
            raw_angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
            angle_deg = raw_angle
            unwrapped = self._unwrap.update(raw_angle)
            cadence_rpm = self._compute_cadence(unwrapped, timestamp)

        if self._last_ts is None:
            self._last_ts = timestamp
        self._last_ts = timestamp

        return CrankState(
            angle_deg   = angle_deg,
            cadence_rpm = cadence_rpm,
            marker_px   = marker_px,
            tracking    = tracking,
            total_revs  = self._unwrap.total_revolutions,
        )

    def draw_overlay(
        self, frame: np.ndarray, state: CrankState
    ) -> np.ndarray:
        """Draw BB centre, marker location, and cadence on frame."""
        bx, by = int(self._bb[0]), int(self._bb[1])
        cv2.drawMarker(frame, (bx, by), (255, 200, 0),
                       cv2.MARKER_CROSS, 20, 2)

        if state.marker_px:
            mx, my = state.marker_px
            colour = (0, 200, 255) if state.tracking == "detected" else (0, 140, 255)
            cv2.circle(frame, (mx, my), 10, colour, 2)
            cv2.line(frame, (bx, by), (mx, my), (100, 100, 255), 1)

        cadence_str = (f"{state.cadence_rpm:.0f} RPM"
                       if state.cadence_rpm else "-- RPM")
        cv2.putText(frame, cadence_str, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        return frame

    # ── internal ─────────────────────────────────────────────────────────────

    def _detect_marker(
        self, hsv: np.ndarray
    ) -> Optional[tuple[int, int]]:
        mask = self._build_mask(hsv)
        if mask is None:
            return None

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        valid = [
            c for c in contours
            if config.MARKER_MIN_AREA_PX
            < cv2.contourArea(c)
            < config.MARKER_MAX_AREA_PX
        ]
        if not valid:
            return None

        best = max(valid, key=cv2.contourArea)
        M    = cv2.moments(best)
        if M["m00"] < 1:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _compute_cadence(
        self, unwrapped: float, ts: float
    ) -> Optional[float]:
        if self._prev_unwrapped is None or self._prev_cadence_ts is None:
            self._prev_unwrapped  = unwrapped
            self._prev_cadence_ts = ts
            return None

        dt = ts - self._prev_cadence_ts
        if dt < 1e-4:
            return self._cadence_rs.median

        angular_velocity_dps = abs(unwrapped - self._prev_unwrapped) / dt  # °/s
        raw_rpm = angular_velocity_dps / 6.0   # 1 RPM = 6 °/s

        self._prev_unwrapped  = unwrapped
        self._prev_cadence_ts = ts

        if config.CADENCE_MIN_RPM < raw_rpm < config.CADENCE_MAX_RPM:
            self._cadence_rs.push(raw_rpm)

        return self._cadence_rs.median

    def _build_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """
        Build a binary detection mask using either HSV colour or brightness.

        MARKER_MODE in config.py controls which method is used:
          'hsv'        — colour sticker (orange, green, etc.)  [default]
          'bright'     — white/silver retroreflective tape or coin
          'dark'       — black marker on a light-coloured crank
        """
        mode   = getattr(config, 'MARKER_MODE', 'hsv')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        if mode == 'bright':
            # Detect the brightest small blob in the frame.
            # Works with white tape, retroreflective strips, or a coin.
            v_chan = hsv[:, :, 2]
            thresh = getattr(config, 'MARKER_BRIGHT_THRESH', 220)
            mask   = cv2.threshold(v_chan, thresh, 255, cv2.THRESH_BINARY)[1]

        elif mode == 'dark':
            v_chan = hsv[:, :, 2]
            thresh = getattr(config, 'MARKER_DARK_THRESH', 40)
            mask   = cv2.threshold(v_chan, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        else:  # 'hsv' — colour sticker
            lo   = np.array(config.MARKER_HSV_LOWER)
            hi   = np.array(config.MARKER_HSV_UPPER)
            mask = cv2.inRange(hsv, lo, hi)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _point_in_frame(x: int, y: int, shape: tuple) -> bool:
        h, w = shape[:2]
        return 0 <= x < w and 0 <= y < h
