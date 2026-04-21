"""
cv/filters.py
─────────────
Lightweight signal processing utilities used across the CV pipeline.

- KalmanFilter1D  : smooth a noisy scalar signal (landmark coordinates,
                    joint angles) without introducing lag
- RollingStats    : online mean / std / median over a sliding window
- AngleUnwrapper  : handle 0°/360° wraparound in crank angle tracking
"""

from collections import deque
from typing import Optional

import numpy as np


# ── 1-D Kalman filter ───────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Constant-velocity 1-D Kalman filter.

    State vector: [position, velocity]
    Suitable for slowly-changing signals like joint angles or
    landmark positions between consecutive video frames.

    Args:
        process_noise:     Q — trust in the motion model (lower = smoother)
        measurement_noise: R — trust in measurements (higher = smoother)
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self._q = process_noise
        self._r = measurement_noise

        # state: [x, dx]
        self._x  = np.zeros(2)
        self._P  = np.eye(2) * 1.0      # initial uncertainty
        self._initialized = False

        # matrices (dt filled in at first update)
        self._H = np.array([[1.0, 0.0]])
        self._I = np.eye(2)

    def update(self, measurement: float, dt: float = 1.0) -> float:
        """
        Feed one new measurement; returns the filtered estimate.
        dt: seconds since last measurement (use 1.0 if unknown / constant rate)
        """
        F = np.array([[1.0, dt], [0.0, 1.0]])   # state transition
        Q = self._q * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2   ],
        ])

        if not self._initialized:
            self._x[0] = measurement
            self._initialized = True
            return measurement

        # Predict
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # Update
        S  = self._H @ P_pred @ self._H.T + self._r
        K  = P_pred @ self._H.T / S[0, 0]
        self._x = x_pred + K.flatten() * (measurement - self._H @ x_pred)
        self._P = (self._I - np.outer(K, self._H)) @ P_pred

        return float(self._x[0])

    def reset(self):
        self._x = np.zeros(2)
        self._P = np.eye(2)
        self._initialized = False


# ── rolling statistics ───────────────────────────────────────────────────────

class RollingStats:
    """
    Incremental mean, std, and median over a fixed-size sliding window.
    O(1) per update for mean/std; O(n log n) for median (acceptable for n≤60).
    """

    def __init__(self, window: int = 30):
        self._buf   = deque(maxlen=window)
        self._sum   = 0.0
        self._sum2  = 0.0

    def push(self, value: float):
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            self._sum  -= old
            self._sum2 -= old * old
        self._buf.append(value)
        self._sum  += value
        self._sum2 += value * value

    @property
    def mean(self) -> Optional[float]:
        n = len(self._buf)
        return self._sum / n if n else None

    @property
    def std(self) -> Optional[float]:
        n = len(self._buf)
        if n < 2:
            return None
        variance = (self._sum2 - self._sum**2 / n) / (n - 1)
        return float(np.sqrt(max(0.0, variance)))

    @property
    def cv(self) -> Optional[float]:
        """Coefficient of variation = std / mean.  Lower → smoother."""
        m = self.mean
        s = self.std
        if m is None or s is None or abs(m) < 1e-6:
            return None
        return s / abs(m)

    @property
    def median(self) -> Optional[float]:
        if not self._buf:
            return None
        return float(np.median(list(self._buf)))

    def __len__(self):
        return len(self._buf)


# ── angle unwrapper ─────────────────────────────────────────────────────────

class AngleUnwrapper:
    """
    Convert a raw crank angle (0–360°, wraps) into a monotonically
    increasing angle suitable for differentiation.

    Also detects full-revolution events so the caller can count RPM.
    """

    def __init__(self):
        self._last:      Optional[float] = None
        self._unwrapped: float = 0.0
        self._revs:      int   = 0
        self._last_rev_unwrapped: float = 0.0

    def update(self, raw_angle_deg: float) -> float:
        """
        Feed raw angle; returns unwrapped (cumulative) angle in degrees.
        """
        if self._last is None:
            self._last      = raw_angle_deg
            self._unwrapped = raw_angle_deg
            return self._unwrapped

        delta = raw_angle_deg - self._last
        # Correct for wrap-around
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        self._unwrapped += delta
        self._last = raw_angle_deg

        # Count completed revolutions
        while self._unwrapped - self._last_rev_unwrapped >= 360:
            self._revs += 1
            self._last_rev_unwrapped += 360

        return self._unwrapped

    @property
    def total_revolutions(self) -> int:
        return self._revs

    def reset(self):
        self._last = None
        self._unwrapped = 0.0
        self._revs = 0
        self._last_rev_unwrapped = 0.0
