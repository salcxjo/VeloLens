"""
tests/test_metrics.py
─────────────────────
Unit tests for filters, angle math, and the metrics/quality score engine.
Run with: python -m pytest tests/ -v
"""

import math
import sys, os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cv.filters     import KalmanFilter1D, RollingStats, AngleUnwrapper
from processing.metrics import MetricsEngine, SessionMetrics


# ── KalmanFilter1D ──────────────────────────────────────────────────────────

def test_kalman_smooths_noise():
    kf = KalmanFilter1D(process_noise=0.01, measurement_noise=0.5)
    noisy = [100 + (i % 2) * 10 for i in range(20)]  # alternates 100, 110
    outputs = [kf.update(v) for v in noisy]
    # after settling, output should cluster between 100 and 110
    assert all(99 <= o <= 111 for o in outputs[5:]), outputs

def test_kalman_tracks_ramp():
    kf = KalmanFilter1D()
    outputs = [kf.update(float(i)) for i in range(50)]
    # should track the ramp (allow 2-step lag)
    assert abs(outputs[-1] - 49) < 3


# ── RollingStats ─────────────────────────────────────────────────────────────

def test_rolling_mean():
    rs = RollingStats(window=4)
    for v in [10, 20, 30, 40]:
        rs.push(v)
    assert abs(rs.mean - 25.0) < 1e-9

def test_rolling_evicts_oldest():
    rs = RollingStats(window=3)
    for v in [10, 20, 30, 999]:   # 999 evicts 10
        rs.push(v)
    # mean of 20, 30, 999
    assert abs(rs.mean - (20 + 30 + 999) / 3) < 1e-9

def test_rolling_cv():
    rs = RollingStats(window=10)
    for _ in range(10):
        rs.push(100.0)     # constant signal
    assert rs.cv is not None
    assert rs.cv < 0.001   # essentially zero variance


# ── AngleUnwrapper ───────────────────────────────────────────────────────────

def test_unwrap_monotonic():
    uw = AngleUnwrapper()
    angles = [350, 355, 0, 5, 10]   # crosses 360→0
    results = [uw.update(a) for a in angles]
    # should be monotonically increasing (after the first)
    for i in range(1, len(results)):
        assert results[i] > results[i - 1], f"Not monotonic at {i}: {results}"

def test_unwrap_counts_revs():
    uw = AngleUnwrapper()
    # simulate 3 full clockwise revolutions (0→360, three times)
    n = 360 * 3 + 1   # +1 to fully complete the 3rd revolution
    for deg in range(0, n, 5):
        uw.update(deg % 360)
    assert uw.total_revolutions == 3


# ── Quality score ────────────────────────────────────────────────────────────

def _make_session(**kwargs) -> SessionMetrics:
    defaults = dict(
        session_id="test",
        start_time=0.0,
        end_time=60.0,
        total_frames=1000,
        pose_frames=900,
        crank_frames=950,
        avg_cadence_rpm=88.0,
        cadence_cv=0.03,
        avg_knee_angle_bdc=145.0,
        torso_drift_deg=1.5,
    )
    defaults.update(kwargs)
    return SessionMetrics(**defaults)

def test_quality_perfect():
    sm = _make_session(
        avg_cadence_rpm=90, cadence_cv=0.01,
        avg_knee_angle_bdc=145, torso_drift_deg=0.5
    )
    score, breakdown = MetricsEngine._compute_quality(sm)
    assert score >= 90, f"Expected >=90 but got {score}"
    assert "cadence_smoothness" in breakdown

def test_quality_poor_cadence():
    sm = _make_session(avg_cadence_rpm=65, cadence_cv=0.12)
    score, _ = MetricsEngine._compute_quality(sm)
    assert score < 70, f"Expected < 70 but got {score}"

def test_quality_missing_pose():
    # If we have no pose data, score should still compute from cadence
    sm = SessionMetrics(
        session_id="test", start_time=0, end_time=60,
        total_frames=500, pose_frames=0, crank_frames=450,
        avg_cadence_rpm=90, cadence_cv=0.03,
    )
    score, breakdown = MetricsEngine._compute_quality(sm)
    # only cadence components available
    assert score is not None
    assert "knee_extension" not in breakdown

def test_quality_no_data():
    sm = SessionMetrics(
        session_id="test", start_time=0, end_time=10,
        total_frames=10, pose_frames=0, crank_frames=0,
    )
    score, breakdown = MetricsEngine._compute_quality(sm)
    assert score is None


# ── joint angle math (inline, no mediapipe dependency) ───────────────────────

def _angle(a, b, c):
    import numpy as np
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(max(-1, min(1, cos_t)))))

def test_straight_leg_is_180():
    # hip at top, knee in middle, ankle at bottom → 180°
    angle = _angle((100, 0), (100, 100), (100, 200))
    assert abs(angle - 180) < 0.1

def test_right_angle_is_90():
    angle = _angle((0, 0), (0, 100), (100, 100))
    assert abs(angle - 90) < 0.1

def test_typical_knee_extension():
    # approximate BDC position
    angle = _angle((100, 50), (130, 200), (140, 350))
    assert 110 < angle < 175


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
