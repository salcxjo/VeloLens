"""
processing/metrics.py
─────────────────────
Per-frame metric computation and end-of-session aggregation.

FrameMetrics   – snapshot of all computed values for one frame
SessionMetrics – aggregated stats for a complete session
MetricsEngine  – stateful accumulator, call update() each frame
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from cv.filters import RollingStats


# ── per-frame snapshot ───────────────────────────────────────────────────────

@dataclass
class FrameMetrics:
    timestamp:      float
    cadence_rpm:    Optional[float] = None
    knee_angle_deg: Optional[float] = None   # right knee at current crank position
    torso_angle_deg:Optional[float] = None   # torso from horizontal
    crank_angle_deg:Optional[float] = None   # 0–360°

    # flags (True = issue detected this frame)
    cadence_low:    bool = False
    cadence_high:   bool = False
    knee_over_ext:  bool = False   # below KNEE_EXTENSION_OPTIMAL_MIN
    knee_under_ext: bool = False   # above KNEE_EXTENSION_OPTIMAL_MAX


# ── session-level aggregates ─────────────────────────────────────────────────

@dataclass
class SessionMetrics:
    session_id:           str
    start_time:           float
    end_time:             float
    total_frames:         int
    pose_frames:          int          # frames where pose was detected
    crank_frames:         int          # frames where crank was tracked

    # cadence
    avg_cadence_rpm:      Optional[float] = None
    max_cadence_rpm:      Optional[float] = None
    cadence_cv:           Optional[float] = None   # coefficient of variation

    # knee
    avg_knee_angle_bdc:   Optional[float] = None   # near BDC (crank 160–200°)
    min_knee_angle:       Optional[float] = None
    max_knee_angle:       Optional[float] = None

    # torso / aero
    avg_torso_angle:      Optional[float] = None
    torso_drift_deg:      Optional[float] = None   # end avg – start avg

    # quality score (0–100)
    quality_score:        Optional[float] = None
    quality_breakdown:    dict            = field(default_factory=dict)

    # event counts
    low_cadence_events:   int = 0
    high_cadence_events:  int = 0
    knee_flag_events:     int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ── engine ───────────────────────────────────────────────────────────────────

class MetricsEngine:
    """
    Accumulates FrameMetrics over a session and computes SessionMetrics
    at the end.

    Usage:
        engine = MetricsEngine(session_id="2024-01-15_001")
        engine.start()
        for each frame:
            fm = engine.update(crank_state, pose_result)
        summary = engine.finish()
    """

    def __init__(self, session_id: str):
        self._id         = session_id
        self._start_time: Optional[float] = None

        # rolling windows for real-time display
        self._cadence_rs  = RollingStats(window=30)
        self._knee_rs     = RollingStats(window=30)
        self._torso_rs    = RollingStats(window=30)

        # long-running history for session aggregation
        self._cadence_hist:  list[float] = []
        self._knee_hist:     list[float] = []
        self._torso_hist:    list[float] = []
        self._knee_bdc:      list[float] = []   # knee angles when crank ≈ BDC

        # counters
        self._total_frames = 0
        self._pose_frames  = 0
        self._crank_frames = 0
        self._low_cad_ev   = 0
        self._high_cad_ev  = 0
        self._knee_flag_ev = 0

    # ── public ───────────────────────────────────────────────────────────────

    def start(self):
        self._start_time = time.monotonic()

    def update(self, crank_state, pose_result) -> FrameMetrics:
        """
        Call once per processed frame.
        crank_state : CrankState (may have None fields if tracking lost)
        pose_result : PoseResult (may be None)
        Returns FrameMetrics for this frame.
        """
        now = time.monotonic()
        self._total_frames += 1

        fm = FrameMetrics(timestamp=now)

        # ── cadence ──────────────────────────────────────────────────────
        if crank_state and crank_state.cadence_rpm is not None:
            self._crank_frames += 1
            rpm = crank_state.cadence_rpm
            fm.cadence_rpm = rpm
            self._cadence_rs.push(rpm)
            self._cadence_hist.append(rpm)
            fm.crank_angle_deg = crank_state.angle_deg
            if rpm < config.CADENCE_MIN_RPM:
                fm.cadence_low  = True
                self._low_cad_ev += 1
            elif rpm > config.CADENCE_MAX_RPM:
                fm.cadence_high  = True
                self._high_cad_ev += 1

        # ── pose-derived metrics ─────────────────────────────────────────
        if pose_result is not None:
            self._pose_frames += 1

            # knee extension
            knee = pose_result.joint_angle(
                config.MP_RIGHT_HIP,
                config.MP_RIGHT_KNEE,
                config.MP_RIGHT_ANKLE,
            )
            if knee is not None:
                fm.knee_angle_deg = knee
                self._knee_rs.push(knee)
                self._knee_hist.append(knee)

                # collect BDC samples (crank 160–200° = pedal at bottom)
                ca = crank_state.angle_deg if crank_state else None
                if ca is not None and 160 <= ca <= 200:
                    self._knee_bdc.append(knee)

                if knee < config.KNEE_EXTENSION_OPTIMAL_MIN:
                    fm.knee_over_ext = True
                    self._knee_flag_ev += 1
                elif knee > config.KNEE_EXTENSION_OPTIMAL_MAX:
                    fm.knee_under_ext = True
                    self._knee_flag_ev += 1

            # torso angle
            torso = pose_result.vector_angle_from_horizontal(
                config.MP_RIGHT_HIP,
                config.MP_RIGHT_SHOULDER,
            )
            if torso is not None:
                fm.torso_angle_deg = torso
                self._torso_rs.push(torso)
                self._torso_hist.append(torso)

        return fm

    def finish(self) -> SessionMetrics:
        """Call at end of session to compute aggregate metrics."""
        end_time = time.monotonic()

        sm = SessionMetrics(
            session_id   = self._id,
            start_time   = self._start_time or end_time,
            end_time     = end_time,
            total_frames = self._total_frames,
            pose_frames  = self._pose_frames,
            crank_frames = self._crank_frames,
            low_cadence_events  = self._low_cad_ev,
            high_cadence_events = self._high_cad_ev,
            knee_flag_events    = self._knee_flag_ev,
        )

        if self._cadence_hist:
            arr = np.array(self._cadence_hist)
            sm.avg_cadence_rpm = float(np.mean(arr))
            sm.max_cadence_rpm = float(np.max(arr))
            # CV needs ≥ 5 samples to be meaningful
            if len(arr) >= 5:
                sm.cadence_cv = float(np.std(arr) / (np.mean(arr) + 1e-9))

        if self._knee_bdc:
            sm.avg_knee_angle_bdc = float(np.mean(self._knee_bdc))

        if self._knee_hist:
            arr = np.array(self._knee_hist)
            sm.min_knee_angle = float(np.min(arr))
            sm.max_knee_angle = float(np.max(arr))

        if self._torso_hist:
            arr = np.array(self._torso_hist)
            sm.avg_torso_angle = float(np.mean(arr))
            # fatigue proxy: compare first 20% vs last 20% of torso samples
            n  = len(arr)
            if n >= 20:
                q = n // 5
                early = np.mean(arr[:q])
                late  = np.mean(arr[-q:])
                sm.torso_drift_deg = float(late - early)

        sm.quality_score, sm.quality_breakdown = self._compute_quality(sm)
        return sm

    # ── quality score ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_quality(sm: SessionMetrics) -> tuple[float, dict]:
        """
        Weighted composite score 0–100.

        Component            Weight  What good looks like
        ─────────────────    ──────  ────────────────────────────────────────
        cadence_smoothness   35%     CV < 0.05 = perfect
        cadence_target       20%     80–95 RPM band
        knee_extension       25%     avg BDC angle 140–150°
        posture_stability    20%     |torso_drift| < 2°
        """
        scores    = {}
        weights   = {}

        # 1. Cadence smoothness
        if sm.cadence_cv is not None:
            # CV 0 → 100 pts; CV 0.15 → 0 pts (linear)
            s = max(0.0, 1.0 - sm.cadence_cv / 0.15) * 100
            scores["cadence_smoothness"]  = round(s, 1)
            weights["cadence_smoothness"] = 0.35

        # 2. Cadence target (80–95 RPM band)
        if sm.avg_cadence_rpm is not None:
            rpm = sm.avg_cadence_rpm
            if 80 <= rpm <= 95:
                s = 100.0
            elif rpm < 80:
                s = max(0.0, 100 - (80 - rpm) * 4)   # -4 pts per RPM below 80
            else:
                s = max(0.0, 100 - (rpm - 95) * 4)
            scores["cadence_target"]  = round(s, 1)
            weights["cadence_target"] = 0.20

        # 3. Knee extension at BDC
        if sm.avg_knee_angle_bdc is not None:
            angle = sm.avg_knee_angle_bdc
            opt_min = config.KNEE_EXTENSION_OPTIMAL_MIN   # 140
            opt_max = config.KNEE_EXTENSION_OPTIMAL_MAX   # 150
            if opt_min <= angle <= opt_max:
                s = 100.0
            elif angle < opt_min:
                s = max(0.0, 100 - (opt_min - angle) * 5)
            else:
                s = max(0.0, 100 - (angle - opt_max) * 5)
            scores["knee_extension"]  = round(s, 1)
            weights["knee_extension"] = 0.25

        # 4. Posture stability
        if sm.torso_drift_deg is not None:
            drift = abs(sm.torso_drift_deg)
            s = max(0.0, 100 - drift * 10)  # -10 pts per degree drift
            scores["posture_stability"]  = round(s, 1)
            weights["posture_stability"] = 0.20

        if not scores:
            return (None, {})

        # normalise weights to available components
        total_w = sum(weights[k] for k in scores)
        composite = sum(
            scores[k] * weights[k] / total_w for k in scores
        )
        return round(composite, 1), scores
