"""
storage/db.py
─────────────
SQLite-backed session storage.

Schema:
  sessions      – one row per session, aggregated metrics + quality score
  frame_events  – downsampled per-frame metrics for trend/fatigue analysis

Designed to be query-friendly from a future Android app or Flask API.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from processing.metrics import SessionMetrics


DB_PATH = Path(config.SESSION_DIR) / "veloLens.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    duration_secs   REAL,

    -- cadence
    avg_cadence_rpm REAL,
    max_cadence_rpm REAL,
    cadence_cv      REAL,

    -- knee
    avg_knee_bdc    REAL,
    min_knee_angle  REAL,
    max_knee_angle  REAL,

    -- torso
    avg_torso_angle REAL,
    torso_drift_deg REAL,

    -- quality
    quality_score   REAL,
    quality_json    TEXT,   -- JSON breakdown dict

    -- counts
    total_frames    INTEGER,
    pose_frames     INTEGER,
    crank_frames    INTEGER,
    low_cad_events  INTEGER,
    high_cad_events INTEGER,
    knee_flag_events INTEGER,

    created_at      REAL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS frame_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    timestamp       REAL NOT NULL,
    cadence_rpm     REAL,
    knee_angle_deg  REAL,
    torso_angle_deg REAL,
    crank_angle_deg REAL
);

CREATE INDEX IF NOT EXISTS idx_fe_session ON frame_events(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA)
    return conn


def save_session(sm: SessionMetrics, conn: Optional[sqlite3.Connection] = None) -> None:
    """Persist a completed SessionMetrics to the database."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    duration = sm.end_time - sm.start_time
    conn.execute("""
        INSERT OR REPLACE INTO sessions (
            id, start_time, end_time, duration_secs,
            avg_cadence_rpm, max_cadence_rpm, cadence_cv,
            avg_knee_bdc, min_knee_angle, max_knee_angle,
            avg_torso_angle, torso_drift_deg,
            quality_score, quality_json,
            total_frames, pose_frames, crank_frames,
            low_cad_events, high_cad_events, knee_flag_events
        ) VALUES (
            :id, :start, :end, :dur,
            :avg_cad, :max_cad, :cad_cv,
            :knee_bdc, :knee_min, :knee_max,
            :torso, :drift,
            :quality, :quality_json,
            :tf, :pf, :cf,
            :lce, :hce, :kfe
        )
    """, dict(
        id          = sm.session_id,
        start       = sm.start_time,
        end         = sm.end_time,
        dur         = duration,
        avg_cad     = sm.avg_cadence_rpm,
        max_cad     = sm.max_cadence_rpm,
        cad_cv      = sm.cadence_cv,
        knee_bdc    = sm.avg_knee_angle_bdc,
        knee_min    = sm.min_knee_angle,
        knee_max    = sm.max_knee_angle,
        torso       = sm.avg_torso_angle,
        drift       = sm.torso_drift_deg,
        quality     = sm.quality_score,
        quality_json= json.dumps(sm.quality_breakdown),
        tf          = sm.total_frames,
        pf          = sm.pose_frames,
        cf          = sm.crank_frames,
        lce         = sm.low_cadence_events,
        hce         = sm.high_cadence_events,
        kfe         = sm.knee_flag_events,
    ))
    if own_conn:
        conn.commit()
        conn.close()


def save_frame_events(
    session_id: str,
    events: list[dict],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Bulk-insert downsampled frame events."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    conn.executemany("""
        INSERT INTO frame_events
            (session_id, timestamp, cadence_rpm,
             knee_angle_deg, torso_angle_deg, crank_angle_deg)
        VALUES
            (:session_id, :timestamp, :cadence_rpm,
             :knee_angle_deg, :torso_angle_deg, :crank_angle_deg)
    """, [{**e, "session_id": session_id} for e in events])
    if own_conn:
        conn.commit()
        conn.close()


def list_sessions(limit: int = 20) -> list[sqlite3.Row]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows
