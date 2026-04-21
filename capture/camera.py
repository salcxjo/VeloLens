"""
capture/camera.py
─────────────────
Threaded camera capture for Raspberry Pi 4 with Pi Camera v2.

Backend priority:
  1. picamera2  — official Pi Camera library for libcamera stack (Bookworm)
  2. v4l2 index — fallback for USB webcams or older setups

Requires system-site-packages venv on Pi OS Bookworm:
  python3 -m venv .venv --system-site-packages
  source .venv/bin/activate
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float      # monotonic seconds


class Camera:
    """
    Non-blocking camera wrapper.
    Tries picamera2 first, falls back to OpenCV VideoCapture.
    Runs a dedicated capture thread; callers use read() without blocking.
    """

    def __init__(self):
        self._picam  = None     # picamera2.Picamera2 instance if used
        self._cap    = None     # cv2.VideoCapture instance if used
        self._backend_label = "unknown"
        self._latest = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._timestamps = deque(maxlen=30)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._open_backend()
        self._thread.start()
        deadline = time.monotonic() + 8.0
        while self._latest is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Camera opened but no frames received within 8 s.\n"
                    "Try: sudo pkill -f libcamera && sudo pkill -f rpicam"
                )
            time.sleep(0.05)
        print(f"[camera] Backend: {self._backend_label}  "
              f"resolution: {self.resolution[0]}x{self.resolution[1]}")
        return self

    def read(self) -> Frame:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("Camera.start() must be called first.")
            return self._latest

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
        if self._cap:
            self._cap.release()

    @property
    def fps(self) -> float:
        ts = list(self._timestamps)
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    @property
    def resolution(self) -> tuple:
        if self._picam:
            cfg = self._picam.camera_configuration()
            if cfg and "main" in cfg:
                return cfg["main"]["size"]
        if self._cap:
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (0, 0)

    # ── backend selection ─────────────────────────────────────────────────────

    def _open_backend(self):
        # 1. Try picamera2 (Pi Camera v2 on libcamera stack)
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            cfg = picam.create_video_configuration(
                main={
                    "size":   (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                    "format": "BGR888",   # OpenCV-native, no conversion needed
                },
                controls={"FrameRate": float(config.TARGET_FPS)},
            )
            picam.configure(cfg)
            picam.start()
            # warm-up: wait for a real frame
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                frame = picam.capture_array("main")
                if frame is not None and frame.size > 0:
                    self._picam = picam
                    self._backend_label = "picamera2"
                    return
            picam.stop()
            picam.close()
            print("[camera] picamera2 opened but produced no frames — trying fallback")
        except Exception as e:
            print(f"[camera] picamera2 unavailable: {e}")

        # 2. OpenCV VideoCapture fallback (USB webcam, etc.)
        print(f"[camera] Trying OpenCV VideoCapture index {config.CAMERA_INDEX} ...")
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        deadline = time.monotonic() + 5.0
        got = 0
        while time.monotonic() < deadline and got < 3:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                got += 1
        if got >= 3:
            self._cap = cap
            self._backend_label = f"v4l2 index {config.CAMERA_INDEX}"
            return
        cap.release()

        raise RuntimeError(
            "No working camera backend found.\n\n"
            "For Pi Camera v2 on Bookworm, ensure your venv inherits system packages:\n"
            "  python3 -m venv .venv --system-site-packages\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt --break-system-packages\n\n"
            "Then verify picamera2 works:\n"
            "  python3 -c \"from picamera2 import Picamera2; print('OK')\"\n\n"
            "For a USB webcam, update CAMERA_INDEX in config.py."
        )

    # ── capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self):
        while not self._stop.is_set():
            try:
                if self._picam:
                    img = self._picam.capture_array("main")
                    # picamera2 BGR888 → already BGR, but may have 4th channel
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    # picamera2 BGR888 format is actually RGB on this stack —
                    # convert to true BGR so OpenCV sees correct colours
                    img = img[:, :, ::-1]
                    # optional flip (e.g. upside-down mount)
                    if config.CAMERA_FLIP is not None:
                        img = cv2.flip(img, config.CAMERA_FLIP)
                else:
                    ok, img = self._cap.read()
                    if not ok or img is None:
                        time.sleep(0.005)
                        continue

                if img is None or img.size == 0:
                    time.sleep(0.005)
                    continue

                ts = time.monotonic()
                self._timestamps.append(ts)
                with self._lock:
                    self._latest = Frame(image=img, timestamp=ts)

            except Exception as e:
                print(f"[camera] Capture error: {e}")
                time.sleep(0.1)
