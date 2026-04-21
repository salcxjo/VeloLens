"""
ble/cadence_server.py
─────────────────────
BLE GATT peripheral broadcasting cadence using the standard
Cycling Speed and Cadence (CSC) profile (UUID 0x1816).

Uses `bless` for the GATT server role on Linux/Pi via BlueZ D-Bus.

Install:
    pip install bless --break-system-packages

The Pi advertises as a CSC sensor — readable by nRF Connect, macOS
Bluetooth Cadence Monitor, or any app that subscribes to:
    Service:        0x1816  (Cycling Speed and Cadence)
    Characteristic: 0x2A5B  (CSC Measurement, Notify)

CSC Measurement packet (5 bytes):
    [0]   flags    = 0x02  (Crank Revolution Data Present)
    [1-2] cumulative crank revolutions  (uint16 LE)
    [3-4] last crank event time         (uint16 LE, 1/1024 s units)
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_UUID_CSC_SERVICE     = "00001816-0000-1000-8000-00805f9b34fb"
_UUID_CSC_MEASUREMENT = "00002a5b-0000-1000-8000-00805f9b34fb"
_UUID_CSC_FEATURE     = "00002a5c-0000-1000-8000-00805f9b34fb"
_CSC_FEATURE_FLAGS    = struct.pack("<H", 0x0002)   # Crank Revolution Data

# Device Information Service — required for MyWhoosh / Zwift to recognise
# the device as a cadence sensor rather than an unknown BLE peripheral
_UUID_DEVINFO_SERVICE = "0000180a-0000-1000-8000-00805f9b34fb"
_UUID_MANUFACTURER    = "00002a29-0000-1000-8000-00805f9b34fb"
_UUID_MODEL           = "00002a24-0000-1000-8000-00805f9b34fb"

# Appearance: 0x0482 = Cycling Cadence Sensor
# Tells the OS to show a bicycle icon and auto-categorise as a sport sensor
_APPEARANCE = struct.pack("<H", 0x0482)


class CadenceBLEServer:
    """
    Thread-safe BLE cadence broadcaster.
    Runs the GATT server in a background thread.
    Call push_cadence() freely from the CV loop — it's non-blocking.
    """

    def __init__(self, device_name: str = "VeloLens"):
        self._name          = device_name
        self._cadence_rpm   = 0.0
        self._cum_cranks    = 0
        self._last_event_t  = 0       # 1/1024 s units
        self._last_push_ts  = 0.0
        self._lock          = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running       = False
        self._server        = None    # bless BlessServer

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            import bless  # noqa: F401
        except ImportError:
            logger.warning(
                "[BLE] bless not installed — cadence broadcast disabled.\n"
                "      pip install bless --break-system-packages"
            )
            return False

        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="ble-cadence"
        )
        self._thread.start()
        logger.info(f"[BLE] Starting GATT server as '{self._name}'")
        return True

    def push_cadence(self, cadence_rpm: float):
        """Feed a new cadence value. Non-blocking, safe to call at 30 fps."""
        now = time.monotonic()
        with self._lock:
            self._cadence_rpm = cadence_rpm
            dt = now - self._last_push_ts if self._last_push_ts else 0.0
            if cadence_rpm > 0 and dt > 0:
                self._cum_cranks = (
                    self._cum_cranks + round(cadence_rpm / 60.0 * dt)
                ) & 0xFFFF
                self._last_event_t = int(now * 1024) & 0xFFFF
            self._last_push_ts = now

        # trigger a notify on the BLE characteristic
        if self._server and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._notify(), self._loop)

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── internal ──────────────────────────────────────────────────────────────

    def _csc_packet(self) -> bytearray:
        with self._lock:
            return bytearray(struct.pack(
                "<BHH", 0x02, self._cum_cranks, self._last_event_t
            ))

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"[BLE] {e}")
        finally:
            self._loop.close()

    async def _serve(self):
        from bless import (
            BlessServer,
            BlessGATTCharacteristic,
            GATTCharacteristicProperties,
            GATTAttributePermissions,
        )

        trigger = asyncio.Event()

        self._server = BlessServer(name=self._name, loop=self._loop)
        self._server.read_request_func  = self._handle_read
        self._server.write_request_func = self._handle_write

        # ── Cycling Speed & Cadence service ──────────────────────────
        await self._server.add_new_service(_UUID_CSC_SERVICE)

        await self._server.add_new_characteristic(
            _UUID_CSC_SERVICE,
            _UUID_CSC_MEASUREMENT,
            GATTCharacteristicProperties.notify,
            None,
            GATTAttributePermissions.readable,
        )
        await self._server.add_new_characteristic(
            _UUID_CSC_SERVICE,
            _UUID_CSC_FEATURE,
            GATTCharacteristicProperties.read,
            bytearray(_CSC_FEATURE_FLAGS),
            GATTAttributePermissions.readable,
        )

        # ── Device Information service (required by MyWhoosh / Zwift) ──
        await self._server.add_new_service(_UUID_DEVINFO_SERVICE)
        await self._server.add_new_characteristic(
            _UUID_DEVINFO_SERVICE,
            _UUID_MANUFACTURER,
            GATTCharacteristicProperties.read,
            bytearray(b"VeloLens"),
            GATTAttributePermissions.readable,
        )
        await self._server.add_new_characteristic(
            _UUID_DEVINFO_SERVICE,
            _UUID_MODEL,
            GATTCharacteristicProperties.read,
            bytearray(b"VeloLens Cadence v1"),
            GATTAttributePermissions.readable,
        )

        await self._server.start()
        logger.info(f"[BLE] Advertising as '{self._name}' — CSC + DevInfo services")
        logger.info(f"[BLE] Appearance: 0x0482 (Cycling Cadence Sensor)")
        logger.info(f"[BLE] MyWhoosh / Zwift: search for cadence sensors in device settings")

        while self._running:
            await asyncio.sleep(1.0)
            await self._notify()

        await self._server.stop()

    async def _notify(self):
        if not self._server:
            return
        try:
            char = self._server.get_characteristic(_UUID_CSC_MEASUREMENT)
            if char:
                char.value = self._csc_packet()
                self._server.update_value(_UUID_CSC_SERVICE, _UUID_CSC_MEASUREMENT)
        except Exception as e:
            logger.debug(f"[BLE] notify error: {e}")

    def _handle_read(self, characteristic, **kwargs) -> bytearray:
        if characteristic.uuid.lower() == _UUID_CSC_FEATURE.lower():
            return bytearray(_CSC_FEATURE_FLAGS)
        return self._csc_packet()

    def _handle_write(self, characteristic, value, **kwargs):
        pass
