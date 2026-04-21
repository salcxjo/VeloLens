"""
strava/client.py
────────────────
Strava API integration for VeloLens.

Two responsibilities:
  1. OAuth2 flow — opens browser for first-time auth, then refreshes
     tokens automatically. Credentials stored in sessions/strava_token.json.

  2. Activity matching — after a session, find the Strava activity that
     overlaps the session window and attach VeloLens metrics to its
     description (or a private note if the activity is already posted).

Requires:
  pip install requests --break-system-packages

Setup (one-time):
  1. Create a Strava API app at https://www.strava.com/settings/api
     Set "Authorization Callback Domain" to localhost
  2. Copy Client ID and Client Secret into config.py:
       STRAVA_CLIENT_ID     = "12345"
       STRAVA_CLIENT_SECRET = "abc..."
  3. Run:  python -m veloLens.strava.client --auth

Usage:
  from strava.client import StravaClient
  client = StravaClient()
  activity = client.match_activity(session_start_unix, session_end_unix)
  client.annotate_activity(activity["id"], session_metrics)
"""

from __future__ import annotations

import json
import logging
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional
from urllib.parse import parse_qs, urlparse, urlencode

import requests

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────
_AUTH_URL    = "https://www.strava.com/oauth/authorize"
_TOKEN_URL   = "https://www.strava.com/oauth/token"
_API_BASE    = "https://www.strava.com/api/v3"
_SCOPE       = "activity:read_all,activity:write"
_REDIRECT    = "http://localhost:8765/callback"
_TOKEN_PATH  = Path(config.SESSION_DIR) / "strava_token.json"
_REDIRECT_PORT = 8765


class StravaClient:
    """
    Strava API client with automatic token refresh.

    Raises StravaNotConfiguredError if STRAVA_CLIENT_ID / SECRET
    are not set in config.py.
    """

    def __init__(self):
        self._client_id     = getattr(config, "STRAVA_CLIENT_ID",     None)
        self._client_secret = getattr(config, "STRAVA_CLIENT_SECRET", None)
        if not self._client_id or not self._client_secret:
            raise StravaNotConfiguredError(
                "Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in config.py.\n"
                "Create an app at https://www.strava.com/settings/api"
            )
        self._token: Optional[dict] = self._load_token()

    # ── auth ──────────────────────────────────────────────────────────────

    def authorize(self):
        """
        Run the OAuth2 PKCE browser flow.
        Opens the browser, starts a local server on port 8765 to capture
        the callback, then exchanges the code for tokens.
        """
        params = {
            "client_id":     self._client_id,
            "redirect_uri":  _REDIRECT,
            "response_type": "code",
            "approval_prompt": "auto",
            "scope":         _SCOPE,
        }
        url = f"{_AUTH_URL}?{urlencode(params)}"
        print(f"\n[Strava] Opening browser for authorisation…\n  {url}\n")
        webbrowser.open(url)

        code = _capture_oauth_code()
        if not code:
            raise RuntimeError("[Strava] No code received — authorisation failed.")

        resp = requests.post(_TOKEN_URL, data={
            "client_id":     self._client_id,
            "client_secret": self._client_secret,
            "code":          code,
            "grant_type":    "authorization_code",
        })
        resp.raise_for_status()
        self._token = resp.json()
        self._save_token()
        print(f"[Strava] Authorised as {self._token['athlete']['firstname']} "
              f"{self._token['athlete']['lastname']}")

    @property
    def is_authorised(self) -> bool:
        return self._token is not None

    # ── activity matching ─────────────────────────────────────────────────

    def match_activity(
        self,
        session_start_unix: float,
        session_end_unix: float,
        tolerance_secs: float = 300,
    ) -> Optional[dict]:
        """
        Find the Strava activity whose start time is within tolerance_secs
        of the VeloLens session window.

        Returns the activity dict or None if not found.
        """
        # Fetch activities in a 2-hour window around the session
        after  = int(session_start_unix) - 3600
        before = int(session_end_unix)   + 3600
        activities = self._get(
            "/athlete/activities",
            params={"after": after, "before": before, "per_page": 10},
        )
        if not activities:
            logger.info("[Strava] No activities found in session window.")
            return None

        best      = None
        best_diff = float("inf")
        for act in activities:
            # parse Strava's ISO 8601 start_date
            import datetime
            start_dt = datetime.datetime.fromisoformat(
                act["start_date"].replace("Z", "+00:00")
            )
            act_start_unix = start_dt.timestamp()
            diff = abs(act_start_unix - session_start_unix)
            if diff < best_diff and diff < tolerance_secs:
                best_diff = diff
                best = act

        if best:
            logger.info(
                f"[Strava] Matched activity '{best['name']}' "
                f"(id={best['id']}, Δt={best_diff:.0f}s)"
            )
        return best

    # ── annotation ────────────────────────────────────────────────────────

    def annotate_activity(
        self, activity_id: int, session_metrics, append: bool = True
    ) -> dict:
        """
        Append a VeloLens metrics summary to the Strava activity description.

        session_metrics : SessionMetrics instance
        append          : if True, append to existing description;
                          if False, replace it.
        Returns the updated activity dict.
        """
        note = _format_strava_note(session_metrics)

        # Fetch current description
        activity = self._get(f"/activities/{activity_id}")
        current_desc = activity.get("description") or ""

        if append and current_desc:
            new_desc = current_desc.rstrip() + "\n\n" + note
        else:
            new_desc = note

        updated = self._put(
            f"/activities/{activity_id}",
            json={"description": new_desc},
        )
        logger.info(f"[Strava] Annotated activity {activity_id}")
        return updated

    def get_recent_activities(self, n: int = 10) -> list[dict]:
        """Fetch the athlete's n most recent activities."""
        return self._get("/athlete/activities", params={"per_page": n}) or []

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> any:
        self._ensure_token_fresh()
        resp = requests.get(
            _API_BASE + path,
            headers=self._auth_header(),
            params=params or {},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, json: dict = None) -> dict:
        self._ensure_token_fresh()
        resp = requests.put(
            _API_BASE + path,
            headers=self._auth_header(),
            json=json or {},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self._token['access_token']}"}

    # ── token management ──────────────────────────────────────────────────

    def _ensure_token_fresh(self):
        if not self._token:
            raise StravaNotAuthorisedError(
                "Not authorised. Run: python -m veloLens.strava.client --auth"
            )
        # Refresh if token expires within 5 minutes
        if self._token.get("expires_at", 0) - time.time() < 300:
            logger.info("[Strava] Refreshing access token…")
            resp = requests.post(_TOKEN_URL, data={
                "client_id":     self._client_id,
                "client_secret": self._client_secret,
                "grant_type":    "refresh_token",
                "refresh_token": self._token["refresh_token"],
            })
            resp.raise_for_status()
            refreshed = resp.json()
            # preserve athlete info which isn't returned on refresh
            refreshed.setdefault("athlete", self._token.get("athlete", {}))
            self._token = refreshed
            self._save_token()

    def _load_token(self) -> Optional[dict]:
        if _TOKEN_PATH.exists():
            try:
                return json.loads(_TOKEN_PATH.read_text())
            except Exception:
                return None
        return None

    def _save_token(self):
        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_PATH.write_text(json.dumps(self._token, indent=2))


# ── formatting ────────────────────────────────────────────────────────────

def _fmt(val, fmt=".1f", suffix="", fallback="–"):
    if val is None:
        return fallback
    return f"{val:{fmt}}{suffix}"


def _format_strava_note(sm) -> str:
    """
    Produce a compact plain-text block suitable for a Strava description.
    Strava descriptions don't support Markdown, so we use unicode symbols.
    """
    qs = sm.quality_score
    quality_str = f"{qs:.0f}/100" if qs is not None else "–"

    lines = [
        "📊 VeloLens Analysis",
        f"  Quality score   {quality_str}",
        f"  Avg cadence     {_fmt(sm.avg_cadence_rpm, suffix=' RPM')}",
        f"  Cadence CV      {_fmt(sm.cadence_cv, fmt='.3f')}  (lower = smoother)",
        f"  Knee @ BDC      {_fmt(sm.avg_knee_angle_bdc, suffix='°')}  (target 140–150°)",
        f"  Torso drift     {_fmt(sm.torso_drift_deg, fmt='+.1f', suffix='°')}",
    ]
    if sm.quality_breakdown:
        lines.append("  Breakdown:")
        for k, v in sm.quality_breakdown.items():
            lines.append(f"    {k.replace('_',' ').title():<22} {v:.0f}")
    lines.append(f"  Session ID      {sm.session_id}")
    return "\n".join(lines)


# ── local OAuth callback server ───────────────────────────────────────────

_oauth_code: Optional[str] = None


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _oauth_code
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if "code" in params:
            _oauth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h2>VeloLens authorised! You can close this tab.</h2>"
            )
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, *args):
        pass   # suppress default access log


def _capture_oauth_code(timeout: int = 60) -> Optional[str]:
    global _oauth_code
    _oauth_code = None
    server = HTTPServer(("localhost", _REDIRECT_PORT), _CallbackHandler)
    server.timeout = timeout
    server.handle_request()   # blocks until one request arrives or timeout
    return _oauth_code


# ── exceptions ────────────────────────────────────────────────────────────

class StravaNotConfiguredError(RuntimeError): pass
class StravaNotAuthorisedError(RuntimeError): pass


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Strava integration tools")
    parser.add_argument("--auth",      action="store_true",
                        help="Run OAuth2 authorisation flow")
    parser.add_argument("--recent",    action="store_true",
                        help="Print 5 most recent activities")
    args = parser.parse_args()

    try:
        client = StravaClient()
    except StravaNotConfiguredError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.auth:
        client.authorize()
    elif args.recent:
        acts = client.get_recent_activities(5)
        for a in acts:
            print(f"  {a['id']:>12}  {a['start_date'][:10]}  "
                  f"{a['name'][:40]}  ({a['type']})")
    else:
        parser.print_help()
