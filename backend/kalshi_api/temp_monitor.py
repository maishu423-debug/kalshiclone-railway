"""
KMIA current-temperature / recorded-high monitor (NOAA MADIS HFMETAR only).

Current temperature = latest valid 5-minute HFMETAR observation.
Recorded high       = highest valid HFMETAR temperature since Miami-local midnight.
There is no other source and no fallback provider.
"""
import threading
import time
from datetime import datetime, timezone

from weather import config as wcfg
from weather.asos_client import fetch_recent_observations
from weather.asos_nowcast import STATUS_UNAVAILABLE, freshness_status, todays_high
from weather.asos_parser import parse_utc

POLL_INTERVAL = 60

_lock = threading.Lock()
_started = False
_latest = None


def fetch_current_high():
    """Returns {"t0_f", "today_high_f", "observed_at", "source", "status"} or None if no HFMETAR obs exists."""
    now = datetime.now(timezone.utc)
    try:
        observations, info = fetch_recent_observations(now=now, reuse_seconds=30)
    except Exception as exc:
        print(f"[temp-monitor] HFMETAR poll failed ({exc})")
        return None
    if not observations:
        return None
    latest = observations[-1]
    age = max(0.0, (now - parse_utc(latest["observed_at"])).total_seconds())
    return {
        "t0_f": latest["temp_f"],
        "today_high_f": todays_high(observations, now)["high_f"],
        "observed_at": latest["observed_at"],
        "source": wcfg.SOURCE_ID,
        "status": freshness_status(age),
    }


def apply_current_high(observation: dict):
    if not observation:
        return
    with _lock:
        global _latest
        _latest = dict(observation)

    from .price_tracker import set_daily_high_nws, update_temp

    # The high is a record of what happened, so it advances even when the feed is delayed;
    # the current temperature is only pushed from a usable (not unavailable) observation.
    if observation.get("status") != STATUS_UNAVAILABLE and observation.get("t0_f") is not None:
        update_temp(float(observation["t0_f"]))
    if observation.get("today_high_f") is not None:
        set_daily_high_nws(float(observation["today_high_f"]))


def refresh_temp_now():
    observation = fetch_current_high()
    if observation is None:
        return None
    apply_current_high(observation)
    from .price_tracker import get_temp_snapshot

    snap = get_temp_snapshot()
    return {
        "t0_f": snap["current_f"],
        "today_high_f": snap["daily_high_f"],
        "observed_at": observation.get("observed_at"),
        "source": observation.get("source"),
    }


def get_latest_observation():
    with _lock:
        return dict(_latest) if _latest else None


def _poll_loop():
    time.sleep(10)
    while True:
        try:
            refresh_temp_now()
        except Exception as exc:
            print(f"[temp-monitor] poll error: {exc}")
        time.sleep(POLL_INTERVAL)


def start_temp_monitor():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    thread = threading.Thread(target=_poll_loop, daemon=True, name="temp-monitor-bg")
    thread.start()
