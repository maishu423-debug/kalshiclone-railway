"""
Fast KMIA current-temperature / recorded-high monitor (NOAA MADIS OMO/HFMETAR only).

Polls the MADIS HFMETAR client every minute (polling cadence), but the selected
observation only changes when MADIS publishes a newer observation timestamp.  Current temperature is only pushed
to the tracker when the latest ASOS observation is fresh enough
(weather.config.ASOS_MAX_CURRENT_AGE_MINUTES); a stale feed is reported as
unavailable instead of being presented as current.
"""
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from weather import config as wcfg
from weather.asos_client import fetch_recent_observations, log_refresh
from weather.asos_nowcast import STATUS_UNAVAILABLE, freshness_status
from weather.asos_parser import parse_utc
from weather.nws_high import get_recorded_high

log = logging.getLogger("temp-monitor")



def seconds_until_next_slot(now=None):
    """Seconds until the next poll slot (second 0 of each POLL_EVERY_MINUTES-th minute;
    with a 5-minute step: HH:01, :06, :11, ...)."""
    now = now or datetime.now(timezone.utc)
    step = wcfg.POLL_EVERY_MINUTES
    minutes_into_cycle = (now.minute - wcfg.POLL_MINUTE_OFFSET) % step
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=step - minutes_into_cycle)
    if minutes_into_cycle == 0 and now.second == 0 and now.microsecond == 0:
        nxt = now
    return max(0.0, (nxt - now).total_seconds())

_lock = threading.Lock()
_started = False
_latest = None


def fetch_current_high(now=None, reuse_seconds=0):
    """
    Returns {"t0_f", "today_high_f", "observed_at", "age_seconds", "status", "source"}
    or None when no ASOS observation could be fetched at all.
    today_high_f = max full-precision (tenths of C) KMIA observation since Miami-local midnight,
    from the NWS observations feed, or None if unavailable.  HFMETAR (whole C) feeds t0_f only.
    """
    now = now or datetime.now(timezone.utc)
    try:
        observations, info = fetch_recent_observations(now=now, reuse_seconds=reuse_seconds)
        log_refresh(observations, info, now, debug=os.getenv("ASOS_DEBUG", "").lower() in {"1", "true", "yes"})
    except Exception as exc:                       # never let a poll problem erase the last good obs
        log.exception("MADIS poll crashed; keeping last good observation")
        observations, info = [], {"errors": [f"poll crashed: {exc}"], "source_status": "fetch_error_using_last_good"}
    if not observations:
        prev = get_latest_observation()
        if prev and prev.get("observed_at"):      # keep serving the last good one, with its real age
            age = max(0.0, (now - parse_utc(prev["observed_at"])).total_seconds())
            return {**prev, "age_seconds": round(age, 1), "status": freshness_status(age),
                    "checked_at": now.isoformat(), "source_status": "fetch_error_using_last_good",
                    "last_poll_success": False}
        log.warning("no KMIA observation available and none held: %s", info.get("errors"))
        return None
    latest = observations[-1]
    age = max(0.0, (now - parse_utc(latest["observed_at"])).total_seconds())
    high = get_recorded_high(now)       # full-precision NWS obs, never the whole-C HFMETAR max
    return {
        "t0_f": latest["temp_f"],
        "today_high_f": high["high_f"] if high else None,
        "observed_at": latest["observed_at"],
        "age_seconds": round(age, 1),
        "status": freshness_status(age),
        "source": wcfg.SOURCE_ID,
        "checked_at": now.isoformat(),
        "source_status": info.get("source_status", "ok"),
        "last_poll_success": info.get("last_poll_success", not info.get("errors")),
    }


def apply_current_high(observation: dict):
    if not observation:
        return
    global _latest
    with _lock:
        prev = _latest
        # Never move the selected observation backwards in time.
        if prev and prev.get("observed_at") and observation.get("observed_at")                 and parse_utc(observation["observed_at"]) < parse_utc(prev["observed_at"]):
            observation = {**prev, "checked_at": observation.get("checked_at"),
                           "age_seconds": prev.get("age_seconds")}
        _latest = dict(observation)

    from .price_tracker import set_daily_high_nws, update_current

    # Today's high is a record of what happened, so it is safe to advance even when
    # the feed is delayed.  Current temperature is only updated from a usable obs,
    # and never advances the recorded high (HFMETAR whole-C can overshoot it).
    if observation.get("status") != STATUS_UNAVAILABLE and observation.get("t0_f") is not None:
        update_current(float(observation["t0_f"]))
    if observation.get("today_high_f") is not None:
        set_daily_high_nws(float(observation["today_high_f"]))


def refresh_temp_now(reuse_seconds=0):
    observation = fetch_current_high(reuse_seconds=reuse_seconds)
    if observation is None:
        return None
    apply_current_high(observation)
    from .price_tracker import get_temp_snapshot

    snap = get_temp_snapshot()
    return {
        "t0_f": snap["current_f"],
        "today_high_f": snap["daily_high_f"],
        "observed_at": observation.get("observed_at"),
        "age_seconds": observation.get("age_seconds"),
        "status": observation.get("status"),
        "source": observation.get("source"),
        "checked_at": observation.get("checked_at"),
        "source_status": observation.get("source_status"),
    }


def get_latest_observation():
    with _lock:
        return dict(_latest) if _latest else None


def _poll_loop():
    """Poll NOAA every MADIS_POLL_MINUTES (default 1) so a newly published observation is picked up fast."""
    try:
        refresh_temp_now()                        # prime the display at startup
    except Exception as exc:
        log.warning("startup poll error: %s", exc)
    while True:
        time.sleep(seconds_until_next_slot() or 1.0)
        try:
            result = refresh_temp_now()
            # with sparse polling (>1 min) retry once if that poll failed or the observation is unusably old
            failed = (result is None or result.get("source_status") != "ok"
                      or result.get("status") == "unavailable")
            if wcfg.POLL_EVERY_MINUTES > 1 and failed:
                time.sleep(wcfg.POLL_RETRY_SECONDS)
                refresh_temp_now()
        except Exception as exc:
            log.warning("poll error: %s", exc)
        time.sleep(1.0)                          # never re-fire inside the same slot


def start_temp_monitor():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    thread = threading.Thread(target=_poll_loop, daemon=True, name="temp-monitor-bg")
    thread.start()
