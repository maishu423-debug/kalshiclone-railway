"""
Recorded daily high, computed the way the NWS KMIA observation page does it.

MADIS HFMETAR carries temperature in whole degrees Celsius (the FAA rounds to
whole C before NWS ever sees it), so 87 F (30.56 C) arrives as 31 C = 87.8 F.
The full-precision readings are the METAR reports (:53 hourly and specials),
which carry tenths of a degree C (30.6 C = 87.1 F).  The 5-minute rows
(minute % 5 == 0) in the NWS observations feed are whole C again and are ignored.

    high_f = max(c * 9/5 + 32) over today's full-precision observations,
             rounded once (to 0.1 F) at the end.

HFMETAR remains the source for the *current* temperature; it is never used for
the recorded high.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from . import config as cfg
from .meteo import c_to_f

log = logging.getLogger("weather.nws_high")

API_URL = "https://api.weather.gov/stations/{station}/observations"
CACHE_SECONDS = 60
TIMEOUT_SECONDS = 10

_lock = threading.Lock()
_cache = {"date": None, "fetched": 0.0, "result": None}


def is_full_precision(observed_at):
    """METAR reports (:53 hourly / specials) have tenths of C; 5-minute rows are whole C."""
    return observed_at.minute % 5 != 0


def high_from_features(features, now, station_tz=None):
    """Pure: NWS observations GeoJSON features -> {"high_f", "high_observed_at", "n_obs", "miami_date"}."""
    tz = station_tz or ZoneInfo(cfg.LOCAL_TZ_NAME)
    today = now.astimezone(tz).date()
    best_c, best_at, n = None, None, 0
    for feat in features:
        props = feat.get("properties") or {}
        value = (props.get("temperature") or {}).get("value")
        stamp = props.get("timestamp")
        if value is None or not stamp:
            continue
        try:
            at = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
            c = float(value)
        except (TypeError, ValueError):
            continue
        if at > now + timedelta(minutes=5) or at.astimezone(tz).date() != today:
            continue
        if not is_full_precision(at):
            continue
        n += 1
        if best_c is None or c > best_c:
            best_c, best_at = c, at.isoformat()
    return {
        "miami_date": today.isoformat(),
        "high_f": None if best_c is None else round(c_to_f(best_c), 1),
        "high_observed_at": best_at,
        "n_obs": n,
    }


def _fetch_features(now):
    tz = ZoneInfo(cfg.LOCAL_TZ_NAME)
    midnight = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    resp = requests.get(
        API_URL.format(station=cfg.ASOS_STATION),
        params={"start": midnight.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": 500},
        headers={"User-Agent": cfg.HTTP_USER_AGENT, "Accept": "application/geo+json"},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json().get("features") or []


def get_recorded_high(now=None):
    """
    Today's recorded high (F, 0.1 precision) or None when it could not be determined.
    Cached for CACHE_SECONDS; a failed fetch keeps today's last good value and never
    falls back to HFMETAR (whose whole-C rounding can overshoot by ~0.7 F).
    """
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo(cfg.LOCAL_TZ_NAME)).date().isoformat()
    with _lock:
        if _cache["date"] == today and time.time() - _cache["fetched"] < CACHE_SECONDS:
            return _cache["result"]
        try:
            result = high_from_features(_fetch_features(now), now)
        except Exception as exc:
            log.warning("NWS observation fetch failed: %s", exc)
            _cache["fetched"] = time.time() - CACHE_SECONDS + 15   # retry in ~15 s
            if _cache["date"] != today:
                _cache.update(date=today, result=None)
            return _cache["result"]
        if result["high_f"] is None and _cache["date"] == today:
            result = _cache["result"]                              # keep last good for the day
        _cache.update(date=today, fetched=time.time(), result=result)
        return result
