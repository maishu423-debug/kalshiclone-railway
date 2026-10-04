import os
import threading
import time
from datetime import datetime

import requests


SYNOPTIC_TOKEN = (
    os.getenv("SYNOPTIC_TOKEN")
    or os.getenv("SYNOPTIC_API_TOKEN")
    or ""
).strip()
SYNOPTIC_STATION = os.getenv("SYNOPTIC_STATION", "KMIA").strip() or "KMIA"
SYNOPTIC_BASE = "https://api.synopticdata.com/v2/stations/timeseries"
POLL_INTERVAL = 60

_lock = threading.Lock()
_started = False
_latest = None


def _metar_temp_f(row: dict):
    try:
        temp_c = row.get("temp")
        if temp_c is None:
            return None
        return float(temp_c) * 9 / 5 + 32
    except (TypeError, ValueError):
        return None


def _metar_observed_at(row: dict):
    raw = row.get("obsTime") or row.get("receiptTime") or row.get("reportTime")
    if isinstance(raw, (int, float)):
        ts = float(raw) / 1000 if raw > 10_000_000_000 else float(raw)
        return datetime.utcfromtimestamp(ts).isoformat() + "Z"
    return raw


def _first_observation_array(observations: dict, prefix: str):
    for key, values in observations.items():
        if key.startswith(prefix) and isinstance(values, list):
            return values
    return None


def _fetch_synoptic_current_high():
    if not SYNOPTIC_TOKEN:
        return None

    from datetime import timezone
    from zoneinfo import ZoneInfo

    miami_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    midnight_miami = now_utc.astimezone(miami_tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_utc = midnight_miami.astimezone(timezone.utc).strftime("%Y%m%d%H%M")
    end_utc = now_utc.strftime("%Y%m%d%H%M")

    params = {
        "stid": SYNOPTIC_STATION,
        "start": start_utc,
        "end": end_utc,
        "vars": "air_temp",
        "units": "temp|F,speed|mph,english",
        "obtimezone": "local",
        "complete": "1",
        "showemptystations": "1",
        "hfmetars": "1",
        "token": SYNOPTIC_TOKEN,
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "kalshi-trading-bot/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    try:
        resp = requests.get(SYNOPTIC_BASE, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        stations = resp.json().get("STATION") or []
        if not stations:
            return None

        observations = (stations[0] or {}).get("OBSERVATIONS") or {}
        times = observations.get("date_time") or []
        temps = _first_observation_array(observations, "air_temp_set_")
        if not times or not temps:
            return None

        current = None
        observed_high = None
        for i, temp in enumerate(temps):
            if temp is None:
                continue
            try:
                temp_f = float(temp)
            except (TypeError, ValueError):
                continue
            current = {
                "t0_f": temp_f,
                "observed_at": times[i] if i < len(times) else None,
            }
            observed_high = temp_f if observed_high is None else max(observed_high, temp_f)

        if current is None:
            return None
        return {
            "t0_f": current["t0_f"],
            "today_high_f": observed_high,
            "observed_at": current["observed_at"],
            "source": "synoptic",
        }
    except Exception as exc:
        print(f"[temp-monitor] Synoptic KMIA fetch failed ({exc}); falling back to AviationWeather/NWS")
        return None


def _fetch_aviation_metar_current():
    url = "https://aviationweather.gov/api/data/metar"
    headers = {
        "Accept": "application/json",
        "User-Agent": "kalshi-trading-bot/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        resp = requests.get(
            url,
            params={"ids": "KMIA", "format": "json", "hours": 24, "_": int(time.time())},
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            return None

        from datetime import timezone
        from zoneinfo import ZoneInfo

        miami_tz = ZoneInfo("America/New_York")
        today_miami = datetime.now(timezone.utc).astimezone(miami_tz).date()
        valid = []
        for row in rows:
            temp_f = _metar_temp_f(row)
            if temp_f is None:
                continue
            observed_at = _metar_observed_at(row)
            if observed_at:
                try:
                    obs_dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                    if obs_dt.astimezone(miami_tz).date() != today_miami:
                        continue
                except Exception:
                    pass
            valid.append({"temp_f": temp_f, "observed_at": observed_at})

        if not valid:
            return None
        valid.sort(key=lambda row: row.get("observed_at") or "", reverse=True)
        return {
            "t0_f": valid[0]["temp_f"],
            "today_high_f": max(row["temp_f"] for row in valid),
            "observed_at": valid[0].get("observed_at"),
            "source": "aviationweather",
        }
    except Exception as exc:
        print(f"[temp-monitor] AviationWeather METAR fetch failed ({exc}); falling back to NWS")
        return None


def _fetch_nws_latest():
    headers = {
        "Accept": "application/geo+json",
        "User-Agent": "kalshi-trading-bot/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        resp = requests.get(
            "https://api.weather.gov/stations/KMIA/observations/latest",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        temp_c = (props.get("temperature") or {}).get("value")
        if temp_c is None:
            return None
        return {
            "t0_f": float(temp_c) * 9 / 5 + 32,
            "observed_at": props.get("timestamp"),
            "source": "nws_latest",
        }
    except Exception as exc:
        print(f"[temp-monitor] NWS /observations/latest failed ({exc})")
        return None


def fetch_current_high():
    synoptic = _fetch_synoptic_current_high()
    if synoptic is not None:
        return synoptic

    metar = _fetch_aviation_metar_current()
    nws_latest = _fetch_nws_latest()

    from datetime import timezone
    from zoneinfo import ZoneInfo

    miami_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    midnight_miami = now_utc.astimezone(miami_tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_utc = midnight_miami.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {
        "Accept": "application/geo+json",
        "User-Agent": "kalshi-trading-bot/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    nws_list_high = None
    nws_list_t0 = None
    nws_list_obs = None
    try:
        resp = requests.get(
            "https://api.weather.gov/stations/KMIA/observations",
            params={"start": start_utc, "limit": 500},
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        features = sorted(
            resp.json().get("features", []),
            key=lambda feat: (feat.get("properties", {}) or {}).get("timestamp") or "",
            reverse=True,
        )
        temps = []
        for feat in features:
            props = feat.get("properties", {})
            temp_c = (props.get("temperature") or {}).get("value")
            if nws_list_obs is None and temp_c is not None:
                nws_list_obs = props.get("timestamp")
            if temp_c is None:
                continue
            try:
                temps.append(float(temp_c) * 9 / 5 + 32)
            except (TypeError, ValueError):
                pass
        if temps:
            nws_list_high = max(temps)
            nws_list_t0 = temps[0]
    except Exception as exc:
        print(f"[temp-monitor] NWS observations list failed ({exc})")

    t0_f = (
        metar.get("t0_f") if metar
        else nws_latest["t0_f"] if nws_latest
        else nws_list_t0
    )
    observed_at = (
        (metar or {}).get("observed_at")
        or (nws_latest or {}).get("observed_at")
        or nws_list_obs
    )
    high_candidates = [
        v for v in (
            nws_list_high,
            metar.get("today_high_f") if metar else None,
            nws_latest["t0_f"] if nws_latest else None,
        )
        if v is not None
    ]

    if t0_f is None and not high_candidates:
        return None

    return {
        "t0_f": t0_f,
        "today_high_f": max(high_candidates) if high_candidates else t0_f,
        "observed_at": observed_at,
        "source": "fallback",
    }


def apply_current_high(observation: dict):
    if not observation:
        return
    with _lock:
        global _latest
        _latest = dict(observation)

    from .price_tracker import set_daily_high_nws, update_temp

    if observation.get("t0_f") is not None:
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
