"""
Background forecast runner.

Spawns all 6 ML model scripts as subprocesses in parallel every 15 minutes,
caches their JSON output in memory, and exposes helpers for the algorithm.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


MODELS_DIR = Path(__file__).resolve().parent.parent / "modelscripts"

_AW_API_KEY  = os.getenv("ACCUWEATHER_API_KEY")
_AW_BASE     = "https://dataservice.accuweather.com"
_AW_LOCATION = "3593859"
_SYNOPTIC_TOKEN = (
    os.getenv("SYNOPTIC_TOKEN")
    or os.getenv("SYNOPTIC_API_TOKEN")
    or ""
).strip()
_SYNOPTIC_STATION = os.getenv("SYNOPTIC_STATION", "KMIA").strip() or "KMIA"
_SYNOPTIC_BASE = "https://api.synopticdata.com/v2/stations/timeseries"

MODELS = [
    {"name": "accuweather_1h", "script": "accuweather_forecast.py"},
    {"name": "accuweather_2h", "script": "accuweather_forecast_2hour.py"},
    {"name": "accuweather_3h", "script": "accuweather_forecast_3hour.py"},
    {"name": "var_1h",         "script": "variable_combined.py"},
    {"name": "var_2h",         "script": "variable_combined_2hour.py"},
    {"name": "var_3h",         "script": "variable_combined_3hour.py"},
]

REFRESH_INTERVAL = 900  # seconds (15 minutes)
STARTUP_DELAY = 30      # let lightweight /ping/ respond before CPU-heavy models start

_lock = threading.Lock()
_started = False
_loop = {
    "started": False,
    "started_at": None,
    "last_cycle_started_at": None,
    "last_cycle_finished_at": None,
    "next_run_at": None,
    "last_error": None,
}
_cache = {
    "results":      {},     # model_name -> dict (or {"error": ...})
    "last_run_at":  None,   # ISO string
    "running":      False,
    "asos_current": None,   # {"t0_f": float, "today_high_f": float} or None
}


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
    """
    Fetch KMIA high-frequency NOAA METAR/ASOS observations from Synoptic.

    The response keeps date_time and variable arrays aligned by index. We
    request today's Miami-local window and compute the recorded high from all
    non-null air temperature values in that window.
    """
    if not _SYNOPTIC_TOKEN:
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

    headers = {
        "Accept": "application/json",
        "User-Agent": "kalshi-trading-bot/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    params = {
        "stid": _SYNOPTIC_STATION,
        "start": start_utc,
        "end": end_utc,
        "vars": "air_temp",
        "units": "temp|F,speed|mph,english",
        "obtimezone": "local",
        "complete": "1",
        "showemptystations": "1",
        "hfmetars": "1",
        "token": _SYNOPTIC_TOKEN,
    }

    try:
        resp = requests.get(_SYNOPTIC_BASE, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        stations = payload.get("STATION") or []
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
            observed_at = times[i] if i < len(times) else None
            current = {"t0_f": temp_f, "observed_at": observed_at}
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
        print(f"[forecast] Synoptic KMIA fetch failed ({exc}); falling back to AviationWeather/NWS")
        return None


def _fetch_aviation_metar_current():
    """
    Fetch KMIA METAR/SPECI history from NOAA AviationWeather.
    This is the same class of station report shown in the weather.gov WRH timeseries.
    """
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
        from zoneinfo import ZoneInfo as _ZI
        _miami = _ZI("America/New_York")
        today_miami = datetime.now(timezone.utc).astimezone(_miami).date()
        valid = []
        for row in rows:
            temp_f = _metar_temp_f(row)
            if temp_f is None:
                continue
            observed_at = _metar_observed_at(row)
            # Only keep today's Miami-date observations — hours=24 can bleed into yesterday
            if observed_at:
                try:
                    obs_dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                    if obs_dt.astimezone(_miami).date() != today_miami:
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
        print(f"[forecast] AviationWeather METAR fetch failed ({exc}); falling back to NWS observations")
        return None


def _fetch_nws_latest() -> dict | None:
    """
    Fetch the single most recent KMIA observation from NWS /observations/latest.
    This endpoint has < 5 min lag (same source as weather.gov/wrh/timeseries).
    Returns {"t0_f": float, "observed_at": str} or None.
    """
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
        props  = resp.json().get("properties", {})
        temp_c = (props.get("temperature") or {}).get("value")
        if temp_c is None:
            return None
        return {
            "t0_f":        float(temp_c) * 9 / 5 + 32,
            "observed_at": props.get("timestamp"),
            "source":      "nws_latest",
        }
    except Exception as exc:
        print(f"[forecast] NWS /observations/latest failed ({exc})")
        return None


def _fetch_asos_current():
    """
    Fetch today's KMIA observations for current temp and today's recorded high.

    t0_f (current temp)  — fastest source first:
        1. NWS /observations/latest  (< 5 min lag, same source as weather.gov timeseries)
        2. AviationWeather METAR     (< 5 min lag)
        3. NWS observations list     (30-90 min lag, last resort)

    today_high_f (recorded high) — max of all today's obs from:
        • NWS observations list  (since midnight Miami, authoritative date boundary)
        • AviationWeather METAR  (today-filtered, more real-time)
    """
    from datetime import timezone
    from zoneinfo import ZoneInfo

    synoptic = _fetch_synoptic_current_high()
    if synoptic is not None:
        return synoptic

    # Fetch fallback sources in sequence.
    metar      = _fetch_aviation_metar_current()
    nws_latest = _fetch_nws_latest()

    MIAMI_TZ      = ZoneInfo("America/New_York")
    now_utc       = datetime.now(timezone.utc)
    midnight_miami = now_utc.astimezone(MIAMI_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc     = midnight_miami.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {
        "Accept": "application/geo+json",
        "User-Agent": "kalshi-trading-bot/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    nws_list_high = None
    nws_list_t0   = None
    nws_list_obs  = None
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
            props  = feat.get("properties", {})
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
            nws_list_t0   = temps[0]   # newest-first after sort
    except Exception as exc:
        print(f"[forecast] NWS observations list failed ({exc})")

    # Best current reading: /latest > METAR > list (priority by recency)
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

    # Today's high: max of all today-bounded sources
    high_candidates = [v for v in (
        nws_list_high,
        metar.get("today_high_f") if metar else None,
        nws_latest["t0_f"] if nws_latest else None,   # current reading is a candidate for today's high
    ) if v is not None]

    if t0_f is None and not high_candidates:
        return None

    return {
        "t0_f":        t0_f,
        "today_high_f": max(high_candidates) if high_candidates else t0_f,
        "observed_at": observed_at,
        "source":      "fallback",
    }


def _apply_asos_current(asos: dict):
    """Push the latest KMIA ASOS reading into the shared temp tracker."""
    if not asos:
        return
    with _lock:
        _cache["asos_current"] = asos

    from .price_tracker import set_daily_high_nws, update_temp

    if asos.get("t0_f") is not None:
        update_temp(float(asos["t0_f"]))
    if asos.get("today_high_f") is not None:
        set_daily_high_nws(float(asos["today_high_f"]))


def _cached_model_high_f():
    with _lock:
        results = dict(_cache.get("results") or {})
    highs = []
    for data in results.values():
        if not isinstance(data, dict) or "error" in data:
            continue
        try:
            t0 = data.get("T0")
            if t0 is not None:
                highs.append(float(t0))
        except (TypeError, ValueError):
            pass
    return max(highs) if highs else None


def _aw_temp_f(row: dict):
    try:
        return float(((row.get("Temperature") or {}).get("Imperial") or {}).get("Value"))
    except (TypeError, ValueError):
        return None


def _fetch_aw_current_and_high():
    """Fallback current/high from AccuWeather current + historical/24."""
    if not _AW_API_KEY:
        return None

    current_f = None
    high_f = None

    curr = requests.get(
        f"{_AW_BASE}/currentconditions/v1/{_AW_LOCATION}",
        params={"apikey": _AW_API_KEY, "details": "false", "language": "en-us"},
        timeout=10,
    )
    curr.raise_for_status()
    curr_rows = curr.json()
    if curr_rows:
        current_f = _aw_temp_f(curr_rows[0])

    hist = requests.get(
        f"{_AW_BASE}/currentconditions/v1/{_AW_LOCATION}/historical/24",
        params={"apikey": _AW_API_KEY, "details": "false", "language": "en-us"},
        timeout=10,
    )
    hist.raise_for_status()
    temps = [t for t in (_aw_temp_f(row) for row in hist.json()) if t is not None]
    if temps:
        high_f = max(temps)

    if current_f is None:
        return None
    return {"t0_f": current_f, "today_high_f": high_f, "observed_at": None}


def _fetch_aw_raw():
    """Fetch AccuWeather current + forecast raw JSON — 2 API calls shared across all models."""
    curr = requests.get(
        f"{_AW_BASE}/currentconditions/v1/{_AW_LOCATION}",
        params={"apikey": _AW_API_KEY, "details": "true", "language": "en-us"},
        timeout=15,
    )
    curr.raise_for_status()

    fore = requests.get(
        f"{_AW_BASE}/forecasts/v1/hourly/12hour/{_AW_LOCATION}",
        params={"apikey": _AW_API_KEY, "details": "true", "metric": "false", "language": "en-us"},
        timeout=15,
    )
    fore.raise_for_status()

    return {"curr_raw": curr.json()[0], "fore_raw": fore.json()[:6]}


def _run_one_model(model, aw_shared_path=None):
    script = MODELS_DIR / model["script"]
    env = {**os.environ}
    if aw_shared_path:
        env["AW_SHARED_DATA"] = aw_shared_path
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--json-output"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(MODELS_DIR),
            env=env,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1200:]
            return model["name"], {"error": f"exit {proc.returncode}: {tail}"}
        data = json.loads(proc.stdout.strip())
        return model["name"], data
    except subprocess.TimeoutExpired:
        return model["name"], {"error": "timeout after 180s"}
    except json.JSONDecodeError as exc:
        return model["name"], {"error": f"JSON parse error: {exc}"}
    except Exception as exc:
        return model["name"], {"error": str(exc)}


def _do_refresh():
    with _lock:
        if _cache["running"]:
            return
        _cache["running"] = True

    # Pre-fetch AccuWeather data once and share it across all model subprocesses.
    # Current temp/high are owned by temp_monitor.py and update on their own loop.
    aw_shared_path = None
    try:
        raw          = _fetch_aw_raw()

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="aw_shared_"
        )
        json.dump(raw, tmp)
        tmp.close()
        aw_shared_path = tmp.name
    except Exception as exc:
        print(f"[forecast] AW pre-fetch failed ({exc}); models will fetch individually")

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_run_one_model, m, aw_shared_path): m for m in MODELS}
            for future in as_completed(futures):
                name, data = future.result()
                results[name] = data
    finally:
        if aw_shared_path:
            try:
                os.unlink(aw_shared_path)
            except OSError:
                pass
        # Always release the running lock — even if an exception killed the executor
        with _lock:
            _cache["results"]      = results
            _cache["last_run_at"]  = datetime.utcnow().isoformat()
            _cache["running"]      = False

    _record_forecast_snapshot(results)


def _record_forecast_snapshot(results: dict):
    """Persist a history row: current temp + ensemble forecast at this refresh cycle."""
    try:
        from .algorithm import get_ensemble_forecast
        from .price_tracker import get_temp_snapshot
        from .sheets_history import append_snapshot

        ensemble = get_ensemble_forecast(results)
        current_f = get_temp_snapshot().get("current_f")
        if not ensemble or current_f is None:
            return
        append_snapshot(
            current_temp_f=current_f,
            model_forecast_f=ensemble.get("mean"),
        )
    except Exception as exc:
        print(f"[forecast] failed to record history snapshot ({exc})")


def _background_loop():
    with _lock:
        _loop["started"] = True
        _loop["started_at"] = datetime.utcnow().isoformat()
        _loop["next_run_at"] = datetime.utcfromtimestamp(
            time.time() + STARTUP_DELAY
        ).isoformat()
    time.sleep(STARTUP_DELAY)
    while True:
        started_at = time.time()
        with _lock:
            _loop["last_cycle_started_at"] = datetime.utcnow().isoformat()
            _loop["next_run_at"] = None
            _loop["last_error"] = None
        try:
            _do_refresh()
        except Exception as exc:
            error = str(exc)
            with _lock:
                _loop["last_error"] = error
            print(f"[forecast] background refresh crashed ({exc}); will retry in {REFRESH_INTERVAL}s")
        elapsed = time.time() - started_at
        sleep_for = max(0, REFRESH_INTERVAL - elapsed)
        with _lock:
            _loop["last_cycle_finished_at"] = datetime.utcnow().isoformat()
            _loop["next_run_at"] = datetime.utcfromtimestamp(
                time.time() + sleep_for
            ).isoformat()
        time.sleep(sleep_for)


def start_background_refresh():
    """Call once at Django startup (AppConfig.ready)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_background_loop, daemon=True, name="forecast-bg")
    t.start()


def trigger_refresh():
    """Fire-and-forget manual refresh (used by API endpoint)."""
    t = threading.Thread(target=_do_refresh, daemon=True, name="forecast-manual")
    t.start()


def _legacy_refresh_temp_now() -> dict | None:
    from .temp_monitor import refresh_temp_now
    return refresh_temp_now()
    """
    Synchronously fetch the latest KMIA reading and push it into price_tracker.
    NWS is the sole source — no AccuWeather involvement.
    Returns {"t0_f": current_f, "today_high_f": daily_high_f, "observed_at": ...} or None.
    """
    asos = _fetch_asos_current()
    if asos is None:
        return None
    _apply_asos_current(asos)
    from .price_tracker import get_temp_snapshot
    snap = get_temp_snapshot()
    return {
        "t0_f":          snap["current_f"],
        "today_high_f":  snap["daily_high_f"],
        "observed_at":   asos.get("observed_at"),
        "source":        asos.get("source"),
    }


def get_forecast_cache():
    """Return a snapshot of the cache (thread-safe copy)."""
    with _lock:
        return {
            "results":      dict(_cache["results"]),
            "last_run_at":  _cache["last_run_at"],
            "running":      _cache["running"],
            "asos_current": _cache["asos_current"],
            "loop":         dict(_loop),
        }


def ensemble_prob_above(strike_f, results=None):
    """
    Return the average P(T_forecast > strike_f) across all model runs that
    succeeded.  Returns None when no model data is available yet.
    """
    if results is None:
        with _lock:
            results = dict(_cache["results"])

    probs = []
    for data in results.values():
        if "error" in data or "temps" not in data:
            continue
        p = sum(prob for t, prob in zip(data["temps"], data["probs"]) if t > strike_f)
        probs.append(p)

    return (sum(probs) / len(probs)) if probs else None
