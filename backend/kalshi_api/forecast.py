"""
Background forecast runner.

Every 15 minutes: fetch KMIA ASOS observations ONCE (NOAA), build the shared
provider-neutral model input (current state + ASOS nowcast), then spawn the 6
model scripts as parallel subprocesses that all read that same JSON.  Results
are cached in memory and exposed to the algorithm.

If there is no sufficiently fresh ASOS observation the models do not run and
every model reports an explicit weather-unavailable error — no other provider
is ever contacted.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from weather import config as wcfg, build_shared_input, fetch_recent_observations, format_log_summary
from weather.nws_high import get_recorded_high
from weather.asos_client import log_refresh
from weather.asos_nowcast import STATUS_UNAVAILABLE
from weather.model_input import SHARED_ENV


MODELS_DIR = Path(__file__).resolve().parent.parent / "modelscripts"

MODELS = [
    {"name": "ou_1h",  "script": "ou_forecast.py",  "args": ["--horizon", "1"]},
    {"name": "ou_2h",  "script": "ou_forecast.py",  "args": ["--horizon", "2"]},
    {"name": "ou_3h",  "script": "ou_forecast.py",  "args": ["--horizon", "3"]},
    {"name": "var_1h", "script": "var_forecast.py", "args": ["--horizon", "1"]},
    {"name": "var_2h", "script": "var_forecast.py", "args": ["--horizon", "2"]},
    {"name": "var_3h", "script": "var_forecast.py", "args": ["--horizon", "3"]},
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
    "asos_current": None,   # {"t0_f", "today_high_f", "observed_at", ...} or None
    "weather":      None,   # shared-input metadata of the last cycle (status, age, source)
}


def _apply_asos_current(shared: dict):
    """Push the latest usable KMIA ASOS reading into the shared temp tracker."""
    cur = shared.get("current")
    if not cur:
        return
    asos = {
        "t0_f": cur.get("temp_f"),
        "today_high_f": (shared.get("today") or {}).get("high_f"),
        "observed_at": cur.get("observed_at"),
        "source": shared["metadata"]["source"],
    }
    with _lock:
        _cache["asos_current"] = asos

    from .price_tracker import set_daily_high_nws, update_current

    if asos["t0_f"] is not None:
        update_current(float(asos["t0_f"]))     # HFMETAR current must not advance the recorded high
    if asos["today_high_f"] is not None:
        set_daily_high_nws(float(asos["today_high_f"]))


def _run_one_model(model, shared_path=None):
    script = MODELS_DIR / model["script"]
    env = {**os.environ}
    if shared_path:
        env[SHARED_ENV] = shared_path
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--json-output",
             *model.get("args", []), "--model-name", model["name"]],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(MODELS_DIR),
            env=env,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1200:]
            return model["name"], {"error": f"exit {proc.returncode}: {tail}"}
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        return model["name"], data
    except subprocess.TimeoutExpired:
        return model["name"], {"error": "timeout after 180s"}
    except json.JSONDecodeError as exc:
        return model["name"], {"error": f"JSON parse error: {exc}"}
    except Exception as exc:
        return model["name"], {"error": str(exc)}


def _fetch_shared_input():
    """One NOAA ASOS fetch per cycle -> normalized shared input (never raises)."""
    now = datetime.now(timezone.utc)
    try:
        observations, info = fetch_recent_observations(now=now, reuse_seconds=240)   # shares the 5-min poll
    except Exception as exc:
        observations, info = [], {"errors": [f"ASOS fetch crashed: {exc}"]}
    log_refresh(observations, info, now, debug=os.getenv("ASOS_DEBUG", "").lower() in {"1", "true", "yes"})
    shared = build_shared_input(observations, info, now=now)
    # HFMETAR is whole-degree C, so its max can overshoot; the recorded high comes from
    # the full-precision NWS observations (what the NWS page shows).
    rec = get_recorded_high(now)
    shared["today"] = {**shared["today"],
                       "high_f": rec["high_f"] if rec else None,
                       "high_observed_at": rec["high_observed_at"] if rec else None,
                       "high_source": "NWS_KMIA_OBS"}
    return shared


def _do_refresh():
    with _lock:
        if _cache["running"]:
            return
        _cache["running"] = True

    results = {}
    shared_path = None
    try:
        shared = _fetch_shared_input()
        for line in format_log_summary(shared):
            print(f"[forecast] {line}")
        for err in shared["metadata"].get("fetch_errors", []):
            print(f"[forecast] ASOS fetch error: {err}")
        weather_md = {**shared["metadata"], "station": shared["station"]}
        with _lock:
            _cache["weather"] = weather_md

        if shared["metadata"]["status"] == STATUS_UNAVAILABLE:
            msg = (f"weather_unavailable: {shared['metadata']['unavailable_reason']}")
            results = {m["name"]: {"error": msg} for m in MODELS}
        else:
            _apply_asos_current(shared)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="asos_shared_", encoding="utf-8")
            json.dump(shared, tmp)
            tmp.close()
            shared_path = tmp.name
            print("[forecast] running forecast models...")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_run_one_model, m, shared_path): m for m in MODELS}
                for future in as_completed(futures):
                    name, data = future.result()
                    results[name] = data
    finally:
        if shared_path:
            try:
                os.unlink(shared_path)
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


def get_forecast_cache():
    """Return a snapshot of the cache (thread-safe copy)."""
    with _lock:
        return {
            "results":      dict(_cache["results"]),
            "last_run_at":  _cache["last_run_at"],
            "running":      _cache["running"],
            "asos_current": _cache["asos_current"],
            "weather":      _cache["weather"],
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
