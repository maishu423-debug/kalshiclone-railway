"""Helpers for model subprocesses to consume the shared ASOS input."""
import json
import math
import os

SHARED_ENV = "ASOS_SHARED_DATA"


class WeatherUnavailable(RuntimeError):
    """No sufficiently fresh ASOS observation — models must not run."""


def load_shared_input():
    """
    Preferred: the orchestrator fetched ASOS once and passed a JSON file path in
    ASOS_SHARED_DATA.  Standalone/dev runs without it build the same structure
    via the weather package (still NOAA ASOS only).
    """
    path = os.environ.get(SHARED_ENV)
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from . import build_live_shared_input
    return build_live_shared_input()


def require_current(shared):
    cur = shared.get("current")
    md = shared.get("metadata", {})
    if not cur or md.get("status") == "unavailable":
        raise WeatherUnavailable(
            f"no current ASOS observation for {shared.get('station')} "
            f"(age={md.get('latest_observation_age_seconds')}s, status={md.get('status')})")
    return cur


def nan(x):
    """None -> NaN so numpy code can use np.isnan."""
    return float("nan") if x is None else float(x)


def has(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def output_metadata(shared):
    md = shared["metadata"]
    return {
        "weather_source": md.get("weather_source", "NOAA MADIS OMO/HFMETAR (KMIA ASOS)"),
        "station": shared.get("station"),
        "observation_time": md.get("latest_observation_at"),
        "observation_age_seconds": md.get("latest_observation_age_seconds"),
        "weather_status": md.get("status"),
        "forecast_input_mode": md.get("forecast_input_mode", "ASOS_NOWCAST"),
    }
