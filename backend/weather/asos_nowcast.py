"""
ASOS-derived current state and 15/30/45/60-minute environmental nowcast.

Pure functions over normalized observations (see asos_parser).  No network.

Method (deliberately simple, no ML dependency):
    value(t) = latest_value + damped_robust_slope * elapsed
  * robust slope  = Theil-Sen median of pairwise slopes over a short window
  * damping       = tau * (1 - exp(-elapsed / tau)), so extrapolation saturates
  * clipping      = slope limited to a physical max rate; result clipped to bounds
  * cloud         = persistence blended with a recency-weighted recent average
  * wind          = U/V trended separately (no 0/360 wrap problem)

Temperature is NOT extrapolated into a forecast here.  Temperature trends are
reported only as features; the OU/VAR Monte-Carlo models forecast temperature.
"""
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import config as cfg
from .asos_parser import parse_utc
from .meteo import uv_to_speed_dir

STATUS_HEALTHY = "healthy"
STATUS_DELAYED = "delayed"
STATUS_UNAVAILABLE = "unavailable"


# ── freshness ────────────────────────────────────────────────────────────────

def freshness_status(age_seconds):
    if age_seconds is None:
        return STATUS_UNAVAILABLE
    if age_seconds <= cfg.ASOS_FRESH_MINUTES * 60:
        return STATUS_HEALTHY
    if age_seconds <= cfg.ASOS_MAX_CURRENT_AGE_MINUTES * 60:
        return STATUS_DELAYED
    return STATUS_UNAVAILABLE


# ── today's observed high ────────────────────────────────────────────────────

def todays_high(observations, now=None):
    """
    Highest valid ASOS temperature (deg F) observed so far on the current
    America/New_York calendar day.  Input is already QC'd and de-duplicated
    (one record per UTC minute); observations in the future are ignored.
    """
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.LOCAL_TZ_NAME)
    today = now.astimezone(tz).date()
    best, best_at, n = None, None, 0
    for o in observations:
        dt = parse_utc(o["observed_at"])
        if dt is None or dt > now + timedelta(minutes=5) or dt.astimezone(tz).date() != today:
            continue
        if o.get("temp_f") is None:
            continue
        n += 1
        if best is None or o["temp_f"] > best:
            best, best_at = o["temp_f"], o["observed_at"]
    return {"miami_date": today.isoformat(), "high_f": best, "high_observed_at": best_at, "n_obs": n}


# ── series helpers ───────────────────────────────────────────────────────────

def _series(observations, key, window_minutes, ref_dt):
    """[(hours_before_ref (<=0), value)] for non-null `key` within the window, ascending."""
    out = []
    for o in observations:
        v = o.get(key)
        dt = parse_utc(o["observed_at"])
        if v is None or dt is None:
            continue
        mins = (ref_dt - dt).total_seconds() / 60.0
        if -1e-6 <= mins <= window_minutes:
            out.append((-mins / 60.0, float(v)))
    return out


def theil_sen(points, min_points=cfg.TREND_MIN_POINTS):
    """Median pairwise slope (units/hour) or None when there is too little data."""
    if len(points) < min_points:
        return None
    t = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    ok = dt > 1e-9
    if not ok.any():
        return None
    return float(np.median((y[j] - y[i])[ok] / dt[ok]))


def _trend(observations, key, window_minutes, ref_dt):
    return theil_sen(_series(observations, key, window_minutes, ref_dt))


def _change(observations, key, minutes, ref_dt):
    """Value now minus value `minutes` earlier (nearest obs within tolerance), else None."""
    cur = next((o.get(key) for o in reversed(observations) if o.get(key) is not None), None)
    if cur is None:
        return None
    tol = min(7.5, minutes / 2.0)
    target = ref_dt - timedelta(minutes=minutes)
    best, best_gap = None, None
    for o in observations:
        v = o.get(key)
        dt = parse_utc(o["observed_at"])
        if v is None or dt is None:
            continue
        gap = abs((dt - target).total_seconds()) / 60.0
        if gap <= tol and (best_gap is None or gap < best_gap):
            best, best_gap = v, gap
    return None if best is None else float(cur) - float(best)


def _round(x, n=4):
    return None if x is None else round(float(x), n)


# ── feature block ────────────────────────────────────────────────────────────

def compute_features(observations, ref_dt):
    """Short-term ASOS trend features (all per-hour unless named otherwise)."""
    obs = observations
    pw = cfg.PRESSURE_TREND_WINDOW_MINUTES
    p_series = _series(obs, "pressure_hpa", pw, ref_dt)
    p_trend = theil_sen(p_series)
    p_accel = None
    half = [p for p in p_series if p[0] <= -pw / 120.0]           # older half
    recent = [p for p in p_series if p[0] > -pw / 120.0]          # newer half
    s_old, s_new = theil_sen(half), theil_sen(recent)
    if s_old is not None and s_new is not None:
        p_accel = (s_new - s_old) / (pw / 120.0)                  # change in slope per hour

    t60 = _series(obs, "temp_f", cfg.TREND_WINDOW_MINUTES, ref_dt)
    t_slope = theil_sen(t60)
    vol = None
    if t_slope is not None and len(t60) >= cfg.TREND_MIN_POINTS:
        t = np.array([p[0] for p in t60]); y = np.array([p[1] for p in t60])
        resid = y - (np.median(y - t_slope * t) + t_slope * t)
        vol = float(np.std(resid))

    return {
        "temp_change_5min_f": _round(_change(obs, "temp_f", 5, ref_dt)),
        "temp_change_15min_f": _round(_change(obs, "temp_f", 15, ref_dt)),
        "temp_trend_30min_f_per_hr": _round(_trend(obs, "temp_f", 30, ref_dt)),
        "temp_trend_60min_f_per_hr": _round(t_slope),
        "temp_volatility_f": _round(vol),
        "dewpoint_trend_f_per_hr": _round(_trend(obs, "dewpoint_f", cfg.TREND_WINDOW_MINUTES, ref_dt)),
        "rh_trend_per_hr": _round(_trend(obs, "relative_humidity", cfg.TREND_WINDOW_MINUTES, ref_dt), 5),
        "pressure_trend_hpa_per_hr": _round(p_trend),
        "pressure_accel_hpa_per_hr2": _round(p_accel),
        "wind_u_trend_mph_per_hr": _round(_trend(obs, "wind_u_mph", cfg.TREND_WINDOW_MINUTES, ref_dt)),
        "wind_v_trend_mph_per_hr": _round(_trend(obs, "wind_v_mph", cfg.TREND_WINDOW_MINUTES, ref_dt)),
        "wind_speed_trend_mph_per_hr": _round(_trend(obs, "wind_speed_mph", cfg.TREND_WINDOW_MINUTES, ref_dt)),
        "cloud_recent_avg": _round(_recent_cloud_avg(obs, ref_dt)),
    }


def _recent_cloud_avg(obs, ref_dt):
    pts = _series(obs, "cloud_fraction", cfg.CLOUD_RECENT_MINUTES, ref_dt)
    if not pts:
        return None
    w = np.arange(1, len(pts) + 1, dtype=float)           # newest weighted most
    return float(np.average([p[1] for p in pts], weights=w))


# ── projection ───────────────────────────────────────────────────────────────

_BOUNDS = {
    "relative_humidity": (0.0, 1.0),
    "dewpoint_f": (-40.0, 95.0),
    "pressure_hpa": cfg.QC_PRESSURE_HPA,
    "wind_u_mph": (-cfg.QC_WIND_MPH[1], cfg.QC_WIND_MPH[1]),
    "wind_v_mph": (-cfg.QC_WIND_MPH[1], cfg.QC_WIND_MPH[1]),
}


def _last_valid(observations, key, ref_dt, max_age_min):
    """(value, dt) of the latest non-null `key` no older than max_age_min, else (None, None)."""
    for o in reversed(observations):
        v = o.get(key)
        dt = parse_utc(o["observed_at"])
        if v is not None and dt is not None and (ref_dt - dt).total_seconds() <= max_age_min * 60:
            return float(v), dt
    return None, None


def project(key, base, slope, elapsed_hours):
    """Damped, rate-limited extrapolation of `base` by `slope` over `elapsed_hours`."""
    if base is None:
        return None
    s = 0.0 if slope is None else float(np.clip(slope, -cfg.MAX_RATE_PER_HOUR[key], cfg.MAX_RATE_PER_HOUR[key]))
    tau = cfg.TREND_DAMPING_TAU_HOURS
    delta = s * tau * (1.0 - math.exp(-max(0.0, elapsed_hours) / tau))
    lo, hi = _BOUNDS[key]
    return float(min(hi, max(lo, base + delta)))


def build_future_features(observations, now, steps=None):
    """List of dicts, one per `minutes_ahead`, measured from `now` (not from the obs time)."""
    steps = steps or cfg.FORECAST_STEPS_MINUTES
    if not observations:
        return []
    latest_dt = parse_utc(observations[-1]["observed_at"])
    max_age = cfg.ASOS_MAX_CURRENT_AGE_MINUTES

    trends = {
        "relative_humidity": _trend(observations, "relative_humidity", cfg.TREND_WINDOW_MINUTES, latest_dt),
        "dewpoint_f": _trend(observations, "dewpoint_f", cfg.TREND_WINDOW_MINUTES, latest_dt),
        "pressure_hpa": _trend(observations, "pressure_hpa", cfg.PRESSURE_TREND_WINDOW_MINUTES, latest_dt),
        "wind_u_mph": _trend(observations, "wind_u_mph", cfg.TREND_WINDOW_MINUTES, latest_dt),
        "wind_v_mph": _trend(observations, "wind_v_mph", cfg.TREND_WINDOW_MINUTES, latest_dt),
    }
    bases = {k: _last_valid(observations, k, latest_dt, max_age) for k in trends}
    cloud_last, _ = _last_valid(observations, "cloud_fraction", latest_dt, max_age)
    cloud_avg = _recent_cloud_avg(observations, latest_dt)

    cloud_future = None
    if cloud_last is not None:
        w = cfg.CLOUD_PERSISTENCE_WEIGHT
        cloud_future = float(np.clip(w * cloud_last + (1 - w) * (cloud_avg if cloud_avg is not None else cloud_last), 0, 1))

    out = []
    for m in steps:
        target = now + timedelta(minutes=m)
        row = {"minutes_ahead": m, "valid_at": target.isoformat()}
        for key, (base, bdt) in bases.items():
            elapsed = ((target - bdt).total_seconds() / 3600.0) if bdt else 0.0
            row[key] = _round(project(key, base, trends[key], elapsed), 4)
        speed, direction = uv_to_speed_dir(row["wind_u_mph"], row["wind_v_mph"])
        row["wind_speed_mph"] = _round(speed, 3)
        row["wind_dir_deg"] = _round(direction, 1)
        row["cloud_fraction"] = _round(cloud_future)
        row["pressure_trend_hpa_per_hr"] = _round(
            None if trends["pressure_hpa"] is None
            else float(np.clip(trends["pressure_hpa"], -cfg.MAX_RATE_PER_HOUR["pressure_hpa"],
                               cfg.MAX_RATE_PER_HOUR["pressure_hpa"])), 4)
        out.append(row)
    return out


# ── shared model input ───────────────────────────────────────────────────────

def build_shared_input(observations, info=None, now=None):
    """
    Provider-neutral JSON handed to every model subprocess (fetched once per cycle).
    Always returns a dict; metadata.status says whether `current` may be used.
    """
    now = now or datetime.now(timezone.utc)
    info = info or {}
    observations = sorted(observations, key=lambda o: o["observed_at"])
    latest = observations[-1] if observations else None
    latest_dt = parse_utc(latest["observed_at"]) if latest else None
    age = max(0.0, (now - latest_dt).total_seconds()) if latest_dt else None
    status = freshness_status(age)
    usable = status != STATUS_UNAVAILABLE

    lookback_start = (latest_dt - timedelta(hours=cfg.ASOS_LOOKBACK_HOURS)) if latest_dt else None
    recent = [o for o in observations if parse_utc(o["observed_at"]) >= lookback_start] if latest_dt else []

    shared = {
        "station": cfg.ASOS_STATION,
        "current": latest if usable else None,
        "last_known": None if usable else latest,   # display only — never feed to a model
        "recent_observations": recent,
        "features": compute_features(recent, latest_dt) if usable else {},
        "future_features": build_future_features(recent, now) if usable else [],
        "today": todays_high(observations, now),
        "metadata": {
            "source": cfg.SOURCE_ID,
            "weather_source": cfg.WEATHER_SOURCE_LABEL,
            "forecast_input_mode": cfg.SHARED_INPUT_MODE,
            "status": status,
            "latest_observation_at": latest["observed_at"] if latest else None,
            "latest_observation_age_seconds": None if age is None else round(age, 1),
            "observations_loaded": len(recent),
            "lookback_minutes": cfg.ASOS_LOOKBACK_HOURS * 60,
            "fetched_sources": info.get("sources", []),
            "fetch_errors": info.get("errors", []),
            "last_poll_success": info.get("last_poll_success", not info.get("errors")),
            "source_status": info.get("source_status") or ("ok" if not info.get("errors") else "fetch_error"),
            "unavailable_reason": None if usable else (
                "no KMIA observation has ever been decoded (and no usable persisted last-good)"
                if latest is None else
                f"last good observation {latest['observed_at']} is {round(age)}s old "
                f"(> {cfg.ASOS_MAX_CURRENT_AGE_MINUTES:g} min)"),
            "rejected_count": len(info.get("rejected", [])),
            "generated_at": now.isoformat(),
        },
    }
    return shared


def format_log_summary(shared):
    """Human-readable lines for the cycle log."""
    md = shared["metadata"]
    lines = [
        f"ASOS station: {shared['station']}",
        f"latest observation: {md['latest_observation_at']}",
        f"observation age: {md['latest_observation_age_seconds']} sec  [{md['status']}]",
        f"observations loaded: {md['observations_loaded']}",
        f"lookback: {md['lookback_minutes']} min",
    ]
    cur = shared.get("current")
    if cur:
        lines.append(
            f"T0: {cur['temp_f']}F  RH0: {cur['relative_humidity']}  pressure: {cur['pressure_hpa']} hPa  "
            f"wind: {cur['wind_speed_mph']} mph @ {cur['wind_dir_deg']}  cloud: {cur['cloud_fraction']}")
    for ff in shared.get("future_features", []):
        lines.append(
            f"{ff['minutes_ahead']}m ASOS-nowcast features: RH={ff['relative_humidity']} "
            f"cloud={ff['cloud_fraction']} wind=({ff['wind_u_mph']},{ff['wind_v_mph']}) "
            f"P={ff['pressure_hpa']}")
    if md["status"] == STATUS_UNAVAILABLE:
        lines.append("weather UNAVAILABLE — no current ASOS observation; models will not run")
    return lines
