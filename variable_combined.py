# -*- coding: utf-8 -*-
"""
Miami Temperature Forecast — Combined Pipeline  v2
===================================================
Target   : Kalshi temperature markets settled on NWS ASOS observation at KMIA
Location : Miami Intl Airport, FL  (lat 25.793, lon -80.291)

Pipeline per 15-min cycle:

  Step 1 — Fetch data
    AccuWeather /currentconditions   → T0 (current temperature + fallback state)
    AccuWeather /historical/24       → temperature OU fitting (fallback)
    AccuWeather /forecasts/hourly    → AccuWeather forecast anchor
    ASOS Iowa Mesonet (90d cache)    → joint variable model fitting + temp OU fitting

  Step 2 — Joint Variable VAR(1) model  [Change 5]
    Fit VAR(1) on 90 days of ASOS data for 5 variables:
      cloud (%), RH (%), wind_u (mph), wind_v (mph), dpdt (hPa/hr)
    Draw N_PATHS correlated joint weather scenarios:
      [C^(m), H^(m), U^(m), V^(m), DP^(m)]  m=1..N
    Variables are correlated — stormy paths get high cloud + high RH + falling pressure together.

  Step 3 — Temperature extended OU with path-specific covariates  [Change 4]
    Fit OU on ASOS 90d history (same as accuweather_forecast.py v6).
    For each MC path m, use the SAME path's weather variables, not a mean.
    Each path sees its own mu^(m) and sigma^(m).
    T0 comes from AccuWeather /currentconditions (ground truth for current state).
    Solar S is deterministic (geometric formula) — same for all paths.

Changes vs v1:
  Change 4 — full variable uncertainty propagated into temperature model
              (previously: collapsed to mean/median → single scenario)
  Change 5 — joint VAR(1) replaces 7 independent OU models
              (previously: cloud, RH, U, V, DP simulated independently)
  Change 1 — 90d ASOS history from Iowa Mesonet (same as accuweather_forecast.py v6)

ASOS variables modelled jointly (5 vars, dewpoint omitted — collinear with RH in Miami):
  cloud_pct, rh_pct, wind_u_mph, wind_v_mph, dpdt_hPa_hr

Usage:
  export ACCUWEATHER_API_KEY="your_key_here"
  python variable_accuweather_combined.py
"""

import math
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

MIAMI_TZ = ZoneInfo("America/New_York")
import os

try:
    from IPython import display as IPydisplay
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

# ══════════════════════════════════════════════════════════════ CONFIG

API_KEY = os.getenv("ACCUWEATHER_API_KEY")

MIAMI_LAT    = 25.793
MIAMI_LON    = -80.291
LOCATION_KEY = "3593859"   # Miami Intl Airport

AW_BASE     = "https://dataservice.accuweather.com"
AW_CURR_URL = f"{AW_BASE}/currentconditions/v1/{{key}}"
AW_HIST_URL = f"{AW_BASE}/currentconditions/v1/{{key}}/historical/24"
AW_FCST_URL = f"{AW_BASE}/forecasts/v1/hourly/12hour/{{key}}"

N_PATHS     = 10_000
REFRESH_SEC = 900        # 15 minutes

# Temperature OU fitting
FIT_DT       = 1.0       # OU fitting dt (hours)
SIM_DT       = 0.25      # MC sub-step (hours)
PHI_FIXED    = 5.0       # fixed solar coefficient
LAMBDA_RIDGE = 0.5       # fallback ridge penalty, used only if there's too
                          # little data to cross-validate [Phase 2] — see
                          # _select_ridge_penalty. The penalty actually used
                          # for a given fit is returned as ridge_used.
RIDGE_CANDIDATES = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
LAMBDA_CANDIDATES = np.geomspace(0.04, 0.70, 15)   # profiled jointly with covariates [Phase 2]
SIGMA_SHRINK = 0.10       # VAR(1) residual-covariance shrinkage toward its diagonal [Phase 2]
THETA_H_HARMONICS = 2     # # of diurnal harmonics for the smooth hour-of-day correction [Phase 2]

# Parameter (coefficient) uncertainty propagation [Phase 4]. Each MC path
# draws its own perturbed copy of the temp-equation coefficients from
# their ridge sampling covariance, instead of every path sharing the same
# point-estimate coefficients. 1.0 = use the analytical sampling
# covariance as-is; scale down if backtested calibration shows this
# overshoots (e.g. widens an already-well-calibrated short horizon too
# much), scale up if long horizons are still overconfident. Tune this
# against reliability_table() output in backtest_variable_combined.py.
PARAM_UNCERTAINTY_SCALE = 1.0

# ASOS config (shared with accuweather_forecast.py v6)
ASOS_STATION     = "MIA"
ASOS_CACHE_PATH  = "kmia_asos_90d.csv"
ASOS_DAYS        = 90
ASOS_REFRESH_SEC = 7 * 24 * 3600
ASOS_MIN_OBS     = 48

# VAR(1) config
# Variables in joint model (order matters — used throughout)
VAR_KEYS  = ["cloud", "rh", "wind_u", "wind_v", "dpdt", "dewpoint"]
VAR_CLIPS = {
    "cloud":    (0.0,   100.0),
    "rh":       (0.0,   100.0),
    "wind_u":   (-80.0,  80.0),
    "wind_v":   (-80.0,  80.0),
    "dpdt":     (-10.0,  10.0),
    "dewpoint": (-20.0,  95.0),
}

# Display
COL_W = 68
SEP   = "  "

# ══════════════════════════════════════════════════════════════ TIME HELPERS

def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))

def miami_today():
    """Return the current date in Miami local time (resets at midnight ET)."""
    return datetime.now(MIAMI_TZ).date()

def parse_aw_epoch(epoch):
    return pd.Timestamp(epoch, unit="s")

# ══════════════════════════════════════════════════════════════ SOLAR

def geometric_solar(ts, lat=MIAMI_LAT, lon=MIAMI_LON):
    doy            = ts.day_of_year
    decl_deg       = 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    solar_noon_utc = 12.0 - lon / 15.0
    utc_h          = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    ha_deg         = 15.0 * (utc_h - solar_noon_utc)
    lat_r  = math.radians(lat)
    decl_r = math.radians(decl_deg)
    ha_r   = math.radians(ha_deg)
    sin_elev = (math.sin(lat_r) * math.sin(decl_r)
                + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    return float(max(0.0, sin_elev))

def solar_path(timestamps):
    return np.array([geometric_solar(ts) for ts in timestamps])

# ══════════════════════════════════════════════════════════════ WIND HELPERS

def uv_components(speed_mph, dir_deg):
    if math.isnan(speed_mph) or math.isnan(dir_deg): return 0.0, 0.0
    r = math.radians(dir_deg)
    return float(-speed_mph * math.sin(r)), float(-speed_mph * math.cos(r))

def uv_arrays(speeds, dirs):
    pairs = [uv_components(s, d) for s, d in zip(speeds, dirs)]
    return (np.array([p[0] for p in pairs]),
            np.array([p[1] for p in pairs]))

def dewpoint_from_t_rh(temp_f, rh_frac):
    """Magnus-Tetens dewpoint approximation. rh_frac in [0,1]."""
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    rh_pct = max(1.0, rh_frac * 100.0)
    a, b = 17.625, 243.04
    gamma = math.log(rh_pct / 100.0) + (a * t_c) / (b + t_c)
    dp_c = (b * gamma) / (a - gamma)
    return dp_c * 9.0 / 5.0 + 32.0

# ══════════════════════════════════════════════════════════════ ASOS FETCH (Change 1)

_SKYC_MAP = {
    "CLR": 0.00, "SKC": 0.00, "NSC": 0.00,
    "FEW": 0.20, "SCT": 0.45, "BKN": 0.75,
    "OVC": 1.00, "VV":  1.00,
}

def _parse_skyc(skyc):
    if not skyc or skyc == "M":
        return np.nan
    return _SKYC_MAP.get(str(skyc).strip().upper()[:3], 0.5)


def fetch_asos_history(days=ASOS_DAYS, station=ASOS_STATION):
    """Fetch hourly ASOS obs from Iowa State Mesonet. Returns standard DataFrame."""
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    url    = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station":     station,
        "data":        "tmpf,dwpf,relh,sknt,drct,mslp,skyc1,skyc2",
        "year1":  start_dt.year,  "month1": start_dt.month,  "day1": start_dt.day,
        "year2":  end_dt.year,    "month2": end_dt.month,    "day2": end_dt.day,
        "tz": "UTC", "format": "onlycomma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no", "report_type": "3",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    lines = [l for l in resp.text.strip().splitlines() if not l.startswith("#")]
    if len(lines) < 2:
        raise ValueError("ASOS returned no data rows.")
    header = [h.strip() for h in lines[0].split(",")]
    rows   = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, [p.strip() for p in parts]))
        def fval(k, fb=np.nan):
            v = row.get(k, "M")
            if v in ("M","T","",None): return fb
            try: return float(v)
            except: return fb
        temp_f   = fval("tmpf")
        if np.isnan(temp_f): continue
        dwpf     = fval("dwpf")
        rh       = fval("relh")
        wind_kts = fval("sknt", 0.0)
        wind_mph = wind_kts * 1.15078
        wind_dir = fval("drct", np.nan)
        mslp     = fval("mslp", np.nan)
        skyc     = row.get("skyc2","M") if row.get("skyc2","M") not in ("M","") else row.get("skyc1","M")
        cloud    = _parse_skyc(skyc)
        try: ts  = pd.Timestamp(row["valid"])
        except: continue
        wu, wv   = uv_components(wind_mph, wind_dir if not np.isnan(wind_dir) else 180.0)
        if np.isnan(dwpf) and not np.isnan(rh):
            dwpf = dewpoint_from_t_rh(temp_f, rh / 100.0)
        rows.append(dict(
            time=ts, temp_f=temp_f, dewpoint_f=dwpf,
            cloud=cloud if not np.isnan(cloud) else 0.3,
            humidity=rh/100.0 if not np.isnan(rh) else np.nan,
            wind_speed=wind_mph, wind_dir=wind_dir if not np.isnan(wind_dir) else 180.0,
            wind_u=wu, wind_v=wv,
            pressure_hpa=mslp, uv_index=0.0,
        ))
    if not rows:
        raise ValueError("ASOS parse produced no valid rows.")
    df = (pd.DataFrame(rows)
            .sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True))
    return df


def load_asos_cache():
    if not os.path.exists(ASOS_CACHE_PATH):
        return None, False
    age = time.time() - os.path.getmtime(ASOS_CACHE_PATH)
    if age > ASOS_REFRESH_SEC:
        return None, True
    try:
        df = pd.read_csv(ASOS_CACHE_PATH, parse_dates=["time"])
        if len(df) < ASOS_MIN_OBS:
            return None, False
        # Cache may have been written by accuweather_forecast.py which omits
        # wind_u/wind_v — compute them on the fly if missing.
        if "wind_u" not in df.columns or "wind_v" not in df.columns:
            ws = df.get("wind_speed", pd.Series([0.0] * len(df))).fillna(0.0)
            wd = df.get("wind_dir",   pd.Series([180.0] * len(df))).fillna(180.0)
            uv = [uv_components(s, d) for s, d in zip(ws, wd)]
            df["wind_u"] = [x[0] for x in uv]
            df["wind_v"] = [x[1] for x in uv]
            df.to_csv(ASOS_CACHE_PATH, index=False)
        if "dewpoint_f" not in df.columns:
            df["dewpoint_f"] = [
                dewpoint_from_t_rh(t, h) for t, h in zip(df["temp_f"], df["humidity"])
            ]
            df.to_csv(ASOS_CACHE_PATH, index=False)
        return df, False
    except Exception:
        pass
    return None, False


def get_asos_data(verbose=True):
    """Return (df_asos, label). Falls back to None if unavailable."""
    df, stale = load_asos_cache()
    if df is not None:
        if verbose: print(f"  [ASOS] Cache: {len(df)} obs")
        return df, f"ASOS cache ({len(df)} obs)"
    reason = "stale" if stale else "missing"
    if verbose: print(f"  [ASOS] Cache {reason} — fetching from Iowa Mesonet …", flush=True)
    try:
        df = fetch_asos_history()
        df.to_csv(ASOS_CACHE_PATH, index=False)
        if verbose: print(f"  [ASOS] Fetched {len(df)} obs → {ASOS_CACHE_PATH}")
        return df, f"ASOS live ({len(df)} obs)"
    except Exception as e:
        if verbose: print(f"  [ASOS] FAILED ({e}) — using AW /historical/24 fallback")
        return None, "AW /historical/24 (ASOS unavailable)"

# ══════════════════════════════════════════════════════════════ ACCUWEATHER FETCH

_CLOUD_ICON_MAP = {
    1:0.00,2:0.05,3:0.10,4:0.15,5:0.20,
    6:0.40,7:0.60,8:0.75,11:0.85,
    12:0.70,13:0.55,14:0.50,
    15:0.85,16:0.80,17:0.75,18:0.90,
    19:0.70,20:0.80,21:0.85,22:0.90,23:0.90,
    24:0.95,25:0.90,26:1.00,27:0.85,28:0.80,29:0.75,
    30:0.00,31:0.00,32:0.00,
    33:0.00,34:0.05,35:0.10,36:0.40,37:0.60,38:0.75,
    39:0.70,40:0.70,41:0.85,42:0.85,43:0.90,44:1.00,
}

def _parse_obs(obs):
    """Parse AccuWeather current-conditions dict → standard row."""
    temp_f = float((obs.get("Temperature") or {}).get("Imperial",{}).get("Value", np.nan))
    cloud_pct = obs.get("CloudCover")
    cloud = (float(cloud_pct)/100.0 if cloud_pct is not None
             else _CLOUD_ICON_MAP.get(int(obs.get("WeatherIcon",1)), 0.5))
    rh_raw = obs.get("RelativeHumidity")
    humidity = float(rh_raw)/100.0 if rh_raw is not None else np.nan
    wind_speed = float((obs.get("Wind") or {}).get("Speed",{}).get("Imperial",{}).get("Value", np.nan))
    wind_dir   = float((obs.get("Wind") or {}).get("Direction",{}).get("Degrees", np.nan))
    pres_inhg  = (obs.get("Pressure") or {}).get("Imperial",{}).get("Value")
    pressure_hpa = float(pres_inhg)*33.8639 if pres_inhg is not None else np.nan
    uv_index = float(obs.get("UVIndex", 0.0))
    wu, wv   = uv_components(wind_speed if not np.isnan(wind_speed) else 0.0,
                              wind_dir   if not np.isnan(wind_dir)   else 180.0)
    return dict(
        time=parse_aw_epoch(obs["EpochTime"]),
        temp_f=temp_f, cloud=cloud, humidity=humidity,
        wind_speed=wind_speed, wind_dir=wind_dir,
        wind_u=wu, wind_v=wv,
        pressure_hpa=pressure_hpa, uv_index=uv_index,
    )


def fetch_historical_24h():
    resp = requests.get(
        AW_HIST_URL.format(key=LOCATION_KEY),
        params={"apikey": API_KEY, "details": "true", "language": "en-us"}, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    if not raw: raise ValueError("/historical/24 empty.")
    rows = [_parse_obs(o) for o in raw]
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def fetch_current():
    resp = requests.get(
        AW_CURR_URL.format(key=LOCATION_KEY),
        params={"apikey": API_KEY, "details": "true", "language": "en-us"}, timeout=15)
    resp.raise_for_status()
    obs_list = resp.json()
    if not obs_list: raise ValueError("Current conditions empty.")
    return _parse_obs(obs_list[0])


def fetch_forecast():
    resp = requests.get(
        AW_FCST_URL.format(key=LOCATION_KEY),
        params={"apikey": API_KEY, "details": "true", "metric": "false", "language": "en-us"},
        timeout=15)
    resp.raise_for_status()
    rows = []
    for p in resp.json()[:6]:
        rh = p.get("RelativeHumidity")
        rh_val = rh.get("Value") if isinstance(rh, dict) else rh
        cc = p.get("CloudCover")
        rows.append(dict(
            time       = parse_aw_epoch(p["EpochDateTime"]),
            temp_f     = float((p.get("Temperature") or {}).get("Value", np.nan)),
            cloud      = float(cc)/100.0 if cc is not None else np.nan,
            humidity   = float(rh_val)/100.0 if rh_val is not None else np.nan,
            wind_speed = float((p.get("Wind") or {}).get("Speed",{}).get("Value", np.nan)),
            wind_dir   = float((p.get("Wind") or {}).get("Direction",{}).get("Degrees", np.nan)),
        ))
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════ ARRAY UTILS

def _fill(arr, fallback):
    arr = arr.copy().astype(float)
    if np.isnan(arr).all(): return np.full_like(arr, fallback)
    return np.where(np.isnan(arr), float(np.nanmean(arr)), arr)


def _last_valid(arr, fallback=np.nan):
    """Most recent non-NaN value in arr (causal — never looks ahead)."""
    idx = np.where(~np.isnan(arr))[0]
    return float(arr[idx[-1]]) if len(idx) else float(fallback)


def _regularize_hourly(df, tol_minutes=25):
    """
    Reindex observations onto a strict top-of-hour grid  [Phase 1, item 3].

    Any two adjacent rows in the returned frame are exactly one hour apart
    by construction. An hour with no observation within `tol_minutes`
    becomes an explicit gap row (every field NaN except `time`) instead of
    being silently skipped — which previously let e.g. a 2-hour gap
    masquerade as a 1-hour VAR/OU transition once NaN rows were dropped.
    """
    if df is None or len(df) == 0:
        return df
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    start = df["time"].iloc[0].floor("h")
    end   = df["time"].iloc[-1].ceil("h")
    grid  = pd.date_range(start, end, freq="h")

    times = df["time"].to_numpy()
    n     = len(df)
    cols  = [c for c in df.columns if c != "time"]
    out_rows = []
    for slot in grid:
        idx = np.searchsorted(times, np.datetime64(slot))
        best_i, best_dt = None, None
        for i in (idx - 1, idx):
            if 0 <= i < n:
                dt_min = abs((df["time"].iat[i] - slot).total_seconds()) / 60.0
                if dt_min <= tol_minutes and (best_dt is None or dt_min < best_dt):
                    best_i, best_dt = i, dt_min
        row = df.iloc[best_i].to_dict() if best_i is not None else {c: np.nan for c in cols}
        row["time"] = slot
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out["time"] = grid
    return out


def _recent_temp_rate(df_hist24, curr_time, curr_temp, target_lag_hours=1.0, tol_hours=0.5):
    """
    Live recent-temperature rate of change (deg F / hour)  [Phase 3, priority 1].

    Computed from the AccuWeather /historical/24 feed rather than the
    weekly-cached ASOS series: df_hist24 is fetched fresh every 15-min
    cycle, so it's the only genuinely live source of "what was the
    temperature about an hour ago" this pipeline has. The model was fit
    on ~1-hour-lag ASOS transitions, so we target the observation closest
    to target_lag_hours before curr_time and rate-normalize by the ACTUAL
    elapsed time — that keeps the live value comparable to the fitted
    coefficient's units regardless of df_hist24's exact reporting cadence.

    Returns None if df_hist24 is unavailable or nothing falls within
    tol_hours of the target lag (caller should fall back to a neutral or
    ASOS-derived value rather than treat None as zero).
    """
    if df_hist24 is None or len(df_hist24) == 0 or "time" not in df_hist24.columns:
        return None
    target_time = curr_time - pd.Timedelta(hours=target_lag_hours)
    gaps = (df_hist24["time"] - target_time).abs().dt.total_seconds() / 3600.0
    idx = gaps.idxmin()
    if gaps.loc[idx] > tol_hours:
        return None
    t_past = df_hist24.loc[idx, "temp_f"]
    if pd.isna(t_past):
        return None
    elapsed = (curr_time - df_hist24.loc[idx, "time"]).total_seconds() / 3600.0
    if elapsed <= 0.05:   # too close together — rate would be noise-dominated
        return None
    return float((curr_temp - float(t_past)) / elapsed)

# ══════════════════════════════════════════════════════════════ JOINT VAR(1) MODEL (Change 5)

def _build_var_matrix(df):
    """
    Build (n, 6) matrix X on a strict hourly grid.
    Columns: cloud%, rh%, wind_u_mph, wind_v_mph, dpdt_hPa/hr, dewpoint_F

    dpdt is a causal 1-hour backward difference [Phase 1, item 2] computed
    on the regularized grid [Phase 1, item 3] — it never uses a future
    pressure reading and never silently spans more than one hour.

    Returns (X, valid) where valid[i] marks whether row i itself has every
    field present. A genuine 1h transition additionally requires
    valid[i-1] & valid[i] — see fit_var1.
    """
    df = _regularize_hourly(df)
    cloud  = df["cloud"].values.astype(float) * 100.0        # 0-1 → %
    rh     = df["humidity"].values.astype(float) * 100.0     # 0-1 → %
    wind_u = df["wind_u"].values.astype(float)
    wind_v = df["wind_v"].values.astype(float)
    pres   = df["pressure_hpa"].values.astype(float)
    dewpt  = df["dewpoint_f"].values.astype(float)

    dpdt = np.full(len(pres), np.nan)
    dpdt[1:] = pres[1:] - pres[:-1]     # causal backward diff, hPa/hr on the 1h grid
    dpdt = np.clip(dpdt, -10.0, 10.0)

    X = np.column_stack([cloud, rh, wind_u, wind_v, dpdt, dewpt])
    valid = ~np.any(np.isnan(X), axis=1)
    return X, valid


def fit_var1(df_asos):
    """
    Fit VAR(1) on ASOS data: X_{t+1} = a + B*X_t + eps_t
    Returns dict with:
      a     : intercept vector (k,)
      B     : transition matrix (k,k)
      Sigma : residual covariance (k,k) — captures cross-variable correlations
      mu    : unconditional mean of each variable (k,)
      k     : number of variables

    Only genuine 1-hour transitions (both endpoints present on the regular
    hourly grid) are used to fit A/B/Sigma [Phase 1, item 3].
    """
    X, valid = _build_var_matrix(df_asos)
    k = X.shape[1]

    pair_ok = valid[:-1] & valid[1:]
    Xt  = X[:-1][pair_ok]   # predictors
    Xt1 = X[1:][pair_ok]    # targets, exactly 1h later
    n   = len(Xt)
    if n < 10:
        raise ValueError(f"fit_var1: only {n} valid causal 1h transitions in the ASOS window.")

    # OLS: [a | B.T] = (A'A)^{-1} A' Y   where A = [1 | Xt]
    A    = np.column_stack([np.ones(n), Xt])   # (n, k+1)
    ATA  = A.T @ A
    ATY  = A.T @ Xt1                            # (k+1, k)
    coef = np.linalg.solve(ATA, ATY)            # (k+1, k)

    a  = coef[0]        # (k,)  intercept
    B  = coef[1:].T     # (k,k) each row = coefficients for one output variable

    # Stability check [Phase 2]: a VAR(1) is only well-behaved if every
    # eigenvalue of B has magnitude < 1. On a noisy or short window OLS can
    # return a B just past that boundary, which wouldn't raise an error but
    # would make the chained trajectory in sample_var1_path blow up or
    # oscillate over 2-3 hours instead of settling. Shrink B uniformly
    # toward zero (preserving its direction) if the spectral radius creeps
    # past a safety margin.
    rho = float(np.max(np.abs(np.linalg.eigvals(B))))
    if rho >= 0.98:
        B = B * (0.97 / rho)
        rho = 0.97

    resid = Xt1 - (a + Xt @ B.T)      # (n, k) — uses the (possibly shrunk) a/B
    Sigma = (resid.T @ resid) / max(n - 1, 1)

    # Covariance shrinkage [Phase 2]: shrink Sigma toward its diagonal by a
    # modest fixed intensity. With only k=6 variables and thousands of
    # hourly observations, the raw sample covariance is already
    # well-conditioned (shrinkage matters most when k approaches n, which
    # isn't the case here), so a small fixed intensity — rather than a
    # fully data-driven Ledoit-Wolf estimate — is enough to damp noisy
    # cross-variable correlation estimates without materially changing the
    # variances that drive each variable's own spread.
    Sigma = (1 - SIGMA_SHRINK) * Sigma + SIGMA_SHRINK * np.diag(np.diag(Sigma))

    # Regularise Sigma: small diagonal nudge for numerical stability
    Sigma += 1e-6 * np.eye(k)

    # Unconditional mean: mu = (I - B)^{-1} a  (if VAR(1) is stationary)
    try:
        mu = np.linalg.solve(np.eye(k) - B, a)
    except np.linalg.LinAlgError:
        mu = np.mean(X[valid], axis=0)

    return dict(a=a, B=B, Sigma=Sigma, mu=mu, k=k,
                n_obs=n, var_keys=VAR_KEYS, rho=rho)


def sample_var1(params_var, x0_vec, N=N_PATHS):
    """
    Draw N correlated joint weather scenarios one step ahead.

    x0_vec : current state vector (k,) in same order as VAR_KEYS:
             [cloud%, rh%, wind_u, wind_v, dpdt, dewpoint]

    Returns dict {key: np.array(N,)} with clipped samples.
    Each sample is one complete joint weather scenario for path m.
    """
    a, B, Sigma = params_var["a"], params_var["B"], params_var["Sigma"]
    k = params_var["k"]

    # Deterministic next-step mean for all paths: same x0
    x_mean = a + B @ x0_vec   # (k,)

    # Draw N correlated shock vectors
    try:
        eps = np.random.multivariate_normal(np.zeros(k), Sigma, size=N)  # (N, k)
    except np.linalg.LinAlgError:
        # Fall back to independent draws if Sigma is numerically bad
        eps = np.random.randn(N, k) * np.sqrt(np.diag(Sigma))

    X_next = x_mean + eps   # (N, k)  broadcast: x_mean shape (k,) → (N,k)

    # Clip each variable to physical bounds
    result = {}
    for i, key in enumerate(VAR_KEYS):
        lo, hi = VAR_CLIPS[key]
        result[key] = np.clip(X_next[:, i], lo, hi)

    return result   # dict of (N,) arrays — one per variable

# ══════════════════════════════════════════════════════════════ TEMPERATURE OU

def _select_ridge_penalty(Xs, Yc, candidates=RIDGE_CANDIDATES, n_folds=5):
    """
    Cross-validate the ridge penalty on STANDARDIZED predictors [Phase 2].

    Uses contiguous (blocked) folds rather than shuffled/random folds: the
    underlying rows are an hourly time series, so a shuffled fold would let
    a training row sit one hour away from its held-out neighbor, leaking
    almost all the signal we're trying to test on. This is a hyperparameter
    search, not the walk-forward performance backtest — a genuine causal
    (train-on-past-only) evaluation of the final model is what the
    walk-forward CRPS backtest is for.

    Falls back to LAMBDA_RIDGE (the fixed default) if there's too little
    data for CV to be reliable.
    """
    n, k = Xs.shape
    if n < n_folds * 10:
        return LAMBDA_RIDGE

    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    errs = np.full(len(candidates), np.inf)
    for ci, lam in enumerate(candidates):
        sse, cnt = 0.0, 0
        for f in range(n_folds):
            lo, hi = bounds[f], bounds[f + 1]
            if hi <= lo:
                continue
            test = np.arange(lo, hi)
            train = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
            if len(train) < k + 2:
                continue
            b = np.linalg.solve(Xs[train].T @ Xs[train] + lam * np.eye(k),
                                 Xs[train].T @ Yc[train])
            resid = Yc[test] - Xs[test] @ b
            sse += float(resid @ resid)
            cnt += len(test)
        if cnt > 0:
            errs[ci] = sse / cnt
    if not np.isfinite(errs).any():
        return LAMBDA_RIDGE
    return float(candidates[int(np.argmin(errs))])


def fit_temp_ou(hist_c, hist_t, hist_h, hist_u, hist_v, hist_s, hist_dp, hist_dew,
                hist_hours=None, dt=FIT_DT, valid_mask=None, hist_dtr=None):
    """
    8-step temperature OU fit: dewpoint plus a recent-temperature-rate
    regressor alongside the original covariates [Phase 3, priority 1].
    hist_dew : dewpoint F array, same length as hist_t.
    hist_dtr : causal recent temperature rate (deg F / hour) — the change
               over the PRIOR hour, known at time t, used to predict the
               t -> t+1 transition. The OU process otherwise only sees the
               current level T, not whether it's currently rising,
               flattening, or falling. If omitted, the trend term drops
               out and psi_trend fits to ~0.

    valid_mask : optional bool array, length len(hist_t)-1. valid_mask[i]
                 marks whether the transition hist_t[i] -> hist_t[i+1] is a
                 genuine 1-hour step [Phase 1, item 3] — i.e. every covariate
                 is present at both endpoints on the regular hourly grid.
                 Rows failing this are excluded so a multi-hour gap can never
                 be fit as if it were a 1-hour temperature change.
    """
    dT = np.diff(hist_t)
    C  = hist_c[:-1];  T  = hist_t[:-1]
    H  = hist_h[:-1];  U  = hist_u[:-1];  V  = hist_v[:-1]
    S  = hist_s[:-1];  DP = hist_dp[:-1]; DEW = hist_dew[:-1]
    DTR = hist_dtr[:-1] if hist_dtr is not None else np.zeros(len(T))

    hours_lag = (hist_hours[:-1].astype(int) % 24
                 if hist_hours is not None and len(hist_hours) == len(hist_t)
                 else np.zeros(len(T), dtype=int))

    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        dT, C, T, H, U, V, S, DP, DEW, DTR, hours_lag = (
            dT[m], C[m], T[m], H[m], U[m], V[m], S[m], DP[m], DEW[m], DTR[m], hours_lag[m]
        )

    CS = C * S

    def cov(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9: return 0.0
        return float(np.cov(a, b)[0, 1])

    # Onshore/offshore wind decomposition [Phase 3]: Miami's coastline runs
    # roughly north-south with the Atlantic to the east, so cross-shore
    # flow is closely approximated by the east-west wind component alone.
    # In this codebase's u/v convention (u = eastward velocity), wind
    # blowing in FROM the east — the sea breeze — has u<0, so onshore = -U.
    # Sea breeze tends to cap/cool the afternoon high (cooler, moister
    # ocean air triggers convection that limits solar heating); offshore
    # flow (dry mainland/Everglades air, or simply no sea breeze) tends to
    # let the high run hotter. These are physically distinct mechanisms,
    # not mirror images of each other, so — unlike a single linear
    # delta_u*U term — onshore and offshore each get their own coefficient.
    # wind_v (alongshore) stays a simple linear term since it doesn't carry
    # the same maritime-air-mass story.
    ONSHORE      = -U
    onshore_pos  = np.maximum(ONSHORE, 0.0)
    offshore_pos = np.maximum(-ONSHORE, 0.0)

    phi = PHI_FIXED
    X = np.column_stack([-C, -CS, -H, -onshore_pos, offshore_pos, -V, DP, DEW, DTR])
    k = X.shape[1]

    def _fit_covariates(lam_try, ridge_penalty):
        """Ridge-fit the covariates conditional on a candidate lambda, and
        report the actual dT-space residual SSE (not the Y-space ridge
        objective, which isn't comparable across different lambdas since Y
        itself is rescaled by 1/sc)."""
        sc_try = lam_try * dt + 1e-9
        Y_try  = dT / sc_try + T - phi * S
        Xmean, Ymean = X.mean(axis=0), Y_try.mean()
        Xc_try, Yc_try = X - Xmean, Y_try - Ymean
        x_std = Xc_try.std(axis=0)
        x_std[x_std < 1e-8] = 1.0
        Xs_try = Xc_try / x_std
        beta_std = np.linalg.solve(Xs_try.T @ Xs_try + ridge_penalty * np.eye(k), Xs_try.T @ Yc_try)
        beta_try = beta_std / x_std
        dT_hat = sc_try * ((Xc_try @ beta_try + Ymean) - T + phi * S)
        sse = float(np.sum((dT - dT_hat) ** 2))
        return beta_try, sse

    # Joint (profiled) fit of lambda and the covariates [Phase 2]. Lambda
    # previously came from T's own autocovariance alone
    # (cov(dT,T)/var(T)), ignoring every other covariate, then the
    # covariates were fit conditional on that fixed value — a valid but not
    # fully efficient two-step estimator. Instead, profile a grid of
    # candidate lambdas: at each one, ridge-fit the covariates and measure
    # the actual dT prediction error, then keep the lambda that minimizes
    # it. This lets the covariates inform the mean-reversion rate estimate.
    # (Solving for lambda algebraically by dividing ridge-shrunk
    # coefficients was considered and rejected — that ratio is numerically
    # fragile whenever ridge pulls the denominator coefficient toward 0.)
    best_lam, best_sse = LAMBDA_CANDIDATES[0], np.inf
    for lam_try in LAMBDA_CANDIDATES:
        _, sse_try = _fit_covariates(lam_try, LAMBDA_RIDGE)
        if sse_try < best_sse:
            best_lam, best_sse = lam_try, sse_try
    lam = float(best_lam)
    sc  = lam * dt + 1e-9

    # Final fit at the chosen lambda, with the ridge penalty itself
    # cross-validated [Phase 2] rather than fixed.
    Y    = dT / sc + T - phi * S
    Xc   = X - X.mean(axis=0);  Yc = Y - Y.mean()

    # Standardize each predictor to unit variance before ridge [Phase 2]:
    # cloud (0-1), wind (mph), dewpoint (deg F ~50-80) and the rest live on
    # wildly different raw scales, so a single fixed penalty on the
    # unstandardized columns regularizes them very unevenly — a
    # large-magnitude column like dewpoint barely feels the penalty while a
    # small one like cloud fraction gets shrunk hard, for no reason
    # connected to how informative either actually is.
    x_std = Xc.std(axis=0)
    x_std[x_std < 1e-8] = 1.0   # guard a constant/degenerate column
    Xs = Xc / x_std

    ridge_used = _select_ridge_penalty(Xs, Yc)
    A          = Xs.T @ Xs + ridge_used * np.eye(k)
    A_inv      = np.linalg.inv(A)
    beta_std   = A_inv @ (Xs.T @ Yc)
    beta       = beta_std / x_std   # back to native-unit coefficients

    # Sampling covariance of beta [Phase 4], for per-path parameter-
    # uncertainty draws in the MC — see monte_carlo_temp_vectorised. Ridge
    # sandwich formula: Var(beta_std) = sigma^2 * A^-1 (Xs'Xs) A^-1, using
    # the fit's own residual variance for sigma^2. Rescaled from
    # standardized to native units by the same diagonal x_std transform
    # used to un-standardize beta itself.
    resid_std      = Yc - Xs @ beta_std
    n              = len(T)
    sigma2_std     = float(np.sum(resid_std ** 2)) / max(n - k, 1)
    beta_std_cov   = sigma2_std * (A_inv @ (Xs.T @ Xs) @ A_inv)
    D_inv          = np.diag(1.0 / x_std)
    beta_cov       = D_inv @ beta_std_cov @ D_inv   # native units

    alpha1       = float(np.clip(beta[0],  0.0, 15.0))
    alpha2       = float(np.clip(beta[1],  0.0, 10.0))
    gamma        = float(np.clip(beta[2],  0.0, 10.0))
    psi_onshore  = float(np.clip(beta[3],  0.0,  5.0))
    psi_offshore = float(np.clip(beta[4],  0.0,  5.0))
    delta_v      = float(np.clip(beta[5], -5.0,  5.0))
    kappa        = float(np.clip(beta[6], -3.0,  3.0))
    epsilon      = float(np.clip(beta[7], -3.0,  3.0))
    psi_trend    = float(np.clip(beta[8], -3.0,  3.0))

    # Bounds, in the same [alpha1,alpha2,gamma,psi_onshore,psi_offshore,
    # delta_v,kappa,epsilon,psi_trend] order as beta — MUST mirror the
    # individual np.clip calls just above. Used again below to clip each
    # MC path's individually-sampled coefficients [Phase 4].
    beta_bounds = np.array([
        (0.0, 15.0), (0.0, 10.0), (0.0, 10.0), (0.0, 5.0), (0.0, 5.0),
        (-5.0, 5.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0),
    ])
    beta_mean = np.array([alpha1, alpha2, gamma, psi_onshore, psi_offshore,
                           delta_v, kappa, epsilon, psi_trend])

    # RH / dewpoint redundancy check [Phase 2]: both stay in the model as
    # separate regressors — empirically they run only ~+0.5 correlated in
    # Miami (not the near-total redundancy that would justify dropping
    # one), and switching to dewpoint depression (T - DEW) instead would
    # trade that for near-total collinearity with T itself. Reported here
    # so a real regime shift (this correlation climbing well past ~0.5)
    # would show up as a visible number rather than silently degrading the
    # fit.
    corr_H_DEW = cov(H, DEW) / (np.std(H) * np.std(DEW) + 1e-9)
    corr_H_DEW = float(np.clip(corr_H_DEW, -1.0, 1.0))

    # Hour-of-day shifts
    mu_pre = (np.mean(T)
              + alpha1*np.mean(C) + alpha2*np.mean(CS) + gamma*np.mean(H)
              + psi_onshore*np.mean(onshore_pos) - psi_offshore*np.mean(offshore_pos)
              + delta_v*np.mean(V)
              - phi*np.mean(S) - kappa*np.mean(DP) - epsilon*np.mean(DEW)
              - psi_trend*np.mean(DTR))
    mu_vec_pre = (mu_pre - alpha1*C - alpha2*CS - gamma*H
                  - psi_onshore*onshore_pos + psi_offshore*offshore_pos
                  - delta_v*V + phi*S + kappa*DP + epsilon*DEW
                  + psi_trend*DTR)
    res_pre = dT - lam*(mu_vec_pre - T)*dt

    # Smooth (Fourier) hour-of-day correction [Phase 2], replacing 24
    # independent per-hour dummies. With a fixed ASOS window, some hours
    # can have only a handful of valid "clear" observations, so an
    # individual dummy per hour is noisy; a low-order harmonic curve
    # borrows strength across neighboring hours while still capturing the
    # same diurnal shape (one broad midday-warming / dawn-cooling cycle).
    raw_theta = res_pre / (lam * dt + 1e-9)
    h_rad = 2 * np.pi * hours_lag / 24.0
    F = np.column_stack(
        [np.ones(len(h_rad))] +
        [fn(j * h_rad) for j in range(1, THETA_H_HARMONICS + 1) for fn in (np.cos, np.sin)]
    )
    harm_coefs, *_ = np.linalg.lstsq(F, raw_theta, rcond=None)
    hg = np.arange(24)
    hg_rad = 2 * np.pi * hg / 24.0
    Fg = np.column_stack(
        [np.ones(24)] +
        [fn(j * hg_rad) for j in range(1, THETA_H_HARMONICS + 1) for fn in (np.cos, np.sin)]
    )
    theta_h = Fg @ harm_coefs
    theta_h -= theta_h.mean()

    mu = float(np.mean(T)
               + alpha1*np.mean(C) + alpha2*np.mean(CS)
               + gamma*np.mean(H)
               + psi_onshore*np.mean(onshore_pos) - psi_offshore*np.mean(offshore_pos)
               + delta_v*np.mean(V)
               - phi*np.mean(S) - kappa*np.mean(DP) - epsilon*np.mean(DEW)
               - psi_trend*np.mean(DTR))

    th_vec  = np.array([theta_h[h] for h in hours_lag])
    mu_full = (mu + th_vec - alpha1*C - alpha2*CS - gamma*H
               - psi_onshore*onshore_pos + psi_offshore*offshore_pos
               - delta_v*V + phi*S + kappa*DP + epsilon*DEW
               + psi_trend*DTR)
    res = dT - lam*(mu_full - T)*dt

    clear = C < 0.5
    s0 = float(np.std(res[clear])  / np.sqrt(dt)) if clear.sum()    > 1 else 0.5
    s1 = float(np.std(res[~clear]) / np.sqrt(dt)) if (~clear).sum() > 1 else s0 + 0.2
    W_mag = np.sqrt(U**2 + V**2)
    eta  = float(np.clip(cov(np.abs(res), W_mag)      / (np.var(W_mag)      + 1e-9), 0.0, 0.10))
    zeta = float(np.clip(cov(np.abs(res), np.abs(DP)) / (np.var(np.abs(DP)) + 1e-9), 0.0,  0.5))

    return dict(
        lambda_=lam, mu_clear=mu,
        alpha1=alpha1, alpha2=alpha2,
        gamma=gamma, psi_onshore=psi_onshore, psi_offshore=psi_offshore, delta_v=delta_v,
        phi=phi, kappa=kappa, epsilon=epsilon, psi_trend=psi_trend, theta_h=theta_h,
        sigma0=float(np.clip(s0, 0.05, 3.0)),
        beta=float(np.clip(s1-s0, 0.0, 2.0)),
        eta=eta, zeta=zeta, ridge_used=ridge_used, corr_H_DEW=corr_H_DEW,
        beta_mean=beta_mean, beta_cov=beta_cov, beta_bounds=beta_bounds,
    )


def _sample_beta_per_path(p, N):
    """
    Draw N per-path realizations of the temp-equation coefficients from
    their ridge sampling covariance [Phase 4], instead of every path
    sharing the same point-estimate coefficients. Each path represents
    "what if the true coefficients were slightly different from the fitted
    point estimate" — drawn ONCE per path (not re-drawn every sub-step),
    since parameter uncertainty is uncertainty about a fixed unknown true
    value, not a stochastic process. A fixed per-path bias then compounds
    across however many sub-steps that path takes, so this naturally
    injects more uncertainty at longer horizons (more steps for the bias
    to accumulate) without a separate by-horizon tuning knob.

    Draws are NOT clipped to beta_bounds. An earlier version clipped each
    draw the same way the point estimate is clipped, which seemed like an
    obvious safety measure but is a biased estimator: whenever a
    coefficient's point estimate sits at or near its own bound (which
    happens routinely here — alpha1/gamma frequently fit right at their
    ceiling), clipping pulls the ENSEMBLE MEAN away from that boundary,
    because the tail that wanted to exceed it piles up exactly at the
    edge while the other tail spreads freely below. Measured directly:
    with a coefficient's point estimate sitting at its own bound and a
    realistic sampling std, ~55% of draws got clipped and the resulting
    ensemble mean was off by several full units from the fitted value —
    a systematic bias in the forecast mean, not just its spread, and the
    likely cause of the calibration regression seen in backtesting after
    Phase 4 first shipped. The coefficient bounds exist to regularize the
    POINT ESTIMATE against an unstable ridge fit on limited data; once
    deliberately sampling to represent genuine uncertainty, clipping every
    draw back to that same bound undoes the sampling. The covariance
    itself (data-derived, and already guarded below) is what should keep
    draws sane, not a hard re-clip.

    Returns a (k, N) array — rows in the same order as beta_mean
    ([alpha1,alpha2,gamma,psi_onshore,psi_offshore,delta_v,kappa,epsilon,
    psi_trend]) — or None if beta_mean/beta_cov aren't available (e.g. an
    older cached params dict) or the covariance isn't usable, in which
    case the caller should fall back to the point estimate for every path.
    """
    beta_mean = p.get("beta_mean")
    beta_cov  = p.get("beta_cov")
    if beta_mean is None or beta_cov is None:
        return None

    k = len(beta_mean)
    cov = beta_cov * (PARAM_UNCERTAINTY_SCALE ** 2) + 1e-9 * np.eye(k)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Covariance not PD (can happen with very little fitting data) --
        # fall back to no injected parameter uncertainty rather than crash
        # or draw something numerically meaningless.
        return None

    return beta_mean[:, None] + L @ np.random.randn(k, N)   # (k, N)


def monte_carlo_temp_vectorised(T0, var_samples, s_path, p,
                                 hour_path=None, N=N_PATHS, dt=SIM_DT, dtrend0=0.0):
    """
    Temperature MC with path-specific weather variables (Change 4).

    var_samples : dict {key: np.array(N,)} — one value per path per variable
                  keys: cloud (%), rh (%), wind_u, wind_v, dpdt, dewpoint
    s_path      : np.array(N_STEPS+1,) — deterministic solar per sub-step
    hour_path   : np.array(N_STEPS+1,) — fractional hour per sub-step
    dtrend0     : recent temperature rate at issue time (deg F / hour)
                  [Phase 3, priority 1]. A single scalar, held constant
                  across the whole horizon — there's no forward model yet
                  for how momentum itself evolves, so this is a fixed
                  "warm start" bias rather than a simulated path.

    The key difference from v1: C, H, U, V, DP are (N,) arrays, not scalars.
    mu and sigma are computed per-path, not shared across paths — and, as
    of [Phase 4], so are the temp-equation coefficients themselves (see
    _sample_beta_per_path): each path uses its own draw from the fitted
    coefficients' sampling covariance rather than the shared point
    estimate, so parameter estimation uncertainty compounds naturally over
    the horizon instead of vanishing entirely.

    Returns (T_end, T_running_max)  [Phase 1, item 1]:
      T_end          — terminal simulated temperature per path (N,)
      T_running_max  — running max of the simulated path per path (N,),
                       including T0. The Kalshi settlement high can be set
                       at any point along the path, not only at the horizon
                       endpoint, so this — not T_end — is the correct
                       quantity to floor against obs_high downstream.
    """
    T      = np.full(N, float(T0), dtype=np.float64)
    T_max  = T.copy()
    N_STEPS = len(s_path) - 1

    # Variable samples in [0,1] fractions or mph as-is
    # cloud in %   → divide by 100 for OU formula (which expects 0-1)
    C_arr  = var_samples["cloud"]  / 100.0          # (N,)
    H_arr  = var_samples["rh"]    / 100.0           # (N,)
    U_arr  = var_samples["wind_u"]                  # (N,)
    V_arr  = var_samples["wind_v"]                  # (N,)
    DP_arr = var_samples["dpdt"]                    # (N,)
    DEW_arr = var_samples["dewpoint"]               # (N,)
    W_arr  = np.sqrt(U_arr**2 + V_arr**2)           # (N,)
    # Onshore/offshore decomposition [Phase 3] — see fit_temp_ou for the
    # coastline-orientation rationale; onshore = -U in this codebase's u/v
    # convention (u = eastward velocity, sea breeze blows in from the east).
    onshore_pos_arr  = np.maximum(-U_arr, 0.0)
    offshore_pos_arr = np.maximum(U_arr, 0.0)

    # Per-path coefficient draws [Phase 4] — see _sample_beta_per_path.
    # Falls back to the shared point estimate (old behavior) if the fitted
    # params don't include a usable sampling covariance.
    beta_paths = _sample_beta_per_path(p, N)
    if beta_paths is not None:
        (alpha1_i, alpha2_i, gamma_i, psi_onshore_i, psi_offshore_i,
         delta_v_i, kappa_i, epsilon_i, psi_trend_i) = beta_paths
    else:
        alpha1_i       = np.full(N, p["alpha1"])
        alpha2_i       = np.full(N, p["alpha2"])
        gamma_i        = np.full(N, p["gamma"])
        psi_onshore_i  = np.full(N, p["psi_onshore"])
        psi_offshore_i = np.full(N, p["psi_offshore"])
        delta_v_i      = np.full(N, p["delta_v"])
        kappa_i        = np.full(N, p["kappa"])
        epsilon_i      = np.full(N, p["epsilon"])
        psi_trend_i    = np.full(N, p.get("psi_trend", 0.0))

    lam = p["lambda_"]

    for i in range(N_STEPS):
        S  = float(s_path[i])
        hr = float(hour_path[i]) if hour_path is not None else None

        # Per-path equilibrium (vectorised — all N paths simultaneously)
        theta = p["theta_h"][int(hr) % 24] if hr is not None else 0.0
        mu_vec = (p["mu_clear"]
                  - alpha1_i * C_arr
                  - alpha2_i * C_arr * S
                  - gamma_i  * H_arr
                  - psi_onshore_i  * onshore_pos_arr
                  + psi_offshore_i * offshore_pos_arr
                  - delta_v_i* V_arr
                  + p["phi"]    * S
                  + kappa_i  * DP_arr
                  + epsilon_i* DEW_arr
                  + psi_trend_i * dtrend0
                  + theta)                          # (N,)

        # Per-path noise scale
        sig_vec = np.maximum(0.01,
                             p["sigma0"]
                             + p["beta"]  * C_arr
                             + p["eta"]   * W_arr
                             + p["zeta"]  * np.abs(DP_arr))   # (N,)
        # Day/night scale
        sig_vec = sig_vec * (1.15 if S > 0.1 else 1.0)

        Z = np.random.randn(N)
        T = T + lam*(mu_vec - T)*dt + sig_vec*math.sqrt(dt)*Z
        T_max = np.maximum(T_max, T)

    return T, T_max

# ══════════════════════════════════════════════════════════════ COMPUTE

def compute(df_asos, asos_label, df_hist24, curr, df_fore, obs_high=None):
    """
    Full combined pipeline with Change 4 + Change 5.

    df_asos   : 90d ASOS DataFrame (or None → fallback to df_hist24)
    df_hist24 : AccuWeather /historical/24 (fallback)
    curr      : parsed current-conditions dict
    df_fore   : AccuWeather hourly forecast DataFrame
    obs_high  : observed daily high so far (°F). MC paths are floored to max(obs_high, T0)
                because the daily high can never decrease.
    """
    # ── Choose fitting dataset ────────────────────────────────────────────
    if df_asos is not None and len(df_asos) >= ASOS_MIN_OBS:
        df_fit     = df_asos
        fit_source = asos_label
    else:
        df_fit     = df_hist24.tail(24).reset_index(drop=True)
        fit_source = f"AW /historical/24 ({len(df_hist24)} obs)"

    # Regularize onto a strict hourly grid [Phase 1, item 3] so that every
    # lag-1 pair used below is a genuine 1-hour transition, not two
    # observations that happen to survive NaN-dropping next to each other.
    df_fit = _regularize_hourly(df_fit)
    if "dewpoint_f" not in df_fit.columns:
        df_fit["dewpoint_f"] = np.nan
    df_fit["dewpoint_f"] = [
        v if not np.isnan(v) else
        (dewpoint_from_t_rh(t, h) if not (np.isnan(t) or np.isnan(h)) else np.nan)
        for v, t, h in zip(df_fit["dewpoint_f"], df_fit["temp_f"], df_fit["humidity"])
    ]

    hist_t   = df_fit["temp_f"].values.astype(float)
    hist_c   = df_fit["cloud"].values.astype(float)
    hist_h   = df_fit["humidity"].values.astype(float)
    hist_u   = df_fit["wind_u"].values.astype(float)
    hist_v   = df_fit["wind_v"].values.astype(float)
    hist_p   = df_fit["pressure_hpa"].values.astype(float)
    hist_s   = solar_path(list(df_fit["time"]))
    hist_dew = df_fit["dewpoint_f"].values.astype(float)
    hist_hours = np.array([ts.hour for ts in df_fit["time"]], dtype=float)

    # Causal 1-hour pressure tendency [Phase 1, item 2]: backward difference
    # on the regular grid — never uses a future pressure reading, unlike the
    # previous np.gradient (centered difference) approach.
    hist_dp = np.full(len(hist_p), np.nan)
    hist_dp[1:] = hist_p[1:] - hist_p[:-1]
    hist_dp = np.clip(hist_dp, -6.0, 6.0)

    # Recent temperature rate [Phase 3, priority 1]: change over the PRIOR
    # hour, deg F/hr, known at time t and used to predict the t -> t+1
    # transition. Captures whether temperature is currently rising,
    # flattening, or falling — information the OU process's current level
    # T alone doesn't carry.
    hist_dtr = np.full(len(hist_t), np.nan)
    hist_dtr[1:] = hist_t[1:] - hist_t[:-1]

    # A transition i-1 -> i is usable only if every covariate is present at
    # both endpoints [Phase 1, item 3]. Feeds fit_temp_ou below.
    row_ok = ~(np.isnan(hist_t) | np.isnan(hist_c) | np.isnan(hist_h) |
               np.isnan(hist_u) | np.isnan(hist_v) | np.isnan(hist_dew))
    # fit_temp_ou uses DP = hist_dp[:-1] and DTR = hist_dtr[:-1] (state
    # known AT time t) as covariates for the t -> t+1 transition, so those
    # are the slices that must be NaN-free — hist_dp[t]/hist_dtr[t]
    # themselves additionally require the row at t-1 valid.
    step_ok = (row_ok[:-1] & row_ok[1:]
               & ~np.isnan(hist_dp[:-1]) & ~np.isnan(hist_dtr[:-1]))

    # ── Current state from AccuWeather (T0, solar, fallback vars) ─────────
    T0    = float(curr["temp_f"])
    S0    = geometric_solar(curr["time"])
    H0_hr = curr["time"].hour
    C0_aw = float(curr["cloud"])      if not np.isnan(curr["cloud"])      else _last_valid(hist_c, 0.3)
    H0_aw = float(curr["humidity"])   if not np.isnan(curr["humidity"])   else _last_valid(hist_h, 0.65)
    U0_aw = float(curr["wind_u"])     if not np.isnan(curr["wind_u"])     else _last_valid(hist_u, 0.0)
    V0_aw = float(curr["wind_v"])     if not np.isnan(curr["wind_v"])     else _last_valid(hist_v, 0.0)
    W0_aw = math.sqrt(U0_aw**2 + V0_aw**2)
    DP0_aw = _last_valid(hist_dp, 0.0)     # causal: most recent valid backward-diff
    P0    = float(curr["pressure_hpa"]) if not np.isnan(curr["pressure_hpa"]) else _last_valid(hist_p, 1015.0)
    UV0   = float(curr["uv_index"])

    # Live recent-temperature rate [Phase 3, priority 1]: prefer the fresh
    # AccuWeather /historical/24 feed (fetched every cycle); fall back to
    # the (possibly stale, weekly-refreshed) ASOS-grid value, then to 0.0
    # ("no information") if neither is available.
    dtrend0 = _recent_temp_rate(df_hist24, curr["time"], T0)
    if dtrend0 is None:
        dtrend0 = _last_valid(hist_dtr, 0.0)

    # ── Step A: Fit temperature OU on ASOS data ───────────────────────────
    temp_params = fit_temp_ou(hist_c, hist_t, hist_h, hist_u, hist_v,
                               hist_s, hist_dp, hist_dew, hist_hours=hist_hours,
                               valid_mask=step_ok, hist_dtr=hist_dtr)

    # ── Step B: Fit VAR(1) on ASOS data (Change 5) ───────────────────────
    var_params = fit_var1(df_fit)

    # Current state vector for VAR(1): [cloud%, rh%, wind_u, wind_v, dpdt, dewpoint]
    x0_cloud  = C0_aw * 100.0   # 0-1 → %
    x0_rh     = H0_aw * 100.0
    x0_wind_u = U0_aw
    x0_wind_v = V0_aw
    x0_dpdt   = DP0_aw
    x0_dewpoint = dewpoint_from_t_rh(T0, H0_aw)
    x0_vec    = np.array([x0_cloud, x0_rh, x0_wind_u, x0_wind_v, x0_dpdt, x0_dewpoint])

    # ── Step C: Sample joint variable scenarios (Change 5) ────────────────
    var_samples = sample_var1(var_params, x0_vec, N=N_PATHS)
    # var_samples: dict {key: (N,) array}

    # ── Step D: Build solar/hour sub-step paths ───────────────────────────
    now_ts    = now_utc()
    N_STEPS   = int(round(1.0 / SIM_DT))
    sub_ts    = [now_ts + pd.Timedelta(hours=i * SIM_DT) for i in range(N_STEPS + 1)]
    s_path    = solar_path(sub_ts)
    hour_path = np.array([ts.hour + ts.minute / 60.0 for ts in sub_ts])

    # ── Step E: Temperature MC with per-path covariates (Change 4) ────────
    T_end, T_path_max = monte_carlo_temp_vectorised(
        T0, var_samples, s_path, temp_params,
        hour_path=hour_path, N=N_PATHS, dt=SIM_DT, dtrend0=dtrend0
    )

    # Daily-high target [Phase 1, item 1]: the settlement high is the max of
    # today's already-observed high and the highest point the simulated path
    # reaches anywhere over the forecast window — not just the value at the
    # horizon endpoint (T_end can dip below a spike that already occurred
    # earlier within the same simulated hour).
    floor   = max(obs_high, T0) if obs_high is not None else T0
    samples = np.maximum(T_path_max, floor)

    rounded      = np.floor(samples + 0.5).astype(int)
    unique, cnts = np.unique(rounded, return_counts=True)
    probs        = cnts / float(N_PATHS)

    # Forecast time label (AccuWeather horizon)
    df_next = df_fore[df_fore["time"] > now_ts].head(1).reset_index(drop=True)
    fore_time = df_next.iloc[0]["time"].strftime("%H:%M UTC") if len(df_next) > 0 else "+1h"

    # Point diagnostics using mean of var_samples for display
    C0_disp  = float(np.mean(var_samples["cloud"]))
    H0_disp  = float(np.mean(var_samples["rh"]))
    U0_disp  = float(np.mean(var_samples["wind_u"]))
    V0_disp  = float(np.mean(var_samples["wind_v"]))
    DP0_disp = float(np.mean(var_samples["dpdt"]))
    DEW0_disp = float(np.mean(var_samples["dewpoint"]))
    mu_now   = (temp_params["mu_clear"]
                - temp_params["alpha1"] * C0_disp/100.0
                - temp_params["alpha2"] * C0_disp/100.0 * S0
                - temp_params["gamma"]  * H0_disp/100.0
                - temp_params["psi_onshore"]  * max(-U0_disp, 0.0)
                + temp_params["psi_offshore"] * max(U0_disp, 0.0)
                - temp_params["delta_v"]* V0_disp
                + temp_params["phi"]    * S0
                + temp_params["kappa"]  * DP0_disp
                + temp_params["epsilon"]* DEW0_disp
                + temp_params["psi_trend"] * dtrend0
                + temp_params["theta_h"][H0_hr])

    # VAR(1) summary for display
    corr_CH  = float(var_params["Sigma"][0,1] /
                     (math.sqrt(var_params["Sigma"][0,0]) *
                      math.sqrt(var_params["Sigma"][1,1]) + 1e-12))
    corr_DPC = float(var_params["Sigma"][4,0] /
                     (math.sqrt(var_params["Sigma"][4,4]) *
                      math.sqrt(var_params["Sigma"][0,0]) + 1e-12))

    return dict(
        T0=T0, S0=S0, UV0=UV0, P0=P0, dtrend0=dtrend0,
        C0_aw=C0_aw*100, H0_aw=H0_aw*100, W0_aw=W0_aw,
        # VAR(1) samples summary
        C0_var=C0_disp, H0_var=H0_disp,
        U0_var=U0_disp, V0_var=V0_disp, DP0_var=DP0_disp,
        W0_var=math.sqrt(U0_disp**2 + V0_disp**2),
        var_samples=var_samples, var_params=var_params,
        corr_CH=corr_CH, corr_DPC=corr_DPC,
        mu_now=mu_now, fore_time=fore_time,
        n_fit=len(df_fit), fit_source=fit_source,
        obs_high=floor,
        temp_params=temp_params,
        samples=samples, temps=unique, probs=probs,
        mean=float(samples.mean()), std=float(samples.std()),
        p10=float(np.percentile(samples,10)), p25=float(np.percentile(samples,25)),
        p75=float(np.percentile(samples,75)), p90=float(np.percentile(samples,90)),
        mode=int(unique[np.argmax(probs)]),
        fetched_at=datetime.now(),
    )

# ══════════════════════════════════════════════════════════════ DISPLAY

def draw(res, iteration, next_fetch):
    secs = max(0, int(next_fetch - time.monotonic()))
    prog = int((REFRESH_SEC - secs) / REFRESH_SEC * 38)
    bar  = "X"*prog + "."*(38-prog)
    ts   = res["fetched_at"].strftime("%H:%M:%S")
    p    = res["temp_params"]
    W    = 72

    def plabel(dp):
        if dp < -1: return "FALLING"
        if dp >  1: return "RISING"
        return "steady"

    pairs = sorted(zip(res["temps"], res["probs"]), key=lambda x: -x[1])
    mp    = pairs[0][1] if pairs else 1.0
    th    = p["theta_h"]
    th_pk = int(np.argmax(th));  th_tr = int(np.argmin(th))

    L = []
    L.append("=" * W)
    L.append(f"  MIAMI KALSHI FORECAST  Combined v2  (Change 4+5)  #{iteration}")
    L.append(f"  {MIAMI_LAT}N {abs(MIAMI_LON):.4f}W  |  Fit: {res['fit_source']}")
    L.append(f"  Paths: {N_PATHS:,}  |  VAR(1) joint variables  |  Path-specific covariates")
    L.append("=" * W)
    L.append(f"  STEP 1 — Current State")
    L.append(f"    Temp     : {res['T0']:.1f} °F  (AccuWeather)   obs daily high: {res['obs_high']:.1f} °F")
    L.append(f"    Solar    : {res['S0']*100:.0f}%  (geometric)")
    L.append(f"  AccuWeather raw  →  Cloud: {res['C0_aw']:.0f}%  RH: {res['H0_aw']:.0f}%  Wind: {res['W0_aw']:.1f}mph")
    L.append(f"  VAR(1) scenario mean  →  Cloud: {res['C0_var']:.0f}%  RH: {res['H0_var']:.0f}%  Wind: {res['W0_var']:.1f}mph")
    L.append(f"    dP/dt (VAR mean): {res['DP0_var']:+.3f} hPa/hr ({plabel(res['DP0_var'])})")
    L.append(f"  VAR Sigma correlations: corr(C,RH)={res['corr_CH']:+.3f}  corr(dP,C)={res['corr_DPC']:+.3f}")
    L.append(f"    VAR(1) spectral radius: {res['var_params']['rho']:.3f}  (stable if < 1)")
    L.append(f"    mu_eq(T0): {res['mu_now']:.1f} °F  (using VAR mean state)")
    L.append("-" * W)
    L.append(f"  STEP 2 — VAR(1) variable spreads (N={N_PATHS:,} correlated paths)")
    for key in VAR_KEYS:
        samps = res["var_samples"][key]
        unit  = "%" if key in ("cloud","rh") else ("mph" if "wind" in key else "hPa/hr")
        L.append(f"    {key:<10}: mean={samps.mean():+7.2f} {unit}  std={samps.std():.2f}  "
                 f"[p10={np.percentile(samps,10):+.1f}, p90={np.percentile(samps,90):+.1f}]")
    L.append("-" * W)
    L.append(f"  STEP 3 — Temperature PDF  →  settlement at {res['fore_time']}")
    L.append(f"  Mode: {res['mode']}°F   Mean: {res['mean']:.1f}°F   Std: {res['std']:.2f}°F")
    L.append(f"  50% CI [{res['p25']:.0f}, {res['p75']:.0f}]°F    80% CI [{res['p10']:.0f}, {res['p90']:.0f}]°F")
    L.append("-" * W)
    L.append(f"  {'degF':<8} {'P(T=k)':>9}   Histogram")
    L.append("-" * W)
    for temp, prob in pairs[:20]:
        blen = int(prob / mp * 36)
        L.append(f"  {temp:<8} {prob*100:>8.2f}%   {'#'*blen}")
    L.append("-" * W)
    L.append(f"  Temp OU params  (ASOS {ASOS_DAYS}d, ridge={p['ridge_used']:.2f} [CV])")
    L.append(f"  lambda={p['lambda_']:.3f}   mu_clear={p['mu_clear']:.1f}°F")
    L.append(f"  alpha1(C)={p['alpha1']:.3f}  alpha2(C*S)={p['alpha2']:.3f}  phi(S)={p['phi']:.3f} [fixed]")
    L.append(f"  gamma(RH)={p['gamma']:.3f}  psi_onshore={p['psi_onshore']:.3f}  psi_offshore={p['psi_offshore']:.3f}  delta_v={p['delta_v']:.3f}  kappa(dP)={p['kappa']:.3f}")
    L.append(f"  psi_trend={p['psi_trend']:.3f}  dtrend0={res['dtrend0']:+.2f}°F/hr")
    L.append(f"  corr(RH,dewpoint)={p['corr_H_DEW']:+.3f}  (kept as separate regressors while |corr|≲0.5-0.6)")
    L.append(f"  sigma0={p['sigma0']:.3f}  beta={p['beta']:.3f}  eta={p['eta']:.4f}  zeta={p['zeta']:.4f}")
    L.append(f"  theta_h: warm={th_pk:02d}h ({th[th_pk]:+.2f}°F)  cool={th_tr:02d}h ({th[th_tr]:+.2f}°F)")
    L.append("-" * W)
    L.append(f"  Fetched: {ts}   Next refresh in: {secs}s")
    L.append(f"  [{bar}]")
    L.append("=" * W)

    if IN_NOTEBOOK:
        IPydisplay.clear_output(wait=True)
    else:
        print("\033c", end="")
    print("\n".join(L))

# ══════════════════════════════════════════════════════════════ MAIN LOOP

def main():
    print("=" * 72)
    print("  Miami Kalshi Forecast — Combined Pipeline  v2")
    print(f"  Change 4: path-specific weather covariates in temperature MC")
    print(f"  Change 5: joint VAR(1) replaces independent variable OUs")
    print(f"  Change 1: {ASOS_DAYS}d ASOS from Iowa Mesonet (cache: {ASOS_CACHE_PATH})")
    print(f"  {N_PATHS:,} MC paths  |  Refresh {REFRESH_SEC}s  |  ASOS re-fetch every 7d")
    print("  Ctrl-C to stop.")
    print("=" * 72)

    print(f"\n  Loading ASOS {ASOS_DAYS}d history …")
    df_asos, asos_label = get_asos_data(verbose=True)
    asos_loaded_at = time.monotonic()

    iteration  = 0
    next_fetch = 0.0
    last_res   = None
    obs_high   = None           # highest T0 seen today (Miami time)
    today_date = miami_today()  # tracks Miami date for midnight reset

    try:
        while True:
            now_m = time.monotonic()

            if df_asos is not None and (now_m - asos_loaded_at) > ASOS_REFRESH_SEC:
                print("\n  [ASOS] Weekly refresh …", flush=True)
                df_asos, asos_label = get_asos_data(verbose=True)
                asos_loaded_at = now_m

            if now_m >= next_fetch:
                iteration += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{ts}] Cycle #{iteration}", flush=True)
                try:
                    print("  [1/3] /historical/24 …", flush=True)
                    df_hist24 = fetch_historical_24h()
                    print("  [2/3] /currentconditions …", flush=True)
                    curr      = fetch_current()

                    # Track observed daily high; reset at Miami midnight
                    _today = miami_today()
                    if _today != today_date:
                        obs_high   = None
                        today_date = _today
                    obs_high = float(curr["temp_f"]) if obs_high is None else max(obs_high, float(curr["temp_f"]))

                    print("  [3/3] /forecasts/hourly …", flush=True)
                    df_fore   = fetch_forecast()

                    last_res   = compute(df_asos, asos_label, df_hist24, curr, df_fore, obs_high=obs_high)
                    next_fetch = time.monotonic() + REFRESH_SEC
                    print(
                        f"  Done — mode={last_res['mode']}°F  "
                        f"mean={last_res['mean']:.1f}°F  "
                        f"std={last_res['std']:.2f}°F",
                        flush=True,
                    )
                except Exception:
                    print("  FAILED — retrying in 60s", flush=True)
                    traceback.print_exc()
                    next_fetch = time.monotonic() + 60

            if last_res is not None:
                draw(last_res, iteration, next_fetch)

            time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopped.")


def run_once_json():
    import json, sys, os

    curr, df_fore = None, None
    _sp = os.environ.get("AW_SHARED_DATA")
    if _sp:
        try:
            with open(_sp) as _f:
                _sh = json.load(_f)
            curr = _parse_obs(_sh["curr_raw"])
            _asos = _sh.get("asos_current") or {}
            if _asos.get("t0_f") is not None:
                curr = {**curr, "temp_f": float(_asos["t0_f"])}
            curr["_asos_today_high_f"] = _asos.get("today_high_f")
            _fr = []
            for _p in _sh["fore_raw"]:
                _rh = _p.get("RelativeHumidity")
                _rv = _rh.get("Value") if isinstance(_rh, dict) else _rh
                _cc = _p.get("CloudCover")
                _fr.append(dict(
                    time=parse_aw_epoch(_p["EpochDateTime"]),
                    temp_f=float((_p.get("Temperature") or {}).get("Value", float("nan"))),
                    cloud=float(_cc)/100.0 if _cc is not None else float("nan"),
                    humidity=float(_rv)/100.0 if _rv is not None else float("nan"),
                    wind_speed=float((_p.get("Wind") or {}).get("Speed", {}).get("Value", float("nan"))),
                    wind_dir=float((_p.get("Wind") or {}).get("Direction", {}).get("Degrees", float("nan"))),
                ))
            df_fore = pd.DataFrame(_fr)
        except Exception:
            curr, df_fore = None, None
    if curr is None:
        curr = fetch_current()
    if df_fore is None:
        df_fore = fetch_forecast()

    try:
        df_asos, asos_label = get_asos_data(verbose=False)
    except Exception:
        df_asos, asos_label = None, "unavailable"

    df_hist24 = (None if df_asos is not None and len(df_asos) >= ASOS_MIN_OBS
                 else fetch_historical_24h())

    res = compute(df_asos, asos_label, df_hist24, curr, df_fore,
                  obs_high=curr.pop("_asos_today_high_f", None))
    out = {
        "model":      "var_1h",
        "T0":         float(res["T0"]),
        "mean":       float(res["mean"]),
        "std":        float(res["std"]),
        "mode":       int(res["mode"]),
        "p10":        float(res["p10"]),
        "p25":        float(res["p25"]),
        "p75":        float(res["p75"]),
        "p90":        float(res["p90"]),
        "temps":      [int(t) for t in res["temps"].tolist()],
        "probs":      [float(p) for p in res["probs"].tolist()],
        "fit_source": str(res["fit_source"]),
        "fetched_at": res["fetched_at"].isoformat(),
    }
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    import sys
    if "--json-output" in sys.argv:
        run_once_json()
    else:
        main()
