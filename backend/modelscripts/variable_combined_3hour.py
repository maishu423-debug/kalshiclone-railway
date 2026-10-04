# -*- coding: utf-8 -*-
"""
Miami Temperature Forecast — Combined Pipeline  v3
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
HORIZON_HOURS = 3        # forecast horizon in hours
PHI_FIXED    = 5.0       # fixed solar coefficient
LAMBDA_RIDGE = 0.5       # ridge penalty on temp OU

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
    """Return current date in Miami local time (resets at midnight ET)."""
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
        # Derive wind_u / wind_v if the cache was written by an older version
        # that didn't include these columns.
        if "wind_u" not in df.columns or "wind_v" not in df.columns:
            spd = df["wind_speed"].fillna(0.0).values.astype(float)
            drc = df["wind_dir"].fillna(180.0).values.astype(float)
            pairs = [uv_components(s, d) for s, d in zip(spd, drc)]
            df["wind_u"] = [p[0] for p in pairs]
            df["wind_v"] = [p[1] for p in pairs]
            # Overwrite cache so this only runs once
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

# ══════════════════════════════════════════════════════════════ JOINT VAR(1) MODEL (Change 5)

def _build_var_matrix(df):
    """
    Build (n, 6) matrix X from ASOS DataFrame.
    Columns: cloud%, rh%, wind_u_mph, wind_v_mph, dpdt_hPa/hr, dewpoint_F

    dpdt computed as gradient of pressure with dt=1h between hourly obs.
    Rows with any NaN dropped.
    """
    cloud  = df["cloud"].values.astype(float) * 100.0        # 0-1 → %
    rh     = df["humidity"].values.astype(float) * 100.0     # 0-1 → %
    wind_u = df["wind_u"].values.astype(float)
    wind_v = df["wind_v"].values.astype(float)
    pres   = _fill(df["pressure_hpa"].values.astype(float), 1015.0)
    dpdt   = np.gradient(pres)   # hPa per hour (hourly ASOS obs spaced ~1h)
    dpdt   = np.clip(dpdt, -10.0, 10.0)
    dewpt  = df["dewpoint_f"].values.astype(float)

    X = np.column_stack([cloud, rh, wind_u, wind_v, dpdt, dewpt])
    # Drop rows with any NaN in cloud or rh
    valid = ~np.any(np.isnan(X), axis=1)
    return X[valid]


def fit_var1(df_asos):
    """
    Fit VAR(1) on ASOS data: X_{t+1} = a + B*X_t + eps_t
    Returns dict with:
      a     : intercept vector (k,)
      B     : transition matrix (k,k)
      Sigma : residual covariance (k,k) — captures cross-variable correlations
      mu    : unconditional mean of each variable (k,)
      k     : number of variables
    """
    X  = _build_var_matrix(df_asos)
    n  = len(X)
    k  = X.shape[1]

    Xt  = X[:-1]   # predictors: X_1 .. X_{n-1}
    Xt1 = X[1:]    # targets:    X_2 .. X_n

    # OLS: [a | B.T] = (A'A)^{-1} A' Y   where A = [1 | Xt]
    A    = np.column_stack([np.ones(n - 1), Xt])   # (n-1, k+1)
    ATA  = A.T @ A
    ATY  = A.T @ Xt1                                # (k+1, k)
    coef = np.linalg.solve(ATA, ATY)                # (k+1, k)

    a  = coef[0]        # (k,)  intercept
    B  = coef[1:].T     # (k,k) each row = coefficients for one output variable

    resid = Xt1 - (A @ coef)          # (n-1, k)
    Sigma = (resid.T @ resid) / (n - 2)

    # Regularise Sigma: small diagonal nudge for numerical stability
    Sigma += 1e-6 * np.eye(k)

    # Unconditional mean: mu = (I - B)^{-1} a  (if VAR(1) is stationary)
    try:
        mu = np.linalg.solve(np.eye(k) - B, a)
    except np.linalg.LinAlgError:
        mu = np.mean(X, axis=0)

    return dict(a=a, B=B, Sigma=Sigma, mu=mu, k=k,
                n_obs=n, var_keys=VAR_KEYS)


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

def _fill_arr(arr, fallback):
    arr = arr.copy().astype(float)
    if np.isnan(arr).all(): return np.full_like(arr, fallback)
    return np.where(np.isnan(arr), float(np.nanmean(arr)), arr)


def fit_temp_ou(hist_c, hist_t, hist_h, hist_u, hist_v, hist_s, hist_dp, hist_dew,
                hist_hours=None, dt=FIT_DT):
    """
    7-step temperature OU fit with dewpoint added alongside the original
    covariates. hist_dew: dewpoint F array, same length as hist_t.
    """
    dT = np.diff(hist_t)
    C  = hist_c[:-1];  T  = hist_t[:-1]
    H  = hist_h[:-1];  U  = hist_u[:-1];  V  = hist_v[:-1]
    S  = hist_s[:-1];  DP = hist_dp[:-1]; DEW = hist_dew[:-1]
    CS = C * S

    hours_lag = (hist_hours[:-1].astype(int) % 24
                 if hist_hours is not None and len(hist_hours) == len(hist_t)
                 else np.zeros(len(T), dtype=int))

    def cov(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9: return 0.0
        return float(np.cov(a, b)[0, 1])

    lam = float(np.clip(-cov(dT, T) / (np.var(T) * dt + 1e-9), 0.04, 0.70))
    sc  = lam * dt + 1e-9
    phi = PHI_FIXED

    X    = np.column_stack([-C, -CS, -H, -U, -V, DP, DEW])
    Y    = dT / sc + T - phi * S
    Xc   = X - X.mean(axis=0);  Yc = Y - Y.mean()
    k    = Xc.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + LAMBDA_RIDGE * np.eye(k), Xc.T @ Yc)

    alpha1  = float(np.clip(beta[0],  0.0, 15.0))
    alpha2  = float(np.clip(beta[1],  0.0, 10.0))
    gamma   = float(np.clip(beta[2],  0.0, 10.0))
    delta_u = float(np.clip(beta[3], -5.0,  5.0))
    delta_v = float(np.clip(beta[4], -5.0,  5.0))
    kappa   = float(np.clip(beta[5], -3.0,  3.0))
    epsilon = float(np.clip(beta[6], -3.0,  3.0))

    # Hour-of-day shifts
    mu_pre = (np.mean(T)
              + alpha1*np.mean(C) + alpha2*np.mean(CS) + gamma*np.mean(H)
              + delta_u*np.mean(U) + delta_v*np.mean(V)
              - phi*np.mean(S) - kappa*np.mean(DP) - epsilon*np.mean(DEW))
    mu_vec_pre = (mu_pre - alpha1*C - alpha2*CS - gamma*H
                  - delta_u*U - delta_v*V + phi*S + kappa*DP + epsilon*DEW)
    res_pre = dT - lam*(mu_vec_pre - T)*dt

    theta_h = np.zeros(24)
    for h in range(24):
        mask = hours_lag == h
        if mask.sum() >= 2:
            theta_h[h] = float(np.mean(res_pre[mask]) / (lam*dt + 1e-9))
    theta_h -= theta_h.mean()

    mu = float(np.mean(T)
               + alpha1*np.mean(C) + alpha2*np.mean(CS)
               + gamma*np.mean(H)
               + delta_u*np.mean(U) + delta_v*np.mean(V)
               - phi*np.mean(S) - kappa*np.mean(DP) - epsilon*np.mean(DEW))

    th_vec  = np.array([theta_h[h] for h in hours_lag])
    mu_full = (mu + th_vec - alpha1*C - alpha2*CS - gamma*H
               - delta_u*U - delta_v*V + phi*S + kappa*DP + epsilon*DEW)
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
        gamma=gamma, delta_u=delta_u, delta_v=delta_v,
        phi=phi, kappa=kappa, epsilon=epsilon, theta_h=theta_h,
        sigma0=float(np.clip(s0, 0.05, 3.0)),
        beta=float(np.clip(s1-s0, 0.0, 2.0)),
        eta=eta, zeta=zeta,
    )


def monte_carlo_temp_vectorised(T0, var_samples, s_path, p,
                                 hour_path=None, N=N_PATHS, dt=SIM_DT):
    """
    Temperature MC with path-specific weather variables (Change 4).

    var_samples : dict {key: np.array(N,)} — one value per path per variable
                  keys: cloud (%), rh (%), wind_u, wind_v, dpdt, dewpoint
    s_path      : np.array(N_STEPS+1,) — deterministic solar per sub-step
    hour_path   : np.array(N_STEPS+1,) — fractional hour per sub-step

    The key difference from v1: C, H, U, V, DP are (N,) arrays, not scalars.
    mu and sigma are computed per-path, not shared across paths.
    """
    T      = np.full(N, float(T0), dtype=np.float64)
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

    lam = p["lambda_"]

    for i in range(N_STEPS):
        S  = float(s_path[i])
        hr = float(hour_path[i]) if hour_path is not None else None

        # Per-path equilibrium (vectorised — all N paths simultaneously)
        theta = p["theta_h"][int(hr) % 24] if hr is not None else 0.0
        mu_vec = (p["mu_clear"]
                  - p["alpha1"] * C_arr
                  - p["alpha2"] * C_arr * S
                  - p["gamma"]  * H_arr
                  - p["delta_u"]* U_arr
                  - p["delta_v"]* V_arr
                  + p["phi"]    * S
                  + p["kappa"]  * DP_arr
                  + p["epsilon"]* DEW_arr
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

    return T

# ══════════════════════════════════════════════════════════════ COMPUTE

def compute(df_asos, asos_label, df_hist24, curr, df_fore, obs_high=None):
    """
    Full combined pipeline with Change 4 + Change 5.

    df_asos   : 90d ASOS DataFrame (or None → fallback to df_hist24)
    df_hist24 : AccuWeather /historical/24 (fallback)
    curr      : parsed current-conditions dict
    df_fore   : AccuWeather hourly forecast DataFrame
    obs_high  : observed daily high so far (°F). MC paths floored to max(obs_high, T0).
    """
    # ── Choose fitting dataset ────────────────────────────────────────────
    if df_asos is not None and len(df_asos) >= ASOS_MIN_OBS:
        df_fit     = df_asos
        fit_source = asos_label
    else:
        df_fit     = df_hist24.tail(24).reset_index(drop=True)
        fit_source = f"AW /historical/24 ({len(df_hist24)} obs)"

    hist_t  = df_fit["temp_f"].values.astype(float)
    hist_c  = _fill_arr(df_fit["cloud"].values.astype(float),        0.3)
    hist_h  = _fill_arr(df_fit["humidity"].values.astype(float),     0.65)
    hist_ws = _fill_arr(df_fit["wind_speed"].values.astype(float),   5.0)
    hist_wd = _fill_arr(df_fit["wind_dir"].values.astype(float),     180.0)
    hist_u  = _fill_arr(df_fit["wind_u"].values.astype(float),       0.0)
    hist_v  = _fill_arr(df_fit["wind_v"].values.astype(float),       0.0)
    hist_p  = _fill_arr(df_fit["pressure_hpa"].values.astype(float), 1015.0)
    hist_s  = solar_path(list(df_fit["time"]))
    hist_dp = np.clip(np.gradient(hist_p) / FIT_DT, -6.0, 6.0)
    if "dewpoint_f" in df_fit.columns:
        hist_dew = _fill_arr(df_fit["dewpoint_f"].values.astype(float), 70.0)
    else:
        hist_dew = np.array(
            [dewpoint_from_t_rh(t, h) for t, h in zip(hist_t, hist_h)],
            dtype=float,
        )
        df_fit = df_fit.copy()
        df_fit["dewpoint_f"] = hist_dew
    hist_hours = np.array([ts.hour for ts in df_fit["time"]], dtype=float)

    # ── Current state from AccuWeather (T0, solar, fallback vars) ─────────
    T0    = float(curr["temp_f"])
    S0    = geometric_solar(curr["time"])
    H0_hr = curr["time"].hour
    C0_aw = float(curr["cloud"])      if not np.isnan(curr["cloud"])      else float(hist_c[-1])
    H0_aw = float(curr["humidity"])   if not np.isnan(curr["humidity"])   else float(hist_h[-1])
    U0_aw = float(curr["wind_u"])     if not np.isnan(curr["wind_u"])     else float(hist_u[-1])
    V0_aw = float(curr["wind_v"])     if not np.isnan(curr["wind_v"])     else float(hist_v[-1])
    W0_aw = math.sqrt(U0_aw**2 + V0_aw**2)
    DP0_aw = float(hist_dp[-1])
    P0    = float(curr["pressure_hpa"]) if not np.isnan(curr["pressure_hpa"]) else float(hist_p[-1])
    UV0   = float(curr["uv_index"])

    # ── Step A: Fit temperature OU on ASOS data ───────────────────────────
    temp_params = fit_temp_ou(hist_c, hist_t, hist_h, hist_u, hist_v,
                               hist_s, hist_dp, hist_dew, hist_hours=hist_hours)

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
    # For HORIZON_HOURS > 1 we chain VAR(1) steps:
    #   x0 → (one step) → x1_mean + noise → (one more step) → x2
    # Each chained step adds independent correlated noise, so the final
    # distribution is wider than a single step — correct behaviour.
    var_samples = sample_var1(var_params, x0_vec, N=N_PATHS)
    if HORIZON_HOURS >= 2:
        # Chain a second VAR step using the per-path x1 draws
        X1 = np.column_stack([var_samples[k] for k in VAR_KEYS])  # (N, k_v)
        a, B, Sigma = var_params["a"], var_params["B"], var_params["Sigma"]
        k_v = var_params["k"]
        x2_mean = (a + X1 @ B.T)   # (N, k_v)
        try:
            eps2 = np.random.multivariate_normal(np.zeros(k_v), Sigma, size=N_PATHS)
        except np.linalg.LinAlgError:
            eps2 = np.random.randn(N_PATHS, k_v) * np.sqrt(np.diag(Sigma))
        X2 = x2_mean + eps2   # (N, k_v)
        for i, key in enumerate(VAR_KEYS):
            lo, hi = VAR_CLIPS[key]
            var_samples[key] = np.clip(X2[:, i], lo, hi)
    if HORIZON_HOURS >= 3:
        # Chain a third VAR step using the per-path x2 draws
        X2 = np.column_stack([var_samples[k] for k in VAR_KEYS])  # (N, k_v)
        x3_mean = (a + X2 @ B.T)   # (N, k_v)
        try:
            eps3 = np.random.multivariate_normal(np.zeros(k_v), Sigma, size=N_PATHS)
        except np.linalg.LinAlgError:
            eps3 = np.random.randn(N_PATHS, k_v) * np.sqrt(np.diag(Sigma))
        X3 = x3_mean + eps3   # (N, k_v)
        for i, key in enumerate(VAR_KEYS):
            lo, hi = VAR_CLIPS[key]
            var_samples[key] = np.clip(X3[:, i], lo, hi)

    # ── Step D: Build solar/hour sub-step paths over HORIZON_HOURS ────────
    # Piecewise-linear waypoint walk through AccuWeather forecast periods.
    # Waypoints: [current, +1h forecast, +2h forecast, ...]
    now_ts    = now_utc()
    df_future = df_fore[df_fore["time"] > now_ts].head(HORIZON_HOURS).reset_index(drop=True)

    wp_c, wp_h, wp_u, wp_v = [C0_aw], [H0_aw], [U0_aw], [V0_aw]
    fore_time = f"+{HORIZON_HOURS}h"

    for i in range(HORIZON_HOURS):
        if i < len(df_future):
            fr  = df_future.iloc[i]
            fc  = float(fr["cloud"])      if not np.isnan(fr["cloud"])      else wp_c[-1]
            fh  = float(fr["humidity"])   if not np.isnan(fr["humidity"])   else wp_h[-1]
            fws = float(fr["wind_speed"]) if not np.isnan(fr["wind_speed"]) else math.sqrt(wp_u[-1]**2+wp_v[-1]**2)
            fwd = (float(fr["wind_dir"])  if not np.isnan(fr["wind_dir"])
                   else math.degrees(math.atan2(-wp_u[-1], -wp_v[-1])) % 360)
            fu, fv = uv_components(fws, fwd)
            if i == HORIZON_HOURS - 1:
                fore_time = fr["time"].strftime("%H:%M UTC")
        else:
            fc, fh, fu, fv = wp_c[-1], wp_h[-1], wp_u[-1], wp_v[-1]
        wp_c.append(fc); wp_h.append(fh); wp_u.append(fu); wp_v.append(fv)

    STEPS_PER_HOUR = int(round(1.0 / SIM_DT))
    N_STEPS_TOTAL  = HORIZON_HOURS * STEPS_PER_HOUR

    # Temperature OU uses AW scalar path for solar/hour; VAR vars are per-path arrays
    sub_ts    = [now_ts + pd.Timedelta(hours=i * SIM_DT) for i in range(N_STEPS_TOTAL + 1)]
    s_path    = solar_path(sub_ts)
    hour_path = np.array([ts.hour + ts.minute / 60.0 for ts in sub_ts])

    # ── Step E: Temperature MC with per-path covariates (Change 4) ────────
    samples = monte_carlo_temp_vectorised(
        T0, var_samples, s_path, temp_params,
        hour_path=hour_path, N=N_PATHS, dt=SIM_DT
    )

    # Hard lower bound: daily high can never fall below what has already been observed.
    floor    = max(obs_high, T0) if obs_high is not None else T0
    samples  = np.maximum(samples, floor)

    rounded      = np.floor(samples + 0.5).astype(int)
    unique, cnts = np.unique(rounded, return_counts=True)
    probs        = cnts / float(N_PATHS)

    # Forecast time label (AccuWeather horizon)
    # (fore_time already set in waypoint loop above)

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
                - temp_params["delta_u"]* U0_disp
                - temp_params["delta_v"]* V0_disp
                + temp_params["phi"]    * S0
                + temp_params["kappa"]  * DP0_disp
                + temp_params["epsilon"]* DEW0_disp
                + temp_params["theta_h"][H0_hr])

    # VAR(1) summary for display
    corr_CH  = float(var_params["Sigma"][0,1] /
                     (math.sqrt(var_params["Sigma"][0,0]) *
                      math.sqrt(var_params["Sigma"][1,1]) + 1e-12))
    corr_DPC = float(var_params["Sigma"][4,0] /
                     (math.sqrt(var_params["Sigma"][4,4]) *
                      math.sqrt(var_params["Sigma"][0,0]) + 1e-12))

    return dict(
        T0=T0, S0=S0, UV0=UV0, P0=P0,
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
    L.append(f"  MIAMI KALSHI FORECAST  Combined v3  (Change 4+5)  #{iteration}")
    L.append(f"  {MIAMI_LAT}N {abs(MIAMI_LON):.4f}W  |  Fit: {res['fit_source']}")
    L.append(f"  Paths: {N_PATHS:,}  |  Horizon: +{HORIZON_HOURS}h  |  VAR(1) joint variables")
    L.append("=" * W)
    L.append(f"  STEP 1 — Current State")
    L.append(f"    Temp     : {res['T0']:.1f} °F  (AccuWeather)   obs daily high: {res['obs_high']:.1f} °F")
    L.append(f"    Solar    : {res['S0']*100:.0f}%  (geometric)")
    L.append(f"  AccuWeather raw  →  Cloud: {res['C0_aw']:.0f}%  RH: {res['H0_aw']:.0f}%  Wind: {res['W0_aw']:.1f}mph")
    L.append(f"  VAR(1) scenario mean  →  Cloud: {res['C0_var']:.0f}%  RH: {res['H0_var']:.0f}%  Wind: {res['W0_var']:.1f}mph")
    L.append(f"    dP/dt (VAR mean): {res['DP0_var']:+.3f} hPa/hr ({plabel(res['DP0_var'])})")
    L.append(f"  VAR Sigma correlations: corr(C,RH)={res['corr_CH']:+.3f}  corr(dP,C)={res['corr_DPC']:+.3f}")
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
    L.append(f"  Temp OU params  (ASOS {ASOS_DAYS}d, ridge={LAMBDA_RIDGE})")
    L.append(f"  lambda={p['lambda_']:.3f}   mu_clear={p['mu_clear']:.1f}°F")
    L.append(f"  alpha1(C)={p['alpha1']:.3f}  alpha2(C*S)={p['alpha2']:.3f}  phi(S)={p['phi']:.3f} [fixed]")
    L.append(f"  gamma(RH)={p['gamma']:.3f}  delta_u={p['delta_u']:.3f}  delta_v={p['delta_v']:.3f}  kappa(dP)={p['kappa']:.3f}")
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
    print("  Miami Kalshi Forecast — Combined Pipeline  v3")
    print(f"  Change 4: path-specific weather covariates in temperature MC")
    print(f"  Change 5: joint VAR(1) replaces independent variable OUs")
    print(f"  Change 1: {ASOS_DAYS}d ASOS from Iowa Mesonet (cache: {ASOS_CACHE_PATH})")
    print(f"  Horizon: +{HORIZON_HOURS}h  |  {N_PATHS:,} MC paths  |  Refresh {REFRESH_SEC}s")
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

            # Reset obs_high at Miami midnight
            _today = miami_today()
            if _today != today_date:
                obs_high   = None
                today_date = _today

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
                    obs_high  = float(curr["temp_f"]) if obs_high is None else max(obs_high, float(curr["temp_f"]))
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
        "model":      "var_3h",
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
