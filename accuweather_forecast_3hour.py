# -*- coding: utf-8 -*-
"""
Miami Temperature Forecast — AccuWeather  v7
============================================
Target   : Kalshi temperature markets settled on NWS ASOS observation at KMIA
Location : Miami Intl Airport, FL  (lat 25.793, lon -80.291)

Data sources per 15-min cycle:

  • ASOS history (Change 1 — parameter fitting)
      Iowa State Mesonet  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
      Station: MIA  |  90 days of hourly obs  |  the exact Kalshi settlement source
      Cached locally to kmia_asos_90d.csv; re-fetched once per week.
      Fallback: AccuWeather /historical/24 if Mesonet unavailable.

  • AccuWeather /currentconditions  — T0 (current state for simulation)
  • AccuWeather /forecasts/hourly/12hour — covariate path for Monte Carlo

Solar radiation:
  Computed geometrically from time-of-day, lat/lon, day-of-year.
  Clear-sky irradiance in [0, 1].

Model (Ornstein-Uhlenbeck, extended — Change 6: time-of-day regimes):
  dT = -lambda*(T - mu(C,H,U,V,S,DP,h)) dt + sigma(C,W,DP,S) dW_t

  mu = mu_clear
       - alpha1*C           (base cloud cooling)
       - alpha2*(C*S)       (cloud×solar interaction: clouds hurt more at noon)
       - gamma*H
       - delta_u*U - delta_v*V
       + phi*S
       + kappa*DP
       + theta_h[h]         (hour-of-day intercept shift, 24 dummies, mean-centred)

  sigma = (sigma0 + beta*C + eta*W + zeta*|DP|) * (1.15 if S>0.1 else 1.0)
          daytime convective hours get 15% wider spread

Fitting strategy (6-step):
  1. lambda via AR(1) on T alone.
  2. phi fixed at PHI_FIXED (breaks C/S collinearity; valid with 90d of data).
  3. Ridge OLS on (C, C*S, H, U, V, DP), demeaned.
  4. Hour-of-day shifts theta_h[0..23] from binned residuals, mean-centred.
  5. mu_clear from equilibrium identity anchored to mean(T).
  6. Noise terms (sigma0, beta, eta, zeta) from final residuals.

With 90 days (~2160 obs) each hour bin has ~90 observations — enough for
stable theta_h and the C*S interaction term.

Output:
  Discrete PDF  P(T=k degF)  over the +3h settlement temperature,
  10,000 Monte Carlo paths, refreshed every 15 minutes.

Usage:
  export ACCUWEATHER_API_KEY="your_key_here"
  python accuweather_forecast.py
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


MIAMI_LAT = 25.793
MIAMI_LON = -80.291

AW_BASE       = "https://dataservice.accuweather.com"
AW_CURR_URL   = f"{AW_BASE}/currentconditions/v1/{{key}}"
AW_HIST24_URL = f"{AW_BASE}/currentconditions/v1/{{key}}/historical/24"
AW_FCST_URL   = f"{AW_BASE}/forecasts/v1/hourly/12hour/{{key}}"

N_PATHS      = 10_000
REFRESH_SEC  = 900        # 15-min forecast cycle
FIT_DT       = 1.0        # OU fitting dt (hours) — ASOS obs are hourly
SIM_DT       = 0.25       # Monte Carlo sub-step (hours)
HORIZON_HOURS = 3         # forecast horizon in hours
PHI_FIXED    = 5.0        # fixed solar coeff (breaks C/S collinearity)
LAMBDA_RIDGE = 0.5        # ridge penalty

# ── Change 1: ASOS long-history config ───────────────────────────────────────
ASOS_STATION     = "MIA"                  # Iowa Mesonet station ID for KMIA
ASOS_CACHE_PATH  = "kmia_asos_90d.csv"   # local cache file
ASOS_DAYS        = 90                     # days of history to fetch
ASOS_REFRESH_SEC = 7 * 24 * 3600         # re-fetch weekly (in seconds)
ASOS_MIN_OBS     = 48                     # min rows before using ASOS over AW 24h

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

# ══════════════════════════════════════════════════════════════ ASOS FETCH (Change 1)

# Iowa Mesonet sky-condition codes → cloud fraction
_SKYC_MAP = {
    "CLR": 0.00, "SKC": 0.00, "NSC": 0.00,
    "FEW": 0.20,
    "SCT": 0.45,
    "BKN": 0.75,
    "OVC": 1.00,
    "VV":  1.00,   # vertical visibility (fog/obscured)
}

def _parse_skyc(skyc):
    """Convert Iowa Mesonet sky condition string to cloud fraction [0,1]."""
    if not skyc or skyc == "M":
        return np.nan
    return _SKYC_MAP.get(str(skyc).strip().upper()[:3], 0.5)


def fetch_asos_history(days=ASOS_DAYS, station=ASOS_STATION):
    """
    Fetch hourly ASOS obs from Iowa State Mesonet for the last `days` days.
    Returns a DataFrame with columns matching the standard fitting schema:
      time, temp_f, cloud, humidity, wind_speed, wind_dir, pressure_hpa, uv_index

    uv_index is not in ASOS — set to 0.0 (only used for display).
    Cloud comes from sky condition codes; pressure from station pressure
    converted to sea-level equivalent via a simple barometric formula.

    Mesonet returns report_type=3 (METAR routine hourly obs).
    Wind speed in knots → converted to mph (×1.15078).
    """
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    url    = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station":     station,
        "data":        "tmpf,relh,sknt,drct,mslp,skyc1,skyc2",
        "year1":       start_dt.year,  "month1": start_dt.month,  "day1": start_dt.day,
        "year2":       end_dt.year,    "month2": end_dt.month,    "day2": end_dt.day,
        "tz":          "UTC",
        "format":      "onlycomma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "3",   # routine hourly METAR
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    lines = [l for l in resp.text.strip().splitlines() if not l.startswith("#")]
    if len(lines) < 2:
        raise ValueError("ASOS returned no data rows.")

    # Header: station,valid,tmpf,relh,sknt,drct,mslp,skyc1,skyc2
    header = [h.strip() for h in lines[0].split(",")]
    rows   = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, [p.strip() for p in parts]))

        def fval(k, fallback=np.nan):
            v = row.get(k, "M")
            if v in ("M", "T", "", None):
                return fallback
            try:
                return float(v)
            except ValueError:
                return fallback

        temp_f     = fval("tmpf")
        rh         = fval("relh")
        wind_kts   = fval("sknt", 0.0)
        wind_mph   = wind_kts * 1.15078
        wind_dir   = fval("drct", np.nan)
        mslp_hpa   = fval("mslp", np.nan)          # already hPa in Mesonet

        # Cloud: prefer skyc2 (upper layer) if available, else skyc1
        skyc = row.get("skyc2", "M")
        if not skyc or skyc == "M":
            skyc = row.get("skyc1", "M")
        cloud = _parse_skyc(skyc)

        try:
            ts = pd.Timestamp(row["valid"])
        except Exception:
            continue

        if np.isnan(temp_f):
            continue   # temperature is mandatory

        rows.append(dict(
            time         = ts,
            temp_f       = temp_f,
            cloud        = cloud if not np.isnan(cloud) else 0.3,
            humidity     = rh / 100.0 if not np.isnan(rh) else np.nan,
            wind_speed   = wind_mph,
            wind_dir     = wind_dir,
            pressure_hpa = mslp_hpa,
            uv_index     = 0.0,
        ))

    if not rows:
        raise ValueError("ASOS parse produced no valid rows.")

    df = (pd.DataFrame(rows)
            .sort_values("time")
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True))
    return df


def load_asos_cache():
    """Load cached ASOS CSV if it exists and is fresh enough."""
    if not os.path.exists(ASOS_CACHE_PATH):
        return None, False
    age_sec = time.time() - os.path.getmtime(ASOS_CACHE_PATH)
    if age_sec > ASOS_REFRESH_SEC:
        return None, True   # stale — signal caller to re-fetch
    try:
        df = pd.read_csv(ASOS_CACHE_PATH, parse_dates=["time"])
        if len(df) >= ASOS_MIN_OBS:
            return df, False
    except Exception:
        pass
    return None, False


def get_asos_fit_data(verbose=True):
    """
    Return (df_asos, source_label).
    Priority:
      1. Fresh local cache (< ASOS_REFRESH_SEC old)
      2. Live fetch from Iowa Mesonet → save to cache
      3. Return None if both fail (caller falls back to AW /historical/24)
    """
    df, stale = load_asos_cache()
    if df is not None:
        label = f"ASOS cache ({len(df)} obs, {ASOS_DAYS}d)"
        if verbose:
            print(f"  [ASOS] Using local cache: {ASOS_CACHE_PATH}  ({len(df)} obs)")
        return df, label

    reason = "stale" if stale else "missing"
    if verbose:
        print(f"  [ASOS] Cache {reason} — fetching {ASOS_DAYS}d from Iowa Mesonet …", flush=True)
    try:
        df = fetch_asos_history()
        df.to_csv(ASOS_CACHE_PATH, index=False)
        label = f"ASOS live ({len(df)} obs, {ASOS_DAYS}d)"
        if verbose:
            print(f"  [ASOS] Fetched {len(df)} obs → saved to {ASOS_CACHE_PATH}")
        return df, label
    except Exception as exc:
        if verbose:
            print(f"  [ASOS] FAILED ({exc}) — will fall back to AW /historical/24")
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
    """Parse one AccuWeather current-conditions object (details=true)."""
    temp_f = float((obs.get("Temperature") or {})
                   .get("Imperial", {}).get("Value", np.nan))
    cloud_pct = obs.get("CloudCover")
    cloud     = (float(cloud_pct) / 100.0 if cloud_pct is not None
                 else _CLOUD_ICON_MAP.get(int(obs.get("WeatherIcon", 1)), 0.5))
    rh_raw   = obs.get("RelativeHumidity")
    humidity = float(rh_raw) / 100.0 if rh_raw is not None else np.nan
    wind_speed = float((obs.get("Wind") or {})
                       .get("Speed", {}).get("Imperial", {}).get("Value", np.nan))
    wind_dir   = float((obs.get("Wind") or {})
                       .get("Direction", {}).get("Degrees", np.nan))
    pres_inhg    = (obs.get("Pressure") or {}).get("Imperial", {}).get("Value")
    pressure_hpa = float(pres_inhg) * 33.8639 if pres_inhg is not None else np.nan
    uv_index = float(obs.get("UVIndex", 0.0))
    return dict(
        time         = parse_aw_epoch(obs["EpochTime"]),
        temp_f       = temp_f,
        cloud        = cloud,
        humidity     = humidity,
        wind_speed   = wind_speed,
        wind_dir     = wind_dir,
        pressure_hpa = pressure_hpa,
        uv_index     = uv_index,
    )


def resolve_location_key():
    return "3593859"   # Miami Intl Airport


def fetch_historical_24h(key):
    resp = requests.get(
        AW_HIST24_URL.format(key=key),
        params={"apikey": API_KEY, "details": "true", "language": "en-us"},
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise ValueError("/historical/24 returned empty list.")
    rows = [_parse_obs(obs) for obs in raw]
    df   = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


def fetch_current(key):
    resp = requests.get(
        AW_CURR_URL.format(key=key),
        params={"apikey": API_KEY, "details": "true", "language": "en-us"},
        timeout=15,
    )
    resp.raise_for_status()
    obs_list = resp.json()
    if not obs_list:
        raise ValueError("Current conditions returned empty list.")
    return _parse_obs(obs_list[0])


def fetch_forecast(key):
    resp = requests.get(
        AW_FCST_URL.format(key=key),
        params={"apikey": API_KEY, "details": "true",
                "metric": "false", "language": "en-us"},
        timeout=15,
    )
    resp.raise_for_status()
    rows = []
    for p in resp.json()[:6]:
        rh     = p.get("RelativeHumidity")
        rh_val = rh.get("Value") if isinstance(rh, dict) else rh
        cc     = p.get("CloudCover")
        rows.append(dict(
            time       = parse_aw_epoch(p["EpochDateTime"]),
            temp_f     = float((p.get("Temperature") or {}).get("Value", np.nan)),
            cloud      = float(cc) / 100.0 if cc is not None else np.nan,
            humidity   = float(rh_val) / 100.0 if rh_val is not None else np.nan,
            wind_speed = float((p.get("Wind") or {}).get("Speed",     {}).get("Value",   np.nan)),
            wind_dir   = float((p.get("Wind") or {}).get("Direction", {}).get("Degrees", np.nan)),
        ))
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════ ARRAY UTILS

def _fill(arr, fallback):
    arr = arr.copy().astype(float)
    if np.isnan(arr).all(): return np.full_like(arr, fallback)
    return np.where(np.isnan(arr), float(np.nanmean(arr)), arr)

# ══════════════════════════════════════════════════════════════ OU MODEL (v6)

def fit_ou(hist_c, hist_t, hist_h, hist_u, hist_v, hist_s, hist_dp,
           hist_hours=None, dt=FIT_DT):
    """
    Six-step OU fit with time-of-day regimes (Change 6).

    hist_hours : array of UTC hour floats, same length as hist_t.
                 If None, theta_h defaults to zero (no regime effect).

    New vs v4:
      • alpha1 (base cloud) + alpha2 (cloud×solar interaction C*S)
      • theta_h[0..23]: hour-of-day intercept shifts, mean-centred
      • sigma multiplied by 1.15 during daytime (in sigma_eq)
    """
    dT = np.diff(hist_t)
    C  = hist_c[:-1];  T  = hist_t[:-1]
    H  = hist_h[:-1];  U  = hist_u[:-1];  V  = hist_v[:-1]
    S  = hist_s[:-1];  DP = hist_dp[:-1]
    CS = C * S   # cloud×solar interaction

    hours_lag = (hist_hours[:-1].astype(int) % 24
                 if hist_hours is not None and len(hist_hours) == len(hist_t)
                 else np.zeros(len(T), dtype=int))

    def cov(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9: return 0.0
        return float(np.cov(a, b)[0, 1])

    # ── Step 1: lambda ────────────────────────────────────────────────────
    lam = float(np.clip(-cov(dT, T) / (np.var(T) * dt + 1e-9), 0.04, 0.70))
    sc  = lam * dt + 1e-9

    # ── Step 2: phi fixed ────────────────────────────────────────────────
    phi = PHI_FIXED

    # ── Step 3: ridge OLS — (C, C*S, H, U, V, DP) ────────────────────────
    X  = np.column_stack([-C, -CS, -H, -U, -V, DP])
    Y  = dT / sc + T - phi * S
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean()
    k  = Xc.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + LAMBDA_RIDGE * np.eye(k), Xc.T @ Yc)

    alpha1  = float(np.clip(beta[0],  0.0, 15.0))
    alpha2  = float(np.clip(beta[1],  0.0, 10.0))
    gamma   = float(np.clip(beta[2],  0.0, 10.0))
    delta_u = float(np.clip(beta[3], -5.0,  5.0))
    delta_v = float(np.clip(beta[4], -5.0,  5.0))
    kappa   = float(np.clip(beta[5], -3.0,  3.0))

    # ── Step 4: hour-of-day intercept shifts ─────────────────────────────
    # Residuals from a preliminary mu anchored at mean(T)
    mu_pre   = (np.mean(T)
                + alpha1 * np.mean(C) + alpha2 * np.mean(CS)
                + gamma  * np.mean(H)
                + delta_u* np.mean(U) + delta_v* np.mean(V)
                - phi    * np.mean(S) - kappa  * np.mean(DP))
    mu_vec_pre = (mu_pre
                  - alpha1*C - alpha2*CS - gamma*H
                  - delta_u*U - delta_v*V + phi*S + kappa*DP)
    res_pre  = dT - lam * (mu_vec_pre - T) * dt

    theta_h = np.zeros(24)
    for h in range(24):
        mask = hours_lag == h
        if mask.sum() >= 2:
            theta_h[h] = float(np.mean(res_pre[mask]) / (lam * dt + 1e-9))
    theta_h -= theta_h.mean()   # mean-centre — keeps mu_clear unbiased

    # ── Step 5: mu_clear from equilibrium identity ────────────────────────
    mu = float(
        np.mean(T)
        + alpha1  * np.mean(C)
        + alpha2  * np.mean(CS)
        + gamma   * np.mean(H)
        + delta_u * np.mean(U)
        + delta_v * np.mean(V)
        - phi     * np.mean(S)
        - kappa   * np.mean(DP)
    )

    # ── Step 6: noise parameters ──────────────────────────────────────────
    th_vec  = np.array([theta_h[h] for h in hours_lag])
    mu_full = (mu + th_vec
               - alpha1*C - alpha2*CS - gamma*H
               - delta_u*U - delta_v*V + phi*S + kappa*DP)
    res = dT - lam * (mu_full - T) * dt

    clear = C < 0.5
    s0 = float(np.std(res[clear])  / np.sqrt(dt)) if clear.sum()    > 1 else 0.5
    s1 = float(np.std(res[~clear]) / np.sqrt(dt)) if (~clear).sum() > 1 else s0 + 0.2
    W_mag = np.sqrt(U**2 + V**2)
    eta  = float(np.clip(cov(np.abs(res), W_mag)      / (np.var(W_mag)      + 1e-9), 0.0, 0.10))
    zeta = float(np.clip(cov(np.abs(res), np.abs(DP)) / (np.var(np.abs(DP)) + 1e-9), 0.0,  0.5))

    return dict(
        lambda_  = lam,
        mu_clear = mu,
        alpha1   = alpha1,
        alpha2   = alpha2,
        gamma    = gamma,
        delta_u  = delta_u,
        delta_v  = delta_v,
        phi      = phi,
        kappa    = kappa,
        theta_h  = theta_h,
        sigma0   = float(np.clip(s0,      0.05, 3.0)),
        beta     = float(np.clip(s1 - s0, 0.0,  2.0)),
        eta      = eta,
        zeta     = zeta,
    )


def mu_eq(C, H, U, V, S, DP, p, hour=None):
    """
    OU equilibrium temperature.
    hour (float or int): if given, adds theta_h[hour] regime shift.
    C*S: cloud×solar interaction — clouds suppress temperature more at noon.
    """
    base = (p["mu_clear"]
            - p["alpha1"] * C
            - p["alpha2"] * C * S
            - p["gamma"]  * H
            - p["delta_u"]* U
            - p["delta_v"]* V
            + p["phi"]    * S
            + p["kappa"]  * DP)
    if hour is not None:
        base += p["theta_h"][int(hour) % 24]
    return base


def sigma_eq(C, W, DP, p, S=0.0):
    """
    Noise scale. 15% wider during daytime convective hours (S > 0.1).
    """
    base  = max(0.01, p["sigma0"] + p["beta"]*C + p["eta"]*W + p["zeta"]*abs(DP))
    return base * (1.15 if S > 0.1 else 1.0)


def monte_carlo(T0, c_path, h_path, u_path, v_path, s_path, dp_path, p,
                hour_path=None, N=N_PATHS, dt=SIM_DT):
    """Simulate N OU paths over len(path)-1 steps of size dt hours."""
    T     = np.full(N, float(T0), dtype=np.float64)
    steps = len(c_path) - 1
    for i in range(steps):
        C  = float(c_path[i]);  H  = float(h_path[i])
        U  = float(u_path[i]);  V  = float(v_path[i])
        S  = float(s_path[i]);  DP = float(dp_path[i])
        W  = math.sqrt(U**2 + V**2)
        hr = float(hour_path[i]) if hour_path is not None else None
        m   = mu_eq(C, H, U, V, S, DP, p, hour=hr)
        sig = sigma_eq(C, W, DP, p, S=S)
        T   = T + p["lambda_"]*(m - T)*dt + sig*math.sqrt(dt)*np.random.randn(N)
    return T

# ══════════════════════════════════════════════════════════════ COMPUTE

def compute(df_asos, asos_label, df_hist24, curr, df_fore, obs_high=None):
    """
    df_asos    : 90-day ASOS DataFrame (or None → fall back to df_hist24)
    asos_label : source description string for display
    df_hist24  : AccuWeather /historical/24 (fallback + always used for AW 24h ref)
    curr       : dict from /currentconditions (T0, current state)
    df_fore    : DataFrame from /hourly/12hour
    obs_high   : observed daily high so far (°F). MC paths floored to max(obs_high, T0).
    """
    # ── Choose fitting dataset ────────────────────────────────────────────
    if df_asos is not None and len(df_asos) >= ASOS_MIN_OBS:
        df_fit     = df_asos
        fit_source = asos_label
    else:
        df_fit     = df_hist24.tail(24).reset_index(drop=True)
        fit_source = f"AW /historical/24 ({len(df_hist24)} obs)"

    hist_t  = df_fit["temp_f"].values.astype(float)
    hist_c  = _fill(df_fit["cloud"].values.astype(float),        0.3)
    hist_h  = _fill(df_fit["humidity"].values.astype(float),     0.65)
    hist_ws = _fill(df_fit["wind_speed"].values.astype(float),   5.0)
    hist_wd = _fill(df_fit["wind_dir"].values.astype(float),     180.0)
    hist_p  = _fill(df_fit["pressure_hpa"].values.astype(float), 1015.0)
    hist_s  = solar_path(list(df_fit["time"]))
    hist_u, hist_v = uv_arrays(hist_ws, hist_wd)
    hist_dp = np.clip(np.gradient(hist_p) / FIT_DT, -6.0, 6.0)

    # Hour-of-day array for Change 6 regime fitting
    hist_hours = np.array([ts.hour for ts in df_fit["time"]], dtype=float)

    # ── Current state (T0) from AccuWeather /currentconditions ───────────
    T0     = float(curr["temp_f"])
    C0     = float(curr["cloud"])      if not np.isnan(curr["cloud"])      else float(hist_c[-1])
    H0     = float(curr["humidity"])   if not np.isnan(curr["humidity"])   else float(hist_h[-1])
    WS0    = float(curr["wind_speed"]) if not np.isnan(curr["wind_speed"]) else float(hist_ws[-1])
    WD0    = float(curr["wind_dir"])   if not np.isnan(curr["wind_dir"])   else float(hist_wd[-1])
    U0, V0 = uv_components(WS0, WD0)
    W0     = math.sqrt(U0**2 + V0**2)
    S0     = geometric_solar(curr["time"])
    P0     = float(curr["pressure_hpa"]) if not np.isnan(curr["pressure_hpa"]) else float(hist_p[-1])
    DP0    = float(hist_dp[-1])
    UV0    = float(curr["uv_index"])
    H0_hr  = curr["time"].hour

    # ── Fit OU on ASOS data ───────────────────────────────────────────────
    params = fit_ou(hist_c, hist_t, hist_h, hist_u, hist_v, hist_s, hist_dp,
                    hist_hours=hist_hours)
    mu_now = mu_eq(C0, H0, U0, V0, S0, DP0, params, hour=H0_hr)

    # ── Forecast path: piecewise-linear walk over HORIZON_HOURS ─────────
    # Fetch up to HORIZON_HOURS future forecast periods from AccuWeather.
    # Build a piecewise-linear path: current → waypoint_1 → waypoint_2 → …
    # Each segment is SIM_DT-spaced sub-steps (4 per hour).
    # If fewer periods than needed are returned, hold last values flat.
    now_ts     = now_utc()
    df_future  = df_fore[df_fore["time"] > now_ts].head(HORIZON_HOURS).reset_index(drop=True)

    # Collect waypoints: index 0 = current state, 1..H = forecast periods
    wp_c, wp_h, wp_u, wp_v = [C0], [H0], [U0], [V0]
    fore_time = f"+{HORIZON_HOURS}h"

    for i in range(HORIZON_HOURS):
        if i < len(df_future):
            fr = df_future.iloc[i]
            fc   = float(fr["cloud"])      if not np.isnan(fr["cloud"])      else wp_c[-1]
            fh   = float(fr["humidity"])   if not np.isnan(fr["humidity"])   else wp_h[-1]
            fws  = float(fr["wind_speed"]) if not np.isnan(fr["wind_speed"]) else math.sqrt(wp_u[-1]**2+wp_v[-1]**2)
            fwd  = (float(fr["wind_dir"])  if not np.isnan(fr["wind_dir"])
                    else math.degrees(math.atan2(-wp_u[-1], -wp_v[-1])) % 360)
            fu, fv = uv_components(fws, fwd)
            if i == HORIZON_HOURS - 1:
                fore_time = fr["time"].strftime("%H:%M UTC")
        else:
            # Hold last waypoint flat
            fc, fh, fu, fv = wp_c[-1], wp_h[-1], wp_u[-1], wp_v[-1]
        wp_c.append(fc); wp_h.append(fh); wp_u.append(fu); wp_v.append(fv)

    # Build full sub-step arrays by concatenating per-hour segments
    STEPS_PER_HOUR = int(round(1.0 / SIM_DT))   # = 4
    N_STEPS_TOTAL  = HORIZON_HOURS * STEPS_PER_HOUR

    c_path = np.empty(N_STEPS_TOTAL + 1)
    h_path = np.empty(N_STEPS_TOTAL + 1)
    u_path = np.empty(N_STEPS_TOTAL + 1)
    v_path = np.empty(N_STEPS_TOTAL + 1)

    for seg in range(HORIZON_HOURS):
        lo = seg * STEPS_PER_HOUR
        hi = lo + STEPS_PER_HOUR + 1
        seg_arr = lambda a, b: np.linspace(a, b, STEPS_PER_HOUR + 1)
        c_path[lo:hi] = seg_arr(wp_c[seg], wp_c[seg+1])
        h_path[lo:hi] = seg_arr(wp_h[seg], wp_h[seg+1])
        u_path[lo:hi] = seg_arr(wp_u[seg], wp_u[seg+1])
        v_path[lo:hi] = seg_arr(wp_v[seg], wp_v[seg+1])

    dp_path   = np.full(N_STEPS_TOTAL + 1, DP0)
    sub_ts    = [now_ts + pd.Timedelta(hours=i * SIM_DT) for i in range(N_STEPS_TOTAL + 1)]
    s_path    = solar_path(sub_ts)
    hour_path = np.array([ts.hour + ts.minute / 60.0 for ts in sub_ts])

    samples      = monte_carlo(T0, c_path, h_path, u_path, v_path,
                               s_path, dp_path, params, hour_path=hour_path)

    # Hard lower bound: daily high can never fall below what has already been observed.
    floor    = max(obs_high, T0) if obs_high is not None else T0
    samples  = np.maximum(samples, floor)

    rounded      = np.floor(samples + 0.5).astype(int)
    unique, cnts = np.unique(rounded, return_counts=True)
    probs        = cnts / float(N_PATHS)

    return dict(
        T0=T0, C0=C0, H0=H0, W0=W0, S0=S0,
        U0=U0, V0=V0, DP0=DP0, UV0=UV0, P0=P0,
        mu_now=mu_now,
        fore_time=fore_time, fore_c=wp_c[-1], fore_h=wp_h[-1],
        fore_ws=math.sqrt(wp_u[-1]**2+wp_v[-1]**2),
        fore_s=float(s_path[-1]), fore_dp=DP0,
        n_asos=len(df_fit), fit_source=fit_source,
        obs_high=floor,
        params=params, samples=samples,
        temps=unique, probs=probs,
        mean=float(samples.mean()), std=float(samples.std()),
        p10=float(np.percentile(samples,10)), p25=float(np.percentile(samples,25)),
        p75=float(np.percentile(samples,75)), p90=float(np.percentile(samples,90)),
        mode=int(unique[np.argmax(probs)]),
        fetched_at=datetime.now(),
    )

# ══════════════════════════════════════════════════════════════ DISPLAY

def draw(res, location_key, iteration, next_fetch):
    secs = max(0, int(next_fetch - time.monotonic()))
    prog = int((REFRESH_SEC - secs) / REFRESH_SEC * 38)
    bar  = "X"*prog + "."*(38-prog)
    ts   = res["fetched_at"].strftime("%H:%M:%S")
    p    = res["params"]
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
    L.append(f"  MIAMI KALSHI TEMPERATURE FORECAST  v7  #{iteration}")
    L.append(f"  LocationKey: {location_key}  |  {MIAMI_LAT}N  {abs(MIAMI_LON):.4f}W")
    L.append(f"  Fit: {res['fit_source']}")
    L.append(f"  Paths: {N_PATHS:,}  |  dt_sim={SIM_DT}h  |  Horizon: +{HORIZON_HOURS}h")
    L.append("=" * W)
    L.append(f"  Current (T0)  —  AccuWeather /currentconditions")
    L.append(f"    Temp     : {res['T0']:.1f} °F   obs daily high: {res['obs_high']:.1f} °F")
    L.append(f"    Cloud    : {res['C0']*100:.0f}%   RH: {res['H0']*100:.0f}%   Wind: {res['W0']:.1f} mph   Solar: {res['S0']*100:.0f}%")
    L.append(f"    Wind U   : {res['U0']:+.1f} mph (east+)    V: {res['V0']:+.1f} mph (north+)")
    L.append(f"    Pressure : {res['P0']:.1f} hPa    dP/dt: {res['DP0']:+.2f} hPa/hr ({plabel(res['DP0'])})")
    L.append(f"    UV Index : {res['UV0']:.0f}")
    L.append(f"    mu_eq(T0): {res['mu_now']:.1f} °F  (OU equilibrium incl. hour regime)")
    L.append("-" * W)
    L.append(f"  Forecast Inputs  —  settlement at {res['fore_time']}  (AccuWeather hourly)")
    L.append(f"    Cloud: {res['fore_c']*100:.0f}%   RH: {res['fore_h']*100:.0f}%   Wind: {res['fore_ws']:.1f} mph   Solar: {res['fore_s']*100:.0f}%   dP/dt: {res['fore_dp']:+.2f}")
    L.append("-" * W)
    L.append(f"  PDF  →  settlement at {res['fore_time']}")
    L.append(f"  Mode: {res['mode']}°F   Mean: {res['mean']:.1f}°F   Std: {res['std']:.2f}°F")
    L.append(f"  50% CI [{res['p25']:.0f}, {res['p75']:.0f}]°F    80% CI [{res['p10']:.0f}, {res['p90']:.0f}]°F")
    L.append("-" * W)
    L.append(f"  {'degF':<8} {'P(T=k)':>9}   Histogram")
    L.append("-" * W)
    for temp, prob in pairs[:20]:
        blen = int(prob / mp * 36)
        L.append(f"  {temp:<8} {prob*100:>8.2f}%   {'#'*blen}")
    L.append("-" * W)
    L.append(f"  OU Parameters  (ASOS {ASOS_DAYS}d fit, dt={FIT_DT}h, ridge={LAMBDA_RIDGE})")
    L.append(f"  lambda={p['lambda_']:.3f}   mu_clear={p['mu_clear']:.1f}°F  (NOT a forecast — baseline only)")
    L.append(f"  alpha1(C)={p['alpha1']:.3f}  alpha2(C*S)={p['alpha2']:.3f}  phi(S)={p['phi']:.3f} [fixed]")
    L.append(f"  gamma(RH)={p['gamma']:.3f}  delta_u={p['delta_u']:.3f}  delta_v={p['delta_v']:.3f}  kappa(dP)={p['kappa']:.3f}")
    L.append(f"  sigma0={p['sigma0']:.3f}  beta={p['beta']:.3f}  eta={p['eta']:.4f}  zeta={p['zeta']:.4f}")
    L.append(f"  theta_h: warm peak={th_pk:02d}h ({th[th_pk]:+.2f}°F)   cool trough={th_tr:02d}h ({th[th_tr]:+.2f}°F)")
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
    print("  Miami Kalshi Temperature Forecast  v6  —  AccuWeather + ASOS")
    print(f"  Change 1: {ASOS_DAYS}d ASOS history from Iowa Mesonet (cache: {ASOS_CACHE_PATH})")
    print(f"  Change 6: time-of-day regimes + cloud×solar interaction")
    print(f"  Horizon: +{HORIZON_HOURS}h  |  {N_PATHS:,} MC paths  |  Refresh {REFRESH_SEC}s")
    print(f"  Ctrl-C to stop.")
    print("=" * 72)

    location_key = resolve_location_key()
    print(f"\n  LocationKey: {location_key} (Miami Intl Airport)")

    # Fetch ASOS once at startup; will use cache on subsequent cycles
    print(f"\n  Loading ASOS {ASOS_DAYS}d history …")
    df_asos, asos_label = get_asos_fit_data(verbose=True)
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

            # Re-fetch ASOS weekly
            if df_asos is not None and (now_m - asos_loaded_at) > ASOS_REFRESH_SEC:
                print("\n  [ASOS] Weekly refresh …", flush=True)
                df_asos, asos_label = get_asos_fit_data(verbose=True)
                asos_loaded_at = now_m

            if now_m >= next_fetch:
                iteration += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{ts}] Cycle #{iteration}", flush=True)
                try:
                    print(f"  [1/3] /historical/24 (fallback + AW ref) …", flush=True)
                    df_hist24 = fetch_historical_24h(location_key)

                    print(f"  [2/3] /currentconditions …", flush=True)
                    curr      = fetch_current(location_key)
                    obs_high  = float(curr["temp_f"]) if obs_high is None else max(obs_high, float(curr["temp_f"]))

                    print(f"  [3/3] /forecasts/hourly/12hour …", flush=True)
                    df_fore   = fetch_forecast(location_key)

                    last_res   = compute(df_asos, asos_label, df_hist24, curr, df_fore, obs_high=obs_high)
                    next_fetch = time.monotonic() + REFRESH_SEC
                    print(
                        f"  Done — mode={last_res['mode']}°F  "
                        f"mean={last_res['mean']:.1f}°F  "
                        f"std={last_res['std']:.2f}°F  "
                        f"fit={last_res['fit_source']}",
                        flush=True,
                    )
                except Exception:
                    print("  FAILED — retrying in 60s", flush=True)
                    traceback.print_exc()
                    next_fetch = time.monotonic() + 60

            if last_res is not None:
                draw(last_res, location_key, iteration, next_fetch)

            time.sleep(60)   # tight loop — draw refreshes every minute, fetch every 15

    except KeyboardInterrupt:
        print("\nStopped.")


def run_once_json():
    import json, sys, os
    location_key = resolve_location_key()

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
        curr = fetch_current(location_key)
    if df_fore is None:
        df_fore = fetch_forecast(location_key)

    try:
        df_asos, asos_label = get_asos_fit_data(verbose=False)
    except Exception:
        df_asos, asos_label = None, "unavailable"

    df_hist24 = (None if df_asos is not None and len(df_asos) >= ASOS_MIN_OBS
                 else fetch_historical_24h(location_key))

    res = compute(df_asos, asos_label, df_hist24, curr, df_fore,
                  obs_high=curr.pop("_asos_today_high_f", None))
    out = {
        "model":      "accuweather_3h",
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
