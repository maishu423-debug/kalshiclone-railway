"""
Miami temperature forecast - OU Monte Carlo on ASOS data (KMIA), horizon via --horizon.

Data (NOAA ASOS only; no other weather provider):
  * fit data   : hourly ASOS history (weather.asos_history)
  * T0 / state : latest KMIA ASOS observation (shared input, fetched once per cycle)
  * 15/30/45/60-min environmental inputs: ASOS nowcast (weather.asos_nowcast)
  * beyond +60 min the +60-min nowcast state is held flat
  * solar: deterministic geometry from time, lat/lon, day of year

Model (Ornstein-Uhlenbeck, time-of-day regimes) - unchanged mathematics:
  dT = -lambda*(T - mu(C,H,U,V,S,DP,h)) dt + sigma(C,W,DP,S) dW_t

Usage (json mode is what the orchestrator runs):
  python ou_forecast.py --json-output [--horizon 1|2|3]
"""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weather import config as wcfg                       # noqa: E402
from weather import meteo                                # noqa: E402
from weather import asos_history                         # noqa: E402
from weather.asos_parser import parse_utc                # noqa: E402
from weather.model_input import (                        # noqa: E402
    load_shared_input, require_current, has, output_metadata)

N_PATHS      = 10_000
FIT_DT       = 1.0        # OU fitting dt (hours) - ASOS history is hourly
SIM_DT       = 0.25       # Monte Carlo sub-step (hours) == nowcast step (15 min)
PHI_FIXED    = 5.0        # fixed solar coeff (breaks C/S collinearity)
LAMBDA_RIDGE = 0.5        # ridge penalty
MIAMI_LAT, MIAMI_LON = wcfg.MIAMI_LAT, wcfg.MIAMI_LON
ASOS_MIN_OBS = asos_history.ASOS_MIN_OBS


def geometric_solar(ts):
    return meteo.geometric_solar(ts, MIAMI_LAT, MIAMI_LON)


def solar_path(timestamps):
    return np.array([geometric_solar(ts) for ts in timestamps])


def uv_components(speed_mph, dir_deg):
    if math.isnan(speed_mph) or math.isnan(dir_deg):
        return 0.0, 0.0
    return meteo.wind_to_uv(speed_mph, dir_deg)


def uv_arrays(speeds, dirs):
    pairs = [uv_components(s, d) for s, d in zip(speeds, dirs)]
    return (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


# ══════════════════════════════════════════════════════════════ OU MODEL

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

def compute(df_fit, fit_source, shared, horizon_hours):
    """
    df_fit : hourly ASOS history (fitting only)
    shared : normalized ASOS shared input (current state + nowcast + today's high)
    """
    cur = require_current(shared)
    now_ts = pd.Timestamp(parse_utc(shared["metadata"]["generated_at"]))
    obs_ts = pd.Timestamp(parse_utc(cur["observed_at"]))

    hist_t  = df_fit["temp_f"].values.astype(float)
    hist_c  = _fill(df_fit["cloud"].values.astype(float),        0.3)
    hist_h  = _fill(df_fit["humidity"].values.astype(float),     0.65)
    hist_ws = _fill(df_fit["wind_speed"].values.astype(float),   5.0)
    hist_wd = _fill(df_fit["wind_dir"].values.astype(float),     180.0)
    hist_p  = _fill(df_fit["pressure_hpa"].values.astype(float), 1015.0)
    hist_s  = solar_path(list(df_fit["time"]))
    hist_u, hist_v = uv_arrays(hist_ws, hist_wd)
    hist_dp = np.clip(np.gradient(hist_p) / FIT_DT, -6.0, 6.0)
    hist_hours = np.array([ts.hour for ts in df_fit["time"]], dtype=float)

    # ── Current state: latest KMIA ASOS observation only ─────────────────
    T0  = float(cur["temp_f"])
    C0  = float(cur["cloud_fraction"])    if has(cur.get("cloud_fraction"))    else float(hist_c[-1])
    H0  = float(cur["relative_humidity"]) if has(cur.get("relative_humidity")) else float(hist_h[-1])
    WS0 = float(cur["wind_speed_mph"])    if has(cur.get("wind_speed_mph"))    else float(hist_ws[-1])
    WD0 = float(cur["wind_dir_deg"])      if has(cur.get("wind_dir_deg"))      else float(hist_wd[-1])
    if has(cur.get("wind_u_mph")) and has(cur.get("wind_v_mph")):
        U0, V0 = float(cur["wind_u_mph"]), float(cur["wind_v_mph"])
    else:
        U0, V0 = uv_components(WS0, WD0)
    W0  = math.sqrt(U0**2 + V0**2)
    S0  = geometric_solar(obs_ts)
    P0  = float(cur["pressure_hpa"]) if has(cur.get("pressure_hpa")) else float(hist_p[-1])
    ptrend = (shared.get("features") or {}).get("pressure_trend_hpa_per_hr")
    DP0 = float(np.clip(ptrend, -6.0, 6.0)) if has(ptrend) else float(hist_dp[-1])
    H0_hr = obs_ts.hour

    params = fit_ou(hist_c, hist_t, hist_h, hist_u, hist_v, hist_s, hist_dp,
                    hist_hours=hist_hours)
    mu_now = mu_eq(C0, H0, U0, V0, S0, DP0, params, hour=H0_hr)

    # ── Covariate path from the ASOS nowcast (15-min steps == SIM_DT) ────
    steps_per_hour = int(round(1.0 / SIM_DT))
    n_steps = horizon_hours * steps_per_hour
    ff = {f["minutes_ahead"]: f for f in shared.get("future_features", [])}
    last_key = max(ff) if ff else None

    def state_at(i):
        """Covariates at sub-step i (15*i min ahead); beyond the nowcast, hold last step flat."""
        if i == 0 or last_key is None:
            return C0, H0, U0, V0, DP0
        f = ff.get(15 * i) or ff[last_key]
        c  = f["cloud_fraction"]    if has(f.get("cloud_fraction"))    else C0
        h  = f["relative_humidity"] if has(f.get("relative_humidity")) else H0
        u  = f["wind_u_mph"]        if has(f.get("wind_u_mph"))        else U0
        v  = f["wind_v_mph"]        if has(f.get("wind_v_mph"))        else V0
        dp = f["pressure_trend_hpa_per_hr"] if has(f.get("pressure_trend_hpa_per_hr")) else DP0
        return float(c), float(h), float(u), float(v), float(np.clip(dp, -6.0, 6.0))

    states  = [state_at(i) for i in range(n_steps + 1)]
    c_path  = np.array([s[0] for s in states])
    h_path  = np.array([s[1] for s in states])
    u_path  = np.array([s[2] for s in states])
    v_path  = np.array([s[3] for s in states])
    dp_path = np.array([s[4] for s in states])
    sub_ts  = [now_ts + pd.Timedelta(hours=i * SIM_DT) for i in range(n_steps + 1)]
    s_path  = solar_path(sub_ts)
    hour_path = np.array([ts.hour + ts.minute / 60.0 for ts in sub_ts])
    fore_time = (now_ts + pd.Timedelta(hours=horizon_hours)).strftime("%H:%M UTC")

    samples = monte_carlo(T0, c_path, h_path, u_path, v_path,
                          s_path, dp_path, params, hour_path=hour_path)

    # Hard lower bound: the day's recorded high can never decrease.
    # obs_high = max valid, de-duplicated ASOS temperature since Miami-local midnight.
    obs_high = (shared.get("today") or {}).get("high_f")
    floor    = max(obs_high, T0) if obs_high is not None else T0
    samples  = np.maximum(samples, floor)

    rounded      = np.floor(samples + 0.5).astype(int)
    unique, cnts = np.unique(rounded, return_counts=True)
    probs        = cnts / float(N_PATHS)

    return dict(
        T0=T0, C0=C0, H0=H0, W0=W0, S0=S0, U0=U0, V0=V0, DP0=DP0, P0=P0,
        mu_now=mu_now, fore_time=fore_time,
        fore_c=float(c_path[-1]), fore_h=float(h_path[-1]),
        fore_ws=float(math.hypot(u_path[-1], v_path[-1])),
        n_asos=len(df_fit), fit_source=fit_source, obs_high=floor,
        params=params, samples=samples, temps=unique, probs=probs,
        mean=float(samples.mean()), std=float(samples.std()),
        p10=float(np.percentile(samples, 10)), p25=float(np.percentile(samples, 25)),
        p75=float(np.percentile(samples, 75)), p90=float(np.percentile(samples, 90)),
        mode=int(unique[np.argmax(probs)]),
        fetched_at=datetime.now(timezone.utc),
    )


def history_days(horizon_hours):
    return 90 if horizon_hours >= 3 else 21


def run_once_json(horizon_hours, model_name):
    shared = load_shared_input()
    require_current(shared)
    df_fit, fit_label = asos_history.get_fit_data(history_days(horizon_hours))
    if len(df_fit) < ASOS_MIN_OBS:
        raise RuntimeError(f"ASOS history too short for fitting ({len(df_fit)} obs)")
    res = compute(df_fit, fit_label, shared, horizon_hours)
    out = {
        "model":      model_name,
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
        **output_metadata(shared),
    }
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    horizon = int(_arg("--horizon", 1))
    if "--json-output" not in sys.argv:
        sys.exit("usage: ou_forecast.py --json-output [--horizon 1|2|3] [--model-name NAME]")
    run_once_json(horizon, _arg("--model-name", f"ou_{horizon}h"))
