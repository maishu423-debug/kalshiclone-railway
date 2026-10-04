"""
Miami temperature forecast - joint VAR(1) weather scenarios + temperature OU
Monte Carlo on ASOS data (KMIA), horizon via --horizon.

Data (NOAA ASOS only; no other weather provider):
  * fit data (VAR(1) + temperature OU): hourly ASOS history (weather.asos_history)
  * T0 and current VAR state: latest KMIA ASOS observation (shared input,
    fetched once per cycle by the orchestrator)
  * dP/dt state: ASOS recent pressure trend
  * solar: deterministic geometry from time, lat/lon, day of year

Mathematics unchanged: VAR(1) draws joint (cloud, RH, wind U/V, dP/dt,
dew point) scenarios chained once per forecast hour; each path then drives the
temperature OU with its own covariates.

Usage (json mode is what the orchestrator runs):
  python var_forecast.py --json-output [--horizon 1|2|3]
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
FIT_DT       = 1.0        # OU fitting dt (hours)
SIM_DT       = 0.25       # MC sub-step (hours)
PHI_FIXED    = 5.0        # fixed solar coefficient
LAMBDA_RIDGE = 0.5        # ridge penalty on temp OU
MIAMI_LAT, MIAMI_LON = wcfg.MIAMI_LAT, wcfg.MIAMI_LON
ASOS_MIN_OBS = asos_history.ASOS_MIN_OBS

# Variables in joint model (order matters - used throughout)
VAR_KEYS  = ["cloud", "rh", "wind_u", "wind_v", "dpdt", "dewpoint"]
VAR_CLIPS = {
    "cloud":    (0.0,   100.0),
    "rh":       (0.0,   100.0),
    "wind_u":   (-80.0,  80.0),
    "wind_v":   (-80.0,  80.0),
    "dpdt":     (-10.0,  10.0),
    "dewpoint": (-20.0,  95.0),
}


def geometric_solar(ts):
    return meteo.geometric_solar(ts, MIAMI_LAT, MIAMI_LON)


def solar_path(timestamps):
    return np.array([geometric_solar(ts) for ts in timestamps])


dewpoint_from_t_rh = meteo.dewpoint_from_t_rh


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

def compute(df_fit, fit_source, shared, horizon_hours):
    """
    df_fit : hourly ASOS history (VAR(1) + temperature OU fitting)
    shared : normalized ASOS shared input (current state + today's high)
    """
    cur = require_current(shared)
    now_ts = pd.Timestamp(parse_utc(shared["metadata"]["generated_at"]))
    obs_ts = pd.Timestamp(parse_utc(cur["observed_at"]))

    hist_t  = df_fit["temp_f"].values.astype(float)
    hist_c  = _fill_arr(df_fit["cloud"].values.astype(float),        0.3)
    hist_h  = _fill_arr(df_fit["humidity"].values.astype(float),     0.65)
    hist_u  = _fill_arr(df_fit["wind_u"].values.astype(float),       0.0)
    hist_v  = _fill_arr(df_fit["wind_v"].values.astype(float),       0.0)
    hist_p  = _fill_arr(df_fit["pressure_hpa"].values.astype(float), 1015.0)
    hist_s  = solar_path(list(df_fit["time"]))
    hist_dp = np.clip(np.gradient(hist_p) / FIT_DT, -6.0, 6.0)
    if "dewpoint_f" in df_fit.columns:
        hist_dew = _fill_arr(df_fit["dewpoint_f"].values.astype(float), 70.0)
    else:
        hist_dew = np.array([dewpoint_from_t_rh(t, h) for t, h in zip(hist_t, hist_h)], dtype=float)
        df_fit = df_fit.copy()
        df_fit["dewpoint_f"] = hist_dew
    hist_hours = np.array([ts.hour for ts in df_fit["time"]], dtype=float)

    # ── Current state: latest KMIA ASOS observation only ─────────────────
    T0    = float(cur["temp_f"])
    S0    = geometric_solar(obs_ts)
    H0_hr = obs_ts.hour
    C0 = float(cur["cloud_fraction"])    if has(cur.get("cloud_fraction"))    else float(hist_c[-1])
    H0 = float(cur["relative_humidity"]) if has(cur.get("relative_humidity")) else float(hist_h[-1])
    U0 = float(cur["wind_u_mph"])        if has(cur.get("wind_u_mph"))        else float(hist_u[-1])
    V0 = float(cur["wind_v_mph"])        if has(cur.get("wind_v_mph"))        else float(hist_v[-1])
    W0 = math.sqrt(U0**2 + V0**2)
    ptrend = (shared.get("features") or {}).get("pressure_trend_hpa_per_hr")
    DP0 = float(np.clip(ptrend, -6.0, 6.0)) if has(ptrend) else float(hist_dp[-1])
    P0  = float(cur["pressure_hpa"]) if has(cur.get("pressure_hpa")) else float(hist_p[-1])
    DEW0 = (float(cur["dewpoint_f"]) if has(cur.get("dewpoint_f"))
            else dewpoint_from_t_rh(T0, H0))

    # ── Fit temperature OU and VAR(1) on ASOS history ────────────────────
    temp_params = fit_temp_ou(hist_c, hist_t, hist_h, hist_u, hist_v,
                              hist_s, hist_dp, hist_dew, hist_hours=hist_hours)
    var_params = fit_var1(df_fit)

    # Current state vector: [cloud%, rh%, wind_u, wind_v, dpdt, dewpoint]
    x0_vec = np.array([C0 * 100.0, H0 * 100.0, U0, V0, DP0, DEW0])

    # ── Joint scenarios: one VAR(1) step per forecast hour, chained ──────
    # Each chained step adds independent correlated noise, so the final
    # distribution widens with horizon.
    var_samples = sample_var1(var_params, x0_vec, N=N_PATHS)
    a, B, Sigma, k_v = var_params["a"], var_params["B"], var_params["Sigma"], var_params["k"]
    for _ in range(horizon_hours - 1):
        X = np.column_stack([var_samples[k] for k in VAR_KEYS])
        x_next = a + X @ B.T
        try:
            eps = np.random.multivariate_normal(np.zeros(k_v), Sigma, size=N_PATHS)
        except np.linalg.LinAlgError:
            eps = np.random.randn(N_PATHS, k_v) * np.sqrt(np.diag(Sigma))
        X_next = x_next + eps
        for i, key in enumerate(VAR_KEYS):
            lo, hi = VAR_CLIPS[key]
            var_samples[key] = np.clip(X_next[:, i], lo, hi)

    # ── Solar / hour sub-step paths over the horizon ─────────────────────
    n_steps   = horizon_hours * int(round(1.0 / SIM_DT))
    sub_ts    = [now_ts + pd.Timedelta(hours=i * SIM_DT) for i in range(n_steps + 1)]
    s_path    = solar_path(sub_ts)
    hour_path = np.array([ts.hour + ts.minute / 60.0 for ts in sub_ts])
    fore_time = (now_ts + pd.Timedelta(hours=horizon_hours)).strftime("%H:%M UTC")

    samples = monte_carlo_temp_vectorised(
        T0, var_samples, s_path, temp_params,
        hour_path=hour_path, N=N_PATHS, dt=SIM_DT)

    # Hard lower bound: the day's recorded high can never decrease.
    # obs_high = max valid, de-duplicated ASOS temperature since Miami-local midnight.
    obs_high = (shared.get("today") or {}).get("high_f")
    floor    = max(obs_high, T0) if obs_high is not None else T0
    samples  = np.maximum(samples, floor)

    rounded      = np.floor(samples + 0.5).astype(int)
    unique, cnts = np.unique(rounded, return_counts=True)
    probs        = cnts / float(N_PATHS)

    return dict(
        T0=T0, S0=S0, P0=P0, C0=C0 * 100, H0=H0 * 100, W0=W0,
        fore_time=fore_time, n_fit=len(df_fit), fit_source=fit_source,
        obs_high=floor, temp_params=temp_params, var_params=var_params,
        samples=samples, temps=unique, probs=probs,
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
        sys.exit("usage: var_forecast.py --json-output [--horizon 1|2|3] [--model-name NAME]")
    run_once_json(horizon, _arg("--model-name", f"var_{horizon}h"))
