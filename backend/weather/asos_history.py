"""
Historical ASOS observations for model fitting (hourly METAR, KMIA).

Source: ASOS archive served by the Iowa Environmental Mesonet (a mirror of the
NOAA/NWS ASOS METAR record).  Cached on disk and refreshed weekly.  There is no
fallback provider: if the archive is unreachable and no cache exists, this
raises AsosHistoryUnavailable so the outage is visible.  A stale cache from the
same source is used (and labelled as stale) rather than silently swapping data.
"""
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as cfg
from .asos_parser import SKY_COVER_FRACTION
from .meteo import dewpoint_from_t_rh, kt_to_mph, wind_to_uv

log = logging.getLogger("weather.asos")

ASOS_REFRESH_SEC = 7 * 24 * 3600
ASOS_MIN_OBS = 48
ARCHIVE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CACHE_DIR = Path(os.getenv("ASOS_CACHE_DIR") or Path(__file__).resolve().parent.parent / "modelscripts")

COLUMNS = ["time", "temp_f", "dewpoint_f", "cloud", "humidity", "wind_speed", "wind_dir",
           "wind_u", "wind_v", "pressure_hpa", "uv_index"]


class AsosHistoryUnavailable(RuntimeError):
    pass


def cache_path(days):
    return CACHE_DIR / f"kmia_asos_{days}d.csv"


def _skyc(code):
    if not code or code == "M":
        return np.nan
    return SKY_COVER_FRACTION.get(str(code).strip().upper()[:3], 0.5)


def fetch_history(days):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "station": cfg.ASOS_ARCHIVE_STATION,
        "data": "tmpf,dwpf,relh,sknt,drct,mslp,skyc1,skyc2",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "UTC", "format": "onlycomma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no",
        "report_type": "3",          # routine hourly METAR
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    lines = [l for l in resp.text.strip().splitlines() if not l.startswith("#")]
    if len(lines) < 2:
        raise ValueError("ASOS archive returned no data rows.")
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, [p.strip() for p in parts]))

        def f(k, fb=np.nan):
            v = row.get(k, "M")
            if v in ("M", "T", "", None):
                return fb
            try:
                return float(v)
            except ValueError:
                return fb

        temp_f = f("tmpf")
        if np.isnan(temp_f):
            continue
        try:
            ts = pd.Timestamp(row["valid"], tz="UTC")
        except Exception:
            continue
        dwpf, rh = f("dwpf"), f("relh")
        mph = kt_to_mph(f("sknt", 0.0))
        wdir = f("drct")
        wdir = 180.0 if np.isnan(wdir) else wdir
        u, v = wind_to_uv(mph, wdir)
        if np.isnan(dwpf) and not np.isnan(rh):
            dwpf = dewpoint_from_t_rh(temp_f, rh / 100.0)
        skyc = row.get("skyc2", "M") if row.get("skyc2", "M") not in ("M", "") else row.get("skyc1", "M")
        cloud = _skyc(skyc)
        rows.append(dict(
            time=ts, temp_f=temp_f, dewpoint_f=dwpf,
            cloud=cloud if not np.isnan(cloud) else 0.3,
            humidity=rh / 100.0 if not np.isnan(rh) else np.nan,
            wind_speed=mph, wind_dir=wdir, wind_u=u, wind_v=v,
            pressure_hpa=f("mslp"), uv_index=0.0,
        ))
    if not rows:
        raise ValueError("ASOS archive parse produced no valid rows.")
    return (pd.DataFrame(rows, columns=COLUMNS).sort_values("time")
            .drop_duplicates(subset=["time"]).reset_index(drop=True))


def _read_cache(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "dewpoint_f" not in df.columns:
        df["dewpoint_f"] = [dewpoint_from_t_rh(t, h) if h == h else np.nan
                            for t, h in zip(df["temp_f"], df["humidity"])]
    if "wind_u" not in df.columns or "wind_v" not in df.columns:
        uv = [wind_to_uv(s if s == s else 0.0, d if d == d else 180.0)
              for s, d in zip(df["wind_speed"], df["wind_dir"])]
        df["wind_u"], df["wind_v"] = [x[0] for x in uv], [x[1] for x in uv]
    return df


def get_fit_data(days, verbose=False):
    """Return (DataFrame[time UTC-aware,...], source_label). Raises AsosHistoryUnavailable."""
    path = cache_path(days)
    stale_df = None
    if path.exists():
        age = time.time() - path.stat().st_mtime
        try:
            df = _read_cache(path)
            if len(df) >= ASOS_MIN_OBS:
                if age <= ASOS_REFRESH_SEC:
                    return df, f"ASOS cache ({len(df)} obs, {days}d)"
                stale_df = df
        except Exception as exc:
            log.warning("ASOS cache unreadable (%s)", exc)
    try:
        df = fetch_history(days)
        out = df.copy()
        out["time"] = out["time"].dt.tz_convert("UTC").dt.tz_localize(None)   # CSV stays UTC-naive text
        out.to_csv(path, index=False)
        return df, f"ASOS live ({len(df)} obs, {days}d)"
    except Exception as exc:
        if stale_df is not None:
            msg = f"ASOS cache STALE ({len(stale_df)} obs, {days}d; refresh failed: {exc})"
            log.warning(msg)
            return stale_df, msg
        raise AsosHistoryUnavailable(f"ASOS history unavailable and no cache: {exc}") from exc
