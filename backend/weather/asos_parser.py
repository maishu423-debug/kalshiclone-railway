"""
Single normalizer for live NOAA ASOS observations.

The only live source is NOAA MADIS OMO/HFMETAR (LDAD/hfmetar), mapped to ONE schema.
No other provider format is accepted.

Normalized observation (all timestamps tz-aware UTC, missing fields are None):

    station, observed_at (ISO-8601 UTC), temp_c, temp_f, dewpoint_c, dewpoint_f,
    relative_humidity (0-1), wind_speed_mph, wind_gust_mph, wind_dir_deg,
    wind_u_mph, wind_v_mph, pressure_hpa, station_pressure_hpa,
    cloud_fraction (0-1), visibility_miles, weather, source, report_source
"""
import logging
from datetime import datetime, timezone

from . import config as cfg
from . import meteo

log = logging.getLogger("weather.asos")

# Sky cover -> fraction (same mapping as the historical ASOS pipeline)
SKY_COVER_FRACTION = {
    "CLR": 0.00, "SKC": 0.00, "NSC": 0.00, "NCD": 0.00, "CAVOK": 0.00,
    "FEW": 0.20, "SCT": 0.45, "BKN": 0.75, "OVC": 1.00, "VV": 1.00,
}

_CORE_FIELDS = ("temp_c", "dewpoint_c", "relative_humidity", "wind_speed_mph",
                "wind_dir_deg", "pressure_hpa", "cloud_fraction", "visibility_miles")


# ── generic helpers ─────────────────────────────────────────────────────────

def _num(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except (TypeError, ValueError):
        return None


def parse_utc(value):
    """ISO string / epoch seconds / epoch ms / datetime -> aware UTC datetime, else None."""
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            ts = float(value)
            dt = datetime.fromtimestamp(ts / 1000.0 if ts > 1e11 else ts, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)   # feeds are UTC; never leave naive
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def cloud_fraction_from_layers(covers):
    """Max sky-cover fraction over reported layers; None if no recognizable layer."""
    fracs = []
    for c in covers:
        if not c:
            continue
        key = str(c).strip().upper()
        if key in SKY_COVER_FRACTION:
            fracs.append(SKY_COVER_FRACTION[key])
        elif key[:3] in SKY_COVER_FRACTION:
            fracs.append(SKY_COVER_FRACTION[key[:3]])
    return max(fracs) if fracs else None


def normalize_madis_record(rec):
    """
    One MADIS HFMETAR record -> raw (unvalidated) observation.

    `rec` uses MADIS native units with fill values already mapped to None:
      station, observation_time (epoch s), temperature (K), dewpoint (K),
      wind_dir (deg), wind_speed / wind_gust (m/s), altimeter (Pa), visibility (m),
      sky_cover (list of 'FEW'/'SCT'/...), present_weather (str), dd (dict var -> MADIS
      data-descriptor flag: Z C S V G prelim/pass, Q questionable, X failed, B bad)
    """
    observed = parse_utc(rec.get("observation_time"))
    if observed is None:
        return None
    dd = rec.get("dd") or {}

    def ok(name, value):
        return None if dd.get(name) in cfg.BAD_QC_FLAGS else _num(value)

    k_t, k_d = ok("temperature", rec.get("temperature")), ok("dewpoint", rec.get("dewpoint"))
    lo, hi = cfg.MADIS_TEMP_K_RANGE
    if k_t is not None and not (lo <= k_t <= hi):
        k_t = None                                  # physically impossible Kelvin value -> no temperature
    return dict(
        station=rec.get("station"),
        observed_at_dt=observed,
        temp_c=None if k_t is None else k_t - 273.15,
        dewpoint_c=None if k_d is None else k_d - 273.15,
        relative_humidity=None,
        wind_speed_mph=meteo.ms_to_mph(ok("windSpeed", rec.get("wind_speed"))),
        wind_gust_mph=meteo.ms_to_mph(ok("windGust", rec.get("wind_gust"))),
        wind_dir_deg=ok("windDir", rec.get("wind_dir")),
        pressure_hpa=meteo.pa_to_hpa(ok("altimeter", rec.get("altimeter"))),   # altimeter setting
        station_pressure_hpa=None,
        cloud_fraction=cloud_fraction_from_layers(rec.get("sky_cover") or []),
        visibility_miles=meteo.m_to_miles(ok("visibility", rec.get("visibility"))),
        weather=rec.get("present_weather") or None,
        report_source="MADIS_HFMETAR",
    )


# ── validation / completion ─────────────────────────────────────────────────

def validate_and_complete(raw, rejected=None):
    """
    Apply sanity checks, derive RH / dew point / U,V, build the public schema.
    Returns the normalized dict, or None if the observation is unusable
    (temperature missing or impossible).  Bad individual fields become None.
    """
    rejected = rejected if rejected is not None else []
    when = raw["observed_at_dt"].isoformat()

    def reject(reason):
        rejected.append({"observed_at": when, "reason": reason})
        log.debug("ASOS obs %s rejected: %s", when, reason)

    def in_range(v, lo_hi):
        return v is not None and lo_hi[0] <= v <= lo_hi[1]

    temp_c = raw["temp_c"]
    if temp_c is None:
        reject("missing temperature")
        return None
    if not in_range(temp_c, cfg.QC_TEMP_C):
        reject(f"temperature out of range ({temp_c})")
        return None

    dew_c = raw["dewpoint_c"]
    if dew_c is not None and (not in_range(dew_c, cfg.QC_DEWPOINT_C)
                              or dew_c > temp_c + cfg.QC_DEWPOINT_MAX_EXCESS_C):
        reject(f"dew point implausible ({dew_c} vs T {temp_c}); field dropped")
        dew_c = None

    rh = raw["relative_humidity"]
    if rh is not None and not in_range(rh, cfg.QC_RH):
        reject(f"relative humidity out of range ({rh}); field dropped")
        rh = None
    if rh is not None:
        rh = min(rh, 1.0)

    # Derive what is derivable — never invent what is not.
    if rh is None and dew_c is not None:
        rh = meteo.rh_from_t_td(temp_c, dew_c)
    if dew_c is None and rh is not None and rh > 0:
        dew_c = meteo.dewpoint_c_from_t_rh(temp_c, rh)

    wspd = raw["wind_speed_mph"]
    if wspd is not None and not in_range(wspd, cfg.QC_WIND_MPH):
        reject(f"wind speed out of range ({wspd}); field dropped")
        wspd = None
    wdir = raw["wind_dir_deg"]
    if wdir is not None and not (0.0 <= wdir <= 360.0):
        reject(f"wind direction out of range ({wdir}); field dropped")
        wdir = None
    gust = raw["wind_gust_mph"]
    if gust is not None and not in_range(gust, cfg.QC_WIND_MPH):
        gust = None

    pres = raw["pressure_hpa"]
    stn_p = raw["station_pressure_hpa"]
    if pres is not None and not in_range(pres, cfg.QC_PRESSURE_HPA):
        reject(f"pressure out of range ({pres}); field dropped")
        pres = None
    if stn_p is not None and not in_range(stn_p, cfg.QC_PRESSURE_HPA):
        stn_p = None
    if pres is None:
        pres = stn_p        # at KMIA's elevation (~3 m) station ≈ sea-level pressure; trend is what matters

    u, v = meteo.wind_to_uv(wspd, wdir)
    temp_f = meteo.c_to_f(temp_c)
    dew_f = meteo.c_to_f(dew_c)

    def r(x, n):
        return None if x is None else round(x, n)

    return dict(
        station=(raw.get("station") or cfg.ASOS_STATION),
        observed_at=raw["observed_at_dt"].isoformat(),
        temp_c=r(temp_c, 2), temp_f=r(temp_f, 2),
        dewpoint_c=r(dew_c, 2), dewpoint_f=r(dew_f, 2),
        relative_humidity=r(rh, 4),
        wind_speed_mph=r(wspd, 2), wind_gust_mph=r(gust, 2), wind_dir_deg=wdir,
        wind_u_mph=r(u, 3), wind_v_mph=r(v, 3),
        pressure_hpa=r(pres, 2), station_pressure_hpa=r(stn_p, 2),
        cloud_fraction=raw["cloud_fraction"],
        visibility_miles=r(raw["visibility_miles"], 2),
        weather=raw.get("weather"),
        source=cfg.SOURCE_ID,
        report_source=raw.get("report_source"),
    )


def _completeness(obs):
    return sum(obs.get(f) is not None for f in _CORE_FIELDS)


_SOURCE_RANK = {"MADIS_HFMETAR": 0}


def deduplicate(observations):
    """
    Sort by UTC time and collapse records that share the same minute.
    Winner = most complete record, then first seen: deterministic.
    """
    best = {}
    for idx, obs in enumerate(observations):
        dt = parse_utc(obs["observed_at"])
        key = dt.replace(second=0, microsecond=0)
        rank = (-_completeness(obs), _SOURCE_RANK.get(obs.get("report_source"), 9), idx)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, obs)
    return [best[k][1] for k in sorted(best)]


def is_public_5min(dt):
    """Public series: observation minute is a multiple of 5 (seconds are not required to be 0)."""
    return dt.minute % cfg.PUBLIC_OBS_MINUTE_STEP == 0


def parse_madis_records(records, station=None, rejected=None, only_5min=True):
    """Raw MADIS records -> validated, de-duplicated, ascending-UTC normalized observations.

    The observation time is the MADIS `observationTime` (never download/file/row order).
    """
    station = station or cfg.ASOS_STATION
    out = []
    for rec in records:
        if (rec.get("station") or "").strip().upper() != station:
            continue
        raw = normalize_madis_record(rec)
        if raw is None:
            continue
        if only_5min and not is_public_5min(raw["observed_at_dt"]):
            log.debug("non-5-minute observation %s ignored", raw["observed_at_dt"].isoformat())
            continue
        obs = validate_and_complete(raw, rejected)
        if obs is not None:
            out.append(obs)
    return deduplicate(out)
