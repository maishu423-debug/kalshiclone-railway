"""Unit conversions, humidity, wind and solar geometry (no I/O)."""
import math

# Magnus-Tetens constants (same family used by the historical pipeline)
_MAG_A, _MAG_B = 17.625, 243.04


def c_to_f(c):
    return None if c is None else c * 9.0 / 5.0 + 32.0


def f_to_c(f):
    return None if f is None else (f - 32.0) * 5.0 / 9.0


def kmh_to_mph(v):
    return None if v is None else v / 1.609344


def ms_to_mph(v):
    return None if v is None else v * 2.236936


def kt_to_mph(v):
    return None if v is None else v * 1.150779


def pa_to_hpa(v):
    return None if v is None else v / 100.0


def inhg_to_hpa(v):
    return None if v is None else v * 33.8639


def m_to_miles(v):
    return None if v is None else v / 1609.344


def rh_from_t_td(temp_c, dewpoint_c):
    """Relative humidity (0-1) from temperature and dew point in deg C."""
    if temp_c is None or dewpoint_c is None:
        return None
    e_td = math.exp(_MAG_A * dewpoint_c / (_MAG_B + dewpoint_c))
    e_t = math.exp(_MAG_A * temp_c / (_MAG_B + temp_c))
    return max(0.0, min(1.0, e_td / e_t))


def dewpoint_c_from_t_rh(temp_c, rh_frac):
    """Dew point (deg C) from temperature (deg C) and RH (0-1)."""
    if temp_c is None or rh_frac is None or rh_frac <= 0:
        return None
    g = math.log(rh_frac) + (_MAG_A * temp_c) / (_MAG_B + temp_c)
    return (_MAG_B * g) / (_MAG_A - g)


def dewpoint_from_t_rh(temp_f, rh_frac):
    """Dew point (deg F) from temperature (deg F) and RH (0-1); NaN-safe floor of 1% RH."""
    rh = max(0.01, rh_frac)
    return c_to_f(dewpoint_c_from_t_rh(f_to_c(temp_f), rh))


def wind_to_uv(speed, direction_deg):
    """
    Meteorological wind (speed, direction wind blows FROM) -> (U east, V north).
    Direction is reduced mod 360 so 0 == 360.  Returns (None, None) when the
    components cannot be known; calm wind (speed == 0) is (0, 0) with any direction.
    """
    if speed is None:
        return None, None
    if speed == 0:
        return 0.0, 0.0
    if direction_deg is None:
        return None, None
    r = math.radians(direction_deg % 360.0)
    return -speed * math.sin(r), -speed * math.cos(r)


def uv_to_speed_dir(u, v):
    if u is None or v is None:
        return None, None
    speed = math.hypot(u, v)
    if speed < 1e-9:
        return 0.0, None
    return speed, math.degrees(math.atan2(-u, -v)) % 360.0


def geometric_solar(ts, lat, lon):
    """Clear-sky solar intensity in [0, 1] from timestamp (UTC), lat, lon, day of year."""
    doy = ts.day_of_year
    decl_deg = 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    solar_noon_utc = 12.0 - lon / 15.0
    utc_h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    ha_deg = 15.0 * (utc_h - solar_noon_utc)
    lat_r, decl_r, ha_r = map(math.radians, (lat, decl_deg, ha_deg))
    sin_elev = (math.sin(lat_r) * math.sin(decl_r)
                + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    return float(max(0.0, sin_elev))
