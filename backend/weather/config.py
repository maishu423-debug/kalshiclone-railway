"""
Central configuration for the ASOS-only weather pipeline.

Weather observation source: NOAA MADIS OMO/HFMETAR (LDAD/hfmetar) for KMIA.
No other live source and no fallback.  Historical model-fitting
data: the ASOS archive mirrored by the Iowa Environmental Mesonet.
No other weather provider is used anywhere in the operational path.
"""
import os

# ── Station ──────────────────────────────────────────────────────────────────
ASOS_STATION = os.getenv("ASOS_STATION", "KMIA").strip().upper() or "KMIA"
# Iowa Mesonet drops the leading "K" of US ICAO ids (KMIA -> MIA).
ASOS_ARCHIVE_STATION = ASOS_STATION[1:] if ASOS_STATION.startswith("K") else ASOS_STATION

MIAMI_LAT = 25.793
MIAMI_LON = -80.291
LOCAL_TZ_NAME = "America/New_York"   # DST handled by zoneinfo, never hardcode UTC-4/-5

# ── Live observation window ──────────────────────────────────────────────────
ASOS_LOOKBACK_HOURS = 3
ASOS_TARGET_INTERVAL_MINUTES = 5
FORECAST_STEPS_MINUTES = [15, 30, 45, 60]

# ── Freshness (configurable) ─────────────────────────────────────────────────
# MADIS_STALE_MINUTES is accepted as an alias for ASOS_FRESH_MINUTES
ASOS_FRESH_MINUTES = float(os.getenv("ASOS_FRESH_MINUTES") or os.getenv("MADIS_STALE_MINUTES") or "15")  # <= healthy
ASOS_MAX_CURRENT_AGE_MINUTES = float(os.getenv("ASOS_MAX_CURRENT_AGE_MINUTES", "25"))  # <= delayed; beyond = unavailable

# ── Nowcast tuning ───────────────────────────────────────────────────────────
TREND_WINDOW_MINUTES = 60          # robust-trend window for T/Td/RH/wind
PRESSURE_TREND_WINDOW_MINUTES = 120
TREND_MIN_POINTS = 4
TREND_DAMPING_TAU_HOURS = 1.0      # extrapolation saturates instead of growing linearly
# Max believable rate of change per hour (clips both the slope and the result)
MAX_RATE_PER_HOUR = {
    "relative_humidity": 0.15,     # fraction / h
    "dewpoint_f": 3.0,
    "pressure_hpa": 1.5,
    "wind_u_mph": 8.0,
    "wind_v_mph": 8.0,
}
CLOUD_RECENT_MINUTES = 60
CLOUD_PERSISTENCE_WEIGHT = 0.5     # weight of latest cloud vs recent weighted average

# ── Quality control (deliberately loose — only reject impossible values) ────
QC_TEMP_C = (-50.0, 55.0)
QC_DEWPOINT_C = (-80.0, 40.0)
QC_DEWPOINT_MAX_EXCESS_C = 1.0     # dewpoint may not exceed temperature by more than this
QC_PRESSURE_HPA = (850.0, 1090.0)
QC_WIND_MPH = (0.0, 200.0)
QC_RH = (0.0, 1.05)                # >1 up to 1.05 is clipped to 1; beyond = rejected
# MADIS data-descriptor (QC) flags that mark a value bad: X=failed, Q=questioned, B=subjectively bad
BAD_QC_FLAGS = {"X", "Q", "B"}

# ── Endpoint (NOAA MADIS, product OMO / HFMETAR, dataset family LDAD/hfmetar) ─
# Public hourly netCDF files YYYYMMDD_HH00.gz; no credentials needed for this path.
MADIS_HFMETAR_URL = os.getenv(
    "MADIS_HFMETAR_URL",
    "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/")
HTTP_USER_AGENT = "kalshi-trading-bot/1.0"
HTTP_TIMEOUT_SEC = 60
# Hourly files at least this recent are re-checked every poll (conditional GET);
# older hours are final and fetched once per process.
MADIS_REFRESH_RECENT_HOURS = 2

SHARED_INPUT_MODE = "ASOS_NOWCAST"
WEATHER_SOURCE_LABEL = "NOAA MADIS OMO/HFMETAR (KMIA ASOS)"
SOURCE_ID = "NOAA_MADIS_ASOS_HFMETAR"
SOURCE_LABEL_SHORT = "NOAA MADIS HFMETAR"

# Public series = observations whose minute is a multiple of 5.
PUBLIC_OBS_MINUTE_STEP = 5
# Valid MADIS temperature range (Kelvin); anything outside is rejected.
MADIS_TEMP_K_RANGE = (180.0, 340.0)
# How often the temperature monitor contacts NOAA (minutes). Completed hours and unchanged files cost
# almost nothing (cache + conditional GET), so 1 minute is cheap; set MADIS_POLL_MINUTES=5 for the old cadence.
POLL_EVERY_MINUTES = max(1, int(os.getenv("MADIS_POLL_MINUTES", "1")))
# Slots: minute % POLL_EVERY_MINUTES == POLL_MINUTE_OFFSET % POLL_EVERY_MINUTES
POLL_MINUTE_OFFSET = 1
POLL_RETRY_SECONDS = 45
