# Kalshi Clone Backend

Small Django API wrapper around Kalshi public market data.

## Run

```powershell
..\kalshibot\Scripts\python.exe manage.py runserver 8000
```

## Endpoints

```txt
GET /api/kalshi/health/
GET /api/kalshi/miami-temperature/
GET /api/kalshi/miami-temperature/?market_ticker=KXHIGHMIA-26JUN15-B93.5
GET /api/kalshi/miami-temperature/stream/
GET /api/kalshi/paper/state/
POST /api/kalshi/paper/order/
POST /api/kalshi/paper/reset/
```

## Kalshi WebSocket credentials

Copy `.env.example` to `.env` and fill in your Kalshi API key id and private key path.

```powershell
Copy-Item .env.example .env
```

## Weather data (NOAA ASOS only)

No weather API key is required.

| Purpose | Source |
| --- | --- |
| Live observations (KMIA, public 5-minute series: obs minute % 5 == 0; time from MADIS `observationTime`) | NOAA MADIS OMO/HFMETAR (`LDAD/hfmetar` hourly netCDF, public, no credentials). Only source; no fallback |
| Model-fitting history (hourly, 21d / 90d, cached weekly) | ASOS archive via Iowa Environmental Mesonet |
| 15/30/45/60-minute environmental inputs | derived from the last ~3 h of KMIA ASOS observations (`weather/asos_nowcast.py`) |
| Solar intensity | computed from time, latitude/longitude, day of year |

ASOS is fetched once per forecast cycle and shared with all model subprocesses.
Freshness is judged from the last valid observation's own timestamp (<=15 min healthy, <=25 min delayed,
older = `weather_unavailable`), never from whether the latest HTTP poll succeeded. A failed poll keeps serving
the last good observation (also persisted to `modelscripts/asos_last_good.json` and restored on restart if still
within the limit) and the API reports `source_status: fetch_error_using_last_good`. No other provider is substituted.

Polling: NOAA is contacted every minute by default (`MADIS_POLL_MINUTES=1`; unchanged hourly files return HTTP 304, so it is cheap; set 5 for HH:01, :06, :11, ... with one retry after 45 s);
the forecast cycle and manual refresh reuse that poll instead of contacting NOAA again.

Optional settings:

```env
CRON_REFRESH_SECRET=choose-a-long-random-string
ASOS_STATION=KMIA
ASOS_FRESH_MINUTES=15           # <= healthy
ASOS_MAX_CURRENT_AGE_MINUTES=25 # <= delayed; older = weather_unavailable
ASOS_DEBUG=1                    # also log the last 10 observation times
```

Diagnostics: `GET /api/kalshi/weather/diagnostics/` (latest observation, age, recent timestamps, cadence histogram).

Tests: `python -m pytest tests`

Do not put Kalshi credentials in the frontend.
