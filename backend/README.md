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

For fast KMIA current temperature and recorded high, also set:

```env
SYNOPTIC_TOKEN=your_synoptic_token
SYNOPTIC_STATION=KMIA
CRON_REFRESH_SECRET=choose-a-long-random-string
```

Do not put Kalshi credentials in the frontend.
