# Railway Hobby deployment

This repository deploys as two Railway services in one project.

## 1. Create the services

Create an empty Railway project and add two services connected to this GitHub
repository:

| Service | Root directory | Config file |
| --- | --- | --- |
| `Backend` | `/backend` | `/backend/railway.json` |
| `Frontend` | `/frontend` | `/frontend/railway.json` |

Set each service's Railway config-file path explicitly. Railway config-file
paths are resolved from the repository root, not from the service root
directory.

Keep the Backend service at one replica. Its SQLite database and background
tasks are intentionally single-instance.

## 2. Add persistent Backend storage

Attach a Railway volume to `Backend` with this mount path:

```text
/data
```

Migrations are part of the Backend start command because Railway volumes are
not mounted in pre-deploy containers.

## 3. Configure Backend variables

Required:

```env
PORT=8000
DJANGO_DEBUG=false
DJANGO_SECRET_KEY=replace-with-a-long-random-value
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,.railway.app,.railway.internal,healthcheck.railway.app
DATABASE_PATH=/data/db.sqlite3
RUN_BACKGROUND_TASKS=true
TZ=UTC
```

Weather integrations:

```env
ACCUWEATHER_API_KEY=
SYNOPTIC_TOKEN=
SYNOPTIC_STATION=KMIA
CRON_REFRESH_SECRET=
```

Optional authenticated Kalshi WebSocket access:

```env
KALSHI_API_KEY_ID=
KALSHI_PRIVATE_KEY_PEM=
```

Use `KALSHI_PRIVATE_KEY_PEM`, not a local key path. The value may contain real
newlines or escaped `\n` sequences. Never commit it.

Optional Google Sheets history:

```env
GOOGLE_SERVICE_ACCOUNT_JSON=
GOOGLE_SHEETS_ID=
GOOGLE_TRADES_SHEET_ID=
```

The Google service-account email must have access to the selected sheets.

## 4. Configure Frontend variables

```env
PORT=3000
NODE_ENV=production
BACKEND_URL=http://${{Backend.RAILWAY_PRIVATE_DOMAIN}}:${{Backend.PORT}}
```

`Backend` in the reference must exactly match the Backend service name.

Generate a public Railway domain for `Frontend`. The Backend can remain private.

## 5. Verify

The Backend deployment health check is:

```text
/api/kalshi/health/
```

Once the Frontend is public, the same endpoint should be reachable through its
rewrite:

```text
https://YOUR-FRONTEND-DOMAIN/api/kalshi/health/
```

It should return:

```json
{"ok": true}
```

Do not enable Railway Serverless for the Backend: sleeping stops its forecast,
temperature-monitor, and price-tracker threads.

The paper-order, paper-reset, and algorithm POST endpoints do not currently
authenticate users. Do not expose live-money order execution through this
application without first adding authentication and authorization.
