"""
NOAA MADIS OMO / HFMETAR client for KMIA (network + netCDF decoding + feed state).

Product : MADIS OMO, dataset family LDAD/hfmetar (high-frequency ASOS)
Access  : public hourly netCDF files
          https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/YYYYMMDD_HH00.gz
          (no credentials required for this path)

This is the ONLY live weather source.  There is deliberately no fallback to the
NWS station API, ordinary METAR, or any other provider.

Reliability rules (AsosFeed):
  * A failed poll is NOT stale weather.  The last valid KMIA observation is kept
    (in memory AND on disk) and its freshness is judged from its own observation
    time, never from whether the latest HTTP request succeeded.
  * The current and previous UTC hour files are re-checked on every poll (MADIS
    reprocesses both every ~5 minutes); older hours are cached once they are final.
  * A downloaded file only replaces/extends the cache after HTTP success,
    gzip decompression, netCDF decoding and record extraction all succeed.
    Cached records are merged by observation time, so they never shrink.
"""
import gzip
import io
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

from . import asos_parser, config as cfg

log = logging.getLogger("weather.asos")

_FILL_ABS = 1e30
STATE_PATH = Path(os.getenv("ASOS_STATE_PATH")
                  or Path(__file__).resolve().parent.parent / "modelscripts" / "asos_last_good.json")


def _headers():
    return {"User-Agent": cfg.HTTP_USER_AGENT, "Cache-Control": "no-cache", "Pragma": "no-cache"}


def hours_needed(now=None, lookback_hours=cfg.ASOS_LOOKBACK_HOURS):
    """Hours back to cover both the nowcast lookback and today's Miami-local day."""
    now = now or datetime.now(timezone.utc)
    midnight = now.astimezone(ZoneInfo(cfg.LOCAL_TZ_NAME)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    since_midnight = (now - midnight.astimezone(timezone.utc)).total_seconds() / 3600.0
    return max(float(lookback_hours), since_midnight) + 0.5


def hourly_files(now, hours):
    """Hourly file stamps (UTC hour starts) covering [now-hours, now], oldest first."""
    start = (now - timedelta(hours=hours)).replace(minute=0, second=0, microsecond=0)
    end = now.replace(minute=0, second=0, microsecond=0)
    out, t = [], start
    while t <= end:
        out.append(t)
        t += timedelta(hours=1)
    return out


def file_name(hour_start):
    return f"{hour_start:%Y%m%d_%H}00.gz"


# ── netCDF decoding ─────────────────────────────────────────────────────────

def _chars(arr):
    """char array (..., n) -> decoded stripped strings along the last axis."""
    a = np.ascontiguousarray(np.asarray(arr))
    n = a.shape[-1]
    flat = a.reshape(-1, n)
    out = [row.tobytes().replace(bytes(1), b"").decode("ascii", "ignore").strip() for row in flat]
    return np.array(out, dtype=object).reshape(a.shape[:-1])


def _val(x):
    if x is np.ma.masked:
        return None
    x = float(x)
    if x != x or abs(x) > _FILL_ABS or x == -9999.0:
        return None
    return x


def decode_hfmetar_ex(raw_gz, station):
    """gzip'd MADIS HFMETAR netCDF bytes -> (records for `station`, total records in file).

    Raises on a truncated/corrupt gzip or netCDF (callers must treat that as a failed download).
    """
    from scipy.io import netcdf_file          # classic netCDF3; no netCDF4/HDF5 dependency

    nc = netcdf_file(io.BytesIO(gzip.decompress(raw_gz)), mmap=False)
    v = nc.variables
    ids = _chars(v["stationId"][:])
    total = int(ids.shape[0])
    idx = np.where(ids == station)[0]
    if idx.size == 0:
        return [], total
    dd_vars = {"temperature": "temperatureDD", "dewpoint": "dewpointDD", "altimeter": "altimeterDD",
               "windDir": "windDirDD", "windSpeed": "windSpeedDD", "windGust": "windGustDD",
               "visibility": "visibilityDD"}
    dd = {name: _chars(v[var][:][idx][:, None]) for name, var in dd_vars.items() if var in v}
    sky = _chars(v["skyCvr"][:][idx]) if "skyCvr" in v else None
    wx = _chars(v["presWx"][:][idx]) if "presWx" in v else None

    def col(name):
        return v[name][:][idx] if name in v else None

    cols = {k: col(k) for k in ("observationTime", "temperature", "dewpoint", "windDir", "windSpeed",
                                "windGust", "altimeter", "visibility")}
    records = []
    for j in range(idx.size):
        t = _val(cols["observationTime"][j])
        if t is None:
            continue
        records.append({
            "station": station,
            "observation_time": t,
            "temperature": _val(cols["temperature"][j]),
            "dewpoint": _val(cols["dewpoint"][j]),
            "wind_dir": _val(cols["windDir"][j]),
            "wind_speed": _val(cols["windSpeed"][j]),
            "wind_gust": _val(cols["windGust"][j]) if cols["windGust"] is not None else None,
            "altimeter": _val(cols["altimeter"][j]),
            "visibility": _val(cols["visibility"][j]),
            "sky_cover": [c for c in (sky[j] if sky is not None else []) if c],
            "present_weather": (str(wx[j]) if wx is not None else "") or None,
            "dd": {name: str(arr[j]) for name, arr in dd.items()},
        })
    return records, total


def decode_hfmetar(raw_gz, station):
    return decode_hfmetar_ex(raw_gz, station)[0]


# ── feed state ──────────────────────────────────────────────────────────────

class AsosFeed:
    """Stateful MADIS HFMETAR poller: per-file record cache + persisted last-good observation."""

    def __init__(self, state_path=None):
        self.lock = threading.RLock()
        self.state_path = Path(state_path) if state_path else STATE_PATH
        self.cache = {}              # filename -> {"last_modified", "records", "final"}
        self.last_good = None        # newest successfully decoded normalized KMIA observation
        self.last_poll = None        # info dict of the most recent poll
        self.failed_at = {}          # filename -> time of last failed fetch (backoff for old hours)
        self._last_fetch_at = None   # when NOAA was last actually contacted
        self._state_loaded = False

    # -- persistence --------------------------------------------------------
    def _load_state(self, now):
        if self._state_loaded:
            return
        self._state_loaded = True
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            obs = data.get("last_good")
            if not obs or obs.get("station") != cfg.ASOS_STATION:
                return
            age = (now - asos_parser.parse_utc(obs["observed_at"])).total_seconds()
            if age <= cfg.ASOS_MAX_CURRENT_AGE_MINUTES * 60:     # real age, timestamp untouched
                self.last_good = obs
                log.info("restored persisted last-good KMIA obs %s (age %.0fs)", obs["observed_at"], age)
            else:
                log.info("persisted last-good KMIA obs %s too old (%.0fs); ignored", obs["observed_at"], age)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.warning("could not read persisted ASOS state %s: %s", self.state_path, exc)

    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"station": cfg.ASOS_STATION, "last_good": self.last_good}),
                           encoding="utf-8")
            os.replace(tmp, self.state_path)
        except Exception as exc:
            log.warning("could not persist ASOS state: %s", exc)

    # -- one file -----------------------------------------------------------
    def _fetch_file(self, name, hs, now, http):
        """Update the cache for one hourly file.  Returns a per-file result dict (never raises)."""
        res = {"file": name, "http_status": None, "bytes": 0, "decoded_records": None,
               "kmia_records": 0, "error": None}
        entry = self.cache.get(name)
        # current + previous hour are still being reprocessed; older hours only once final
        recheck = (now - hs) < timedelta(hours=cfg.MADIS_REFRESH_RECENT_HOURS) or not (entry and entry["final"])
        if entry and not recheck:
            res.update(http_status="cached", kmia_records=len(entry["records"]))
            return res
        is_recent = (now - hs) < timedelta(hours=cfg.MADIS_REFRESH_RECENT_HOURS)
        last_fail = self.failed_at.get(name)
        if not is_recent and last_fail and (now - last_fail) < timedelta(minutes=5):
            res.update(http_status="backoff", kmia_records=len(entry["records"]) if entry else 0)
            return res                           # old hour failed recently; do not hammer NOAA
        headers = _headers()
        if entry and entry.get("last_modified"):
            headers["If-Modified-Since"] = entry["last_modified"]
        try:
            resp = http.get(cfg.MADIS_HFMETAR_URL + name, headers=headers, timeout=cfg.HTTP_TIMEOUT_SEC)
            res["http_status"] = resp.status_code
            if resp.status_code == 304 and entry:
                res["kmia_records"] = len(entry["records"])
                return res
            if resp.status_code == 404:
                res["kmia_records"] = len(entry["records"]) if entry else 0
                return res                       # not published (yet): keep whatever is cached
            resp.raise_for_status()
            content = resp.content
            res["bytes"] = len(content)
            records, total = decode_hfmetar_ex(content, cfg.ASOS_STATION)   # raises on corrupt gzip/netCDF
            res["decoded_records"] = total
            if total == 0:
                raise ValueError("file decoded but contains no records")
            merged = {r["observation_time"]: r for r in (entry["records"] if entry else [])}
            merged.update({r["observation_time"]: r for r in records})
            self.cache[name] = {
                "last_modified": resp.headers.get("Last-Modified"),
                "records": [merged[k] for k in sorted(merged)],
                "final": (now - hs) >= timedelta(hours=2),
            }
            res["kmia_records"] = len(self.cache[name]["records"])
            self.failed_at.pop(name, None)
        except Exception as exc:
            self.failed_at[name] = now
            res["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("MADIS HFMETAR %s failed (%s); keeping previous cache/last-good", name, res["error"])
            if entry:
                res["kmia_records"] = len(entry["records"])
        return res

    # -- poll ---------------------------------------------------------------
    def poll(self, lookback_hours=cfg.ASOS_LOOKBACK_HOURS, now=None, session=None, reuse_seconds=0):
        """
        Returns (observations, info).  observations: QC'd, de-duplicated, ascending UTC KMIA
        public-5-minute observations built from the cached records plus last-good; never emptier
        than before because a download failed.

        reuse_seconds > 0: if NOAA was already contacted that recently, do not contact it again;
        answer from the cache (keeps secondary callers from polling NOAA aggressively).
        """
        now = now or datetime.now(timezone.utc)
        http = session or requests
        with self.lock:
            self._load_state(now)
            if (reuse_seconds and self._last_fetch_at and self.last_poll is not None
                    and (now - self._last_fetch_at).total_seconds() < reuse_seconds):
                info = {**self.last_poll, "reused_last_poll": True}
                return self._assemble(now, info, lookback_hours), info

            hours = hours_needed(now, lookback_hours)
            info = {"poll_started_at": now.isoformat(), "hours": hours, "sources": [], "errors": [],
                    "rejected": [], "files": [], "file_results": [], "requested_files": [],
                    "records_received": 0, "fetched_at": now.isoformat(), "reused_last_poll": False}
            cur_hour = now.replace(minute=0, second=0, microsecond=0)
            must = {cur_hour, cur_hour - timedelta(hours=1)}           # current + previous UTC hour, always
            info["current_file"] = file_name(cur_hour)
            info["previous_file"] = file_name(cur_hour - timedelta(hours=1))
            for hs in sorted(set(hourly_files(now, hours)) | must):
                name = file_name(hs)
                res = self._fetch_file(name, hs, now, http)
                info["requested_files"].append(name)
                info["file_results"].append(res)
                if res["error"]:
                    info["errors"].append(f"MADIS {name}: {res['error']}")
                elif res["http_status"] in (200, 304, "cached"):
                    info["files"].append(name)
            self._last_fetch_at = now
            info["last_poll_success"] = not info["errors"]
            observations = self._assemble(now, info, lookback_hours)
            info["fetch_error"] = "; ".join(info["errors"]) or None
            info["source_status"] = ("fetch_error_using_last_good" if info["errors"] and self.last_good
                                     else "fetch_error_no_observation" if info["errors"] else "ok")
            self.last_poll = info
            return observations, info

    def _assemble(self, now, info, lookback_hours):
        """Cache + last-good -> observations; advance last-good only to a strictly newer observation."""
        records = [r for e in self.cache.values() for r in e["records"]]
        info["records_received"] = len(records)
        rejected = info.setdefault("rejected", [])
        observations = asos_parser.parse_madis_records(records, station=cfg.ASOS_STATION, rejected=rejected)
        horizon = now + timedelta(minutes=5)
        observations = [o for o in observations if asos_parser.parse_utc(o["observed_at"]) <= horizon]
        info["kmia_found"] = len(observations)
        newest_downloaded = observations[-1] if observations else None
        info["newest_downloaded_observation"] = newest_downloaded["observed_at"] if newest_downloaded else None
        if records:
            info.setdefault("sources", [])
            if "MADIS_HFMETAR" not in info["sources"]:
                info["sources"].append("MADIS_HFMETAR")

        info["cache_updated"] = False
        if newest_downloaded is not None and (
                self.last_good is None or asos_parser.parse_utc(newest_downloaded["observed_at"])
                > asos_parser.parse_utc(self.last_good["observed_at"])):
            self.last_good = newest_downloaded
            self._save_state()
            info["cache_updated"] = True
        if self.last_good is not None:
            observations = asos_parser.deduplicate(observations + [self.last_good])
        info["last_good_observation"] = self.last_good["observed_at"] if self.last_good else None
        info["last_good_age_seconds"] = (
            round(max(0.0, (now - asos_parser.parse_utc(self.last_good["observed_at"])).total_seconds()), 1)
            if self.last_good else None)
        return observations

    def reset(self, state_path=None):
        with self.lock:
            self.cache.clear()
            self.last_good = None
            self.last_poll = None
            self.failed_at.clear()
            self._last_fetch_at = None
            self._state_loaded = False
            if state_path is not None:
                self.state_path = Path(state_path)


_feed = AsosFeed()
_cache = _feed.cache        # kept for tests / introspection


def get_feed():
    return _feed


def fetch_recent_observations(lookback_hours=cfg.ASOS_LOOKBACK_HOURS, now=None, session=None, reuse_seconds=0):
    return _feed.poll(lookback_hours=lookback_hours, now=now, session=session, reuse_seconds=reuse_seconds)


def get_latest_asos_observation(station_id=None, now=None, session=None, poll=True, reuse_seconds=0):
    """
    Public interface: the latest valid public 5-minute MADIS HFMETAR observation, or None if none has
    ever been obtained.  Callers need not know about gzip/netCDF/filenames.

    Never returns None merely because a download failed: the last good observation is kept, and
    `is_stale` says whether it is older than MADIS_STALE_MINUTES.
    """
    station_id = (station_id or cfg.ASOS_STATION).upper()
    if station_id != cfg.ASOS_STATION:
        raise ValueError(f"only the configured station {cfg.ASOS_STATION} is supported, got {station_id}")
    now = now or datetime.now(timezone.utc)
    if poll:
        _feed.poll(now=now, session=session, reuse_seconds=reuse_seconds)
    with _feed.lock:
        obs = _feed.last_good
        fetched = _feed._last_fetch_at
    if obs is None:
        return None
    obs_time = asos_parser.parse_utc(obs["observed_at"])
    age = max(0.0, (now - obs_time).total_seconds())
    return {
        "station": cfg.ASOS_STATION,
        "temperature_f": obs["temp_f"],
        "temperature_k": round(obs["temp_c"] + 273.15, 2),
        "observation_time_utc": obs_time,
        "fetched_at_utc": fetched or now,
        "age_seconds": round(age, 1),
        "source": cfg.SOURCE_LABEL_SHORT,
        "is_stale": age > cfg.ASOS_FRESH_MINUTES * 60,
    }


# ── diagnostics ─────────────────────────────────────────────────────────────

def cadence_distribution(observations):
    """Share of consecutive-observation gaps that are 1, 5, 10, 60 minutes or other."""
    ts = sorted(asos_parser.parse_utc(o["observed_at"]) for o in observations)
    gaps = [round((b - a).total_seconds() / 60.0, 2) for a, b in zip(ts, ts[1:])]
    n = len(gaps)
    buckets = {"1min": 0, "5min": 0, "10min": 0, "60min": 0, "other": 0}
    for g in gaps:
        key = {1: "1min", 5: "5min", 10: "10min", 60: "60min"}.get(round(g))
        if key is None or abs(g - round(g)) > 0.1:
            key = "other"
        buckets[key] += 1
    return {"intervals": n, "counts": buckets,
            "percent": {k: (round(100.0 * c / n, 1) if n else None) for k, c in buckets.items()}}


def diagnostics(now=None, observations=None, info=None, n_recent=10):
    from .asos_nowcast import freshness_status
    now = now or datetime.now(timezone.utc)
    if observations is None:
        observations, info = fetch_recent_observations(now=now, reuse_seconds=30)
    info = info or {}
    latest = observations[-1] if observations else None
    latest_dt = asos_parser.parse_utc(latest["observed_at"]) if latest else None
    age = round((now - latest_dt).total_seconds(), 1) if latest_dt else None
    return {
        "station": cfg.ASOS_STATION,
        "source": cfg.SOURCE_ID,
        "server_checked_at": now.isoformat(),
        "last_poll_at": info.get("poll_started_at", now.isoformat()),
        "last_poll_success": info.get("last_poll_success", not info.get("errors")),
        "last_poll_error": info.get("fetch_error") or (
            "; ".join(info["errors"]) if info.get("errors") else None),
        "source_status": info.get("source_status"),
        "requested_files": info.get("requested_files", []),
        "file_results": info.get("file_results", []),
        "kmia_records_found": info.get("records_received", len(observations)),
        "newest_downloaded_observation": info.get("newest_downloaded_observation"),
        "last_good_observation": ({"observed_at": latest["observed_at"], "temp_f": latest["temp_f"]}
                                  if latest else None),
        "latest_observed_at": latest["observed_at"] if latest else None,
        "observation_age_seconds": age,
        "freshness_status": freshness_status(age),
        "temperature_f": latest["temp_f"] if latest else None,
        "recent_observation_times": [o["observed_at"] for o in observations[-n_recent:]],
        "observations_received": len(observations),
        "cadence": cadence_distribution(observations),
        "errors": info.get("errors", []),
    }


def log_refresh(observations, info, now, debug=False):
    """[MADIS] diagnostic lines for every poll, including exactly why current would be None."""
    info = info or {}
    d = diagnostics(now, observations, info)
    P = "[MADIS]"
    lines = [f"station={d['station']}"]
    if info.get("reused_last_poll"):
        lines.append("reusing the last NOAA poll (queried recently)")
    lines += [f"checking current file={info.get('current_file')}",
              f"checking previous file={info.get('previous_file')}"]
    for r in info.get("file_results", []):
        if r["file"] in (info.get("current_file"), info.get("previous_file")):
            extra = f" error={r['error']}" if r["error"] else ""
            lines.append(f"{r['file']}: http={r['http_status']} bytes={r['bytes']} "
                         f"decoded_records={r['decoded_records']} kmia_records={r['kmia_records']}{extra}")
    for e in info.get("errors", []):
        lines.append(f"download failed: {e}")
    lines.append(f"found {info.get('kmia_found', len(observations))} {d['station']} public 5-minute observations")
    lg = info.get("last_good_observation")
    if lg is not None and observations:
        newest = observations[-1]
        if info.get("cache_updated"):
            lines += [f"latest valid 5m observation={newest['observed_at']}",
                      f"temperature={newest['temp_f']}F",
                      f"observation age={d['observation_age_seconds']}s",
                      "cache updated"]
        else:
            lines += [f"latest NOAA observation still {asos_parser.parse_utc(lg).strftime('%H:%MZ')}",
                      "cached observation retained"]
            if info.get("errors"):
                lines.append(f"retaining cached observation {asos_parser.parse_utc(lg).strftime('%H:%MZ')} / "
                             f"{newest['temp_f']}F (age {d['observation_age_seconds']}s)")
    lines.append(f"freshness={d['freshness_status']} source_status={info.get('source_status')}")
    if d["freshness_status"] == "unavailable":
        if not observations:
            lines.append("current=None because: no KMIA observation has ever been decoded and no usable "
                         "persisted last-good exists")
        else:
            lines.append(f"models paused: last good observation {d['latest_observed_at']} is "
                         f"{d['observation_age_seconds']}s old (> {cfg.ASOS_MAX_CURRENT_AGE_MINUTES:g} min); "
                         "temperature kept for display, marked stale")
    if debug:
        lines.append("last_10_observation_times=" + ",".join(
            asos_parser.parse_utc(o["observed_at"]).strftime("%H:%M") for o in observations[-10:]))
    for line in lines:
        print(f"{P} {line}", flush=True)
