import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from weather import asos_nowcast, asos_parser, config as cfg, meteo

BACKEND = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)   # 14:00 Miami (EDT)


# ── fixtures ────────────────────────────────────────────────────────────────

def madis_rec(ts, temp_c=30.0, dew_c=24.0, wdir=90, wspd_kmh=18.0, slp_pa=101500,
              clouds=("FEW",), qc=None, vis_m=16093, station="KMIA"):
    """One MADIS HFMETAR record in native units (K, m/s, Pa, m) as decode_hfmetar yields it."""
    return {
        "station": station,
        "observation_time": ts.timestamp(),
        "temperature": None if temp_c is None else temp_c + 273.15,
        "dewpoint": None if dew_c is None else dew_c + 273.15,
        "wind_dir": wdir,
        "wind_speed": None if wspd_kmh is None else wspd_kmh / 3.6,
        "wind_gust": None,
        "altimeter": slp_pa,
        "visibility": vis_m,
        "sky_cover": list(clouds),
        "present_weather": None,
        "dd": {"temperature": qc or "V", "dewpoint": "V", "windDir": "V", "windSpeed": "V",
               "altimeter": "V", "visibility": "V"},
    }


def floor5(dt):
    """Floor to the 5-minute observation boundary."""
    return dt.replace(minute=dt.minute // 5 * 5, second=0, microsecond=0)


def parse(recs, **kw):
    return asos_parser.parse_madis_records(recs, **kw)


def normalized(n=36, fn=None, **base):
    recs = []
    for i in range(n):
        ts = NOW - timedelta(minutes=5 + 5 * (n - 1 - i))
        kw = dict(base)
        if fn:
            kw.update(fn(i))
        recs.append(madis_rec(ts, **kw))
    return parse(recs)


# ── parsing / conversions ───────────────────────────────────────────────────

def test_parse_madis_fields():
    obs = parse([madis_rec(NOW)])[0]
    assert obs["observed_at"] == NOW.isoformat()
    assert obs["temp_c"] == 30.0 and obs["temp_f"] == 86.0
    assert obs["dewpoint_f"] == pytest.approx(75.2, abs=0.01)
    assert obs["wind_speed_mph"] == pytest.approx(11.18, abs=0.01)     # 18 km/h
    assert obs["wind_dir_deg"] == 90
    assert obs["pressure_hpa"] == 1015.0
    assert obs["cloud_fraction"] == 0.2
    assert obs["visibility_miles"] == pytest.approx(10.0, abs=0.01)
    assert obs["source"] == "NOAA_MADIS_ASOS_HFMETAR" and obs["station"] == "KMIA"
    assert obs["report_source"] == "MADIS_HFMETAR"


def test_other_stations_are_ignored():
    assert parse([madis_rec(NOW, station="KFLL", temp_c=10.0)]) == []


def test_unit_conversions():
    assert meteo.c_to_f(100) == 212 and meteo.f_to_c(32) == 0
    assert meteo.c_to_f(-40) == -40
    assert meteo.ms_to_mph(10) == pytest.approx(22.369, abs=1e-3)
    assert meteo.kmh_to_mph(1.609344) == pytest.approx(1.0)
    assert meteo.kt_to_mph(10) == pytest.approx(11.5078, abs=1e-3)
    assert meteo.pa_to_hpa(101325) == pytest.approx(1013.25)
    assert meteo.inhg_to_hpa(29.92) == pytest.approx(1013.2, abs=0.1)


def test_rh_from_temp_dewpoint():
    assert meteo.rh_from_t_td(25, 25) == pytest.approx(1.0)
    assert meteo.rh_from_t_td(30, 24) == pytest.approx(0.7, abs=0.02)
    # HFMETAR carries no RH: derived from T/Td, never invented without a dew point
    obs = parse([madis_rec(NOW)])[0]
    assert obs["relative_humidity"] == pytest.approx(0.7, abs=0.02)
    assert parse([madis_rec(NOW, dew_c=None)])[0]["relative_humidity"] is None
    # round trip
    assert meteo.dewpoint_c_from_t_rh(30, meteo.rh_from_t_td(30, 24)) == pytest.approx(24, abs=0.01)


@pytest.mark.parametrize("d", [0, 1, 359, 360])
def test_wind_uv_near_north(d):
    u, v = meteo.wind_to_uv(10.0, d)
    assert math.hypot(u, v) == pytest.approx(10.0)
    assert v < 0                                  # wind from the north blows southward
    assert abs(u) <= 10 * math.sin(math.radians(1.0)) + 1e-9


def test_wind_uv_wraparound_and_cardinals():
    assert meteo.wind_to_uv(10, 0) == pytest.approx(meteo.wind_to_uv(10, 360), abs=1e-9)
    u359, v359 = meteo.wind_to_uv(10, 359)
    u1, v1 = meteo.wind_to_uv(10, 1)
    assert u359 == pytest.approx(-u1) and v359 == pytest.approx(v1)     # symmetric about north
    u, v = meteo.wind_to_uv(10, 90)                                      # from the east
    assert (u, v) == pytest.approx((-10.0, 0.0), abs=1e-9)
    spd, direc = meteo.uv_to_speed_dir(*meteo.wind_to_uv(10, 359))
    assert spd == pytest.approx(10) and direc == pytest.approx(359)
    assert meteo.wind_to_uv(0, None) == (0.0, 0.0)           # calm needs no direction
    assert meteo.wind_to_uv(5, None) == (None, None)         # variable: do not invent


# ── dedup / QC ──────────────────────────────────────────────────────────────

def test_dedup_prefers_most_complete_and_is_deterministic():
    sparse = madis_rec(NOW, clouds=(), slp_pa=None)
    full = madis_rec(NOW + timedelta(seconds=20))                     # same minute
    a = parse([sparse, full])
    b = parse([full, sparse])
    assert len(a) == 1 and a == b
    assert a[0]["pressure_hpa"] is not None and a[0]["cloud_fraction"] is not None


def test_dedup_sorts_by_observed_at_and_removes_duplicates():
    recs = [madis_rec(NOW - timedelta(minutes=m)) for m in (10, 0, 5, 5, 0, 10)]   # shuffled + duplicated
    out = parse(recs)
    times = [o["observed_at"] for o in out]
    assert len(out) == 3 and times == sorted(times)


def test_qc_rejects_bad_values_without_dropping_valid_extremes():
    rej = []
    bad_t = madis_rec(NOW, temp_c=80)                       # impossible
    flagged = madis_rec(NOW - timedelta(minutes=5), qc="X")  # MADIS data-descriptor X = failed QC
    ok_hot = madis_rec(NOW - timedelta(minutes=10), temp_c=41.0, dew_c=25.0)   # valid extreme
    out = parse([bad_t, flagged, ok_hot], rejected=rej)
    assert [o["temp_c"] for o in out] == [41.0]
    assert len(rej) == 2 and all("reason" in r for r in rej)
    # bad dew point drops only that field
    rej = []
    obs = parse([madis_rec(NOW, dew_c=35.0)], rejected=rej)[0]
    assert obs["dewpoint_c"] is None and obs["temp_c"] == 30.0 and rej


def test_naive_timestamps_become_utc_aware():
    assert asos_parser.parse_utc("2026-07-15T18:00:00").tzinfo is not None
    assert asos_parser.parse_utc(NOW.timestamp()) == NOW


# ── freshness ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("age_min,status", [
    (2, "healthy"), (15, "healthy"), (15.5, "delayed"), (25, "delayed"), (25.5, "unavailable"), (90, "unavailable")])
def test_freshness_thresholds(age_min, status):
    assert asos_nowcast.freshness_status(age_min * 60) == status
    assert cfg.ASOS_FRESH_MINUTES == 15 and cfg.ASOS_MAX_CURRENT_AGE_MINUTES == 25


def test_shared_input_stale_is_unavailable_and_not_fed_to_models():
    obs = normalized()
    shared = asos_nowcast.build_shared_input(obs, now=NOW + timedelta(minutes=45))
    assert shared["metadata"]["status"] == "unavailable"
    assert shared["current"] is None and shared["last_known"] is not None
    assert shared["future_features"] == []
    assert shared["metadata"]["latest_observation_age_seconds"] > 25 * 60


def test_shared_input_no_observations():
    shared = asos_nowcast.build_shared_input([], now=NOW)
    assert shared["metadata"]["status"] == "unavailable" and shared["current"] is None


def test_shared_input_fresh_schema():
    shared = asos_nowcast.build_shared_input(normalized(), {"sources": ["MADIS_HFMETAR"]}, now=NOW)
    md = shared["metadata"]
    assert md["status"] == "healthy" and md["source"] == "NOAA_MADIS_ASOS_HFMETAR"
    assert md["forecast_input_mode"] == "ASOS_NOWCAST" and md["latest_observation_age_seconds"] == 300
    assert shared["station"] == "KMIA" and shared["current"]["temp_f"] is not None
    assert len(shared["recent_observations"]) == 36
    json.dumps(shared)                                      # must be JSON-serializable


# ── nowcast ─────────────────────────────────────────────────────────────────

def test_future_features_15_30_45_60():
    shared = asos_nowcast.build_shared_input(normalized(), now=NOW)
    ff = shared["future_features"]
    assert [f["minutes_ahead"] for f in ff] == [15, 30, 45, 60]
    for f in ff:
        assert 0 <= f["relative_humidity"] <= 1 and 0 <= f["cloud_fraction"] <= 1
        assert 850 < f["pressure_hpa"] < 1090
        assert f["wind_u_mph"] is not None and f["wind_speed_mph"] >= 0


def test_trend_extrapolation_is_damped_and_clipped():
    # RH rising 1%/5min = 12%/h; pressure falling 1 hPa/5min = 12 hPa/h (absurd -> clipped)
    obs = normalized(fn=lambda i: {"dew_c": 20 + 0.0 * i, "slp_pa": 101500 - 100 * i})
    shared = asos_nowcast.build_shared_input(obs, now=NOW)
    p0 = shared["current"]["pressure_hpa"]
    p60 = shared["future_features"][-1]["pressure_hpa"]
    assert p0 - p60 <= cfg.MAX_RATE_PER_HOUR["pressure_hpa"] * cfg.TREND_DAMPING_TAU_HOURS + 1e-6
    assert p60 < p0                                           # direction preserved
    rh = [f["relative_humidity"] for f in shared["future_features"]]
    assert all(0 <= r <= 1 for r in rh)


def test_rh_clipped_at_one():
    obs = normalized(fn=lambda i: {"temp_c": 25.0, "dew_c": 20 + 0.14 * i})   # near saturation, rising
    obs = [o for o in obs]
    shared = asos_nowcast.build_shared_input(obs, now=NOW)
    assert all(f["relative_humidity"] <= 1.0 for f in shared["future_features"])


def test_wind_trend_uses_uv_not_direction():
    # Direction oscillating across the 359/1 boundary must not look like a 358 deg swing.
    obs = normalized(fn=lambda i: {"wdir": 359 if i % 2 else 1, "wspd_kmh": 18.0})
    shared = asos_nowcast.build_shared_input(obs, now=NOW)
    ff = shared["future_features"]
    for f in ff:
        assert f["wind_speed_mph"] == pytest.approx(11.18, abs=0.7)
        assert f["wind_v_mph"] < -10                           # still from the north
    assert abs(shared["features"]["wind_u_trend_mph_per_hr"]) < 1.0


def test_missing_fields_do_not_crash_and_are_not_fabricated():
    obs = normalized(fn=lambda i: {"clouds": (), "slp_pa": None, "wdir": None})
    shared = asos_nowcast.build_shared_input(obs, now=NOW)
    cur = shared["current"]
    assert cur["cloud_fraction"] is None and cur["pressure_hpa"] is None
    assert cur["wind_dir_deg"] is None and cur["wind_u_mph"] is None
    assert cur["relative_humidity"] is not None                # derivable from T/Td
    for f in shared["future_features"]:
        assert f["cloud_fraction"] is None and f["pressure_hpa"] is None
        assert f["wind_u_mph"] is None and f["relative_humidity"] is not None
    json.dumps(shared)


def test_sparse_history_gives_no_trend_but_still_works():
    shared = asos_nowcast.build_shared_input(normalized(n=2), now=NOW)
    assert shared["features"]["temp_trend_60min_f_per_hr"] is None
    assert len(shared["future_features"]) == 4                 # persistence only


def test_features_temperature_trend_signs():
    obs = normalized(fn=lambda i: {"temp_c": 25 + 0.05 * i})   # warming 0.6 C/h
    f = asos_nowcast.build_shared_input(obs, now=NOW)["features"]
    assert f["temp_trend_60min_f_per_hr"] > 0 and f["temp_change_15min_f"] > 0
    assert f["temp_volatility_f"] is not None


# ── today's high / time zones ───────────────────────────────────────────────

def test_todays_high_uses_miami_day_not_utc_day_and_dst():
    # Miami is UTC-4 in July: 03:30Z on Jul 15 is still Jul 14 locally (yesterday).
    yday = parse([
        madis_rec(datetime(2026, 7, 15, 3, 30, tzinfo=timezone.utc), temp_c=38.0, dew_c=24.0)])
    today = parse([
        madis_rec(datetime(2026, 7, 15, 5, 0, tzinfo=timezone.utc), temp_c=27.0, dew_c=24.0),
        madis_rec(datetime(2026, 7, 15, 17, 0, tzinfo=timezone.utc), temp_c=33.0),
        madis_rec(datetime(2026, 7, 15, 17, 0, 30, tzinfo=timezone.utc), temp_c=33.0)])   # duplicate minute
    h = asos_nowcast.todays_high(yday + today, NOW)
    assert h["miami_date"] == "2026-07-15" and h["high_f"] == meteo.c_to_f(33.0) and h["n_obs"] == 2
    # Winter (EST, UTC-5): 04:30Z is still the previous local day.
    winter_now = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)
    w = parse([
        madis_rec(datetime(2026, 1, 15, 4, 30, tzinfo=timezone.utc), temp_c=30.0, dew_c=20.0),
        madis_rec(datetime(2026, 1, 15, 5, 30, tzinfo=timezone.utc), temp_c=20.0, dew_c=15.0)])
    assert asos_nowcast.todays_high(w, winter_now)["high_f"] == meteo.c_to_f(20.0)


# ── no AccuWeather / no other provider ──────────────────────────────────────

def test_operational_code_has_no_accuweather_or_other_providers():
    banned = ["accu" + "weather", "ACCU" + "WEATHER", "synoptic" + "data", "SYNOP" + "TIC",
              "open" + "weathermap", "api.weather" + ".gov", "aviation" + "weather.gov", "stations/KM" + "IA/observations", "weather" + "api.com", "tomorrow" + ".io", "visual" + "crossing"]
    roots = [BACKEND / "weather", BACKEND / "modelscripts", BACKEND / "kalshi_api",
             BACKEND / "config", BACKEND / "railway.json", BACKEND / "requirements.txt",
             BACKEND.parent / "frontend" / "app"]
    offenders = []
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for p in files:
            if p.suffix in {".py", ".json", ".txt", ".tsx", ".ts", ".md"} and "__pycache__" not in p.parts:
                text = p.read_text(encoding="utf-8", errors="ignore")
                # weather/nws_high.py is the one sanctioned NWS reader (recorded high only)
                allowed = {"api.weather" + ".gov", "stations/KM" + "IA/observations"} if p.name == "nws_high.py" else set()
                offenders += [f"{p}: {b}" for b in banned if b not in allowed and b.lower() in text.lower()]
    assert not offenders, offenders


# ── model end-to-end ────────────────────────────────────────────────────────

def _history_df(n=200):
    import pandas as pd
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    hours = t.hour.values
    temp = 80 + 8 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.6, n)
    return pd.DataFrame(dict(
        time=t, temp_f=temp, dewpoint_f=74 + rng.normal(0, 1, n), cloud=rng.uniform(0, 1, n),
        humidity=rng.uniform(0.5, 0.95, n), wind_speed=rng.uniform(2, 14, n), wind_dir=rng.uniform(0, 360, n),
        wind_u=rng.normal(0, 5, n), wind_v=rng.normal(0, 5, n),
        pressure_hpa=1015 + np.cumsum(rng.normal(0, 0.3, n)) * 0.2, uv_index=0.0))


def _assert_valid_distribution(res, shared):
    assert np.isfinite(res["mean"]) and np.isfinite(res["std"])
    probs = np.array(res["probs"]) if "probs" in res else None
    assert probs is not None and np.all(np.isfinite(probs)) and np.all(probs >= 0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    temps = np.array(res["temps"])
    assert temps.min() >= -20 and temps.max() <= 135
    assert temps.min() >= math.floor((shared["today"]["high_f"] or -99) + 0.5) - 1   # floor logic holds


@pytest.mark.parametrize("script,horizon", [("ou_forecast", 1), ("ou_forecast", 3), ("var_forecast", 1), ("var_forecast", 3)])
def test_models_run_on_asos_shared_input(script, horizon, tmp_path, monkeypatch):
    sys.path.insert(0, str(BACKEND / "modelscripts"))
    mod = __import__(script)
    shared = asos_nowcast.build_shared_input(normalized(), now=NOW)
    # Model "now" is the shared generated_at, so sub-step hours/timezones are consistent.
    res = mod.compute(_history_df(), "test history", shared, horizon)
    assert res["T0"] == shared["current"]["temp_f"]
    _assert_valid_distribution({"mean": res["mean"], "std": res["std"], "probs": res["probs"],
                                "temps": res["temps"]}, shared)


def test_models_survive_missing_cloud_wind_pressure():
    sys.path.insert(0, str(BACKEND / "modelscripts"))
    import ou_forecast, var_forecast
    obs = normalized(fn=lambda i: {"clouds": (), "slp_pa": None, "wdir": None})
    shared = asos_nowcast.build_shared_input(obs, now=NOW)
    for mod in (ou_forecast, var_forecast):
        res = mod.compute(_history_df(), "test history", shared, 2)
        assert np.isfinite(res["mean"])


def test_model_refuses_stale_weather():
    sys.path.insert(0, str(BACKEND / "modelscripts"))
    import ou_forecast
    from weather.model_input import WeatherUnavailable
    shared = asos_nowcast.build_shared_input(normalized(), now=NOW + timedelta(hours=2))
    with pytest.raises(WeatherUnavailable):
        ou_forecast.compute(_history_df(), "x", shared, 1)


def test_model_subprocess_contract(tmp_path):
    """Run the real script via the orchestrator contract: ASOS_SHARED_DATA json -> stdout json."""
    now = floor5(datetime.now(timezone.utc)) + timedelta(minutes=2)       # obs stay on 5-minute marks
    shift = floor5(now) - NOW                    # re-time observations to be fresh against the wall clock
    obs = normalized()
    for o in obs:
        o["observed_at"] = (asos_parser.parse_utc(o["observed_at"]) + shift).isoformat()
    shared = asos_nowcast.build_shared_input(obs, now=now)
    assert shared["metadata"]["status"] == "healthy"
    p = tmp_path / "shared.json"
    p.write_text(json.dumps(shared))
    import pandas as pd
    hist = _history_df(300)
    hist["time"] = hist["time"].dt.tz_localize(None)
    hist.to_csv(tmp_path / "kmia_asos_21d.csv", index=False)
    env = {**os.environ, "ASOS_SHARED_DATA": str(p), "ASOS_CACHE_DIR": str(tmp_path)}
    proc = subprocess.run([sys.executable, str(BACKEND / "modelscripts" / "ou_forecast.py"), "--json-output",
                           "--horizon", "1", "--model-name", "ou_1h"],
                          capture_output=True, text=True, env=env, timeout=120, cwd=str(BACKEND / "modelscripts"))
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["model"] == "ou_1h" and out["forecast_input_mode"] == "ASOS_NOWCAST"
    assert out["weather_source"].startswith("NOAA MADIS OMO/HFMETAR") and out["station"] == "KMIA"
    assert sum(out["probs"]) == pytest.approx(1.0, abs=1e-6)


# ══ MADIS HFMETAR specifics ═════════════════════════════════════════════════

T13 = datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc)
HF_FIXTURE = [(0, 84.2), (5, 84.2), (10, 84.4), (15, 84.4), (20, 84.6)]     # minute, deg F


def hf_records(shuffle=False):
    recs = [madis_rec(T13 + timedelta(minutes=m), temp_c=meteo.f_to_c(f)) for m, f in HF_FIXTURE]
    return list(reversed(recs)) if shuffle else recs


def test_newest_observation_wins_even_if_it_arrives_first_or_last():
    for shuffle in (False, True):
        obs = parse(hf_records(shuffle))
        assert [o["observed_at"][11:16] for o in obs] == ["13:00", "13:05", "13:10", "13:15", "13:20"]
        shared = asos_nowcast.build_shared_input(obs, now=T13 + timedelta(minutes=26))
        assert shared["current"]["temp_f"] == pytest.approx(84.6, abs=0.01)
        assert shared["current"]["observed_at"] == (T13 + timedelta(minutes=20)).isoformat()
        assert shared["current"]["observed_at"].endswith("+00:00")          # tz-aware UTC


def test_hf_fixture_duplicates_removed_and_all_timestamps_utc_aware():
    obs = parse(hf_records() + hf_records())
    assert len(obs) == 5
    assert all(asos_parser.parse_utc(o["observed_at"]).utcoffset() == timedelta(0) for o in obs)


def _fake_madis_session(calls, fail=False):
    class Resp:
        status_code = 404
        headers = {}
        content = b""

        def raise_for_status(self):
            pass

    class S:
        def get(self, url, **kw):
            calls.append(url)
            if fail:
                raise RuntimeError("MADIS down")
            return Resp()
    return S()


def test_fetch_uses_only_madis_and_never_falls_back_when_madis_is_down():
    from weather import asos_client
    asos_client._cache.clear()
    calls = []
    obs, info = asos_client.fetch_recent_observations(now=T13 + timedelta(minutes=6),
                                                      session=_fake_madis_session(calls, fail=True))
    assert obs == [] and info["errors"] and info["sources"] == []
    assert calls and all(u.startswith(cfg.MADIS_HFMETAR_URL) for u in calls)       # nothing else was contacted
    shared = asos_nowcast.build_shared_input(obs, info, now=T13 + timedelta(minutes=6))
    assert shared["metadata"]["status"] == "unavailable" and shared["current"] is None


def test_decode_hfmetar_netcdf_roundtrip():
    """Build a classic-netCDF HFMETAR-like file and decode it with the production decoder."""
    import gzip
    import io
    from scipy.io import netcdf_file
    from weather import asos_client

    stations = ["KFLL", "KMIA", "KMIA", "KMIA"]
    times = [T13 + timedelta(minutes=m) for m in (0, 0, 5, 10)]
    buf = io.BytesIO()
    nc = netcdf_file(buf, "w")
    nc.createDimension("recNum", 4)
    nc.createDimension("staIdLen", 6)
    nc.createDimension("sky", 3)
    nc.createDimension("skyLen", 8)

    def put(name, dims, dtype, data):
        v = nc.createVariable(name, dtype, dims)
        v[:] = data

    ids = np.array([list(s.ljust(6, chr(0))) for s in stations], dtype="S1")
    put("stationId", ("recNum", "staIdLen"), "c", ids)
    put("observationTime", ("recNum",), "d", np.array([t.timestamp() for t in times]))
    put("temperature", ("recNum",), "f", np.array([300.0, 302.15, 302.15, 302.35], dtype="f4"))
    put("dewpoint", ("recNum",), "f", np.array([298.15, 298.15, 3.4028235e38, 298.15], dtype="f4"))
    put("windDir", ("recNum",), "f", np.array([110, 110, 110, 100], dtype="f4"))
    put("windSpeed", ("recNum",), "f", np.array([5, 5, 5, 5], dtype="f4"))
    put("altimeter", ("recNum",), "f", np.array([101490.0] * 4, dtype="f4"))
    put("visibility", ("recNum",), "f", np.array([16093.0] * 4, dtype="f4"))
    put("temperatureDD", ("recNum",), "c", np.array([b"V", b"V", b"V", b"X"], dtype="S1"))
    sky = np.zeros((4, 3, 8), dtype="S1")
    sky[:, 0, :3] = np.array([b"F", b"E", b"W"], dtype="S1")
    put("skyCvr", ("recNum", "sky", "skyLen"), "c", sky)
    nc.flush()
    raw = buf.getvalue()
    nc.close()
    recs = asos_client.decode_hfmetar(gzip.compress(raw), "KMIA")
    assert len(recs) == 3 and all(r["station"] == "KMIA" for r in recs)
    assert recs[0]["temperature"] == pytest.approx(302.15, abs=0.01) and recs[0]["sky_cover"] == ["FEW"]
    assert recs[1]["dewpoint"] is None                       # fill value -> missing, not fabricated
    obs = parse(recs)                                        # X-flagged temperature record dropped by QC
    assert [o["observed_at"][11:16] for o in obs] == ["13:00", "13:05"]
    assert obs[0]["temp_c"] == pytest.approx(29.0, abs=0.01)
    assert obs[0]["dewpoint_c"] == pytest.approx(25.0, abs=0.01) and obs[1]["dewpoint_c"] is None


def test_cadence_distribution():
    from weather import asos_client
    obs = parse([madis_rec(T13 + timedelta(minutes=m)) for m in (0, 5, 10, 15, 25, 85, 90)])
    d = asos_client.cadence_distribution(obs)
    assert d["intervals"] == 6
    assert d["counts"] == {"1min": 0, "5min": 4, "10min": 1, "60min": 1, "other": 0}


def test_diagnostics_shape():
    from weather import asos_client
    now = T13 + timedelta(minutes=26)
    d = asos_client.diagnostics(now, parse(hf_records()), {"errors": []})
    assert d["station"] == "KMIA" and d["source"] == "NOAA_MADIS_ASOS_HFMETAR"
    assert d["temperature_f"] == pytest.approx(84.6, abs=0.01) and d["observation_age_seconds"] == 360
    assert len(d["recent_observation_times"]) == 5


# ── orchestrator: one authoritative observation shared by all six models ────

def test_all_six_models_receive_the_same_current_observation(monkeypatch):
    from kalshi_api import forecast
    now = floor5(datetime.now(timezone.utc)) + timedelta(minutes=2)
    recs = [madis_rec(floor5(now) - timedelta(minutes=5 * k), temp_c=meteo.f_to_c(84.0 + 0.2 * (6 - k)))
            for k in range(6)]
    obs = parse(recs)
    info = {"errors": [], "sources": ["MADIS_HFMETAR"], "rejected": []}
    monkeypatch.setattr(forecast, "fetch_recent_observations", lambda now=None, **kw: (obs, info))
    seen = {}

    def fake_run(model, shared_path=None):
        with open(shared_path, encoding="utf-8") as f:
            seen[model["name"]] = json.load(f)["current"]
        return model["name"], {"model": model["name"], "T0": seen[model["name"]]["temp_f"],
                               "temps": [84], "probs": [1.0]}

    monkeypatch.setattr(forecast, "_run_one_model", fake_run)
    monkeypatch.setattr(forecast, "_record_forecast_snapshot", lambda r: None)
    monkeypatch.setattr(forecast, "_apply_asos_current", lambda s: None)
    forecast._cache["running"] = False
    forecast._do_refresh()
    assert sorted(seen) == sorted(m["name"] for m in forecast.MODELS) and len(seen) == 6
    newest = max(obs, key=lambda o: o["observed_at"])
    assert all(c == newest for c in seen.values())                 # identical object for every model
    assert forecast._cache["results"]["ou_1h"]["T0"] == newest["temp_f"]   # current.temp_f -> model T0
    assert newest["temp_f"] == pytest.approx(85.2, abs=0.01)


def test_stale_madis_makes_models_not_run(monkeypatch):
    from kalshi_api import forecast
    obs = parse([madis_rec(floor5(datetime.now(timezone.utc)) - timedelta(minutes=40))])
    info = {"errors": [], "sources": ["MADIS_HFMETAR"], "rejected": []}
    monkeypatch.setattr(forecast, "fetch_recent_observations", lambda now=None, **kw: (obs, info))
    ran = []
    monkeypatch.setattr(forecast, "_run_one_model", lambda m, p=None: ran.append(m) or (m["name"], {}))
    monkeypatch.setattr(forecast, "_record_forecast_snapshot", lambda r: None)
    forecast._cache["running"] = False
    forecast._do_refresh()
    assert ran == [] and all("weather_unavailable" in v["error"] for v in forecast._cache["results"].values())
    assert forecast._cache["weather"]["status"] == "unavailable"


def test_model_T0_equals_current_temp_f():
    sys.path.insert(0, str(BACKEND / "modelscripts"))
    import ou_forecast
    import var_forecast
    shared = asos_nowcast.build_shared_input(parse(hf_records()), now=T13 + timedelta(minutes=26))
    for mod in (ou_forecast, var_forecast):
        res = mod.compute(_history_df(), "t", shared, 1)
        assert res["T0"] == shared["current"]["temp_f"] == pytest.approx(84.6, abs=0.01)


# ══ Resilience: last-good observation survives failed polls ═════════════════

import gzip as _gzip
import io as _io

T10 = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)
POLL_NOW = datetime(2026, 10, 3, 10, 24, tzinfo=timezone.utc)      # current file 1000, previous 0900


def hf_gz(rows):
    """rows: [(station, datetime, temp_f)] -> gzip'd classic-netCDF HFMETAR-like bytes."""
    from scipy.io import netcdf_file
    n = len(rows)
    buf = _io.BytesIO()
    nc = netcdf_file(buf, "w")
    nc.createDimension("recNum", n)
    nc.createDimension("staIdLen", 6)

    def put(name, dims, dtype, data):
        v = nc.createVariable(name, dtype, dims)
        v[:] = data

    put("stationId", ("recNum", "staIdLen"), "c",
        np.array([list(r[0].ljust(6, chr(0))) for r in rows], dtype="S1"))
    put("observationTime", ("recNum",), "d", np.array([r[1].timestamp() for r in rows]))
    put("temperature", ("recNum",), "f", np.array([meteo.f_to_c(r[2]) + 273.15 for r in rows], dtype="f4"))
    put("dewpoint", ("recNum",), "f", np.array([298.15] * n, dtype="f4"))
    put("windDir", ("recNum",), "f", np.array([110.0] * n, dtype="f4"))
    put("windSpeed", ("recNum",), "f", np.array([5.0] * n, dtype="f4"))
    put("altimeter", ("recNum",), "f", np.array([101490.0] * n, dtype="f4"))
    put("visibility", ("recNum",), "f", np.array([16093.0] * n, dtype="f4"))
    nc.flush()
    raw = buf.getvalue()
    nc.close()
    return _gzip.compress(raw)


class FakeMadis:
    """name -> bytes (200) | int (HTTP status) | Exception.  Unlisted files are 404."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.requested = []

    def get(self, url, headers=None, timeout=None):
        assert url.startswith(cfg.MADIS_HFMETAR_URL), url        # only MADIS may ever be contacted
        name = url.rsplit("/", 1)[-1]
        self.requested.append(name)
        item = self.files.get(name, 404)
        if isinstance(item, Exception):
            raise item

        class R:
            status_code = 200
            headers = {"Last-Modified": "x"}
            content = b""

            def raise_for_status(self_inner):
                if self_inner.status_code >= 400:
                    raise RuntimeError(f"HTTP {self_inner.status_code}")
        r = R()
        if isinstance(item, int):
            r.status_code = item
        else:
            r.content = item
        return r


@pytest.fixture
def feed(tmp_path):
    from weather import asos_client
    f = asos_client.AsosFeed(state_path=tmp_path / "last_good.json")
    return f


def current_temp(obs, now, info=None):
    shared = asos_nowcast.build_shared_input(obs, info, now=now)
    return shared, shared["current"]


def test_poll_requests_current_and_previous_hour_files(feed):
    srv = FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])})
    obs, info = feed.poll(now=POLL_NOW, session=srv)
    assert "20261003_1000.gz" in info["requested_files"] and "20261003_0900.gz" in info["requested_files"]
    assert obs[-1]["temp_f"] == pytest.approx(84.2, abs=0.01)


def test_http_500_after_good_poll_keeps_last_good(feed):
    good = FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])})
    feed.poll(now=POLL_NOW, session=good)
    down = FakeMadis({"20261003_1000.gz": 500, "20261003_0900.gz": 500})
    obs, info = feed.poll(now=POLL_NOW + timedelta(minutes=1), session=down)
    shared, cur = current_temp(obs, POLL_NOW + timedelta(minutes=1), info)
    assert cur is not None and cur["temp_f"] == pytest.approx(84.2, abs=0.01)
    assert cur["observed_at"] == (T10 + timedelta(minutes=10)).isoformat()      # timestamp not faked
    assert info["errors"] and info["source_status"] == "fetch_error_using_last_good"
    assert shared["metadata"]["latest_observation_age_seconds"] == pytest.approx(15 * 60, abs=1)   # 10:10 -> 10:25
    assert shared["metadata"]["status"] == "healthy" and shared["metadata"]["source_status"] == "fetch_error_using_last_good"


def test_network_exception_keeps_last_good(feed):
    feed.poll(now=POLL_NOW, session=FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=15), 84.4)])}))
    boom = FakeMadis({"20261003_1000.gz": ConnectionError("reset"), "20261003_0900.gz": ConnectionError("reset")})
    obs, info = feed.poll(now=POLL_NOW + timedelta(minutes=2), session=boom)
    assert obs[-1]["temp_f"] == pytest.approx(84.4, abs=0.01) and info["last_good_age_seconds"] is not None


def test_corrupt_or_truncated_gzip_keeps_last_good_and_cache(feed):
    good_bytes = hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2), ("KMIA", T10 + timedelta(minutes=15), 84.4)])
    feed.poll(now=POLL_NOW, session=FakeMadis({"20261003_1000.gz": good_bytes}))
    for bad in (b"not a gzip at all", good_bytes[: len(good_bytes) // 2]):             # garbage + truncated
        obs, info = feed.poll(now=POLL_NOW + timedelta(minutes=1),
                              session=FakeMadis({"20261003_1000.gz": bad}))
        assert any("20261003_1000.gz" in e for e in info["errors"])
        assert [o["observed_at"][11:16] for o in obs[-2:]] == ["10:10", "10:15"]        # cache intact
        assert obs[-1]["temp_f"] == pytest.approx(84.4, abs=0.01)


def test_file_without_kmia_does_not_erase_cached_records(feed):
    feed.poll(now=POLL_NOW, session=FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])}))
    other = hf_gz([("KFLL", T10 + timedelta(minutes=20), 80.0)])                       # valid file, no KMIA yet
    obs, info = feed.poll(now=POLL_NOW + timedelta(minutes=1), session=FakeMadis({"20261003_1000.gz": other}))
    assert obs[-1]["temp_f"] == pytest.approx(84.2, abs=0.01)


def test_current_hour_missing_uses_previous_hour(feed):
    prev = hf_gz([("KMIA", T10 - timedelta(minutes=10), 83.8)])                        # 09:50
    srv = FakeMadis({"20261003_0900.gz": prev})                                        # 1000 -> 404
    now = T10 + timedelta(minutes=3)                                                    # 10:03
    obs, info = feed.poll(now=now, session=srv)
    shared, cur = current_temp(obs, now)
    assert cur is not None and cur["observed_at"][11:16] == "09:50"
    assert cur["temp_f"] == pytest.approx(83.8, abs=0.01)
    assert info["last_poll_success"] is True       # a 404 for an unpublished hour is not a fetch error


def test_current_hour_is_rechecked_every_poll_never_frozen(feed):
    name = "20261003_1000.gz"
    feed.poll(now=POLL_NOW, session=FakeMadis({name: hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])}))
    srv = FakeMadis({name: hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2), ("KMIA", T10 + timedelta(minutes=15), 84.6)])})
    obs, _ = feed.poll(now=POLL_NOW + timedelta(minutes=1), session=srv)
    assert name in srv.requested and "20261003_0900.gz" in srv.requested
    assert obs[-1]["temp_f"] == pytest.approx(84.6, abs=0.01)
    assert feed.cache[name]["final"] is False


def test_truly_stale_last_good_is_unavailable(feed):
    feed.poll(now=POLL_NOW, session=FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])}))
    later = T10 + timedelta(minutes=10 + 26)                                            # last good is 26 min old
    obs, info = feed.poll(now=later, session=FakeMadis({"20261003_1000.gz": 500, "20261003_0900.gz": 500}))
    shared, cur = current_temp(obs, later, info)
    assert cur is None and shared["metadata"]["status"] == "unavailable"
    assert "26" in shared["metadata"]["unavailable_reason"] or "1560" in shared["metadata"]["unavailable_reason"]
    assert shared["last_known"]["temp_f"] == pytest.approx(84.2, abs=0.01)             # still shown for display only


def test_freshness_boundaries_15_and_25_minutes():
    assert asos_nowcast.freshness_status(15 * 60) == "healthy"
    assert asos_nowcast.freshness_status(15 * 60 + 1) == "delayed"
    assert asos_nowcast.freshness_status(25 * 60) == "delayed"
    assert asos_nowcast.freshness_status(25 * 60 + 1) == "unavailable"


def test_restart_restores_persisted_last_good_with_real_age(tmp_path):
    from weather import asos_client
    path = tmp_path / "last_good.json"
    f1 = asos_client.AsosFeed(state_path=path)
    f1.poll(now=POLL_NOW, session=FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])}))
    assert path.exists() and json.loads(path.read_text())["last_good"]["temp_f"] == pytest.approx(84.2, abs=0.01)
    restart = T10 + timedelta(minutes=18)                                               # 8 min after the obs
    f2 = asos_client.AsosFeed(state_path=path)                                          # new process, empty memory
    obs, info = f2.poll(now=restart, session=FakeMadis({"20261003_1000.gz": 500, "20261003_0900.gz": 500}))
    shared, cur = current_temp(obs, restart)
    assert cur is not None and cur["observed_at"] == (T10 + timedelta(minutes=10)).isoformat()   # not extended
    assert shared["metadata"]["latest_observation_age_seconds"] == pytest.approx(8 * 60, abs=1)
    assert shared["metadata"]["status"] == "healthy"


def test_restart_ignores_persisted_last_good_that_is_too_old(tmp_path):
    from weather import asos_client
    path = tmp_path / "last_good.json"
    f1 = asos_client.AsosFeed(state_path=path)
    f1.poll(now=POLL_NOW, session=FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2)])}))
    restart = T10 + timedelta(minutes=10 + 40)
    f2 = asos_client.AsosFeed(state_path=path)
    obs, info = f2.poll(now=restart, session=FakeMadis({"20261003_1000.gz": 500, "20261003_0900.gz": 500}))
    assert obs == [] and f2.last_good is None


def test_last_good_never_moves_backwards(feed):
    name = "20261003_1000.gz"
    feed.poll(now=POLL_NOW, session=FakeMadis({name: hf_gz([("KMIA", T10 + timedelta(minutes=15), 84.6)])}))
    older = hf_gz([("KMIA", T10 + timedelta(minutes=5), 80.0)])
    feed.cache.clear()                                   # simulate a fresh cache receiving only an older record
    obs, _ = feed.poll(now=POLL_NOW + timedelta(minutes=1), session=FakeMadis({name: older}))
    assert obs[-1]["observed_at"][11:16] == "10:15" and obs[-1]["temp_f"] == pytest.approx(84.6, abs=0.01)


def test_diagnostics_reports_poll_and_last_good(feed):
    srv = FakeMadis({"20261003_1000.gz": hf_gz([("KMIA", T10 + timedelta(minutes=10), 84.2),
                                                ("KMIA", T10 + timedelta(minutes=15), 84.4)])})
    obs, info = feed.poll(now=POLL_NOW, session=srv)
    from weather import asos_client
    d = asos_client.diagnostics(POLL_NOW, obs, info)
    assert d["last_poll_success"] is True and d["last_poll_error"] is None
    assert d["requested_files"][-2:] == ["20261003_0900.gz", "20261003_1000.gz"]
    assert d["kmia_records_found"] == 2
    assert d["last_good_observation"] == {"observed_at": (T10 + timedelta(minutes=15)).isoformat(), "temp_f": 84.4}
    assert d["observation_age_seconds"] == 9 * 60 and d["freshness_status"] == "healthy"


# ══ Spec checklist: station rows, 5-minute rule, Kelvin, fills, failures, merge, stale ═════════════

U21 = datetime(2026, 10, 3, 21, 0, tzinfo=timezone.utc)                  # 21:00Z hour
NOW_2108 = datetime(2026, 10, 3, 21, 8, 12, tzinfo=timezone.utc)


def at(h, m, sec=0):
    return datetime(2026, 10, 3, h, m, sec, tzinfo=timezone.utc)


def two_hour_server(rows_prev=(), rows_cur=(), **extra):
    files = {}
    if rows_prev:
        files["20261003_2000.gz"] = hf_gz(list(rows_prev))
    if rows_cur:
        files["20261003_2100.gz"] = hf_gz(list(rows_cur))
    files.update(extra)
    return FakeMadis(files)


def test_current_hour_file_contains_station(feed):
    obs, info = feed.poll(now=NOW_2108, session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)]))
    assert obs[-1]["observed_at"][11:16] == "21:05" and info["current_file"] == "20261003_2100.gz"
    assert info["previous_file"] == "20261003_2000.gz"


def test_previous_hour_fallback_right_after_the_hour(feed):
    now = at(21, 3)                                      # 21:03Z: 21:00 file not yet written
    obs, _ = feed.poll(now=now, session=two_hour_server(rows_prev=[("KMIA", at(20, 55), 85.5)]))
    assert obs[-1]["observed_at"][11:16] == "20:55" and obs[-1]["temp_f"] == pytest.approx(85.5, abs=0.01)


def test_current_and_previous_files_merge(feed):
    srv = two_hour_server(rows_prev=[("KMIA", at(20, 50), 84.5), ("KMIA", at(20, 55), 85.0)],
                          rows_cur=[("KMIA", at(21, 0), 85.5), ("KMIA", at(21, 5), 86.0)])
    obs, info = feed.poll(now=NOW_2108, session=srv)
    assert [o["observed_at"][11:16] for o in obs] == ["20:50", "20:55", "21:00", "21:05"]
    assert obs[-1]["temp_f"] == pytest.approx(86.0, abs=0.01)


def test_acceptance_fixture_2055_2100_2105(feed):
    srv = two_hour_server(rows_prev=[("KMIA", at(20, 55), 85.0)],
                          rows_cur=[("KMIA", at(21, 0), 85.5), ("KMIA", at(21, 5), 86.0)])
    obs, _ = feed.poll(now=NOW_2108, session=srv)
    assert obs[-1]["observed_at"][11:16] == "21:05" and obs[-1]["temp_f"] == pytest.approx(86.0, abs=0.01)


def test_greatest_observation_time_wins_regardless_of_row_order(feed):
    rows = [("KMIA", at(21, 5), 86.0), ("KMIA", at(21, 0), 85.5), ("KMIA", at(21, 10), 86.4), ("KMIA", at(20, 55), 85.0)]
    obs, _ = feed.poll(now=at(21, 12), session=two_hour_server(rows_cur=rows))
    assert obs[-1]["observed_at"][11:16] == "21:10" and obs[-1]["temp_f"] == pytest.approx(86.4, abs=0.01)


@pytest.mark.parametrize("minute", [0, 5, 10])
def test_5_minute_marks_accepted(minute):
    assert len(parse([madis_rec(at(21, minute))])) == 1


@pytest.mark.parametrize("minute", [3, 1, 7, 59])
def test_non_5_minute_marks_rejected(minute):
    assert parse([madis_rec(at(21, minute))]) == []


def test_5_minute_filter_allows_nonzero_seconds():
    assert len(parse([madis_rec(at(21, 5, 40))])) == 1


def test_kelvin_to_fahrenheit():
    obs = parse([madis_rec(at(21, 5), temp_c=30.0)])[0]                       # 303.15 K
    assert obs["temp_f"] == pytest.approx((303.15 - 273.15) * 9 / 5 + 32, abs=0.01) == pytest.approx(86.0, abs=0.01)


def _rec_with_temp(value):
    r = madis_rec(at(21, 5))
    r["temperature"] = value
    return r


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 3.4028235e38, -9999.0, 100.0, 400.0, None])
def test_invalid_temperatures_rejected(bad):
    assert parse([_rec_with_temp(bad)]) == []


def test_masked_value_rejected_at_decode():
    from weather import asos_client
    assert asos_client._val(np.ma.masked) is None
    assert asos_client._val(np.float32("nan")) is None and asos_client._val(3.4028235e38) is None


def test_http_404_keeps_cached_observation(feed):
    feed.poll(now=NOW_2108, session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)]))
    obs, info = feed.poll(now=NOW_2108 + timedelta(minutes=5), session=FakeMadis({}))      # everything 404
    assert obs[-1]["temp_f"] == pytest.approx(86.0, abs=0.01) and obs[-1]["observed_at"][11:16] == "21:05"


def test_timeout_keeps_cached_observation(feed):
    import requests
    feed.poll(now=NOW_2108, session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)]))
    slow = FakeMadis({"20261003_2100.gz": requests.Timeout("timed out"), "20261003_2000.gz": requests.Timeout("t")})
    obs, info = feed.poll(now=NOW_2108 + timedelta(minutes=5), session=slow)
    assert obs[-1]["temp_f"] == pytest.approx(86.0, abs=0.01) and info["errors"]


def test_regression_http_500_keeps_2105_86F(feed):
    feed.poll(now=NOW_2108, session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)]))
    obs, info = feed.poll(now=NOW_2108 + timedelta(minutes=5),
                          session=FakeMadis({"20261003_2100.gz": 500, "20261003_2000.gz": 500}))
    assert obs[-1]["temp_f"] == pytest.approx(86.0, abs=0.01) and obs[-1]["observed_at"] == at(21, 5).isoformat()
    assert feed.last_good["observed_at"] == at(21, 5).isoformat()


def test_older_observation_does_not_replace_newer_cache(feed):
    feed.poll(now=NOW_2108, session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)]))
    feed.cache.clear()
    obs, _ = feed.poll(now=NOW_2108 + timedelta(minutes=1), session=two_hour_server(rows_cur=[("KMIA", at(21, 0), 85.5)]))
    assert feed.last_good["observed_at"] == at(21, 5).isoformat() and obs[-1]["temp_f"] == pytest.approx(86.0, abs=0.01)


def test_same_timestamp_does_not_replace_cache(feed):
    srv = two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)])
    _, info1 = feed.poll(now=NOW_2108, session=srv)
    first = feed.last_good
    _, info2 = feed.poll(now=NOW_2108 + timedelta(minutes=1), session=srv)
    assert info1["cache_updated"] is True and info2["cache_updated"] is False and feed.last_good is first


def test_newer_observation_replaces_cache(feed):
    feed.poll(now=NOW_2108, session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)]))
    _, info = feed.poll(now=NOW_2108 + timedelta(minutes=5),
                        session=two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0), ("KMIA", at(21, 10), 86.4)]))
    assert info["cache_updated"] is True and feed.last_good["observed_at"] == at(21, 10).isoformat()


def test_get_latest_asos_observation_interface_and_stale_flag():
    from weather import asos_client
    srv = two_hour_server(rows_cur=[("KMIA", at(21, 5), 86.0)])
    cur = asos_client.get_latest_asos_observation("KMIA", now=NOW_2108, session=srv)
    assert cur["station"] == "KMIA" and cur["source"] == "NOAA MADIS HFMETAR"
    assert cur["temperature_f"] == pytest.approx(86.0, abs=0.01) and cur["temperature_k"] == pytest.approx(303.15, abs=0.01)
    assert cur["observation_time_utc"] == at(21, 5) and cur["fetched_at_utc"] is not None
    assert cur["age_seconds"] == pytest.approx(192, abs=1) and cur["is_stale"] is False
    # NOAA keeps failing; 20 minutes later the same observation is still returned, now marked stale
    down = FakeMadis({"20261003_2100.gz": 500, "20261003_2000.gz": 500})
    later = at(21, 5) + timedelta(minutes=20)
    stale = asos_client.get_latest_asos_observation("KMIA", now=later, session=down)
    assert stale is not None and stale["temperature_f"] == pytest.approx(86.0, abs=0.01)
    assert stale["observation_time_utc"] == at(21, 5) and stale["is_stale"] is True
    with pytest.raises(ValueError):
        asos_client.get_latest_asos_observation("KFLL", now=later, poll=False)


def test_none_only_when_nothing_was_ever_obtained():
    from weather import asos_client
    assert asos_client.get_latest_asos_observation("KMIA", now=NOW_2108, session=FakeMadis({})) is None


def test_stale_observation_remains_visible_in_tracker_snapshot(monkeypatch):
    """Display keeps the last temperature even when the observation is stale/unavailable."""
    from kalshi_api import temp_monitor, price_tracker
    price_tracker.update_temp(86.0)
    monkeypatch.setattr(temp_monitor, "fetch_recent_observations",
                        lambda now=None, **kw: ([], {"errors": ["down"], "source_status": "fetch_error_no_observation"}))
    temp_monitor._latest = {"observed_at": at(21, 5).isoformat(), "t0_f": 86.0, "status": "healthy"}
    obs = temp_monitor.fetch_current_high(now=at(21, 5) + timedelta(minutes=40))
    assert obs["t0_f"] == 86.0 and obs["status"] == "unavailable" and obs["source_status"] == "fetch_error_using_last_good"
    assert price_tracker.get_temp_snapshot()["current_f"] == 86.0
    temp_monitor._latest = None


def test_poll_slots_every_minute_by_default():
    from kalshi_api.temp_monitor import seconds_until_next_slot as f
    assert cfg.POLL_EVERY_MINUTES == 1
    t = datetime(2026, 10, 3, 21, 3, 30, tzinfo=timezone.utc)
    assert (t + timedelta(seconds=f(t))).strftime("%H:%M:%S") == "21:04:00"
    t = datetime(2026, 10, 3, 21, 59, 59, tzinfo=timezone.utc)
    assert (t + timedelta(seconds=f(t))).strftime("%H:%M:%S") == "22:00:00"


def test_poll_slots_are_minute_01_06_11(monkeypatch):
    monkeypatch.setattr(cfg, "POLL_EVERY_MINUTES", 5)
    from kalshi_api.temp_monitor import seconds_until_next_slot as f

    def nxt(h, m, sec):
        t = datetime(2026, 10, 3, h, m, sec, tzinfo=timezone.utc)
        return t + timedelta(seconds=f(t))

    assert nxt(21, 0, 0).strftime("%H:%M:%S") == "21:01:00"
    assert nxt(21, 1, 0).strftime("%H:%M:%S") == "21:01:00"
    assert nxt(21, 3, 30).strftime("%H:%M:%S") == "21:06:00"
    assert nxt(21, 56, 1).strftime("%H:%M:%S") == "22:01:00"
    assert nxt(21, 59, 59).strftime("%H:%M:%S") == "22:01:00"


def test_scheduled_poll_updates_current_temp_and_recorded_high(monkeypatch):
    """Each 5-minute poll that sees a newer observation advances both Current Temp and the recorded high."""
    from kalshi_api import temp_monitor, price_tracker
    price_tracker._current_f = price_tracker._daily_high_f = price_tracker._date_str = None
    temp_monitor._latest = None
    now = floor5(datetime.now(timezone.utc)) + timedelta(minutes=2)
    series = [(now - timedelta(minutes=12), 84.0), (now - timedelta(minutes=7), 85.0)]
    nws = {"high_f": 85.0}      # recorded high now comes from the NWS full-precision obs, not HFMETAR
    monkeypatch.setattr(temp_monitor, "get_recorded_high",
                        lambda now=None: {"high_f": nws["high_f"], "high_observed_at": None})

    def serve(rows):
        obs = parse([madis_rec(t, temp_c=meteo.f_to_c(f)) for t, f in rows])
        monkeypatch.setattr(temp_monitor, "fetch_recent_observations",
                            lambda now=None, **kw: (obs, {"errors": [], "source_status": "ok", "last_poll_success": True}))

    serve(series)
    r1 = temp_monitor.refresh_temp_now()
    snap = price_tracker.get_temp_snapshot()
    assert snap["current_f"] == pytest.approx(85.0, abs=0.01) and snap["daily_high_f"] == pytest.approx(85.0, abs=0.01)
    # next 5-minute poll: a new, hotter observation appears
    series.append((now - timedelta(minutes=2), 86.0))
    nws["high_f"] = 86.0
    serve(series)
    temp_monitor.refresh_temp_now()
    snap = price_tracker.get_temp_snapshot()
    assert snap["current_f"] == pytest.approx(86.0, abs=0.01) and snap["daily_high_f"] == pytest.approx(86.0, abs=0.01)
    # a later, cooler observation moves Current Temp but never lowers the recorded high
    series.append((now + timedelta(minutes=3), 85.0))
    serve(series)
    temp_monitor.refresh_temp_now()
    snap = price_tracker.get_temp_snapshot()
    assert snap["daily_high_f"] == pytest.approx(86.0, abs=0.01)
    temp_monitor._latest = None


def test_hfmetar_current_never_advances_recorded_high(monkeypatch):
    """87 F arrives as 31 C = 87.8 F in HFMETAR; only the NWS high may set the record (87.1)."""
    from kalshi_api import temp_monitor, price_tracker
    price_tracker._current_f = price_tracker._daily_high_f = price_tracker._date_str = None
    temp_monitor._latest = None
    now = floor5(datetime.now(timezone.utc)) + timedelta(minutes=2)
    obs = parse([madis_rec(now - timedelta(minutes=2), temp_c=31.0)])
    monkeypatch.setattr(temp_monitor, "fetch_recent_observations",
                        lambda now=None, **kw: (obs, {"errors": [], "source_status": "ok", "last_poll_success": True}))
    monkeypatch.setattr(temp_monitor, "get_recorded_high",
                        lambda now=None: {"high_f": 87.1, "high_observed_at": None})
    temp_monitor.refresh_temp_now()
    snap = price_tracker.get_temp_snapshot()
    assert snap["current_f"] == pytest.approx(87.8, abs=0.01)
    assert snap["daily_high_f"] == pytest.approx(87.1, abs=0.01)
    assert price_tracker.nws_round_temp_f(snap["daily_high_f"]) == 87
    temp_monitor._latest = None
