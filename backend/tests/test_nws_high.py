from datetime import datetime, timezone

import pytest

from weather import nws_high

NOW = datetime(2026, 10, 4, 4, 56, tzinfo=timezone.utc)   # Oct 4, 00:56 EDT -> Miami day is Oct 4
DAY = datetime(2026, 10, 3, 23, 30, tzinfo=timezone.utc)  # Oct 3, 19:30 EDT


def feat(ts, c):
    return {"properties": {"timestamp": ts, "temperature": {"value": c}}}


def test_oct3_high_is_87_1_not_hfmetar_87_8():
    """Yesterday: whole-C 5-minute rows hit 31 C (87.8 F) but the :53 METARs peaked at 30.6 C (87.1 F)."""
    features = [feat("2026-10-03T17:50:00+00:00", 31), feat("2026-10-03T17:55:00+00:00", 31),
                feat("2026-10-03T17:53:00+00:00", 30.6), feat("2026-10-03T16:53:00+00:00", 30.6),
                feat("2026-10-03T21:53:00+00:00", 29.4)]
    r = nws_high.high_from_features(features, DAY)
    assert r["high_f"] == 87.1
    assert r["high_observed_at"].startswith("2026-10-03T17:53") or r["high_observed_at"].startswith("2026-10-03T16:53")
    assert r["n_obs"] == 3


def test_five_minute_whole_c_rows_never_count():
    r = nws_high.high_from_features([feat("2026-10-03T17:55:00+00:00", 31)], DAY)
    assert r["high_f"] is None and r["n_obs"] == 0


def test_special_metar_off_five_minute_counts():
    r = nws_high.high_from_features([feat("2026-10-03T21:58:00+00:00", 29.4)], DAY)
    assert r["high_f"] == 84.9


def test_only_the_miami_calendar_day_counts():
    features = [feat("2026-10-03T23:53:00+00:00", 30.6), feat("2026-10-04T04:53:00+00:00", 26.7)]
    assert nws_high.high_from_features(features, NOW)["high_f"] == 80.1   # Oct 4 EDT only


def test_failed_fetch_keeps_last_good_and_never_uses_hfmetar(monkeypatch):
    nws_high._cache.update(date=None, fetched=0.0, result=None)
    monkeypatch.setattr(nws_high, "_fetch_features", lambda now: [feat("2026-10-03T17:53:00+00:00", 30.6)])
    assert nws_high.get_recorded_high(DAY)["high_f"] == 87.1

    def boom(now):
        raise RuntimeError("down")
    nws_high._cache["fetched"] = 0.0
    monkeypatch.setattr(nws_high, "_fetch_features", boom)
    assert nws_high.get_recorded_high(DAY)["high_f"] == 87.1
