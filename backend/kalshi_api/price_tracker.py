"""
Real-time price velocity tracker.

Records YES/NO ask prices for every market every time the algorithm
state endpoint is hit (every 30 s from the frontend) plus once a
minute from its own background thread.

Velocity = price change in ¢/min over a rolling 5-minute window.
Daily high is tracked as the running maximum of T0 readings from the
forecast models, resetting automatically at midnight.
"""
import threading
import time
import math
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

_lock = threading.Lock()

# ticker → deque of {ts, yes_bid, yes_ask}
_price_history = defaultdict(lambda: deque(maxlen=60))

_current_f    = None
_daily_high_f = None
_date_str     = None   # YYYY-MM-DD, used to detect midnight rollover

_started = False

VELOCITY_WINDOW_SEC = 300   # 5-minute rolling window
MIAMI_TZ = ZoneInfo("America/New_York")


def _today_key() -> str:
    return datetime.now(MIAMI_TZ).date().isoformat()


def nws_round_temp_f(temp_f: float) -> int:
    """NWS whole-degree temperature rounding: .5 and above rounds up."""
    return int(math.floor(float(temp_f) + 0.5))


# ── Write ────────────────────────────────────────────────────────────────────

def record_price(ticker: str, yes_bid_cents, yes_ask_cents):
    """Call every time you get a fresh price for a market."""
    with _lock:
        _price_history[ticker].append({
            "ts":      time.time(),
            "yes_bid": yes_bid_cents,
            "yes_ask": yes_ask_cents,
        })


def update_temp(temp_f: float):
    """Update current temperature and advance daily_high when this reading is hotter."""
    global _current_f, _daily_high_f, _date_str
    if temp_f is None:
        return
    with _lock:
        today = _today_key()
        if _date_str != today:
            _date_str = today
            _daily_high_f = temp_f   # reset for new day — don't carry yesterday's high
        elif _daily_high_f is None or temp_f > _daily_high_f:
            _daily_high_f = temp_f
        _current_f = temp_f


def update_current(temp_f: float):
    """Update current temperature without changing daily_high."""
    global _current_f, _date_str
    if temp_f is None:
        return
    with _lock:
        _date_str = _today_key()
        _current_f = temp_f


def update_daily_high(high_f: float):
    """Advance today's daily high — only ever increases, never lowers."""
    global _daily_high_f, _date_str
    if high_f is None:
        return
    with _lock:
        today = _today_key()
        if _date_str != today:
            _date_str = today
        if _daily_high_f is None or high_f > _daily_high_f:
            _daily_high_f = high_f


def set_daily_high_nws(high_f: float):
    """Set today's recorded high from the combined NWS + METAR authoritative source.
    Resets on a new day; otherwise only ever advances (never lowers)."""
    global _daily_high_f, _date_str
    if high_f is None:
        return
    with _lock:
        today = _today_key()
        if _date_str != today:
            _date_str = today
            _daily_high_f = high_f          # new day: reset to first reading
        elif _daily_high_f is None or high_f > _daily_high_f:
            _daily_high_f = high_f          # same day: only advance


# ── Read ─────────────────────────────────────────────────────────────────────

def get_temp_snapshot() -> dict:
    with _lock:
        return {
            "current_f":    _current_f,
            "daily_high_f": _daily_high_f,
        }


def get_yes_velocity(ticker: str) -> float:
    """YES ask price ¢/min over the last 5 minutes."""
    with _lock:
        hist = list(_price_history[ticker])
    pts = [(h["ts"], h["yes_ask"]) for h in hist if h["yes_ask"] is not None]
    return _velocity(pts)


def get_no_velocity(ticker: str) -> float:
    """
    NO ask ≈ 100 − YES bid.
    Positive = NO price rising (YES becoming cheaper / less likely).
    """
    with _lock:
        hist = list(_price_history[ticker])
    pts = [(h["ts"], 100 - h["yes_bid"]) for h in hist if h["yes_bid"] is not None]
    return _velocity(pts)


def _velocity(pts: list) -> float:
    now    = time.time()
    recent = [(ts, v) for ts, v in pts if now - ts <= VELOCITY_WINDOW_SEC]
    if len(recent) < 2:
        return 0.0
    dt_min = (recent[-1][0] - recent[0][0]) / 60.0
    if dt_min < 0.1:
        return 0.0
    return (recent[-1][1] - recent[0][1]) / dt_min


def get_all_velocities(tickers) -> dict:
    return {
        t: {
            "yes_velocity": round(get_yes_velocity(t), 3),
            "no_velocity":  round(get_no_velocity(t),  3),
        }
        for t in tickers
    }


# ── Background polling ────────────────────────────────────────────────────────

def _poll_loop():
    time.sleep(15)   # let Django finish starting
    while True:
        _poll_once()
        time.sleep(60)


def _poll_once():
    try:
        from .client import fetch_json, get_today_event_ticker, normalize_market
        payload = fetch_json(f"/events/{get_today_event_ticker()}")
        for raw in payload.get("markets", []):
            m = normalize_market(raw)
            record_price(m["ticker"], m.get("yes_bid_cents"), m.get("yes_ask_cents"))
    except Exception:
        pass

    # Temperature is owned by the 5-min ASOS poll loop in forecast.py.
    # Removed stale-model-T0 update here to prevent oscillation.


def start_price_tracking():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_poll_loop, daemon=True, name="price-tracker-bg")
    t.start()
