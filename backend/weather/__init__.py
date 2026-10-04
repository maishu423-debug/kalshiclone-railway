"""ASOS-only weather pipeline: client -> parser -> nowcast -> shared model input."""
from datetime import datetime, timezone

from . import config
from .asos_client import fetch_recent_observations
from .asos_nowcast import build_shared_input, format_log_summary


def build_live_shared_input(now=None):
    """Fetch KMIA ASOS once and return the provider-neutral model input (see asos_nowcast)."""
    now = now or datetime.now(timezone.utc)
    observations, info = fetch_recent_observations(now=now)
    return build_shared_input(observations, info, now=now)


__all__ = ["config", "build_live_shared_input", "build_shared_input",
           "fetch_recent_observations", "format_log_summary"]
