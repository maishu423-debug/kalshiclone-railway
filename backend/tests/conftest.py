import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest


@pytest.fixture(autouse=True)
def _isolated_asos_feed(tmp_path):
    """Each test gets an empty feed with its own last-good file (never touches the real one)."""
    from weather import asos_client
    asos_client.get_feed().reset(state_path=tmp_path / "asos_last_good.json")
    yield
    asos_client.get_feed().reset()
