import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.playlist import AGENTS, MAX_LENGTH, MIN_LENGTH, PlaylistService

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def svc():
    return PlaylistService()


def test_cold_start_under_10s(svc):
    assert svc.load_seconds < 10


@pytest.mark.parametrize("agent", list(AGENTS))
@pytest.mark.parametrize("length", [MIN_LENGTH, 12, MAX_LENGTH])
def test_playlist_is_valid_fast_and_leak_free(svc, agent, length):
    user = 12345
    built = svc.build(user, length, agent)
    items = built.tracks["item_idx"].tolist()
    assert len(items) == length and len(set(items)) == length            # no duplicates
    train = set(svc.train_items[svc.train_offsets[user]:svc.train_offsets[user + 1]].tolist())
    assert not (set(items) & train)                                       # never re-recommend a train track
    assert built.seconds < 3
    assert built.heldout_hits == sum(i in svc.holdout_of(user) for i in items)


def test_serving_is_deterministic(svc):
    a = svc.build(777, 10, "Random arm").tracks["item_idx"].tolist()
    b = svc.build(777, 10, "Random arm").tracks["item_idx"].tolist()
    assert a == b  # shared agents are deep-copied per request, so state never leaks between requests


def test_greedy_is_the_top_of_the_hybrid_ranking(svc):
    built = svc.build(4321, 10, "Hybrid (Greedy)")
    assert sorted(built.tracks["hybrid_rank"].tolist()) == list(range(1, 11))


def test_streamlit_app_runs_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    at.slider[0].set_value(8).run()
    assert not at.exception
    at.selectbox[1].set_value("SARSA").run()
    assert not at.exception
    assert len(at.markdown) > 8
