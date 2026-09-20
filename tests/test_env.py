"""Phase 5 acceptance unit tests for src/rl/environment.py (docs/SPEC.md
Phase 5 task 6): reward determinism given (user, action sequence, seed);
artist cap violations incur w_rep and are reported in the reward
decomposition (the cap is a soft penalty, not a hard filter — see
environment.py's module docstring); no duplicate track ever enters a
playlist (that one IS a hard filter). Uses the real Phase 3/4 artifacts on
disk (core content matrix, candidates.parquet, holdout.parquet) rather than
synthetic fixtures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from config import DATA_PROCESSED
from src.recommenders.common import load_core_content_matrix
from src.rl.environment import GROUNDED, GROUNDED_SIMULATED, EnvConfig, PlaylistEnv, _load_item_metadata


@pytest.fixture(scope="module")
def content_matrix():
    return load_core_content_matrix()


@pytest.fixture(scope="module")
def candidates_df():
    return pd.read_parquet(DATA_PROCESSED / "candidates.parquet")


@pytest.fixture(scope="module")
def holdout_df():
    return pd.read_parquet(DATA_PROCESSED / "holdout.parquet")


def _user_candidates(candidates_df, user_idx):
    return candidates_df[candidates_df["user_idx"] == user_idx][["item_idx", "hybrid_score"]]


def _user_holdout(holdout_df, user_idx):
    rows = holdout_df[holdout_df["user_idx"] == user_idx]
    return dict(zip(rows["item_idx"], rows["aff_ui"]))


def _sample_user(candidates_df, holdout_df, n=1, seed=0):
    users = np.intersect1d(candidates_df["user_idx"].unique(), holdout_df["user_idx"].unique())
    rng = np.random.default_rng(seed)
    return rng.choice(users, size=n, replace=False)


def _make_env(content_matrix, candidates_df, holdout_df, user_idx, config=None):
    cand = _user_candidates(candidates_df, user_idx)
    holdout = _user_holdout(holdout_df, user_idx)
    return PlaylistEnv(int(user_idx), cand, content_matrix, holdout, config)


def _run_random_episode(env, rng):
    """Runs one episode with a uniform-random arm policy, returns the reward
    sequence and the final playlist."""
    rewards = []
    done = False
    env.reset()
    while not done:
        arm = int(rng.integers(0, env.config.n_arms))
        _, reward, done, info = env.step(arm)
        if info.get("pool_exhausted"):
            break
        rewards.append(reward)
    return rewards, list(env.playlist)


@pytest.mark.parametrize("reward_mode", [GROUNDED, GROUNDED_SIMULATED])
def test_reward_deterministic_given_seed(content_matrix, candidates_df, holdout_df, reward_mode):
    """PlaylistEnv has no internal randomness (clusters are a fixed global
    lookup, not fit per episode — see environment.py's EnvConfig docstring),
    so determinism given (user, action sequence) holds unconditionally; this
    still exercises the Phase 5 acceptance criterion as originally stated."""
    user_idx = int(_sample_user(candidates_df, holdout_df, seed=1)[0])
    action_sequence = [0, 3, 7, 2, 5, 1, 9, 4, 6, 8, 0, 2, 5, 7, 1]

    config = EnvConfig(reward_mode=reward_mode)
    env_a = _make_env(content_matrix, candidates_df, holdout_df, user_idx, config)
    env_b = _make_env(content_matrix, candidates_df, holdout_df, user_idx, EnvConfig(reward_mode=reward_mode))

    rewards_a, playlist_a = [], []
    for arm in action_sequence:
        _, r, done, info = env_a.step(arm)
        if info.get("pool_exhausted"):
            break
        rewards_a.append(r)
        if done:
            break
    playlist_a = list(env_a.playlist)

    rewards_b, playlist_b = [], []
    for arm in action_sequence:
        _, r, done, info = env_b.step(arm)
        if info.get("pool_exhausted"):
            break
        rewards_b.append(r)
        if done:
            break
    playlist_b = list(env_b.playlist)

    assert rewards_a == rewards_b
    assert playlist_a == playlist_b


def test_artist_cap_violation_incurs_penalty():
    """The artist cap is a soft constraint (see environment.py's module
    docstring): a candidate that would exceed it stays selectable, but its
    selection must incur the w_rep penalty and show up as repeat=1 in the
    reward decomposition. Constructs a controlled 3-item, single-artist pool
    (real content vectors, real artist metadata) and drives it with
    step_song() (direct position control, bypassing arm-cluster resolution
    entirely) so the 3rd pick of the same artist deterministically exceeds
    artist_cap=2 regardless of which global cluster these items fall into."""
    content_matrix = load_core_content_matrix()
    meta = _load_item_metadata()
    counts = meta["artist"].value_counts()
    artist = counts[counts >= 3].index[0]
    item_idxs = meta.index[meta["artist"] == artist][:3].tolist()

    candidates = pd.DataFrame({"item_idx": item_idxs, "hybrid_score": [3.0, 2.0, 1.0]})
    config = EnvConfig(playlist_length=3, artist_cap=2, reward_mode=GROUNDED)
    env = PlaylistEnv(user_idx=0, candidates=candidates, content_matrix=content_matrix, holdout={}, config=config)

    decompositions = []
    for pos in range(3):
        _, reward, done, info = env.step_song(pos)
        decompositions.append(info)

    assert len(env.playlist) == 3
    assert len(set(env.playlist)) == 3  # the hard duplicate filter still holds

    repeats = [d["repeat"] for d in decompositions]
    assert repeats == [0, 0, 1], f"expected the artist cap (2) to be exceeded on the 3rd pick, got {repeats}"

    third = decompositions[2]
    expected_reward = (config.w_rel * third["rel"] + config.w_div * third["div"]
                        - config.w_rep * third["repeat"] - config.w_dup * third["dup"])
    assert third["reward"] == pytest.approx(expected_reward)
    assert third["reward"] < config.w_rel * third["rel"] + config.w_div * third["div"], \
        "w_rep penalty was not actually applied to the reward"


def test_no_duplicate_tracks(content_matrix, candidates_df, holdout_df):
    users = _sample_user(candidates_df, holdout_df, n=20, seed=3)
    rng = np.random.default_rng(456)
    for user_idx in users:
        env = _make_env(content_matrix, candidates_df, holdout_df, int(user_idx), EnvConfig())
        _, playlist = _run_random_episode(env, rng)
        assert len(playlist) == len(set(playlist)), f"duplicate track for user {user_idx}: {playlist}"


def test_step_song_rejects_reused_position(content_matrix, candidates_df, holdout_df):
    """step_song (Phase 6's LinUCB song-level control) must hard-reject an
    already-used position rather than silently substituting another one."""
    user_idx = int(_sample_user(candidates_df, holdout_df, seed=4)[0])
    env = _make_env(content_matrix, candidates_df, holdout_df, user_idx, EnvConfig())
    unused = env.unused_positions()
    pos = int(unused[0])
    env.step_song(pos)
    assert pos not in env.unused_positions()
    with pytest.raises(ValueError):
        env.step_song(pos)


def test_song_contexts_shape(content_matrix, candidates_df, holdout_df):
    user_idx = int(_sample_user(candidates_df, holdout_df, seed=5)[0])
    env = _make_env(content_matrix, candidates_df, holdout_df, user_idx, EnvConfig())
    contexts = env.song_contexts()
    expected_dim = env.user_profile.shape[1] + content_matrix.shape[1] + 5  # profile + content + state features
    assert contexts.shape == (len(env.candidate_items), expected_dim)
    assert np.isfinite(contexts).all()
