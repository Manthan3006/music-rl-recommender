"""Phase 5 — RL playlist-construction environment (Gym-style).

`PlaylistEnv` builds one episode's playlist for a single user out of that
user's Phase 4 candidate pool. It implements the two-level action space from
B.7: the agent picks an **arm** (one of N_ARMS=10 *fixed, global* content
clusters — see fit_global_clusters.py; every item_idx has one persistent
cluster label used by every user and every episode, not a per-episode
refit), and the environment deterministically resolves that arm to the
highest-hybrid-score unused candidate in it (B.7 level 2). Reward is B.12's
grounded (Tier 1) or grounded+simulated (Tier 1 + Tier 2) formula, selected
via `EnvConfig.reward_mode`.

Global (not per-candidate-set) clusters are required for Phase 6: arm
identity has to mean the same thing across all 20,000 training episodes for
EpsilonGreedyMAB/UCB1's arm-value estimates and the Q-table to accumulate
anything coherent across episodes — a value learned for "arm 3" from one
user's episode must transfer to the next user's episode, which per-episode
KMeans refitting (arbitrary, independently-assigned labels each time) cannot
give you.

Only the no-duplicate constraint is a hard filter on candidate selection — a
playlist containing the same track twice is a bug, not a trade-off. The
artist cap is a **soft** constraint: a candidate that would exceed it stays
selectable and is charged the `w_rep` penalty in the reward instead of being
removed from consideration. This is deliberate, not an oversight: the
repetition penalty is the primary source of sequential credit assignment in
this MDP — the cost of repeating an artist at step t is paid at step t, but
the benefit of having deferred it only shows up in the diversity/relevance
terms of later steps. If repetition were structurally impossible (as an
earlier version of this environment did, by hard-filtering it), the agent
would have nothing to learn to defer, and the whole argument for using RL
here over a stateless contextual bandit (which just maximises immediate
reward) would weaken to nothing. So the penalty must be a cost the agent can
choose to pay, not a wall it cannot reach. See docs/SPEC.md Phase 5.

State (B.7, |S|=162): a 5-tuple of small discretised buckets, chosen so a
tabular Q-table is meaningful across users and candidate pools — nothing in
the state is a raw item id.
"""

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import ARTIST_CAP, DATA_PROCESSED, N_ARMS, PLAYLIST_LENGTH, SEED, SIMULATED_REL_LAMBDA, W_DIV, W_DUP, W_REL, W_REP
from src.recommenders.common import load_idx_maps

GROUNDED = "grounded"
GROUNDED_SIMULATED = "grounded_simulated"

# Fixed tertile-style bucket edges. Computed once from the core catalogue's
# own energy/valence distribution so state semantics are stable across every
# user and episode (required for a tabular Q-table to mean the same thing
# everywhere) rather than being sample-window dependent.
_ITEM_METADATA = None
_ENERGY_EDGES = None
_VALENCE_EDGES = None
_USER_PROFILES = None


def _load_item_metadata() -> pd.DataFrame:
    """artist / energy / valence per item_idx, core catalogue, cached."""
    global _ITEM_METADATA, _ENERGY_EDGES, _VALENCE_EDGES
    if _ITEM_METADATA is not None:
        return _ITEM_METADATA

    _, item_track_ids = load_idx_maps()
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet").set_index("track_id")
    meta = items.loc[item_track_ids, ["artist", "energy", "valence"]].reset_index(drop=True)
    meta.index.name = "item_idx"

    _ENERGY_EDGES = np.quantile(meta["energy"].to_numpy(), [1 / 3, 2 / 3])
    _VALENCE_EDGES = np.quantile(meta["valence"].to_numpy(), [1 / 3, 2 / 3])
    _ITEM_METADATA = meta
    return _ITEM_METADATA


def _load_user_profiles() -> sp.csr_matrix:
    global _USER_PROFILES
    if _USER_PROFILES is None:
        _USER_PROFILES = sp.load_npz(DATA_PROCESSED / "user_profiles.npz").tocsr()
    return _USER_PROFILES


def _bucket(value: float, edges: np.ndarray) -> int:
    """0=low, 1=mid, 2=high."""
    return int(np.searchsorted(edges, value, side="right"))


_GLOBAL_CLUSTERS = None


def _load_global_clusters() -> np.ndarray:
    """item_idx -> fixed global cluster label (0..N_ARMS-1), see
    fit_global_clusters.py. Computed once and persisted to disk; every
    PlaylistEnv shares this same lookup instead of fitting its own KMeans."""
    global _GLOBAL_CLUSTERS
    if _GLOBAL_CLUSTERS is None:
        from src.rl.fit_global_clusters import load as load_global_clusters
        _GLOBAL_CLUSTERS = load_global_clusters()
    return _GLOBAL_CLUSTERS


@dataclass
class EnvConfig:
    playlist_length: int = PLAYLIST_LENGTH
    artist_cap: int = ARTIST_CAP
    n_arms: int = N_ARMS
    reward_mode: str = GROUNDED_SIMULATED
    lambda_sim: float = SIMULATED_REL_LAMBDA
    w_rel: float = W_REL
    w_div: float = W_DIV
    w_rep: float = W_REP
    w_dup: float = W_DUP
    seed: int = SEED


class PlaylistEnv:
    """One user's playlist-construction episode.

    Parameters
    ----------
    user_idx : int
    candidates : pd.DataFrame with columns ['item_idx', 'hybrid_score'] —
        this user's Phase 4 candidate pool (e.g. a slice of
        candidates.parquet filtered to user_idx).
    content_matrix : sp.csr_matrix, row i == item_idx i (core catalogue),
        e.g. src.recommenders.common.load_core_content_matrix().
    holdout : dict {item_idx: aff_ui} — this user's held-out tracks (H_u)
        and their observed affinity, e.g. from holdout.parquet.
    config : EnvConfig
    """

    def __init__(self, user_idx: int, candidates: pd.DataFrame, content_matrix: sp.csr_matrix,
                 holdout: dict, config: EnvConfig = None):
        self.user_idx = user_idx
        self.content_matrix = content_matrix
        self.holdout = holdout
        self.config = config or EnvConfig()

        meta = _load_item_metadata()
        cand = candidates.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        self.candidate_items = cand["item_idx"].to_numpy()
        self.candidate_scores = cand["hybrid_score"].to_numpy()
        self.candidate_artists = meta["artist"].to_numpy()[self.candidate_items]

        self.cluster_of = _load_global_clusters()[self.candidate_items]
        # per-arm candidate positions, pre-sorted by hybrid_score descending
        # (candidate_items is already sorted, so this preserves that order)
        self.arms = {k: [i for i in range(len(self.candidate_items)) if self.cluster_of[i] == k]
                     for k in range(self.config.n_arms)}

        # Always loaded (not just for GROUNDED_SIMULATED): LinUCB's song-level
        # context (B.7) includes the user profile regardless of reward mode.
        self.user_profile = _load_user_profiles()[user_idx]

        self.reward_log = []
        self.reset()

    def reset(self):
        self.used_positions = set()
        self.playlist = []          # list of item_idx, in play order
        self.artist_counts = Counter()
        self.reward_log = []
        return self._state()

    def _state(self) -> tuple:
        recent = self.playlist[-3:]
        if recent:
            meta = _load_item_metadata()
            energy = meta["energy"].to_numpy()[recent].mean()
            valence = meta["valence"].to_numpy()[recent].mean()
            energy_bucket = _bucket(energy, _ENERGY_EDGES)
            valence_bucket = _bucket(valence, _VALENCE_EDGES)
        else:
            energy_bucket = 1  # mid, no history yet
            valence_bucket = 1

        if len(self.playlist) >= 2:
            # content_matrix rows are already L2-normalised (Phase 2), so a
            # plain dot product is the cosine similarity.
            vecs = self.content_matrix[self.playlist].toarray()
            sims = vecs @ vecs.T
            iu = np.triu_indices(len(self.playlist), k=1)
            mean_dissim = 1.0 - sims[iu].mean()
            diversity_level = _bucket(mean_dissim, np.array([1 / 3, 2 / 3]))
        else:
            diversity_level = 1  # mid, undefined with <2 tracks

        near_cap = 1 if self.artist_counts and max(self.artist_counts.values()) >= self.config.artist_cap else 0

        frac = len(self.playlist) / self.config.playlist_length
        position_bucket = _bucket(frac, np.array([1 / 3, 2 / 3]))

        return (energy_bucket, valence_bucket, diversity_level, near_cap, position_bucket)

    def _select_candidate(self, arm_k: int):
        """Best unused candidate in cluster arm_k; falls back to the best
        unused candidate in any cluster if arm_k is exhausted, so an episode
        never deadlocks while unused candidates remain elsewhere in the pool.

        The artist cap is NOT enforced here — a candidate that would exceed
        it stays selectable and is charged the w_rep penalty in step()
        instead. Only "already used" (duplicate) is a hard filter; the
        artist cap is a cost the agent can choose to pay."""
        def is_valid(pos):
            return pos not in self.used_positions

        for pos in self.arms.get(arm_k, []):
            if is_valid(pos):
                return pos

        for pos in range(len(self.candidate_items)):
            if is_valid(pos):
                return pos

        return None

    def _relevance(self, item_idx: int) -> float:
        if item_idx in self.holdout:
            return float(self.holdout[item_idx])
        if self.config.reward_mode == GROUNDED_SIMULATED:
            # Both content_matrix rows and user_profiles rows are already
            # L2-normalised (Phase 2), so a plain dot product is the cosine.
            content_vec = self.content_matrix[item_idx]
            cos = float(content_vec.dot(self.user_profile.T).toarray()[0, 0])
            return self.config.lambda_sim * cos
        return 0.0

    def _apply_action(self, pos: int, arm_k: int):
        cfg = self.config
        item_idx = int(self.candidate_items[pos])
        artist = self.candidate_artists[pos]

        rel = self._relevance(item_idx)

        if self.playlist:
            # content_matrix rows are already L2-normalised, so vecs @ v is cosine.
            vecs = self.content_matrix[self.playlist].toarray()
            v = self.content_matrix[item_idx].toarray().ravel()
            max_sim = float((vecs @ v).max())
            diversity = 1.0 - max_sim
        else:
            diversity = 1.0

        repeat_flag = 1 if self.artist_counts[artist] >= cfg.artist_cap else 0
        dup_flag = 1 if item_idx in self.playlist else 0

        reward = (cfg.w_rel * rel + cfg.w_div * diversity
                  - cfg.w_rep * repeat_flag - cfg.w_dup * dup_flag)

        self.used_positions.add(pos)
        self.playlist.append(item_idx)
        self.artist_counts[artist] += 1

        decomposition = {
            "step": len(self.playlist), "item_idx": item_idx, "artist": artist,
            "arm": arm_k, "rel": rel, "div": diversity, "repeat": repeat_flag, "dup": dup_flag,
            "reward": reward,
        }
        self.reward_log.append(decomposition)

        done = len(self.playlist) >= cfg.playlist_length
        return self._state(), reward, done, decomposition

    def step(self, arm_k: int):
        """Level-1 (tabular/bandit) action: pick an arm, environment resolves
        it to a specific candidate (B.7 level 2)."""
        pos = self._select_candidate(arm_k)
        if pos is None:
            return self._state(), 0.0, True, {"pool_exhausted": True}
        return self._apply_action(pos, arm_k)

    def step_song(self, position: int):
        """Song-level action (B.7's LinUCB): directly select a specific
        candidate position, bypassing arm-cluster resolution entirely.
        `position` must be unused this episode — see unused_positions();
        the no-duplicate constraint is still a hard filter, it is just
        enforced by the caller not offering a used position rather than by
        this method silently substituting another one (an agent choosing
        among only the unused positions is the whole point of song-level
        control)."""
        if position in self.used_positions:
            raise ValueError(f"position {position} already used this episode")
        return self._apply_action(position, arm_k=int(self.cluster_of[position]))

    def unused_positions(self) -> np.ndarray:
        return np.fromiter(
            (p for p in range(len(self.candidate_items)) if p not in self.used_positions), dtype=int)

    def _playlist_state_features(self) -> np.ndarray:
        """Small continuous feature vector summarising the playlist so far,
        for LinUCB's song-level context (B.7) — richer than the discretised
        tabular _state() buckets, appropriate since LinUCB handles
        high-dimensional continuous context directly."""
        recent = self.playlist[-3:]
        if recent:
            meta = _load_item_metadata()
            energy = float(meta["energy"].to_numpy()[recent].mean())
            valence = float(meta["valence"].to_numpy()[recent].mean())
        else:
            energy, valence = 0.5, 0.5

        if len(self.playlist) >= 2:
            vecs = self.content_matrix[self.playlist].toarray()
            sims = vecs @ vecs.T
            iu = np.triu_indices(len(self.playlist), k=1)
            diversity_so_far = float(1.0 - sims[iu].mean())
        else:
            diversity_so_far = 1.0

        near_cap = 1.0 if self.artist_counts and max(self.artist_counts.values()) >= self.config.artist_cap else 0.0
        position_frac = len(self.playlist) / self.config.playlist_length

        return np.array([position_frac, energy, valence, diversity_so_far, near_cap])

    def song_contexts(self) -> np.ndarray:
        """(n_candidates, d) LinUCB context matrix, one row per candidate
        position: user_profile ⊕ content_vec(a) ⊕ playlist_state_features
        (B.7). Includes used positions too — mask with unused_positions()."""
        profile = self.user_profile.toarray().ravel()
        content_vecs = self.content_matrix[self.candidate_items].toarray()
        n = len(self.candidate_items)
        profile_block = np.tile(profile, (n, 1))
        state_block = np.tile(self._playlist_state_features(), (n, 1))
        return np.hstack([profile_block, content_vecs, state_block])


def make_env_for_user(user_idx: int, candidates_df: pd.DataFrame, content_matrix, holdout_by_user_aff: dict,
                       config: EnvConfig = None) -> PlaylistEnv:
    """Convenience factory: slices candidates.parquet and holdout aff lookup
    down to one user before constructing PlaylistEnv."""
    user_candidates = candidates_df[candidates_df["user_idx"] == user_idx][["item_idx", "hybrid_score"]]
    user_holdout = holdout_by_user_aff.get(user_idx, {})
    return PlaylistEnv(user_idx, user_candidates, content_matrix, user_holdout, config)
