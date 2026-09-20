"""Phase 6 — RL agents (docs/SPEC.md Phase 6), all subclasses of BaseAgent.

BaseAgent (task 0) declares exactly two required methods:
`select_action(state) -> arm` and `update(state, action, reward, next_state,
done)`. Two extensions were needed to fit all six agents into one uniform
interface without the training loop reaching into any agent's internals:

1. `select_action(self, state, info=None)` — an optional `info` dict the
   training loop always populates with the same keys (`episode`,
   `arm_best_score`, and for song-level agents `song_contexts` /
   `unused_positions`); every agent reads only the keys it needs and ignores
   the rest. Needed because GreedyAgent ("always highest hybrid score") and
   LinUCB (song-level context, B.7) both need information beyond the 5-tuple
   tabular state to act, and QLearning/SARSA's epsilon decay is keyed to the
   training episode index, not something derivable from `state` alone.
2. `update(self, state, action, reward, next_state, done, next_action=None)`
   — SARSA is on-policy and its TD target needs the actual next action
   taken, which the fixed 5-argument signature has no room for. `next_action`
   defaults to None and every other agent ignores it.

Two class-level declarations tell the training loop how to drive an agent,
without it inspecting any agent's internal state:
  - `ACTION_LEVEL` ('arm', the default, or 'song' for LinUCB) — whether the
    loop calls `env.step(action)` or `env.step_song(action)`.
  - `NEEDS_NEXT_ACTION` (False by default, True for SARSA) — whether the
    loop must pre-select the next episode's action before calling `update`,
    so the on-policy target uses the action actually about to be taken.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from config import N_ARMS, SEED

N_ENERGY_BUCKETS = 3
N_VALENCE_BUCKETS = 3
N_DIVERSITY_BUCKETS = 3
N_ARTIST_FLAGS = 2
N_POSITION_BUCKETS = 3
N_STATES = N_ENERGY_BUCKETS * N_VALENCE_BUCKETS * N_DIVERSITY_BUCKETS * N_ARTIST_FLAGS * N_POSITION_BUCKETS  # 162


def state_to_idx(state: tuple) -> int:
    """B.7's 5-tuple -> a single 0..161 index for the Q-table."""
    energy, valence, diversity, near_cap, position = state
    idx = energy
    idx = idx * N_VALENCE_BUCKETS + valence
    idx = idx * N_DIVERSITY_BUCKETS + diversity
    idx = idx * N_ARTIST_FLAGS + near_cap
    idx = idx * N_POSITION_BUCKETS + position
    return idx


class BaseAgent(ABC):
    ACTION_LEVEL = "arm"
    NEEDS_NEXT_ACTION = False

    @abstractmethod
    def select_action(self, state, info: dict = None):
        ...

    @abstractmethod
    def update(self, state, action, reward, next_state, done, next_action=None):
        ...


class RandomAgent(BaseAgent):
    """Control: uniform-random arm every step."""

    def __init__(self, n_arms: int = N_ARMS, seed: int = SEED):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, info: dict = None):
        return int(self.rng.integers(0, self.n_arms))

    def update(self, state, action, reward, next_state, done, next_action=None):
        pass


class GreedyAgent(BaseAgent):
    """Control: always the arm containing the highest-hybrid-score unused
    candidate — the ablation that shows whether RL adds anything at all. No
    learning; deterministic given `info["arm_best_score"]`."""

    def select_action(self, state, info: dict = None):
        return int(np.argmax(info["arm_best_score"]))

    def update(self, state, action, reward, next_state, done, next_action=None):
        pass


class UCB1(BaseAgent):
    """Arm-level, no state: canonical UCB1, bonus = sqrt(2 ln(t) / n_a).
    `counts` (n_a) and `t` are instance attributes on the one agent object
    the training loop reuses for every episode of a run, so they accumulate
    across episodes and are never reset per episode. Every arm is pulled
    once (in random order) before the UCB rule takes over."""

    def __init__(self, n_arms: int = N_ARMS, seed: int = SEED):
        self.n_arms = n_arms
        self.q = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
        self.t = 0
        self.rng = np.random.default_rng(seed)  # untried-arm tie-breaking only

    def select_action(self, state, info: dict = None):
        self.t += 1
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(self.rng.choice(untried))
        ucb = self.q + np.sqrt(2.0 * np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, state, action, reward, next_state, done, next_action=None):
        self.counts[action] += 1
        self.q[action] += (reward - self.q[action]) / self.counts[action]


class LinUCB(BaseAgent):
    """Song-level (B.7): context = user_profile (+) content_vec(a) (+)
    playlist_state_features, a single shared ridge-regression model updated
    via the Sherman-Morrison rank-1 identity (no per-step matrix inversion).
    alpha controls the exploration-bonus width; B.7 sweeps
    alpha in {0.5, 1.0, 2.0}."""

    ACTION_LEVEL = "song"

    def __init__(self, dim: int, alpha: float = 1.0, seed: int = SEED):
        self.dim = dim
        self.alpha = alpha
        self.A_inv = np.eye(dim)
        self.b = np.zeros(dim)
        self.rng = np.random.default_rng(seed)  # tie-breaking only
        self._last_context = None

    def select_action(self, state, info: dict = None):
        contexts = info["song_contexts"]        # (n_candidates, dim)
        unused = info["unused_positions"]        # positions still available
        theta = self.A_inv @ self.b
        X = contexts[unused]                     # (m, dim)
        mean = X @ theta
        # diag(X @ A_inv @ X.T) without forming the full m x m matrix
        bonus = self.alpha * np.sqrt(np.einsum("ij,ij->i", X @ self.A_inv, X).clip(min=0))
        ucb = mean + bonus
        # tie-break randomly among the (near-)best, not always the first
        best = np.flatnonzero(ucb >= ucb.max() - 1e-12)
        choice = int(self.rng.choice(best))
        self._last_context = X[choice]
        return int(unused[choice])

    def update(self, state, action, reward, next_state, done, next_action=None):
        x = self._last_context
        Ainv_x = self.A_inv @ x
        denom = 1.0 + x @ Ainv_x
        self.A_inv -= np.outer(Ainv_x, Ainv_x) / denom
        self.b += reward * x


class TabularAgent(BaseAgent):
    """Shared 162x10 Q-table + epsilon-greedy behaviour policy for
    QLearningAgent/SARSAAgent (B.7/task 4-5). Epsilon decays linearly from
    eps_start to eps_end over n_episodes, keyed off info["episode"] rather
    than internal step counting, since the training loop is what knows the
    current episode index."""

    def __init__(self, n_states: int = N_STATES, n_arms: int = N_ARMS,
                 alpha: float = 0.1, gamma: float = 0.9,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 n_episodes: int = 20_000, seed: int = SEED):
        self.q = np.zeros((n_states, n_arms))
        self.n_arms = n_arms
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.n_episodes = n_episodes
        self.rng = np.random.default_rng(seed)

    def _epsilon(self, episode: int) -> float:
        frac = min(episode / max(self.n_episodes - 1, 1), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(self, state, info: dict = None):
        episode = (info or {}).get("episode", self.n_episodes)
        s = state_to_idx(state)
        if self.rng.random() < self._epsilon(episode):
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.q[s]))


class QLearningAgent(TabularAgent):
    """Off-policy: TD target bootstraps off max_a' Q(s', a')."""

    def update(self, state, action, reward, next_state, done, next_action=None):
        s, s2 = state_to_idx(state), state_to_idx(next_state)
        target = reward if done else reward + self.gamma * self.q[s2].max()
        self.q[s, action] += self.alpha * (target - self.q[s, action])


class SARSAAgent(TabularAgent):
    """On-policy: TD target uses the action actually taken next
    (next_action, supplied by the training loop — see NEEDS_NEXT_ACTION)."""

    NEEDS_NEXT_ACTION = True

    def update(self, state, action, reward, next_state, done, next_action=None):
        s, s2 = state_to_idx(state), state_to_idx(next_state)
        if done or next_action is None:
            target = reward
        else:
            target = reward + self.gamma * self.q[s2, next_action]
        self.q[s, action] += self.alpha * (target - self.q[s, action])
