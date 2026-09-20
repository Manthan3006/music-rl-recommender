"""Phase 6 — train and compare the six agents (docs/SPEC.md Phase 6).

Trains RandomAgent, GreedyAgent, UCB1, LinUCB, QLearningAgent, and
SARSAAgent for 5,000 episodes x 3 seeds each, against the SAME fixed
5,000-user episode curriculum for every agent/seed (sampled once, shared)
so learning-curve differences reflect the agent, not which users happened
to appear. (EpsilonGreedyMAB was dropped as redundant with UCB1.)

LinUCB's alpha is fixed at 1.0 — NOT tuned (the alpha grid search was
dropped due to compute cost; see docs/SPEC.md Phase 6).

Each reward mode is a full independent run written to its own subdirectory.
GROUNDED (Tier 1 only) is the PRIMARY experiment and the default.
GROUNDED_SIMULATED (Tier 1+2, B.12) is a SECONDARY experiment, deferred to
a single-seed run after Phase 7, to document that the simulated-relevance
term can be gamed. Usage: `python -m src.rl.train [grounded|grounded_simulated]`.
Per-episode holdout hits are logged alongside reward so Precision@15 /
Recall@15 — the primary metrics — come straight from the training run.

The environment/training loop depend only on
BaseAgent's interface (src/rl/agents.py's module docstring explains the two
small, uniform extensions to that interface all seven agents share).
"""

import os

# Pin BLAS to one thread BEFORE numpy is imported. These matrices are tiny
# (~90 x 223), so multi-threaded BLAS is pure overhead — and with several
# processes running it oversubscribes the cores: LinUCB measured 0.020
# s/episode single-threaded vs 0.108 default vs 0.64 with two parallel
# processes (the original ~64-hour estimate).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, MODELS, N_ARMS, RESULTS, SEED
from src.recommenders.common import load_core_content_matrix
from src.rl.agents import (
    GreedyAgent,
    LinUCB,
    QLearningAgent,
    RandomAgent,
    SARSAAgent,
    UCB1,
)
from src.rl.environment import GROUNDED, GROUNDED_SIMULATED, EnvConfig, PlaylistEnv

N_EPISODES = 5_000
SEEDS = [42, 43, 44]
ALL_MODES = [GROUNDED, GROUNDED_SIMULATED]
REWARD_MODES = [GROUNDED]  # default run; grounded_simulated is deferred (see docstring)
LINUCB_ALPHA = 1.0  # fixed, not tuned
FINAL_WINDOW = 500  # episodes (10% of the run) for "final performance" numbers


def models_dir(reward_mode):
    return MODELS / "rl" / reward_mode


def results_dir(reward_mode):
    return RESULTS / "phase6" / reward_mode


def curves_path(reward_mode):
    return RESULTS / f"rl_curves_{reward_mode}.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    content_matrix = load_core_content_matrix()
    candidates_df = pd.read_parquet(DATA_PROCESSED / "candidates.parquet")
    holdout_df = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")

    log("Pre-grouping candidates/holdout by user for O(1) per-episode lookup ...")
    candidates_by_user = {
        u: g[["item_idx", "hybrid_score"]].reset_index(drop=True)
        for u, g in candidates_df.groupby("user_idx")
    }
    holdout_by_user = {
        u: dict(zip(g["item_idx"], g["aff_ui"]))
        for u, g in holdout_df.groupby("user_idx")
    }
    return content_matrix, candidates_by_user, holdout_by_user


def sample_curriculum(users, n_episodes, seed):
    rng = np.random.default_rng(seed)
    if n_episodes <= len(users):
        return rng.choice(users, size=n_episodes, replace=False)
    return rng.choice(users, size=n_episodes, replace=True)


def arm_best_scores(env) -> np.ndarray:
    scores = np.full(env.config.n_arms, -np.inf)
    used = env.used_positions
    for k, positions in env.arms.items():
        avail = [p for p in positions if p not in used]
        if avail:
            scores[k] = env.candidate_scores[avail].max()
    return scores


def make_env(content_matrix, candidates_by_user, holdout_by_user, user_idx, reward_mode):
    cand = candidates_by_user.get(user_idx)
    holdout = holdout_by_user.get(user_idx, {})
    config = EnvConfig(reward_mode=reward_mode)
    return PlaylistEnv(int(user_idx), cand, content_matrix, holdout, config)


def run_episode(env, agent, episode_idx):
    state = env.reset()
    total_reward = 0.0
    pending_action = None
    done = False
    while not done:
        info = {"episode": episode_idx}
        if agent.ACTION_LEVEL == "song":
            info["song_contexts"] = env.song_contexts()
            info["unused_positions"] = env.unused_positions()
        else:
            info["arm_best_score"] = arm_best_scores(env)

        action = pending_action if pending_action is not None else agent.select_action(state, info=info)

        if agent.ACTION_LEVEL == "song":
            next_state, reward, done, decomp = env.step_song(action)
        else:
            next_state, reward, done, decomp = env.step(action)

        if agent.NEEDS_NEXT_ACTION:
            if done:
                next_action = None
            else:
                next_info = {"episode": episode_idx, "arm_best_score": arm_best_scores(env)}
                next_action = agent.select_action(next_state, info=next_info)
            agent.update(state, action, reward, next_state, done, next_action=next_action)
            pending_action = next_action
        else:
            agent.update(state, action, reward, next_state, done)
            pending_action = None

        state = next_state
        total_reward += reward

    hits = len(set(env.playlist) & set(env.holdout))
    return total_reward, hits, len(env.holdout)


def train_agent(agent_factory, agent_name, users, content_matrix, candidates_by_user, holdout_by_user,
                 n_episodes, seed, curriculum, reward_mode, log_every=2000):
    agent = agent_factory(seed)
    rewards = np.empty(n_episodes)
    hits = np.empty(n_episodes, dtype=int)
    holdout_sizes = np.empty(n_episodes, dtype=int)
    t0 = time.time()
    for i in range(n_episodes):
        user_idx = int(curriculum[i])
        env = make_env(content_matrix, candidates_by_user, holdout_by_user, user_idx, reward_mode)
        rewards[i], hits[i], holdout_sizes[i] = run_episode(env, agent, i)
        if (i + 1) % log_every == 0:
            recent_mean = rewards[max(0, i + 1 - log_every):i + 1].mean()
            log(f"  [{agent_name} seed={seed}] episode {i + 1}/{n_episodes} "
                f"rolling_mean_reward({log_every})={recent_mean:.4f} ({time.time() - t0:.0f}s elapsed)")
    return agent, rewards, hits, holdout_sizes


def run_mode(reward_mode, content_matrix, candidates_by_user, holdout_by_user, users, curriculum):
    results_dir(reward_mode).mkdir(parents=True, exist_ok=True)
    models_dir(reward_mode).mkdir(parents=True, exist_ok=True)
    log(f"===== reward_mode={reward_mode} =====")

    dim = content_matrix.shape[1] * 2 + 5  # user_profile + content_vec + playlist_state_features

    agent_factories = {
        "Random": lambda seed: RandomAgent(n_arms=N_ARMS, seed=seed),
        "Greedy": lambda seed: GreedyAgent(),
        "UCB1": lambda seed: UCB1(n_arms=N_ARMS, seed=seed),
        "LinUCB": lambda seed: LinUCB(dim=dim, alpha=LINUCB_ALPHA, seed=seed),
        "QLearning": lambda seed: QLearningAgent(n_arms=N_ARMS, n_episodes=N_EPISODES, seed=seed),
        "SARSA": lambda seed: SARSAAgent(n_arms=N_ARMS, n_episodes=N_EPISODES, seed=seed),
    }

    curve_frames = []
    ucb1_arm_rows = []
    for agent_name, factory in agent_factories.items():
        for seed in SEEDS:
            log(f"[{reward_mode}] Training {agent_name} (seed={seed}) for {N_EPISODES:,} episodes ...")
            t0 = time.time()
            agent, rewards, hits, holdout_sizes = train_agent(
                factory, agent_name, users, content_matrix, candidates_by_user, holdout_by_user,
                N_EPISODES, seed, curriculum, reward_mode,
            )
            dt = time.time() - t0
            log(f"  {agent_name} seed={seed} done in {dt:.0f}s, "
                f"final rolling_mean_reward({FINAL_WINDOW})={rewards[-FINAL_WINDOW:].mean():.4f}, "
                f"final Precision@15={hits[-FINAL_WINDOW:].sum() / (FINAL_WINDOW * 15):.4f}")

            curve_frames.append(pd.DataFrame({
                "agent": agent_name, "seed": seed, "episode": np.arange(N_EPISODES),
                "reward": rewards, "hits": hits, "holdout_size": holdout_sizes,
            }))

            if agent_name == "UCB1":
                frac = agent.counts / agent.counts.sum()
                ucb1_arm_rows.append({"seed": seed, **{f"arm_{k}": float(frac[k]) for k in range(N_ARMS)}})
                log(f"  UCB1 seed={seed} cumulative arm-pull distribution (t={agent.t:,}): "
                    + " ".join(f"{k}:{frac[k]:.3f}" for k in range(N_ARMS)))

            with open(models_dir(reward_mode) / f"{agent_name}_seed{seed}.pkl", "wb") as f:
                pickle.dump(agent, f)

            # checkpoint after every (agent, seed) in case of interruption
            pd.concat(curve_frames, ignore_index=True).to_csv(curves_path(reward_mode), index=False)
            if ucb1_arm_rows:
                pd.DataFrame(ucb1_arm_rows).to_csv(results_dir(reward_mode) / "ucb1_arm_distribution.csv", index=False)

    log(f"[{reward_mode}] Saved {curves_path(reward_mode)} and {models_dir(reward_mode)}/*.pkl.")


def main():
    modes = [sys.argv[1]] if len(sys.argv) > 1 else REWARD_MODES
    assert all(m in ALL_MODES for m in modes), f"reward mode must be one of {ALL_MODES}"

    content_matrix, candidates_by_user, holdout_by_user = load_data()
    users = np.array(sorted(set(candidates_by_user) & set(holdout_by_user)))
    log(f"{len(users):,} users available for training.")

    curriculum = sample_curriculum(users, N_EPISODES, seed=SEED)
    log(f"Sampled a fixed {N_EPISODES}-episode user curriculum (seed={SEED}), "
        f"shared across every agent, training seed, and reward mode.")

    for reward_mode in modes:
        run_mode(reward_mode, content_matrix, candidates_by_user, holdout_by_user, users, curriculum)


if __name__ == "__main__":
    main()
