"""Phase 6 table of record: every agent re-evaluated with exploration OFF on
1,000 held-out episodes (grounded reward mode, the primary experiment).

The training-time numbers in evaluate_phase6.py are measured while the
agents are still exploring and learning; this script freezes each trained
checkpoint (models/rl/grounded/<agent>_seed<s>.pkl, 3 seeds) and evaluates
it with no further updates on users the agents never trained on:

  - Q-Learning / SARSA: epsilon = 0 (pure argmax over the learned Q-table).
  - UCB1: no confidence bonus — always the arm with the highest estimated
    value (argmax q).
  - LinUCB: alpha = 0 — the exploration bonus is removed, leaving the
    ridge-regression mean estimate.
  - Random / Greedy: unchanged (they have no exploration to disable).

Held-out episodes: 1,000 users sampled (SEED) from the users NOT in the
5,000-user training curriculum; asserted disjoint from it (leakage check).
Metrics per playlist: Precision@15 = hits/15, Recall@15 = hits/5, and total
grounded episode reward; reported as mean +/- std across the 3 training seeds.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import PLAYLIST_LENGTH, SEED
from src.rl.environment import GROUNDED
from src.rl.train import (
    N_EPISODES,
    SEEDS,
    arm_best_scores,
    load_data,
    make_env,
    models_dir,
    results_dir,
    sample_curriculum,
)

N_EVAL = 1_000
AGENT_ORDER = ["Random", "Greedy", "UCB1", "LinUCB", "QLearning", "SARSA"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def disable_exploration(name, agent):
    if name in ("QLearning", "SARSA"):
        agent.eps_start = agent.eps_end = 0.0
    elif name == "LinUCB":
        agent.alpha = 0.0
    elif name == "UCB1":
        agent.select_action = lambda state, info=None: int(np.argmax(agent.q))
    return agent


def run_frozen_episode(env, agent):
    """select_action only — the agent is never updated."""
    state = env.reset()
    total_reward, done = 0.0, False
    while not done:
        info = {"episode": N_EPISODES}
        if agent.ACTION_LEVEL == "song":
            info["song_contexts"] = env.song_contexts()
            info["unused_positions"] = env.unused_positions()
        else:
            info["arm_best_score"] = arm_best_scores(env)
        action = agent.select_action(state, info=info)
        if agent.ACTION_LEVEL == "song":
            state, reward, done, _ = env.step_song(action)
        else:
            state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, len(set(env.playlist) & set(env.holdout)), len(env.holdout)


def heldout_users(users, curriculum):
    train_users = set(int(u) for u in curriculum)
    pool = np.array([u for u in users if int(u) not in train_users])
    chosen = np.random.default_rng(SEED).choice(pool, size=N_EVAL, replace=False)
    assert not (set(int(u) for u in chosen) & train_users), "LEAKAGE: eval user appears in the RL training curriculum"
    return chosen


def main():
    content_matrix, candidates_by_user, holdout_by_user = load_data()
    users = np.array(sorted(set(candidates_by_user) & set(holdout_by_user)))
    curriculum = sample_curriculum(users, N_EPISODES, seed=SEED)
    eval_users = heldout_users(users, curriculum)
    log(f"{len(eval_users)} held-out users (disjoint from the {N_EPISODES}-user training curriculum).")

    rows = []
    for name in AGENT_ORDER:
        for seed in SEEDS:
            with open(models_dir(GROUNDED) / f"{name}_seed{seed}.pkl", "rb") as f:
                agent = disable_exploration(name, pickle.load(f))
            rewards, hits, sizes = [], [], []
            for u in eval_users:
                env = make_env(content_matrix, candidates_by_user, holdout_by_user, int(u), GROUNDED)
                r, h, s = run_frozen_episode(env, agent)
                rewards.append(r); hits.append(h); sizes.append(s)
            hits, sizes = np.array(hits), np.array(sizes)
            rows.append({
                "agent": name, "seed": seed,
                "precision_at_15": hits.sum() / (len(hits) * PLAYLIST_LENGTH),
                "recall_at_15": float(np.mean(hits / sizes)),
                "mean_hits_per_playlist": float(hits.mean()),
                "mean_reward": float(np.mean(rewards)),
            })
            log(f"  {name} seed={seed}: P@15={rows[-1]['precision_at_15']:.4f} "
                f"R@15={rows[-1]['recall_at_15']:.4f} reward={rows[-1]['mean_reward']:.4f}")

    per_seed = pd.DataFrame(rows)
    out = results_dir(GROUNDED)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "final_eval_no_exploration_per_seed.csv", index=False)

    metrics = ["precision_at_15", "recall_at_15", "mean_hits_per_playlist", "mean_reward"]
    agg = per_seed.groupby("agent")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reindex(AGENT_ORDER)
    agg.to_csv(out / "final_eval_no_exploration.csv")

    print(f"\n=== TABLE OF RECORD: exploration disabled, {N_EVAL} held-out episodes, "
          f"{len(SEEDS)} training seeds (mean +/- std) ===")
    disp = pd.DataFrame({
        "Precision@15": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(agg["precision_at_15_mean"], agg["precision_at_15_std"])],
        "Recall@15": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(agg["recall_at_15_mean"], agg["recall_at_15_std"])],
        "hits/15": [f"{a:.3f}" for a in agg["mean_hits_per_playlist_mean"]],
        "mean reward": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(agg["mean_reward_mean"], agg["mean_reward_std"])],
    }, index=agg.index)
    print(disp.to_string())
    print(f"\nSaved to {out}/final_eval_no_exploration*.csv")


if __name__ == "__main__":
    main()
