"""Phase 5 acceptance — random-policy baseline over 1,000 episodes.

Runs a uniform-random arm policy for 1,000 episodes (1,000 distinct sampled
users, one episode each) in both reward modes (B.12 Tier 1 grounded-only,
Tier 1+2 grounded+simulated — task 5), and reports per-episode total reward
mean/std plus the mean per-step reward decomposition (rel/div/repeat/dup) so
it's clear which term dominates (task 6 / acceptance).

The artist cap is a soft penalty, not a hard filter (see environment.py's
module docstring), so a uniform-random policy is expected to trigger some
non-zero rate of cap violations (repeat=1 steps) — this run reports that
rate directly (mean_repeat = fraction of steps that violate the cap;
pct_episodes_with_violation = fraction of the 1,000 episodes with >=1
violation), rather than the structural zero a hard filter would force.

pytest unit tests (reward determinism, cap-violation penalty, no
duplicates) live in tests/test_env.py and are run separately
(`python -m pytest tests/test_env.py`).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, N_ARMS, RESULTS, SEED
from src.recommenders.common import load_core_content_matrix
from src.rl.environment import GROUNDED, GROUNDED_SIMULATED, EnvConfig, PlaylistEnv

N_EPISODES = 1000
OUT_DIR = RESULTS / "phase5"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    content_matrix = load_core_content_matrix()
    candidates_df = pd.read_parquet(DATA_PROCESSED / "candidates.parquet")
    holdout_df = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")
    return content_matrix, candidates_df, holdout_df


def sample_users(candidates_df, holdout_df, n, seed):
    users = np.intersect1d(candidates_df["user_idx"].unique(), holdout_df["user_idx"].unique())
    rng = np.random.default_rng(seed)
    return rng.choice(users, size=n, replace=False)


def run_episode(content_matrix, candidates_df, holdout_df, user_idx, reward_mode, seed):
    cand = candidates_df[candidates_df["user_idx"] == user_idx][["item_idx", "hybrid_score"]]
    rows = holdout_df[holdout_df["user_idx"] == user_idx]
    holdout = dict(zip(rows["item_idx"], rows["aff_ui"]))

    config = EnvConfig(reward_mode=reward_mode)
    env = PlaylistEnv(int(user_idx), cand, content_matrix, holdout, config)

    rng = np.random.default_rng(seed)
    total_reward = 0.0
    done = False
    while not done:
        arm = int(rng.integers(0, N_ARMS))
        _, reward, done, info = env.step(arm)
        if info.get("pool_exhausted"):
            break
        total_reward += reward

    return total_reward, env.reward_log


def run_baseline(content_matrix, candidates_df, holdout_df, users, reward_mode):
    episode_rewards = []
    decomposition_rows = []
    for i, user_idx in enumerate(users):
        total_reward, reward_log = run_episode(
            content_matrix, candidates_df, holdout_df, user_idx, reward_mode, seed=SEED + i
        )
        episode_rewards.append(total_reward)
        for d in reward_log:
            d["episode"] = i
        decomposition_rows.extend(reward_log)
        if (i + 1) % 200 == 0:
            log(f"  [{reward_mode}] {i + 1}/{len(users)} episodes")

    return np.array(episode_rewards), pd.DataFrame(decomposition_rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content_matrix, candidates_df, holdout_df = load_data()
    users = sample_users(candidates_df, holdout_df, N_EPISODES, seed=SEED)
    log(f"Sampled {len(users):,} users for the random-policy baseline.")

    summary_rows = []
    for reward_mode in [GROUNDED, GROUNDED_SIMULATED]:
        log(f"Running {N_EPISODES} random-policy episodes, reward_mode={reward_mode} ...")
        t0 = time.time()
        episode_rewards, decomposition = run_baseline(content_matrix, candidates_df, holdout_df, users, reward_mode)
        dt = time.time() - t0

        mean_r, std_r = float(episode_rewards.mean()), float(episode_rewards.std())
        log(f"  {reward_mode}: mean total reward={mean_r:.4f} std={std_r:.4f} "
            f"(n={len(episode_rewards)}, {dt:.1f}s)")

        pd.DataFrame({"episode": np.arange(len(episode_rewards)), "total_reward": episode_rewards}) \
            .to_csv(OUT_DIR / f"random_policy_episode_rewards_{reward_mode}.csv", index=False)

        mean_decomp = decomposition[["rel", "div", "repeat", "dup", "reward"]].mean()
        pct_episodes_with_violation = float(decomposition.groupby("episode")["repeat"].max().mean())
        log(f"  {reward_mode}: mean per-step decomposition -> "
            f"rel={mean_decomp['rel']:.4f} div={mean_decomp['div']:.4f} "
            f"repeat={mean_decomp['repeat']:.4f} dup={mean_decomp['dup']:.4f}")
        log(f"  {reward_mode}: artist-cap violation rate -> "
            f"{mean_decomp['repeat']:.4f} of steps, "
            f"{pct_episodes_with_violation:.4f} of episodes have >=1 violation")
        decomposition.to_csv(OUT_DIR / f"random_policy_step_decomposition_{reward_mode}.csv", index=False)

        summary_rows.append({
            "reward_mode": reward_mode, "n_episodes": len(episode_rewards),
            "mean_total_reward": mean_r, "std_total_reward": std_r,
            "mean_rel": mean_decomp["rel"], "mean_div": mean_decomp["div"],
            "mean_repeat": mean_decomp["repeat"], "mean_dup": mean_decomp["dup"],
            "pct_episodes_with_cap_violation": pct_episodes_with_violation,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "random_policy_summary.csv", index=False)

    print("\n=== PHASE 5 RANDOM-POLICY BASELINE (1,000 episodes) ===")
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved per-episode rewards, per-step decompositions, and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
