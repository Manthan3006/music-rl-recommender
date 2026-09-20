"""Phase 6 diagnosis — why do EpsilonGreedyMAB/UCB1/QLearning/SARSA fail to
beat Random (see results/phase6/final_performance.csv, evaluate_phase6.py)?

Loads the seed=42 checkpoint for Greedy, QLearning, LinUCB, UCB1, Random
from models/rl/, freezes each policy (select_action only, no further
`update` calls — this is a read-only diagnostic, not additional training),
and replays it on the *same* final 1,000 users of the fixed 20,000-episode
curriculum used in training (src/rl/train.py's sample_curriculum, same seed),
at their original episode indices (19,000-19,999) so epsilon-decay-dependent
agents (QLearning) see the same near-fully-decayed epsilon they had at the
end of training.

For each agent, records the full per-step reward decomposition (already
emitted by PlaylistEnv._apply_action: arm, rel, div, repeat, dup, reward)
and the finished 15-track playlist, then aggregates:

  1. results/phase6/diagnosis/arm_distribution.csv — how often each of the
     10 arms is chosen, per agent.
  2. results/phase6/diagnosis/reward_decomposition.csv — mean per-step rel /
     div / repeat-rate / dup-rate / reward, per agent.
  3. results/phase6/diagnosis/greedy_vs_random.csv — Greedy's mean distinct
     arms used per episode, and its mean div/repeat vs Random's.
  4. results/phase6/diagnosis/precision_recall.csv — Precision@15 and
     Recall@15 (against the 5-track holdout) per agent.

CAVEAT (found afterwards): freezing an online bandit and replaying it with no
`update` calls makes UCB1 pick one arm forever (its q/counts never change,
and all arms' UCB scores were within ~0.001 of each other), so the "UCB1
selects arm 3 on 100% of steps" result here is an artifact of this frozen
replay, NOT a training bug — UCB1's saved training-time pull counts were
spread across all 10 arms. Trust the training-time counts (train.py logs
them to results/phase6/<mode>/ucb1_arm_distribution.csv), not that column.
This script targeted the superseded first run (models/rl/*.pkl, W_REL=1.0).

Nothing here retrains or overwrites any existing model, curve, or Phase 6
acceptance result — this only reads models/rl/*.pkl and writes new files
under results/phase6/diagnosis/.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, MODELS, RESULTS, SEED
from src.recommenders.common import load_core_content_matrix
from src.rl.environment import GROUNDED_SIMULATED, EnvConfig, PlaylistEnv
from src.rl.train import arm_best_scores, sample_curriculum

N_EPISODES = 20_000
N_FINAL = 1_000
DIAG_SEED = 42
AGENTS = ["Greedy", "QLearning", "LinUCB", "UCB1", "Random"]
MODELS_DIR = MODELS / "rl"
OUT_DIR = RESULTS / "phase6" / "diagnosis"


def load_data():
    content_matrix = load_core_content_matrix()
    candidates_df = pd.read_parquet(DATA_PROCESSED / "candidates.parquet")
    holdout_df = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")
    candidates_by_user = {
        u: g[["item_idx", "hybrid_score"]].reset_index(drop=True)
        for u, g in candidates_df.groupby("user_idx")
    }
    holdout_by_user = {
        u: dict(zip(g["item_idx"], g["aff_ui"]))
        for u, g in holdout_df.groupby("user_idx")
    }
    return content_matrix, candidates_by_user, holdout_by_user


def make_env(content_matrix, candidates_by_user, holdout_by_user, user_idx):
    cand = candidates_by_user.get(user_idx)
    holdout = holdout_by_user.get(user_idx, {})
    config = EnvConfig(reward_mode=GROUNDED_SIMULATED)
    return PlaylistEnv(int(user_idx), cand, content_matrix, holdout, config)


def run_frozen_episode(env, agent, episode_idx):
    """select_action only — no agent.update() — so this is a read-only
    evaluation of the already-trained (loaded) policy."""
    state = env.reset()
    decomps = []
    done = False
    while not done:
        info = {"episode": episode_idx}
        if agent.ACTION_LEVEL == "song":
            info["song_contexts"] = env.song_contexts()
            info["unused_positions"] = env.unused_positions()
        else:
            info["arm_best_score"] = arm_best_scores(env)

        action = agent.select_action(state, info=info)
        if agent.ACTION_LEVEL == "song":
            next_state, reward, done, decomp = env.step_song(action)
        else:
            next_state, reward, done, decomp = env.step(action)

        if "pool_exhausted" not in decomp:
            decomps.append(decomp)
        state = next_state

    return decomps, list(env.playlist)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content_matrix, candidates_by_user, holdout_by_user = load_data()
    users = np.array(sorted(set(candidates_by_user) & set(holdout_by_user)))
    curriculum = sample_curriculum(users, N_EPISODES, seed=SEED)
    final_curriculum = curriculum[-N_FINAL:]
    base_episode_idx = N_EPISODES - N_FINAL

    all_steps = []
    pr_rows = []

    for agent_name in AGENTS:
        with open(MODELS_DIR / f"{agent_name}_seed{DIAG_SEED}.pkl", "rb") as f:
            agent = pickle.load(f)

        total_hits, total_holdout, total_precision_num = 0, 0, 0
        distinct_arms_per_episode = []

        for i, user_idx in enumerate(final_curriculum):
            episode_idx = base_episode_idx + i
            env = make_env(content_matrix, candidates_by_user, holdout_by_user, int(user_idx))
            decomps, playlist = run_frozen_episode(env, agent, episode_idx)

            for d in decomps:
                all_steps.append({"agent": agent_name, "episode": i, **d})
            distinct_arms_per_episode.append(len({d["arm"] for d in decomps}))

            holdout_items = set(holdout_by_user.get(int(user_idx), {}))
            hits = len(set(playlist) & holdout_items)
            total_hits += hits
            total_holdout += len(holdout_items)
            total_precision_num += hits

        pr_rows.append({
            "agent": agent_name,
            "n_episodes": len(final_curriculum),
            "mean_hits_per_playlist": total_hits / len(final_curriculum),
            "precision_at_15": total_precision_num / (len(final_curriculum) * 15),
            "recall_at_15": total_hits / total_holdout,
            "mean_distinct_arms_per_episode": float(np.mean(distinct_arms_per_episode)),
        })
        print(f"  done: {agent_name}")

    steps = pd.DataFrame(all_steps)

    # 1. Arm-selection distribution
    arm_counts = steps.groupby(["agent", "arm"]).size().rename("count").reset_index()
    arm_counts["pct"] = arm_counts["count"] / arm_counts.groupby("agent")["count"].transform("sum")
    arm_dist = arm_counts.pivot(index="arm", columns="agent", values="pct").reindex(columns=AGENTS).fillna(0.0)
    arm_dist.to_csv(OUT_DIR / "arm_distribution.csv")
    print("\n=== ARM-SELECTION DISTRIBUTION (fraction of steps, final 1,000 episodes) ===")
    print(arm_dist.to_string(float_format="%.3f"))

    # 2. Reward decomposition
    decomp = steps.groupby("agent").agg(
        mean_rel=("rel", "mean"), mean_div=("div", "mean"),
        mean_repeat_rate=("repeat", "mean"), mean_dup_rate=("dup", "mean"),
        mean_reward=("reward", "mean"),
    ).reindex(AGENTS)
    decomp.to_csv(OUT_DIR / "reward_decomposition.csv")
    print("\n=== MEAN PER-STEP REWARD DECOMPOSITION ===")
    print(decomp.to_string(float_format="%.4f"))

    # 3. Greedy vs Random
    pr_df = pd.DataFrame(pr_rows).set_index("agent")
    greedy_vs_random = pd.DataFrame({
        "Greedy": [pr_df.loc["Greedy", "mean_distinct_arms_per_episode"],
                   decomp.loc["Greedy", "mean_div"], decomp.loc["Greedy", "mean_repeat_rate"]],
        "Random": [pr_df.loc["Random", "mean_distinct_arms_per_episode"],
                   decomp.loc["Random", "mean_div"], decomp.loc["Random", "mean_repeat_rate"]],
    }, index=["mean_distinct_arms_per_episode", "mean_div", "mean_repeat_rate"])
    greedy_vs_random.to_csv(OUT_DIR / "greedy_vs_random.csv")
    print("\n=== GREEDY vs RANDOM ===")
    print(greedy_vs_random.to_string(float_format="%.4f"))

    # 4. Precision@15 / Recall@15
    pr_out = pr_df[["n_episodes", "mean_hits_per_playlist", "precision_at_15", "recall_at_15"]].reindex(AGENTS)
    pr_out.to_csv(OUT_DIR / "precision_recall.csv")
    print("\n=== PRECISION@15 / RECALL@15 (vs 5-track holdout, final 1,000 episodes) ===")
    print(pr_out.to_string(float_format="%.4f"))

    print(f"\nSaved diagnostics to {OUT_DIR}/")


if __name__ == "__main__":
    main()
