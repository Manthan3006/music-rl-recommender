"""Phase 7 ablation 2 — grounded-only vs grounded+simulated reward (B.12).

Every agent trained under each reward mode (grounded: 3 seeds, the Phase 6
primary run; grounded_simulated: 1 seed, the deferred secondary run) is frozen
(exploration disabled, exactly as in the Phase 6 table of record) and rolled
out once on the SAME 1,000 held-out users as that table, in the
grounded+simulated environment. From that one rollout we score:

  * hold-out quality: Precision@15, Recall@15 (real, observed playcounts)
  * simulated reward: the full step reward with the Tier-2 content-similarity
    relevance term (lambda=0.3)
  * grounded reward: recomputed from the same per-step decomposition with
    relevance = 0 for every non-held-out track
  * the reward's components: mean diversity term, artist-repeat rate, and the
    mean Tier-2 (simulated) relevance the agent collected per step

The point is B.12/Viva Q2: a reward that includes a modelled term can be
raised without landing a single real hold-out track. If an agent's reward
rank and its hit-rate rank disagree, the reward is being gamed.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PLAYLIST_LENGTH, RESULTS, SEED, W_DIV, W_DUP, W_REL, W_REP
from src.rl.environment import GROUNDED, GROUNDED_SIMULATED, EnvConfig, PlaylistEnv
from src.rl.evaluate_phase6_final import AGENT_ORDER, disable_exploration, heldout_users
from src.rl.train import N_EPISODES, SEEDS, SEEDS_BY_MODE, arm_best_scores, load_data, log, models_dir, sample_curriculum

OUT = RESULTS / "phase7"


def rollout(env, agent):
    state, done = env.reset(), False
    while not done:
        info = {"episode": N_EPISODES}
        if agent.ACTION_LEVEL == "song":
            info["song_contexts"], info["unused_positions"] = env.song_contexts(), env.unused_positions()
        else:
            info["arm_best_score"] = arm_best_scores(env)
        action = agent.select_action(state, info=info)
        state, _, done, _ = (env.step_song if agent.ACTION_LEVEL == "song" else env.step)(action)
    log_ = env.reward_log
    sim_reward = sum(r["reward"] for r in log_)
    grounded_rel = [float(env.holdout[r["item_idx"]]) if r["item_idx"] in env.holdout else 0.0 for r in log_]
    grounded_reward = sum(W_REL * g + W_DIV * r["div"] - W_REP * r["repeat"] - W_DUP * r["dup"]
                          for g, r in zip(grounded_rel, log_))
    hits = sum(1 for r in log_ if r["item_idx"] in env.holdout)
    n = len(log_)
    tier2 = [r["rel"] for r in log_ if r["item_idx"] not in env.holdout]
    return {
        "hits": hits, "holdout_size": len(env.holdout),
        "sim_reward": sim_reward, "grounded_reward": grounded_reward,
        "mean_div": float(np.mean([r["div"] for r in log_])),
        "repeat_rate": float(np.mean([r["repeat"] for r in log_])),
        "tier2_rel_per_step": float(np.sum(tier2) / n),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    content_matrix, candidates_by_user, holdout_by_user = load_data()
    users = np.array(sorted(set(candidates_by_user) & set(holdout_by_user)))
    eval_users = heldout_users(users, sample_curriculum(users, N_EPISODES, seed=SEED))
    log(f"{len(eval_users)} held-out users (asserted disjoint from the training curriculum).")

    rows = []
    for mode in (GROUNDED, GROUNDED_SIMULATED):
        for name in AGENT_ORDER:
            for seed in SEEDS_BY_MODE[mode]:
                with open(models_dir(mode) / f"{name}_seed{seed}.pkl", "rb") as f:
                    agent = disable_exploration(name, pickle.load(f))
                res = []
                for u in eval_users:
                    cand, hold = candidates_by_user[int(u)], holdout_by_user[int(u)]
                    env = PlaylistEnv(int(u), cand, content_matrix, hold, EnvConfig(reward_mode=GROUNDED_SIMULATED))
                    res.append(rollout(env, agent))
                d = pd.DataFrame(res)
                rows.append({
                    "trained_with": mode, "agent": name, "seed": seed,
                    "precision_at_15": d["hits"].sum() / (len(d) * PLAYLIST_LENGTH),
                    "recall_at_15": float(np.mean(d["hits"] / d["holdout_size"])),
                    **{c: float(d[c].mean()) for c in ["sim_reward", "grounded_reward", "mean_div", "repeat_rate", "tier2_rel_per_step"]},
                })
                log(f"  {mode:>18} {name:>9} seed={seed}: P@15={rows[-1]['precision_at_15']:.4f} "
                    f"sim_reward={rows[-1]['sim_reward']:.3f} grounded_reward={rows[-1]['grounded_reward']:.3f}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "ablation_reward_per_seed.csv", index=False)
    metrics = [c for c in per_seed.columns if c not in ("trained_with", "agent", "seed")]
    agg = per_seed.groupby(["trained_with", "agent"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["agent"] = pd.Categorical(agg["agent"], AGENT_ORDER, ordered=True)
    agg = agg.sort_values(["trained_with", "agent"])
    agg.to_csv(OUT / "ablation_reward.csv", index=False)

    corr_rows = []
    for mode, g in agg.groupby("trained_with"):
        for reward_col in ("sim_reward_mean", "grounded_reward_mean"):
            rho, _ = spearmanr(g[reward_col], g["precision_at_15_mean"])
            corr_rows.append({"trained_with": mode, "reward": reward_col.replace("_mean", ""),
                              "spearman_rho_vs_precision_at_15": rho, "n_agents": len(g)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "reward_vs_quality_correlation.csv", index=False)

    show = agg[["trained_with", "agent", "precision_at_15_mean", "recall_at_15_mean", "sim_reward_mean",
                "grounded_reward_mean", "mean_div_mean", "tier2_rel_per_step_mean"]]
    print("\n=== Reward ablation: every agent scored by hold-out quality and by both rewards ===")
    print(show.round(4).to_string(index=False))
    print("\n=== Spearman rank correlation, reward vs Precision@15, across the 6 agents ===")
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
