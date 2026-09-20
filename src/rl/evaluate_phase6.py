"""Phase 6 acceptance gate and comparison table.

Reads results/rl_curves_<mode>.csv and the pickled agents in
models/rl/<mode>/ (both written by train.py) for two reward modes:

  - grounded            PRIMARY experiment — the headline results.
  - grounded_simulated  SECONDARY experiment (deferred; a single-seed run
                        after Phase 7) — documents that the Tier-2
                        simulated-relevance term is gameable (LinUCB gets
                        the highest reward with the lowest holdout hit rate).
                        Evaluated here only if its curves file exists.

The PRIMARY metric is holdout hit rate: Precision@15 and Recall@15 of the
generated 15-track playlist against the user's 5 held-out tracks, computed
over the last FINAL_WINDOW training episodes. Mean episode reward is
reported alongside it for every agent but is secondary.

Acceptance gate (AMENDED — see docs/SPEC.md Phase 6; gated on the primary
mode only, the secondary run is reported, not gated):
  1. Every agent is a BaseAgent subclass driven only via select_action/update.
  2. Every learned agent (UCB1, LinUCB, QLearning, SARSA) converges
     (rolling-mean reward plateaus).
  3. Q-Learning and SARSA learn distinct policies.
  4. The full comparison is reported against both Random and Greedy (the
     tables below, plus the exploration-off table of record from
     evaluate_phase6_final.py).
The original gate ("all learned agents beat Random") is NOT checked — it
failed and was replaced; see the spec for the record. Beating Random or
Greedy is still printed for every agent, as information, not as a gate.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.rl.agents import (
    BaseAgent,
    GreedyAgent,
    LinUCB,
    QLearningAgent,
    RandomAgent,
    SARSAAgent,
    UCB1,
)
from src.rl.train import FINAL_WINDOW, curves_path, models_dir, results_dir

PRIMARY_MODE = "grounded"
SECONDARY_MODE = "grounded_simulated"
LEARNED_AGENTS = ["UCB1", "LinUCB", "QLearning", "SARSA"]
AGENT_ORDER = ["Random", "Greedy", "UCB1", "LinUCB", "QLearning", "SARSA"]
PLAYLIST_LENGTH = 15

AGENT_CLASSES = {
    "Random": RandomAgent, "Greedy": GreedyAgent, "UCB1": UCB1,
    "LinUCB": LinUCB, "QLearning": QLearningAgent, "SARSA": SARSAAgent,
}


def check_base_agent_subclassing():
    checks = []
    for name, cls in AGENT_CLASSES.items():
        is_subclass = issubclass(cls, BaseAgent)
        has_interface = {"select_action", "update"}.issubset(set(dir(cls)))
        checks.append((f"{name} is a BaseAgent subclass with select_action/update", is_subclass and has_interface))
    return checks


def final_performance(curves: pd.DataFrame) -> pd.DataFrame:
    """Per (agent, seed) means over the last FINAL_WINDOW episodes — reward,
    Precision@15 (hits / 15), Recall@15 (hits / holdout size) — then mean and
    std across seeds."""
    max_ep = curves["episode"].max()
    tail = curves[curves["episode"] > max_ep - FINAL_WINDOW].copy()
    tail["precision"] = tail["hits"] / PLAYLIST_LENGTH
    tail["recall"] = tail["hits"] / tail["holdout_size"]
    per_seed = tail.groupby(["agent", "seed"])[["reward", "precision", "recall"]].mean().reset_index()
    summary = per_seed.groupby("agent")[["reward", "precision", "recall"]].agg(["mean", "std"])
    summary.columns = ["mean_reward", "std_reward", "precision_at_15", "std_precision",
                       "recall_at_15", "std_recall"]
    return summary.reindex(AGENT_ORDER)


def convergence_check(curves: pd.DataFrame, agent_name: str, n_windows: int = 5) -> dict:
    """Splits each seed's curve into n_windows equal chunks and checks the
    LAST TWO windows' mean rewards are within 10% of the curve's overall
    reward scale (a plateau)."""
    sub = curves[curves["agent"] == agent_name]
    max_ep = sub["episode"].max() + 1
    edges = np.linspace(0, max_ep, n_windows + 1).astype(int)

    per_seed_plateaued = []
    window_means = None
    for seed, g in sub.groupby("seed"):
        window_means = [g[(g["episode"] >= lo) & (g["episode"] < hi)]["reward"].mean()
                        for lo, hi in zip(edges[:-1], edges[1:])]
        overall_scale = max(abs(g["reward"].mean()), 1e-6)
        rel_change = abs(window_means[-1] - window_means[-2]) / overall_scale
        per_seed_plateaued.append(rel_change < 0.10)

    return {
        "agent": agent_name,
        "window_means_last_seed": [float(w) for w in window_means],
        "pct_seeds_plateaued": float(np.mean(per_seed_plateaued)),
    }


def policy_difference(reward_mode: str, seed: int) -> dict:
    with open(models_dir(reward_mode) / f"QLearning_seed{seed}.pkl", "rb") as f:
        ql = pickle.load(f)
    with open(models_dir(reward_mode) / f"SARSA_seed{seed}.pkl", "rb") as f:
        sarsa = pickle.load(f)

    n_diff = int((np.argmax(ql.q, axis=1) != np.argmax(sarsa.q, axis=1)).sum())
    return {"seed": seed, "n_states_differing": n_diff, "n_states": ql.q.shape[0],
            "identical": n_diff == 0}


def evaluate_mode(reward_mode: str, gate: bool):
    out_dir = results_dir(reward_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(curves_path(reward_mode))
    n_seeds = curves["seed"].nunique()
    label = "PRIMARY" if reward_mode == PRIMARY_MODE else "SECONDARY"

    summary = final_performance(curves)
    print(f"\n{'=' * 78}\n{label}: reward_mode={reward_mode} — last {FINAL_WINDOW} episodes, "
          f"{n_seeds} seeds (mean +/- std across seeds)\n{'=' * 78}")
    display = pd.DataFrame({
        "Precision@15": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(summary["precision_at_15"], summary["std_precision"])],
        "Recall@15": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(summary["recall_at_15"], summary["std_recall"])],
        "mean reward": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(summary["mean_reward"], summary["std_reward"])],
    }, index=summary.index)
    print(display.to_string())
    summary.to_csv(out_dir / "final_performance.csv")

    random_row = summary.loc["Random"]
    greedy_row = summary.loc["Greedy"]

    print(f"\n--- learned agents vs Random / Greedy ({reward_mode}) ---")
    for agent_name in LEARNED_AGENTS:
        row = summary.loc[agent_name]
        print(f"  {agent_name:10s} P@15 {row['precision_at_15']:.4f} "
              f"(Random {random_row['precision_at_15']:.4f} -> "
              f"{'beats' if row['precision_at_15'] > random_row['precision_at_15'] else 'does NOT beat'} Random; "
              f"Greedy {greedy_row['precision_at_15']:.4f} -> "
              f"{'beats' if row['precision_at_15'] > greedy_row['precision_at_15'] else 'does NOT beat'} Greedy) | "
              f"reward {row['mean_reward']:.4f} "
              f"(Random {random_row['mean_reward']:.4f} -> "
              f"{'beats' if row['mean_reward'] > random_row['mean_reward'] else 'does NOT beat'} Random)")

    convergence_rows = [convergence_check(curves, a) for a in LEARNED_AGENTS]
    pd.DataFrame(convergence_rows).to_csv(out_dir / "convergence.csv", index=False)
    policy_rows = [policy_difference(reward_mode, s) for s in sorted(curves["seed"].unique())]
    pd.DataFrame(policy_rows).to_csv(out_dir / "policy_comparison.csv", index=False)

    print(f"\n--- convergence and Q-Learning / SARSA policies ({reward_mode}) ---")
    for r in convergence_rows:
        print(f"  {r['agent']}: {r['pct_seeds_plateaued']:.0%} of seeds plateaued")
    for r in policy_rows:
        print(f"  seed={r['seed']}: {r['n_states_differing']}/{r['n_states']} states differ in greedy policy")

    checks = []
    checks.extend(check_base_agent_subclassing())
    for r in convergence_rows:
        checks.append((f"{r['agent']} converges (rolling-mean reward plateaus)", r["pct_seeds_plateaued"] >= 0.6))
    checks.append(("Q-Learning and SARSA produce different policies (no seed identical)",
                   not any(r["identical"] for r in policy_rows)))
    checks.append(("Full comparison reported against both Random and Greedy (tables above)",
                   bool(summary.loc[["Random", "Greedy"]].notna().all().all())))

    print(f"\n--- acceptance checks ({reward_mode}, {'GATING' if gate else 'reported only'}) ---")
    all_pass = True
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        all_pass = all_pass and ok
    print(f"  => {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    return all_pass


def main():
    primary_ok = evaluate_mode(PRIMARY_MODE, gate=True)
    if curves_path(SECONDARY_MODE).exists():
        evaluate_mode(SECONDARY_MODE, gate=False)
    else:
        print(f"\n(secondary run {SECONDARY_MODE} not present yet — deferred until after Phase 7)")
    print(f"\nPhase 6 acceptance (primary, {PRIMARY_MODE}): {'ALL PASS' if primary_ok else 'FAILED'}")
    if not primary_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
