"""Phase 7 figures -> results/figures/.

  model_comparison.png   Recall@K and NDCG@K bars, +/-1 std across the 5 folds
  rl_learning_curves.png rolling reward and Precision@15 over training, 95% CI
                         over the 3 grounded seeds (t-interval)
  coverage_vs_novelty.png catalogue coverage vs novelty, K=10 and K=20
  reward_hacking.png     simulated reward vs real Precision@15, agents trained
                         on the grounded+simulated reward (single seed)

Colour follows the model family (fixed order, never rank), every mark is also
labelled directly, and the data behind each figure is a CSV in results/.
Palette: the validated reference categorical order (blue, orange, aqua,
yellow) on a light surface; baselines are neutral grey.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS

FIG = RESULTS / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8d8c86"
FAMILY = {
    "Random": ("baseline", GREY), "Popularity": ("baseline", GREY),
    "Content": ("single recommender", BLUE), "ItemCF": ("single recommender", BLUE), "ALS": ("single recommender", BLUE),
    "Hybrid": ("hybrid (= Greedy)", ORANGE),
    "Hybrid+RandomArm": ("hybrid + RL agent", AQUA), "Hybrid+UCB1": ("hybrid + RL agent", AQUA),
    "Hybrid+LinUCB": ("hybrid + RL agent", AQUA), "Hybrid+QLearning": ("hybrid + RL agent", AQUA),
    "Hybrid+SARSA": ("hybrid + RL agent", AQUA),
}
FAMILY_ORDER = ["baseline", "single recommender", "hybrid (= Greedy)", "hybrid + RL agent"]
FAMILY_COLOR = {f: next(c for ff, c in FAMILY.values() if ff == f) for f in FAMILY_ORDER}
MARKER = {"baseline": "s", "single recommender": "o", "hybrid (= Greedy)": "D", "hybrid + RL agent": "^"}
SHORT = {"Hybrid+RandomArm": "+RandomArm", "Hybrid+UCB1": "+UCB1", "Hybrid+LinUCB": "+LinUCB",
         "Hybrid+QLearning": "+QLearning", "Hybrid+SARSA": "+SARSA"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def family_legend(ax, loc="lower right"):
    handles = [plt.Line2D([], [], marker=MARKER[f], color=FAMILY_COLOR[f], linestyle="", markersize=7, label=f)
               for f in FAMILY_ORDER]
    ax.legend(handles=handles, loc=loc, fontsize=9)


def model_comparison(fc):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True)
    order = list(FAMILY)
    for r, metric in enumerate(["recall", "ndcg"]):
        for c, k in enumerate([10, 20]):
            ax = axes[r, c]
            d = fc[(fc.K == k) & (fc.metric == metric)].set_index("config").loc[order]
            y = np.arange(len(order))[::-1]
            ax.barh(y, d["mean"], xerr=d["std"], height=0.62, color=[FAMILY[n][1] for n in order],
                    error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2.5})
            ax.set_yticks(y)
            ax.set_yticklabels(order)
            ax.set_title(f"{'Recall' if metric == 'recall' else 'NDCG'}@{k}")
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, d["mean"].max() * 1.18)
    family_legend(axes[1, 1])
    fig.suptitle("Model comparison on 5 held-out tracks per user  (bars: mean; whiskers: ±1 std across 5 folds of 1,000 users)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "model_comparison.png", dpi=170)
    plt.close(fig)


def rolling_ci(curves, col, window=500):
    """rolling mean per (agent, seed), then mean and 95% t-interval across seeds."""
    rows = []
    for agent, g in curves.groupby("agent"):
        per_seed = []
        for _, gs in g.groupby("seed"):
            s = gs.sort_values("episode")[col].rolling(window, min_periods=window).mean().to_numpy()
            per_seed.append(s)
        a = np.vstack(per_seed)
        mean = a.mean(axis=0)
        se = a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])
        half = stats.t.ppf(0.975, a.shape[0] - 1) * se
        rows.append(pd.DataFrame({"agent": agent, "episode": np.arange(a.shape[1]), "mean": mean, "ci95": half}))
    return pd.concat(rows)


def rl_learning_curves():
    curves = pd.read_csv(RESULTS / "rl_curves_grounded.csv")
    curves["precision_at_15"] = curves["hits"] / 15
    style_of = {"Random": (GREY, "-"), "Greedy": (INK, "--"), "UCB1": (BLUE, "-"), "LinUCB": (ORANGE, "-"),
                "QLearning": (AQUA, "-"), "SARSA": (YELLOW, "-")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    tables = []
    for ax, col, title in [(axes[0], "reward", "Mean episode reward (grounded)"),
                           (axes[1], "precision_at_15", "Precision@15 vs held-out tracks")]:
        ci = rolling_ci(curves, col)
        ci["metric"] = col
        tables.append(ci)
        for agent, (color, ls) in style_of.items():
            g = ci[ci.agent == agent].dropna()
            ax.plot(g["episode"], g["mean"], color=color, linestyle=ls, linewidth=1.8, label=agent)
            ax.fill_between(g["episode"], g["mean"] - g["ci95"], g["mean"] + g["ci95"], color=color, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("training episode (500-episode rolling mean)")
        if col == "reward":
            ax.set_ylim(top=4.3)
            ax.legend(fontsize=9, ncol=3, loc="upper right")
        else:
            ax.legend(fontsize=9, ncol=3, loc="center right", bbox_to_anchor=(1.0, 0.36))
    fig.suptitle("RL training curves, exploration ON, 3 seeds, shaded = 95% CI. Reward and hit rate rank the agents differently.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "rl_learning_curves.png", dpi=170)
    plt.close(fig)
    pd.concat(tables).to_csv(RESULTS / "phase7" / "rl_learning_curves_data.csv", index=False)


def coverage_vs_novelty(fc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    # Hybrid and four RL agents (and at K=10 nearly ItemCF) sit on top of one
    # another: they get ONE bracketed label; exact values are in
    # results/final_comparison.csv. Everything else is labelled in place.
    cluster = ["Hybrid", "Hybrid+RandomArm", "Hybrid+QLearning", "Hybrid+SARSA"]
    inplace = {"Hybrid+LinUCB": (7, -3), "Hybrid+UCB1": (7, -12), "ItemCF": (7, 5), "ALS": (7, 4), "Content": (7, 4),
               "Popularity": (7, 4), "Random": (-8, -14)}
    for ax, k in zip(axes, [10, 20]):
        d = fc[fc.K == k].pivot(index="config", columns="metric", values="mean")
        for name, (fam, color) in FAMILY.items():
            x, y = d.loc[name, "catalog_coverage"], d.loc[name, "novelty"]
            ax.scatter(x, y, s=64, color=color, marker=MARKER[fam], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            if name in inplace:
                ax.annotate(SHORT.get(name, name), (x, y), xytext=inplace[name], textcoords="offset points",
                            fontsize=8.5, color=INK2)
        cx, cy = d.loc[cluster, "catalog_coverage"].mean(), d.loc[cluster, "novelty"].mean()
        ax.annotate("Hybrid, +RandomArm,\n+QLearning, +SARSA\n(overlapping)", (cx, cy), xytext=(0.62, 0.30),
                    textcoords="axes fraction", fontsize=8.5, color=INK2, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 1.0, "shrinkB": 6})
        ax.set_title(f"K = {k}")
        ax.set_xlabel("catalogue coverage (fraction of 11,613 tracks shown to 1,000 users)")
    axes[0].set_ylabel("novelty (mean −log₂ train popularity; higher = rarer)")
    family_legend(axes[0], loc="upper left")
    fig.suptitle("Coverage vs novelty — Random and Popularity bracket the trade-off; accuracy lives in the middle",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "coverage_vs_novelty.png", dpi=170)
    plt.close(fig)


def reward_hacking():
    d = pd.read_csv(RESULTS / "phase7" / "ablation_reward.csv")
    d = d[d.trained_with == "grounded_simulated"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    colors = {"Random": GREY, "Greedy": INK, "UCB1": BLUE, "LinUCB": ORANGE, "QLearning": AQUA, "SARSA": YELLOW}
    off = {"Greedy": (-8, 8), "UCB1": (8, 4), "LinUCB": (-6, 10)}
    stacked = {"SARSA": (7.34, 0.1005), "Random": (7.34, 0.0955), "QLearning": (7.34, 0.0770)}  # crowded cluster near x=7.2
    for _, r in d.iterrows():
        ax.scatter(r["sim_reward_mean"], r["precision_at_15_mean"], s=80, color=colors[r["agent"]], edgecolor=SURFACE,
                   linewidth=1.3, zorder=3)
        xy = (r["sim_reward_mean"], r["precision_at_15_mean"])
        if r["agent"] in stacked:
            ax.annotate(r["agent"], xy, xytext=stacked[r["agent"]], textcoords="data", fontsize=9, color=INK2, va="center",
                        arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.9, "shrinkA": 1, "shrinkB": 5})
        else:
            ax.annotate(r["agent"], xy, xytext=off[r["agent"]], textcoords="offset points", fontsize=9, color=INK2,
                        ha="right" if off[r["agent"]][0] < 0 else "left")
    ax.set_xlabel("mean simulated (Tier 1 + 2) episode reward  →  what the agent optimised")
    ax.set_ylabel("Precision@15 on real held-out tracks  →  what we care about")
    ax.set_title("Agents trained on the simulated reward: higher reward, worse playlists")
    fig.text(0.01, 0.005, "1,000 held-out users, exploration off. Single seed (42): the secondary reward-hacking run, not a ranking of agents.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "reward_hacking.png", dpi=170)
    plt.close(fig)


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    fc = pd.read_csv(RESULTS / "final_comparison.csv")
    model_comparison(fc)
    rl_learning_curves()
    coverage_vs_novelty(fc)
    reward_hacking()
    print("Saved:", *sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
