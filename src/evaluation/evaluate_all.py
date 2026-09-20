"""Phase 7 — the full comparison table (docs/SPEC.md Phase 7, B.15).

Configurations (K = 10 and 20, 5 evaluation folds, mean +/- std across folds):

  Random       uniform draw from the catalogue minus the user's train items
  Popularity   global top-K by distinct train listeners
  Content      cosine(user profile, catalogue)
  ItemCF       item-item CF, k=5 neighbours
  ALS          implicit ALS with confidence weighting
  Hybrid       NDCG-weighted RRF over the four above = top-K of the Phase 4
               pool. This IS the Greedy agent's playlist (asserted below).
  Hybrid+RandomArm / +UCB1 / +LinUCB / +QLearning / +SARSA
               the Phase 6 agents, exploration disabled, playlist_length = K,
               built by src/evaluation/rl_playlists.py

All configurations are scored on the same 5 disjoint folds x 1,000 users, none
of whom is in the RL training curriculum (eval_users.py). For RL rows each fold
value is the mean over the 3 trained checkpoints, so the reported std is the
across-fold (user-sampling) std; the across-checkpoint spread is written to
results/phase7/rl_by_train_seed.csv.

Outputs: results/final_comparison.csv (long: config, K, metric, mean, std,
n_folds), results/phase7/{final_comparison_wide,per_fold_metrics,
rl_by_train_seed,leakage_checks,greedy_equals_hybrid}.csv/.json and a row in
results/metrics.csv. Rating-prediction error metrics are deliberately absent
(no ratings exist, B.2).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, RESULTS, SEED
from src.evaluation import beyond_accuracy as ba
from src.evaluation import ranking_metrics as rm
from src.evaluation.eval_users import FOLD_SEEDS, assert_no_leakage, get_eval_folds
from src.evaluation.rl_playlists import PLAYLISTS_PATH, REWARDS_PATH
from src.recommenders import als_model, content, item_cf, popularity
from src.recommenders.common import (
    load_core_content_matrix,
    load_holdout_by_user,
    load_train_csr,
)
from src.rl.environment import _load_item_metadata

KS = [10, 20]
OUT = RESULTS / "phase7"
BASELINES = ["Random", "Popularity", "Content", "ItemCF", "ALS", "Hybrid"]
RL_CONFIGS = {"RandomArm": "Random", "UCB1": "UCB1", "LinUCB": "LinUCB", "QLearning": "QLearning", "SARSA": "SARSA"}
CONFIG_ORDER = BASELINES + [f"Hybrid+{c}" for c in RL_CONFIGS]
LIST_METRICS = ["precision", "recall", "f1", "ndcg", "map", "hit_rate",
                "intra_list_diversity", "novelty", "artist_entropy"]
POP_METRICS = ["catalog_coverage", "gini"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Context:
    def __init__(self):
        self.train_csr = load_train_csr()
        self.n_users, self.n_items = self.train_csr.shape
        self.content = load_core_content_matrix()
        self.artist_of = _load_item_metadata()["artist"].to_numpy()
        pop = popularity.load()
        self.item_pop = np.maximum(pop.listeners, 1.0) / self.n_users  # train listener fraction
        self.holdout = load_holdout_by_user()
        cand = pd.read_parquet(DATA_PROCESSED / "candidates.parquet").sort_values(["user_idx", "rank"])
        self.cand = {u: (g["item_idx"].to_numpy(), g["hybrid_score"].to_numpy()) for u, g in cand.groupby("user_idx")}
        self.recs = {"Popularity": pop, "Content": content.load(), "ItemCF": item_cf.load(), "ALS": als_model.load()}

    def seen(self, u):
        return self.train_csr[int(u)].indices


def baseline_lists(ctx, folds):
    """{config: {fold: {user: [top-20 item_idx]}}}"""
    out = {c: {} for c in BASELINES}
    for f, users in enumerate(folds):
        rng = np.random.default_rng(FOLD_SEEDS[f])
        for c in BASELINES:
            out[c][f] = {}
        for u in users:
            u = int(u)
            seen = ctx.seen(u)
            pool = np.setdiff1d(np.arange(ctx.n_items), seen)
            out["Random"][f][u] = [int(i) for i in rng.choice(pool, size=max(KS), replace=False)]
            for c, rec in ctx.recs.items():
                out[c][f][u] = [i for i, _ in rec.recommend(u, max(KS), exclude_seen=True, seen=seen)]
            out["Hybrid"][f][u] = [int(i) for i in ctx.cand[u][0][:max(KS)]]
    return out


def score_fold(lists_by_user, k, ctx):
    rows = {m: [] for m in LIST_METRICS}
    all_items = []
    for u, lst in lists_by_user.items():
        lst = lst[:k]
        assert len(lst) == k and len(set(lst)) == k, f"user {u}: list is not {k} distinct items"
        assert not (set(lst) & set(ctx.seen(u).tolist())), f"LEAKAGE: user {u} was recommended a train item"
        rel = ctx.holdout[u]
        for m, v in rm.all_ranking_metrics(lst, rel, k).items():
            rows[m].append(v)
        rows["intra_list_diversity"].append(ba.intra_list_diversity(lst, ctx.content))
        rows["novelty"].append(ba.novelty(lst, ctx.item_pop))
        rows["artist_entropy"].append(ba.artist_entropy(lst, ctx.artist_of))
        all_items += lst
    out = {m: float(np.mean(v)) for m, v in rows.items()}
    out["catalog_coverage"] = ba.catalog_coverage(all_items, ctx.n_items)
    out["gini"] = ba.gini_index(all_items, ctx.n_items)
    return out


def rl_lists(playlists, folds):
    """{(agent, train_seed, K): {fold: {user: [items in play order]}}}"""
    out = {}
    fold_of = {int(u): f for f, users in enumerate(folds) for u in users}
    playlists = playlists.sort_values(["agent", "train_seed", "K", "user_idx", "position"])
    for (agent, seed, k, u), g in playlists.groupby(["agent", "train_seed", "K", "user_idx"], sort=False):
        out.setdefault((agent, int(seed), int(k)), {}).setdefault(fold_of[int(u)], {})[int(u)] = g["item_idx"].tolist()
    return out


def check_greedy_equals_hybrid(rl, ctx, folds):
    """Greedy always takes the highest-hybrid-score unused candidate, so its
    playlist is the pool's top-K. Compared on hybrid SCORES so that exact
    ties in the RRF score (which the environment and the rank column may order
    differently) don't trigger a false alarm."""
    rows = []
    for k in KS:
        n_same_items = n_same_scores = n = 0
        for f, users in enumerate(folds):
            for u in users:
                u = int(u)
                items, scores = ctx.cand[u]
                g = rl[("Greedy", 42, k)][f][u]
                score_of = dict(zip(items.tolist(), scores.tolist()))
                n += 1
                n_same_items += int(g == items[:k].tolist())
                n_same_scores += int(np.allclose([score_of[i] for i in g], scores[:k]))
        rows.append({"K": k, "users": n, "identical_item_lists": n_same_items, "identical_hybrid_score_sequences": n_same_scores})
        assert n_same_scores == n, "Greedy playlist is not the Hybrid pool's top-K"
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds, curriculum = get_eval_folds()
    ctx = Context()
    leak = assert_no_leakage(folds, curriculum, ctx.holdout)
    json.dump(leak, open(OUT / "leakage_checks.json", "w"), indent=2)
    log(f"Leakage assertions passed: {leak}")

    log("Building baseline / hybrid lists ...")
    base = baseline_lists(ctx, folds)
    playlists = pd.read_parquet(PLAYLISTS_PATH)
    rewards = pd.read_parquet(REWARDS_PATH)
    rl = rl_lists(playlists, folds)
    greedy_check = check_greedy_equals_hybrid(rl, ctx, folds)
    greedy_check.to_csv(OUT / "greedy_equals_hybrid.csv", index=False)
    log(f"Greedy == Hybrid top-K check:\n{greedy_check.to_string(index=False)}")

    fold_of = {int(u): f for f, users in enumerate(folds) for u in users}
    rewards = rewards.assign(fold=rewards["user_idx"].map(fold_of))

    rows = []  # config, K, fold, train_seed, metric, value
    for c in BASELINES:
        for k in KS:
            for f in range(len(folds)):
                for m, v in score_fold(base[c][f], k, ctx).items():
                    rows.append((f"{c}", k, f, -1, m, v))
        log(f"  scored {c}")
    for cfg, agent in RL_CONFIGS.items():
        for k in KS:
            for seed in sorted({s for (a, s, kk) in rl if a == agent and kk == k}):
                for f in range(len(folds)):
                    for m, v in score_fold(rl[(agent, seed, k)][f], k, ctx).items():
                        rows.append((f"Hybrid+{cfg}", k, f, seed, m, v))
                    r = rewards[(rewards.agent == agent) & (rewards.train_seed == seed) & (rewards.K == k) & (rewards.fold == f)]
                    rows.append((f"Hybrid+{cfg}", k, f, seed, "grounded_reward", float(r["reward"].mean())))
        log(f"  scored Hybrid+{cfg}")
    # Greedy's grounded reward is the Hybrid row's reward (same playlist)
    for k in KS:
        for f in range(len(folds)):
            r = rewards[(rewards.agent == "Greedy") & (rewards.K == k) & (rewards.fold == f)]
            rows.append(("Hybrid", k, f, -1, "grounded_reward", float(r["reward"].mean())))

    df = pd.DataFrame(rows, columns=["config", "K", "fold", "train_seed", "metric", "value"])
    df["fold_seed"] = df["fold"].map(lambda f: FOLD_SEEDS[f])
    df.to_csv(OUT / "per_fold_metrics.csv", index=False)

    # RL: mean over trained checkpoints within each fold, then mean/std across folds
    per_fold = df.groupby(["config", "K", "fold", "metric"], as_index=False)["value"].mean()
    agg = per_fold.groupby(["config", "K", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"count": "n_folds"})
    agg["config"] = pd.Categorical(agg["config"], CONFIG_ORDER, ordered=True)
    agg = agg.sort_values(["K", "metric", "config"]).reset_index(drop=True)
    assert agg["std"].notna().all(), "every reported number must carry a std"
    agg.to_csv(RESULTS / "final_comparison.csv", index=False)

    wide = agg.assign(cell=lambda d: d["mean"].map("{:.4f}".format) + " ± " + d["std"].map("{:.4f}".format))
    wide = wide.pivot_table(index=["K", "config"], columns="metric", values="cell", aggfunc="first", observed=True)
    wide.to_csv(OUT / "final_comparison_wide.csv")

    by_seed = df[df.train_seed >= 0].groupby(["config", "K", "train_seed", "metric"])["value"].agg(["mean", "std"]).reset_index()
    by_seed.to_csv(OUT / "rl_by_train_seed.csv", index=False)

    # rating-prediction error metrics must not appear anywhere in the outputs
    assert not agg["metric"].str.contains("rmse|mae$", case=False, regex=True).any()

    cfg_blob = {"folds": FOLD_SEEDS, "Ks": KS, "n_eval_users": len(np.concatenate(folds)), "seed": SEED}
    cfg_hash = hashlib.sha1(json.dumps(cfg_blob, sort_keys=True).encode()).hexdigest()[:10]
    headline = agg[(agg.K == 20) & (agg.metric == "ndcg")].set_index("config")["mean"].round(4).to_dict()
    row = pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": "phase7_final_comparison",
                         "config_hash": cfg_hash, "summary": json.dumps({"ndcg@20": headline})}])
    metrics_path = RESULTS / "metrics.csv"
    row.to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)

    for k in KS:
        sub = wide.loc[k][["precision", "recall", "ndcg", "map", "hit_rate"]]
        print(f"\n=== K={k} (mean ± std across 5 folds) ===")
        print(sub.to_string())
    log("Saved results/final_comparison.csv and results/phase7/*")


if __name__ == "__main__":
    main()
