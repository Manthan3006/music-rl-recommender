"""Phase 4 diagnostic — fusion-strategy comparison.

Investigates why Hybrid-RRF (unweighted, k=60) failed the Phase 4 acceptance
check (NDCG@20 below solo ItemCF — see results/phase4/phase4_metrics.csv) by
comparing five candidate-pool variants over the same 50/30/15/5
ItemCF/ALS/Content/Popularity pool composition (B.13):

  1. rrf_unweighted_k60  — current hybrid.py default, score = sum 1/(60+rank)
  2. rrf_unweighted_k10  — smaller RRF constant -> sharper rank decay
  3. rrf_unweighted_k20  — smaller RRF constant -> sharper rank decay
  4. rrf_weighted_k60    — score = sum w_m/(60+rank), w_m proportional to
                           each model's solo NDCG@20 (ItemCF 0.3345,
                           ALS 0.1740, Content 0.0454, Popularity 0.0331),
                           normalised to sum to 1
  5. itemcf_only_top100  — ItemCF's own top-100 list, no pooling at all;
                           baseline for how much the other three models
                           actually add to candidate-pool coverage

For each variant: Recall@20, NDCG@20, candidate recall@100, and the average
number of distinct items surviving dedup per user (out of the 100-item
budget). This is diagnostic only — it does NOT change hybrid.py's default
or overwrite data/processed/candidates.parquet.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import RESULTS
from src.recommenders import als_model, content, item_cf, popularity
from src.recommenders.common import load_holdout_by_user, load_train_csr, ndcg_at_k, recall_at_k

OUT_PATH = RESULTS / "phase4" / "fusion_comparison.csv"

POOL_SIZES = {"item_cf": 50, "als": 30, "content": 15, "popularity": 5}
SOLO_NDCG20 = {"item_cf": 0.3345, "als": 0.1740, "content": 0.0454, "popularity": 0.0331}
_total = sum(SOLO_NDCG20.values())
NDCG_WEIGHTS = {name: v / _total for name, v in SOLO_NDCG20.items()}

VARIANTS = [
    {"name": "rrf_unweighted_k60", "rrf_const": 60, "weighted": False},
    {"name": "rrf_unweighted_k10", "rrf_const": 10, "weighted": False},
    {"name": "rrf_unweighted_k20", "rrf_const": 20, "weighted": False},
    {"name": "rrf_weighted_k60", "rrf_const": 60, "weighted": True},
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fuse(component_ranks: dict, rrf_const: int, weighted: bool) -> dict:
    """component_ranks: {model_name: [(item_idx, rank), ...]} -> {item_idx: score}"""
    scores = {}
    for name, pairs in component_ranks.items():
        w = NDCG_WEIGHTS[name] if weighted else 1.0
        for item_idx, rank in pairs:
            scores[item_idx] = scores.get(item_idx, 0.0) + w / (rrf_const + rank)
    return scores


def main():
    train_csr = load_train_csr()
    n_users = train_csr.shape[0]
    holdout_by_user = load_holdout_by_user()

    log("Loading fitted models ...")
    models = {
        "item_cf": item_cf.load(),
        "als": als_model.load(),
        "content": content.load(),
        "popularity": popularity.load(),
    }

    # accumulators keyed by variant name (the 4 fused variants) + itemcf_only
    recalls20 = {v["name"]: [] for v in VARIANTS}
    ndcgs20 = {v["name"]: [] for v in VARIANTS}
    cand_recalls100 = {v["name"]: [] for v in VARIANTS}
    pool_sizes_after_dedup = {v["name"]: [] for v in VARIANTS}

    itemcf_recalls20, itemcf_ndcgs20, itemcf_recalls100, itemcf_pool_sizes = [], [], [], []

    t0 = time.time()
    for u in range(n_users):
        relevant = holdout_by_user.get(u)
        if not relevant:
            continue
        seen = train_csr[u].indices

        # ItemCF top-100 computed once; its top-50 prefix is reused for pooling.
        itemcf_top100 = models["item_cf"].recommend(u, 100, exclude_seen=True, seen=seen)
        als_top30 = models["als"].recommend(u, POOL_SIZES["als"], exclude_seen=True, seen=seen)
        content_top15 = models["content"].recommend(u, POOL_SIZES["content"], exclude_seen=True, seen=seen)
        pop_top5 = models["popularity"].recommend(u, POOL_SIZES["popularity"], exclude_seen=True, seen=seen)

        component_ranks = {
            "item_cf": [(i, r) for r, (i, _) in enumerate(itemcf_top100[:POOL_SIZES["item_cf"]], start=1)],
            "als": [(i, r) for r, (i, _) in enumerate(als_top30, start=1)],
            "content": [(i, r) for r, (i, _) in enumerate(content_top15, start=1)],
            "popularity": [(i, r) for r, (i, _) in enumerate(pop_top5, start=1)],
        }

        for variant in VARIANTS:
            name = variant["name"]
            scores = fuse(component_ranks, variant["rrf_const"], variant["weighted"])
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            rec_list = [(i, s) for i, s in ranked]
            recalls20[name].append(recall_at_k(rec_list, relevant, 20))
            ndcgs20[name].append(ndcg_at_k(rec_list, relevant, 20))
            cand_set = set(i for i, _ in ranked[:100])
            cand_recalls100[name].append(len(cand_set & relevant) / len(relevant))
            pool_sizes_after_dedup[name].append(len(cand_set))

        itemcf_rec_list = itemcf_top100
        itemcf_recalls20.append(recall_at_k(itemcf_rec_list, relevant, 20))
        itemcf_ndcgs20.append(ndcg_at_k(itemcf_rec_list, relevant, 20))
        itemcf_cand_set = set(i for i, _ in itemcf_rec_list[:100])
        itemcf_recalls100.append(len(itemcf_cand_set & relevant) / len(relevant))
        itemcf_pool_sizes.append(len(itemcf_cand_set))

        if (u + 1) % 10_000 == 0:
            log(f"  processed {u + 1:,}/{n_users:,} users ({time.time() - t0:.0f}s elapsed)")

    rows = []
    for variant in VARIANTS:
        name = variant["name"]
        rows.append({
            "variant": name,
            "recall@20": float(np.nanmean(recalls20[name])),
            "ndcg@20": float(np.nanmean(ndcgs20[name])),
            "candidate_recall@100": float(np.nanmean(cand_recalls100[name])),
            "avg_distinct_candidates": float(np.mean(pool_sizes_after_dedup[name])),
        })
    rows.append({
        "variant": "itemcf_only_top100",
        "recall@20": float(np.nanmean(itemcf_recalls20)),
        "ndcg@20": float(np.nanmean(itemcf_ndcgs20)),
        "candidate_recall@100": float(np.nanmean(itemcf_recalls100)),
        "avg_distinct_candidates": float(np.mean(itemcf_pool_sizes)),
    })

    results = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    log(f"Saved {OUT_PATH}")

    print("\n=== PHASE 4 FUSION STRATEGY COMPARISON ===")
    print(results.to_string(index=False, float_format="%.4f"))
    print(f"\n(Weighted variant weights: {NDCG_WEIGHTS})")


if __name__ == "__main__":
    main()
