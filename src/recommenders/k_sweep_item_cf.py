"""Phase 3 follow-up — ItemCF top-k neighbour sweep.

Truncates the existing top-50 item-item similarity matrix (models/item_cf_top50.npz)
to smaller neighbourhood sizes and re-evaluates Recall@20 / NDCG@20 / catalog
coverage@20 on the real Phase 1 holdout for each k. Truncating top-50 to top-k
(k<=50) is exact — the top-k neighbours by similarity are a subset of the
top-50 neighbours already stored, so this reproduces what a from-scratch build
at that k would find, without re-running the O(n_items^2) similarity
computation per k.

Catalog coverage@20 = number of distinct items appearing in any evaluated
user's top-20 recommendation list, out of n_items in the core catalogue.

Feeds the k-selection evidence for Phase 9 (docs/SPEC.md Phase 3 default). No
change to the ItemCF default is made here — this script only measures.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import MODELS, RESULTS
from src.recommenders.common import (
    load_holdout_by_user,
    load_train_aff_csr,
    load_train_csr,
    ndcg_at_k,
    recall_at_k,
)
from src.recommenders.item_cf import ItemCFRecommender

SIM_PATH = MODELS / "item_cf_top50.npz"
K_GRID = [2, 5, 10, 15, 20, 30, 40, 50]
OUT_PATH = RESULTS / "phase3" / "item_cf_k_sweep.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def truncate_topk(sim: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k largest-similarity neighbours per row of an already
    row-topk sim matrix. Exact for k <= the matrix's existing per-row k."""
    sim = sim.tocsr()
    rows, cols, data = [], [], []
    for i in range(sim.shape[0]):
        start, end = sim.indptr[i], sim.indptr[i + 1]
        row_idx = sim.indices[start:end]
        row_val = sim.data[start:end]
        if len(row_val) > k:
            top = np.argpartition(-row_val, k - 1)[:k]
        else:
            top = np.arange(len(row_val))
        rows.extend([i] * len(top))
        cols.extend(row_idx[top].tolist())
        data.extend(row_val[top].tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=sim.shape)


def main():
    if not SIM_PATH.exists():
        raise FileNotFoundError(f"{SIM_PATH} not found — run item_cf.py first.")

    sim_top50 = sp.load_npz(SIM_PATH).tocsr()
    aff_csr = load_train_aff_csr()
    train_csr = load_train_csr()
    holdout_by_user = load_holdout_by_user()
    n_items = sim_top50.shape[0]
    users = np.array(sorted(holdout_by_user.keys()))

    log(f"Sweeping k over {K_GRID} using top-50 sim matrix (shape={sim_top50.shape}) ...")
    rows = []
    for k in K_GRID:
        t0 = time.time()
        sim_k = truncate_topk(sim_top50, k)
        rec = ItemCFRecommender(sim_k, aff_csr)

        recalls, ndcgs = [], []
        recommended_items = set()
        for u in users:
            relevant = holdout_by_user.get(int(u))
            if not relevant:
                continue
            seen = train_csr[u].indices
            rec_list = rec.recommend(int(u), 20, exclude_seen=True, seen=seen)
            recalls.append(recall_at_k(rec_list, relevant, 20))
            ndcgs.append(ndcg_at_k(rec_list, relevant, 20))
            recommended_items.update(i for i, _ in rec_list)

        dt = time.time() - t0
        rows.append({
            "k_neighbours": k,
            "mean_neighbours_per_item": round(sim_k.nnz / sim_k.shape[0], 2),
            "recall@20": float(np.nanmean(recalls)),
            "ndcg@20": float(np.nanmean(ndcgs)),
            "coverage@20": len(recommended_items) / n_items,
            "n_users": len(recalls),
            "seconds": round(dt, 1),
        })
        log(f"  k={k:<3} Recall@20={rows[-1]['recall@20']:.4f} "
            f"NDCG@20={rows[-1]['ndcg@20']:.4f} "
            f"Coverage@20={rows[-1]['coverage@20']:.4f}  ({dt:.1f}s)")

    results = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    log(f"Saved k-sweep table to {OUT_PATH}")

    print("\n=== ITEMCF TOP-K NEIGHBOUR SWEEP ===")
    print(results.to_string(index=False, float_format="%.4f"))

    best = results.loc[results["recall@20"].idxmax()]
    log(f"Best by Recall@20: k={int(best['k_neighbours'])}")


if __name__ == "__main__":
    main()
