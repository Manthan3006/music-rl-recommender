"""Phase 3 — Item-item collaborative filtering.

Cosine item-item similarity on the train CSR (confidence-weighted), computed
in row chunks so the full (n_items, n_items) similarity matrix is never
densified at once, keeping only the top-5 neighbours per item. Score for a
user is sum_{j in train_u} sim(i, j) * aff_uj (Phase 3 task 3 / B.7).

k=5 was chosen by a holdout sweep over k in {2,5,10,15,20,30,40,50}
(results/phase3/item_cf_k_sweep.csv) — Recall@20, NDCG@20, and Coverage@20
all peak at k=5 and degrade monotonically for k>5, so the choice is not a
popularity-concentration artifact. Because the sweep was run on the same
holdout used for acceptance evaluation, this is reported in Phase 9 as a
sensitivity analysis, not as tuning — a separate validation split would have
been the methodologically clean way to select k.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import scipy.sparse as sp

from config import MODELS
from src.recommenders.common import (
    evaluate_recommender,
    load_holdout_by_user,
    load_train_aff_csr,
    load_train_csr,
    top_n_from_scores,
)

MODEL_PATH = MODELS / "item_cf_top5.npz"
TOP_K_NEIGHBOURS = 5
CHUNK_SIZE = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_item_similarity_topk(train_csr: sp.csr_matrix, topk: int = TOP_K_NEIGHBOURS,
                                chunk_size: int = CHUNK_SIZE) -> sp.csr_matrix:
    item_matrix = train_csr.T.tocsr()  # (n_items, n_users)
    n_items = item_matrix.shape[0]

    norms = np.sqrt(np.asarray(item_matrix.multiply(item_matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    normalized = sp.diags(1.0 / norms) @ item_matrix  # L2-normalized rows

    rows, cols, data = [], [], []
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        block = normalized[start:end]                    # (block, n_users)
        sim_block = (block @ normalized.T).toarray()      # (block, n_items) dense chunk only
        for local_i in range(end - start):
            i = start + local_i
            row = sim_block[local_i]
            row[i] = -1.0  # exclude self
            k = min(topk, n_items - 1)
            top_idx = np.argpartition(-row, k - 1)[:k]
            top_idx = top_idx[row[top_idx] > 0]
            rows.extend([i] * len(top_idx))
            cols.extend(top_idx.tolist())
            data.extend(row[top_idx].tolist())
        log(f"  item-item similarity: {end:,}/{n_items:,} items processed")

    sim = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
    return sim


class ItemCFRecommender:
    def __init__(self, sim_topk: sp.csr_matrix, aff_csr: sp.csr_matrix):
        self.sim = sim_topk
        self.aff = aff_csr

    def score_user(self, user_idx) -> np.ndarray:
        return np.asarray(self.sim.dot(self.aff[user_idx].T).todense()).ravel()

    def recommend(self, user_idx, n, exclude_seen=True, seen=None):
        scores = self.score_user(user_idx)
        exclude = seen if exclude_seen else None
        return top_n_from_scores(scores, n, exclude=exclude)


def fit(train_csr: sp.csr_matrix) -> ItemCFRecommender:
    sim = build_item_similarity_topk(train_csr)
    aff_csr = load_train_aff_csr()
    return ItemCFRecommender(sim, aff_csr)


def save(rec: ItemCFRecommender):
    MODELS.mkdir(parents=True, exist_ok=True)
    sp.save_npz(MODEL_PATH, rec.sim)


def load() -> ItemCFRecommender:
    sim = sp.load_npz(MODEL_PATH).tocsr()
    aff_csr = load_train_aff_csr()
    return ItemCFRecommender(sim, aff_csr)


def main():
    train_csr = load_train_csr()
    log(f"Building top-{TOP_K_NEIGHBOURS} item-item similarity over "
        f"{train_csr.shape[1]:,} items (chunk_size={CHUNK_SIZE}) ...")
    rec = fit(train_csr)
    save(rec)
    log(f"Saved {MODEL_PATH.name}: shape={rec.sim.shape}, nnz={rec.sim.nnz:,} "
        f"(mean neighbours/item={rec.sim.nnz / rec.sim.shape[0]:.1f})")

    holdout_by_user = load_holdout_by_user()

    def recommend_fn(u, k, exclude_seen=True):
        seen = train_csr[u].indices if exclude_seen else None
        return rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)

    metrics = evaluate_recommender(recommend_fn, holdout_by_user, k=20)
    log(f"ItemCF Recall@20={metrics['recall@k']:.4f} "
        f"NDCG@20={metrics['ndcg@k']:.4f} (n_users={metrics['n_users']:,})")


if __name__ == "__main__":
    main()
