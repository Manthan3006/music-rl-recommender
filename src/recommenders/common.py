"""Shared loaders and evaluation helpers for the Phase 3 recommenders.

All four recommenders (popularity, content, item_cf, als_model) operate over
the same item_idx / user_idx space defined by data/processed/idx_maps.parquet
(the interaction-filtered core: 39,825 users x 11,613 tracks), so they share
one interface — recommend(user_idx, n, exclude_seen=True) ->
List[(item_idx, score)] — and are directly comparable and poolable in Phase 4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import DATA_PROCESSED

CORE_CONTENT_MATRIX_PATH = DATA_PROCESSED / "core_content_matrix.npz"


def load_idx_maps():
    idx_maps = pd.read_parquet(DATA_PROCESSED / "idx_maps.parquet")
    users = idx_maps[idx_maps.entity_type == "user"].sort_values("idx")["raw_id"].to_numpy()
    items = idx_maps[idx_maps.entity_type == "item"].sort_values("idx")["raw_id"].to_numpy()
    return users, items


def load_train_csr() -> sp.csr_matrix:
    """train_csr.npz from Phase 1: values are c_ui (confidence), already
    alpha-scaled per B.11. Its sparsity pattern also defines each user's
    train ('seen') items."""
    return sp.load_npz(DATA_PROCESSED / "train_csr.npz").tocsr()


def load_train_df() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "train.parquet")


def load_train_aff_csr() -> sp.csr_matrix:
    """Same shape/sparsity pattern as train_csr but with aff_ui values —
    needed for item-CF scoring (B.7/Phase 3 task 3: score = sum sim*aff)."""
    train = load_train_df()
    users, items = load_idx_maps()
    n_users, n_items = len(users), len(items)
    return sp.csr_matrix(
        (train["aff_ui"].to_numpy(), (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    )


def load_holdout_by_user() -> dict:
    holdout = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")
    return holdout.groupby("user_idx")["item_idx"].apply(set).to_dict()


def load_core_content_matrix() -> sp.csr_matrix:
    """content_matrix.npz (Phase 2, full 49,703-track catalogue, row i ==
    items.parquet row i) restricted + reindexed to the core item_idx space
    (0..n_items-1). Cached to disk on first build so Phase 5's RL clustering
    can reuse it directly."""
    if CORE_CONTENT_MATRIX_PATH.exists():
        return sp.load_npz(CORE_CONTENT_MATRIX_PATH).tocsr()

    items_df = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    content = sp.load_npz(DATA_PROCESSED / "content_matrix.npz").tocsr()
    _, core_track_ids = load_idx_maps()

    row_of_track = pd.Series(np.arange(len(items_df)), index=items_df["track_id"])
    rows = row_of_track.loc[core_track_ids].to_numpy()
    core_content = content[rows]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    sp.save_npz(CORE_CONTENT_MATRIX_PATH, core_content)
    return core_content


def top_n_from_scores(scores: np.ndarray, n: int, exclude: np.ndarray = None):
    """argpartition-based top-N, excluding `exclude` indices. Returns a
    score-descending list of (item_idx, score) tuples."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    if exclude is not None and len(exclude):
        scores[exclude] = -np.inf
    n_valid = int(np.isfinite(scores).sum())
    n = min(n, n_valid)
    if n <= 0:
        return []
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top]


# --- minimal ranking metrics, only what Phase 3's acceptance criteria need.
# Phase 7 builds the full P/R/F1/NDCG/MAP/HR/beyond-accuracy evaluation suite.

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    topk = {i for i, _ in recommended[:k]}
    return len(topk & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    dcg = sum(1.0 / np.log2(rank + 1) for rank, (i, _) in enumerate(recommended[:k], start=1)
              if i in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else np.nan


def evaluate_recommender(recommend_fn, holdout_by_user: dict, k: int = 20,
                          users: np.ndarray = None) -> dict:
    """recommend_fn(user_idx, k, exclude_seen=True) -> List[(item_idx, score)]"""
    if users is None:
        users = np.array(sorted(holdout_by_user.keys()))
    recalls, ndcgs = [], []
    for u in users:
        relevant = holdout_by_user.get(int(u))
        if not relevant:
            continue
        rec = recommend_fn(int(u), k, exclude_seen=True)
        recalls.append(recall_at_k(rec, relevant, k))
        ndcgs.append(ndcg_at_k(rec, relevant, k))
    return {
        "recall@k": float(np.nanmean(recalls)),
        "ndcg@k": float(np.nanmean(ndcgs)),
        "n_users": len(recalls),
    }
