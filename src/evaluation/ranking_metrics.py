"""Phase 7 — accuracy metrics (B.15): Precision/Recall/F1/NDCG/MAP/HitRate @K.

Every function takes one user's ranked list `recommended` (item_idx, best
first) and the set of that user's held-out `relevant` items, and looks only at
the first K entries. There are no ratings in this dataset (implicit feedback,
B.2), so nothing here predicts a rating — these are all top-K ranking metrics.
"""

import numpy as np


def _hits(recommended, relevant, k):
    return [1 if i in relevant else 0 for i in list(recommended)[:k]]


def precision_at_k(recommended, relevant, k) -> float:
    return sum(_hits(recommended, relevant, k)) / k


def recall_at_k(recommended, relevant, k) -> float:
    return sum(_hits(recommended, relevant, k)) / len(relevant)


def f1_at_k(recommended, relevant, k) -> float:
    p, r = precision_at_k(recommended, relevant, k), recall_at_k(recommended, relevant, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def hit_rate_at_k(recommended, relevant, k) -> float:
    return 1.0 if sum(_hits(recommended, relevant, k)) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k) -> float:
    hits = _hits(recommended, relevant, k)
    dcg = sum(h / np.log2(rank + 1) for rank, h in enumerate(hits, start=1))
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def average_precision_at_k(recommended, relevant, k) -> float:
    """AP@K normalised by min(|relevant|, K), the standard cut-off form."""
    hits = _hits(recommended, relevant, k)
    running, total = 0, 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            running += 1
            total += running / rank
    return total / min(len(relevant), k)


METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "f1": f1_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit_rate": hit_rate_at_k,
}


def all_ranking_metrics(recommended, relevant, k) -> dict:
    return {name: fn(recommended, relevant, k) for name, fn in METRICS.items()}
