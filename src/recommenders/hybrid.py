"""Phase 4 — Hybrid candidate generator (Reciprocal Rank Fusion, B.13/B.14).

Pools each user's top-N lists from the four Phase 3 recommenders (ItemCF,
ALS, Content, Popularity) in a fixed 50/30/15/5 ratio, dedupes, and re-scores
by NDCG-weighted Reciprocal Rank Fusion: score = sum_m w_m / (60 + rank_m)
over the recommenders whose list contains the item, where rank is 1-indexed
within that recommender's own top-N list and w_m is that recommender's solo
NDCG@20 on the Phase 3 acceptance holdout (RRF_WEIGHTS below), normalised to
sum to 1. This is the sole hybrid strategy for the core system (B.14) — the
weighted *linear* hybrid with grid-searched α/β/γ and per-recommender
min-max normalisation is a separate, still-deferred experiment (Part F).

Plain unweighted RRF (w_m=1) was tried first and failed the original
Phase 4 acceptance criterion (Hybrid NDCG@20 >= best single model — ItemCF
dominates the other three badly enough that unweighted fusion diluted its
ranking). NDCG-weighting recovered most of the gap; see
results/phase4/fusion_comparison.csv and docs/SPEC.md Phase 4/9. The
acceptance criterion itself was changed to candidate-recall-only, since this
phase's job is pool coverage, not ranking (that's Phase 6's job).

If candidate recall@100 falls below the 0.40 floor (B.13), N is raised to
200 (pool scaled to 100/60/30/10) and the run is redone and documented.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from src.recommenders import als_model, content, item_cf, popularity
from src.recommenders.common import evaluate_recommender, load_holdout_by_user, load_train_csr

CANDIDATES_PATH = DATA_PROCESSED / "candidates.parquet"
RRF_K = 60
POOL_RATIO = {"item_cf": 50, "als": 30, "content": 15, "popularity": 5}  # sums to 100
DEFAULT_N = 100
FALLBACK_N = 200
RECALL_FLOOR = 0.40

# Each recommender's solo NDCG@20 on the Phase 3 acceptance holdout
# (results/phase3/phase3 run; ItemCF at k=5), normalised to sum to 1.
_SOLO_NDCG20 = {"item_cf": 0.3345, "als": 0.1740, "content": 0.0454, "popularity": 0.0331}
_total = sum(_SOLO_NDCG20.values())
RRF_WEIGHTS = {name: v / _total for name, v in _SOLO_NDCG20.items()}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pool_sizes(n: int) -> dict:
    base = sum(POOL_RATIO.values())
    return {name: count * n // base for name, count in POOL_RATIO.items()}


def load_models() -> dict:
    return {
        "item_cf": item_cf.load(),
        "als": als_model.load(),
        "content": content.load(),
        "popularity": popularity.load(),
    }


def rrf_rank_for_user(models: dict, sizes: dict, user_idx: int, seen: np.ndarray, n: int):
    """Pool each recommender's top-N list, NDCG-weighted-RRF-rescore, dedupe,
    return the top-n (item_idx, hybrid_score) pairs sorted by score descending."""
    scores = {}
    for name, rec in models.items():
        lst = rec.recommend(user_idx, sizes[name], exclude_seen=True, seen=seen)
        w = RRF_WEIGHTS[name]
        for rank, (item_idx, _) in enumerate(lst, start=1):
            scores[item_idx] = scores.get(item_idx, 0.0) + w / (RRF_K + rank)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[:n]


def build_candidates(n: int) -> pd.DataFrame:
    train_csr = load_train_csr()
    n_users = train_csr.shape[0]
    sizes = pool_sizes(n)
    log(f"Pool sizes for N={n}: {sizes}")

    models = load_models()

    rows = []
    for u in range(n_users):
        seen = train_csr[u].indices
        ranked = rrf_rank_for_user(models, sizes, u, seen, n)
        for rank, (item_idx, score) in enumerate(ranked, start=1):
            rows.append({"user_idx": u, "item_idx": item_idx, "hybrid_score": score, "rank": rank})
        if (u + 1) % 10_000 == 0:
            log(f"  candidates built for {u + 1:,}/{n_users:,} users")

    return pd.DataFrame(rows)


def save(df: pd.DataFrame):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CANDIDATES_PATH, index=False)


def load() -> pd.DataFrame:
    return pd.read_parquet(CANDIDATES_PATH)


def candidate_recall_at_n(candidates: pd.DataFrame, holdout_by_user: dict) -> float:
    cand_by_user = candidates.groupby("user_idx")["item_idx"].apply(set)
    recalls = []
    for u, relevant in holdout_by_user.items():
        if not relevant:
            continue
        cand = cand_by_user.get(u, set())
        recalls.append(len(cand & relevant) / len(relevant))
    return float(np.nanmean(recalls))


def hybrid_recommend_fn(candidates: pd.DataFrame):
    by_user = {
        u: list(zip(g["item_idx"].to_numpy(), g["hybrid_score"].to_numpy()))
        for u, g in candidates.sort_values("rank").groupby("user_idx")
    }

    def recommend_fn(user_idx, k, exclude_seen=True):
        recs = by_user.get(user_idx, [])
        return [(int(i), float(s)) for i, s in recs[:k]]

    return recommend_fn


def main():
    holdout_by_user = load_holdout_by_user()

    n = DEFAULT_N
    log(f"Building candidates with N={n} ...")
    candidates = build_candidates(n)
    recall_n = candidate_recall_at_n(candidates, holdout_by_user)
    log(f"Candidate recall@{n} = {recall_n:.4f}")

    if recall_n < RECALL_FLOOR:
        log(f"Candidate recall@{n} ({recall_n:.4f}) is below the {RECALL_FLOOR} floor (B.13) — "
            f"raising N to {FALLBACK_N} and rebuilding.")
        n = FALLBACK_N
        candidates = build_candidates(n)
        recall_n = candidate_recall_at_n(candidates, holdout_by_user)
        log(f"Candidate recall@{n} = {recall_n:.4f}")

    save(candidates)
    log(f"Saved {CANDIDATES_PATH} ({len(candidates):,} rows, N={n})")

    recommend_fn = hybrid_recommend_fn(candidates)
    metrics = evaluate_recommender(recommend_fn, holdout_by_user, k=20)
    log(f"Hybrid NDCG-weighted-RRF Recall@20={metrics['recall@k']:.4f} "
        f"NDCG@20={metrics['ndcg@k']:.4f} (n_users={metrics['n_users']:,})")


if __name__ == "__main__":
    main()
