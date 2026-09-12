"""Phase 3 acceptance gate.

Loads the four fitted recommenders (Popularity, Content, ItemCF, ALS) plus a
Random baseline, evaluates Recall@20 / NDCG@20 on the real Phase 1 holdout,
and checks the three acceptance criteria from docs/SPEC.md Phase 3:

  1. Each model beats Random on Recall@20.
  2. ALS beats Popularity on NDCG@20.
  3. Item-CF beats Content on Recall@20.

Run popularity.py, content.py, item_cf.py and als_model.py first — this
script only loads their saved artifacts, it does not refit anything.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import SEED
from src.recommenders import als_model, content, item_cf, popularity
from src.recommenders.common import (
    evaluate_recommender,
    load_holdout_by_user,
    load_train_csr,
    top_n_from_scores,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RandomRecommender:
    def __init__(self, n_items: int, seed: int = SEED):
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def recommend(self, user_idx, n, exclude_seen=True, seen=None):
        scores = self.rng.random(self.n_items)
        exclude = seen if exclude_seen else None
        return top_n_from_scores(scores, n, exclude=exclude)


def main():
    train_csr = load_train_csr()
    n_users, n_items = train_csr.shape
    holdout_by_user = load_holdout_by_user()

    def with_seen_exclusion(rec):
        def recommend_fn(u, k, exclude_seen=True):
            seen = train_csr[u].indices if exclude_seen else None
            return rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)
        return recommend_fn

    log("Loading fitted models ...")
    models = {
        "Random": RandomRecommender(n_items),
        "Popularity": popularity.load(),
        "Content": content.load(),
        "ItemCF": item_cf.load(),
        "ALS": als_model.load(),
    }

    log("Evaluating Recall@20 / NDCG@20 on the Phase 1 holdout "
        f"({len(holdout_by_user):,} users) ...")
    rows = []
    for name, rec in models.items():
        t0 = time.time()
        metrics = evaluate_recommender(with_seen_exclusion(rec), holdout_by_user, k=20)
        dt = time.time() - t0
        rows.append({"model": name, **metrics, "seconds": round(dt, 1)})
        log(f"  {name:<12} Recall@20={metrics['recall@k']:.4f} "
            f"NDCG@20={metrics['ndcg@k']:.4f}  ({dt:.1f}s)")

    results = pd.DataFrame(rows).set_index("model")

    print("\n=== PHASE 3 ACCEPTANCE ===")
    print(results[["recall@k", "ndcg@k", "n_users"]].to_string(float_format="%.4f"))

    checks = []
    for name in ["Popularity", "Content", "ItemCF", "ALS"]:
        ok = results.loc[name, "recall@k"] > results.loc["Random", "recall@k"]
        checks.append((f"{name} beats Random on Recall@20", ok))

    ok = results.loc["ALS", "ndcg@k"] > results.loc["Popularity", "ndcg@k"]
    checks.append(("ALS beats Popularity on NDCG@20", ok))

    ok = results.loc["ItemCF", "recall@k"] > results.loc["Content", "recall@k"]
    checks.append(("ItemCF beats Content on Recall@20", ok))

    print()
    all_pass = True
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        all_pass = all_pass and ok

    print(f"\nPhase 3 acceptance: {'ALL PASS' if all_pass else 'FAILED'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
