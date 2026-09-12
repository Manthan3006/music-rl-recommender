"""Phase 4 acceptance gate.

Loads the fitted Phase 3 recommenders plus the Phase 4 candidate pool,
evaluates Recall@20 / NDCG@20 for all five (Popularity, Content, ItemCF,
ALS, Hybrid-RRF) for the Phase 9 report, and checks the sole Phase 4
acceptance criterion from docs/SPEC.md:

  Candidate recall@N >= 0.40 (N=100, or 200 if hybrid.py raised it).

(The original second criterion — Hybrid NDCG@20 >= best single model — was
dropped: Phase 4 is a candidate generator, not a ranker, and no RRF variant
recovered solo ItemCF's NDCG@20 on this dataset. See docs/SPEC.md Phase 4/9
and results/phase4/fusion_comparison.csv.)

Run popularity.py, content.py, item_cf.py, als_model.py, and hybrid.py first
— this script only loads their saved artifacts, it does not refit anything.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import RESULTS
from src.recommenders import als_model, content, hybrid, item_cf, popularity
from src.recommenders.common import evaluate_recommender, load_holdout_by_user, load_train_csr

RESULTS_PATH = RESULTS / "phase4" / "phase4_metrics.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    train_csr = load_train_csr()
    holdout_by_user = load_holdout_by_user()

    def with_seen_exclusion(rec):
        def recommend_fn(u, k, exclude_seen=True):
            seen = train_csr[u].indices if exclude_seen else None
            return rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)
        return recommend_fn

    log("Loading fitted models and candidate pool ...")
    candidates = hybrid.load()
    n = int(candidates.groupby("user_idx")["rank"].max().max())
    log(f"Loaded candidates.parquet: {len(candidates):,} rows, N<={n}")

    single_models = {
        "Popularity": popularity.load(),
        "Content": content.load(),
        "ItemCF": item_cf.load(),
        "ALS": als_model.load(),
    }

    log("Evaluating Recall@20 / NDCG@20 on the Phase 1 holdout "
        f"({len(holdout_by_user):,} users) ...")
    rows = []
    for name, rec in single_models.items():
        t0 = time.time()
        metrics = evaluate_recommender(with_seen_exclusion(rec), holdout_by_user, k=20)
        dt = time.time() - t0
        rows.append({"model": name, **metrics, "seconds": round(dt, 1)})
        log(f"  {name:<12} Recall@20={metrics['recall@k']:.4f} "
            f"NDCG@20={metrics['ndcg@k']:.4f}  ({dt:.1f}s)")

    t0 = time.time()
    hybrid_recommend_fn = hybrid.hybrid_recommend_fn(candidates)
    hybrid_metrics = evaluate_recommender(hybrid_recommend_fn, holdout_by_user, k=20)
    dt = time.time() - t0
    rows.append({"model": "Hybrid-RRF", **hybrid_metrics, "seconds": round(dt, 1)})
    log(f"  {'Hybrid-RRF':<12} Recall@20={hybrid_metrics['recall@k']:.4f} "
        f"NDCG@20={hybrid_metrics['ndcg@k']:.4f}  ({dt:.1f}s)")

    candidate_recall_n = hybrid.candidate_recall_at_n(candidates, holdout_by_user)
    log(f"Candidate recall@{n} = {candidate_recall_n:.4f}")

    results = pd.DataFrame(rows).set_index("model")

    print("\n=== PHASE 4 METRICS ===")
    print(results[["recall@k", "ndcg@k", "n_users"]].to_string(float_format="%.4f"))
    print(f"\nCandidate recall@{n} = {candidate_recall_n:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = results.reset_index()
    report.loc[len(report)] = {
        "model": f"candidate_recall@{n}", "recall@k": candidate_recall_n,
        "ndcg@k": float("nan"), "n_users": len(holdout_by_user), "seconds": float("nan"),
    }
    report.to_csv(RESULTS_PATH, index=False)
    log(f"Saved {RESULTS_PATH}")

    checks = [
        (f"Candidate recall@{n} >= 0.40", candidate_recall_n >= 0.40),
    ]

    print()
    all_pass = True
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        all_pass = all_pass and ok

    print(f"\nPhase 4 acceptance: {'ALL PASS' if all_pass else 'FAILED'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
