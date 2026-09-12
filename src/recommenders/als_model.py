"""Phase 3 — ALS with confidence weighting (B.2, implicit library).

train_csr.npz already IS the confidence matrix C_ui = 1 + ALS_ALPHA *
log1p(playcount / eps) (Phase 1 / B.11) — alpha is already baked in. implicit
also multiplies its input by its own `alpha` constructor arg before fitting
(implicit.cpu.als: `if self.alpha != 1.0: Cui = self.alpha * Cui`), so we fit
with alpha=1.0 here to avoid double-scaling an already-alpha-weighted matrix.

Grid-searches factors in {32, 64, 128} x regularization in {0.01, 0.05, 0.1}
on an internal validation slice carved out of train (2 held-out items/user,
disjoint from the real Phase 1 holdout used for final acceptance), then
refits the best config on the full train set.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from config import MODELS, SEED
from src.recommenders.common import (
    evaluate_recommender,
    load_holdout_by_user,
    load_train_csr,
    load_train_df,
    top_n_from_scores,
)

MODEL_PATH = MODELS / "als_factors.npz"
FACTORS_GRID = [32, 64, 128]
REG_GRID = [0.01, 0.05, 0.1]
ALS_ITERATIONS = 20
VAL_ITEMS_PER_USER = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ALSRecommender:
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray):
        self.user_factors = user_factors
        self.item_factors = item_factors

    def score_user(self, user_idx) -> np.ndarray:
        return self.item_factors @ self.user_factors[user_idx]

    def recommend(self, user_idx, n, exclude_seen=True, seen=None):
        scores = self.score_user(user_idx)
        exclude = seen if exclude_seen else None
        return top_n_from_scores(scores, n, exclude=exclude)


def fit_als(train_csr: sp.csr_matrix, factors: int, regularization: float,
            iterations: int = ALS_ITERATIONS, show_progress: bool = False) -> ALSRecommender:
    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, alpha=1.0,
        iterations=iterations, random_state=SEED,
    )
    model.fit(train_csr, show_progress=show_progress)
    return ALSRecommender(np.asarray(model.user_factors), np.asarray(model.item_factors))


def make_validation_split(train: pd.DataFrame, n_users: int, n_items: int, seed=SEED):
    """Carve VAL_ITEMS_PER_USER items/user out of train for grid-search model
    selection, kept separate from the real Phase 1 holdout used for the
    Phase 3 acceptance gate."""
    val = train.groupby("user_idx", group_keys=False).sample(
        n=VAL_ITEMS_PER_USER, random_state=seed
    )
    train_for_grid = train.drop(val.index)

    train_for_grid_csr = sp.csr_matrix(
        (train_for_grid["c_ui"].to_numpy(),
         (train_for_grid["user_idx"].to_numpy(), train_for_grid["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    )
    val_by_user = val.groupby("user_idx")["item_idx"].apply(set).to_dict()
    return train_for_grid_csr, val_by_user


def grid_search(train_for_grid_csr, val_by_user, val_users_sample):
    results = []
    for factors in FACTORS_GRID:
        for reg in REG_GRID:
            t0 = time.time()
            rec = fit_als(train_for_grid_csr, factors, reg)
            seen_lookup = train_for_grid_csr

            def recommend_fn(u, k, exclude_seen=True, _rec=rec, _seen=seen_lookup):
                seen = _seen[u].indices if exclude_seen else None
                return _rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)

            metrics = evaluate_recommender(recommend_fn, val_by_user, k=20, users=val_users_sample)
            dt = time.time() - t0
            log(f"  factors={factors:>3} reg={reg:<5} -> "
                f"val NDCG@20={metrics['ndcg@k']:.4f} Recall@20={metrics['recall@k']:.4f} "
                f"({dt:.1f}s)")
            results.append({"factors": factors, "regularization": reg, **metrics})
    return pd.DataFrame(results)


def save(rec: ALSRecommender):
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_PATH, user_factors=rec.user_factors, item_factors=rec.item_factors)


def load() -> ALSRecommender:
    d = np.load(MODEL_PATH)
    return ALSRecommender(d["user_factors"], d["item_factors"])


def main():
    train = load_train_df()
    train_csr = load_train_csr()
    n_users, n_items = train_csr.shape

    log("Carving internal validation slice out of train for grid search "
        f"({VAL_ITEMS_PER_USER} items/user, disjoint from the Phase 1 holdout) ...")
    train_for_grid_csr, val_by_user = make_validation_split(train, n_users, n_items)

    rng = np.random.default_rng(SEED)
    val_users_sample = rng.choice(sorted(val_by_user.keys()), size=5000, replace=False)

    log(f"Grid search: factors={FACTORS_GRID} x regularization={REG_GRID} "
        f"(evaluated on {len(val_users_sample):,} sampled users) ...")
    grid_df = grid_search(train_for_grid_csr, val_by_user, val_users_sample)
    MODELS.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(MODELS / "als_grid_search.csv", index=False)

    best = grid_df.loc[grid_df["ndcg@k"].idxmax()]
    best_factors, best_reg = int(best["factors"]), float(best["regularization"])
    log(f"Best config: factors={best_factors}, regularization={best_reg} "
        f"(val NDCG@20={best['ndcg@k']:.4f})")

    log(f"Refitting on the full train set ({train_csr.nnz:,} interactions) ...")
    rec = fit_als(train_csr, best_factors, best_reg, show_progress=True)
    save(rec)
    log(f"Saved als_factors.npz: user_factors={rec.user_factors.shape}, "
        f"item_factors={rec.item_factors.shape}")

    holdout_by_user = load_holdout_by_user()

    def recommend_fn(u, k, exclude_seen=True):
        seen = train_csr[u].indices if exclude_seen else None
        return rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)

    metrics = evaluate_recommender(recommend_fn, holdout_by_user, k=20)
    log(f"ALS (factors={best_factors}, reg={best_reg}) Recall@20={metrics['recall@k']:.4f} "
        f"NDCG@20={metrics['ndcg@k']:.4f} (n_users={metrics['n_users']:,})")


if __name__ == "__main__":
    main()
