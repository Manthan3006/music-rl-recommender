"""Phase 3 — Popularity recommender (baseline).

Global top-N by distinct listeners and by total playcount. No
personalization beyond excluding a user's own train items.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import MODELS
from src.recommenders.common import (
    evaluate_recommender,
    load_holdout_by_user,
    load_train_csr,
    load_train_df,
    top_n_from_scores,
)

MODEL_PATH = MODELS / "popularity_scores.parquet"


class PopularityRecommender:
    """by='listeners' (default) ranks by distinct-listener count, which is
    more robust to a handful of power-users inflating one track's playcount;
    by='playcount' ranks by total raw playcount. Both are always computed and
    saved; `by` selects which one recommend() uses."""

    def __init__(self, listeners: np.ndarray, playcount: np.ndarray, by: str = "listeners"):
        self.listeners = listeners
        self.playcount = playcount
        self.by = by

    def scores(self) -> np.ndarray:
        return self.listeners if self.by == "listeners" else self.playcount

    def recommend(self, user_idx, n, exclude_seen=True, seen=None):
        exclude = seen if exclude_seen else None
        return top_n_from_scores(self.scores(), n, exclude=exclude)


def fit(train: pd.DataFrame, n_items: int) -> PopularityRecommender:
    listeners = np.zeros(n_items)
    playcount = np.zeros(n_items)
    g = train.groupby("item_idx")
    listeners_s = g["user_idx"].nunique()
    playcount_s = g["playcount"].sum()
    listeners[listeners_s.index.to_numpy()] = listeners_s.to_numpy()
    playcount[playcount_s.index.to_numpy()] = playcount_s.to_numpy()
    return PopularityRecommender(listeners, playcount)


def save(rec: PopularityRecommender):
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "item_idx": np.arange(len(rec.listeners)),
        "listeners": rec.listeners,
        "playcount": rec.playcount,
    })
    df.to_parquet(MODEL_PATH, index=False)


def load() -> PopularityRecommender:
    df = pd.read_parquet(MODEL_PATH)
    return PopularityRecommender(df["listeners"].to_numpy(), df["playcount"].to_numpy())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    train = load_train_df()
    train_csr = load_train_csr()
    n_items = train_csr.shape[1]

    rec = fit(train, n_items)
    save(rec)
    log(f"Fitted popularity scores over {n_items:,} items "
        f"({train['user_idx'].nunique():,} users).")

    top10_listeners = np.argsort(-rec.listeners)[:10]
    top10_playcount = np.argsort(-rec.playcount)[:10]
    log(f"Top-10 by listeners: {top10_listeners.tolist()}")
    log(f"Top-10 by playcount: {top10_playcount.tolist()}")

    holdout_by_user = load_holdout_by_user()

    def recommend_fn(u, k, exclude_seen=True):
        seen = train_csr[u].indices if exclude_seen else None
        return rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)

    metrics = evaluate_recommender(recommend_fn, holdout_by_user, k=20)
    log(f"Popularity (by={rec.by}) Recall@20={metrics['recall@k']:.4f} "
        f"NDCG@20={metrics['ndcg@k']:.4f} (n_users={metrics['n_users']:,})")


if __name__ == "__main__":
    main()
