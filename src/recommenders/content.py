"""Phase 3 — Content-based recommender.

cosine(user_profile[u], core_content_matrix). Both matrices are already
L2-normalized per row (Phase 2), so a plain dot product is the cosine
similarity. score_batch() chunks over users for batch scoring so the full
(n_users, n_items) dense score matrix is never materialized at once.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import scipy.sparse as sp

from config import DATA_PROCESSED
from src.recommenders.common import (
    evaluate_recommender,
    load_core_content_matrix,
    load_holdout_by_user,
    load_train_csr,
    top_n_from_scores,
)

USER_PROFILES_PATH = DATA_PROCESSED / "user_profiles.npz"


class ContentRecommender:
    def __init__(self, user_profiles: sp.csr_matrix, content: sp.csr_matrix):
        self.user_profiles = user_profiles
        self.content = content

    def score_user(self, user_idx) -> np.ndarray:
        profile = self.user_profiles[user_idx]
        return np.asarray(self.content.dot(profile.T).todense()).ravel()

    def score_batch(self, user_idxs, chunk_size=2000):
        for start in range(0, len(user_idxs), chunk_size):
            chunk = user_idxs[start:start + chunk_size]
            profiles_chunk = self.user_profiles[chunk]
            scores_chunk = profiles_chunk.dot(self.content.T).toarray()
            yield chunk, scores_chunk

    def recommend(self, user_idx, n, exclude_seen=True, seen=None):
        scores = self.score_user(user_idx)
        exclude = seen if exclude_seen else None
        return top_n_from_scores(scores, n, exclude=exclude)


def load() -> ContentRecommender:
    user_profiles = sp.load_npz(USER_PROFILES_PATH).tocsr()
    content = load_core_content_matrix()
    return ContentRecommender(user_profiles, content)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    rec = load()
    log(f"Loaded content recommender: user_profiles={rec.user_profiles.shape}, "
        f"core_content_matrix={rec.content.shape}")

    train_csr = load_train_csr()
    holdout_by_user = load_holdout_by_user()

    def recommend_fn(u, k, exclude_seen=True):
        seen = train_csr[u].indices if exclude_seen else None
        return rec.recommend(u, k, exclude_seen=exclude_seen, seen=seen)

    metrics = evaluate_recommender(recommend_fn, holdout_by_user, k=20)
    log(f"Content Recall@20={metrics['recall@k']:.4f} "
        f"NDCG@20={metrics['ndcg@k']:.4f} (n_users={metrics['n_users']:,})")


if __name__ == "__main__":
    main()
