"""Phase 6 prerequisite — fixed global song clusters (B.7).

B.7 offers two ways to build the K=10 arms: "KMeans on the normalised
audio-feature + tag vector, refit per candidate set, or use fixed global
clusters." Phase 5 initially refit KMeans per user's own candidate pool.
That has to change for Phase 6: EpsilonGreedyMAB, UCB1, and even the
Q-table's convergence over 20,000 episodes all require arm identity to mean
the same thing across every episode, so a persistent value/Q estimate for
"arm 3" learned from user A's episodes is meaningful when user B's episode
comes along. Per-candidate-set refit can't give you that — KMeans labels are
arbitrary and independently assigned per fit, so "cluster 3" for user A and
"cluster 3" for user B need not share any content meaning.

This script fits ONE KMeans(K=N_ARMS) over the full core catalogue's content
vectors and persists the resulting item_idx -> cluster_label array. Every
PlaylistEnv now looks up clusters here instead of fitting its own.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.cluster import KMeans

from config import MODELS, N_ARMS, SEED
from src.recommenders.common import load_core_content_matrix

CLUSTERS_PATH = MODELS / "global_song_clusters.npy"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_and_save() -> np.ndarray:
    content = load_core_content_matrix()
    log(f"Fitting global KMeans(K={N_ARMS}) over {content.shape[0]:,} core-catalogue items ...")
    km = KMeans(n_clusters=N_ARMS, random_state=SEED, n_init=10)
    labels = km.fit_predict(content.toarray())

    MODELS.mkdir(parents=True, exist_ok=True)
    np.save(CLUSTERS_PATH, labels)

    sizes = np.bincount(labels, minlength=N_ARMS)
    log(f"Saved {CLUSTERS_PATH}. Cluster sizes: {sizes.tolist()}")
    return labels


def load() -> np.ndarray:
    if not CLUSTERS_PATH.exists():
        return fit_and_save()
    return np.load(CLUSTERS_PATH)


if __name__ == "__main__":
    fit_and_save()
