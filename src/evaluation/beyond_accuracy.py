"""Phase 7 — beyond-accuracy metrics (B.15).

Per-list metrics (one user's playlist): intra-list diversity, novelty, artist
entropy. Population metrics (all lists in an evaluation fold): catalogue
coverage and the Gini index of item exposure.
"""

import numpy as np


def intra_list_diversity(items, content) -> float:
    """1 - mean pairwise content cosine. `content` rows are L2-normalised
    (Phase 2), so a plain dot product is the cosine."""
    if len(items) < 2:
        return np.nan
    vecs = content[list(items)].toarray()
    sims = vecs @ vecs.T
    iu = np.triu_indices(len(items), k=1)
    return float(1.0 - sims[iu].mean())


def novelty(items, item_popularity) -> float:
    """Mean self-information -log2(p_i), p_i = fraction of TRAIN users who
    listened to item i. Higher = less popular items."""
    return float(np.mean(-np.log2(item_popularity[list(items)])))


def artist_entropy(items, artist_of) -> float:
    """Shannon entropy (bits) of the artist distribution inside one list;
    max is log2(K) when every track has a distinct artist."""
    _, counts = np.unique(artist_of[list(items)], return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def catalog_coverage(all_items, n_items) -> float:
    """Fraction of the catalogue appearing in at least one list."""
    return len(set(int(i) for i in all_items)) / n_items


def gini_index(all_items, n_items) -> float:
    """Gini of item exposure across the whole catalogue (0 = every item shown
    equally often, ->1 = exposure concentrated on a few items). Items never
    recommended count as zero exposure."""
    counts = np.bincount(np.asarray(list(all_items), dtype=int), minlength=n_items).astype(float)
    counts.sort()
    n = len(counts)
    cum = np.cumsum(counts)
    if cum[-1] == 0:
        return np.nan
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
