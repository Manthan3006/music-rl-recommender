"""Phase 2 — Content features.

Builds a single dense-ish item-content matrix (9 QuantileTransformed audio
features + 100-dim fixed-vocabulary TF-IDF tags, weighted 0.45/0.55) and
affinity-weighted user profile vectors from TRAIN interactions only. See
docs/SPEC.md Phase 2 for the full task list and acceptance criteria.

content_matrix.npz row i corresponds to items.parquet row i (same order,
covers the full cleaned catalogue, not just the interaction core) — this
lets the content recommender reach the 20,224 orphan tracks that have no
listening history at all.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import QuantileTransformer, normalize

from config import (
    AUDIO_FEATURE_WEIGHT,
    DATA_PROCESSED,
    MUSIC_INFO_CSV,
    MUSIC_INFO_ENCODING,
    SEED,
    TAG_FEATURE_WEIGHT,
)

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_log",
]

NEIGHBOUR_CHECK_TRACKS = ["Come as You Are", "Mr. Brightside", "Creep"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_tag_vocabulary() -> list:
    """Fixed 100-term controlled tag vocabulary, derived from the full raw
    catalogue (Part A: exactly 100 terms, each appearing >=50 times)."""
    raw = pd.read_csv(MUSIC_INFO_CSV, encoding=MUSIC_INFO_ENCODING, usecols=["tags"])
    tag_lists = raw["tags"].dropna().apply(lambda s: [t.strip() for t in s.split(",")])
    counts = pd.Series([t for tl in tag_lists for t in tl]).value_counts()
    vocab = sorted(counts.index.tolist())
    log(f"Tag vocabulary: {len(vocab)} terms (min count {int(counts.min())}).")
    assert len(vocab) == 100, f"expected 100 controlled tags, got {len(vocab)}"
    return vocab


def tag_tokenizer(s):
    return [t.strip() for t in s.split(",")] if isinstance(s, str) else []


def build_audio_block(items: pd.DataFrame) -> np.ndarray:
    df = items.copy()
    df["duration_log"] = np.log1p(df["duration_ms"])
    X = df[AUDIO_FEATURES].to_numpy(dtype=float)
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    X = qt.fit_transform(X)
    return normalize(X, norm="l2", axis=1)


def build_tag_block(items: pd.DataFrame, vocab: list) -> sp.csr_matrix:
    tags = items["tags"].fillna("")
    vec = TfidfVectorizer(
        vocabulary=vocab, tokenizer=tag_tokenizer, lowercase=False, token_pattern=None,
    )
    X = vec.fit_transform(tags)
    return normalize(X, norm="l2", axis=1)


def nearest_neighbours_check(items: pd.DataFrame, content: sp.csr_matrix,
                              query_names: list, k: int = 8):
    name_to_row = {}
    for i, n in enumerate(items["name"]):
        name_to_row.setdefault(n, i)

    print("\n=== QUALITATIVE NEAREST-NEIGHBOUR CHECK (Phase 2 acceptance) ===")
    for qname in query_names:
        if qname not in name_to_row:
            print(f"  [skip] {qname!r} not found in catalogue")
            continue
        qi = name_to_row[qname]
        sims = cosine_similarity(content[qi], content).ravel()
        order = [j for j in np.argsort(-sims) if j != qi][:k]
        q = items.iloc[qi]
        print(f"\n  Query: {q['name']!r} — {q['artist']} ({q['year']}, genre={q['genre']})")
        for j in order:
            row = items.iloc[j]
            print(f"    {sims[j]:.3f}  {row['name']!r} — {row['artist']} "
                  f"({row['year']}, genre={row['genre']})")


def build_user_profiles(items: pd.DataFrame, content: sp.csr_matrix) -> sp.csr_matrix:
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    holdout = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")
    idx_maps = pd.read_parquet(DATA_PROCESSED / "idx_maps.parquet")
    n_users = int(idx_maps.loc[idx_maps["entity_type"] == "user", "idx"].max()) + 1

    track_row = pd.Series(np.arange(len(items)), index=items["track_id"])
    train = train.copy()
    train["content_row"] = train["track_id"].map(track_row)
    missing = train["content_row"].isna().sum()
    if missing:
        log(f"WARNING: {missing} train rows reference a track_id absent from "
            f"items.parquet (dropped during Phase 1 catalog cleaning); excluding.")
        train = train.dropna(subset=["content_row"])
    train["content_row"] = train["content_row"].astype(int)

    # Vectorised affinity-weighted sum via sparse matmul:
    # profiles[n_users, dims] = A[n_users, n_train] @ content[train_rows, dims]
    n_train = len(train)
    A = sp.csr_matrix(
        (train["aff_ui"].to_numpy(), (train["user_idx"].to_numpy(), np.arange(n_train))),
        shape=(n_users, n_train),
    )
    content_train = content[train["content_row"].to_numpy()]
    profiles = A @ content_train
    profiles = normalize(profiles, norm="l2", axis=1)

    # Acceptance: no holdout track may have contributed to any profile.
    holdout_pairs = set(zip(holdout["user_idx"].tolist(), holdout["track_id"].tolist()))
    train_pairs = set(zip(train["user_idx"].tolist(), train["track_id"].tolist()))
    overlap = holdout_pairs & train_pairs
    print(f"\nHoldout tracks used in any user profile: {len(overlap)} "
          f"(requirement: 0) -> {'PASS' if len(overlap) == 0 else 'FAIL'}")

    empty_profiles = int((profiles.getnnz(axis=1) == 0).sum())
    log(f"User profiles: {profiles.shape[0]:,} users, {empty_profiles} empty profiles.")
    return profiles


def main():
    t0 = time.time()
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    log(f"Loaded {len(items):,} catalogue tracks.")

    vocab = build_tag_vocabulary()
    audio = build_audio_block(items)
    tags = build_tag_block(items, vocab)
    log(f"audio block shape={audio.shape}, tag block shape={tags.shape}")

    audio_w = sp.csr_matrix(audio) * AUDIO_FEATURE_WEIGHT
    tags_w = tags.multiply(TAG_FEATURE_WEIGHT).tocsr()
    content = sp.hstack([audio_w, tags_w], format="csr")
    content = normalize(content, norm="l2", axis=1)
    log(f"content_matrix shape={content.shape}, nnz={content.nnz:,} "
        f"(weights: audio={AUDIO_FEATURE_WEIGHT}, tags={TAG_FEATURE_WEIGHT})")

    sp.save_npz(DATA_PROCESSED / "content_matrix.npz", content)
    log("Wrote content_matrix.npz (row i == items.parquet row i).")

    nearest_neighbours_check(items, content, NEIGHBOUR_CHECK_TRACKS)

    profiles = build_user_profiles(items, content)
    sp.save_npz(DATA_PROCESSED / "user_profiles.npz", profiles)
    log(f"Wrote user_profiles.npz shape={profiles.shape}")

    log(f"Phase 2 complete in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
