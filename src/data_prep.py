"""Phase 1 — Preprocessing & filtering.

Merges the 9 duplicate track pairs, drops bad-audio tracks, applies iterative
k-core filtering, subsamples to 40,000 users, computes p_ui/c_ui/aff_ui, and
writes the leave-5-out train/holdout split plus the train confidence CSR
matrix. See docs/SPEC.md Phase 1 for the full task list and acceptance
criteria.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import (
    ALS_ALPHA,
    CONFIDENCE_EPS,
    DATA_PROCESSED,
    MIN_TRACK_LISTENERS,
    MIN_USER_INTERACTIONS,
    MUSIC_INFO_CSV,
    MUSIC_INFO_ENCODING,
    POST_SUBSAMPLE_MIN_TRACK_LISTENERS,
    SEED,
    SUBSAMPLE_USERS,
)

RAW_HISTORY_PARQUET = DATA_PROCESSED / "listening_history_raw.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    log("Loading Music_Info.csv ...")
    music = pd.read_csv(MUSIC_INFO_CSV, encoding=MUSIC_INFO_ENCODING)

    log("Loading cached listening_history_raw.parquet ...")
    hist = pd.read_parquet(RAW_HISTORY_PARQUET)
    return music, hist


def merge_duplicate_tracks(music: pd.DataFrame, hist: pd.DataFrame):
    """Merge the 9 duplicate spotify_id pairs: keep the more-listened
    track_id as canonical, sum playcounts under it, drop the other row from
    the catalogue."""
    dup_mask = music.duplicated("spotify_id", keep=False) & music["spotify_id"].notna()
    dup = music.loc[dup_mask, ["spotify_id", "track_id"]]

    listeners = hist["track_id"].value_counts()

    mapping = {}
    drop_track_ids = set()
    for sid, group in dup.groupby("spotify_id"):
        tids = group["track_id"].tolist()
        counts = [(tid, listeners.get(tid, 0)) for tid in tids]
        counts.sort(key=lambda x: (-x[1], x[0]))  # highest listeners wins, tie -> smaller id
        canonical = counts[0][0]
        for tid, _ in counts[1:]:
            mapping[tid] = canonical
            drop_track_ids.add(tid)

    log(f"Duplicate merge: {len(mapping)} track_ids remapped to their canonical twin.")

    # Remap history track_id (categorical-safe, vectorized via category array).
    cats = hist["track_id"].cat.categories.to_numpy()
    remapped_cats = np.array([mapping.get(c, c) for c in cats])
    codes = hist["track_id"].cat.codes.to_numpy()
    hist = hist.copy()
    hist["track_id"] = pd.Series(remapped_cats[codes], index=hist.index).astype("category")

    # Collapse any (user_id, track_id) rows that collided after remapping.
    before = len(hist)
    hist = (
        hist.groupby(["user_id", "track_id"], observed=True)["playcount"]
        .sum()
        .reset_index()
    )
    after = len(hist)
    log(f"Post-merge collision collapse: {before:,} -> {after:,} rows "
        f"({before - after} rows absorbed).")

    # Drop the non-canonical duplicate rows from the catalogue.
    music = music[~music["track_id"].isin(drop_track_ids)].copy()

    return music, hist, mapping


def clean_catalog(music: pd.DataFrame) -> pd.DataFrame:
    n0 = len(music)
    bad_tempo = music["tempo"] == 0
    bad_duration = (music["duration_ms"] < 60_000) | (music["duration_ms"] > 600_000)
    bad = bad_tempo | bad_duration
    log(f"Dropping {bad_tempo.sum()} tempo==0 tracks and {bad_duration.sum()} "
        f"duration-out-of-range tracks ({bad.sum()} unique rows total).")
    music = music[~bad].copy()

    music.loc[music["year"] == 1900, "year"] = np.nan
    music["decade"] = (music["year"] // 10 * 10)

    log(f"Catalogue: {n0:,} -> {len(music):,} tracks after cleaning.")
    return music


def filter_history_to_catalog(hist: pd.DataFrame, valid_track_ids: set) -> pd.DataFrame:
    before = len(hist)
    hist = hist[hist["track_id"].isin(valid_track_ids)].copy()
    hist["track_id"] = hist["track_id"].cat.remove_unused_categories()
    log(f"Filtered history to cleaned catalogue: {before:,} -> {len(hist):,} rows.")
    return hist


def iterative_k_core(hist: pd.DataFrame, min_track=MIN_TRACK_LISTENERS,
                      min_user=MIN_USER_INTERACTIONS, label="") -> pd.DataFrame:
    hist = hist.copy()
    it = 0
    while True:
        it += 1
        n_before = len(hist)

        track_counts = hist.groupby("track_id", observed=True).size()
        keep_tracks = track_counts.index[track_counts >= min_track]
        hist = hist[hist["track_id"].isin(keep_tracks)]

        user_counts = hist.groupby("user_id", observed=True).size()
        keep_users = user_counts.index[user_counts >= min_user]
        hist = hist[hist["user_id"].isin(keep_users)]

        n_after = len(hist)
        log(f"  [{label} k-core iter {it}] {n_before:,} -> {n_after:,} rows, "
            f"{hist['user_id'].nunique():,} users, {hist['track_id'].nunique():,} tracks")
        if n_after == n_before:
            break

    hist["track_id"] = hist["track_id"].cat.remove_unused_categories()
    hist["user_id"] = hist["user_id"].cat.remove_unused_categories()
    return hist


def summarize(hist: pd.DataFrame, label: str) -> dict:
    n_users = hist["user_id"].nunique()
    n_tracks = hist["track_id"].nunique()
    n_rows = len(hist)
    density = n_rows / (n_users * n_tracks) if n_users and n_tracks else 0.0
    summary = {
        "label": label,
        "interactions": n_rows,
        "users": n_users,
        "tracks": n_tracks,
        "density_pct": round(100 * density, 4),
    }
    log(f"=== {label}: interactions={n_rows:,}  users={n_users:,}  "
        f"tracks={n_tracks:,}  density={summary['density_pct']}% ===")
    return summary


def subsample_users(hist: pd.DataFrame, n_users=SUBSAMPLE_USERS, seed=SEED) -> pd.DataFrame:
    unique_users = np.sort(hist["user_id"].cat.categories.to_numpy())
    # categories may include some not actually present after prior filtering;
    # restrict to those with actual rows.
    present = hist["user_id"].unique()
    unique_users = np.sort(np.asarray(present, dtype=str))
    rng = np.random.default_rng(seed)
    if len(unique_users) <= n_users:
        log(f"Only {len(unique_users)} users available, skipping subsample.")
        return hist
    chosen = rng.choice(unique_users, size=n_users, replace=False)
    chosen_set = set(chosen.tolist())
    hist = hist[hist["user_id"].isin(chosen_set)].copy()
    hist["user_id"] = hist["user_id"].cat.remove_unused_categories()
    log(f"Subsampled to {n_users:,} users (seed={seed}): {len(hist):,} rows remain "
        f"before refilter.")
    return hist


def compute_confidence_fields(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    hist["p_ui"] = 1
    hist["c_ui"] = 1 + ALS_ALPHA * np.log1p(hist["playcount"] / CONFIDENCE_EPS)
    log1p_pc = np.log1p(hist["playcount"])
    user_max = log1p_pc.groupby(hist["user_id"], observed=True).transform("max")
    hist["aff_ui"] = log1p_pc / user_max
    return hist


def build_idx_maps(hist: pd.DataFrame):
    users = np.sort(hist["user_id"].unique().astype(str))
    items = np.sort(hist["track_id"].unique().astype(str))
    user_idx = pd.Series(np.arange(len(users)), index=users)
    item_idx = pd.Series(np.arange(len(items)), index=items)
    hist = hist.copy()
    hist["user_idx"] = hist["user_id"].astype(str).map(user_idx).astype("int32")
    hist["item_idx"] = hist["track_id"].astype(str).map(item_idx).astype("int32")
    return hist, users, items


def leave_five_out_split(hist: pd.DataFrame, seed=SEED):
    holdout = hist.groupby("user_id", observed=True, group_keys=False).sample(
        n=5, random_state=seed
    )
    train = hist.drop(holdout.index)
    return train, holdout


def run_leakage_assertions(train: pd.DataFrame, holdout: pd.DataFrame):
    print("\n=== LEAKAGE / INVARIANT ASSERTIONS ===")

    train_counts = train.groupby("user_id", observed=True).size()
    min_train = int(train_counts.min())
    print(f"train.groupby('user_id').size().min() = {min_train}  "
          f"(requirement: >= 15) -> {'PASS' if min_train >= 15 else 'FAIL'}")

    holdout_counts = holdout.groupby("user_id", observed=True).size()
    holdout_size_dist = holdout_counts.value_counts().to_dict()
    print(f"holdout.groupby('user_id').size().value_counts() = {holdout_size_dist}  "
          f"(requirement: every user has exactly 5) -> "
          f"{'PASS' if set(holdout_size_dist.keys()) == {5} else 'FAIL'}")

    train_pairs = set(zip(train["user_id"].astype(str), train["track_id"].astype(str)))
    holdout_pairs = set(zip(holdout["user_id"].astype(str), holdout["track_id"].astype(str)))
    overlap = train_pairs & holdout_pairs
    print(f"len(train_pairs & holdout_pairs) = {len(overlap)}  "
          f"(requirement: 0) -> {'PASS' if len(overlap) == 0 else 'FAIL'}")

    train_users = set(train["user_id"].unique())
    holdout_users = set(holdout["user_id"].unique())
    print(f"train_users == holdout_users: {train_users == holdout_users}  "
          f"(len train_users={len(train_users)}, len holdout_users={len(holdout_users)})")

    total_train_plus_holdout = len(train) + len(holdout)
    print(f"len(train) + len(holdout) = {total_train_plus_holdout}")


def main():
    t_start = time.time()
    music, hist = load_inputs()

    music, hist, dup_mapping = merge_duplicate_tracks(music, hist)
    music = clean_catalog(music)

    valid_track_ids = set(music["track_id"])
    hist = filter_history_to_catalog(hist, valid_track_ids)

    log("Running initial iterative k-core filter (user>=20, track>=50) ...")
    hist_core = iterative_k_core(hist, label="pre-subsample")
    pre_subsample_summary = summarize(hist_core, "PRE-SUBSAMPLE k-core result")

    hist_sub = subsample_users(hist_core, n_users=SUBSAMPLE_USERS, seed=SEED)
    log(f"Re-running full iterative k-core after subsample (user>={MIN_USER_INTERACTIONS}, "
        f"track>={POST_SUBSAMPLE_MIN_TRACK_LISTENERS}) — NOT the population-scale "
        f"track>={MIN_TRACK_LISTENERS} threshold; see docs/SPEC.md Phase 1 task 5.")
    hist_final = iterative_k_core(
        hist_sub, min_track=POST_SUBSAMPLE_MIN_TRACK_LISTENERS,
        min_user=MIN_USER_INTERACTIONS, label="post-subsample",
    )
    post_subsample_summary = summarize(hist_final, "POST-SUBSAMPLE final result")

    n_artists = music.set_index("track_id").loc[
        hist_final["track_id"].unique(), "artist"
    ].nunique()
    log(f"Distinct artists in final core: {n_artists:,}")

    print("\n=== K-CORE BEFORE vs AFTER 40,000-USER SUBSAMPLE ===")
    print(f"{'':25s}{'interactions':>15s}{'users':>12s}{'tracks':>10s}{'density%':>10s}{'artists':>10s}")
    for s in (pre_subsample_summary, post_subsample_summary):
        artists_col = f"{n_artists:>10,d}" if s is post_subsample_summary else f"{'':>10s}"
        print(f"{s['label']:25s}{s['interactions']:>15,d}{s['users']:>12,d}"
              f"{s['tracks']:>10,d}{s['density_pct']:>10.4f}{artists_col}")

    log("Computing p_ui, c_ui, aff_ui ...")
    hist_final = compute_confidence_fields(hist_final)

    log("Building integer index maps ...")
    hist_final, users, items = build_idx_maps(hist_final)

    log("Splitting leave-5-out per user ...")
    train, holdout = leave_five_out_split(hist_final, seed=SEED)

    run_leakage_assertions(train, holdout)

    # --- Persist outputs ---
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    music["track_id_in_core"] = music["track_id"].isin(set(items.tolist()))
    music_out = music.copy()
    music_out.to_parquet(DATA_PROCESSED / "items.parquet", index=False)
    log(f"Wrote items.parquet ({len(music_out):,} catalogue tracks, "
        f"{music_out['track_id_in_core'].sum():,} in the interaction core).")

    keep_cols = ["user_id", "track_id", "user_idx", "item_idx", "playcount",
                 "p_ui", "c_ui", "aff_ui"]
    hist_final[keep_cols].to_parquet(DATA_PROCESSED / "interactions.parquet", index=False)
    train[keep_cols].to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    holdout[keep_cols].to_parquet(DATA_PROCESSED / "holdout.parquet", index=False)
    log("Wrote interactions.parquet, train.parquet, holdout.parquet.")

    idx_map_df = pd.concat([
        pd.DataFrame({"entity_type": "user", "raw_id": users, "idx": np.arange(len(users))}),
        pd.DataFrame({"entity_type": "item", "raw_id": items, "idx": np.arange(len(items))}),
    ], ignore_index=True)
    idx_map_df.to_parquet(DATA_PROCESSED / "idx_maps.parquet", index=False)
    log(f"Wrote idx_maps.parquet ({len(users):,} users + {len(items):,} items).")

    n_users, n_items = len(users), len(items)
    train_csr = sp.csr_matrix(
        (train["c_ui"].to_numpy(), (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    )
    sp.save_npz(DATA_PROCESSED / "train_csr.npz", train_csr)
    log(f"Wrote train_csr.npz, shape={train_csr.shape}, nnz={train_csr.nnz:,}.")

    log(f"Phase 1 complete in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
