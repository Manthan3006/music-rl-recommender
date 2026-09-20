"""Phase 7 — the evaluation user folds and the leakage assertions.

Population: users NOT in the Phase 6 RL training curriculum (the fixed
5,000-user curriculum the agents trained on, src/rl/train.py). The RL reward
is computed from each curriculum user's own held-out tracks, so evaluating an
agent on a curriculum user would score it on data it was trained against.
Excluding them from *every* configuration keeps all rows of the comparison on
the same users.

5,000 of those users are drawn with SEED and split into 5 disjoint folds of
1,000. The "5 seeds" of B.15 are these 5 folds: each configuration is scored
once per fold and reported as mean +/- std across folds, so the std measures
sampling variability over users (plus, for RL rows, the mean over the 3
trained checkpoints — see evaluate_all.py).

`assert_no_leakage` raises (AssertionError) on any of:
  1. an eval user who is in the RL training curriculum;
  2. a user whose held-out track also appears in their TRAIN matrix row;
  3. a stored user profile that is not reproducible from train rows alone
     (and, as a power check, that WOULD change if holdout rows were included).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from config import DATA_PROCESSED, SEED
from src.recommenders.common import load_core_content_matrix, load_train_aff_csr, load_train_csr

N_EVAL_USERS = 5_000
N_FOLDS = 5
FOLD_SEEDS = [42, 43, 44, 45, 46]  # fold f <-> "seed" FOLD_SEEDS[f]


def curriculum_users() -> np.ndarray:
    """Re-derive the exact Phase 6 training curriculum (same function, same
    seed, same user list as src/rl/train.py)."""
    from src.rl.train import N_EPISODES, sample_curriculum

    cand_users = pd.read_parquet(DATA_PROCESSED / "candidates.parquet", columns=["user_idx"])["user_idx"].unique()
    hold_users = pd.read_parquet(DATA_PROCESSED / "holdout.parquet", columns=["user_idx"])["user_idx"].unique()
    users = np.array(sorted(set(cand_users.tolist()) & set(hold_users.tolist())))
    return sample_curriculum(users, N_EPISODES, seed=SEED), users


def get_eval_folds():
    curriculum, users = curriculum_users()
    pool = np.array(sorted(set(users.tolist()) - set(int(u) for u in curriculum)))
    chosen = np.random.default_rng(SEED).choice(pool, size=N_EVAL_USERS, replace=False)
    fold_size = N_EVAL_USERS // N_FOLDS
    folds = [np.sort(chosen[f * fold_size:(f + 1) * fold_size]) for f in range(N_FOLDS)]
    return folds, curriculum


def assert_no_leakage(folds, curriculum, holdout_by_user, n_profile_check=200):
    eval_users = np.concatenate(folds)
    train_users = set(int(u) for u in curriculum)
    assert len(set(eval_users.tolist())) == len(eval_users) == N_EVAL_USERS, "eval folds overlap"
    assert not (set(eval_users.tolist()) & train_users), "LEAKAGE: eval user is in the RL training curriculum"

    train_csr = load_train_csr()
    for u in eval_users:
        seen = set(train_csr[int(u)].indices.tolist())
        assert len(holdout_by_user[int(u)]) == 5, f"user {u} does not have exactly 5 holdout tracks"
        assert not (seen & holdout_by_user[int(u)]), f"LEAKAGE: user {u} has a holdout track in train"

    # user profiles must be reproducible from TRAIN rows only
    profiles = sp.load_npz(DATA_PROCESSED / "user_profiles.npz").tocsr()
    content = load_core_content_matrix()
    aff = load_train_aff_csr()
    holdout = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")
    sample = np.random.default_rng(SEED).choice(eval_users, size=n_profile_check, replace=False)
    n_would_change = 0
    for u in sample:
        u = int(u)
        expected = normalize(aff[u] @ content, norm="l2")
        stored = profiles[u]
        assert abs(expected - stored).max() < 1e-6, f"LEAKAGE: stored profile of user {u} is not train-only"
        h = holdout[holdout["user_idx"] == u]
        with_holdout = aff[u].toarray().ravel()
        with_holdout[h["item_idx"].to_numpy()] += h["aff_ui"].to_numpy()
        contaminated = normalize(sp.csr_matrix(with_holdout) @ content, norm="l2")
        n_would_change += int(abs(contaminated - stored).max() > 1e-6)
    assert n_would_change == n_profile_check, "profile leakage check has no power (holdout would not change profiles)"
    return {
        "eval_users": len(eval_users),
        "overlap_with_rl_curriculum": 0,
        "users_with_train_holdout_overlap": 0,
        "profiles_checked": n_profile_check,
        "profiles_train_only": n_profile_check,
    }


if __name__ == "__main__":
    from src.recommenders.common import load_holdout_by_user

    folds, curriculum = get_eval_folds()
    print("fold sizes:", [len(f) for f in folds])
    print("leakage checks:", assert_no_leakage(folds, curriculum, load_holdout_by_user()))
