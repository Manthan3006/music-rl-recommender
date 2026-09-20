"""Phase 7/9 diagnostic — what a frozen (bonus-free) UCB1 actually plays.

With exploration disabled UCB1 requests argmax(q) at every step, i.e. one fixed
content cluster. PlaylistEnv._select_candidate falls back to the best unused
candidate of ANY cluster once that cluster's candidates in the user's pool are
exhausted, so how many pool candidates the chosen arm holds decides how much of
the playlist is really "UCB1" and how much is the hybrid ranking (Greedy).
Writes results/phase7/ucb1_frozen_arm.csv.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, RESULTS
from src.rl.environment import GROUNDED
from src.rl.train import SEEDS, models_dir


def main():
    cand = pd.read_parquet(DATA_PROCESSED / "candidates.parquet", columns=["user_idx", "item_idx"])
    cand["arm"] = np.load(Path(__file__).resolve().parent.parent.parent / "models" / "global_song_clusters.npy")[cand["item_idx"].to_numpy()]
    per_arm = cand.groupby(["user_idx", "arm"]).size().unstack(fill_value=0)
    rows = []
    for s in SEEDS:
        q = pickle.load(open(models_dir(GROUNDED) / f"UCB1_seed{s}.pkl", "rb")).q
        arm = int(np.argmax(q))
        rows.append({"seed": s, "frozen_arm": arm, "q_of_arm": float(q[arm]), "q_min_all_arms": float(q.min()),
                     "q_max_all_arms": float(q.max()), "median_pool_candidates_in_arm": float(per_arm[arm].median()),
                     "mean_pool_candidates_in_arm": float(per_arm[arm].mean()),
                     "share_users_arm_holds_ge_10": float((per_arm[arm] >= 10).mean()),
                     "share_users_arm_holds_ge_15": float((per_arm[arm] >= 15).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phase7" / "ucb1_frozen_arm.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
