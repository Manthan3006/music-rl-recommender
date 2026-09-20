"""Phase 7 — generate frozen-policy RL playlists for the evaluation folds.

For every (agent, trained checkpoint, K) it loads the Phase 6 grounded
checkpoint, disables exploration exactly as the Phase 6 table of record does
(epsilon=0 for Q-Learning/SARSA, argmax-q for UCB1, alpha=0 for LinUCB), and
builds one playlist of length K per evaluation user with select_action only —
no agent is ever updated here. The environment is run with
`playlist_length = K` so the discretised position bucket and LinUCB's
position feature keep their meaning (early/mid/late thirds of the playlist)
even though the agents were trained at length 15.

Output (data/processed/, intermediate):
  phase7_rl_playlists.parquet — agent, train_seed, K, user_idx, position, item_idx
  phase7_rl_rewards.parquet   — agent, train_seed, K, user_idx, reward (grounded)
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from src.evaluation.eval_users import get_eval_folds
from src.recommenders.common import load_core_content_matrix
from src.rl.environment import GROUNDED, EnvConfig, PlaylistEnv
from src.rl.evaluate_phase6_final import disable_exploration
from src.rl.train import SEEDS, arm_best_scores, models_dir

KS = [10, 20]
LEARNED_AND_RANDOM = ["Random", "UCB1", "LinUCB", "QLearning", "SARSA"]
PLAYLISTS_PATH = DATA_PROCESSED / "phase7_rl_playlists.parquet"
REWARDS_PATH = DATA_PROCESSED / "phase7_rl_rewards.parquet"

_G = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _init(eval_users):
    _G["content"] = load_core_content_matrix()
    cand = pd.read_parquet(DATA_PROCESSED / "candidates.parquet")
    cand = cand[cand["user_idx"].isin(eval_users)]
    hold = pd.read_parquet(DATA_PROCESSED / "holdout.parquet")
    hold = hold[hold["user_idx"].isin(eval_users)]
    _G["cand"] = {u: g[["item_idx", "hybrid_score"]].reset_index(drop=True) for u, g in cand.groupby("user_idx")}
    _G["hold"] = {u: dict(zip(g["item_idx"], g["aff_ui"])) for u, g in hold.groupby("user_idx")}
    _G["users"] = sorted(int(u) for u in eval_users)  # fixed order: Random-arm outcomes depend on it


def _episode(env, agent):
    state = env.reset()
    total, done = 0.0, False
    while not done:
        info = {"episode": 10 ** 9}  # past every decay schedule
        if agent.ACTION_LEVEL == "song":
            info["song_contexts"] = env.song_contexts()
            info["unused_positions"] = env.unused_positions()
        else:
            info["arm_best_score"] = arm_best_scores(env)
        action = agent.select_action(state, info=info)
        step = env.step_song if agent.ACTION_LEVEL == "song" else env.step
        state, reward, done, _ = step(action)
        total += reward
    return total


def _run_task(task):
    name, seed, k = task
    t0 = time.time()
    with open(models_dir(GROUNDED) / f"{name}_seed{seed}.pkl", "rb") as f:
        agent = disable_exploration(name, pickle.load(f))
    cfg = EnvConfig(playlist_length=k, reward_mode=GROUNDED)
    play_rows, rew_rows = [], []
    for u in _G["users"]:
        u = int(u)
        env = PlaylistEnv(u, _G["cand"][u], _G["content"], _G["hold"][u], cfg)
        reward = _episode(env, agent)
        assert len(env.playlist) == k and len(set(env.playlist)) == k, "duplicate or short playlist"
        for pos, item in enumerate(env.playlist):
            play_rows.append((name, seed, k, u, pos, int(item)))
        rew_rows.append((name, seed, k, u, reward))
    return task, play_rows, rew_rows, time.time() - t0


def main():
    folds, _ = get_eval_folds()
    eval_users = np.concatenate(folds)
    tasks = [(n, s, k) for k in KS for n in LEARNED_AND_RANDOM for s in SEEDS] + [("Greedy", SEEDS[0], k) for k in KS]
    # slowest (LinUCB) first so the pool stays busy
    tasks.sort(key=lambda t: t[0] != "LinUCB")
    log(f"{len(tasks)} tasks x {len(eval_users):,} eval users, 7 workers")

    play_all, rew_all = [], []
    with Pool(7, initializer=_init, initargs=(set(int(u) for u in eval_users),)) as pool:
        for task, play_rows, rew_rows, dt in pool.imap_unordered(_run_task, tasks):
            log(f"  done {task} in {dt:.0f}s")
            play_all += play_rows
            rew_all += rew_rows

    cols = ["agent", "train_seed", "K", "user_idx", "position", "item_idx"]
    pd.DataFrame(play_all, columns=cols).to_parquet(PLAYLISTS_PATH, index=False)
    pd.DataFrame(rew_all, columns=["agent", "train_seed", "K", "user_idx", "reward"]).to_parquet(REWARDS_PATH, index=False)
    log(f"Saved {PLAYLISTS_PATH.name} ({len(play_all):,} rows) and {REWARDS_PATH.name}")


if __name__ == "__main__":
    main()
