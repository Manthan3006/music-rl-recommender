"""Playlist construction service (Part C's last stage) used by the Streamlit UI.

Everything here is inference-only: the Phase 6 agents are loaded from
models/rl/grounded/ (training seed 42), frozen with the SAME exploration-off
rules as the Phase 6 table of record (`disable_exploration`), and driven
through PlaylistEnv with `playlist_length = the requested length`. Nothing is
trained or fitted at request time.

Serving uses `holdout={}` in the environment, so no agent can see a held-out
track. Held-out tracks are exposed only through `holdout_of(user)` for the
UI's opt-in "reveal held-out tracks" evaluation view.
"""

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from src.evaluation import beyond_accuracy as ba
from src.evaluation.eval_users import curriculum_users
from src.recommenders.common import load_core_content_matrix, load_idx_maps
from src.rl.environment import GROUNDED, EnvConfig, PlaylistEnv
from src.rl.evaluate_phase6_final import disable_exploration, run_frozen_episode
from src.rl.train import models_dir

SERVING_SEED = 42
MIN_LENGTH, MAX_LENGTH = 5, 20
# UI label -> (checkpoint name, one-line description)
AGENTS = {
    "Hybrid (Greedy)": ("Greedy", "Top of the hybrid ranking, no RL. The strongest playlist by hit rate in our evaluation."),
    "Random arm": ("Random", "Hybrid pool, but a uniformly random content cluster each step (the RL control)."),
    "UCB1": ("UCB1", "Bandit over 10 content clusters, no state."),
    "LinUCB": ("LinUCB", "Song-level contextual bandit (user profile + track + playlist state)."),
    "Q-Learning": ("QLearning", "Tabular off-policy RL over 162 playlist states x 10 clusters."),
    "SARSA": ("SARSA", "Tabular on-policy RL over 162 playlist states x 10 clusters."),
}


@dataclass
class BuiltPlaylist:
    tracks: pd.DataFrame        # one row per position, with display metadata
    scores: dict                # playlist-level diversity / novelty numbers
    seconds: float
    heldout_hits: int           # tracks that are among the user's 5 held-out (evaluation view only)


class PlaylistService:
    def __init__(self):
        t0 = time.time()
        self.content = load_core_content_matrix()
        _, core_track_ids = load_idx_maps()
        self.n_items = len(core_track_ids)

        items = pd.read_parquet(DATA_PROCESSED / "items.parquet",
                                columns=["track_id", "name", "artist", "genre", "tags", "year", "spotify_preview_url"])
        self.meta = items.set_index("track_id").loc[core_track_ids].reset_index(drop=True)
        self.artist_of = self.meta["artist"].to_numpy()

        cand = pd.read_parquet(DATA_PROCESSED / "candidates.parquet").sort_values(["user_idx", "rank"])
        self.n_users = int(cand["user_idx"].max()) + 1
        self.cand_items = cand["item_idx"].to_numpy()
        self.cand_scores = cand["hybrid_score"].to_numpy()
        self.cand_offsets = np.searchsorted(cand["user_idx"].to_numpy(), np.arange(self.n_users + 1))

        train = pd.read_parquet(DATA_PROCESSED / "train.parquet", columns=["user_idx", "item_idx", "playcount"])
        train = train.sort_values(["user_idx", "playcount"], ascending=[True, False], kind="stable")
        self.train_items = train["item_idx"].to_numpy()
        self.train_playcounts = train["playcount"].to_numpy()
        self.train_offsets = np.searchsorted(train["user_idx"].to_numpy(), np.arange(self.n_users + 1))
        listeners = np.bincount(self.train_items, minlength=self.n_items).astype(float)
        self.item_pop = np.maximum(listeners, 1.0) / self.n_users

        hold = pd.read_parquet(DATA_PROCESSED / "holdout.parquet", columns=["user_idx", "item_idx"]).sort_values("user_idx")
        self.hold_items = hold["item_idx"].to_numpy()
        self.hold_offsets = np.searchsorted(hold["user_idx"].to_numpy(), np.arange(self.n_users + 1))

        curriculum, _ = curriculum_users()
        self.curriculum = set(int(u) for u in curriculum)

        self._raw_agents = {}
        for label, (ckpt, _) in AGENTS.items():
            import pickle
            with open(models_dir(GROUNDED) / f"{ckpt}_seed{SERVING_SEED}.pkl", "rb") as f:
                self._raw_agents[label] = pickle.load(f)

        self.build(0, 10, "Hybrid (Greedy)")  # warm the environment's lazy module-level caches
        self.load_seconds = time.time() - t0

    # -- lookups -----------------------------------------------------------
    def n_train_tracks(self, user_idx) -> int:
        return int(self.train_offsets[user_idx + 1] - self.train_offsets[user_idx])

    def in_rl_curriculum(self, user_idx) -> bool:
        return int(user_idx) in self.curriculum

    def holdout_of(self, user_idx) -> set:
        return set(self.hold_items[self.hold_offsets[user_idx]:self.hold_offsets[user_idx + 1]].tolist())

    def _display(self, items) -> pd.DataFrame:
        m = self.meta.iloc[list(items)]
        return pd.DataFrame({
            "item_idx": list(items), "track": m["name"].to_numpy(), "artist": m["artist"].to_numpy(),
            "genre": m["genre"].fillna("").to_numpy(), "tags": m["tags"].fillna("").to_numpy(),
            "year": m["year"].to_numpy(), "preview_url": m["spotify_preview_url"].to_numpy(),
        })

    def top_train_tracks(self, user_idx, n=10) -> pd.DataFrame:
        lo = self.train_offsets[user_idx]
        items = self.train_items[lo:lo + n] if self.n_train_tracks(user_idx) else []
        df = self._display(items)
        df["playcount"] = self.train_playcounts[lo:lo + len(items)]
        return df

    # -- inference ---------------------------------------------------------
    def build(self, user_idx: int, length: int, agent_label: str) -> BuiltPlaylist:
        assert MIN_LENGTH <= length <= MAX_LENGTH, f"playlist length must be {MIN_LENGTH}-{MAX_LENGTH}"
        t0 = time.time()
        lo, hi = self.cand_offsets[user_idx], self.cand_offsets[user_idx + 1]
        cand = pd.DataFrame({"item_idx": self.cand_items[lo:hi], "hybrid_score": self.cand_scores[lo:hi]})
        env = PlaylistEnv(int(user_idx), cand, self.content, {}, EnvConfig(playlist_length=length, reward_mode=GROUNDED))
        ckpt = AGENTS[agent_label][0]
        agent = disable_exploration(ckpt, copy.deepcopy(self._raw_agents[agent_label]))  # copy first: the lambda for UCB1 closes over the agent
        run_frozen_episode(env, agent)
        items = list(env.playlist)
        assert len(items) == length and len(set(items)) == length

        tracks = self._display(items)
        tracks.insert(0, "position", np.arange(1, length + 1))
        tracks["hybrid_rank"] = [int(np.flatnonzero(cand["item_idx"].to_numpy() == i)[0]) + 1 for i in items]
        held = self.holdout_of(user_idx)
        tracks["is_heldout"] = [i in held for i in items]

        scores = {
            "diversity": ba.intra_list_diversity(items, self.content),
            "novelty": ba.novelty(items, self.item_pop),
            "artist_entropy": ba.artist_entropy(items, self.artist_of),
            "distinct_artists": int(len(set(self.artist_of[items]))),
            "max_tracks_one_artist": int(pd.Series(self.artist_of[items]).value_counts().max()),
        }
        return BuiltPlaylist(tracks, scores, time.time() - t0, int(tracks["is_heldout"].sum()))
