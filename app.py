"""Streamlit demo — `streamlit run app.py`.

Needs the pipeline artefacts already on disk (data/processed/, models/); it
trains and fits nothing. Models load once (`st.cache_resource`), candidates
are precomputed (data/processed/candidates.parquet), and each playlist is a
single frozen-policy rollout (milliseconds).
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.playlist import AGENTS, MAX_LENGTH, MIN_LENGTH, PlaylistService

st.set_page_config(page_title="Adaptive Hybrid Playlists", page_icon="🎧", layout="wide")


@st.cache_resource(show_spinner="Loading models and candidates (one-off) ...")
def get_service() -> PlaylistService:
    return PlaylistService()


@st.cache_resource
def user_labels(_service: PlaylistService) -> list:
    return [f"user {u:05d}  ·  {_service.n_train_tracks(u)} listened tracks" for u in range(_service.n_users)]


svc = get_service()
labels = user_labels(svc)

st.title("Adaptive Hybrid Music Playlist Recommendation")
st.caption("A hybrid recommender (item-CF + ALS + content + popularity, fused with RRF) proposes ~90 candidates per listener; "
           "a reinforcement-learning agent then builds the playlist one track at a time.")

# ---------------------------------------------------------------- controls
with st.sidebar:
    st.header("Build a playlist")
    if "user_idx" not in st.session_state:
        st.session_state.user_idx = 20
    if st.button("🎲 Random listener", use_container_width=True):
        st.session_state.user_idx = random.randrange(svc.n_users)
    user_idx = st.selectbox("Listener (type to search, e.g. `user 01234`)", options=range(svc.n_users),
                            format_func=labels.__getitem__, key="user_idx")
    length = st.slider("Playlist length", MIN_LENGTH, MAX_LENGTH, 15)
    agent_label = st.selectbox("Agent", list(AGENTS), help="Who chooses the next track at each step.")
    st.caption(AGENTS[agent_label][1])
    reveal = st.checkbox("Reveal held-out tracks (evaluation view)", value=False,
                         help="Each listener has 5 tracks hidden from every model. Ticking this marks which of them the "
                              "playlist happened to recover. Nothing here is visible to the agents.")

built = svc.build(user_idx, length, agent_label)
tracks = built.tracks

# ---------------------------------------------------------------- playlist
main, side = st.columns([2.2, 1], gap="large")
with main:
    st.subheader(f"{agent_label} playlist for user {user_idx:05d}")
    if svc.in_rl_curriculum(user_idx):
        st.caption("ℹ️ This listener was in the RL agents' training curriculum.")
    st.caption(f"Built in {built.seconds * 1000:.0f} ms — one frozen-policy rollout, no training at request time.")
    for _, r in tracks.iterrows():
        with st.container(border=True):
            c_num, c_main, c_meta, c_audio = st.columns([0.5, 3.2, 2.6, 2.6])
            c_num.markdown(f"### {r['position']}")
            badge = "  ✅ *held-out*" if reveal and r["is_heldout"] else ""
            c_main.markdown(f"**{r['track']}**{badge}  \n{r['artist']}")
            year = f"{int(r['year'])}" if pd.notna(r["year"]) else "year n/a"
            genre = r["genre"] or "genre n/a"
            tags = ", ".join(r["tags"].split(",")[:5]) if r["tags"] else "no tags"
            c_meta.markdown(f"{genre} · {year}  \n<span style='opacity:.7'>{tags}</span>", unsafe_allow_html=True)
            if isinstance(r["preview_url"], str) and r["preview_url"]:
                c_audio.audio(r["preview_url"])
            else:
                c_audio.caption("no preview available")
    if reveal:
        st.success(f"This playlist recovers **{built.heldout_hits} of the listener's 5 held-out tracks**.")

# ---------------------------------------------------------------- side panel
with side:
    st.subheader("Playlist scores")
    s = built.scores
    m1, m2 = st.columns(2)
    m1.metric("Diversity", f"{s['diversity']:.2f}", help="1 − mean pairwise content cosine. Higher = more varied.")
    m2.metric("Novelty", f"{s['novelty']:.2f}", help="Mean −log₂(share of listeners who played the track). Higher = rarer tracks.")
    m3, m4 = st.columns(2)
    m3.metric("Distinct artists", f"{s['distinct_artists']}/{length}")
    m4.metric("Most from one artist", s["max_tracks_one_artist"], help="The artist cap of 2 is a soft penalty in the reward, not a hard filter.")

    st.subheader("What they listen to")
    st.caption("Top 10 training tracks by play count — a sanity check on the recommendations.")
    top = svc.top_train_tracks(user_idx)
    for _, r in top.iterrows():
        st.markdown(f"**{r['track']}** — {r['artist']}  \n<span style='opacity:.7'>{r['playcount']} plays</span>", unsafe_allow_html=True)

    with st.expander("How much to trust this"):
        st.markdown(
            "- In our offline evaluation (5 held-out tracks per listener) the plain hybrid ranking — **Hybrid (Greedy)** — "
            "recovered the most held-out tracks; none of the RL agents beat it.\n"
            "- The dataset has no timestamps or skips, so the sequence is the *playlist the agent builds*, not an observed listening order.\n"
            "- Previews come from Spotify URLs stored in the dataset; some may no longer resolve.\n"
            "- Full numbers and caveats: `report/FINDINGS.md`."
        )
