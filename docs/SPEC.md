# Adaptive Hybrid Music Playlist Recommendation using Reinforcement Learning
## Technical Review + Build Specification

This document is both (a) a critical review of the proposed plan and (b) a build
spec that can be handed directly to Claude Code as a project prompt.

---

# PART A — VERIFIED DATASET FACTS

All numbers below were computed from the actual uploaded files. Do not re-derive
them from assumptions; re-verify them in Phase 0 and print them into the report.

## A.1 `Music_Info.csv`

| Property | Value |
|---|---|
| Rows | 50,683 |
| Columns | 21 |
| Unique `track_id` | 50,683 (no duplicates) |
| Unique `name` | 50,683 — **every title is globally unique** (source was deduped on title) |
| Unique `artist` | 8,317 (median 2 tracks/artist, p90 = 16) |
| Unique `spotify_id` | 50,674 → **9 duplicate pairs** |
| `genre` missing | 28,335 (55.9%), only 15 distinct values |
| `tags` missing | 1,127 (2.2%) |
| Neither tag nor genre | 695 tracks |
| `tags` vocabulary | exactly 100 controlled terms, every one appears ≥50 times |
| Tags per track | mean 3.66, median 3, max 52 |
| `year` | 1900–2022, median 2006, 8 rows use 1900 as a null placeholder |

**Audio feature anomalies**

- `tempo == 0` on 10 tracks; `time_signature == 0` on the same 10 → invalid rows.
- `duration_ms`: min 1,439 ms (1.4 s), max 3,816,373 ms (63 min); 575 tracks > 10 min.
- Zero-inflation: `instrumentalness < 0.01` on 52.8% of tracks, `acousticness < 0.01` on 38.4%.
- Skew: `speechiness` 3.86, `liveness` 2.03, `loudness` −1.70.

**Feature redundancy (Pearson)**

- `energy` ↔ `loudness` = **+0.78**
- `energy` ↔ `acousticness` = **−0.73**
- `loudness` ↔ `acousticness` = −0.61
- `danceability` ↔ `valence` = +0.54

**Genre label reliability** — for tracks that *have* a genre, how often does the
genre string appear among that track's own tags?

| Genre | Agreement | n |
|---|---|---|
| New Age | 0.814 | 237 |
| Punk | 0.640 | 383 |
| Country | 0.629 | 607 |
| Folk | 0.611 | 355 |
| Electronic | 0.578 | 3,710 |
| Pop | 0.547 | 1,145 |
| Blues | 0.540 | 189 |
| Reggae | 0.507 | 691 |
| Rap | 0.481 | 821 |
| Jazz | 0.441 | 793 |
| Rock | 0.437 | 9,965 |
| Metal | 0.334 | 2,516 |
| RnB | 0.270 | 696 |
| **World** | **0.000** | 140 |
| **Latin** | **0.000** | 100 |
| **Overall** | **0.465** | 22,348 |

Spot-checked examples of mislabelling: Nirvana's *Come as You Are* → `RnB`;
Radiohead's *Creep* → `RnB`; Placebo → `Rap`.

**Duplicate track pairs (same `spotify_id`, different `track_id`)** — must be
merged or they will both surface in one playlist:

```
Samuel Barber       Adagio for Strings, Op. 11 / Adagio For Strings
2Pac                How Do U Want It / How Do You Want It
Amadou & Mariam     Je pense à toi / Je Pense A Toi
The Specials        Too Much Too Young (Live) / Too Much Too Young
Radiohead           There, There / There There
Broken Social Scene hHallmark / Hallmark
The Breeders        Do You Love Me Now? / Do You Love Me Now
Annie               Greatest Hit / The Greatest Hit
Matchbox Twenty     3 AM / 3AM
```

## A.2 `User Listening History.csv`

| Property | Value |
|---|---|
| Rows | **9,711,301** |
| Columns | `track_id`, `user_id`, `playcount` |
| Nulls | 0 |
| Exact duplicate rows | 0 |
| Duplicate `(user_id, track_id)` pairs | 0 |
| Unique users | **962,037** |
| Unique tracks | **30,459** |
| `playcount` range | 1 – 2,948 (mean 2.63, median 1) |
| `playcount == 1` | 60.8% of all rows |
| `playcount >= 10` | 4.2% of all rows |
| Matrix density | **0.0331%** (sparsity 99.9669%) |
| `user_id` format | 40-char SHA-1 hex (Echo Nest Taste Profile) |
| `track_id` format | Million Song Dataset IDs (`TR…`) |
| **Timestamps** | **NONE** |

**Interactions per user** — mean 10.09, median 5.

| Threshold | Users | % |
|---|---|---|
| ≥5 tracks | 535,041 | 55.6% |
| ≥10 tracks | 290,898 | 30.2% |
| ≥15 tracks | 185,418 | 19.3% |
| ≥20 tracks | 127,174 | 13.2% |
| ≥30 tracks | 66,766 | 6.9% |
| ≥50 tracks | 23,795 | 2.5% |

**Listeners per track** — mean 318.8, median 57, max 80,656.
27,049 tracks have ≥5 listeners; 12,398 have ≥100.

## A.3 Join analysis (this is Section 8 of the plan, now answered)

```
tracks in listening history        : 30,459
tracks in Music_Info               : 50,683
intersection                       : 30,459
history tracks missing metadata    : 0        ← perfect, 100% of rows joinable
Music_Info tracks with 0 listens   : 20,224   ← 39.9% of the catalogue is dead weight
```

**Both directions matter and they are asymmetric.** Every interaction can be
enriched with audio features — excellent. But two-fifths of the music catalogue
has no interaction data at all, which means those 20,224 tracks are invisible to
collaborative filtering and can only ever be reached content-side.

---

# PART B — WHAT THE DATA CHANGES IN YOUR PLAN

## 🔴 NECESSARY CHANGES

### B.1 The problem statement's "sequential" claim is currently false

There are **no timestamps and no session boundaries**. `playcount` is an
aggregate over an unknown window. You cannot observe the order in which a user
played anything, so you cannot learn from real listening sequences, and you
cannot do a temporal train/test split.

This does not kill the project — but the framing must be corrected, or a viva
examiner will find it in thirty seconds.

**Correct framing:** the MDP is defined over the **playlist the agent
constructs**, not over the user's observed listening order. State transitions
come from the agent's own previous actions (what it has already placed in the
playlist), which is genuine sequential decision-making with genuine credit
assignment — a track chosen at position 3 constrains what is good at position 7
through diversity and artist-repetition terms. Say exactly this in the report.

Delete from the problem statement: *"the selection of one song may influence the
quality and relevance of subsequent songs"* framed as an observed user
phenomenon. Replace with the playlist-construction framing.

### B.2 SVD / SVD++ must be dropped — there are no ratings

`playcount` is implicit feedback with a hard floor at 1. There is no negative
signal and no rating scale. SVD++ is defined for explicit ratings with observed
negatives; applying it here and reporting RMSE is the single most common error in
this project genre.

**Replace with:** ALS with confidence weighting (Hu, Koren & Volinsky, 2008) —
this *is* the "matrix factorization / latent factors / regularization / implicit
feedback" content from your syllabus, done correctly. BPR (Rendle et al., 2009)
as a pairwise-ranking contrast is **deferred** — add it only as an optional
experiment (Part F) after the core system is complete and evaluated.

Use the `implicit` library. Do not hand-roll ALS.

### B.3 Drop user-based collaborative filtering

962,037 users. A user–user similarity matrix is 962,037² — roughly 3.7 TB in
float32. Even after filtering it stays wasteful, and it teaches you nothing that
item–item doesn't.

**Keep item–item CF only.** After filtering (see B.4) the item–item matrix is
12,193² ≈ 0.59 GB in float32 — comfortable in RAM. Item–item also covers the
"neighborhood methods" syllabus topic fully.

### B.4 You must apply iterative k-core filtering before anything else

The raw matrix is 962,037 × 30,459 at 0.0331% density. Median user has 5
interactions — too few to build a profile, evaluate a hold-out, or run an
episode. Median track has 57 listeners, but the tail is unusable.

Tested thresholds:

| Filter | Interactions | Users | Tracks | Density | Item–item RAM |
|---|---|---|---|---|---|
| user≥10, track≥20 | 6,958,707 | 287,456 | 18,989 | 0.127% | 1.44 GB |
| user≥15, track≥30 | 5,623,673 | 180,120 | 15,787 | 0.198% | 1.00 GB |
| **user≥20, track≥50** | **4,446,456** | **118,153** | **12,193** | **0.309%** | **0.59 GB** |
| user≥25, track≥50 | 3,601,426 | 81,722 | 11,146 | 0.395% | 0.50 GB |
| user≥30, track≥100 | 2,248,735 | 45,648 | 5,835 | 0.844% | 0.14 GB |

**Use `user≥20, track≥50` applied iteratively until stable.** Rationale: ≥20
interactions per user allows a 15/5 train/hold-out split with a meaningful
profile; ≥50 listeners per track gives item–item CF enough co-occurrence signal;
12,193 tracks × 2,863 artists is plenty for artist-diversity constraints; and
98.8% of those tracks still carry tags.

Then **subsample to 30,000–50,000 users** for RL experiments. The full 118k core
is fine for training CF/ALS, but RL episodes over 118k users is wasted compute.
Fix and report the random seed.

**Important:** the `≥50` track threshold above is tuned against the full
118,153-user population. Do not re-apply it verbatim after subsampling to
40,000 users — it is disproportionately strict against a population with
roughly 1/3 the listeners per track, and iterative k-core cascades that into
losing ~49% of the catalogue. Phase 1 instead re-derives the post-subsample
threshold (`≥17`) empirically; see Phase 1, task 5.

### B.5 `genre` cannot be a primary content feature

55.9% missing, only 46.5% agreement with the track's own tags, and 0% agreement
for World and Latin. It is noisy and mostly absent.

**Use `tags` as the primary semantic signal.** The tag vocabulary is a clean,
capped set of exactly 100 terms with 97.8% coverage (98.8% in the filtered core),
every term appearing at least 50 times. TF-IDF over that vocabulary is a strong,
honest content representation.

Keep `genre` only as a low-weight (≤0.10) one-hot block with `UNK` for missing,
or drop it entirely. Report the ablation either way — a "we tested it and it
hurt" result is worth more marks than silently including it.

### B.6 DQN is not necessary — your instinct is correct

Justification to state in the viva, not just "it was too complex":

1. **The environment is fully known and deterministic.** Transitions are the
   agent's own playlist edits. There is no unknown stochastic dynamics model to
   approximate.
2. **The horizon is short** (playlist length 10–20). Deep credit assignment
   across long horizons is what deep RL buys you; you don't need it.
3. **Function approximation is only needed if the state space is too large to
   tabulate.** With the two-level design in B.7 it isn't.
4. **DQN on ~100 discrete actions with sparse offline reward will be unstable**
   and you will spend the semester debugging replay buffers instead of
   demonstrating RL understanding.

State what *would* justify DQN: continuous or very high-dimensional state, an
action space in the thousands, or a learned user-simulator with stochastic
transitions. Offer it as declared future work.

### B.7 Your tabular Q-Learning / SARSA action space needs restructuring

This is the subtle one. If the action is "pick one specific song from a
100-candidate set", then |A| = 100 and the action identity changes meaning for
every user and every step — a tabular Q-table over that is meaningless, because
`Q(s, song_47)` doesn't transfer across users.

**Fix — two-level policy:**

- **Level 1 (RL, tabular):** the agent picks an **arm**, not a song. Cluster the
  full catalogue into K = 8–12 buckets (KMeans on the normalised audio-feature +
  tag vector) **once, globally** — not refit per candidate set, so arm
  identity is stable across every user and episode (required for
  EpsilonGreedyMAB/UCB1/Q-table learning to accumulate anything across
  episodes; see Phase 5/6). The action is "draw the next track from bucket
  k". |A| = K = 10.
- **Level 2 (deterministic):** within the chosen bucket, take the
  highest-hybrid-score track not yet used. The artist cap is enforced as a
  reward penalty (B.12's `w_rep` term), not as a selection filter here — see
  Phase 5 for why it must stay a cost the agent can choose to pay rather
  than a wall it cannot reach.

State (discretised, transferable across users):

```
s = ( energy_bucket(recent 3 tracks)      ∈ {low, mid, high}          3
    , valence_bucket(recent 3 tracks)     ∈ {low, mid, high}          3
    , diversity_level(playlist so far)    ∈ {low, mid, high}          3
    , artist_repeat_flag                  ∈ {clear, near_cap}         2
    , playlist_position_bucket            ∈ {early, mid, late}        3
    )
```

|S| = 3·3·3·2·3 = **162 states**, |A| = 10 → a 162 × 10 Q-table. Tabular
Q-Learning and SARSA will actually converge on this, and the on-policy vs
off-policy comparison becomes a real, visible result.

Run LinUCB at the *fine-grained song level* in parallel — it handles the
high-dimensional contextual case that the tabular agent cannot. That contrast is
itself a result worth reporting.

### B.8 Merge the 9 duplicate track pairs

Same `spotify_id`, different `track_id`. Merge before building the interaction
matrix — sum playcounts, keep the more-listened `track_id` as canonical. Without
this, "There, There" and "There There" can both appear in one demo playlist.

### B.9 Clean the audio-feature anomalies

- Drop or impute the 10 tracks with `tempo == 0` and `time_signature == 0`.
- Set `year == 1900` to null (8 rows), bucket year into decades rather than using it raw.
- Winsorise `duration_ms` at the 1st/99th percentile, or drop tracks < 60 s and > 10 min from the recommendable pool (575 + 2 tracks).
- **Drop `loudness`** (r = 0.78 with `energy` — redundant), and drop `key`, `mode`, `time_signature` (categorical, no meaningful contribution to similarity).

Final content feature set: `danceability`, `energy`, `speechiness`,
`acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`,
`duration_log`, `decade` + 100-dim TF-IDF tags.

### B.10 Use `QuantileTransformer`, not `StandardScaler`

`speechiness` (skew 3.86), `liveness` (2.03), `instrumentalness` (52.8% near
zero), `acousticness` (38.4% near zero). Z-scoring a zero-inflated,
heavily-skewed variable produces a similarity space dominated by outliers.
`QuantileTransformer(output_distribution='normal')` fixes this in one line.

---

## 🟡 RECOMMENDED IMPROVEMENTS

### B.11 Playcount → preference conversion

Do **not** min-max scale playcount into a pseudo-rating. Raw skew is 46.31; after
`log1p` it drops to 2.11.

Use the Hu–Koren–Volinsky split into *preference* and *confidence*:

```
p_ui = 1                                (binary: interaction observed)
c_ui = 1 + α · log(1 + playcount_ui / ε)     α = 40, ε = 1e-8
```

For the reward and for ranking features, additionally compute a **per-user
normalised affinity** so heavy users don't dominate:

```
aff_ui = log1p(playcount_ui) / max_i( log1p(playcount_ui) )     ∈ (0, 1]
```

`aff_ui` is what feeds the RL relevance reward. `c_ui` is what feeds ALS.

### B.12 Reward design — the honest answer

You asked not to invent unrealistic feedback. Here is what the data actually
permits, in descending order of defensibility.

**Tier 1 — grounded, no invention (use this as the core).**
Hold out 5 tracks per user (`H_u`). Reward for recommending track `a`:

```
rel(u, a) = aff_ua        if a ∈ H_u        (grounded in real playcount)
          = 0             otherwise
```

This is *replay evaluation* (Li et al., 2011). Everything about it is real. Its
weakness is extreme sparsity — 5 hits in a 100-candidate set — which makes RL
learning curves noisy.

**Tier 2 — declared simulator (use this to densify, and label it as such).**
For `a ∉ H_u`, back off to a content-based proxy:

```
rel(u, a) = λ · cos( user_profile_vector_u , content_vector_a )      λ = 0.3
```

where `user_profile_vector_u` is the affinity-weighted mean of the content
vectors of the user's **training** tracks only. This is a **simulated user
model**, and you must say so in the report — "we assume a user's affinity for an
unseen track is proportional to its content similarity to their listening
profile; this is a modelling assumption, not observed feedback." Declared
honestly, it is completely academically valid. Undeclared, it is fabrication.

**Full step reward:**

```
r_t = w_rel · rel(u, a_t)
    + w_div · ( 1 − max_{j ∈ playlist_{<t}} cos(content_a_t, content_j) )
    − w_rep · 1[ artist(a_t) already appears ≥ 2 times in playlist_{<t} ]
    − w_dup · 1[ a_t already in playlist_{<t} ]

w_rel = 1.0,  w_div = 0.3,  w_rep = 0.5,  w_dup = 1.0
```

The diversity and repetition terms are what make this a genuine MDP rather than a
bandit — they depend on the agent's own history, so a myopic choice at step 3 is
punished at step 7.

The `w_div` sensitivity sweep (0.0 / 0.15 / 0.3 / 0.5) — a cheap experiment that
produces a real accuracy-vs-diversity trade-off curve, exactly the
"beyond-accuracy evaluation" your RS syllabus asks for — is **deferred** to
Part F, run only after the core system is complete and evaluated.

**What you must state as a limitation:** there is no skip data, no dwell time, no
explicit negatives, and no timestamps. Do not simulate skips. Say so.

### B.13 Candidate set size

Candidate generation should reduce 12,193 tracks to **N = 100** per user, per
episode. Rationale: 100 is small enough for LinUCB to explore within a
20-step episode, large enough to contain most of the 5 held-out tracks
(measure and report **candidate recall@100** — if it is below ~0.4 the RL layer
is being starved and you should raise N to 200).

Composition: 50 from item–item CF, 30 from ALS, 15 from content, 5 from
popularity — then dedupe and re-score with the hybrid weights.

### B.14 Hybrid weighting and score normalisation

Different recommenders produce scores on incomparable scales (cosine ∈ [0,1],
ALS dot products unbounded, popularity counts in the thousands).

**Normalise by rank, not by value.** For each recommender, convert its top-N list
to a reciprocal-rank score `1/(60 + rank)` and sum — this is Reciprocal Rank
Fusion, it is scale-free, and it is far more robust than min-max blending. Report
it as your primary hybrid.

The weighted linear hybrid you proposed (α·CF + β·content + γ·popularity,
α+β+γ=1) with per-recommender min-max normalisation and weights tuned by grid
search on a validation split is **deferred** — build it as an optional
experiment (Part F) once RRF is validated as the core hybrid, not in the first
implementation.

Expected outcome to check: popularity weight should come out small. The top 10%
of tracks already hold 47.0% of interactions, so a popularity-heavy hybrid will
score well on Precision@K and terribly on novelty and coverage. That tension is a
finding, not a bug — put it in the report.

### B.15 Evaluation protocol

**Split.** Per-user leave-5-out, random (not temporal — no timestamps exist).
Only over users with ≥20 interactions (guaranteed by the k-core). Fix the seed.
State explicitly in the report that a temporal split is impossible with this data
and why.

**Valid metrics** (K = 10 and K = 20):

| Family | Metrics |
|---|---|
| Accuracy | Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, HitRate@K |
| Beyond-accuracy | Catalog Coverage, Intra-List Diversity (1 − mean pairwise content cosine), Novelty (−log₂ item popularity), Artist Entropy, Gini index |
| RL-specific | Mean cumulative episode reward, learning curve with 95% CI over seeds, cumulative regret vs a greedy oracle, action-distribution entropy |

**Invalid — do not use:** RMSE, MAE, or any rating-prediction metric. There are
no ratings. You were right to flag this.

**Baselines to beat, in order:** Random → Popularity → Content-only →
ItemCF-only → ALS-only → Hybrid → Hybrid + Bandit → Hybrid + RL.
Run ≥5 seeds and report mean ± std. A single-seed RL result is not evidence.

### B.16 Context-aware extension — deferred to optional experiments

Not part of the first implementation. Once the core system (Popularity,
Content-based, Item-CF, ALS, Hybrid-RRF, and the RL agents) is complete and
evaluated, add this as an optional experiment (Part F): derive a **mood
quadrant** from `valence` × `energy` (happy / energetic / calm / sad) and
expose it in the Streamlit UI as a **candidate-set filter**. That is textbook
*contextual pre-filtering* — name it as such in the report. Do not build a
contextual modelling approach; do not invent time-of-day or activity data you
do not have.

---

## 🟢 GOOD AS PLANNED — KEEP UNCHANGED

- **The overall two-stage architecture** (traditional recommenders generate
  candidates → RL selects/orders). This is the correct and standard industrial
  pattern, and it is exactly the right way to make the RS and RL components both
  load-bearing rather than bolted together.
- **Refusing to use RMSE/MAE.** Correct.
- **Refusing DQN by default.** Correct, and now you have the technical
  justification in B.6.
- **Popularity as a baseline.** Keep it. With 47% of interactions in the top 10%
  of tracks, it will be a genuinely hard baseline and beating it will mean
  something.
- **Content-based filtering with cosine similarity on audio features + tags.**
  Verified working — a spot-check on the real data returned Meat Puppets, Alice
  in Chains and Foo Fighters for *Come as You Are*, and Kings of Leon for
  *Mr. Brightside*. The signal is real.
- **Streamlit UI last.** Correct sequencing.
- **Comparing 5 model configurations.** Keep it; once the pipeline exists each
  additional row of the results table is nearly free.

### Title

Use: **"Adaptive Hybrid Music Playlist Recommendation using Reinforcement
Learning"**

Your option 1 minus the emoji. Option 3 is redundant ("Reinforcement
Learning-Based … Using Reinforcement Learning" reads awkwardly once expanded).
Option 2 is accurate but long. If the department wants the sequential aspect
foregrounded: *"Sequential Playlist Generation using a Hybrid Recommender and
Reinforcement Learning"*.

---

# PART C — FINAL ARCHITECTURE

```
  Music_Info.csv (50,683)          User Listening History.csv (9.71M)
         │                                      │
         └──────────────┬───────────────────────┘
                        │  join on track_id (100% of interactions covered)
                        ▼
        ┌───────────────────────────────────────┐
        │  PHASE 0-1  PREPROCESSING             │
        │  • merge 9 duplicate track pairs      │
        │  • drop tempo=0 / bad duration        │
        │  • iterative k-core (u≥20, t≥50)      │
        │    → 118,153 users × 12,193 tracks    │
        │  • subsample 40,000 users (seeded)    │
        │  • playcount → p_ui, c_ui, aff_ui     │
        │  • QuantileTransform audio feats      │
        │  • TF-IDF 100-term tag vocabulary     │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  ITEM & USER REPRESENTATION           │
        │  content_vec: 10 audio + 100 tags     │
        │  user_profile: aff-weighted mean of   │
        │                TRAIN tracks only      │
        └───────────────────┬───────────────────┘
                            ▼
        ┌──────────┬────────┴────────┬──────────┐
        ▼          ▼                 ▼          ▼
   Popularity  Content-Based    Item-Item CF   ALS
   (baseline)  (cosine)         (cosine on     (implicit,
                                 co-occurrence) confidence-
                                                weighted)
        └──────────┴────────┬────────┴──────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  HYBRID CANDIDATE GENERATOR           │
        │  Reciprocal Rank Fusion (core)        │
        │  + weighted linear (deferred, Part F) │
        │  → top-100 candidates per user        │
        │  report candidate recall@100          │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  RL LAYER                             │
        │  cluster candidates → K=10 arms       │
        │                                       │
        │  Agents (all compared):               │
        │   • Random / Greedy      (controls)   │
        │   • ε-greedy MAB         (RL course)  │
        │   • UCB1                 (RL course)  │
        │   • LinUCB               (contextual, │
        │                       song-level)     │
        │   • Q-Learning (tabular, 162×10)      │
        │   • SARSA      (tabular, 162×10)      │
        │                                       │
        │  reward = relevance + diversity       │
        │           − artist repeat − duplicate │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  PLAYLIST CONSTRUCTION (len 10-20)    │
        │  artist cap ≤2, no duplicates         │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  EVALUATION + STREAMLIT UI            │
        └───────────────────────────────────────┘
```

## Final algorithm selection

| Component | Decision |
|---|---|
| Baseline | Popularity (play-count weighted) |
| Content-based | QuantileTransformed audio (9 feats) + TF-IDF tags (100), cosine |
| Neighborhood CF | **Item–item only**, cosine on the confidence matrix, top-50 neighbours |
| Model-based CF | **ALS with confidence weighting** (`implicit`), factors=64, reg=0.05, α=40. BPR — **deferred**, optional experiment (Part F). |
| Dropped | User-based CF, SVD, SVD++, NCF, DQN (documented only, see Part F) |
| Hybrid | Reciprocal Rank Fusion (core). Grid-searched weighted linear — **deferred**, optional experiment (Part F). |
| RL primary | **LinUCB** contextual bandit (song-level) |
| RL sequential | **Tabular Q-Learning vs SARSA** over 162 states × 10 arms |
| RL baselines | ε-greedy MAB, UCB1, Random, Greedy |
| Reward | `w_rel·rel + w_div·div − w_rep·repeat − w_dup·dup` |
| Evaluation | P/R/F1/NDCG/MAP/HR @10,20 + Coverage/ILD/Novelty/Gini + RL curves |
| Context | Mood quadrant candidate pre-filter — **deferred**, optional experiment (Part F). |

---

# PART D — DEVELOPMENT ROADMAP / CLAUDE CODE BUILD PROMPT

> Everything below can be pasted into Claude Code as the project brief.
> Work phase by phase. Do not start a phase until the previous phase's
> acceptance criteria pass.

**First-implementation scope (Phases 0–9):** Popularity, Content-Based,
Item-Based CF, Implicit ALS, Hybrid-RRF, and the RL agents Random, Greedy,
ε-greedy MAB, UCB1, LinUCB, Q-Learning, SARSA. BPR, DQN, the weighted-linear
hybrid with grid-searched α/β/γ, the context-aware mood extension, and the
`w_div` sensitivity sweep are **deferred** to Part F and are only attempted
after the core system above is complete and evaluated.

## Global conventions

- Python 3.11. Deps: `pandas`, `numpy`, `scipy`, `scikit-learn`, `implicit`,
  `matplotlib`, `seaborn`, `streamlit`, `pyarrow`, `tqdm`.
- All randomness seeded with `SEED = 42`, set in `src/config.py`.
- Intermediate artefacts are Parquet in `data/processed/`, never re-derived.
- Every module has a `if __name__ == "__main__"` entry point and writes its
  outputs to disk. No notebook-only code.
- Every experiment writes a row to `results/metrics.csv` with the config hash.
- Do not use `localStorage`. Do not use deep learning. Do not use SVD++.

```
project/
├── config.py
├── data/
│   ├── raw/            Music_Info.csv, User Listening History.csv
│   └── processed/
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── recommenders/
│   │   ├── popularity.py
│   │   ├── content.py
│   │   ├── item_cf.py
│   │   ├── als_model.py
│   │   └── hybrid.py
│   ├── rl/
│   │   ├── environment.py
│   │   ├── agents.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── ranking_metrics.py
│   │   └── beyond_accuracy.py
│   └── playlist.py
├── results/
├── app.py
└── report/
```

---

### PHASE 0 — Verification & EDA

**Goal.** Independently reproduce every number in Part A before building on them.

**Tasks.**
1. Load both CSVs (note `Music_Info.csv` has a UTF-8 BOM — use `encoding='utf-8-sig'`).
2. Reproduce: 50,683 tracks; 9,711,301 interactions; 962,037 users; 30,459
   distinct tracks in history; 0 unjoinable interactions; 20,224 orphan catalogue
   tracks; 0.0331% density.
3. Plot: playcount distribution (log-log), interactions-per-user CDF,
   listeners-per-track CDF, audio-feature histograms, feature correlation
   heatmap, tag frequency top-40, genre missingness.
4. Recompute the genre-vs-tags agreement table (expect overall ≈ 0.465).
5. Identify the 9 duplicate `spotify_id` pairs programmatically.

**Output.** `results/eda/` with figures + `results/eda/dataset_stats.json`.

**Acceptance.** Every figure regenerates from a clean checkout. Stats match
Part A. If any number differs, stop and investigate before proceeding.

**Dependencies.** None.

---

### PHASE 1 — Preprocessing & filtering

**Goal.** Produce the clean, filtered interaction matrix and item catalogue.

**Tasks.**
1. Merge the 9 duplicate track pairs (sum playcounts, keep the higher-listener `track_id`).
2. Drop tracks with `tempo == 0`, `duration_ms < 60000`, `duration_ms > 600000`.
3. Set `year == 1900` → null; add `decade` bucket.
4. Iterative k-core: repeat {drop tracks with <50 listeners; drop users with <20
   tracks} until fixed point. Expect ≈ 4.45M rows / 118,153 users / 12,193 tracks.
5. Seeded random subsample to 40,000 users, then **re-run the full iterative
   k-core to convergence** (user≥20, track≥17 — not a single re-application of
   the population-scale ≥50 threshold). Expect **39,825 users / 11,613 tracks /
   1,489,288 interactions / 0.322% density / 2,760 distinct artists**.

   **Why ≥17, not ≥50, post-subsample:** ≥50 was tuned against the full
   118,153-user population (B.4). Re-applying it to a 40,000-user subsample is
   disproportionately strict — the subsample already has ~1/3 the listeners
   per track, so a fixed absolute threshold cascades through the iterative
   filter and costs ~49% of the catalogue (6,107 of 11,944 tracks), leaving a
   denser but much smaller, less diverse core (1,544 artists). ≥17 rescales
   the threshold to the subsampled population instead of the full one; it
   removes only ~3% of tracks and keeps artist diversity (2,760) and coverage
   intact while still enforcing the same k-core invariant (user≥20 track≥17)
   to convergence.
6. Compute `p_ui`, `c_ui` (α=40), `aff_ui` per B.11.
7. Build integer index maps `user_idx`, `item_idx`; save as Parquet.
8. Leave-5-out split per user → `train.parquet`, `holdout.parquet`.
9. Build `scipy.sparse.csr_matrix` for train confidence; save as `.npz`.

**Output.** `data/processed/{interactions,train,holdout,items,idx_maps}.parquet`,
`data/processed/train_csr.npz`.

**Acceptance.** No user in train has <15 interactions. Every holdout user has
exactly 5 holdout tracks. No leakage: `train ∩ holdout = ∅`. Print the final
shape and density.

**Dependencies.** Phase 0.

---

### PHASE 2 — Content features

**Goal.** A single dense item-content matrix and per-user profile vectors.

**Tasks.**
1. `QuantileTransformer(output_distribution='normal')` on the 9 retained audio
   features (`danceability, energy, speechiness, acousticness, instrumentalness,
   liveness, valence, tempo, duration_log`). **`loudness`, `key`, `mode`,
   `time_signature` are excluded.**
2. TF-IDF on `tags` with a fixed 100-term vocabulary, comma tokeniser, no
   lowercasing beyond what exists.
3. L2-normalise each block separately, then hstack with weights
   `audio 0.45 / tags 0.55`. (Optional low-weight genre one-hot ≤0.10 — ablate it.)
4. L2-normalise the concatenation.
5. `user_profile[u] = normalize( Σ_{i ∈ train_u} aff_ui · content[i] )` —
   **train tracks only**, never holdout.

**Output.** `data/processed/content_matrix.npz`, `user_profiles.npz`.

**Acceptance.** Run the qualitative check: nearest neighbours of *Come as You
Are*, *Mr. Brightside*, *Creep* must be same-genre, same-era tracks. If they are
not, the weighting is wrong. Assert no holdout track contributed to any profile.

**Dependencies.** Phase 1.

---

### PHASE 3 — Individual recommenders

**Goal.** Four working top-N recommenders with a shared interface
`recommend(user_idx, N, exclude_seen=True) -> List[(item_idx, score)]`.

**Tasks.**
1. `popularity.py` — global top-N by distinct listeners and by total playcount.
2. `content.py` — cosine(`user_profile[u]`, `content_matrix`), chunked.
3. `item_cf.py` — cosine item–item on the train CSR, **keep only top-5
   neighbours per item** (chunked `argpartition`, never materialise 12,193²
   densely at once), then score `Σ_{j ∈ train_u} sim(i,j) · aff_uj`.
   k=5 is a swept choice, not an arbitrary default: a sweep over
   k ∈ {2,5,10,15,20,30,40,50} (`results/phase3/item_cf_k_sweep.csv`) shows
   Recall@20, NDCG@20, and Coverage@20 all peak at k=5 and fall monotonically
   for k>5 — coverage *rising* as k falls rules out popularity concentration
   as the explanation. Report this in Phase 9 as a **sensitivity analysis**,
   not as tuning: the sweep was run on the same holdout used for acceptance
   evaluation, so a separate validation split would have been the
   methodologically rigorous way to select k.
4. `als_model.py` — `implicit.als.AlternatingLeastSquares(factors=64,
   regularization=0.05, alpha=40, iterations=20)`. Grid `factors ∈ {32,64,128}`,
   `reg ∈ {0.01,0.05,0.1}` on a validation slice.

**Output.** Fitted models in `models/`, each producing a top-100 list per user.

**Acceptance.** Each model beats Random on Recall@20. ALS beats Popularity on
NDCG@20. Item-CF beats Content on Recall@20 (if not, the CF is broken).

**Dependencies.** Phase 2.

---

### PHASE 4 — Hybrid candidate generator

**Goal.** Top-100 candidates per user that the RL layer can work with.

**Tasks.**
1. Implement Reciprocal Rank Fusion, `score = Σ_m w_m / (60 + rank_m)` — the
   sole hybrid strategy for the core system. `w_m` is each recommender's solo
   NDCG@20 on the Phase 3 acceptance holdout, normalised to sum to 1 (ItemCF
   0.3345, ALS 0.1740, Content 0.0454, Popularity 0.0331 at the time of
   writing). This is *not* the deferred weighted linear hybrid below — it
   reweights the RRF rank terms by a fixed, already-measured quantity, no
   grid search, no per-recommender score normalisation. Plain unweighted RRF
   (`w_m=1`) was tried first and failed the original NDCG-based acceptance
   criterion; see the Phase 9 finding below and
   `results/phase4/fusion_comparison.csv`.
2. Pool: 50 ItemCF + 30 ALS + 15 Content + 5 Popularity → dedupe → re-rank → top-100.
3. **Measure and report candidate recall@100.** If < 0.40, raise N to 200 and
   document the change.

*(The weighted **linear** hybrid — α·CF + β·content + γ·popularity with
per-recommender min-max score normalisation and α/β/γ grid-searched on a
validation split — is a separate, still-deferred experiment; see Part F. Do
not implement it in this phase.)*

**Output.** `data/processed/candidates.parquet` (user_idx, item_idx, hybrid_score, rank).

**Acceptance.** Candidate recall@100 ≥ 0.40 only.

Originally this phase also required Hybrid (RRF) NDCG@20 ≥ best single
model. That criterion assumed the pooled recommenders would be roughly
comparable in strength; on this dataset ItemCF dominates ALS/Content/
Popularity badly enough that no RRF variant (unweighted, reweighted RRF
constant, or NDCG-weighted) fully recovered ItemCF's solo NDCG@20 — see the
Phase 9 finding below. The criterion is dropped: **Phase 4 is a candidate
generator, not a ranker** — its job is to guarantee the held-out tracks are
*in* the pool at all; ranking within the pool is the RL agent's job
(Phase 6), which is exactly the layer the flat 100-song action space (B.7)
exists to make tractable. An NDCG bar on the pool itself was the wrong gate.

**Dependencies.** Phase 3.

---

### PHASE 5 — RL environment

**Goal.** A Gym-style environment over playlist construction.

**Tasks.**
1. `PlaylistEnv(user_idx, candidates, content_matrix, holdout, config)`.
2. `reset()` → initial state tuple (see B.7 discretisation) + empty playlist.
3. `step(arm_k)` → pick best unused candidate in cluster k → compute reward
   (B.12) → update state → return `(s', r, done, info)`. `done` when
   `len(playlist) == playlist_length` (default 15). **The artist cap is a
   soft constraint, not a hard filter on candidate selection**: a candidate
   that would exceed it stays selectable and is charged the `w_rep` penalty
   instead of being removed from consideration. Only the no-duplicate
   constraint is a hard filter — a playlist containing the same track twice
   is a bug, not a trade-off.

   *Why soft, not hard (revised after Phase 5 was first built with a hard
   cap):* the repetition penalty is the primary source of sequential credit
   assignment in this MDP — the cost of repeating an artist is paid at the
   step it happens, but the benefit of having deferred it only shows up in
   later steps' diversity/relevance terms. If repetition is structurally
   impossible, the agent has nothing to learn to defer, and the argument for
   tabular RL over a stateless contextual bandit (which just maximises
   immediate reward and would behave identically under a hard cap) weakens
   to nothing. The penalty must be a cost the agent can choose to pay.
4. Cluster into K=10 arms with KMeans on `content_matrix` rows, seeded, using
   **fixed global clusters** (one KMeans fit over the whole core catalogue,
   `src/rl/fit_global_clusters.py`, persisted to
   `models/global_song_clusters.npy`) rather than refitting per user's
   candidate pool. Phase 5 was first built with per-candidate-set refit,
   which B.7 also allows — it had to be revised once Phase 6 made the
   requirement concrete: EpsilonGreedyMAB, UCB1, and the Q-table's
   convergence over 20,000 episodes (Phase 6) all require arm identity to
   carry the same content meaning across every episode, so a value learned
   for "arm 3" on one user's episode transfers to the next user's. Per-
   episode KMeans refitting can't give you that — its cluster labels are
   arbitrary and independently assigned each time, so "cluster 3" means
   nothing consistent across users.
5. Expose both a **grounded-only** reward mode (Tier 1) and a
   **grounded+simulated** mode (Tier 1 + Tier 2) via a config flag. Run both;
   report both.
6. Unit-test: reward is deterministic given (user, action sequence, seed);
   artist cap violations incur `w_rep` and are reported in the reward
   decomposition; no duplicate track ever enters a playlist.

**Output.** `src/rl/environment.py` + `tests/test_env.py`.

**Acceptance.** All unit tests pass. A random policy over 1,000 episodes produces
a stable mean reward with sane variance. The reward decomposition
(rel / div / repeat) is logged per step so you can debug which term dominates
— under a random policy, expect to see some non-zero rate of artist-cap
violations (report it), since nothing prevents a uniform-random arm choice
from repeating an artist.

**Dependencies.** Phase 4.

---

### PHASE 6 — RL agents

**Goal.** Seven agents, all subclasses of one `BaseAgent` interface, trained
and compared.

**Tasks.**
0. Define `BaseAgent` (abstract base class) with exactly two required methods:
   `select_action(state) -> arm` and `update(state, action, reward, next_state,
   done)`. The environment and training loop (Phase 5, `train.py`) must depend
   only on this interface, never on a concrete agent's internals — this is
   what lets the deferred agents (Part F) be added later without touching
   either.
1. `RandomAgent`, `GreedyAgent` (always highest hybrid score — the ablation that
   shows whether RL adds anything at all).
2. `EpsilonGreedyMAB`, `UCB1` — arm-level, no state.
3. `LinUCB` — song-level, context = `[user_profile ⊕ content_vec(a) ⊕
   playlist_state_features]`, α ∈ {0.5, 1.0, 2.0}.
4. `QLearningAgent` — tabular 162×10, α=0.1, γ=0.9, ε decaying 1.0→0.05.
5. `SARSAAgent` — same table, on-policy update.
6. Train each for 20,000 episodes × 5 seeds. Log per-episode reward.

**Output.** `models/rl/*.pkl`, `results/rl_curves.csv`.

**Acceptance.** Every agent is a subclass of `BaseAgent` and is driven only
through `select_action`/`update`. Q-Learning and SARSA both converge
(rolling-mean reward plateaus). Q-Learning and SARSA produce *different*
policies — if the learned Q-tables are identical, the on/off-policy
distinction is not being implemented correctly. All learned agents beat
Random. Report whether they beat Greedy — **if they do not, that is a
legitimate finding**, and the honest analysis of *why* (sparse reward, short
horizon, strong hybrid prior) is worth more than a fabricated win.

**Dependencies.** Phase 5.

---

### PHASE 7 — Evaluation

**Goal.** The comparison table and figures that carry the report.

**Tasks.**
1. `ranking_metrics.py` — Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, HitRate@K.
2. `beyond_accuracy.py` — Catalog Coverage, Intra-List Diversity, Novelty,
   Artist Entropy, Gini.
3. Evaluate all 8 configurations (Random, Popularity, Content, ItemCF, ALS,
   Hybrid, Hybrid+LinUCB, Hybrid+QL) at K=10 and K=20, 5 seeds, mean ± std.
4. Ablation: with/without the genre block; grounded-only vs grounded+simulated reward.
5. Figures: model comparison bars with error bars, RL learning curves with 95%
   CI, coverage vs novelty.

*(The `w_div` sensitivity sweep and its accuracy-vs-diversity trade-off curve
are deferred — see Part F. Run them only after this phase's core evaluation
is complete.)*

**Output.** `results/final_comparison.csv`, `results/figures/`.

**Acceptance.** No RMSE or MAE anywhere in the results. Every number has a
standard deviation. Every claim in the report traces to a row in the CSV.

**Dependencies.** Phase 6.

---

### PHASE 8 — Streamlit UI

**Goal.** A demo that runs in under 3 seconds per playlist.

**Tasks.**
1. Inputs: user selector (searchable), playlist length slider (5–20), agent
   selector.
2. Output: numbered playlist with track, artist, genre/tags, year, plus a
   Spotify preview `st.audio` where `spotify_preview_url` exists.
3. Side panel: the user's top-10 training tracks (so a viewer can judge whether
   the recommendations make sense), and the playlist's diversity/novelty scores.
4. Pre-compute and cache candidates; `@st.cache_resource` for models.

*(The mood-quadrant candidate filter — B.16 — is deferred; add it as an
optional experiment, Part F, once the core UI works.)*

**Acceptance.** Cold start < 10 s, per-playlist < 3 s. No model training at
request time. Runs from a clean checkout with one command.

**Dependencies.** Phase 7.

---

### PHASE 9 — Report

**Goal.** The document that survives the viva.

**Must explicitly contain:**
- The dataset statistics table from Part A, independently reproduced.
- The join analysis in **both directions**, including the 20,224 orphan tracks.
- **A limitations section stating:** no timestamps (hence no temporal split and
  no observed listening order), no skip/negative signal, no session boundaries,
  no demographics; and a precise statement of which parts of the reward are
  grounded in observed playcounts versus which are a declared content-similarity
  simulator.
- The justification for choosing tabular RL over DQN (B.6) and the conditions
  under which DQN *would* be warranted.
- The justification for ALS-with-confidence over SVD++ (B.2).
- The two-level action-space design and why a flat 100-song tabular action space
  would have been invalid (B.7).
- The ItemCF top-k neighbour sweep (`results/phase3/item_cf_k_sweep.csv`),
  reported as a **sensitivity analysis, not as tuning** — k was swept on the
  same holdout used for Phase 3 acceptance evaluation, so the honest framing
  is "this is how sensitive the model is to k," not "k was optimised";
  note that a separate validation split would have been the rigorous approach
  to actually tune k.
- **Phase 4 finding: ALS/Content/Popularity are largely redundant with the
  ItemCF neighbourhood signal on this dataset, not complementary to it.**
  ItemCF alone reaches candidate recall@100 = 0.5562; pooling all four
  models only reaches 0.5572 (`results/phase4/fusion_comparison.csv`,
  `results/phase4/phase4_metrics.csv`) — a 0.001 gain for three extra
  models. Plain (unweighted) RRF fusion also *failed* to beat solo ItemCF's
  NDCG@20 (0.2689 vs 0.3345); reweighting RRF by each model's solo NDCG@20
  recovered most of the gap and beat ItemCF on Recall@20 (0.3976 vs 0.3923)
  but still trailed on NDCG@20 (0.3168 vs 0.3345) — see the acceptance
  criterion change above. Candidate coverage was never the issue; ranking
  quality was. Attribute this to the aggressive k-core filtering
  (Phase 1): the resulting core interaction matrix is dense enough
  (0.322% of the user×item grid has an interaction — Phase 1's own
  reproduced stat — vs a typical <0.01% for unfiltered implicit-feedback
  catalogues) that item co-occurrence alone
  captures most of the recoverable signal, leaving little room for
  ALS/Content/Popularity to add. State plainly that this is a property of
  the filtered 39,825-user/11,613-item subset used here, not a general
  claim that hybrid recommenders are redundant. Also note k=5 (chosen on
  the acceptance holdout, see above) may be partly responsible for how
  dominant ItemCF appears here — the two findings are not independent.

---

# PART F — DEFERRED / OPTIONAL EXPERIMENTS (post-core only)

Do not start any of these until the core system — Phases 0–9, covering
Popularity, Content-Based, Item-Based CF, Implicit ALS, Hybrid-RRF, and the
RL agents Random, Greedy, ε-greedy MAB, UCB1, LinUCB, Q-Learning, SARSA — is
complete and evaluated (Phase 7's comparison table exists and is reported).
These are optional, additive experiments, not blockers for the core
deliverable or the viva.

1. **BPR** (Rendle et al., 2009) — pairwise-ranking contrast to ALS (B.2).
   Fit alongside the existing ALS model; report BPR vs ALS on the Phase 7
   metrics.
2. **Weighted linear hybrid** (α·CF + β·content + γ·popularity, α+β+γ=1),
   per-recommender min-max normalisation, weights tuned by grid search on the
   validation split (B.14). Report side by side with the core RRF hybrid.
3. **Context-aware mood extension** (B.16) — mood quadrant from
   valence×energy as a Streamlit candidate-set pre-filter.
4. **`w_div` sensitivity sweep** — {0.0, 0.15, 0.3, 0.5} → accuracy-vs-diversity
   trade-off curve (B.12).
5. **DQN — documented only, never implemented.** B.6 explains why tabular RL
   is the right choice here; the project's hard constraint (no deep learning)
   means DQN stays declared future work for the report's limitations
   section, not something to build even as an optional experiment.

Because every core agent (Phase 6) subclasses `BaseAgent`, adding new bandit
or RL variants later requires only a new subclass — the environment and
training loop do not change.

---

# PART E — VIVA PREPARATION

Answer these before your defence. They are the questions the data invites.

1. *"You call this sequential — where is the sequence in your data?"*
   → It is not in the data. There are no timestamps. The MDP is over the playlist
   the agent constructs; transitions come from the agent's own prior actions.

2. *"Your reward uses content similarity for tracks the user never played. Isn't
   that circular — rewarding the recommender for agreeing with itself?"*
   → Partially, yes, and that is why the grounded-only (Tier 1) results are
   reported alongside. Point to both columns of the results table.

3. *"Why not SVD++, it's in your syllabus?"*
   → No ratings exist. `playcount` has a floor of 1 and no observed negatives.
   ALS with confidence weighting is the implicit-feedback member of the same
   latent-factor family.

4. *"Why did you throw away 88% of your users?"*
   → Median user has 5 interactions; a 15/5 split needs ≥20. Quote the k-core
   table and the resulting density improvement, 0.0331% → 0.309%.

5. *"Does the RL layer actually beat the hybrid on its own?"*
   → Answer with your real numbers. If it doesn't, explain why (5 relevant items
   in 100 candidates, 15-step horizon, strong hybrid prior) and what would change
   it (denser feedback, longer playlists, real skip signal).

6. *"What does Q-Learning learn that UCB doesn't?"*
   → UCB has no state. Q-Learning conditions on playlist context, so it can
   learn to hold a high-relevance arm back early to preserve diversity headroom
   later. Show the learned Q-table for `position=late, diversity=low`.
