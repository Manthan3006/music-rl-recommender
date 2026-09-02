# Project

Adaptive Hybrid Music Playlist Recommendation using Reinforcement Learning.
Final-year mini-project spanning two courses: Recommendation Systems and
Reinforcement Learning.

**The full build specification is `docs/SPEC.md`. Read it before writing code.**
Part D of that file is the phase-by-phase roadmap. Work one phase at a time and
do not start a phase until the previous phase's acceptance criteria pass.

## Hard constraints
- No deep learning. No DQN, no NCF, no transformers.
- No SVD or SVD++ — the data is implicit feedback with no ratings.
- Never report RMSE or MAE.
- SEED = 42 everywhere, set in config.py.
- Every module writes its outputs to disk. No notebook-only code.

## Data
- data/raw/Music_Info.csv — 50,683 tracks, UTF-8 BOM (use encoding='utf-8-sig')
- data/raw/User Listening History.csv — 9,711,301 rows, ~600MB, no timestamps
