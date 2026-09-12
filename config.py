"""Global configuration and constants for the project. Single source of truth."""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

MUSIC_INFO_CSV = DATA_RAW / "Music_Info.csv"
LISTENING_HISTORY_CSV = DATA_RAW / "User Listening History.csv"

MUSIC_INFO_ENCODING = "utf-8-sig"  # Music_Info.csv has a UTF-8 BOM

# k-core filtering thresholds (Part B.4 of docs/SPEC.md)
MIN_USER_INTERACTIONS = 20
MIN_TRACK_LISTENERS = 50  # population-scale threshold, used pre-subsample
SUBSAMPLE_USERS = 40_000
# Post-subsample re-filter threshold. NOT the same as MIN_TRACK_LISTENERS:
# re-applying the population-scale >=50 threshold to a 40,000-user subsample
# is disproportionately strict and cascades through the iterative k-core,
# costing ~49% of the catalogue. See docs/SPEC.md Phase 1, task 5.
POST_SUBSAMPLE_MIN_TRACK_LISTENERS = 17

# Playcount -> preference/confidence (Part B.11)
ALS_ALPHA = 40
CONFIDENCE_EPS = 1e-8

# Content feature blend weights (Phase 2)
AUDIO_FEATURE_WEIGHT = 0.45
TAG_FEATURE_WEIGHT = 0.55

# Reward weights (Part B.12)
W_REL = 1.0
W_DIV = 0.3
W_REP = 0.5
W_DUP = 1.0
SIMULATED_REL_LAMBDA = 0.3  # Tier 2 backoff weight (B.12)

# RL environment (Phase 5 / B.7)
PLAYLIST_LENGTH = 15
ARTIST_CAP = 2
N_ARMS = 10

# Candidate generation (Part B.13)
CANDIDATE_POOL_SIZE = 100
CANDIDATE_RECALL_MIN = 0.40

for _dir in (DATA_PROCESSED, RESULTS, MODELS):
    _dir.mkdir(parents=True, exist_ok=True)
