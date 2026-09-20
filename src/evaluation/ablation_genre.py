"""Phase 7 ablation 1 — with / without the genre block (B.5, Phase 2 task 3).

B.5: genre is 55.9% missing and only 46.5% self-consistent with tags, so the
core content vector excludes it. This ablation tests that decision instead of
asserting it: rebuild the item-content matrix with a one-hot genre block
(UNK for missing) at weight 0.10, the spec's ceiling, on top of the existing
audio 0.45 / tags 0.55 blocks, then compare on the SAME 5 evaluation folds:

  * Content-only recommender (K=10, 20)
  * Hybrid-RRF (the four recommenders re-pooled with the genre-variant content
    recommender; ItemCF/ALS/Popularity unchanged), K=10, 20 and candidate
    recall@100.

Diversity is always measured with the BASELINE content matrix so both variants
are scored by the same yardstick. The RL agents were trained on the baseline
matrix and are not retrained here — this ablation covers the recommender/
candidate layers only.

The no-genre rebuild is first asserted to reproduce data/processed/
content_matrix.npz, so the baseline arm is the real Phase 2 baseline.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from config import AUDIO_FEATURE_WEIGHT, DATA_PROCESSED, RESULTS, TAG_FEATURE_WEIGHT
from src import features
from src.evaluation.eval_users import get_eval_folds
from src.evaluation.evaluate_all import KS, Context, log, score_fold
from src.recommenders import hybrid
from src.recommenders.common import load_idx_maps, load_train_aff_csr
from src.recommenders.content import ContentRecommender

GENRE_WEIGHT = 0.10
OUT = RESULTS / "phase7"


def build_variants():
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    audio = sp.csr_matrix(features.build_audio_block(items))
    tags = features.build_tag_block(items, features.build_tag_vocabulary())
    genre = sp.csr_matrix(pd.get_dummies(items["genre"].fillna("UNK")).to_numpy(dtype=float))
    base_parts = [audio * AUDIO_FEATURE_WEIGHT, tags.multiply(TAG_FEATURE_WEIGHT).tocsr()]
    base = normalize(sp.hstack(base_parts, format="csr"), norm="l2", axis=1)
    with_genre = normalize(sp.hstack(base_parts + [genre * GENRE_WEIGHT], format="csr"), norm="l2", axis=1)

    stored = sp.load_npz(DATA_PROCESSED / "content_matrix.npz").tocsr()
    assert abs(base - stored).max() < 1e-9, "no-genre rebuild does not reproduce the Phase 2 content matrix"
    log(f"No-genre rebuild reproduces content_matrix.npz exactly; genre one-hot has {genre.shape[1]} columns "
        f"(incl. UNK for {int((items['genre'].isna()).sum()):,} tracks)")

    _, core_track_ids = load_idx_maps()
    row_of_track = pd.Series(np.arange(len(items)), index=items["track_id"])
    rows = row_of_track.loc[core_track_ids].to_numpy()
    return base[rows], with_genre[rows]


def make_content_rec(core_content, aff):
    profiles = normalize(aff @ core_content, norm="l2", axis=1)  # TRAIN affinities only
    return ContentRecommender(profiles.tocsr(), core_content)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds, _ = get_eval_folds()
    ctx = Context()
    base_core, genre_core = build_variants()
    aff = load_train_aff_csr()
    recs = {"no_genre": make_content_rec(base_core, aff), "genre_0.10": make_content_rec(genre_core, aff)}

    sizes = hybrid.pool_sizes(hybrid.DEFAULT_N)
    hybrid_models = {n: ctx.recs[m] for n, m in [("item_cf", "ItemCF"), ("als", "ALS"), ("popularity", "Popularity")]}

    # the rebuilt no-genre pool must equal the stored Phase 4 candidates
    check_users = [int(u) for u in folds[0][:200]]
    for u in check_users:
        ranked = hybrid.rrf_rank_for_user({**hybrid_models, "content": recs["no_genre"]}, sizes, u, ctx.seen(u), hybrid.DEFAULT_N)
        assert [i for i, _ in ranked] == ctx.cand[u][0].tolist(), "rebuilt no-genre hybrid pool differs from candidates.parquet"
    log(f"Rebuilt no-genre hybrid pool matches candidates.parquet for {len(check_users)} users.")

    rows = []
    for variant, content_rec in recs.items():
        for f, users in enumerate(folds):
            content_lists, hybrid_lists, cand_recalls = {}, {}, []
            for u in users:
                u = int(u)
                seen = ctx.seen(u)
                content_lists[u] = [i for i, _ in content_rec.recommend(u, max(KS), exclude_seen=True, seen=seen)]
                if variant == "no_genre":
                    pool = ctx.cand[u][0].tolist()
                else:
                    ranked = hybrid.rrf_rank_for_user({**hybrid_models, "content": content_rec}, sizes, u, seen, hybrid.DEFAULT_N)
                    pool = [i for i, _ in ranked]
                hybrid_lists[u] = pool[:max(KS)]
                cand_recalls.append(len(set(pool) & ctx.holdout[u]) / len(ctx.holdout[u]))
            for model, lists in (("Content", content_lists), ("Hybrid", hybrid_lists)):
                for k in KS:
                    for m, v in score_fold(lists, k, ctx).items():
                        rows.append((variant, model, k, f, m, v))
            rows.append((variant, "Hybrid", 100, f, "candidate_recall", float(np.mean(cand_recalls))))
        log(f"  scored variant {variant}")

    df = pd.DataFrame(rows, columns=["variant", "model", "K", "fold", "metric", "value"])
    df.to_csv(OUT / "ablation_genre_per_fold.csv", index=False)

    agg = df.groupby(["model", "K", "metric", "variant"])["value"].agg(["mean", "std"]).reset_index()
    piv = df.pivot_table(index=["model", "K", "metric", "fold"], columns="variant", values="value").reset_index()
    piv["diff"] = piv["genre_0.10"] - piv["no_genre"]
    paired = piv.groupby(["model", "K", "metric"])["diff"].agg(["mean", "std"]).rename(
        columns={"mean": "diff_mean", "std": "diff_std"}).reset_index()
    paired["folds_genre_better"] = piv.assign(pos=piv["diff"] > 0).groupby(["model", "K", "metric"])["pos"].sum().to_numpy()
    out = agg.merge(paired, on=["model", "K", "metric"])
    out.to_csv(OUT / "ablation_genre.csv", index=False)

    show = out[out.metric.isin(["recall", "ndcg", "intra_list_diversity", "candidate_recall"])]
    p = show.pivot_table(index=["model", "K", "metric"], columns="variant", values="mean")
    p["diff (genre - no_genre)"] = show.drop_duplicates(["model", "K", "metric"]).set_index(["model", "K", "metric"])["diff_mean"]
    p["diff std"] = show.drop_duplicates(["model", "K", "metric"]).set_index(["model", "K", "metric"])["diff_std"]
    print("\n=== Genre-block ablation (mean across 5 folds) ===")
    print(p.round(4).to_string())


if __name__ == "__main__":
    main()
