"""Phase 0 — Verification & EDA.

Independently reproduces every number claimed in docs/SPEC.md Part A from the
raw CSVs, and writes figures + a stats JSON to results/eda/. No number here is
assumed — everything is recomputed from data/raw on every run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    LISTENING_HISTORY_CSV,
    MUSIC_INFO_CSV,
    MUSIC_INFO_ENCODING,
    RESULTS,
    SEED,
)

np.random.seed(SEED)

EDA_DIR = RESULTS / "eda"
FIG_DIR = EDA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "loudness", "tempo",
]


def load_music_info() -> pd.DataFrame:
    df = pd.read_csv(MUSIC_INFO_CSV, encoding=MUSIC_INFO_ENCODING)
    return df


def load_listening_history() -> pd.DataFrame:
    dtypes = {"track_id": "string", "user_id": "string", "playcount": "int32"}
    df = pd.read_csv(LISTENING_HISTORY_CSV, dtype=dtypes)
    return df


def music_info_stats(music: pd.DataFrame) -> dict:
    stats = {}
    stats["rows"] = int(len(music))
    stats["columns"] = int(music.shape[1])
    stats["unique_track_id"] = int(music["track_id"].nunique())
    stats["unique_name"] = int(music["name"].nunique())
    stats["unique_artist"] = int(music["artist"].nunique())
    stats["unique_spotify_id"] = int(music["spotify_id"].nunique())
    stats["duplicate_spotify_id_pairs"] = stats["rows"] - stats["unique_spotify_id"]

    stats["genre_missing"] = int(music["genre"].isna().sum())
    stats["genre_missing_pct"] = round(100 * stats["genre_missing"] / stats["rows"], 1)
    stats["genre_distinct_values"] = int(music["genre"].nunique())

    stats["tags_missing"] = int(music["tags"].isna().sum())
    stats["tags_missing_pct"] = round(100 * stats["tags_missing"] / stats["rows"], 1)

    neither = music["genre"].isna() & music["tags"].isna()
    stats["neither_tag_nor_genre"] = int(neither.sum())

    tag_lists = music["tags"].dropna().apply(lambda s: [t.strip() for t in s.split(",")])
    vocab = set()
    for tl in tag_lists:
        vocab.update(tl)
    stats["tags_vocab_size"] = len(vocab)
    tag_counts = pd.Series([t for tl in tag_lists for t in tl]).value_counts()
    stats["tags_vocab_min_count"] = int(tag_counts.min())
    # Denominator is ALL tracks (missing tags -> 0), matching docs/SPEC.md Part A.
    n_tags_all = music["tags"].apply(lambda s: 0 if pd.isna(s) else len(s.split(",")))
    stats["tags_per_track_mean"] = round(float(n_tags_all.mean()), 2)
    stats["tags_per_track_median"] = float(n_tags_all.median())
    stats["tags_per_track_max"] = int(n_tags_all.max())

    stats["year_min"] = int(music["year"].min())
    stats["year_max"] = int(music["year"].max())
    stats["year_median"] = float(music["year"].median())
    stats["year_1900_rows"] = int((music["year"] == 1900).sum())

    stats["tempo_zero"] = int((music["tempo"] == 0).sum())
    stats["time_signature_zero"] = int((music["time_signature"] == 0).sum())
    stats["tempo_and_timesig_zero_same_rows"] = int(
        ((music["tempo"] == 0) & (music["time_signature"] == 0)).sum()
    )

    stats["duration_ms_min"] = int(music["duration_ms"].min())
    stats["duration_ms_max"] = int(music["duration_ms"].max())
    stats["duration_over_10min"] = int((music["duration_ms"] > 600_000).sum())

    stats["instrumentalness_near_zero_pct"] = round(
        100 * (music["instrumentalness"] < 0.01).mean(), 1
    )
    stats["acousticness_near_zero_pct"] = round(
        100 * (music["acousticness"] < 0.01).mean(), 1
    )

    skew = music[AUDIO_FEATURES].skew()
    stats["skew"] = {k: round(float(v), 2) for k, v in skew.items()}

    corr = music[AUDIO_FEATURES].corr()
    stats["corr_energy_loudness"] = round(float(corr.loc["energy", "loudness"]), 2)
    stats["corr_energy_acousticness"] = round(float(corr.loc["energy", "acousticness"]), 2)
    stats["corr_loudness_acousticness"] = round(float(corr.loc["loudness", "acousticness"]), 2)
    stats["corr_danceability_valence"] = round(float(corr.loc["danceability", "valence"]), 2)

    return stats, tag_lists, tag_counts, corr


def genre_tag_agreement(music: pd.DataFrame, tag_lists: pd.Series) -> pd.DataFrame:
    has_genre = music["genre"].notna()
    sub = music.loc[has_genre, ["genre"]].copy()
    sub["tag_list"] = tag_lists.reindex(sub.index).apply(
        lambda x: x if isinstance(x, list) else []
    )

    def normalize(s):
        return s.strip().lower().replace(" ", "_")

    def agrees(row):
        return normalize(row["genre"]) in {normalize(t) for t in row["tag_list"]}

    sub["agree"] = sub.apply(agrees, axis=1)
    table = sub.groupby("genre")["agree"].agg(["mean", "count"]).rename(
        columns={"mean": "agreement", "count": "n"}
    ).sort_values("agreement", ascending=False)
    overall = sub["agree"].mean()
    return table, overall, len(sub)


def find_duplicate_tracks(music: pd.DataFrame) -> pd.DataFrame:
    dup = music[music.duplicated("spotify_id", keep=False) & music["spotify_id"].notna()]
    dup = dup.sort_values("spotify_id")
    return dup[["spotify_id", "track_id", "artist", "name"]]


def listening_history_stats(hist: pd.DataFrame) -> dict:
    stats = {}
    stats["rows"] = int(len(hist))
    stats["nulls"] = int(hist.isna().sum().sum())
    stats["exact_duplicate_rows"] = int(hist.duplicated().sum())
    stats["duplicate_user_track_pairs"] = int(hist.duplicated(["user_id", "track_id"]).sum())
    stats["unique_users"] = int(hist["user_id"].nunique())
    stats["unique_tracks"] = int(hist["track_id"].nunique())

    pc = hist["playcount"]
    stats["playcount_min"] = int(pc.min())
    stats["playcount_max"] = int(pc.max())
    stats["playcount_mean"] = round(float(pc.mean()), 2)
    stats["playcount_median"] = float(pc.median())
    stats["playcount_eq1_pct"] = round(100 * (pc == 1).mean(), 1)
    stats["playcount_ge10_pct"] = round(100 * (pc >= 10).mean(), 1)
    stats["playcount_skew"] = round(float(pc.skew()), 2)
    stats["playcount_log1p_skew"] = round(float(np.log1p(pc).skew()), 2)

    density = stats["rows"] / (stats["unique_users"] * stats["unique_tracks"])
    stats["density_pct"] = round(100 * density, 4)

    per_user = hist.groupby("user_id").size()
    stats["interactions_per_user_mean"] = round(float(per_user.mean()), 2)
    stats["interactions_per_user_median"] = float(per_user.median())

    thresholds = [5, 10, 15, 20, 30, 50]
    threshold_table = {}
    for t in thresholds:
        n = int((per_user >= t).sum())
        threshold_table[t] = {"users": n, "pct": round(100 * n / stats["unique_users"], 1)}
    stats["users_by_min_interactions"] = threshold_table

    per_track = hist.groupby("track_id").size()
    stats["listeners_per_track_mean"] = round(float(per_track.mean()), 1)
    stats["listeners_per_track_median"] = float(per_track.median())
    stats["listeners_per_track_max"] = int(per_track.max())
    stats["tracks_ge5_listeners"] = int((per_track >= 5).sum())
    stats["tracks_ge100_listeners"] = int((per_track >= 100).sum())

    return stats, per_user, per_track


def join_analysis(music: pd.DataFrame, hist: pd.DataFrame, hist_stats: dict) -> dict:
    music_ids = set(music["track_id"])
    hist_ids = set(hist["track_id"].unique())
    intersection = music_ids & hist_ids
    missing_metadata = hist_ids - music_ids
    orphan_catalogue = music_ids - hist_ids
    return {
        "tracks_in_history": len(hist_ids),
        "tracks_in_music_info": len(music_ids),
        "intersection": len(intersection),
        "history_tracks_missing_metadata": len(missing_metadata),
        "music_info_tracks_zero_listens": len(orphan_catalogue),
    }


def make_plots(music, hist, per_user, per_track, tag_counts, corr):
    # Playcount distribution, log-log
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = hist["playcount"].value_counts().sort_index()
    ax.loglog(counts.index, counts.values, marker=".", linestyle="none")
    ax.set_xlabel("playcount")
    ax.set_ylabel("frequency")
    ax.set_title("Playcount distribution (log-log)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "playcount_distribution_loglog.png", dpi=150)
    plt.close(fig)

    # Interactions-per-user CDF
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_vals = np.sort(per_user.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf)
    ax.set_xscale("log")
    ax.set_xlabel("interactions per user")
    ax.set_ylabel("CDF")
    ax.set_title("Interactions-per-user CDF")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "interactions_per_user_cdf.png", dpi=150)
    plt.close(fig)

    # Listeners-per-track CDF
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_vals = np.sort(per_track.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf)
    ax.set_xscale("log")
    ax.set_xlabel("listeners per track")
    ax.set_ylabel("CDF")
    ax.set_title("Listeners-per-track CDF")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "listeners_per_track_cdf.png", dpi=150)
    plt.close(fig)

    # Audio feature histograms
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, feat in zip(axes.flat, AUDIO_FEATURES):
        ax.hist(music[feat].dropna(), bins=50)
        ax.set_title(feat)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "audio_feature_histograms.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(AUDIO_FEATURES)))
    ax.set_xticklabels(AUDIO_FEATURES, rotation=90)
    ax.set_yticks(range(len(AUDIO_FEATURES)))
    ax.set_yticklabels(AUDIO_FEATURES)
    fig.colorbar(im)
    ax.set_title("Audio feature correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Tag frequency top-40
    fig, ax = plt.subplots(figsize=(8, 8))
    top40 = tag_counts.sort_values(ascending=False).head(40)
    ax.barh(top40.index[::-1], top40.values[::-1])
    ax.set_title("Top-40 tags by frequency")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tag_frequency_top40.png", dpi=150)
    plt.close(fig)

    # Genre missingness
    fig, ax = plt.subplots(figsize=(4, 4))
    missing = music["genre"].isna().sum()
    present = len(music) - missing
    ax.pie([present, missing], labels=["has genre", "missing genre"], autopct="%1.1f%%")
    ax.set_title("Genre missingness")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "genre_missingness.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading Music_Info.csv ...")
    music = load_music_info()
    print("Loading User Listening History.csv (this is ~600MB, may take a while) ...")
    hist = load_listening_history()

    print("Computing Music_Info stats ...")
    mi_stats, tag_lists, tag_counts, corr = music_info_stats(music)

    print("Computing genre/tag agreement ...")
    agreement_table, overall_agreement, n_both = genre_tag_agreement(music, tag_lists)

    print("Finding duplicate spotify_id pairs ...")
    dup_tracks = find_duplicate_tracks(music)

    print("Computing listening history stats ...")
    hist_stats, per_user, per_track = listening_history_stats(hist)

    print("Computing join analysis ...")
    join_stats = join_analysis(music, hist, hist_stats)

    print("Generating figures ...")
    make_plots(music, hist, per_user, per_track, tag_counts, corr)

    all_stats = {
        "music_info": mi_stats,
        "genre_tag_agreement": {
            "overall": round(float(overall_agreement), 3),
            "n_tracks_with_both": int(n_both),
            "by_genre": {
                g: {"agreement": round(float(r["agreement"]), 3), "n": int(r["n"])}
                for g, r in agreement_table.iterrows()
            },
        },
        "duplicate_spotify_id_pairs": {
            "count": int(dup_tracks["spotify_id"].nunique()),
            "pairs": dup_tracks.to_dict(orient="records"),
        },
        "listening_history": hist_stats,
        "join_analysis": join_stats,
    }

    out_path = EDA_DIR / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== SUMMARY ===")
    print(f"Music_Info rows: {mi_stats['rows']} (expect 50683)")
    print(f"Unique spotify_id: {mi_stats['unique_spotify_id']} -> {mi_stats['duplicate_spotify_id_pairs']} dup pairs (expect 9)")
    print(f"Genre missing: {mi_stats['genre_missing']} (expect 28335)")
    print(f"Tags missing: {mi_stats['tags_missing']} (expect 1127)")
    print(f"Tags vocab size: {mi_stats['tags_vocab_size']} (expect 100)")
    print(f"Genre/tag overall agreement: {overall_agreement:.3f} (expect ~0.465)")
    print(f"History rows: {hist_stats['rows']} (expect 9711301)")
    print(f"Unique users: {hist_stats['unique_users']} (expect 962037)")
    print(f"Unique tracks in history: {hist_stats['unique_tracks']} (expect 30459)")
    print(f"Density: {hist_stats['density_pct']}% (expect 0.0331%)")
    print(f"Join: history tracks missing metadata: {join_stats['history_tracks_missing_metadata']} (expect 0)")
    print(f"Join: catalogue tracks w/ 0 listens: {join_stats['music_info_tracks_zero_listens']} (expect 20224)")


if __name__ == "__main__":
    main()
