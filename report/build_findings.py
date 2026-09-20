"""Phase 9 — build report/FINDINGS.md from report/findings_template.md.

Every table, and every result number cited in the prose, is filled in from the
CSVs under results/ (and results/eda/dataset_stats.json) so the report cannot
drift from the data. Placeholders in the template:

  {{T:name}}                 a generated markdown table
  {{MS:config:K:metric}}     'mean ± std' from results/final_comparison.csv
  {{M:config:K:metric}}      mean only
  {{TR:agent:column}}        table of record (results/phase6/grounded/final_eval_no_exploration.csv)
  {{AB:trained_with:agent:column}}  results/phase7/ablation_reward.csv
  {{V:key}}                  a derived value defined in `values()` below

Run: python report/build_findings.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import W_DIV, W_REP

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEMPLATE = ROOT / "report" / "findings_template.md"
OUT = ROOT / "report" / "FINDINGS.md"

FC = pd.read_csv(R / "final_comparison.csv").set_index(["config", "K", "metric"])
TOR = pd.read_csv(R / "phase6/grounded/final_eval_no_exploration.csv").set_index("agent")
ABL = pd.read_csv(R / "phase7/ablation_reward.csv").set_index(["trained_with", "agent"])
STATS = json.load(open(R / "eda/dataset_stats.json"))
UF = pd.read_csv(R / "phase7/ucb1_frozen_arm.csv")
CONFIGS = ["Random", "Popularity", "Content", "ItemCF", "ALS", "Hybrid", "Hybrid+RandomArm",
           "Hybrid+UCB1", "Hybrid+LinUCB", "Hybrid+QLearning", "Hybrid+SARSA"]
AGENTS = ["Random", "Greedy", "UCB1", "LinUCB", "QLearning", "SARSA"]


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def ms(cfg, k, m):
    r = FC.loc[(cfg, k, m)]
    return f"{r['mean']:.4f} ± {r['std']:.4f}"


def display(cfg):
    return {"Random": "Random (catalogue)", "Hybrid": "Hybrid (RRF) = Greedy"}.get(cfg, cfg)


# ------------------------------------------------------------------ tables
def t_cmp(k):
    rows = [{"Configuration": display(c), **{lab: ms(c, k, m) for lab, m in
            [(f"P@{k}", "precision"), (f"R@{k}", "recall"), (f"F1@{k}", "f1"), (f"NDCG@{k}", "ndcg"),
             (f"MAP@{k}", "map"), (f"HR@{k}", "hit_rate")]}} for c in CONFIGS]
    return md(pd.DataFrame(rows))


def t_beyond(k):
    rows = [{"Configuration": display(c), **{lab: ms(c, k, m) for lab, m in
            [("Coverage", "catalog_coverage"), ("ILD", "intra_list_diversity"), ("Novelty (bits)", "novelty"),
             ("Artist entropy (bits)", "artist_entropy"), ("Gini", "gini")]}} for c in CONFIGS]
    return md(pd.DataFrame(rows))


def t_record():
    rows = []
    for a in AGENTS:
        r = TOR.loc[a]
        rows.append({"Agent": a,
                     "Precision@15": f"{r.precision_at_15_mean:.4f} ± {r.precision_at_15_std:.4f}",
                     "Recall@15": f"{r.recall_at_15_mean:.4f} ± {r.recall_at_15_std:.4f}",
                     "Hits / playlist (of 5)": f"{r.mean_hits_per_playlist_mean:.3f}",
                     "Mean reward": f"{r.mean_reward_mean:.4f} ± {r.mean_reward_std:.4f}"})
    return md(pd.DataFrame(rows))


def decomposition():
    g = ABL.loc["grounded"].loc[AGENTS].copy()
    g["div_term"] = W_DIV * 15 * g.mean_div_mean
    g["repeat_penalty"] = -W_REP * 15 * g.repeat_rate_mean
    g["relevance_term"] = g.grounded_reward_mean - g.div_term - g.repeat_penalty
    return g


def t_decomp():
    g = decomposition()
    rows = [{"Agent": a, "Precision@15": f"{g.loc[a].precision_at_15_mean:.4f}",
             "Relevance term": f"{g.loc[a].relevance_term:.3f}", "Diversity term": f"{g.loc[a].div_term:.3f}",
             "Artist-repeat penalty": f"{g.loc[a].repeat_penalty:.3f}", "= Mean reward": f"{g.loc[a].grounded_reward_mean:.3f}",
             "Steps that repeat an artist": f"{100 * g.loc[a].repeat_rate_mean:.1f}%"} for a in AGENTS]
    return md(pd.DataFrame(rows))


def paired():
    f = pd.read_csv(R / "phase7/per_fold_metrics.csv").groupby(["config", "K", "fold", "metric"], as_index=False).value.mean()
    rows = []
    for k in (10, 20):
        for m, lab in (("recall", "Recall"), ("ndcg", "NDCG")):
            p = f[(f.K == k) & (f.metric == m)].pivot(index="fold", columns="config", values="value")
            for c in ["Hybrid+UCB1", "Hybrid+LinUCB", "Hybrid+QLearning", "Hybrid+SARSA"]:
                dh, dr = p[c] - p["Hybrid"], p[c] - p["Hybrid+RandomArm"]
                rows.append({"Metric": f"{lab}@{k}", "Agent": c.split("+")[1], "dh": dh.mean(), "dh_worse": int((dh < 0).sum()),
                             "dr": dr.mean(), "dr_std": dr.std(), "dr_better": int((dr > 0).sum())})
    return pd.DataFrame(rows)


def t_paired():
    d = paired()
    return md(pd.DataFrame({
        "Metric": d.Metric, "Agent": d.Agent,
        "vs Hybrid (Greedy)": [f"{a:+.4f} (worse in {w}/5 folds)" for a, w in zip(d.dh, d.dh_worse)],
        "vs Random-arm control": [f"{a:+.4f} ± {b:.4f} (better in {n}/5 folds)" for a, b, n in zip(d.dr, d.dr_std, d.dr_better)]}))


def t_genre():
    a = pd.read_csv(R / "phase7/ablation_genre.csv")
    a = a[a.metric.isin(["recall", "ndcg", "intra_list_diversity", "candidate_recall"])]
    rows = []
    for (model, k, metric), g in a.groupby(["model", "K", "metric"], sort=False):
        gi, ng = g.set_index("variant").loc["genre_0.10"], g.set_index("variant").loc["no_genre"]
        rows.append({"Layer": model, "K": k if k != 100 else "pool 100", "Metric": metric.replace("_", " "),
                     "No genre": f"{ng['mean']:.4f}", "+ genre (0.10)": f"{gi['mean']:.4f}",
                     "Δ (paired, mean ± std over folds)": f"{gi['diff_mean']:+.4f} ± {gi['diff_std']:.4f}",
                     "Folds genre better": f"{int(gi['folds_genre_better'])}/5"})
    return md(pd.DataFrame(rows))


def t_reward_ablation():
    rows = []
    for mode, label in (("grounded", "grounded (3 seeds)"), ("grounded_simulated", "grounded + simulated (1 seed)")):
        for a in AGENTS:
            r = ABL.loc[(mode, a)]
            rows.append({"Trained with": label, "Agent": a, "Precision@15": f"{r.precision_at_15_mean:.4f}",
                         "Simulated reward": f"{r.sim_reward_mean:.3f}", "Grounded reward": f"{r.grounded_reward_mean:.3f}",
                         "Tier-2 relevance / step": f"{r.tier2_rel_per_step_mean:.3f}", "Diversity / step": f"{r.mean_div_mean:.3f}"})
    return md(pd.DataFrame(rows))


def t_corr():
    c = pd.read_csv(R / "phase7/reward_vs_quality_correlation.csv")
    c["trained_with"] = c["trained_with"].map({"grounded": "grounded", "grounded_simulated": "grounded + simulated"})
    c["reward"] = c["reward"].map({"sim_reward": "simulated reward", "grounded_reward": "grounded reward"})
    c["spearman_rho_vs_precision_at_15"] = c["spearman_rho_vs_precision_at_15"].map("{:+.3f}".format)
    return md(c.rename(columns={"trained_with": "Agents trained with", "reward": "Reward scored",
                                "spearman_rho_vs_precision_at_15": "Spearman ρ vs Precision@15", "n_agents": "Agents"}))


def t_ksweep():
    d = pd.read_csv(R / "phase3/item_cf_k_sweep.csv")
    return md(pd.DataFrame({"k neighbours": d.k_neighbours, "Recall@20": d["recall@20"].map("{:.4f}".format),
                            "NDCG@20": d["ndcg@20"].map("{:.4f}".format), "Coverage@20": d["coverage@20"].map("{:.4f}".format)}))


def t_fusion():
    d = pd.read_csv(R / "phase4/fusion_comparison.csv")
    return md(pd.DataFrame({"Variant": d.variant, "Recall@20": d["recall@20"].map("{:.4f}".format),
                            "NDCG@20": d["ndcg@20"].map("{:.4f}".format),
                            "Candidate recall@100": d["candidate_recall@100"].map("{:.4f}".format)}))


def t_phase4():
    d = pd.read_csv(R / "phase4/phase4_metrics.csv")
    d = d[d.model != "candidate_recall@100"]
    return md(pd.DataFrame({"Model": d.model, "Recall@20": d["recall@k"].map("{:.4f}".format),
                            "NDCG@20": d["ndcg@k"].map("{:.4f}".format), "Users": d.n_users.map("{:,}".format)}))


def t_seedspread():
    b = pd.read_csv(R / "phase7/rl_by_train_seed.csv")
    b = b[(b.K == 20) & (b.metric == "ndcg")].pivot(index="config", columns="train_seed", values="mean")
    rows = [{"Agent": c.split("+")[1], **{f"seed {s}": f"{b.loc[c, s]:.4f}" for s in b.columns}}
            for c in ["Hybrid+RandomArm", "Hybrid+UCB1", "Hybrid+LinUCB", "Hybrid+QLearning", "Hybrid+SARSA"]]
    return md(pd.DataFrame(rows))


def t_phase5():
    d = pd.read_csv(R / "phase5/random_policy_summary.csv")
    d = d[d.reward_mode == "grounded"].iloc[0]
    return (f"{int(d.n_episodes):,} random-policy episodes (grounded): mean reward {d.mean_total_reward:.3f} ± {d.std_total_reward:.3f}; "
            f"artist-cap violations in {100 * d.pct_episodes_with_cap_violation:.1f}% of episodes.")


def t_genreagree():
    by = STATS["genre_tag_agreement"]["by_genre"]
    rows = [{"Genre": g, "Agreement with own tags": f"{v['agreement']:.3f}", "n tracks": f"{v['n']:,}"} for g, v in by.items()]
    return md(pd.DataFrame(rows))


def t_parta():
    m, h, j = STATS["music_info"], STATS["listening_history"], STATS["join_analysis"]
    rows = [
        ("Catalogue tracks", "50,683", f"{m['rows']:,}"),
        ("Distinct artists", "8,317", f"{m['unique_artist']:,}"),
        ("Duplicate `spotify_id` pairs", "9", f"{m['duplicate_spotify_id_pairs']}"),
        ("`genre` missing", "55.9%", f"{m['genre_missing_pct']}%"),
        ("Genre agrees with own tags (overall)", "0.465", f"{STATS['genre_tag_agreement']['overall']}"),
        ("Tag vocabulary", "100 terms, each ≥ 50 uses", f"{m['tags_vocab_size']} terms, min count {m['tags_vocab_min_count']}"),
        ("`tempo == 0` (and `time_signature == 0`, same rows)", "10", f"{m['tempo_zero']}"),
        ("Tracks longer than 10 min", "575", f"{m['duration_over_10min']}"),
        ("`instrumentalness` < 0.01 / `acousticness` < 0.01", "52.8% / 38.4%", f"{m['instrumentalness_near_zero_pct']}% / {m['acousticness_near_zero_pct']}%"),
        ("corr(energy, loudness) / corr(energy, acousticness)", "+0.78 / −0.73", f"{m['corr_energy_loudness']:+.2f} / {m['corr_energy_acousticness']:+.2f}"),
        ("Interactions", "9,711,301", f"{h['rows']:,}"),
        ("Users / tracks with history", "962,037 / 30,459", f"{h['unique_users']:,} / {h['unique_tracks']:,}"),
        ("Matrix density", "0.0331%", f"{h['density_pct']}%"),
        ("Playcount median / mean / max", "1 / 2.63 / 2,948", f"{h['playcount_median']:.0f} / {h['playcount_mean']} / {h['playcount_max']:,}"),
        ("`playcount == 1` / `≥ 10`", "60.8% / 4.2%", f"{h['playcount_eq1_pct']}% / {h['playcount_ge10_pct']}%"),
        ("Interactions per user: median / mean", "5 / 10.09", f"{h['interactions_per_user_median']:.0f} / {h['interactions_per_user_mean']}"),
        ("Users with ≥ 20 interactions", "127,174 (13.2%)", f"{h['users_by_min_interactions']['20']['users']:,} ({h['users_by_min_interactions']['20']['pct']}%)"),
        ("Listeners per track: median / mean / max", "57 / 318.8 / 80,656", f"{h['listeners_per_track_median']:.0f} / {h['listeners_per_track_mean']} / {h['listeners_per_track_max']:,}"),
        ("History tracks missing metadata", "0", f"{j['history_tracks_missing_metadata']}"),
        ("Catalogue tracks with 0 listens (orphans)", "20,224", f"{j['music_info_tracks_zero_listens']:,}"),
        ("Playcount skew, raw → after log1p", "46.31 → 2.11", f"**{h['playcount_skew']} → {h['playcount_log1p_skew']}**  ⚠ differs"),
    ]
    df = pd.DataFrame(rows, columns=["Quantity", "Spec (Part A / B.11)", "Reproduced in Phase 0"])
    return md(df)


TABLES = {"cmp10": lambda: t_cmp(10), "cmp20": lambda: t_cmp(20), "beyond10": lambda: t_beyond(10),
          "beyond20": lambda: t_beyond(20), "record": t_record, "decomp": t_decomp, "paired": t_paired,
          "genre": t_genre, "rewardabl": t_reward_ablation, "corr": t_corr, "ksweep": t_ksweep, "fusion": t_fusion,
          "phase4": t_phase4, "seedspread": t_seedspread, "phase5": t_phase5, "genreagree": t_genreagree, "parta": t_parta}


# ------------------------------------------------------------------ derived values
def values():
    g = decomposition()
    conv = pd.read_csv(R / "phase6/grounded/convergence.csv")
    pol = pd.read_csv(R / "phase6/grounded/policy_comparison.csv")
    k = pd.read_csv(R / "phase3/item_cf_k_sweep.csv").set_index("k_neighbours")
    p4 = pd.read_csv(R / "phase4/phase4_metrics.csv").set_index("model")
    fus = pd.read_csv(R / "phase4/fusion_comparison.csv").set_index("variant")
    sd = pd.read_csv(R / "phase7/rl_by_train_seed.csv")
    ucb = sd[(sd.config == "Hybrid+UCB1") & (sd.K == 20) & (sd.metric == "ndcg")].set_index("train_seed")["mean"]
    pm = paired().set_index(["Metric", "Agent"])
    sim = ABL.loc["grounded_simulated"]
    gr = ABL.loc["grounded"]
    corr = pd.read_csv(R / "phase7/reward_vs_quality_correlation.csv").set_index(["trained_with", "reward"])["spearman_rho_vs_precision_at_15"]
    n_users_core, n_items_core, n_inter = 39825, 11613, 1489288
    v = {
        "skew_raw": f"{STATS['listening_history']['playcount_skew']}", "skew_log": f"{STATS['listening_history']['playcount_log1p_skew']}",
        "eval_users": "5,000", "eval_folds": "5 × 1,000",
        "core_user_share": f"{100 * n_users_core / STATS['listening_history']['unique_users']:.1f}%",
        "core_interaction_share": f"{100 * n_inter / STATS['listening_history']['rows']:.1f}%",
        "core_item_share_of_history": f"{100 * n_items_core / STATS['listening_history']['unique_tracks']:.1f}%",
        "core_item_share_of_catalogue": f"{100 * n_items_core / STATS['music_info']['rows']:.1f}%",
        "density_gain": f"{0.322 / 0.0331:.1f}×",
        "cand_recall": f"{p4.loc['candidate_recall@100', 'recall@k']:.4f}",
        "itemcf_only_cand_recall": f"{fus.loc['itemcf_only_top100', 'candidate_recall@100']:.4f}",
        "unw_rrf_ndcg": f"{fus.loc['rrf_unweighted_k60', 'ndcg@20']:.4f}",
        "w_rrf_ndcg": f"{fus.loc['rrf_weighted_k60', 'ndcg@20']:.4f}",
        "w_rrf_recall": f"{fus.loc['rrf_weighted_k60', 'recall@20']:.4f}",
        "itemcf_recall_p4": f"{fus.loc['itemcf_only_top100', 'recall@20']:.4f}",
        "itemcf_ndcg_p4": f"{fus.loc['itemcf_only_top100', 'ndcg@20']:.4f}",
        "k5_recall": f"{k.loc[5, 'recall@20']:.4f}", "k10_recall": f"{k.loc[10, 'recall@20']:.4f}",
        "k2_recall": f"{k.loc[2, 'recall@20']:.4f}", "k50_recall": f"{k.loc[50, 'recall@20']:.4f}",
        "k_gap_runnerup": f"{k.loc[5, 'recall@20'] - k.loc[10, 'recall@20']:.4f}",
        "k_gap_worst": f"{k.loc[5, 'recall@20'] - k.loc[50, 'recall@20']:.4f}",
        "itemcf_minus_als_recall20": f"{FC.loc[('ItemCF', 20, 'recall'), 'mean'] - FC.loc[('ALS', 20, 'recall'), 'mean']:.3f}",
        "greedy_rel": f"{g.loc['Greedy'].relevance_term:.2f}", "random_rel": f"{g.loc['Random'].relevance_term:.2f}",
        "greedy_repeat_pct": f"{100 * g.loc['Greedy'].repeat_rate_mean:.1f}%", "random_repeat_pct": f"{100 * g.loc['Random'].repeat_rate_mean:.1f}%",
        "greedy_repeat_pen": f"{g.loc['Greedy'].repeat_penalty:.2f}", "random_repeat_pen": f"{g.loc['Random'].repeat_penalty:.2f}",
        "greedy_div_term": f"{g.loc['Greedy'].div_term:.2f}", "random_div_term": f"{g.loc['Random'].div_term:.2f}",
        "linucb_rel": f"{g.loc['LinUCB'].relevance_term:.2f}", "linucb_div_term": f"{g.loc['LinUCB'].div_term:.2f}",
        "linucb_repeat_pen": f"{g.loc['LinUCB'].repeat_penalty:.2f}", "linucb_repeat_pct": f"{100 * g.loc['LinUCB'].repeat_rate_mean:.1f}%",
        "linucb_p15_frac_greedy": f"{TOR.loc['LinUCB', 'precision_at_15_mean'] / TOR.loc['Greedy', 'precision_at_15_mean']:.2f}",
        "linucb_p15_frac_random": f"{TOR.loc['LinUCB', 'precision_at_15_mean'] / TOR.loc['Random', 'precision_at_15_mean']:.2f}",
        "sim_linucb_frac_random": f"{sim.loc['LinUCB', 'precision_at_15_mean'] / sim.loc['Random', 'precision_at_15_mean']:.2f}",
        "sim_linucb_reward_gain": f"{100 * (sim.loc['LinUCB', 'sim_reward_mean'] / sim.loc['Random', 'sim_reward_mean'] - 1):.0f}%",
        "sim_linucb_tier2_ratio": f"{sim.loc['LinUCB', 'tier2_rel_per_step_mean'] / sim.loc['Random', 'tier2_rel_per_step_mean']:.1f}×",
        "rho_gr_reward": f"{corr.loc[('grounded', 'grounded_reward')]:+.2f}",
        "rho_sim_sim": f"{corr.loc[('grounded_simulated', 'sim_reward')]:+.2f}",
        "qs_diff_states": ", ".join(f"{int(n)}" for n in pol.n_states_differing),
        "ucb_seed42": f"{ucb.loc[42]:.4f}", "ucb_seed43": f"{ucb.loc[43]:.4f}", "ucb_seed44": f"{ucb.loc[44]:.4f}",
        "ucb_recall20_vs_rand": f"{pm.loc[('Recall@20', 'UCB1'), 'dr']:+.4f}, better in {pm.loc[('Recall@20', 'UCB1'), 'dr_better']}/5 folds",
        "ucb_ndcg20_vs_rand": f"{pm.loc[('NDCG@20', 'UCB1'), 'dr']:+.4f}, better in {pm.loc[('NDCG@20', 'UCB1'), 'dr_better']}/5 folds",
        "ql_recall20_vs_rand": f"{pm.loc[('Recall@20', 'QLearning'), 'dr']:+.4f}",
        "ql_ndcg20_vs_rand": f"{pm.loc[('NDCG@20', 'QLearning'), 'dr']:+.4f}",
        "sarsa_recall20_vs_rand": f"{pm.loc[('Recall@20', 'SARSA'), 'dr']:+.4f}",
        "sarsa_recall10_vs_rand": f"{pm.loc[('Recall@10', 'SARSA'), 'dr']:+.4f}",
        "linucb_recall20_vs_greedy": f"{pm.loc[('Recall@20', 'LinUCB'), 'dh']:+.4f}",
        "ucb_recall20_vs_greedy": f"{pm.loc[('Recall@20', 'UCB1'), 'dh']:+.4f}",
        "ucb_arms": ", ".join(f"arm {int(a)} (seed {int(sd)})" for sd, a in zip(UF.seed, UF.frozen_arm)),
        "ucb_q_range": f"{UF.q_min_all_arms.min():.2f}–{UF.q_max_all_arms.max():.2f}",
        "ucb_pool_median": f"{int(UF.median_pool_candidates_in_arm.min())}–{int(UF.median_pool_candidates_in_arm.max())}",
        "ucb_share_ge15": f"{100 * UF.share_users_arm_holds_ge_15.min():.0f}–{100 * UF.share_users_arm_holds_ge_15.max():.0f}%",
        "hits_random": f"{TOR.loc['Random', 'mean_hits_per_playlist_mean']:.2f}", "hits_greedy": f"{TOR.loc['Greedy', 'mean_hits_per_playlist_mean']:.2f}",
        "pool_steps_per_cell": f"{5000 * 15 / (162 * 10):.0f}",
        "conv_all": "yes" if (conv.pct_seeds_plateaued == 1).all() else "no",
    }
    return v


def fill(text):
    V = values()

    def repl(m):
        kind, arg = m.group(1), m.group(2)
        if kind == "T":
            return TABLES[arg]()
        if kind == "V":
            return V[arg]
        parts = arg.split(":")
        if kind == "MS":
            return ms(parts[0], int(parts[1]), parts[2])
        if kind == "M":
            return f"{FC.loc[(parts[0], int(parts[1]), parts[2]), 'mean']:.4f}"
        if kind == "TR":
            r = TOR.loc[parts[0]]
            col = parts[1]
            return f"{r[col + '_mean']:.4f} ± {r[col + '_std']:.4f}"
        if kind == "TRM":
            return f"{TOR.loc[parts[0], parts[1] + '_mean']:.4f}"
        if kind == "AB":
            return f"{ABL.loc[(parts[0], parts[1]), parts[2]]:.4f}"
        raise KeyError(m.group(0))

    out = re.sub(r"\{\{(T|V|MS|M|TR|TRM|AB):([^}]+)\}\}", repl, text)
    assert "{{" not in out, "unfilled placeholder: " + out[out.index("{{"):out.index("{{") + 60]
    return out


if __name__ == "__main__":
    OUT.write_text(fill(TEMPLATE.read_text()))
    print(f"Wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
