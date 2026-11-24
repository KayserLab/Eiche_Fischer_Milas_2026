#!/usr/bin/env python3
"""Compare median(mean within-strategy simulation ranks) by experimental cohort."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir
from figures.Suppl_Figures.TTF_TTE_Matched_InVitro_DPD import EVENTS, LABELS
from figures.Suppl_Figures.plot_TTF_TTE_Matched_Simulation_Strategies import (
    validate_values, build_rank_table,
)
from figures.Suppl_Figures.TTF_TTE_Experimental_Cohort_Initial_Conditions import (
    GROUPS, join_experimental_cohorts,
)
from shared.strategy_colors import get_color

BASE = figure_data_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False)
PLOT_DIR = figure_plot_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False)


def compute_scores(simulations, pairs):
    # Preserve the ranking population used for the existing simulation heatmap.
    ranks = build_rank_table(validate_values(simulations))
    ranks["raw_within_strategy_rank"] = np.nan
    for (_, _), group in ranks.loc[ranks["rank_eligible"]].groupby(["event", "strategy"]):
        ranks.loc[group.index, "raw_within_strategy_rank"] = group["time"].rank(method="average")
    cohorts, joined = join_experimental_cohorts(pairs, ranks)
    scores = joined.groupby(["event", "initial_condition", "experimental_strategy"], sort=True).agg(
        mean_rank_percentile=("rank_percentile", "mean"),
        mean_raw_rank=("raw_within_strategy_rank", "mean"),
        n_observed_strategies=("rank_eligible", "sum"),
        n_available_strategies=("strategy", "nunique"),
    ).reset_index()
    scores["score_status"] = np.where(scores["n_observed_strategies"] > 0,
                                      "included", "no_observed_events")
    return cohorts, joined, scores


def summarize_scores(scores):
    rows = []
    for event in EVENTS:
        for group in GROUPS:
            available = scores.loc[scores["event"].eq(event)
                                   & scores["experimental_strategy"].eq(group)]
            scored = available.loc[available["score_status"].eq("included")]
            values = scored["mean_rank_percentile"]
            rows.append({
                "event": event, "experimental_strategy": group,
                "n_initial_conditions": len(available), "n_scored": len(scored),
                "n_without_observed_events": len(available) - len(scored),
                "median_mean_rank_percentile": values.median(),
                "mean_mean_rank_percentile": values.mean(),
                "q25_mean_rank_percentile": values.quantile(0.25),
                "q75_mean_rank_percentile": values.quantile(0.75),
                "median_mean_raw_rank": scored["mean_raw_rank"].median(),
                "min_contributing_strategies": scored["n_observed_strategies"].min(),
                "median_contributing_strategies": scored["n_observed_strategies"].median(),
                "max_contributing_strategies": scored["n_observed_strategies"].max(),
            })
    return pd.DataFrame(rows)


def median_range(*samples, axis=-1):
    medians = np.stack([np.median(sample, axis=axis) for sample in samples], axis=0)
    return np.max(medians, axis=0) - np.min(medians, axis=0)


def median_difference(x, y, axis=-1):
    return np.median(x, axis=axis) - np.median(y, axis=axis)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    output = np.full(len(pvalues), np.nan)
    indices = np.flatnonzero(np.isfinite(pvalues))
    order = indices[np.argsort(pvalues[indices])]
    if len(order):
        output[order] = np.minimum(1, np.maximum.accumulate(
            pvalues[order] * np.arange(len(order), 0, -1)))
    return output


def compare_medians(scores, *, n_permutations=19999, seed=20261028):
    omnibus, pairwise = [], []
    for event_index, event in enumerate(EVENTS):
        data = scores.loc[scores["event"].eq(event) & scores["score_status"].eq("included")]
        groups = [g for g in GROUPS if data["experimental_strategy"].eq(g).any()]
        samples = [data.loc[data["experimental_strategy"].eq(g),
                            "mean_rank_percentile"].to_numpy() for g in groups]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 200, event_index]))
        result = permutation_test(tuple(samples), median_range,
                                  permutation_type="independent", vectorized=True,
                                  n_resamples=n_permutations, batch=512,
                                  alternative="greater", random_state=rng)
        omnibus.append({"event": event, "statistic": "range_of_group_medians",
                        "median_range_percentile_points": float(result.statistic),
                        "n_scored_initial_conditions": len(data), "n_groups": len(groups),
                        "n_permutations_used": len(result.null_distribution),
                        "p_permutation": float(result.pvalue)})
        for i, a in enumerate(groups):
            for j in range(i + 1, len(groups)):
                b = groups[j]
                rng = np.random.default_rng(np.random.SeedSequence([seed, 201, event_index, i, j]))
                test = permutation_test((samples[i], samples[j]), median_difference,
                                        permutation_type="independent", vectorized=True,
                                        n_resamples=n_permutations, batch=512,
                                        alternative="two-sided", random_state=rng)
                pairwise.append({
                    "event": event, "group_a": a, "group_b": b,
                    "n_a": len(samples[i]), "n_b": len(samples[j]),
                    "median_a_minus_b_percentile_points": float(test.statistic),
                    "n_resamples_used": len(test.null_distribution),
                    "test_mode": "exact" if math.comb(len(samples[i]) + len(samples[j]),
                                                       len(samples[i])) <= n_permutations else "Monte_Carlo",
                    "p_permutation_two_sided": float(test.pvalue),
                })
    omnibus, pairwise = pd.DataFrame(omnibus), pd.DataFrame(pairwise)
    omnibus["p_holm_two_events"] = holm_adjust(omnibus["p_permutation"])
    pairwise["p_holm_all_pairs_both_events"] = holm_adjust(pairwise["p_permutation_two_sided"])
    return omnibus, pairwise


def plot_scores(scores, summary, omnibus, output):
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.0), sharey=True)
    for ax, event, letter in zip(axes, EVENTS, ("A", "B")):
        panel = scores.loc[scores["event"].eq(event) & scores["score_status"].eq("included")]
        for x, group in enumerate(GROUPS):
            data = panel.loc[panel["experimental_strategy"].eq(group)].sort_values("initial_condition")
            offsets = np.linspace(-0.14, 0.14, len(data)) if len(data) > 1 else np.zeros(len(data))
            ax.scatter(x + offsets, data["mean_rank_percentile"],
                       s=8 + 6 * data["n_observed_strategies"],
                       c=get_color(group), edgecolors="black", linewidths=0.4, zorder=3)
            row = summary.loc[summary["event"].eq(event)
                              & summary["experimental_strategy"].eq(group)].iloc[0]
            median = row["median_mean_rank_percentile"]
            ax.hlines(median, x - 0.24, x + 0.24, color="black", lw=1.5, zorder=4)
            ax.text(x, 110, f"Median {median:.1f}\nn={int(row['n_scored'])}/{int(row['n_initial_conditions'])}",
                    ha="center", va="center", fontsize=5.8)
        p = omnibus.loc[omnibus["event"].eq(event), "p_permutation"].iloc[0]
        ax.text(0.5, -0.23, f"Global median-comparison p={p:.3f}",
                transform=ax.transAxes, ha="center", fontsize=6)
        ax.set(xticks=np.arange(len(GROUPS)), xticklabels=[LABELS[g] for g in GROUPS],
               xlim=(-0.5, len(GROUPS) - 0.5), ylim=(0, 120),
               yticks=[0, 25, 50, 75, 100],
               xlabel="Original experimental strategy")
        ax.set_title(f"{letter}   {event}", loc="left", fontsize=7, fontweight="bold")
        for tick, group in zip(ax.get_xticklabels(), GROUPS):
            tick.set_color(get_color(group))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean within-strategy simulation rank percentile")
    fig.text(0.5, 0.075, "Each dot: one initial condition; bar: group median. Higher = longer simulated event times.",
             ha="center", fontsize=6)
    fig.text(0.5, 0.033, "Dot size indicates 1-5 contributing strategies. Only observed event times receive ranks.",
             ha="center", fontsize=6)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.30, wspace=0.25)
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)


def write_methods(path, scores, summary):
    path.write_text(f"""# Median of mean simulation ranks by experimental assignment

For each event (TTF/TTE), the requested statistic is:

1. Rank initial conditions within each simulated strategy using observed times.
   Ties receive average ranks. Longer event times have higher ranks.
2. Normalize ranks to 0-100 with 100*(rank-1)/(n_observed-1), preserving the scale
   and ranking population of the existing simulation rank heatmap (including
   all available simulation seeds, before selecting retained experimental wells).
3. For each initial condition, take the arithmetic mean of its available
   within-strategy rank percentiles, equally weighting simulated strategies.
4. Link the initial condition to its exact retained experimental well and
   original experimental treatment assignment.
5. Compare the median of those per-condition mean rank percentiles between
   experimental strategy groups.

Scores are computed separately for TTF and TTE. The CSV also includes the mean
raw rank and median(mean raw rank) as a literal unscaled alternative; raw ranks
are less comparable because the strategies have different observed sample sizes.
A singleton within-strategy rank would receive percentile 50. No such special
case is interpreted as establishing relative performance.

All five simulated strategies are considered (CT, AT50, AT100, AT+, ATagent).
An initial condition need not have all five events observed: average only its
available observed ranks. One observed strategy is enough to obtain a score.
The number of available and contributing strategies is exported for every seed
and represented by dot size. Initial conditions with zero observed events
have no score and remain in the audit, not at zero. Censored and invalid
endpoints are never treated as known event times or assigned ranks.
The included score counts are:

{summary[['event', 'experimental_strategy', 'n_initial_conditions', 'n_scored', 'min_contributing_strategies', 'median_contributing_strategies', 'max_contributing_strategies']].to_string(index=False)}

The primary descriptive comparison is exactly the median of per-condition
mean ranks, not a median of all strategy runs pooled together. Each initial
condition contributes at most one score per event. Original experimental
ATagent membership is not represented by the exact matched seed set.

Global exploratory permutation tests use the range of group medians as their
statistic (max median minus min median), with up to 19,999 independent label
permutations, fixed group sizes, and an upper-tail test. Pairwise permutation
tests use median(group A) minus median(group B) with two-sided p-values.
Small pairwise tests enumerate every distinct allocation when feasible;
the CSV records exact versus Monte Carlo mode and the number of resamples.
SciPy uses its conservative +1 correction for randomized tests. Holm correction
is exported across the two global event tests, and separately across all
twelve pairwise/event comparisons. The figure shows unadjusted global p-values.
No claim of equivalence follows from a nonsignificant result.

The permutation null assumes exchangeable score distributions across original
experimental groups. This is a descriptive, selected matched cohort, not a
randomization analysis of the original allocation. Different strategy coverage
and informative event censoring can affect mean rank scores. In particular,
averaging different subsets of strategies is not a balanced five-strategy
comparison. These results are not censoring-adjusted and cannot establish
whether initial conditions were causally better/worse in vitro.

References:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-events-csv", type=Path,
                        default=BASE / "ttf_tte_matched_simulation_strategies/simulation_event_times.csv")
    parser.add_argument("--matched-pairs-csv", type=Path,
                        default=BASE / "ttf_tte_matched/matched_event_times.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "experimental_cohort_mean_ranks")
    parser.add_argument("--output-plot-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()
    cohorts, ranked_outcomes, scores = compute_scores(pd.read_csv(args.simulation_events_csv),
                                                      pd.read_csv(args.matched_pairs_csv))
    summary = summarize_scores(scores)
    omnibus, pairwise = compare_medians(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)
    ranked_outcomes.to_csv(args.output_dir / "within_strategy_rank_audit.csv", index=False)
    scores.to_csv(args.output_dir / "mean_ranks_per_initial_condition.csv", index=False)
    summary.to_csv(args.output_dir / "median_mean_ranks_by_experimental_group.csv", index=False)
    omnibus.to_csv(args.output_dir / "global_median_comparisons.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_median_comparisons.csv", index=False)
    plot_scores(scores, summary, omnibus,
                args.output_plot_dir / "TTF_TTE_Experimental_Cohorts_Mean_Ranks.pdf")
    write_methods(args.output_dir / "METHODS.md", scores, summary)
    print(summary.to_string(index=False))
    print(omnibus.to_string(index=False))
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
