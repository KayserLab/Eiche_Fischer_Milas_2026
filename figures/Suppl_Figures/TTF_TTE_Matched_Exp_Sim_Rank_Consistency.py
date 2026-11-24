#!/usr/bin/env python3
"""Rank agreement between exact matched experimental wells and DPD simulations.

Compute ranks within each treatment strategy and each source. Correlations
use only pairs with that event observed in both sources. Event-free endpoints
are marked separately. PDFs go to the supplementary share plot directory;
tables go to the corresponding supplementary data directory; inputs are unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import figtools as ft
from shared.data_paths import figure_plot_dir, figure_data_dir
from figures.Suppl_Figures.TTF_TTE_Matched_InVitro_DPD import (
    LABELS, EVENTS, pearson_correlation, pearson_statistics, format_pearson_p,
)
from figures.Suppl_Figures.plot_TTF_TTE_Matched_Simulation_Strategies import STRATEGIES
from figures.Suppl_Figures.plot_TTF_TTE_Matched_Simulation_Strategies import (
    pair_statistics, benjamini_hochberg,
)
from shared.strategy_colors import get_color

DEFAULT_INPUT = figure_data_dir(
    "Suppl_Figures", "ttf_tte_matched_analysis", "ttf_tte_matched", create=False
) / "matched_event_times.csv"
DEFAULT_OUTPUT = figure_data_dir(
    "Suppl_Figures", "ttf_tte_matched_analysis", "ttf_tte_matched_exp_sim_rank_consistency",
    create=False)


def prepare_pairs(pairs):
    required = {"event", "strategy", "initial_condition", "exp_sample", "exp_plate",
                "sim_sample", "match_status", "exp_time", "sim_time", "exp_observed",
                "sim_observed", "exp_reason", "sim_reason"}
    if required - set(pairs):
        raise ValueError(f"Missing columns: {sorted(required - set(pairs))}")
    pairs = pairs.loc[pairs["event"].isin(EVENTS) & pairs["strategy"].isin(STRATEGIES)].copy()
    if not pairs["match_status"].eq("exact_match").all():
        raise ValueError("Input contains non-exact matches")
    if not pairs["initial_condition"].eq(pairs["exp_sample"]).all():
        raise ValueError("Recorded simulation seed differs from experimental sample ID")
    if pairs.duplicated(["event", "strategy", "exp_plate", "exp_sample"]).any():
        raise ValueError("Repeated experimental well would inflate independent pair counts")
    if pairs.duplicated(["event", "strategy", "sim_sample"]).any():
        raise ValueError("Repeated simulation outcome")
    for prefix in ("exp", "sim"):
        pairs[f"{prefix}_time"] = pd.to_numeric(pairs[f"{prefix}_time"], errors="coerce")
        pairs[f"{prefix}_valid"] = (
            pairs[f"{prefix}_observed"].eq(True)
            & pairs[f"{prefix}_reason"].fillna("").eq("")
            & np.isfinite(pairs[f"{prefix}_time"]))
    pairs["both_observed"] = pairs["exp_valid"] & pairs["sim_valid"]
    return pairs


def rank_table(pairs):
    records = []
    for row in pairs.to_dict("records"):
        for source in ("exp", "sim"):
            valid = row[f"{source}_valid"]
            reason = row[f"{source}_reason"]
            status = ("observed" if valid else (
                "event_not_observed" if pd.isna(reason) or reason == "" else str(reason)))
            records.append({"event": row["event"], "strategy": row["strategy"],
                            "initial_condition": row["initial_condition"],
                            "exp_plate": row["exp_plate"], "exp_sample": row["exp_sample"],
                            "sim_sample": row["sim_sample"], "source": source,
                            "time": row[f"{source}_time"], "rank_eligible": valid,
                            "endpoint_status": status, "rank_percentile": np.nan})
    ranks = pd.DataFrame(records)
    for (_event, _strategy, _source), group in ranks.loc[ranks["rank_eligible"]].groupby(
            ["event", "strategy", "source"]):
        n = len(group)
        percentile = ((group["time"].rank(method="average") - 1) / (n - 1) * 100
                      if n > 1 else pd.Series(50.0, index=group.index))
        ranks.loc[group.index, "rank_percentile"] = percentile
    return ranks


def strategy_correlations(pairs, *, seed=20261028, n_bootstrap=9999, n_permutations=19999):
    rows = []
    for event_index, event in enumerate(EVENTS):
        for strategy_index, strategy in enumerate((*STRATEGIES, "All points")):
            group = pairs.loc[pairs["event"].eq(event)]
            if strategy != "All points":
                group = group.loc[group["strategy"].eq(strategy)]
            complete = group.loc[group["both_observed"]].sort_values("initial_condition")
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, 100, event_index, strategy_index]))
            stat = pair_statistics(complete["exp_time"].to_numpy(),
                                   complete["sim_time"].to_numpy(), rng=rng,
                                   n_bootstrap=n_bootstrap, n_permutations=n_permutations)
            rows.append({"event": event, "strategy": strategy, "n_exact_pairs": len(group),
                         "n_pairs_not_both_observed": len(group) - len(complete),
                         **pearson_statistics(complete["exp_time"], complete["sim_time"]),
                         **stat})
    table = pd.DataFrame(rows)
    table["q_bh_both_events"] = np.nan
    within = table["strategy"].ne("All points")
    table.loc[within, "q_bh_both_events"] = benjamini_hochberg(
        table.loc[within, "p_permutation_two_sided"].to_numpy())
    return table


def plot_ranks(ranks, output):
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    order, group_ranges = [], []
    for strategy in STRATEGIES:
        group = ranks.loc[ranks["strategy"].eq(strategy)]
        if group.empty:
            continue
        exp = group.loc[group["source"].eq("exp")]
        mean_rank = exp.groupby("initial_condition")["rank_percentile"].mean()
        seeds = mean_rank.sort_values(ascending=False, kind="stable", na_position="last").index.tolist()
        start = len(order)
        order.extend((strategy, seed) for seed in seeds)
        group_ranges.append((strategy, start, len(order)))
    index = pd.MultiIndex.from_tuples(order, names=["strategy", "initial_condition"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 7.3))
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("#eeeeee")
    for ax, event, letter in zip(axes, EVENTS, ("A", "B")):
        panel = ranks.loc[ranks["event"].eq(event)]
        matrix = panel.pivot(index=["strategy", "initial_condition"], columns="source",
                             values="rank_percentile").reindex(index=index, columns=["exp", "sim"])
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), vmin=0, vmax=100,
                          cmap=cmap, aspect="auto", interpolation="nearest")
        lookup = panel.set_index(["strategy", "initial_condition", "source"])
        for i, (strategy, condition) in enumerate(order):
            for j, source in enumerate(("exp", "sim")):
                key = (strategy, condition, source)
                if key not in lookup.index:
                    symbol = "-"
                else:
                    row = lookup.loc[key]
                    symbol = ("" if row["rank_eligible"] else
                              "C" if row["endpoint_status"] == "event_not_observed" else "x")
                if symbol:
                    ax.text(j, i, symbol, ha="center", va="center", fontsize=5.5, color="0.35")
        ax.set(xticks=[0, 1], xticklabels=["Exp", "Sim"],
               yticks=np.arange(len(order)),
               yticklabels=[condition for _, condition in order] if event == "TTF" else [])
        ax.set_title(f"{letter}   {event}", loc="left", fontsize=7, fontweight="bold", pad=8)
        ax.tick_params(length=0, axis="both")
        ax.tick_params(axis="y", labelsize=5.4)
        for tick, (strategy, _) in zip(ax.get_yticklabels(), order):
            tick.set_color(get_color(strategy))
        for strategy, start, end in group_ranges:
            if start:
                ax.axhline(start - 0.5, color="white", lw=2.2)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.4)
        ax.tick_params(which="minor", length=0)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Exact matched well (grouped by treatment strategy)", labelpad=4)
    fig.subplots_adjust(left=0.19, right=0.86, top=0.94, bottom=0.12, wspace=0.15)
    cax = fig.add_axes([0.89, 0.24, 0.018, 0.56])
    bar = fig.colorbar(image, cax=cax, ticks=[0, 25, 50, 75, 100])
    bar.set_label("Rank percentile within strategy and source", fontsize=6)
    fig.text(0.52, 0.065, "Higher percentile = longer observed TTF/TTE within that strategy and source.",
             ha="center", fontsize=6)
    fig.text(0.52, 0.038, "C: event not observed   x: invalid event data. Ranks do not measure absolute agreement.",
             ha="center", fontsize=6)
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)


def plot_agreement(table, output):
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.3))
    displayed = (*STRATEGIES, "All points")
    positions = np.arange(len(displayed))
    for ax, event, letter in zip(axes, EVENTS, ("A", "B")):
        panel = table.loc[table["event"].eq(event)].set_index("strategy")
        ticklabels = []
        ax.axvline(0, color="0.6", lw=0.7, linestyle="--", zorder=0)
        for i, strategy in enumerate(displayed):
            row = panel.loc[strategy]
            n = int(row["n_pairs"])
            ticklabels.append(f"{LABELS.get(strategy, strategy)}  (n={n})")
            if not np.isfinite(row["rho"]):
                label = "No exact matches" if row["n_exact_pairs"] == 0 else "Too few pairs / constant times"
                ax.text(0, i, label, ha="center", va="center", fontsize=5.4, color="0.4")
                continue
            color = "#222222" if strategy == "All points" else get_color(strategy)
            if np.isfinite(row["ci_low"]):
                ax.hlines(i, row["ci_low"], row["ci_high"], color=color, lw=1.2, zorder=1)
                ax.vlines([row["ci_low"], row["ci_high"]], i - 0.07, i + 0.07,
                          color=color, lw=0.8, zorder=1)
            ax.scatter(row["rho"], i, s=22, c=color, edgecolors="black", linewidths=0.4, zorder=2)
            label = f"rho={row['rho']:.2f}; r={row['pearson_r']:.2f}; p(r)={format_pearson_p(row['pearson_p_two_sided'])}"
            ax.text(0.98, i - 0.25, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=5.1)
        ax.axhline(len(STRATEGIES) - 0.5, color="0.8", lw=0.6)
        ax.set(yticks=positions, yticklabels=ticklabels, xlim=(-1.08, 1.08),
                ylim=(len(displayed) - 0.5, -0.6), xticks=[-1, -0.5, 0, 0.5, 1],
                xlabel="Exp-Sim correlation")
        ax.set_title(f"{letter}   {event}", loc="left", fontsize=7, fontweight="bold", pad=8)
        for tick, strategy in zip(ax.get_yticklabels(), displayed):
            tick.set_color("#222222" if strategy == "All points" else get_color(strategy))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.89, bottom=0.25, wspace=0.72)
    fig.text(0.5, 0.10, "Points/bars: Spearman rho and 95% CI. Labels r, p(r): Pearson r and unadjusted two-sided p.",
             ha="center", fontsize=6)
    fig.text(0.5, 0.055, "All points pools strategies; between-strategy differences can contribute to this association.",
             ha="center", fontsize=6)
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)


def write_methods(path, source, stats):
    path.write_text(f"""# Exact matched experiment-simulation rank agreement

Input: `{source}`. All new graphics are PDF. Only exact experimental sample IDs
and the same treatment strategy are accepted, using the existing matched table.
Experimental exclusions and valid-observation cutoffs are inherited from it.

Each rank heatmap cell is a within-event, within-strategy, within-source rank
percentile: 100 * (average-tie rank - 1) / (n_observed - 1). All valid observed
events in the exact-matched cohort are ranked independently for Exp and Sim.
A singleton rank is displayed at 50, without establishing relative performance.
An observed simulation can be colored even when its matched experiment is
censored. C is an unobserved event, x is invalid data; neither is ranked.
Rows are grouped by strategy (CT, AT50, AT100, AT+, ATagent) and ordered within
strategy by descending mean experimental rank across TTF/TTE. Both panels
have the same row order. Row colors identify strategies. ATagent has no
exact same-strategy experimental matches and thus has no heatmap rows.

Correlation compares Exp with Sim separately for each strategy and event,
using only exact pairs with valid observed events in BOTH sources. Times are
re-ranked within that jointly observed subset, so heatmap ranks and the
correlation subset can differ. An additional All points row pools the same
jointly observed pairs across strategies, as requested. Between-strategy time
differences can contribute to this pooled association, which is not a
strategy-adjusted estimate. Each event has one pooled result (n=20 TTF, n=25 TTE).
No cross-strategy correlations are attempted because each experimental well
received only its recorded treatment.

Pearson r is also computed from the same jointly observed event-time pairs
within each strategy. It measures linear association in days, while Spearman
rho compares ranks. Pearson r is suppressed below n=3 or for constant times.
Pearson values appear in the labels and CSV; the forest points and interval
bars continue to represent Spearman rho only. Labels p(r) report the unadjusted,
two-sided Pearson p-value from scipy.stats.pearsonr, under the usual independent
bivariate-normal null model. No Pearson confidence interval is added. Neither
coefficient measures absolute agreement. Pooled estimates are exploratory.

The resampling implementation is shared with the simulation-only analysis:
9,999 paired BCa bootstrap resamples, with a documented percentile fallback
when BCa is degenerate and enough bootstrap replicates remain finite;
19,999 two-sided Monte Carlo random-pairing permutations with a +1 correction.
The CSV records finite bootstrap counts and method. Rho is suppressed for
n<3 or constant times; confidence intervals are suppressed for n<4.
BH q-values adjust only the estimable within-strategy Spearman tests across
TTF and TTE; pooled rows are excluded from that family. Pearson p-values remain
unadjusted and are labelled as such. Confidence intervals are pointwise.

Small n, especially n=3 or 4, gives unstable estimates; n=2 is deliberately
shown as insufficient. Positive rho means wells retain their relative ordering,
not that absolute event times agree. One-to-one well/run uniqueness is checked
to avoid treating repeated uses of a well as independent observations.

These observed-event results do not account for censoring. Event-free
trajectories may represent better control; excluding them can bias rank
agreement. Weak agreement does not establish randomness. No survival model
or stochastic noise-vs-interaction decomposition is fitted here.

Statistical definitions:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-event-times-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-plot-dir", type=Path,
                        default=figure_plot_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False),
                        help="PDF destination; --output-dir stores analysis tables and methods.")
    args = parser.parse_args()
    pairs = prepare_pairs(pd.read_csv(args.matched_event_times_csv))
    ranks = rank_table(pairs)
    stats = strategy_correlations(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.output_dir / "matched_rank_percentiles_and_status.csv", index=False)
    stats.to_csv(args.output_dir / "exp_sim_rank_correlations_by_strategy.csv", index=False)
    pairs.loc[pairs["both_observed"]].to_csv(args.output_dir / "jointly_observed_pairs.csv", index=False)
    plot_ranks(ranks, args.output_plot_dir / "TTF_TTE_Matched_Exp_Sim_Rank_Heatmaps.pdf")
    plot_agreement(stats, args.output_plot_dir / "TTF_TTE_Matched_Exp_Sim_Rank_Correlations.pdf")
    write_methods(args.output_dir / "METHODS.md", args.matched_event_times_csv, stats)
    print(stats[["event", "strategy", "n_exact_pairs", "n_pairs", "rho", "pearson_r",
                 "ci_low", "ci_high", "p_permutation_two_sided"]].to_string(index=False))
    print(f"Saved PDFs to {args.output_plot_dir}; source tables to {args.output_dir}")


if __name__ == "__main__":
    main()
