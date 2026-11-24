#!/usr/bin/env python3
"""Day-1 cell counts in the retained exact-matched experimental cohorts.

Compare original experimental assignments using experimental day-1 counts.
Read matched simulation day-1 counts as an initialization consistency check.
No selection on later TTF/TTE occurrence is applied. Graphics are PDF only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir, resolve_stored_data_path
from analysis.figure_2.Event_Time_Analysis import _coordinate_count
from figures.Suppl_Figures.TTF_TTE_Matched_InVitro_DPD import LABELS
from figures.Suppl_Figures.TTF_TTE_Experimental_Cohort_Initial_Conditions import GROUPS
from figures.Suppl_Figures.TTF_TTE_Experimental_Cohort_Mean_Ranks import (
    median_range, median_difference, holm_adjust,
)
from shared.strategy_colors import get_color

BASE = figure_data_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False)
PLOT_DIR = figure_plot_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False)
METRICS = {
    "total_cells": ("Total cells", "Total cells [thousands]", 1000.0),
    "sensitive_cells": ("Sensitive cells", "Sensitive cells [thousands]", 1000.0),
    "resistant_cells": ("Resistant cells", "Resistant cells", 1.0),
    "resistant_percent": ("Resistant fraction", "Resistant cells [% of total]", 1.0),
}


def day1_cell_count(handle, group_name):
    if group_name not in handle:
        raise ValueError(f"Missing cell population group: {group_name}")
    population = handle[group_name]
    keys = []
    for key in population.keys():
        match = re.search(r"\d+", key)
        if match and int(match.group()) == 1:
            keys.append(key)
    if len(keys) != 1:
        raise ValueError(f"Expected one day-1 record for {group_name}; found {keys}")
    obj = population[keys[0]]
    if isinstance(obj, h5py.Group):
        if "block0_values" in obj:
            obj = obj["block0_values"]
        elif "values" in obj:
            obj = obj["values"]
        else:
            raise ValueError(f"No coordinate dataset for day-1 {group_name}")
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"Invalid day-1 coordinate dataset: {group_name}")
    return _coordinate_count(obj)


def read_day1(path):
    with h5py.File(path, "r") as handle:
        sensitive = day1_cell_count(handle, "sus")
        resistant = day1_cell_count(handle, "res")
    total = sensitive + resistant
    if total <= 0:
        raise ValueError(f"Nonpositive day-1 total: {path}")
    return {"total_cells": total, "sensitive_cells": sensitive,
            "resistant_cells": resistant, "resistant_percent": 100 * resistant / total,
            "resistant_fraction": resistant / total}


def load_matched_day1(pairs):
    if not pairs["match_status"].eq("exact_match").all():
        raise ValueError("Only exact experimental-simulation matches are allowed")
    if not pairs["initial_condition"].eq(pairs["exp_sample"]).all():
        raise ValueError("Initial-condition metadata and experimental sample disagree")
    columns = ["strategy", "initial_condition", "exp_plate", "exp_sample",
               "sim_sample", "exp_input", "sim_input"]
    wells = pairs[columns].drop_duplicates()
    if wells.duplicated(["exp_plate", "exp_sample"]).any():
        raise ValueError("An experimental well has more than one simulation match")
    if wells["initial_condition"].duplicated().any():
        raise ValueError("Repeated initial-condition identity")
    rows = []
    for index, well in enumerate(wells.to_dict("records"), 1):
        row = {**well, "experimental_strategy": well["strategy"], "acquisition_day": 1}
        for source in ("exp", "sim"):
            path = resolve_stored_data_path(well[f"{source}_input"])
            counts = read_day1(path)
            row.update({f"{source}_{name}": value for name, value in counts.items()})
        for metric in METRICS:
            row[f"sim_minus_exp_{metric}"] = row[f"sim_{metric}"] - row[f"exp_{metric}"]
        rows.append(row)
        if index % 10 == 0:
            print(f"Read day-1 counts for {index}/{len(wells)} matched wells", flush=True)
    return pd.DataFrame(rows)


def group_summary(counts):
    rows = []
    for metric in METRICS:
        for group in GROUPS:
            values = counts.loc[counts["experimental_strategy"].eq(group), f"exp_{metric}"]
            rows.append({"metric": metric, "experimental_strategy": group, "n_wells": len(values),
                         "median": values.median(), "mean": values.mean(),
                         "sd": values.std(ddof=1), "q25": values.quantile(0.25),
                         "q75": values.quantile(0.75), "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def comparisons(counts, n_permutations=19999, seed=20261028):
    global_rows, pair_rows = [], []
    for metric_index, metric in enumerate(METRICS):
        samples = [counts.loc[counts["experimental_strategy"].eq(g), f"exp_{metric}"].to_numpy()
                   for g in GROUPS]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 300, metric_index]))
        result = permutation_test(tuple(samples), median_range, vectorized=True,
                                  permutation_type="independent", n_resamples=n_permutations,
                                  batch=512, alternative="greater", random_state=rng)
        global_rows.append({"metric": metric, "n_wells": len(counts),
                            "range_of_group_medians": float(result.statistic),
                            "p_permutation": float(result.pvalue)})
        for i, a in enumerate(GROUPS):
            for j in range(i + 1, len(GROUPS)):
                b = GROUPS[j]
                rng = np.random.default_rng(np.random.SeedSequence([seed, 301, metric_index, i, j]))
                result = permutation_test((samples[i], samples[j]), median_difference, vectorized=True,
                                          permutation_type="independent", n_resamples=n_permutations,
                                          batch=512, alternative="two-sided", random_state=rng)
                pair_rows.append({
                    "metric": metric, "group_a": a, "group_b": b,
                    "n_a": len(samples[i]), "n_b": len(samples[j]),
                    "median_a_minus_b": float(result.statistic),
                    "p_permutation_two_sided": float(result.pvalue),
                    "n_resamples_used": len(result.null_distribution),
                    "test_mode": "exact" if math.comb(len(samples[i]) + len(samples[j]),
                                                     len(samples[i])) <= n_permutations else "Monte_Carlo",
                })
    global_tests, pairwise = pd.DataFrame(global_rows), pd.DataFrame(pair_rows)
    global_tests["p_holm_four_metrics"] = holm_adjust(global_tests["p_permutation"])
    pairwise["p_holm_all_24_comparisons"] = holm_adjust(pairwise["p_permutation_two_sided"])
    return global_tests, pairwise


def plot_counts(counts, summary, tests, output):
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.3))
    for ax, (metric, (title, ylabel, scale)), letter in zip(axes.ravel(), METRICS.items(), "ABCD"):
        values = counts[f"exp_{metric}"] / scale
        lower, upper = float(values.min()), float(values.max())
        span = max(upper - lower, 0.1 * upper, 0.01)
        plot_top = upper + 0.45 * span
        for x, group in enumerate(GROUPS):
            data = counts.loc[counts["experimental_strategy"].eq(group)].sort_values("initial_condition")
            y = data[f"exp_{metric}"] / scale
            offsets = np.linspace(-0.14, 0.14, len(data))
            ax.scatter(x + offsets, y, s=18, c=get_color(group),
                       edgecolors="black", linewidths=0.4, zorder=3)
            row = summary.loc[summary["metric"].eq(metric)
                              & summary["experimental_strategy"].eq(group)].iloc[0]
            median = row["median"] / scale
            ax.hlines(median, x - 0.24, x + 0.24, color="black", lw=1.5, zorder=4)
            median_label = (f"{median:.4f}" if metric == "resistant_percent" else
                            f"{median:.1f}" if metric == "resistant_cells" else f"{median:.2f}")
            ax.text(x, plot_top - 0.12 * span, f"{median_label}\nn={len(data)}",
                    ha="center", va="center", fontsize=5.7)
        p = tests.loc[tests["metric"].eq(metric), "p_permutation"].iloc[0]
        p_text = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
        ax.text(0.5, -0.31, f"Global median-comparison p={p_text}",
                transform=ax.transAxes, ha="center", fontsize=6)
        ax.set(xticks=np.arange(len(GROUPS)), xticklabels=[LABELS[g] for g in GROUPS],
               xlim=(-0.5, len(GROUPS) - 0.5),
               ylim=(max(0, lower - 0.12 * span), plot_top),
               xlabel="Original experimental strategy", ylabel=ylabel)
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=7, fontweight="bold")
        for tick, group in zip(ax.get_xticklabels(), GROUPS):
            tick.set_color(get_color(group))
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.055, "Experimental acquisition day 1. Each dot is one retained matched well; bars mark medians.",
             ha="center", fontsize=6)
    fig.text(0.5, 0.027, "Numbers above groups are medians in the axis units. Global p-values are unadjusted permutation tests.",
             ha="center", fontsize=6)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.17, hspace=0.70, wspace=0.35)
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)


def write_methods(path, counts):
    match_checks = []
    for metric in ("total_cells", "sensitive_cells", "resistant_cells"):
        difference = counts[f"sim_minus_exp_{metric}"]
        match_checks.append(f"- {metric}: {int(difference.eq(0).sum())}/{len(counts)} exactly equal; "
                            f"maximum absolute difference {difference.abs().max():g} cells.")
    path.write_text(f"""# Day-1 counts by original experimental assignment

Population: the same {len(counts)} retained experimental wells with exact
experimental-simulation initial-condition matches used in the cohort analyses.
Include all these wells regardless of subsequent TTF/TTE occurrence.
The original experimental ATagent group is not covered by this seed set.

Read sensitive (sus) and resistant (res) coordinates from exactly acquisition
day 1 in the canonical experimental HDF5. Counts use dataset shapes, with the
same coordinate-count convention as Event_Time_Analysis.py.
The experimental inputs are the prepared *_total_poisson_padded.h5 point
clouds used to initialize the matched simulations; this is not a count from
unprocessed microscope images.
Pandas-style block0_values/values datasets and direct datasets are supported. Missing day-1
records are rejected rather than substituted with the first available day.
No raw inputs are modified. An explicitly stored empty coordinate dataset
represents zero cells; a missing population/day is invalid.

Total = sensitive + resistant. Resistant percent is 100 * resistant / total,
computed per well before summarizing. The figure and group tests use the
EXPERIMENTAL counts grouped by original experimental treatment assignment.
Matched simulation day-1 counts are read only as an initialization check and
are exported alongside the experimental values:

{chr(10).join(match_checks)}

Descriptive tables contain means, medians, standard deviations, quartiles and
ranges. The PDF contains dots for individual wells and black median bars.
Total/sensitive panels display thousands of cells; resistant counts are shown
as individual cells. The CSV retains exact counts.
The resistant-fraction panel displays percent. More starting cells or a higher
resistant fraction need not imply worse outcomes on their own.

Exploratory global tests compare the range of the four group medians by
independent label permutations (19,999 draws). Pairwise tests use differences
of medians, two-sided. Small pairwise tests enumerate all allocations when
feasible; the CSV records exact versus Monte Carlo mode and resample count.
SciPy applies +1 corrections for randomized tests. Global p-values in the
figure are unadjusted; CSVs additionally report Holm adjustment across the
four global metrics and separately across all 24 pairwise comparisons.
The metrics are related; they are not four independent biological measurements.

The permutation null assumes exchangeable measurement distributions across
groups. This matched subset is selected and has 8-10 wells per group; results
do not establish causal effects or equivalence across the full experiment.
No outcomes or mean-rank scores were used to select these baseline records.

References:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-pairs-csv", type=Path,
                        default=BASE / "ttf_tte_matched/matched_event_times.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "experimental_cohort_day1_counts")
    parser.add_argument("--output-plot-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()
    counts = load_matched_day1(pd.read_csv(args.matched_pairs_csv))
    summary = group_summary(counts)
    tests, pairwise = comparisons(counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output_dir / "day1_counts_per_matched_well.csv", index=False)
    summary.to_csv(args.output_dir / "day1_counts_by_experimental_group.csv", index=False)
    tests.to_csv(args.output_dir / "global_day1_median_comparisons.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_day1_median_comparisons.csv", index=False)
    checks = pd.DataFrame([
        {"metric": metric, "n_wells": len(counts),
         "n_exact_exp_sim": int(counts[f"sim_minus_exp_{metric}"].eq(0).sum()),
         "max_absolute_difference": counts[f"sim_minus_exp_{metric}"].abs().max()}
        for metric in ("total_cells", "sensitive_cells", "resistant_cells")
    ])
    checks.to_csv(args.output_dir / "exp_sim_day1_initialization_check.csv", index=False)
    plot_counts(counts, summary, tests, args.output_plot_dir / "Day1_Cell_Counts_By_Experimental_Group.pdf")
    write_methods(args.output_dir / "METHODS.md", counts)
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))
    print(checks.to_string(index=False))
    print(pairwise.loc[pairwise["p_permutation_two_sided"] < 0.05].to_string(index=False))


if __name__ == "__main__":
    main()
