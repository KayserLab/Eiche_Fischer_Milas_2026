#!/usr/bin/env python3
"""Matched simulation event times and strategy-to-strategy rank correlations.

Each line joins the available observed events from one recorded
initial_condition across strategies. Missing or unobserved outcomes do not
exclude the other observed events. Single observed outcomes are shown as dots.
No experimental exclusions are applied to simulations.
The combined 2x2 figure has TTE left and TTF right, matched times above and
strategy-to-strategy correlation heatmaps below. PDFs go to the share,
and audit CSVs go to the corresponding supplementary figure-data directory.

Run directly to derive events from canonical inputs, or use --event-times-csv
to reuse the shared simulation_event_times.csv. Rank audits and pairwise
correlation statistics retain their existing supplementary data paths.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import figtools as ft
from analysis.figure_2.Event_Time_Analysis import (
    ESCAPE_MARGIN_CSV, SIMULATION_ROOT, SIMULATION_H5_SUFFIX, load_escape_margins,
)
from shared.data_paths import COMBINED_RESULTS_CSV, figure_plot_dir, figure_data_dir
from figures.Suppl_Figures.TTF_TTE_Matched_InVitro_DPD import (
    EVENTS, failure_record, escape_record,
)
from shared.strategy_colors import get_color, get_display_name, normalize_strategy

STRATEGIES = ("CT", "AT50", "AT100", "AT100+", "LV-Agent")
LABELS = {strategy: get_display_name(strategy, superscript=True) for strategy in STRATEGIES}


def discover_runs(metadata: pd.DataFrame, simulation_root: Path) -> pd.DataFrame:
    metadata = metadata[["protocol", "run", "initial_condition"]].drop_duplicates().copy()
    metadata["strategy"] = metadata["protocol"].map(normalize_strategy)
    metadata["run"] = metadata["run"].astype(str).str.replace(r"\.0$", "", regex=True)
    metadata = metadata[["strategy", "run", "initial_condition"]].drop_duplicates()
    if metadata.duplicated(["strategy", "run"]).any():
        raise ValueError("Conflicting simulation initial-condition metadata")
    lookup = metadata.set_index(["strategy", "run"])
    rows = []
    for path in sorted(simulation_root.glob(f"Plate_*/hdf5/*{SIMULATION_H5_SUFFIX}")):
        plate = path.parent.parent.name
        strategy = normalize_strategy(plate.removeprefix("Plate_"))
        if strategy not in STRATEGIES:
            continue
        sample = path.name.removesuffix(SIMULATION_H5_SUFFIX)
        match = re.match(r"^R(\d+)_", sample)
        if not match or int(match[1]) < 10:
            continue
        run = str(int(match[1]))
        seed, reason = "", ""
        if (strategy, run) not in lookup.index:
            reason = "missing_initial_condition_metadata"
        else:
            value = lookup.loc[(strategy, run), "initial_condition"]
            if pd.isna(value) or str(value).strip().lower() in {"", "unknown", "artificial"}:
                reason = "invalid_initial_condition_metadata"
            else:
                seed = str(value).strip()
        rows.append({"initial_condition": seed, "strategy": strategy, "run": run,
                     "plate": plate, "sample": sample, "input_file": str(path),
                     "metadata_status": reason or "included"})
    if not rows:
        raise RuntimeError("No simulation population inputs were found")
    runs = pd.DataFrame(rows)
    eligible = runs.loc[runs["metadata_status"].eq("included")]
    if eligible.duplicated(["initial_condition", "strategy"]).any():
        raise ValueError("Multiple runs for a seed/strategy; cannot choose one arbitrarily")
    return runs


def derive_events(runs: pd.DataFrame, margins: pd.DataFrame, horizon: float) -> pd.DataFrame:
    rows = []
    eligible = runs.loc[runs["metadata_status"].eq("included")]
    for index, run in enumerate(eligible.to_dict("records"), 1):
        args = ("simulation", run["plate"], run["sample"], run["strategy"], horizon)
        for event in EVENTS:
            record = (failure_record(Path(run["input_file"]), *args) if event == "TTF"
                      else escape_record(margins, *args))
            rows.append({**run, "event": event, "time": record.get("time", np.nan),
                         "observed": record.get("observed", False),
                         "last_day": record.get("last_day", np.nan),
                         "initial_total": record.get("initial_total", np.nan),
                         "event_status": record["reason"] or (
                             "observed" if record["observed"] else "event_not_observed")})
        if index % 20 == 0:
            print(f"Read {index}/{len(eligible)} simulation trajectories", flush=True)
    return pd.DataFrame(rows)


def select_observed_cohorts(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every valid observed event, allowing incomplete matched sets."""
    if values.duplicated(["event", "initial_condition", "strategy"]).any():
        raise ValueError("Duplicate seed/strategy/event outcomes")
    rows = []
    for event in EVENTS:
        for seed in sorted(values["initial_condition"].unique()):
            group = values.loc[values["event"].eq(event)
                               & values["initial_condition"].eq(seed)]
            present = set(group["strategy"])
            missing = [s for s in STRATEGIES if s not in present]
            valid = group.loc[group["observed"].eq(True)
                              & group["event_status"].eq("observed")
                              & np.isfinite(group["time"])]
            observed = set(valid["strategy"])
            unobserved = [s for s in STRATEGIES if s in present and s not in observed]
            complete = not missing and not unobserved
            rows.append({"event": event, "initial_condition": seed,
                          "included": not valid.empty, "complete_all_strategies": complete,
                          "n_strategies_present": len(present),
                         "n_events_observed": len(observed),
                         "missing_strategies": "; ".join(missing),
                         "unobserved_or_invalid_strategies": "; ".join(unobserved)})
    audit = pd.DataFrame(rows)
    annotated = values.merge(audit[["event", "initial_condition", "included"]],
                             on=["event", "initial_condition"], validate="many_to_one")
    selected = (annotated["observed"].eq(True)
                & annotated["event_status"].eq("observed")
                & np.isfinite(annotated["time"]))
    return annotated.loc[selected].copy(), audit


DEFAULT_INPUT = figure_data_dir(
    "Suppl_Figures", "ttf_tte_matched_analysis", "ttf_tte_matched_simulation_strategies",
    create=False) / "simulation_event_times.csv"
DEFAULT_OUTPUT = figure_data_dir(
    "Suppl_Figures", "ttf_tte_matched_analysis", "ttf_tte_simulation_rank_consistency",
    create=False)


def validate_values(values: pd.DataFrame) -> pd.DataFrame:
    required = {"event", "initial_condition", "strategy", "time", "observed", "event_status"}
    if required - set(values):
        raise ValueError(f"Missing columns: {sorted(required - set(values))}")
    values = values.loc[values["event"].isin(EVENTS)
                        & values["strategy"].isin(STRATEGIES)].copy()
    if values["initial_condition"].isna().any():
        raise ValueError("Missing initial-condition IDs")
    if values.duplicated(["event", "initial_condition", "strategy"]).any():
        raise ValueError("Duplicate event/initial-condition/strategy records")
    values["time"] = pd.to_numeric(values["time"], errors="coerce")
    values["rank_eligible"] = (values["observed"].eq(True)
                              & values["event_status"].eq("observed")
                              & np.isfinite(values["time"]))
    if not values["rank_eligible"].any():
        raise ValueError("No valid observed events")
    return values


def build_rank_table(values: pd.DataFrame) -> pd.DataFrame:
    rows = values.copy()
    rows["rank_percentile"] = np.nan
    for (_event, _strategy), group in rows.loc[rows["rank_eligible"]].groupby(
            ["event", "strategy"]):
        n = len(group)
        percentile = ((group["time"].rank(method="average") - 1) / (n - 1) * 100
                      if n > 1 else pd.Series(50.0, index=group.index))
        rows.loc[group.index, "rank_percentile"] = percentile
    return rows


def spearman_statistic(x, y, axis=-1):
    """Spearman's rho with average ranks, vectorized for paired resampling."""
    rx = rankdata(x, axis=axis)
    ry = rankdata(y, axis=axis)
    rx = rx - np.mean(rx, axis=axis, keepdims=True)
    ry = ry - np.mean(ry, axis=axis, keepdims=True)
    denominator = np.sqrt(np.sum(rx * rx, axis=axis) * np.sum(ry * ry, axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(np.sum(rx * ry, axis=axis) / denominator, -1, 1)


def pair_statistics(x, y, *, rng, n_bootstrap, n_permutations):
    n = len(x)
    result = {"n_pairs": n, "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
              "ci_method": "not_estimable", "n_finite_bootstrap": 0,
              "p_permutation_two_sided": np.nan}
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return result
    rho = float(spearman_statistic(x, y))
    result["rho"] = rho
    if n >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from scipy.stats import DegenerateDataWarning
            warnings.simplefilter("ignore", DegenerateDataWarning)
            boot = bootstrap((x, y), spearman_statistic, paired=True,
                             vectorized=True, axis=-1, n_resamples=n_bootstrap,
                             batch=512, confidence_level=0.95, method="BCa",
                             random_state=rng)
        finite = boot.bootstrap_distribution[np.isfinite(boot.bootstrap_distribution)]
        result["n_finite_bootstrap"] = len(finite)
        low, high = float(boot.confidence_interval.low), float(boot.confidence_interval.high)
        if np.isfinite(low) and np.isfinite(high):
            result.update(ci_low=low, ci_high=high, ci_method="paired_BCa_bootstrap")
        elif len(finite) >= max(100, int(0.95 * n_bootstrap)):
            low, high = np.quantile(finite, [0.025, 0.975])
            result.update(ci_low=float(low), ci_high=float(high),
                          ci_method="paired_percentile_bootstrap_fallback")
    # Randomize pairings of the average ranks. Re-ranking is unnecessary here
    # because permutation preserves the ranks and ties within each variable.
    rx = rankdata(x) - (n + 1) / 2
    ry = rankdata(y) - (n + 1) / 2
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    extreme = 0
    remaining = n_permutations
    while remaining:
        batch = min(512, remaining)
        shuffled = rng.permuted(np.broadcast_to(ry, (batch, n)), axis=1)
        null = np.sum(shuffled * rx, axis=1) / denominator
        extreme += int((np.abs(null) >= abs(rho) - 1e-12).sum())
        remaining -= batch
    result["p_permutation_two_sided"] = (extreme + 1) / (n_permutations + 1)
    return result


def benjamini_hochberg(pvalues):
    """Adjust the finite p-values as one family across both event types."""
    adjusted = np.full(len(pvalues), np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    order = finite[np.argsort(np.asarray(pvalues)[finite])]
    m = len(order)
    if m:
        raw = np.asarray(pvalues)[order] * m / np.arange(1, m + 1)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    return adjusted


def correlations(values, *, seed=20261028, n_bootstrap=9999, n_permutations=19999):
    rows, joined_rows = [], []
    for event_index, event in enumerate(EVENTS):
        observed = values.loc[values["event"].eq(event) & values["rank_eligible"]]
        for i, a in enumerate(STRATEGIES):
            for j in range(i + 1, len(STRATEGIES)):
                b = STRATEGIES[j]
                left = observed.loc[observed["strategy"].eq(a),
                                    ["initial_condition", "time"]]
                right = observed.loc[observed["strategy"].eq(b),
                                     ["initial_condition", "time"]]
                joined = left.merge(right, on="initial_condition", suffixes=("_a", "_b"),
                                    validate="one_to_one").sort_values("initial_condition")
                rng = np.random.default_rng(np.random.SeedSequence([seed, event_index, i, j]))
                stat = pair_statistics(joined["time_a"].to_numpy(),
                                       joined["time_b"].to_numpy(), rng=rng,
                                       n_bootstrap=n_bootstrap, n_permutations=n_permutations)
                rows.append({"event": event, "strategy_a": a, "strategy_b": b, **stat})
                for pair in joined.to_dict("records"):
                    joined_rows.append({"event": event, "strategy_a": a, "strategy_b": b, **pair})
    table = pd.DataFrame(rows)
    table["q_bh_both_events"] = benjamini_hochberg(
        table["p_permutation_two_sided"].to_numpy())
    return table, pd.DataFrame(joined_rows)


def color_strategy_ticks(ax):
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        if len(ticks) == len(STRATEGIES):
            for tick, strategy in zip(ticks, STRATEGIES):
                tick.set_color(get_color(strategy))


def draw_matched_events(ax, plotted, event, offsets, upper):
    panel = plotted.loc[plotted["event"].eq(event)]
    for initial, group in panel.groupby("initial_condition", sort=True):
        group = group.set_index("strategy")
        available = [strategy for strategy in STRATEGIES if strategy in group.index]
        group = group.loc[available]
        x = np.array([STRATEGIES.index(strategy) for strategy in available]) + offsets[initial]
        if len(group) > 1:
            ax.plot(x, group["time"], color="0.60", lw=0.55, alpha=0.45, zorder=1)
        ax.scatter(x, group["time"], s=17, c=[get_color(strategy) for strategy in available],
                   edgecolors="black", linewidths=0.4, alpha=0.9, zorder=2)
    ax.text(0.03, 0.97,
            f"{panel['initial_condition'].nunique()} initial conditions; {len(panel)} observed events",
            transform=ax.transAxes, va="top", fontsize=6)
    if panel.empty:
        ax.text(0.5, 0.5, "No observed events", transform=ax.transAxes,
                ha="center", va="center", fontsize=6)
    ax.set(xlim=(-0.4, len(STRATEGIES) - 0.6), ylim=(0, upper),
           xticks=np.arange(len(STRATEGIES)), xticklabels=[LABELS[s] for s in STRATEGIES],
           xlabel="Treatment strategy", ylabel=f"{event} in silico [d]")
    for tick, strategy in zip(ax.get_xticklabels(), STRATEGIES):
        tick.set_color(get_color(strategy))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")


def draw_strategy_correlations(ax, table, values, event):
    """The existing strategy-pair rho/CI matrix, drawn into a supplied axis."""
    panel = table.loc[table["event"].eq(event)]
    matrix = np.eye(len(STRATEGIES))
    annotations = {}
    for i, a in enumerate(STRATEGIES):
        n = int((values["event"].eq(event) & values["strategy"].eq(a)
                 & values["rank_eligible"]).sum())
        annotations[i, i] = f"1.00\nn={n}"
        for j in range(i + 1, len(STRATEGIES)):
            b = STRATEGIES[j]
            row = panel.loc[panel["strategy_a"].eq(a) & panel["strategy_b"].eq(b)].iloc[0]
            matrix[i, j] = matrix[j, i] = row["rho"]
            annotations[j, i] = (f"{row['rho']:.2f}\nn={row['n_pairs']}"
                                 if np.isfinite(row["rho"]) else f"NA\nn={row['n_pairs']}")
            annotations[i, j] = (f"[{row['ci_low']:.2f},\n{row['ci_high']:.2f}]"
                                 if np.isfinite(row["ci_low"]) else "CI: NA")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    heatmap = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1)
    for (i, j), label in annotations.items():
        ax.text(j, i, label, ha="center", va="center", fontsize=5.6,
                color="white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 0.65 else "black")
    ax.set(xticks=np.arange(len(STRATEGIES)), yticks=np.arange(len(STRATEGIES)),
           xticklabels=[LABELS[s] for s in STRATEGIES],
           yticklabels=[LABELS[s] for s in STRATEGIES])
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=45)
    color_strategy_ticks(ax)
    ax.set_xticks(np.arange(-0.5, len(STRATEGIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(STRATEGIES), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.spines[:].set_visible(False)
    return heatmap


def plot_matched_and_correlations(plotted, values, table, output):
    """Matched TTE/TTF times above their respective strategy correlations."""
    if Path(output).suffix.lower() != ".pdf":
        raise ValueError("This figure is exported as PDF only")
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    seeds = sorted(values["initial_condition"].unique())
    offsets = dict(zip(seeds, np.linspace(-0.065, 0.065, len(seeds))
                       if len(seeds) > 1 else np.zeros(len(seeds))))
    upper = (max(20, np.ceil(plotted["time"].max() * 1.15 / 20) * 20)
             if not plotted.empty else 100)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.4))
    for col, event in enumerate(("TTE", "TTF")):
        draw_matched_events(axes[0, col], plotted, event, offsets, upper)
        heatmap = draw_strategy_correlations(axes[1, col], table, values, event)
        for row, title in enumerate((f"{event}: matched initial conditions",
                                     f"{event}: strategy correlations")):
            axes[row, col].set_title(title, loc="left",
                                     fontsize=7, fontweight="bold", pad=8)
            axes[row, col].annotate("abcd"[2 * row + col], xy=(0, 1),
                                    xycoords="axes fraction", xytext=(-26, 8),
                                    textcoords="offset points", ha="left", va="baseline",
                                    fontsize=8, fontweight="bold", annotation_clip=False)
    fig.subplots_adjust(left=0.085, right=0.87, top=0.94, bottom=0.10,
                        wspace=0.24, hspace=0.26)
    matrix_box = axes[1, 1].get_position()
    cax = fig.add_axes([matrix_box.x1 + 0.015, matrix_box.y0, 0.016, matrix_box.height])
    bar = fig.colorbar(heatmap, cax=cax, ticks=[-1, 0, 1])
    bar.set_label("Spearman rho", fontsize=6, rotation=270, va="bottom")
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)


def write_methods(path, source, table, seed, n_bootstrap, n_permutations):
    text = f"""# Simulation initial-condition rank consistency

Source: `{source}`. Strategy order: CT, AT50, AT100, AT+, ATagent.
These are exploratory observed-event analyses, separately for TTF and TTE.

## Combined matched-event / rank figure

`TTF_TTE_Matched_Simulation_Strategies.pdf` has TTE in the left column and TTF
in the right column. Top panels (a, b) connect all available observed events from
the exact same initial condition across strategies, including incomplete
sets and single observed outcomes. Bottom panels (c, d) are strategy-by-strategy
Spearman correlation matrices for the corresponding endpoint. The lower
triangle shows rho and the number of exact paired initial conditions; the
upper triangle shows 95% paired-bootstrap confidence intervals. Matrix colors
encode rho on a shared -1 to +1 scale. Diagonals show self-correlation and the
number of observed events within each strategy. The figure contains no
well-by-well rank heatmaps. A shared colorbar is aligned beside the bottom row.
One combined PDF is produced.

## Retained rank audit

The rank CSV retains average-tie percentiles within each strategy and event:
100 * (rank - 1) / (n_observed - 1). A singleton gets 50. Unobserved or invalid
events have no rank. These audit ranks are separate from the pairwise
re-ranking used to calculate the strategy correlations below.

## Strategy correlations and uncertainty

For each of the ten strategy pairs and each event, join by the exact recorded
initial_condition. Keep only valid observed events under BOTH strategies;
other strategies may be missing. Compute Spearman rho by re-ranking the times
inside this pairwise subset, with average ranks for ties. Pairwise sample sizes
can differ; matrices assembled this way need not be positive semidefinite.

95% confidence intervals resample whole initial-condition pairs with replacement
({n_bootstrap} replicates), using BCa intervals. If a degenerate bootstrap makes
BCa unavailable, a percentile interval is used only if at least 95% and at
least 100 resamples are finite. The CSV records the interval method and finite
bootstrap count. No interval is reported below n=4 or for constant times.

Two-sided Monte Carlo permutation p-values shuffle one strategy's ranks among
the matched initial conditions ({n_permutations} random permutations). They
use (1 + count(abs(rho_permuted) >= abs(rho_observed))) / (1 + n_permutations).
The CSV also reports Benjamini-Hochberg q-values across all {len(table)} tested
event/strategy-pair comparisons. These are exploratory; no stars are plotted.
Seed: {seed}. The confidence intervals are pointwise, not simultaneous.

The heatmap lower triangles show rho and paired n; upper triangles show the
95% confidence limits. Diagonals show self-correlation and per-strategy n.

## Interpretation and limitations

A positive rho suggests initial conditions tend to keep their ordering across
that strategy pair among the jointly observed outcomes. Weak or negative
correlation can reflect strategy-specific responses, sampling uncertainty, or
simulation noise. It does not prove randomness. Only one run per seed/strategy
is available here; repeated stochastic runs would be needed to separate noise
from initial-condition-by-strategy interaction.

Because event-free outcomes can represent better control, excluding censored
endpoints can bias these observed-event correlations. Do not interpret them as
a censoring-adjusted analysis of all simulations. C is not assigned the last
observation day as if it were a known event. A survival model with a shared
initial-condition effect would be a separate analysis and has not been fitted.
Seeds are treated as exchangeable units for resampling; inference beyond these
selected initial conditions requires that assumption.

Definitions follow the existing figure: TTF is first acquisition day with
resistant population >= 50% of initial total; TTE is first positive signed
escape margin. Invalid baseline/event records remain excluded.

## Statistical references

- [SciPy Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
- [SciPy paired bootstrap and BCa](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [SciPy permutation of pairings](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=COMBINED_RESULTS_CSV)
    parser.add_argument("--escape-margin-csv", type=Path, default=ESCAPE_MARGIN_CSV)
    parser.add_argument("--simulation-root", type=Path, default=SIMULATION_ROOT)
    parser.add_argument("--event-times-csv", type=Path,
                        help="Use previously derived event times instead of rereading trajectories.")
    parser.add_argument("--output-dir", type=Path,
                        default=figure_data_dir("Suppl_Figures", "ttf_tte_matched_analysis",
                                                "ttf_tte_matched_simulation_strategies", create=False))
    parser.add_argument("--rank-output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Rank tables, pairwise statistics and methods; preserves existing data paths.")
    parser.add_argument("--output-plot-dir", type=Path,
                        default=figure_plot_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False))
    parser.add_argument("--horizon-days", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--bootstrap-resamples", type=int, default=9999)
    parser.add_argument("--permutation-resamples", type=int, default=19999)
    args = parser.parse_args()
    if not np.isfinite(args.horizon_days) or args.horizon_days <= 0:
        raise ValueError("--horizon-days must be finite and positive")
    if args.bootstrap_resamples < 100 or args.permutation_resamples < 100:
        raise ValueError("Use at least 100 resamples")
    runs = None
    if args.event_times_csv is not None:
        values = pd.read_csv(args.event_times_csv)
    else:
        metadata = pd.read_csv(args.metadata_csv, usecols=["protocol", "run", "initial_condition"],
                               dtype=str)
        runs = discover_runs(metadata, args.simulation_root)
        margins = load_escape_margins(args.escape_margin_csv)
        values = derive_events(runs, margins, args.horizon_days)
    if values.empty:
        raise RuntimeError("No simulation runs have usable initial-condition metadata")
    validated = validate_values(values)
    plotted, audit = select_observed_cohorts(validated)
    ranks = build_rank_table(validated)
    table, paired = correlations(validated, seed=args.seed,
                                 n_bootstrap=args.bootstrap_resamples,
                                 n_permutations=args.permutation_resamples)
    for directory in (args.output_dir, args.rank_output_dir, args.output_plot_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if runs is not None:
        runs.to_csv(args.output_dir / "simulation_run_audit.csv", index=False)
    # Keep the reusable event table independent of derived display/rank columns.
    values.to_csv(args.output_dir / "simulation_event_times.csv", index=False)
    audit.to_csv(args.output_dir / "cohort_audit.csv", index=False)
    plotted.to_csv(args.output_dir / "plotted_event_times.csv", index=False)
    summary = pd.DataFrame([
        {"event": event, "n_initial_conditions": plotted.loc[
            plotted["event"].eq(event), "initial_condition"].nunique(),
         "n_strategy_outcomes": int(plotted["event"].eq(event).sum())}
        for event in EVENTS])
    summary.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    ranks.to_csv(args.rank_output_dir / "rank_percentiles_and_endpoint_status.csv", index=False)
    table.to_csv(args.rank_output_dir / "pairwise_spearman_statistics.csv", index=False)
    paired.to_csv(args.rank_output_dir / "pairwise_matched_event_times.csv", index=False)
    pdf = args.output_plot_dir / "TTF_TTE_Matched_Simulation_Strategies.pdf"
    plot_matched_and_correlations(plotted, validated, table, pdf)
    write_methods(args.rank_output_dir / "METHODS.md",
                   args.event_times_csv or "Canonical simulation trajectories and escape margins",
                   table, args.seed, args.bootstrap_resamples, args.permutation_resamples)
    print(summary.to_string(index=False))
    print(f"Saved combined 2x2 figure to {pdf}")
    print(f"Rank statistics and methods: {args.rank_output_dir}")


if __name__ == "__main__":
    main()
