#!/usr/bin/env python3
"""Compare the simulated AT+ and AT+100 schedules by KM, TTF and TTE.

Failure times come from the prepared point clouds. Escape times come from
the Figure 2 signed escape-margin table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import figtools as ft
from shared.data_paths import COMBINED_RESULTS_CSV, figure_data_dir, figure_plot_dir
from analysis.figure_2.Event_Time_Analysis import (
    ESCAPE_MARGIN_CSV,
    SIMULATION_ROOT,
    derive_tte,
    kaplan_meier_median,
    load_escape_margins,
    load_population_counts,
    simulation_failure_times,
)
from analysis.shared_initial_condition_statistics import (
    format_survival_contrast,
    initial_condition,
    survival_contrasts,
)
from shared.strategy_colors import get_color, get_display_name

OLD_ATPLUS = "ATplusdayone"
NEW_ATPLUS = "AT100+"  # canonical name for raw strategy Dens
STRATEGIES = (NEW_ATPLUS, OLD_ATPLUS)
LABELS = {s: get_display_name(s, superscript=True) for s in STRATEGIES}
COLORS = {s: get_color(s) for s in STRATEGIES}
DEFAULT_OUTPUT = figure_plot_dir("Suppl_Figures", "km_simulation_atplus", create=False)
OUTPUT_PDF = "km_ttf_tte_simulation_atplus.pdf"
MEDIAN_COLOR = "#b000b5"


def load_events(simulation_root: Path = SIMULATION_ROOT, min_run: int = 10) -> pd.DataFrame:
    """Use the main-figure endpoint for all six cohorts, retaining input paths."""
    counts = load_population_counts(
        dataset_type="simulation", strategies=STRATEGIES,
        simulation_root=simulation_root, simulation_min_run=min_run,
    )
    if counts.empty:
        raise ValueError(f"No prepared simulation trajectories found under {simulation_root}")
    values = simulation_failure_times(
        STRATEGIES, simulation_min_run=min_run, counts=counts,
    ).rename(columns={"ttp": "time", "progressed": "event"})
    missing = set(STRATEGIES) - set(values.strategy)
    if missing:
        raise ValueError(f"Missing eligible simulation cohorts: {sorted(missing)}. "
                         "Prepare both data-DB_ATplusdayone and data-DB_Dens; "
                         "each run needs a day-1 baseline and follow-up.")
    values["event"] = values.event.astype(int)
    values["sample_key"] = values["sample"]
    return values


def load_escape_events(path: Path, min_run: int) -> pd.DataFrame:
    values = derive_tte(
        load_escape_margins(path), dataset_type="simulation", strategies=STRATEGIES,
        simulation_min_run=min_run,
    )
    missing = set(STRATEGIES) - set(values.strategy)
    if missing:
        raise ValueError(f"Missing eligible TTE cohorts in {path}: {sorted(missing)}. "
                         "Refresh the signed escape-margin table for both AT+ cohorts.")
    return values.rename(columns={"tte": "time", "escaped": "event"})


def endpoint_statistics(values: pd.DataFrame, overview_csv: Path) -> list[dict]:
    simulation = values.assign(initial_condition=[
        initial_condition(strategy, sample, path=overview_csv)
        for strategy, sample in zip(values.strategy, values["sample"])
    ])
    return survival_contrasts(
        values.iloc[:0], simulation, time="time", event="event", strategies=STRATEGIES,
    )


def km_curve(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Product-limit estimate; tied failures precede removal of censored runs."""
    at_risk, survival = len(group), 1.0
    times, fractions = [0.0], [1.0]
    for time, observations in group.groupby("time", sort=True):
        survival *= 1.0 - observations.event.sum() / at_risk
        times.append(float(time))
        fractions.append(float(survival))
        at_risk -= len(observations)
    return np.asarray(times), np.asarray(fractions)


def plot_km(ax, values: pd.DataFrame, x_max: float) -> None:
    for strategy in STRATEGIES:
        group = values.loc[values.strategy == strategy]
        times, survival = km_curve(group)
        ax.step(times, survival, where="post", color=COLORS[strategy],
                linestyle="--",
                linewidth=1.5, label=f"{LABELS[strategy]} (n={len(group)})")
        censored = group.loc[group.event == 0, "time"].to_numpy()
        ax.plot(censored, survival[np.searchsorted(times, censored, side="right") - 1],
                linestyle="none", marker="|", markersize=5, color=COLORS[strategy])
    ax.set(xlabel="Acquisition day", ylabel="Failure-free fraction",
           xlim=(0, x_max), ylim=(0, 1.05))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, borderaxespad=0)


def plot_event_times(ax, values: pd.DataFrame, ylabel: str, p: float, y_max: float) -> None:
    rng = np.random.default_rng(0)
    for x, strategy in enumerate(STRATEGIES, start=1):
        group = values.loc[values.strategy == strategy]
        observed = group.loc[group.event.astype(bool), "time"]
        ax.scatter(x + rng.uniform(-0.08, 0.08, len(observed)), observed,
                   s=8, color=COLORS[strategy], alpha=0.85, marker="x", linewidths=0.8)
        median = kaplan_meier_median(group.time.to_numpy(), group.event.to_numpy())
        if np.isfinite(median):
            ax.hlines(median, x - 0.10, x + 0.10, color=MEDIAN_COLOR, linewidth=1.1, zorder=4)
        ax.annotate(f"(n={len(group)})", (x, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -20), textcoords="offset points", ha="center", va="top",
                    fontsize=6, color=COLORS[strategy])
    ax.set_xticks([1, 2], [LABELS[strategy] for strategy in STRATEGIES], va="baseline")
    ax.tick_params(axis="x", pad=8)
    for label, strategy in zip(ax.get_xticklabels(), STRATEGIES):
        label.set_color(COLORS[strategy])
    ax.set(xlim=(0.6, 2.4), ylim=(0, y_max), ylabel=ylabel)
    if p < 0.05:
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.hlines(0.91 * y_max, 1, 2, color="black", linewidth=0.9)
        ax.annotate(stars, (1.5, 0.91 * y_max), (0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5)


def plot(failure: pd.DataFrame, escape: pd.DataFrame, contrasts: dict,
         output_dir: Path, x_max: float, show: bool) -> None:
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = ft.new_subplots_variable(
        1, 3, ax_widths_mm=(58, 25, 25), ax_heights_mm=(40,),
        margins=(13, 3, 12, 10), wspace_mm=15, dpi=300, apply_style=False,
    )
    plot_km(axes[0], failure, x_max)
    last_day = max(failure.time.max(), escape.time.max())
    y_max = max(160, np.ceil(last_day / 18) * 20)
    for ax, endpoint, label, values in (
        (axes[1], "TTF", "Time to failure, TTF [d]", failure),
        (axes[2], "TTE", "Time to escape, TTE [d]", escape),
    ):
        rows = contrasts[endpoint]
        plot_event_times(ax, values, label, rows[0]["p"] if rows else np.nan, y_max)
    axes[2].legend(
        handles=[Line2D([], [], color=MEDIAN_COLOR, linewidth=1.1, label="KM median")],
        loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, borderaxespad=0,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output_dir / OUTPUT_PDF)
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", type=Path, default=SIMULATION_ROOT,
                        help="Shared prepared_pointclouds directory")
    parser.add_argument("--min-run", type=int, default=10,
                        help="Minimum run number for every cohort; Figure 4 uses 10.")
    parser.add_argument("--escape-margin-csv", type=Path, default=ESCAPE_MARGIN_CSV,
                        help="Figure 2 signed escape-margin table.")
    parser.add_argument("--simulation-overview", type=Path, default=COMBINED_RESULTS_CSV,
                        help="Simulation overview CSV with shared initial-condition IDs.")
    parser.add_argument("--x-max", type=float, default=130,
                        help="Displayed acquisition-day limit; does not change censoring.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-data-dir", type=Path, default=figure_data_dir(
        "Suppl_Figures", "km_simulation_atplus", create=False))
    parser.add_argument("--hide-figure", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.x_max) or args.x_max <= 0 or args.min_run < 0:
        parser.error("x-max must be positive and finite; min-run must be nonnegative")
    try:
        values = load_events(args.simulation_root, args.min_run)
        escape = load_escape_events(args.escape_margin_csv, args.min_run)
        contrasts = {
            "TTF": endpoint_statistics(values, args.simulation_overview),
            "TTE": endpoint_statistics(escape, args.simulation_overview),
        }
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    values["display_label"] = values.strategy.map(LABELS)
    values.to_csv(args.output_data_dir / "km_simulation_atplus_events.csv", index=False)
    summary = values.groupby("strategy", sort=False).agg(
        n=("event", "size"), failures=("event", "sum"), last_followup=("time", "max"))
    summary["censored"] = summary.n - summary.failures
    summary.to_csv(args.output_data_dir / "km_simulation_atplus_summary.csv")
    escape["display_label"] = escape.strategy.map(LABELS)
    escape.to_csv(args.output_data_dir / "tte_simulation_atplus_events.csv", index=False)
    statistics = []
    for endpoint, rows in contrasts.items():
        print(f"[STATS] {endpoint} (shared initial conditions)")
        for row in rows:
            print(format_survival_contrast(row))
            statistics.append({"endpoint": endpoint, **row})
        if not rows:
            print("  At least two runs per cohort are required for comparison.")
    pd.DataFrame(statistics).to_csv(
        args.output_data_dir / "ttf_tte_simulation_atplus_statistics.csv", index=False,
    )
    plot(values, escape, contrasts, args.output_dir, args.x_max, show=not args.hide_figure)
    print(f"Saved AT+ and AT+100 comparison to {args.output_dir / OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
