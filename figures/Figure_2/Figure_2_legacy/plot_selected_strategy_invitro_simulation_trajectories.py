"""Plot selected in-vitro and simulation confinement/escape trajectories together.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

This is a focused companion to
``plot_all_well_confinement_escape_trajectories.py``.  It keeps the original
trajectory construction and input conventions, but combines the experimental
and simulation datasets in one figure.  Strategy colors identify CT, AT50,
and AT100; solid lines are in-vitro data and dashed lines are simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from figures.Suppl_Figures.Legacy import plot_all_well_confinement_escape_trajectories as source
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name


DEFAULT_OUTPUT_DIR = (
    ft.paper_figure_data_dir(__file__)
    / "Selected_CT_AT50_AT100_InVitro_Simulation_Trajectories"
)
OUTPUT_PDF = "selected_CT_AT50_AT100_invitro_simulation_trajectories.pdf"
OUTPUT_CSV = "selected_CT_AT50_AT100_invitro_simulation_trajectories.csv"
AVERAGE_OUTPUT_PDF = "selected_CT_AT50_AT100_strategy_average_invitro_simulation.pdf"
AVERAGE_OUTPUT_CSV = "selected_CT_AT50_AT100_strategy_average_invitro_simulation.csv"
MEDIAN_OUTPUT_PDF = "selected_CT_AT50_AT100_strategy_median_invitro_simulation.pdf"
MEDIAN_OUTPUT_CSV = "selected_CT_AT50_AT100_strategy_median_invitro_simulation.csv"
CARRY_FORWARD_OUTPUT_PDF = "selected_CT_AT50_AT100_invitro_simulation_trajectories_carry_forward.pdf"
CARRY_FORWARD_OUTPUT_CSV = "selected_CT_AT50_AT100_invitro_simulation_trajectories_carry_forward.csv"
CARRY_FORWARD_AVERAGE_OUTPUT_PDF = "selected_CT_AT50_AT100_strategy_average_invitro_simulation_carry_forward.pdf"
CARRY_FORWARD_AVERAGE_OUTPUT_CSV = "selected_CT_AT50_AT100_strategy_average_invitro_simulation_carry_forward.csv"
CARRY_FORWARD_MEDIAN_OUTPUT_PDF = "selected_CT_AT50_AT100_strategy_median_invitro_simulation_carry_forward.pdf"
CARRY_FORWARD_MEDIAN_OUTPUT_CSV = "selected_CT_AT50_AT100_strategy_median_invitro_simulation_carry_forward.csv"

SELECTED_STRATEGIES = {"CT", "AT50", "AT100"}
X_MIN = 0.0
X_MAX = 125.0
Y_MIN = -80000.0
Y_MAX = 20000.0

FIGURE_AXES_WIDTH_MM = 85.0
FIGURE_AXES_HEIGHT_MM = 55.0
FIGURE_MARGINS_MM = (18.0, 5.0, 13.0, 5.0)
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
LINEWIDTH = 0.75
LINE_ALPHA = 0.42
ZERO_LINEWIDTH = 0.8
AVERAGE_LINEWIDTH = 1.5
UNCERTAINTY_ALPHA = 0.16

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CT, AT50, and AT100 confinement/escape trajectories for "
            "in-vitro and simulation data in one figure."
        )
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=source.DEFAULT_SIMULATION_ROOT,
        help=f"Simulation density run. Default: {source.DEFAULT_SIMULATION_ROOT}",
    )
    parser.add_argument(
        "--simulation-quantification-dir",
        type=Path,
        default=None,
        help=(
            "Simulation aggregate CSV directory. Default: "
            "<simulation-root>/Escape_Quantification."
        ),
    )
    parser.add_argument(
        "--confinement-csv",
        type=Path,
        default=source.DEFAULT_CONFINEMENT_CSV,
        help=f"Per-well in-vitro confinement table. Default: {source.DEFAULT_CONFINEMENT_CSV}",
    )
    parser.add_argument(
        "--escape-summary-csv",
        type=Path,
        default=source.DEFAULT_ESCAPE_SUMMARY_CSV,
        help=f"In-vitro escape summary. Default: {source.DEFAULT_ESCAPE_SUMMARY_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Ignore full exclusions and cutoff days from shared/experiment_filters.py.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def load_trajectories(args: argparse.Namespace) -> pd.DataFrame:
    """Load, filter, and combine the two datasets using the original pipeline."""
    experimental_confinement = source.load_confinement(
        args.confinement_csv,
        args.include_excluded,
    )
    experimental_metadata, experimental_escaped = source.load_escape_data(
        args.escape_summary_csv,
        args.include_excluded,
    )
    experimental = source.build_signed_trajectories(
        experimental_confinement,
        experimental_metadata,
        experimental_escaped,
    )
    experimental = source.carry_forward_truncated_wells(
        experimental,
        enabled=True,
    )
    experimental["dataset_type"] = "In vitro"

    quantification_dir = args.simulation_quantification_dir or (
        args.simulation_root / source.SIMULATION_QUANTIFICATION_SUBDIR
    )
    simulation_confinement = source.load_confinement(
        quantification_dir / source.SIMULATION_CONFINEMENT_FILENAME,
        include_excluded=True,
    )
    simulation_metadata, simulation_escaped = source.load_simulation_escape_data(
        quantification_dir / source.SIMULATION_ESCAPE_COUNTS_FILENAME
    )
    simulation_confinement = source.filter_simulation_initial_condition_runs(
        simulation_confinement
    )
    simulation_metadata = source.filter_simulation_initial_condition_runs(
        simulation_metadata
    )
    simulation_escaped = source.filter_simulation_initial_condition_runs(
        simulation_escaped
    )
    simulation = source.build_signed_trajectories(
        simulation_confinement,
        simulation_metadata,
        simulation_escaped,
    )
    simulation = source.carry_forward_truncated_wells(
        simulation,
        enabled=True,
    )
    simulation["dataset_type"] = "Simulation"

    trajectories = pd.concat([experimental, simulation], ignore_index=True)
    trajectories = trajectories[
        trajectories["strategy"].isin(SELECTED_STRATEGIES)
    ].copy()
    if trajectories.empty:
        raise ValueError(
            "No CT, AT50, or AT100 trajectories remain after filtering."
        )
    return trajectories.sort_values(
        ["dataset_type", "strategy", "plate", "sample", "day", "phase"]
    )


def ordered_strategies(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    return [strategy for strategy in STRATEGY_ORDER if strategy in present]


def apply_axes_format(ax, *, ylabel: str = "Confinement thickness (+) / escaped cells (-)") -> None:
    ax.axhline(0.0, color="#222222", linewidth=ZERO_LINEWIDTH, zorder=3)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(direction="out", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_legends(ax, trajectories: pd.DataFrame) -> None:
    strategy_handles = [
        Line2D(
            [], [], color=get_color(strategy, "#666666"), linewidth=1.4,
            label=get_display_name(strategy),
        )
        for strategy in ordered_strategies(trajectories["strategy"])
    ]
    dataset_handles = [
        Line2D([], [], color="#333333", linewidth=1.2, linestyle="-", label="In vitro"),
        Line2D([], [], color="#333333", linewidth=1.2, linestyle="--", label="Simulation"),
    ]
    strategy_legend = ax.legend(
        handles=strategy_handles, loc="upper right", frameon=False,
        ncol=1, handlelength=1.5, columnspacing=0.9,
    )
    ax.add_artist(strategy_legend)
    ax.legend(
        handles=dataset_handles, loc="lower right", frameon=False,
        ncol=1, handlelength=1.5,
    )


def save_figure(fig, output_path: Path, show: bool) -> None:
    ft.savefig(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_trajectories(
    trajectories: pd.DataFrame,
    output_path: Path,
    show: bool,
    title: str,
    ylabel: str = "Confinement thickness (+) / escaped cells (-)",
) -> None:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM, height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM, apply_style=False,
    )
    if title:
        ax.set_title(title, loc="left", pad=4.0, fontsize=TEXT_SIZE_PT)
    for (_, _, _), group in trajectories.groupby(
        ["dataset_type", "plate", "sample"], sort=True,
    ):
        group = group.sort_values(["day", "phase"])
        strategy = str(group["strategy"].iloc[0])
        dataset_type = str(group["dataset_type"].iloc[0])
        ax.plot(
            group["day"].to_numpy(dtype=float),
            group["signed_value"].to_numpy(dtype=float),
            color=get_color(strategy, "#666666"), linewidth=LINEWIDTH,
            alpha=LINE_ALPHA,
            linestyle="--" if dataset_type == "Simulation" else "-",
            solid_capstyle="round", dash_capstyle="round", zorder=2,
        )
    apply_axes_format(ax)
    add_legends(ax, trajectories)
    save_figure(fig, output_path, show)


def plot_summary(
    summary: pd.DataFrame,
    value_column: str,
    lower_column: str,
    upper_column: str,
    output_path: Path,
    show: bool,
    title: str,
    ylabel: str = "Confinement thickness (+) / escaped cells (-)",
) -> None:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM, height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM, apply_style=False,
    )
    if title:
        ax.set_title(title, loc="left", pad=4.0, fontsize=TEXT_SIZE_PT)
    for (dataset_type, strategy), group in summary.groupby(
        ["dataset_type", "strategy"], sort=True,
    ):
        group = group.sort_values("day")
        x = group["day"].to_numpy(dtype=float)
        color = get_color(str(strategy), "#666666")
        linestyle = "--" if dataset_type == "Simulation" else "-"
        ax.fill_between(
            x,
            group[lower_column].to_numpy(dtype=float),
            group[upper_column].to_numpy(dtype=float),
            color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1,
        )
        ax.plot(
            x, group[value_column].to_numpy(dtype=float), color=color,
            linewidth=AVERAGE_LINEWIDTH, linestyle=linestyle,
            solid_capstyle="round", dash_capstyle="round", zorder=2,
        )
    apply_axes_format(ax, ylabel=ylabel)
    add_legends(ax, summary)
    save_figure(fig, output_path, show)


def summarize_by_dataset(df: pd.DataFrame, summarize_function) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for dataset_type, group in df.groupby("dataset_type", sort=True):
        summary = summarize_function(group)
        summary["dataset_type"] = dataset_type
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def make_rank_carry_forward(
    raw: pd.DataFrame,
    include_excluded: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_type, group in raw.groupby("dataset_type", sort=True):
        carried = source.rank_carry_forward_truncated_wells(
            group,
            enabled=True,
            rank_filter_cutoffs=(dataset_type == "In vitro" and not include_excluded),
        )
        carried["dataset_type"] = dataset_type
        frames.append(carried)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    all_carry_forward = load_trajectories(args)
    raw = all_carry_forward[
        ~all_carry_forward["carried_forward"].astype(bool)
    ].copy()
    rank_carry_forward = make_rank_carry_forward(raw, args.include_excluded)
    summaries = {
        "average": summarize_by_dataset(raw, source.summarize_strategy_averages),
        "median": summarize_by_dataset(raw, source.summarize_strategy_medians),
        "carry_forward_average": summarize_by_dataset(
            all_carry_forward, source.summarize_strategy_averages,
        ),
        "carry_forward_median": summarize_by_dataset(
            rank_carry_forward, source.summarize_strategy_medians,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_frames = {
        OUTPUT_CSV: raw,
        CARRY_FORWARD_OUTPUT_CSV: all_carry_forward,
        AVERAGE_OUTPUT_CSV: summaries["average"],
        MEDIAN_OUTPUT_CSV: summaries["median"],
        CARRY_FORWARD_AVERAGE_OUTPUT_CSV: summaries["carry_forward_average"],
        CARRY_FORWARD_MEDIAN_OUTPUT_CSV: summaries["carry_forward_median"],
    }
    for filename, frame in output_frames.items():
        frame.to_csv(args.output_dir / filename, index=False)

    plot_trajectories(
        raw, args.output_dir / OUTPUT_PDF, args.show,
        "CT, AT50, and AT100: in-vitro and simulation",
    )
    plot_summary(
        summaries["average"], "mean_signed_value", "mean_minus_std", "mean_plus_std",
        args.output_dir / AVERAGE_OUTPUT_PDF, args.show,
        "CT, AT50, and AT100: strategy averages",
    )
    plot_summary(
        summaries["median"], "median_signed_value", "median_ci_lower", "median_ci_upper",
        args.output_dir / MEDIAN_OUTPUT_PDF, args.show,
        "CT, AT50, and AT100: strategy medians",
    )
    plot_trajectories(
        all_carry_forward, args.output_dir / CARRY_FORWARD_OUTPUT_PDF, args.show,
        "CT, AT50, and AT100: in-vitro and simulation, carry-forward",
    )
    plot_summary(
        summaries["carry_forward_average"], "mean_signed_value",
        "mean_minus_std", "mean_plus_std",
        args.output_dir / CARRY_FORWARD_AVERAGE_OUTPUT_PDF, args.show,
        "",
        "escaped cell (-) / confinement margin (+)",
    )
    plot_summary(
        summaries["carry_forward_median"], "median_signed_value",
        "median_ci_lower", "median_ci_upper",
        args.output_dir / CARRY_FORWARD_MEDIAN_OUTPUT_PDF, args.show,
        "CT, AT50, and AT100: strategy medians, carry-forward",
    )

    for filename in (*output_frames.keys(), OUTPUT_PDF, AVERAGE_OUTPUT_PDF,
                     MEDIAN_OUTPUT_PDF, CARRY_FORWARD_OUTPUT_PDF,
                     CARRY_FORWARD_AVERAGE_OUTPUT_PDF, CARRY_FORWARD_MEDIAN_OUTPUT_PDF):
        print(f"[saved] {args.output_dir / filename}")


if __name__ == "__main__":
    main()
