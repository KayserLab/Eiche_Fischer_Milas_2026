#!/usr/bin/env python3
"""Plot well-weighted resistant-growth master curves from the pixel-store cache.

For every data set and growth definition, the figure contains:
* one thin median curve per well/simulation run;
* one colored median curve per strategy, with an interquartile band across
  well/run curves;
* one black master curve equal to the binwise mean of the strategy curves,
  with a min-max band across strategies.

Pixels from all eligible intervals are pooled within a well before its binned
median is calculated, so timepoints are not averaged first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from analysis.figure_3.Legacy import aggregate_well_weighted_master_curve as well_aggregation
from analysis.figure_3.Legacy.aggregate_well_weighted_master_curve import (
    PERIOD as WELL_CURVE_PERIOD,
    SUPPORT_MODE as WELL_CURVE_SUPPORT_MODE,
    X_BIN_WIDTH,
    ensure_well_curve_table,
)
from shared.data_paths import figure_data_dir, figure_plot_dir
from shared import figtools as ft
from figures.Figure_3.Figure_3_Legacy import plot_master_curve_sim_exp as master
from shared.strategy_colors import get_color

OUTPUT_DIR = figure_plot_dir(
    "Figure_3", "density_mastercurve_all", "well_weighted_master_curves", create=False,
)
if WELL_CURVE_SUPPORT_MODE == "one_cell_t0":
    OUTPUT_DIR = OUTPUT_DIR / "one_cell_t0"
STRATEGIES = ("CT", "AT50", "AT100")
DATASETS = (("simulation", True), ("experiment", False))
GROWTH_MODES = ("relative", "absolute")
PARTIAL_PIXEL_STORE = figure_data_dir(
    "Figure_3", "master_curve_160_pixel_store", create=False,
) / "master_curve_pixels.building.h5"
PARTIAL_CURVE_TABLE = figure_data_dir(
    "Figure_3", "master_curve_160_well_weighted_experiment_partial", create=False,
) / "well_curve_medians.tsv"

MIN_PIXELS_PER_WELL_BIN = 100
MIN_WELLS_PER_STRATEGY_BIN = 2
REQUIRE_ALL_STRATEGIES_FOR_MASTER = True

WELL_LINEWIDTH = 0.45
WELL_ALPHA = 0.5
STRATEGY_LINEWIDTH = 1.25
STRATEGY_BAND_ALPHA = 0.12
MASTER_LINEWIDTH = 2.2
MASTER_BAND_ALPHA = 0.18
# The well-weighted experimental absolute curves occupy a substantially tighter
# visible range than the pixel-weighted Figure 3 master curve.
WELL_WEIGHTED_ABSOLUTE_Y_LIMITS = (-0.05, 0.10)
WELL_WEIGHTED_FIGURE_HEIGHT_IN = 2.0
# Shared ranges for the completed simulation-versus-experiment export.  They
# contain every strategy IQR and the master min-max band in both datasets.
COMPARISON_Y_LIMITS = {
    "relative": (-0.10, 1.10),
    "absolute": (-0.01, 0.20),
}
BIN_TOKEN = f"{X_BIN_WIDTH:g}".replace(".", "p")
BIN_SUFFIX = "" if np.isclose(X_BIN_WIDTH, 0.10) else f"_xbin_{BIN_TOKEN}"
PERIOD_SUFFIX = {
    "before_escape": "",
    "after_escape": "_after_escape",
    "all_timepoints": "_all_timepoints",
}[WELL_CURVE_PERIOD]


def strategy_statistics(well_curves: pd.DataFrame) -> pd.DataFrame:
    """Take the median and IQR across equally weighted well/run curves."""
    eligible = well_curves[
        well_curves["n_pixels"] >= MIN_PIXELS_PER_WELL_BIN
        ].copy()
    rows: list[dict[str, object]] = []
    groups = ["strategy", "x_bin_center_cells_per_1000_um2"]
    for (strategy, center), group in eligible.groupby(groups, sort=True):
        values = group["median"].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) < MIN_WELLS_PER_STRATEGY_BIN:
            continue
        rows.append({
            "strategy": str(strategy),
            "x_bin_center_cells_per_1000_um2": float(center),
            "n_wells": int(len(values)),
            "center": float(np.median(values)),
            "lower": float(np.quantile(values, 0.25)),
            "upper": float(np.quantile(values, 0.75)),
        })
    return pd.DataFrame(rows)


def master_statistics(strategy_curves: pd.DataFrame) -> pd.DataFrame:
    """Take the binwise mean and min-max range across strategy medians."""
    rows: list[dict[str, object]] = []
    for center, group in strategy_curves.groupby(
            "x_bin_center_cells_per_1000_um2", sort=True
    ):
        group = group.drop_duplicates("strategy")
        values = group["center"].to_numpy(float)
        values = values[np.isfinite(values)]
        if REQUIRE_ALL_STRATEGIES_FOR_MASTER and len(values) != len(STRATEGIES):
            continue
        if len(values) < 2:
            continue
        rows.append({
            "x_bin_center_cells_per_1000_um2": float(center),
            "n_strategies": int(len(values)),
            "center": float(np.mean(values)),
            "lower": float(np.min(values)),
            "upper": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def _plot_contiguous(
        ax: plt.Axes,
        data: pd.DataFrame,
        y_column: str,
        *,
        color: str,
        linewidth: float,
        alpha: float,
        linestyle: str,
        zorder: int,
) -> None:
    """Plot separate segments when a curve has an empty density bin."""
    if data.empty:
        return
    ordered = data.sort_values("x_bin_center_cells_per_1000_um2")
    x = ordered["x_bin_center_cells_per_1000_um2"].to_numpy(float)
    y = ordered[y_column].to_numpy(float)
    breaks = np.flatnonzero(np.diff(x) > 1.5 * X_BIN_WIDTH) + 1
    for indices in np.split(np.arange(len(x)), breaks):
        if len(indices) < 2:
            continue
        ax.plot(
            x[indices],
            y[indices],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
        )


def _fill_contiguous(
        ax: plt.Axes,
        data: pd.DataFrame,
        *,
        color: str,
        alpha: float,
        zorder: int,
) -> None:
    if data.empty:
        return
    ordered = data.sort_values("x_bin_center_cells_per_1000_um2")
    x = ordered["x_bin_center_cells_per_1000_um2"].to_numpy(float)
    lower = ordered["lower"].to_numpy(float)
    upper = ordered["upper"].to_numpy(float)
    breaks = np.flatnonzero(np.diff(x) > 1.5 * X_BIN_WIDTH) + 1
    for indices in np.split(np.arange(len(x)), breaks):
        if len(indices) < 2:
            continue
        ax.fill_between(
            x[indices],
            lower[indices],
            upper[indices],
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )


def legend_handles(is_simulation: bool) -> list[Line2D]:
    linestyle = "--" if is_simulation else "-"
    handles = [
        Line2D(
            [0], [0],
            color="#777777",
            linewidth=WELL_LINEWIDTH,
            alpha=WELL_ALPHA,
            linestyle=linestyle,
            label="Individual well/run",
        )
    ]
    handles.extend(
        Line2D(
            [0], [0],
            color=get_color(strategy, "#666666"),
            linewidth=STRATEGY_LINEWIDTH,
            linestyle=linestyle,
            label=master.get_display_name(strategy),
        )
        for strategy in STRATEGIES
    )
    handles.append(
        Line2D(
            [0], [0],
            color="black",
            linewidth=MASTER_LINEWIDTH,
            linestyle=linestyle,
            label="Master",
        )
    )
    return handles


def write_plot(
        well_curves: pd.DataFrame,
        *,
        dataset: str,
        growth_mode: str,
        is_simulation: bool,
        y_limits: tuple[float, float] | None = None,
        output_dataset: str | None = None,
) -> Path:
    linestyle = "--" if is_simulation else "-"
    strategy_curves = strategy_statistics(well_curves)
    pooled_curve = master_statistics(strategy_curves)
    if strategy_curves.empty:
        raise RuntimeError(f"No strategy curves remain for {dataset}/{growth_mode}")
    if pooled_curve.empty:
        raise RuntimeError(f"No master-curve bins remain for {dataset}/{growth_mode}")

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.3,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    fig, ax = plt.subplots(
        figsize=(master.SINGLE_DATASET_FIGSIZE[0], WELL_WEIGHTED_FIGURE_HEIGHT_IN)
    )
    fig.subplots_adjust(
        left=master.DENSITY_MAP_GRID_LEFT,
        right=master.DENSITY_MAP_GRID_RIGHT,
        bottom=0.27,
        top=0.84,
    )

    eligible_wells = well_curves[
        well_curves["n_pixels"] >= MIN_PIXELS_PER_WELL_BIN
        ]
    for strategy in STRATEGIES:
        color = get_color(strategy, "#666666")
        strategy_wells = eligible_wells[
            eligible_wells["strategy"] == strategy
            ]
        for _, curve in strategy_wells.groupby("replicate", sort=False):
            _plot_contiguous(
                ax,
                curve,
                "median",
                color=color,
                linewidth=WELL_LINEWIDTH,
                alpha=WELL_ALPHA,
                linestyle=linestyle,
                zorder=1,
            )

        curve = strategy_curves[strategy_curves["strategy"] == strategy]
        _fill_contiguous(
            ax,
            curve,
            color=color,
            alpha=STRATEGY_BAND_ALPHA,
            zorder=3,
        )
        _plot_contiguous(
            ax,
            curve,
            "center",
            color=color,
            linewidth=STRATEGY_LINEWIDTH,
            alpha=1.0,
            linestyle=linestyle,
            zorder=4,
        )

    _fill_contiguous(
        ax,
        pooled_curve,
        color="black",
        alpha=MASTER_BAND_ALPHA,
        zorder=6,
    )
    _plot_contiguous(
        ax,
        pooled_curve,
        "center",
        color="black",
        linewidth=MASTER_LINEWIDTH,
        alpha=1.0,
        linestyle=linestyle,
        zorder=7,
    )

    ax.axvspan(
        master.USER_X_LIMITS[0],
        master.GREYED_X_MAX,
        color="white",
        alpha=1.0 - master.GREYED_ALPHA,
        linewidth=0,
        zorder=20,
        antialiased=False,
    )
    ax.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=30)
    ax.set_xlim(*master.USER_X_LIMITS)
    if y_limits is None:
        y_limits = (
            master.RELATIVE_Y_LIMITS
            if growth_mode == "relative"
            else WELL_WEIGHTED_ABSOLUTE_Y_LIMITS
        )
    ax.set_ylim(*y_limits)
    ax.set_xlabel(
        master.DENSITY_X_LABEL,
        fontsize=7,
        labelpad=master.XLABEL_PAD,
    )
    ax.set_ylabel(
        master.master_curve_ylabel(growth_mode, "resistant"),
        fontsize=7,
        labelpad=master.YLABEL_PAD,
        color=master.MASTER_CURVE_Y_AXIS_COLOR,
    )
    ft.format_axes(ax)
    ax.spines["left"].set_color(master.MASTER_CURVE_Y_AXIS_COLOR)
    ax.tick_params(axis="y", colors=master.MASTER_CURVE_Y_AXIS_COLOR)
    master.format_one_decimal_ticks(ax)
    if growth_mode == "absolute":
        lower, upper = y_limits
        ax.set_yticks(np.arange(lower, upper + 1.0e-9, 0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(
        handles=legend_handles(is_simulation),
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.35,
    )

    output_name = dataset if output_dataset is None else output_dataset
    output_dir = OUTPUT_DIR / output_name / growth_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_token = (
        "daily_intervals" if dataset == "simulation" else "experimental_intervals"
    )
    period_suffix = "" if output_dataset is not None else PERIOD_SUFFIX
    path = output_dir / (
        f"master_curve_{output_name}_{growth_mode}_well_weighted_"
        f"{interval_token}{period_suffix}{BIN_SUFFIX}.pdf"
    )
    ft.savefig(fig, path, transparent=False)
    plt.close(fig)
    return path


def partial_experiment_curve_table() -> Path:
    """Reduce experiment rows already present in a stopped partial pixel store."""
    if PARTIAL_CURVE_TABLE.is_file():
        return PARTIAL_CURVE_TABLE
    if not PARTIAL_PIXEL_STORE.is_file():
        raise FileNotFoundError(
            f"Missing {PARTIAL_PIXEL_STORE}. Stop the builder before using this mode."
        )
    curves = well_aggregation.build_well_curve_table_parallel(
        PARTIAL_PIXEL_STORE,
        ("experiment",),
    )
    PARTIAL_CURVE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PARTIAL_CURVE_TABLE.with_suffix(".building.tsv")
    curves.to_csv(temporary, sep="\t", index=False)
    temporary.replace(PARTIAL_CURVE_TABLE)
    return PARTIAL_CURVE_TABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-from-partial-store",
        action="store_true",
        help="Use experiment pixels from a stopped master_curve_pixels.building.h5.",
    )
    parser.add_argument(
        "--exclude-artificial-simulation-runs",
        action="store_true",
        help="Export simulation curves excluding artificial run_0 through run_9.",
    )
    parser.add_argument(
        "--one-cell-threshold",
        action="store_true",
        help=(
            "Require resistant density at the first endpoint to be at least "
            "one 160-um-kernel cell density; writes a separate output family."
        ),
    )
    period_group = parser.add_mutually_exclusive_group()
    period_group.add_argument(
        "--all-timepoints",
        action="store_true",
        help="Export curves from all timepoints, including post-escape intervals.",
    )
    period_group.add_argument(
        "--after-escape",
        action="store_true",
        help="Export curves from after-escape intervals only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_period = (
        "after_escape" if args.after_escape
        else "all_timepoints" if args.all_timepoints
        else "before_escape"
    )
    if args.experiment_from_partial_store:
        table_path = partial_experiment_curve_table()
        datasets = (("experiment", False),)
    else:
        table_path = ensure_well_curve_table()
        datasets = DATASETS

    curves = pd.read_csv(table_path, sep="\t")
    curves = curves[curves["strategy"].isin(STRATEGIES)]
    if args.exclude_artificial_simulation_runs:
        artificial = curves["replicate"].astype(str).str.contains(r"/run_[0-9]$")
        curves = curves[~((curves["dataset"] == "simulation") & artificial)]
        datasets = (("simulation", True),)
    comparison_limits = (
        COMPARISON_Y_LIMITS
        if (
            len(datasets) == 2
            or args.exclude_artificial_simulation_runs
            or args.all_timepoints
            or args.after_escape
        ) else None
    )

    for dataset, is_simulation in datasets:
        for growth_mode in GROWTH_MODES:
            selected = curves[
                (curves["dataset"] == dataset)
                & (curves["growth_mode"] == growth_mode)
                ]
            saved = write_plot(
                selected,
                dataset=dataset,
                growth_mode=growth_mode,
                is_simulation=is_simulation,
                y_limits=(
                    comparison_limits[growth_mode]
                    if comparison_limits is not None else None
                ),
                output_dataset=(
                    (
                        "simulation_after_no_artificial"
                        if args.after_escape
                        else ("simulation_all_no_artificial"
                              if args.all_timepoints else "simulation_no_artificial")
                    )
                    if args.exclude_artificial_simulation_runs
                    and dataset == "simulation"
                    else (
                        (f"{dataset}_after_escape" if args.after_escape
                         else f"{dataset}_all_timepoints" if args.all_timepoints
                         else None)
                    )
                ),
            )
            print(f"[saved] {saved}")


if __name__ == "__main__":
    main()
