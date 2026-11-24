#!/usr/bin/env python3
"""Retired pooled-pixel binned density-change summary plotting workflow.

For experimental and simulation runs, every resistant-density gate, relative
and absolute-per-day change, and overall/before/after-escape phase, this script
writes one multi-panel PDF containing:

- an all-strategy pooled-pixel mean +/- one sample standard deviation panel;
- an all-strategy pooled-pixel median with a 95% percentile-bootstrap interval
  panel;
- separate mean/SD and median/CI panels for every strategy.

The overview panels omit NT, KPP-Agent, and Pulse. Individual panels retain
all strategies present in each run. Relative-change figures use the displayed
Y range configured below; absolute-change figures retain the YAML scale.

All strategy lines share each panel. X bins have a fixed width of
0.2 cells (1000 µm$^2$)$^{-1}$ by default. Mean and SD are accumulated exactly. To avoid
holding hundreds of millions of pixel values in memory, median statistics are
evaluated from a 4096-level empirical Y histogram by default.

Pixels are treated as independent observations, as explicitly requested for
this analysis. The resulting uncertainty is therefore pixel-level rather than
replicate-level uncertainty. Every analysis is plotted twice: the primary
version omits bins below the configurable minimum pixel count, while the
mirrored all_bins version plots every nonempty bin.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.stats import beta
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.figure_3 import density_change_data_utils as base
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name as _get_display_name


def get_display_name(strategy: str) -> str:
    """Use the Figure 3 terminology without changing source-data keys."""
    display_name = _get_display_name(strategy)
    return {
        "LV-Agent": "ATagent",
        "AT100+": r"AT$^{+}$",
        r"AT100$^{+}$": r"AT$^{+}$",
    }.get(display_name, display_name)


DEFAULT_EXPERIMENTAL_RUN_DIR = Path(
    r"D:\2025_08_CoreSeed_Large_Testdataset\output\run_20260630_145305_320_32"
)
DEFAULT_SIMULATION_RUN_DIR = Path(
    r"F:\2025_08_CoreSeed_Simulation_Density_Output"
    r"\run_sim_20260709_145640_320_32"
)
DEFAULT_RUN_DIRS = (
    DEFAULT_EXPERIMENTAL_RUN_DIR,
    DEFAULT_SIMULATION_RUN_DIR,
)
DEFAULT_OUTPUT_SUBDIR = Path(
    "summary_plots/requested_variations_20260716/binned_strategy_lines"
)
DEFAULT_GATES = (
    (2.0e-5, 4.375e-4),
    (2.0e-5, 0.001),
    (2.0e-5, 0.002),
)
CHANGE_TYPES = ("rel", "abs_per_day")
PHASES = ("overall", "before_escape", "after_escape")
DEFAULT_X_BIN_SIZE = 0.2
DEFAULT_Y_HIST_BINS = 4096
DEFAULT_MIN_BIN_PIXELS = 100
OVERVIEW_STRATEGIES = tuple(
    strategy
    for strategy in STRATEGY_ORDER
    if strategy not in {"NT", "KPP-Agent", "Pulse"}
)
RELATIVE_DISPLAY_YLIM = (-0.25, 0.6)
SUMMARY_COLUMNS = (
    "phase",
    "change_type",
    "gate_lower_cells_per_um2",
    "gate_upper_cells_per_um2",
    "strategy",
    "x_bin_left_cells_per_1000_um2",
    "x_bin_right_cells_per_1000_um2",
    "x_bin_center_cells_per_1000_um2",
    "n_pixels",
    "mean",
    "sd",
    "median",
    "median_ci95_low",
    "median_ci95_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pooled-pixel mean/SD and median/95% CI lines by strategy "
            "for coarse-grained density-change maps."
        )
    )
    parser.add_argument(
        "--config",
        default=os.fspath(base.DEFAULT_CONFIG_PATH),
        help=f"Density-map YAML configuration. Default: {base.DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        default=None,
        help=(
            "Coarse-grained run directory. Repeat to process multiple runs. "
            "By default both the configured experimental and simulation runs "
            "are processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory override. Valid only with one --run-dir. "
            f"Default: <run-dir>/{DEFAULT_OUTPUT_SUBDIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--x-bin-size",
        type=float,
        default=DEFAULT_X_BIN_SIZE,
        help=(
            "Total-density X-bin width in cells (1000 µm$^2$)$^{-1}$. "
            f"Default: {DEFAULT_X_BIN_SIZE}"
        ),
    )
    parser.add_argument(
        "--y-hist-bins",
        type=int,
        default=DEFAULT_Y_HIST_BINS,
        help=(
            "Empirical Y-histogram resolution used for median and CI. "
            f"Default: {DEFAULT_Y_HIST_BINS}"
        ),
    )
    parser.add_argument(
        "--min-bin-pixels",
        type=int,
        default=DEFAULT_MIN_BIN_PIXELS,
        help=(
            "Minimum qualifying pixels for the primary plots. A second "
            "all_bins plot is always produced without this filter. "
            f"Default: {DEFAULT_MIN_BIN_PIXELS}"
        ),
    )
    parser.add_argument(
        "--annotate-n-pixels",
        action="store_true",
        help=(
            "Label points in the individual-strategy panels with compact "
            "qualifying-pixel counts (for example 4.9k or 3.2M)."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PHASES,
        default=list(PHASES),
        help="Phases to produce. Default: overall before_escape after_escape.",
    )
    parser.add_argument(
        "--experimental-escape-summary",
        default=None,
        help=(
            "Experimental escape summary CSV. By default it is resolved from "
            "the density-map YAML configuration."
        ),
    )
    parser.add_argument(
        "--simulation-escape-csv",
        default=None,
        help=(
            "Simulation CSV containing protocol, run, and TTE. Defaults to "
            "simulation_escape_csv in the YAML, then the original analysis "
            "script's default."
        ),
    )
    return parser.parse_args()


def gate_name(gate: tuple[float, float]) -> str:
    return base.density_window_token(gate).removeprefix("_")


def change_name(change_type: str) -> str:
    return "relative" if change_type == "rel" else "absolute_per_day"


def strategies_in_run(run_dir: Path) -> list[str]:
    found = {
        base.strategy_from_plate_and_well(
            changes_dir.parent.parent.name,
            changes_dir.parent.name,
        )
        for changes_dir in run_dir.glob("Plate_*/*/density_changes")
    }
    strategies = [strategy for strategy in STRATEGY_ORDER if strategy in found]
    if not strategies:
        raise RuntimeError(f"No recognized strategy density outputs found in {run_dir}")
    return strategies


def resolve_escape_cutoffs(
    run_dir: Path,
    config: dict,
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], tuple[float, float]], bool, Path]:
    simulation_run = base.is_simulation_density_run(run_dir)
    if simulation_run:
        configured = (
            args.simulation_escape_csv
            or config.get("simulation_escape_csv")
            or base.DEFAULT_SIMULATION_ESCAPE_CSV
        )
        source_path = Path(os.fspath(configured)).expanduser()
        cutoffs = base.load_simulation_escape_cutoffs(run_dir, source_path)
    else:
        if args.experimental_escape_summary:
            source_path = Path(args.experimental_escape_summary).expanduser()
        else:
            experiment_path = base.resolve_experiment_path(
                experiment_path=config.get("experiment_path"),
                mount_root=config.get("mount_root"),
                hdf5_path=config.get("hdf5_path"),
            )
            source_path = base.resolve_escape_summary_path(
                config,
                experiment_path,
            )
        cutoffs = base.load_escape_cutoffs(source_path)

    lookup = {
        (str(row.plate), str(row.well)): (
            float(row.escape_day),
            float(row.pre_escape_cutoff_day),
        )
        for row in cutoffs.itertuples(index=False)
    }
    return lookup, simulation_run, source_path


def interval_phases(
    *,
    selected_phases: tuple[str, ...],
    plate: str,
    well: str,
    first_day: float,
    next_day: float,
    escape_cutoffs: dict[tuple[str, str], tuple[float, float]],
    simulation_run: bool,
) -> tuple[str, ...]:
    phases = ["overall"] if "overall" in selected_phases else []
    cutoff = escape_cutoffs.get((plate, well))
    if cutoff is None:
        return tuple(phases)

    escape_day, pre_escape_cutoff_day = cutoff
    phase_allowed = True
    if not simulation_run and well in base.EXCLUDE_SAMPLES:
        sample_cutoff = base.EXCLUDE_SAMPLES[well]
        phase_allowed = (
            sample_cutoff is not None
            and np.isfinite(next_day)
            and next_day <= float(sample_cutoff)
        )
    if not phase_allowed:
        return tuple(phases)

    if (
        "before_escape" in selected_phases
        and np.isfinite(next_day)
        and np.isfinite(pre_escape_cutoff_day)
        and next_day <= pre_escape_cutoff_day
    ):
        phases.append("before_escape")
    if (
        "after_escape" in selected_phases
        and np.isfinite(first_day)
        and np.isfinite(escape_day)
        and first_day >= escape_day
    ):
        phases.append("after_escape")
    return tuple(phases)


def plot_settings(config: dict, change_type: str) -> dict[str, object]:
    spec = base.CHANGE_SPECS[change_type]
    return {
        "change_type": change_type,
        "xlim": base.parse_float_limits(
            config,
            "relative_density_change_scatter_xlim",
            base.DEFAULT_XLIM,
        ),
        "ylim": base.parse_float_limits(
            config,
            spec["ylim_key"],
            spec["default_ylim"],
        ),
        "xscale": base.parse_axis_scale(
            config,
            "relative_density_change_scatter_xscale",
        ),
        "yscale": base.parse_axis_scale(config, spec["yscale_key"]),
        "y_label": str(spec["ylabel"]),
    }


@dataclass
class Aggregator:
    x_edges: np.ndarray
    y_edges: np.ndarray
    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    histogram: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        n_strategies: int,
        x_edges: np.ndarray,
        ylim: tuple[float, float],
        n_y_bins: int,
    ) -> "Aggregator":
        n_x_bins = len(x_edges) - 1
        return cls(
            x_edges=x_edges,
            y_edges=np.linspace(float(ylim[0]), float(ylim[1]), n_y_bins + 1),
            count=np.zeros((n_strategies, n_x_bins), dtype=np.int64),
            total=np.zeros((n_strategies, n_x_bins), dtype=np.float64),
            total_sq=np.zeros((n_strategies, n_x_bins), dtype=np.float64),
            histogram=np.zeros(
                (n_strategies, n_x_bins, n_y_bins),
                dtype=np.int64,
            ),
        )

    @property
    def n_x_bins(self) -> int:
        return len(self.x_edges) - 1

    @property
    def n_y_bins(self) -> int:
        return len(self.y_edges) - 1

    def add(
        self,
        strategy_index: int,
        x_bins: np.ndarray,
        y_values: np.ndarray,
    ) -> None:
        if y_values.size == 0:
            return
        self.count[strategy_index] += np.bincount(
            x_bins,
            minlength=self.n_x_bins,
        )
        self.total[strategy_index] += np.bincount(
            x_bins,
            weights=y_values,
            minlength=self.n_x_bins,
        )
        self.total_sq[strategy_index] += np.bincount(
            x_bins,
            weights=y_values * y_values,
            minlength=self.n_x_bins,
        )

        y_min, y_max = self.y_edges[0], self.y_edges[-1]
        y_bins = np.floor(
            (y_values - y_min) * self.n_y_bins / (y_max - y_min)
        ).astype(np.int64)
        np.clip(y_bins, 0, self.n_y_bins - 1, out=y_bins)
        flat_bins = x_bins * self.n_y_bins + y_bins
        hist = np.bincount(
            flat_bins,
            minlength=self.n_x_bins * self.n_y_bins,
        )
        self.histogram[strategy_index] += hist.reshape(
            self.n_x_bins,
            self.n_y_bins,
        )


def histogram_quantile(
    counts: np.ndarray,
    edges: np.ndarray,
    quantile: float,
) -> float:
    n = int(counts.sum())
    if n == 0 or not np.isfinite(quantile):
        return np.nan
    q = float(np.clip(quantile, 0.0, 1.0))
    cumulative = np.cumsum(counts)
    target = q * max(n - 1, 0) + 1.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(counts) - 1)
    previous = int(cumulative[index - 1]) if index else 0
    within = int(counts[index])
    fraction = (
        0.5
        if within <= 0
        else float(np.clip((target - previous) / within, 0.0, 1.0))
    )
    return float(
        edges[index] + fraction * (edges[index + 1] - edges[index])
    )


def summarize(
    aggregator: Aggregator,
    strategies: list[str],
    gate: tuple[float, float],
    change_type: str,
    phase: str,
) -> pd.DataFrame:
    x_centers = 0.5 * (aggregator.x_edges[:-1] + aggregator.x_edges[1:])
    rows: list[dict[str, object]] = []
    for strategy_index, strategy in enumerate(strategies):
        for x_index, x_center in enumerate(x_centers):
            n = int(aggregator.count[strategy_index, x_index])
            if n == 0:
                continue
            total = float(aggregator.total[strategy_index, x_index])
            total_sq = float(aggregator.total_sq[strategy_index, x_index])
            mean = total / n
            variance = np.nan
            if n > 1:
                variance = max(
                    0.0,
                    (total_sq - total * total / n) / (n - 1),
                )
            sd = float(np.sqrt(variance)) if np.isfinite(variance) else np.nan
            hist = aggregator.histogram[strategy_index, x_index]
            median = histogram_quantile(hist, aggregator.y_edges, 0.5)
            if n > 1:
                shape = 0.5 * (n + 1.0)
                q_low, q_high = beta.ppf((0.025, 0.975), shape, shape)
                ci_low = histogram_quantile(hist, aggregator.y_edges, q_low)
                ci_high = histogram_quantile(hist, aggregator.y_edges, q_high)
            else:
                ci_low = median
                ci_high = median
            rows.append(
                {
                    "phase": phase,
                    "change_type": change_type,
                    "gate_lower_cells_per_um2": gate[0],
                    "gate_upper_cells_per_um2": gate[1],
                    "strategy": strategy,
                    "x_bin_left_cells_per_1000_um2": aggregator.x_edges[x_index],
                    "x_bin_right_cells_per_1000_um2": aggregator.x_edges[x_index + 1],
                    "x_bin_center_cells_per_1000_um2": x_center,
                    "n_pixels": n,
                    "mean": mean,
                    "sd": sd,
                    "median": median,
                    "median_ci95_low": ci_low,
                    "median_ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compact_count(value: int) -> str:
    """Format a non-negative count with a short, plot-friendly unit."""
    count = int(value)
    for scale, suffix in ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "k")):
        if count < scale:
            continue
        scaled = count / scale
        precision = 1 if scaled < 10 else 0
        return f"{scaled:.{precision}f}{suffix}"
    return str(count)


def annotate_pixel_counts(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    *,
    color: str,
    display_ylim: tuple[float, float],
) -> None:
    """Add compact count labels without expanding the data limits."""
    lower, upper = (float(value) for value in display_ylim)
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        return
    label_color = tuple(0.55 * channel for channel in to_rgb(color))
    for index, (x_value, y_value, count) in enumerate(zip(x, y, counts)):
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        if y_value > upper - 0.13 * span:
            place_above = False
        elif y_value < lower + 0.13 * span:
            place_above = True
        else:
            place_above = index % 2 == 0
        offset = 4.0 if place_above else -4.0
        annotation = ax.annotate(
            compact_count(int(count)),
            xy=(x_value, y_value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if place_above else "top",
            color=label_color,
            fontsize=3.4,
            clip_on=True,
            zorder=5,
        )
        annotation.set_path_effects(
            [path_effects.withStroke(linewidth=1.0, foreground="white")]
        )


def plot_summary(
    summary: pd.DataFrame,
    output_path: Path,
    settings: dict[str, object],
    gate: tuple[float, float],
    phase: str,
    strategies: list[str],
    min_bin_pixels: int,
    annotate_n_pixels: bool = False,
) -> None:
    base.ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=6,
        overrides={"axes.linewidth": 0.8, "lines.linewidth": 1.1},
    )
    n_strategy_rows = max(1, int(np.ceil(len(strategies) / 2)))
    fig = plt.figure(
        figsize=(7.2, 2.0 + 2.0 * n_strategy_rows),
        dpi=base.DPI,
    )
    grid = fig.add_gridspec(1 + n_strategy_rows, 4)
    ax_mean = fig.add_subplot(grid[0, :2])
    ax_median = fig.add_subplot(grid[0, 2:], sharex=ax_mean, sharey=ax_mean)
    strategy_axis_pairs = [
        (
            fig.add_subplot(
                grid[1 + index // 2, 2 * (index % 2)],
                sharex=ax_mean,
                sharey=ax_mean,
            ),
            fig.add_subplot(
                grid[1 + index // 2, 2 * (index % 2) + 1],
                sharex=ax_mean,
                sharey=ax_mean,
            ),
        )
        for index in range(len(strategies))
    ]
    strategy_axes = [
        ax
        for axis_pair in strategy_axis_pairs
        for ax in axis_pair
    ]
    all_axes = [ax_mean, ax_median, *strategy_axes]
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.065,
        top=0.875,
        wspace=0.34,
        hspace=0.62,
    )
    handles = []
    labels = []

    plotted = summary.copy()
    insufficient = plotted["n_pixels"] < int(min_bin_pixels)
    plotted.loc[
        insufficient,
        [
            "mean",
            "sd",
            "median",
            "median_ci95_low",
            "median_ci95_high",
        ],
    ] = np.nan

    for strategy in OVERVIEW_STRATEGIES:
        sub = plotted[plotted["strategy"] == strategy].sort_values(
            "x_bin_center_cells_per_1000_um2"
        )
        if sub.empty or not (
            np.isfinite(sub["mean"]).any()
            or np.isfinite(sub["median"]).any()
        ):
            continue
        x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        sd = sub["sd"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        ci_low = sub["median_ci95_low"].to_numpy(dtype=float)
        ci_high = sub["median_ci95_high"].to_numpy(dtype=float)
        color = get_color(strategy, "#000000")
        ax_mean.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.07,
            linewidth=0,
        )
        line, = ax_mean.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=2.0,
            linewidth=1.0,
            zorder=2,
        )
        ax_median.fill_between(
            x,
            ci_low,
            ci_high,
            color=color,
            alpha=0.11,
            linewidth=0,
        )
        ax_median.plot(
            x,
            median,
            color=color,
            marker="o",
            markersize=2.0,
            linewidth=1.0,
            zorder=2,
        )
        handles.append(line)
        labels.append(get_display_name(strategy))

    for strategy, (strategy_mean_ax, strategy_median_ax) in zip(
        strategies,
        strategy_axis_pairs,
    ):
        sub = plotted[plotted["strategy"] == strategy].sort_values(
            "x_bin_center_cells_per_1000_um2"
        )
        display_name = get_display_name(strategy)
        strategy_mean_ax.set_title(
            f"{display_name}: Mean +/- SD",
            fontsize=6.3,
            pad=2,
        )
        strategy_median_ax.set_title(
            f"{display_name}: Median with 95% CI",
            fontsize=6.3,
            pad=2,
        )
        if sub.empty or not (
            np.isfinite(sub["mean"]).any()
            or np.isfinite(sub["median"]).any()
        ):
            continue
        x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        sd = sub["sd"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        ci_low = sub["median_ci95_low"].to_numpy(dtype=float)
        ci_high = sub["median_ci95_high"].to_numpy(dtype=float)
        counts = sub["n_pixels"].to_numpy(dtype=np.int64)
        color = get_color(strategy, "#000000")
        strategy_mean_ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        strategy_mean_ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=2.0,
            linewidth=1.0,
        )
        strategy_median_ax.fill_between(
            x,
            ci_low,
            ci_high,
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        strategy_median_ax.plot(
            x,
            median,
            color=color,
            marker="o",
            markersize=2.0,
            linewidth=1.0,
        )
        if annotate_n_pixels:
            annotation_ylim = (
                RELATIVE_DISPLAY_YLIM
                if settings["change_type"] == "rel"
                else settings["ylim"]
            )
            annotate_pixel_counts(
                strategy_mean_ax,
                x,
                mean,
                counts,
                color=color,
                display_ylim=annotation_ylim,
            )
            annotate_pixel_counts(
                strategy_median_ax,
                x,
                median,
                counts,
                color=color,
                display_ylim=annotation_ylim,
            )

    display_ylim = (
        RELATIVE_DISPLAY_YLIM
        if settings["change_type"] == "rel"
        else settings["ylim"]
    )
    for ax in all_axes:
        ax.set_xscale(settings["xscale"])
        ax.set_yscale(settings["yscale"])
        ax.set_xlim(*settings["xlim"])
        ax.set_ylim(*display_ylim)
        if display_ylim[0] < 0 < display_ylim[1]:
            ax.axhline(
                0,
                color="#777777",
                linewidth=0.45,
                alpha=0.45,
                zorder=0,
            )
    ax_mean.set_xlabel(base.X_LABEL)
    ax_median.set_xlabel(base.X_LABEL)
    ax_mean.set_ylabel(settings["y_label"])
    ax_median.tick_params(labelleft=False)
    ax_mean.set_title("Mean +/- SD", fontsize=7, pad=3)
    ax_median.set_title("Median with 95% CI", fontsize=7, pad=3)
    last_row_start = 4 * (n_strategy_rows - 1)
    for index, ax in enumerate(strategy_axes):
        if index >= last_row_start:
            ax.set_xlabel(base.X_LABEL, fontsize=5.5)
        else:
            ax.tick_params(labelbottom=False)
        if index % 4 != 0:
            ax.tick_params(labelleft=False)
    fig.suptitle(
        f"{phase.replace('_', ' ').title()} | First-frame resistant density: "
        f"{gate[0]:g} to {gate[1]:g} cells µm$^{{-2}}$",
        fontsize=7,
        y=0.988,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.957),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        handletextpad=0.35,
        fontsize=6,
    )
    if annotate_n_pixels:
        fig.text(
            0.985,
            0.012,
            "Strategy-panel labels: qualifying n_px (k = 10^3, M = 10^6)",
            ha="right",
            va="bottom",
            fontsize=5.0,
            color="#444444",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.ft.savefig(fig, output_path)
    plt.close(fig)


def read_and_accumulate(
    run_dir: Path,
    config: dict,
    aggregators: dict[tuple[str, str, str], Aggregator],
    x_edges: np.ndarray,
    strategies: list[str],
    selected_phases: tuple[str, ...],
    escape_cutoffs: dict[tuple[str, str], tuple[float, float]],
    simulation_run: bool,
) -> int:
    strategy_index = {
        strategy: index for index, strategy in enumerate(strategies)
    }
    settings_by_type = {
        kind: plot_settings(config, kind) for kind in CHANGE_TYPES
    }
    x_bin_size = float(x_edges[1] - x_edges[0])
    n_x_bins = len(x_edges) - 1
    completed = 0

    for changes_dir in run_dir.glob("Plate_*/*/density_changes"):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        strategy = base.strategy_from_plate_and_well(plate, well)
        if strategy not in strategy_index:
            print(f"[WARN] Unknown strategy {strategy} for {plate}/{well}")
            continue
        maps_dir = changes_dir.parent / "density_maps"

        for rel_path in sorted(
            changes_dir.glob(base.CHANGE_SPECS["rel"]["glob"])
        ):
            match = base.REL_CHANGE_RE.match(rel_path.name)
            if match is None:
                continue
            first_tp = match.group("first_tp")
            next_tp = match.group("next_tp")
            first_day = base.timepoint_to_day(first_tp)
            next_day = base.timepoint_to_day(next_tp)
            active_phases = interval_phases(
                selected_phases=selected_phases,
                plate=plate,
                well=well,
                first_day=first_day,
                next_day=next_day,
                escape_cutoffs=escape_cutoffs,
                simulation_run=simulation_run,
            )
            if not active_phases:
                continue
            total_path = base.density_map_path(maps_dir, well, first_tp)
            res_path = base.resistant_density_map_path(
                maps_dir,
                well,
                first_tp,
            )
            abs_path = rel_path.with_name(
                rel_path.name.replace("_rel_diff.tif", "_abs_diff.tif")
            )
            if not all(
                path.exists() for path in (total_path, res_path, abs_path)
            ):
                print(f"[WARN] Missing pair for {rel_path}")
                continue

            total_density = tiff.imread(total_path).astype(
                np.float32,
                copy=False,
            ).ravel()
            res_density = tiff.imread(res_path).astype(
                np.float32,
                copy=False,
            ).ravel()
            rel_change = tiff.imread(rel_path).astype(
                np.float32,
                copy=False,
            ).ravel()
            abs_change = tiff.imread(abs_path).astype(
                np.float32,
                copy=False,
            ).ravel()
            arrays = (total_density, res_density, rel_change, abs_change)
            if len({array.shape for array in arrays}) != 1:
                raise ValueError(f"Shape mismatch for {rel_path}")

            x = total_density * base.X_DISPLAY_SCALE
            x_valid = (
                np.isfinite(x)
                & (x >= x_edges[0])
                & (x <= x_edges[-1])
            )
            x_bins = np.zeros(x.shape, dtype=np.int64)
            finite_x = np.isfinite(x)
            x_bins[finite_x] = np.floor(
                (x[finite_x] - x_edges[0]) / x_bin_size
            ).astype(np.int64)
            np.clip(x_bins, 0, n_x_bins - 1, out=x_bins)
            changes = {
                "rel": rel_change,
                "abs_per_day": base.change_values_for_plot(
                    abs_change,
                    first_tp=first_tp,
                    next_tp=next_tp,
                    change_type="abs_per_day",
                ),
            }

            for gate in DEFAULT_GATES:
                gate_mask = (
                    np.isfinite(res_density)
                    & (res_density >= gate[0])
                    & (res_density <= gate[1])
                )
                for kind, y in changes.items():
                    if kind not in settings_by_type:
                        continue
                    settings = settings_by_type[kind]
                    valid = (
                        x_valid
                        & gate_mask
                        & np.isfinite(y)
                        & (y >= settings["ylim"][0])
                        & (y <= settings["ylim"][1])
                    )
                    if settings["xscale"] == "log":
                        valid &= x > 0
                    if settings["yscale"] == "log":
                        valid &= y > 0
                    valid_x_bins = x_bins[valid]
                    valid_y = y[valid].astype(np.float64, copy=False)
                    for phase in active_phases:
                        aggregators[(phase, gate_name(gate), kind)].add(
                            strategy_index[strategy],
                            valid_x_bins,
                            valid_y,
                        )
            completed += 1
            if completed % 250 == 0:
                print(f"[scan] {completed} intervals")
    return completed


def write_outputs(
    *,
    output_root: Path,
    config: dict,
    aggregators: dict[tuple[str, str, str], Aggregator],
    strategies: list[str],
    selected_phases: tuple[str, ...],
    min_bin_pixels: int,
    annotate_n_pixels: bool,
    simulation_run: bool,
    escape_source: Path | None,
) -> int:
    inventory = []
    variants = (
        ("minimum_bin_pixels", output_root, int(min_bin_pixels)),
        ("all_bins", output_root / "all_bins", 1),
    )
    for phase in selected_phases:
        for gate in DEFAULT_GATES:
            for kind in CHANGE_TYPES:
                summary = summarize(
                    aggregators[(phase, gate_name(gate), kind)],
                    strategies,
                    gate,
                    kind,
                    phase,
                )
                for variant, variant_root, variant_min_pixels in variants:
                    phase_root = (
                        variant_root
                        if phase == "overall"
                        else variant_root / phase
                    )
                    output_dir = (
                        phase_root / change_name(kind) / gate_name(gate)
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    table_path = (
                        output_dir / "binned_strategy_statistics.tsv"
                    )
                    summary.to_csv(table_path, sep="\t", index=False)
                    pdf_path = (
                        output_dir
                        / "binned_mean_median_all_strategies.pdf"
                    )
                    plot_summary(
                        summary,
                        pdf_path,
                        plot_settings(config, kind),
                        gate,
                        phase,
                        strategies,
                        variant_min_pixels,
                        annotate_n_pixels=annotate_n_pixels,
                    )
                    inventory.append(
                        {
                            "variant": variant,
                            "minimum_bin_pixels": variant_min_pixels,
                            "phase": phase,
                            "pdf": os.fspath(
                                pdf_path.relative_to(output_root)
                            ),
                            "bytes": pdf_path.stat().st_size,
                        }
                    )
                    print(f"[saved] {pdf_path}")

    pd.DataFrame(inventory).to_csv(
        output_root / "plot_inventory.tsv",
        sep="\t",
        index=False,
    )
    run_type = "simulation" if simulation_run else "experimental"
    escape_text = (
        os.fspath(escape_source)
        if escape_source is not None
        else "not loaded (overall phase only)"
    )
    annotation_text = (
        "Individual-strategy point labels report qualifying pixel counts "
        "using k = 10^3 and M = 10^6.\n"
        if annotate_n_pixels
        else ""
    )
    (output_root / "README.txt").write_text(
        f"{run_type.title()} pooled-pixel binned strategy lines.\n\n"
        "The two top panels show CT, AT50, AT100, ATagent, and AT^+. They "
        "report mean +/- one sample SD and median with a 95% percentile-"
        "bootstrap interval evaluated from the empirical Y distribution.\n"
        "The lower panels separately show mean/SD and median/CI for each "
        "strategy present in the run.\n"
        f"Primary plots omit bins with fewer than {min_bin_pixels} pixels.\n"
        "The all_bins folder mirrors every plot without that threshold and "
        "therefore shows every nonempty bin.\n"
        f"Phases: {', '.join(selected_phases)}.\n"
        "Before escape requires next_day <= pre_escape_cutoff_day; after "
        "escape requires first_day >= escape_day.\n"
        f"Escape metadata: {escape_text}.\n"
        f"Relative-change plots display Y from {RELATIVE_DISPLAY_YLIM[0]:g} "
        f"to {RELATIVE_DISPLAY_YLIM[1]:g}; absolute-change "
        "plots retain their configured scale.\n"
        "All qualifying pixels are included and treated as independent.\n"
        "The resistant-density gate and X density use the interval first frame.\n"
        f"{annotation_text}",
        encoding="ascii",
    )
    (output_root / "all_bins" / "README.txt").write_text(
        "All nonempty bins are plotted in this mirrored output tree.\n"
        "See ../README.txt for the shared analysis definitions.\n",
        encoding="ascii",
    )
    return len(inventory)


def process_run(
    *,
    run_dir: Path,
    output_root: Path,
    config: dict,
    args: argparse.Namespace,
    selected_phases: tuple[str, ...],
) -> int:
    strategies = strategies_in_run(run_dir)
    simulation_run = base.is_simulation_density_run(run_dir)
    escape_cutoffs: dict[tuple[str, str], tuple[float, float]] = {}
    escape_source: Path | None = None
    if any(phase != "overall" for phase in selected_phases):
        escape_cutoffs, simulation_run, escape_source = resolve_escape_cutoffs(
            run_dir,
            config,
            args,
        )
        print(f"[escape] {escape_source}")

    run_type = "simulation" if simulation_run else "experimental"
    print(f"[run] {run_type}: {run_dir}")
    print(f"[strategies] {', '.join(strategies)}")
    output_root.mkdir(parents=True, exist_ok=True)
    settings_by_type = {
        kind: plot_settings(config, kind) for kind in CHANGE_TYPES
    }
    x_max = max(float(settings["xlim"][1]) for settings in settings_by_type.values())
    n_x_bins = int(np.ceil(x_max / float(args.x_bin_size)))
    x_edges = np.arange(n_x_bins + 1, dtype=float) * float(args.x_bin_size)
    aggregators = {
        (phase, gate_name(gate), kind): Aggregator.create(
            n_strategies=len(strategies),
            x_edges=x_edges,
            ylim=settings_by_type[kind]["ylim"],
            n_y_bins=int(args.y_hist_bins),
        )
        for phase in selected_phases
        for gate in DEFAULT_GATES
        for kind in CHANGE_TYPES
    }

    completed = read_and_accumulate(
        run_dir,
        config,
        aggregators,
        x_edges,
        strategies,
        selected_phases,
        escape_cutoffs,
        simulation_run,
    )
    print(f"[scan] {completed} intervals total")
    return write_outputs(
        output_root=output_root,
        config=config,
        aggregators=aggregators,
        strategies=strategies,
        selected_phases=selected_phases,
        min_bin_pixels=int(args.min_bin_pixels),
        annotate_n_pixels=bool(args.annotate_n_pixels),
        simulation_run=simulation_run,
        escape_source=escape_source,
    )


def main() -> None:
    args = parse_args()
    if args.x_bin_size <= 0:
        raise ValueError("--x-bin-size must be positive")
    if args.y_hist_bins < 32:
        raise ValueError("--y-hist-bins must be at least 32")
    if args.min_bin_pixels < 1:
        raise ValueError("--min-bin-pixels must be at least 1")

    run_dirs = [
        Path(path).expanduser()
        for path in (args.run_dirs or DEFAULT_RUN_DIRS)
    ]
    if args.output_dir and len(run_dirs) != 1:
        raise ValueError("--output-dir requires exactly one --run-dir")
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    selected_phases = tuple(dict.fromkeys(args.phases))
    config = base.load_config(args.config)
    total_pdfs = 0
    for run_dir in run_dirs:
        output_root = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else run_dir / DEFAULT_OUTPUT_SUBDIR
        )
        total_pdfs += process_run(
            run_dir=run_dir,
            output_root=output_root,
            config=config,
            args=args,
            selected_phases=selected_phases,
        )
    print(f"[done] {total_pdfs} PDFs across {len(run_dirs)} run(s)")


if __name__ == "__main__":
    main()
