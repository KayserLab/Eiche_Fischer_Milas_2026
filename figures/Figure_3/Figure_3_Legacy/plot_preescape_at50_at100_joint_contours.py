#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot pre-escape AT50/AT100 density-change contours with marginals.

The script reuses the exact 2-D histograms written by
``aggregate_average_symmetric_condition_matrix_160.py``.  It creates one
two-panel figure for experiments and one for simulations.  Each panel contains
the strategy's joint density, marginal x/y histograms, and the corresponding
pooled five-strategy master curve.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as analysis
from shared import figtools as ft
from shared.strategy_colors import get_color


# The publication condition used by the referenced master-curve script.
CONDITION_THRESHOLD = 1.0e-8
STRATEGIES = ("AT50", "AT100")
X_LIMITS = (0.0, 2.5)
RELATIVE_Y_LIMITS = (-0.75, 2.0)
ABSOLUTE_LIMIT_QUANTILES = (0.001, 0.999)
ABSOLUTE_LIMIT_STEP = 0.05
MARGINAL_Y_BINS = 40
MAX_SCATTER_POINTS = 8000
SCATTER_SIZE = 1.00
SCATTER_ALPHA = 0.22
MIN_MASTER_BIN_PIXELS = 100
CONTOUR_MASS_LEVELS = (0.90, 0.70, 0.50, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    contour_cache_root = analysis.DEFAULT_CACHE_ROOT.with_name(
        analysis.DEFAULT_CACHE_ROOT.name + "_contour_xbin_0p1"
    )
    parser.add_argument("--cache-root", type=Path, default=contour_cache_root)
    parser.add_argument("--data-root", type=Path, default=analysis.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
            / "AT50_AT100_contour_density"
        ),
    )
    parser.add_argument("--statistic", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--smooth-sigma", type=float, default=1.0,
        help="Gaussian smoothing in displayed histogram-bin units (default: 1.0).",
    )
    return parser.parse_args()


def publication_condition() -> analysis.Condition:
    matches = [
        condition for condition in analysis.CONDITIONS
        if condition.cutoff == "before_escape"
        and (
            (analysis.GROWTH_MODE == "absolute" and condition.support_mode == "average")
            or (
                analysis.GROWTH_MODE == "relative"
                and analysis.SUPPORT_POLICY == "both_endpoints"
                and condition.support_mode == "both"
            )
            or (
                analysis.GROWTH_MODE == "relative"
                and analysis.SUPPORT_POLICY == "t0_positive"
                and condition.support_mode == "t0"
            )
        )
        and condition.gate_lower is None
        and condition.gate_upper is None
        and math.isclose(
            condition.threshold,
            CONDITION_THRESHOLD if analysis.SUPPORT_POLICY == "both_endpoints" else 0.0,
            rel_tol=1.0e-12, abs_tol=0.0
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify the before-escape condition with both endpoints "
            f"> {CONDITION_THRESHOLD:g}. Check the aggregation settings."
        )
    return matches[0]


def master_curve(path: Path, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pooled master-curve table: {path}\n"
            "Run aggregate_average_symmetric_condition_matrix_160.py first."
        )
    table = pd.read_csv(path, sep="\t")
    valid = (table["n_pixels"] >= MIN_MASTER_BIN_PIXELS) & np.isfinite(table[statistic])
    table = table.loc[valid].sort_values("x_bin_center_cells_per_1000_um2")
    if table.empty:
        raise RuntimeError(f"No valid {statistic} master-curve bins in {path}")
    return (
        table["x_bin_center_cells_per_1000_um2"].to_numpy(float),
        table[statistic].to_numpy(float),
    )


def displayed_histogram(
    histogram: np.ndarray,
    y_edges: np.ndarray,
    y_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the very fine stored y bins into plot-resolution bins."""
    target_edges = np.linspace(y_limits[0], y_limits[1], MARGINAL_Y_BINS + 1)
    source_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    target_index = np.searchsorted(target_edges, source_centers, side="right") - 1
    valid = (target_index >= 0) & (target_index < len(target_edges) - 1)
    reduced = np.zeros((histogram.shape[0], len(target_edges) - 1), dtype=float)
    for source_index in np.flatnonzero(valid):
        reduced[:, target_index[source_index]] += histogram[:, source_index]
    return reduced, target_edges, 0.5 * (target_edges[:-1] + target_edges[1:])


def density_levels(density: np.ndarray) -> np.ndarray:
    """Return density thresholds enclosing the requested probability masses."""
    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size < 4:
        raise RuntimeError("Too few occupied bins for contour plotting")
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = []
    for mass in CONTOUR_MASS_LEVELS:
        thresholds.append(ordered[min(np.searchsorted(cumulative, mass), len(ordered) - 1)])
    levels = np.unique(np.asarray(thresholds, dtype=float))
    if levels.size < 2:
        levels = np.linspace(positive.min(), positive.max(), 5)[1:]
    return levels


def absolute_y_limits(aggregator, strategies: list[str]) -> tuple[float, float]:
    """Robust limits shared by AT50/AT100 within one absolute-growth figure."""
    counts = sum(
        aggregator.histogram[strategies.index(strategy)].sum(axis=0).astype(float)
        for strategy in STRATEGIES
    )
    if counts.sum() <= 0:
        raise RuntimeError("No absolute-growth values available for y-axis scaling")
    cumulative = np.cumsum(counts) / counts.sum()
    centers = 0.5 * (aggregator.y_edges[:-1] + aggregator.y_edges[1:])
    lower = float(centers[np.searchsorted(cumulative, ABSOLUTE_LIMIT_QUANTILES[0])])
    upper = float(centers[np.searchsorted(cumulative, ABSOLUTE_LIMIT_QUANTILES[1])])
    lower, upper = min(0.0, lower), max(0.0, upper)
    padding = 0.08 * max(upper - lower, ABSOLUTE_LIMIT_STEP)
    lower = np.floor((lower - padding) / ABSOLUTE_LIMIT_STEP) * ABSOLUTE_LIMIT_STEP
    upper = np.ceil((upper + padding) / ABSOLUTE_LIMIT_STEP) * ABSOLUTE_LIMIT_STEP
    return float(lower), float(upper)

def sample_histogram_points(
    histogram: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    y_limits: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a reproducible pixel-weighted sample from the cached 2-D counts."""
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    visible_y = (y_centers >= y_limits[0]) & (y_centers <= y_limits[1])
    counts = histogram.astype(np.float64, copy=True)
    counts[:, ~visible_y] = 0.0
    total = int(counts.sum())
    if total <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    n_points = min(total, MAX_SCATTER_POINTS)
    probabilities = counts.ravel() / counts.sum()
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(
        probabilities.size, size=n_points, replace=True, p=probabilities
    )
    x_index, y_index = np.unravel_index(flat_indices, counts.shape)
    x = rng.uniform(x_edges[x_index], x_edges[x_index + 1])
    y = rng.uniform(y_edges[y_index], y_edges[y_index + 1])
    return x, y

def draw_panel(
    fig: plt.Figure,
    cell,
    strategy: str,
    histogram: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    master_x: np.ndarray,
    master_y: np.ndarray,
    smooth_sigma: float,
    show_ylabel: bool,
    y_limits: tuple[float, float],
    top_count_limit: float,
    right_count_limit: float,
) -> tuple[plt.Axes, int]:
    nested = cell.subgridspec(
        2, 2, width_ratios=(5.0, 1.25), height_ratios=(1.25, 5.0),
        wspace=0.16, hspace=0.12,
    )
    ax_top = fig.add_subplot(nested[0, 0])
    ax = fig.add_subplot(nested[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(nested[1, 1], sharey=ax)
    ax_blank = fig.add_subplot(nested[0, 1])
    ax_blank.axis("off")

    reduced, displayed_y_edges, y_centers = displayed_histogram(histogram, y_edges, y_limits)
    density = gaussian_filter(reduced, sigma=(smooth_sigma, smooth_sigma), mode="nearest")
    if density.sum() <= 0:
        raise RuntimeError(f"No displayed data for {strategy}")
    density /= density.sum()
    color = get_color(strategy, "#666666")
    scatter_x, scatter_y = sample_histogram_points(
        histogram, x_edges, y_edges, y_limits,
        seed=7050 if strategy == "AT50" else 7100,
    )
    ax.scatter(
        scatter_x, scatter_y, s=SCATTER_SIZE, c=color, alpha=SCATTER_ALPHA,
        linewidths=0, rasterized=True, zorder=4,
    )
    levels = density_levels(density)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    fill_levels = np.r_[levels, density.max() * (1.0 + 1.0e-9)]
    alphas = np.linspace(0.13, 0.50, len(fill_levels) - 1)
    fill_colors = [matplotlib.colors.to_rgba(color, alpha) for alpha in alphas]
    ax.contourf(x_centers, y_centers, density.T, levels=fill_levels,
                colors=fill_colors, antialiased=True, zorder=2)
    ax.contour(x_centers, y_centers, density.T, levels=levels,
               colors=[color], linewidths=0.65, zorder=3)
    ax.plot(master_x, master_y, color="black", linewidth=1.2, marker="o",
            markersize=2.4, markeredgecolor="white", markeredgewidth=0.3, zorder=8)
    ax.axhline(0, color="#B8B8B8", linewidth=0.6, zorder=0)

    total_count = reduced.sum()
    x_counts = reduced.sum(axis=1) / total_count
    y_counts = reduced.sum(axis=0) / total_count
    ax_top.bar(x_centers, x_counts, width=np.diff(x_edges), color=color,
               alpha=0.55, edgecolor=color, linewidth=0.35)
    ax_right.barh(y_centers, y_counts, height=np.diff(displayed_y_edges), color=color,
                  alpha=0.55, edgecolor=color, linewidth=0.35)
    ax_top.set_title(strategy, fontsize=7, fontweight="normal", pad=2)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*y_limits)
    if not show_ylabel:
        ax.tick_params(axis="y", which="both", labelleft=False)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    ax_top.set_ylim(0.0, top_count_limit)
    ax_right.set_xlim(0.0, right_count_limit)
    ax_top.tick_params(
        axis="x", which="both", bottom=True, labelbottom=False,
        labelsize=6, pad=1,
    )
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_top.set_ylabel("Fraction", fontsize=6, labelpad=2)
    ax_right.set_xlabel("Fraction", fontsize=6, labelpad=1)
    ax_top.tick_params(axis="y", labelsize=6, pad=2, left=True)
    ax_right.tick_params(axis="x", labelsize=6, pad=2, bottom=True)
    ax_top.yaxis.set_major_locator(MaxNLocator(3))
    right_tick_upper = MaxNLocator(nbins=2).tick_values(0.0, right_count_limit)[-1]
    ax_right.set_xlim(0.0, right_tick_upper)
    ax_right.set_xticks([0.0, right_tick_upper])
    for target in (ax, ax_top, ax_right):
        ft.format_axes(target)
    for marginal_ax in (ax_top, ax_right):
        marginal_ax.spines["left"].set_visible(True)
        marginal_ax.spines["bottom"].set_visible(True)
        marginal_ax.spines["left"].set_linewidth(0.8)
        marginal_ax.spines["bottom"].set_linewidth(0.8)
    return ax, int(reduced.sum())


def write_figure(
    dataset_label: str,
    filename_token: str,
    strategies: list[str],
    aggregator,
    master: tuple[np.ndarray, np.ndarray],
    output_dir: Path,
    statistic: str,
    smooth_sigma: float,
    shared_top_count_limit: float | None = None,
    shared_right_count_limit: float | None = None,
) -> Path:
    ft.apply_nature_style(
        text_pt=7, tick_pt=6, legend_pt=6,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    fig = plt.figure(figsize=(3.28, 1.96), dpi=300)
    outer = fig.add_gridspec(1, 2, left=0.17, right=0.97, bottom=0.27,
                             top=0.90, wspace=0.31)
    missing = [strategy for strategy in STRATEGIES if strategy not in strategies]
    y_limits = (
        RELATIVE_Y_LIMITS
        if analysis.GROWTH_MODE == "relative"
        else absolute_y_limits(aggregator, strategies)
    )
    if missing:
        raise RuntimeError(f"{', '.join(missing)} absent from the {dataset_label} cache")
    reduced_histograms = [
        displayed_histogram(
            aggregator.histogram[strategies.index(strategy)],
            aggregator.y_edges,
            y_limits,
        )[0]
        for strategy in STRATEGIES
    ]
    top_count_limit = 1.05 * max(
        float((histogram.sum(axis=1) / histogram.sum()).max())
        for histogram in reduced_histograms
    )
    right_count_limit = 1.05 * max(
        float((histogram.sum(axis=0) / histogram.sum()).max())
        for histogram in reduced_histograms
    )
    if shared_top_count_limit is not None:
        top_count_limit = shared_top_count_limit
    if shared_right_count_limit is not None:
        right_count_limit = shared_right_count_limit
    counts: list[int] = []
    for panel_index, strategy in enumerate(STRATEGIES):
        if strategy not in strategies:
            raise RuntimeError(f"{strategy} is absent from the {dataset_label} cache")
        strategy_index = strategies.index(strategy)
        _ax, count = draw_panel(
            fig, outer[0, panel_index], strategy,
            aggregator.histogram[strategy_index], aggregator.x_edges,
            aggregator.y_edges, master[0], master[1], smooth_sigma,
            show_ylabel=(panel_index == 0),
            y_limits=y_limits,
            top_count_limit=top_count_limit,
            right_count_limit=right_count_limit,
        )
        counts.append(count)
    statistic_label = "mean" if statistic == "mean" else "median"
    y_label = (
        r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$" "\\n" r"[day$^{-1}$]"
        if analysis.GROWTH_MODE == "relative"
        else "Absolute resistant growth\\n" r"[cells (1000 µm$^2$)$^{-1}$ day$^{-1}$]"
    )
    fig.supxlabel(r"Local Density [cells (1000 µm$^2$)$^{-1}$]", fontsize=7, y=0.075)
    fig.supylabel(y_label, fontsize=7, x=0.040)
    handle = Line2D(
        [0], [0], color="black", marker="o", markersize=2.4,
        linewidth=1.2, markeredgecolor="white", markeredgewidth=0.3,
        label=f"Pooled master curve ({statistic_label})",
    )
    fig.legend(handles=[handle], loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=1, frameon=False, handlelength=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    growth_token = "" if analysis.GROWTH_MODE == "relative" else "_absolute"
    stem = output_dir / (
        f"preescape_AT50_AT100_joint_contours{growth_token}_{filename_token}_{statistic}"
    )
    pdf_path = stem.with_suffix(".pdf")
    ft.savefig(fig, pdf_path, transparent=False)
    plt.close(fig)
    return pdf_path


def main() -> None:
    args = parse_args()
    condition = publication_condition()
    condition_dir = args.data_root.expanduser() / condition.output_relative
    cache_root = args.cache_root.expanduser()
    inputs = (
        (
            "Experiments", "experiments", cache_root / "in_vitro_all_conditions.npz",
            condition_dir / "in_vitro_pooled_master_binned_statistics.tsv",
        ),
        (
            "Simulations", "simulations", cache_root / "in_silico_all_conditions.npz",
            condition_dir / "in_silico_pooled_master_binned_statistics.tsv",
        ),
    )
    loaded_inputs = []
    top_limits: list[float] = []
    right_limits: list[float] = []
    for label, token, cache_path, master_path in inputs:
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing aggregation cache: {cache_path}\n"
                "Run aggregate_average_symmetric_condition_matrix_160.py first."
            )
        strategies, aggregators, _metadata = analysis.load_bundle(cache_path)
        aggregator = aggregators[condition.slug]
        y_limits = (
            RELATIVE_Y_LIMITS
            if analysis.GROWTH_MODE == "relative"
            else absolute_y_limits(aggregator, strategies)
        )
        reduced_histograms = [
            displayed_histogram(
                aggregator.histogram[strategies.index(strategy)],
                aggregator.y_edges,
                y_limits,
            )[0]
            for strategy in STRATEGIES
        ]
        top_limits.append(
            1.05 * max(
                float((histogram.sum(axis=1) / histogram.sum()).max())
                for histogram in reduced_histograms
            )
        )
        right_limits.append(
            1.05 * max(
                float((histogram.sum(axis=0) / histogram.sum()).max())
                for histogram in reduced_histograms
            )
        )
        master = master_curve(master_path, args.statistic)
        loaded_inputs.append((label, token, strategies, aggregator, master))

    shared_top_count_limit = max(top_limits)
    shared_right_count_limit = max(right_limits)
    for label, token, strategies, aggregator, master in loaded_inputs:
        path = write_figure(
            label, token, strategies, aggregator, master,
            args.output_dir.expanduser(), args.statistic, max(0.0, args.smooth_sigma),
            shared_top_count_limit=shared_top_count_limit,
            shared_right_count_limit=shared_right_count_limit,
        )
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
