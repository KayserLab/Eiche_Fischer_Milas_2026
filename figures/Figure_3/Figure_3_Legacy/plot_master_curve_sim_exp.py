#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot Figure 3 master curves from precomputed TSV plotting tables."""

# =============================================================================
# UPSTREAM DATA AND ANALYSIS
# =============================================================================
# This is a plotting-only script. It reads binned TSV tables plus histogram NPZ
# caches produced by:
#   analysis/figure_3/Legacy/aggregate_average_symmetric_condition_matrix_160.py
# That aggregation reads the 160 um / 16x experimental density-map TIFFs from
# the simulation-equivalent experimental point-filter run, and creates matching
# simulation density maps directly from the simulation HDF5 snapshots through:
#   analysis/basic/density_maps/Density_Maps_Epanechnikov_coarsegrained.py
#   analysis/basic/density_maps/Density_Maps_Epanechnikov_simulations_coarsegrained.py
# The experimental Figure-2 mask filter is applied upstream by:
#   analysis/figure_3/generate_figure2_mask_filtered_density_maps.py
# Re-run the aggregation script after changing density maps, filtering, time
# intervals, growth/support settings, or bin definitions; then re-run this file.

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for import_path in (PROJECT_ROOT,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as analysis
from shared import figtools as ft
from shared.strategy_colors import get_color, get_display_name as _get_display_name
from shared.data_paths import figure_data_dir, figure_plot_dir, intermediate_dir


def get_display_name(strategy: str) -> str:
    """Use Figure 3 display terminology without altering data keys."""
    display_name = _get_display_name(strategy)
    return {
        "LV-Agent": "ATagent",
        "AT100+": r"AT$^{+}$",
        r"AT100$^{+}$": r"AT$^{+}$",
    }.get(display_name, display_name)


# =============================================================================
# USER PLOTTING SETTINGS -- no density analysis is performed by this script.
# =============================================================================

# Select which histogram cache to use:
#   0.05 -> aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p05
#   0.10 -> aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p1
#   0.20 -> aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p2
HISTOGRAM_CACHE_BIN_WIDTH = 0.10

HISTOGRAM_CACHE_DIRECTORIES = {
    0.05: "aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p05",
    0.10: "aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p1",
    0.20: "aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p2",
}

if HISTOGRAM_CACHE_BIN_WIDTH not in HISTOGRAM_CACHE_DIRECTORIES:
    raise ValueError(
        "HISTOGRAM_CACHE_BIN_WIDTH must be one of 0.05, 0.10, or 0.20"
    )

USER_PLOTTING_DATA_ROOT = figure_data_dir(
    "Figure_3", "master_curve_160", create=False
)
USER_PLOT_OUTPUT_ROOT = figure_plot_dir("Figure_3", "density_mastercurve", create=False)
USER_FIGURE3_OUTPUT = figure_plot_dir("Figure_3", "density_mastercurve", create=False)
USER_SENSITIVE_FIGURE3_OUTPUT = figure_plot_dir(
    "Figure_3", "density_mastercurve_all", create=False
)
USER_BIN_STATISTICS = ("mean", "median")  # mean with SD and median with IQR
# Default export: separate simulation and experimental all-run curves only.
# Set any of the following switches to True to export the corresponding
# additional plot family without changing the plotting code.
PLOT_RESISTANT = True
# Enable this only when the optional NT-sensitive master curve is needed.
PLOT_SENSITIVE = True
PLOT_RELATIVE_GROWTH = True
PLOT_ABSOLUTE_GROWTH = True
PLOT_ALL_SIMULATION_RUNS = True
PLOT_ESCAPING_SIMULATION_RUNS = False
PLOT_SIMULATION_ONLY = True
PLOT_EXPERIMENT_ONLY = True
PLOT_SIMULATION_VS_EXPERIMENT = True
# Additional simulation-only master curve using the experimental d1-to-d3,
# d3-to-d5, ... acquisition schedule rather than daily simulation intervals.
PLOT_SIMULATION_EXPERIMENTAL_INTERVALS = True
# Additional matched-interval simulation curve: each available strategy has the
# same weight in a density bin, independent of its valid pixel count.
PLOT_EQUAL_STRATEGY_WEIGHTED_SIMULATION = True
# Set True to restore the density-PDF panel beneath each master curve.
SHOW_DENSITY_PDF = False
USER_X_LIMITS = (0.0, 1.75)
RELATIVE_Y_LIMITS = (-0.1, 0.9)
ABSOLUTE_Y_LIMITS = (-0.1, 0.5)
USER_MIN_BIN_PIXELS = 100
YLABEL_PAD = 2
XLABEL_PAD = 0
# Matches the resistant-growth y axis in plot_at50_density_trajectories.py.
MASTER_CURVE_Y_AXIS_COLOR = "#b40426"
DENSITY_X_LABEL = r"$\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
KDE_BANDWIDTH = 0.10  # cells / 1000 um2
# Density points used for the mean-curve contrast logged for the Figure 3 text.
# They lie above the plotted support threshold and are represented in both data
# sets.  The reported contrast is the high-density mean minus the low-density
# mean, in day^-1.
TREND_CONTRAST_LOW_DENSITY = 0.45
TREND_CONTRAST_HIGH_DENSITY = 1.55
GREYED_X_MAX = 0.4
# Retain 20% opacity below the local-density support threshold so the
# unsupported region is visually distinct in both curves and marginals.
GREYED_ALPHA = 0.20
USER_HISTOGRAM_CACHE_ROOT = intermediate_dir(
    "combined",
    "figure_3",
    "master_curve_160",
    create=False,
)
HISTOGRAM_STRATEGIES = ("CT", "AT50", "AT100")
COMBINED_FIGSIZE = (6.66, 4.45)
# Match the complete visible span of the three 0.85-inch density-map panels,
# including their two 0.06-panel-width gaps.  This is the full master-curve
# x-range, including the greyed low-density region.
DENSITY_MAP_PANEL_IN = 0.85
DENSITY_MAP_GRID_GAP = 0.06
DENSITY_MAP_THREE_PANEL_WIDTH_IN = DENSITY_MAP_PANEL_IN * (
    3 + DENSITY_MAP_GRID_GAP * 2
)
DENSITY_MAP_GRID_LEFT = 0.18
DENSITY_MAP_GRID_RIGHT = 0.945
DENSITY_MAP_FIGURE_WIDTH_IN = DENSITY_MAP_THREE_PANEL_WIDTH_IN / (
    DENSITY_MAP_GRID_RIGHT - DENSITY_MAP_GRID_LEFT
)
STACKED_DATASET_FIGSIZE = (DENSITY_MAP_FIGURE_WIDTH_IN, 5.37)
# Keep the separate simulation and experimental panels compact while retaining
# their common density-map-matched width.
SINGLE_DATASET_FIGSIZE = (
    STACKED_DATASET_FIGSIZE[0],
    2.35 if SHOW_DENSITY_PDF else 1.60,
)
PANEL_HEIGHT_RATIOS = (6.0, 1.7)

# Strategies shown in the publication master curves and KDEs.
STRATEGIES = ("CT", "AT50", "AT100")
SENSITIVE_STRATEGIES = ("NT",)


@dataclass(frozen=True)
class PlotVariant:
    target_population: str
    growth_mode: str
    simulation_cohort: str
    data_directory: str
    cache_directory: str
    condition_directory: Path
    condition_slug: str
    simulation_interval_mode: str = "daily"
    simulation_only: bool = False

    @property
    def filename_suffix(self) -> str:
        population = "" if self.target_population == "resistant" else "_sensitive"
        cohort = "" if self.simulation_cohort == "escaping" else "_all_runs"
        interval = (
            "" if self.simulation_interval_mode == "daily"
            else "_experimental_intervals"
        )
        return population + cohort + interval

    @property
    def y_limits(self) -> tuple[float, float]:
        return RELATIVE_Y_LIMITS if self.growth_mode == "relative" else ABSOLUTE_Y_LIMITS

    @property
    def strategies(self) -> tuple[str, ...]:
        return SENSITIVE_STRATEGIES if self.target_population == "sensitive" else STRATEGIES


def plot_variants() -> tuple[PlotVariant, ...]:
    resistant_relative_cache = HISTOGRAM_CACHE_DIRECTORIES[HISTOGRAM_CACHE_BIN_WIDTH]
    resistant_absolute_cache = resistant_relative_cache.replace(
        "before_escape_all_both_endpoints",
        "absolute_before_escape_average_endpoints",
    )
    variants = []
    target_populations = tuple(
        population
        for population, enabled in (
            ("resistant", PLOT_RESISTANT),
            ("sensitive", PLOT_SENSITIVE),
        )
        if enabled
    )
    growth_definitions = tuple(
        definition
        for definition in ("relative", "absolute")
        if (
            (definition == "relative" and PLOT_RELATIVE_GROWTH)
            or (definition == "absolute" and PLOT_ABSOLUTE_GROWTH)
        )
    )
    cohorts = tuple(
        cohort
        for cohort, enabled in (
            ("escaping", PLOT_ESCAPING_SIMULATION_RUNS),
            ("all", PLOT_ALL_SIMULATION_RUNS),
        )
        if enabled
    )
    if not target_populations or not growth_definitions or not cohorts:
        raise ValueError("Enable at least one population, growth definition, and cohort.")
    if not any((PLOT_SIMULATION_ONLY, PLOT_EXPERIMENT_ONLY, PLOT_SIMULATION_VS_EXPERIMENT)):
        raise ValueError("Enable at least one master-curve output type.")
    for target_population in target_populations:
        population_suffix = "" if target_population == "resistant" else "_sensitive"
        for growth_mode, data_directory, cache_directory, support, condition_slug in (
            (
                "relative", "plots_before_escape_gate_matrix", resistant_relative_cache,
                "both_t0_t1_gt_1e-08", "before_escape__both_gt_1e-08",
            ),
            (
                "absolute", "plots_absolute_before_escape_gate_matrix", resistant_absolute_cache,
                "average_t0_t1_gt_1e-08", "before_escape__average_gt_1e-08",
            ),
        ):
            if growth_mode not in growth_definitions:
                continue
            for cohort in cohorts:
                cohort_suffix = "" if cohort == "escaping" else "_all_runs"
                variants.append(PlotVariant(
                    target_population=target_population,
                    growth_mode=growth_mode,
                    simulation_cohort=cohort,
                    data_directory=data_directory + population_suffix + cohort_suffix,
                    cache_directory=cache_directory + population_suffix + cohort_suffix,
                    condition_directory=Path("no_gating") / support,
                    condition_slug=condition_slug,
                ))
    if PLOT_SIMULATION_EXPERIMENTAL_INTERVALS:
        variants.append(PlotVariant(
            target_population="resistant",
            growth_mode="relative",
            simulation_cohort="all",
            data_directory="plots_before_escape_gate_matrix_all_runs_experimental_schedule",
            cache_directory=resistant_relative_cache + "_all_runs_experimental_schedule",
            condition_directory=Path("no_gating") / "both_t0_t1_gt_1e-08",
            condition_slug="before_escape__both_gt_1e-08",
            simulation_interval_mode="experimental_schedule",
            simulation_only=True,
        ))
    return tuple(variants)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=USER_PLOTTING_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=USER_HISTOGRAM_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=USER_PLOT_OUTPUT_ROOT)
    parser.add_argument("--figure3-output", type=Path, default=USER_FIGURE3_OUTPUT)
    return parser.parse_args()


def compact_count(value: int) -> str:
    count = int(value)
    for scale, suffix in ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "k")):
        if count >= scale:
            scaled = count / scale
            return f"{scaled:.{1 if scaled < 10 else 0}f}{suffix}"
    return str(count)


def read_condition_tables(
    condition_dir: Path,
) -> tuple[pd.DataFrame, ...]:
    names = (
        "in_vitro_strategy_binned_statistics.tsv",
        "in_vitro_pooled_master_binned_statistics.tsv",
        "in_silico_strategy_binned_statistics.tsv",
        "in_silico_pooled_master_binned_statistics.tsv",
    )
    paths = tuple(condition_dir / name for name in names)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required plotting tables are missing. Run "
            "analysis/figure_3/Legacy/aggregate_average_symmetric_condition_matrix_160.py first:\n  "
            + joined
        )
    return tuple(pd.read_csv(path, sep="\t") for path in paths)


def read_histogram_cache(
    cache_root: Path,
    condition_slug: str,
    strategies_to_plot: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Load x-marginals from the selected dedicated aggregation cache."""
    result: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    for dataset, filename in (
        ("in_vitro", "in_vitro_all_conditions.npz"),
        ("in_silico", "in_silico_all_conditions.npz"),
    ):
        path = cache_root / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {HISTOGRAM_CACHE_BIN_WIDTH:.2f}-bin histogram cache: {path}\n"
                "Run aggregate_average_symmetric_condition_matrix_160.py with "
                f"CORESEED_X_BIN_WIDTH={HISTOGRAM_CACHE_BIN_WIDTH:.2f} "
                "and this cache directory first."
            )

        with np.load(path, allow_pickle=False) as data:
            strategies = [
                analysis.canonical_strategy_label(str(value))
                for value in data["strategies"].tolist()
            ]
            metadata = json.loads(str(data["metadata_json"][0]))
            x_edges = data["x_edges"].copy()
            if "condition_slugs" in data:
                condition_slugs = [str(value) for value in data["condition_slugs"].tolist()]
                try:
                    condition_index = condition_slugs.index(condition_slug)
                except ValueError as error:
                    raise RuntimeError(
                        f"Condition {condition_slug!r} is absent from {path}"
                    ) from error
            else:
                # Compatibility with the existing escaping/relative cache.
                condition_index = 9
            counts = data[f"c{condition_index}_count"].copy()

        if not np.allclose(
            np.diff(x_edges),
            HISTOGRAM_CACHE_BIN_WIDTH,
        ):
            raise RuntimeError(
                f"Histogram cache does not use "
                f"{HISTOGRAM_CACHE_BIN_WIDTH:.2f}-wide bins: {path}"
            )

        if dataset == "in_vitro":
            expected_run = str(analysis.DEFAULT_INVITRO_RUN.expanduser()).lower()
            actual_run = str(metadata.get("run_dir", "")).lower()

            if (
                actual_run != expected_run
                or metadata.get("input_filter")
                != "figure2_clean_sensitive_and_all_clone_resistant"
            ):
                raise RuntimeError(
                    "KDE cache provenance does not match the required Figure 2-mask-filtered "
                    f"experimental input: {path}"
                )

        elif (
            str(metadata.get("input_root", "")).lower()
            != str(analysis.DEFAULT_SIM_ROOT.expanduser()).lower()
        ):
            raise RuntimeError(
                "KDE cache provenance does not match the configured simulation database: "
                f"{path}"
            )

        result[dataset] = (
            x_edges,
            {
                strategy: counts[strategies.index(strategy)].astype(float)
                for strategy in strategies_to_plot
            },
        )

    return result


def rows_for_plot(data: pd.DataFrame, statistic: str) -> pd.DataFrame:
    if statistic == "mean":
        valid = (
            (data["n_pixels"] >= USER_MIN_BIN_PIXELS)
            & np.isfinite(data["mean"])
            & np.isfinite(data["sd"])
        )
    elif statistic == "median":
        valid = (
            (data["n_pixels"] >= USER_MIN_BIN_PIXELS)
            & np.isfinite(data["q1"])
            & np.isfinite(data["median"])
            & np.isfinite(data["q3"])
        )
    else:
        raise ValueError(f"Unsupported statistic={statistic!r}")

    return data[valid].sort_values("x_bin_center_cells_per_1000_um2")


def format_one_decimal_ticks(ax: plt.Axes) -> None:
    """Use physical density values on selected x-axis tick labels."""
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                (f"{value:.1f}" if np.isclose(value, 0.0) else rf"${value:.1f}\times 10^{{-3}}$")
                if np.isclose(value % 0.5, 0.0, atol=1.0e-9)
                else ""
            )
        )
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def log_mean_curve_contrasts(
    dataset: str,
    strategy_data: pd.DataFrame,
    pooled_data: pd.DataFrame,
) -> None:
    """Print the Figure 3 mean-curve contrasts used in the manuscript text.

    These are descriptive values from the plotted binned means.  They are not
    pixel-level inferential statistics; confidence intervals should instead be
    calculated with a replicate/run-level cluster bootstrap.
    """
    low = TREND_CONTRAST_LOW_DENSITY
    high = TREND_CONTRAST_HIGH_DENSITY

    print(
        "[mean-curve contrast] "
        f"dataset={dataset}; density={low:.2f}->{high:.2f} "
        "cells (1000 µm$^2$)$^{-1}$; contrast=mean(high)-mean(low) [day^-1]"
    )

    for label, data in (
        *[(strategy, strategy_data[strategy_data["strategy"] == strategy])
          for strategy in STRATEGIES],
        ("Pooled", pooled_data),
    ):
        valid = rows_for_plot(data, "mean")
        centers = valid["x_bin_center_cells_per_1000_um2"].to_numpy(float)
        low_match = np.flatnonzero(np.isclose(centers, low))
        high_match = np.flatnonzero(np.isclose(centers, high))

        if low_match.size != 1 or high_match.size != 1:
            print(
                f"  {label}: unavailable "
                f"(required bins at {low:.2f} and {high:.2f})"
            )
            continue

        low_row = valid.iloc[int(low_match[0])]
        high_row = valid.iloc[int(high_match[0])]
        low_mean = float(low_row["mean"])
        high_mean = float(high_row["mean"])
        print(
            f"  {label}: mean({low:.2f})={low_mean:.4f} "
            f"[n_pixels={int(low_row['n_pixels'])}]; "
            f"mean({high:.2f})={high_mean:.4f} "
            f"[n_pixels={int(high_row['n_pixels'])}]; "
            f"high-low={high_mean - low_mean:.4f} day^-1"
        )


def log_mean_curve_inference(condition_dir: Path) -> None:
    """Print replicate/run-level Figure 3 contrast inference from aggregation."""
    path = condition_dir / "mean_curve_contrast_statistics.tsv"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing mean-curve inferential statistics. Re-run "
            "analysis/figure_3/Legacy/aggregate_average_symmetric_condition_matrix_160.py "
            f"with --overwrite-cache first: {path}"
        )

    data = pd.read_csv(path, sep="\t")
    required = {
        "dataset",
        "strategy",
        "n_independent_units",
        "mean_contrast_day_minus_1",
        "bootstrap_ci_95_low_day_minus_1",
        "bootstrap_ci_95_high_day_minus_1",
        "wilcoxon_one_sided_p",
        "wilcoxon_one_sided_p_holm",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Malformed mean-curve statistics table {path}: {sorted(missing)}")

    print(
        "[mean-curve inference] independent units are experimental wells or "
        "simulation runs; CI=95% bootstrap CI of the unit-level mean contrast; "
        "p=one-sided Wilcoxon signed-rank test, Holm-adjusted within dataset"
    )
    for dataset in ("in_silico", "in_vitro"):
        print(f"  dataset={dataset}")
        subset = data[data["dataset"] == dataset]
        for strategy in STRATEGIES:
            row = subset[subset["strategy"] == strategy]
            if len(row) != 1:
                raise ValueError(
                    f"Expected exactly one {dataset}/{strategy} row in {path}"
                )
            values = row.iloc[0]
            print(
                f"    {strategy}: n={int(values['n_independent_units'])}; "
                f"mean_high-low={float(values['mean_contrast_day_minus_1']):.4f} "
                "day^-1; "
                f"95% CI=[{float(values['bootstrap_ci_95_low_day_minus_1']):.4f}, "
                f"{float(values['bootstrap_ci_95_high_day_minus_1']):.4f}]; "
                f"p={float(values['wilcoxon_one_sided_p']):.3g}; "
                f"p_holm={float(values['wilcoxon_one_sided_p_holm']):.3g}"
            )


def plot_faded_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    linewidth: float,
    zorder: int,
    linestyle: str = "-",
) -> None:
    """Draw a continuous line, faded below the density-support threshold."""
    if len(x) < 2:
        return

    crossing = np.flatnonzero(
        (x[:-1] < GREYED_X_MAX) & (x[1:] >= GREYED_X_MAX)
    )
    if crossing.size == 0:
        ax.plot(
            x, y, color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=GREYED_ALPHA if np.all(x < GREYED_X_MAX) else 1.0,
            zorder=zorder,
        )
        return

    index = int(crossing[0])
    fraction = (GREYED_X_MAX - x[index]) / (x[index + 1] - x[index])
    y_threshold = y[index] + fraction * (y[index + 1] - y[index])
    for x_segment, y_segment, alpha in (
        (
            np.r_[x[:index + 1], GREYED_X_MAX],
            np.r_[y[:index + 1], y_threshold],
            GREYED_ALPHA,
        ),
        (
            np.r_[GREYED_X_MAX, x[index + 1:]],
            np.r_[y_threshold, y[index + 1:]],
            1.0,
        ),
    ):
        ax.plot(
            x_segment, y_segment, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=alpha, zorder=zorder,
        )


def fill_faded_band(
    ax: plt.Axes,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    """Fill a continuous uncertainty band with an exact opacity change at 0.4."""
    crossing = np.flatnonzero(
        (x[:-1] < GREYED_X_MAX) & (x[1:] >= GREYED_X_MAX)
    )

    if crossing.size == 0:
        band_alpha = (
            alpha * GREYED_ALPHA
            if np.all(x < GREYED_X_MAX)
            else alpha
        )
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=band_alpha,
            linewidth=0,
            zorder=zorder,
        )
        return

    index = int(crossing[0])
    fraction = (
        (GREYED_X_MAX - x[index])
        / (x[index + 1] - x[index])
    )
    lower_threshold = (
        lower[index]
        + fraction * (lower[index + 1] - lower[index])
    )
    upper_threshold = (
        upper[index]
        + fraction * (upper[index + 1] - upper[index])
    )
    x_threshold = np.array([GREYED_X_MAX])

    ax.fill_between(
        np.r_[x[:index + 1], x_threshold],
        np.r_[lower[:index + 1], lower_threshold],
        np.r_[upper[:index + 1], upper_threshold],
        color=color,
        alpha=alpha * GREYED_ALPHA,
        linewidth=0,
        zorder=zorder,
    )

    ax.fill_between(
        np.r_[x_threshold, x[index + 1:]],
        np.r_[lower_threshold, lower[index + 1:]],
        np.r_[upper_threshold, upper[index + 1:]],
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
    )


def equal_strategy_weighted_master(
    strategy_data: pd.DataFrame,
    statistic: str,
    strategies: tuple[str, ...],
) -> pd.DataFrame:
    """Average strategy-level bin centers so each strategy contributes equally."""
    rows = []
    for center, group in strategy_data.groupby("x_bin_center_cells_per_1000_um2"):
        values = []
        for strategy in strategies:
            sub = rows_for_plot(group[group["strategy"] == strategy], statistic)
            if not sub.empty:
                values.append(float(sub.iloc[0][statistic]))
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        rows.append({
            "x_bin_center_cells_per_1000_um2": float(center),
            "n_pixels": int(len(values)),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
        })
    return pd.DataFrame(rows)

def plot_dataset(
    ax: plt.Axes,
    strategy_data: pd.DataFrame,
    pooled_data: pd.DataFrame,
    statistic: str,
    is_simulation: bool,
    strategies_to_plot: tuple[str, ...],
    show_pooled: bool = True,
    y_limits: tuple[float, float] = RELATIVE_Y_LIMITS,
) -> tuple[list[str], pd.DataFrame]:
    plotted: list[str] = []

    for strategy in strategies_to_plot:
        sub = rows_for_plot(
            strategy_data[strategy_data["strategy"] == strategy],
            statistic,
        )

        if sub.empty:
            continue

        plotted.append(strategy)
        x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(float)
        center = sub[statistic].to_numpy(float)

        if statistic == "mean":
            spread = sub["sd"].to_numpy(float)
            lower, upper = center - spread, center + spread
        else:
            lower = sub["q1"].to_numpy(float)
            upper = sub["q3"].to_numpy(float)

        color = get_color(strategy, "#666666")

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.075,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            center,
            color=color,
            linewidth=1.0,
            linestyle="--" if is_simulation else "-",
            zorder=2,
        )

    master = pd.DataFrame()
    if show_pooled:
        master = rows_for_plot(pooled_data, statistic)
        if master.empty:
            raise RuntimeError("No pooled bins remain")

        x = master["x_bin_center_cells_per_1000_um2"].to_numpy(float)
        center = master[statistic].to_numpy(float)
        if statistic == "mean":
            spread = master["sd"].to_numpy(float)
            lower, upper = center - spread, center + spread
        else:
            lower = master["q1"].to_numpy(float)
            upper = master["q3"].to_numpy(float)

        ax.fill_between(
            x, lower, upper, color="black", alpha=0.14,
            linewidth=0, zorder=7,
        )
        ax.plot(
            x, center, color="black",
            linewidth=2.2, linestyle="--" if is_simulation else "-",
            zorder=10,
        )

    # A single overlay gives every curve and uncertainty band an exactly vertical
    # opacity boundary at the requested local-density threshold. Disable edge
    # antialiasing so the full-opacity curve cannot show through as a thin
    # stripe at the boundary.
    ax.axvspan(
        USER_X_LIMITS[0],
        GREYED_X_MAX,
        color="white",
        alpha=1.0 - GREYED_ALPHA,
        zorder=20,
        linewidth=0,
        antialiased=False,
    )

    ax.axhline(
        0.0,
        color="#B8B8B8",
        linewidth=0.6,
        zorder=30,
    )
    ax.axvline(0.4, color="#B8B8B8",
        linewidth=0.6, linestyle="--", zorder=30)


    for spine in ax.spines.values():
        spine.set_zorder(30)

    ax.tick_params(axis="both", zorder=30)
    ax.set_xlim(*USER_X_LIMITS)
    ax.set_ylim(*y_limits)
    ft.format_axes(ax)
    ax.spines["left"].set_color(MASTER_CURVE_Y_AXIS_COLOR)
    ax.tick_params(axis="y", colors=MASTER_CURVE_Y_AXIS_COLOR)
    format_one_decimal_ticks(ax)

    return plotted, master


def kde_from_binned_counts(
    x_edges: np.ndarray,
    counts: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Return a unit-area Gaussian probability-density estimate."""
    total = counts.sum()

    if total <= 0:
        return np.zeros_like(x)

    centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    scaled_distance = (
        (x[:, None] - centers[None, :])
        / KDE_BANDWIDTH
    )
    kernels = np.exp(-0.5 * scaled_distance ** 2)

    return kernels @ counts / (
        total
        * KDE_BANDWIDTH
        * np.sqrt(2.0 * np.pi)
    )


def plot_strategy_kdes(
    ax: plt.Axes,
    x_edges: np.ndarray,
    counts_by_strategy: dict[str, np.ndarray],
    y_max: float,
    is_simulation: bool,
    strategies_to_plot: tuple[str, ...],
    show_ylabel: bool = True,
    show_yticklabels: bool = True,
) -> None:
    """Plot CT, AT50, and AT100 KDEs with a shared y-axis scale."""
    x = np.linspace(*USER_X_LIMITS, 501)

    for strategy in strategies_to_plot:
        density = kde_from_binned_counts(
            x_edges,
            counts_by_strategy[strategy],
            x,
        )

        plot_faded_line(
            ax,
            x,
            density,
            get_color(strategy, "#666666"),
            linewidth=1.2,
            zorder=2,
            linestyle="--" if is_simulation else "-",
        )

    ax.set_xlim(*USER_X_LIMITS)
    ax.set_ylim(0.0, y_max)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.axvline(0.4, color="#B8B8B8",
        linewidth=0.6, linestyle="--")
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)

    if show_ylabel:
        ax.set_ylabel(
            "PDF",
            fontsize=7,
            labelpad=YLABEL_PAD,
        )

    ft.format_axes(ax)
    format_one_decimal_ticks(ax)


def common_kde_y_limit(
    histograms: tuple[
        tuple[np.ndarray, dict[str, np.ndarray]],
        ...
    ],
    strategies_to_plot: tuple[str, ...],
) -> float:
    """Return one KDE ceiling for matched simulation and experiment panels."""
    x = np.linspace(*USER_X_LIMITS, 501)

    maximum = max(
        float(
            kde_from_binned_counts(
                edges,
                counts[strategy],
                x,
            ).max()
        )
        for edges, counts in histograms
        for strategy in strategies_to_plot
    )

    return maximum * 1.10 if maximum > 0 else 1.0


def legend_handles(
    plotted: set[str],
    is_simulation: bool = False,
    show_pooled: bool = True,
    pooled_label: str = "Pooled",
) -> list[Line2D]:
    linestyle = "--" if is_simulation else "-"
    handles = [
        Line2D(
            [0],
            [0],
            color=get_color(strategy, "#666666"),
            linewidth=1.0,
            linestyle=linestyle,
            label=get_display_name(strategy),
        )
        for strategy in STRATEGIES
        if strategy in plotted
    ]

    if show_pooled:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.2,
                linestyle=(0, (1.4, 0.7)) if is_simulation else linestyle,
                label=pooled_label,
            )
        )

    return handles


def draw_below_symbol_legend(
    ax: plt.Axes,
    plotted: set[str],
) -> None:
    """Draw the compact legend in unused high-density simulation-panel space."""
    entries = [
        (
            "pooled",
            "Pooled",
            "black",
        )
    ]
    entries.extend(
        [
        (
            strategy,
            get_display_name(strategy),
            get_color(strategy, "#666666"),
        )
        for strategy in STRATEGIES
        if strategy in plotted
        ]
    )

    # Stack entries vertically in the empty high-density region of the
    # simulation panel, with each label beside its dashed-line-and-square symbol.
    x_position = 1.90
    symbol_ys = np.linspace(0.76, 0.52, len(entries))
    label_offset = 0.085

    for symbol_y, (strategy, label, color) in zip(
        symbol_ys,
        entries,
    ):
        linewidth = (
            2.4
            if strategy == "pooled"
            else 1.2
        )

        line = Line2D(
            (x_position - 0.055, x_position + 0.055),
            (symbol_y, symbol_y),
            transform=ax.transData,
            color=color,
            linewidth=linewidth,
            linestyle="--",
        )
        ax.add_line(line)

        ax.text(
            x_position + label_offset,
            symbol_y,
            label,
            transform=ax.transData,
            ha="left",
            va="center",
            fontsize=6,
        )


def master_curve_ylabel(
    growth_mode: str = "relative",
    target_population: str = "resistant",
) -> str:
    """Compact shared y-axis label for the master-curve panels."""
    subscript = "res" if target_population == "resistant" else "sen"
    if growth_mode == "relative":
        return rf"$\Delta\tilde{{\rho}}_\mathrm{{{subscript}}}/\Delta t$ [day$^{{-1}}$]"

    return rf"$d\rho_\mathrm{{{subscript}}}/dt$ [cells (1000 µm$^2$)$^{{-1}}$ day$^{{-1}}$]"


def write_combined_plot(
    output_dir: Path,
    condition: analysis.Condition,
    tables: tuple[pd.DataFrame, ...],
    histograms: dict[
        str,
        tuple[np.ndarray, dict[str, np.ndarray]]
    ],
    statistic: str,
    filename: str = "master_curve_in_vitro_vs_in_silico.pdf",
) -> Path:
    vitro_strategy, vitro_pooled, silico_strategy, silico_pooled = tables

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={
            "axes.titlesize": 8,
            "axes.labelsize": 7,
        },
    )

    fig = plt.figure(figsize=COMBINED_FIGSIZE)

    outer = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.99,
        bottom=0.19,
        top=0.90,
        wspace=0.08,
    )

    left = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=PANEL_HEIGHT_RATIOS,
        hspace=0.06,
    )

    right = outer[0, 1].subgridspec(
        2,
        1,
        height_ratios=PANEL_HEIGHT_RATIOS,
        hspace=0.06,
    )

    ax_silico = fig.add_subplot(left[0, 0])
    ax_silico_kde = fig.add_subplot(
        left[1, 0],
        sharex=ax_silico,
    )

    ax_vitro = fig.add_subplot(
        right[0, 0],
        sharex=ax_silico,
        sharey=ax_silico,
    )

    ax_vitro_kde = fig.add_subplot(
        right[1, 0],
        sharex=ax_vitro,
    )

    silico, _ = plot_dataset(
        ax_silico,
        silico_strategy,
        silico_pooled,
        statistic,
        is_simulation=True,
        strategies_to_plot=STRATEGIES,
    )

    vitro, _ = plot_dataset(
        ax_vitro,
        vitro_strategy,
        vitro_pooled,
        statistic,
        is_simulation=False,
        strategies_to_plot=STRATEGIES,
    )

    silico_edges, silico_counts = histograms["in_silico"]
    vitro_edges, vitro_counts = histograms["in_vitro"]

    kde_y_limit = common_kde_y_limit(
        (
            (silico_edges, silico_counts),
            (vitro_edges, vitro_counts),
        )
    )

    plot_strategy_kdes(
        ax_silico_kde,
        silico_edges,
        silico_counts,
        kde_y_limit,
        is_simulation=True,
        strategies_to_plot=STRATEGIES,
    )

    plot_strategy_kdes(
        ax_vitro_kde,
        vitro_edges,
        vitro_counts,
        kde_y_limit,
        is_simulation=False,
        strategies_to_plot=STRATEGIES,
        show_ylabel=False,
        show_yticklabels=False,
    )

    ax_silico.set_ylabel(
        master_curve_ylabel(),
        fontsize=7,
        labelpad=YLABEL_PAD,
        color=MASTER_CURVE_Y_AXIS_COLOR,
    )

    plt.setp(
        ax_silico.get_xticklabels(),
        visible=False,
    )
    plt.setp(
        ax_vitro.get_xticklabels(),
        visible=False,
    )

    ax_silico_kde.set_xlabel(
        DENSITY_X_LABEL,
        fontsize=7,
        labelpad=XLABEL_PAD,
    )

    ax_vitro_kde.set_xlabel(
        DENSITY_X_LABEL,
        fontsize=7,
        labelpad=XLABEL_PAD,
    )

    ax_vitro.tick_params(labelleft=False)

    handles = legend_handles(
        set(vitro) | set(silico),
        is_simulation=True,
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / filename

    ft.savefig(
        fig,
        path,
        transparent=False,
    )

    plt.close(fig)

    return path


def write_stacked_dataset_plot(
    output_dir: Path,
    filename: str,
    vitro_strategy: pd.DataFrame,
    vitro_pooled: pd.DataFrame,
    vitro_histogram: tuple[
        np.ndarray,
        dict[str, np.ndarray],
    ],
    silico_strategy: pd.DataFrame,
    silico_pooled: pd.DataFrame,
    silico_histogram: tuple[
        np.ndarray,
        dict[str, np.ndarray],
    ],
    kde_y_limit: float,
    statistic: str,
    growth_mode: str,
    target_population: str,
    y_limits: tuple[float, float],
    strategies_to_plot: tuple[str, ...],
) -> Path:
    """Write matched simulation (top) and experiment (bottom) panels."""
    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={
            "axes.titlesize": 8,
            "axes.labelsize": 7,
        },
    )

    fig = plt.figure(
        figsize=STACKED_DATASET_FIGSIZE
    )

    panels = fig.add_gridspec(
        5,
        1,
        left=DENSITY_MAP_GRID_LEFT,
        right=DENSITY_MAP_GRID_RIGHT,
        bottom=0.06,
        # Leave room for the two-row legend above the upper master-curve panel.
        top=0.90,
        height_ratios=(
            *PANEL_HEIGHT_RATIOS,
            0.75,
            *PANEL_HEIGHT_RATIOS,
        ),
        hspace=0.14,
    )

    ax_silico = fig.add_subplot(
        panels[0, 0]
    )

    ax_silico_kde = fig.add_subplot(
        panels[1, 0],
        sharex=ax_silico,
    )

    ax_vitro = fig.add_subplot(
        panels[3, 0],
        sharex=ax_silico,
        sharey=ax_silico,
    )

    ax_vitro_kde = fig.add_subplot(
        panels[4, 0],
        sharex=ax_vitro,
    )

    silico, _ = plot_dataset(
        ax_silico,
        silico_strategy,
        silico_pooled,
        statistic,
        is_simulation=True,
        y_limits=y_limits,
        strategies_to_plot=strategies_to_plot,
    )

    vitro, _ = plot_dataset(
        ax_vitro,
        vitro_strategy,
        vitro_pooled,
        statistic,
        is_simulation=False,
        y_limits=y_limits,
        strategies_to_plot=strategies_to_plot,
    )

    silico_edges, silico_counts = silico_histogram
    vitro_edges, vitro_counts = vitro_histogram

    plot_strategy_kdes(
        ax_silico_kde,
        silico_edges,
        silico_counts,
        kde_y_limit,
        is_simulation=True,
        strategies_to_plot=strategies_to_plot,
    )

    plot_strategy_kdes(
        ax_vitro_kde,
        vitro_edges,
        vitro_counts,
        kde_y_limit,
        is_simulation=False,
        strategies_to_plot=strategies_to_plot,
    )

    ylabel = master_curve_ylabel(growth_mode, target_population)

    for ax in (
        ax_silico,
        ax_vitro,
    ):
        ax.set_ylabel(
            ylabel,
            fontsize=7,
            labelpad=YLABEL_PAD,
            color=MASTER_CURVE_Y_AXIS_COLOR,
        )

    plt.setp(
        ax_silico.get_xticklabels(),
        visible=False,
    )

    plt.setp(
        ax_vitro.get_xticklabels(),
        visible=False,
    )

    ax_silico_kde.set_xlabel(
        DENSITY_X_LABEL,
        fontsize=7,
        labelpad=XLABEL_PAD,
    )

    ax_vitro_kde.set_xlabel(
        DENSITY_X_LABEL,
        fontsize=7,
        labelpad=XLABEL_PAD,
    )

    for ax, is_simulation in ((ax_silico, True), (ax_vitro, False)):
        ax.legend(
            handles=legend_handles(set(STRATEGIES), is_simulation=is_simulation),
            loc="upper right",
            frameon=False,
            ncol=1,
            handlelength=1.8,
            handletextpad=0.35,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / filename

    ft.savefig(
        fig,
        path,
        transparent=False,
    )

    plt.close(fig)

    return path


def write_single_dataset_plot(
    output_dir: Path,
    filename: str,
    strategy_table: pd.DataFrame,
    pooled_table: pd.DataFrame,
    histogram: tuple[np.ndarray, dict[str, np.ndarray]],
    kde_y_limit: float,
    statistic: str,
    growth_mode: str,
    target_population: str,
    y_limits: tuple[float, float],
    strategies_to_plot: tuple[str, ...],
    *,
    is_simulation: bool,
    pooled_label: str = "Pooled",
) -> Path:
    """Write one simulation-only or experiment-only master-curve panel."""
    show_pooled = target_population != "sensitive"
    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    fig = plt.figure(figsize=SINGLE_DATASET_FIGSIZE)
    panels = fig.add_gridspec(
        2 if SHOW_DENSITY_PDF else 1,
        1,
        left=DENSITY_MAP_GRID_LEFT,
        right=DENSITY_MAP_GRID_RIGHT,
        # Keep simulation and experiment panels geometrically identical, while
        # reserving room for the x-axis label and the optional legend.
        bottom=0.24 if SHOW_DENSITY_PDF else 0.27,
        top=0.84,
        height_ratios=PANEL_HEIGHT_RATIOS if SHOW_DENSITY_PDF else None,
        hspace=0.14,
    )
    ax_curve = fig.add_subplot(panels[0, 0])
    ax_kde = (
        fig.add_subplot(panels[1, 0], sharex=ax_curve)
        if SHOW_DENSITY_PDF else None
    )
    plot_dataset(
        ax_curve,
        strategy_table,
        pooled_table,
        statistic,
        is_simulation=is_simulation,
        y_limits=y_limits,
        strategies_to_plot=strategies_to_plot,
        show_pooled=show_pooled,
    )
    if SHOW_DENSITY_PDF:
        edges, counts = histogram
        plot_strategy_kdes(
            ax_kde,
            edges,
            counts,
            kde_y_limit,
            is_simulation=is_simulation,
            strategies_to_plot=strategies_to_plot,
        )
    ax_curve.set_ylabel(
        master_curve_ylabel(growth_mode, target_population),
        fontsize=7,
        labelpad=YLABEL_PAD,
        color=MASTER_CURVE_Y_AXIS_COLOR,
    )
    if SHOW_DENSITY_PDF:
        plt.setp(ax_curve.get_xticklabels(), visible=False)
        ax_kde.set_xlabel(
            DENSITY_X_LABEL,
            fontsize=7,
            labelpad=XLABEL_PAD,
        )
    else:
        ax_curve.set_xlabel(
            DENSITY_X_LABEL,
            fontsize=7,
            labelpad=XLABEL_PAD,
        )
    ax_curve.legend(
        handles=legend_handles(
            set(strategies_to_plot),
            is_simulation=is_simulation,
            show_pooled=show_pooled,
            pooled_label=pooled_label,
        ),
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.35,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    ft.savefig(fig, path, transparent=False)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser()
    cache_root = args.cache_root.expanduser()
    figure_output = args.figure3_output.expanduser()
    written: set[tuple[str, str, str, str]] = set()

    for variant in plot_variants():
        variant_figure_output = (
            USER_SENSITIVE_FIGURE3_OUTPUT
            if variant.target_population == "sensitive"
            else figure_output
        )
        condition_dir = (
            data_root / variant.data_directory / variant.condition_directory
        )
        tables = read_condition_tables(
            condition_dir,
        )

        (
            vitro_strategy,
            vitro_pooled,
            silico_strategy,
            silico_pooled,
        ) = tables

        histograms = None
        if SHOW_DENSITY_PDF:
            histograms = read_histogram_cache(
                cache_root / variant.cache_directory,
                variant.condition_slug,
                variant.strategies,
            )

        log_mean_curve_contrasts(
            f"in_silico/{variant.target_population}/"
            f"{variant.simulation_cohort}/{variant.growth_mode}",
            silico_strategy,
            silico_pooled,
        )
        if not variant.simulation_only:
            log_mean_curve_contrasts(
                f"in_vitro/{variant.target_population}/{variant.growth_mode}",
                vitro_strategy,
                vitro_pooled,
            )
        if variant.growth_mode == "relative" and not variant.simulation_only:
            log_mean_curve_inference(
                condition_dir,
            )

        for statistic in USER_BIN_STATISTICS:
            statistic_figure_output = variant_figure_output
            if variant.target_population == "sensitive":
                statistic_figure_output = variant_figure_output / "sensitive" / statistic
            comparison_figure_output = (
                USER_SENSITIVE_FIGURE3_OUTPUT / "simulations_above_experiments" / statistic
            )
            if variant.target_population == "sensitive":
                comparison_figure_output = (
                    USER_SENSITIVE_FIGURE3_OUTPUT / "sensitive"
                    / "simulations_above_experiments" / statistic
                )
            kde_y_limit = 1.0
            if SHOW_DENSITY_PDF:
                assert histograms is not None
                kde_y_limit = common_kde_y_limit(
                    (
                        histograms["in_silico"],
                        histograms["in_vitro"],
                    ),
                    variant.strategies,
                )

            criteria_token = variant.condition_directory.name
            filename_tail = (
                f"{variant.growth_mode}_{statistic}_before_escape_"
                f"{criteria_token}{variant.filename_suffix}.pdf"
            )
            if PLOT_SIMULATION_VS_EXPERIMENT and not variant.simulation_only:
                saved = write_stacked_dataset_plot(
                    comparison_figure_output,
                    vitro_strategy=vitro_strategy,
                    vitro_pooled=vitro_pooled,
                    vitro_histogram=histograms["in_vitro"] if histograms is not None else None,
                    silico_strategy=silico_strategy,
                    silico_pooled=silico_pooled,
                    silico_histogram=histograms["in_silico"] if histograms is not None else None,
                    kde_y_limit=kde_y_limit,
                    statistic=statistic,
                    growth_mode=variant.growth_mode,
                    target_population=variant.target_population,
                    y_limits=variant.y_limits,
                    strategies_to_plot=variant.strategies,
                    filename=(
                        "master_curve_simulations_above_experiments_"
                        f"{filename_tail}"
                    ),
                )
                print(f"[saved] {saved}")

            if PLOT_SIMULATION_ONLY:
                simulation_saved = write_single_dataset_plot(
                    statistic_figure_output,
                    filename=f"master_curve_simulation_{filename_tail}",
                    strategy_table=silico_strategy,
                    pooled_table=silico_pooled,
                    histogram=histograms["in_silico"] if histograms is not None else None,
                    kde_y_limit=kde_y_limit,
                    statistic=statistic,
                    growth_mode=variant.growth_mode,
                    target_population=variant.target_population,
                    y_limits=variant.y_limits,
                    strategies_to_plot=variant.strategies,
                    is_simulation=True,
                )
                print(f"[saved] {simulation_saved}")
                if (
                    PLOT_EQUAL_STRATEGY_WEIGHTED_SIMULATION
                    and variant.simulation_interval_mode == "experimental_schedule"
                ):
                    equal_master = equal_strategy_weighted_master(
                        silico_strategy, statistic, variant.strategies,
                    )
                    equal_saved = write_single_dataset_plot(
                        statistic_figure_output / "equal_strategy_weighted",
                        filename=(
                            "master_curve_simulation_"
                            f"{filename_tail.removesuffix('.pdf')}_equal_strategy_weighted.pdf"
                        ),
                        strategy_table=silico_strategy,
                        pooled_table=equal_master,
                        histogram=histograms["in_silico"] if histograms is not None else None,
                        kde_y_limit=kde_y_limit,
                        statistic=statistic,
                        growth_mode=variant.growth_mode,
                        target_population=variant.target_population,
                        y_limits=variant.y_limits,
                        strategies_to_plot=variant.strategies,
                        is_simulation=True,
                        pooled_label="Equal-weight pooled",
                    )
                    print(f"[saved] {equal_saved}")

            if PLOT_EXPERIMENT_ONLY and not variant.simulation_only:
                experiment_saved = write_single_dataset_plot(
                    statistic_figure_output,
                    filename=f"master_curve_experiment_{filename_tail}",
                    strategy_table=vitro_strategy,
                    pooled_table=vitro_pooled,
                    histogram=histograms["in_vitro"] if histograms is not None else None,
                    kde_y_limit=kde_y_limit,
                    statistic=statistic,
                    growth_mode=variant.growth_mode,
                    target_population=variant.target_population,
                    y_limits=variant.y_limits,
                    strategies_to_plot=variant.strategies,
                    is_simulation=False,
                )
                print(f"[saved] {experiment_saved}")
            written.add((
                variant.target_population,
                variant.growth_mode,
                variant.simulation_cohort,
                statistic,
            ))

    expected = len(plot_variants()) * len(USER_BIN_STATISTICS)
    if len(written) != expected:
        raise RuntimeError(f"Expected {expected} master-curve plots, wrote {len(written)}")


if __name__ == "__main__":
    main()
