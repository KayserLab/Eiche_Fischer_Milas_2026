#!/usr/bin/env python3
"""Plot the 120 µm simulation master curve after excluding artificial runs 0–9."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from analysis.figure_3.HDF5_analysis.constants import (
    MIN_PROTOCOLS_PER_BIN,
    MIN_RUNS_PER_BIN,
    exp_scaling,
    prolif_n_rho_bins,
    prolif_rho_max,
    px_to_micron,
)
from shared.strategy_colors import get_color, get_display_name
from shared.data_paths import figure_data_dir, figure_plot_dir
from analysis.figure_3.HDF5_analysis.constants import validate_kernel_settings


STRATEGIES = ("CT", "AT50", "AT100")
# HDF5 tables are stored in the renamed analysis data directory.
SOURCE_DATA_SUBDIR = "HDF5_analysis"
DATA_ROOT = figure_data_dir("Figure_3", "HDF5_analysis", "flux_vs_density",
    "kernel_radius_120um", "simulation", create=False)
OUTPUT_PATH = figure_plot_dir("Figure_3", "density_mastercurve", create=False) / (
    "master_curve_simulation_relative_all_timepoints_radius_120um_non_artificial_runs.pdf"
)

X_LIMITS = (0.25, 1.5)
Y_LIMITS = (0.0, 0.6)
GREYED_X_MAX = 0.4
GREYED_ALPHA = 0.20
FIGURE_WIDTH_IN = 0.85 * (3 + 2 * 0.06) / (0.945 - 0.18)
FIGURE_SIZE = (FIGURE_WIDTH_IN, 1.92)


def is_artificial_run(run_name: str) -> bool:
    match = re.fullmatch(r"run_(\d+)", run_name)
    return match is not None and 0 <= int(match.group(1)) <= 9


def read_strategy_curves(strategy: str) -> np.ndarray:
    path = DATA_ROOT / f"flux_vs_density_{strategy}.h5"
    with h5py.File(path, "r") as handle:
        validate_kernel_settings(handle.attrs, 120.0, source=path)
        names = [name.decode() for name in handle["run_names"][:]]
        keep = np.asarray([not is_artificial_run(name) for name in names])
        curves = np.asarray(handle["all_timepoints"]["median_rel"][:], dtype=float)
    print(f"{strategy}: kept {keep.sum()} of {len(keep)} runs; excluded {(~keep).sum()} artificial runs")
    return curves[keep]


def percentile(values: np.ndarray, q: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(values, q, axis=0)


def strategy_statistics(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.isfinite(curves).sum(axis=0)
    arrays = (percentile(curves, 50), percentile(curves, 25), percentile(curves, 75))
    return tuple(np.where(support >= MIN_RUNS_PER_BIN, array, np.nan) for array in arrays)


def master_statistics(
    statistics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.stack([statistics[strategy][0] for strategy in STRATEGIES])
    support = np.isfinite(centers).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        arrays = (np.nanmean(centers, axis=0), np.nanmin(centers, axis=0), np.nanmax(centers, axis=0))
    return tuple(np.where(support >= MIN_PROTOCOLS_PER_BIN, array, np.nan) for array in arrays)


def density_axis() -> np.ndarray:
    edges = np.linspace(0.0, prolif_rho_max, prolif_n_rho_bins + 1)
    native_centers = 0.5 * (edges[:-1] + edges[1:])
    return native_centers * 1000.0 / (exp_scaling * px_to_micron) ** 2


def format_density_ticks(axis: plt.Axes) -> None:
    axis.set_xticks(np.arange(X_LIMITS[0], X_LIMITS[1] + 0.001, 0.25))

    def label(value: float, _position: float) -> str:
        if np.isclose(value % 0.5, 0.0, atol=1e-9) and not np.isclose(value, 0.0):
            return rf"${value:.1f}\times10^{{-3}}$"
        return ""

    axis.xaxis.set_major_formatter(FuncFormatter(label))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.1f}"))


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=get_color(strategy, "#666666"), linewidth=1.0,
               linestyle="--", label=get_display_name(strategy, superscript=True))
        for strategy in STRATEGIES
    ]
    master = Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label="Master")
    dash_pattern = matplotlib.rcParams["lines.dashed_pattern"]
    if matplotlib.rcParams["lines.scale_dashes"]:
        dash_pattern = tuple(length / master.get_linewidth() for length in dash_pattern)
    master.set_dashes(dash_pattern)
    handles.append(master)
    return handles


def main() -> None:
    curves = {strategy: read_strategy_curves(strategy) for strategy in STRATEGIES}
    statistics = {strategy: strategy_statistics(values) for strategy, values in curves.items()}
    master = master_statistics(statistics)
    x = density_axis()

    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=5.6,
                          overrides={"axes.titlesize": 8, "axes.labelsize": 7})
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(1, 1, left=0.18, right=0.945, bottom=0.27, top=0.84)
    axis = fig.add_subplot(grid[0, 0])

    for strategy in STRATEGIES:
        center, lower, upper = statistics[strategy]
        color = get_color(strategy, "#666666")
        axis.fill_between(x, lower, upper, color=color, alpha=0.20, linewidth=0, zorder=1)
        axis.plot(x, center, color=color, linewidth=1.0, linestyle="--", zorder=2)
    center, lower, upper = master
    axis.fill_between(x, lower, upper, color="black", alpha=0.15, linewidth=0, zorder=7)
    axis.plot(x, center, color="black", linewidth=2.2, linestyle="--", zorder=10)

    axis.axvspan(X_LIMITS[0], GREYED_X_MAX, color="white", alpha=1.0 - GREYED_ALPHA,
                linewidth=0, antialiased=False, zorder=20)
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=30)
    axis.axvline(GREYED_X_MAX, color="#B8B8B8", linewidth=0.6, linestyle="--", zorder=30)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlabel(r"$\rho_\mathrm{tot}$ [cells µm$^{-2}$]", fontsize=7, labelpad=0)
    axis.set_ylabel(r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$ [day$^{-1}$]", fontsize=7, labelpad=2)
    ft.format_axes(axis)
    format_density_ticks(axis)
    axis.legend(handles=legend_handles(), loc="upper right", frameon=False,
                ncol=1, handlelength=1.8, handletextpad=0.35)
    for spine in axis.spines.values():
        spine.set_zorder(30)
    axis.tick_params(axis="both", zorder=30)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, OUTPUT_PATH, transparent=False)
    plt.close(fig)
    print(f"[saved] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
