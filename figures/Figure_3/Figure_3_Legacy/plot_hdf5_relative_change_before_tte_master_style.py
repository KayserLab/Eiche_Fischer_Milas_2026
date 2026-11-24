#!/usr/bin/env python3
"""Plot HDF5-style relative-growth master curves before TTE.

The script reads the per-run before_tte/median_rel curves written by the
HDF5-analysis computation and exports separate simulation and experimental
PDFs using the visual style of plot_master_curve_sim_exp.py.

Examples
--------
Standard 160 um radius analysis:
    python figures/Figure_3/plot_hdf5_relative_change_before_tte_master_style.py

Additional 80 um radius analysis:
    python figures/Figure_3/plot_hdf5_relative_change_before_tte_master_style.py \
        --kernel-radius-um 80
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HDF5_ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "figure_3" / "HDF5_analysis"
for import_path in (PROJECT_ROOT, HDF5_ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from shared import figtools as ft
from analysis.figure_3.HDF5_analysis.constants import (
    AREA_SCALE,
    MIN_PROTOCOLS_PER_BIN,
    MIN_RUNS_PER_BIN,
    exp_scaling,
    prolif_n_rho_bins,
    prolif_rho_max,
    px_to_micron,
)
from shared.data_paths import figure_data_dir, figure_plot_dir
from shared.strategy_colors import get_color, get_display_name


STRATEGIES = ("CT", "AT50", "AT100")
SOURCE_DATA_SUBDIR = "HDF5_analysis"
OUTPUT_SUBDIR = "HDF5_analysis"
PERIOD = "before_tte"
RELATIVE_Y_LIMITS = (0.0, 2.0)
X_LIMITS = (0.0, 1.75)
GREYED_X_MAX = 0.4
GREYED_ALPHA = 0.20
MASTER_CURVE_Y_AXIS_COLOR = "#b40426"
DENSITY_X_LABEL = r"$\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
RELATIVE_Y_LABEL = r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$ [day$^{-1}$]"

DENSITY_MAP_PANEL_IN = 0.85
DENSITY_MAP_GRID_GAP = 0.06
DENSITY_MAP_GRID_LEFT = 0.18
DENSITY_MAP_GRID_RIGHT = 0.945
DENSITY_MAP_THREE_PANEL_WIDTH_IN = DENSITY_MAP_PANEL_IN * (
    3 + 2 * DENSITY_MAP_GRID_GAP
)
FIGURE_WIDTH_IN = DENSITY_MAP_THREE_PANEL_WIDTH_IN / (
    DENSITY_MAP_GRID_RIGHT - DENSITY_MAP_GRID_LEFT
)
FIGURE_SIZE = (FIGURE_WIDTH_IN, 1.60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot separate simulation and experimental relative-growth master "
            "curves before TTE in the Figure 3 master-curve style."
        )
    )
    parser.add_argument(
        "--kernel-radius-um",
        type=float,
        default=float(os.environ.get("HDF5_KERNEL_RADIUS_UM", os.environ.get("MAJA_KERNEL_RADIUS_UM", "160"))),
        help="Kernel radius used during computation (default: 160).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional flux_vs_density data root containing simulation/ and "
            "experiment/ directories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the two PDFs.",
    )
    return parser.parse_args()


def radius_directory(radius_um: float) -> str | None:
    if np.isclose(radius_um, 160.0):
        return None
    token = f"{float(radius_um):g}".replace(".", "p")
    return f"kernel_radius_{token}um"


def default_roots(radius_um: float) -> tuple[Path, Path]:
    data_root = figure_data_dir(
        "Figure_3", SOURCE_DATA_SUBDIR, "flux_vs_density", create=False
    )
    output_root = figure_plot_dir(
        "Figure_3", OUTPUT_SUBDIR, "flux_vs_density", create=False
    )
    variant = radius_directory(radius_um)
    if variant is not None:
        data_root = data_root / variant
        output_root = output_root / variant
    return data_root, output_root / "master_style_before_tte"


def input_path(data_root: Path, dataset: str, strategy: str) -> Path:
    return data_root / dataset / f"flux_vs_density_{strategy}.h5"


def load_run_curves(
    data_root: Path,
    dataset: str,
    radius_um: float,
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for strategy in STRATEGIES:
        path = input_path(data_root, dataset, strategy)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {dataset} input for {strategy}: {path}\n"
                "Run the corresponding HDF5 compute script first."
            )
        with h5py.File(path, "r") as handle:
            stored_radius = handle.attrs.get("kernel_radius_um")
            if stored_radius is not None and not np.isclose(
                float(stored_radius), radius_um
            ):
                raise ValueError(
                    f"{path} was computed with radius {stored_radius} um, "
                    f"not {radius_um:g} um."
                )
            if PERIOD not in handle:
                raise KeyError(
                    f"{path} has no {PERIOD!r} group. Rerun the corresponding "
                    "HDF5 compute script."
                )
            group = handle[PERIOD]
            if "median_rel" not in group:
                raise KeyError(f"{path}/{PERIOD} has no 'median_rel' dataset.")
            values = np.asarray(group["median_rel"][:], dtype=float)
            if values.ndim != 2:
                raise ValueError(
                    f"{path}/{PERIOD}/median_rel must be two-dimensional; "
                    f"found shape {values.shape}."
                )
            curves[strategy] = values
    return curves


def finite_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(values, percentile, axis=0)


def strategy_statistics(
    run_curves: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = {}
    for strategy, values in run_curves.items():
        support = np.isfinite(values).sum(axis=0)
        median = finite_percentile(values, 50)
        lower = finite_percentile(values, 25)
        upper = finite_percentile(values, 75)
        enough = support >= MIN_RUNS_PER_BIN
        result[strategy] = tuple(
            np.where(enough, array, np.nan)
            for array in (median, lower, upper)
        )
    return result


def master_statistics(
    strategy_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack([strategy_curves[strategy][0] for strategy in STRATEGIES])
    support = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmean(values, axis=0)
        lower = np.nanmin(values, axis=0)
        upper = np.nanmax(values, axis=0)
    enough = support >= MIN_PROTOCOLS_PER_BIN
    return tuple(
        np.where(enough, array, np.nan)
        for array in (center, lower, upper)
    )


def density_axis(dataset: str) -> np.ndarray:
    edges = np.linspace(0.0, prolif_rho_max, prolif_n_rho_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if dataset == "simulation":
        return centers * 1000.0 / (exp_scaling * px_to_micron) ** 2
    if dataset == "experiment":
        return (centers / AREA_SCALE) * 1000.0 / px_to_micron**2
    raise ValueError(f"Unknown dataset: {dataset!r}")


def format_density_ticks(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                (
                    f"{value:.1f}"
                    if np.isclose(value, 0.0)
                    else rf"${value:.1f}\times 10^{{-3}}$"
                )
                if np.isclose(value % 0.5, 0.0, atol=1.0e-9)
                else ""
            )
        )
    )
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))


def legend_handles(is_simulation: bool) -> list[Line2D]:
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
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            linestyle=linestyle,
            label="Mean",
        )
    )
    return handles


def draw_plot(
    dataset: str,
    stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    master: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
) -> None:
    is_simulation = dataset == "simulation"
    linestyle = "--" if is_simulation else "-"
    x = density_axis(dataset)

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(
        1,
        1,
        left=DENSITY_MAP_GRID_LEFT,
        right=DENSITY_MAP_GRID_RIGHT,
        bottom=0.27,
        top=0.84,
    )
    axis = fig.add_subplot(grid[0, 0])

    for strategy in STRATEGIES:
        center, lower, upper = stats[strategy]
        color = get_color(strategy, "#666666")
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.075,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            center,
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
            zorder=2,
        )

    center, lower, upper = master
    axis.fill_between(
        x,
        lower,
        upper,
        color="black",
        alpha=0.14,
        linewidth=0,
        zorder=7,
    )
    axis.plot(
        x,
        center,
        color="black",
        linewidth=2.2,
        linestyle=linestyle,
        zorder=10,
    )

    axis.axvspan(
        X_LIMITS[0],
        GREYED_X_MAX,
        color="white",
        alpha=1.0 - GREYED_ALPHA,
        linewidth=0,
        antialiased=False,
        zorder=20,
    )
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=30)
    axis.axvline(
        GREYED_X_MAX,
        color="#B8B8B8",
        linewidth=0.6,
        linestyle="--",
        zorder=30,
    )
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*RELATIVE_Y_LIMITS)
    axis.set_xlabel(DENSITY_X_LABEL, fontsize=7, labelpad=0)
    axis.set_ylabel(
        RELATIVE_Y_LABEL,
        fontsize=7,
        labelpad=2,
        color=MASTER_CURVE_Y_AXIS_COLOR,
    )
    ft.format_axes(axis)
    axis.spines["left"].set_color(MASTER_CURVE_Y_AXIS_COLOR)
    axis.tick_params(axis="y", colors=MASTER_CURVE_Y_AXIS_COLOR)
    format_density_ticks(axis)
    axis.legend(
        handles=legend_handles(is_simulation),
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.35,
    )
    for spine in axis.spines.values():
        spine.set_zorder(30)
    axis.tick_params(axis="both", zorder=30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output_path, transparent=False)
    plt.close(fig)
    print(f"[saved] {output_path}")


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.kernel_radius_um) or args.kernel_radius_um <= 0:
        raise ValueError("--kernel-radius-um must be positive.")

    default_data_root, default_output_root = default_roots(args.kernel_radius_um)
    data_root = (
        default_data_root
        if args.data_root is None
        else args.data_root.expanduser()
    )
    output_root = (
        default_output_root
        if args.output_dir is None
        else args.output_dir.expanduser()
    )

    for dataset in ("simulation", "experiment"):
        run_curves = load_run_curves(data_root, dataset, args.kernel_radius_um)
        stats = strategy_statistics(run_curves)
        master = master_statistics(stats)
        draw_plot(
            dataset,
            stats,
            master,
            output_root
            / f"relative_change_before_tte_{dataset}_radius_{args.kernel_radius_um:g}um.pdf",
        )


if __name__ == "__main__":
    main()
