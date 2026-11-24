#!/usr/bin/env python3
"""Export HDF5 master curves using the shared Figure 3 defaults.

The default inputs are the per-run/per-well t0-density diagnostics written by
``compute_master_curves.py``. For each selected time period and growth
definition, this script exports one simulation and one experiment panel.  Its
defaults save 160 µm all-timepoints relative plots with density at t0 in
the shared Figure 3 ``density_mastercurve`` directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
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
    rho_homeostatic_res,
    MASTER_KERNEL_RADIUS_UM, MASTER_GROWTH_MODE, MASTER_DENSITY_COORDINATE, MASTER_PERIOD,
    is_non_artificial_run,
)
from shared.data_paths import figure_data_dir, figure_plot_dir
from analysis.figure_3.HDF5_analysis.constants import validate_kernel_settings
from shared.strategy_colors import get_color, get_display_name

STRATEGIES = ("CT", "AT50", "AT100")
PERIODS = ("before_tte", "after_tte", "all_timepoints", "before_ttf")
PERIOD_LABELS = {
    "before_tte": "Before TTE",
    "after_tte": "After TTE",
    "all_timepoints": "All timepoints",
    "before_ttf": "Before TTF",
}
GROWTH_MODES = ("relative", "absolute")
DENSITY_COORDINATES = ("first_endpoint", "endpoint_mean")
SOURCE_DATA_SUBDIR = "HDF5_analysis"
OUTPUT_SUBDIR = "HDF5_analysis"

X_LIMITS = (0.25, 1.5)
GREYED_X_MAX = 0.4
GREYED_ALPHA = 0.20
RELATIVE_Y_LIMITS = (0.0, 0.6)
MASTER_CURVE_Y_AXIS_COLOR = "black"
DEFAULT_OUTPUT_DIR = figure_plot_dir("Figure_3", "density_mastercurve", create=False)
DENSITY_X_LABEL = r"Local total density, $\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
FIRST_ENDPOINT_DENSITY_X_LABEL = (
    r"Local total density, $\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
)
RELATIVE_Y_LABEL = (
    "Rel resistant density change,\n"
    r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$ [day$^{-1}$]"
)
ABSOLUTE_Y_LABEL = r"$\Delta\rho_\mathrm{res}/\Delta t$ [cells µm$^{-2}$ day$^{-1}$]"

DENSITY_MAP_PANEL_IN = 0.85
DENSITY_MAP_GRID_GAP = 0.06
DENSITY_MAP_GRID_LEFT = 0.18
DENSITY_MAP_GRID_RIGHT = 0.945
FIGURE_WIDTH_IN = (
        DENSITY_MAP_PANEL_IN * (3 + 2 * DENSITY_MAP_GRID_GAP)
        / (DENSITY_MAP_GRID_RIGHT - DENSITY_MAP_GRID_LEFT)
)
FIGURE_SIZE = (FIGURE_WIDTH_IN, 1.92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-radius-um",
        type=float,
        default=None,
        help="One kernel radius; omitted: plot both 120 and 160 um.",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=PERIODS,
        default=(MASTER_PERIOD,),
    )
    parser.add_argument(
        "--growth-modes",
        nargs="+",
        choices=GROWTH_MODES,
        default=(MASTER_GROWTH_MODE,),
    )
    parser.add_argument(
        "--density-coordinate",
        choices=DENSITY_COORDINATES,
        default=MASTER_DENSITY_COORDINATE,
        help=(
            "Density used for binning. The default reads the t0-density curves "
            "from the selection-diagnostics HDF5."
        ),
    )
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=None,
        help="Optional selection-diagnostics HDF5 for first_endpoint mode.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        default=True,
        help="Write PDFs directly into --output-dir instead of period/growth subdirectories.",
    )
    parser.add_argument(
        "--nested-output",
        action="store_false",
        dest="flat_output",
        help="Write PDFs into period/growth subdirectories.",
    )
    return parser.parse_args()


def radius_directory(radius_um: float) -> str | None:
    if np.isclose(radius_um, 160.0):
        return None
    return f"kernel_radius_{float(radius_um):g}".replace(".", "p") + "um"


def default_roots(radius_um: float) -> tuple[Path, Path]:
    data_root = figure_data_dir("Figure_3", SOURCE_DATA_SUBDIR, "flux_vs_density", create=False)
    variant = radius_directory(radius_um)
    if variant is not None:
        data_root /= variant
    # Radius-specific input locations do not change the plotting destination.
    return data_root, DEFAULT_OUTPUT_DIR


def input_path(data_root: Path, dataset: str, strategy: str) -> Path:
    return data_root / dataset / f"flux_vs_density_{strategy}.h5"


def default_selection_input(radius_um: float) -> Path:
    token = f"{radius_um:g}".replace(".", "p")
    return figure_data_dir(
        "Figure_3",
        SOURCE_DATA_SUBDIR,
        "flux_vs_density",
        f"kernel_radius_{token}um",
        "selection_diagnostics",
        create=False,
    ) / f"master_curve_selection_diagnostics_radius_{token}um.h5"


def density_scale(dataset: str) -> float:
    if dataset == "simulation":
        return 1000.0 / (exp_scaling * px_to_micron) ** 2
    if dataset == "experiment":
        return 1000.0 / px_to_micron ** 2
    raise ValueError(f"Unknown dataset: {dataset!r}")


def density_axis(dataset: str) -> np.ndarray:
    edges = np.linspace(0.0, prolif_rho_max, prolif_n_rho_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if dataset == "simulation":
        return centers * density_scale(dataset)
    return centers / AREA_SCALE * density_scale(dataset)


def load_run_curves(
        data_root: Path,
        dataset: str,
        period: str,
        growth_mode: str,
        radius_um: float,
) -> dict[str, np.ndarray]:
    dataset_name = "median_rel" if growth_mode == "relative" else "median"
    curves: dict[str, np.ndarray] = {}
    for strategy in STRATEGIES:
        path = input_path(data_root, dataset, strategy)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset} input for {strategy}: {path}")
        with h5py.File(path, "r") as handle:
            validate_kernel_settings(handle.attrs, radius_um, source=path)
            if period not in handle:
                raise KeyError(f"{path} has no {period!r} group. Rerun the HDF5 computation.")
            values = np.asarray(handle[period][dataset_name][:], dtype=float)
            if values.ndim != 2:
                raise ValueError(f"{path}/{period}/{dataset_name} must be two-dimensional.")
            if dataset == "simulation":
                if "run_names" not in handle:
                    raise KeyError(f"{path} has no run_names dataset for selecting runs >9.")
                run_names = [name.decode() for name in handle["run_names"][:]]
                keep = np.asarray([is_non_artificial_run(name) for name in run_names])
                if len(keep) != len(values):
                    raise ValueError(f"{path} run_names does not match the curve rows.")
                print(f"{strategy}: kept {keep.sum()} of {len(keep)} simulation runs (run >9)")
                values = values[keep]
            curves[strategy] = values
    return curves


def load_first_endpoint_run_curves(
        selection_path: Path,
        dataset: str,
        period: str,
        growth_mode: str,
        radius_um: float,
) -> dict[str, np.ndarray]:
    dataset_name = "median_rel" if growth_mode == "relative" else "median"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Missing first-endpoint diagnostic input: {selection_path}"
        )
    curves = {}
    with h5py.File(selection_path, "r") as handle:
        validate_kernel_settings(handle.attrs, radius_um, source=selection_path)
        for strategy in STRATEGIES:
            strategy_group = handle[dataset][strategy]
            try:
                values = np.asarray(
                    strategy_group[period]["t0_density"][dataset_name][:],
                    dtype=float,
                )
            except KeyError as error:
                raise KeyError(
                    f"{selection_path} has no {dataset}/{strategy}/{period}/"
                    f"t0_density/{dataset_name}. Rerun the selection-diagnostics "
                    "computation."
                ) from error
            if values.ndim != 2:
                raise ValueError(
                    f"{selection_path}/{dataset}/{strategy}/{period}/t0_density/"
                    f"{dataset_name} must be two-dimensional."
                )
            if dataset == "simulation":
                run_names = [
                    name.decode() if isinstance(name, bytes) else str(name)
                    for name in strategy_group["run_names"][:]
                ]
                keep = np.asarray([is_non_artificial_run(name) for name in run_names])
                if len(keep) != len(values):
                    raise ValueError(
                        f"{selection_path} run_names does not match the curve rows."
                    )
                print(
                    f"{strategy}: kept {keep.sum()} of {len(keep)} simulation "
                    "runs (run >9)"
                )
                values = values[keep]
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
        center = finite_percentile(values, 50)
        lower = finite_percentile(values, 25)
        upper = finite_percentile(values, 75)
        result[strategy] = tuple(
            np.where(support >= MIN_RUNS_PER_BIN, array, np.nan)
            for array in (center, lower, upper)
        )
    return result


def master_statistics(
        strategy_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean strategy curve and mean +/- sample SD across strategies per bin."""
    values = np.stack([strategy_curves[strategy][0] for strategy in STRATEGIES])
    support = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmean(values, axis=0)
        standard_deviation = np.nanstd(values, axis=0, ddof=1)
        lower = center - standard_deviation
        upper = center + standard_deviation
    return tuple(
        np.where(support >= MIN_PROTOCOLS_PER_BIN, array, np.nan)
        for array in (center, lower, upper)
    )


def common_y_limits(
        statistics: dict[str, tuple[
            dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray]]],
        growth_mode: str,
) -> tuple[float, float]:
    if growth_mode == "relative":
        return RELATIVE_Y_LIMITS
    values = []
    for dataset, (strategy_stats, master) in statistics.items():
        x = density_axis(dataset)
        region = (x >= GREYED_X_MAX) & (x <= X_LIMITS[1])
        scale = density_scale(dataset)
        for arrays in (*strategy_stats.values(), master):
            selected = np.asarray(arrays, dtype=float)[..., region] * scale
            values.extend(selected[np.isfinite(selected)])
    if not values:
        return (-0.05, 0.10)
    lower, upper = min(values), max(values)
    lower = min(0.0, float(lower))
    upper = max(0.0, float(upper))
    padding = 0.08 * (upper - lower) if upper > lower else 0.05
    return lower - padding, upper + padding


def format_density_ticks(axis: plt.Axes) -> None:
    axis.set_xticks(np.arange(X_LIMITS[0], X_LIMITS[1] + 0.001, 0.25))

    def density_tick_label(value: float, _position: float) -> str:
        if np.isclose(value % 0.5, 0.0, atol=1e-9) and not np.isclose(value, 0.0):
            return rf"${value:.1f}\times 10^{{-3}}$"
        return ""

    axis.xaxis.set_major_formatter(
        FuncFormatter(density_tick_label)
    )
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))


def legend_handles(is_simulation: bool) -> list[Line2D]:
    linestyle = "--" if is_simulation else "-"
    handles = [
        Line2D([0], [0], color=get_color(strategy, "#666666"), linewidth=1.0,
               linestyle=linestyle, label=get_display_name(strategy, superscript=True))
        for strategy in STRATEGIES
    ]
    master = Line2D([0], [0], color="black", linewidth=2.2,
                    linestyle=linestyle, label="Mean")
    if is_simulation:
        # Keep the black proxy's physical dash length and gap equal to the
        # strategy proxies while retaining its thicker stroke.
        dash_pattern = matplotlib.rcParams["lines.dashed_pattern"]
        if matplotlib.rcParams["lines.scale_dashes"]:
            dash_pattern = tuple(length / master.get_linewidth() for length in dash_pattern)
        master.set_dashes(dash_pattern)
    handles.append(master)
    return handles


def draw_plot(
        dataset: str,
        period: str,
        growth_mode: str,
        strategy_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        master: tuple[np.ndarray, np.ndarray, np.ndarray],
        y_limits: tuple[float, float],
        output_path: Path,
        radius_um: float,
        density_coordinate: str,
) -> None:
    is_simulation = dataset == "simulation"
    linestyle = "--" if is_simulation else "-"
    scale = 1.0 if growth_mode == "relative" else density_scale(dataset)
    x = density_axis(dataset)
    ylabel = RELATIVE_Y_LABEL if growth_mode == "relative" else ABSOLUTE_Y_LABEL

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={
            "axes.titlesize": 8,
            "axes.labelsize": 7,
        },
    )
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(1, 1, left=DENSITY_MAP_GRID_LEFT, right=DENSITY_MAP_GRID_RIGHT,
                            bottom=0.27, top=0.84)
    axis = fig.add_subplot(grid[0, 0])
    for strategy in STRATEGIES:
        center, lower, upper = strategy_stats[strategy]
        color = get_color(strategy, "#666666")
        axis.fill_between(x, lower * scale, upper * scale, color=color, alpha=0.20,
                          linewidth=0, zorder=1)
        axis.plot(x, center * scale, color=color, linewidth=1.0, linestyle=linestyle, zorder=2)
    center, lower, upper = master
    axis.fill_between(x, lower * scale, upper * scale, color="black", alpha=0.15,
                      linewidth=0, zorder=7)
    axis.plot(x, center * scale, color="black", linewidth=2.2, linestyle=linestyle, zorder=10)
    if is_simulation:
        # Non-confluent fading and its cutoff apply only to Figure 3h.
        axis.axvspan(X_LIMITS[0], GREYED_X_MAX, color="white", alpha=1.0 - GREYED_ALPHA,
                     linewidth=0, antialiased=False, zorder=20)
        # Match the resistant homeostatic reference and fading in
        # b_plot_cell_size_vs_density.py (x is in cells / 1000 um^2).
        homeostatic_x = rho_homeostatic_res * density_scale("simulation")
        axis.axvspan(homeostatic_x, X_LIMITS[1], color="white",
                     alpha=1.0 - GREYED_ALPHA, linewidth=0,
                     antialiased=False, zorder=20)
        axis.vlines(homeostatic_x, ymin=0.0, ymax=0.5,
                    color="black", linestyle="--",
                    linewidth=ft.NATURE_DEFAULT_AXIS_LINEWIDTH_PT, zorder=30)
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=30)
    if is_simulation:
        axis.axvline(GREYED_X_MAX, color="#B8B8B8",
                     linewidth=ft.NATURE_DEFAULT_AXIS_LINEWIDTH_PT,
                     linestyle="--", zorder=30)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*y_limits)
    xlabel = (
        FIRST_ENDPOINT_DENSITY_X_LABEL
        if density_coordinate == "first_endpoint"
        else DENSITY_X_LABEL
    )
    axis.set_xlabel(xlabel, fontsize=7, labelpad=0)
    axis.set_ylabel(ylabel, fontsize=7, labelpad=2, color=MASTER_CURVE_Y_AXIS_COLOR)
    ft.format_axes(axis)
    axis.spines["left"].set_color(MASTER_CURVE_Y_AXIS_COLOR)
    axis.tick_params(axis="y", colors=MASTER_CURVE_Y_AXIS_COLOR)
    format_density_ticks(axis)
    legend = axis.legend(handles=legend_handles(is_simulation), loc="upper right", frameon=False,
                         ncol=1, handlelength=1.8, handletextpad=0.35)
    legend.set_zorder(40)
    for spine in axis.spines.values():
        spine.set_zorder(30)
    axis.tick_params(axis="both", zorder=30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output_path, transparent=False)
    plt.close(fig)
    print(f"[saved] {output_path}")


def plot_radius(args) -> None:
    if not np.isfinite(args.kernel_radius_um) or args.kernel_radius_um <= 0:
        raise ValueError("--kernel-radius-um must be positive.")
    default_data_root, default_output_root = default_roots(args.kernel_radius_um)
    data_root = default_data_root if args.data_root is None else args.data_root.expanduser()
    output_root = default_output_root if args.output_dir is None else args.output_dir.expanduser()
    selection_input = (
        default_selection_input(args.kernel_radius_um)
        if args.selection_input is None
        else args.selection_input.expanduser()
    )
    print(f"[density coordinate] {args.density_coordinate}")
    for period in args.periods:
        for growth_mode in args.growth_modes:
            results = {}
            for dataset in ("simulation", "experiment"):
                if args.density_coordinate == "first_endpoint":
                    curves = load_first_endpoint_run_curves(
                        selection_input,
                        dataset,
                        period,
                        growth_mode,
                        args.kernel_radius_um,
                    )
                else:
                    curves = load_run_curves(
                        data_root,
                        dataset,
                        period,
                        growth_mode,
                        args.kernel_radius_um,
                    )
                stats = strategy_statistics(curves)
                results[dataset] = (stats, master_statistics(stats))
            y_limits = common_y_limits(results, growth_mode)
            for dataset, (stats, master) in results.items():
                filename = (
                    f"master_curve_{dataset}_{growth_mode}_{period}_"
                    f"radius_{args.kernel_radius_um:g}um.pdf"
                )
                output_path = (
                    output_root / filename
                    if args.flat_output
                    else output_root / period / growth_mode / filename
                )
                draw_plot(
                    dataset, period, growth_mode, stats, master, y_limits,
                    output_path,
                    args.kernel_radius_um,
                    args.density_coordinate,
                )


def main() -> None:
    args = parse_args()
    radii = (args.kernel_radius_um,) if args.kernel_radius_um is not None else (120.0, 160.0)
    for radius in radii:
        plot_radius(argparse.Namespace(**(vars(args) | {"kernel_radius_um": radius})))


if __name__ == "__main__":
    main()
