#!/usr/bin/env python3
"""Plot all selected kernel-radius diagnostics in one multipanel figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HDF5_ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "figure_3" / "HDF5_analysis"
for import_path in (PROJECT_ROOT, HDF5_ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from shared import figtools as ft
import plot_hdf5_master_curves as base
from shared.data_paths import figure_data_dir
from shared.strategy_colors import get_color

DEFAULT_KERNEL_RADII_UM = (80.0, 100.0, 120.0, 160.0)
STRATEGIES = ("CT", "AT50", "AT100")
PERIODS = ("all_timepoints", "before_tte")
VARIANTS = (
    "current",
    "t0_density",
)
PANEL_ROWS = (
    ("experiment", "current", "Experiment\nCurrent"),
    ("experiment", "t0_density", "Experiment\nFirst-endpoint density"),
    ("simulation", "current", "Simulation\nCurrent"),
    ("simulation", "t0_density", "Simulation\nFirst-endpoint density"),
)
DEFAULT_DATA_ROOT = figure_data_dir(
    "Figure_3",
    "HDF5_analysis",
    "flux_vs_density",
    create=False,
)
DEFAULT_OUTPUT_DIR = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\figure_plots\Figure_3\density_mastercurve"
)
X_LIMITS = (0.0, 1.5)
DIAGNOSTIC_DENSITY_X_LABEL = r"$\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
DIAGNOSTIC_RELATIVE_Y_LABEL = (
    r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$ [day$^{-1}$]"
)
DIAGNOSTIC_ABSOLUTE_Y_LABEL = (
    r"$\Delta\rho_\mathrm{res}/\Delta t$ [cells µm$^{-2}$ day$^{-1}$]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Plot one explicitly selected diagnostic HDF5. Its stored kernel "
            "radius is used unless --kernel-radii-um supplies one value."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--kernel-radii-um",
        nargs="+",
        type=float,
        default=None,
        help="Kernel radii to plot (default: 80 100 120 160).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--growth-modes",
        nargs="+",
        choices=("relative", "absolute"),
        default=("relative", "absolute"),
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=PERIODS,
        default=PERIODS,
    )
    return parser.parse_args()


def radius_token(radius_um: float) -> str:
    return f"{radius_um:g}".replace(".", "p")


def default_input_path(data_root: Path, radius_um: float) -> Path:
    token = radius_token(radius_um)
    return (
            data_root
            / f"kernel_radius_{token}um"
            / "selection_diagnostics"
            / f"master_curve_selection_diagnostics_radius_{token}um.h5"
    )


def stored_radius(input_path: Path) -> float:
    candidates = (input_path, input_path.with_suffix(".building.h5"))
    for candidate in candidates:
        try:
            with h5py.File(candidate, "r") as handle:
                return float(handle.attrs["kernel_radius_um"])
        except (KeyError, OSError):
            continue
    raise RuntimeError(f"Cannot read the kernel radius from {input_path}")


def supports_requests(
        input_path: Path,
        growth_modes: list[str] | tuple[str, ...],
        periods: list[str] | tuple[str, ...],
) -> bool:
    dataset_names = {
        "relative": "median_rel",
        "absolute": "median",
    }
    try:
        with h5py.File(input_path, "r") as handle:
            for dataset in ("experiment", "simulation"):
                for strategy in STRATEGIES:
                    for period in periods:
                        for variant in VARIANTS:
                            group = handle[dataset][strategy][period][variant]
                            if any(
                                    dataset_names[mode] not in group
                                    for mode in growth_modes
                            ):
                                return False
    except (KeyError, OSError):
        return False
    return True


def load_statistics(
        input_path: Path,
        growth_mode: str,
        period: str,
        radius_um: float,
) -> dict[str, dict[str, tuple[dict[str, tuple[np.ndarray, ...]], tuple[np.ndarray, ...]]]]:
    dataset_name = "median_rel" if growth_mode == "relative" else "median"
    results = {}
    with h5py.File(input_path, "r") as handle:
        actual_radius = float(handle.attrs["kernel_radius_um"])
        if not np.isclose(actual_radius, radius_um):
            raise ValueError(
                f"Expected a {radius_um:g} um file, got {actual_radius:g} um: {input_path}"
            )
        for dataset in ("experiment", "simulation"):
            results[dataset] = {}
            for variant in VARIANTS:
                run_curves = {
                    strategy: np.asarray(
                        handle[dataset][strategy][period][variant][dataset_name][:],
                        dtype=float,
                    )
                    for strategy in STRATEGIES
                }
                strategy_stats = base.strategy_statistics(run_curves)
                results[dataset][variant] = (
                    strategy_stats,
                    base.master_statistics(strategy_stats),
                )
    return results


def draw_panel(
        axis: plt.Axes,
        dataset: str,
        strategy_stats: dict[str, tuple[np.ndarray, ...]],
        master: tuple[np.ndarray, ...],
        title: str,
        show_legend: bool,
        growth_mode: str,
        y_limits: tuple[float, float],
) -> None:
    linestyle = "--" if dataset == "simulation" else "-"
    x = base.density_axis(dataset)
    scale = 1.0 if growth_mode == "relative" else base.density_scale(dataset)
    for strategy in STRATEGIES:
        center, lower, upper = strategy_stats[strategy]
        color = get_color(strategy, "#666666")
        axis.fill_between(
            x, lower * scale, upper * scale,
            color=color, alpha=0.20, linewidth=0, zorder=1
        )
        axis.plot(
            x, center * scale,
            color=color, linewidth=1.0, linestyle=linestyle, zorder=2
        )
    center, lower, upper = master
    axis.fill_between(
        x, lower * scale, upper * scale,
        color="black", alpha=0.15, linewidth=0, zorder=7
    )
    axis.plot(
        x, center * scale,
        color="black", linewidth=2.2, linestyle=linestyle, zorder=10
    )
    axis.axvspan(
        X_LIMITS[0],
        base.GREYED_X_MAX,
        color="white",
        alpha=1.0 - base.GREYED_ALPHA,
        linewidth=0,
        antialiased=False,
        zorder=20,
    )
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=30)
    axis.axvline(
        base.GREYED_X_MAX,
        color="#B8B8B8",
        linewidth=0.6,
        linestyle="--",
        zorder=30,
    )
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*y_limits)
    axis.set_title(title, fontsize=7, pad=3)
    ft.format_axes(axis)
    format_density_ticks(axis)
    axis.spines["left"].set_color("black")
    axis.tick_params(axis="y", colors="black")
    if show_legend:
        axis.legend(
            handles=base.legend_handles(dataset == "simulation"),
            loc="upper right",
            frameon=False,
            ncol=1,
            handlelength=1.8,
            handletextpad=0.35,
        )


def format_density_ticks(axis: plt.Axes) -> None:
    ticks = np.arange(X_LIMITS[0], X_LIMITS[1] + 0.001, 0.25)
    labels = []
    for value in ticks:
        if np.isclose(value, 0.0):
            labels.append("0.0")
        elif np.isclose(value % 0.5, 0.0, atol=1e-9):
            labels.append(rf"${value:.1f}\times 10^{{-3}}$")
        else:
            labels.append("")
    axis.set_xticks(ticks, labels)


def common_y_limits(
        statistics_collection: list[
            dict[
                str,
                dict[
                    str,
                    tuple[dict[str, tuple[np.ndarray, ...]], tuple[np.ndarray, ...]],
                ],
            ]
        ],
        growth_mode: str,
) -> tuple[float, float]:
    if growth_mode == "relative":
        return base.RELATIVE_Y_LIMITS
    values = []
    for statistics in statistics_collection:
        for dataset, variant_statistics in statistics.items():
            x = base.density_axis(dataset)
            region = (x >= base.GREYED_X_MAX) & (x <= X_LIMITS[1])
            scale = base.density_scale(dataset)
            for strategy_stats, master in variant_statistics.values():
                for arrays in (*strategy_stats.values(), master):
                    selected = np.asarray(arrays, dtype=float)[..., region] * scale
                    values.extend(selected[np.isfinite(selected)])
    if not values:
        return -0.05, 0.10
    lower = min(0.0, float(min(values)))
    upper = max(0.0, float(max(values)))
    padding = 0.08 * (upper - lower) if upper > lower else 0.05
    return lower - padding, upper + padding


def output_path(output_dir: Path, growth_mode: str, period: str) -> Path:
    return output_dir / (
        f"master_curve_selection_diagnostics_{growth_mode}_"
        f"{period}_all_kernel_radii.pdf"
    )


def resolve_input(
        input_path: Path,
        radius_um: float,
        growth_modes: list[str] | tuple[str, ...],
        periods: list[str] | tuple[str, ...],
) -> Path:
    building_path = input_path.with_suffix(".building.h5")
    if not supports_requests(input_path, growth_modes, periods):
        if supports_requests(building_path, growth_modes, periods):
            print(f"[input] using completed building file: {building_path}")
            input_path = building_path
        else:
            raise RuntimeError(
                f"The {radius_um:g} um diagnostic HDF5 does not contain every "
                "requested growth mode. Rerun "
                "compute_master_curves.py."
            )
    return input_path


def draw_combined_figure(
        statistics_by_radius: dict[
            float,
            dict[
                tuple[str, str],
                dict[
                    str,
                    dict[
                        str,
                        tuple[
                            dict[str, tuple[np.ndarray, ...]],
                            tuple[np.ndarray, ...],
                        ],
                    ],
                ],
            ],
        ],
        radii: tuple[float, ...],
        growth_mode: str,
        period: str,
        y_limits: tuple[float, float],
        output_dir: Path,
) -> None:
    ylabel = (
        DIAGNOSTIC_RELATIVE_Y_LABEL
        if growth_mode == "relative"
        else DIAGNOSTIC_ABSOLUTE_Y_LABEL
    )
    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={"axes.titlesize": 7, "axes.labelsize": 7},
    )
    fig, axes = plt.subplots(
        len(PANEL_ROWS),
        len(radii),
        figsize=(1.9 * len(radii), 6.9),
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={
            "left": 0.105,
            "right": 0.985,
            "bottom": 0.09,
            "top": 0.965,
            "wspace": 0.18,
            "hspace": 0.25,
        },
    )
    for column, radius_um in enumerate(radii):
        statistics = statistics_by_radius[radius_um][(period, growth_mode)]
        for row, (dataset, variant, row_label) in enumerate(PANEL_ROWS):
            strategy_stats, master = statistics[dataset][variant]
            draw_panel(
                axes[row, column],
                dataset,
                strategy_stats,
                master,
                title=f"{radius_um:g} µm" if row == 0 else "",
                show_legend=column == 0 and row in (0, 2),
                growth_mode=growth_mode,
                y_limits=y_limits,
            )
            if column != 0:
                axes[row, column].tick_params(labelleft=False)
            else:
                axes[row, column].set_ylabel(
                    f"{row_label}\n{ylabel}",
                    fontsize=7,
                    labelpad=2,
                    color="black",
                )
    for axis in axes[-1, :]:
        axis.set_xlabel(DIAGNOSTIC_DENSITY_X_LABEL, fontsize=7, labelpad=0)

    selected_output = output_path(output_dir, growth_mode, period)
    ft.savefig(fig, selected_output, transparent=False)
    plt.close(fig)
    print(f"[saved] {selected_output}")


def plot_all_radii(
        jobs: tuple[tuple[float, Path], ...],
        output_dir: Path,
        growth_modes: list[str] | tuple[str, ...],
        periods: list[str] | tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    radii = tuple(radius_um for radius_um, _ in jobs)
    statistics_by_radius = {}
    for radius_um, candidate_path in jobs:
        print(f"[radius] loading {radius_um:g} um")
        input_path = resolve_input(
            candidate_path, radius_um, growth_modes, periods
        )
        statistics_by_radius[radius_um] = {
            (period, growth_mode): load_statistics(
                input_path, growth_mode, period, radius_um
            )
            for period in periods
            for growth_mode in growth_modes
        }

    y_limits_by_growth_mode = {
        growth_mode: common_y_limits(
            [
                statistics_by_radius[radius_um][(period, growth_mode)]
                for radius_um in radii
                for period in periods
            ],
            growth_mode,
        )
        for growth_mode in growth_modes
    }
    for period in periods:
        for growth_mode in growth_modes:
            draw_combined_figure(
                statistics_by_radius=statistics_by_radius,
                radii=radii,
                growth_mode=growth_mode,
                period=period,
                y_limits=y_limits_by_growth_mode[growth_mode],
                output_dir=output_dir,
            )


def main() -> None:
    args = parse_args()
    requested_radii = (
        tuple(args.kernel_radii_um)
        if args.kernel_radii_um is not None
        else DEFAULT_KERNEL_RADII_UM
    )
    if any(not np.isfinite(radius) or radius <= 0 for radius in requested_radii):
        raise ValueError("Kernel radii must be finite positive values.")

    if args.input is not None:
        if args.kernel_radii_um is not None and len(requested_radii) != 1:
            raise ValueError("--input accepts exactly one --kernel-radii-um value.")
        input_path = args.input.expanduser()
        radius_um = (
            requested_radii[0]
            if args.kernel_radii_um is not None
            else stored_radius(input_path)
        )
        jobs = ((radius_um, input_path),)
    else:
        data_root = args.data_root.expanduser()
        jobs = tuple(
            (radius_um, default_input_path(data_root, radius_um))
            for radius_um in requested_radii
        )

    plot_all_radii(
        jobs=jobs,
        output_dir=args.output_dir.expanduser(),
        growth_modes=args.growth_modes,
        periods=args.periods,
    )


if __name__ == "__main__":
    main()
