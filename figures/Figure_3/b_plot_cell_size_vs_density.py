#!/usr/bin/env python3
"""Plot single-cell expansion against local density from pressure-analysis HDF5 files.

Direct inputs (one file per treatment protocol, for each radius)::

    Data/figure_data/Figure_3/cell_size_vs_density/kernel_radius_<120|160>um/AT50.h5
    Data/figure_data/Figure_3/cell_size_vs_density/kernel_radius_<120|160>um/AT100.h5
    Data/figure_data/Figure_3/cell_size_vs_density/kernel_radius_<120|160>um/CT.h5

Each file must have the collaborator pipeline's hierarchy
``{run}/cells/{window}/{cell_type,rho_tot,delta_d}``. By default, the script
writes the main PDF to ``Data/figure_plots/Figure_3/cell_size_vs_density``.
Generate these inputs with ``analysis/figure_3/HDF5_analysis/compute_cell_size_density.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from scipy.stats import binned_statistic


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [os.fspath(ROOT)]

from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir
from shared.strategy_colors import FIGURE_COLOR_MAP
from analysis.figure_3.HDF5_analysis import constants as shared


# Input/output locations follow the repository-wide data layout. The input
# directory is not created here: missing analysis data should fail explicitly.
DEFAULT_KERNEL_RADIUS_UM = 160.0
DEFAULT_INPUT_DIR = figure_data_dir(
    "Figure_3", "cell_size_vs_density", f"kernel_radius_{DEFAULT_KERNEL_RADIUS_UM:g}um", create=False)
DEFAULT_OUTPUT_DIR = figure_plot_dir("Figure_3", "cell_size_vs_density", create=False)


def input_directory(radius_um):
    return DEFAULT_INPUT_DIR.parent / f"kernel_radius_{radius_um:g}um"

PROTOCOLS = ("AT50", "AT100", "CT")

# Values carried over from the pressure-analysis metadata that generated the
# HDF5 files. They determine only unit conversion, binning, and reference lines.
PX_TO_MICRON = shared.px_to_micron
A_MAX_EXP_PX2 = shared.A_max_exp
A_MAX_SIM = shared.A_max_sim
EXP_SCALING = np.sqrt(A_MAX_EXP_PX2 / A_MAX_SIM)
UNIT_SCALE_UM = EXP_SCALING * PX_TO_MICRON
DENSITY_SCALE = 1000.0 / UNIT_SCALE_UM**2
OBSERVATION_HOURS = 12.0

RHO_MAX = shared.prolif_rho_max
RHO_EDGES: np.ndarray = np.linspace(0.0, RHO_MAX, shared.prolif_n_rho_bins + 1)
RHO_CENTERS: np.ndarray = np.asarray(
    0.5 * (RHO_EDGES[:-1] + RHO_EDGES[1:]), dtype=float
)
MIN_BIN_COUNT = 100
GREYED_X_MAX = 0.4
X_AXIS_MIN = 0.25
X_AXIS_MAX = 1.5
Y_AXIS_MAX = 1.8
GREYED_ALPHA = 0.20

RHO_HOMEOSTATIC_SENSITIVE = 0.291241
RHO_HOMEOSTATIC_RES = 0.199453

# CellType values written by the collaborator's simulation model.
CELL_TYPE_SENSITIVE = 0
CELL_TYPE_RESISTANT = 1

COLOR_SENSITIVE = FIGURE_COLOR_MAP["blue"]
COLOR_RESISTANT = FIGURE_COLOR_MAP["yellow"]
REFERENCE_COLOR = "#7F7F7F"
LINEWIDTH = ft.NATURE_DEFAULT_AXIS_LINEWIDTH_PT
DENSITY_AXIS_LABEL = r"Local total density, $\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
def format_density_x_ticks(ax: plt.Axes) -> None:
    """Label alternate density ticks while retaining all tick marks."""
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                (
                    f"{value:.1f}"
                    if np.isclose(value, 0.0)
                    else (
                        rf"${value:.2f}\times 10^{{-3}}$"
                        if np.isclose(value, X_AXIS_MIN)
                        else rf"${value:.1f}\times 10^{{-3}}$"
                    )
                )
                if np.isclose(value % 0.5, 0.0, atol=1.0e-9)
                else ""
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-radius-um", type=float, choices=shared.MASTER_KERNEL_RADII_UM,
                        default=None, help="One radius; omitted: plot both 120 and 160 um.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing AT50.h5, AT100.h5, and CT.h5 for one radius; with both radii, parent of kernel_radius_*um directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for the main PDF (and optional verification PDF).",
    )
    parser.add_argument(
        "--save-verification",
        action="store_true",
        help="Also write the verification distance-distributions PDF.",
    )
    return parser.parse_args()


def _required_input_files(input_dir: Path) -> dict[str, Path]:
    paths = {protocol: input_dir / f"{protocol}.h5" for protocol in PROTOCOLS}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing pressure-analysis input files:\n"
            f"{formatted}\n"
            "Place the three derived HDF5 files in the default input directory "
            "or pass --input-dir."
        )
    return paths


def load_pooled_data(input_dir: Path, kernel_radius_um=DEFAULT_KERNEL_RADIUS_UM) -> dict[str, np.ndarray]:
    """Pool finite density/expansion observations across protocols and runs."""
    paths = _required_input_files(input_dir)
    pooled: dict[str, list[np.ndarray]] = {
        "rho_sus": [],
        "delta_sus": [],
        "rho_res": [],
        "delta_res": [],
    }
    observed_cell_types: set[int] = set()

    for protocol, path in paths.items():
        with h5py.File(path, "r") as handle:
            shared.validate_cell_density_settings(handle.attrs, kernel_radius_um, source=path)
            for run_name in handle:
                if "cells" not in handle[run_name]:
                    raise KeyError(f"Missing group '{run_name}/cells' in {path}")
                for window_name, group in handle[run_name]["cells"].items():
                    required = {"cell_type", "rho_tot", "delta_d"}
                    missing = required.difference(group.keys())
                    if missing:
                        location = f"{run_name}/cells/{window_name}"
                        raise KeyError(
                            f"Missing datasets {sorted(missing)} in {path}:{location}"
                        )

                    cell_type = np.asarray(group["cell_type"][:])
                    rho_tot = np.asarray(group["rho_tot"][:], dtype=float)
                    delta_d = np.asarray(group["delta_d"][:], dtype=float)
                    if not (cell_type.shape == rho_tot.shape == delta_d.shape):
                        raise ValueError(
                            f"Shape mismatch in {path}:{run_name}/cells/{window_name}"
                        )

                    observed_cell_types.update(
                        int(value) for value in np.unique(cell_type)
                    )
                    valid = np.isfinite(rho_tot) & np.isfinite(delta_d)
                    sensitive = valid & (cell_type == CELL_TYPE_SENSITIVE)
                    resistant = valid & (cell_type == CELL_TYPE_RESISTANT)
                    pooled["rho_sus"].append(rho_tot[sensitive])
                    pooled["delta_sus"].append(delta_d[sensitive])
                    pooled["rho_res"].append(rho_tot[resistant])
                    pooled["delta_res"].append(delta_d[resistant])

        print(f"[loaded] {protocol}: {path}")

    expected_types = {CELL_TYPE_SENSITIVE, CELL_TYPE_RESISTANT}
    if not expected_types.issubset(observed_cell_types):
        raise ValueError(
            "Expected cell_type codes 0 (sensitive) and 1 (resistant), but "
            f"found {sorted(observed_cell_types)}."
        )

    result = {
        name: np.concatenate(arrays) if arrays else np.empty(0)
        for name, arrays in pooled.items()
    }
    if result["delta_sus"].size == 0 or result["delta_res"].size == 0:
        raise ValueError("No finite sensitive or resistant observations were found.")

    print(
        "[pooled] "
        f"sensitive={result['delta_sus'].size:,}, "
        f"resistant={result['delta_res'].size:,}"
    )
    return result


def binned_summary(rho: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    q25 = lambda sample: np.percentile(sample, 25)
    q75 = lambda sample: np.percentile(sample, 75)
    median = np.asarray(
        binned_statistic(rho, values, statistic="median", bins=RHO_EDGES)[0],
        dtype=float,
    )
    lower = np.asarray(
        binned_statistic(rho, values, statistic=q25, bins=RHO_EDGES)[0],
        dtype=float,
    )
    upper = np.asarray(
        binned_statistic(rho, values, statistic=q75, bins=RHO_EDGES)[0],
        dtype=float,
    )
    count = np.asarray(
        binned_statistic(rho, values, statistic="count", bins=RHO_EDGES)[0],
        dtype=float,
    )
    enough = count >= MIN_BIN_COUNT
    return {
        "median": np.where(enough, median, np.nan),
        "q25": np.where(enough, lower, np.nan),
        "q75": np.where(enough, upper, np.nan),
        "count": count,
    }


def _plot_median_iqr(
    ax: plt.Axes,
    x: np.ndarray,
    summary: dict[str, np.ndarray],
    color: str,
    label: str,
) -> None:
    median = summary["median"] * UNIT_SCALE_UM / OBSERVATION_HOURS
    lower = summary["q25"] * UNIT_SCALE_UM / OBSERVATION_HOURS
    upper = summary["q75"] * UNIT_SCALE_UM / OBSERVATION_HOURS

    ax.plot(
        x,
        median,
        color=color,
        linestyle="--",
        linewidth=2 * LINEWIDTH,
        label=label,
    )
    ax.fill_between(
        x, lower, upper, color=color, alpha=0.25, linewidth=0
    )


def plot_main_figure(
    data: dict[str, np.ndarray],
    summaries: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    kernel_radius_um=DEFAULT_KERNEL_RADIUS_UM,
) -> Path:
    fig, ax_curve = plt.subplots()
    ft.set_axes_size_mm(
        ax_curve,
        width_mm=50,
        height_mm=30,
        left_mm=14,
        right_mm=6,
        bottom_mm=13,
        top_mm=5,
    )

    scaled_centers = RHO_CENTERS * DENSITY_SCALE
    _plot_median_iqr(
        ax_curve,
        scaled_centers,
        summaries["res"],
        COLOR_RESISTANT,
        "Resistant",
    )
    for density, color in ((RHO_HOMEOSTATIC_RES, COLOR_RESISTANT),):
        ax_curve.vlines(
            density * DENSITY_SCALE,
            ymin=-0.2 * UNIT_SCALE_UM / OBSERVATION_HOURS,
            ymax=1.5,
            color="black",
            linestyle="--",
            linewidth=LINEWIDTH,
            zorder=30,
        )
    ax_curve.set_ylabel(
        r"Cell expansion rate, $\mathit{\lambda}$ [µm h$^{-1}$]"
    )

    first_valid = np.flatnonzero(
        np.isfinite(summaries["res"]["median"])
    )
    if first_valid.size == 0:
        raise ValueError(f"No density bin contains at least {MIN_BIN_COUNT} observations.")
    x_limits = (
        X_AXIS_MIN,
        X_AXIS_MAX,
    )
    ax_curve.axhline(0.0, color=REFERENCE_COLOR, linewidth=LINEWIDTH, zorder=30)
    ax_curve.set_xlim(*x_limits)
    ax_curve.set_xlabel(DENSITY_AXIS_LABEL, labelpad=0)
    ft.format_axes(ax_curve)
    format_density_x_ticks(ax_curve)
    ax_curve.set_ylim(
        -0.2 * UNIT_SCALE_UM / OBSERVATION_HOURS,
        Y_AXIS_MAX * UNIT_SCALE_UM / OBSERVATION_HOURS,
    )
    for left, right in (
        (x_limits[0], GREYED_X_MAX),
        (RHO_HOMEOSTATIC_RES * DENSITY_SCALE, x_limits[1]),
    ):
        ax_curve.axvspan(
            left,
            right,
            color="white",
            alpha=1.0 - GREYED_ALPHA,
            zorder=20,
            linewidth=0,
            antialiased=False,
        )
    ax_curve.axvline(
        GREYED_X_MAX,
        color="#B8B8B8",
        linewidth=LINEWIDTH,
        linestyle="--",
        zorder=30,
    )
    for spine in ax_curve.spines.values():
        spine.set_zorder(30)
    ax_curve.tick_params(axis="both", zorder=30)

    output_path = output_dir / f"delta_d_vs_density_radius_{kernel_radius_um:g}um.pdf"
    ft.savefig(fig, output_path)
    plt.close(fig)
    return output_path


def plot_verification_figure(
    data: dict[str, np.ndarray],
    summaries: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    kernel_radius_um=DEFAULT_KERNEL_RADIUS_UM,
) -> Path:
    valid_bins = np.flatnonzero(
        (summaries["sus"]["count"] >= MIN_BIN_COUNT)
        | (summaries["res"]["count"] >= MIN_BIN_COUNT)
    )
    if valid_bins.size == 0:
        raise ValueError("No sufficiently populated bins are available for verification.")
    selected_bins = np.unique(
        valid_bins[np.linspace(0, valid_bins.size - 1, 8).astype(int)]
    )

    ncols = 4
    nrows = int(np.ceil(selected_bins.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    ft.set_axes_grid_size_mm(
        axes,
        nrows=nrows,
        ncols=ncols,
        ax_width_mm=36,
        ax_height_mm=30,
        left_mm=13,
        right_mm=4,
        bottom_mm=12,
        top_mm=5,
        wspace_mm=5,
        hspace_mm=8,
    )
    flat_axes = axes.reshape(-1)

    for ax, raw_bin_index in zip(flat_axes, selected_bins):
        bin_index = int(raw_bin_index)
        lower_edge, upper_edge = RHO_EDGES[bin_index : bin_index + 2]
        select_sus = (
            (data["rho_sus"] >= lower_edge) & (data["rho_sus"] < upper_edge)
        )
        select_res = (
            (data["rho_res"] >= lower_edge) & (data["rho_res"] < upper_edge)
        )
        if np.any(select_sus):
            ax.hist(
                data["delta_sus"][select_sus], bins=40, density=True,
                color=COLOR_SENSITIVE, alpha=0.4, label="Sensitive",
            )
        if np.any(select_res):
            ax.hist(
                data["delta_res"][select_res], bins=40, density=True,
                color=COLOR_RESISTANT, alpha=0.4, label="Resistant",
            )

        y_max = ax.get_ylim()[1]
        for summary, color, y_fraction in (
            (summaries["sus"], COLOR_SENSITIVE, 0.92),
            (summaries["res"], COLOR_RESISTANT, 0.84),
        ):
            median = summary["median"][bin_index]
            if np.isfinite(median):
                ax.axvline(
                    median, color=color, linestyle="--", linewidth=LINEWIDTH
                )
                ax.hlines(
                    y_fraction * y_max,
                    summary["q25"][bin_index],
                    summary["q75"][bin_index],
                    color="black",
                    linewidth=2 * LINEWIDTH,
                )

        scaled_density = float(RHO_CENTERS[bin_index] * DENSITY_SCALE)
        ax.set_title(rf"{scaled_density:.1f} cells (1000 µm$^2$)$^{{-1}}$")
        ax.set_xlabel(r"$\Delta d$ [simulation units]")
        ft.format_axes(ax)

    for ax in flat_axes[selected_bins.size :]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False)

    output_path = output_dir / f"verification_distance_distributions_radius_{kernel_radius_um:g}um.pdf"
    ft.savefig(fig, output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    # Apply this after the shared style so every plot element uses Arial.
    plt.rcParams.update({"font.family": "Arial"})

    radii = (args.kernel_radius_um,) if args.kernel_radius_um is not None else shared.MASTER_KERNEL_RADII_UM
    for radius in radii:
        input_dir = input_directory(radius)
        if args.input_dir is not None:
            input_dir = args.input_dir.expanduser()
            if len(radii) > 1:
                input_dir /= input_directory(radius).name
        data = load_pooled_data(input_dir, radius)
        summaries = {
            "sus": binned_summary(data["rho_sus"], data["delta_sus"]),
            "res": binned_summary(data["rho_res"], data["delta_res"]),
        }
        output_dir = args.output_dir.expanduser()
        main_path = plot_main_figure(data, summaries, output_dir, radius)
        print(f"[saved] {main_path}")
        if args.save_verification:
            verification_path = plot_verification_figure(data, summaries, output_dir, radius)
            print(f"[saved] {verification_path}")


if __name__ == "__main__":
    main()
