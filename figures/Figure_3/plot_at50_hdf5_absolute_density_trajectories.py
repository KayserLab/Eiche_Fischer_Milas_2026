#!/usr/bin/env python3
"""Plot D5/run_18 trajectories with the shared Figure 3 master-curve parameters.

The original spatial mean/SD, centered four-point running average, and 40-day
view are retained. Only the density fields and support use the shared Figure 3
kernel, neighbour, and resistant-density threshold settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
HDF5_DIR = ROOT / "analysis" / "figure_3" / "HDF5_analysis"
for import_path in (ROOT, HDF5_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.figure_3.HDF5_analysis import compute_flux_vs_density as hdf5_sim
from analysis.figure_3.HDF5_analysis import compute_flux_vs_density_experiment as hdf5_exp
from analysis.figure_3.HDF5_analysis.constants import (
    MIN_RESISTANT_NUMERATOR,
    kernel_settings, resistant_support, resistant_change_rate,
    MASTER_KERNEL_RADIUS_UM, MASTER_DENSITY_COORDINATE,
    MASTER_PERIOD, is_non_artificial_run,
)
from shared.data_paths import figure_data_dir, figure_plot_dir
from shared import figtools as ft

EXPERIMENT_RUN = "AT50/D5"
SIMULATION_PROTOCOL = "AT50"
SIMULATION_RUN = "run_18"
TRAJECTORY_GROWTH_MODE = "absolute"
PLOT_END_DAY = 40.0
BASE_KERNEL_RADIUS_UM = MASTER_KERNEL_RADIUS_UM
BASE_MIN_NEIGHBORS = hdf5_sim.prolif_min_neighbors
BASE_MIN_RESISTANT_NUMERATOR = MIN_RESISTANT_NUMERATOR
VARIANT_KERNEL_RADIUS_UM = 120.0
VARIANT_MIN_NEIGHBORS = hdf5_sim.prolif_min_neighbors
VARIANT_MIN_RESISTANT_NUMERATOR = MIN_RESISTANT_NUMERATOR
if VARIANT_MIN_NEIGHBORS != hdf5_exp.prolif_min_neighbors:
    raise ValueError("Experimental and simulation neighbour thresholds differ")
if not np.isclose(
    VARIANT_MIN_RESISTANT_NUMERATOR,
    hdf5_exp.MIN_RES_DENSITY * hdf5_exp.prolif_bin_area,
):
    raise ValueError("Experimental and simulation MIN_RES_DENSITY settings differ")

CACHE_DIR = figure_data_dir(
    "Figure_3", "HDF5_analysis", "at50_absolute_trajectories", create=True,
)
EXPERIMENT_CACHE = CACHE_DIR / (
    "D5_AT50_experiment_absolute_trajectory_mean_sd_total_t0_160um_minres2p25_minneighbors14.tsv"
)
SIMULATION_CACHE = CACHE_DIR / (
    "AT50_run18_simulation_absolute_trajectory_mean_sd_total_t0_160um_minres2p25_minneighbors14.tsv"
)
VARIANT_EXPERIMENT_CACHE = CACHE_DIR / (
    "D5_AT50_experiment_absolute_trajectory_mean_sd_total_t0_120um_minres2p25_minneighbors14.tsv"
)
VARIANT_SIMULATION_CACHE = CACHE_DIR / (
    "AT50_run18_simulation_absolute_trajectory_mean_sd_total_t0_120um_minres2p25_minneighbors14.tsv"
)

OUTPUT_DIR = figure_plot_dir("Figure_3", "density_trajectories", create=True)
EXPERIMENT_OUTPUT = (
    OUTPUT_DIR
    / "D5_AT50_experiment_absolute_resistant_density_change_trajectories_mean_sd_"
      "HDF5_160um_minres2p25_minneighbors14.pdf"
)
SIMULATION_OUTPUT = (
    OUTPUT_DIR
    / "D5_AT50_simulation_run18_absolute_resistant_density_change_trajectories_mean_sd_"
      "HDF5_160um_minres2p25_minneighbors14.pdf"
)
VARIANT_EXPERIMENT_OUTPUT = (
    OUTPUT_DIR
    / "D5_AT50_experiment_absolute_resistant_density_change_trajectories_mean_sd_"
      "HDF5_120um_minres2p25_minneighbors14.pdf"
)
VARIANT_SIMULATION_OUTPUT = (
    OUTPUT_DIR
    / "D5_AT50_simulation_run18_absolute_resistant_density_change_trajectories_mean_sd_"
      "HDF5_120um_minres2p25_minneighbors14.pdf"
)
ABSOLUTE_CHANGE_LABEL = (
    "⟨" r"$\Delta\rho_{\mathrm{res}}$" "⟩" r"$_A/\Delta t$"
    "\n"
    r"$[\mathrm{cells}\,\mathrm{µm}^{-2}\,\mathrm{day}^{-1}]$"
)
TOTAL_DENSITY_LABEL = (
    "⟨" r"$\rho_{\mathrm{tot}}$" "⟩" r"$_A$"
    r" $[\mathrm{cells}\,\mathrm{µm}^{-2}]$"
)
ABSOLUTE_CHANGE_COLUMN = "Mean_Resistant_Density_Absolute_Change"
ABSOLUTE_CHANGE_STD_COLUMN = "Std_Resistant_Density_Absolute_Change"
RATE_COLOR = "#b40426"
TOTAL_COLOR = "black"
RUNNING_AVERAGE_WINDOW = 4
SHOW_FIGURE = False

ft.apply_nature_style(
    text_pt=7.0,
    tick_pt=6.0,
    legend_pt=6.0,
    overrides={
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    },
)


def _nice_ticks(limits: tuple[float, float], target_count: int = 4) -> np.ndarray:
    """Return readable 1/2/5-step ticks spanning the requested range."""
    lower, upper = sorted(map(float, limits))
    if not np.isfinite((lower, upper)).all() or lower == upper:
        return np.array([lower])
    candidates = np.asarray(
        [base * 10.0 ** exponent for exponent in range(-6, 5) for base in (1, 2, 5, 10)],
        dtype=float,
    )
    counts = np.floor(upper / candidates) - np.ceil(lower / candidates) + 1
    step = candidates[np.argmin(np.abs(counts - target_count))]
    first = np.ceil(lower / step) * step
    last = np.floor(upper / step) * step
    return np.arange(first, last + 0.5 * step, step)


def appropriate_change_limits(data: pd.DataFrame, column: str) -> tuple[float, float]:
    """Choose robust rate limits from the displayed interval summaries."""
    values = data[column].to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (-0.5, 0.5)
    lower, upper = np.quantile(values, (0.01, 0.99))
    lower, upper = min(0.0, float(lower)), max(0.0, float(upper))
    span = max(upper - lower, 0.1)
    padding = 0.12 * span
    raw_step = (span + 2.0 * padding) / 4.0
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    step = min((1.0, 2.0, 5.0, 10.0), key=lambda value: abs(value * magnitude - raw_step)) * magnitude
    return (
        float(np.floor((lower - padding) / step) * step),
        float(np.ceil((upper + padding) / step) * step),
    )


def _running_mean_and_propagated_std(
    values: pd.Series, standard_deviations: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    rolling = values.astype(float).rolling(
        RUNNING_AVERAGE_WINDOW, center=True, min_periods=1,
    )
    count = rolling.count().to_numpy()
    mean = rolling.mean().to_numpy()
    propagated_std = np.sqrt(
        (standard_deviations.astype(float) ** 2).rolling(
            RUNNING_AVERAGE_WINDOW, center=True, min_periods=1,
        ).sum().to_numpy()
    ) / count
    return mean, propagated_std


def plot_trajectory(
    data: pd.DataFrame,
    change_limits: tuple[float, float],
    output: Path,
    *,
    is_simulation: bool,
    change_column: str,
    change_std_column: str,
    change_label: str,
    total_density_upper_limit: float,
) -> None:
    """Plot one HDF5-derived AT50 trajectory without the legacy renderer."""
    fig, ax = ft.new_subplots(
        1, 1, ax_width_mm=0.85 * (3 + 0.06 * 2) * 25.4, ax_height_mm=14.0,
        # Keep the 38 mm canvas and 14 mm axes height unchanged while moving
        # the plotting area 1.5 mm upward.
        margins=(21.5, 15.0, 8.5, 15.5), dpi=300, apply_style=False,
    )
    rate_lower, rate_upper = map(float, change_limits)
    total_lower, total_upper = 0.0, float(total_density_upper_limit)

    def rate_to_total(values):
        return total_lower + (np.asarray(values) - rate_lower) * (
            total_upper - total_lower
        ) / (rate_upper - rate_lower)

    def total_to_rate(values):
        return rate_lower + (np.asarray(values) - total_lower) * (
            rate_upper - rate_lower
        ) / (total_upper - total_lower)

    summary = data.groupby("day", as_index=False).agg(
        total=("Mean_Total_Density", "mean"),
        total_std=("Std_Total_Density", "mean"),
        rate=(change_column, "mean"),
        rate_std=(change_std_column, "mean"),
    )
    total_mean, total_std = _running_mean_and_propagated_std(
        summary["total"], summary["total_std"],
    )
    rate_mean, rate_std = _running_mean_and_propagated_std(
        summary["rate"], summary["rate_std"],
    )
    line_style = "--" if is_simulation else "-"
    ax.fill_between(
        summary["day"], total_mean - total_std, total_mean + total_std,
        color=TOTAL_COLOR, alpha=0.28, linewidth=0, zorder=0.5,
    )
    ax.fill_between(
        summary["day"], rate_to_total(rate_mean - rate_std),
        rate_to_total(rate_mean + rate_std), color=RATE_COLOR, alpha=0.28,
        linewidth=0, zorder=0.5,
    )
    ax.plot(summary["day"], total_mean, color=TOTAL_COLOR, linestyle=line_style, linewidth=1.0)
    ax.plot(
        summary["day"], rate_to_total(rate_mean), color=RATE_COLOR,
        linestyle=line_style, linewidth=1.0,
    )
    ax.set(xlim=(0, 40), ylim=(total_lower, total_upper), xlabel="Time [d]")
    ax.set_xticks((5, 15, 25, 35), minor=True)
    ax.set_yticks(np.arange(0.0, 2.01, 0.5))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: str(int(round(value))) if np.isclose(value, round(value)) else "",
    ))
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(TOTAL_COLOR)
    ax.tick_params(axis="x", which="minor", length=2.5, width=0.8)

    rate_axis = ax.secondary_yaxis("left", functions=(total_to_rate, rate_to_total))
    rate_axis.spines["left"].set_color(RATE_COLOR)
    rate_axis.tick_params(axis="y", colors=RATE_COLOR, direction="out", length=2.5, width=0.8, pad=1)
    rate_axis.set_ylabel(change_label, color=RATE_COLOR, labelpad=1)
    rate_axis.yaxis.label.set_fontfamily("Segoe UI Symbol")
    rate_axis.set_yticks(_nice_ticks(change_limits))
    rate_axis.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: str(int(round(value * 10))) if np.isclose(value * 10, round(value * 10)) else "",
    ))

    ax.text(1.015, 1.025, r"$\times 10^{-3}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6, color=TOTAL_COLOR, clip_on=False)
    ax.text(0.0, 1.025, r"$\times 10^{-4}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6, color=RATE_COLOR, clip_on=False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_bbox = ax.get_window_extent(renderer)
    red_label_bbox = rate_axis.yaxis.label.get_window_extent(renderer)
    total_label = fig.text(
        0.0, 0.0, TOTAL_DENSITY_LABEL, color=TOTAL_COLOR, rotation=270,
        ha="center", va="center", fontfamily="Segoe UI Symbol",
    )
    fig.canvas.draw()
    total_label_bbox = total_label.get_window_extent(fig.canvas.get_renderer())
    black_ticks = [label.get_window_extent(renderer) for label in ax.get_yticklabels() if label.get_text()]
    red_ticks = [label.get_window_extent(renderer) for label in rate_axis.get_yticklabels() if label.get_text()]
    black_tick_right = max((box.x1 for box in black_ticks), default=plot_bbox.x1)
    red_tick_left = min((box.x0 for box in red_ticks), default=plot_bbox.x0)
    label_gap = max(red_tick_left - red_label_bbox.x1, 0.0)
    total_center_x = black_tick_right + label_gap + total_label_bbox.width / 2
    total_x = fig.transFigure.inverted().transform((total_center_x, plot_bbox.y0))[0]
    total_y = fig.transFigure.inverted().transform((plot_bbox.x0, 0.5 * (plot_bbox.y0 + plot_bbox.y1)))[1]
    total_label.set_position((total_x, total_y))

    ft.savefig(fig, output)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    print(f"[done] Saved: {output}")


def _through_end_day(items, day_of):
    """Items up to PLOT_END_DAY, plus the next one.

    Each interval is plotted at its start day, and a rate at day X needs a
    later endpoint to exist.  Keeping one item beyond the cut-off therefore
    lets the curve reach PLOT_END_DAY instead of stopping at the previous
    acquisition.
    """
    if PLOT_END_DAY is None:
        return list(items)
    kept = [item for item in items if day_of(item) <= PLOT_END_DAY]
    later = [item for item in items if day_of(item) > PLOT_END_DAY]
    if later:
        kept.append(later[0])
    return kept


def _summary_row(
    *,
    start_day: float,
    end_day: float,
    total_t0: np.ndarray,
    resistant_t0: np.ndarray,
    total_t1: np.ndarray,
    resistant_t1: np.ndarray,
    valid_t0: np.ndarray,
    valid_t1: np.ndarray,
    support_threshold: float,
    density_scale: float,
    source: str,
) -> dict[str, object] | None:
    """Return the spatial mean and SD using HDF5's absolute-change support."""
    delta_days = float(end_day - start_day)
    valid = resistant_support(
        total_t0, resistant_t0, total_t1, resistant_t1,
        valid_t0, valid_t1, threshold=support_threshold,
    )
    if delta_days <= 0 or not np.any(valid):
        return None

    total = total_t0[valid] * density_scale
    absolute_change = (
        (resistant_t1[valid] - resistant_t0[valid])
        / delta_days
        * density_scale
    )
    ddof = 1 if len(total) > 1 else 0
    return {
        "plate": source,
        "sample": EXPERIMENT_RUN if source == "experiment" else SIMULATION_RUN,
        "strategy": "AT50",
        # Both the trajectory position and displayed total density use t0.
        # The rate itself still uses the full t0-to-t1 interval.
        "day": float(start_day),
        "Mean_Total_Density": float(np.mean(total)),
        "Std_Total_Density": float(np.std(total, ddof=ddof)),
        "Mean_Resistant_Density_Absolute_Change": float(np.mean(absolute_change)),
        "Std_Resistant_Density_Absolute_Change": float(
            np.std(absolute_change, ddof=ddof)
        ),
        "n_pixels": int(np.count_nonzero(valid)),
        "start_day": float(start_day),
        "end_day": float(end_day),
    }


def _experimental_fixed_grid(handle, *, kernel_radius_um):
    return hdf5_exp._fixed_well_grid(handle, kernel_radius_um=kernel_radius_um)


def _experimental_local_densities(handle, day_key, grid_points, grid_shape, *, kernel_radius_um, min_neighbors):
    return hdf5_exp.local_densities(handle, day_key, grid_points, grid_shape,
        kernel_radius_um=kernel_radius_um, min_neighbors=min_neighbors)


def _simulation_local_densities(run_dir, timestep, *, kernel_radius_um, min_neighbors):
    return hdf5_sim.local_densities(run_dir, timestep,
        kernel_radius_um=kernel_radius_um, min_neighbors=min_neighbors)


def build_experiment_trajectory(
    *, kernel_radius_um=BASE_KERNEL_RADIUS_UM,
    min_neighbors=BASE_MIN_NEIGHBORS,
    min_resistant_numerator=BASE_MIN_RESISTANT_NUMERATOR,
) -> pd.DataFrame:
    """Recompute D5 from the raw experimental point-cloud HDF5."""
    runs = hdf5_exp.discover_runs()
    try:
        h5_path = Path(runs["AT50"][EXPERIMENT_RUN])
    except KeyError as error:
        raise RuntimeError(f"HDF5 input does not contain {EXPERIMENT_RUN}") from error

    _, rule = hdf5_exp._sample_and_rule(h5_path)
    cutoff = None if rule is ... else float(rule)
    density_scale = 1000.0 / hdf5_exp.px_to_micron ** 2
    rows: list[dict[str, object]] = []
    with h5py.File(h5_path, "r") as handle:
        grid_points, grid_shape = _experimental_fixed_grid(
            handle, kernel_radius_um=kernel_radius_um,
        )
        days = hdf5_exp._sorted_days(handle)
        if cutoff is not None:
            days = [(day, key) for day, key in days if day <= cutoff]
        days = _through_end_day(days, lambda item: item[0])

        previous = None
        previous_valid = None
        previous_day = None
        for index, (day, day_key) in enumerate(days, start=1):
            total, _, resistant, valid = _experimental_local_densities(
                handle, day_key, grid_points, grid_shape,
                kernel_radius_um=kernel_radius_um,
                min_neighbors=min_neighbors,
            )
            print(f"[experiment HDF5] {index}/{len(days)} day {day}", flush=True)
            if previous is not None:
                row = _summary_row(
                    start_day=float(previous_day),
                    end_day=float(day),
                    total_t0=previous[0],
                    resistant_t0=previous[1],
                    total_t1=total,
                    resistant_t1=resistant,
                    valid_t0=previous_valid,
                    valid_t1=valid,
                    support_threshold=(
                        min_resistant_numerator
                        / (np.pi * (kernel_radius_um / hdf5_exp.px_to_micron) ** 2)
                    ),
                    density_scale=density_scale,
                    source="experiment",
                )
                if row is not None:
                    rows.append(row)
            previous = (total, resistant)
            previous_valid = valid
            previous_day = day
    if not rows:
        raise RuntimeError("No valid experimental HDF5 intervals were found.")
    return pd.DataFrame(rows)


def build_simulation_trajectory(
    *, kernel_radius_um=BASE_KERNEL_RADIUS_UM,
    min_neighbors=BASE_MIN_NEIGHBORS,
    min_resistant_numerator=BASE_MIN_RESISTANT_NUMERATOR,
) -> pd.DataFrame:
    """Recompute run 18 from its native daily simulation HDF5 snapshots."""
    if not is_non_artificial_run(SIMULATION_RUN):
        raise ValueError("The master curve excludes simulation runs 0 through 9")
    run_dir = Path(hdf5_sim.protocol_outputs_dir(SIMULATION_PROTOCOL, SIMULATION_RUN))
    timesteps = hdf5_sim._selected_timesteps(str(run_dir))
    day_of = lambda timestep: (
        timestep / hdf5_sim.time_dump + hdf5_sim.escape_core.sim.DEFAULT_DAY_OFFSET
    )
    selected = _through_end_day(timesteps, day_of)
    density_scale = 1000.0 / (
        hdf5_sim.exp_scaling * hdf5_sim.px_to_micron
    ) ** 2
    rows: list[dict[str, object]] = []
    previous = None
    previous_valid = None
    previous_day = None
    for index, timestep in enumerate(selected, start=1):
        total, _, resistant, valid = _simulation_local_densities(
            str(run_dir), timestep,
            kernel_radius_um=kernel_radius_um,
            min_neighbors=min_neighbors,
        )
        day = timestep / hdf5_sim.time_dump + hdf5_sim.escape_core.sim.DEFAULT_DAY_OFFSET
        print(f"[simulation HDF5] {index}/{len(selected)} day {day:g}", flush=True)
        if previous is not None:
            row = _summary_row(
                start_day=float(previous_day),
                end_day=float(day),
                total_t0=previous[0],
                resistant_t0=previous[1],
                total_t1=total,
                resistant_t1=resistant,
                valid_t0=previous_valid,
                valid_t1=valid,
                support_threshold=(
                    min_resistant_numerator / (
                        np.pi * (
                            kernel_radius_um
                            / (hdf5_sim.exp_scaling * hdf5_sim.px_to_micron)
                        ) ** 2
                    )
                ),
                density_scale=density_scale,
                source="simulation",
            )
            if row is not None:
                rows.append(row)
        previous = (total, resistant)
        previous_valid = valid
        previous_day = day
    if not rows:
        raise RuntimeError("No valid simulation HDF5 intervals were found.")
    return pd.DataFrame(rows)


def _load_or_build(
    path: Path, builder, *, force: bool, kernel_radius_um: float,
) -> pd.DataFrame:
    expected = {
        **kernel_settings(kernel_radius_um), "cache_version": 4,
        "growth_mode": TRAJECTORY_GROWTH_MODE, "density_coordinate": MASTER_DENSITY_COORDINATE,
        "period": MASTER_PERIOD, "summary_statistic": "mean", "spread_statistic": "sample_standard_deviation",
        "temporal_smoothing_window": RUNNING_AVERAGE_WINDOW,
        "source_experiment": EXPERIMENT_RUN, "source_simulation": SIMULATION_RUN,
        "plot_end_day": "all" if PLOT_END_DAY is None else str(PLOT_END_DAY),
    }
    if path.is_file() and not force:
        cached = pd.read_csv(path, sep="\t")
        if not cached.empty and all(
            name in cached and (np.all(cached[name].astype(str) == value) if isinstance(value, str)
                                else np.all(np.isclose(cached[name], value, rtol=1e-12, atol=1e-12)))
            for name, value in expected.items()
        ):
            print(f"[cache] Reusing {path}")
            return cached
        print(f"[cache] Rebuilding {path}: missing or different density settings")
    result = builder()
    for name, value in expected.items():
        result[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, sep="\t", index=False)
    print(f"[cache] Saved {path}")
    return result


def main() -> None:
    force = "--recompute" in sys.argv
    configurations = (
        (
            EXPERIMENT_CACHE, SIMULATION_CACHE,
            EXPERIMENT_OUTPUT, SIMULATION_OUTPUT,
            BASE_KERNEL_RADIUS_UM,
            BASE_MIN_NEIGHBORS, BASE_MIN_RESISTANT_NUMERATOR,
        ),
        (
            VARIANT_EXPERIMENT_CACHE, VARIANT_SIMULATION_CACHE,
            VARIANT_EXPERIMENT_OUTPUT, VARIANT_SIMULATION_OUTPUT,
            VARIANT_KERNEL_RADIUS_UM,
            VARIANT_MIN_NEIGHBORS, VARIANT_MIN_RESISTANT_NUMERATOR,
        ),
    )
    for (
        experiment_cache, simulation_cache,
        experiment_output, simulation_output,
        kernel_radius_um, min_neighbors, min_resistant_numerator,
    ) in configurations:
        experimental = _load_or_build(
            experiment_cache,
            lambda: build_experiment_trajectory(
                kernel_radius_um=kernel_radius_um,
                min_neighbors=min_neighbors,
                min_resistant_numerator=min_resistant_numerator,
            ),
            force=force,
            kernel_radius_um=kernel_radius_um,
        )
        simulation = _load_or_build(
            simulation_cache,
            lambda: build_simulation_trajectory(
                kernel_radius_um=kernel_radius_um,
                min_neighbors=min_neighbors,
                min_resistant_numerator=min_resistant_numerator,
            ),
            force=force,
            kernel_radius_um=kernel_radius_um,
        )
        combined = pd.concat([experimental, simulation], ignore_index=True)
        limits = appropriate_change_limits(combined, ABSOLUTE_CHANGE_COLUMN)
        for data, output, is_simulation in (
            (experimental, experiment_output, False),
            (simulation, simulation_output, True),
        ):
            plot_trajectory(
                data,
                limits,
                output,
                is_simulation=is_simulation,
                change_column=ABSOLUTE_CHANGE_COLUMN,
                change_std_column=ABSOLUTE_CHANGE_STD_COLUMN,
                change_label=ABSOLUTE_CHANGE_LABEL,
                total_density_upper_limit=2.3,
            )

if __name__ == "__main__":
    main()
