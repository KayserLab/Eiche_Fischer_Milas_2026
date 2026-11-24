#!/usr/bin/env python3
"""Plot experimental and native-timepoint simulated AT50 density trajectories.

The panels use the Plate 4 / D5_AT50-D5 experiment and its associated
simulation run 18. Each uses its own observation time points, the 160 um kernel, 16x-coarsened density maps,
interval summaries, and visual style of the retained Figure 3 trajectory plot.
"""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import figtools as ft
from shared.data_paths import figure_plot_dir
from figures.Figure_3.Figure_3_Legacy import (
    Density_Correlation_Examples_coarsegrained_from_Figure_3 as legacy,
)

# Keep the trajectory script independent of the much larger density-map figure
# module. This is the same 1.05-cell, 160-um Epanechnikov threshold used there.
RIM_DENSITY_THRESHOLD = 1.05 * 2.0 / (pi * 160.0 ** 2)


PLATE = "Plate_4"
WELL = "D5_AT50-D5"
SIMULATION_SOURCE = "AT50"
SIMULATION_RUN = "run_18"
# Matplotlib's ``coolwarm`` maximum, matching the positive end of the
# resistant-density-rate colourbar in plot_at50_density_maps.py.
RATE_COLOR = "#b40426"
# Full width of the three 0.85-inch density-map panels, including the two map
# gaps.  The trajectory axes use this physical width when figures are stacked.
DENSITY_MAP_THREE_PANEL_WIDTH_MM = 0.85 * (3 + 0.06 * 2) * 25.4
EXPERIMENT_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_experiment_density_trajectories_160um_16x.pdf"
)
SIMULATION_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_simulation_run18_density_trajectories_160um_16x.pdf"
)
MATCHED_EXPERIMENT_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_experiment_density_trajectories_160um_16x_matched_intervals.pdf"
)
MATCHED_SIMULATION_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_simulation_run18_density_trajectories_160um_16x_matched_experimental_intervals.pdf"
)
ABSOLUTE_EXPERIMENT_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_experiment_absolute_resistant_density_change_trajectories_160um_16x.pdf"
)
ABSOLUTE_SIMULATION_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_simulation_run18_absolute_resistant_density_change_trajectories_160um_16x.pdf"
)
ABSOLUTE_MATCHED_EXPERIMENT_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_experiment_absolute_resistant_density_change_trajectories_160um_16x_matched_intervals.pdf"
)
ABSOLUTE_MATCHED_SIMULATION_OUTPUT = (
    figure_plot_dir("Figure_3", create=True)
    / "D5_AT50_simulation_run18_absolute_resistant_density_change_trajectories_160um_16x_matched_experimental_intervals.pdf"
)
RIM_THRESHOLD_TAG = "rim1cell"
ONE_CELL_OUTPUT_DIR = figure_plot_dir("Figure_3", "trajectories_all", create=True)
THRESHOLDED_EXPERIMENT_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_experiment_density_trajectories_160um_16x_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_SIMULATION_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_simulation_run18_density_trajectories_160um_16x_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_MATCHED_EXPERIMENT_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_experiment_density_trajectories_160um_16x_matched_intervals_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_MATCHED_SIMULATION_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_simulation_run18_density_trajectories_160um_16x_matched_experimental_intervals_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_ABSOLUTE_EXPERIMENT_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_experiment_absolute_resistant_density_change_trajectories_160um_16x_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_ABSOLUTE_SIMULATION_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_simulation_run18_absolute_resistant_density_change_trajectories_160um_16x_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_ABSOLUTE_MATCHED_EXPERIMENT_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_experiment_absolute_resistant_density_change_trajectories_160um_16x_matched_intervals_{RIM_THRESHOLD_TAG}.pdf"
)
THRESHOLDED_ABSOLUTE_MATCHED_SIMULATION_OUTPUT = (
    ONE_CELL_OUTPUT_DIR
    / f"D5_AT50_simulation_run18_absolute_resistant_density_change_trajectories_160um_16x_matched_experimental_intervals_{RIM_THRESHOLD_TAG}.pdf"
)

RELATIVE_CHANGE_COLUMN = "Mean_Resistant_Density_Change"
RELATIVE_CHANGE_STD_COLUMN = "Std_Resistant_Density_Change"
ABSOLUTE_CHANGE_COLUMN = "Mean_Resistant_Density_Absolute_Change"
ABSOLUTE_CHANGE_STD_COLUMN = "Std_Resistant_Density_Absolute_Change"
RELATIVE_CHANGE_LABEL = (
    r"$\Delta\langle\tilde{\rho}_\mathrm{res}\rangle_A/\Delta t$ [day$^{-1}$]"
)
ABSOLUTE_CHANGE_LABEL = (
    r"$\Delta\langle\rho_\mathrm{res}\rangle_A/\Delta t$"
    "\n"
    r"[cells µm$^{-2}$ day$^{-1}$]"
)
# Horizontal position of the absolute-change exponent, in axes coordinates.
# The rate axis and its tick labels sit outside the left spine, so a negative
# offset is needed to place the multiplier above them instead of over the
# plotting area.
ABSOLUTE_EXPONENT_X = 0.0
# The absolute rate ticks land on 0.05 steps, which one decimal cannot
# resolve.  Labelling them ten-fold against a 1e-4 multiplier instead of
# 1e-3 puts them on 0.5 steps, so one decimal is exact and nothing is lost.
ABSOLUTE_RATE_LABEL_SCALE = 10.0
ABSOLUTE_RATE_EXPONENT = r"$\times 10^{-4}$"


def load_at50_trajectory(*, rim_threshold_only: bool = False):
    """Return interval-wise whole-well mean summaries for the selected AT50 well."""
    original_examples = legacy.EXAMPLES_2H
    try:
        # Limit the legacy loader to D5 so this focused plot does not read the
        # NT and CT examples that appeared alongside it in the old figure.
        legacy.EXAMPLES_2H = {
            "AT50": {"plate": PLATE, "sample": WELL},
        }
        return legacy.load_wellwide_data(
            legacy.INPUT_RUN,
            spatial_summary=legacy.finite_mean,
            growth_area_weighted=True,
            resistant_threshold=(RIM_DENSITY_THRESHOLD if rim_threshold_only else None),
            threshold_first_endpoint_only=rim_threshold_only,
        )
    finally:
        legacy.EXAMPLES_2H = original_examples


def load_simulation_trajectory(*, rim_threshold_only: bool = False):
    """Return interval-wise whole-domain mean summaries for matched run 18."""
    original_examples = legacy.SIMULATION_EXAMPLES
    try:
        legacy.SIMULATION_EXAMPLES = {
            "AT50": {"source": SIMULATION_SOURCE, "run": SIMULATION_RUN},
        }
        return legacy.load_simulation_wellwide_data(
            legacy.SIMULATION_ROOT,
            spatial_summary=legacy.finite_mean,
            growth_area_weighted=True,
            resistant_threshold=(RIM_DENSITY_THRESHOLD if rim_threshold_only else None),
            threshold_first_endpoint_only=rim_threshold_only,
        )
    finally:
        legacy.SIMULATION_EXAMPLES = original_examples


def load_simulation_trajectory_matching_experimental_intervals(
    *, rim_threshold_only: bool = False,
):
    """Summarise simulation over the same endpoints as experimental maps."""
    maps_dir = legacy.density_analysis.DEFAULT_INVITRO_RUN / PLATE / WELL / "density_maps"
    prefix = f"{WELL}_"
    suffix = "_total_density_epanechnikov.tif"
    experimental_days = sorted(
        int(candidate)
        for path in maps_dir.glob(f"{WELL}_*_total_density_epanechnikov.tif")
        if (candidate := path.name.removeprefix(prefix).removesuffix(suffix)).isdigit()
    )
    intervals = [
        (start_day, end_day)
        for start_day, end_day in zip(experimental_days, experimental_days[1:])
        if end_day <= legacy.X_LIM[1]
    ]
    if not intervals:
        raise RuntimeError("No experimental intervals are available within the trajectory time range.")

    sim_args = legacy.density_analysis.sweep.simulation_args(
        str(legacy.SIMULATION_ROOT), SIMULATION_SOURCE, SIMULATION_RUN,
    )
    plans = legacy.density_analysis.sim.discover_run_plans(sim_args)
    if len(plans) != 1:
        raise RuntimeError(
            f"Expected one simulation plan for {SIMULATION_SOURCE}/{SIMULATION_RUN}, found {len(plans)}"
        )
    files = [item for item in plans[0].files if float(item.day) <= legacy.X_LIM[1]]
    loaded = [
        legacy.density_analysis.sim.load_filtered_cells(item.path, sim_args)
        for item in files
    ]
    canvas = legacy.density_analysis.sweep.canvas_from_loaded(
        loaded, float(legacy.DENSITY_KERNEL_UM),
    )

    def density_at_or_after(day: int):
        try:
            index = next(index for index, item in enumerate(files) if float(item.day) >= day)
        except StopIteration as error:
            raise RuntimeError(f"Simulation has no density data at or after day {day}") from error
        positions_um, type_values = loaded[index]
        resistant, total = legacy.density_analysis.simulation_density(
            positions_um, type_values, canvas,
        )
        return float(files[index].day), resistant, total

    rows = []
    for start_day, end_day in intervals:
        actual_start, resistant_start, total_start = density_at_or_after(start_day)
        actual_end, resistant_end, total_end = density_at_or_after(end_day)
        delta_days = actual_end - actual_start
        if delta_days <= 0:
            continue
        average_total = 0.5 * (total_start + total_end)
        valid = (
            legacy.np.isfinite(average_total)
            & legacy.np.isfinite(resistant_start)
            & legacy.np.isfinite(resistant_end)
        )
        if rim_threshold_only:
            valid &= resistant_start > RIM_DENSITY_THRESHOLD
        else:
            valid &= resistant_start >= legacy.RESISTANT_ENDPOINT_THRESHOLD
            valid &= resistant_end >= legacy.RESISTANT_ENDPOINT_THRESHOLD
        if not legacy.np.any(valid):
            continue
        relative_growth = (
            (resistant_end[valid] - resistant_start[valid])
            / resistant_start[valid]
            / delta_days
        )
        absolute_growth = (
            (resistant_end[valid] - resistant_start[valid])
            / delta_days
            * legacy.DENSITY_DISPLAY_SCALE
        )
        rows.append({
            "plate": SIMULATION_SOURCE,
            "sample": SIMULATION_RUN,
            "strategy": "AT50",
            "day": 0.5 * (actual_start + actual_end),
            "Mean_Total_Density": legacy.finite_mean(
                average_total[valid] * legacy.DENSITY_DISPLAY_SCALE
            ),
            "Std_Total_Density": float(legacy.np.std(
                average_total[valid] * legacy.DENSITY_DISPLAY_SCALE, ddof=1,
            )),
            "Mean_Resistant_Density_Change": legacy.density_weighted_mean(
                relative_growth, resistant_start[valid],
            ),
            "Std_Resistant_Density_Change": legacy.density_weighted_std(
                relative_growth, resistant_start[valid],
            ),
            "Mean_Resistant_Density_Absolute_Change": legacy.finite_mean(
                absolute_growth,
            ),
            "Std_Resistant_Density_Absolute_Change": float(legacy.np.std(
                absolute_growth, ddof=1,
            )),
            "n_pixels": int(legacy.np.count_nonzero(valid)),
        })
    if not rows:
        raise RuntimeError("No valid matched-interval simulation summaries were calculated.")
    return legacy.pd.DataFrame(rows)


def plot_trajectory(
    data, change_limits, output, *, is_simulation: bool, show_markers: bool = False,
    change_column: str = RELATIVE_CHANGE_COLUMN,
    change_std_column: str = RELATIVE_CHANGE_STD_COLUMN,
    change_label: str = RELATIVE_CHANGE_LABEL,
    integer_total_tick_labels: bool = False,
    integer_rate_tick_labels: bool = False,
    total_density_upper_limit: float = 2.1,
) -> None:
    # The absolute label is two lines, so it needs more room than the
    # single-line relative one.
    left_margin_mm = 21.5 if change_column == ABSOLUTE_CHANGE_COLUMN else 12.0
    fig, ax = ft.new_subplots(
        1,
        1,
        ax_width_mm=DENSITY_MAP_THREE_PANEL_WIDTH_MM,
        ax_height_mm=14.0,
        # Shift the axes down while leaving room for the density multiplier.
        margins=(left_margin_mm, 15.0, 7.0, 17.0),
        dpi=legacy.DPI,
        apply_style=False,
    )
    original_rate_color = legacy.COLOR_CHG
    # The density-map tables contain both relative and absolute change
    # columns. Assign into the legacy names instead of renaming, which would
    # otherwise create duplicate column labels and let the legacy renderer use
    # the original relative values for an absolute plot.
    plot_data = data.copy()
    plot_data[RELATIVE_CHANGE_COLUMN] = data[change_column].to_numpy(float)
    plot_data[RELATIVE_CHANGE_STD_COLUMN] = data[change_std_column].to_numpy(float)
    try:
        legacy.COLOR_CHG = RATE_COLOR
        legacy.plot_one_panel(
            ax, plot_data, "", change_limits, treatment_days=(),
        )
    finally:
        legacy.COLOR_CHG = original_rate_color
    if not show_markers:
        for collection in ax.collections:
            collection.remove()
    if is_simulation:
        for line in ax.lines:
            line.set_linestyle("--")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, total_density_upper_limit)
    ax.set_xticks((5, 15, 25, 35), minor=True)
    ax.tick_params(axis="x", which="minor", length=2.5, width=0.8)
    ax.set_xlabel("Time [days]", labelpad=2)
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(legacy.COLOR_TOTAL)
    ax.set_ylabel("")
    # Replace the legacy right-hand secondary axis with the resistant-rate
    # axis on the left, leaving total density on the right.
    ax.child_axes[-1].remove()
    total_lo, total_hi = map(float, legacy.Y_LIM_TOTAL)
    rate_lo, rate_hi = map(float, change_limits)

    def total_to_rate(values):
        values = legacy.np.asarray(values)
        return rate_lo + (values - total_lo) * (rate_hi - rate_lo) / (total_hi - total_lo)

    def rate_to_total(values):
        values = legacy.np.asarray(values)
        return total_lo + (values - rate_lo) * (total_hi - total_lo) / (rate_hi - rate_lo)

    rate_axis = ax.secondary_yaxis(
        "left",
        functions=(total_to_rate, rate_to_total),
    )
    uncertainty = plot_data.groupby("day", as_index=False).agg(
        total=("Mean_Total_Density", "mean"),
        total_std=("Std_Total_Density", "mean"),
        growth=("Mean_Resistant_Density_Change", "mean"),
        growth_std=("Std_Resistant_Density_Change", "mean"),
    )

    def running_mean_and_propagated_std(values, standard_deviations):
        """Match the centred running mean and propagate its spatial spread."""
        values = legacy.pd.Series(values, dtype=float)
        standard_deviations = legacy.pd.Series(standard_deviations, dtype=float)
        rolling = values.rolling(
            legacy.RUNNING_AVG_WINDOW, center=True, min_periods=1,
        )
        count = rolling.count().to_numpy()
        mean = rolling.mean().to_numpy()
        # For an average of N independent interval summaries, variances add
        # before division by N squared: sigma_mean = sqrt(sum(sigma_i^2)) / N.
        propagated_std = (
            legacy.np.sqrt(
                (standard_deviations ** 2).rolling(
                    legacy.RUNNING_AVG_WINDOW, center=True, min_periods=1,
                ).sum().to_numpy()
            )
            / count
        )
        return mean, propagated_std

    total_mean, total_std = running_mean_and_propagated_std(
        uncertainty["total"], uncertainty["total_std"],
    )
    growth_mean, growth_std = running_mean_and_propagated_std(
        uncertainty["growth"], uncertainty["growth_std"],
    )
    ax.fill_between(
        uncertainty["day"],
        total_mean - total_std,
        total_mean + total_std,
        color=legacy.COLOR_TOTAL, alpha=0.28, linewidth=0, zorder=0.5,
    )
    ax.fill_between(
        uncertainty["day"],
        rate_to_total(growth_mean - growth_std),
        rate_to_total(growth_mean + growth_std),
        color=RATE_COLOR, alpha=0.28, linewidth=0, zorder=1.0,
    )
    rate_axis.spines["left"].set_color(RATE_COLOR)
    rate_axis.tick_params(
        axis="y", colors=RATE_COLOR, direction="out", length=2.5,
        width=0.8, pad=1,
    )
    rate_axis.set_ylabel(
        change_label,
        color=RATE_COLOR,
        labelpad=1,
    )
    decimal_ticks = FormatStrFormatter("%.1f")
    if integer_total_tick_labels:
        ax.set_yticks(legacy.np.arange(0.0, 2.01, 0.5))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _position: (
                str(int(round(value)))
                if legacy.np.isclose(value, round(value))
                else ""
            )
        ))
    else:
        ax.set_yticks((0.0, 1.0, 2.0))
        ax.yaxis.set_major_formatter(decimal_ticks)
    ax.text(
        1.015, 1.025, r"$\times 10^{-3}$", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=6, color=legacy.COLOR_TOTAL,
        clip_on=False,
    )
    if change_column == ABSOLUTE_CHANGE_COLUMN:
        ax.text(
            ABSOLUTE_EXPONENT_X, 1.025, ABSOLUTE_RATE_EXPONENT,
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6,
            color=RATE_COLOR, clip_on=False,
        )
    rate_axis.set_yticks(
        legacy.ticks_divisible_by_5_or_10(change_limits, target_n=4)
    )
    if change_column == ABSOLUTE_CHANGE_COLUMN:
        if integer_rate_tick_labels:
            rate_axis.yaxis.set_major_formatter(FuncFormatter(
                lambda value, _position: (
                    str(int(round(value * ABSOLUTE_RATE_LABEL_SCALE)))
                    if legacy.np.isclose(
                        value * ABSOLUTE_RATE_LABEL_SCALE,
                        round(value * ABSOLUTE_RATE_LABEL_SCALE),
                    )
                    else ""
                )
            ))
        else:
            rate_axis.yaxis.set_major_formatter(FuncFormatter(
                lambda value, _position:
                f"{value * ABSOLUTE_RATE_LABEL_SCALE:.1f}"
            ))
    else:
        rate_axis.yaxis.set_major_formatter(decimal_ticks)
    # Match the clearance the red label keeps beyond its own tick labels,
    # rather than mirroring its whole spine-to-label gap.  The red ticks are
    # much wider than the black ones, so mirroring the gap pushed the black
    # label needlessly far from its axis.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_bbox = ax.get_window_extent(renderer)
    red_label_bbox = rate_axis.yaxis.label.get_window_extent(renderer)
    black_label = fig.text(
        0.0,
        0.0,
        r"$\langle\rho_\mathrm{tot}\rangle_A$ [cells µm$^{-2}$]",
        color=legacy.COLOR_TOTAL,
        rotation=270,
        ha="center",
        va="center",
    )
    fig.canvas.draw()
    black_label_bbox = black_label.get_window_extent(fig.canvas.get_renderer())
    def _tick_span(axes):
        boxes = [
            label.get_window_extent(renderer)
            for label in axes.get_yticklabels()
            if label.get_text()
        ]
        return boxes or None

    red_ticks = _tick_span(rate_axis)
    black_ticks = _tick_span(ax)
    red_tick_left = min(b.x0 for b in red_ticks) if red_ticks else plot_bbox.x0
    black_tick_right = max(b.x1 for b in black_ticks) if black_ticks else plot_bbox.x1
    label_gap = max(red_tick_left - red_label_bbox.x1, 0.0)
    black_center_x = black_tick_right + label_gap + black_label_bbox.width / 2
    black_label_x = fig.transFigure.inverted().transform(
        (black_center_x, plot_bbox.y0)
    )[0]
    black_label_y = fig.transFigure.inverted().transform(
        (plot_bbox.x0, 0.5 * (plot_bbox.y0 + plot_bbox.y1))
    )[1]
    black_label.set_position((black_label_x, black_label_y))
    ft.savefig(fig, output)
    if legacy.SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    print(f"[done] Saved: {output}")


def appropriate_change_limits(data, change_column: str):
    """Choose limits from the requested change metric only.

    The density-map tables retain both relative and absolute change columns.
    Renaming the selected absolute column onto the relative-column name created
    duplicate labels, causing the old absolute axis limits to include relative
    rates even though the red line itself used absolute rates.
    """
    limits_data = legacy.pd.DataFrame({
        RELATIVE_CHANGE_COLUMN: data[change_column].to_numpy(float),
    })
    return legacy.appropriate_growth_limits(limits_data)


def save_one_cell_density_map_trajectories() -> None:
    """Save the retained one-cell density-map trajectory variants only."""
    experimental = load_at50_trajectory(rim_threshold_only=True)
    simulation = load_simulation_trajectory(rim_threshold_only=True)
    combined = legacy.pd.concat([experimental, simulation], ignore_index=True)
    relative_limits = appropriate_change_limits(combined, RELATIVE_CHANGE_COLUMN)
    absolute_limits = appropriate_change_limits(combined, ABSOLUTE_CHANGE_COLUMN)

    plot_trajectory(experimental, relative_limits, THRESHOLDED_EXPERIMENT_OUTPUT, is_simulation=False)
    plot_trajectory(simulation, relative_limits, THRESHOLDED_SIMULATION_OUTPUT, is_simulation=True)
    plot_trajectory(
        experimental, relative_limits, THRESHOLDED_MATCHED_EXPERIMENT_OUTPUT,
        is_simulation=False, show_markers=True,
    )
    plot_trajectory(
        experimental, absolute_limits, THRESHOLDED_ABSOLUTE_EXPERIMENT_OUTPUT,
        is_simulation=False, change_column=ABSOLUTE_CHANGE_COLUMN,
        change_std_column=ABSOLUTE_CHANGE_STD_COLUMN, change_label=ABSOLUTE_CHANGE_LABEL,
    )
    plot_trajectory(
        simulation, absolute_limits, THRESHOLDED_ABSOLUTE_SIMULATION_OUTPUT,
        is_simulation=True, change_column=ABSOLUTE_CHANGE_COLUMN,
        change_std_column=ABSOLUTE_CHANGE_STD_COLUMN, change_label=ABSOLUTE_CHANGE_LABEL,
    )
    plot_trajectory(
        experimental, absolute_limits, THRESHOLDED_ABSOLUTE_MATCHED_EXPERIMENT_OUTPUT,
        is_simulation=False, show_markers=True, change_column=ABSOLUTE_CHANGE_COLUMN,
        change_std_column=ABSOLUTE_CHANGE_STD_COLUMN, change_label=ABSOLUTE_CHANGE_LABEL,
    )


def main() -> None:
    """Save both raw-HDF5 parameter sets and retained density-map variants."""
    legacy.SHOW_FIGURE = False
    from figures.Figure_3 import plot_at50_hdf5_absolute_density_trajectories

    plot_at50_hdf5_absolute_density_trajectories.main()
    save_one_cell_density_map_trajectories()


# The standard density-map variants without the one-cell threshold remain disabled.


if __name__ == "__main__":
    main()
