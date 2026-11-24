"""Plot HDF5-style master curves separately before TTE, after TTE, and over all timepoints.

Each period produces a strategy/master summary and a second PDF containing
every individual simulation-run and experimental-well curve.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis/figure_3/HDF5_analysis")]


import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    AREA_SCALE,
    EXPERIMENT_OUTPUT_DIR,
    FIGURES_OUTPUT_DIR,
    FIG_WIDTH_MM,
    MIN_PROTOCOLS_PER_BIN,
    MIN_RUNS_PER_BIN,
    NATURE_AXIS_LINEWIDTH_PT,
    NATURE_DPI,
    NATURE_FONT_STACK,
    NATURE_LEGEND_PT,
    NATURE_TEXT_PT,
    NATURE_TICK_LENGTH_PT,
    NATURE_TICK_PT,
    SIMULATION_OUTPUT_DIR,
    color_master,
    color_res,
    exp_scaling,
    prolif_n_rho_bins,
    prolif_radius_um,
    prolif_rho_max,
    protocol_colors,
    px_to_micron,
    rho_confluent_res,
    rho_homeostatic_res,
)

STRATEGIES = ("AT50", "AT100", "CT")
PERIODS = ("before_tte", "after_tte", "all_timepoints", "before_ttf")
PERIOD_TITLES = {
    "before_tte": "Before TTE",
    "after_tte": "After TTE",
    "all_timepoints": "All timepoints",
    "before_ttf": "Before TTF",
}
DATASETS = ("simulation", "experiment")
DATASET_TITLES = {"simulation": "Simulation", "experiment": "Experiment"}

SIMULATION_OUTPUT_DIR = os.fspath(SIMULATION_OUTPUT_DIR)
EXPERIMENT_OUTPUT_DIR = os.fspath(EXPERIMENT_OUTPUT_DIR)
OUTPUT_ROOT = os.path.join(
    os.fspath(FIGURES_OUTPUT_DIR), "pressure_analysis", "tte_periods"
)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": NATURE_FONT_STACK,
    "font.size": NATURE_TEXT_PT,
    "axes.labelsize": NATURE_TEXT_PT,
    "axes.titlesize": NATURE_TEXT_PT,
    "xtick.labelsize": NATURE_TICK_PT,
    "ytick.labelsize": NATURE_TICK_PT,
    "legend.fontsize": NATURE_LEGEND_PT,
    "axes.linewidth": NATURE_AXIS_LINEWIDTH_PT,
    "xtick.major.size": NATURE_TICK_LENGTH_PT,
    "ytick.major.size": NATURE_TICK_LENGTH_PT,
})

rho_edges = np.linspace(0, prolif_rho_max, prolif_n_rho_bins + 1)
rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
rho_centers_exp = rho_centers / AREA_SCALE

simulation_density_scale = 1000 / (exp_scaling * px_to_micron) ** 2
experiment_density_scale = 1000 / px_to_micron ** 2
simulation_x = rho_centers * simulation_density_scale
experiment_x = rho_centers_exp * experiment_density_scale
rho_max_display = prolif_rho_max * simulation_density_scale
confluent_display = rho_confluent_res * simulation_density_scale
homeostatic_display = rho_homeostatic_res * simulation_density_scale

XLABEL = r"local density $\rho_\mathrm{tot}$ [cells (1000 µm$^2$)$^{-1}$]"
ABS_YLABEL = r"$\Delta\rho_\mathrm{res}$ [24h, cells (1000 µm$^2$)$^{-1}$]"
REL_YLABEL = r"$\Delta\rho_\mathrm{res} / \rho_\mathrm{res}(t_0)$ [24h$^{-1}$]"


def input_path(dataset, strategy):
    root = SIMULATION_OUTPUT_DIR if dataset == "simulation" else EXPERIMENT_OUTPUT_DIR
    return os.path.join(root, f"flux_vs_density_{strategy}.h5")


def load_curves(dataset, period):
    absolute, relative = {}, {}
    for strategy in STRATEGIES:
        path = input_path(dataset, strategy)
        with h5py.File(path, "r") as handle:
            if period not in handle:
                raise KeyError(
                    f"{path} has no {period!r} group. Rerun the corresponding "
                    "compute_flux_vs_density script."
                )
            group = handle[period]
            absolute[strategy] = group["median"][:]
            relative[strategy] = group["median_rel"][:]
    return absolute, relative


def strategy_statistics(run_curves):
    result = {}
    for strategy, values in run_curves.items():
        count = np.isfinite(values).sum(axis=0)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(values, axis=0)
            q25 = np.nanpercentile(values, 25, axis=0)
            q75 = np.nanpercentile(values, 75, axis=0)
        enough = count >= MIN_RUNS_PER_BIN
        result[strategy] = tuple(
            np.where(enough, array, np.nan) for array in (median, q25, q75)
        )
    return result


def master_statistics(strategy_curves):
    values = np.stack([strategy_curves[name][0] for name in STRATEGIES])
    count = np.isfinite(values).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
        lower = np.nanmin(values, axis=0)
        upper = np.nanmax(values, axis=0)
    enough = count >= MIN_PROTOCOLS_PER_BIN
    return tuple(
        np.where(enough, array, np.nan) for array in (mean, lower, upper)
    )


def style_axis(axis, ylabel):
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(confluent_display, color=color_res, linestyle=":", linewidth=0.8)
    axis.axvline(homeostatic_display, color=color_res, linestyle="--", linewidth=0.8)
    axis.set_xlim(0, rho_max_display)
    axis.set_xlabel(XLABEL)
    axis.set_ylabel(ylabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def region_ylim(specifications):
    """Return limits from data between the two density references."""
    values = []
    lower_x, upper_x = sorted((confluent_display, homeostatic_display))
    for x, arrays, scale in specifications:
        in_region = (x >= lower_x) & (x <= upper_x)
        for array in arrays:
            data = np.asarray(array, dtype=float)
            selected = data[..., in_region] * scale
            values.extend(selected[np.isfinite(selected)].ravel())
    if not values:
        return None
    lower = min(0.0, float(np.min(values)))
    upper = max(0.0, float(np.max(values)))
    span = upper - lower
    padding = 0.08 * span if span > 0 else max(abs(lower), abs(upper), 1.0) * 0.08
    return lower - padding, upper + padding


def set_shared_region_ylim(axes, specifications, fixed_limits=None):
    """Apply one y-range to paired simulation and experimental panels."""
    limits = fixed_limits if fixed_limits is not None else region_ylim(specifications)
    if limits is None:
        return
    for axis in axes:
        axis.set_ylim(*limits)


RELATIVE_Y_LIMITS = (0.0, 1.0)


def summary_arrays(curves, master):
    return [
        array
        for strategy in STRATEGIES
        for array in curves[strategy]
    ] + list(master)


def individual_arrays(curves, master):
    return [curves[strategy] for strategy in STRATEGIES] + [master[0]]


def draw_summary(axis, curves, master, x, y_scale, ylabel, linestyle):
    for strategy in STRATEGIES:
        median, lower, upper = curves[strategy]
        color = protocol_colors[strategy]
        axis.plot(
            x, median * y_scale, color=color, linestyle=linestyle, label=strategy
        )
        axis.fill_between(
            x, lower * y_scale, upper * y_scale, color=color, alpha=0.2, linewidth=0
        )
    mean, lower, upper = master
    axis.plot(
        x,
        mean * y_scale,
        color=color_master,
        linestyle=linestyle,
        linewidth=1.5,
        label="mean",
    )
    axis.fill_between(
        x, lower * y_scale, upper * y_scale, color=color_master, alpha=0.15, linewidth=0
    )
    style_axis(axis, ylabel)
    axis.legend()


def draw_individual(axis, run_curves, master, x, y_scale, ylabel, linestyle):
    for strategy in STRATEGIES:
        color = protocol_colors[strategy]
        for curve in run_curves[strategy]:
            axis.plot(
                x,
                curve * y_scale,
                color=color,
                linestyle=linestyle,
                linewidth=0.6,
                alpha=0.35,
            )
        axis.plot([], [], color=color, linestyle=linestyle, label=strategy)
    axis.plot(
        x,
        master[0] * y_scale,
        color=color_master,
        linestyle=linestyle,
        linewidth=1.5,
        label="mean",
    )
    style_axis(axis, ylabel)
    axis.legend()


def save(fig, subdirectory, filename):
    directory = os.path.join(OUTPUT_ROOT, subdirectory)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def calculate_global_y_limits():
    """Use all three periods to define fixed summary and individual limits."""
    specifications = {
        "summary_absolute": [],
        "summary_relative": [],
        "individual_absolute": [],
        "individual_relative": [],
    }
    for period in PERIODS:
        (
            sim_abs, sim_rel, exp_abs, exp_rel,
            sim_abs_stats, sim_rel_stats, exp_abs_stats, exp_rel_stats,
            sim_abs_master, sim_rel_master, exp_abs_master, exp_rel_master,
        ) = load_period_bundle(period)
        specifications["summary_absolute"].extend((
            (simulation_x, summary_arrays(sim_abs_stats, sim_abs_master), simulation_density_scale),
            (experiment_x, summary_arrays(exp_abs_stats, exp_abs_master), experiment_density_scale),
        ))
        specifications["summary_relative"].extend((
            (simulation_x, summary_arrays(sim_rel_stats, sim_rel_master), 1.0),
            (experiment_x, summary_arrays(exp_rel_stats, exp_rel_master), 1.0),
        ))
        specifications["individual_absolute"].extend((
            (simulation_x, individual_arrays(sim_abs, sim_abs_master), simulation_density_scale),
            (experiment_x, individual_arrays(exp_abs, exp_abs_master), experiment_density_scale),
        ))
        specifications["individual_relative"].extend((
            (simulation_x, individual_arrays(sim_rel, sim_rel_master), 1.0),
            (experiment_x, individual_arrays(exp_rel, exp_rel_master), 1.0),
        ))
    return {name: region_ylim(items) for name, items in specifications.items()}


def load_period_bundle(period):
    """Curves, per-strategy statistics, and master statistics for one period."""
    sim_abs, sim_rel = load_curves("simulation", period)
    exp_abs, exp_rel = load_curves("experiment", period)
    sim_abs_stats = strategy_statistics(sim_abs)
    sim_rel_stats = strategy_statistics(sim_rel)
    exp_abs_stats = strategy_statistics(exp_abs)
    exp_rel_stats = strategy_statistics(exp_rel)
    sim_abs_master = master_statistics(sim_abs_stats)
    sim_rel_master = master_statistics(sim_rel_stats)
    exp_abs_master = master_statistics(exp_abs_stats)
    exp_rel_master = master_statistics(exp_rel_stats)
    return (
        sim_abs, sim_rel, exp_abs, exp_rel,
        sim_abs_stats, sim_rel_stats, exp_abs_stats, exp_rel_stats,
        sim_abs_master, sim_rel_master, exp_abs_master, exp_rel_master,
    )


def plot_period(period, global_y_limits):
    (
        sim_abs, sim_rel, exp_abs, exp_rel,
        sim_abs_stats, sim_rel_stats, exp_abs_stats, exp_rel_stats,
        sim_abs_master, sim_rel_master, exp_abs_master, exp_rel_master,
    ) = load_period_bundle(period)

    size = (FIG_WIDTH_MM / 25.4 * 0.8, FIG_WIDTH_MM / 25.4 * 0.64)
    fig, axes = plt.subplots(2, 2, figsize=size, sharey="col")
    draw_summary(
        axes[0, 0], sim_abs_stats, sim_abs_master, simulation_x,
        simulation_density_scale, ABS_YLABEL, "--",
    )
    draw_summary(
        axes[0, 1], sim_rel_stats, sim_rel_master, simulation_x,
        1.0, REL_YLABEL, "--",
    )
    draw_summary(
        axes[1, 0], exp_abs_stats, exp_abs_master, experiment_x,
        experiment_density_scale, ABS_YLABEL, "-",
    )
    draw_summary(
        axes[1, 1], exp_rel_stats, exp_rel_master, experiment_x,
        1.0, REL_YLABEL, "-",
    )
    set_shared_region_ylim(
        (axes[0, 0], axes[1, 0]),
        (
            (simulation_x, summary_arrays(sim_abs_stats, sim_abs_master), simulation_density_scale),
            (experiment_x, summary_arrays(exp_abs_stats, exp_abs_master), experiment_density_scale),
        ),
        global_y_limits["summary_absolute"],
    )
    set_shared_region_ylim(
        (axes[0, 1], axes[1, 1]),
        (
            (simulation_x, summary_arrays(sim_rel_stats, sim_rel_master), 1.0),
            (experiment_x, summary_arrays(exp_rel_stats, exp_rel_master), 1.0),
        ),
        RELATIVE_Y_LIMITS,
    )
    axes[0, 0].set_title(f"Simulation {PERIOD_TITLES[period]}", loc="left")
    axes[1, 0].set_title(f"Experiment {PERIOD_TITLES[period]}", loc="left")
    fig.tight_layout()
    save(fig, "summary", f"flux_vs_density_{period}.pdf")

    fig, axes = plt.subplots(2, 2, figsize=size, sharey="col")
    draw_individual(
        axes[0, 0], sim_abs, sim_abs_master, simulation_x,
        simulation_density_scale, ABS_YLABEL, "--",
    )
    draw_individual(
        axes[0, 1], sim_rel, sim_rel_master, simulation_x,
        1.0, REL_YLABEL, "--",
    )
    draw_individual(
        axes[1, 0], exp_abs, exp_abs_master, experiment_x,
        experiment_density_scale, ABS_YLABEL, "-",
    )
    draw_individual(
        axes[1, 1], exp_rel, exp_rel_master, experiment_x,
        1.0, REL_YLABEL, "-",
    )
    set_shared_region_ylim(
        (axes[0, 0], axes[1, 0]),
        (
            (simulation_x, individual_arrays(sim_abs, sim_abs_master), simulation_density_scale),
            (experiment_x, individual_arrays(exp_abs, exp_abs_master), experiment_density_scale),
        ),
        global_y_limits["individual_absolute"],
    )
    set_shared_region_ylim(
        (axes[0, 1], axes[1, 1]),
        (
            (simulation_x, individual_arrays(sim_rel, sim_rel_master), 1.0),
            (experiment_x, individual_arrays(exp_rel, exp_rel_master), 1.0),
        ),
        RELATIVE_Y_LIMITS,
    )
    axes[0, 0].set_title(f"Simulation â€” {PERIOD_TITLES[period]}", loc="left")
    axes[1, 0].set_title(f"Experiment â€” {PERIOD_TITLES[period]}", loc="left")
    fig.tight_layout()
    save(
        fig,
        "single_trajectories",
        f"flux_vs_density_{period}_single_trajectories.pdf",
    )


# The three periods sit across one full two-column page width.
PERIOD_COMPARISON_SIZE = (FIG_WIDTH_MM / 25.4 * 4 / 3, FIG_WIDTH_MM / 25.4 * 0.58)


def period_panel(bundle, dataset, quantity):
    """Return the curves, statistics, and drawing settings for one panel."""
    (
        sim_abs, sim_rel, exp_abs, exp_rel,
        sim_abs_stats, sim_rel_stats, exp_abs_stats, exp_rel_stats,
        sim_abs_master, sim_rel_master, exp_abs_master, exp_rel_master,
    ) = bundle
    table = {
        ("simulation", "absolute"): (
            sim_abs, sim_abs_stats, sim_abs_master, simulation_x,
            simulation_density_scale, "--",
        ),
        ("simulation", "relative"): (
            sim_rel, sim_rel_stats, sim_rel_master, simulation_x, 1.0, "--",
        ),
        ("experiment", "absolute"): (
            exp_abs, exp_abs_stats, exp_abs_master, experiment_x,
            experiment_density_scale, "-",
        ),
        ("experiment", "relative"): (
            exp_rel, exp_rel_stats, exp_rel_master, experiment_x, 1.0, "-",
        ),
    }
    return table[(dataset, quantity)]


def tidy_comparison_grid(axes):
    """Keep the outer axis labels and a single legend."""
    rows, columns = axes.shape
    for row in range(rows):
        for column in range(columns):
            axis = axes[row, column]
            if row != rows - 1:
                axis.set_xlabel("")
            if column != 0:
                axis.set_ylabel("")
            legend = axis.get_legend()
            if legend is not None and (row, column) != (0, 0):
                legend.remove()


def plot_period_comparison(global_y_limits):
    """Draw the three periods side by side so they can be compared directly.

    One figure per quantity and per curve kind: rows are simulation and
    experiment, columns are the periods.  All panels of a figure share one
    y-range, which is what makes the periods comparable by eye.
    """
    bundles = {period: load_period_bundle(period) for period in PERIODS}
    variants = (
        ("summary", draw_summary, True),
        ("single_trajectories", draw_individual, False),
    )
    quantities = (
        ("absolute", ABS_YLABEL, "summary_absolute", "individual_absolute"),
        ("relative", REL_YLABEL, None, None),
    )
    for kind, draw, use_statistics in variants:
        for quantity, ylabel, summary_key, individual_key in quantities:
            if quantity == "relative":
                limits = RELATIVE_Y_LIMITS
            else:
                key = summary_key if kind == "summary" else individual_key
                limits = global_y_limits[key]
            fig, axes = plt.subplots(
                len(DATASETS), len(PERIODS), figsize=PERIOD_COMPARISON_SIZE,
            )
            for row, dataset in enumerate(DATASETS):
                for column, period in enumerate(PERIODS):
                    runs, stats, master, x, y_scale, linestyle = period_panel(
                        bundles[period], dataset, quantity,
                    )
                    axis = axes[row, column]
                    draw(
                        axis,
                        stats if use_statistics else runs,
                        master, x, y_scale, ylabel, linestyle,
                    )
                    axis.set_title(
                        f"{DATASET_TITLES[dataset]} - {PERIOD_TITLES[period]}",
                        loc="left",
                    )
            if limits is not None:
                for axis in axes.ravel():
                    axis.set_ylim(*limits)
            tidy_comparison_grid(axes)
            fig.suptitle(
                f"Kernel radius: {prolif_radius_um:g} µm",
                y=1.02,
            )
            fig.tight_layout()
            suffix = "" if kind == "summary" else "_single_trajectories"
            save(
                fig,
                "period_comparison",
                f"flux_vs_density_periods_{quantity}{suffix}.pdf",
            )


if __name__ == "__main__":
    shared_y_limits = calculate_global_y_limits()
    for requested_period in PERIODS:
        plot_period(requested_period, shared_y_limits)
    plot_period_comparison(shared_y_limits)
