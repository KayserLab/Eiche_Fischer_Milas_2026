"""Plot direct-HDF5 experimental and simulated AT50 resistant-density maps.

Rows distinguish experiment and simulation; columns show the Figure 3 start
days 1, 12, and 24. The maps use the 120 and 160 um Figure 3 analysis settings
and one shared resistant-density colour scale.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir
from figures.Figure_3 import plot_at50_density_maps as density_maps


OUTPUT_DIR = figure_plot_dir(
    "Suppl_Figures", "at50_resistant_density_maps", create=True,
)
CACHE_DIRECTORY = figure_data_dir(
    "Suppl_Figures", "at50_resistant_density_maps", "cache", create=True,
)
CACHE_VERSION = 2
KERNEL_RADII_UM = (
    density_maps.HDF5_BASE_KERNEL_RADIUS_UM,
    density_maps.HDF5_VARIANT_KERNEL_RADIUS_UM,
)
MIN_NEIGHBORS = density_maps.hdf5_constants.MIN_KERNEL_NEIGHBORS
MIN_RESISTANT_NUMERATOR = density_maps.hdf5_constants.MIN_RESISTANT_NUMERATOR


def full_interval_source(
    field0, field1, elapsed_days, *, kernel_radius_um, simulation=False,
):
    """Keep full fields and the Figure 3 support decision in native units."""
    total0, resistant0, valid0 = field0
    total1, resistant1, valid1 = field1
    physical_pixel = density_maps.hdf5_constants.px_to_micron
    if simulation:
        physical_pixel *= density_maps.hdf5_constants.exp_scaling
    radius = kernel_radius_um / physical_pixel
    support = density_maps.hdf5_constants.resistant_support(
        total0, resistant0, total1, resistant1, valid0, valid1,
        threshold=MIN_RESISTANT_NUMERATOR / (np.pi * radius ** 2),
    )
    maps = (
        total0 / physical_pixel ** 2,
        total1 / physical_pixel ** 2,
        resistant0 / physical_pixel ** 2,
        resistant1 / physical_pixel ** 2,
    )
    if simulation:
        maps = tuple(values.T for values in maps)
        support = support.T
    # The cache's neighbour_valid slot carries the complete native support
    # decision, including finite-data checks and the inclusive t0 threshold.
    return (*maps, elapsed_days, support)


def load_hdf5_sources(*, kernel_radius_um):
    """Load or build full fields with the same parameters as Figure 3."""
    cache_file = CACHE_DIRECTORY / (
        f"D5_AT50_full_resistant_density_hdf5_{kernel_radius_um:g}um_cache.npz"
    )
    intervals = density_maps.AT50_ADDITIONAL_INTERVALS
    cached = density_maps._load_hdf5_cache(
        cache_file=cache_file,
        cache_version=CACHE_VERSION,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=MIN_NEIGHBORS,
        min_resistant_numerator=MIN_RESISTANT_NUMERATOR,
        intervals=intervals,
    )
    if cached is not None:
        return cached

    experimental_days = [
        int(day) for interval in intervals for day in interval[:2]
    ]
    simulation_days = [
        int(day) for interval in intervals for day in interval[2:]
    ]
    experimental_fields, experimental_extent = (
        density_maps._hdf5_experimental_fields(
            experimental_days,
            kernel_radius_um=kernel_radius_um,
            min_neighbors=MIN_NEIGHBORS,
        )
    )
    simulation_fields, simulation_extent = density_maps._hdf5_simulation_fields(
        simulation_days,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=MIN_NEIGHBORS,
    )
    sources = []
    for exp_start, exp_end, sim_start, sim_end in intervals:
        exp_start, exp_end = int(exp_start), int(exp_end)
        sim_start, sim_end = int(sim_start), int(sim_end)
        experiment = full_interval_source(
            experimental_fields[exp_start], experimental_fields[exp_end],
            exp_end - exp_start,
            kernel_radius_um=kernel_radius_um,
        )
        simulation = full_interval_source(
            simulation_fields[sim_start], simulation_fields[sim_end],
            sim_end - sim_start,
            kernel_radius_um=kernel_radius_um,
            simulation=True,
        )
        sources.append((
            experiment, experimental_extent, simulation, simulation_extent,
        ))
    density_maps._save_hdf5_cache(
        sources,
        cache_file=cache_file,
        cache_version=CACHE_VERSION,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=MIN_NEIGHBORS,
        min_resistant_numerator=MIN_RESISTANT_NUMERATOR,
        intervals=intervals,
    )
    return sources


def build_panels(interval_sources):
    """Prepare density-only panels from direct-HDF5 interval sources."""
    panels = []
    for experiment, experiment_extent, simulation, simulation_extent in interval_sources:
        panel = {}
        for source_name, source, extent in (
            ("experiment", experiment, experiment_extent),
            ("simulation", simulation, simulation_extent),
        ):
            total_density = source[0]
            support = np.asarray(source[5], dtype=bool)
            outline_endpoints = (
                np.where(support, source[2], np.nan),
                np.where(support, source[3], np.nan),
            )
            data = (total_density, None, None, outline_endpoints, None)
            fov = density_maps.magnified_fov(
                density_maps.density_centroid(total_density, extent), extent,
            )
            panel[source_name] = {
                "data": data,
                "resistant_density": source[2],
                "support_mask": support,
                "extent": extent,
                "fov": fov,
            }
        panels.append(panel)
    return panels


def draw_resistant_density_maps(panels, output: Path, *, kernel_radius_um) -> None:
    """Draw experiment/simulation rows with a shared resistant-density scale."""
    source_names = ("simulation", "experiment")
    rim_density_threshold = MIN_RESISTANT_NUMERATOR / (np.pi * kernel_radius_um ** 2)
    display_days = tuple(
        int(interval[0]) for interval in density_maps.AT50_ADDITIONAL_INTERVALS
    )
    resistant_vmax = density_maps.POISSON_SAMPLING_DENSITY_CAP

    timepoint_count = len(panels)
    # Specify square maps and gaps in inches: GridSpec's wspace and hspace
    # are fractions of different average cell sizes (the bar is narrower).
    panel_inches, gap_inches = 1.55, 0.07
    width_ratios = (*([1.0] * timepoint_count), 0.06)
    column_count, row_count = len(width_ratios), len(source_names)
    left, right, bottom, top = 0.12, 0.91, 0.07, 0.91
    figure_width = (
        panel_inches * sum(width_ratios) + gap_inches * (column_count - 1)
    ) / (right - left)
    figure_height = (
        panel_inches * row_count + gap_inches * (row_count - 1)
    ) / (top - bottom)
    figure = plt.figure(figsize=(figure_width, figure_height))
    grid = figure.add_gridspec(
        row_count, column_count, width_ratios=width_ratios,
        wspace=gap_inches / (panel_inches * sum(width_ratios) / column_count),
        hspace=gap_inches / panel_inches,
        left=left, right=right, bottom=bottom, top=top,
    )
    zoom_fraction = density_maps.fitted_rim_zoom_fraction(
        panels, source_names, threshold=rim_density_threshold,
    )
    density_image = None
    map_axes = []
    for row, source_name in enumerate(source_names):
        for column, panel in enumerate(panels):
            source = panel[source_name]
            extent = source["extent"]
            resistant_endpoints = source["data"][3]
            resistant_density = np.asarray(
                source["resistant_density"], dtype=float,
            )
            rim_valid = density_maps.panel_support_mask(
                source, threshold=rim_density_threshold,
            )
            zoom_fov = density_maps.zoomed_fov(
                density_maps.density_centroid(
                    np.where(
                        rim_valid,
                        density_maps.resistant_endpoint_mean(resistant_endpoints),
                        0.0,
                    ), extent,
                ),
                source["fov"],
                zoom_fraction,
            )
            axis = figure.add_subplot(grid[row, column])
            map_axes.append(axis)
            density_image = axis.imshow(
                resistant_density * density_maps.DENSITY_DISPLAY_SCALE,
                cmap="inferno", origin="lower", extent=extent,
                vmin=0.0,
                vmax=resistant_vmax * density_maps.DENSITY_DISPLAY_SCALE,
            )
            # Share Figure 3's yellow colour, short dashes, and line width.
            density_maps.add_resistant_rim(
                axis,
                density_maps.resistant_rim_maps(resistant_endpoints),
                extent,
                threshold=rim_density_threshold,
                colour=density_maps.RESISTANT_OUTLINE_COLOUR,
                support_mask=rim_valid,
            )
            axis.set(
                xlim=zoom_fov[:2], ylim=zoom_fov[2:],
                xticks=[], yticks=[], aspect="equal",
            )
            if row == 0:
                axis.set_title(f"Day {display_days[column]}", fontsize=8, pad=4)
            if column == 0:
                axis.set_ylabel(source_name.capitalize(), fontsize=8, labelpad=7)
                density_maps.add_scale_bar(
                    axis, zoom_fov, colour="white", compact=True, right=True,
                )

    colourbar_axis = figure.add_subplot(grid[:, -1])
    # Equal-aspect maps shrink inside their GridSpec cells. Align the bar to
    # the rendered map edges rather than to the taller, unadjusted cells.
    figure.canvas.draw()
    map_bottom = min(axis.get_position().y0 for axis in map_axes)
    map_top = max(axis.get_position().y1 for axis in map_axes)
    bar_position = colourbar_axis.get_position()
    colourbar_axis.set_position((
        bar_position.x0, map_bottom, bar_position.width, map_top - map_bottom,
    ))
    colourbar = figure.colorbar(density_image, cax=colourbar_axis)
    colourbar.set_label(
        r"$\rho_\mathrm{res}$ [cells µm$^{-2}$]", fontsize=8, labelpad=7,
        rotation=270, verticalalignment="bottom",
    )
    colourbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    colourbar.ax.tick_params(labelsize=7)
    colourbar.ax.annotate(
        r"$\times 10^{-3}$", xy=(0.5, 1.0), xycoords="axes fraction",
        xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=7, annotation_clip=False,
    )
    figure.savefig(output, dpi=600, transparent=False)
    plt.close(figure)


def main() -> None:
    ft.apply_nature_style(
        text_pt=7, tick_pt=6, legend_pt=5.6,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    numerator_tag = f"{MIN_RESISTANT_NUMERATOR:g}".replace(".", "p")
    for radius in KERNEL_RADII_UM:
        panels = build_panels(load_hdf5_sources(kernel_radius_um=radius))
        output = OUTPUT_DIR / (
            "D5_AT50_exp_sim_resistant_density_maps_d01_d12_d24_"
            f"hdf5_{radius:g}um_minres{numerator_tag}_minneighbors{MIN_NEIGHBORS}.pdf"
        )
        draw_resistant_density_maps(panels, output, kernel_radius_um=radius)


if __name__ == "__main__":
    main()
