#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot matched D5 AT50 experiment/simulation maps from the shared 160 um, 16x pipeline."""

# =============================================================================
# UPSTREAM DATA AND ANALYSIS
# =============================================================================
# Experimental inputs:
# - raw fluorescence TIFFs in
#   Data/exp_raw_data/Plate_4/Tifs;
# - 160 um / 16x total and resistant-density TIFFs generated from the Figure 2
#   cleaned sensitive masks and all resistant-clone masks. The run is shared
#   with the Figure 3 master-curve analysis.
# Simulation inputs are the AT50 run_18 HDF5 snapshots.  They are filtered and
# converted to matching 160 um / 16x density maps in memory using:
#   analysis/basic/density_maps/Density_Maps_Epanechnikov_simulations_coarsegrained.py
#   analysis/figure_3/Legacy/aggregate_average_symmetric_condition_matrix_160.py
# This temporary variant displays the resistant relative change weighted by the
# susceptible fraction of the endpoint-average local total density.
# Historical note only: the former AT100 companion data were Plate_5 /
# A2_AT100-A2 (matched to simulation R20_AT100-R20).  They are no longer
# loaded or plotted by this script.

from __future__ import annotations
import os, sys
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import ndimage
import tifffile as tiff


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [os.fspath(ROOT)]
from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as analysis
from shared import figtools as ft
from shared.strategy_colors import FIGURE_COLOR_MAP
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, figure_plot_dir

OUT = figure_plot_dir(
    "Figure_3", "density_maps", "temporary_susceptible_weighted_relative_change",
    create=False,
)
ALL_TIMEPOINTS_OUT = OUT / "all_timepoints"
# Only the final AT50 d1/d12/d26 comparison is written to the shared drive.
# Set to True only when the auxiliary, previously generated figures are wanted.
SAVE_AUXILIARY_SHARED_FIGURES = False
# Set to False to skip the wide experimental/simulation all-timepoint overview
# while still producing the regular Figure 3 density-map exports.
PLOT_ALL_EXPERIMENTAL_TIMEPOINTS = True
PLATE, WELL, TP0, TP1 = "Plate_4", "D5_AT50-D5", "17", "19"
OUTPUT_DAY_TAG = f"d{int(TP0):02d}"
RELATIVE_EXPERIMENT_PDF = (
    f"D5_AT50_experiment_relative_resistant_density_change_{OUTPUT_DAY_TAG}_160um_16x.pdf"
)
RELATIVE_SIMULATION_PDF = (
    f"D5_AT50_simulation_relative_resistant_density_change_{OUTPUT_DAY_TAG}_160um_16x.pdf"
)
SIMULATION_CHANGE_TIMECOURSE_PDF = "D5_AT50_simulation_resistant_density_change_timecourse.pdf"
SIM_TP0, SIM_TP1 = "17", "18"
AT50_ADDITIONAL_INTERVALS = (
    # (experimental start, experimental end, simulation start, simulation end)
    ("01", "03", "01", "02"),
    ("12", "15", "12", "13"),
    ("26", "29", "26", "27"),
)
SCALE_BAR_UM = 1000.0
# The overexposure correction assumes 0.00286 cells per image pixel².  Convert
# it to the density-map physical unit using the 1.095 µm image-pixel size.
POISSON_OVEREXPOSURE_DENSITY_PER_PIXEL = 0.00286
POISSON_SAMPLING_DENSITY_CAP = (
    POISSON_OVEREXPOSURE_DENSITY_PER_PIXEL / analysis.sim.PIXEL_SIZE_UM ** 2
)  # cells / µm²; displayed as ≈2.385 cells / 1000 µm²
# Both supplied 8200 x 8200 simulation PNGs share this physical calibration.
# It was fitted from the user-selected d12/d26 landmark pairs (5–6 µm RMS).
SIMULATION_PNG_UM_PER_PIXEL = 1.3472772618
SIMULATION_PNG_ORIGIN_UM = (2253.1472934, 1598.0875438)
# Keep large computed data beside the other Figure 3 data, never beside code.
CACHE_DIRECTORY = figure_data_dir("Figure_3", "density_maps", "cache", create=False)
SIMULATION_PNG_DIRECTORY = figure_data_dir("Figure_3", "density_maps", create=False)
CACHE_FILE = CACHE_DIRECTORY / (
    f"d5_at50_density_maps_exp{TP0}_{TP1}_sim{SIM_TP0}_{SIM_TP1}_cache.npz"
)
CACHE_VERSION = 6
COMPARISON_CACHE_FILE = CACHE_DIRECTORY / (
    "d5_at50_density_maps_exp01_03_exp12_15_exp26_29_"
    "sim01_02_sim12_13_sim26_27_cache.npz"
)
COMPARISON_CACHE_VERSION = 13
# Keep this False for fast redraws after plot-only edits.  Set it to True once
# only when the raw images, density maps, or simulation input have changed.
FORCE_REBUILD_CACHE = False
RAW_EXPERIMENT_DIR = EXP_RAW_DATA / "Plate_4" / "Tifs"
EXPERIMENTAL_DENSITY_RUN = analysis.DEFAULT_INVITRO_RUN
COARSENING_FACTOR = 16
COARSE_PIXEL_SIZE_UM = analysis.sim.PIXEL_SIZE_UM * COARSENING_FACTOR
DENSITY_DISPLAY_SCALE = 1000.0
RAW_BACKGROUND_PERCENTILE = 62.0
RAW_SIGNAL_PERCENTILE = 100
BLUE_SIGNAL_PERCENTILE = 96.5
YELLOW_BACKGROUND_PERCENTILE = 90.0
# A mild display-only lift makes the dim sensitive/blue fluorescence visible
# without changing the underlying image data or the resistant/yellow channel.
BLUE_SIGNAL_GAMMA = 0.45
BLUE_DISPLAY_FLOOR = 0.25
YELLOW_DISPLAY_FLOOR = 0.49
YELLOW_SIGNAL_GAMMA = 0.20
RAW_DISPLAY_CACHE_SIGNATURE = np.asarray((
    RAW_BACKGROUND_PERCENTILE,
    RAW_SIGNAL_PERCENTILE,
    BLUE_SIGNAL_PERCENTILE,
    BLUE_SIGNAL_GAMMA,
    BLUE_DISPLAY_FLOOR,
    YELLOW_BACKGROUND_PERCENTILE,
    YELLOW_DISPLAY_FLOOR,
    YELLOW_SIGNAL_GAMMA,
), dtype=float)
DISPLAY_FOV_FRACTION = 0.45
# Lower-row fields of view: smaller fraction means a stronger zoom-in.
CHANGE_PANEL_ZOOM_FRACTION = 0.35
# The compact two-timepoint comparison needs a slightly wider context.
COMPARISON_ZOOM_FRACTION = CHANGE_PANEL_ZOOM_FRACTION * 1.05 * 1.05
RIM_OUTWARD_OFFSET_PIXELS = 2
MIN_RIM_COMPONENT_PIXELS = 2
RESISTANT_OUTLINE_COLOUR = "#DAA520"
RATE_LABEL_COLOUR = "black"
RELATIVE_DENSITY_FLOOR = 1e-10
RELATIVE_CHANGE_MIN_PER_DAY = -1.05
RELATIVE_CHANGE_MAX_PER_DAY = 100.0
RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY = 5.0
LOCAL_DENSITY_LABEL = r"Local density [cells µm$^{-2}$]"
DENSITY_COLORBAR_LABEL = r"$\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL = (
    r"$(\Delta\tilde{\rho}_\mathrm{res}/\Delta t)"
    r"(\bar{\rho}_\mathrm{sus}/\bar{\rho}_\mathrm{tot})$ [day$^{-1}$]"
)
CHANGE_DISPLAY_PERCENTILE = 99.8
SIMULATION_CHANGE_INTERVALS = (("15", "16"), ("16", "17"), ("17", "18"),
                               ("18", "19"), ("19", "20"), ("20", "21"))


def raw_fluorescence_composite(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Return the sensitive (blue) / resistant (yellow) fluorescence composite."""
    image = tiff.imread(path).astype(float)
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError(f"Expected a CYX three-channel microscopy TIFF, got {image.shape}: {path}")

    # The acquisition order is blue/sensitive, yellow/resistant, gray/brightfield.
    # Deliberately omit the gray/brightfield channel from this fluorescence panel.
    # Suppress the diffuse low-intensity background in each fluorescence channel.
    def scale(channel, signal_percentile=RAW_SIGNAL_PERCENTILE, gamma=1.0,
              background_percentile=RAW_BACKGROUND_PERCENTILE):
        low = np.nanpercentile(channel, background_percentile)
        high = np.nanpercentile(channel, signal_percentile)
        if high <= low:
            return np.zeros_like(channel, dtype=float)
        return np.clip((channel - low) / (high - low), 0, 1) ** gamma

    sensitive = scale(image[0], BLUE_SIGNAL_PERCENTILE, BLUE_SIGNAL_GAMMA)
    resistant = scale(
        image[1],
        gamma=YELLOW_SIGNAL_GAMMA,
        background_percentile=YELLOW_BACKGROUND_PERCENTILE,
    )
    # Suppress weak diffuse fluorescence after the display transforms, while
    # retaining genuine blue/yellow overlap at stronger intensities.
    sensitive[sensitive < BLUE_DISPLAY_FLOOR] = 0
    resistant[resistant < YELLOW_DISPLAY_FLOOR] = 0
    # Add both fluorescence channels so overlap remains visible rather than
    # replacing one cell type with whichever signal happens to be brighter.
    composite = np.clip(
        sensitive[..., np.newaxis] * np.asarray(to_rgb(FIGURE_COLOR_MAP["blue"]))
        + resistant[..., np.newaxis] * np.asarray(to_rgb(FIGURE_COLOR_MAP["yellow"])),
        0, 1,
    )
    height, width = composite.shape[:2]
    target_height, target_width = target_shape
    if height > target_height or width > target_width:
        raise ValueError(f"Raw image {composite.shape[:2]} exceeds density-map canvas {target_shape}")
    pad_y = target_height - height
    pad_x = target_width - width
    return np.pad(
        composite,
        ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (0, 0)),
    ).astype(np.float32)

def resistant_change(previous, current, elapsed_days, *, relative):
    if relative:
        result = np.full_like(current, np.nan, dtype=float)
        valid = (
            np.isfinite(previous)
            & np.isfinite(current)
            & (previous > RELATIVE_DENSITY_FLOOR)
        )
        # Preserve the established map definition: relative change from the
        # first endpoint, normalized by the elapsed number of days.
        result[valid] = (current[valid] - previous[valid]) / previous[valid] / elapsed_days
        result[
            (result < RELATIVE_CHANGE_MIN_PER_DAY)
            | (result > RELATIVE_CHANGE_MAX_PER_DAY)
        ] = np.nan
        return result
    return (current - previous) / elapsed_days


def susceptible_weighted_relative_change(
    total_t0, total_t1, resistant_t0, resistant_t1, elapsed_days,
):
    """Return Δrho_res/rho_res(t0) times the endpoint-average sus fraction.

    The weight uses bar(rho_sus)/bar(rho_tot), where each bar is the mean of
    t0 and t1, matching the total-density map displayed beside the change map.
    """
    total_average = 0.5 * (total_t0 + total_t1)
    resistant_average = 0.5 * (resistant_t0 + resistant_t1)
    susceptible_average = np.maximum(total_average - resistant_average, 0.0)
    result = np.full_like(total_average, np.nan, dtype=float)
    valid = (
        np.isfinite(total_average)
        & np.isfinite(resistant_t0)
        & np.isfinite(resistant_t1)
        & (total_average > 0)
        & (resistant_t0 > RELATIVE_DENSITY_FLOOR)
    )
    result[valid] = (
        (resistant_t1[valid] - resistant_t0[valid])
        / resistant_t0[valid]
        / elapsed_days
        * susceptible_average[valid]
        / total_average[valid]
    )
    result[
        (result < RELATIVE_CHANGE_MIN_PER_DAY)
        | (result > RELATIVE_CHANGE_MAX_PER_DAY)
    ] = np.nan
    return result


def remove_single_pixel_change_outliers(change: np.ndarray) -> np.ndarray:
    """Mask isolated nonzero coarse-grid pixels in a displayed change map."""
    valid = np.isfinite(change) & ~np.isclose(change, 0)
    labels, count = ndimage.label(valid)
    if count == 0:
        return change
    component_sizes = np.bincount(labels.ravel())
    isolated = (labels > 0) & (component_sizes[labels] == 1)
    if not np.any(isolated):
        return change
    filtered = change.copy()
    filtered[isolated] = np.nan
    return filtered


def raw_experiment_path(day: str, *, well: str = WELL,
                        raw_directory: Path = RAW_EXPERIMENT_DIR,
                        plate: str = PLATE) -> Path:
    """Return the raw microscopy image at the first day of the interval."""
    plate_number = int(plate.removeprefix("Plate_"))
    matches = sorted(raw_directory.glob(
        f"*_d{int(float(day)):02d}_Plate{plate_number}_MIPs_{well}.tif"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one raw TIFF for {well} at d{day}, found: {matches}"
        )
    return matches[0]


def experimental(*, plate: str = PLATE, well: str = WELL,
                 raw_directory: Path = RAW_EXPERIMENT_DIR,
                 tp0: str = TP0, tp1: str = TP1):
    maps = EXPERIMENTAL_DENSITY_RUN / plate / well / "density_maps"
    total_t0 = tiff.imread(analysis.base.density_map_path(maps, well, tp0)).astype(float)
    total_t1 = tiff.imread(analysis.base.density_map_path(maps, well, tp1)).astype(float)
    r0 = tiff.imread(analysis.base.resistant_density_map_path(maps, well, tp0)).astype(float)
    r1 = tiff.imread(analysis.base.resistant_density_map_path(maps, well, tp1)).astype(float)
    elapsed_days = float(tp1) - float(tp0)
    if elapsed_days <= 0:
        raise ValueError("TP1 must be later than TP0")
    raw_image = raw_fluorescence_composite(
        raw_experiment_path(tp0, well=well, raw_directory=raw_directory, plate=plate),
        (total_t1.shape[0] * COARSENING_FACTOR, total_t1.shape[1] * COARSENING_FACTOR),
    )
    return total_t0, total_t1, r0, r1, raw_image, elapsed_days

def load_simulation_history(strategy: str = "AT50"):
    """Load the simulation snapshots once for every output in this script."""
    args = analysis.sweep.simulation_args(
        os.fspath(analysis.DEFAULT_SIM_ROOT), strategy, "run_18",
    )
    plan = analysis.sim.discover_run_plans(args)[0]
    loaded = [analysis.sim.load_filtered_cells(x.path, args) for x in plan.files]
    canvas = analysis.sweep.canvas_from_loaded(loaded, float(analysis.KERNEL_UM))
    return args, plan, loaded, canvas, {}


def simulation(history, *, tp0: str = SIM_TP0, tp1: str = SIM_TP1):
    _, plan, loaded, canvas, density_cache = history
    index0 = next(i for i, x in enumerate(plan.files) if float(x.day) >= float(tp0))
    index1 = next(i for i, x in enumerate(plan.files) if float(x.day) >= float(tp1))
    def density(index):
        if index not in density_cache:
            density_cache[index] = analysis.simulation_density(*loaded[index], canvas)
        return density_cache[index]

    r0, t0 = density(index0)
    r1, t1 = density(index1)
    elapsed_days = float(plan.files[index1].day) - float(plan.files[index0].day)
    if elapsed_days <= 0:
        raise ValueError("Simulation end day must be later than start day")
    return t0, t1, r0, r1, (loaded[index0][0], loaded[index0][1], canvas), elapsed_days


def orient_simulation_map(array):
    """Rotate 90° clockwise, then mirror across the horizontal axis.

    With images displayed using ``origin='lower'``, this requested visual
    transformation is a transpose of the displayed array.
    """
    return array.T


def orient_simulation_source(source_data):
    """Apply the requested display orientation to raw cells and all map data."""
    total_t0, total_t1, r0, r1, image, elapsed_days = source_data
    positions, cell_types, canvas = image
    local_positions = positions - np.asarray([canvas.min_x_um, canvas.min_y_um])
    transformed_positions = local_positions[:, [1, 0]]
    transformed_canvas = SimpleNamespace(
        min_x_um=0.0,
        min_y_um=0.0,
        width_px=canvas.height_px,
        height_px=canvas.width_px,
    )
    return (
        orient_simulation_map(total_t0),
        orient_simulation_map(total_t1),
        orient_simulation_map(r0),
        orient_simulation_map(r1),
        (transformed_positions, cell_types, transformed_canvas),
        elapsed_days,
    )


def panel_data(source_data, *, relative):
    total_t0, total_t1, r0, r1, image, elapsed_days = source_data
    # These are the master curve's x and resistant-density definitions.
    total = 0.5 * (total_t0 + total_t1)
    resistant_average = 0.5 * (r0 + r1)
    change = (
        susceptible_weighted_relative_change(total_t0, total_t1, r0, r1, elapsed_days)
        if relative else resistant_change(r0, r1, elapsed_days, relative=False)
    )
    # The rim must follow the same change quantity that is visualised: absolute
    # for absolute maps and relative for relative-growth maps.
    return total, change, image, resistant_average, change


def simulation_png(day: str):
    """Load the supplied simulation image in the requested display orientation."""
    image_path = SIMULATION_PNG_DIRECTORY / f"day_{int(day)}_sim.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing simulation panel image: {image_path}")
    # The required orientation is algebraically equivalent to one clockwise
    # 90° rotation of the raster.
    return np.rot90(plt.imread(image_path), -1)


def simulation_png_extent(source_data, image, map_extent):
    """Return the shared, landmark-calibrated physical extent of a PNG."""
    height, width = image.shape[:2]
    scale = SIMULATION_PNG_UM_PER_PIXEL
    offset = SIMULATION_PNG_ORIGIN_UM
    return (offset[0], offset[0] + scale * width,
            offset[1], offset[1] + scale * height)


def simulation_panel_data(source_data, *, relative, day: str):
    """Use the supplied simulation PNG as the raw-image panel background."""
    total, change, _image, resistant_average, boundary_change = panel_data(
        source_data, relative=relative,
    )
    return total, change, simulation_png(day), resistant_average, boundary_change


def plot_simulation_change_timecourse(output: Path, history):
    """Diagnostic side-by-side simulation change maps over matched intervals."""
    _, plan, loaded, canvas, density_cache = history
    extent = (0, canvas.height_px * analysis.sim.PIXEL_SIZE_UM,
              0, canvas.width_px * analysis.sim.PIXEL_SIZE_UM)

    def density(index):
        if index not in density_cache:
            density_cache[index] = analysis.simulation_density(*loaded[index], canvas)
        return density_cache[index]

    panels = []
    for first_day, next_day in SIMULATION_CHANGE_INTERVALS:
        first_index = next(i for i, item in enumerate(plan.files) if float(item.day) >= float(first_day))
        next_index = next(i for i, item in enumerate(plan.files) if float(item.day) >= float(next_day))
        r0, total_t0 = density(first_index)
        r1, total_t1 = density(next_index)
        elapsed_days = float(plan.files[next_index].day) - float(plan.files[first_index].day)
        panels.append((
            f"d{first_day}→d{next_day}",
            orient_simulation_map(resistant_change(r0, r1, elapsed_days, relative=False)),
            orient_simulation_map(susceptible_weighted_relative_change(
                total_t0, total_t1, r0, r1, elapsed_days,
            )),
            orient_simulation_map(r1),
        ))

    absolute_values = np.concatenate([np.abs(panel[1]).ravel() for panel in panels])
    absolute_limit = max(np.nanpercentile(absolute_values[np.isfinite(absolute_values)], CHANGE_DISPLAY_PERCENTILE), 1e-8)
    relative_norm = TwoSlopeNorm(vmin=RELATIVE_CHANGE_MIN_PER_DAY, vcenter=0.0,
                                 vmax=RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY)
    change_cmap = plt.get_cmap("coolwarm").copy()
    change_cmap.set_bad("white")
    fig, axes = plt.subplots(2, len(panels), figsize=(13.2, 4.4), constrained_layout=True)
    for column, (label, absolute, relative, resistant) in enumerate(panels):
        arrays = (
            (np.ma.masked_where(np.isclose(absolute, 0), absolute), None),
            (np.ma.masked_invalid(relative), relative_norm),
        )
        for row, (array, norm) in enumerate(arrays):
            ax = axes[row, column]
            imshow_args = dict(cmap=change_cmap, origin="lower", extent=extent)
            if norm is None:
                imshow_args.update(vmin=-absolute_limit, vmax=absolute_limit)
            else:
                imshow_args["norm"] = norm
            image = ax.imshow(array, **imshow_args)
            ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=8)
            if column == 0:
                ax.set_ylabel(
                    "Absolute change\nper day" if row == 0
                    else RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL,
                    fontsize=7,
                )
            add_scale_bar(ax, extent, colour="#222222")
            if column == len(panels) - 1:
                colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
                colorbar.set_label(
                    "Resistant density change [cells µm$^{-2}$ day$^{-1}$]" if row == 0
                    else RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL,
                    fontsize=7,
                )
    fig.suptitle(
        "AT50 simulation (run 18): susceptible-weighted resistant relative change",
        y=1.02,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def save_plot_cache(experimental_source, simulation_source, simulation_history):
    """Save the expensive input preparation for fast plot-only redraws."""
    total_t0, total_t1, r0, r1, raw_image, elapsed_days = experimental_source
    sim_total_t0, sim_total_t1, sim_r0, sim_r1, (positions, cell_types, canvas), sim_elapsed_days = simulation_source
    _, plan, loaded, canvas, density_cache = simulation_history
    daily_resistant = []
    for day in range(15, 22):
        index = next(i for i, item in enumerate(plan.files) if float(item.day) >= day)
        # These daily maps feed the optional time-course cache.  The main
        # panels may not have requested every one yet, so calculate a missing
        # map here instead of assuming it is already cached.
        if index not in density_cache:
            density_cache[index] = analysis.simulation_density(*loaded[index], canvas)
        daily_resistant.append(density_cache[index][0])
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_FILE,
        cache_version=np.asarray(CACHE_VERSION),
        experimental_total_t0=total_t0,
        experimental_total_t1=total_t1,
        experimental_r0=r0,
        experimental_r1=r1,
        experimental_raw=raw_image,
        experimental_elapsed_days=np.asarray(elapsed_days),
        simulation_total_t0=sim_total_t0,
        simulation_total_t1=sim_total_t1,
        simulation_r0=sim_r0,
        simulation_r1=sim_r1,
        simulation_positions=np.asarray(positions),
        simulation_cell_types=np.asarray(cell_types),
        simulation_elapsed_days=np.asarray(sim_elapsed_days),
        canvas_min_x_um=np.asarray(canvas.min_x_um),
        canvas_min_y_um=np.asarray(canvas.min_y_um),
        canvas_width_px=np.asarray(canvas.width_px),
        canvas_height_px=np.asarray(canvas.height_px),
        daily_resistant=np.stack(daily_resistant),
    )


def load_plot_cache():
    """Return cached sources, or None when a complete rebuild is necessary."""
    if FORCE_REBUILD_CACHE or not CACHE_FILE.exists():
        return None
    try:
        with np.load(CACHE_FILE, allow_pickle=False) as cached:
            if int(cached["cache_version"]) != CACHE_VERSION:
                return None
            canvas = SimpleNamespace(
                min_x_um=float(cached["canvas_min_x_um"]),
                min_y_um=float(cached["canvas_min_y_um"]),
                width_px=int(cached["canvas_width_px"]),
                height_px=int(cached["canvas_height_px"]),
            )
            experimental_source = (
                cached["experimental_total_t0"], cached["experimental_total_t1"],
                cached["experimental_r0"], cached["experimental_r1"],
                cached["experimental_raw"], float(cached["experimental_elapsed_days"]),
            )
            simulation_source = (
                cached["simulation_total_t0"], cached["simulation_total_t1"],
                cached["simulation_r0"], cached["simulation_r1"],
                (cached["simulation_positions"], cached["simulation_cell_types"], canvas),
                float(cached["simulation_elapsed_days"]),
            )
            # The time-course plot only needs resistant-density maps. Rebuild
            # a minimal history whose density cache already contains them.
            files = [SimpleNamespace(day=str(day)) for day in range(15, 22)]
            densities = {
                index: (resistant, np.empty_like(resistant))
                for index, resistant in enumerate(cached["daily_resistant"])
            }
            history = (None, SimpleNamespace(files=files), (), canvas, densities)
            return experimental_source, simulation_source, history
    except (KeyError, OSError, ValueError):
        return None


def save_comparison_cache(sources):
    """Cache the experimental/simulation source pairs used in the final figure."""
    cached = {
        "cache_version": np.asarray(COMPARISON_CACHE_VERSION),
        "raw_display_signature": RAW_DISPLAY_CACHE_SIGNATURE,
    }
    for index, (experiment_source, simulation_source) in enumerate(sources):
        total_t0, total_t1, r0, r1, raw_image, elapsed_days = experiment_source
        prefix = f"experiment_{index}_"
        cached.update({
            prefix + "total_t0": total_t0,
            prefix + "total_t1": total_t1,
            prefix + "r0": r0,
            prefix + "r1": r1,
            prefix + "raw": raw_image,
            prefix + "elapsed_days": np.asarray(elapsed_days),
        })
        total_t0, total_t1, r0, r1, (positions, cell_types, canvas), elapsed_days = simulation_source
        prefix = f"simulation_{index}_"
        cached.update({
            prefix + "total_t0": total_t0,
            prefix + "total_t1": total_t1,
            prefix + "r0": r0,
            prefix + "r1": r1,
            prefix + "positions": positions,
            prefix + "cell_types": cell_types,
            prefix + "elapsed_days": np.asarray(elapsed_days),
            prefix + "canvas_min_x_um": np.asarray(canvas.min_x_um),
            prefix + "canvas_min_y_um": np.asarray(canvas.min_y_um),
            prefix + "canvas_width_px": np.asarray(canvas.width_px),
            prefix + "canvas_height_px": np.asarray(canvas.height_px),
        })
    COMPARISON_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COMPARISON_CACHE_FILE, **cached)


def load_comparison_cache():
    """Load the d1/d12/d26 comparison sources, if their cache is available."""
    if FORCE_REBUILD_CACHE or not COMPARISON_CACHE_FILE.exists():
        return None
    try:
        with np.load(COMPARISON_CACHE_FILE, allow_pickle=False) as cached:
            if int(cached["cache_version"]) != COMPARISON_CACHE_VERSION:
                return None
            if not np.array_equal(
                cached.get("raw_display_signature"), RAW_DISPLAY_CACHE_SIGNATURE,
            ):
                return None
            sources = []
            for index in range(len(AT50_ADDITIONAL_INTERVALS)):
                prefix = f"experiment_{index}_"
                experiment_source = (
                    cached[prefix + "total_t0"], cached[prefix + "total_t1"],
                    cached[prefix + "r0"], cached[prefix + "r1"],
                    cached[prefix + "raw"], float(cached[prefix + "elapsed_days"]),
                )
                prefix = f"simulation_{index}_"
                canvas = SimpleNamespace(
                    min_x_um=float(cached[prefix + "canvas_min_x_um"]),
                    min_y_um=float(cached[prefix + "canvas_min_y_um"]),
                    width_px=int(cached[prefix + "canvas_width_px"]),
                    height_px=int(cached[prefix + "canvas_height_px"]),
                )
                simulation_source = (
                    cached[prefix + "total_t0"], cached[prefix + "total_t1"],
                    cached[prefix + "r0"], cached[prefix + "r1"],
                    (cached[prefix + "positions"], cached[prefix + "cell_types"], canvas),
                    float(cached[prefix + "elapsed_days"]),
                )
                sources.append((experiment_source, simulation_source))
            return sources
    except (KeyError, OSError, ValueError):
        return None


def full_fov_extent(experiment_data):
    height, width = experiment_data[0].shape
    return (0, width * COARSE_PIXEL_SIZE_UM, 0, height * COARSE_PIXEL_SIZE_UM)


def centered_simulation_extent(canvas, density_shape, fov_extent):
    x0, x1, y0, y1 = fov_extent
    # The density grid is ceil-rounded during coarsening. Use its physical
    # extent so raw scatter coordinates and map pixels share one exact scale.
    height_px, width_px = density_shape
    width = width_px * COARSE_PIXEL_SIZE_UM
    height = height_px * COARSE_PIXEL_SIZE_UM
    return (
        x0 + ((x1 - x0) - width) / 2,
        x0 + ((x1 - x0) + width) / 2,
        y0 + ((y1 - y0) - height) / 2,
        y0 + ((y1 - y0) + height) / 2,
    )


def density_centroid(density, extent):
    weights = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0, None)
    total = weights.sum()
    if total <= 0:
        return ((extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2)
    y, x = np.indices(weights.shape)
    x_um = extent[0] + (x + 0.5) * (extent[1] - extent[0]) / weights.shape[1]
    y_um = extent[2] + (y + 0.5) * (extent[3] - extent[2]) / weights.shape[0]
    return (float((x_um * weights).sum() / total), float((y_um * weights).sum() / total))


def magnified_fov(centre, full_extent):
    full_width = full_extent[1] - full_extent[0]
    full_height = full_extent[3] - full_extent[2]
    # Keep each displayed FOV square.  This is immaterial for the square
    # original datasets, but preserves equal panel boxes for rectangular
    # matched DPD canvases such as R20_AT100-R20.
    width = height = min(full_width, full_height) * DISPLAY_FOV_FRACTION
    x0 = np.clip(centre[0] - width / 2, full_extent[0], full_extent[1] - width)
    y0 = np.clip(centre[1] - height / 2, full_extent[2], full_extent[3] - height)
    return (float(x0), float(x0 + width), float(y0), float(y0 + height))


def zoomed_fov(centre, outer_extent, fraction):
    """Return a centred square crop confined to ``outer_extent``."""
    x0, x1, y0, y1 = outer_extent
    width = (x1 - x0) * fraction
    height = (y1 - y0) * fraction
    left = np.clip(centre[0] - width / 2, x0, x1 - width)
    bottom = np.clip(centre[1] - height / 2, y0, y1 - height)
    return (float(left), float(left + width), float(bottom), float(bottom + height))


def add_zoom_indicator(source_ax, zoom_ax, zoom_extent, *, colour="#b0b0b0",
                       vertical=False):
    """Mark ``zoom_extent`` in the source panel and connect it to its zoom."""
    x0, x1, y0, y1 = zoom_extent
    source_ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor=colour, linewidth=1.1, zorder=20,
    ))
    # The two short leaders make the otherwise differently coloured second
    # panel unambiguously read as the framed region at higher magnification.
    endpoints = (
        (((x0, y0), (0, 1)), ((x1, y0), (1, 1)))
        if vertical else (((x1, y0), (0, 0)), ((x1, y1), (0, 1)))
    )
    for source_point, zoom_point in endpoints:
        source_ax.add_artist(ConnectionPatch(
            xyA=source_point, coordsA=source_ax.transData,
            xyB=zoom_point, coordsB=zoom_ax.transAxes,
            color=colour, linewidth=0.8, alpha=1,
            clip_on=False, zorder=19,
        ))


def shared_colour_limits(*datasets):
    totals = np.concatenate([data[0].ravel() for data in datasets])
    changes = np.concatenate([data[1].ravel() for data in datasets])
    finite_change_magnitudes = np.abs(changes[np.isfinite(changes) & (changes != 0)])
    return (
        max(np.nanmax(totals), 1e-8),
        max(np.nanpercentile(finite_change_magnitudes, CHANGE_DISPLAY_PERCENTILE), 1e-8),
    )


def add_scale_bar(ax, fov_extent, *, colour, compact=False, right=False):
    x0, x1, y0, y1 = fov_extent
    width = x1 - x0
    height = y1 - y0
    bar_x = x1 - 0.08 * width - SCALE_BAR_UM if right else x0 + 0.08 * width
    bar_y = y0 + 0.08 * height
    ax.plot([bar_x, bar_x + SCALE_BAR_UM], [bar_y, bar_y], color=colour,
            lw=1.1 if compact else 2.0,
            solid_capstyle="butt", zorder=10)


def add_resistant_rim(ax, density_change, image_extent, *, colour="white"):
    """Outline all connected regions of valid resistant-density change."""
    # Use the same numerical-zero tolerance as the displayed change map.
    # This excludes imperceptible density-kernel tails from the clone rim.
    valid = np.isfinite(density_change) & ~np.isclose(density_change, 0)
    labels, count = ndimage.label(valid)
    if count:
        component_sizes = np.bincount(labels.ravel())
        valid = (
            (labels > 0)
            & (component_sizes[labels] >= MIN_RIM_COMPONENT_PIXELS)
        )
    if not np.any(valid):
        return
    # Place the line outside the valid region, rather than through its outer
    # data pixels. One coarse-grid pixel is 17.52 µm here.
    outer_edge = ndimage.binary_dilation(
        valid, iterations=RIM_OUTWARD_OFFSET_PIXELS,
    )
    ax.contour(
        outer_edge.astype(float),
        levels=[0.5],
        colors=colour,
        linestyles="--",
        linewidths=0.5,
        origin="lower",
        extent=image_extent,
        zorder=8,
    )


def draw(name, data, path, *, fov_extent, image_extent, density_vmax, change_vmax,
         change_label, relative_change=False):
    total, change, image, resistant_density, boundary_change = data
    # 204 pt wide (85% of the original 240 pt Illustrator placement width).
    # Typography remains at the established Figure 3 point sizes.
    fig = plt.figure(figsize=(204 / 72, 2.34 * 0.85))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 1.0, 0.055), wspace=0.02, hspace=0.10,
        left=0.03, right=0.81, bottom=0.04, top=0.99,
    )
    raw_full = fig.add_subplot(grid[0, 0])
    total_ax = fig.add_subplot(grid[0, 1])
    density_zoom = fig.add_subplot(grid[1, 0])
    change_ax = fig.add_subplot(grid[1, 1])
    total_cbar_ax = fig.add_subplot(grid[0, 2])
    change_cbar_ax = fig.add_subplot(grid[1, 2])
    # The simulation canvas is smaller than the shared comparison FOV.  Make
    # the surrounding area look like each panel's zero/background value.
    raw_full.set_facecolor("black")
    total_ax.set_facecolor(plt.get_cmap("inferno")(0.0))
    density_zoom.set_facecolor(plt.get_cmap("inferno")(0.0))
    change_ax.set_facecolor("white")
    if isinstance(image, tuple):
        positions, cell_types, canvas = image
        # Use the same physical origin as the density grid. The coarsened map
        # may have a small ceil-rounded pad at its far edge, which must not
        # rescale raw HDF5 coordinates.
        x = image_extent[0] + positions[:, 0] - canvas.min_x_um
        y = image_extent[2] + positions[:, 1] - canvas.min_y_um
        raw_full.scatter(x[cell_types == 0], y[cell_types == 0], s=.05, alpha=1, c=FIGURE_COLOR_MAP["blue"], label='Sensitive')
        raw_full.scatter(x[cell_types == 1], y[cell_types == 1], s=.05, alpha=1, c=FIGURE_COLOR_MAP["yellow"], label='Resistant')
    else:
        raw_full.imshow(image, origin='lower', extent=image_extent)
    raw_full.set_xticks([]); raw_full.set_yticks([])
    density_zoom.set_xticks([]); density_zoom.set_yticks([])
    change_cmap = plt.get_cmap("coolwarm").copy()
    change_cmap.set_bad("white")
    signed_change = np.ma.masked_where(np.isclose(change, 0), change)
    relative_norm = (
        TwoSlopeNorm(
            vmin=RELATIVE_CHANGE_MIN_PER_DAY,
            vcenter=0.0,
            vmax=RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY,
        )
        if relative_change else None
    )
    specs=[
        (
            total * DENSITY_DISPLAY_SCALE,
            "Local density",
            "inferno",
            0,
            density_vmax * DENSITY_DISPLAY_SCALE,
            LOCAL_DENSITY_LABEL,
        ),
        (
            signed_change,
            change_label,
            change_cmap,
            -change_vmax,
            change_vmax,
            RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL if relative_change
            else "Resistant density\nchange [cells µm$^{-2}$ day$^{-1}$]",
        ),
    ]
    # The lower-left panel is a zoom of the total density, on exactly the
    # same colour scale as the full total-density panel.
    density_zoom.imshow(
        total * DENSITY_DISPLAY_SCALE, cmap="inferno", origin="lower",
        extent=image_extent, vmin=0,
        vmax=density_vmax * DENSITY_DISPLAY_SCALE,
    )
    colourbars = []
    for index, (ax, cbar_ax, (arr, title, cmap, vmin, vmax_, colorbar_label)) in enumerate(
        zip((total_ax, change_ax), (total_cbar_ax, change_cbar_ax), specs)
    ):
        imshow_args = dict(cmap=cmap, origin='lower', extent=image_extent)
        if index == 1 and relative_norm is not None:
            imshow_args['norm'] = relative_norm
        else:
            imshow_args.update(vmin=vmin, vmax=vmax_)
        im=ax.imshow(arr, **imshow_args)
        ax.set_xticks([]); ax.set_yticks([])
        colorbar = fig.colorbar(im, cax=cbar_ax)
        if index == 1 and relative_change:
            colorbar.set_ticks([-1.0, 0.0, 2.5, 5.0])
        colorbar.set_label(colorbar_label, fontsize=7, labelpad=3)
        colorbar.ax.tick_params(labelsize=6)
        colourbars.append((ax, colorbar))
    change_fov = zoomed_fov(
        density_centroid(resistant_density, image_extent), fov_extent,
        CHANGE_PANEL_ZOOM_FRACTION,
    )
    for ax in (raw_full, total_ax):
        ax.set(xlim=fov_extent[:2], ylim=fov_extent[2:], aspect='equal')
    for ax in (density_zoom, change_ax):
        ax.set(xlim=change_fov[:2], ylim=change_fov[2:], aspect='equal')
    add_zoom_indicator(raw_full, density_zoom, change_fov)
    for ax in (raw_full, total_ax, density_zoom):
        add_resistant_rim(ax, boundary_change, image_extent)
    add_resistant_rim(change_ax, boundary_change, image_extent, colour="#222222")
    add_scale_bar(raw_full, fov_extent, colour="white", compact=True)
    add_scale_bar(total_ax, fov_extent, colour="white", compact=True)
    add_scale_bar(density_zoom, change_fov, colour="white", compact=True)
    add_scale_bar(change_ax, change_fov, colour="#222222", compact=True)
    # Equal-aspect axes leave some unused width in their GridSpec cells. Shift
    # the intact right column left by just that excess so the *visible* gaps
    # are equal in the exported figure, while every image box stays identical.
    fig.canvas.draw()
    vertical_gap = raw_full.get_position().y0 - density_zoom.get_position().y1
    horizontal_gap = vertical_gap * (fig.get_figheight() / fig.get_figwidth())
    for left_ax, right_ax in ((raw_full, total_ax), (density_zoom, change_ax)):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        right_ax.set_position((
            left_box.x1 + horizontal_gap, right_box.y0,
            right_box.width, right_box.height,
        ))
    # Each colourbar follows the exact frame height of its paired map.
    for ax, colorbar in colourbars:
        panel_box = ax.get_position()
        bar_box = colorbar.ax.get_position()
        colorbar.ax.set_position((
            panel_box.x1 + 0.015, panel_box.y0, bar_box.width, panel_box.height,
        ))
    # Do not use a tight bounding box: it would change the 240 pt page width.
    # Preserve more detail in the raster raw-image and density-map layers.
    fig.savefig(path, dpi=600, transparent=False); plt.close(fig)


def _draw_at50_two_timepoint_comparison_stacked(
    timepoint_panels, path, *, density_vmax, source_names, show_colourbars,
):
    """Draw modalities vertically, with all requested days adjacent per source."""
    timepoint_count = len(timepoint_panels)
    if timepoint_count < 1:
        raise ValueError("At least one comparison timepoint is required")
    has_colourbars = show_colourbars
    # For figures with bars, reserve the first column for them.  This keeps
    # the combined panel arrangement unchanged and puts a standalone bar left
    # of its simulation panels.
    colourbar_column = 0 if has_colourbars else None
    source_column_offset = int(has_colourbars)
    source_columns = {
        name: source_column_offset + index * timepoint_count
        for index, name in enumerate(source_names)
    }
    width_ratios = (
        ([0.10] if has_colourbars else [])
        + [1] * (len(source_names) * timepoint_count)
    )
    grid_left, grid_right = 0.18, 0.945
    column_count = len(width_ratios)
    # Keep the reflowed panels at the compact physical size used before the
    # layout change, with equal physical grid gaps in both directions.
    panel_inches = 0.85
    panel_ratio_total = sum(width_ratios)
    # Preserve square 0.85-inch maps while using equal, slightly roomier
    # physical gaps in both directions.  The broad left margin prevents the
    # left-hand colourbar labels from being cropped.
    grid_gap = 0.06
    figure_width = panel_inches * (panel_ratio_total + grid_gap * (column_count - 1)) / (grid_right - grid_left)
    figure_height = panel_inches * (3 + 2 * grid_gap) / 0.945
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        3, column_count, width_ratios=width_ratios, height_ratios=(1, 1, 1),
        wspace=grid_gap, hspace=grid_gap, left=grid_left, right=grid_right,
        bottom=0.035, top=0.98,
    )
    change_cmap = plt.get_cmap("coolwarm").copy()
    change_cmap.set_bad("white")
    relative_norm = TwoSlopeNorm(vmin=RELATIVE_CHANGE_MIN_PER_DAY, vcenter=0.0,
                                 vmax=RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY)
    density_im = change_im = None
    for source_name in source_names:
        base_column = source_columns[source_name]
        for time_index in range(timepoint_count):
            panel = timepoint_panels[time_index][source_name]
            total, change, image, resistant_density, boundary_change = panel["data"]
            fov_extent, image_extent = panel["fov"], panel["extent"]
            raw_image_extent = panel["raw_extent"]
            raw_ax = fig.add_subplot(grid[0, base_column + time_index])
            density_ax = fig.add_subplot(grid[1, base_column + time_index])
            change_ax = fig.add_subplot(grid[2, base_column + time_index])
            raw_ax.set_facecolor("black")
            density_ax.set_facecolor(plt.get_cmap("inferno")(0.0))
            change_ax.set_facecolor("white")
            if isinstance(image, tuple):
                positions, cell_types, canvas = image
                x = image_extent[0] + positions[:, 0] - canvas.min_x_um
                y = image_extent[2] + positions[:, 1] - canvas.min_y_um
                raw_ax.scatter(x[cell_types == 0], y[cell_types == 0], s=.05,
                               alpha=1, c=FIGURE_COLOR_MAP["blue"])
                raw_ax.scatter(x[cell_types == 1], y[cell_types == 1], s=.05,
                               alpha=1, c=FIGURE_COLOR_MAP["yellow"])
            else:
                raw_ax.imshow(image, origin="lower", extent=raw_image_extent)
            zoom_fov = zoomed_fov(
                density_centroid(resistant_density, image_extent), fov_extent,
                COMPARISON_ZOOM_FRACTION,
            )
            density_im = density_ax.imshow(
                total * DENSITY_DISPLAY_SCALE, cmap="inferno", origin="lower",
                extent=image_extent, vmin=0,
                vmax=(density_vmax[source_name] if isinstance(density_vmax, dict) else density_vmax)
                * DENSITY_DISPLAY_SCALE,
            )
            display_change = remove_single_pixel_change_outliers(change)
            change_im = change_ax.imshow(
                np.ma.masked_where(np.isclose(display_change, 0), display_change), cmap=change_cmap,
                norm=relative_norm, origin="lower", extent=image_extent,
            )
            raw_ax.set(xlim=fov_extent[:2], ylim=fov_extent[2:], aspect="equal")
            for ax in (density_ax, change_ax):
                ax.set(xlim=zoom_fov[:2], ylim=zoom_fov[2:], aspect="equal")
            add_zoom_indicator(raw_ax, density_ax, zoom_fov, vertical=True)
            for ax in (raw_ax, density_ax, change_ax):
                add_resistant_rim(
                    ax, boundary_change, image_extent,
                    colour=RESISTANT_OUTLINE_COLOUR,
                )
            if time_index == 0:  # Day 1, the first requested timepoint.
                add_scale_bar(raw_ax, fov_extent, colour="white", compact=True, right=True)
                add_scale_bar(density_ax, zoom_fov, colour="white", compact=True, right=True)
                add_scale_bar(change_ax, zoom_fov, colour="#222222", compact=True, right=True)
            for ax in (raw_ax, density_ax, change_ax):
                ax.set_xticks([]); ax.set_yticks([])
    if has_colourbars:
        density_cbar = fig.colorbar(density_im, cax=fig.add_subplot(grid[1, colourbar_column]))
        density_cbar.set_label(DENSITY_COLORBAR_LABEL,
                               fontsize=7, labelpad=8, rotation=90)
        density_cbar.ax.yaxis.set_ticks_position("left")
        density_cbar.ax.yaxis.set_label_position("left")
        density_cbar.ax.yaxis.set_label_coords(-2.7, 0.5)
        density_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        density_cbar.ax.text(
            1.0, 1.02, r"$\times 10^{-3}$", transform=density_cbar.ax.transAxes,
            ha="right", va="bottom", fontsize=6, clip_on=False,
        )
        density_cbar.ax.tick_params(labelsize=6)
        # Move the two tick labels that face the neighbouring colourbar into
        # their own bars, rather than increasing the shared panel spacing.
        density_cbar.ax.get_yticklabels()[0].set_verticalalignment("bottom")
        change_cbar = fig.colorbar(change_im, cax=fig.add_subplot(grid[2, colourbar_column]))
        change_cbar.set_ticks([-1.0, 0.0, 2.5, 5.0])
        change_cbar.set_label(RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL,
                              fontsize=7, labelpad=8, rotation=90,
                              color=RATE_LABEL_COLOUR)
        change_cbar.ax.yaxis.set_ticks_position("left")
        change_cbar.ax.yaxis.set_label_position("left")
        change_cbar.ax.yaxis.set_label_coords(-2.7, 0.5)
        change_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        change_cbar.ax.tick_params(labelsize=6, colors="black", labelcolor="black")
        change_cbar.ax.get_yticklabels()[-1].set_verticalalignment("top")
    fig.savefig(path, dpi=600, transparent=False)
    plt.close(fig)


def draw_at50_two_timepoint_comparison(
    timepoint_panels, path, *, density_vmax,
    source_names=("simulation", "experiment"),
    show_colourbars=True,
):
    """Draw the requested-day density maps for one or both matched data sources."""
    source_names = tuple(source_names)
    if not source_names or any(
        name not in {"simulation", "experiment"} for name in source_names
    ):
        raise ValueError(f"Unexpected comparison sources: {source_names}")
    return _draw_at50_two_timepoint_comparison_stacked(
        timepoint_panels, path, density_vmax=density_vmax,
        source_names=source_names, show_colourbars=show_colourbars,
    )
    # Make the 3x4 layout modestly smaller while retaining the established
    # point-size typography.
    # Extra canvas above accommodates the two-line horizontal-bar labels;
    # the GridSpec bounds below retain the existing physical panel size.
    combined = len(source_names) == 2
    # Retain the combined figure's column width and give a single-source page
    # enough height for width-constrained square panels.  This keeps every map
    # at the same physical size in the combined and separate exports.  The
    # experimental export retains the colourbar row as blank space so its map
    # geometry exactly matches the simulation export.
    figure_height = 5.59 if combined else 3.022
    fig = plt.figure(figsize=(7.2 * 0.70 * 0.85, figure_height))
    height_ratios = [0.13]
    row_sources = []
    next_row = 1
    for source_index, source_name in enumerate(source_names):
        height_ratios.extend((1, 1))
        row_sources.extend(((next_row, 0, source_name), (next_row + 1, 1, source_name)))
        next_row += 2
        if source_index < len(source_names) - 1:
            height_ratios.append(0.18)
            next_row += 1
    grid = fig.add_gridspec(
        len(height_ratios), 3, width_ratios=(1, 1, 1),
        height_ratios=height_ratios,
        # GridSpec fractions are relative to a cell's width/height.  At this
        # portrait figure aspect, 0.05 in each direction made the horizontal
        # physical gap wider; 0.021 yields the same ~3.2 pt frame gap as y.
        wspace=0.021, hspace=0.10,
        # The image axes are square and height-constrained.  Narrowing the
        # grid (rather than only wspace) matches the physical x/y frame gaps.
        # With the side colorbars removed, this narrower centered grid gives
        # the same physical horizontal gap as the ~2.5 pt vertical gap.
        left=0.082, right=0.885,
        bottom=0.018 if combined else 0.035,
        top=0.918 if combined else 0.845,
    )
    # Shared horizontal bars sit above their respective map columns.
    if show_colourbars:
        density_cbar_ax = fig.add_subplot(grid[0, 1])
        change_cbar_ax = fig.add_subplot(grid[0, 2])
    change_cmap = plt.get_cmap("coolwarm").copy()
    change_cmap.set_bad("white")
    relative_norm = TwoSlopeNorm(
        vmin=RELATIVE_CHANGE_MIN_PER_DAY,
        vcenter=0.0,
        vmax=RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY,
    )
    density_im = change_im = None

    for row, time_index, source_name in row_sources:
        panel = timepoint_panels[time_index]
        data = panel[source_name]["data"]
        fov_extent = panel[source_name]["fov"]
        image_extent = panel[source_name]["extent"]
        raw_image_extent = panel[source_name]["raw_extent"]
        total, change, image, resistant_density, boundary_change = data
        raw_ax = fig.add_subplot(grid[row, 0])
        density_ax = fig.add_subplot(grid[row, 1])
        change_ax = fig.add_subplot(grid[row, 2])
        raw_ax.set_facecolor("black")
        density_ax.set_facecolor(plt.get_cmap("inferno")(0.0))
        change_ax.set_facecolor("white")

        if isinstance(image, tuple):
            positions, cell_types, canvas = image
            x = image_extent[0] + positions[:, 0] - canvas.min_x_um
            y = image_extent[2] + positions[:, 1] - canvas.min_y_um
            raw_ax.scatter(x[cell_types == 0], y[cell_types == 0], s=.05,
                           alpha=1, c=FIGURE_COLOR_MAP["blue"])
            raw_ax.scatter(x[cell_types == 1], y[cell_types == 1], s=.05,
                           alpha=1, c=FIGURE_COLOR_MAP["yellow"])
        else:
            raw_ax.imshow(image, origin="lower", extent=raw_image_extent)

        zoom_fov = zoomed_fov(
            density_centroid(resistant_density, image_extent), fov_extent,
            COMPARISON_ZOOM_FRACTION,
        )
        density_im = density_ax.imshow(
            total * DENSITY_DISPLAY_SCALE, cmap="inferno", origin="lower",
            extent=image_extent, vmin=0,
            vmax=density_vmax * DENSITY_DISPLAY_SCALE,
        )
        change_im = change_ax.imshow(
            np.ma.masked_where(np.isclose(change, 0), change),
            cmap=change_cmap, norm=relative_norm, origin="lower",
            extent=image_extent,
        )
        raw_ax.set(xlim=fov_extent[:2], ylim=fov_extent[2:], aspect="equal")
        for ax in (density_ax, change_ax):
            ax.set(xlim=zoom_fov[:2], ylim=zoom_fov[2:], aspect="equal")
        add_zoom_indicator(raw_ax, density_ax, zoom_fov)
        for ax in (raw_ax, density_ax):
            add_resistant_rim(ax, boundary_change, image_extent)
        add_resistant_rim(change_ax, boundary_change, image_extent, colour="#222222")
        add_scale_bar(raw_ax, fov_extent, colour="white", compact=True)
        add_scale_bar(density_ax, zoom_fov, colour="white", compact=True)
        add_scale_bar(change_ax, zoom_fov, colour="#222222", compact=True)
        for ax in (raw_ax, density_ax, change_ax):
            ax.set_xticks([])
            ax.set_yticks([])

    if not show_colourbars:
        fig.savefig(path, dpi=600, transparent=False)
        plt.close(fig)
        return

    density_cbar = fig.colorbar(density_im, cax=density_cbar_ax, orientation="horizontal")
    density_cbar.set_label(
        r"$\rho_\mathrm{tot}$" "\n" r"[cells µm$^{-2}$]", fontsize=7, labelpad=2,
    )
    density_cbar.ax.xaxis.set_label_position("top")
    density_cbar.ax.xaxis.set_ticks_position("top")
    density_cbar.ax.tick_params(labelsize=6, pad=1)
    change_cbar = fig.colorbar(change_im, cax=change_cbar_ax, orientation="horizontal")
    change_cbar.set_ticks([-1.0, 0.0, 2.5, 5.0])
    change_cbar.set_label(
        RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL, fontsize=7, labelpad=2,
    )
    change_cbar.ax.xaxis.set_label_position("top")
    change_cbar.ax.xaxis.set_ticks_position("top")
    change_cbar.ax.tick_params(labelsize=6, pad=1)
    # The square image panels are narrower than their GridSpec cells.  Match
    # each bar to the actual panel frame, and make it deliberately slim.
    fig.canvas.draw()
    for panel_ax, colorbar in ((density_ax, density_cbar), (change_ax, change_cbar)):
        panel_box = panel_ax.get_position()
        bar_box = colorbar.ax.get_position()
        bar_height = bar_box.height * 0.62
        colorbar.ax.set_position((
            panel_box.x0, bar_box.y0, panel_box.width, bar_height,
        ))
    fig.canvas.draw()
    # Use the regular centered tick alignment for the leftmost −1 label.
    change_tick_labels = change_cbar.ax.get_xticklabels()
    if change_tick_labels:
        change_tick_labels[0].set_horizontalalignment("center")
    fig.savefig(path, dpi=600, transparent=False)
    plt.close(fig)


def overview_raw_display_image(image: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
    """Reduce very large RGB display assets before Matplotlib copies them.

    This affects only the raw-fluorescence preview in the all-timepoint
    overview; its physical extent is retained and no density data are changed.
    """
    stride = max(1, int(np.ceil(max(image.shape[:2]) / max_dimension)))
    return image[::stride, ::stride]


def draw_at50_all_experimental_timepoints_stacked(
    path: Path, *, simulation_matches_experimental_intervals: bool = False,
    relative_change: bool = True,
) -> None:
    """Draw every experimental timepoint with experiment above simulation.

    Density and density-change maps are always drawn. Raw-image panels are
    drawn only when the matching microscopy TIFF or supplied simulation PNG is
    available, so missing display assets never discard the quantitative maps.
    When requested, each simulation change uses the same start/end acquisition
    days as its experimental counterpart rather than a one-day interval.
    """
    maps = EXPERIMENTAL_DENSITY_RUN / PLATE / WELL / "density_maps"
    prefix = f"{WELL}_"
    suffix = "_total_density_epanechnikov.tif"
    timepoints = sorted(
        int(candidate)
        for image_path in maps.glob(f"{WELL}_*_total_density_epanechnikov.tif")
        if (candidate := image_path.name.removeprefix(prefix).removesuffix(suffix)).isdigit()
    )
    if not timepoints:
        raise FileNotFoundError(f"No experimental total-density maps found in {maps}")

    simulation_history = load_simulation_history("AT50")
    panels = []
    for index, day in enumerate(timepoints):
        next_day = timepoints[index + 1] if index + 1 < len(timepoints) else day
        day_tag = f"{day:02d}"
        next_day_tag = f"{next_day:02d}"
        total_t0 = tiff.imread(analysis.base.density_map_path(maps, WELL, day_tag)).astype(float)
        resistant_t0 = tiff.imread(
            analysis.base.resistant_density_map_path(maps, WELL, day_tag)
        ).astype(float)
        if next_day == day:
            total_t1, resistant_t1, elapsed_days = total_t0, resistant_t0, 1.0
        else:
            total_t1 = tiff.imread(
                analysis.base.density_map_path(maps, WELL, next_day_tag)
            ).astype(float)
            resistant_t1 = tiff.imread(
                analysis.base.resistant_density_map_path(maps, WELL, next_day_tag)
            ).astype(float)
            elapsed_days = float(next_day - day)
        experiment_data = panel_data(
            (total_t0, total_t1, resistant_t0, resistant_t1, None, elapsed_days),
            relative=relative_change,
        )
        try:
            experiment_raw_path = raw_experiment_path(day_tag)
        except FileNotFoundError:
            pass
        else:
            experiment_image = raw_fluorescence_composite(
                experiment_raw_path,
                (total_t1.shape[0] * COARSENING_FACTOR,
                 total_t1.shape[1] * COARSENING_FACTOR),
            )
            experiment_data = (*experiment_data[:2], experiment_image, *experiment_data[3:])

        extent = full_fov_extent(experiment_data)
        experiment_fov = magnified_fov(
            density_centroid(experiment_data[0], extent), extent,
        )
        simulation_end_day = (
            next_day
            if simulation_matches_experimental_intervals and next_day != day
            else day + 1
        )
        try:
            simulation_source = orient_simulation_source(
                simulation(
                    simulation_history, tp0=day_tag,
                    tp1=f"{simulation_end_day:02d}",
                )
            )
        except StopIteration:
            simulation_data = None
            simulation_extent = extent
            simulation_fov = experiment_fov
            simulation_raw_extent = extent
        else:
            simulation_data = panel_data(simulation_source, relative=relative_change)
            simulation_image_path = SIMULATION_PNG_DIRECTORY / f"day_{day}_sim.png"
            if simulation_image_path.exists():
                simulation_data = (
                    *simulation_data[:2], simulation_png(str(day)), *simulation_data[3:]
                )
            else:
                simulation_data = (
                    *simulation_data[:2], None, *simulation_data[3:]
                )
            simulation_extent = centered_simulation_extent(
                simulation_data[2][2] if isinstance(simulation_data[2], tuple)
                else simulation_source[4][2],
                simulation_data[0].shape,
                extent,
            )
            simulation_fov = magnified_fov(
                density_centroid(simulation_data[0], simulation_extent), extent,
            )
            simulation_raw_extent = (
                simulation_png_extent(
                    simulation_source, simulation_data[2], simulation_extent,
                ) if simulation_image_path.exists() else simulation_extent
            )

        panels.append({
            "experiment": {
                "data": experiment_data,
                "extent": extent,
                "raw_extent": extent,
                "fov": experiment_fov,
            },
            "simulation": {
                "data": simulation_data,
                "extent": simulation_extent,
                "raw_extent": simulation_raw_extent,
                "fov": simulation_fov,
            },
        })

    density_vmax, change_vmax = shared_colour_limits(
        *(source["data"]
          for panel in panels for source in panel.values()
          if source["data"] is not None)
    )
    timepoint_count = len(panels)
    grid_gap = 0.06
    panel_inches = 0.85
    grid_left, grid_right = 0.12, 0.985
    width_ratios = [0.10] + [1] * timepoint_count
    figure_width = panel_inches * (
        sum(width_ratios) + grid_gap * (len(width_ratios) - 1)
    ) / (grid_right - grid_left)
    figure_height = panel_inches * (6 + grid_gap * 5) / 0.965
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        6, len(width_ratios), width_ratios=width_ratios,
        wspace=grid_gap, hspace=grid_gap, left=grid_left, right=grid_right,
        bottom=0.02, top=0.985,
    )
    change_cmap = plt.get_cmap("coolwarm").copy()
    change_cmap.set_bad("white")
    change_norm = TwoSlopeNorm(
        vmin=RELATIVE_CHANGE_MIN_PER_DAY if relative_change else -change_vmax,
        vcenter=0.0,
        vmax=RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY if relative_change else change_vmax,
    )
    density_im = change_im = None
    for source_index, source_name in enumerate(("experiment", "simulation")):
        row_start = 3 * source_index
        for time_index, panel in enumerate(panels):
            data = panel[source_name]
            if data["data"] is None:
                for row in range(3):
                    unavailable_ax = fig.add_subplot(grid[row_start + row, time_index + 1])
                    unavailable_ax.set_facecolor("white")
                    unavailable_ax.set_xticks([])
                    unavailable_ax.set_yticks([])
                    for spine in unavailable_ax.spines.values():
                        spine.set_visible(False)
                continue
            total, change, image, resistant_density, boundary_change = data["data"]
            image_extent, fov_extent = data["extent"], data["fov"]
            raw_ax = fig.add_subplot(grid[row_start, time_index + 1])
            density_ax = fig.add_subplot(grid[row_start + 1, time_index + 1])
            change_ax = fig.add_subplot(grid[row_start + 2, time_index + 1])
            if source_index == 0:
                raw_ax.set_title(f"Day {timepoints[time_index]}", fontsize=7, pad=2)
            raw_ax.set_facecolor("black")
            density_ax.set_facecolor(plt.get_cmap("inferno")(0.0))
            change_ax.set_facecolor("white")
            has_raw_image = image is not None
            if has_raw_image:
                raw_ax.imshow(
                    overview_raw_display_image(image), origin="lower",
                    extent=data["raw_extent"],
                )
                raw_ax.set(xlim=fov_extent[:2], ylim=fov_extent[2:], aspect="equal")
            else:
                raw_ax.set_facecolor("white")
                for spine in raw_ax.spines.values():
                    spine.set_visible(False)
            zoom_fov = zoomed_fov(
                density_centroid(resistant_density, image_extent), fov_extent,
                COMPARISON_ZOOM_FRACTION,
            )
            density_im = density_ax.imshow(
                total * DENSITY_DISPLAY_SCALE, cmap="inferno", origin="lower",
                extent=image_extent, vmin=0, vmax=density_vmax * DENSITY_DISPLAY_SCALE,
            )
            display_change = remove_single_pixel_change_outliers(change)
            change_im = change_ax.imshow(
                np.ma.masked_where(np.isclose(display_change, 0), display_change),
                cmap=change_cmap, norm=change_norm, origin="lower", extent=image_extent,
            )
            for axis in (density_ax, change_ax):
                axis.set(xlim=zoom_fov[:2], ylim=zoom_fov[2:], aspect="equal")
            for axis in (density_ax, change_ax):
                add_resistant_rim(
                    axis, boundary_change, image_extent,
                    colour=RESISTANT_OUTLINE_COLOUR,
                )
            if has_raw_image:
                add_resistant_rim(
                    raw_ax, boundary_change, image_extent,
                    colour=RESISTANT_OUTLINE_COLOUR,
                )
            for axis in (raw_ax, density_ax, change_ax):
                axis.set_xticks([])
                axis.set_yticks([])
            if time_index == 0:
                if has_raw_image:
                    add_scale_bar(raw_ax, fov_extent, colour="white", compact=True, right=True)
                add_scale_bar(density_ax, zoom_fov, colour="white", compact=True, right=True)
                add_scale_bar(change_ax, zoom_fov, colour="#222222", compact=True, right=True)

    density_cbar = fig.colorbar(density_im, cax=fig.add_subplot(grid[4, 0]))
    density_cbar.set_label(DENSITY_COLORBAR_LABEL, fontsize=7, labelpad=8, rotation=90)
    density_cbar.ax.yaxis.set_ticks_position("left")
    density_cbar.ax.yaxis.set_label_position("left")
    density_cbar.ax.yaxis.set_label_coords(-2.7, 0.5)
    density_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    density_cbar.ax.text(0.0, 1.02, r"$\times 10^{-3}$", transform=density_cbar.ax.transAxes,
                         ha="left", va="bottom", fontsize=6, clip_on=False)
    density_cbar.ax.tick_params(labelsize=6)
    change_cbar = fig.colorbar(change_im, cax=fig.add_subplot(grid[5, 0]))
    if relative_change:
        change_cbar.set_ticks([-1.0, 0.0, 2.5, 5.0])
    change_cbar.set_label(
                          RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL if relative_change
                          else r"$\Delta\rho_\mathrm{res}/\Delta t$ [cells µm$^{-2}$ day$^{-1}$]",
                          fontsize=7, labelpad=8, rotation=90, color=RATE_LABEL_COLOUR)
    change_cbar.ax.yaxis.set_ticks_position("left")
    change_cbar.ax.yaxis.set_label_position("left")
    change_cbar.ax.yaxis.set_label_coords(-2.7, 0.5)
    change_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    change_cbar.ax.tick_params(labelsize=6, colors="black", labelcolor="black")
    # This helper is also useful when called directly, so do not rely on the
    # module entry point to have prepared its output directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, transparent=False)
    plt.close(fig)


if __name__ == '__main__':
    # Use the same Arial-based Figure 3 typography as the master curves so
    # axis and colourbar labels have matching character spacing.
    ft.apply_nature_style(
        text_pt=7, tick_pt=6, legend_pt=5.6,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    # The d17/d19 cache and its outputs are legacy auxiliaries.  Do not
    # initialise or write them during the standard d12/d26 figure build.
    if SAVE_AUXILIARY_SHARED_FIGURES:
        cached_sources = load_plot_cache()
        if cached_sources is None:
            experimental_source = experimental()
            simulation_history = load_simulation_history()
            simulation_source = simulation(simulation_history)
            rebuilt_cache = True
        else:
            experimental_source, simulation_source, simulation_history = cached_sources
            rebuilt_cache = False
        simulation_source_for_cache = simulation_source
        simulation_source = orient_simulation_source(simulation_source)
        absolute_experiment = panel_data(experimental_source, relative=False)
        absolute_simulation = panel_data(simulation_source, relative=False)
        relative_experiment = panel_data(experimental_source, relative=True)
        relative_simulation = panel_data(simulation_source, relative=True)
        shared_density_vmax, shared_absolute_change_vmax = shared_colour_limits(
            absolute_experiment, absolute_simulation,
        )
        full_extent = full_fov_extent(absolute_experiment)
        simulation_extent = centered_simulation_extent(
            absolute_simulation[2][2], absolute_simulation[0].shape, full_extent,
        )
        experiment_fov = magnified_fov(
            density_centroid(absolute_experiment[0], full_extent), full_extent,
        )
        simulation_fov = magnified_fov(
            density_centroid(absolute_simulation[0], simulation_extent), full_extent,
        )
        for experiment_data, simulation_data, experiment_output, simulation_output, change_label, relative_change in (
            (absolute_experiment, absolute_simulation,
             OUT/f'D5_AT50_experiment_density_maps_{OUTPUT_DAY_TAG}_160um_16x.pdf',
             OUT/f'D5_AT50_simulation_density_maps_{OUTPUT_DAY_TAG}_160um_16x.pdf',
             'Absolute resistant density change per day', False),
            (relative_experiment, relative_simulation,
             OUT/RELATIVE_EXPERIMENT_PDF, OUT/RELATIVE_SIMULATION_PDF,
              r'$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$', True),
        ):
            draw('Experiment', experiment_data, experiment_output,
                 fov_extent=experiment_fov, image_extent=full_extent,
                 density_vmax=shared_density_vmax, change_vmax=shared_absolute_change_vmax,
                 change_label=change_label, relative_change=relative_change)
            draw('Associated simulation (run 18)', simulation_data, simulation_output,
                 fov_extent=simulation_fov, image_extent=simulation_extent,
                 density_vmax=shared_density_vmax, change_vmax=shared_absolute_change_vmax,
                 change_label=change_label, relative_change=relative_change)
        plot_simulation_change_timecourse(OUT / SIMULATION_CHANGE_TIMECOURSE_PDF, simulation_history)
        if rebuilt_cache:
            save_plot_cache(experimental_source, simulation_source_for_cache, simulation_history)

    # Additional AT50 starting days.  The microscope was acquired every
    # 2–3 days, so experiment uses the next recorded acquisition while the
    # simulation retains the requested daily interval.
    comparison_sources = load_comparison_cache()
    if comparison_sources is None:
        alternate_at50_history = load_simulation_history("AT50")
        comparison_sources = [
            (
                experimental(tp0=exp_start, tp1=exp_end),
                simulation(alternate_at50_history, tp0=sim_start, tp1=sim_end),
            )
            for exp_start, exp_end, sim_start, sim_end in AT50_ADDITIONAL_INTERVALS
        ]
        save_comparison_cache(comparison_sources)

    alternate_at50_panels = []
    for ((exp_start, exp_end, sim_start, sim_end),
         (alternate_experiment_source, alternate_simulation_source)) in zip(
             AT50_ADDITIONAL_INTERVALS, comparison_sources,
         ):
        alternate_simulation_source = orient_simulation_source(alternate_simulation_source)
        alternate_absolute_experiment = panel_data(alternate_experiment_source, relative=False)
        alternate_relative_experiment = panel_data(alternate_experiment_source, relative=True)
        alternate_absolute_simulation = simulation_panel_data(
            alternate_simulation_source, relative=False, day=sim_start,
        )
        alternate_relative_simulation = simulation_panel_data(
            alternate_simulation_source, relative=True, day=sim_start,
        )
        alternate_extent = full_fov_extent(alternate_absolute_experiment)
        alternate_experiment_fov = magnified_fov(
            density_centroid(alternate_absolute_experiment[0], alternate_extent),
            alternate_extent,
        )
        alternate_simulation_extent = centered_simulation_extent(
            alternate_absolute_simulation[2][2],
            alternate_absolute_simulation[0].shape,
            alternate_extent,
        )
        alternate_simulation_fov = magnified_fov(
            density_centroid(alternate_absolute_simulation[0], alternate_simulation_extent),
            alternate_extent,
        )
        alternate_simulation_png_extent = simulation_png_extent(
            alternate_simulation_source, alternate_relative_simulation[2],
            alternate_simulation_extent,
        )
        interval_tag = f"d{int(exp_start):02d}_to_d{int(exp_end):02d}"
        sim_interval_tag = f"d{int(sim_start):02d}_to_d{int(sim_end):02d}"
        alternate_at50_panels.append((
            alternate_absolute_experiment, alternate_relative_experiment,
            alternate_absolute_simulation, alternate_relative_simulation,
            alternate_extent, alternate_experiment_fov,
            alternate_simulation_extent, alternate_simulation_fov,
            alternate_simulation_png_extent,
            interval_tag, sim_interval_tag,
        ))

    # Use identical density and absolute-change scales across the complete
    # d1/d12/d26 AT50 comparison.
    alternate_density_vmax, alternate_change_vmax = shared_colour_limits(
        *(panel for values in alternate_at50_panels for panel in (values[0], values[2]))
    )
    experiment_density_vmax, _ = shared_colour_limits(
        *(values[0] for values in alternate_at50_panels)
    )
    simulation_density_vmax, _ = shared_colour_limits(
        *(values[2] for values in alternate_at50_panels)
    )
    for (alternate_absolute_experiment, alternate_relative_experiment,
         alternate_absolute_simulation, alternate_relative_simulation,
         alternate_extent, alternate_experiment_fov,
         alternate_simulation_extent, alternate_simulation_fov,
         _alternate_simulation_png_extent,
         interval_tag, sim_interval_tag) in (
        alternate_at50_panels if SAVE_AUXILIARY_SHARED_FIGURES else ()
    ):
        for (experiment_data, simulation_data, experiment_output, simulation_output,
             change_label, relative_change) in (
            (alternate_absolute_experiment, alternate_absolute_simulation,
             OUT / f"D5_AT50_experiment_density_maps_{interval_tag}_160um_16x.pdf",
             OUT / f"D5_AT50_simulation_density_maps_{sim_interval_tag}_160um_16x.pdf",
             "Absolute resistant density change per day", False),
            (alternate_relative_experiment, alternate_relative_simulation,
             OUT / f"D5_AT50_experiment_relative_resistant_density_change_{interval_tag}_160um_16x.pdf",
             OUT / f"D5_AT50_simulation_relative_resistant_density_change_{sim_interval_tag}_160um_16x.pdf",
              r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$", True),
        ):
            draw('D5 AT50 experiment', experiment_data, experiment_output,
                 fov_extent=alternate_experiment_fov, image_extent=alternate_extent,
                 density_vmax=alternate_density_vmax, change_vmax=alternate_change_vmax,
                 change_label=change_label, relative_change=relative_change)
            draw('D5 AT50 simulation (run 18)', simulation_data, simulation_output,
                 fov_extent=alternate_simulation_fov,
                 image_extent=alternate_simulation_extent,
                 density_vmax=alternate_density_vmax, change_vmax=alternate_change_vmax,
                 change_label=change_label, relative_change=relative_change)

    comparison_panels = []
    for (absolute_experiment, relative_experiment,
         absolute_simulation, relative_simulation,
         extent, experiment_fov, simulation_extent, simulation_fov,
         panel_simulation_png_extent,
         _interval_tag, _sim_interval_tag) in alternate_at50_panels:
        comparison_panels.append({
            "simulation": {
                "data": relative_simulation,
                "extent": simulation_extent,
                "raw_extent": panel_simulation_png_extent,
                "fov": simulation_fov,
            },
            "experiment": {
                "data": relative_experiment,
                "extent": extent,
                "raw_extent": extent,
                "fov": experiment_fov,
            },
        })
    draw_at50_two_timepoint_comparison(
        comparison_panels,
        OUT / (
            "D5_AT50_simulation_experiment_d01_d12_d26_relative_density_maps_"
            f"160um_16x_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=alternate_density_vmax,
    )
    draw_at50_two_timepoint_comparison(
        comparison_panels,
        OUT / (
            "D5_AT50_experiment_d01_d12_d26_relative_density_maps_"
            f"160um_16x_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=alternate_density_vmax,
        source_names=("experiment",),
        show_colourbars=False,
    )
    draw_at50_two_timepoint_comparison(
        comparison_panels,
        OUT / (
            "D5_AT50_simulation_run18_d01_d12_d26_relative_density_maps_"
            f"160um_16x_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=alternate_density_vmax,
        source_names=("simulation",),
        show_colourbars=True,
    )
    # Keep simulation-maximum-normalized counterparts beside the main source
    # plots, with the same source-specific colourbar arrangement.
    draw_at50_two_timepoint_comparison(
        comparison_panels,
        OUT / (
            "D5_AT50_experiment_d01_d12_d26_relative_density_maps_"
            f"simulation_max_shared_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=simulation_density_vmax,
        source_names=("experiment",),
        show_colourbars=False,
    )
    draw_at50_two_timepoint_comparison(
        comparison_panels,
        OUT / (
            "D5_AT50_simulation_run18_d01_d12_d26_relative_density_maps_"
            f"simulation_max_shared_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=simulation_density_vmax,
        source_names=("simulation",),
        show_colourbars=True,
    )
    # Alternative separate-source density scales, kept apart from the final
    # shared-scale exports for direct comparison.
    alternative_scales_out = OUT / "alternative_density_scales"
    alternative_scales_out.mkdir(parents=True, exist_ok=True)

    # Day 19 is the first day of this interval, just like d01, d12, and d26
    # in the standard figures.  It replaces the latest d26/d29 panel only in
    # this extra pair of exports.
    day19_experiment_source = experimental(tp0="19", tp1="22")
    day19_simulation_source = orient_simulation_source(
        simulation(load_simulation_history("AT50"), tp0="19", tp1="20")
    )
    day19_absolute_experiment = panel_data(day19_experiment_source, relative=False)
    day19_relative_experiment = panel_data(day19_experiment_source, relative=True)
    day19_has_simulation_png = (
        SIMULATION_PNG_DIRECTORY / "day_19_sim.png"
    ).exists()
    if day19_has_simulation_png:
        day19_absolute_simulation = simulation_panel_data(
            day19_simulation_source, relative=False, day="19",
        )
        day19_relative_simulation = simulation_panel_data(
            day19_simulation_source, relative=True, day="19",
        )
    else:
        # Fall back to the simulation positions/type data if the optional
        # display PNG has not been provided yet.
        day19_absolute_simulation = panel_data(day19_simulation_source, relative=False)
        day19_relative_simulation = panel_data(day19_simulation_source, relative=True)
    day19_extent = full_fov_extent(day19_absolute_experiment)
    day19_experiment_fov = magnified_fov(
        density_centroid(day19_absolute_experiment[0], day19_extent), day19_extent,
    )
    day19_simulation_extent = centered_simulation_extent(
        day19_absolute_simulation[2][2], day19_absolute_simulation[0].shape,
        day19_extent,
    )
    day19_simulation_fov = magnified_fov(
        density_centroid(day19_absolute_simulation[0], day19_simulation_extent),
        day19_extent,
    )
    day19_comparison_panels = comparison_panels[:2] + [{
        "simulation": {
            "data": day19_relative_simulation,
            "extent": day19_simulation_extent,
            "raw_extent": (
                simulation_png_extent(
                    day19_simulation_source, day19_relative_simulation[2],
                    day19_simulation_extent,
                )
                if day19_has_simulation_png else day19_simulation_extent
            ),
            "fov": day19_simulation_fov,
        },
        "experiment": {
            "data": day19_relative_experiment,
            "extent": day19_extent,
            "raw_extent": day19_extent,
            "fov": day19_experiment_fov,
        },
    }]
    day19_density_vmax, _ = shared_colour_limits(
        *(source["data"]
          for panel in day19_comparison_panels
          for source in panel.values())
    )
    for source_name, filename_stem in (
        ("experiment", "D5_AT50_experiment"),
        ("simulation", "D5_AT50_simulation_run18"),
    ):
        draw_at50_two_timepoint_comparison(
            day19_comparison_panels,
            alternative_scales_out / (
                f"{filename_stem}_d01_d12_d19_relative_density_maps_"
                f"shared_max_scalebar{int(SCALE_BAR_UM)}um.pdf"
            ),
            density_vmax=day19_density_vmax,
            source_names=(source_name,),
            show_colourbars=True,
        )

    for scale_tag, density_limits in (
        ("source_max", {
            "experiment": experiment_density_vmax,
            "simulation": simulation_density_vmax,
        }),
        # Use the simulation maximum as one shared limit, including for the
        # experimental counterpart, to compare both sources directly.
        ("simulation_max_shared", {
            "experiment": simulation_density_vmax,
            "simulation": simulation_density_vmax,
        }),
        ("poisson_cap_2p385", {
            "experiment": POISSON_SAMPLING_DENSITY_CAP,
            "simulation": POISSON_SAMPLING_DENSITY_CAP,
        }),
    ):
        for source_name, filename_stem in (
            ("experiment", "D5_AT50_experiment"),
            ("simulation", "D5_AT50_simulation_run18"),
        ):
            draw_at50_two_timepoint_comparison(
                comparison_panels,
                alternative_scales_out / (
                    f"{filename_stem}_d01_d12_d26_relative_density_maps_"
                    f"{scale_tag}_scalebar{int(SCALE_BAR_UM)}um.pdf"
                ),
                density_vmax=density_limits,
                source_names=(source_name,),
                show_colourbars=True,
            )
    if PLOT_ALL_EXPERIMENTAL_TIMEPOINTS:
        ALL_TIMEPOINTS_OUT.mkdir(parents=True, exist_ok=True)
        draw_at50_all_experimental_timepoints_stacked(
            ALL_TIMEPOINTS_OUT / "D5_AT50_all_timepoints_"
            f"relative_density_maps_160um_16x_scalebar{int(SCALE_BAR_UM)}um.pdf"
        )
        draw_at50_all_experimental_timepoints_stacked(
            ALL_TIMEPOINTS_OUT / "D5_AT50_all_timepoints_"
            f"sim_matched_intervals_160um_16x_"
            f"scalebar{int(SCALE_BAR_UM)}um.pdf",
            simulation_matches_experimental_intervals=True,
        )
        draw_at50_all_experimental_timepoints_stacked(
            ALL_TIMEPOINTS_OUT / "D5_AT50_all_timepoints_"
            f"absolute_density_change_160um_16x_scalebar{int(SCALE_BAR_UM)}um.pdf",
            relative_change=False,
        )
        draw_at50_all_experimental_timepoints_stacked(
            ALL_TIMEPOINTS_OUT / "D5_AT50_all_timepoints_"
            f"absolute_density_change_sim_matched_intervals_"
            f"160um_16x_scalebar{int(SCALE_BAR_UM)}um.pdf",
            simulation_matches_experimental_intervals=True,
            relative_change=False,
        )
