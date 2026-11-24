#!/usr/bin/env python3
"""Plot D5/run_18 HDF5 maps using the Figure 3 master-curve parameters.

Default Run exports absolute change and total density at t0 for 120 and 160 um.
Historical TIFF/coarse-grid helpers remain available for archived callers only.
"""

from __future__ import annotations
import math
import os, sys
from pathlib import Path
from types import SimpleNamespace
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import tifffile as tiff


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [os.fspath(ROOT)]
from analysis.figure_3 import density_common as analysis
from analysis.figure_3.HDF5_analysis import constants as hdf5_constants
from analysis.figure_3.HDF5_analysis import compute_flux_vs_density as hdf5_sim
from analysis.figure_3.HDF5_analysis import compute_flux_vs_density_experiment as hdf5_exp
from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import density_canvas_shape
from shared import figtools as ft
from shared.strategy_colors import FIGURE_COLOR_MAP
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, figure_plot_dir

OUT = figure_plot_dir("Figure_3", "density_maps", create=False)
MAP_GROWTH_MODE = "absolute"
ADDITIONAL_OUT = OUT / "additional_plots"
ALL_TIMEPOINTS_OUT = ADDITIONAL_OUT / "all_timepoints"
# Both 120-um and 160-um HDF5 pairs are written directly to density_maps.
# Historical auxiliary exports go below additional_plots.
# Set to True to also run the historical TIFF-based auxiliary exports.
SAVE_AUXILIARY_SHARED_FIGURES = False
# Set to False to skip the wide experimental/simulation all-timepoint overview
# while still producing the regular Figure 3 density-map exports.
PLOT_ALL_EXPERIMENTAL_TIMEPOINTS = False
# Also export the absolute-change pair rebuilt directly from the raw HDF5
# point clouds with the HDF5 master-curve kernel and support rules.
PLOT_HDF5_ABSOLUTE = True
# Brief rim-threshold token for output filenames: the rim requires more than
# one cell's worth of local resistant density (ONE_CELL_DENSITY_THRESHOLD
# below, the one-cell peak of the 160-um kernel), evaluated at the interval's
# first endpoint only.  See RIM_DENSITY_THRESHOLD and resistant_rim_maps().
RIM_THRESHOLD_TAG = "rim1cell"
PLATE, WELL, TP0, TP1 = "Plate_4", "D5_AT50-D5", "17", "19"
OUTPUT_DAY_TAG = f"d{int(TP0):02d}"
RELATIVE_EXPERIMENT_PDF = (
    f"D5_AT50_exp_relative_resistant_density_change_{OUTPUT_DAY_TAG}_{RIM_THRESHOLD_TAG}.pdf"
)
RELATIVE_SIMULATION_PDF = (
    f"D5_AT50_sim_relative_resistant_density_change_{OUTPUT_DAY_TAG}_{RIM_THRESHOLD_TAG}.pdf"
)
SIMULATION_CHANGE_TIMECOURSE_PDF = "D5_AT50_sim_resistant_density_change_timecourse.pdf"
SIM_TP0, SIM_TP1 = "17", "18"
AT50_ADDITIONAL_INTERVALS = (
    # (experimental start, experimental end, simulation start, simulation end)
    ("01", "03", "01", "02"),
    ("12", "15", "12", "13"),
    ("24", "26", "24", "25"),
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
    "d5_at50_density_maps_exp01_03_exp12_15_exp24_26_"
    "sim01_02_sim12_13_sim24_25_cache.npz"
)
COMPARISON_CACHE_VERSION = 14
HDF5_CACHE_FILE = CACHE_DIRECTORY / (
    "d5_at50_hdf5_160um_minres2p25_minneighbors14_exp01_03_exp12_15_exp24_26_"
    "sim01_02_sim12_13_sim24_25_cache.npz"
)
HDF5_CACHE_VERSION = 7
HDF5_VARIANT_CACHE_FILE = CACHE_DIRECTORY / (
    "d5_at50_hdf5_120um_minres2p25_minneighbors14_"
    "exp01_03_exp12_15_exp24_26_sim01_02_sim12_13_sim24_25_cache.npz"
)
HDF5_VARIANT_CACHE_VERSION = 6
HDF5_FILTER_SERIES_CACHE_FILE = CACHE_DIRECTORY / (
    "d5_at50_hdf5_120um_minneighbors14_"
    "all_experimental_timepoints_through_d24_cache.npz"
)
HDF5_FILTER_SERIES_CACHE_VERSION = 3
HDF5_BASE_KERNEL_RADIUS_UM = 160.0
HDF5_BASE_MIN_NEIGHBORS = hdf5_constants.prolif_min_neighbors
HDF5_BASE_MIN_RESISTANT_NUMERATOR = hdf5_constants.MIN_RESISTANT_NUMERATOR
# Both radii use the same support rules; only the kernel radius changes.
HDF5_VARIANT_MIN_NEIGHBORS = hdf5_constants.prolif_min_neighbors
HDF5_VARIANT_MIN_RESISTANT_NUMERATOR = hdf5_constants.MIN_RESISTANT_NUMERATOR
HDF5_VARIANT_KERNEL_RADIUS_UM = 120.0
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
# Use the same, slightly wider threshold-region zoom for experimental and
# simulation panels so the new resistant-density outline is not cropped.
# Fallback only: the comparison figure now fits its zoom to the widest rim
# it actually draws (see fitted_rim_zoom_fraction).  This value is used when
# no panel has a rim at all.
RIM_ZOOM_FRACTION = 0.45
# Breathing room around the widest rim, so the dashed line is not flush
# against the panel edge.
RIM_ZOOM_MARGIN = 1.06
# Share of each colourbar cell left empty above the bar, so the power of
# ten has somewhere to sit that is not the bar above it.
COLOURBAR_HEADROOM = 0.17
COLOURBAR_MULTIPLIER_PAD = 0.04
# Labelling only the integer ticks narrows the tick column, which is what
# lets the rotated label sit this close to the bar.
COLOURBAR_LABEL_X = -1.35
COLOURBAR_LABEL_LINESPACING = 1.2
# Gap between a colourbar's tick labels and its rotated label, in points.
# The label is placed by measurement rather than a fixed offset, because a
# two-line label is nearly twice as wide as a one-line one.
COLOURBAR_LABEL_GAP_PT = 1.5
# The experiment and simulation comparison figures are written by separate
# calls, so the crop is measured over every source present in the panel data,
# not just the one being drawn.  The two figures then agree, and the widest
# rim of either source still fits.
RIM_OUTWARD_OFFSET_PIXELS = 1
MIN_RIM_COMPONENT_PIXELS = 2
RESISTANT_OUTLINE_COLOUR = "#DAA520"
RATE_LABEL_COLOUR = "black"
# Maximum density contribution of one isolated cell at the centre of a
# 160-µm Epanechnikov kernel.  Keep this in physical density units.
ONE_CELL_DENSITY_THRESHOLD = 2.0 / (np.pi * analysis.KERNEL_UM ** 2)
# The rim encloses regions holding more than one resistant cell, so an
# isolated cell is excluded.  The multiple sits just above 1 rather than at
# exactly 1: ONE_CELL_DENSITY_THRESHOLD *is* the one-cell kernel peak, so a
# lone cell would otherwise land on the strict ">" boundary, where float32 FFT
# rounding (~6e-8 relative, and dependent on canvas size) decides the outcome.
# Two cells within one kernel radius reach ~1.99x, far above this cut.
RIM_MIN_CELL_MULTIPLE = 1.05
RIM_DENSITY_THRESHOLD = RIM_MIN_CELL_MULTIPLE * ONE_CELL_DENSITY_THRESHOLD
RELATIVE_DENSITY_FLOOR = 1e-10
RELATIVE_CHANGE_MIN_PER_DAY = -1.05
RELATIVE_CHANGE_MAX_PER_DAY = 100.0
RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY = 1.0
# Display absolute density changes in units of 10^-4 cells/µm²/day, matching
# the explicit multiplier treatment used by the density colourbar.
ABSOLUTE_CHANGE_DISPLAY_SCALE = 1.0e4
# Fixed linear range of the absolute-change colourbar, in displayed 1e-4 units.
# Values outside the range saturate at the colour ends.
ABSOLUTE_CHANGE_DISPLAY_MIN = -2.0
ABSOLUTE_CHANGE_DISPLAY_MAX = 2.0
LOCAL_DENSITY_LABEL = r"Local density [cells µm$^{-2}$]"
DENSITY_COLORBAR_LABEL = r"$\rho_\mathrm{tot}$ [cells µm$^{-2}$]"
RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL = r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$ [day$^{-1}$]"
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
    total_t0 = tiff.imread(analysis.density_map_path(maps, well, tp0)).astype(float)
    total_t1 = tiff.imread(analysis.density_map_path(maps, well, tp1)).astype(float)
    r0 = tiff.imread(analysis.resistant_density_map_path(maps, well, tp0)).astype(float)
    r1 = tiff.imread(analysis.resistant_density_map_path(maps, well, tp1)).astype(float)
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
    args = analysis.simulation_args(
        os.fspath(analysis.DEFAULT_SIM_ROOT), strategy, "run_18",
    )
    plan = analysis.sim.discover_run_plans(args)[0]
    loaded = [analysis.sim.load_filtered_cells(x.path, args) for x in plan.files]
    canvas = analysis.canvas_from_loaded(loaded, float(analysis.KERNEL_UM))
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


def _hdf5_kernel_density_field(positions, cell_types, grid_points, radius, grid_shape, *, min_neighbors):
    total, _, resistant, valid = hdf5_sim._kernel_density_field(
        positions, cell_types, grid_points, radius, 2.0/(np.pi*radius**2),
        grid_shape, min_neighbors)
    return total, resistant, valid


def _hdf5_experimental_hdf5_path():
    return Path(hdf5_exp.discover_runs()["AT50"]["AT50/D5"])


def _hdf5_experimental_positions(h5_file, day_key):
    susceptible = np.asarray(h5_file["sus"][day_key]["block0_values"])[:, [1, 0]]
    resistant = np.asarray(h5_file["res"][day_key]["block0_values"])[:, [1, 0]]
    positions = np.vstack((susceptible, resistant))
    cell_types = np.concatenate((
        np.full(len(susceptible), hdf5_constants.CELL_TYPE_SUSCEPTIBLE),
        np.full(len(resistant), hdf5_constants.CELL_TYPE_RESISTANT),
    ))
    return positions, cell_types


def _hdf5_experimental_grid(h5_file, *, kernel_radius_um):
    points, shape = hdf5_exp._fixed_well_grid(h5_file, kernel_radius_um=kernel_radius_um)
    spacing = hdf5_exp._SIM_GRID_SPACING
    extent_px = (points[:,0].min()-spacing/2, points[:,0].max()+spacing/2,
                 points[:,1].min()-spacing/2, points[:,1].max()+spacing/2)
    return points, shape, tuple(value*hdf5_constants.px_to_micron for value in extent_px)


def _hdf5_experimental_fields(requested_days, *, kernel_radius_um, min_neighbors):
    path = _hdf5_experimental_hdf5_path()
    _, rule = hdf5_exp._sample_and_rule(path)
    cutoff = None if rule is ... else float(rule)
    with h5py.File(path, "r") as handle:
        points, shape, extent = _hdf5_experimental_grid(handle, kernel_radius_um=kernel_radius_um)
        days = {day: key for day, key in hdf5_exp._sorted_days(handle)
                if cutoff is None or day <= cutoff}
        fields = {}
        for day in sorted(set(requested_days)):
            total, _, resistant, valid = hdf5_exp.local_densities(
                handle, days[day], points, shape, kernel_radius_um=kernel_radius_um,
                min_neighbors=min_neighbors)
            fields[day] = total, resistant, valid
    return fields, extent


def _hdf5_read_simulation_snapshot(path):
    with h5py.File(path, "r") as h5_file:
        data = h5_file["data"]

        def read(name):
            for key in (name, f'"{name}"'):
                if key in data:
                    return np.asarray(data[key])
            raise KeyError(f"{name!r} not found in {path}")

        active = read("cell_flags").reshape(-1) == hdf5_constants.CELL_FLAG_ACTIVE
        positions = read("cell_nucleus")[active, :2]
        cell_types = read("cell_type")[active].reshape(-1)

    scaling = hdf5_constants.exp_scaling
    centre = np.asarray(hdf5_constants.filter_robust_center, dtype=float) / scaling
    radial = (
        np.linalg.norm(positions - centre, axis=1)
        <= hdf5_constants.filter_outlier_boundary / scaling
    )
    positions = positions[radial]
    cell_types = cell_types[radial]
    tree = cKDTree(positions)
    neighbours = tree.query_ball_point(
        positions, hdf5_constants.filter_neighbour_radius / scaling,
    )
    supported = np.asarray([
        len(indices) >= hdf5_constants.filter_min_neighbours
        for indices in neighbours
    ])
    return positions[supported], cell_types[supported]


def _hdf5_simulation_grid():
    points = hdf5_sim.GRID_PTS
    shape = hdf5_sim.N_GRID
    spacing = np.diff(np.unique(points[:,0]))[0]
    pixel = hdf5_constants.exp_scaling*hdf5_constants.px_to_micron
    extent = (points[:,0].min()-spacing/2, points[:,0].max()+spacing/2,
              points[:,1].min()-spacing/2, points[:,1].max()+spacing/2)
    return points, shape, tuple(value*pixel for value in extent)


def _hdf5_simulation_fields(requested_days, *, kernel_radius_um, min_neighbors):
    run = "run_18"
    if not hdf5_constants.is_non_artificial_run(run):
        raise ValueError("The master curve excludes simulation runs 0 through 9")
    run_dir = hdf5_sim.protocol_outputs_dir("AT50", run)
    selected = hdf5_sim._selected_timesteps(run_dir)
    by_day = {ts/hdf5_sim.time_dump+hdf5_sim.escape_core.sim.DEFAULT_DAY_OFFSET: ts
              for ts in selected}
    _, _, extent = _hdf5_simulation_grid()
    fields = {}
    for day in sorted(set(requested_days)):
        # An exact native day is required: no silent substitution of snapshots.
        total, _, resistant, valid = hdf5_sim.local_densities(
            run_dir, by_day[day], kernel_radius_um=kernel_radius_um,
            min_neighbors=min_neighbors)
        fields[day] = total, resistant, valid
    return fields, extent


def _hdf5_interval_source(
    field0, field1, elapsed_days, *, kernel_radius_um,
    min_resistant_numerator, simulation=False,
):
    total0, resistant0, valid0 = field0
    total1, resistant1, valid1 = field1
    physical_pixel = hdf5_constants.px_to_micron
    if simulation:
        physical_pixel *= hdf5_constants.exp_scaling
    radius = kernel_radius_um / physical_pixel
    minimum_resistant_density = min_resistant_numerator / (np.pi * radius ** 2)
    # This is also the yellow-outline rule. Keep it strictly identical so
    # every outlined pixel is retained for the resistant-density-change maps,
    # and every excluded pixel lies outside the outline.
    support = hdf5_constants.resistant_support(
        total0, resistant0, total1, resistant1, valid0, valid1,
        threshold=minimum_resistant_density)
    maps = (
        # Show local *total* density everywhere. The minimum-neighbour rule
        # defines only which resistant-density-change pixels are retained;
        # using it here created black holes in the density-map display.
        total0 / physical_pixel ** 2,
        total1 / physical_pixel ** 2,
        np.where(support, resistant0 / physical_pixel ** 2, np.nan),
        np.where(support, resistant1 / physical_pixel ** 2, np.nan),
    )
    neighbour_valid = valid0 & valid1
    if simulation:
        maps = tuple(values.T for values in maps)
        neighbour_valid = neighbour_valid.T
    return (*maps, elapsed_days, neighbour_valid)


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


def resistant_rim_maps(resistant_endpoints):
    """Endpoint maps the rim threshold must hold in.

    Only the interval's first endpoint is required.  The experimental and
    simulation panels of one figure span different numbers of days (the
    experiment follows the acquisition schedule), and requiring both endpoints
    penalised the longer interval: a region had to survive 2-3 days in vitro
    against 1 day in silico.  Return ``resistant_endpoints`` unchanged here to
    restore the both-timepoints rule.
    """
    return resistant_endpoints[:1]


def resistant_endpoint_mean(resistant_endpoints):
    """Mean of an interval's endpoint resistant-density maps."""
    r0, r1 = resistant_endpoints
    return 0.5 * (np.asarray(r0, dtype=float) + np.asarray(r1, dtype=float))


def panel_data(source_data, *, relative):
    total_t0, total_t1, r0, r1, image, elapsed_days = source_data
    # Display density at the interval's first endpoint. The second endpoint is
    # used only for calculating resistant-density change.
    total = total_t0
    # Both endpoints are retained so that the rim can require its density
    # threshold at each timepoint of the interval.  Use
    # resistant_endpoint_mean() wherever the former endpoint-average map is
    # wanted, such as the zoom centroid.
    resistant_endpoints = (r0, r1)
    change = resistant_change(r0, r1, elapsed_days, relative=relative)
    # The rim must follow the same change quantity that is visualised: absolute
    # for absolute maps and relative for relative-growth maps.
    return total, change, image, resistant_endpoints, change


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
    total, change, _image, resistant_endpoints, boundary_change = panel_data(
        source_data, relative=relative,
    )
    return total, change, simulation_png(day), resistant_endpoints, boundary_change


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
        r0, _ = density(first_index)
        r1, _ = density(next_index)
        elapsed_days = float(plan.files[next_index].day) - float(plan.files[first_index].day)
        panels.append((
            f"d{first_day}→d{next_day}",
            orient_simulation_map(resistant_change(r0, r1, elapsed_days, relative=False)),
            orient_simulation_map(resistant_change(r0, r1, elapsed_days, relative=True)),
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
    fig.suptitle("AT50 simulation (run 18): resistant-density change time course", y=1.02)
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


def _save_hdf5_cache(
    sources, *, cache_file, cache_version, kernel_radius_um, min_neighbors,
    min_resistant_numerator, intervals=AT50_ADDITIONAL_INTERVALS,
):
    cached = {
        "cache_version": np.asarray(cache_version),
        "analysis_signature": np.asarray((
            kernel_radius_um,
            min_neighbors,
            min_resistant_numerator,
        ), dtype=float),
        "intervals": np.asarray(intervals, dtype=int),
    }
    for index, (experiment, experiment_extent, simulation, simulation_extent) in enumerate(sources):
        for source_name, source, extent in (
            ("experiment", experiment, experiment_extent),
            ("simulation", simulation, simulation_extent),
        ):
            prefix = f"{source_name}_{index}_"
            total0, total1, resistant0, resistant1, elapsed_days, neighbour_valid = source
            cached.update({
                prefix + "total0": total0,
                prefix + "total1": total1,
                prefix + "resistant0": resistant0,
                prefix + "resistant1": resistant1,
                prefix + "elapsed_days": np.asarray(elapsed_days),
                prefix + "neighbour_valid": neighbour_valid,
                prefix + "extent": np.asarray(extent),
            })
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **cached)


def _load_hdf5_cache(
    *, cache_file, cache_version, kernel_radius_um, min_neighbors,
    min_resistant_numerator, intervals=AT50_ADDITIONAL_INTERVALS,
):
    if FORCE_REBUILD_CACHE or not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as cached:
            if int(cached["cache_version"]) != cache_version:
                return None
            expected_signature = np.asarray((
                kernel_radius_um,
                min_neighbors,
                min_resistant_numerator,
            ), dtype=float)
            if not np.array_equal(cached["analysis_signature"], expected_signature):
                return None
            if "intervals" in cached:
                if not np.array_equal(
                    cached["intervals"], np.asarray(intervals, dtype=int),
                ):
                    return None
            elif tuple(intervals) != AT50_ADDITIONAL_INTERVALS:
                # Legacy three-interval caches predate the interval signature
                # and remain valid only for the established three-row figure.
                return None
            sources = []
            for index in range(len(intervals)):
                pair = []
                for source_name in ("experiment", "simulation"):
                    prefix = f"{source_name}_{index}_"
                    source = (
                        cached[prefix + "total0"], cached[prefix + "total1"],
                        cached[prefix + "resistant0"], cached[prefix + "resistant1"],
                        float(cached[prefix + "elapsed_days"]),
                        cached[prefix + "neighbour_valid"].astype(bool),
                    )
                    pair.extend((source, tuple(cached[prefix + "extent"])))
                sources.append(tuple(pair))
            return sources
    except (KeyError, OSError, ValueError):
        return None


def hdf5_interval_sources(
    *, cache_file=HDF5_CACHE_FILE,
    cache_version=HDF5_CACHE_VERSION,
    kernel_radius_um=HDF5_BASE_KERNEL_RADIUS_UM,
    min_neighbors=HDF5_BASE_MIN_NEIGHBORS,
    min_resistant_numerator=HDF5_BASE_MIN_RESISTANT_NUMERATOR,
    intervals=AT50_ADDITIONAL_INTERVALS,
):
    """Return direct-HDF5 experiment/simulation interval pairs."""
    cached = _load_hdf5_cache(
        cache_file=cache_file,
        cache_version=cache_version,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=min_neighbors,
        min_resistant_numerator=min_resistant_numerator,
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
    experimental_fields, experimental_extent = _hdf5_experimental_fields(
        experimental_days,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=min_neighbors,
    )
    simulation_fields, simulation_extent = _hdf5_simulation_fields(
        simulation_days,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=min_neighbors,
    )
    sources = []
    for exp_start, exp_end, sim_start, sim_end in intervals:
        exp_start, exp_end = int(exp_start), int(exp_end)
        sim_start, sim_end = int(sim_start), int(sim_end)
        experiment = _hdf5_interval_source(
            experimental_fields[exp_start], experimental_fields[exp_end],
            exp_end - exp_start,
            kernel_radius_um=kernel_radius_um,
            min_resistant_numerator=min_resistant_numerator,
        )
        simulation = _hdf5_interval_source(
            simulation_fields[sim_start], simulation_fields[sim_end],
            sim_end - sim_start,
            kernel_radius_um=kernel_radius_um,
            min_resistant_numerator=min_resistant_numerator,
            simulation=True,
        )
        sources.append((
            experiment, experimental_extent, simulation, simulation_extent,
        ))
    _save_hdf5_cache(
        sources,
        cache_file=cache_file,
        cache_version=cache_version,
        kernel_radius_um=kernel_radius_um,
        min_neighbors=min_neighbors,
        min_resistant_numerator=min_resistant_numerator,
        intervals=intervals,
    )
    return sources


def hdf5_filter_series_intervals(maximum_day=24):
    """Pair every displayed experimental day with its next acquisition day."""
    with h5py.File(_hdf5_experimental_hdf5_path(), "r") as h5_file:
        experimental_days = sorted(
            int(key.split("_")[1]) for key in h5_file["sus"].keys()
        )
    displayed_days = [day for day in experimental_days if 1 <= day <= maximum_day]
    intervals = []
    for day in displayed_days:
        following_days = [candidate for candidate in experimental_days if candidate > day]
        if not following_days:
            raise ValueError(f"No experimental timepoint follows day {day}")
        intervals.append((day, following_days[0], day, day + 1))
    return tuple(intervals)


def hdf5_absolute_comparison_panels(
    interval_sources, comparison_sources=None, alternate_at50_panels=None,
    *, relative=None,
):
    """Attach the established raw panels and physical extents to HDF5 maps."""
    if relative is None:
        relative = MAP_GROWTH_MODE == "relative"
    if comparison_sources is None:
        first_day = AT50_ADDITIONAL_INTERVALS[0][0]
        # The HDF5 coordinates use the padded well canvas. Round it to the
        # coarse grid exactly as the original density-map TIFFs did.
        shape = density_canvas_shape(
            EXP_RAW_DATA / PLATE, WELL, first_day, _hdf5_experimental_hdf5_path(),
        )
        if shape is not None:
            target_shape = tuple(
                math.ceil(size / COARSENING_FACTOR) * COARSENING_FACTOR for size in shape
            )
        else:
            density_path = analysis.density_map_path(
                EXPERIMENTAL_DENSITY_RUN / PLATE / WELL / "density_maps", WELL, first_day,
            )
            if not density_path.is_file():
                raise FileNotFoundError(f"Padded image canvas missing for {PLATE}/{WELL}")
            with tiff.TiffFile(density_path) as density_file:
                target_shape = tuple(
                    size * COARSENING_FACTOR for size in density_file.series[0].shape[-2:]
                )
        pixel = hdf5_constants.px_to_micron
        experiment_raw_extent = (0., target_shape[1] * pixel, 0., target_shape[0] * pixel)

    panels = []
    datasets = []
    for index, (
        experiment_source, experiment_extent,
        simulation_source, simulation_extent,
    ) in enumerate(interval_sources):
        exp_start, _exp_end, sim_start, _sim_end = AT50_ADDITIONAL_INTERVALS[index]

        if comparison_sources is None:
            raw = raw_fluorescence_composite(raw_experiment_path(exp_start), target_shape)
        else:
            raw = comparison_sources[index][0][4]
            experiment_raw_extent = alternate_at50_panels[index][4]
        experiment_source = (*experiment_source[:4], raw, experiment_source[4])
        experiment_data = panel_data(experiment_source, relative=relative)
        experiment_fov = magnified_fov(
            density_centroid(experiment_data[0], experiment_extent),
            experiment_raw_extent,
        )

        simulation_raw = simulation_png(sim_start)
        simulation_raw_extent = simulation_png_extent(
            None, simulation_raw, simulation_extent,
        )
        map_width = simulation_extent[1] - simulation_extent[0]
        map_height = simulation_extent[3] - simulation_extent[2]
        raw_centre_x = 0.5 * (simulation_raw_extent[0] + simulation_raw_extent[1])
        raw_centre_y = 0.5 * (simulation_raw_extent[2] + simulation_raw_extent[3])
        simulation_extent = (
            raw_centre_x - map_width / 2, raw_centre_x + map_width / 2,
            raw_centre_y - map_height / 2, raw_centre_y + map_height / 2,
        )
        simulation_source = (
            *simulation_source[:4], simulation_raw, simulation_source[4],
        )
        simulation_data = panel_data(simulation_source, relative=relative)
        simulation_centre = density_centroid(
            simulation_data[0], simulation_extent,
        )
        # Display experiment and simulation with exactly the same physical
        # field-of-view dimensions, irrespective of the PNG canvas size.
        fov_width = min(
            experiment_fov[1] - experiment_fov[0],
            simulation_raw_extent[1] - simulation_raw_extent[0],
        )
        fov_height = min(
            experiment_fov[3] - experiment_fov[2],
            simulation_raw_extent[3] - simulation_raw_extent[2],
        )
        simulation_left = np.clip(
            simulation_centre[0] - fov_width / 2,
            simulation_raw_extent[0], simulation_raw_extent[1] - fov_width,
        )
        simulation_bottom = np.clip(
            simulation_centre[1] - fov_height / 2,
            simulation_raw_extent[2], simulation_raw_extent[3] - fov_height,
        )
        simulation_fov = (
            float(simulation_left), float(simulation_left + fov_width),
            float(simulation_bottom), float(simulation_bottom + fov_height),
        )
        datasets.extend((experiment_data, simulation_data))
        panels.append({
            "experiment": {
                "data": experiment_data,
                "support_mask": np.isfinite(experiment_source[2]),
                "extent": experiment_extent,
                "raw_extent": experiment_raw_extent,
                "fov": experiment_fov,
            },
            "simulation": {
                "data": simulation_data,
                "support_mask": np.isfinite(simulation_source[2]),
                "extent": simulation_extent,
                "raw_extent": simulation_raw_extent,
                "fov": simulation_fov,
            },
        })
    return panels, datasets


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


def resistant_density_mask(resistant_density, *, threshold=RIM_DENSITY_THRESHOLD):
    """Return pixels above the resistant-density threshold at every endpoint."""
    candidates = (
        (resistant_density,) if np.ndim(resistant_density) == 2
        else tuple(resistant_density)
    )
    valid = np.ones(np.shape(candidates[0]), dtype=bool)
    for density_map in candidates:
        density_map = np.asarray(density_map, dtype=float)
        valid &= np.isfinite(density_map) & (density_map >= threshold)
    return valid


def panel_support_mask(panel, *, threshold):
    """Keep the native support decision through unit conversion and rendering."""
    if "support_mask" in panel:
        return np.asarray(panel["support_mask"], dtype=bool)
    return resistant_density_mask(resistant_rim_maps(panel["data"][3]), threshold=threshold)


def drawn_rim_mask(rim_valid):
    """Return the region ``add_resistant_rim`` outlines, or None if empty.

    This repeats the component filter and the outward dilation of the rim so
    that a zoom fitted to this mask matches the dashed line on the figure.
    """
    valid = np.asarray(rim_valid, dtype=bool)
    labels, count = ndimage.label(valid)
    if count:
        component_sizes = np.bincount(labels.ravel())
        valid = (labels > 0) & (component_sizes[labels] >= MIN_RIM_COMPONENT_PIXELS)
    if not np.any(valid):
        return None
    return ndimage.binary_dilation(valid, iterations=RIM_OUTWARD_OFFSET_PIXELS)


def mask_physical_extent(mask, image_extent):
    """Bounding box of ``mask`` in the units of ``image_extent``."""
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    x0, x1, y0, y1 = image_extent
    height, width = mask.shape
    pixel_width = (x1 - x0) / width
    pixel_height = (y1 - y0) / height
    return (
        x0 + columns[0] * pixel_width,
        x0 + (columns[-1] + 1) * pixel_width,
        y0 + rows[0] * pixel_height,
        y0 + (rows[-1] + 1) * pixel_height,
    )


def containing_zoom_fraction(centre, bounds, fov_extent):
    """Smallest ``zoomed_fov`` fraction whose crop still contains ``bounds``.

    ``zoomed_fov`` clips its window to the outer extent, so the relationship
    between fraction and crop is not a plain scaling near an edge.  A bisection
    on the real function avoids having to reproduce that clipping.
    """
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        x0, x1, y0, y1 = zoomed_fov(centre, fov_extent, max(middle, 1e-6))
        if x0 <= bounds[0] and bounds[1] <= x1 and y0 <= bounds[2] and bounds[3] <= y1:
            high = middle
        else:
            low = middle
    return high


def fitted_rim_zoom_fraction(timepoint_panels, source_names, *, threshold):
    """One zoom fraction for every panel: the widest rim of any source.

    The fraction is the smallest one that crops no rim at all, plus a margin,
    measured over every source in ``timepoint_panels`` rather than only the
    one being drawn.  The separately written experiment and simulation
    figures therefore share one crop and stay physically comparable, and the
    source with the largest rims -- the experiment -- sets it.  Only the
    panel centres differ.
    """
    available = set()
    for panel_set in timepoint_panels:
        available.update(panel_set)
    reference_names = tuple(available) if available else tuple(source_names)
    required = []
    for panel_set in timepoint_panels:
        for source_name in reference_names:
            panel = panel_set[source_name]
            endpoints = panel["data"][3]
            rim_valid = panel_support_mask(panel, threshold=threshold)
            mask = drawn_rim_mask(rim_valid)
            if mask is None:
                continue
            centre = density_centroid(
                np.where(rim_valid, resistant_endpoint_mean(endpoints), 0.0),
                panel["extent"],
            )
            required.append(containing_zoom_fraction(
                centre,
                mask_physical_extent(mask, panel["extent"]),
                panel["fov"],
            ))
    if not required:
        return RIM_ZOOM_FRACTION
    return float(np.clip(max(required) * RIM_ZOOM_MARGIN, 0.02, 1.0))


def integer_colourbar_ticks(value, _position):
    """Label whole numbers only, leaving the half-step ticks unlabelled."""
    nearest = round(float(value))
    if not np.isclose(value, nearest, atol=1e-9):
        return ""
    return f"{nearest:d}"


def style_colourbar_label(colourbar):
    """Move the rotated label in against the narrowed tick column."""
    colourbar.ax.yaxis.set_label_coords(COLOURBAR_LABEL_X, 0.5)
    colourbar.ax.yaxis.label.set_linespacing(COLOURBAR_LABEL_LINESPACING)


def align_colourbar_label(fig, colourbar, *, gap_pt=None):
    """Shift the rotated label to sit one gap left of its tick labels.

    A fixed offset cannot suit both bars: the two-line change label is nearly
    twice as wide as the one-line density label, and set_label_coords places
    a label by its centre.  Measuring the drawn tick column instead gives both
    bars the same visual gap, as close in as the labels allow.
    """
    gap = COLOURBAR_LABEL_GAP_PT if gap_pt is None else gap_pt
    axes = colourbar.ax
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [
        text.get_window_extent(renderer)
        for text in axes.get_yticklabels()
        if text.get_text()
    ]
    if not ticks:
        return
    label = axes.yaxis.label
    target_right = min(box.x0 for box in ticks) - gap * fig.dpi / 72.0
    shift = target_right - label.get_window_extent(renderer).x1
    width = axes.get_window_extent(renderer).width
    if width <= 0:
        return
    x, _y = label.get_position()
    axes.yaxis.set_label_coords(x + shift / width, 0.5)


def colourbar_axes(fig, cell):
    """Axes for a colourbar, shortened from the top inside its grid cell.

    The stacked bars leave no usable space between them, so the multiplier
    printed above a bar would otherwise sit on the bar above.  Trimming each
    bar by COLOURBAR_HEADROOM of its cell keeps every bar in place and in
    proportion while opening a strip above it for that multiplier.
    """
    axes = fig.add_subplot(cell)
    box = axes.get_position()
    axes.set_position((
        box.x0, box.y0, box.width, box.height * (1.0 - COLOURBAR_HEADROOM),
    ))
    return axes


def colourbar_multiplier(colourbar, exponent, *, colour="black", align="right"):
    """Print the power of ten in the strip above the bar."""
    # Right-aligned on the bar: the text is several times wider than the bar,
    # so anchoring it on the left pushed it over the image panel beside the
    # colourbar column, where black text on a black panel vanished.
    colourbar.ax.text(
        0.0 if align == "left" else 1.0, 1.0 + COLOURBAR_MULTIPLIER_PAD,
        rf"$\times 10^{{{exponent}}}$",
        transform=colourbar.ax.transAxes,
        ha=align, va="bottom", fontsize=6, color=colour, clip_on=False,
        zorder=12,
    )


def add_resistant_rim(
    ax, resistant_density, image_extent, *, threshold=RIM_DENSITY_THRESHOLD,
    colour="white", support_mask=None,
):
    """Outline connected regions above a resistant-density threshold.

    ``resistant_density`` is either one map or a sequence of maps.  When
    several maps are given the threshold must be exceeded in *every* one of
    them, so an interval's rim encloses only the regions that are populated at
    both of its endpoints, instead of regions that reach the threshold on the
    endpoint average alone.
    """
    valid = (resistant_density_mask(resistant_density, threshold=threshold)
             if support_mask is None else np.asarray(support_mask, dtype=bool))
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
    # Keep short dashes in physical points, independent of outline thickness.
    with plt.rc_context({"lines.scale_dashes": False}):
        ax.contour(
            outer_edge.astype(float),
            levels=[0.5],
            colors=colour,
            linestyles=[(0, (3.0, 1.0))],
            linewidths=1.2,
            origin="lower",
            extent=image_extent,
            zorder=8,
        )


def draw(name, data, path, *, fov_extent, image_extent, density_vmax, change_vmax,
         change_label, relative_change=False):
    total, change, image, resistant_endpoints, boundary_change = data
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
        density_centroid(resistant_endpoint_mean(resistant_endpoints), image_extent), fov_extent,
        CHANGE_PANEL_ZOOM_FRACTION,
    )
    for ax in (raw_full, total_ax):
        ax.set(xlim=fov_extent[:2], ylim=fov_extent[2:], aspect='equal')
    for ax in (density_zoom, change_ax):
        ax.set(xlim=change_fov[:2], ylim=change_fov[2:], aspect='equal')
    add_zoom_indicator(raw_full, density_zoom, change_fov)
    for ax in (raw_full, total_ax, density_zoom):
        add_resistant_rim(ax, resistant_rim_maps(resistant_endpoints), image_extent)
    add_resistant_rim(change_ax, resistant_rim_maps(resistant_endpoints), image_extent, colour="#222222")
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
    relative_change=True, change_vmax=None,
    rim_density_threshold=RIM_DENSITY_THRESHOLD,
    full_map_view=False,
):
    """Draw modalities vertically, with all requested days adjacent per source."""
    timepoint_count = len(timepoint_panels)
    if timepoint_count < 1:
        raise ValueError("At least one comparison timepoint is required")
    has_colourbars = show_colourbars
    # Put the shared colourbars after the maps, with descriptive row labels
    # on the left and quantitative colourbar labels on the right.
    colourbar_column = len(source_names) * timepoint_count if has_colourbars else None
    source_columns = {
        name: index * timepoint_count
        for index, name in enumerate(source_names)
    }
    width_ratios = (
        [1] * (len(source_names) * timepoint_count)
        + ([0.10] if has_colourbars else [])
    )
    grid_left, grid_right = (0.12, 0.80) if has_colourbars else (0.18, 0.945)
    column_count = len(width_ratios)
    # Keep the reflowed panels at the compact physical size used before the
    # layout change, with equal physical grid gaps in both directions.
    panel_inches = 0.85
    panel_ratio_total = sum(width_ratios)
    # Preserve square 0.85-inch maps while using equal, slightly roomier
    # physical gaps in both directions. Reserve space on the right for the
    # colourbar ticks and units, and on the left for the row descriptions.
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
    change_display_scale = 1.0 if relative_change else ABSOLUTE_CHANGE_DISPLAY_SCALE
    if relative_change:
        change_norm = TwoSlopeNorm(
            vmin=RELATIVE_CHANGE_MIN_PER_DAY,
            vcenter=0.0,
            vmax=RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY,
        )
    else:
        if change_vmax is None or not np.isfinite(change_vmax) or change_vmax <= 0:
            raise ValueError("A positive change_vmax is required for absolute-change maps")
        # The range is fixed rather than taken from change_vmax; the argument
        # is still required so that callers keep supplying a sane data scale.
        change_norm = Normalize(
            vmin=ABSOLUTE_CHANGE_DISPLAY_MIN,
            vmax=ABSOLUTE_CHANGE_DISPLAY_MAX,
        )
    density_im = change_im = None
    # Fit one zoom to the widest rim in the figure, rather than using a fixed
    # fraction that has to be re-tuned whenever the threshold changes.
    zoom_fraction = fitted_rim_zoom_fraction(
        timepoint_panels, source_names, threshold=rim_density_threshold,
    )
    for source_name in source_names:
        base_column = source_columns[source_name]
        for time_index in range(timepoint_count):
            panel = timepoint_panels[time_index][source_name]
            total, change, image, resistant_endpoints, boundary_change = panel["data"]
            fov_extent, image_extent = panel["fov"], panel["extent"]
            raw_image_extent = panel["raw_extent"]
            raw_ax = fig.add_subplot(grid[0, base_column + time_index])
            density_ax = fig.add_subplot(grid[1, base_column + time_index])
            change_ax = fig.add_subplot(grid[2, base_column + time_index])
            if has_colourbars and base_column == 0 and time_index == 0:
                density_ax.set_ylabel("Local total\ndensity", fontsize=7, labelpad=7)
                change_ax.set_ylabel("Resistant density\nchange", fontsize=7, labelpad=7)
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
            rim_density = resistant_endpoint_mean(resistant_endpoints)
            rim_valid = panel_support_mask(panel, threshold=rim_density_threshold)
            zoom_fov = zoomed_fov(
                density_centroid(np.where(rim_valid, rim_density, 0.0), image_extent),
                fov_extent, zoom_fraction,
            )
            # The direct-HDF5 variants show the complete total-density field.
            # Resistant density change is deliberately restricted to the
            # outlined region, and retains the established close-up view.
            density_view = image_extent if full_map_view else zoom_fov
            change_view = zoom_fov
            density_im = density_ax.imshow(
                total * DENSITY_DISPLAY_SCALE, cmap="inferno", origin="lower",
                extent=image_extent, vmin=0,
                vmax=(density_vmax[source_name] if isinstance(density_vmax, dict) else density_vmax)
                * DENSITY_DISPLAY_SCALE,
            )
            display_change = remove_single_pixel_change_outliers(change)
            change_im = change_ax.imshow(
                np.ma.masked_where(
                    np.isclose(display_change, 0) | ~rim_valid,
                    display_change * change_display_scale,
                ),
                cmap=change_cmap,
                norm=change_norm, origin="lower", extent=image_extent,
            )
            raw_ax.set(xlim=fov_extent[:2], ylim=fov_extent[2:], aspect="equal")
            density_ax.set(
                xlim=density_view[:2], ylim=density_view[2:], aspect="equal",
            )
            change_ax.set(
                xlim=change_view[:2], ylim=change_view[2:], aspect="equal",
            )
            if not full_map_view:
                add_zoom_indicator(raw_ax, density_ax, zoom_fov, vertical=True)
            for ax in (raw_ax, density_ax, change_ax):
                add_resistant_rim(
                    ax, resistant_rim_maps(resistant_endpoints), image_extent,
                    threshold=rim_density_threshold,
                    colour=RESISTANT_OUTLINE_COLOUR,
                    support_mask=rim_valid,
                )
            if time_index == 0:  # Day 1, the first requested timepoint.
                add_scale_bar(raw_ax, fov_extent, colour="white", compact=True, right=True)
                add_scale_bar(density_ax, density_view, colour="white", compact=True, right=True)
                add_scale_bar(change_ax, change_view, colour="#222222", compact=True, right=True)
            for ax in (raw_ax, density_ax, change_ax):
                ax.set_xticks([]); ax.set_yticks([])
    if has_colourbars:
        density_cbar = fig.colorbar(density_im, cax=colourbar_axes(fig, grid[1, colourbar_column]))
        density_cbar.set_label(DENSITY_COLORBAR_LABEL,
                               fontsize=7, labelpad=1, rotation=270,
                               verticalalignment="bottom")
        density_cbar.ax.yaxis.set_ticks_position("right")
        density_cbar.ax.yaxis.set_label_position("right")
        density_cbar.ax.yaxis.label.set_verticalalignment("bottom")
        density_cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(integer_colourbar_ticks)
        )
        colourbar_multiplier(density_cbar, -3, align="left")
        density_cbar.ax.tick_params(labelsize=6)
        # Move the two tick labels that face the neighbouring colourbar into
        # their own bars, rather than increasing the shared panel spacing.
        density_cbar.ax.get_yticklabels()[0].set_verticalalignment("bottom")
        change_cbar = fig.colorbar(change_im, cax=colourbar_axes(fig, grid[2, colourbar_column]))
        if relative_change:
            change_cbar.set_ticks(
                [-1.0, 0.0, RELATIVE_CHANGE_DISPLAY_MAX_PER_DAY]
            )
            change_label = RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL
        else:
            change_cbar.set_ticks([
                ABSOLUTE_CHANGE_DISPLAY_MIN, 0.0, ABSOLUTE_CHANGE_DISPLAY_MAX,
            ])
            # Two lines: on one line this label is taller than its bar
            # and overflowed past the bottom edge of the figure.
            change_label = (
                "$\\Delta\\rho_\\mathrm{res}/\\Delta t$\n"
                "[cells µm$^{-2}$ day$^{-1}$]"
            )
        change_cbar.set_label(
            change_label, fontsize=7, labelpad=1, rotation=270,
            color=RATE_LABEL_COLOUR,
        )
        change_cbar.ax.yaxis.set_ticks_position("right")
        change_cbar.ax.yaxis.set_label_position("right")
        change_cbar.ax.yaxis.label.set_verticalalignment("bottom")
        change_cbar.ax.yaxis.label.set_linespacing(COLOURBAR_LABEL_LINESPACING)
        change_cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(integer_colourbar_ticks)
        )
        if not relative_change:
            colourbar_multiplier(change_cbar, -4, colour=RATE_LABEL_COLOUR, align="left")
        change_cbar.ax.tick_params(labelsize=6, colors="black", labelcolor="black")
        change_cbar.ax.get_yticklabels()[-1].set_verticalalignment("top")
    fig.savefig(path, dpi=600, transparent=False)
    plt.close(fig)


def draw_at50_two_timepoint_comparison(
    timepoint_panels, path, *, density_vmax,
    source_names=("simulation", "experiment"),
    show_colourbars=True, relative_change=True, change_vmax=None,
    rim_density_threshold=RIM_DENSITY_THRESHOLD,
    full_map_view=False,
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
        relative_change=relative_change, change_vmax=change_vmax,
        rim_density_threshold=rim_density_threshold, full_map_view=full_map_view,
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
        total, change, image, resistant_endpoints, boundary_change = data
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
            density_centroid(resistant_endpoint_mean(resistant_endpoints), image_extent), fov_extent,
            COMPARISON_ZOOM_FRACTION,
        )
        map_view = image_extent if full_map_view else zoom_fov
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
            ax.set(xlim=map_view[:2], ylim=map_view[2:], aspect="equal")
        if not full_map_view:
            add_zoom_indicator(raw_ax, density_ax, zoom_fov)
        for ax in (raw_ax, density_ax):
            add_resistant_rim(ax, resistant_rim_maps(resistant_endpoints), image_extent)
        add_resistant_rim(change_ax, resistant_rim_maps(resistant_endpoints), image_extent, colour="#222222")
        add_scale_bar(raw_ax, fov_extent, colour="white", compact=True)
        add_scale_bar(density_ax, map_view, colour="white", compact=True)
        add_scale_bar(change_ax, map_view, colour="#222222", compact=True)
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


def draw_hdf5_density_filter_series(
    interval_sources, path, *, density_vmax, display_days,
):
    """Compare unfiltered and neighbour-filtered densities side by side."""
    columns = (
        ("Experiment\nunfiltered", 0, False),
        ("Experiment\n≥14 neighbours", 0, True),
        ("Simulation\nunfiltered", 2, False),
        ("Simulation\n≥14 neighbours", 2, True),
    )
    # Keep every map large when additional timepoints are present rather than
    # compressing all rows into a fixed-height page.
    figure = plt.figure(figsize=(9.6, 1.9 * len(interval_sources) + 0.9))
    grid = figure.add_gridspec(
        len(interval_sources), len(columns) + 1,
        width_ratios=(*([1.0] * len(columns)), 0.055),
        wspace=0.055, hspace=0.12,
        left=0.075, right=0.91, bottom=0.065, top=0.90,
    )
    density_image = None
    for row, interval in enumerate(interval_sources):
        day = int(display_days[row])
        for column, (title, source_index, apply_filter) in enumerate(columns):
            source = interval[source_index]
            extent = interval[source_index + 1]
            total = np.asarray(source[0], dtype=float)
            # Keep a pixel only when the unweighted cell count meets the
            # 14-neighbour requirement at both interval endpoints.
            retained = source[5]
            display = np.ma.masked_where(~retained, total) if apply_filter else total
            axis = figure.add_subplot(grid[row, column])
            density_image = axis.imshow(
                display * DENSITY_DISPLAY_SCALE, cmap="inferno", origin="lower",
                extent=extent, vmin=0,
                vmax=density_vmax * DENSITY_DISPLAY_SCALE,
            )
            add_resistant_rim(
                axis, (retained.astype(float),), extent, threshold=0.5,
                colour=RESISTANT_OUTLINE_COLOUR,
            )
            axis.set(xticks=[], yticks=[], aspect="equal")
            if row == 0:
                axis.set_title(title, fontsize=10, pad=7)
            if column == 0:
                axis.set_ylabel(f"day {day}", fontsize=10, rotation=90, labelpad=10)
    density_cbar = figure.colorbar(
        density_image, cax=figure.add_subplot(grid[:, -1]),
    )
    density_cbar.set_label(DENSITY_COLORBAR_LABEL, fontsize=10, labelpad=8)
    density_cbar.ax.tick_params(labelsize=9)
    density_cbar.ax.yaxis.set_major_formatter(FuncFormatter(integer_colourbar_ticks))
    colourbar_multiplier(density_cbar, -3)
    figure.savefig(path, dpi=600, transparent=False)
    plt.close(figure)


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
        total_t0 = tiff.imread(analysis.density_map_path(maps, WELL, day_tag)).astype(float)
        resistant_t0 = tiff.imread(
            analysis.resistant_density_map_path(maps, WELL, day_tag)
        ).astype(float)
        if next_day == day:
            total_t1, resistant_t1, elapsed_days = total_t0, resistant_t0, 1.0
        else:
            total_t1 = tiff.imread(
                analysis.density_map_path(maps, WELL, next_day_tag)
            ).astype(float)
            resistant_t1 = tiff.imread(
                analysis.resistant_density_map_path(maps, WELL, next_day_tag)
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
            total, change, image, resistant_endpoints, boundary_change = data["data"]
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
                density_centroid(resistant_endpoint_mean(resistant_endpoints), image_extent), fov_extent,
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
                    axis, resistant_rim_maps(resistant_endpoints), image_extent,
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

    density_cbar = fig.colorbar(density_im, cax=colourbar_axes(fig, grid[4, 0]))
    density_cbar.set_label(DENSITY_COLORBAR_LABEL, fontsize=7, labelpad=8, rotation=90)
    density_cbar.ax.yaxis.set_ticks_position("left")
    density_cbar.ax.yaxis.set_label_position("left")
    style_colourbar_label(density_cbar)
    density_cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(integer_colourbar_ticks)
    )
    colourbar_multiplier(density_cbar, -3)
    density_cbar.ax.tick_params(labelsize=6)
    change_cbar = fig.colorbar(change_im, cax=colourbar_axes(fig, grid[5, 0]))
    if relative_change:
        change_cbar.set_ticks([-1.0, 0.0, 2.5, 5.0])
    change_cbar.set_label(
                          RELATIVE_RESISTANT_DENSITY_CHANGE_LABEL if relative_change
                          else r"$\Delta\rho_\mathrm{res}/\Delta t$ [cells µm$^{-2}$ day$^{-1}$]",
                          fontsize=7, labelpad=8, rotation=90, color=RATE_LABEL_COLOUR)
    change_cbar.ax.yaxis.set_ticks_position("left")
    change_cbar.ax.yaxis.set_label_position("left")
    style_colourbar_label(change_cbar)
    change_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    change_cbar.ax.tick_params(labelsize=6, colors="black", labelcolor="black")
    fig.savefig(path, dpi=600, transparent=False)
    plt.close(fig)


def plot_legacy_auxiliaries():
    OUT.mkdir(parents=True, exist_ok=True)
    ADDITIONAL_OUT.mkdir(parents=True, exist_ok=True)
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
             ADDITIONAL_OUT/f'D5_AT50_exp_density_maps_{OUTPUT_DAY_TAG}_{RIM_THRESHOLD_TAG}.pdf',
             ADDITIONAL_OUT/f'D5_AT50_sim_density_maps_{OUTPUT_DAY_TAG}_{RIM_THRESHOLD_TAG}.pdf',
             'Absolute resistant density change per day', False),
            (relative_experiment, relative_simulation,
             ADDITIONAL_OUT/RELATIVE_EXPERIMENT_PDF,
             ADDITIONAL_OUT/RELATIVE_SIMULATION_PDF,
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
        plot_simulation_change_timecourse(
            ADDITIONAL_OUT / SIMULATION_CHANGE_TIMECOURSE_PDF, simulation_history,
        )
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
             ADDITIONAL_OUT / f"D5_AT50_exp_density_maps_{interval_tag}_{RIM_THRESHOLD_TAG}.pdf",
             ADDITIONAL_OUT / f"D5_AT50_sim_density_maps_{sim_interval_tag}_{RIM_THRESHOLD_TAG}.pdf",
             "Absolute resistant density change per day", False),
            (alternate_relative_experiment, alternate_relative_simulation,
             ADDITIONAL_OUT / f"D5_AT50_exp_relative_resistant_density_change_{interval_tag}_{RIM_THRESHOLD_TAG}.pdf",
             ADDITIONAL_OUT / f"D5_AT50_sim_relative_resistant_density_change_{sim_interval_tag}_{RIM_THRESHOLD_TAG}.pdf",
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

    relative_comparison_panels = []
    absolute_comparison_panels = []
    for (absolute_experiment, relative_experiment,
         absolute_simulation, relative_simulation,
         extent, experiment_fov, simulation_extent, simulation_fov,
         panel_simulation_png_extent,
         _interval_tag, _sim_interval_tag) in alternate_at50_panels:
        for comparison_panels, experiment_data, simulation_data in (
            (relative_comparison_panels, relative_experiment, relative_simulation),
            (absolute_comparison_panels, absolute_experiment, absolute_simulation),
        ):
            comparison_panels.append({
                "simulation": {
                    "data": simulation_data,
                    "extent": simulation_extent,
                    "raw_extent": panel_simulation_png_extent,
                    "fov": simulation_fov,
                },
                "experiment": {
                    "data": experiment_data,
                    "extent": extent,
                    "raw_extent": extent,
                    "fov": experiment_fov,
                },
            })

    # These auxiliary exports use the density assumed when extrapolating cell
    # numbers in overexposed experimental regions as their shared density cap.
    draw_at50_two_timepoint_comparison(
        relative_comparison_panels,
        ADDITIONAL_OUT / (
            "D5_AT50_exp_d01_d12_d24_relative_density_maps_"
            f"{RIM_THRESHOLD_TAG}_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=POISSON_SAMPLING_DENSITY_CAP,
        source_names=("experiment",),
        show_colourbars=False,
        relative_change=True,
    )
    draw_at50_two_timepoint_comparison(
        relative_comparison_panels,
        ADDITIONAL_OUT / (
            "D5_AT50_sim_d01_d12_d24_relative_density_maps_"
            f"{RIM_THRESHOLD_TAG}_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=POISSON_SAMPLING_DENSITY_CAP,
        source_names=("simulation",),
        show_colourbars=True,
        relative_change=True,
    )
    draw_at50_two_timepoint_comparison(
        absolute_comparison_panels,
        ADDITIONAL_OUT / (
            "D5_AT50_exp_d01_d12_d24_absolute_density_change_maps_"
            f"{RIM_THRESHOLD_TAG}_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=POISSON_SAMPLING_DENSITY_CAP,
        source_names=("experiment",),
        show_colourbars=False,
        relative_change=False,
        change_vmax=alternate_change_vmax,
    )

    if PLOT_HDF5_ABSOLUTE:
        hdf5_configurations = (
            (
                "hdf5_160um_minres2p25_minneighbors14",
                hdf5_interval_sources(),
                HDF5_BASE_MIN_RESISTANT_NUMERATOR,
                HDF5_BASE_KERNEL_RADIUS_UM,
            ),
            (
                "hdf5_120um_minres2p25_minneighbors14",
                hdf5_interval_sources(
                    cache_file=HDF5_VARIANT_CACHE_FILE,
                    cache_version=HDF5_VARIANT_CACHE_VERSION,
                    kernel_radius_um=HDF5_VARIANT_KERNEL_RADIUS_UM,
                    min_neighbors=HDF5_VARIANT_MIN_NEIGHBORS,
                    min_resistant_numerator=HDF5_VARIANT_MIN_RESISTANT_NUMERATOR,
                ),
                HDF5_VARIANT_MIN_RESISTANT_NUMERATOR,
                HDF5_VARIANT_KERNEL_RADIUS_UM,
            ),
        )
        for (
            output_tag, interval_sources, minimum_resistant_numerator,
            kernel_radius_um,
        ) in hdf5_configurations:
            output_directory = OUT
            hdf5_absolute_panels, hdf5_absolute_datasets = (
                hdf5_absolute_comparison_panels(
                    interval_sources, comparison_sources, alternate_at50_panels,
                    relative=False,
                )
            )
            _, hdf5_change_vmax = shared_colour_limits(*hdf5_absolute_datasets)
            rim_density_threshold = minimum_resistant_numerator / (
                np.pi * kernel_radius_um ** 2
            )
            for source_name in ("experiment", "simulation"):
                source_tag = "exp" if source_name == "experiment" else "sim"
                draw_at50_two_timepoint_comparison(
                    hdf5_absolute_panels,
                    output_directory / (
                        f"D5_AT50_{source_tag}_d01_d12_d24_"
                        f"absolute_density_change_maps_{output_tag}_"
                        f"scalebar{int(SCALE_BAR_UM)}um.pdf"
                    ),
                    density_vmax=POISSON_SAMPLING_DENSITY_CAP,
                    source_names=(source_name,),
                    show_colourbars=(source_name == "simulation"),
                    relative_change=False,
                    change_vmax=hdf5_change_vmax,
                    rim_density_threshold=rim_density_threshold,
                )
            if output_tag == "hdf5_120um_minres2p25_minneighbors14":
                filter_series_intervals = hdf5_filter_series_intervals(24)
                filter_series_sources = hdf5_interval_sources(
                    cache_file=HDF5_FILTER_SERIES_CACHE_FILE,
                    cache_version=HDF5_FILTER_SERIES_CACHE_VERSION,
                    kernel_radius_um=HDF5_VARIANT_KERNEL_RADIUS_UM,
                    min_neighbors=HDF5_VARIANT_MIN_NEIGHBORS,
                    min_resistant_numerator=HDF5_VARIANT_MIN_RESISTANT_NUMERATOR,
                    intervals=filter_series_intervals,
                )
                draw_hdf5_density_filter_series(
                    filter_series_sources,
                    ADDITIONAL_OUT / (
                        "D5_AT50_exp_sim_density_map_filter_comparison_"
                        "all_timepoints_through_d24_120um_"
                        "minneighbors14.pdf"
                    ),
                    density_vmax=POISSON_SAMPLING_DENSITY_CAP,
                    display_days=[interval[0] for interval in filter_series_intervals],
                )
    draw_at50_two_timepoint_comparison(
        absolute_comparison_panels,
        ADDITIONAL_OUT / (
            "D5_AT50_sim_d01_d12_d24_absolute_density_change_maps_"
            f"{RIM_THRESHOLD_TAG}_scalebar{int(SCALE_BAR_UM)}um.pdf"
        ),
        density_vmax=POISSON_SAMPLING_DENSITY_CAP,
        source_names=("simulation",),
        show_colourbars=True,
        relative_change=False,
        change_vmax=alternate_change_vmax,
    )


def main():
    """Export only HDF5 panels with the shared master-curve profile."""
    # Match the typography of the master-curve panels.
    ft.apply_nature_style(
        text_pt=7, tick_pt=6, legend_pt=5.6,
        overrides={"axes.titlesize": 8, "axes.labelsize": 7},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    configurations = (
        (HDF5_BASE_KERNEL_RADIUS_UM, HDF5_CACHE_FILE, HDF5_CACHE_VERSION),
        (HDF5_VARIANT_KERNEL_RADIUS_UM, HDF5_VARIANT_CACHE_FILE, HDF5_VARIANT_CACHE_VERSION),
    )
    relative = MAP_GROWTH_MODE == "relative"
    for radius, cache, version in configurations:
        sources = hdf5_interval_sources(cache_file=cache, cache_version=version,
            kernel_radius_um=radius, min_neighbors=hdf5_constants.MIN_KERNEL_NEIGHBORS,
            min_resistant_numerator=hdf5_constants.MIN_RESISTANT_NUMERATOR)
        panels, datasets = hdf5_absolute_comparison_panels(sources, relative=relative)
        finite = [np.abs(data[1][np.isfinite(data[1])]) for data in datasets]
        magnitudes = np.concatenate(finite)
        change_vmax = max(float(np.max(magnitudes)), 1e-8) if len(magnitudes) else 1e-8
        for source_name in ("experiment", "simulation"):
            path = OUT / (f"D5_AT50_{source_name}_d01_d12_d24_"
                f"{MAP_GROWTH_MODE}_growth_total_t0_hdf5_{radius:g}um.pdf")
            draw_at50_two_timepoint_comparison(panels, path,
                density_vmax=POISSON_SAMPLING_DENSITY_CAP,
                source_names=(source_name,), show_colourbars=source_name=="simulation",
                relative_change=relative, change_vmax=change_vmax,
                rim_density_threshold=hdf5_constants.MIN_RESISTANT_NUMERATOR/(np.pi*radius**2),
                full_map_view=False)


if __name__ == "__main__":
    main()
    if SAVE_AUXILIARY_SHARED_FIGURES:
        plot_legacy_auxiliaries()
