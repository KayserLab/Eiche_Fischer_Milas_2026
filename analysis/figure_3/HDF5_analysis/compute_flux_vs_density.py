"""
Resistant flux (Δρ_res per 24h) vs local density ρ_tot, per simulation run.
Runnable directly or importable through the analysis.figure_3 package.

Output: OUT_DIR/flux_vs_density_{protocol}.h5 -- per run, median/q25/q75/count
of absolute Δρ_res and relative Δρ_res/ρ_res(t0), binned by ρ_tot.

Run: python compute_flux_vs_density.py
"""

import os
import re

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if os.fspath(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPO_ROOT))
from analysis.figure_3.density_common import run_kernel_radii

# PyCharm Run calculates both radii. Edit this tuple to select just one.
DEFAULT_KERNEL_RADII_UM = (120.0, 160.0)
if __name__ == "__main__":
    run_kernel_radii(Path(__file__).resolve(), DEFAULT_KERNEL_RADII_UM)

from analysis.figure_3.HDF5_analysis.constants import (
    exp_scaling, px_to_micron, time_dump,
    filter_robust_center, filter_outlier_boundary, filter_neighbour_radius, filter_min_neighbours,
    prolif_radius_um, prolif_min_neighbors, prolif_rho_max, prolif_n_rho_bins, min_prolif_bin_count,
    CELL_FLAG_ACTIVE, CELL_TYPE_SUSCEPTIBLE, CELL_TYPE_RESISTANT, N_JOBS,
    KERNEL_QUERY_CHUNK_SIZE,
    SIMULATION_INPUT_DIR, SIMULATION_OUTPUT_DIR,
    MIN_RESISTANT_NUMERATOR,
    is_non_artificial_run, resistant_support, resistant_change_rate,
)
from analysis.figure_3.HDF5_analysis.constants import kernel_settings
from analysis.figure_3 import density_common as escape_core

# ======================================================
# CONFIG -- edit these to point at your own data layout
# ======================================================

def discover_simulation_protocols():
    """Return every simulation strategy with an available outputs directory."""
    prefix = "data-DB_"
    return sorted(
        path.name.removeprefix(prefix)
        for path in SIMULATION_INPUT_DIR.glob(f"{prefix}*")
        if path.is_dir() and (path / "outputs").is_dir()
    )


REQUESTED_STRATEGIES = frozenset(
    label.strip().upper() for label in os.environ.get("HDF5_STRATEGIES", os.environ.get("MAJA_STRATEGIES", "")).split(",")
    if label.strip()
)
SIMULATION_PROTOCOLS = tuple(
    protocol for protocol in discover_simulation_protocols()
    if not REQUESTED_STRATEGIES or protocol.upper() in REQUESTED_STRATEGIES
)

# Defaults come from EICHE_FISCHER_DATA_ROOT. These can be overridden through
# HDF5_SIMULATION_INPUT_DIR and FLUX_OUTPUT_DIR.
OUT_DIR = os.fspath(SIMULATION_OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


def protocol_outputs_dir(protocol, run):
    return os.fspath(SIMULATION_INPUT_DIR / f"data-DB_{protocol}" / "outputs" / run)


def flux_vs_density_path(protocol):
    """Return path to the per-protocol resistant flux-vs-density H5 file."""
    return os.path.join(OUT_DIR, f"flux_vs_density_{protocol}.h5")


# ======================================================
# DERIVED CONSTANTS (grid/radius, this run's units: sim-px)
# ======================================================

prolif_radius      = prolif_radius_um / (exp_scaling * px_to_micron)   # sim-px
prolif_bin_area    = np.pi * prolif_radius ** 2                         # sim-px^2
prolif_grid_extent = 520   # sim-px half-range [-extent, +extent]

prolif_rho_edges = np.linspace(0, prolif_rho_max, prolif_n_rho_bins + 1)

N_GRID      = 520
GRID_EXTENT = prolif_grid_extent
T_OBS_DAYS  = 1.0   # 24h windows
PERIODS = ("all_timepoints", "before_tte", "after_tte", "before_ttf")

# 2D Epanechnikov kernel K(u) = (2/pi)(1 - u^2); (1/h^2) * sum K(d/h) is a density
_EPANECHNIKOV_NORM = 2.0 / (np.pi * prolif_radius ** 2)

X_GRID   = np.linspace(-GRID_EXTENT, GRID_EXTENT, N_GRID)
Y_GRID   = np.linspace(-GRID_EXTENT, GRID_EXTENT, N_GRID)
_xx, _yy = np.meshgrid(X_GRID, Y_GRID, indexing="xy")
GRID_PTS = np.column_stack([_xx.ravel(), _yy.ravel()])

rho_edges       = prolif_rho_edges
MIN_RES_DENSITY = MIN_RESISTANT_NUMERATOR / prolif_bin_area

_q25 = lambda x: np.percentile(x, 25)
_q75 = lambda x: np.percentile(x, 75)


# ======================================================
# SNAPSHOT LOADING (copied from scripts/utils.py, scripts/paper_figures/fig_single_movie.py)
# ======================================================

def _load_data_from_timestep(dump_folder, timestep, filter_active=True):
    """Load one raw simulation snapshot dump."""
    input_file_path = f"{dump_folder}/simulation_{timestep:09d}.h5"
    with h5py.File(input_file_path, "r") as f:
        data = f["data"]

        def read(name):
            for key in (name, f'"{name}"'):
                if key in data:
                    return np.asarray(data[key])
            raise KeyError(f"{name!r} not found in {input_file_path}")

        cell_flags = read("cell_flags")
        mask = (cell_flags == CELL_FLAG_ACTIVE) if filter_active else slice(None)
        nuclei = read("cell_nucleus")[mask]
        cell_type = read("cell_type")[mask]

    nuclei_positions = nuclei[:, :2]
    xs = nuclei_positions[:, 0]
    ys = nuclei_positions[:, 1]

    return {
        "timestep":  timestep,
        "xs":        xs,
        "ys":        ys,
        "cell_type": cell_type,
    }


def _get_scaling(simulation_units):
    return exp_scaling if simulation_units else 1.0


def _remove_outliers(cells, simulation_units=False):
    """Radial + local-density outlier filter (scripts/analysis.py: remove_outliers)."""
    scaling = _get_scaling(simulation_units)
    center_robust = np.array(filter_robust_center / scaling, dtype=float)

    # --- Step 1: radial filter ---
    distances    = np.linalg.norm(cells - center_robust, axis=1)
    radial_mask  = distances <= filter_outlier_boundary / scaling
    cells_radial = cells[radial_mask]

    # --- Step 2: density filter ---
    if len(cells_radial) == 0:
        return cells_radial, radial_mask

    tree      = cKDTree(cells_radial)
    neighbors = tree.query_ball_point(cells_radial, r=filter_neighbour_radius / scaling)
    density_mask = np.array([len(n) >= filter_min_neighbours for n in neighbors])

    filtered_cells = cells_radial[density_mask]

    # --- build full mask ---
    full_mask = np.zeros(len(cells), dtype=bool)
    radial_indices = np.where(radial_mask)[0]
    full_mask[radial_indices[density_mask]] = True

    return filtered_cells, full_mask


def _prepare_snapshot(snap, remove_outliers=True):
    """Convert a raw snapshot dict into (positions, cell_type) plotting arrays."""
    cells = np.column_stack((snap["xs"], snap["ys"]))

    if remove_outliers:
        cells, mask = _remove_outliers(cells, simulation_units=True)
        cell_types = snap["cell_type"][mask]
    else:
        cell_types = snap["cell_type"]

    return cells, cell_types


# ======================================================
# LOCAL DENSITY FIELDS -- shared with compute_flux_vs_density_experiment.py
# ======================================================

def _bin_groups(points, width):
    """Group point indices by square spatial bin."""
    coordinates = np.floor(points / width).astype(np.int64)
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    ordered = coordinates[order]
    boundaries = np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1
    return {tuple(coordinates[group[0]]): group
            for group in np.split(order, boundaries) if group.size}


def _binned_neighbours(pos, grid_pts, radius, budget):
    """Yield radius neighbours in input order without invoking cKDTree."""
    positions = np.asarray(pos, dtype=np.float64)
    grid = np.asarray(grid_pts, dtype=np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(grid).all():
        raise ValueError('Positions and grid points must be finite')
    width = np.nextafter(radius, np.inf)
    cell_bins = _bin_groups(positions, width)
    radius_squared = radius ** 2
    for (x, y), indices in _bin_groups(grid, width).items():
        nearby = [cell_bins[(x + dx, y + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if (x + dx, y + dy) in cell_bins]
        if not nearby:
            continue
        candidates = np.sort(np.concatenate(nearby))
        cells = positions[candidates]
        block_size = max(1, budget // len(candidates))
        for start in range(0, len(indices), block_size):
            batch = indices[start:start + block_size]
            dx = grid[batch, 0, None] - cells[None, :, 0]
            dy = grid[batch, 1, None] - cells[None, :, 1]
            within = dx * dx + dy * dy <= radius_squared
            for index, mask in zip(batch, within):
                yield index, candidates[mask]


def _tree_neighbours(pos, grid_pts, radius, budget):
    tree = cKDTree(pos)
    # A fixed number of grid points alone does not bound memory in dense frames.
    for start in range(0, len(grid_pts), KERNEL_QUERY_CHUNK_SIZE):
        points = grid_pts[start:start + KERNEL_QUERY_CHUNK_SIZE]
        counts = tree.query_ball_point(points, radius, return_length=True)
        cumulative = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        cursor = 0
        while cursor < len(points):
            end = int(np.searchsorted(cumulative, cumulative[cursor] + budget,
                                      side='right') - 1)
            end = max(cursor + 1, min(end, len(points)))
            neighbors = tree.query_ball_point(points[cursor:end], radius, return_sorted=True)
            for index, items in enumerate(neighbors, start=start + cursor):
                yield index, items
            del neighbors
            cursor = end


def _kernel_density_field(pos, cell_type, grid_pts, radius, epanechnikov_norm, n_grid,
                          min_neighbors, *, use_spatial_bins=False):
    """Epanechnikov-weighted kernel density (tot/sus/res) at each
    grid point, plus a `valid` mask (>= min_neighbors)."""
    pos = np.ascontiguousarray(pos)
    cell_type = np.ascontiguousarray(cell_type)
    grid_pts = np.asarray(grid_pts)
    if pos.ndim != 2 or pos.shape[1] != 2 or cell_type.shape != (len(pos),):
        raise ValueError('Positions must be (N, 2) and cell types must be (N,)')
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError('Grid points must be (M, 2)')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Kernel radius must be finite and positive')
    susceptible = cell_type == CELL_TYPE_SUSCEPTIBLE
    neighbour_budget = max(1, int(os.environ.get('HDF5_KERNEL_MAX_NEIGHBOR_ENTRIES', '250000')))
    n_neighbors = np.zeros(len(grid_pts), dtype=np.int32)
    kernel_tot = np.zeros(len(grid_pts))
    kernel_sus = np.zeros(len(grid_pts))
    query = _binned_neighbours if use_spatial_bins else _tree_neighbours
    for index, neighbors in query(pos, grid_pts, radius, neighbour_budget):
        n_neighbors[index] = len(neighbors)
        if not len(neighbors):
            continue
        # Both queries return sorted cell indices to preserve summation order.
        indices = np.asarray(neighbors, dtype=np.intp)
        sus = susceptible[indices]
        d = np.linalg.norm(pos[indices] - grid_pts[index], axis=1)
        w = epanechnikov_norm * (1.0 - (d / radius) ** 2)
        kernel_tot[index] = w.sum()
        kernel_sus[index] = w[sus].sum()

    grid_shape = (
        (int(n_grid), int(n_grid))
        if np.isscalar(n_grid)
        else tuple(int(value) for value in n_grid)
    )
    valid      = (n_neighbors >= min_neighbors).reshape(grid_shape)
    kernel_tot = kernel_tot.reshape(grid_shape)
    kernel_sus = kernel_sus.reshape(grid_shape)
    kernel_res = kernel_tot - kernel_sus
    return kernel_tot, kernel_sus, kernel_res, valid


def local_densities(run_dir, ts, *, kernel_radius_um=None, min_neighbors=None,
                    use_spatial_bins=False):
    """Per-grid-point kernel (rho_tot, rho_sus, rho_res), plus a `valid` mask."""
    snap = _load_data_from_timestep(run_dir, ts)
    pos, cell_type = _prepare_snapshot(snap)
    radius = prolif_radius if kernel_radius_um is None else kernel_radius_um / (exp_scaling * px_to_micron)
    minimum = prolif_min_neighbors if min_neighbors is None else min_neighbors
    norm = 2.0 / (np.pi * radius ** 2)
    return _kernel_density_field(pos, cell_type, GRID_PTS, radius, norm, N_GRID, minimum,
                                 use_spatial_bins=use_spatial_bins)


def _selected_timesteps(run_dir):
    """Timesteps 24h apart, in this run's own dump spacing."""
    pattern = re.compile(r"simulation_(\d+)\.h5")
    all_timesteps = sorted(int(pattern.match(f).group(1)) for f in os.listdir(run_dir) if pattern.match(f))
    if len(all_timesteps) < 2:
        return []

    spacing      = all_timesteps[1] - all_timesteps[0]
    window_steps = round(T_OBS_DAYS * time_dump)
    if window_steps % spacing != 0:
        raise ValueError(f"{run_dir}: 24h window ({window_steps} steps) is not an exact "
                          f"multiple of this run's {spacing}-step dump spacing.")

    t0 = all_timesteps[0]
    return [ts for ts in all_timesteps if (ts - t0) % window_steps == 0]

def _time_to_failure(run_dir, timesteps):
    """First observed day with resistant cells >= 50% of the initial total."""
    initial_total = None
    for timestep in timesteps:
        snapshot = _load_data_from_timestep(run_dir, timestep)
        cell_types = snapshot["cell_type"]
        if initial_total is None:
            initial_total = len(cell_types)
        if initial_total and np.count_nonzero(cell_types == CELL_TYPE_RESISTANT) >= 0.5 * initial_total:
            return timestep / time_dump + escape_core.sim.DEFAULT_DAY_OFFSET
    return None


# ======================================================
# FLUX VS DENSITY -- _bin_curve shared with compute_flux_vs_density_experiment.py
# ======================================================

def _bin_curve(rho_tot, values, rho_edges):
    """median/q25/q75/count of `values` per ρ_tot bin (NaN below min_prolif_bin_count)."""
    median, _, _ = binned_statistic(rho_tot, values, statistic="median", bins=rho_edges)
    q25,    _, _ = binned_statistic(rho_tot, values, statistic=_q25,     bins=rho_edges)
    q75,    _, _ = binned_statistic(rho_tot, values, statistic=_q75,     bins=rho_edges)
    count,  _, _ = binned_statistic(rho_tot, values, statistic="count",  bins=rho_edges)

    enough = count >= min_prolif_bin_count
    return (np.where(enough, median, np.nan), np.where(enough, q25, np.nan),
            np.where(enough, q75, np.nan), count)


def _run_curve(protocol, run):
    """This run's (absolute, relative) Δρ_res-vs-ρ_tot curves, each a (median, q25, q75, count) tuple."""
    run_dir   = protocol_outputs_dir(protocol, run)
    timesteps = _selected_timesteps(run_dir)
    ttf = _time_to_failure(run_dir, timesteps)
    print(f"{protocol}_{run}: {len(timesteps)} snapshots -> {len(timesteps) - 1} windows")

    rho_tot_pool, delta_res_pool         = [], []
    rho_tot_rel_pool, delta_res_rel_pool = [], []
    prev_fields, prev_valid = None, None
    for ts in timesteps:
        rho_tot_1, _, rho_res_1, valid = local_densities(run_dir, ts)

        if prev_fields is not None:
            rho_tot_0, rho_res_0 = prev_fields

            rho_tot_avg = 0.5 * (rho_tot_0 + rho_tot_1)
            delta_res   = rho_res_1 - rho_res_0

            # pool valid bins
            valid_res = resistant_support(rho_tot_0, rho_res_0, rho_tot_1, rho_res_1,
                                          prev_valid, valid, threshold=MIN_RES_DENSITY)
            rho_tot_pool.append(rho_tot_avg[valid_res])
            delta_res_pool.append(delta_res[valid_res])

            # Absolute and relative changes use the same t0 one-cell support.
            valid_rel = valid_res
            rho_tot_rel_pool.append(rho_tot_avg[valid_rel])
            delta_res_rel_pool.append(delta_res[valid_rel] / rho_res_0[valid_rel])

        prev_fields, prev_valid = (rho_tot_1, rho_res_1), valid

    absolute = _bin_curve(np.concatenate(rho_tot_pool), np.concatenate(delta_res_pool), rho_edges)
    relative = _bin_curve(np.concatenate(rho_tot_rel_pool), np.concatenate(delta_res_rel_pool), rho_edges)
    return absolute, relative


def _empty_curve(edges=None):
    selected_edges = rho_edges if edges is None else edges
    size = len(selected_edges) - 1
    return (
        np.full(size, np.nan),
        np.full(size, np.nan),
        np.full(size, np.nan),
        np.zeros(size),
    )


def _curve_from_pool(x_values, y_values, edges=None):
    selected_edges = rho_edges if edges is None else edges
    if not x_values:
        return _empty_curve(selected_edges)
    pooled_x = np.concatenate(x_values)
    pooled_y = np.concatenate(y_values)
    finite = np.isfinite(pooled_x) & np.isfinite(pooled_y)
    if not np.any(finite):
        return _empty_curve(selected_edges)
    return _bin_curve(
        pooled_x[finite], pooled_y[finite], selected_edges
    )


def _run_curve_by_period(protocol, run, tte):
    """Calculate one simulation run separately for each requested TTE period."""
    run_dir = protocol_outputs_dir(protocol, run)
    timesteps = _selected_timesteps(run_dir)
    ttf = _time_to_failure(run_dir, timesteps)
    print(
        f"{protocol}_{run}: {len(timesteps)} snapshots -> {len(timesteps) - 1} windows; "
        f"TTE={tte if tte is not None else 'not observed'}; "
        f"TTF={ttf if ttf is not None else 'not observed'}"
    )
    pools = {
        period: {"x_abs": [], "y_abs": [], "x_rel": [], "y_rel": []}
        for period in PERIODS
    }
    previous = None
    previous_valid = None
    previous_day = None
    for timestep in timesteps:
        rho_total_1, _, rho_resistant_1, valid = local_densities(run_dir, timestep)
        day = timestep / time_dump + escape_core.sim.DEFAULT_DAY_OFFSET
        if previous is not None:
            rho_total_0, rho_resistant_0 = previous
            rho_total_average = 0.5 * (rho_total_0 + rho_total_1)
            delta_resistant = rho_resistant_1 - rho_resistant_0
            valid_absolute = resistant_support(
                rho_total_0, rho_resistant_0, rho_total_1, rho_resistant_1,
                previous_valid, valid, threshold=MIN_RES_DENSITY)
            valid_relative = valid_absolute

            selected_periods = ["all_timepoints"]
            if tte is None or day < tte:
                selected_periods.append("before_tte")
            if tte is not None and previous_day > tte:
                selected_periods.append("after_tte")
            # The interval upper endpoint must be strictly before the first
            # acquisition at which resistant cells reach 50% of the initial total.
            if ttf is None or day < ttf:
                selected_periods.append("before_ttf")
            for period in selected_periods:
                pool = pools[period]
                pool["x_abs"].append(rho_total_average[valid_absolute])
                pool["y_abs"].append(delta_resistant[valid_absolute])
                pool["x_rel"].append(rho_total_average[valid_relative])
                pool["y_rel"].append(
                    delta_resistant[valid_relative] / rho_resistant_0[valid_relative]
                )
        previous = (rho_total_1, rho_resistant_1)
        previous_valid = valid
        previous_day = day

    return {
        period: (
            _curve_from_pool(pool["x_abs"], pool["y_abs"]),
            _curve_from_pool(pool["x_rel"], pool["y_rel"]),
        )
        for period, pool in pools.items()
    }


if __name__ == "__main__":
    escape_days = escape_core.simulation_escape_days(
        escape_core.DEFAULT_SIM_ESCAPE,
        escape_core.DEFAULT_ESCAPE_MARGIN_CSV,
    )
    if not SIMULATION_PROTOCOLS:
        raise SystemExit(
            f"No data-DB_*/outputs directories found under {SIMULATION_INPUT_DIR}"
        )
    print(f"Simulation strategies: {', '.join(SIMULATION_PROTOCOLS)}")
    for protocol in SIMULATION_PROTOCOLS:
        outputs_base = protocol_outputs_dir(protocol, "")
        if not os.path.isdir(outputs_base):
            raise FileNotFoundError(
                f"Simulation outputs directory does not exist for {protocol}: {outputs_base}"
            )
        discovered_run_folders = sorted(
            d for d in os.listdir(outputs_base)
            if os.path.isdir(os.path.join(outputs_base, d)) and is_non_artificial_run(d)
        )
        run_folders = []
        for run in discovered_run_folders:
            selected = _selected_timesteps(protocol_outputs_dir(protocol, run))
            if len(selected) < 2:
                print(
                    f"[skip] {protocol}/{run}: only {len(selected)} usable "
                    "daily snapshots; at least 2 are required"
                )
                continue
            run_folders.append(run)
        if not run_folders:
            print(f"[skip] {protocol}: no runs contain at least two daily snapshots")
            continue

        strategy_key = escape_core.canonical_strategy_label(
            escape_core.SOURCE_TO_DISPLAY.get(protocol, protocol)
        )
        results_by_period = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_run_curve_by_period)(
                protocol,
                run,
                escape_days.get(
                    (strategy_key, escape_core.normalize_simulation_run_id(run))
                ),
            )
            for run in run_folders
        )
        results = [result["all_timepoints"] for result in results_by_period]

        median     = np.stack([r[0][0] for r in results])
        q25        = np.stack([r[0][1] for r in results])
        q75        = np.stack([r[0][2] for r in results])
        count      = np.stack([r[0][3] for r in results])
        median_rel = np.stack([r[1][0] for r in results])
        q25_rel    = np.stack([r[1][1] for r in results])
        q75_rel    = np.stack([r[1][2] for r in results])
        count_rel  = np.stack([r[1][3] for r in results])
        print(f"{protocol}: {len(run_folders)} run curves, "
              f"{np.isfinite(median).any(axis=1).sum()} with >=1 valid absolute bin, "
              f"{np.isfinite(median_rel).any(axis=1).sum()} with >=1 valid relative bin")

        out_path = flux_vs_density_path(protocol)
        with h5py.File(out_path, "w") as f:
            f.attrs.update(kernel_settings(prolif_radius_um))
            f.attrs["resistant_support"] = (
                "rho_resistant(t0) >= 2/(pi*kernel_radius^2) "
                "for absolute and relative change"
            )
            f.attrs["minimum_resistant_density_native"] = MIN_RES_DENSITY
            f.attrs["period_definitions"] = (
                "before_tte: upper endpoint < TTE; "
                "after_tte: lower endpoint > TTE; "
                "before_ttf: upper endpoint < TTF (first day resistant cells reach 50% of initial total); "
                "all_timepoints: every interval"
            )
            f.create_dataset("run_names",  data=np.array(run_folders, dtype="S"))
            f.create_dataset("median",     data=median,     compression="gzip")
            f.create_dataset("q25",        data=q25,        compression="gzip")
            f.create_dataset("q75",        data=q75,        compression="gzip")
            f.create_dataset("count",      data=count,      compression="gzip")
            f.create_dataset("median_rel", data=median_rel, compression="gzip")
            f.create_dataset("q25_rel",    data=q25_rel,    compression="gzip")
            f.create_dataset("q75_rel",    data=q75_rel,    compression="gzip")
            f.create_dataset("count_rel",  data=count_rel,  compression="gzip")
            for period in PERIODS:
                period_results = [result[period] for result in results_by_period]
                group = f.create_group(period)
                for name, result_index, curve_index in (
                    ("median", 0, 0), ("q25", 0, 1), ("q75", 0, 2), ("count", 0, 3),
                    ("median_rel", 1, 0), ("q25_rel", 1, 1),
                    ("q75_rel", 1, 2), ("count_rel", 1, 3),
                ):
                    group.create_dataset(
                        name,
                        data=np.stack([
                            result[result_index][curve_index]
                            for result in period_results
                        ]),
                        compression="gzip",
                    )
        print(f"Saved: {out_path}")
