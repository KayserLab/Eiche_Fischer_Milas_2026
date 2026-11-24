"""
Resistant flux (Δρ_res per 24h) vs local density ρ_tot, computed from raw
experimental point clouds. Same procedure as compute_flux_vs_density.py
(grid, Epanechnikov kernel, flux masks, per-run binning; kernel/binning
functions imported from there), adapted for real data: no outlier removal.
One fixed absolute-coordinate grid is used for every day in a well. Raw
coordinates are already in exp-px, so no exp_scaling conversion; Δρ_res divided by
actual elapsed days, since real snapshots are 2-3 days apart, not a fixed 24h.

Each well uses one absolute-coordinate grid covering all of its acquisitions,
with the same physical grid spacing as the simulation script.

Output: OUT_DIR/flux_vs_density_{strategy}.h5 (same schema as the
simulation version, own directory so the two never collide).

Run: python compute_flux_vs_density_experiment.py
"""

import glob
import os
import re

import h5py
import numpy as np
from joblib import Parallel, delayed

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
    exp_scaling, px_to_micron, AREA_SCALE,
    prolif_radius_um, prolif_min_neighbors, prolif_rho_max, prolif_n_rho_bins,
    CELL_TYPE_SUSCEPTIBLE, CELL_TYPE_RESISTANT, EXPERIMENT_N_JOBS,
    KERNEL_QUERY_CHUNK_SIZE,
    EXPERIMENT_INPUT_DIR, EXPERIMENT_OUTPUT_DIR,
    MIN_RESISTANT_NUMERATOR,
    resistant_support, resistant_change_rate,
)
from analysis.figure_3.HDF5_analysis.constants import kernel_settings
from analysis.figure_3.HDF5_analysis.compute_flux_vs_density import (
    PERIODS,
    _bin_curve,
    _curve_from_pool,
    _kernel_density_field,
)
from shared.experiment_filters import EXCLUDE_SAMPLES
from analysis.figure_3 import density_common as escape_core

# ======================================================
# CONFIG -- edit these to point at your own data/output layout
# ======================================================

RAW_EXP_DATA_DIR = os.fspath(EXPERIMENT_INPUT_DIR)
REQUESTED_STRATEGIES = frozenset(
    label.strip().upper() for label in os.environ.get("HDF5_STRATEGIES", os.environ.get("MAJA_STRATEGIES", "")).split(",")
    if label.strip()
)

OUT_DIR = os.fspath(EXPERIMENT_OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Filenames look like G418_a2a4_B3_AT100-B3_total_poisson_padded.h5.
# Strategy and well identify each experimental run in this dataset.
FILENAME_RE = re.compile(
    r"^G418_a2a4_(?P<well>[A-Za-z]+[0-9]+)_(?P<strategy>.+)-(?P=well)_total.*\.h5$"
)


def _sample_and_rule(path):
    """Return the canonical experiment-filter sample name and its rule."""
    match = FILENAME_RE.match(os.path.basename(path))
    if match is None:
        raise ValueError(f"Unexpected experimental HDF5 filename: {path}")
    sample = f"{match['well']}_{match['strategy']}-{match['well']}"
    return sample, EXCLUDE_SAMPLES.get(sample, ...)


def _plate_name(path):
    for parent in os.path.abspath(path).split(os.sep)[::-1]:
        if re.fullmatch(r"Plate_[0-9]+", parent, flags=re.IGNORECASE):
            return parent
    raise ValueError(f"Could not determine Plate_N directory for {path}")


def discover_runs():
    """Return {strategy: {strategy/well: filepath}} for all input files."""
    runs = {}
    excluded = 0
    truncated = 0
    for path in sorted(glob.glob(os.path.join(RAW_EXP_DATA_DIR, "**", "*.h5"), recursive=True)):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        if REQUESTED_STRATEGIES and m["strategy"].upper() not in REQUESTED_STRATEGIES:
            continue
        sample, rule = _sample_and_rule(path)
        if rule is None:
            excluded += 1
            print(f"[filter] Excluding {sample}")
            continue
        if rule is not ...:
            truncated += 1
        run_name = f"{m['strategy']}/{m['well']}"
        strategy_runs = runs.setdefault(m["strategy"], {})
        if run_name in strategy_runs:
            raise ValueError(
                f"Duplicate experimental run {run_name!r} for {m['strategy']!r}: "
                f"{strategy_runs[run_name]} and {path}"
            )
        strategy_runs[run_name] = path
    print(
        f"[filter] shared/experiment_filters.py: {excluded} wells excluded; "
        f"{truncated} wells truncated"
    )
    return runs


def flux_vs_density_path(strategy):
    """Return path to the per-strategy resistant flux-vs-density H5 file."""
    return os.path.join(OUT_DIR, f"flux_vs_density_{strategy}.h5")


# ======================================================
# DERIVED CONSTANTS (grid/radius, this run's units: exp-px)
# ======================================================

prolif_radius   = prolif_radius_um / px_to_micron   # exp-px (kernel bandwidth h)
prolif_bin_area = np.pi * prolif_radius ** 2         # exp-px^2

prolif_rho_max_exp = prolif_rho_max / AREA_SCALE     # exp-px^-2 (converted from sim-px^-2)
prolif_rho_edges   = np.linspace(0, prolif_rho_max_exp, prolif_n_rho_bins + 1)

# Grid: same physical spacing as the simulation's
_SIM_GRID_SPACING = 2 * 520 / (520 - 1) * exp_scaling   # ~23.8 exp-px

_EPANECHNIKOV_NORM = 2.0 / (np.pi * prolif_radius ** 2)

rho_edges       = prolif_rho_edges
MIN_RES_DENSITY = MIN_RESISTANT_NUMERATOR / prolif_bin_area


# ======================================================
# SNAPSHOT LOADING
# ======================================================

def _sorted_days(f):
    """[(day_num, day_key), ...] sorted by day_num, from the file's 'sus' group."""
    return sorted((int(k.split("_")[1]), k) for k in f["sus"].keys())


def _raw_day_positions(f, day_key):
    """Return susceptible and resistant positions in image coordinates."""
    pos_sus = f["sus"][day_key]["block0_values"][:][:, [1, 0]]
    pos_res = f["res"][day_key]["block0_values"][:][:, [1, 0]]
    return pos_sus, pos_res



def _time_to_failure(f, days):
    """First observed day with resistant cells >= 50% of the initial total."""
    initial_total = None
    for day, day_key in days:
        positions_sus, positions_res = _raw_day_positions(f, day_key)
        if initial_total is None:
            initial_total = len(positions_sus) + len(positions_res)
        if initial_total and len(positions_res) >= 0.5 * initial_total:
            return float(day)
    return None
def _fixed_well_grid(f, *, kernel_radius_um=None):
    """Create one absolute-coordinate grid covering every acquisition day."""
    minimum = np.array([np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf])
    for _, day_key in _sorted_days(f):
        pos_sus, pos_res = _raw_day_positions(f, day_key)
        positions = np.vstack([pos_sus, pos_res])
        minimum = np.minimum(minimum, positions.min(axis=0))
        maximum = np.maximum(maximum, positions.max(axis=0))
    radius = prolif_radius if kernel_radius_um is None else kernel_radius_um / px_to_micron
    minimum -= radius
    maximum += radius
    x_grid = np.arange(
        minimum[0], maximum[0] + 0.5 * _SIM_GRID_SPACING, _SIM_GRID_SPACING
    )
    y_grid = np.arange(
        minimum[1], maximum[1] + 0.5 * _SIM_GRID_SPACING, _SIM_GRID_SPACING
    )
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]), xx.shape


def _load_day(f, day_key):
    """Load one calendar day's (positions, cell_type).

    block0_values columns are [y, x] (pandas HDF store), swapped to (x, y)
    below. Coordinates remain in their original absolute image frame.
    """
    pos_sus, pos_res = _raw_day_positions(f, day_key)

    pos = np.vstack([pos_sus, pos_res])
    cell_type = np.concatenate([
        np.full(len(pos_sus), CELL_TYPE_SUSCEPTIBLE),
        np.full(len(pos_res), CELL_TYPE_RESISTANT),
    ])
    return pos, cell_type


def local_densities(f, day_key, grid_points, grid_shape, *, kernel_radius_um=None, min_neighbors=None):
    """Per-grid-point kernel (rho_tot, rho_sus, rho_res), plus a `valid` mask."""
    pos, cell_type = _load_day(f, day_key)
    radius = prolif_radius if kernel_radius_um is None else kernel_radius_um / px_to_micron
    minimum = prolif_min_neighbors if min_neighbors is None else min_neighbors
    return _kernel_density_field(
        pos,
        cell_type,
        grid_points,
        radius,
        2.0 / (np.pi * radius ** 2),
        grid_shape,
        minimum,
    )


# ======================================================
# FLUX VS DENSITY (_bin_curve imported from compute_flux_vs_density.py)
# ======================================================

def _run_curve(h5_path, run_name):
    """Return one filtered well's absolute and relative growth curves."""
    _, rule = _sample_and_rule(h5_path)
    cutoff = None if rule is ... else float(rule)
    with h5py.File(h5_path, "r") as f:
        grid_points, grid_shape = _fixed_well_grid(f)
        days = _sorted_days(f)
        original_day_count = len(days)
        if cutoff is not None:
            days = [(day, key) for day, key in days if day <= cutoff]
        print(
            f"{run_name}: {len(days)}/{original_day_count} days -> "
            f"{max(len(days) - 1, 0)} windows"
            + (f" (truncated at day {cutoff:g})" if cutoff is not None else "")
        )

        rho_tot_pool, delta_res_pool         = [], []
        rho_tot_rel_pool, delta_res_rel_pool = [], []
        prev_fields, prev_valid, prev_day = None, None, None
        for day_num, day_key in days:
            rho_tot_1, _, rho_res_1, valid = local_densities(
                f, day_key, grid_points, grid_shape
            )

            if prev_fields is not None:
                rho_tot_0, rho_res_0 = prev_fields
                delta_t_days = day_num - prev_day   # real gaps are 2-3 days here, not a fixed 24h

                rho_tot_avg = 0.5 * (rho_tot_0 + rho_tot_1)
                delta_res   = (rho_res_1 - rho_res_0) / delta_t_days   # per calendar day

                # pool valid bins
                valid_res = resistant_support(rho_tot_0, rho_res_0, rho_tot_1, rho_res_1,
                                              prev_valid, valid, threshold=MIN_RES_DENSITY)
                rho_tot_pool.append(rho_tot_avg[valid_res])
                delta_res_pool.append(delta_res[valid_res])

                # Absolute and relative changes use the same t0 one-cell support.
                valid_rel = valid_res
                rho_tot_rel_pool.append(rho_tot_avg[valid_rel])
                delta_res_rel_pool.append(delta_res[valid_rel] / rho_res_0[valid_rel])

            prev_fields, prev_valid, prev_day = (rho_tot_1, rho_res_1), valid, day_num

    absolute = _bin_curve(np.concatenate(rho_tot_pool), np.concatenate(delta_res_pool), rho_edges)
    relative = _bin_curve(np.concatenate(rho_tot_rel_pool), np.concatenate(delta_res_rel_pool), rho_edges)
    return absolute, relative


def _run_curve_by_period(h5_path, run_name, escape_info):
    """Calculate one experimental well for all requested TTE periods."""
    _, rule = _sample_and_rule(h5_path)
    manual_cutoff = None if rule is ... else float(rule)
    escape_day, pre_escape_day = (
        (None, None) if escape_info is None else escape_info
    )
    pools = {
        period: {"x_abs": [], "y_abs": [], "x_rel": [], "y_rel": []}
        for period in PERIODS
    }
    with h5py.File(h5_path, "r") as f:
        grid_points, grid_shape = _fixed_well_grid(f)
        days = _sorted_days(f)
        original_day_count = len(days)
        if manual_cutoff is not None:
            days = [(day, key) for day, key in days if day <= manual_cutoff]
        ttf = _time_to_failure(f, days)
        print(
            f"{run_name}: {len(days)}/{original_day_count} days -> "
            f"{max(len(days) - 1, 0)} windows; "
            f"TTE={escape_day if escape_day is not None else 'not observed'}" + "; "
            + f"TTF={ttf if ttf is not None else 'not observed'}"
        )
        previous = None
        previous_valid = None
        previous_day = None
        for day, day_key in days:
            rho_total_1, _, rho_resistant_1, valid = local_densities(
                f, day_key, grid_points, grid_shape
            )
            if previous is not None:
                rho_total_0, rho_resistant_0 = previous
                delta_days = day - previous_day
                rho_total_average = 0.5 * (rho_total_0 + rho_total_1)
                delta_resistant = (
                    (rho_resistant_1 - rho_resistant_0) / delta_days
                )
                valid_absolute = resistant_support(
                    rho_total_0, rho_resistant_0, rho_total_1, rho_resistant_1,
                    previous_valid, valid, threshold=MIN_RES_DENSITY)
                valid_relative = valid_absolute
                selected_periods = ["all_timepoints"]
                if escape_day is None or day <= pre_escape_day:
                    selected_periods.append("before_tte")
                if escape_day is not None and previous_day > escape_day:
                    selected_periods.append("after_tte")
                # Do not include the interval ending at the first observed
                # failure day, when resistant cells reach 50% of the initial total.
                if ttf is None or day < ttf:
                    selected_periods.append("before_ttf")
                for period in selected_periods:
                    pool = pools[period]
                    pool["x_abs"].append(rho_total_average[valid_absolute])
                    pool["y_abs"].append(delta_resistant[valid_absolute])
                    pool["x_rel"].append(rho_total_average[valid_relative])
                    pool["y_rel"].append(
                        delta_resistant[valid_relative]
                        / rho_resistant_0[valid_relative]
                    )
            previous = (rho_total_1, rho_resistant_1)
            previous_valid = valid
            previous_day = day

    return {
        period: (
            _curve_from_pool(pool["x_abs"], pool["y_abs"], rho_edges),
            _curve_from_pool(pool["x_rel"], pool["y_rel"], rho_edges),
        )
        for period, pool in pools.items()
    }


if __name__ == "__main__":
    print(
        f"[parallel] {EXPERIMENT_N_JOBS} experimental workers; "
        f"KD-tree query chunks of {KERNEL_QUERY_CHUNK_SIZE:,} grid points"
    )
    escape_table = escape_core.experimental_escape_cutoffs(
        escape_core.DEFAULT_EXPERIMENT_ESCAPE,
        escape_core.DEFAULT_ESCAPE_MARGIN_CSV,
    )
    escape_lookup = {
        (str(row.plate), str(row.well)): (
            float(row.escape_day),
            float(row.pre_escape_cutoff_day),
        )
        for row in escape_table.itertuples(index=False)
    }
    runs_by_strategy = discover_runs()
    if not runs_by_strategy:
        raise SystemExit(f"No matching H5 files found under {RAW_EXP_DATA_DIR}")

    for strategy, plates in sorted(runs_by_strategy.items()):
        plate_names = sorted(plates.keys())

        results_by_period = Parallel(n_jobs=EXPERIMENT_N_JOBS, backend="loky")(
            delayed(_run_curve_by_period)(
                plates[run_name],
                run_name,
                escape_lookup.get(
                    (_plate_name(plates[run_name]), _sample_and_rule(plates[run_name])[0])
                ),
            )
            for run_name in plate_names
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
        print(f"{strategy}: {len(plate_names)} run curves, "
              f"{np.isfinite(median).any(axis=1).sum()} with >=1 valid absolute bin, "
              f"{np.isfinite(median_rel).any(axis=1).sum()} with >=1 valid relative bin")

        out_path = flux_vs_density_path(strategy)
        with h5py.File(out_path, "w") as f:
            f.attrs.update(kernel_settings(prolif_radius_um))
            f.attrs["experiment_filters_applied"] = True
            f.attrs["experiment_filter_source"] = "experiment_filters.EXCLUDE_SAMPLES"
            f.attrs["coordinate_grid"] = (
                "one fixed absolute image-coordinate grid per well, covering "
                "all acquisition days with one kernel-radius padding"
            )
            f.attrs["resistant_support"] = (
                "rho_resistant(t0) >= 2/(pi*kernel_radius^2) "
                "for absolute and relative change"
            )
            f.attrs["minimum_resistant_density_native"] = MIN_RES_DENSITY
            f.attrs["period_definitions"] = (
                "before_tte: upper endpoint is no later than last confined day; "
                "after_tte: lower endpoint > TTE; "
                "before_ttf: upper endpoint < TTF (first day resistant cells reach 50% of initial total); "
                "all_timepoints: every interval allowed by manual truncation"
            )
            f.create_dataset("run_names",  data=np.array(plate_names, dtype="S"))
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
