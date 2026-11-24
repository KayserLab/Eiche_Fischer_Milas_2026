#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Stream missing in-vitro strategies into a 40-um all-phase IQR cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as density_base
from analysis.figure_3 import master_curve_iqr_common as common
import plot_density_change_datapoints_coarsegrained as base
from shared.strategy_colors import normalize_strategy


DEFAULT_EXPERIMENT_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
DEFAULT_ESCAPE_CSV = DEFAULT_EXPERIMENT_ROOT / (
    r"Figures\Distance_to_front_analysis\frontcontact_summary_ALL.csv"
)
DEFAULT_OUTPUT_ROOT = Path(r"F:\CoreSeed_Master_Curve_Linear_IQR_Cache\in_vitro")
DEFAULT_STRATEGIES = ("NT", "LV-Agent", "AT100+")
KERNEL_UM = 40
COARSENING_FACTOR = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--escape-csv", type=Path, default=DEFAULT_ESCAPE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--overwrite-well-cache", action="store_true")
    return parser.parse_args()


def discover_days(path: Path) -> list[str]:
    discovered: set[str] = set()
    with h5py.File(path, "r") as handle:
        for group in ("sus", "res"):
            if group not in handle:
                continue
            for key in handle[group].keys():
                if key.startswith("d_") and key[2:].isdigit():
                    discovered.add(key[2:])
    return sorted(discovered, key=lambda value: int(value))


def resistant_and_total_density(
    susceptible,
    resistant,
    *,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    res_image = density_base.rasterize_points_to_coarse_grid(
        resistant,
        height,
        width,
        COARSENING_FACTOR,
    )
    sus_image = density_base.rasterize_points_to_coarse_grid(
        susceptible,
        height,
        width,
        COARSENING_FACTOR,
    )
    coarse_radius = density_base.radius_um_to_coarse_px(
        KERNEL_UM,
        COARSENING_FACTOR,
    )
    res_density = density_base.get_local_density_fft(res_image, coarse_radius)
    total_density = density_base.get_local_density_fft(res_image + sus_image, coarse_radius)
    coarse_area = (density_base.PIXEL_SIZE_UM * COARSENING_FACTOR) ** 2
    res_density = density_base.zero_density_fft_noise(res_density / coarse_area)
    total_density = density_base.zero_density_fft_noise(total_density / coarse_area)
    return res_density, total_density


def interval_phases(
    well: str,
    first_day: float,
    next_day: float,
    escape_day: float | None,
    preescape_cutoff: float | None,
) -> tuple[str, ...]:
    phases = ["overall"]
    if well in base.EXCLUDE_SAMPLES:
        sample_cutoff = base.EXCLUDE_SAMPLES[well]
        phase_allowed = (
            sample_cutoff is not None
            and np.isfinite(next_day)
            and next_day <= float(sample_cutoff)
        )
        if not phase_allowed:
            return tuple(phases)
    if preescape_cutoff is not None and np.isfinite(preescape_cutoff):
        if next_day <= preescape_cutoff:
            phases.append("before_escape")
    if escape_day is not None and np.isfinite(escape_day):
        if first_day >= escape_day:
            phases.append("after_escape")
    return tuple(phases)


def process_well(
    plate: str,
    hdf5_path_text: str,
    strategy: str,
    escape_day: float | None,
    preescape_cutoff: float | None,
    output_root_text: str,
) -> tuple[str, str, str]:
    hdf5_path = Path(hdf5_path_text)
    experiment_root = hdf5_path.parent.parent.parent
    plate_path = experiment_root / plate
    well = density_base.well_identifier_from_pointcloud_filename(hdf5_path.name)
    days = discover_days(hdf5_path)
    if len(days) < 2:
        raise RuntimeError(f"Need at least two timepoints for {plate}/{well}")
    image_shape = density_base.density_canvas_shape(
        os.fspath(plate_path),
        well,
        days[0],
        os.fspath(hdf5_path),
    )
    if image_shape is None:
        raise RuntimeError(f"No density canvas for {plate}/{well}")

    bundle = common.AggregationBundle(
        [strategy],
        common.create_aggregators([strategy]),
        {},
    )
    previous: tuple[np.ndarray, np.ndarray, float] | None = None
    counts = {phase: 0 for phase in common.PHASES}
    for day_text in days:
        susceptible, resistant = density_base.pointcloud_read(
            os.fspath(hdf5_path),
            day_text,
            require_timepoint=True,
        )
        res_density, total_density = resistant_and_total_density(
            susceptible,
            resistant,
            image_shape=image_shape,
        )
        day = float(day_text)
        if previous is not None:
            previous_res, previous_total, previous_day = previous
            phases = interval_phases(
                well,
                previous_day,
                day,
                escape_day,
                preescape_cutoff,
            )
            common.add_linear_interval(
                bundle.aggregators,
                0,
                previous_total,
                previous_res,
                res_density,
                day - previous_day,
                phases,
            )
            for phase in phases:
                counts[phase] += 1
        previous = (res_density, total_density, day)

    cache_path = (
        Path(output_root_text)
        / "run_caches_40um_streamed"
        / plate
        / f"{well}.npz"
    )
    bundle.metadata = {
        "plate": plate,
        "well": well,
        "strategy": strategy,
        "kernel_um": KERNEL_UM,
        "coarsening_factor": COARSENING_FACTOR,
        "escape_day": escape_day,
        "preescape_cutoff_day": preescape_cutoff,
        "intervals": counts,
    }
    common.save_bundle(cache_path, bundle)
    return plate, well, os.fspath(cache_path)


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser()
    escape_csv = args.escape_csv.expanduser()
    output_root = args.output_root.expanduser()
    if not experiment_root.is_dir():
        raise FileNotFoundError(experiment_root)
    if not escape_csv.is_file():
        raise FileNotFoundError(escape_csv)
    requested = set(args.strategies) & set(common.ALLOWED_STRATEGIES)
    cutoffs_frame = base.load_escape_cutoffs(escape_csv)
    cutoffs = {
        (str(row.plate), str(row.well)): (
            float(row.escape_day),
            float(row.pre_escape_cutoff_day),
        )
        for row in cutoffs_frame.itertuples(index=False)
    }

    tasks: list[tuple[str, Path, str, float | None, float | None]] = []
    for plate_index in range(1, 9):
        plate = f"Plate_{plate_index}"
        hdf5_dir = experiment_root / plate / "hdf5"
        for filename in density_base.pointcloud_filenames(
            os.fspath(hdf5_dir),
            poisson=True,
            include_patterns=["*_total_poisson_padded.h5"],
        ):
            raw_strategy = density_base.strategy_name_from_pointcloud_filename(filename)
            strategy = normalize_strategy(raw_strategy)
            if strategy not in requested:
                continue
            hdf5_path = hdf5_dir / filename
            well = density_base.well_identifier_from_pointcloud_filename(filename)
            cutoff = cutoffs.get((plate, well))
            escape_day = cutoff[0] if cutoff is not None else None
            preescape_cutoff = cutoff[1] if cutoff is not None else None
            tasks.append((plate, hdf5_path, strategy, escape_day, preescape_cutoff))
    if not tasks:
        raise RuntimeError("No matching in-vitro HDF5 wells found")
    task_keys = [
        (plate, density_base.well_identifier_from_pointcloud_filename(path.name))
        for plate, path, _strategy, _escape, _pre in tasks
    ]
    if len(task_keys) != len(set(task_keys)):
        raise RuntimeError("Duplicate plate/well tasks remain after HDF5 selection")
    print(f"[wells] {len(tasks)}; strategies={sorted(requested)}")

    def cache_path_for(task) -> Path:
        plate, hdf5_path, _strategy, _escape, _pre = task
        well = density_base.well_identifier_from_pointcloud_filename(hdf5_path.name)
        return output_root / "run_caches_40um_streamed" / plate / f"{well}.npz"

    pending = [
        task for task in tasks
        if args.overwrite_well_cache or not cache_path_for(task).is_file()
    ]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, int(args.workers)), len(pending))
        ) as pool:
            futures = {
                pool.submit(
                    process_well,
                    plate,
                    os.fspath(hdf5_path),
                    strategy,
                    escape_day,
                    preescape_cutoff,
                    os.fspath(output_root),
                ): (plate, hdf5_path.name)
                for plate, hdf5_path, strategy, escape_day, preescape_cutoff in pending
            }
            for future in as_completed(futures):
                plate, well, cache_path = future.result()
                print(f"[complete] {plate}/{well} -> {cache_path}", flush=True)

    paths = [cache_path_for(task) for task in tasks]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} well caches")
    merged = common.merge_bundles(paths)
    merged.metadata.update(
        {
            "dataset": "in_vitro",
            "kernel_um": KERNEL_UM,
            "coarsening_factor": COARSENING_FACTOR,
            "n_wells": len(tasks),
            "strategies": sorted(requested),
            "escape_csv": os.fspath(escape_csv),
        }
    )
    output_path = output_root / "kernel_40um_streamed_other_strategies.npz"
    common.save_bundle(output_path, merged)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
