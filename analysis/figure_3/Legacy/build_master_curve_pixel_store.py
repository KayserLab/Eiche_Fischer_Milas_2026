#!/usr/bin/env python3
"""Build a reusable pixel-level HDF5 store for Figure 3 master curves.

The store retains endpoint density values, interval metadata, and compact
per-well density-binned curves. Downstream
scripts can therefore apply pixel-, interval-, or well-level weighting without
reopening TIFF files or simulation snapshots. Experimental intervals are
written first; independent simulation runs are computed in parallel in local
temporary HDF5 fragments and merged into the same final store.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

ROOT = Path(__file__).resolve().parents[3]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as core
from analysis.figure_3.Legacy import aggregate_well_weighted_master_curve as well_aggregation
from shared.data_paths import COMBINED_RESULTS_CSV, SIMULATION_DATABASE, figure_data_dir
from shared.experiment_filters import EXCLUDE_SAMPLES

OUTPUT_DIR = figure_data_dir(
    "Figure_3", "master_curve_160_pixel_store", create=False,
)
OUTPUT_PATH = OUTPUT_DIR / "master_curve_pixels.h5"
BUILDING_PATH = OUTPUT_DIR / "master_curve_pixels.building.h5"
STRATEGIES = ("CT", "NT", "AT50", "AT100", "LV-Agent", "AT+")
SIMULATION_SOURCES = ("CT", "NT", "AT50", "AT100", "Agent", "Dens")
SUMMARY_SUPPORT_THRESHOLD = 1.0e-8
COMPRESSION_LIBRARY = "blosc:zstd"
COMPRESSION_LEVEL = 7

# Independent simulation runs are computed concurrently.  Keep this modest:
# each worker reads several source HDF5 files and writes a large temporary
# fragment.  Set CORESEED_MASTER_CURVE_WORKERS to override the default.
# Experimental wells are independent and use the same local-fragment workflow.
EXPERIMENT_WORKERS = max(
    1,
    int(os.environ.get("CORESEED_EXPERIMENT_MASTER_CURVE_WORKERS", min(4, os.cpu_count() or 1))),
)

SIMULATION_WORKERS = max(
    1,
    int(os.environ.get("CORESEED_MASTER_CURVE_WORKERS", min(4, os.cpu_count() or 1))),
)
MERGE_CHUNK_ROWS = 500_000


def interval_frame(
        *,
        dataset: str,
        strategy: str,
        replicate: str,
        interval_mode: str,
        start_day: float,
        end_day: float,
        before_escape: bool,
        analysis_allowed: bool,
        total_t0: np.ndarray,
        resistant_t0: np.ndarray,
        total_t1: np.ndarray,
        resistant_t1: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    if len({a.shape for a in (total_t0, resistant_t0, total_t1, resistant_t1)}) != 1:
        raise ValueError(
            f"Density shape mismatch for {dataset}/{replicate} {start_day}->{end_day}"
        )
    delta_days = float(end_day - start_day)
    if delta_days <= 0:
        return pd.DataFrame(), None

    total0 = total_t0.astype(np.float32, copy=False)
    total1 = total_t1.astype(np.float32, copy=False)
    res0 = resistant_t0.astype(np.float32, copy=False)
    res1 = resistant_t1.astype(np.float32, copy=False)
    total_average = 0.5 * (total0 + total1)
    resistant_average = 0.5 * (res0 + res1)
    growth = np.full(total_average.shape, np.nan, dtype=np.float32)
    positive = np.isfinite(res0) & (res0 > 0)
    growth[positive] = (
            (res1[positive] - res0[positive])
            / res0[positive]
            / delta_days
    )
    finite = (
            np.isfinite(total0)
            & np.isfinite(total1)
            & np.isfinite(res0)
            & np.isfinite(res1)
    )
    if not np.any(finite):
        return pd.DataFrame(), None

    non_isolated = core.non_isolated_change_pixels(res1 - res0)
    row_index, column_index = np.nonzero(finite)
    frame = pd.DataFrame({
        "dataset": dataset,
        "strategy": strategy,
        "replicate": replicate,
        "interval_mode": interval_mode,
        "start_day": np.float32(start_day),
        "end_day": np.float32(end_day),
        "midpoint_day": np.float32(0.5 * (start_day + end_day)),
        "delta_days": np.float32(delta_days),
        "before_escape": bool(before_escape),
        "analysis_allowed": bool(analysis_allowed),
        "row": row_index.astype(np.int16),
        "column": column_index.astype(np.int16),
        "total_density_t0": total0[finite],
        "total_density_t1": total1[finite],
        "resistant_density_t0": res0[finite],
        "resistant_density_t1": res1[finite],
        "total_density_average": total_average[finite],
        "resistant_density_average": resistant_average[finite],
        "relative_resistant_growth_per_day": growth[finite],
        "non_isolated_change": non_isolated[finite],
    })

    summary_valid = (
            finite
            & non_isolated
            & (res0 >= SUMMARY_SUPPORT_THRESHOLD)
            & (res1 >= SUMMARY_SUPPORT_THRESHOLD)
    )
    if not np.any(summary_valid):
        return frame, None
    weights = resistant_average[summary_valid].astype(float)
    values = growth[summary_valid].astype(float)
    summary = {
        "dataset": dataset,
        "strategy": strategy,
        "replicate": replicate,
        "interval_mode": interval_mode,
        "start_day": float(start_day),
        "end_day": float(end_day),
        "midpoint_day": 0.5 * (start_day + end_day),
        "delta_days": delta_days,
        "before_escape": bool(before_escape),
        "analysis_allowed": bool(analysis_allowed),
        "total_density_cells_per_1000_um2": float(
            np.mean(total_average[summary_valid]) * core.base.X_DISPLAY_SCALE
        ),
        "relative_resistant_growth_per_day": float(
            np.average(values, weights=weights)
        ),
        "n_valid_pixels": int(np.count_nonzero(summary_valid)),
        "resistant_density_weight_sum": float(weights.sum()),
    }
    return frame, summary


class StoreWriter:
    def __init__(self, path: Path, *, write_provenance: bool = True):
        self.store = pd.HDFStore(
            path,
            mode="w",
            complevel=COMPRESSION_LEVEL,
            complib=COMPRESSION_LIBRARY,
        )
        self.interval_rows: list[dict[str, object]] = []
        self.pixel_rows = 0
        self.well_curve_rows: list[dict[str, object]] = []
        self.write_provenance = write_provenance

    def append_pixels(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.store.append(
            "pixels",
            frame,
            format="table",
            index=False,
            data_columns=["dataset", "strategy", "replicate", "interval_mode"],
            min_itemsize={
                "dataset": 10,
                "strategy": 16,
                "replicate": 96,
                "interval_mode": 24,
            },
        )
        self.pixel_rows += len(frame)

    def append(self, frame: pd.DataFrame, summary: dict[str, object] | None) -> None:
        self.append_pixels(frame)
        if summary is not None:
            self.interval_rows.append(summary)

    def close(self) -> None:
        if self.well_curve_rows:
            well_curves = pd.DataFrame(self.well_curve_rows)
            self.store.put(
                "well_curves",
                well_curves,
                format="table",
                data_columns=["dataset", "strategy", "replicate", "growth_mode"],
            )

        intervals = pd.DataFrame(self.interval_rows)
        self.store.put("intervals", intervals, format="table", data_columns=True)
        if self.write_provenance:
            # A uniform string column avoids PyTables' mixed-object pickle warning.
            provenance = pd.Series({
                "kernel_um": core.KERNEL_UM,
                "coarsening_factor": core.COARSENING_FACTOR,
                "summary_support_threshold": SUMMARY_SUPPORT_THRESHOLD,
                "pixel_density_unit": "cells_per_um2",
                "summary_density_unit": "cells_per_1000_um2",
                "growth_unit": "per_day",
                "experimental_source": os.fspath(core.DEFAULT_INVITRO_RUN),
                "simulation_source": os.fspath(SIMULATION_DATABASE),
            }).astype(str)
            self.store.put("provenance", provenance.to_frame("value"), format="fixed")
        self.store.close()


def add_fragment_well_curves(
    fragment_path: Path,
    dataset: str,
) -> int:
    """Store the compact per-well curves beside one completed fragment."""
    with pd.HDFStore(fragment_path, mode="r") as fragment:
        if "/pixels" not in fragment.keys():
            return 0
    curves = well_aggregation.build_well_curve_table((dataset,), fragment_path)
    if curves.empty:
        return 0
    with pd.HDFStore(fragment_path, mode="a") as fragment:
        fragment.put(
            "well_curves",
            curves,
            format="table",
            data_columns=["dataset", "strategy", "replicate", "growth_mode"],
        )
    return len(curves)


def discover_experiment_tasks() -> list[tuple[str, str, str, Path, Path, float]]:
    cutoffs = core.experimental_escape_cutoffs(
        core.DEFAULT_EXPERIMENT_ESCAPE,
        core.DEFAULT_ESCAPE_MARGIN_CSV,
    )
    cutoff_lookup = {
        (str(row.plate), str(row.well)): float(row.pre_escape_cutoff_day)
        for row in cutoffs.itertuples(index=False)
    }
    tasks = []
    for changes_dir in sorted(
            core.DEFAULT_INVITRO_RUN.glob("Plate_*/*/density_changes")
    ):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        strategy = core.canonical_strategy_label(
            core.base.strategy_from_plate_and_well(plate, well)
        )
        if strategy not in STRATEGIES:
            continue
        tasks.append((
            plate,
            well,
            strategy,
            changes_dir,
            changes_dir.parent / "density_maps",
            cutoff_lookup.get((plate, well), -np.inf),
        ))
    return tasks


def append_experiment_well(
        writer: StoreWriter,
        task: tuple[str, str, str, Path, Path, float],
) -> int:
    """Append every usable interval from one experimental well."""
    plate, well, strategy, changes_dir, maps_dir, cutoff = task
    configured = EXCLUDE_SAMPLES.get(well, ...)
    sample_enabled = configured is not None
    missing_pairs = 0
    for change_path in sorted(
            changes_dir.glob(core.base.CHANGE_SPECS["rel"]["glob"])
    ):
        match = core.base.REL_CHANGE_RE.match(change_path.name)
        if match is None:
            continue
        first_tp, next_tp = match.group("first_tp"), match.group("next_tp")
        first_day = core.base.timepoint_to_day(first_tp)
        next_day = core.base.timepoint_to_day(next_tp)
        allowed = sample_enabled and (
                configured is ... or next_day <= float(configured)
        )
        paths = (
            core.base.density_map_path(maps_dir, well, first_tp),
            core.base.resistant_density_map_path(maps_dir, well, first_tp),
            core.base.density_map_path(maps_dir, well, next_tp),
            core.base.resistant_density_map_path(maps_dir, well, next_tp),
        )
        if not all(path.is_file() for path in paths):
            missing_pairs += 1
            continue
        arrays = [
            tiff.imread(path).astype(np.float32, copy=False)
            for path in paths
        ]
        frame, summary = interval_frame(
            dataset="experiment",
            strategy=strategy,
            replicate=f"{plate}/{well}",
            interval_mode="experimental_schedule",
            start_day=first_day,
            end_day=next_day,
            before_escape=bool(next_day <= cutoff),
            analysis_allowed=bool(allowed),
            total_t0=arrays[0],
            resistant_t0=arrays[1],
            total_t1=arrays[2],
            resistant_t1=arrays[3],
        )
        writer.append(frame, summary)
    return missing_pairs


def write_experiment_fragment(
        task: tuple[str, str, str, Path, Path, float],
        fragment_path: Path,
) -> tuple[Path, str, str, int, int, int]:
    """Build one local, self-contained HDF5 fragment for an experimental well."""
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[experiment worker] {task[0]}/{task[1]}: reading intervals", flush=True)
    writer = StoreWriter(fragment_path, write_provenance=False)
    try:
        missing_pairs = append_experiment_well(writer, task)
        pixels = writer.pixel_rows
        intervals = len(writer.interval_rows)
        writer.close()
        print(
            f"[experiment worker] {task[0]}/{task[1]}: raw fragment ready "
            f"({pixels:,} pixels, {intervals} intervals); reducing curves",
            flush=True,
        )
        curve_bins = add_fragment_well_curves(fragment_path, "experiment")
        print(
            f"[experiment worker] {task[0]}/{task[1]}: ready "
            f"({curve_bins:,} compact bins)",
            flush=True,
        )
    except BaseException:
        writer.store.close()
        raise
    return fragment_path, task[0], task[1], pixels, intervals, missing_pairs


def append_experiments(writer: StoreWriter) -> None:
    tasks = discover_experiment_tasks()
    temp_root = Path(tempfile.mkdtemp(prefix="eiche_fischer_master_curve_experiment_"))
    print(
        f"[experiment] {len(tasks)} wells; {EXPERIMENT_WORKERS} workers; "
        f"local fragments in {temp_root}",
        flush=True,
    )
    completed = 0
    missing_pairs = 0
    try:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=EXPERIMENT_WORKERS,
                mp_context=context,
        ) as executor:
            futures = []
            for index, task in enumerate(tasks):
                fragment_path = temp_root / f"{index:03d}_{task[0]}_{task[1]}.h5"
                futures.append(executor.submit(
                    write_experiment_fragment,
                    task,
                    fragment_path,
                ))
            for future in concurrent.futures.as_completed(futures):
                fragment_path, plate, well, pixels, intervals, missing = future.result()
                merge_simulation_fragment(writer, fragment_path)
                completed += 1
                missing_pairs += missing
                print(
                    f"[experiment] {completed}/{len(tasks)} {plate}/{well}; "
                    f"{pixels:,} pixels, {intervals} intervals; "
                    f"{writer.pixel_rows:,} total pixels",
                    flush=True,
                )
    finally:
        try:
            temp_root.rmdir()
        except OSError:
            pass
    if missing_pairs:
        print(f"[experiment] {missing_pairs} missing density pairs", flush=True)


def append_simulation_run(
        writer: StoreWriter,
        source: str,
        plan: object,
        escape_days: dict[tuple[str, str], float],
) -> None:
    sim_args = core.sweep.simulation_args(
        os.fspath(SIMULATION_DATABASE),
        source,
        plan.run_id,
    )
    loaded = [
        core.sim.load_filtered_cells(item.path, sim_args)
        for item in plan.files
    ]
    if len(loaded) < 2:
        return

    canvas = core.sweep.canvas_from_loaded(loaded, float(core.KERNEL_UM))
    snapshots = {
        float(item.day): core.simulation_density(positions, types, canvas)
        for item, (positions, types) in zip(plan.files, loaded)
    }
    days = sorted(snapshots)
    key = (
        core.canonical_strategy_label(plan.strategy),
        core.base.normalize_simulation_run_id(plan.run_id),
    )
    escape_day = escape_days.get(key)
    # Follow the experimental pre-escape convention: the first escape-day
    # snapshot is excluded, so an interval cannot end at or begin from it.
    upper_day = (
        max(days)
        if escape_day is None
        else max((day for day in days if day < escape_day), default=-np.inf)
    )
    strategy = core.canonical_strategy_label(
        core.sweep.SOURCE_TO_DISPLAY[source]
    )
    pair_sets = {
        "daily": tuple(zip(days, days[1:])),
        "experimental_schedule": core.EXPERIMENTAL_INTERVAL_DAY_PAIRS,
    }
    for interval_mode, pairs in pair_sets.items():
        for start_day, end_day in pairs:
            start_day, end_day = float(start_day), float(end_day)
            if start_day not in snapshots or end_day not in snapshots:
                continue
            resistant_t0, total_t0 = snapshots[start_day]
            resistant_t1, total_t1 = snapshots[end_day]
            frame, summary = interval_frame(
                dataset="simulation",
                strategy=strategy,
                replicate=f"{source}/{plan.run_id}",
                interval_mode=interval_mode,
                start_day=start_day,
                end_day=end_day,
                before_escape=bool(end_day <= upper_day),
                analysis_allowed=True,
                total_t0=total_t0,
                resistant_t0=resistant_t0,
                total_t1=total_t1,
                resistant_t1=resistant_t1,
            )
            writer.append(frame, summary)


def write_simulation_fragment(
        source: str,
        plan: object,
        escape_days: dict[tuple[str, str], float],
        fragment_path: Path,
) -> tuple[Path, str, str, int, int]:
    """Build one local, self-contained HDF5 fragment for a simulation run."""
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[simulation worker] {source}/{plan.run_id}: reading snapshots", flush=True)
    writer = StoreWriter(fragment_path, write_provenance=False)
    try:
        append_simulation_run(writer, source, plan, escape_days)
        pixels = writer.pixel_rows
        intervals = len(writer.interval_rows)
        writer.close()
        print(
            f"[simulation worker] {source}/{plan.run_id}: raw fragment ready "
            f"({pixels:,} pixels, {intervals} intervals); reducing curves",
            flush=True,
        )
        curve_bins = add_fragment_well_curves(fragment_path, "simulation")
        print(
            f"[simulation worker] {source}/{plan.run_id}: ready "
            f"({curve_bins:,} compact bins)",
            flush=True,
        )
    except BaseException:
        writer.store.close()
        raise
    return fragment_path, source, plan.run_id, pixels, intervals


def merge_simulation_fragment(writer: StoreWriter, fragment_path: Path) -> int:
    """Append one completed local fragment to the final shared-drive store."""
    with pd.HDFStore(fragment_path, mode="r") as fragment:
        if "/pixels" in fragment.keys():
            for frame in fragment.select("pixels", chunksize=MERGE_CHUNK_ROWS):
                writer.append_pixels(frame)
        if "/intervals" in fragment.keys():
            intervals = fragment["intervals"]
            writer.interval_rows.extend(intervals.to_dict("records"))
        if "/well_curves" in fragment.keys():
            well_curves = fragment["well_curves"]
            writer.well_curve_rows.extend(well_curves.to_dict("records"))
    fragment_path.unlink()
    return len(writer.interval_rows)


def append_simulations(writer: StoreWriter) -> None:
    escape_days = core.simulation_escape_days(
        COMBINED_RESULTS_CSV,
        core.DEFAULT_ESCAPE_MARGIN_CSV,
    )
    tasks: list[tuple[str, object]] = []
    for source in SIMULATION_SOURCES:
        plans = core.sim.discover_run_plans(
            core.sweep.simulation_args(os.fspath(SIMULATION_DATABASE), source)
        )
        tasks.extend((source, plan) for plan in plans)

    temp_root = Path(
        tempfile.mkdtemp(prefix="eiche_fischer_master_curve_simulation_")
    )
    print(
        f"[simulation] {len(tasks)} runs; {SIMULATION_WORKERS} workers; "
        f"local fragments in {temp_root}",
        flush=True,
    )
    completed = 0
    try:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=SIMULATION_WORKERS,
                mp_context=context,
        ) as executor:
            futures = []
            for index, (source, plan) in enumerate(tasks):
                fragment_path = temp_root / (
                    f"{index:03d}_{source}_{plan.run_id}.h5"
                )
                futures.append(executor.submit(
                    write_simulation_fragment,
                    source,
                    plan,
                    escape_days,
                    fragment_path,
                ))

            for future in concurrent.futures.as_completed(futures):
                fragment_path, source, run_id, pixels, intervals = future.result()
                merge_simulation_fragment(writer, fragment_path)
                completed += 1
                print(
                    f"[simulation] {completed}/{len(tasks)} {source}/{run_id}; "
                    f"{pixels:,} pixels, {intervals} intervals; "
                    f"{writer.pixel_rows:,} total pixels",
                    flush=True,
                )
    finally:
        try:
            temp_root.rmdir()
        except OSError:
            pass


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Move or remove it before rebuilding."
        )
    if BUILDING_PATH.exists():
        raise FileExistsError(
            f"Interrupted build found at {BUILDING_PATH}. "
            "Move or remove it before rebuilding."
        )
    writer = StoreWriter(BUILDING_PATH)
    try:
        append_experiments(writer)
        append_simulations(writer)
        writer.close()
    except BaseException:
        writer.store.close()
        raise
    BUILDING_PATH.replace(OUTPUT_PATH)
    print(
        f"[saved] {OUTPUT_PATH} "
        f"({writer.pixel_rows:,} pixels, {len(writer.interval_rows):,} intervals)"
    )


if __name__ == "__main__":
    main()
