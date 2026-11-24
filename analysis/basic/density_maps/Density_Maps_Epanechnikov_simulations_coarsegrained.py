#!/usr/bin/env python3
"""Create simulation density maps from prepared simulation point clouds.

This is the density-only stage of the simulation pipeline. It no longer reads
raw simulation snapshots and no longer performs active-cell selection,
outlier filtering, coordinate scaling, or point-cloud HDF5 creation. Those
shared operations belong to ``Prepare_Simulation_Pointclouds.py``.

Input
-----
The canonical prepared-point-cloud manifest and its HDF5 files::

    intermediate_analysis_data/simulation/basic/prepared_pointclouds/
        simulation_pointcloud_run_manifest.tsv
        Plate_<strategy>/hdf5/*_simulation.h5

Output
------
The established density layout is preserved for downstream figure code::

    intermediate_analysis_data/simulation/basic/density_maps/
        simulation_density_run_manifest.tsv
        simulation_density_run_summary.csv
        Plate_<strategy>/<well_identifier>/density_maps/
        Plate_<strategy>/<well_identifier>/density_changes/

For each day, prepared sensitive and resistant coordinates are shifted onto a
common padded canvas and converted to coarse-grained Epanechnikov density TIFFs.
Density-change maps are generated with the same helper and thresholds as the
experimental pipeline.

Parallelism
-----------
``--workers N`` processes N independent simulation runs concurrently. The
default is 1. A progress bar reports completed runs.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

# Set True to replace simulation density-map outputs from PyCharm.
OVERWRITE = True
import tifffile as tiff

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DENSITY_ZERO_FLOOR_UM2,
    PIXEL_SIZE_UM,
    density,
    density_tiff_description,
    load_config,
    output_type_name,
    resolve_coarsening_factor,
    resolve_radius_um,
    write_test_density_changes,
)
from analysis.basic.density_maps import Prepare_Simulation_Pointclouds as pointcloud_prep  # noqa: E402
from shared.data_paths import intermediate_dir, resolve_stored_data_path  # noqa: E402


DEFAULT_OUTPUT_ROOT = intermediate_dir(
    "simulation", "basic", "density_maps", create=False
)
DENSITY_MANIFEST = "simulation_density_run_manifest.tsv"
DENSITY_SUMMARY = "simulation_density_run_summary.csv"
FALLBACK_RADIUS_UM = 320.0
FALLBACK_COARSENING_FACTOR = 32
DEFAULT_DENSITY_MAPS_NAME = "density_maps"
DEFAULT_DENSITY_CHANGES_NAME = "density_changes"
TIF_COMPRESSION = "deflate"
DAY_RE = re.compile(r"^d_(\d+)$")

# Backward-compatible public names used by Figure 3's in-memory kernel-sweep
# utilities. Their implementation now lives in the shared preparation module.
DEFAULT_PREPARED_ROOT = pointcloud_prep.DEFAULT_OUTPUT_ROOT
POINTCLOUD_MANIFEST = pointcloud_prep.POINTCLOUD_MANIFEST
DEFAULT_TIME_DUMP = pointcloud_prep.DEFAULT_TIME_DUMP
DEFAULT_DAY_OFFSET = pointcloud_prep.DEFAULT_DAY_OFFSET
DEFAULT_ACTIVE_FLAG = pointcloud_prep.DEFAULT_ACTIVE_FLAG
DEFAULT_TYPE_DATASET = pointcloud_prep.DEFAULT_TYPE_DATASET
DEFAULT_SENSITIVE_VALUE = pointcloud_prep.DEFAULT_SENSITIVE_VALUE
DEFAULT_RESISTANT_VALUE = pointcloud_prep.DEFAULT_RESISTANT_VALUE
DEFAULT_POSITION_SCALE_PX = pointcloud_prep.DEFAULT_POSITION_SCALE_PX
DEFAULT_FILTER_CENTER = pointcloud_prep.DEFAULT_FILTER_CENTER
DEFAULT_FILTER_OUTLIER_BOUNDARY = pointcloud_prep.DEFAULT_FILTER_OUTLIER_BOUNDARY
DEFAULT_FILTER_NEIGHBOUR_RADIUS = pointcloud_prep.DEFAULT_FILTER_NEIGHBOUR_RADIUS
DEFAULT_FILTER_MIN_NEIGHBOURS = pointcloud_prep.DEFAULT_FILTER_MIN_NEIGHBOURS
SimulationFile = pointcloud_prep.SimulationFile
RunPlan = pointcloud_prep.RunPlan
discover_run_plans = pointcloud_prep.discover_run_plans


@dataclass(frozen=True)
class PreparedRun:
    plate: str
    well_identifier: str
    strategy: str
    run_id: str
    source_run_dir: str
    hdf5_path: Path
    timepoints: tuple[str, ...]


@dataclass(frozen=True)
class PixelCanvasSpec:
    min_x_px: float
    min_y_px: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class CanvasSpec:
    """Legacy um-origin canvas used by in-memory Figure 3 utilities."""

    min_x_um: float
    min_y_um: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class DensitySettings:
    radius_um: float
    coarsening_factor: int
    canvas_padding_um: float
    density_maps_name: str
    density_changes_name: str
    zero_initial_res_density_threshold_um2: float
    res_growth_threshold_um2: float
    max_timepoints: int
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create simulation density maps from prepared point clouds."
    )
    parser.add_argument("--config", default=os.fspath(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT,
        help=f"Prepared simulation point-cloud root. Default: {DEFAULT_PREPARED_ROOT}",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Optional manifest; default: <prepared-root>/simulation_pointcloud_run_manifest.tsv.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"Density output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--max-runs-per-strategy", type=int, default=0)
    parser.add_argument("--max-timepoints", type=int, default=0)
    parser.add_argument("--radius-um", type=float, default=None)
    parser.add_argument("--coarsening-factor", type=int, default=None)
    parser.add_argument("--canvas-padding-um", type=float, default=None)
    parser.add_argument("--density-maps-name", default=None)
    parser.add_argument("--density-changes-name", default=None)
    parser.add_argument(
        "--zero-initial-res-density-threshold-um2", type=float, default=None
    )
    parser.add_argument("--res-growth-threshold-um2", type=float, default=None)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Independent simulation runs processed concurrently. Default: 1.",
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                        help="Replace existing simulation density-map outputs.")
    return parser.parse_args()


def apply_config_defaults(args: argparse.Namespace, config: dict) -> None:
    if args.radius_um is None:
        args.radius_um = resolve_radius_um(config) if config else FALLBACK_RADIUS_UM
    if args.coarsening_factor is None:
        args.coarsening_factor = (
            resolve_coarsening_factor(config, float(args.radius_um))
            if config else FALLBACK_COARSENING_FACTOR
        )
    if args.canvas_padding_um is None:
        args.canvas_padding_um = float(args.radius_um)
    if args.density_maps_name is None:
        args.density_maps_name = output_type_name(
            config, "density_maps_output_type", DEFAULT_DENSITY_MAPS_NAME
        )
    if args.density_changes_name is None:
        args.density_changes_name = output_type_name(
            config, "density_changes_output_type", DEFAULT_DENSITY_CHANGES_NAME
        )
    if args.zero_initial_res_density_threshold_um2 is None:
        args.zero_initial_res_density_threshold_um2 = float(
            config.get("zero_initial_res_density_threshold_um2", DENSITY_ZERO_FLOOR_UM2)
            if config else DENSITY_ZERO_FLOOR_UM2
        )
    if args.res_growth_threshold_um2 is None:
        args.res_growth_threshold_um2 = float(
            config.get("res_growth_threshold_um2", DENSITY_ZERO_FLOOR_UM2)
            if config else DENSITY_ZERO_FLOOR_UM2
        )


def _normalized_filters(values: Iterable[str] | None, prefix: str) -> set[str]:
    normalized: set[str] = set()
    for value in values or ():
        text = str(value).strip()
        if text:
            normalized.add(text)
            if not text.startswith(prefix):
                normalized.add(f"{prefix}{text}")
    return normalized


def resolve_position_scale_um(args: argparse.Namespace) -> float:
    """Compatibility helper for raw-snapshot in-memory Figure 3 analyses."""
    if getattr(args, "position_scale_um", None) is not None:
        return float(args.position_scale_um)
    return float(args.position_scale_px) * PIXEL_SIZE_UM


def _legacy_preparation_settings(args: argparse.Namespace) -> pointcloud_prep.PreparationSettings:
    return pointcloud_prep.PreparationSettings(
        active_flag=int(args.active_flag),
        type_dataset=str(args.type_dataset),
        sensitive_value=int(args.sensitive_value),
        resistant_value=int(args.resistant_value),
        position_scale_px=resolve_position_scale_um(args) / PIXEL_SIZE_UM,
        outlier_filter=not bool(args.no_outlier_filter),
        filter_center=(float(args.filter_center[0]), float(args.filter_center[1])),
        filter_outlier_boundary=float(args.filter_outlier_boundary),
        filter_neighbour_radius=float(args.filter_neighbour_radius),
        filter_min_neighbours=int(args.filter_min_neighbours),
        overwrite=False,
    )


def load_filtered_cells(
    path: Path, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning filtered positions in um.

    The shared filtering/scaling implementation is owned by
    ``Prepare_Simulation_Pointclouds.py``. This wrapper keeps existing
    in-memory Figure 3 kernel sweeps working without duplicating that logic.
    """
    points_px, classes = pointcloud_prep.load_filtered_points_px(
        Path(path), _legacy_preparation_settings(args)
    )
    return points_px * PIXEL_SIZE_UM, classes


def positions_to_density_inputs(
    positions_um: np.ndarray,
    type_values: np.ndarray,
    canvas: CanvasSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility conversion used by in-memory Figure 3 analyses."""
    x_px = (positions_um[:, 0] - canvas.min_x_um) / PIXEL_SIZE_UM
    y_px = (positions_um[:, 1] - canvas.min_y_um) / PIXEL_SIZE_UM
    points = pd.DataFrame({"x": x_px, "y": y_px, "type": type_values})
    sensitive = points[points["type"] == 0][["x", "y"]].copy()
    resistant = points[points["type"] == 1][["x", "y"]].copy()
    return sensitive, resistant


def timepoints_in_h5(path: Path) -> tuple[str, ...]:
    with h5py.File(path, mode="r") as handle:
        if "sus" not in handle or "res" not in handle:
            raise ValueError(f"Prepared HDF5 lacks /sus or /res: {path}")
        common = set(handle["sus"]).intersection(handle["res"])
    parsed: list[tuple[int, str]] = []
    for key in common:
        match = DAY_RE.match(key)
        if match:
            parsed.append((int(match.group(1)), match.group(1)))
    return tuple(value for _, value in sorted(parsed))


def discover_prepared_runs(args: argparse.Namespace) -> list[PreparedRun]:
    prepared_root = Path(args.prepared_root).expanduser()
    manifest = (
        Path(args.manifest).expanduser()
        if args.manifest else prepared_root / POINTCLOUD_MANIFEST
    )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Prepared simulation manifest does not exist: {manifest}. "
            "Run Prepare_Simulation_Pointclouds.py first."
        )
    table = pd.read_csv(manifest, sep="\t")
    required = {
        "plate", "well_identifier", "strategy", "run_id", "source_run_dir",
        "prepared_hdf5_path", "status",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Point-cloud manifest is missing columns: {sorted(missing)}")

    strategy_filter = _normalized_filters(args.strategies, "data-DB_")
    run_filter = _normalized_filters(args.runs, "run_")
    runs: list[PreparedRun] = []
    for record in table.itertuples(index=False):
        if str(record.status) not in {"ok", "skipped_exists"}:
            continue
        strategy = str(record.strategy)
        if (
            strategy_filter and strategy not in strategy_filter
            and f"data-DB_{strategy}" not in strategy_filter
        ):
            continue
        run_id = str(record.run_id)
        if run_filter and run_id not in run_filter:
            continue
        path = resolve_stored_data_path(str(record.prepared_hdf5_path))
        if not path.is_file():
            raise FileNotFoundError(f"Prepared point-cloud HDF5 is missing: {path}")
        timepoints = timepoints_in_h5(path)
        if len(timepoints) < 2:
            raise ValueError(f"Need at least two timepoints in {path}")
        runs.append(
            PreparedRun(
                plate=str(record.plate),
                well_identifier=str(record.well_identifier),
                strategy=strategy,
                run_id=run_id,
                source_run_dir=str(resolve_stored_data_path(str(record.source_run_dir))),
                hdf5_path=path,
                timepoints=timepoints,
            )
        )
    runs.sort(key=lambda item: (item.strategy, item.run_id))
    if int(args.max_runs_per_strategy) > 0:
        limited: list[PreparedRun] = []
        for strategy in sorted({item.strategy for item in runs}):
            group = [item for item in runs if item.strategy == strategy]
            limited.extend(group[: int(args.max_runs_per_strategy)])
        runs = limited
    return runs


def load_xy(handle: h5py.File, channel: str, timepoint: str) -> np.ndarray:
    path = f"{channel}/d_{timepoint}/block0_values"
    if path not in handle:
        return np.empty((0, 2), dtype=float)
    values = np.asarray(handle[path])
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Unexpected point-cloud shape at /{path}: {values.shape}")
    return values[:, [1, 0]].astype(np.float64, copy=False)


def canvas_for_run(run: PreparedRun, settings: DensitySettings) -> PixelCanvasSpec:
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    selected = run.timepoints[: settings.max_timepoints or None]
    with h5py.File(run.hdf5_path, mode="r") as handle:
        for timepoint in selected:
            for channel in ("sus", "res"):
                points = load_xy(handle, channel, timepoint)
                if points.size:
                    minima.append(np.nanmin(points, axis=0))
                    maxima.append(np.nanmax(points, axis=0))
    if not minima:
        raise RuntimeError(f"No prepared cells found in {run.hdf5_path}")
    padding_px = float(settings.canvas_padding_um) / PIXEL_SIZE_UM
    minimum = np.nanmin(np.vstack(minima), axis=0) - padding_px
    maximum = np.nanmax(np.vstack(maxima), axis=0) + padding_px
    span = np.maximum(maximum - minimum, 1.0)
    return PixelCanvasSpec(
        min_x_px=float(minimum[0]), min_y_px=float(minimum[1]),
        width_px=int(math.ceil(span[0])), height_px=int(math.ceil(span[1])),
    )


def points_on_canvas(points: np.ndarray, canvas: PixelCanvasSpec) -> pd.DataFrame:
    if points.size == 0:
        return pd.DataFrame({"x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
    return pd.DataFrame(
        {"x": points[:, 0] - canvas.min_x_px, "y": points[:, 1] - canvas.min_y_px}
    )


def _atomic_tiff(path: Path, image: np.ndarray, description: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        tiff.imwrite(
            temporary, image.astype(np.float32, copy=False),
            description=description, compression=TIF_COMPRESSION,
        )
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def create_density_run(
    run: PreparedRun, settings: DensitySettings, output_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir = output_root / run.plate / run.well_identifier
    maps_dir = output_dir / settings.density_maps_name
    changes_dir = output_dir / settings.density_changes_name
    row: dict[str, object] = {
        "plate": run.plate, "well_identifier": run.well_identifier,
        "strategy": run.strategy, "run_id": run.run_id,
        "source_run_dir": run.source_run_dir,
        "prepared_hdf5_path": os.fspath(run.hdf5_path),
        "n_timepoints": 0, "n_maps_written": 0, "status": "ok",
        "elapsed_s": 0.0, "error": "",
    }
    try:
        timepoints = run.timepoints[: settings.max_timepoints or None]
        pointcloud_prep.validate_prepared_h5(run.hdf5_path, timepoints)
        maps_dir.mkdir(parents=True, exist_ok=True)
        changes_dir.mkdir(parents=True, exist_ok=True)
        canvas = canvas_for_run(run, settings)
        description_base = density_tiff_description(
            radius_um=settings.radius_um,
            coarsening_factor=settings.coarsening_factor,
        )
        print(
            f"[{run.strategy}/{run.run_id}] canvas={canvas.height_px}x{canvas.width_px}px "
            f"origin_px=({canvas.min_x_px:.3f}, {canvas.min_y_px:.3f})"
        )
        with h5py.File(run.hdf5_path, mode="r") as handle:
            position_scale_px = float(handle.attrs.get("position_scale_px", np.nan))
            for timepoint in timepoints:
                base = f"{run.well_identifier}_{timepoint}"
                expected = (
                    maps_dir / f"{base}_sus_density_epanechnikov.tif",
                    maps_dir / f"{base}_res_density_epanechnikov.tif",
                    maps_dir / f"{base}_total_density_epanechnikov.tif",
                )
                if not settings.overwrite and all(path.is_file() for path in expected):
                    continue
                sensitive = points_on_canvas(load_xy(handle, "sus", timepoint), canvas)
                resistant = points_on_canvas(load_xy(handle, "res", timepoint), canvas)
                sus_density, res_density, total_density = density(
                    sensitive, resistant, radius_um=settings.radius_um,
                    image_shape=(canvas.height_px, canvas.width_px),
                    coarsening_factor=settings.coarsening_factor,
                )
                group = handle[f"sus/d_{timepoint}"]
                timestep = int(group.attrs.get("timestep", -1))
                day = int(group.attrs.get("day", int(timepoint)))
                description = (
                    f"{description_base}; source=prepared_simulation_pointcloud; "
                    f"timestep={timestep}; simulation_day={day}; "
                    f"position_scale_px={position_scale_px}; "
                    f"origin_x_px={canvas.min_x_px}; origin_y_px={canvas.min_y_px}"
                )
                for path, image in zip(
                    expected, (sus_density, res_density, total_density), strict=True
                ):
                    _atomic_tiff(path, image, description)
                    row["n_maps_written"] = int(row["n_maps_written"]) + 1
                print(
                    f"[{run.strategy}/{run.run_id}] wrote day {day:03d}: "
                    f"sus={len(sensitive):,} res={len(resistant):,}"
                )

        write_test_density_changes(
            directory_path=output_dir,
            well_identifier=run.well_identifier,
            timepoints=list(timepoints),
            density_maps_folder=settings.density_maps_name,
            density_changes_folder=settings.density_changes_name,
            channels=("sus", "res", "total"),
            write_zero_initial_growth_channel=True,
            zero_initial_density_threshold=settings.zero_initial_res_density_threshold_um2,
            measurable_growth_threshold=settings.res_growth_threshold_um2,
        )
        row["n_timepoints"] = len(timepoints)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_s"] = float(time.perf_counter() - started)
    return row


def _atomic_write_table(table: pd.DataFrame, path: Path, *, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    table.to_csv(temporary, index=False, sep=sep, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def write_outputs(rows: list[dict[str, object]], output_root: Path) -> None:
    table = pd.DataFrame(rows).sort_values(["plate", "run_id"]).reset_index(drop=True)
    _atomic_write_table(table, output_root / DENSITY_SUMMARY)
    columns = [
        "plate", "well_identifier", "strategy", "run_id", "source_run_dir",
        "prepared_hdf5_path", "n_timepoints", "status",
    ]
    _atomic_write_table(table[columns], output_root / DENSITY_MANIFEST, sep="\t")


def run(args: argparse.Namespace) -> Path:
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    config = load_config(args.config)
    apply_config_defaults(args, config)
    runs = discover_prepared_runs(args)
    if not runs:
        raise RuntimeError("No prepared simulation runs matched the requested filters")
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    settings = DensitySettings(
        radius_um=float(args.radius_um),
        coarsening_factor=int(args.coarsening_factor),
        canvas_padding_um=float(args.canvas_padding_um),
        density_maps_name=str(args.density_maps_name),
        density_changes_name=str(args.density_changes_name),
        zero_initial_res_density_threshold_um2=float(
            args.zero_initial_res_density_threshold_um2
        ),
        res_growth_threshold_um2=float(args.res_growth_threshold_um2),
        max_timepoints=max(0, int(args.max_timepoints)),
        overwrite=bool(args.overwrite),
    )
    print(
        f"[info] runs={len(runs)}, workers={int(args.workers)}, "
        f"radius_um={settings.radius_um:g}, coarsening_factor={settings.coarsening_factor}"
    )
    print(f"[info] reading prepared point clouds from: {Path(args.prepared_root)}")
    print(f"[info] writing density outputs to: {output_root}")

    rows: list[dict[str, object]] = []
    if int(args.workers) == 1:
        iterator = runs
        if tqdm is not None:
            iterator = tqdm(runs, desc="Simulation density maps", unit="run")
        for prepared_run in iterator:
            row = create_density_run(prepared_run, settings, output_root)
            rows.append(row)
            print(
                f"[run] {prepared_run.strategy}/{prepared_run.run_id} "
                f"status={row['status']} timepoints={row['n_timepoints']} "
                f"elapsed={float(row['elapsed_s']):.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(create_density_run, item, settings, output_root): item
                for item in runs
            }
            completed = as_completed(futures)
            if tqdm is not None:
                completed = tqdm(
                    completed, total=len(futures),
                    desc="Simulation density maps", unit="run",
                )
            for future in completed:
                prepared_run = futures[future]
                row = future.result()
                rows.append(row)
                print(
                    f"[run] {prepared_run.strategy}/{prepared_run.run_id} "
                    f"status={row['status']} timepoints={row['n_timepoints']} "
                    f"elapsed={float(row['elapsed_s']):.1f}s"
                )

    write_outputs(rows, output_root)
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    print(f"[save] {output_root / DENSITY_MANIFEST}")
    print(f"[save] {output_root / DENSITY_SUMMARY}")
    print(f"[done] status counts: {counts}")
    if counts.get("error", 0):
        raise RuntimeError(
            f"{counts['error']} density run(s) failed; inspect {output_root / DENSITY_SUMMARY}"
        )
    return output_root / DENSITY_MANIFEST


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
