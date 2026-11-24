#!/usr/bin/env python3
"""Prepare simulation cell point clouds for shared downstream analyses.

Purpose
-------
Raw simulation snapshots contain all model cells in native simulation units.
This script performs the shared, non-density preprocessing exactly once:

1. discover ``data-DB_<strategy>/outputs/run_<n>/simulation_<timestep>.h5``;
2. retain active cells (``cell_flags == 1`` by default);
3. retain known sensitive/resistant types and split them into two channels;
4. apply the established radial/local-neighbour outlier filter;
5. convert native coordinates to experimental pixel coordinates;
6. map timesteps to analysis days; and
7. write one atomic ``*_simulation.h5`` point-cloud file per simulation run.

Inputs (read-only)
------------------
``<EICHE_FISCHER_DATA_ROOT>/sim_raw_data/data-DB``

Outputs
-------
``intermediate_analysis_data/simulation/basic/prepared_pointclouds/``

The output layout is::

    Plate_<strategy>/hdf5/<run-id>_<strategy>-<run-id>_simulation.h5
    simulation_pointcloud_run_manifest.tsv
    Prepare_Simulation_Pointclouds_summary.csv

Each HDF5 contains ``/sus/d_XXX/block0_values`` and
``/res/d_XXX/block0_values``. Columns follow the historical experimental HDF
convention: column 0 is y and column 1 is x. Root attributes record the full
preprocessing configuration and source run.

The simulation overview uses this same preparation and saves a companion
``*.overview.json`` with raw/filtered counts and treatment state. Both stages
reuse completed runs; their expensive spatial filtering is performed once.

Downstream consumers
--------------------
* ``Escape_Margin_Analysis.py`` reads these point clouds directly.
* ``Density_Maps_Epanechnikov_simulations_coarsegrained.py`` reads the same
  point clouds to make density TIFFs and density-change maps.

Parallelism
-----------
``--workers N`` processes N independent simulation runs concurrently. The
PyCharm default is 4, a safe balance between CPU utilisation and shared-drive
I/O. Progress is shown per completed simulation run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
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

# Set True to replace prepared simulation point-cloud outputs from PyCharm.
OVERWRITE = False
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional presentation dependency
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (  # noqa: E402
    PIXEL_SIZE_UM,
)
from shared.data_paths import SIMULATION_DATABASE, intermediate_dir  # noqa: E402


DEFAULT_INPUT_ROOT = SIMULATION_DATABASE
DEFAULT_OUTPUT_ROOT = intermediate_dir(
    "simulation", "basic", "prepared_pointclouds", create=False
)
POINTCLOUD_MANIFEST = "simulation_pointcloud_run_manifest.tsv"
SUMMARY_CSV = "Prepare_Simulation_Pointclouds_summary.csv"
PREPARED_SUFFIX = "_simulation.h5"
PREPARATION_VERSION = 1

DEFAULT_TIME_DUMP = 100_000
DEFAULT_DAY_OFFSET = 1
DEFAULT_ACTIVE_FLAG = 1
DEFAULT_SENSITIVE_VALUE = 0
DEFAULT_RESISTANT_VALUE = 1
DEFAULT_TYPE_DATASET = "cell_type"
DEFAULT_POSITION_SCALE_PX = 11.89

# Original experiment_fitting/scripts/metadata.py parameters, converted from
# experimental pixels to native simulation units with the metadata area scale.
FILTER_A_MAX_EXPERIMENT_PX2 = 2591.0
FILTER_A_MAX_SIMULATION = 18.34
DEFAULT_FILTER_SIMULATION_SCALE = math.sqrt(
    FILTER_A_MAX_EXPERIMENT_PX2 / FILTER_A_MAX_SIMULATION
)
DEFAULT_FILTER_CENTER = (0.0, 0.0)
DEFAULT_FILTER_OUTLIER_BOUNDARY = 6000.0 / DEFAULT_FILTER_SIMULATION_SCALE
DEFAULT_FILTER_NEIGHBOUR_RADIUS = 360.0 / DEFAULT_FILTER_SIMULATION_SCALE
DEFAULT_FILTER_MIN_NEIGHBOURS = 5

SIM_FILE_RE = re.compile(r"^simulation_(?P<timestep>\d+)\.h5$")


@dataclass(frozen=True)
class SimulationFile:
    path: Path
    timestep: int
    day: int
    timepoint: str


@dataclass(frozen=True)
class RunPlan:
    strategy_dir: Path
    strategy: str
    run_dir: Path
    run_id: str
    plate: str
    well_identifier: str
    files: tuple[SimulationFile, ...]


@dataclass(frozen=True)
class PreparationSettings:
    active_flag: int
    type_dataset: str
    sensitive_value: int
    resistant_value: int
    position_scale_px: float
    outlier_filter: bool
    filter_center: tuple[float, float]
    filter_outlier_boundary: float
    filter_neighbour_radius: float
    filter_min_neighbours: int
    overwrite: bool


def default_settings():
    return PreparationSettings(
        DEFAULT_ACTIVE_FLAG, DEFAULT_TYPE_DATASET, DEFAULT_SENSITIVE_VALUE,
        DEFAULT_RESISTANT_VALUE, DEFAULT_POSITION_SCALE_PX, True,
        DEFAULT_FILTER_CENTER, DEFAULT_FILTER_OUTLIER_BOUNDARY,
        DEFAULT_FILTER_NEIGHBOUR_RADIUS, DEFAULT_FILTER_MIN_NEIGHBOURS, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare filtered simulation point-cloud HDF5 files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Read-only simulation database root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Prepared point-cloud root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Optional strategy filter, e.g. AT50 AT100 Agent.",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Optional run filter, e.g. run_0 run_10 or 0 10.",
    )
    parser.add_argument("--max-runs-per-strategy", type=int, default=0)
    parser.add_argument("--max-timepoints", type=int, default=0)
    parser.add_argument("--time-dump", type=int, default=DEFAULT_TIME_DUMP)
    parser.add_argument("--day-offset", type=int, default=DEFAULT_DAY_OFFSET)
    parser.add_argument("--active-flag", type=int, default=DEFAULT_ACTIVE_FLAG)
    parser.add_argument("--type-dataset", default=DEFAULT_TYPE_DATASET)
    parser.add_argument("--sensitive-value", type=int, default=DEFAULT_SENSITIVE_VALUE)
    parser.add_argument("--resistant-value", type=int, default=DEFAULT_RESISTANT_VALUE)
    parser.add_argument(
        "--position-scale-px",
        type=float,
        default=DEFAULT_POSITION_SCALE_PX,
        help="Experimental pixels per native simulation unit.",
    )
    parser.add_argument(
        "--position-scale-um",
        type=float,
        default=None,
        help="Optional um per native unit; overrides --position-scale-px.",
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="Disable the established radial/local-neighbour outlier filter.",
    )
    parser.add_argument(
        "--filter-center", nargs=2, type=float,
        default=list(DEFAULT_FILTER_CENTER), metavar=("X", "Y"),
    )
    parser.add_argument(
        "--filter-outlier-boundary",
        type=float,
        default=DEFAULT_FILTER_OUTLIER_BOUNDARY,
    )
    parser.add_argument(
        "--filter-neighbour-radius",
        type=float,
        default=DEFAULT_FILTER_NEIGHBOUR_RADIUS,
    )
    parser.add_argument(
        "--filter-min-neighbours",
        type=int,
        default=DEFAULT_FILTER_MIN_NEIGHBOURS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Independent simulation runs processed concurrently. Default: 4.",
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                        help="Replace existing prepared point-cloud outputs.")
    return parser.parse_args()


def strategy_name_from_dir(path: Path) -> str:
    return path.name.removeprefix("data-DB_")


def normalize_filter_values(
    values: Iterable[str] | None, prefix: str
) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized.add(text)
        if not text.startswith(prefix):
            normalized.add(f"{prefix}{text}")
    return normalized


def path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)


def run_designator(run_name: str) -> str:
    suffix = run_name.split("_", 1)[-1]
    return f"R{suffix}" if suffix.isdigit() else path_token(run_name)


def simulation_files(
    run_dir: Path, time_dump: int, day_offset: int
) -> list[SimulationFile]:
    files: list[SimulationFile] = []
    for path in run_dir.glob("simulation_*.h5"):
        match = SIM_FILE_RE.match(path.name)
        if match is None:
            continue
        timestep = int(match.group("timestep"))
        day = timestep // int(time_dump) + int(day_offset)
        files.append(
            SimulationFile(
                path=path,
                timestep=timestep,
                day=day,
                timepoint=f"{day:03d}",
            )
        )
    return sorted(files, key=lambda item: item.timestep)


def discover_run_plans(args: argparse.Namespace) -> list[RunPlan]:
    input_root = Path(args.input_root).expanduser()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Simulation database root does not exist: {input_root}")
    strategy_filter = normalize_filter_values(args.strategies, "data-DB_")
    run_filter = normalize_filter_values(args.runs, "run_")
    plans: list[RunPlan] = []

    for strategy_dir in sorted(input_root.glob("data-DB_*")):
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_name_from_dir(strategy_dir)
        if (
            strategy_filter
            and strategy_dir.name not in strategy_filter
            and strategy not in strategy_filter
        ):
            continue
        outputs_dir = strategy_dir / "outputs"
        if not outputs_dir.is_dir():
            continue
        run_dirs = sorted(
            (path for path in outputs_dir.glob("run_*") if path.is_dir()),
            key=lambda path: (
                (0, int(path.name.split("_")[-1]))
                if path.name.split("_")[-1].isdigit()
                else (1, path.name)
            ),
        )
        if run_filter:
            run_dirs = [path for path in run_dirs if path.name in run_filter]
        if int(args.max_runs_per_strategy) > 0:
            run_dirs = run_dirs[: int(args.max_runs_per_strategy)]

        for run_dir in run_dirs:
            files = simulation_files(run_dir, args.time_dump, args.day_offset)
            if int(args.max_timepoints) > 0:
                files = files[: int(args.max_timepoints)]
            if len(files) < 2:
                print(
                    f"[warn] Skipping {strategy_dir.name}/{run_dir.name}; "
                    "need at least two timepoints."
                )
                continue
            designator = run_designator(run_dir.name)
            strategy_token = path_token(strategy)
            plans.append(
                RunPlan(
                    strategy_dir=strategy_dir,
                    strategy=strategy,
                    run_dir=run_dir,
                    run_id=run_dir.name,
                    plate=f"Plate_{strategy_token}",
                    well_identifier=f"{designator}_{strategy_token}-{designator}",
                    files=tuple(files),
                )
            )
    return plans


def _dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    if name in group:
        return group[name]
    quoted = f'"{name}"'
    if quoted in group:
        return group[quoted]
    raise KeyError(f"Dataset {name!r} not found in {group.file.filename}")


def load_active_cells(
    path: Path, settings: PreparationSettings, *, with_overview=False
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, mode="r") as handle:
        data = handle["data"]
        flags = np.asarray(_dataset(data, "cell_flags"))
        active = flags == int(settings.active_flag)
        nuclei = np.asarray(_dataset(data, "cell_nucleus"))[active, :2].astype(
            np.float64, copy=False
        )
        overview = None
        if with_overview:
            try:
                canonical_types = np.asarray(_dataset(data, 'cell_type'))[active]
                treatment = int(bool(_dataset(data, 'therapy_switch')[()]))
                overview = {'types': canonical_types, 'positions': nuclei, 'treatment': treatment}
            except KeyError:
                pass  # Point clouds alone do not require overview metadata.
        try:
            cell_type = (canonical_types if overview is not None and settings.type_dataset == 'cell_type'
                         else np.asarray(_dataset(data, settings.type_dataset))[active])
        except KeyError:
            if settings.type_dataset != "cell_id":
                print(
                    f"[warn] {settings.type_dataset!r} missing in {path}; "
                    "falling back to 'cell_id'."
                )
            cell_type = np.asarray(_dataset(data, "cell_id"))[active]

    classes = np.full(cell_type.shape, -1, dtype=np.int8)
    classes[cell_type == int(settings.sensitive_value)] = 0
    classes[cell_type == int(settings.resistant_value)] = 1
    known = classes >= 0
    if with_overview:
        return nuclei[known], classes[known], overview
    return nuclei[known], classes[known]


def outlier_mask(
    cells: np.ndarray,
    *,
    center: tuple[float, float],
    boundary: float,
    neighbour_radius: float,
    min_neighbours: int,
) -> np.ndarray:
    if cells.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(cells.shape[0], dtype=bool)
    if boundary > 0:
        distances = np.linalg.norm(cells - np.asarray(center, dtype=float), axis=1)
        keep &= distances <= float(boundary)
    if neighbour_radius > 0 and min_neighbours > 1 and np.any(keep):
        indices = np.flatnonzero(keep)
        radial_cells = cells[indices]
        counts = cKDTree(radial_cells).query_ball_point(
            radial_cells, r=float(neighbour_radius), return_length=True
        )
        dense = counts >= int(min_neighbours)
        updated = np.zeros_like(keep)
        updated[indices[dense]] = True
        keep = updated
    return keep


def load_filtered_points_px(
    path: Path, settings: PreparationSettings, *, with_overview=False
) -> tuple[np.ndarray, np.ndarray]:
    if with_overview:
        positions, classes, overview = load_active_cells(path, settings, with_overview=True)
    else:
        positions, classes = load_active_cells(path, settings)
    if settings.outlier_filter:
        keep = outlier_mask(
            positions,
            center=settings.filter_center,
            boundary=settings.filter_outlier_boundary,
            neighbour_radius=settings.filter_neighbour_radius,
            min_neighbours=settings.filter_min_neighbours,
        )
        positions = positions[keep]
        classes = classes[keep]
    if with_overview:
        counts = overview_counts(overview, classes, settings)
        return positions * float(settings.position_scale_px), classes, counts
    return positions * float(settings.position_scale_px), classes


def overview_counts(overview, filtered_classes, settings):
    if overview is None:
        return None
    types = overview['types']
    raw_sensitive, raw_resistant = int(np.count_nonzero(types == 0)), int(np.count_nonzero(types == 1))
    # The legacy overview includes unknown cell types as neighbours, while
    # point-cloud preparation excludes them. Preserve that exceptional case.
    if (raw_sensitive + raw_resistant != len(types)
            or not uses_overview_filter(settings)):
        defaults = default_settings()
        kept = outlier_mask(overview['positions'], center=defaults.filter_center,
                            boundary=defaults.filter_outlier_boundary,
                            neighbour_radius=defaults.filter_neighbour_radius,
                            min_neighbours=defaults.filter_min_neighbours)
        filtered_sensitive = int(np.count_nonzero(kept & (types == 0)))
        filtered_resistant = int(np.count_nonzero(kept & (types == 1)))
    else:
        filtered_sensitive = int(np.count_nonzero(filtered_classes == 0))
        filtered_resistant = int(np.count_nonzero(filtered_classes == 1))
    return [raw_sensitive, raw_resistant, filtered_sensitive, filtered_resistant, overview['treatment']]


def uses_overview_filter(settings):
    defaults = default_settings()
    names = ('active_flag', 'type_dataset', 'sensitive_value', 'resistant_value',
             'outlier_filter', 'filter_center', 'filter_outlier_boundary',
             'filter_neighbour_radius', 'filter_min_neighbours')
    return all(getattr(settings, name) == getattr(defaults, name) for name in names)


def source_signature(plan):
    records = [(os.fspath(item.path.resolve()), item.timestep, item.day,
                item.path.stat().st_size, item.path.stat().st_mtime_ns)
               for item in plan.files]
    return hashlib.sha256(json.dumps(records).encode('utf-8')).hexdigest()


def settings_signature(settings):
    values = asdict(settings)
    values.pop('overwrite')
    return json.loads(json.dumps(values))


def overview_cache_path(output_path):
    return output_path.with_suffix('.overview.json')


def read_overview_cache(output_path, plan, settings):
    path = overview_cache_path(output_path)
    if not path.is_file():
        return None
    try:
        cache = json.loads(path.read_text(encoding='utf-8'))
        stat = output_path.stat()
        if (cache['version'] == 1 and cache['sources'] == source_signature(plan)
                and cache['settings'] == settings_signature(settings)
                and cache['coordinates'] == [stat.st_size, stat.st_mtime_ns]
                and len(cache['counts']) == len(plan.files)):
            return cache['counts']
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def write_overview_cache(output_path, plan, settings, counts):
    stat = output_path.stat()
    cache = {'version': 1, 'sources': source_signature(plan),
             'settings': settings_signature(settings),
             'coordinates': [stat.st_size, stat.st_mtime_ns], 'counts': counts}
    path = overview_cache_path(output_path)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(cache), encoding='utf-8')
    os.replace(temporary, path)


def validate_run_cache(output_path, plan, settings):
    validate_prepared_h5(output_path, (item.timepoint for item in plan.files))
    with h5py.File(output_path, 'r') as handle:
        for item in plan.files:
            for channel in ('sus', 'res'):
                group = handle[f'{channel}/d_{item.timepoint}']
                if group.attrs.get('timestep') != item.timestep or group.attrs.get('day') != item.day:
                    raise ValueError(f'{output_path}: incompatible timepoints; use --overwrite to rebuild')
        for name, expected in asdict(settings).items():
            if name == 'overwrite':
                continue
            observed = handle.attrs.get(name)
            if observed is None or not np.array_equal(observed, expected):
                raise ValueError(f'{output_path}: incompatible {name}; use --overwrite to rebuild')
        recorded = handle.attrs.get('source_signature')
        if recorded is not None and recorded != source_signature(plan):
            raise ValueError(f'{output_path}: raw snapshots changed; use --overwrite to rebuild')
        if recorded is None and any(item.path.stat().st_mtime_ns > output_path.stat().st_mtime_ns
                                    for item in plan.files):
            raise ValueError(f'{output_path}: raw snapshots newer than coordinates; use --overwrite to rebuild')


def upgrade_overview_cache(output_path, plan, settings):
    """Add missing summary metadata to old coordinates without filtering again."""
    counts = []
    with h5py.File(output_path, 'r') as prepared:
        for item in plan.files:
            with h5py.File(item.path, 'r') as raw:
                data = raw['data']
                active = np.asarray(_dataset(data, 'cell_flags')) == settings.active_flag
                try:
                    types = np.asarray(_dataset(data, 'cell_type'))[active]
                    therapy = int(bool(_dataset(data, 'therapy_switch')[()]))
                except KeyError:
                    counts.append(None)
                    continue
                raw_sen, raw_res = int(np.count_nonzero(types == 0)), int(np.count_nonzero(types == 1))
                filtered_sen = prepared[f'sus/d_{item.timepoint}/block0_values'].shape[0]
                filtered_res = prepared[f'res/d_{item.timepoint}/block0_values'].shape[0]
                if raw_sen + raw_res != len(types) or not uses_overview_filter(settings):
                    overview = {'types': types, 'positions': np.asarray(_dataset(data, 'cell_nucleus'))[active, :2],
                                'treatment': therapy}
                    counts.append(overview_counts(overview, np.empty(0), settings))
                else:
                    counts.append([raw_sen, raw_res, filtered_sen, filtered_res, therapy])
    write_overview_cache(output_path, plan, settings, counts)
    return counts


def _pointcloud_values(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return points_xy[:, [1, 0]].astype(np.float32, copy=False)


def validate_prepared_h5(path: Path, expected_timepoints: Iterable[str]) -> None:
    expected_timepoints = tuple(expected_timepoints)
    with h5py.File(path, mode="r") as handle:
        if int(handle.attrs.get("preparation_version", -1)) != PREPARATION_VERSION:
            raise ValueError(f"Unexpected preparation version in {path}")
        for channel in ("sus", "res"):
            if channel not in handle:
                raise ValueError(f"Missing /{channel} in {path}")
            for timepoint in expected_timepoints:
                dataset_path = f"{channel}/d_{timepoint}/block0_values"
                if dataset_path not in handle:
                    raise ValueError(f"Missing /{dataset_path} in {path}")
                dataset = handle[dataset_path]
                if dataset.ndim != 2 or dataset.shape[1] != 2:
                    raise ValueError(
                        f"Unexpected shape at /{dataset_path}: {dataset.shape}"
                    )


def output_path_for_plan(plan: RunPlan, output_root: Path) -> Path:
    return output_root / plan.plate / "hdf5" / f"{plan.well_identifier}{PREPARED_SUFFIX}"


def prepare_run(
    plan: RunPlan, settings: PreparationSettings, output_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    output_path = output_path_for_plan(plan, output_root)
    row: dict[str, object] = {
        "plate": plan.plate,
        "well_identifier": plan.well_identifier,
        "strategy": plan.strategy,
        "run_id": plan.run_id,
        "source_run_dir": os.fspath(plan.run_dir),
        "prepared_hdf5_path": os.fspath(output_path),
        "n_timepoints": len(plan.files),
        "n_sensitive": 0,
        "n_resistant": 0,
        "status": "ok",
        "elapsed_s": 0.0,
        "error": "",
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        replacing_incomplete = False
        if output_path.is_file() and not settings.overwrite:
            try:
                validate_run_cache(output_path, plan, settings)
                if read_overview_cache(output_path, plan, settings) is None:
                    upgrade_overview_cache(output_path, plan, settings)
                row["status"] = "skipped_exists"
                return row
            except Exception as error:
                replacing_incomplete = True
                print(f'[rebuild] {plan.strategy}/{plan.run_id}: {error}', flush=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, mode="w") as handle:
            handle.attrs["preparation_version"] = PREPARATION_VERSION
            handle.attrs["plate"] = plan.plate
            handle.attrs["well_identifier"] = plan.well_identifier
            handle.attrs["strategy"] = plan.strategy
            handle.attrs["run_id"] = plan.run_id
            handle.attrs["source_run_dir"] = os.fspath(plan.run_dir)
            handle.attrs["coordinate_system"] = "experimental_pixels_unshifted"
            handle.attrs["position_scale_px"] = settings.position_scale_px
            handle.attrs["active_flag"] = settings.active_flag
            handle.attrs["type_dataset"] = settings.type_dataset
            handle.attrs["sensitive_value"] = settings.sensitive_value
            handle.attrs["resistant_value"] = settings.resistant_value
            handle.attrs["outlier_filter"] = settings.outlier_filter
            handle.attrs["filter_center"] = settings.filter_center
            handle.attrs["filter_outlier_boundary"] = settings.filter_outlier_boundary
            handle.attrs["filter_neighbour_radius"] = settings.filter_neighbour_radius
            handle.attrs["filter_min_neighbours"] = settings.filter_min_neighbours
            original_signature = source_signature(plan)
            handle.attrs['source_signature'] = original_signature

            overview_rows = []
            for sim_file in plan.files:
                positions_px, classes, counts = load_filtered_points_px(sim_file.path, settings, with_overview=True)
                overview_rows.append(counts)
                sensitive = positions_px[classes == 0]
                resistant = positions_px[classes == 1]
                row["n_sensitive"] = int(row["n_sensitive"]) + len(sensitive)
                row["n_resistant"] = int(row["n_resistant"]) + len(resistant)
                for channel, points in (("sus", sensitive), ("res", resistant)):
                    group = handle.require_group(f"{channel}/d_{sim_file.timepoint}")
                    group.attrs["day"] = sim_file.day
                    group.attrs["timestep"] = sim_file.timestep
                    group.attrs["source_file"] = os.fspath(sim_file.path)
                    group.create_dataset(
                        "block0_values",
                        data=_pointcloud_values(points),
                        compression="gzip",
                    )

        validate_prepared_h5(temporary, (item.timepoint for item in plan.files))
        if source_signature(plan) != original_signature:
            raise RuntimeError(f'Raw snapshots changed while preparing {plan.run_dir}')
        if output_path.exists() and not settings.overwrite and not replacing_incomplete:
            raise FileExistsError(f"Output appeared during processing: {output_path}")
        os.replace(temporary, output_path)
        write_overview_cache(output_path, plan, settings, overview_rows)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
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
    _atomic_write_table(table, output_root / SUMMARY_CSV)
    manifest_columns = [
        "plate",
        "well_identifier",
        "strategy",
        "run_id",
        "source_run_dir",
        "prepared_hdf5_path",
        "n_timepoints",
        "status",
    ]
    _atomic_write_table(
        table[manifest_columns], output_root / POINTCLOUD_MANIFEST, sep="\t"
    )


def run(args: argparse.Namespace) -> Path:
    if int(args.time_dump) <= 0:
        raise ValueError("--time-dump must be positive")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if float(args.position_scale_px) <= 0 and args.position_scale_um is None:
        raise ValueError("--position-scale-px must be positive")
    scale_px = (
        float(args.position_scale_um) / PIXEL_SIZE_UM
        if args.position_scale_um is not None
        else float(args.position_scale_px)
    )
    if scale_px <= 0:
        raise ValueError("Resolved position scale must be positive")

    plans = discover_run_plans(args)
    if not plans:
        raise RuntimeError("No simulation runs matched the requested filters")
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    settings = PreparationSettings(
        active_flag=int(args.active_flag),
        type_dataset=str(args.type_dataset),
        sensitive_value=int(args.sensitive_value),
        resistant_value=int(args.resistant_value),
        position_scale_px=scale_px,
        outlier_filter=not bool(args.no_outlier_filter),
        filter_center=(float(args.filter_center[0]), float(args.filter_center[1])),
        filter_outlier_boundary=float(args.filter_outlier_boundary),
        filter_neighbour_radius=float(args.filter_neighbour_radius),
        filter_min_neighbours=int(args.filter_min_neighbours),
        overwrite=bool(args.overwrite),
    )
    print(
        f"[info] runs={len(plans)}, workers={int(args.workers)}, "
        f"position_scale_px={scale_px:g}, outlier_filter={settings.outlier_filter}"
    )
    print(f"[info] writing prepared point clouds to: {output_root}")

    rows: list[dict[str, object]] = []
    if int(args.workers) == 1:
        iterator = plans
        if tqdm is not None:
            iterator = tqdm(plans, desc="Prepare simulation point clouds", unit="run")
        for plan in iterator:
            row = prepare_run(plan, settings, output_root)
            rows.append(row)
            print(
                f"[run] {plan.strategy}/{plan.run_id} status={row['status']} "
                f"timepoints={row['n_timepoints']} elapsed={float(row['elapsed_s']):.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(prepare_run, plan, settings, output_root): plan
                for plan in plans
            }
            completed = as_completed(futures)
            if tqdm is not None:
                completed = tqdm(
                    completed,
                    total=len(futures),
                    desc="Prepare simulation point clouds",
                    unit="run",
                )
            for future in completed:
                plan = futures[future]
                row = future.result()
                rows.append(row)
                print(
                    f"[run] {plan.strategy}/{plan.run_id} status={row['status']} "
                    f"timepoints={row['n_timepoints']} elapsed={float(row['elapsed_s']):.1f}s"
                )

    write_outputs(rows, output_root)
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    print(f"[save] {output_root / POINTCLOUD_MANIFEST}")
    print(f"[save] {output_root / SUMMARY_CSV}")
    print(f"[done] status counts: {counts}")
    if counts.get("error", 0):
        raise RuntimeError(
            f"{counts['error']} simulation run(s) failed; inspect {output_root / SUMMARY_CSV}"
        )
    return output_root / POINTCLOUD_MANIFEST


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
