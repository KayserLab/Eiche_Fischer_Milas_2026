#!/usr/bin/env python3
"""Rebuild a daily, per-run summary from the raw simulation HDF5 dumps.

The expected input layout is::

    <input-root>/data-DB_<protocol>/outputs/run_<run>/simulation_<timestep>.h5

Only living/active cells (``cell_flags == 1``) are counted.  Cell type 0 is
sensitive and cell type 1 is resistant, which is the same selection used by
the repository's simulation density-map analysis.  Dump 0 is exported as day
1 so that the output follows the historical ``combined_results.csv`` calendar.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_paths import COMBINED_RESULTS_CSV, SIMULATION_DATABASE
from analysis.basic.density_maps import Prepare_Simulation_Pointclouds as pointclouds


DEFAULT_INPUT_ROOT = SIMULATION_DATABASE
DEFAULT_OUTPUT = COMBINED_RESULTS_CSV
# Set to True when running this file with PyCharm's play button to replace an
# existing summary.  The command-line --overwrite option does the same.
OVERWRITE = False
TIME_DUMP = 100_000
# Original experiment_fitting/scripts/metadata.py values are specified in
# experimental pixels.  The HDF5 coordinates used here are simulation units.
FILTER_A_MAX_EXPERIMENT_PX2 = 2591.0
FILTER_A_MAX_SIMULATION = 18.34
FILTER_SIMULATION_SCALE = np.sqrt(FILTER_A_MAX_EXPERIMENT_PX2 / FILTER_A_MAX_SIMULATION)
FILTER_CENTER = np.asarray((0.0, 0.0))
FILTER_OUTLIER_BOUNDARY = 6000.0 / FILTER_SIMULATION_SCALE
FILTER_NEIGHBOUR_RADIUS = 360.0 / FILTER_SIMULATION_SCALE
FILTER_MIN_NEIGHBOURS = 5

# Transcribed from Data_Maja/initial_condition_DB/mapping.png and used by the
# Figure 4 DPD/in-vitro matching analysis.
RUN_TO_INITIAL_CONDITION = {
    10: "A1_AT50-A1", 11: "B2_AT50-B2", 12: "B6_AT50-B6",
    13: "C2_AT50-C2", 14: "C5_AT50-C5", 15: "D1_AT50-D1",
    16: "D3_AT50-D3", 17: "D4_AT50-D4", 18: "D5_AT50-D5",
    19: "D6_AT50-D6", 20: "A2_AT100-A2", 21: "A5_AT100-A5",
    22: "B3_AT100-B3", 23: "B5_AT100-B5", 24: "B6_AT100-B6",
    25: "C1_AT100-C1", 26: "C2_AT100-C2", 27: "C4_AT100-C4",
    28: "C6_AT100-C6", 29: "D4_AT100-D4", 30: "A1_CT-A1",
    31: "A3_CT-A3", 32: "B2_CT-B2", 33: "B3_CT-B3",
    34: "C1_CT-C1", 35: "C2_CT-C2", 36: "C4_CT-C4",
    37: "C5_CT-C5", 38: "D2_CT-D2", 39: "D3_CT-D3",
    40: "D4_CT-D4", 41: "D5_CT-D5", 42: "B2_Dens-B2",
    43: "B3_Dens-B3", 44: "B4_Dens-B4", 45: "C2_Dens-C2",
    46: "C3_Dens-C3", 47: "C4_Dens-C4", 48: "D3_Dens-D3",
    49: "D5_Dens-D5",
}

COLUMNS = [
    "protocol", "run", "initial_condition", "day", "TTF",
    "res_cell_count", "sen_cell_count", "treatment",
    "normalized_tot", "normalized_sus", "normalized_res",
    "filtered_res_cell_count", "filtered_sen_cell_count",
    "filtered_normalized_tot", "filtered_normalized_sus", "filtered_normalized_res",
]
SIMULATION_RE = re.compile(r"^simulation_(\d+)\.h5$")


@dataclass(frozen=True)
class Dump:
    path: Path
    timestep: int


@dataclass(frozen=True)
class RunPlan:
    protocol: str
    run: str
    initial_condition: str
    dumps: list[Dump]


def dataset(data: h5py.Group, name: str) -> h5py.Dataset:
    """Return a hickle dataset whether its key is quoted or not."""
    for key in (name, f'"{name}"'):
        if key in data:
            return data[key]
    raise KeyError(f"{name!r} not found in {data.file.filename}")


def initial_condition_for_run(run: str) -> str:
    numeric = re.fullmatch(r"\d+", run)
    if numeric:
        run_number = int(run)
        if run_number < 10:
            return "artificial"
        return RUN_TO_INITIAL_CONDITION.get(run_number, "unknown")
    # Non-treatment controls use folders such as run_B1; their initial well is
    # encoded directly in the run name.
    return run


def run_sort_key(run: object) -> tuple[int, int, str]:
    """Sort numeric run labels numerically, then retain stable text-run ordering."""
    label = str(run)
    if label.isdigit():
        return 0, int(label), label
    return 1, 0, label


def discover_runs(input_root: Path, protocol_filter: set[str] | None) -> list[RunPlan]:
    plans: list[RunPlan] = []
    for strategy_dir in sorted(input_root.glob("data-DB_*")):
        if not strategy_dir.is_dir():
            continue
        protocol = strategy_dir.name.removeprefix("data-DB_")
        if protocol_filter and protocol.upper() not in protocol_filter:
            continue
        outputs = strategy_dir / "outputs"
        for run_dir in sorted(outputs.glob("run_*"), key=lambda path: path.name):
            if not run_dir.is_dir():
                continue
            dumps = []
            for path in run_dir.glob("simulation_*.h5"):
                match = SIMULATION_RE.match(path.name)
                if match:
                    dumps.append(Dump(path, int(match.group(1))))
            if not dumps:
                continue
            run = run_dir.name.removeprefix("run_")
            plans.append(RunPlan(protocol, run, initial_condition_for_run(run), sorted(dumps, key=lambda dump: dump.timestep)))
    return plans


def spatial_filter_mask(positions: np.ndarray) -> np.ndarray:
    """Apply the shared simulation density-map outlier filter to cell positions."""
    if positions.size == 0:
        return np.zeros(0, dtype=bool)
    mask = np.linalg.norm(positions - FILTER_CENTER, axis=1) <= FILTER_OUTLIER_BOUNDARY
    kept = np.flatnonzero(mask)
    if kept.size:
        tree = cKDTree(positions[kept])
        neighbour_lists = tree.query_ball_point(
            positions[kept], r=FILTER_NEIGHBOUR_RADIUS
        )
        mask[kept] = np.fromiter(
            (len(neighbours) >= FILTER_MIN_NEIGHBOURS for neighbours in neighbour_lists),
            dtype=bool,
            count=kept.size,
        )
    return mask


def read_dump(dump: Dump) -> tuple[int, int, int, int, int]:
    """Return raw and spatially filtered counts plus the treatment state."""
    with h5py.File(dump.path, "r") as handle:
        data = handle["data"]
        flags = np.asarray(dataset(data, "cell_flags"))
        types = np.asarray(dataset(data, "cell_type"))
        active = flags == 1
        sensitive = int(np.count_nonzero(active & (types == 0)))
        resistant = int(np.count_nonzero(active & (types == 1)))
        positions = np.asarray(dataset(data, "cell_nucleus"))[active, :2]
        active_types = types[active]
        kept = spatial_filter_mask(positions)
        filtered_sensitive = int(np.count_nonzero(kept & (active_types == 0)))
        filtered_resistant = int(np.count_nonzero(kept & (active_types == 1)))
        treatment = int(bool(dataset(data, "therapy_switch")[()]))
    return sensitive, resistant, filtered_sensitive, filtered_resistant, treatment


def prepared_plan(plan: RunPlan, day_offset: int) -> pointclouds.RunPlan:
    run_dir = plan.dumps[0].path.parent
    strategy_dir = run_dir.parent.parent
    strategy = pointclouds.path_token(plan.protocol)
    designator = pointclouds.run_designator(run_dir.name)
    files = tuple(
        pointclouds.SimulationFile(dump.path, dump.timestep,
                                  dump.timestep // TIME_DUMP + day_offset,
                                  f'{dump.timestep // TIME_DUMP + day_offset:03d}')
        for dump in plan.dumps
    )
    return pointclouds.RunPlan(strategy_dir, plan.protocol, run_dir, run_dir.name,
                               f'Plate_{strategy}', f'{designator}_{strategy}-{designator}', files)


def rows_for_run(plan: RunPlan, day_offset: int,
                 prepared_root: Path | None = None) -> list[dict[str, object]]:
    # One run is read serially: this avoids opening many large HDF5 files at
    # once on the network drive, while runs are processed in parallel.
    if prepared_root is None:
        # Retain independent raw-data reading for verification callers.
        daily = [read_dump(dump) for dump in plan.dumps]
    else:
        shared_plan = prepared_plan(plan, day_offset)
        settings = pointclouds.default_settings()
        result = pointclouds.prepare_run(shared_plan, settings, prepared_root)
        if result['status'] == 'error':
            raise RuntimeError(result['error'])
        path = pointclouds.output_path_for_plan(shared_plan, prepared_root)
        daily = pointclouds.read_overview_cache(path, shared_plan, settings)
        if daily is None or any(item is None for item in daily):
            raise ValueError(f'Missing cell_type or therapy_switch metadata in {plan.protocol}/run_{plan.run}')
    initial_total = sum(daily[0][:2])
    filtered_initial_total = sum(daily[0][2:4])
    if initial_total == 0:
        raise ValueError(f"Initial active-cell count is zero for {plan.protocol}/run_{plan.run}")
    if filtered_initial_total == 0:
        raise ValueError(f"Initial filtered cell count is zero for {plan.protocol}/run_{plan.run}")
    resistant_normalized = np.asarray([row[1] / initial_total for row in daily])
    crossing = np.flatnonzero(resistant_normalized > 0.5)
    ttf = int(plan.dumps[int(crossing[0])].timestep // TIME_DUMP + day_offset) if crossing.size else -1
    rows = []
    for dump, (sensitive, resistant, filtered_sensitive, filtered_resistant, treatment) in zip(plan.dumps, daily):
        day = dump.timestep // TIME_DUMP + day_offset
        rows.append({
            "protocol": plan.protocol,
            "run": plan.run,
            "initial_condition": plan.initial_condition,
            "day": day,
            "TTF": ttf,
            "res_cell_count": resistant,
            "sen_cell_count": sensitive,
            "treatment": treatment,
            "normalized_tot": (sensitive + resistant) / initial_total,
            "normalized_sus": sensitive / initial_total,
            "normalized_res": resistant / initial_total,
            "filtered_res_cell_count": filtered_resistant,
            "filtered_sen_cell_count": filtered_sensitive,
            "filtered_normalized_tot": (filtered_sensitive + filtered_resistant) / filtered_initial_total,
            "filtered_normalized_sus": filtered_sensitive / filtered_initial_total,
            "filtered_normalized_res": filtered_resistant / filtered_initial_total,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepared-root", type=Path, default=pointclouds.DEFAULT_OUTPUT_ROOT,
                        help="Shared filtered point clouds; reused by simulation_points.")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                        help="Replace an existing output CSV.")
    parser.add_argument("--protocols", nargs="+", help="Optional protocols, e.g. CT AT50 AT100")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent runs to read (default: 4).")
    parser.add_argument("--max-runs", type=int, default=0, help="Testing limit; 0 includes every run.")
    parser.add_argument("--max-dumps", type=int, default=0, help="Testing limit per run; 0 includes every daily dump.")
    parser.add_argument("--day-offset", type=int, default=1, help="Day assigned to simulation_000000000.h5.")
    return parser.parse_args()


def validate_summary(path: Path, plans: list[RunPlan], day_offset: int = 1) -> None:
    """Require every current run/day, valid counts, and consistent normalizations."""
    expected = {(plan.protocol, plan.run, dump.timestep // TIME_DUMP + day_offset): plan
                for plan in plans for dump in plan.dumps}
    rows = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if set(COLUMNS) - set(reader.fieldnames or ()):
            raise ValueError('Summary columns are missing')
        for row in reader:
            key = row['protocol'], row['run'], int(row['day'])
            if key in rows or key not in expected:
                raise ValueError(f'Duplicate or unexpected simulation day: {key}')
            if row['initial_condition'] != expected[key].initial_condition:
                raise ValueError(f'Wrong initial condition: {key}')
            for name in COLUMNS[3:]:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite {name}: {key}')
            for name in ('res_cell_count', 'sen_cell_count', 'filtered_res_cell_count',
                         'filtered_sen_cell_count', 'day', 'TTF', 'treatment'):
                value = float(row[name])
                if value != int(value) or (value < 0 and name != 'TTF'):
                    raise ValueError(f'Invalid {name}: {key}')
            if int(row['treatment']) not in (0, 1):
                raise ValueError(f'Invalid treatment: {key}')
            rows[key] = row
    if set(rows) != set(expected) or not rows:
        raise ValueError(f'Missing simulation days: expected {len(expected)}, found {len(rows)}')
    for plan in plans:
        daily = [rows[plan.protocol, plan.run, dump.timestep // TIME_DUMP + day_offset]
                 for dump in plan.dumps]
        initial = float(daily[0]['res_cell_count']) + float(daily[0]['sen_cell_count'])
        filtered_initial = float(daily[0]['filtered_res_cell_count']) + float(daily[0]['filtered_sen_cell_count'])
        if initial <= 0 or filtered_initial <= 0:
            raise ValueError(f'Invalid baseline for {plan.protocol}/{plan.run}')
        ttf = next((int(row['day']) for row in daily
                    if float(row['res_cell_count']) / initial > .5), -1)
        for row in daily:
            if int(row['TTF']) != ttf:
                raise ValueError(f'Inconsistent failure time: {plan.protocol}/{plan.run}')
            for prefix, denominator in (('', initial), ('filtered_', filtered_initial)):
                res, sen = float(row[prefix + 'res_cell_count']), float(row[prefix + 'sen_cell_count'])
                if prefix and (res > float(row['res_cell_count']) or sen > float(row['sen_cell_count'])):
                    raise ValueError('Filtered count exceeds raw count')
                for suffix, value in (('tot', res + sen), ('sus', sen), ('res', res)):
                    if not math.isclose(float(row[prefix + 'normalized_' + suffix]), value / denominator,
                                        rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f'Inconsistent normalization: {plan.protocol}/{plan.run}')
    if not all(path.stat().st_mtime_ns >= dump.path.stat().st_mtime_ns
               for plan in plans for dump in plan.dumps):
        raise ValueError('Raw simulation snapshots are newer than the summary')


def main() -> None:
    args = parse_args()
    protocols = {item.upper() for item in args.protocols} if args.protocols else None
    plans = discover_runs(args.input_root, protocols)
    if args.max_runs:
        plans = plans[:args.max_runs]
    if args.max_dumps:
        plans = [RunPlan(plan.protocol, plan.run, plan.initial_condition, plan.dumps[:args.max_dumps]) for plan in plans]
    if not plans:
        raise FileNotFoundError(f"No simulation runs found below {args.input_root}")
    if args.output.exists() and not args.overwrite:
        try:
            validate_summary(args.output, plans, args.day_offset)
            print(f'[skip] verified complete simulation overview: {args.output}', flush=True)
            return
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            print(f'[rebuild] simulation overview is incomplete or stale: {error}', flush=True)

    print(f"Reading {len(plans)} runs from {args.input_root}", flush=True)
    print(f"Reusing/saving filtered point clouds in {args.prepared_root}", flush=True)

    rows: list[dict[str, object]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(rows_for_run, plan, args.day_offset, args.prepared_root): plan
            for plan in plans
        }
        for future in as_completed(futures):
            plan = futures[future]
            run_rows = future.result()
            rows.extend(run_rows)
            completed += 1
            print(
                f"Completed {completed}/{len(plans)}: "
                f"{plan.protocol}/run_{plan.run} ({len(run_rows)} daily rows)",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            str(row["protocol"]),
            *run_sort_key(row["run"]),
            int(row["day"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f'.{args.output.name}.tmp-{os.getpid()}')
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output)
    print(f"Wrote {len(rows):,} daily rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
