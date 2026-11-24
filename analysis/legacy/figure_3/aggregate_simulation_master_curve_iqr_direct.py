#!/usr/bin/env python3
"""Retired simulation-only master-curve cache workflow; retained for provenance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
from analysis.figure_3 import master_curve_iqr_common as common
from analysis.figure_3 import density_change_data_utils as base
from analysis.figure_3 import simulation_density_utils as sweep
from shared.strategy_colors import normalize_strategy


DEFAULT_INPUT_ROOT = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\data-DB"
)
DEFAULT_ESCAPE_CSV = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\combined_results.csv"
)
DEFAULT_OUTPUT_ROOT = Path(r"F:\CoreSeed_Master_Curve_Linear_IQR_Cache\in_silico")
DEFAULT_KERNELS = ((40, 4), (160, 16))


def parse_kernel(value: str) -> tuple[int, int]:
    pieces = value.split(":", 1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Kernel must be KERNEL_UM:COARSENING")
    kernel_um, coarsening = (int(piece) for piece in pieces)
    if kernel_um <= 0 or coarsening <= 0:
        raise argparse.ArgumentTypeError("Kernel values must be positive")
    return kernel_um, coarsening


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--escape-csv", type=Path, default=DEFAULT_ESCAPE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--kernel",
        action="append",
        type=parse_kernel,
        default=None,
        help="Repeatable KERNEL_UM:COARSENING; default 40:4 and 160:16.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite-run-cache", action="store_true")
    return parser.parse_args()


def load_positive_tte(path: Path) -> dict[tuple[str, str], float]:
    summary = pd.read_csv(path, dtype={"protocol": str, "run": str})
    required = {"protocol", "run", "TTE"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    data = summary[["protocol", "run", "TTE"]].copy()
    data["strategy_key"] = data["protocol"].astype(str).map(normalize_strategy)
    data["run_key"] = data["run"].map(base.normalize_simulation_run_id)
    data["escape_day"] = pd.to_numeric(data["TTE"], errors="coerce")
    finite = data[np.isfinite(data["escape_day"]) & (data["escape_day"] > 0)]
    conflicts = finite.groupby(["strategy_key", "run_key"])["escape_day"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting positive TTE values in {path}")
    finite = finite.groupby(["strategy_key", "run_key"], as_index=False)[
        "escape_day"
    ].first()
    return {
        (str(row.strategy_key), str(row.run_key)): float(row.escape_day)
        for row in finite.itertuples(index=False)
    }


def active_phases(first_day: float, next_day: float, cutoff: float | None) -> tuple[str, ...]:
    phases = ["overall"]
    if cutoff is not None and np.isfinite(cutoff):
        if next_day <= cutoff:
            phases.append("before_escape")
        if first_day >= cutoff:
            phases.append("after_escape")
    return tuple(phases)


def process_run(
    source: str,
    run_name: str,
    display_strategy: str,
    cutoff: float | None,
    input_root: str,
    cache_root: str,
    kernels: tuple[tuple[int, int], ...],
) -> tuple[str, str, dict[int, str]]:
    sim_args = sweep.simulation_args(input_root, source, run_name)
    plans = sim.discover_run_plans(sim_args)
    if len(plans) != 1:
        raise RuntimeError(f"Expected one plan for {source}/{run_name}, found {len(plans)}")
    plan = plans[0]
    loaded = [sim.load_filtered_cells(item.path, sim_args) for item in plan.files]
    canvases = {
        kernel_um: sweep.canvas_from_loaded(loaded, float(kernel_um))
        for kernel_um, _coarsening in kernels
    }
    bundles = {
        kernel_um: common.AggregationBundle(
            [display_strategy],
            common.create_aggregators([display_strategy]),
            {},
        )
        for kernel_um, _coarsening in kernels
    }
    previous: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    interval_counts = {
        kernel_um: {phase: 0 for phase in common.PHASES}
        for kernel_um, _coarsening in kernels
    }
    for sim_file, (positions_um, type_values) in zip(plan.files, loaded):
        for kernel_um, coarsening in kernels:
            sus, res = sim.positions_to_density_inputs(
                positions_um,
                type_values,
                canvases[kernel_um],
            )
            res_density, total_density = sweep.resistant_and_total_density(
                sus,
                res,
                radius_um=float(kernel_um),
                canvas=canvases[kernel_um],
                coarsening_factor=coarsening,
            )
            if kernel_um in previous:
                previous_res, previous_total, previous_day = previous[kernel_um]
                phases = active_phases(
                    float(previous_day),
                    float(sim_file.day),
                    cutoff,
                )
                common.add_linear_interval(
                    bundles[kernel_um].aggregators,
                    0,
                    previous_total,
                    previous_res,
                    res_density,
                    float(sim_file.day - previous_day),
                    phases,
                )
                for phase in phases:
                    interval_counts[kernel_um][phase] += 1
            previous[kernel_um] = (res_density, total_density, sim_file.day)

    saved: dict[int, str] = {}
    for kernel_um, coarsening in kernels:
        cache_path = (
            Path(cache_root)
            / "run_caches"
            / source
            / f"{run_name}_kernel_{kernel_um}.npz"
        )
        bundles[kernel_um].metadata = {
            "source": source,
            "run": run_name,
            "kernel_um": kernel_um,
            "coarsening_factor": coarsening,
            "positive_tte": cutoff,
            "intervals": interval_counts[kernel_um],
        }
        common.save_bundle(cache_path, bundles[kernel_um])
        saved[kernel_um] = os.fspath(cache_path)
    return source, run_name, saved


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser()
    escape_csv = args.escape_csv.expanduser()
    output_root = args.output_root.expanduser()
    kernels = tuple(args.kernel or DEFAULT_KERNELS)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not escape_csv.is_file():
        raise FileNotFoundError(escape_csv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    tte = load_positive_tte(escape_csv)

    tasks: list[tuple[str, str, str, float | None]] = []
    excluded_tte = 0
    for source, display in sweep.SOURCE_TO_DISPLAY.items():
        if display not in common.ALLOWED_STRATEGIES:
            continue
        plans = sim.discover_run_plans(sweep.simulation_args(os.fspath(input_root), source))
        for plan in plans:
            cutoff = tte.get(
                (
                    normalize_strategy(plan.strategy),
                    base.normalize_simulation_run_id(plan.run_id),
                )
            )
            if cutoff is None:
                excluded_tte += 1
            tasks.append((source, plan.run_id, display, cutoff))
    print(f"[runs] total={len(tasks)}; without_positive_TTE={excluded_tte}")

    def cache_paths(source: str, run_name: str) -> list[Path]:
        return [
            output_root / "run_caches" / source / f"{run_name}_kernel_{kernel_um}.npz"
            for kernel_um, _coarsening in kernels
        ]

    pending = [
        task for task in tasks
        if args.overwrite_run_cache
        or not all(path.is_file() for path in cache_paths(task[0], task[1]))
    ]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, int(args.workers)), len(pending))
        ) as pool:
            futures = {
                pool.submit(
                    process_run,
                    source,
                    run_name,
                    display,
                    cutoff,
                    os.fspath(input_root),
                    os.fspath(output_root),
                    kernels,
                ): (source, run_name)
                for source, run_name, display, cutoff in pending
            }
            for future in as_completed(futures):
                source, run_name, saved = future.result()
                print(f"[complete] {source}/{run_name}: {saved}", flush=True)

    for kernel_um, coarsening in kernels:
        paths = [
            output_root / "run_caches" / source / f"{run_name}_kernel_{kernel_um}.npz"
            for source, run_name, _display, _cutoff in tasks
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} run caches for kernel {kernel_um}")
        merged = common.merge_bundles(paths)
        merged.metadata.update(
            {
                "dataset": "in_silico",
                "kernel_um": kernel_um,
                "coarsening_factor": coarsening,
                "n_runs": len(tasks),
                "n_runs_without_positive_TTE": excluded_tte,
                "escape_csv": os.fspath(escape_csv),
            }
        )
        output_path = output_root / f"kernel_{kernel_um}um_all_phases_iqr.npz"
        common.save_bundle(output_path, merged)
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
