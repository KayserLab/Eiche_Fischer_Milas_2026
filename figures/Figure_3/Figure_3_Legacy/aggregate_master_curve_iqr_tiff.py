#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Aggregate all-phase linear-growth/IQR caches from density-map TIFF runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.figure_3 import master_curve_iqr_common as common
import plot_density_change_datapoints_coarsegrained as base
import plot_binned_density_change_strategy_lines_coarsegrained as plotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG_PATH)
    parser.add_argument("--escape-table", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(common.ALLOWED_STRATEGIES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    config_path = args.config.expanduser()
    escape_table = args.escape_table.expanduser()
    output_cache = args.output_cache.expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not escape_table.is_file():
        raise FileNotFoundError(escape_table)

    config = base.load_config(config_path)
    available = plotter.strategies_in_run(run_dir)
    requested = set(args.strategies) & set(common.ALLOWED_STRATEGIES)
    strategies = [
        strategy for strategy in common.ALLOWED_STRATEGIES
        if strategy in available and strategy in requested
    ]
    if not strategies:
        raise RuntimeError(f"No requested strategies found in {run_dir}")
    strategy_index = {name: index for index, name in enumerate(strategies)}
    aggregators = common.create_aggregators(strategies)
    resolution_args = argparse.Namespace(
        experimental_escape_summary=(None if args.simulation else os.fspath(escape_table)),
        simulation_escape_csv=(os.fspath(escape_table) if args.simulation else None),
    )
    cutoffs, simulation_run, escape_source = plotter.resolve_escape_cutoffs(
        run_dir,
        config,
        resolution_args,
    )
    if bool(simulation_run) != bool(args.simulation):
        raise ValueError(f"Unexpected run type: simulation={simulation_run}")

    intervals = {phase: 0 for phase in common.PHASES}
    below = {gate_name: 0 for gate_name, _gate in common.GATES}
    above = {gate_name: 0 for gate_name, _gate in common.GATES}
    scanned = 0
    for changes_dir in run_dir.glob("Plate_*/*/density_changes"):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        strategy = base.strategy_from_plate_and_well(plate, well)
        if strategy not in strategy_index:
            continue
        maps_dir = changes_dir.parent / "density_maps"
        for rel_path in sorted(changes_dir.glob(base.CHANGE_SPECS["rel"]["glob"])):
            match = base.REL_CHANGE_RE.match(rel_path.name)
            if match is None:
                continue
            first_tp = match.group("first_tp")
            next_tp = match.group("next_tp")
            first_day = base.timepoint_to_day(first_tp)
            next_day = base.timepoint_to_day(next_tp)
            active_phases = plotter.interval_phases(
                selected_phases=common.PHASES,
                plate=plate,
                well=well,
                first_day=first_day,
                next_day=next_day,
                escape_cutoffs=cutoffs,
                simulation_run=simulation_run,
            )
            if not active_phases:
                continue
            total_path = base.density_map_path(maps_dir, well, first_tp)
            res_path = base.resistant_density_map_path(maps_dir, well, first_tp)
            abs_path = rel_path.with_name(
                rel_path.name.replace("_rel_diff.tif", "_abs_diff.tif")
            )
            if not all(path.is_file() for path in (total_path, res_path, abs_path)):
                print(f"[WARN] Missing density/change pair for {rel_path}")
                continue
            previous_total = tiff.imread(total_path).astype(np.float32, copy=False)
            previous_res = tiff.imread(res_path).astype(np.float32, copy=False)
            abs_change = tiff.imread(abs_path).astype(np.float32, copy=False)
            next_res = previous_res + abs_change
            tails = common.add_linear_interval(
                aggregators,
                strategy_index[strategy],
                previous_total,
                previous_res,
                next_res,
                float(next_day - first_day),
                active_phases,
            )
            for gate_name, (low, high) in tails.items():
                below[gate_name] += low
                above[gate_name] += high
            for phase in active_phases:
                intervals[phase] += 1
            scanned += 1
            if scanned % 500 == 0:
                print(f"[scan] {scanned} intervals", flush=True)

    metadata = {
        "run_dir": os.fspath(run_dir),
        "config": os.fspath(config_path),
        "escape_source": os.fspath(escape_source),
        "simulation": bool(simulation_run),
        "kernel_um": float(config.get("radius_um", 0)),
        "coarsening_factor": int(config.get("coarsening_factor", 0)),
        "intervals": intervals,
        "below_histogram": below,
        "above_histogram": above,
        "excluded_strategies": ["Pulse", "KPP-Agent"],
    }
    common.save_bundle(
        output_cache,
        common.AggregationBundle(strategies, aggregators, metadata),
    )
    print(f"[done] strategies={strategies}; intervals={intervals}")
    print(f"[tails] below={below}; above={above}")
    print(f"[saved] {output_cache}")


if __name__ == "__main__":
    main()
