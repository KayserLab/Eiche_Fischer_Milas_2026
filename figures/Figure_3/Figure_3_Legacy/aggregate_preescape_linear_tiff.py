#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Aggregate pre-escape linear resistant growth from density-map TIFF runs.

The plotted response is the average fractional change per day,

    ((rho_next - rho_previous) / rho_previous) / delta_days

where rho is the resistant-density map.  Binning and resistant-density gating
use the previous-frame maps, matching the existing relative-growth analysis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

import plot_density_change_datapoints_coarsegrained as base
import plot_binned_density_change_strategy_lines_coarsegrained as plotter


GATE = (2.0e-5, 4.375e-4)
GATE_TOKEN = "gate_2e-05_0p0004375"
CHANGE_TYPE = "linear_rel_per_day"
DEFAULT_X_BIN_SIZE = 0.2
DEFAULT_Y_HIST_BINS = 32768
DEFAULT_Y_HIST_RANGE = (-1.05, 12.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG_PATH)
    parser.add_argument("--escape-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--x-bin-size", type=float, default=DEFAULT_X_BIN_SIZE)
    parser.add_argument("--y-hist-bins", type=int, default=DEFAULT_Y_HIST_BINS)
    return parser.parse_args()


def pooled_aggregator(source: plotter.Aggregator) -> plotter.Aggregator:
    pooled = plotter.Aggregator.create(
        n_strategies=1,
        x_edges=source.x_edges,
        ylim=(float(source.y_edges[0]), float(source.y_edges[-1])),
        n_y_bins=source.n_y_bins,
    )
    pooled.count[0] = source.count.sum(axis=0)
    pooled.total[0] = source.total.sum(axis=0)
    pooled.total_sq[0] = source.total_sq.sum(axis=0)
    pooled.histogram[0] = source.histogram.sum(axis=0)
    return pooled


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    config_path = args.config.expanduser()
    escape_table = args.escape_table.expanduser()
    output_dir = args.output_dir.expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Density run does not exist: {run_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    if not escape_table.is_file():
        raise FileNotFoundError(f"Escape table does not exist: {escape_table}")
    if args.x_bin_size <= 0:
        raise ValueError("--x-bin-size must be positive")
    if args.y_hist_bins < 1024:
        raise ValueError("--y-hist-bins must be at least 1024")

    config = base.load_config(config_path)
    strategies = plotter.strategies_in_run(run_dir)
    resolution_args = argparse.Namespace(
        experimental_escape_summary=(
            None if args.simulation else os.fspath(escape_table)
        ),
        simulation_escape_csv=(
            os.fspath(escape_table) if args.simulation else None
        ),
    )
    cutoffs, simulation_run, escape_source = plotter.resolve_escape_cutoffs(
        run_dir,
        config,
        resolution_args,
    )
    if bool(simulation_run) != bool(args.simulation):
        raise ValueError(
            f"Unexpected run type for {run_dir}: simulation={simulation_run}"
        )

    settings = plotter.plot_settings(config, "rel")
    x_max = float(settings["xlim"][1])
    n_x_bins = int(np.ceil(x_max / float(args.x_bin_size)))
    x_edges = np.arange(n_x_bins + 1, dtype=float) * float(args.x_bin_size)
    aggregator = plotter.Aggregator.create(
        n_strategies=len(strategies),
        x_edges=x_edges,
        ylim=DEFAULT_Y_HIST_RANGE,
        n_y_bins=int(args.y_hist_bins),
    )
    strategy_index = {name: index for index, name in enumerate(strategies)}
    n_x_bins = aggregator.n_x_bins
    x_bin_size = float(x_edges[1] - x_edges[0])
    completed = 0
    below_hist = 0
    above_hist = 0

    for changes_dir in run_dir.glob("Plate_*/*/density_changes"):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        strategy = base.strategy_from_plate_and_well(plate, well)
        if strategy not in strategy_index:
            print(f"[WARN] Unknown strategy {strategy} for {plate}/{well}")
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
            phases = plotter.interval_phases(
                selected_phases=("before_escape",),
                plate=plate,
                well=well,
                first_day=first_day,
                next_day=next_day,
                escape_cutoffs=cutoffs,
                simulation_run=simulation_run,
            )
            if not phases:
                continue
            delta_days = next_day - first_day
            if not np.isfinite(delta_days) or delta_days <= 0:
                raise ValueError(f"Invalid interval {first_tp}->{next_tp}")

            total_path = base.density_map_path(maps_dir, well, first_tp)
            res_path = base.resistant_density_map_path(maps_dir, well, first_tp)
            abs_path = rel_path.with_name(
                rel_path.name.replace("_rel_diff.tif", "_abs_diff.tif")
            )
            if not all(path.is_file() for path in (total_path, res_path, abs_path)):
                print(f"[WARN] Missing density/change pair for {rel_path}")
                continue

            total_density = tiff.imread(total_path).astype(np.float32, copy=False).ravel()
            res_density = tiff.imread(res_path).astype(np.float32, copy=False).ravel()
            abs_change = tiff.imread(abs_path).astype(np.float32, copy=False).ravel()
            if len({total_density.shape, res_density.shape, abs_change.shape}) != 1:
                raise ValueError(f"Shape mismatch for {rel_path}")

            x = total_density.astype(np.float64, copy=False) * base.X_DISPLAY_SCALE
            x_valid = np.isfinite(x) & (x >= x_edges[0]) & (x <= x_edges[-1])
            x_bins = np.zeros(x.shape, dtype=np.int64)
            finite_x = np.isfinite(x)
            x_bins[finite_x] = np.floor(
                (x[finite_x] - x_edges[0]) / x_bin_size
            ).astype(np.int64)
            np.clip(x_bins, 0, n_x_bins - 1, out=x_bins)

            y = np.full(res_density.shape, np.nan, dtype=np.float32)
            valid_growth = (
                np.isfinite(res_density)
                & np.isfinite(abs_change)
                & (res_density > 0)
            )
            y[valid_growth] = (
                abs_change[valid_growth]
                / res_density[valid_growth]
                / np.float32(delta_days)
            )
            valid = (
                x_valid
                & valid_growth
                & np.isfinite(y)
                & (res_density >= GATE[0])
                & (res_density <= GATE[1])
            )
            values = y[valid].astype(np.float64, copy=False)
            below_hist += int(np.count_nonzero(values < DEFAULT_Y_HIST_RANGE[0]))
            above_hist += int(np.count_nonzero(values > DEFAULT_Y_HIST_RANGE[1]))
            aggregator.add(
                strategy_index[strategy],
                x_bins[valid],
                values,
            )
            completed += 1
            if completed % 250 == 0:
                print(f"[scan] {completed} pre-escape intervals")

    if completed == 0:
        raise RuntimeError(f"No pre-escape intervals accumulated from {run_dir}")

    strategy_summary = plotter.summarize(
        aggregator,
        strategies,
        GATE,
        CHANGE_TYPE,
        "before_escape",
    )
    master_summary = plotter.summarize(
        pooled_aggregator(aggregator),
        ["All strategies"],
        GATE,
        CHANGE_TYPE,
        "before_escape",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = output_dir / "binned_strategy_statistics.tsv"
    master_path = output_dir / "pooled_all_strategies_statistics.tsv"
    strategy_summary.to_csv(strategy_path, sep="\t", index=False)
    master_summary.to_csv(master_path, sep="\t", index=False)
    total_pixels = int(master_summary["n_pixels"].sum())
    print(
        f"[done] intervals={completed}; pixels={total_pixels:,}; "
        f"below_hist={below_hist:,}; above_hist={above_hist:,}; "
        f"escape={escape_source}"
    )
    print(f"[saved] {strategy_path}")
    print(f"[saved] {master_path}")


if __name__ == "__main__":
    main()
