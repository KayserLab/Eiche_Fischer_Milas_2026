#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Stream simulation source data into pre-escape linear-growth summaries.

This avoids retaining density-map TIFFs for a requested simulation kernel.
Only intervals whose end day is no later than the run's positive TTE are
included.  Linear growth is the average fractional resistant-density change
per day: ((rho_next - rho_previous) / rho_previous) / delta_days.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
import plot_density_change_datapoints_coarsegrained as base
import plot_binned_density_change_strategy_lines_coarsegrained as plotter
import plot_simulation_kernel_sweep_direct as sweep
from shared.strategy_colors import normalize_strategy


DEFAULT_INPUT_ROOT = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\data-DB"
)
DEFAULT_ESCAPE_CSV = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\combined_results.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\CoreSeed_InSilico_PreEscape_160\run_kernel_160_coarse_16"
    r"\summary_plots\gate_matrix_with_n\before_escape"
    r"\linear_relative_per_day\gate_2e-05_0p0004375"
)
GATE = (2.0e-5, 4.375e-4)
CHANGE_TYPE = "linear_rel_per_day"
Y_HIST_RANGE = (-1.05, 12.0)
DEFAULT_Y_HIST_BINS = 32768
DEFAULT_X_BIN_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--escape-csv", type=Path, default=DEFAULT_ESCAPE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kernel-um", type=int, default=160)
    parser.add_argument("--coarsening-factor", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--x-bin-size", type=float, default=DEFAULT_X_BIN_SIZE)
    parser.add_argument("--y-hist-bins", type=int, default=DEFAULT_Y_HIST_BINS)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=tuple(sweep.SOURCE_TO_DISPLAY),
        default=list(sweep.SOURCE_TO_DISPLAY),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
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
    finite = data[np.isfinite(data["escape_day"]) & (data["escape_day"] > 0)].copy()
    conflicts = finite.groupby(["strategy_key", "run_key"])["escape_day"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        examples = ", ".join(
            f"{strategy}/run_{run}" for strategy, run in conflicts.index[:5]
        )
        raise ValueError(f"Conflicting TTE values in {path}: {examples}")
    finite = finite.groupby(["strategy_key", "run_key"], as_index=False)[
        "escape_day"
    ].first()
    return {
        (str(row.strategy_key), str(row.run_key)): float(row.escape_day)
        for row in finite.itertuples(index=False)
    }


def create_aggregator(
    x_edges: np.ndarray,
    n_y_bins: int,
    *,
    n_strategies: int = 1,
) -> plotter.Aggregator:
    return plotter.Aggregator.create(
        n_strategies=n_strategies,
        x_edges=x_edges,
        ylim=Y_HIST_RANGE,
        n_y_bins=n_y_bins,
    )


def accumulate_interval(
    aggregator: plotter.Aggregator,
    previous_total: np.ndarray,
    previous_res: np.ndarray,
    next_res: np.ndarray,
    delta_days: float,
    x_edges: np.ndarray,
) -> tuple[int, int]:
    x = previous_total.ravel().astype(np.float64, copy=False) * base.X_DISPLAY_SCALE
    previous = previous_res.ravel()
    nxt = next_res.ravel()
    valid_growth = (
        np.isfinite(previous)
        & np.isfinite(nxt)
        & (previous > 0)
    )
    y = np.full(previous.shape, np.nan, dtype=np.float32)
    y[valid_growth] = (
        (nxt[valid_growth] - previous[valid_growth])
        / previous[valid_growth]
        / np.float32(delta_days)
    )

    x_bin_size = float(x_edges[1] - x_edges[0])
    n_x_bins = len(x_edges) - 1
    x_valid = np.isfinite(x) & (x >= x_edges[0]) & (x <= x_edges[-1])
    x_bins = np.zeros(x.shape, dtype=np.int64)
    finite_x = np.isfinite(x)
    x_bins[finite_x] = np.floor(
        (x[finite_x] - x_edges[0]) / x_bin_size
    ).astype(np.int64)
    np.clip(x_bins, 0, n_x_bins - 1, out=x_bins)
    valid = (
        x_valid
        & valid_growth
        & np.isfinite(y)
        & (previous >= GATE[0])
        & (previous <= GATE[1])
    )
    values = y[valid].astype(np.float64, copy=False)
    aggregator.add(0, x_bins[valid], values)
    return (
        int(np.count_nonzero(values < Y_HIST_RANGE[0])),
        int(np.count_nonzero(values > Y_HIST_RANGE[1])),
    )


def save_cache(
    path: Path,
    aggregator: plotter.Aggregator,
    intervals: int,
    below_hist: int,
    above_hist: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        intervals=np.asarray([intervals], dtype=np.int64),
        below_hist=np.asarray([below_hist], dtype=np.int64),
        above_hist=np.asarray([above_hist], dtype=np.int64),
        count=aggregator.count[0],
        total=aggregator.total[0],
        total_sq=aggregator.total_sq[0],
        histogram=aggregator.histogram[0],
    )


def process_run(
    source: str,
    run_name: str,
    cutoff_day: float,
    input_root: str,
    cache_path: str,
    kernel_um: int,
    coarsening_factor: int,
    x_edges: np.ndarray,
    n_y_bins: int,
) -> tuple[str, str, int, str]:
    sim_args = sweep.simulation_args(input_root, source, run_name)
    plans = sim.discover_run_plans(sim_args)
    if len(plans) != 1:
        raise RuntimeError(
            f"Expected one plan for {source}/{run_name}, found {len(plans)}"
        )
    plan = plans[0]
    loaded = [sim.load_filtered_cells(item.path, sim_args) for item in plan.files]
    canvas = sweep.canvas_from_loaded(loaded, float(kernel_um))
    aggregator = create_aggregator(x_edges, n_y_bins)
    previous: tuple[np.ndarray, np.ndarray, int] | None = None
    intervals = 0
    below_hist = 0
    above_hist = 0
    for sim_file, (positions_um, type_values) in zip(plan.files, loaded):
        if sim_file.day > cutoff_day:
            break
        sus, res = sim.positions_to_density_inputs(
            positions_um,
            type_values,
            canvas,
        )
        res_density, total_density = sweep.resistant_and_total_density(
            sus,
            res,
            radius_um=float(kernel_um),
            canvas=canvas,
            coarsening_factor=coarsening_factor,
        )
        if previous is not None:
            previous_res, previous_total, previous_day = previous
            low, high = accumulate_interval(
                aggregator,
                previous_total,
                previous_res,
                res_density,
                float(sim_file.day - previous_day),
                x_edges,
            )
            below_hist += low
            above_hist += high
            intervals += 1
        previous = (res_density, total_density, sim_file.day)
    save_cache(
        Path(cache_path),
        aggregator,
        intervals,
        below_hist,
        above_hist,
    )
    return source, run_name, intervals, cache_path


def merge_cache(
    path: Path,
    target: plotter.Aggregator,
    strategy_index: int,
) -> tuple[int, int, int]:
    with np.load(path) as data:
        target.count[strategy_index] += data["count"]
        target.total[strategy_index] += data["total"]
        target.total_sq[strategy_index] += data["total_sq"]
        target.histogram[strategy_index] += data["histogram"]
        return (
            int(data["intervals"][0]),
            int(data["below_hist"][0]),
            int(data["above_hist"][0]),
        )


def pooled_aggregator(source: plotter.Aggregator) -> plotter.Aggregator:
    pooled = create_aggregator(
        source.x_edges,
        source.n_y_bins,
        n_strategies=1,
    )
    pooled.count[0] = source.count.sum(axis=0)
    pooled.total[0] = source.total.sum(axis=0)
    pooled.total_sq[0] = source.total_sq.sum(axis=0)
    pooled.histogram[0] = source.histogram.sum(axis=0)
    return pooled


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser()
    escape_csv = args.escape_csv.expanduser()
    output_dir = args.output_dir.expanduser()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not escape_csv.is_file():
        raise FileNotFoundError(escape_csv)
    if args.kernel_um <= 0 or args.coarsening_factor <= 0:
        raise ValueError("Kernel and coarsening factor must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    config = base.load_config(sim.DEFAULT_CONFIG_PATH)
    settings = plotter.plot_settings(config, "rel")
    x_max = float(settings["xlim"][1])
    n_x_bins = int(np.ceil(x_max / float(args.x_bin_size)))
    x_edges = np.arange(n_x_bins + 1, dtype=float) * float(args.x_bin_size)
    tte = load_positive_tte(escape_csv)
    selected_sources = list(dict.fromkeys(args.strategies))
    selected_display = [
        display
        for display in sweep.DISPLAY_ORDER
        if display in {sweep.SOURCE_TO_DISPLAY[source] for source in selected_sources}
    ]
    global_aggregator = create_aggregator(
        x_edges,
        int(args.y_hist_bins),
        n_strategies=len(selected_display),
    )

    cache_root = output_dir / "aggregation_cache"
    tasks: list[tuple[str, str, float, Path]] = []
    unmatched = 0
    for source in selected_sources:
        plans = sim.discover_run_plans(sweep.simulation_args(os.fspath(input_root), source))
        for plan in plans:
            key = (
                normalize_strategy(plan.strategy),
                base.normalize_simulation_run_id(plan.run_id),
            )
            cutoff = tte.get(key)
            if cutoff is None:
                unmatched += 1
                continue
            tasks.append(
                (
                    source,
                    plan.run_id,
                    float(cutoff),
                    cache_root / source / f"{plan.run_id}_linear_preescape.npz",
                )
            )
    if not tasks:
        raise RuntimeError("No simulation runs matched a positive TTE")
    print(f"[TTE] matched={len(tasks)}; excluded_without_positive_TTE={unmatched}")

    pending = [task for task in tasks if args.overwrite_cache or not task[3].is_file()]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, int(args.workers)), len(pending))
        ) as pool:
            futures = {
                pool.submit(
                    process_run,
                    source,
                    run_name,
                    cutoff,
                    os.fspath(input_root),
                    os.fspath(cache_path),
                    int(args.kernel_um),
                    int(args.coarsening_factor),
                    x_edges,
                    int(args.y_hist_bins),
                ): (source, run_name)
                for source, run_name, cutoff, cache_path in pending
            }
            for future in as_completed(futures):
                source, run_name, intervals, cache_path = future.result()
                print(
                    f"[complete] {source}/{run_name}: {intervals} intervals -> "
                    f"{cache_path}",
                    flush=True,
                )

    total_intervals = 0
    below_hist = 0
    above_hist = 0
    for source, _run_name, _cutoff, cache_path in tasks:
        display = sweep.SOURCE_TO_DISPLAY[source]
        strategy_index = selected_display.index(display)
        intervals, low, high = merge_cache(
            cache_path,
            global_aggregator,
            strategy_index,
        )
        total_intervals += intervals
        below_hist += low
        above_hist += high

    strategy_summary = plotter.summarize(
        global_aggregator,
        selected_display,
        GATE,
        CHANGE_TYPE,
        "before_escape",
    )
    master_summary = plotter.summarize(
        pooled_aggregator(global_aggregator),
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
        f"[done] intervals={total_intervals}; pixels={total_pixels:,}; "
        f"below_hist={below_hist:,}; above_hist={above_hist:,}"
    )
    print(f"[saved] {strategy_path}")
    print(f"[saved] {master_path}")


if __name__ == "__main__":
    main()
