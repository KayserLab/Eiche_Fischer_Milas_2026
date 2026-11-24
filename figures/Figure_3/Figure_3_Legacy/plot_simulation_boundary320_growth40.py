#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot 40-um growth against a 320-um density environment for simulations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as density_base
from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
import plot_binned_density_change_strategy_lines_coarsegrained as producer
import plot_simulation_kernel_sweep_direct as common


BOUNDARY_KERNEL_UM = 320
BOUNDARY_COARSE_FACTOR = 32
GROWTH_KERNEL_UM = 40
GROWTH_COARSE_FACTOR = 4
DEFAULT_OUTPUT_ROOT = (
    common.DEFAULT_OUTPUT_ROOT / "run_boundary_kernel_320_growth_kernel_40"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=os.fspath(common.DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=os.fspath(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config", default=os.fspath(sim.DEFAULT_CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--x-bin-size", type=float, default=0.2)
    parser.add_argument("--y-hist-bins", type=int, default=4096)
    parser.add_argument("--min-bin-pixels", type=int, default=100)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=tuple(common.SOURCE_TO_DISPLAY),
        default=list(common.SOURCE_TO_DISPLAY),
    )
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--max-timepoints", type=int, default=0)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def resistant_density(
    res,
    *,
    radius_um: float,
    canvas: sim.CanvasSpec,
    coarsening_factor: int,
) -> np.ndarray:
    image = density_base.rasterize_points_to_coarse_grid(
        res,
        canvas.height_px,
        canvas.width_px,
        coarsening_factor,
    )
    coarse_radius = density_base.radius_um_to_coarse_px(
        radius_um,
        coarsening_factor,
    )
    result = density_base.get_local_density_fft(image, coarse_radius)
    coarse_pixel_area_um2 = (sim.PIXEL_SIZE_UM * coarsening_factor) ** 2
    return density_base.zero_density_fft_noise(result / coarse_pixel_area_um2)


def boundary_indices_for_growth_grid(
    boundary_canvas: sim.CanvasSpec,
    boundary_shape: tuple[int, int],
    growth_canvas: sim.CanvasSpec,
    growth_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    growth_pixel_um = sim.PIXEL_SIZE_UM * GROWTH_COARSE_FACTOR
    boundary_pixel_um = sim.PIXEL_SIZE_UM * BOUNDARY_COARSE_FACTOR
    growth_y = (
        growth_canvas.min_y_um
        + (np.arange(growth_shape[0], dtype=float) + 0.5) * growth_pixel_um
    )
    growth_x = (
        growth_canvas.min_x_um
        + (np.arange(growth_shape[1], dtype=float) + 0.5) * growth_pixel_um
    )
    boundary_y = np.floor(
        (growth_y - boundary_canvas.min_y_um) / boundary_pixel_um
    ).astype(np.int64)
    boundary_x = np.floor(
        (growth_x - boundary_canvas.min_x_um) / boundary_pixel_um
    ).astype(np.int64)
    if (
        boundary_y.min(initial=0) < 0
        or boundary_x.min(initial=0) < 0
        or boundary_y.max(initial=-1) >= boundary_shape[0]
        or boundary_x.max(initial=-1) >= boundary_shape[1]
    ):
        raise ValueError("Growth grid extends outside the 320-um boundary grid")
    return boundary_y, boundary_x


def create_aggregators(
    x_edges: np.ndarray,
    ylim: tuple[float, float],
    n_y_bins: int,
) -> dict[int, producer.Aggregator]:
    return {
        gate_index: producer.Aggregator.create(
            n_strategies=1,
            x_edges=x_edges,
            ylim=ylim,
            n_y_bins=n_y_bins,
        )
        for gate_index, _gate in enumerate(common.GATES)
    }


def accumulate_mixed_interval(
    aggregators: dict[int, producer.Aggregator],
    previous_boundary_total: np.ndarray,
    previous_boundary_res: np.ndarray,
    previous_growth_res: np.ndarray,
    next_growth_res: np.ndarray,
    boundary_y: np.ndarray,
    boundary_x: np.ndarray,
    delta_days: float,
    x_edges: np.ndarray,
    ylim: tuple[float, float],
) -> None:
    previous_growth = previous_growth_res.ravel()
    next_growth = next_growth_res.ravel()
    y = np.zeros(previous_growth.shape, dtype=np.float32)
    valid_growth = (
        np.isfinite(previous_growth)
        & np.isfinite(next_growth)
        & (previous_growth > sim.DENSITY_ZERO_FLOOR_UM2)
        & (next_growth > sim.DENSITY_ZERO_FLOOR_UM2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        y[valid_growth] = (
            np.log(next_growth[valid_growth] / previous_growth[valid_growth])
            / float(delta_days)
        ).astype(np.float32, copy=False)

    boundary_res = previous_boundary_res[
        boundary_y[:, None],
        boundary_x[None, :],
    ].ravel()
    x = (
        previous_boundary_total[
            boundary_y[:, None],
            boundary_x[None, :],
        ].ravel().astype(np.float64, copy=False)
        * producer.base.X_DISPLAY_SCALE
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
    finite_y = np.isfinite(y) & (y >= ylim[0]) & (y <= ylim[1])

    for gate_index, gate in enumerate(common.GATES):
        valid = (
            x_valid
            & finite_y
            & np.isfinite(boundary_res)
            & (boundary_res >= gate[0])
            & (boundary_res <= gate[1])
        )
        aggregators[gate_index].add(
            0,
            x_bins[valid],
            y[valid].astype(np.float64, copy=False),
        )


def save_cache(
    path: Path,
    aggregators: dict[int, producer.Aggregator],
    intervals: int,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "intervals": np.asarray([intervals], dtype=np.int64)
    }
    for gate_index, aggregator in aggregators.items():
        prefix = f"g{gate_index}"
        arrays[f"{prefix}_count"] = aggregator.count[0]
        arrays[f"{prefix}_total"] = aggregator.total[0]
        arrays[f"{prefix}_total_sq"] = aggregator.total_sq[0]
        arrays[f"{prefix}_histogram"] = aggregator.histogram[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def process_run(
    strategy: str,
    run_name: str,
    input_root: str,
    cache_path: str,
    x_edges: np.ndarray,
    ylim: tuple[float, float],
    n_y_bins: int,
    max_timepoints: int,
) -> tuple[str, int, str]:
    args = common.simulation_args(
        input_root,
        strategy,
        run_name,
        max_timepoints,
    )
    plans = sim.discover_run_plans(args)
    if len(plans) != 1:
        raise RuntimeError(f"Expected one plan for {strategy}/{run_name}, got {len(plans)}")
    plan = plans[0]
    loaded = [sim.load_filtered_cells(item.path, args) for item in plan.files]
    boundary_canvas = common.canvas_from_loaded(
        loaded,
        float(BOUNDARY_KERNEL_UM),
    )
    growth_canvas = common.canvas_from_loaded(
        loaded,
        float(GROWTH_KERNEL_UM),
    )
    aggregators = create_aggregators(x_edges, ylim, n_y_bins)
    previous = None
    grid_indices = None
    intervals = 0
    for sim_file, (positions_um, type_values) in zip(plan.files, loaded):
        boundary_sus, boundary_res_points = sim.positions_to_density_inputs(
            positions_um,
            type_values,
            boundary_canvas,
        )
        boundary_res, boundary_total = common.resistant_and_total_density(
            boundary_sus,
            boundary_res_points,
            radius_um=float(BOUNDARY_KERNEL_UM),
            canvas=boundary_canvas,
            coarsening_factor=BOUNDARY_COARSE_FACTOR,
        )
        _growth_sus, growth_res_points = sim.positions_to_density_inputs(
            positions_um,
            type_values,
            growth_canvas,
        )
        growth_res = resistant_density(
            growth_res_points,
            radius_um=float(GROWTH_KERNEL_UM),
            canvas=growth_canvas,
            coarsening_factor=GROWTH_COARSE_FACTOR,
        )
        if grid_indices is None:
            grid_indices = boundary_indices_for_growth_grid(
                boundary_canvas,
                boundary_res.shape,
                growth_canvas,
                growth_res.shape,
            )
        if previous is not None:
            previous_boundary_res, previous_boundary_total, previous_growth_res, previous_day = previous
            accumulate_mixed_interval(
                aggregators,
                previous_boundary_total,
                previous_boundary_res,
                previous_growth_res,
                growth_res,
                grid_indices[0],
                grid_indices[1],
                float(sim_file.day - previous_day),
                x_edges,
                ylim,
            )
            intervals += 1
        previous = (
            boundary_res,
            boundary_total,
            growth_res,
            sim_file.day,
        )
    save_cache(Path(cache_path), aggregators, intervals)
    print(f"[{strategy}/{run_name}] {intervals} intervals complete", flush=True)
    return strategy, intervals, cache_path


def merge_cache(
    path: Path,
    strategy_index: int,
    aggregators: dict[int, producer.Aggregator],
) -> int:
    with np.load(path) as data:
        intervals = int(data["intervals"][0])
        for gate_index, aggregator in aggregators.items():
            prefix = f"g{gate_index}"
            aggregator.count[strategy_index] += data[f"{prefix}_count"]
            aggregator.total[strategy_index] += data[f"{prefix}_total"]
            aggregator.total_sq[strategy_index] += data[f"{prefix}_total_sq"]
            aggregator.histogram[strategy_index] += data[f"{prefix}_histogram"]
    return intervals


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = producer.base.load_config(args.config)
    settings = producer.plot_settings(config, "rel")
    n_x_bins = int(math.ceil(float(settings["xlim"][1]) / float(args.x_bin_size)))
    x_edges = np.arange(n_x_bins + 1, dtype=float) * float(args.x_bin_size)
    selected_sources = list(dict.fromkeys(args.strategies))
    selected_display = [
        display
        for display in common.DISPLAY_ORDER
        if display in {common.SOURCE_TO_DISPLAY[item] for item in selected_sources}
    ]
    aggregators = {
        gate_index: producer.Aggregator.create(
            n_strategies=len(selected_display),
            x_edges=x_edges,
            ylim=settings["ylim"],
            n_y_bins=int(args.y_hist_bins),
        )
        for gate_index, _gate in enumerate(common.GATES)
    }

    cache_dir = output_root / "aggregation_cache"
    tasks: list[tuple[str, str, Path]] = []
    requested_runs = (
        {
            value if str(value).startswith("run_") else f"run_{value}"
            for value in args.runs
        }
        if args.runs
        else None
    )
    for source in selected_sources:
        plans = sim.discover_run_plans(
            common.simulation_args(os.fspath(input_root), source)
        )
        if requested_runs is not None:
            plans = [plan for plan in plans if plan.run_id in requested_runs]
        for plan in plans:
            tasks.append(
                (
                    source,
                    plan.run_id,
                    cache_dir / source / f"{plan.run_id}_mixed_320_40.npz",
                )
            )
    pending = [
        task
        for task in tasks
        if args.overwrite_cache or not task[2].exists()
    ]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, args.workers), len(pending))
        ) as pool:
            futures = {
                pool.submit(
                    process_run,
                    source,
                    run_name,
                    os.fspath(input_root),
                    os.fspath(cache_path),
                    x_edges,
                    tuple(settings["ylim"]),
                    int(args.y_hist_bins),
                    int(args.max_timepoints),
                ): (source, run_name)
                for source, run_name, cache_path in pending
            }
            for future in as_completed(futures):
                source, intervals, cache_path = future.result()
                print(
                    f"[complete] {source}/{futures[future][1]}: "
                    f"{intervals} intervals -> {cache_path}",
                    flush=True,
                )

    interval_counts = {display: 0 for display in selected_display}
    for source, _run_name, cache_path in tasks:
        display = common.SOURCE_TO_DISPLAY[source]
        strategy_index = selected_display.index(display)
        interval_counts[display] += merge_cache(
            cache_path,
            strategy_index,
            aggregators,
        )

    producer.DEFAULT_GATES = common.GATES
    producer.CHANGE_TYPES = ("rel",)
    output_aggregators = {
        ("overall", producer.gate_name(gate), "rel"): aggregators[gate_index]
        for gate_index, gate in enumerate(common.GATES)
    }
    plot_root = output_root / "summary_plots" / "gate_matrix_with_n"
    n_pdfs = producer.write_outputs(
        output_root=plot_root,
        config=config,
        aggregators=output_aggregators,
        strategies=selected_display,
        selected_phases=("overall",),
        min_bin_pixels=int(args.min_bin_pixels),
        annotate_n_pixels=True,
        simulation_run=True,
        escape_source=None,
    )
    (output_root / "mixed_kernel_manifest.txt").write_text(
        "Simulation mixed-kernel summary generated directly from source HDF5.\n"
        "Gate and total-density x-axis: 320-um kernel, coarsening factor 32.\n"
        "$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$: 40-um kernel, coarsening factor 4.\n"
        "Each 40-um-grid pixel uses the containing 320-um-grid pixel's density.\n"
        f"Strategies and intervals: {interval_counts}\n"
        f"PDFs: {n_pdfs}\n",
        encoding="ascii",
    )
    print(f"[done] outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
