#!/usr/bin/env python3
"""Retired simulation density kernel-sweep plotting workflow.

This reproduced the pooled-pixel relative-change gate-matrix plots without
retaining the very large intermediate TIFF trees for the finer kernels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as density_base
from analysis.figure_3 import master_curve_binning as producer


DEFAULT_INPUT_ROOT = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\data-DB"
)
DEFAULT_OUTPUT_ROOT = Path(r"F:\CoreSeed_Kernel_Sweep_InSilico")
KERNELS = ((40, 4), (80, 8), (160, 16), (320, 32))
GATES = (
    (2.0e-5, 4.375e-4),
    (2.0e-5, 1.0e-3),
    (2.0e-5, 2.0e-3),
    (1.0e-4, 4.375e-4),
    (1.0e-4, 1.0e-3),
    (1.0e-4, 2.0e-3),
    (5.0e-4, 1.0e-3),
    (5.0e-4, 2.0e-3),
)
SOURCE_TO_DISPLAY = {
    "CT": "CT",
    "NT": "NT",
    "AT50": "AT50",
    "AT100": "AT100",
    "Agent": "ATagent",
    "Dens": r"AT$^{+}$",
}
DISPLAY_ORDER = ("CT", "NT", "AT50", "AT100", "ATagent", r"AT$^{+}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=os.fspath(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=os.fspath(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--config",
        default=os.fspath(sim.DEFAULT_CONFIG_PATH),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--x-bin-size", type=float, default=0.2)
    parser.add_argument("--y-hist-bins", type=int, default=4096)
    parser.add_argument("--min-bin-pixels", type=int, default=100)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=tuple(SOURCE_TO_DISPLAY),
        default=list(SOURCE_TO_DISPLAY),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--runs", nargs="+", default=None, help="Optional simulation run filter.")
    parser.add_argument("--max-timepoints", type=int, default=0, help="Testing limit; 0 means all.")
    return parser.parse_args()


def simulation_args(
    input_root: str,
    strategy: str,
    run_name: str | None = None,
    max_timepoints: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_root=input_root,
        strategies=[strategy],
        runs=[run_name] if run_name else None,
        max_runs_per_strategy=0,
        max_timepoints=int(max_timepoints),
        time_dump=sim.DEFAULT_TIME_DUMP,
        day_offset=sim.DEFAULT_DAY_OFFSET,
        active_flag=sim.DEFAULT_ACTIVE_FLAG,
        type_dataset=sim.DEFAULT_TYPE_DATASET,
        sensitive_value=sim.DEFAULT_SENSITIVE_VALUE,
        resistant_value=sim.DEFAULT_RESISTANT_VALUE,
        position_scale_px=sim.DEFAULT_POSITION_SCALE_PX,
        position_scale_um=None,
        no_outlier_filter=False,
        filter_center=list(sim.DEFAULT_FILTER_CENTER),
        filter_outlier_boundary=sim.DEFAULT_FILTER_OUTLIER_BOUNDARY,
        filter_neighbour_radius=sim.DEFAULT_FILTER_NEIGHBOUR_RADIUS,
        filter_min_neighbours=sim.DEFAULT_FILTER_MIN_NEIGHBOURS,
    )


def canvas_from_loaded(
    loaded: list[tuple[np.ndarray, np.ndarray]],
    radius_um: float,
) -> sim.CanvasSpec:
    nonempty = [positions for positions, _types in loaded if positions.size]
    if not nonempty:
        raise RuntimeError("No filtered cells in simulation run")
    min_xy = np.nanmin(np.vstack([np.nanmin(item, axis=0) for item in nonempty]), axis=0)
    max_xy = np.nanmax(np.vstack([np.nanmax(item, axis=0) for item in nonempty]), axis=0)
    min_xy -= radius_um
    max_xy += radius_um
    span = np.maximum(max_xy - min_xy, sim.PIXEL_SIZE_UM)
    return sim.CanvasSpec(
        min_x_um=float(min_xy[0]),
        min_y_um=float(min_xy[1]),
        width_px=int(math.ceil(span[0] / sim.PIXEL_SIZE_UM)),
        height_px=int(math.ceil(span[1] / sim.PIXEL_SIZE_UM)),
    )


def resistant_and_total_density(
    sus,
    res,
    *,
    radius_um: float,
    canvas: sim.CanvasSpec,
    coarsening_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    res_image = density_base.rasterize_points_to_coarse_grid(
        res,
        canvas.height_px,
        canvas.width_px,
        coarsening_factor,
    )
    sus_image = density_base.rasterize_points_to_coarse_grid(
        sus,
        canvas.height_px,
        canvas.width_px,
        coarsening_factor,
    )
    coarse_radius = density_base.radius_um_to_coarse_px(radius_um, coarsening_factor)
    res_density = density_base.get_local_density_fft(res_image, coarse_radius)
    total_density = density_base.get_local_density_fft(res_image + sus_image, coarse_radius)
    coarse_pixel_area_um2 = (sim.PIXEL_SIZE_UM * coarsening_factor) ** 2
    res_density = density_base.zero_density_fft_noise(res_density / coarse_pixel_area_um2)
    total_density = density_base.zero_density_fft_noise(total_density / coarse_pixel_area_um2)
    return res_density, total_density


def create_aggregators(
    x_edges: np.ndarray,
    ylim: tuple[float, float],
    n_y_bins: int,
) -> dict[tuple[int, int], producer.Aggregator]:
    return {
        (kernel_um, gate_index): producer.Aggregator.create(
            n_strategies=1,
            x_edges=x_edges,
            ylim=ylim,
            n_y_bins=n_y_bins,
        )
        for kernel_um, _coarse in KERNELS
        for gate_index, _gate in enumerate(GATES)
    }


def accumulate_interval(
    aggregators: dict[tuple[int, int], producer.Aggregator],
    kernel_um: int,
    previous_total: np.ndarray,
    previous_res: np.ndarray,
    next_res: np.ndarray,
    delta_days: float,
    x_edges: np.ndarray,
    ylim: tuple[float, float],
) -> None:
    x = previous_total.ravel().astype(np.float64, copy=False) * producer.base.X_DISPLAY_SCALE
    previous = previous_res.ravel()
    nxt = next_res.ravel()
    y = np.zeros(previous.shape, dtype=np.float32)
    valid_growth = (
        np.isfinite(previous)
        & np.isfinite(nxt)
        & (previous > sim.DENSITY_ZERO_FLOOR_UM2)
        & (nxt > sim.DENSITY_ZERO_FLOOR_UM2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        y[valid_growth] = (
            np.log(nxt[valid_growth] / previous[valid_growth]) / float(delta_days)
        ).astype(np.float32, copy=False)

    x_bin_size = float(x_edges[1] - x_edges[0])
    n_x_bins = len(x_edges) - 1
    x_valid = np.isfinite(x) & (x >= x_edges[0]) & (x <= x_edges[-1])
    x_bins = np.zeros(x.shape, dtype=np.int64)
    finite_x = np.isfinite(x)
    x_bins[finite_x] = np.floor((x[finite_x] - x_edges[0]) / x_bin_size).astype(np.int64)
    np.clip(x_bins, 0, n_x_bins - 1, out=x_bins)
    finite_y = np.isfinite(y) & (y >= ylim[0]) & (y <= ylim[1])

    for gate_index, gate in enumerate(GATES):
        valid = (
            x_valid
            & finite_y
            & np.isfinite(previous)
            & (previous >= gate[0])
            & (previous <= gate[1])
        )
        aggregators[(kernel_um, gate_index)].add(
            0,
            x_bins[valid],
            y[valid].astype(np.float64, copy=False),
        )


def save_strategy_cache(
    path: Path,
    aggregators: dict[tuple[int, int], producer.Aggregator],
    intervals: int,
) -> None:
    arrays: dict[str, np.ndarray] = {"intervals": np.asarray([intervals], dtype=np.int64)}
    for (kernel_um, gate_index), aggregator in aggregators.items():
        prefix = f"k{kernel_um}_g{gate_index}"
        arrays[f"{prefix}_count"] = aggregator.count[0]
        arrays[f"{prefix}_total"] = aggregator.total[0]
        arrays[f"{prefix}_total_sq"] = aggregator.total_sq[0]
        arrays[f"{prefix}_histogram"] = aggregator.histogram[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def process_strategy(
    strategy: str,
    run_name: str,
    input_root: str,
    cache_path: str,
    x_edges: np.ndarray,
    ylim: tuple[float, float],
    n_y_bins: int,
    max_timepoints: int,
) -> tuple[str, int, str]:
    args = simulation_args(input_root, strategy, run_name, max_timepoints)
    plans = sim.discover_run_plans(args)
    aggregators = create_aggregators(x_edges, ylim, n_y_bins)
    intervals = 0
    for plan_index, plan in enumerate(plans, start=1):
        loaded = [sim.load_filtered_cells(item.path, args) for item in plan.files]
        canvases = {
            kernel_um: canvas_from_loaded(loaded, float(kernel_um))
            for kernel_um, _coarse in KERNELS
        }
        previous: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for sim_file, (positions_um, type_values) in zip(plan.files, loaded):
            for kernel_um, coarse_factor in KERNELS:
                sus, res = sim.positions_to_density_inputs(
                    positions_um,
                    type_values,
                    canvases[kernel_um],
                )
                res_density, total_density = resistant_and_total_density(
                    sus,
                    res,
                    radius_um=float(kernel_um),
                    canvas=canvases[kernel_um],
                    coarsening_factor=coarse_factor,
                )
                if kernel_um in previous:
                    previous_res, previous_total, previous_day = previous[kernel_um]
                    accumulate_interval(
                        aggregators,
                        kernel_um,
                        previous_total,
                        previous_res,
                        res_density,
                        float(sim_file.day - previous_day),
                        x_edges,
                        ylim,
                    )
                previous[kernel_um] = (res_density, total_density, sim_file.day)
            if previous:
                intervals += 1 if len(previous) and sim_file is not plan.files[0] else 0
        print(
            f"[{strategy}] run {plan_index}/{len(plans)} {plan.run_id} complete",
            flush=True,
        )
    save_strategy_cache(Path(cache_path), aggregators, intervals)
    return strategy, intervals, cache_path


def merge_cache(
    path: Path,
    strategy_index: int,
    aggregators: dict[tuple[int, int], producer.Aggregator],
) -> int:
    with np.load(path) as data:
        intervals = int(data["intervals"][0])
        for (kernel_um, gate_index), aggregator in aggregators.items():
            prefix = f"k{kernel_um}_g{gate_index}"
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
    x_max = float(settings["xlim"][1])
    n_x_bins = int(math.ceil(x_max / float(args.x_bin_size)))
    x_edges = np.arange(n_x_bins + 1, dtype=float) * float(args.x_bin_size)
    selected_sources = list(dict.fromkeys(args.strategies))
    selected_display = [
        display
        for display in DISPLAY_ORDER
        if display in {SOURCE_TO_DISPLAY[source] for source in selected_sources}
    ]
    global_aggregators = {
        (kernel_um, gate_index): producer.Aggregator.create(
            n_strategies=len(selected_display),
            x_edges=x_edges,
            ylim=settings["ylim"],
            n_y_bins=int(args.y_hist_bins),
        )
        for kernel_um, _coarse in KERNELS
        for gate_index, _gate in enumerate(GATES)
    }

    cache_dir = output_root / "aggregation_cache"
    tasks: list[tuple[str, str, Path]] = []
    for source in selected_sources:
        plans = sim.discover_run_plans(simulation_args(os.fspath(input_root), source))
        if args.runs:
            requested_runs = {
                value if str(value).startswith("run_") else f"run_{value}"
                for value in args.runs
            }
            plans = [plan for plan in plans if plan.run_id in requested_runs]
        for plan in plans:
            tasks.append(
                (
                    source,
                    plan.run_id,
                    cache_dir / source / f"{plan.run_id}_kernel_sweep.npz",
                )
            )
    pending = [
        task
        for task in tasks
        if args.overwrite_cache or not task[2].exists()
    ]
    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(pending))) as pool:
            futures = {
                pool.submit(
                    process_strategy,
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
                    f"[complete] {futures[future][0]}/{futures[future][1]}: "
                    f"{intervals} intervals -> {cache_path}",
                    flush=True,
                )

    interval_counts = {display: 0 for display in selected_display}
    for source, _run_name, cache_path in tasks:
        display = SOURCE_TO_DISPLAY[source]
        strategy_index = selected_display.index(display)
        interval_counts[display] += merge_cache(
            cache_path,
            strategy_index,
            global_aggregators,
        )

    producer.DEFAULT_GATES = GATES
    producer.CHANGE_TYPES = ("rel",)
    for kernel_um, coarse_factor in KERNELS:
        run_root = output_root / f"run_kernel_{kernel_um}_coarse_{coarse_factor}"
        plot_root = run_root / "summary_plots" / "gate_matrix_with_n"
        kernel_aggregators = {
            ("overall", producer.gate_name(gate), "rel"): global_aggregators[
                (kernel_um, gate_index)
            ]
            for gate_index, gate in enumerate(GATES)
        }
        n_pdfs = producer.write_outputs(
            output_root=plot_root,
            config=config,
            aggregators=kernel_aggregators,
            strategies=selected_display,
            selected_phases=("overall",),
            min_bin_pixels=int(args.min_bin_pixels),
            annotate_n_pixels=True,
            simulation_run=True,
            escape_source=None,
        )
        (run_root / "streamed_plot_manifest.txt").write_text(
            "Simulation kernel summary generated directly from source HDF5.\n"
            "Intermediate density-map TIFFs were not retained.\n"
            f"Kernel radius: {kernel_um} um\n"
            f"Coarsening factor: {coarse_factor}\n"
            f"Strategies and intervals: {interval_counts}\n"
            f"PDFs: {n_pdfs}\n",
            encoding="ascii",
        )
    print(f"[done] outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
