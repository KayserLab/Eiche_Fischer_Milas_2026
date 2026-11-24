#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Trace low-density, gated, pre-escape pixels for the 160-um master curve."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, FIGURE_DIR, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as density_base
from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
from analysis.figure_3 import aggregate_simulation_master_curve_iqr_direct as sim_aggregate
from analysis.figure_3 import master_curve_iqr_common as common
import plot_binned_density_change_strategy_lines_coarsegrained as plotter
import plot_density_change_datapoints_coarsegrained as base
import plot_simulation_kernel_sweep_direct as sweep
from shared.strategy_colors import normalize_strategy


KERNEL_UM = 160
COARSENING_FACTOR = 16
COARSE_PIXEL_UM = density_base.PIXEL_SIZE_UM * COARSENING_FACTOR
GATE = (2.0e-5, 4.375e-4)
X_EDGES = np.arange(5, dtype=float) * 0.2
BIN_LABELS = tuple(f"{X_EDGES[i]:.1f}-{X_EDGES[i + 1]:.1f}" for i in range(4))
ZERO_FLOOR = common.ZERO_FLOOR

DEFAULT_INVITRO_RUN = Path(
    r"F:\CoreSeed_InVitro_PreEscape_160\run_kernel_160_coarse_16"
)
DEFAULT_CONFIG = ANALYSIS_DIR / "density_maps_epanechnikov_config_160_all_strategies_all_plates.yaml"
DEFAULT_EXPERIMENT_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
DEFAULT_ESCAPE_SUMMARY = (
    DEFAULT_EXPERIMENT_ROOT
    / "Figures"
    / "Distance_to_front_analysis"
    / "frontcontact_summary_ALL.csv"
)
DEFAULT_SIM_ROOT = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\data-DB"
)
DEFAULT_SIM_ESCAPE = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\combined_results.csv"
)
DEFAULT_OUTPUT = (
    Path.home() / "Desktop" / "CoreSeed_160um_PreEscape_LowDensity_Diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-vitro-run", type=Path, default=DEFAULT_INVITRO_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--escape-summary", type=Path, default=DEFAULT_ESCAPE_SUMMARY)
    parser.add_argument("--simulation-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--simulation-escape", type=Path, default=DEFAULT_SIM_ESCAPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulation-workers", type=int, default=4)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-mask-spatial", action="store_true")
    return parser.parse_args()


def x_bin_indices(total_density: np.ndarray) -> np.ndarray:
    x = total_density.astype(np.float64, copy=False) * base.X_DISPLAY_SCALE
    indices = np.floor(x / 0.2).astype(np.int16, copy=False)
    indices[~np.isfinite(x)] = -1
    return indices


def project_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.shape == target_shape:
        return source
    factor_y = max(1, math.ceil(source.shape[0] / target_shape[0]))
    factor_x = max(1, math.ceil(source.shape[1] / target_shape[1]))
    row_starts = np.arange(0, source.shape[0], factor_y)
    col_starts = np.arange(0, source.shape[1], factor_x)
    reduced_y = np.logical_or.reduceat(source, row_starts, axis=0)
    reduced = np.logical_or.reduceat(reduced_y, col_starts, axis=1)
    return reduced[: target_shape[0], : target_shape[1]]


def signed_distance_um(mask: np.ndarray) -> np.ndarray:
    inside = np.asarray(mask, dtype=bool)
    if not np.any(inside):
        return np.full(inside.shape, np.nan, dtype=np.float32)
    outside_distance = distance_transform_edt(~inside)
    inside_distance = distance_transform_edt(inside)
    signed = np.where(inside, -inside_distance, outside_distance)
    return (signed * COARSE_PIXEL_UM).astype(np.float32, copy=False)


def mask_frame(
    path: Path,
    frame_index: int,
    target_shape: tuple[int, int],
) -> np.ndarray:
    frame = tiff.imread(path, key=int(frame_index))
    return project_mask(frame > 0, target_shape)


def attach_experimental_spatial(
    points: pd.DataFrame,
    *,
    experiment_root: Path,
    plate: str,
    well_identifier: str,
    frame_index: int,
    target_shape: tuple[int, int],
) -> pd.DataFrame:
    if points.empty:
        return points
    clone_path = (
        experiment_root
        / plate
        / "clone_masks_stacks"
        / f"{well_identifier}_res_clone_masks.tif"
    )
    colony_path = (
        experiment_root
        / plate
        / "colony_masks_stacks"
        / f"{well_identifier}_total_colony_masks.tif"
    )
    if not clone_path.is_file() or not colony_path.is_file():
        points["spatial_masks_available"] = False
        return points
    clone_mask = mask_frame(clone_path, frame_index, target_shape)
    colony_mask = mask_frame(colony_path, frame_index, target_shape)
    clone_distance = signed_distance_um(clone_mask)
    colony_distance = signed_distance_um(colony_mask)
    rows = points["row"].to_numpy(dtype=np.int64)
    cols = points["col"].to_numpy(dtype=np.int64)
    points["spatial_masks_available"] = True
    points["inside_clone_mask"] = clone_mask[rows, cols]
    points["inside_colony_mask"] = colony_mask[rows, cols]
    points["signed_distance_to_clone_um"] = clone_distance[rows, cols]
    points["signed_distance_to_colony_um"] = colony_distance[rows, cols]
    return points


def interval_points(
    previous_total: np.ndarray,
    previous_res: np.ndarray,
    next_res: np.ndarray,
    delta_days: float,
) -> pd.DataFrame:
    x = previous_total.astype(np.float64, copy=False) * base.X_DISPLAY_SCALE
    bins = x_bin_indices(previous_total)
    valid = (
        np.isfinite(x)
        & np.isfinite(previous_res)
        & np.isfinite(next_res)
        & (previous_res > ZERO_FLOOR)
        & (previous_res >= GATE[0])
        & (previous_res <= GATE[1])
        & (bins >= 0)
        & (bins < len(BIN_LABELS))
    )
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return pd.DataFrame()
    prev = previous_res[rows, cols].astype(np.float64, copy=False)
    nxt = next_res[rows, cols].astype(np.float64, copy=False)
    growth = (nxt / prev - 1.0) / float(delta_days)
    result = pd.DataFrame(
        {
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "x_bin_index": bins[rows, cols].astype(np.int8),
            "total_density_cells_per_1000_um2": x[rows, cols].astype(np.float32),
            "previous_res_density": prev.astype(np.float32),
            "next_res_density": nxt.astype(np.float32),
            "growth_per_day": growth.astype(np.float32),
        }
    )
    result["x_bin"] = result["x_bin_index"].map(dict(enumerate(BIN_LABELS)))
    result["resistant_fraction_of_total"] = (
        result["previous_res_density"]
        / (result["total_density_cells_per_1000_um2"] / base.X_DISPLAY_SCALE)
    )
    result["complete_signal_loss"] = result["next_res_density"] <= ZERO_FLOOR
    return result


def extract_in_vitro(
    run_dir: Path,
    config_path: Path,
    escape_summary: Path,
    experiment_root: Path,
    *,
    add_mask_spatial: bool,
) -> pd.DataFrame:
    config = base.load_config(config_path)
    resolution_args = argparse.Namespace(
        experimental_escape_summary=os.fspath(escape_summary),
        simulation_escape_csv=None,
    )
    cutoffs, simulation_run, _source = plotter.resolve_escape_cutoffs(
        run_dir,
        config,
        resolution_args,
    )
    if simulation_run:
        raise ValueError(f"Expected experimental run: {run_dir}")

    chunks: list[pd.DataFrame] = []
    scanned = 0
    for changes_dir in sorted(run_dir.glob("Plate_*/*/density_changes")):
        plate = changes_dir.parent.parent.name
        well_identifier = changes_dir.parent.name
        strategy = base.strategy_from_plate_and_well(plate, well_identifier)
        if strategy not in common.ALLOWED_STRATEGIES:
            continue
        maps_dir = changes_dir.parent / "density_maps"
        total_paths = sorted(maps_dir.glob("*_total_density_epanechnikov.tif"))
        timepoint_pattern = re.compile(
            rf"^{re.escape(well_identifier)}_(?P<tp>[^_]+)_total_density_epanechnikov\.tif$"
        )
        timepoints = []
        for path in total_paths:
            match = timepoint_pattern.match(path.name)
            if match is not None:
                timepoints.append(match.group("tp"))
        frame_by_timepoint = {value: index for index, value in enumerate(timepoints)}

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
                well=well_identifier,
                first_day=first_day,
                next_day=next_day,
                escape_cutoffs=cutoffs,
                simulation_run=False,
            )
            if "before_escape" not in phases:
                continue
            cutoff = cutoffs.get((plate, well_identifier), (np.nan, np.nan))[1]
            total_path = base.density_map_path(maps_dir, well_identifier, first_tp)
            res_path = base.resistant_density_map_path(maps_dir, well_identifier, first_tp)
            abs_path = rel_path.with_name(
                rel_path.name.replace("_rel_diff.tif", "_abs_diff.tif")
            )
            previous_total = tiff.imread(total_path).astype(np.float32, copy=False)
            previous_res = tiff.imread(res_path).astype(np.float32, copy=False)
            abs_change = tiff.imread(abs_path).astype(np.float32, copy=False)
            next_res = previous_res + abs_change
            points = interval_points(
                previous_total,
                previous_res,
                next_res,
                float(next_day - first_day),
            )
            scanned += 1
            if points.empty:
                continue
            points.insert(0, "dataset", "in_vitro")
            points.insert(1, "strategy", strategy)
            points.insert(2, "replicate", f"{plate}/{well_identifier}")
            points.insert(3, "plate", plate)
            points.insert(4, "well", well_identifier)
            points["first_day"] = first_day
            points["next_day"] = next_day
            points["delta_days"] = next_day - first_day
            points["escape_cutoff_day"] = cutoff
            points["days_before_escape_cutoff"] = cutoff - next_day
            if add_mask_spatial and first_tp in frame_by_timepoint:
                points = attach_experimental_spatial(
                    points,
                    experiment_root=experiment_root,
                    plate=plate,
                    well_identifier=well_identifier,
                    frame_index=frame_by_timepoint[first_tp],
                    target_shape=previous_total.shape,
                )
            chunks.append(points)
    if not chunks:
        raise RuntimeError("No in-vitro low-density pixels were selected")
    result = pd.concat(chunks, ignore_index=True)
    print(
        f"[in vitro] intervals={scanned}; selected pixels={len(result):,}; "
        f"replicates={result['replicate'].nunique()}"
    )
    return result


def simulation_density_and_occupancy(
    positions_um: np.ndarray,
    type_values: np.ndarray,
    canvas: sim.CanvasSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    susceptible, resistant = sim.positions_to_density_inputs(
        positions_um,
        type_values,
        canvas,
    )
    resistant_image = density_base.rasterize_points_to_coarse_grid(
        resistant,
        canvas.height_px,
        canvas.width_px,
        COARSENING_FACTOR,
    )
    susceptible_image = density_base.rasterize_points_to_coarse_grid(
        susceptible,
        canvas.height_px,
        canvas.width_px,
        COARSENING_FACTOR,
    )
    coarse_radius = density_base.radius_um_to_coarse_px(
        KERNEL_UM,
        COARSENING_FACTOR,
    )
    resistant_density = density_base.get_local_density_fft(
        resistant_image,
        coarse_radius,
    )
    total_image = resistant_image + susceptible_image
    total_density = density_base.get_local_density_fft(
        total_image,
        coarse_radius,
    )
    coarse_area = COARSE_PIXEL_UM ** 2
    resistant_density = density_base.zero_density_fft_noise(
        resistant_density / coarse_area
    )
    total_density = density_base.zero_density_fft_noise(total_density / coarse_area)
    return resistant_density, total_density, resistant_image, total_image


def attach_simulation_spatial(
    points: pd.DataFrame,
    resistant_occupancy: np.ndarray,
    total_occupancy: np.ndarray,
) -> pd.DataFrame:
    if points.empty:
        return points
    resistant_mask = resistant_occupancy > 0
    total_mask = total_occupancy > 0
    resistant_distance = signed_distance_um(resistant_mask)
    total_distance = signed_distance_um(total_mask)
    rows = points["row"].to_numpy(dtype=np.int64)
    cols = points["col"].to_numpy(dtype=np.int64)
    points["spatial_masks_available"] = True
    points["inside_clone_mask"] = resistant_mask[rows, cols]
    points["inside_colony_mask"] = total_mask[rows, cols]
    points["signed_distance_to_clone_um"] = resistant_distance[rows, cols]
    points["signed_distance_to_colony_um"] = total_distance[rows, cols]
    points["raw_resistant_cells_in_coarse_pixel"] = resistant_occupancy[rows, cols]
    points["raw_total_cells_in_coarse_pixel"] = total_occupancy[rows, cols]
    return points


def extract_one_simulation_run(
    input_root: str,
    run_name: str,
    cutoff_day: float,
) -> pd.DataFrame:
    sim_args = sweep.simulation_args(input_root, "CT", run_name)
    plans = sim.discover_run_plans(sim_args)
    if len(plans) != 1:
        raise RuntimeError(f"Expected one CT plan for {run_name}, found {len(plans)}")
    plan = plans[0]
    loaded = [sim.load_filtered_cells(item.path, sim_args) for item in plan.files]
    canvas = sweep.canvas_from_loaded(loaded, float(KERNEL_UM))
    previous: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ] | None = None
    chunks: list[pd.DataFrame] = []
    for sim_file, (positions_um, type_values) in zip(plan.files, loaded):
        if float(sim_file.day) > float(cutoff_day):
            break
        current = simulation_density_and_occupancy(
            positions_um,
            type_values,
            canvas,
        )
        if previous is not None:
            previous_res, previous_total, previous_res_image, previous_total_image, previous_day = previous
            delta_days = float(sim_file.day) - previous_day
            points = interval_points(
                previous_total,
                previous_res,
                current[0],
                delta_days,
            )
            if not points.empty:
                points = attach_simulation_spatial(
                    points,
                    previous_res_image,
                    previous_total_image,
                )
                points.insert(0, "dataset", "in_silico")
                points.insert(1, "strategy", "CT")
                points.insert(2, "replicate", run_name)
                points.insert(3, "plate", "simulation_CT")
                points.insert(4, "well", run_name)
                points["first_day"] = previous_day
                points["next_day"] = float(sim_file.day)
                points["delta_days"] = delta_days
                points["escape_cutoff_day"] = float(cutoff_day)
                points["days_before_escape_cutoff"] = float(cutoff_day) - float(sim_file.day)
                chunks.append(points)
        previous = (current[0], current[1], current[2], current[3], float(sim_file.day))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def extract_in_silico(
    input_root: Path,
    escape_csv: Path,
    workers: int,
) -> pd.DataFrame:
    tte = sim_aggregate.load_positive_tte(escape_csv)
    plans = sim.discover_run_plans(sweep.simulation_args(os.fspath(input_root), "CT"))
    tasks: list[tuple[str, float]] = []
    for plan in plans:
        key = (
            normalize_strategy(plan.strategy),
            base.normalize_simulation_run_id(plan.run_id),
        )
        cutoff = tte.get(key)
        if cutoff is not None and np.isfinite(cutoff) and cutoff > 0:
            tasks.append((plan.run_id, float(cutoff)))
    if not tasks:
        raise RuntimeError("No CT simulation runs matched a positive TTE")

    chunks: list[pd.DataFrame] = []
    if workers <= 1:
        for index, (run_name, cutoff) in enumerate(tasks, start=1):
            chunks.append(extract_one_simulation_run(os.fspath(input_root), run_name, cutoff))
            print(f"[in silico] {index}/{len(tasks)} {run_name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = {
                pool.submit(
                    extract_one_simulation_run,
                    os.fspath(input_root),
                    run_name,
                    cutoff,
                ): run_name
                for run_name, cutoff in tasks
            }
            completed = 0
            for future in as_completed(futures):
                run_name = futures[future]
                chunks.append(future.result())
                completed += 1
                print(f"[in silico] {completed}/{len(tasks)} {run_name}", flush=True)
    nonempty = [chunk for chunk in chunks if not chunk.empty]
    if not nonempty:
        raise RuntimeError("No in-silico low-density pixels were selected")
    result = pd.concat(nonempty, ignore_index=True)
    print(
        f"[in silico] selected pixels={len(result):,}; "
        f"runs={result['replicate'].nunique()}"
    )
    return result


def validate_simulation_counts(points: pd.DataFrame) -> None:
    cache_path = (
        Path(r"F:\CoreSeed_Master_Curve_Linear_IQR_Cache")
        / "in_silico"
        / "kernel_160um_all_phases_iqr.npz"
    )
    bundle = common.load_bundle(cache_path)
    strategy_index = bundle.strategies.index("CT")
    expected = bundle.aggregators[("gate_2e-05_0p0004375", "before_escape")].count[
        strategy_index, : len(BIN_LABELS)
    ]
    observed = (
        points.groupby("x_bin_index").size().reindex(range(len(BIN_LABELS)), fill_value=0).to_numpy()
    )
    if not np.array_equal(expected, observed):
        raise AssertionError(
            f"Simulation count mismatch: expected={expected.tolist()}, observed={observed.tolist()}"
        )
    print(f"[validated] simulation counts={observed.tolist()}")


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = ["dataset", "strategy", "x_bin_index", "x_bin"]
    for keys, group in points.groupby(grouping, observed=True, sort=True):
        y = group["growth_per_day"].to_numpy(dtype=float)
        row: dict[str, object] = dict(zip(grouping, keys))
        row.update(
            n_pixels=len(group),
            n_replicates=group["replicate"].nunique(),
            median=float(np.median(y)),
            q1=float(np.quantile(y, 0.25)),
            q3=float(np.quantile(y, 0.75)),
            mean=float(np.mean(y)),
            fraction_negative=float(np.mean(y < 0)),
            fraction_near_zero=float(np.mean(np.abs(y) <= 0.05)),
            fraction_growth_ge_1=float(np.mean(y >= 1.0)),
            fraction_growth_ge_2=float(np.mean(y >= 2.0)),
            fraction_complete_signal_loss=float(group["complete_signal_loss"].mean()),
            median_first_day=float(group["first_day"].median()),
            median_days_before_escape=float(group["days_before_escape_cutoff"].median()),
            median_resistant_fraction=float(group["resistant_fraction_of_total"].median()),
        )
        if "spatial_masks_available" in group and group["spatial_masks_available"].fillna(False).any():
            spatial = group[group["spatial_masks_available"].fillna(False)]
            row.update(
                n_spatial_pixels=len(spatial),
                fraction_outside_clone_mask=float((~spatial["inside_clone_mask"].astype(bool)).mean()),
                fraction_outside_colony_mask=float((~spatial["inside_colony_mask"].astype(bool)).mean()),
                median_signed_clone_distance_um=float(spatial["signed_distance_to_clone_um"].median()),
                median_signed_colony_distance_um=float(spatial["signed_distance_to_colony_um"].median()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    in_vitro = extract_in_vitro(
        args.in_vitro_run.expanduser(),
        args.config.expanduser(),
        args.escape_summary.expanduser(),
        args.experiment_root.expanduser(),
        add_mask_spatial=not args.skip_mask_spatial,
    )
    in_vitro.to_csv(
        output_dir / "in_vitro_selected_low_density_pixels.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    summary = summarize(in_vitro)
    summary.to_csv(output_dir / "in_vitro_summary_by_strategy_bin.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    if args.skip_simulation:
        print(f"[done] {output_dir}")
        return
    in_silico = extract_in_silico(
        args.simulation_root.expanduser(),
        args.simulation_escape.expanduser(),
        max(1, int(args.simulation_workers)),
    )
    validate_simulation_counts(in_silico)
    in_silico.to_csv(
        output_dir / "in_silico_CT_selected_low_density_pixels.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    combined_summary = pd.concat(
        [summarize(in_vitro), summarize(in_silico)],
        ignore_index=True,
    )
    combined_summary.to_csv(
        output_dir / "summary_by_dataset_strategy_bin.tsv",
        sep="\t",
        index=False,
    )
    print(combined_summary.to_string(index=False))
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
