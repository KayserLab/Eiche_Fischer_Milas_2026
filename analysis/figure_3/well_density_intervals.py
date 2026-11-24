"""Experimental and matched simulation intervals for supplementary density plots.

Retains the 160-um kernel, 16-fold coarsening, cohort and growth definitions.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
from shared import experiment_filters
from analysis.figure_3 import density_common as base
from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
from shared.data_paths import SIMULATION_DATABASE, intermediate_dir

DEFAULT_RUN_DIR = intermediate_dir(
    "experimental", "figure_3", "figure2_mask_filtered_mastercurve",
    "run_kernel_160_coarse_16_figure2_masks_all_clones", create=False,
)
DEFAULT_SIMULATION_RUN_DIR = SIMULATION_DATABASE
KERNEL_UM = 160
COARSENING_FACTOR = 16

SIMULATION_MATCHED_RANGES = {
    'Plate_AT50': range(10, 20),
    'Plate_AT100': range(20, 30),
    'Plate_CT': range(30, 42),
    'Plate_Dens': range(42, 50),
}
SIMULATION_INITIAL_WELLS = {
    'Plate_AT50': ('B1', 'B2', 'B6', 'C2', 'C5', 'D1', 'D3', 'D4', 'D5', 'D6'),
    'Plate_AT100': ('A2', 'A5', 'B3', 'B5', 'B6', 'C1', 'C2', 'C4', 'C6', 'D4'),
    'Plate_CT': ('A1', 'A3', 'B2', 'B3', 'C1', 'C2', 'C4', 'C5', 'D2', 'D3', 'D4', 'D5'),
    'Plate_Dens': ('B2', 'B3', 'B4', 'C2', 'C3', 'C4', 'D3', 'D5'),
}
REL_RE = re.compile(
    r"^(?P<well>.+)_(?P<first>[^_]+)_to_(?P<next>[^_]+)_"
    r"res_density_epanechnikov_rel_diff\.tif$"
)


def simulation_density(
    positions_um: np.ndarray,
    type_values: np.ndarray,
    canvas: sim.CanvasSpec,
) -> tuple[np.ndarray, np.ndarray]:
    susceptible, resistant = sim.positions_to_density_inputs(
        positions_um,
        type_values,
        canvas,
    )
    return base.resistant_and_total_density(
        susceptible,
        resistant,
        radius_um=float(KERNEL_UM),
        canvas=canvas,
        coarsening_factor=COARSENING_FACTOR,
    )



def filter_rule(well: str) -> tuple[str, float | None]:
    rules = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    if well not in rules:
        return "keep", None
    value = rules[well]
    if value is None:
        return "exclude", None
    try:
        return "truncate", float(value)
    except (TypeError, ValueError):
        return "keep", None



def summarize(values: np.ndarray, statistic: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if statistic == "median":
        return float(np.median(finite))
    return float(np.mean(finite))



def density_weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the density-weighted spatial standard deviation."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    values, weights = values[valid], weights[valid]
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))



def is_matched_simulation_run(plate: str, well: str) -> bool:
    match = re.match(r'^R(?P<replicate>\d+)_', well)
    allowed = SIMULATION_MATCHED_RANGES.get(plate)
    return bool(match and allowed is not None and int(match.group('replicate')) in allowed)



def scan_intervals(
    run_dir: Path, gate: tuple[float, float],
    only_plate: str | None = None, only_well: str | None = None,
    matching_simulation_runs: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lower, upper = gate
    for changes_dir in sorted(run_dir.glob("Plate_*/*/density_changes")):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        if only_plate is not None and plate != only_plate:
            continue
        if only_well is not None and well != only_well:
            continue
        if matching_simulation_runs and not is_matched_simulation_run(plate, well):
            continue
        status, cutoff = filter_rule(well)
        if status == "exclude":
            continue
        strategy = base.strategy_from_plate_and_well(plate, well)
        maps_dir = changes_dir.parent / "density_maps"

        for rel_path in sorted(changes_dir.glob("*_res_density_epanechnikov_rel_diff.tif")):
            match = REL_RE.match(rel_path.name)
            if match is None or match.group("well") != well:
                continue
            first_tp = match.group("first")
            next_tp = match.group("next")
            first_day = base.timepoint_to_day(first_tp)
            next_day = base.timepoint_to_day(next_tp)
            delta_days = next_day - first_day
            if not np.isfinite(delta_days) or delta_days <= 0:
                continue
            if status == "truncate" and cutoff is not None and next_day > cutoff:
                continue

            first_total_path = maps_dir / f"{well}_{first_tp}_total_density_epanechnikov.tif"
            next_total_path = maps_dir / f"{well}_{next_tp}_total_density_epanechnikov.tif"
            first_res_path = maps_dir / f"{well}_{first_tp}_res_density_epanechnikov.tif"
            next_res_path = maps_dir / f"{well}_{next_tp}_res_density_epanechnikov.tif"
            paths = (first_total_path, next_total_path, first_res_path, next_res_path)
            if not all(path.is_file() for path in paths):
                continue

            first_total = tiff.imread(first_total_path).astype(np.float32, copy=False).ravel()
            next_total = tiff.imread(next_total_path).astype(np.float32, copy=False).ravel()
            first_res = tiff.imread(first_res_path).astype(np.float32, copy=False).ravel()
            next_res = tiff.imread(next_res_path).astype(np.float32, copy=False).ravel()
            if len({array.shape for array in (first_total, next_total, first_res, next_res)}) != 1:
                raise ValueError(f"Shape mismatch for {plate}/{well} {first_tp}->{next_tp}")

            midpoint_total = 0.5 * (first_total + next_total)
            valid = (
                np.isfinite(midpoint_total)
                & np.isfinite(first_res)
                & np.isfinite(next_res)
                & (first_res >= lower)
                & (first_res <= upper)
                & (next_res >= lower)
                & (next_res <= upper)
            )
            if not np.any(valid):
                continue
            total_values = midpoint_total[valid] * base.X_DISPLAY_SCALE
            abs_per_day = (next_res[valid] - first_res[valid]) / np.float32(delta_days) * base.X_DISPLAY_SCALE
            symmetric_rel_per_day = ((next_res[valid] - first_res[valid]) / (0.5 * (first_res[valid] + next_res[valid])) / np.float32(delta_days))

            row: dict[str, object] = {
                "plate": plate,
                "well": well,
                "strategy": strategy,
                "first_tp": first_tp,
                "next_tp": next_tp,
                "first_day": first_day,
                "next_day": next_day,
                "mid_day": 0.5 * (first_day + next_day),
                "delta_days": delta_days,
                "n_pixels": int(np.count_nonzero(valid)),
            }
            for statistic in ("median", "mean"):
                row[f"density_{statistic}"] = summarize(total_values, statistic)
                row[f"relative_{statistic}"] = summarize(symmetric_rel_per_day, statistic)
                row[f"absolute_{statistic}"] = summarize(abs_per_day, statistic)
            row["density_area_mean"] = float(np.mean(total_values))
            row["density_area_std"] = float(np.std(total_values, ddof=1))
            row["net_relative_resistant_growth"] = float(
                np.sum(next_res[valid] - first_res[valid])
                / np.sum(0.5 * (first_res[valid] + next_res[valid]))
                / delta_days
            )
            row["net_relative_resistant_growth_std"] = density_weighted_std(
                symmetric_rel_per_day, 0.5 * (first_res[valid] + next_res[valid]),
            )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No complete well/timepoint pairs found under {run_dir}")
    return pd.DataFrame(rows)



def scan_matched_simulation_intervals(
    simulation_root: Path, gate: tuple[float, float],
) -> pd.DataFrame:
    """Recalculate the matched simulations directly at 160 um / 16x."""
    lower, upper = gate
    rows: list[dict[str, object]] = []
    source_by_plate = {
        "Plate_AT50": "AT50",
        "Plate_AT100": "AT100",
        "Plate_CT": "CT",
        "Plate_Dens": "Dens",
    }
    display_strategy = {
        "Plate_AT50": "AT50",
        "Plate_AT100": "AT100",
        "Plate_CT": "CT",
        "Plate_Dens": r"AT$^{+}$",
    }
    for plate, source in source_by_plate.items():
        args = base.simulation_args(os.fspath(simulation_root), source)
        for plan in sim.discover_run_plans(args):
            match = re.fullmatch(r"run_(\d+)", str(plan.run_id))
            if match is None or int(match.group(1)) not in SIMULATION_MATCHED_RANGES[plate]:
                continue
            print(f"[simulation] {plate}/{plan.run_id}", flush=True)
            loaded = [
                sim.load_filtered_cells(item.path, args)
                for item in plan.files
            ]
            canvas = base.canvas_from_loaded(
                loaded, float(KERNEL_UM)
            )
            previous: tuple[object, np.ndarray, np.ndarray] | None = None
            for item, (positions_um, type_values) in zip(plan.files, loaded):
                resistant, total = simulation_density(
                    positions_um, type_values, canvas
                )
                if previous is None:
                    previous = (item, resistant, total)
                    continue
                first_item, first_resistant, first_total = previous
                first_day = float(first_item.day)
                next_day = float(item.day)
                delta_days = next_day - first_day
                if not np.isfinite(delta_days) or delta_days <= 0:
                    previous = (item, resistant, total)
                    continue
                first_res = first_resistant.ravel()
                next_res = resistant.ravel()
                midpoint_total = 0.5 * (first_total.ravel() + total.ravel())
                valid = (
                    np.isfinite(midpoint_total)
                    & np.isfinite(first_res)
                    & np.isfinite(next_res)
                    & (first_res >= lower) & (first_res <= upper)
                    & (next_res >= lower) & (next_res <= upper)
                )
                if np.any(valid):
                    total_values = midpoint_total[valid] * base.X_DISPLAY_SCALE
                    absolute_per_day = (
                        (next_res[valid] - first_res[valid]) / delta_days
                        * base.X_DISPLAY_SCALE
                    )
                    relative_per_day = (
                        (next_res[valid] - first_res[valid])
                        / (0.5 * (first_res[valid] + next_res[valid]))
                        / delta_days
                    )
                    row: dict[str, object] = {
                        "plate": plate,
                        "well": f"R{match.group(1)}_{plan.run_id}",
                        "strategy": display_strategy[plate],
                        "first_tp": str(first_item.timepoint),
                        "next_tp": str(item.timepoint),
                        "first_day": first_day,
                        "next_day": next_day,
                        "mid_day": 0.5 * (first_day + next_day),
                        "delta_days": delta_days,
                        "n_pixels": int(np.count_nonzero(valid)),
                    }
                    for statistic in ("median", "mean"):
                        row[f"density_{statistic}"] = summarize(total_values, statistic)
                        row[f"relative_{statistic}"] = summarize(relative_per_day, statistic)
                        row[f"absolute_{statistic}"] = summarize(absolute_per_day, statistic)
                    row["density_area_mean"] = float(np.mean(total_values))
                    row["density_area_std"] = float(np.std(total_values, ddof=1))
                    row["net_relative_resistant_growth"] = float(
                        np.sum(next_res[valid] - first_res[valid])
                        / np.sum(0.5 * (first_res[valid] + next_res[valid]))
                        / delta_days
                    )
                    row["net_relative_resistant_growth_std"] = density_weighted_std(
                        relative_per_day, 0.5 * (first_res[valid] + next_res[valid]),
                    )
                    rows.append(row)
                previous = (item, resistant, total)
    if not rows:
        raise RuntimeError(f"No valid matched simulation intervals found under {simulation_root}")
    return pd.DataFrame(rows)


