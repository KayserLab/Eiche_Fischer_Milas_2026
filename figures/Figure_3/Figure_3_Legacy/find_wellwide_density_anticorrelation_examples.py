#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Find wells whose density change anticorrelates with total density over time.

For every consecutive timepoint pair, this script computes the pixelwise mean
total density across the two endpoints.  Relative change uses the linear
per-day correction ``(res_next - res_first) / res_first / delta_days`` and
absolute change is also expressed per day.  Each interval is then represented
by either the spatial median or spatial mean across the complete well.

The output consists of a ranked CSV and a multipage PDF containing density-
change plots for the best wells.  Both spatial statistics are always written
to the CSV so candidate selection can be checked for robustness.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from shared import experiment_filters
from analysis.figure_3 import density_common as base
from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as density_analysis


DEFAULT_RUN_DIR = density_analysis.DEFAULT_INVITRO_RUN
DEFAULT_SIMULATION_RUN_DIR = density_analysis.DEFAULT_SIM_ROOT
DEFAULT_OUTPUT_DIR = (
    Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
    / "wellwide_anticorrelation"
)
DEFAULT_GATE = (1.0e-8, float("inf"))
TIME_SERIES_EXAMPLES = (
    ('Plate_5', 'A5_AT100-A5'),
    ('Plate_2', 'B2_AT50-B2'),
    ('Plate_6', 'A3_ATAgent1-A3'),
    ('Plate_4', 'D2_AT50-D2'),
    ('Plate_4', 'D5_AT50-D5'),
    ('Plate_7', 'A4_ATAgent2-A4'),
    ('Plate_5', 'C6_NT-C6'),
    ('Plate_5', 'D2_CT-D2'),
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--associated-simulation-run-dir", type=Path,
        default=DEFAULT_SIMULATION_RUN_DIR,
        help="Simulation density-map run used for the paired comparison PDF.",
    )
    parser.add_argument(
        "--skip-combined", action="store_true",
        help="Do not create the paired experimental/simulation PDF.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--statistic", choices=("median", "mean"), default="median")
    parser.add_argument("--gate-lower", type=float, default=DEFAULT_GATE[0])
    parser.add_argument("--gate-upper", type=float, default=DEFAULT_GATE[1])
    parser.add_argument("--min-intervals", type=int, default=8)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument('--only-plate')
    parser.add_argument('--only-well')
    parser.add_argument('--matching-simulation-runs', action='store_true')
    parser.add_argument('--all-wells', action='store_true')
    return parser.parse_args()


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


def simulation_initial_condition(plate: str, well: str) -> str | None:
    match = re.match(r'^R(?P<replicate>\d+)_', well)
    allowed = SIMULATION_MATCHED_RANGES.get(plate)
    initial_wells = SIMULATION_INITIAL_WELLS.get(plate)
    if match is None or allowed is None or initial_wells is None:
        return None
    replicate = int(match.group('replicate'))
    if replicate not in allowed:
        return None
    strategy = plate.removeprefix('Plate_')
    experimental_well = initial_wells[replicate - allowed.start]
    return f'{experimental_well} ({strategy})'


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
        args = density_analysis.sweep.simulation_args(os.fspath(simulation_root), source)
        for plan in density_analysis.sim.discover_run_plans(args):
            match = re.fullmatch(r"run_(\d+)", str(plan.run_id))
            if match is None or int(match.group(1)) not in SIMULATION_MATCHED_RANGES[plate]:
                continue
            print(f"[simulation] {plate}/{plan.run_id}", flush=True)
            loaded = [
                density_analysis.sim.load_filtered_cells(item.path, args)
                for item in plan.files
            ]
            canvas = density_analysis.sweep.canvas_from_loaded(
                loaded, float(density_analysis.KERNEL_UM)
            )
            previous: tuple[object, np.ndarray, np.ndarray] | None = None
            for item, (positions_um, type_values) in zip(plan.files, loaded):
                resistant, total = density_analysis.simulation_density(
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


def correlation_table(intervals: pd.DataFrame, statistic: str) -> pd.DataFrame:
    x_col = f"density_{statistic}"
    rel_col = f"relative_{statistic}"
    abs_col = f"absolute_{statistic}"
    rows: list[dict[str, object]] = []
    for (plate, well, strategy), sub in intervals.groupby(["plate", "well", "strategy"]):
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, rel_col, abs_col])
        if sub.empty:
            continue
        rel_pearson = sub[x_col].corr(sub[rel_col])
        rel_spearman = sub[x_col].corr(sub[rel_col], method="spearman")
        abs_pearson = sub[x_col].corr(sub[abs_col])
        abs_spearman = sub[x_col].corr(sub[abs_col], method="spearman")
        density_span = float(sub[x_col].max() - sub[x_col].min())
        # Favor wells that are negative for both response definitions, cover a
        # useful density range, and contain enough independent intervals.
        strength = np.mean(
            np.maximum(0.0, -np.asarray([rel_pearson, rel_spearman, abs_pearson, abs_spearman]))
        )
        score = float(strength * math.log1p(len(sub)) * min(density_span, 2.0))
        rows.append(
            {
                "plate": plate,
                "well": well,
                "strategy": strategy,
                "n_intervals": len(sub),
                "first_day": float(sub["first_day"].min()),
                "last_day": float(sub["next_day"].max()),
                "density_span": density_span,
                "relative_pearson": rel_pearson,
                "relative_spearman": rel_spearman,
                "absolute_pearson": abs_pearson,
                "absolute_spearman": abs_spearman,
                "score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def plot_candidates(
    intervals: pd.DataFrame,
    ranking: pd.DataFrame,
    statistic: str,
    output_path: Path,
    top: int,
) -> None:
    x_col = f"density_{statistic}"
    columns = ((f"relative_{statistic}", "Linear relative change [day$^{-1}$]"),
               (f"absolute_{statistic}", "Absolute change [cells µm$^{-2}$ day$^{-1}$]"))
    with PdfPages(output_path) as pdf:
        for _, candidate in ranking.head(top).iterrows():
            sub = intervals[
                (intervals["plate"] == candidate["plate"])
                & (intervals["well"] == candidate["well"])
            ].sort_values("mid_day")
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=150)
            for ax, (y_col, ylabel) in zip(axes, columns):
                points = ax.scatter(
                    sub[x_col], sub[y_col], c=sub["mid_day"], cmap="viridis",
                    s=28, edgecolor="none", zorder=3,
                )
                ax.plot(sub[x_col], sub[y_col], color="0.75", linewidth=0.7, zorder=2)
                ax.axhline(0, color="0.65", linewidth=0.6)
                ax.set_xlabel(f"Endpoint-averaged total density\n[timepoint {statistic}; cells (1000 µm$^2$)$^{{-1}}$]")
                ax.set_ylabel(ylabel)
                ax.grid(color="0.92", linewidth=0.5)
                fig.colorbar(points, ax=ax, label="Interval midpoint [day]")
            fig.suptitle(
                f"{candidate['plate']} / {candidate['well']} ({candidate['strategy']})   "
                f"n={int(candidate['n_intervals'])}\n"
                f"relative r={candidate['relative_pearson']:.2f}, rho={candidate['relative_spearman']:.2f}; "
                f"absolute r={candidate['absolute_pearson']:.2f}, rho={candidate['absolute_spearman']:.2f}",
                fontsize=10,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    span = high - low
    pad = fraction * span if span > 0 else max(abs(low) * fraction, 1.0e-6)
    return low - pad, high + pad


def plot_one_time_series(sub, statistic, density_col, change_specs):
    strategy = str(sub['strategy'].iloc[0])
    density_ylim = padded_limits(sub[density_col].to_numpy(float))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=180, sharex=True)
    for ax, (change_col, change_label, change_color) in zip(axes, change_specs):
        ax_change = ax.twinx()
        time = sub['mid_day'].to_numpy(float)
        density = sub[density_col].to_numpy(float)
        change = sub[change_col].to_numpy(float)
        ax.plot(time, density, 'o-', color='black', markersize=3.2, linewidth=1.1, zorder=3)
        ax_change.plot(
            time, change, 'o-', color=change_color,
            markersize=3.2, linewidth=1.1, zorder=2,
        )
        ax.set_ylim(*density_ylim)
        ax_change.set_ylim(*padded_limits(change))
        ax.set_xlabel('Time [day; interval midpoint]')
        ax.set_ylabel('Total density [cells (1000 µm$^2$)$^{-1}$]', color='black')
        ax_change.set_ylabel(change_label, color=change_color)
        ax.tick_params(axis='y', colors='black')
        ax_change.tick_params(axis='y', colors=change_color)
        ax_change.spines['right'].set_color(change_color)
        ax.grid(axis='x', color='0.92', linewidth=0.5)
        pearson = sub[density_col].corr(sub[change_col])
        spearman = sub[density_col].corr(sub[change_col], method='spearman')
        name = 'relative' if change_col.startswith('relative') else 'absolute'
        ax.set_title(f'{name}: r={pearson:.2f}, rho={spearman:.2f}', fontsize=9)
    plate, well = sub[['plate', 'well']].iloc[0]
    initial_condition = simulation_initial_condition(plate, well)
    initial_label = (
        f'; experimental initial condition: {initial_condition}'
        if initial_condition is not None else ''
    )
    fig.suptitle(
        f'{plate} / {well} ({strategy}){initial_label}; whole-well spatial {statistic}; '
        f'n={len(sub)} intervals', fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_time_series_examples(
    intervals: pd.DataFrame, statistic: str, output_path: Path,
    examples=TIME_SERIES_EXAMPLES,
) -> None:
    density_col = f'density_{statistic}'
    change_specs = (
        (f'relative_{statistic}', 'Linear relative change [day$^{-1}$]', '#DAA520'),
        (f'absolute_{statistic}', 'Absolute change [cells µm$^{-2}$ day$^{-1}$]', '#B4464B'),
    )
    with PdfPages(output_path) as pdf:
        for plate, well in examples:
            sub = intervals[
                (intervals['plate'] == plate) & (intervals['well'] == well)
            ].sort_values('mid_day')
            if sub.empty:
                continue
            fig = plot_one_time_series(sub, statistic, density_col, change_specs)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def paired_experimental_subset(
    experimental: pd.DataFrame, simulation_plate: str, simulation_well: str,
) -> pd.DataFrame:
    match = re.match(r"^R(?P<replicate>\d+)_", simulation_well)
    allowed = SIMULATION_MATCHED_RANGES.get(simulation_plate)
    initial_wells = SIMULATION_INITIAL_WELLS.get(simulation_plate)
    if match is None or allowed is None or initial_wells is None:
        return experimental.iloc[0:0].copy()
    replicate = int(match.group("replicate"))
    if replicate not in allowed:
        return experimental.iloc[0:0].copy()
    initial_well = initial_wells[replicate - allowed.start]
    strategy = {
        "Plate_AT50": "AT50", "Plate_AT100": "AT100",
        "Plate_CT": "CT", "Plate_Dens": "AT100+",
    }[simulation_plate]
    candidates = experimental[
        (experimental["strategy"].astype(str) == strategy)
        & experimental["well"].astype(str).str.startswith(initial_well + "_")
    ].copy()
    if candidates.empty:
        return candidates
    selected_well = sorted(candidates["well"].astype(str).unique())[0]
    return candidates[candidates["well"].astype(str) == selected_well].copy()


def plot_paired_page(
    experimental: pd.DataFrame, simulation: pd.DataFrame, statistic: str,
) -> plt.Figure:
    density_col = f"density_{statistic}"
    change_specs = (
        (f"relative_{statistic}", "Linear relative change [day$^{-1}$]", "#DAA520"),
        (f"absolute_{statistic}", "Absolute change [cells µm$^{-2}$ day$^{-1}$]", "#B4464B"),
    )
    density_limits = padded_limits(
        np.r_[experimental[density_col].to_numpy(float), simulation[density_col].to_numpy(float)]
    )
    change_limits = {
        column: padded_limits(
            np.r_[experimental[column].to_numpy(float), simulation[column].to_numpy(float)]
        )
        for column, _label, _color in change_specs
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=180, sharex="col")
    for column_index, (title, data) in enumerate((
        ("Experiment", experimental), ("Associated simulation", simulation),
    )):
        data = data.sort_values("mid_day")
        for row_index, (change_col, change_label, change_color) in enumerate(change_specs):
            ax = axes[row_index, column_index]
            ax_change = ax.twinx()
            time = data["mid_day"].to_numpy(float)
            ax.plot(time, data[density_col], "o-", color="black", markersize=3.0, linewidth=1.0)
            ax_change.plot(
                time, data[change_col], "o-", color=change_color,
                markersize=3.0, linewidth=1.0,
            )
            ax.set_ylim(*density_limits)
            ax_change.set_ylim(*change_limits[change_col])
            ax.set_ylabel("Total density [cells (1000 µm$^2$)$^{-1}$]", color="black")
            ax_change.set_ylabel(change_label, color=change_color)
            ax.tick_params(axis="y", colors="black")
            ax_change.tick_params(axis="y", colors=change_color)
            ax_change.spines["right"].set_color(change_color)
            ax.grid(axis="x", color="0.92", linewidth=0.5)
            if row_index == 0:
                ax.set_title(title)
            if row_index == 1:
                ax.set_xlabel("Time [day; interval midpoint]")
    exp_plate, exp_well = experimental[["plate", "well"]].iloc[0]
    sim_plate, sim_well = simulation[["plate", "well"]].iloc[0]
    fig.suptitle(f"{exp_plate}/{exp_well}  |  {sim_plate}/{sim_well}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig



def plot_paired_experiment_simulation(
    experimental: pd.DataFrame, simulation: pd.DataFrame, statistic: str,
    output_path: Path,
) -> int:
    pages = 0
    with PdfPages(output_path) as pdf:
        for (plate, well), sim_sub in simulation.groupby(["plate", "well"], sort=True):
            exp_sub = paired_experimental_subset(experimental, str(plate), str(well))
            if exp_sub.empty:
                continue
            fig = plot_paired_page(exp_sub, sim_sub, statistic)
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
    if pages == 0:
        raise RuntimeError("No experimental wells could be paired to matched simulations")
    return pages


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if args.gate_lower < 0 or args.gate_upper <= args.gate_lower:
        raise ValueError("The resistant-density gate must satisfy 0 <= lower < upper")
    output_dir.mkdir(parents=True, exist_ok=True)

    intervals = scan_intervals(
        run_dir, (args.gate_lower, args.gate_upper),
        only_plate=args.only_plate, only_well=args.only_well,
        matching_simulation_runs=args.matching_simulation_runs,
    )
    median_ranking = correlation_table(intervals, "median")
    mean_ranking = correlation_table(intervals, "mean")
    ranking = median_ranking if args.statistic == "median" else mean_ranking
    eligible = ranking[ranking["n_intervals"] >= args.min_intervals].copy()
    if eligible.empty:
        raise RuntimeError(f"No wells contain at least {args.min_intervals} valid intervals")

    intervals.to_csv(output_dir / "wellwide_interval_summaries.csv", index=False)
    median_ranking.to_csv(output_dir / "wellwide_ranked_median.csv", index=False)
    mean_ranking.to_csv(output_dir / "wellwide_ranked_mean.csv", index=False)
    pdf_path = output_dir / f"top_wellwide_anticorrelation_{args.statistic}.pdf"
    plot_candidates(intervals, eligible, args.statistic, pdf_path, args.top)
    time_series_path = output_dir / f'selected_wellwide_time_series_{args.statistic}.pdf'
    if args.only_plate and args.only_well:
        examples = ((args.only_plate, args.only_well),)
    elif args.matching_simulation_runs or args.all_wells:
        examples = tuple(
            intervals[['plate', 'well']].drop_duplicates().sort_values(
                ['plate', 'well'], key=lambda col: col.map(str)
            ).itertuples(index=False, name=None)
        )
    else:
        examples = TIME_SERIES_EXAMPLES
    plot_time_series_examples(intervals, args.statistic, time_series_path, examples)

    if not args.skip_combined and not args.matching_simulation_runs:
        simulation_run_dir = args.associated_simulation_run_dir.expanduser()
        if not simulation_run_dir.is_dir():
            raise FileNotFoundError(
                f"Associated simulation run does not exist: {simulation_run_dir}"
            )
        simulation_intervals = scan_matched_simulation_intervals(
            simulation_run_dir, (args.gate_lower, args.gate_upper),
        )
        combined_path = (
            output_dir
            / f"combined_experimental_vs_associated_simulation_{args.statistic}.pdf"
        )
        combined_pages = plot_paired_experiment_simulation(
            intervals, simulation_intervals, args.statistic, combined_path
        )
        print(f"[saved] {combined_path} ({combined_pages} paired pages)")

    print(f"[saved] {output_dir / 'wellwide_interval_summaries.csv'}")
    print(f"[saved] {output_dir / 'wellwide_ranked_median.csv'}")
    print(f"[saved] {output_dir / 'wellwide_ranked_mean.csv'}")
    print(f"[saved] {pdf_path}")
    print(eligible.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
