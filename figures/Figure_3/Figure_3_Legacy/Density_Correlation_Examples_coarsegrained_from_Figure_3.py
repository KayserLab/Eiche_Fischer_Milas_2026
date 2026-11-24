#!/usr/bin/env python
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
# -*- coding: utf-8 -*-
"""
TL;DR
-----
This retained Figure 3 copy plots selected whole-well trajectories from the
same 160-um kernel, 16x coarse-grained density maps used for the master curve
and AT50/AT100 contour plots.

Detailed
--------
Inputs:
- The filtered experimental 160-um/16x density-map run.
- Plate-level treatment schedules used to annotate treatment timing.

Outputs:
- Example Figure 3 panels for chosen wells.

How it works:
- Calculates whole-well spatial medians for the requested wells.
- Applies the shared exclusion logic and converts timepoints to real day
  coordinates.
- Reads treatment schedules for the relevant plates and overlays them on the
  example trajectory panels.

Pipeline role:
- Figure 3 support plot for mechanistic example trajectories.
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import math
import re
import sys

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared import experiment_filters
from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as density_analysis


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset\Figures\Figure_2")
DATA_DIR = BASE_DIR / "Data"
FIGURE3_DATA_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
DENSITY_KERNEL_UM = 160
DENSITY_COARSENING_FACTOR = 16
INPUT_RUN = density_analysis.DEFAULT_INVITRO_RUN
SIMULATION_ROOT = density_analysis.DEFAULT_SIM_ROOT
SIMULATION_RESULTS = density_analysis.DEFAULT_SIM_ESCAPE
RESISTANT_ENDPOINT_THRESHOLD = 1.0e-8
DATASET_ROOT = BASE_DIR.parents[1]
FIGURE3_DIR = DATASET_ROOT / r"Figures\Figure_3"
SCHEDULE_BASENAME = "treatment_schedule.txt"
_RX_DIGITS = re.compile(r"(\d+)")

OUTPUT_DIR = ft.paper_figure_data_dir(__file__)
EXPERIMENT_OUTPUT = OUTPUT_DIR / "Density_Correlation_Examples_experiments_160um_16x_whole_well_median.pdf"
SIMULATION_OUTPUT = OUTPUT_DIR / "Density_Correlation_Examples_simulations_160um_16x_whole_well_median.pdf"


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CHANGE_TYPE = "rel"
DENSITY_DISPLAY_SCALE = 1000.0
TOTAL_DENSITY_COL = "Mean_Total_Density_cells_per_um2"
LEGACY_TOTAL_DENSITY_COL = "Mean_Total_Density"
TOTAL_DENSITY_LABEL = "Local Density [cells (1000 µm$^2$)$^{-1}$]"
RELATIVE_GROWTH_LABEL = r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$ [day$^{-1}$]"

# Example selection (order matters)
EXAMPLES_2H: Dict[str, Dict[str, object]] = {
    "NT": {
        "plate": "Plate_4",
        "sample": "B1_NT-B1",
    },
    "AT50": {
        "plate": "Plate_4",
        "sample": "D5_AT50-D5",
    },
    "CT": {
        "plate": "Plate_2",
        "sample": "A3_CT-A3",
    },
}
SIMULATION_EXAMPLES: Dict[str, Dict[str, object]] = {
    "NT": {"source": "NT", "run": "run_B1"},
    "AT50": {"source": "AT50", "run": "run_18"},
    "CT": {"source": "CT", "run": "run_31"},
}

# Axis limits (requested)
X_LIM = (0.0, 48.0)
Y_LIM_TOTAL_PHYSICAL = (0.0, 0.0021)
Y_LIM_TOTAL = tuple(v * DENSITY_DISPLAY_SCALE for v in Y_LIM_TOTAL_PHYSICAL)

# Ticks (requested)
X_TICKS = [0, 12, 24, 36, 48]
Y_TICKS_TOTAL = [0, 4, 8, 12]            # requested
Y_TICKS_CHG = [-1, 0, 1, 2, 3]

# Dot styling (plain dots, smaller)
S_TOTAL = 9
S_CHG = 8
ALPHA_PTS = 0.80
COLOR_TOTAL = "black"
COLOR_CHG = "#DAA520"
MARKER = "o"

RUNNING_AVG_WINDOW = 4  # days

# -------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# -------------------------------------------------------------------
# Width and height refer to axes boxes: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 16.758
FIGURE_AXES_HEIGHT_MM = 8.239
FIGURE_MARGINS_MM = (9.000, 11.000, 10.800, 10.857)  # left, right, bottom, top
PANEL_GAP_MM = 1.813
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
SHOW_FIGURE = True

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    },
)



# -------------------------------------------------------------------
# Exclusion helper
# -------------------------------------------------------------------
def true_timepoint_to_day(tp: int) -> int:
    """Map true timepoint index (starting at 1) to real day: 1,3,5,8,10,12,..."""
    if tp < 1:
        raise ValueError(f"true_timepoint_to_day expects tp >= 1, got {tp}")
    day = 1
    for k in range(2, tp + 1):
        step = 2 if ((k - 2) % 3) in (0, 1) else 3
        day += step
    return day


def timepoint0_to_day(tp0: int) -> int:
    """Map zero-based timepoint to real day. timepoint 0 -> day 1."""
    if tp0 < 0:
        raise ValueError(f"timepoint0_to_day expects tp0 >= 0, got {tp0}")
    return true_timepoint_to_day(tp0 + 1)


def interval_days_from_timepoint0(tp0: int) -> float:
    """
    Interval length represented by this timepoint's change value.
    Assumes change at tp0 compares tp0-1 -> tp0, so tp0=0 has no interval.
    """
    if tp0 <= 0:
        return np.nan
    return float(timepoint0_to_day(tp0) - timepoint0_to_day(tp0 - 1))


def get_filter_rule(sample: str) -> Tuple[str, Optional[float]]:
    """
    Uses experiment_filters.EXCLUDE_SAMPLES:
      - None    -> excluded
      - number  -> truncation cutoff in real days
    """
    rules: Dict[str, Any] = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    if sample not in rules:
        return "keep", None
    val = rules[sample]
    if val is None:
        return "exclude", None
    try:
        return "truncate", float(val)
    except Exception:
        return "keep", None


def apply_experiment_filters(df: pd.DataFrame, sample_col: str = "sample") -> pd.DataFrame:
    if sample_col not in df.columns:
        print(f"[exclude WARN] Missing '{sample_col}' column; skipping experiment_filters.")
        return df.copy()

    keep_mask = pd.Series(True, index=df.index)
    excluded_rows = 0
    truncated_rows = 0

    for sample, idx in df.groupby(sample_col).groups.items():
        status, cutoff = get_filter_rule(str(sample))
        if status == "exclude":
            keep_mask.loc[idx] = False
            excluded_rows += len(idx)
            continue
        if status != "truncate" or cutoff is None:
            continue

        if "day" in df.columns:
            day_vals = pd.to_numeric(df.loc[idx, "day"], errors="coerce")
        elif "timepoint" in df.columns:
            tp_vals = pd.to_numeric(df.loc[idx, "timepoint"], errors="coerce")
            day_vals = tp_vals.apply(
                lambda v: float(timepoint0_to_day(int(v))) if np.isfinite(v) and int(v) >= 0 else np.nan
            )
        else:
            continue

        bad = day_vals > float(cutoff)
        drop_idx = day_vals.index[bad.fillna(False)]
        keep_mask.loc[drop_idx] = False
        truncated_rows += int(bad.fillna(False).sum())

    out = df[keep_mask].copy()
    print(
        f"[exclude] After experiment_filters: kept {len(out)}/{len(df)} rows "
        f"(excluded={excluded_rows}, truncated={truncated_rows})"
    )
    return out


# -------------------------------------------------------------------
# Load 2h data
# -------------------------------------------------------------------
def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def density_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan")


def density_weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    values, weights = values[valid], weights[valid]
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def load_wellwide_data(
    run_dir: Path, spatial_summary=finite_median, growth_summary=finite_median,
    growth_area_weighted: bool = False, resistant_threshold: float | None = None,
    threshold_first_endpoint_only: bool = False,
) -> pd.DataFrame:
    """Calculate whole-well spatial summaries from the shared 160-um/16x maps."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"160-um input run not found: {run_dir}")
    rows: List[Dict[str, object]] = []
    for label, spec in EXAMPLES_2H.items():
        plate, sample = str(spec["plate"]), str(spec["sample"])
        status, cutoff = get_filter_rule(sample)
        if status == "exclude":
            print(f"[exclude] {plate}/{sample}")
            continue
        well_dir = run_dir / plate / sample
        maps_dir = well_dir / "density_maps"
        changes_dir = well_dir / "density_changes"
        for rel_path in sorted(changes_dir.glob("*_res_density_epanechnikov_rel_diff.tif")):
            match = density_analysis.base.REL_CHANGE_RE.match(rel_path.name)
            if match is None:
                continue
            first_tp, next_tp = match.group("first_tp"), match.group("next_tp")
            first_day = density_analysis.base.timepoint_to_day(first_tp)
            next_day = density_analysis.base.timepoint_to_day(next_tp)
            delta_days = float(next_day - first_day)
            if delta_days <= 0 or (status == "truncate" and cutoff is not None and next_day > cutoff):
                continue
            paths = (
                density_analysis.base.density_map_path(maps_dir, sample, first_tp),
                density_analysis.base.density_map_path(maps_dir, sample, next_tp),
                density_analysis.base.resistant_density_map_path(maps_dir, sample, first_tp),
                density_analysis.base.resistant_density_map_path(maps_dir, sample, next_tp),
            )
            if not all(path.is_file() for path in paths):
                continue
            total_t0, total_t1, res_t0, res_t1 = (
                tiff.imread(path).astype(np.float64, copy=False).ravel() for path in paths
            )
            if len({array.shape for array in (total_t0, total_t1, res_t0, res_t1)}) != 1:
                raise ValueError(f"Density-map shape mismatch for {plate}/{sample} {first_tp}->{next_tp}")
            average_total = 0.5 * (total_t0 + total_t1)
            valid = (
                np.isfinite(average_total)
                & np.isfinite(res_t0)
                & np.isfinite(res_t1)
            )
            if resistant_threshold is None:
                valid &= (res_t0 >= RESISTANT_ENDPOINT_THRESHOLD)
                valid &= (res_t1 >= RESISTANT_ENDPOINT_THRESHOLD)
            else:
                valid &= res_t0 > resistant_threshold
                if not threshold_first_endpoint_only:
                    valid &= res_t1 > resistant_threshold
            if not np.any(valid):
                continue
            relative_growth = (
                (res_t1[valid] - res_t0[valid])
                / res_t0[valid]
                / delta_days
            )
            absolute_growth = (
                (res_t1[valid] - res_t0[valid])
                / delta_days
                * DENSITY_DISPLAY_SCALE
            )
            growth_value = (
                density_weighted_mean(relative_growth, res_t0[valid])
                if growth_area_weighted else growth_summary(relative_growth)
            )
            total_values = average_total[valid] * DENSITY_DISPLAY_SCALE
            rows.append({
                "plate": plate,
                "sample": sample,
                "strategy": label,
                "day": 0.5 * (first_day + next_day),
                "Mean_Total_Density": spatial_summary(total_values),
                "Std_Total_Density": float(np.std(total_values, ddof=1)),
                "Mean_Resistant_Density_Change": growth_value,
                "Std_Resistant_Density_Change": (
                    density_weighted_std(relative_growth, res_t0[valid])
                    if growth_area_weighted else float(np.std(relative_growth, ddof=1))
                ),
                "Mean_Resistant_Density_Absolute_Change": spatial_summary(absolute_growth),
                "Std_Resistant_Density_Absolute_Change": float(
                    np.std(absolute_growth, ddof=1)
                ),
                "n_pixels": int(np.count_nonzero(valid)),
            })
    if not rows:
        raise RuntimeError(f"No valid whole-well intervals found under {run_dir}")
    return pd.DataFrame(rows)


def load_simulation_wellwide_data(
    input_root: Path, spatial_summary=finite_median, growth_summary=finite_median,
    growth_area_weighted: bool = False, resistant_threshold: float | None = None,
    threshold_first_endpoint_only: bool = False,
) -> pd.DataFrame:
    """Calculate whole-domain spatial summaries from representative simulation runs."""
    rows: List[Dict[str, object]] = []
    for label, spec in SIMULATION_EXAMPLES.items():
        source, run_name = str(spec["source"]), str(spec["run"])
        sim_args = density_analysis.sweep.simulation_args(str(input_root), source, run_name)
        plans = density_analysis.sim.discover_run_plans(sim_args)
        if len(plans) != 1:
            raise RuntimeError(
                f"Expected one simulation plan for {source}/{run_name}, found {len(plans)}"
            )
        plan = plans[0]
        loaded = [
            density_analysis.sim.load_filtered_cells(item.path, sim_args)
            for item in plan.files
            if float(item.day) <= X_LIM[1]
        ]
        files = [item for item in plan.files if float(item.day) <= X_LIM[1]]
        canvas = density_analysis.sweep.canvas_from_loaded(
            loaded, float(DENSITY_KERNEL_UM)
        )
        previous = None
        for sim_file, (positions_um, type_values) in zip(files, loaded):
            current_day = float(sim_file.day)
            current_resistant, current_total = density_analysis.simulation_density(
                positions_um, type_values, canvas
            )
            if previous is not None:
                previous_resistant, previous_total, previous_day = previous
                delta_days = current_day - previous_day
                average_total = 0.5 * (previous_total + current_total)
                valid = (
                    np.isfinite(average_total)
                    & np.isfinite(previous_resistant)
                    & np.isfinite(current_resistant)
                )
                if resistant_threshold is None:
                    valid &= (previous_resistant >= RESISTANT_ENDPOINT_THRESHOLD)
                    valid &= (current_resistant >= RESISTANT_ENDPOINT_THRESHOLD)
                else:
                    valid &= previous_resistant > resistant_threshold
                    if not threshold_first_endpoint_only:
                        valid &= current_resistant > resistant_threshold
                if delta_days > 0 and np.any(valid):
                    relative_growth = (
                        (current_resistant[valid] - previous_resistant[valid])
                        / previous_resistant[valid]
                        / delta_days
                    )
                    absolute_growth = (
                        (current_resistant[valid] - previous_resistant[valid])
                        / delta_days
                        * DENSITY_DISPLAY_SCALE
                    )
                    growth_value = (
                        density_weighted_mean(relative_growth, previous_resistant[valid])
                        if growth_area_weighted else growth_summary(relative_growth)
                    )
                    total_values = average_total[valid] * DENSITY_DISPLAY_SCALE
                    rows.append({
                        "plate": source,
                        "sample": run_name,
                        "strategy": label,
                        "day": 0.5 * (previous_day + current_day),
                        "Mean_Total_Density": spatial_summary(total_values),
                        "Std_Total_Density": float(np.std(total_values, ddof=1)),
                        "Mean_Resistant_Density_Change": growth_value,
                        "Std_Resistant_Density_Change": (
                            density_weighted_std(relative_growth, previous_resistant[valid])
                            if growth_area_weighted else float(np.std(relative_growth, ddof=1))
                        ),
                        "Mean_Resistant_Density_Absolute_Change": spatial_summary(absolute_growth),
                        "Std_Resistant_Density_Absolute_Change": float(
                            np.std(absolute_growth, ddof=1)
                        ),
                        "n_pixels": int(np.count_nonzero(valid)),
                    })
            previous = current_resistant, current_total, current_day
    if not rows:
        raise RuntimeError(f"No valid simulation intervals found under {input_root}")
    return pd.DataFrame(rows)


def extract_example_2h(df_2h: pd.DataFrame, plate: str, sample: str) -> pd.DataFrame:
    sub = df_2h[(df_2h["plate"] == plate) & (df_2h["sample"] == sample)].copy()
    return sub.sort_values("day").reset_index(drop=True)


def ticks_divisible_by_5_or_10(lim: Tuple[float, float], target_n: int = 5) -> np.ndarray:
    lo, hi = float(lim[0]), float(lim[1])
    if hi < lo:
        lo, hi = hi, lo

    candidates: List[float] = []
    for k in range(-6, 5):
        scale = 10 ** k
        candidates.extend([1.0 * scale, 2.0 * scale, 5.0 * scale, 10.0 * scale])

    def n_ticks(step: float) -> int:
        start = math.ceil(lo / step) * step
        end = math.floor(hi / step) * step
        if end < start:
            return 0
        return int(round((end - start) / step)) + 1

    best_step = min(candidates, key=lambda s: (abs(n_ticks(s) - target_n), s))
    start = math.ceil(lo / best_step) * best_step
    end = math.floor(hi / best_step) * best_step
    if end < start:
        return np.array([], dtype=float)

    ticks = np.arange(start, end + 0.5 * best_step, best_step)
    if np.allclose(ticks, np.round(ticks)):
        return np.round(ticks).astype(int)
    return ticks


def format_tick_label(value: float) -> str:
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    abs_value = abs(float(value))
    if abs_value < 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if abs_value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1f}".rstrip("0").rstrip(".")


def parse_treated_days(tokens: List[str]) -> List[int]:
    """
    Parse treated days from one schedule line.
    Tokens containing 's' are stop tokens and are ignored.
    """
    out: List[int] = []
    for t in tokens:
        if not t or "s" in t:
            continue
        m = _RX_DIGITS.search(t)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def load_plate_schedule_map(dataset_root: Path, plate: str) -> Dict[str, List[int]]:
    """
    Load one plate schedule file and return sample -> treated days list.
    """
    sched_path = dataset_root / plate / SCHEDULE_BASENAME
    if not sched_path.is_file():
        print(f"[WARN] Missing schedule file: {sched_path}")
        return {}

    sched_map: Dict[str, List[int]] = {}
    with open(sched_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            rep = parts[0]
            sched_map[rep] = parse_treated_days(parts[1:])
    return sched_map


def get_treatment_days_for_sample(plate_map: Dict[str, List[int]], sample: str) -> List[int]:
    """
    Resolve treatment days for a sample name.
    Supports exact sample keys and keys with experiment prefixes.
    """
    if sample in plate_map:
        return plate_map[sample]

    # Fallback for keys like "G418_a2a4_B3_AT100-B3" -> sample "B3_AT100-B3"
    for key, days in plate_map.items():
        if str(key).endswith(sample):
            return days
    return []


def appropriate_growth_limits(data: pd.DataFrame) -> Tuple[float, float]:
    values = data["Mean_Resistant_Density_Change"].to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -0.5, 0.5
    lower, upper = np.quantile(values, [0.01, 0.99])
    lower, upper = min(0.0, float(lower)), max(0.0, float(upper))
    span = max(upper - lower, 0.1)
    padding = 0.12 * span
    raw_step = (span + 2.0 * padding) / 4.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = min((1.0, 2.0, 5.0, 10.0), key=lambda value: abs(value * magnitude - raw_step)) * magnitude
    return (
        float(math.floor((lower - padding) / step) * step),
        float(math.ceil((upper + padding) / step) * step),
    )


def plot_one_panel(
    axL: plt.Axes, sub: pd.DataFrame, label: str, growth_limits: Tuple[float, float],
    add_legend: bool = False, treatment_days: Optional[List[int]] = None,
) -> None:
    ft.format_axes(axL, spines=("top", "right"))

    total_lo, total_hi = map(float, Y_LIM_TOTAL)
    chg_lo, chg_hi = map(float, growth_limits)

    def chg_to_total_axis(y):
        return total_lo + (np.asarray(y) - chg_lo) * (total_hi - total_lo) / (chg_hi - chg_lo)

    def total_axis_to_chg(y):
        return chg_lo + (np.asarray(y) - total_lo) * (chg_hi - chg_lo) / (total_hi - total_lo)

    if sub.empty:
        axL.text(0.5, 0.5, f"{label}\n(no data)", transform=axL.transAxes,
                 ha="center", va="center")
        axL.set_axis_off()
        return

    if treatment_days:
        shade_max_day = float(sub["day"].max()) if "day" in sub.columns and not sub.empty else X_LIM[1]
        for d in treatment_days:
            # Treatment ON from day d to d+1.
            x0 = max(float(d), X_LIM[0])
            x1 = min(float(d + 1), X_LIM[1], shade_max_day)
            if x1 > x0:
                axL.axvspan(x0, x1, color="#bfbfbf", alpha=1.0, zorder=0)

    grouped = sub.groupby("day", as_index=False).agg(
        Mean_Total_Density=("Mean_Total_Density", "mean"),
        Mean_Resistant_Density_Change=("Mean_Resistant_Density_Change", "mean"),
    )

    # running averages (per-day means -> rolling mean)
    t = grouped["day"].to_numpy()

    tot = grouped["Mean_Total_Density"].to_numpy()
    chg = grouped["Mean_Resistant_Density_Change"].to_numpy()

    tot_ra = pd.Series(tot).rolling(RUNNING_AVG_WINDOW, center=True, min_periods=1).mean().to_numpy()
    chg_ra = pd.Series(chg).rolling(RUNNING_AVG_WINDOW, center=True, min_periods=1).mean().to_numpy()
    chg_plot = chg_to_total_axis(chg)
    chg_ra_plot = chg_to_total_axis(chg_ra)

    axL.plot(t, tot_ra, linestyle="-", linewidth=1.0, alpha=0.9, color=COLOR_TOTAL)
    axL.plot(t, chg_ra_plot, linestyle="-", linewidth=1.0, alpha=0.9, color=COLOR_CHG)

    # Plain dot markers (no ring): edgecolors="none"
    axL.scatter(
        grouped["day"],
        grouped["Mean_Total_Density"],
        s=S_TOTAL,
        alpha=ALPHA_PTS,
        color=COLOR_TOTAL,
        marker=MARKER,
        edgecolors="none",
    )
    axL.scatter(
        grouped["day"],
        chg_plot,
        s=S_CHG,
        alpha=ALPHA_PTS,
        color=COLOR_CHG,
        marker=MARKER,
        edgecolors="none",
    )

    # Limits + ticks
    axL.set_xlim(*X_LIM)
    axL.set_ylim(*Y_LIM_TOTAL)

    x_ticks = ticks_divisible_by_5_or_10(X_LIM, target_n=6)
    axL.set_xticks(x_ticks)

    y_ticks_total = ticks_divisible_by_5_or_10(Y_LIM_TOTAL, target_n=4)
    axL.set_yticks(y_ticks_total)
    axL.set_yticklabels([format_tick_label(float(y)) for y in y_ticks_total])

    axR = axL.secondary_yaxis("right", functions=(total_axis_to_chg, chg_to_total_axis))
    y_ticks_chg = ticks_divisible_by_5_or_10(growth_limits, target_n=4)
    axR.set_yticks(y_ticks_chg)
    axR.set_yticklabels([format_tick_label(float(y)) for y in y_ticks_chg])
    axR.spines["right"].set_color(COLOR_CHG)

    axL.tick_params(axis="y", colors=COLOR_TOTAL)
    axR.tick_params(axis="y", colors=COLOR_CHG, direction="out", length=2.5, width=0.8, pad=1)

    # Compact padding
    axL.tick_params(axis="x", pad=1)
    axL.tick_params(axis="y", pad=1)

    # Strategy label in-panel (top-right)
    axL.text(
        0.98, 0.92, label,
        transform=axL.transAxes,
        ha="right", va="top",
        zorder=10
    )

    axL.grid(False)


def simulation_treatment_days(source: str, run_name: str, table: pd.DataFrame) -> List[int]:
    if source == "NT":
        return []
    match = re.search(r"(\d+)$", run_name)
    if match is None:
        return []
    selected = table[
        (table["protocol"].astype(str) == source)
        & (pd.to_numeric(table["run"], errors="coerce") == int(match.group(1)))
        & (pd.to_numeric(table["treatment_on"], errors="coerce") > 0)
    ]
    return sorted(pd.to_numeric(selected["day"], errors="coerce").dropna().astype(int).unique())


def write_examples(
    data: pd.DataFrame, specs: Dict[str, Dict[str, object]], output_path: Path,
    simulation: bool, growth_limits: Tuple[float, float],
) -> Tuple[float, float]:
    fig, axes = ft.new_subplots(
        3,
        1,
        ax_width_mm=FIGURE_AXES_WIDTH_MM,
        ax_height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        hspace_mm=PANEL_GAP_MM,
        dpi=DPI,
        sharex=True,
        apply_style=False,
    )

    # one shared y-label per side
    fig.text(0.035, 0.500, TOTAL_DENSITY_LABEL, va="center", rotation="vertical", color=COLOR_TOTAL, fontsize=7)
    fig.text(0.900, 0.500, RELATIVE_GROWTH_LABEL, va="center", rotation="vertical", color=COLOR_CHG, fontsize=7)

    order = ["NT", "AT50", "CT"]
    plate_schedule_cache: Dict[str, Dict[str, List[int]]] = {}
    simulation_table = pd.read_csv(SIMULATION_RESULTS) if simulation else None
    for idx, (ax, name) in enumerate(zip(axes, order)):
        spec = specs[name]
        plate = str(spec["source"] if simulation else spec["plate"])
        sample = str(spec["run"] if simulation else spec["sample"])
        sub = extract_example_2h(
            data,
            plate=plate,
            sample=sample,
        )
        if simulation:
            treatment_days = simulation_treatment_days(
                plate, sample, simulation_table
            )
        else:
            if plate not in plate_schedule_cache:
                plate_schedule_cache[plate] = load_plate_schedule_map(DATASET_ROOT, plate)
            treatment_days = get_treatment_days_for_sample(plate_schedule_cache[plate], sample)
        if name != "NT" and not treatment_days:
            print(f"[WARN] No treatment days found for {plate}/{sample}; no shading will be shown for this panel.")
        # Put legend only once (top panel)
        plot_one_panel(
            ax, sub, name, growth_limits, add_legend=(idx == 0),
            treatment_days=treatment_days,
        )

    # X label only on bottom subplot
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Days")

    # Margins tuned so both left/right y-labels are NOT cut
    # Save
    ft.savefig(fig, output_path)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"[done] Saved: {output_path}; resistant-growth ylim={growth_limits}")
    return growth_limits



def main() -> None:
    FIGURE3_DIR.mkdir(parents=True, exist_ok=True)
    experimental = load_wellwide_data(INPUT_RUN)
    simulation = load_simulation_wellwide_data(SIMULATION_ROOT)
    shared_growth_limits = appropriate_growth_limits(
        pd.concat([experimental, simulation], ignore_index=True)
    )
    write_examples(
        experimental, EXAMPLES_2H, EXPERIMENT_OUTPUT,
        simulation=False, growth_limits=shared_growth_limits,
    )
    write_examples(
        simulation, SIMULATION_EXAMPLES, SIMULATION_OUTPUT,
        simulation=True, growth_limits=shared_growth_limits,
    )


if __name__ == "__main__":
    main()
