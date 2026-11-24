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
This script plots selected example trajectories from the density-correlation
analysis so individual clone behaviors can be inspected in detail.

Detailed
--------
Inputs:
- Figure 2h local-density data.
- Plate-level treatment schedules used to annotate treatment timing.

Outputs:
- Example Figure 3 panels for chosen clones/replicates.

How it works:
- Loads the Figure 2h tables and extracts the requested example clone
  trajectories.
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared import experiment_filters


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset\Figures\Figure_2")
DATA_DIR = BASE_DIR / "Data"
FIGURE3_DATA_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
CSV_2H = FIGURE3_DATA_DIR / "Figure2h_ring_change_vs_ring_total_density_timeseries_epanechnikov_40pxboundary.csv"
DATASET_ROOT = BASE_DIR.parents[1]
FIGURE3_DIR = DATASET_ROOT / r"Figures\Figure_3"
SCHEDULE_BASENAME = "treatment_schedule.txt"
_RX_DIGITS = re.compile(r"(\d+)")

OUT_PDF = ft.paper_figure_data_dir(__file__) / "Figure_2_Density_Correlation_Examples.pdf"


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CHANGE_TYPE = "rel"
DENSITY_DISPLAY_SCALE = 1000.0
TOTAL_DENSITY_COL = "Mean_Total_Density_cells_per_um2"
LEGACY_TOTAL_DENSITY_COL = "Mean_Total_Density"
TOTAL_DENSITY_LABEL = "Boundary Density [cells (1000 µm$^2$)$^{-1}$]"
RELATIVE_GROWTH_LABEL = r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$"

# Example selection (order matters)
EXAMPLES_2H: Dict[str, Dict[str, object]] = {
    "NT": {
        "plate": "Plate_4",
        "sample": "B1_NT-B1",
        "clone_label": 2,
    },
    "AT50": {
        "plate": "Plate_4",
        "sample": "D5_AT50-D5",
        "clone_label": 2,
    },
    "CT": {
        "plate": "Plate_2",
        "sample": "A3_CT-A3",
        "clone_label": 2,
    },
}

# Axis limits (requested)
X_LIM = (0.0, 48.0)
Y_LIM_TOTAL_PHYSICAL = (0.0, 0.0021)
Y_LIM_TOTAL = tuple(v * DENSITY_DISPLAY_SCALE for v in Y_LIM_TOTAL_PHYSICAL)
Y_LIM_CHG = (-1.0, 7)

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
NORMALIZE_3DAY_INTERVAL_CHANGE = True

# -------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# -------------------------------------------------------------------
# Width and height refer to axes boxes: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 23.94
FIGURE_AXES_HEIGHT_MM = 11.77
FIGURE_MARGINS_MM = (8.82, 9.24, 8.50, 1.00)  # left, right, bottom, top
PANEL_GAP_MM = 2.59
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
def load_2h_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"2h CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "plate",
        "sample",
        "strategy",
        "clone_label",
        "sector",
        "change_type",
        "timepoint",
        "Mean_Resistant_Density_Change",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"2h CSV missing columns: {sorted(missing)}")
    source_density_col = TOTAL_DENSITY_COL if TOTAL_DENSITY_COL in df.columns else LEGACY_TOTAL_DENSITY_COL
    if source_density_col not in df.columns:
        raise ValueError(
            f"2h CSV must contain {TOTAL_DENSITY_COL!r} or {LEGACY_TOTAL_DENSITY_COL!r}: {csv_path}"
        )

    df = apply_experiment_filters(df, sample_col="sample")
    df = df[df["change_type"].astype(str) == str(CHANGE_TYPE)].copy()
    if df.empty:
        raise ValueError(
            f"No rows with change_type={CHANGE_TYPE!r} found in {csv_path}. "
            "Generate the Figure 2h table with CHANGE_KIND='rel' before running this plot."
        )

    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce").astype(int)
    df["day"] = df["timepoint"].apply(lambda tp: int(timepoint0_to_day(int(tp))))
    df["Mean_Total_Density"] = pd.to_numeric(df[source_density_col], errors="coerce")
    df["Mean_Total_Density"] = df["Mean_Total_Density"] * DENSITY_DISPLAY_SCALE
    df["Mean_Resistant_Density_Change"] = pd.to_numeric(
        df["Mean_Resistant_Density_Change"], errors="coerce"
    )

    if NORMALIZE_3DAY_INTERVAL_CHANGE:
        interval_days = df["timepoint"].apply(lambda tp: interval_days_from_timepoint0(int(tp)))
        is_three_day = interval_days == 3.0
        n_three = int(is_three_day.sum())
        if n_three > 0:
            df.loc[is_three_day, "Mean_Resistant_Density_Change"] = (
                df.loc[is_three_day, "Mean_Resistant_Density_Change"] / 1.5
            )
        print(f"[normalize] 3-day interval correction applied to {n_three} rows (divide by 1.5).")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Mean_Total_Density", "Mean_Resistant_Density_Change"]
    )
    return df


def extract_example_2h(df_2h: pd.DataFrame, plate: str, sample: str, clone_label: int) -> pd.DataFrame:
    sub = df_2h[
        (df_2h["plate"] == plate)
        & (df_2h["sample"] == sample)
        & (df_2h["clone_label"] == int(clone_label))
    ].copy()
    return sub.sort_values(["day", "sector"]).reset_index(drop=True)


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


def plot_one_panel(
    axL: plt.Axes, sub: pd.DataFrame, label: str, add_legend: bool = False, treatment_days: Optional[List[int]] = None
) -> None:
    ft.format_axes(axL, spines=("top", "right"))

    total_lo, total_hi = map(float, Y_LIM_TOTAL)
    chg_lo, chg_hi = map(float, Y_LIM_CHG)

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
    y_ticks_chg = ticks_divisible_by_5_or_10(Y_LIM_CHG, target_n=4)
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


def main() -> None:
    FIGURE3_DIR.mkdir(parents=True, exist_ok=True)

    df_2h = load_2h_data(CSV_2H)

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
    fig.text(0.005, 0.575, TOTAL_DENSITY_LABEL, va="center", rotation="vertical", color=COLOR_TOTAL, fontsize=7)
    fig.text(0.86, 0.575, RELATIVE_GROWTH_LABEL, va="center", rotation="vertical", color=COLOR_CHG, fontsize=7)

    order = ["NT", "AT50", "CT"]
    plate_schedule_cache: Dict[str, Dict[str, List[int]]] = {}
    for idx, (ax, name) in enumerate(zip(axes, order)):
        spec = EXAMPLES_2H[name]
        plate = str(spec["plate"])
        sample = str(spec["sample"])
        sub = extract_example_2h(
            df_2h,
            plate=plate,
            sample=sample,
            clone_label=int(spec["clone_label"]),
        )
        if plate not in plate_schedule_cache:
            plate_schedule_cache[plate] = load_plate_schedule_map(DATASET_ROOT, plate)
        treatment_days = get_treatment_days_for_sample(plate_schedule_cache[plate], sample)
        if name != "NT" and not treatment_days:
            print(f"[WARN] No treatment days found for {plate}/{sample}; no shading will be shown for this panel.")
        # Put legend only once (top panel)
        plot_one_panel(ax, sub, name, add_legend=(idx == 0), treatment_days=treatment_days)

    # X label only on bottom subplot
    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Days")

    # Margins tuned so both left/right y-labels are NOT cut
    # Save
    ft.savefig(fig, OUT_PDF)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"[done] Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
