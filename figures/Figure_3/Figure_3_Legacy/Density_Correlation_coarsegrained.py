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
This script plots the main relationship between local clone-adjacent density
and resistant growth behavior using the Figure 2h analysis outputs.

Detailed
--------
Inputs:
- Figure 2h local-density CSVs.
- Replicate exclusion/truncation rules and the shared strategy color mapping.

Outputs:
- The Figure 3 density-correlation panels, including highlighted behavioral
  trajectories.

How it works:
- Converts Figure 2h timepoint data into day-aligned trajectories.
- Filters replicates according to the shared manual exclusion/truncation table.
- Detects characteristic behaviors such as fluctuating trajectories or
  irreversible escape candidates.
- Visualizes density-versus-growth relationships using per-strategy colors and
  annotated trajectory panels.

Pipeline role:
- Main Figure 3 plotting script for the density-correlation story.
"""


from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared import experiment_filters
from shared.strategy_colors import get_color


def strategy_color(strategy: str, default: str = "#000000") -> str:
    return get_color(strategy, default)


# =============================================================================
# PATHS & CONFIG
# =============================================================================
BASE_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset\Figures\Figure_2")
DATA_DIR = BASE_DIR / "Data"
FIGURE3_DATA_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
IN_CSV = FIGURE3_DATA_DIR / "Figure2h_ring_change_vs_ring_total_density_timeseries_epanechnikov_40pxboundary.csv"
DATASET_ROOT = BASE_DIR.parents[1]

RES_DENSITY_WINDOW = (2.0e-5, 0.001)


def density_window_token(res_density_window: tuple[float, float]) -> str:
    def fmt(value: float) -> str:
        return f"{value:.6g}".replace(".", "p")

    lower, upper = res_density_window
    return f"_res_window_{fmt(lower)}_to_{fmt(upper)}"


OUT_DIR = ft.paper_figure_data_dir(__file__)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SUFFIX = density_window_token(RES_DENSITY_WINDOW)
OUT_BASENAME = OUT_DIR / f"Density_Correlation{OUTPUT_SUFFIX}"
OUT_BASENAME_4P = OUT_DIR / f"Density_Correlation_4panel_trajectories{OUTPUT_SUFFIX}"
OUT_BASENAME_ALL_BEHAVIOR = (
    OUT_DIR / f"Density_Correlation_all_irreversible_escape_trajectories{OUTPUT_SUFFIX}"
)
OUT_BASENAME_ALL_BEHAVIOR_PANELS = (
    OUT_DIR / f"Density_Correlation_irreversible_escape_panels{OUTPUT_SUFFIX}"
)

CHANGE_TYPE = "rel"     # "abs" for absolute change in resistant density; "rel" for relative growth (fold-change)
X_DENSITY_DISPLAY_SCALE = 1000.0
TOTAL_DENSITY_COL = "Mean_Total_Density_cells_per_um2"
LEGACY_TOTAL_DENSITY_COL = "Mean_Total_Density"
RESISTANT_DENSITY_COLS = (
    "Mean_Resistant_Density_cells_per_um2",
    "Mean_Resistant_Density",
    "Mean_Initial_Resistant_Density_cells_per_um2",
    "Mean_Initial_Resistant_Density",
)
X_AXIS_LABEL = "Boundary Density [cells (1000 µm$^2$)$^{-1}$]"

CHANGE_TYPE_CONFIG = {
    "abs": {
        "y_axis_label": r"Resistant density change (cells µm$^{-2}$)",
        "y_lim": (-0.00025, 0.0008),
        "fluct_rise_min": 0.00031578947368421053,
        "fluct_fall_min": 0.0002105263157894737,
        "escape_y_min": 0.0010526315789473684,
        "return_y_max": 0.0004210526315789474,
    },
    "rel": {
        "y_axis_label": r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$",
        "y_lim": (-2.0, 10.0),
        "fluct_rise_min": 4.0,
        "fluct_fall_min": 2.0,
        "escape_y_min": 3.0,
        "return_y_max": 3.5,
    },
}
if CHANGE_TYPE not in CHANGE_TYPE_CONFIG:
    raise ValueError(f"Unsupported CHANGE_TYPE={CHANGE_TYPE!r}; choose one of {sorted(CHANGE_TYPE_CONFIG)}.")

_CHANGE_CFG = CHANGE_TYPE_CONFIG[CHANGE_TYPE]
Y_AXIS_LABEL = _CHANGE_CFG["y_axis_label"]

# Strategies to include (raw names or aliases are allowed; they get normalized below).
INCLUDE_STRATEGIES = ("NT", "CT", "AT100", "AT50")

X_LIM_PHYSICAL = (0.0, 0.003)
X_LIM = tuple(v * X_DENSITY_DISPLAY_SCALE for v in X_LIM_PHYSICAL)
Y_LIM = _CHANGE_CFG["y_lim"]

GRID_N = 140
CONTOUR_QUANTILES = (0.80, 0.90, 0.95)
RESTRICT_TO_PLOTTING_WINDOW = False
NORMALIZE_3DAY_INTERVAL_CHANGE = True

FILL_ALPHA_MIN = 0.10
FILL_ALPHA_MAX = 0.80

OUTLINE_LW = 0.9
OUTLINE_ALPHA = 0.95
_CONTOUR_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

SHOW_LEGEND = True
LEGEND_LOC = "upper right"

# Automatic trajectory overlay: wells that show both
# 1) fluctuation (drop/rise followed by rebound/fall), and
# 2) irreversible escape (enter low-density/high-growth and never recover to high-density/low-growth).
SHOW_BEHAVIOR_TRAJECTORIES_ON_MAIN = False
SHOW_BEHAVIOR_TRAJECTORIES_IN_PANELS = True
TRAJECTORY_COLOR = "#000000"
TRAJECTORY_LW = 0.9
TRAJECTORY_ALPHA_START = 0.90
TRAJECTORY_ALPHA_END = 0.25
ANNOTATE_BEHAVIOR_TRAJECTORIES = False
BEHAVIOR_PANEL_NCOLS = 4
PANEL_DAY_MARKERS: Dict[Tuple[str, str, Optional[int]], List[int]] = {
    ("Plate_2", "B4_AT50-B4", 3): [8, 15, 19],
}
PANEL_DAY_MARKER_SIZE = 18
PANEL_DAY_MARKER_FACECOLOR = "#ffffff"
PANEL_DAY_MARKER_TEXT_OFFSET = (2, 2)

MIN_TRAJECTORY_DAYS = 6
FLUCT_DROP_MIN = 0.631578947368421
FLUCT_RISE_MIN = _CHANGE_CFG["fluct_rise_min"]
FLUCT_REBOUND_MIN = 0.42105263157894735
FLUCT_FALL_MIN = _CHANGE_CFG["fluct_fall_min"]

# Very-soft behavior thresholds in displayed x units (cells/1000 um2).
ESCAPE_X_MAX = 1.5
ESCAPE_Y_MIN = _CHANGE_CFG["escape_y_min"]
RETURN_X_MIN = 1.3
RETURN_Y_MAX = _CHANGE_CFG["return_y_max"]


# =============================================================================
# EDITABLE FIGURE PARAMETERS
# =============================================================================
# Width and height refer to axes boxes: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 24.0
FIGURE_AXES_HEIGHT_MM = 36.0
FIGURE_MARGINS_MM = (10.0, 8.0, 13.0, 5.0)  # left, right, bottom, top
PANEL_AXES_WIDTH_MM = 24.0
PANEL_AXES_HEIGHT_MM = 36.0
PANEL_MARGINS_MM = (10.0, 8.0, 13.0, 5.0)  # left, right, bottom, top
PANEL_GAP_MM = 6.0
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
SHOW_FIGURE = False

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


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    return ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

# =============================================================================
# STRATEGY NORMALIZATION
# =============================================================================
STRATEGY_ALIASES = {
    "ATAgent1": "KPP-Agent",
    "ATAgent":  "KPP-Agent",
    "ATAgent2": "ATagent",
    "Dens":     r"AT$^{+}$",
}

def normalize_strategy(s: str) -> str:
    return STRATEGY_ALIASES.get(str(s), str(s))


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


def apply_experiment_filters(df: pd.DataFrame) -> pd.DataFrame:
    sample_col = "sample" if "sample" in df.columns else ("replicate" if "replicate" in df.columns else None)
    if sample_col is None:
        print("[exclude WARN] No sample/replicate column found; skipping experiment_filters.")
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


# =============================================================================
# HELPERS
# =============================================================================
def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_n: int,
):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    xx = np.linspace(xlim[0], xlim[1], grid_n)
    yy = np.linspace(ylim[0], ylim[1], grid_n)
    Xg, Yg = np.meshgrid(xx, yy)

    pts = np.vstack([x, y])
    positions = np.vstack([Xg.ravel(), Yg.ravel()])

    try:
        kde = gaussian_kde(pts)
        Z = kde(positions).reshape(Xg.shape)
        return Xg, Yg, Z
    except LinAlgError:
        jitter = 1e-6
        pts_j = np.vstack([
            x + np.random.normal(0, jitter, size=x.size),
            y + np.random.normal(0, jitter, size=y.size),
        ])
        kde = gaussian_kde(pts_j)
        Z = kde(positions).reshape(Xg.shape)
        return Xg, Yg, Z


def ticks_divisible_by_5_or_10(lim: Tuple[float, float], target_n: int = 6) -> np.ndarray:
    lo, hi = float(lim[0]), float(lim[1])
    if hi < lo:
        lo, hi = hi, lo

    # Candidate steps remain simple round values across the current data scale.
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


def build_contour_payload(x: np.ndarray, y: np.ndarray, base_color: str) -> Optional[Dict[str, Any]]:
    if x.size < 10:
        return None

    Xg, Yg, Z = compute_density_grid(x, y, X_LIM, Y_LIM, GRID_N)

    vals = Z[np.isfinite(Z) & (Z > 0)]
    if vals.size == 0:
        return None

    core_levels = np.quantile(vals, CONTOUR_QUANTILES)
    levels = np.concatenate((core_levels, [vals.max() * 1.001]))

    n_regions = len(levels) - 1
    base_rgba = np.array(mcolors.to_rgba(base_color))

    colors = []
    for i in range(n_regions):
        a = FILL_ALPHA_MIN + (FILL_ALPHA_MAX - FILL_ALPHA_MIN) * (i / max(1, n_regions - 1))
        rgba = base_rgba.copy()
        rgba[-1] = a
        colors.append(tuple(rgba))

    return {
        "Xg": Xg,
        "Yg": Yg,
        "Z": Z,
        "levels": levels,
        "core_levels": core_levels,
        "colors": colors,
        "base_color": base_color,
    }


def filled_contours_on_ax(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    base_color: str,
    zorder: int,
    cache_key: Optional[str] = None,
) -> bool:
    if cache_key is None:
        payload = build_contour_payload(x, y, base_color)
    else:
        if cache_key not in _CONTOUR_CACHE:
            _CONTOUR_CACHE[cache_key] = build_contour_payload(x, y, base_color)
        payload = _CONTOUR_CACHE[cache_key]

    if payload is None:
        return False

    ax.contourf(
        payload["Xg"],
        payload["Yg"],
        payload["Z"],
        levels=payload["levels"],
        colors=payload["colors"],
        zorder=zorder,
    )
    ax.contour(
        payload["Xg"],
        payload["Yg"],
        payload["Z"],
        levels=payload["core_levels"],
        colors=[payload["base_color"]],
        linewidths=OUTLINE_LW,
        alpha=OUTLINE_ALPHA,
        zorder=zorder + 1,
    )

    return True


def build_daily_trajectory_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return per-day averaged trajectory points for each strategy/plate/sample/clone."""
    if "timepoint" not in df_in.columns:
        return pd.DataFrame()

    work = df_in.copy()
    tp = pd.to_numeric(work["timepoint"], errors="coerce")
    work = work[tp.notna()].copy()
    work["timepoint"] = tp[tp.notna()].astype(int)
    work["day"] = work["timepoint"].map(timepoint0_to_day)

    group_cols = ["strategy", "plate", "sample"]
    if "clone_label" in work.columns:
        clone_vals = pd.to_numeric(work["clone_label"], errors="coerce")
        work = work[clone_vals.notna()].copy()
        work["clone_label"] = clone_vals[clone_vals.notna()].astype(int)
        group_cols.append("clone_label")
    group_cols.append("day")

    out = work.groupby(group_cols, as_index=False).agg(
        x=("Mean_Total_Density", "mean"),
        y=("Mean_Resistant_Density_Change", "mean"),
    )
    return out.sort_values(group_cols).reset_index(drop=True)


def get_highlight_points(traj_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return per-day averaged points for one configured trajectory from the trajectory table."""
    if traj_df.empty:
        return pd.DataFrame()

    sub = traj_df[
        (traj_df["strategy"] == str(cfg["strategy"]))
        & (traj_df["plate"] == str(cfg["plate"]))
        & (traj_df["sample"] == str(cfg["sample"]))
    ].copy()
    if "clone_label" in traj_df.columns and cfg.get("clone_label") is not None:
        sub = sub[sub["clone_label"] == int(cfg["clone_label"])].copy()
    if sub.empty:
        return pd.DataFrame()

    days_cfg = cfg.get("days", None)
    if days_cfg is None:
        out = sub.copy()
    else:
        out = sub[sub["day"].isin(days_cfg)].copy()
    return out.sort_values("day").reset_index(drop=True)


def has_fluctuation_pattern(points: List[Tuple[float, float, float]]) -> bool:
    """Detect a loop-like trajectory with drop/rise followed by rebound/fall."""
    if len(points) < MIN_TRAJECTORY_DAYS:
        return False

    for ia in range(len(points) - 2):
        _, xa, ya = points[ia]
        for ib in range(ia + 1, len(points) - 1):
            _, xb, yb = points[ib]
            if (xa - xb) < FLUCT_DROP_MIN or (yb - ya) < FLUCT_RISE_MIN:
                continue
            for ic in range(ib + 1, len(points)):
                _, xc, yc = points[ic]
                if (xc - xb) >= FLUCT_REBOUND_MIN and (yb - yc) >= FLUCT_FALL_MIN:
                    return True
    return False


def has_irreversible_escape(points: List[Tuple[float, float, float]]) -> Tuple[bool, Optional[float]]:
    """Detect high/low switching with a final low-density/high-growth state."""
    if not points:
        return False, None

    final_day, final_x, final_y = points[-1]
    if final_x > ESCAPE_X_MAX or final_y < ESCAPE_Y_MIN:
        return False, None

    states: List[str] = []
    for _, x, y in points:
        if x >= RETURN_X_MIN and y <= RETURN_Y_MAX:
            state = "H"  # high density / low growth
        elif x <= ESCAPE_X_MAX and y >= ESCAPE_Y_MIN:
            state = "L"  # low density / high growth
        else:
            continue
        if not states or states[-1] != state:
            states.append(state)

    sequence = "".join(states)
    if "HLHL" not in sequence and not sequence.endswith("HL"):
        return False, None

    return True, float(final_day)


def select_behavior_wells(traj_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Select all clone trajectories showing both the fluctuation pattern and irreversible escape."""
    if traj_df.empty:
        return []

    selected: List[Dict[str, Any]] = []
    group_cols = ["strategy", "plate", "sample"]
    use_clone_level = "clone_label" in traj_df.columns
    if use_clone_level:
        group_cols.append("clone_label")

    for group_key, sub in traj_df.groupby(group_cols, sort=False):
        if use_clone_level:
            strategy, plate, sample, clone_label = group_key
            clone_label = int(clone_label)
        else:
            strategy, plate, sample = group_key
            clone_label = None

        points = [
            (float(r.day), float(r.x), float(r.y))
            for r in sub.sort_values("day").itertuples(index=False)
        ]
        if not has_fluctuation_pattern(points):
            continue

        is_irreversible, escape_day = has_irreversible_escape(points)
        if not is_irreversible:
            continue

        selected.append({
            "strategy": str(strategy),
            "plate": str(plate),
            "sample": str(sample),
            "clone_label": clone_label,
            "label": (
                f"{str(sample).split('-')[0]} clone {clone_label}"
                if clone_label is not None
                else str(sample).split("-")[0]
            ),
            "color": TRAJECTORY_COLOR,
            "escape_day": escape_day,
        })

    selected.sort(
        key=lambda cfg: (
            float(cfg["escape_day"]),
            cfg["strategy"],
            cfg["sample"],
            -1 if cfg.get("clone_label") is None else int(cfg["clone_label"]),
        )
    )
    return selected


def plot_trajectory_on_ax(
    ax: plt.Axes,
    hp: pd.DataFrame,
    color: str,
    label: Optional[str] = None,
    annotate: bool = False,
) -> None:
    """Plot one trajectory as connected line segments with later segments more transparent."""
    if hp.empty:
        return

    day_vals = hp["day"].to_numpy(dtype=float)
    if float(day_vals.max()) > float(day_vals.min()):
        alpha_vals = TRAJECTORY_ALPHA_START - (
            (TRAJECTORY_ALPHA_START - TRAJECTORY_ALPHA_END)
            * (day_vals - day_vals.min())
            / (day_vals.max() - day_vals.min())
        )
    else:
        alpha_vals = np.full_like(day_vals, TRAJECTORY_ALPHA_START, dtype=float)

    if annotate and label:
        first = hp.iloc[0]
        ax.annotate(
            label,
            (float(first["x"]), float(first["y"])),
            xytext=(2, 3),
            textcoords="offset points",
            fontsize=5,
            color=color,
            zorder=55,
        )

    for i in range(len(hp) - 1):
        x0, y0 = float(hp.iloc[i]["x"]), float(hp.iloc[i]["y"])
        x1, y1 = float(hp.iloc[i + 1]["x"]), float(hp.iloc[i + 1]["y"])
        seg_rgba = mcolors.to_rgba(color, alpha=float(alpha_vals[i + 1]))
        ax.plot([x0, x1], [y0, y1], color=seg_rgba, lw=TRAJECTORY_LW, zorder=52)


def get_panel_day_markers(cfg: Dict[str, Any]) -> List[int]:
    key = (str(cfg["plate"]), str(cfg["sample"]), cfg.get("clone_label"))
    return PANEL_DAY_MARKERS.get(key, [])


def plot_panel_day_markers(
    ax: plt.Axes,
    hp: pd.DataFrame,
    days: List[int],
    color: str,
) -> None:
    if hp.empty or not days:
        return

    marked = hp[hp["day"].isin(days)].copy()
    if marked.empty:
        return

    ax.scatter(
        marked["x"],
        marked["y"],
        s=PANEL_DAY_MARKER_SIZE,
        facecolors=PANEL_DAY_MARKER_FACECOLOR,
        edgecolors=color,
        linewidths=0.8,
        zorder=56,
    )

    for row in marked.itertuples(index=False):
        ax.annotate(
            f"{int(row.day)}",
            (float(row.x), float(row.y)),
            xytext=PANEL_DAY_MARKER_TEXT_OFFSET,
            textcoords="offset points",
            fontsize=5,
            color=color,
            zorder=57,
        )


# =============================================================================
# LOAD & FILTER DATA
# =============================================================================
if not IN_CSV.is_file():
    raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

df = pd.read_csv(IN_CSV)
df = apply_experiment_filters(df)
df = df[df["change_type"] == CHANGE_TYPE].copy()
if df.empty:
    raise ValueError(
        f"No rows with change_type={CHANGE_TYPE!r} found in {IN_CSV}. "
        "Generate the Figure 2h table with CHANGE_KIND='rel' before running this plot."
    )
df["strategy"] = df["strategy"].map(normalize_strategy)

source_density_col = TOTAL_DENSITY_COL if TOTAL_DENSITY_COL in df.columns else LEGACY_TOTAL_DENSITY_COL
if source_density_col not in df.columns:
    raise ValueError(
        f"Input CSV must contain {TOTAL_DENSITY_COL!r} or {LEGACY_TOTAL_DENSITY_COL!r}: {IN_CSV}"
    )
source_res_density_col = next((col for col in RESISTANT_DENSITY_COLS if col in df.columns), None)
if source_res_density_col is None:
    raise ValueError(
        "Input CSV must contain a first-frame resistant-density column for gated plotting. "
        f"Tried: {', '.join(RESISTANT_DENSITY_COLS)}. CSV: {IN_CSV}"
    )
df["Mean_Total_Density"] = pd.to_numeric(df[source_density_col], errors="coerce")
df["Mean_Total_Density"] = df["Mean_Total_Density"] * X_DENSITY_DISPLAY_SCALE
df["Mean_Resistant_Density"] = pd.to_numeric(df[source_res_density_col], errors="coerce")
df["Mean_Resistant_Density_Change"] = pd.to_numeric(
    df["Mean_Resistant_Density_Change"], errors="coerce"
)

lower_res, upper_res = RES_DENSITY_WINDOW
pre_gate_n = len(df)
df = df[
    df["Mean_Resistant_Density"].between(lower_res, upper_res, inclusive="both")
].copy()
print(
    "[gate] Kept "
    f"{len(df)}/{pre_gate_n} rows with first-frame resistant density "
    f"{lower_res:g} to {upper_res:g} cells µm$^{{-2}}$."
)
if df.empty:
    raise ValueError(
        "No rows remain after applying the first-frame resistant-density gate "
        f"{lower_res:g} to {upper_res:g} cells µm$^{{-2}}$."
    )

if NORMALIZE_3DAY_INTERVAL_CHANGE and "timepoint" in df.columns:
    tp_vals = pd.to_numeric(df["timepoint"], errors="coerce")
    interval_days = tp_vals.apply(
        lambda v: interval_days_from_timepoint0(int(v)) if np.isfinite(v) and int(v) >= 0 else np.nan
    )
    is_three_day = interval_days == 3.0
    n_three = int(is_three_day.sum())
    if n_three > 0:
        df.loc[is_three_day, "Mean_Resistant_Density_Change"] = (
            df.loc[is_three_day, "Mean_Resistant_Density_Change"] / 1.5
        )
    print(f"[normalize] 3-day interval correction applied to {n_three} rows (divide by 1.5).")

df = df.dropna(subset=["Mean_Total_Density", "Mean_Resistant_Density_Change"])
if RESTRICT_TO_PLOTTING_WINDOW:
    # Optional truncation mode: use only in-frame samples for KDE.
    df = df[
        df["Mean_Total_Density"].between(*X_LIM)
        & df["Mean_Resistant_Density_Change"].between(*Y_LIM)
    ].copy()

# Keep a full post-cleaning table for trajectory overlays (including excluded strategies).
df_highlight = df.copy()
traj_df = build_daily_trajectory_table(df_highlight)

# Keep contour plotting restricted to requested strategies only.
include_strategies_norm = list(dict.fromkeys(normalize_strategy(s) for s in INCLUDE_STRATEGIES))
df = df[df["strategy"].isin(set(include_strategies_norm))].copy()

strategies = [s for s in include_strategies_norm if s in set(df["strategy"].unique())]
print(f"Plotting strategies: {strategies}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    behavior_wells = select_behavior_wells(traj_df)
    print(
        "[highlight] Plotting "
        f"{len(behavior_wells)} clone trajectories with fluctuation + irreversible escape."
    )
    for cfg in behavior_wells:
        if cfg.get("clone_label") is None:
            print(f"[highlight] {cfg['sample']} ({cfg['strategy']} / {cfg['plate']})")
        else:
            print(
                f"[highlight] {cfg['sample']} clone {cfg['clone_label']} "
                f"({cfg['strategy']} / {cfg['plate']})"
            )

    fig, ax = new_figure()

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    yticks = ticks_divisible_by_5_or_10(Y_LIM, target_n=6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([format_tick_label(float(y)) for y in yticks])

    xticks = ticks_divisible_by_5_or_10(X_LIM, target_n=6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_tick_label(float(x)) for x in xticks])

    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL, labelpad=1)

    legend_handles: List[plt.Line2D] = []

    for i, strat in enumerate(strategies):
        sub = df[df["strategy"] == strat]
        col = strategy_color(strat)

        ok = filled_contours_on_ax(
            ax,
            sub["Mean_Total_Density"].to_numpy(),
            sub["Mean_Resistant_Density_Change"].to_numpy(),
            base_color=col,
            zorder=1 + 5 * i,
            cache_key=strat,
        )

        if ok:
            legend_handles.append(
                plt.Line2D([0], [0], color=col, lw=2.2, label=strat)
            )

    if SHOW_LEGEND and legend_handles:
        ax.legend(
            handles=legend_handles,
            loc=LEGEND_LOC,
            frameon=False,
            ncol=1,
            handletextpad=0.4,
            columnspacing=0.8,
            borderaxespad=0.2,
            fontsize=5,
        )

    if SHOW_BEHAVIOR_TRAJECTORIES_ON_MAIN:
        for cfg in behavior_wells:
            hp = get_highlight_points(traj_df, cfg)
            if hp.empty:
                continue
            plot_trajectory_on_ax(
                ax,
                hp,
                color=str(cfg["color"]),
                label=str(cfg.get("label", "")),
                annotate=ANNOTATE_BEHAVIOR_TRAJECTORIES,
            )

    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)

    ft.savefig(fig, OUT_BASENAME.with_suffix(".pdf"))

    n_panels = len(behavior_wells)
    if n_panels > 0:
        ncols = min(BEHAVIOR_PANEL_NCOLS, n_panels)
        nrows = int(math.ceil(n_panels / ncols))
        fig_panels, axes = ft.new_subplots(
            nrows,
            ncols,
            ax_width_mm=PANEL_AXES_WIDTH_MM,
            ax_height_mm=PANEL_AXES_HEIGHT_MM,
            margins=PANEL_MARGINS_MM,
            wspace_mm=PANEL_GAP_MM,
            hspace_mm=PANEL_GAP_MM,
            dpi=DPI,
            sharex=True,
            sharey=True,
            apply_style=False,
        )
        axes_arr = np.atleast_1d(axes).ravel()

        yticks = ticks_divisible_by_5_or_10(Y_LIM, target_n=6)
        xticks = ticks_divisible_by_5_or_10(X_LIM, target_n=6)
        ytick_labels = [format_tick_label(float(y)) for y in yticks]
        xtick_labels = [format_tick_label(float(x)) for x in xticks]

        panel_legend_handles: List[plt.Line2D] = []
        for panel_idx, (ax_panel, cfg) in enumerate(zip(axes_arr, behavior_wells)):
            row_idx = panel_idx // ncols
            col_idx = panel_idx % ncols

            ax_panel.set_xlim(*X_LIM)
            ax_panel.set_ylim(*Y_LIM)
            ax_panel.set_xticks(xticks)
            ax_panel.set_yticks(yticks)

            if row_idx == nrows - 1:
                ax_panel.set_xticklabels(xtick_labels)
                ax_panel.set_xlabel(X_AXIS_LABEL)
            else:
                ax_panel.set_xticklabels([])
                ax_panel.set_xlabel("")

            if col_idx == 0:
                ax_panel.set_yticklabels(ytick_labels)
                ax_panel.set_ylabel(Y_AXIS_LABEL, labelpad=1)
            else:
                ax_panel.set_yticklabels([])
                ax_panel.set_ylabel("")

            ax_panel.set_title(str(cfg["label"]), fontsize=6, pad=2)

            for i, strat in enumerate(strategies):
                sub = df[df["strategy"] == strat]
                col = strategy_color(strat)
                ok = filled_contours_on_ax(
                    ax_panel,
                    sub["Mean_Total_Density"].to_numpy(),
                    sub["Mean_Resistant_Density_Change"].to_numpy(),
                    base_color=col,
                    zorder=1 + 5 * i,
                    cache_key=strat,
                )
                if panel_idx == 0 and ok:
                    panel_legend_handles.append(
                        plt.Line2D([0], [0], color=col, lw=2.2, label=strat)
                    )

            hp = get_highlight_points(traj_df, cfg)
            if hp.empty:
                target = f"{cfg['plate']} / {cfg['sample']}"
                if cfg.get("clone_label") is not None:
                    target += f" / clone {cfg['clone_label']}"
                print(f"[highlight WARN] No highlight points found for {target}.")
                continue

            if SHOW_BEHAVIOR_TRAJECTORIES_IN_PANELS:
                plot_trajectory_on_ax(
                    ax_panel,
                    hp,
                    color=str(cfg["color"]),
                    label=str(cfg.get("label", "")),
                    annotate=False,
                )
                plot_panel_day_markers(
                    ax_panel,
                    hp,
                    days=get_panel_day_markers(cfg),
                    color=str(cfg["color"]),
                )

            ax_panel.tick_params(axis="x", pad=1)
            ax_panel.tick_params(axis="y", pad=1)

        for ax_panel in axes_arr[n_panels:]:
            ax_panel.set_visible(False)

        if SHOW_LEGEND and panel_legend_handles:
            axes_arr[0].legend(
                handles=panel_legend_handles,
                loc=LEGEND_LOC,
                frameon=False,
                ncol=1,
                handletextpad=0.4,
                columnspacing=0.8,
                borderaxespad=0.2,
                fontsize=5,
            )

        ft.savefig(fig_panels, OUT_BASENAME_ALL_BEHAVIOR_PANELS.with_suffix(".pdf"))
        plt.close(fig_panels)

    plt.close(fig)


if __name__ == "__main__":
    main()
