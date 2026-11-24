#!/usr/bin/env python
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - Y:\Membe
# Upstream/related script reference(s):
# - analysis/legacy/Figure_5_TTP_vs_Density_analysis.py
# - import matplotlib.py
# =============================================================================
# -*- coding: utf-8 -*-
"""
TL;DR
-----
This script compares replicate-level TTP with replicate-level pre-TTP density
summaries around resistant clones.

Detailed
--------
Inputs:
- The replicate-level Figure 4 TTP-versus-density table from
  `analysis/legacy/Figure_5_TTP_vs_Density_analysis.py`.
- Shared strategy colors.

Outputs:
- The Figure 4 TTP-versus-density panel with replicate-level points plus
  strategy-level median summaries.

How it works:
- Loads replicate-level density/TTP summaries already filtered upstream for
  observability and canonical experiment-end handling.
- Uses the active upstream replicate metric, which is the median of replicate
  timepoint-level density medians in cells um^-2 across the observable pre-TTP
  window, plotted as cells per 1000 um^2.
- Plots all retained replicate summaries colored by strategy.
- Summarizes each strategy by the median replicate density and median TTP and
  overlays those medians as crosses.

Pipeline role:
- Figure 4 plotting endpoint connecting replicate-level density context to
  progression timing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.strategy_colors import get_color, get_display_name


def strategy_color_and_label(strategy: str) -> tuple[str, str]:
    s = str(strategy).strip()
    return get_color(s, "#000000"), get_display_name(s)


# =============================================================================
# EDITABLE FIGURE PARAMETERS
# =============================================================================
# Width and height refer to the axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 35.0
FIGURE_AXES_HEIGHT_MM = 35.0
FIGURE_MARGINS_MM = (11.0, 1.0, 14.0, 1.0)  # left, right, bottom, top
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


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    return ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )



# =============================================================================
# Paths
# =============================================================================
IN_CSV = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_5_Data"
    r"\Figure5c_density_vs_TTP.csv"
)

PUB_DIR = ft.paper_figure_data_dir(__file__)
PUB_DIR.mkdir(parents=True, exist_ok=True)
OUT_MEDIAN_SUMMARY = PUB_DIR / "Figure_5_TTP_vs_Density.pdf"
PUB_MEDIAN_SUMMARY = OUT_MEDIAN_SUMMARY

REPLICATE_POINT_SIZE = 14.0
REPLICATE_POINT_ALPHA = 0.7
MEDIAN_MARKER_SIZE = 55.0
DENSITY_PLOT_SCALE = 1000.0

YLIM: Optional[Tuple[float, float]] = None


# =============================================================================
# Data loading
# =============================================================================
def load_df() -> pd.DataFrame:
    if not IN_CSV.is_file():
        raise FileNotFoundError(f"Cannot find analysis CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    for col in ["median_density", "ttp_day", "n_density_rows", "event"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["median_density", "ttp_day"]).copy()
    return df


def padded_limits(values: np.ndarray, *, lower_zero: bool = False) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    pad = 0.05 * (v_max - v_min) if v_max > v_min else 0.5
    lo = v_min - pad
    hi = v_max + pad
    if lower_zero:
        lo = max(0.0, lo)
    return lo, hi


# =============================================================================
# Plot
# =============================================================================
def make_strategy_median_summary_plot(df: pd.DataFrame) -> None:
    df = df[df["strategy"].astype(str).str.upper() != "ATAGENT1"].copy()

    strategies = sorted(df["strategy"].dropna().astype(str).unique().tolist())

    xs, ys = [], []
    labels, colors = [], []

    for strat in strategies:
        sub = df[df["strategy"].astype(str) == strat].copy()
        density_plot = sub["median_density"].to_numpy(dtype=float) * DENSITY_PLOT_SCALE
        x_center = float(np.median(density_plot))
        y_center = float(np.median(sub["ttp_day"].to_numpy(dtype=float)))
        if not (np.isfinite(x_center) and np.isfinite(y_center)):
            continue

        c, lab = strategy_color_and_label(strat)
        xs.append(x_center)
        ys.append(y_center)
        labels.append(f"{lab} (n={len(sub)})")
        colors.append(c)

    fig, ax = new_figure()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    for strat in strategies:
        sub = df[df["strategy"].astype(str) == strat].copy()
        if sub.empty:
            continue
        c, _ = strategy_color_and_label(strat)
        ax.scatter(
            sub["median_density"].to_numpy(dtype=float) * DENSITY_PLOT_SCALE,
            sub["ttp_day"].to_numpy(dtype=float),
            s=22.0,
            color=c,
            alpha=0.8,
            edgecolors="none",
            zorder=2,
        )

    for mx, my, c in zip(
        xs,
        ys,
        colors,
    ):
        ax.scatter(
            [mx],
            [my],
            s=MEDIAN_MARKER_SIZE,
            marker="x",
            color=c,
            linewidths=1.4,
            zorder=4,
        )

    ax.set_xlabel(r"Median Pre-TTP Local Density (cells / 1000 µm$^2$)")
    ax.set_ylabel("Time to Progression")
    x_limits = padded_limits(
        df["median_density"].to_numpy(dtype=float) * DENSITY_PLOT_SCALE,
        lower_zero=True,
    )
    y_limits = padded_limits(df["ttp_day"].to_numpy(dtype=float), lower_zero=True)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if YLIM is not None:
        ax.set_ylim(*YLIM)

    handles = []
    seen = set()
    for lab, c in zip(labels, colors):
        if lab in seen:
            continue
        seen.add(lab)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                markersize=5,
                markeredgewidth=1.2,
                color=c,
                label=lab,
            )
        )
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=LEGEND_SIZE_PT)

    OUT_MEDIAN_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, OUT_MEDIAN_SUMMARY)
    ft.savefig(fig, PUB_MEDIAN_SUMMARY)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"[SAVE] Wrote: {OUT_MEDIAN_SUMMARY}")
    print(f"[SAVE] Wrote: {PUB_MEDIAN_SUMMARY}")


def main() -> None:
    df = load_df()
    make_strategy_median_summary_plot(df)


if __name__ == "__main__":
    main()
