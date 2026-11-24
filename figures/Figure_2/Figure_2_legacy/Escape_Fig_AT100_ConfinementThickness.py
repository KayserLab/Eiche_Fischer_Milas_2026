"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script shows representative AT100 confinement-thickness trajectories over
time, with treatment days marked along the trajectory.

Detailed
--------
Inputs:
- The confinement-thickness CSV from `analysis/figure_2/Escape_ConfinementThickness.py`.
- Treatment schedules for the corresponding replicates.

Outputs:
- A figure of representative AT100 confinement-thickness traces.

How it works:
- Loads treatment schedules and maps them onto day indices.
- Filters the confinement-thickness table to the relevant AT100 replicates.
- Selects representative wells and plots their confinement-thickness evolution,
  coloring line segments according to treatment timing.

Pipeline role:
- Figure 2 plotting script for representative confinement-thickness dynamics.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared import experiment_filters


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = DATA_ROOT / r"Figures\Escape_Quantification"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_SHARED = ft.paper_figure_data_dir(__file__)
OUT_DIR_SHARED.mkdir(parents=True, exist_ok=True)
IN_CSV = OUT_DIR / "Escape_ConfinementThickness_all_replicates.csv"

DEBUG_VISUALIZE_C5 = False
DEBUG_REPLICATE = "C5_AT50-C5"

KEEP_STRATEGY = "AT100"
HIGHLIGHT_REPLICATE = "D4_AT100-D4"
LOW_INITIAL_MAX = 5000.0
DAY50_TARGET = 2000.0
OTHER_ALPHA = 0.12

COLOR_OFF = "#92278f"
COLOR_ON = "#d3d3d3"
LINEWIDTH = 0.9
ALPHA_LINE = 1.0
GRID_ALPHA = 0.25
SCHEDULE_BASENAME = "treatment_schedule.txt"
_RX_DIGITS = re.compile(r"(\d+)")


USE_THIN_FIGURE = True

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis
# length. Margins are extra space for tick labels and axis labels.
FIGURE_AXES_WIDTH_MM = 50.0
FIGURE_AXES_HEIGHT_MM = 25.0
# Keep the plot box fixed, but give the rotated y-axis title enough room.
FIGURE_MARGINS_MM = (18.0, 4.0, 8.0, 4.0) if USE_THIN_FIGURE else (18.0, 4.0, 5.0, 2.0)
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
SHOW_FIGURE = DEBUG_VISUALIZE_C5

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


def parse_strategy(rep: str) -> str:
    m = re.match(r"^[A-H]\d{1,2}_([^-_]+)-[A-H]\d{1,2}$", str(rep))
    return m.group(1) if m else "UNKNOWN"


def find_schedule_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if SCHEDULE_BASENAME in filenames:
            out.append(Path(dirpath) / SCHEDULE_BASENAME)
    return out


def parse_treated_days(tokens: list[str]) -> set[int]:
    treated: set[int] = set()
    for t in tokens:
        if not t:
            continue
        if "s" in t:
            continue
        m = _RX_DIGITS.search(t)
        if m:
            treated.add(int(m.group(1)))
    return treated


def load_treatment_map(root: Path) -> dict[str, set[int]]:
    sched_files = find_schedule_files(root)
    out: dict[str, set[int]] = {}
    for sched in sched_files:
        with open(sched, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                out[parts[0]] = parse_treated_days(parts[1:])
    return out


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    rules: Dict[str, Any] = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    keep = np.ones(len(df), dtype=bool)
    reps = df["replicate"].astype(str).to_numpy()
    day = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=float)
    for i, s in enumerate(reps):
        if s not in rules:
            continue
        cutoff = rules[s]
        if cutoff is None:
            keep[i] = False
        else:
            try:
                if np.isfinite(day[i]) and day[i] > float(cutoff):
                    keep[i] = False
            except Exception:
                pass
    return df[keep].copy()


def add_treatment_colored_line(ax, x: np.ndarray, y: np.ndarray, alpha: float, treated_days: set[int]) -> None:
    if len(x) < 2:
        return
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_colors = []
    for i in range(len(x) - 1):
        d0 = int(np.round(float(x[i])))
        seg_colors.append(COLOR_ON if d0 in treated_days else COLOR_OFF)
    lc = LineCollection(segs, colors=seg_colors, linewidths=LINEWIDTH, alpha=float(alpha), capstyle="round", joinstyle="round")
    ax.add_collection(lc)


def select_representative_replicates(df: pd.DataFrame) -> list[str]:
    chosen: list[str] = []
    all_reps = [str(r) for r in df["replicate"].dropna().unique()]

    if HIGHLIGHT_REPLICATE in all_reps:
        chosen.append(HIGHLIGHT_REPLICATE)

    per_rep_initial: list[tuple[str, float]] = []
    per_rep_day50: list[tuple[str, float]] = []
    for rep, g in df.groupby("replicate"):
        g = g.sort_values("day")
        rep_name = str(rep)
        init_val = float(g["confinement_thickness"].iloc[0])
        per_rep_initial.append((rep_name, init_val))

        i50 = int(np.argmin(np.abs(g["day"].to_numpy(dtype=float) - 50.0)))
        d50_val = float(g["confinement_thickness"].iloc[i50])
        per_rep_day50.append((rep_name, d50_val))

    low_candidates = [(r, v) for (r, v) in per_rep_initial if v < LOW_INITIAL_MAX and r not in chosen]
    if low_candidates:
        low_rep = min(low_candidates, key=lambda t: t[1])[0]
        chosen.append(low_rep)

    d50_candidates = [(r, v) for (r, v) in per_rep_day50 if r not in chosen]
    if d50_candidates:
        d50_rep = min(d50_candidates, key=lambda t: abs(t[1] - DAY50_TARGET))[0]
        chosen.append(d50_rep)

    if len(chosen) < 3:
        for rep in all_reps:
            if rep not in chosen:
                chosen.append(rep)
            if len(chosen) >= 3:
                break

    return chosen[:3]


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(IN_CSV)
    df = pd.read_csv(IN_CSV)
    if "replicate" not in df.columns:
        raise ValueError("Missing replicate in confinement CSV")
    df["strategy"] = df["replicate"].astype(str).map(parse_strategy)
    df = df[df["strategy"] == KEEP_STRATEGY].copy()
    df = apply_filters(df)
    if DEBUG_VISUALIZE_C5:
        df = df[df["replicate"] == DEBUG_REPLICATE].copy()
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["confinement_thickness"] = pd.to_numeric(df["confinement_thickness"], errors="coerce")
    df = df[np.isfinite(df["day"]) & np.isfinite(df["confinement_thickness"])].copy()
    selected_reps = select_representative_replicates(df)
    print(f"[select] Replicates plotted: {selected_reps}")
    out_csv = OUT_DIR / "Escape_Fig_AT100_ConfinementThickness_data.csv"
    df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} rows={len(df)}")
    if df.empty:
        return

    treat_map = load_treatment_map(DATA_ROOT)
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    for rep, g in df.groupby("replicate"):
        g = g.sort_values("day")
        x = g["day"].to_numpy(dtype=float)
        y = g["confinement_thickness"].to_numpy(dtype=float)
        alpha = ALPHA_LINE if str(rep) in selected_reps else OTHER_ALPHA
        add_treatment_colored_line(ax, x, y, alpha=alpha, treated_days=treat_map.get(str(rep), set()))

    ax.set_xlabel("Day")
    ax.set_ylabel("Confinement Thickness")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)

    # Explicit limits are needed for LineCollection-only plots.
    xmin = float(df["day"].min())
    xmax = float(df["day"].max())
    ymin = float(df["confinement_thickness"].min())
    ymax = float(df["confinement_thickness"].max())
    yr = (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(0, 60)
    ax.set_ylim(max(0.0, ymin - 0.05 * yr), ymax + 0.05 * yr)

    out_pdf = OUT_DIR_SHARED / "Escape_Fig_AT100_ConfinementThickness.pdf"
    ft.savefig(fig, out_pdf)
    print(f"[save] {out_pdf}")
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
