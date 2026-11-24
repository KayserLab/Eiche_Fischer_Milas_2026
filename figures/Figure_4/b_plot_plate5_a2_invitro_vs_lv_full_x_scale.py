#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Export separate full-horizon experimental and LV AT100 trajectories.

The two Figure 4b panels retain the same axes dimensions and scales.  The
experimental panel shows only the measured trajectory, whereas the LV panel
shows only its fitted LV trajectory.
"""
from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Plate 5 A2 Results trajectory, fitted_lv_params.json, treatment
# schedule, and all_escape_margin.csv resolved by the base plotting script.
# Scripts: analysis/figure_4/plot_plate5_a2_invitro_vs_lv.py,
# analysis/figure_4/plot_replicate_resistant_invitro_dpd_lv.py, and shared/figtools.py.
# Outputs: full-horizon Plate 5 A2 in-vitro/LV PDF.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
FIGURE_DIR = THIS_FILE.parent
for path in (REPO_ROOT, FIGURE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared import figtools as ft
from analysis.figure_4 import plot_plate5_a2_invitro_vs_lv as base
from shared.strategy_colors import get_color


OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "Example_Trajectories"
EXPERIMENT_OUTPUT_PDF = OUTPUT_DIR / "Plate_5_A2_AT100-A2_experiment.pdf"
LV_OUTPUT_PDF = OUTPUT_DIR / "Plate_5_A2_AT100-A2_LV.pdf"
LV_LINESTYLE = ":"
PROGRESSION_LINESTYLE = "--"
TREATMENT_COLOR = get_color("AT100")
TREATMENT_LINEWIDTH = 1.5
EXPERIMENT_AXES_WIDTH_MM = 15.0
EXPERIMENT_AXES_HEIGHT_MM = 10.0
FIGURE_MARGINS_MM = (12.0, 3.0, 10.0, 5.0)


def plot() -> None:
    exp, lv, _escape_day, initial_total = base.load_trajectories()
    # Full x scale: retain all experimentally observed points and simulate LV
    # through that same final observation day.
    exp = exp.sort_values("day").copy()
    lv = lv[lv["day"] <= float(exp["day"].max())].copy()
    ymax = max(1.05, float(max(exp["sensitive"].max(), exp["resistant"].max(), lv["lv_sensitive"].max(), lv["lv_resistant"].max()) / initial_total) * 1.08)
    plot_trajectory(
        exp, initial_total, ymax=ymax, output_pdf=EXPERIMENT_OUTPUT_PDF,
        sensitive_column="sensitive", resistant_column="resistant", linestyle="-",
        treatment_intervals=intervals_from_treatment_days(
            base.read_invitro_treatment_days(), float(exp["day"].max())
        ),
        axes_width_mm=EXPERIMENT_AXES_WIDTH_MM,
        axes_height_mm=EXPERIMENT_AXES_HEIGHT_MM,
        treatment_linestyle="-",
    )
    plot_trajectory(
        lv, initial_total, ymax=ymax, output_pdf=LV_OUTPUT_PDF,
        sensitive_column="lv_sensitive", resistant_column="lv_resistant", linestyle=LV_LINESTYLE,
        treatment_intervals=intervals_from_lv_trajectory(lv),
        axes_width_mm=base.AXES_WIDTH_MM,
        axes_height_mm=base.AXES_HEIGHT_MM,
        treatment_linestyle=LV_LINESTYLE,
        y_label=r"Normalized count, $\eta$",
    )


def intervals_from_treatment_days(
    treatment_days: set[int], final_day: float
) -> list[tuple[float, float]]:
    """Coalesce consecutive experimental treatment days into exposure intervals."""
    days = sorted(day for day in treatment_days if day < final_day)
    if not days:
        return []
    intervals: list[tuple[float, float]] = []
    start = previous = days[0]
    for day in days[1:] + [None]:
        if day is None or day != previous + 1:
            intervals.append((float(start), min(float(previous + 1), final_day)))
            if day is None:
                break
            start = day
        previous = day
    return intervals


def intervals_from_lv_trajectory(trajectory) -> list[tuple[float, float]]:
    """Return contiguous LV treatment-on intervals through the displayed horizon."""
    intervals: list[tuple[float, float]] = []
    for current, following in zip(trajectory.iloc[:-1].itertuples(), trajectory.iloc[1:].itertuples()):
        if not current.treatment_on:
            continue
        start, stop = float(current.day), float(following.day)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], stop)
        else:
            intervals.append((start, stop))
    return intervals


def draw_treatment_timing(
    ax: plt.Axes,
    intervals: list[tuple[float, float]],
    *,
    linestyle: str,
) -> None:
    """Draw AT100 exposure intervals immediately above the data axes."""
    for start, stop in intervals:
        ax.plot(
            [start, stop], [1.045, 1.045], transform=ax.get_xaxis_transform(),
            color=TREATMENT_COLOR, lw=TREATMENT_LINEWIDTH, linestyle=linestyle,
            solid_capstyle="butt", clip_on=False, zorder=8,
        )


def plot_trajectory(
    trajectory,
    initial_total: float,
    *,
    ymax: float,
    output_pdf: Path,
    sensitive_column: str,
    resistant_column: str,
    linestyle: str,
    treatment_intervals: list[tuple[float, float]],
    axes_width_mm: float,
    axes_height_mm: float,
    treatment_linestyle: str,
    y_label: str = "Normalized\n" + r"count, $\eta$",
) -> None:
    """Export one source-specific trajectory panel with treatment timing above it."""
    fig, ax = ft.new_figure(
        width=axes_width_mm, height_mm=axes_height_mm,
        margins=FIGURE_MARGINS_MM, dpi=300, apply_style=False,
    )
    ax.plot(
        trajectory["day"], trajectory[sensitive_column] / initial_total,
        color=base.SENSITIVE_COLOR, linewidth=1.0, linestyle=linestyle,
        marker=None,
    )
    ax.plot(
        trajectory["day"], trajectory[resistant_column] / initial_total,
        color=base.RESISTANT_COLOR, linewidth=1.0, linestyle=linestyle,
        marker=None,
    )
    ax.axhline(
        base.PROGRESSION_THRESHOLD, color="#555555", linewidth=0.8,
        linestyle=PROGRESSION_LINESTYLE, zorder=0,
    )
    draw_treatment_timing(ax, treatment_intervals, linestyle=treatment_linestyle)
    ax.set(
        xlim=(0, float(trajectory["day"].max())), ylim=(0, ymax),
        xlabel="Time [d]",
    )
    ax.set_ylabel(y_label, linespacing=1.05)
    handles = [
        Line2D([], [], color=base.SENSITIVE_COLOR, linewidth=1.0, linestyle=linestyle, label="sen"),
        Line2D([], [], color=base.RESISTANT_COLOR, linewidth=1.0, linestyle=linestyle, label="res"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=1, loc="upper left", handlelength=1.6,
              columnspacing=0.8, handletextpad=0.35)
    ft.format_axes(ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output_pdf, dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    plot()
    print(f"Saved: {EXPERIMENT_OUTPUT_PDF}")
    print(f"Saved: {LV_OUTPUT_PDF}")
