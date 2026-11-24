"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - Y:\Membe
# Upstream/related script reference(s):
# - shared/figtools.py
# - import matplotlib.py
# =============================================================================
TL;DR
-----
This script generates the side-by-side supplementary comparison of relative
growth for alpha2 versus alpha4 conditions.

Detailed
--------
Inputs:
- Plate-reader XLSX tables for the relevant alpha2/alpha4 sheets.

Outputs:
- A supplementary side-by-side plot comparing the two conditions.

How it works:
- Loads the selected plate-reader tables for each condition.
- Builds the custom diverging color map used in the panel.
- Draws the side-by-side summary in the manuscript’s chosen layout.

Pipeline role:
- Supplementary plotting endpoint for the alpha2/alpha4 plate-reader branch.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: 2025_01_Plate_Reader_Results.xlsx (alpha2/alpha4 worksheets).
# Scripts: shared/figtools.py for publication sizing and export.
# Outputs: supplementary alpha2/alpha4 relative-growth comparison PDF.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared.data_paths import EXP_RAW_DATA

LEGEND_FONTSIZE = 5
LEGEND_COLS = 2
LEGEND_TITLE_SIZE = 5

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to each axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 65.0
FIGURE_AXES_HEIGHT_MM = 42.0
FIGURE_MARGINS_MM = (12.0, 4.0, 12.0, 6.0)  # left, right, bottom, top
FIGURE_PANEL_GAP_MM = 8.0
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
DPI = 300
SHOW_FIGURE = True

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_FONTSIZE,
)

XLSX = EXP_RAW_DATA / "alamar_blue" / "2025_01_Plate_Reader_Results.xlsx"
SHEETS = ("G418 alpha-2", "G418 alpha-4")
SHEET_TITLES = {
    "G418 alpha-2": "Sensitive Cells",
    "G418 alpha-4": "Resistant Cells",
}

ROW_SLICE_MEANS = slice(281, 294)
ROW_SLICE_SDS = slice(296, 309)
COLS = [0, 1, 2, 3, 4]
COLNAMES = ["Condition", "0", "2", "5", "7"]
X_DAYS = np.array([0, 2, 5, 7], dtype=float)

PALETTES = {
    "purple_orange": ("#6A3D9A", "#FF7F0E"),
    "teal_magenta": ("#008B8B", "#C51B7D"),
    "navy_coral": ("#1F3A93", "#FF6F61"),
    "slate_rose": ("#3E4C59", "#D81B60"),
}
PALETTE_CHOICE = "purple_orange"
OUTPUT_PDF = ft.paper_figure_data_dir(__file__) / "Figure 3.2 RelativeGrowth_alpha2_alpha4_side_by_side.pdf"


def make_diverging_cmap(n_colors: int, name: str = "custom_div") -> list[tuple[float, float, float, float]]:
    c0, c1 = PALETTES[PALETTE_CHOICE]
    cmap = LinearSegmentedColormap.from_list(name, [c0, c1])
    return [cmap(i) for i in np.linspace(0.05, 0.95, n_colors)]


def load_tables(sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(XLSX, sheet_name=sheet_name, header=None)

    means = (
        df.iloc[ROW_SLICE_MEANS, COLS]
        .copy()
        .set_axis(COLNAMES, axis=1)
        .dropna(subset=["Condition"])
    )
    sds = (
        df.iloc[ROW_SLICE_SDS, COLS]
        .copy()
        .set_axis(COLNAMES, axis=1)
        .dropna(subset=["Condition"])
        .set_index("Condition")
    )

    for col in COLNAMES[1:]:
        means[col] = pd.to_numeric(means[col], errors="coerce")
        sds[col] = pd.to_numeric(sds[col], errors="coerce")

    return means, sds


def main() -> None:
    fig, axes = ft.new_subplots(
        1,
        2,
        ax_width_mm=FIGURE_AXES_WIDTH_MM,
        ax_height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        wspace_mm=FIGURE_PANEL_GAP_MM,
        dpi=DPI,
        sharex=True,
        sharey=True,
        apply_style=False,
    )

    panel_labels = ["a", "b"]
    for ax, label in zip(axes, panel_labels):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=TEXT_SIZE_PT,
            fontweight="bold",
            va="top",
            ha="right",
        )

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylim(0.03, 40)
        ax.set_xticks([0, 2, 5, 7])
        ax.set_xlabel("Time after treatment start [d]")

    axes[0].set_ylabel("Relative cell count change")

    for ax, sheet in zip(axes, SHEETS):
        means, sds = load_tables(sheet)
        conditions = means["Condition"].tolist()
        colors = make_diverging_cmap(len(conditions))

        for color, (_, row) in zip(colors, means.iterrows()):
            condition = row["Condition"]
            y = row[["0", "2", "5", "7"]].to_numpy()
            if condition in sds.index:
                yerr = sds.loc[condition, ["0", "2", "5", "7"]].to_numpy()
            else:
                yerr = np.zeros_like(y)

            ax.errorbar(
                X_DAYS,
                y,
                yerr=yerr,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                elinewidth=1.0,
                capsize=2,
                color=color,
                label=str(condition),
            )

        ax.set_title(SHEET_TITLES.get(sheet, sheet))
        ax.legend(
            frameon=False,
            ncols=LEGEND_COLS,
            fontsize=LEGEND_FONTSIZE,
            title="G418 concentration [mg/ml]",
            title_fontsize=LEGEND_TITLE_SIZE,
            handlelength=1.4,
            columnspacing=0.8,
            handletextpad=0.4,
        )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, OUTPUT_PDF)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    print("Saved PDF:", OUTPUT_PDF)


if __name__ == "__main__":
    main()
