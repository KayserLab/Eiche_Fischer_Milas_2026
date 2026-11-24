"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script plots resistant growth as a function of escape grade using the
newer canonical Figure 2 analysis branch.

Detailed
--------
Inputs:
- The main and QC CSVs written by
  `analysis/legacy/Escape_Fig_Growth_over_Grade_of_Escape.py`.

Outputs:
- Combined and per-strategy growth-versus-escape-grade plots.
- Binned summary plots and optional QC/debug overlays.

How it works:
- Reads the precomputed growth-versus-escape table and maps replicates to the
  canonical strategy vocabulary.
- Generates manuscript-style split plots, plus binned median/IQR summaries
  over escape grade.
- Optionally exports debug overlays for rows that violate expected fraction
  constraints.

Pipeline role:
- Figure 2 plotting endpoint for the growth-over-escape-grade branch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.strategy_colors import get_color, get_display_name


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = DATA_ROOT / r"Figures\Escape_Quantification"
OUT_DIR_SHARED = ft.paper_figure_data_dir(__file__)
OUT_DIR_SHARED_PER_STRAT = OUT_DIR_SHARED / "Escape_Fig_Growth_over_Grade_of_Escape_replot_new_strategy_specific"
IN_CSV = OUT_DIR / "Escape_Fig_Growth_over_Grade_of_Escape_data_new.csv"

OUT_PDF_ALL_LENGTH = OUT_DIR_SHARED / "Escape_Fig_Growth_over_Grade_of_Escape_replot_all_length_new.pdf"
OUT_PDF_ALL_FRACTION = OUT_DIR_SHARED / "Escape_Fig_Growth_over_Grade_of_Escape_replot_all_fraction_new.pdf"
OUT_PDF_COMBINED_BINNED = OUT_DIR_SHARED / "Escape_Fig_Growth_over_Grade_of_Escape_replot_combined_binned_new.pdf"
OUT_PDF_PER_STRAT_PREFIX = OUT_DIR_SHARED_PER_STRAT / "Escape_Fig_Growth_over_Grade_of_Escape_replot_new"
OUT_NEG_CSV = OUT_DIR / "Escape_Fig_Growth_over_Grade_of_Escape_negative_growth_rows_new.csv"
OUT_FRAC_GT1_CSV = OUT_DIR / "Escape_Fig_Growth_over_Grade_of_Escape_fraction_gt1_rows_new.csv"
DEBUG_DIR = OUT_DIR / "Debug_Growth_over_Grade_new"
OUT_PDF_LENGTH_ONLY = OUT_DIR_SHARED / "Escape_Fig_Growth_over_Grade_of_Escape_replot_combined_binned_length_only_new.pdf"

SUMMARY_STRATEGIES = ["CT", "AT50", "AT100"]
SAMPLE_RE = re.compile(r"^[A-H]\d+_([^-]+)-[A-H]\d+$", re.IGNORECASE)

XMIN_ABS = 0.0
XMIN_FRAC = 0.0
XMAX_FRAC = 1.0
YMIN_FIXED = -100.0
YMAX_FIXED = 10000.0
N_BINS_COMBINED = 10
SAVE_DEBUG_OVERLAYS = False
MAX_DEBUG_OVERLAYS = 20

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to axes boxes: the physical x-axis and y-axis
# length. Margins are extra space for tick labels, labels, and titles.
MAIN_AXES_WIDTH_MM = 30.0
MAIN_AXES_HEIGHT_MM = 40.0
MAIN_MARGINS_MM = (7.32, 2.84, 20.12, 2.24)  # left, right, bottom, top
COMBINED_AXES_WIDTH_MM = 30.0
COMBINED_AXES_HEIGHT_MM = 40.0
COMBINED_MARGINS_MM = (16.76, 4.19, 20.12, 2.24)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
COMBINED_YLABEL_X = 0.17
COMBINED_YLABEL_Y = 0.66
COMBINED_YLABEL_FONTSIZE = 7
COMBINED_TICK_LABELSIZE = 6
COMBINED_LABEL_FONTSIZE = 7
COMBINED_YLABEL_TEXT = "Res. proliferation per day"
COMBINED_YTICK_PAD = 2
COMBINED_BOTTOM_XTICKS: tuple[float, ...] = (0.0, 0.5, 1.0)
COMBINED_TOP_XTICKS: tuple[float, ...] = (100.0, 200.0, 300.0)
COMBINED_LENGTH_LABEL_TEXT = "Front len. [10^3 px] / "
COMBINED_RATIO_LABEL_TEXT = "ratio"
COMBINED_LABEL_Y = -0.15

PX_TO_MM = 1.095e-3
LINE_COLOR = "#222222"
XTICKS = (0.0, 100.0, 200.0, 300.0)
YTICKS = (0.0, 1000.0, 2000.0, 3000.0, 4000.0)
X_LABEL = "Escape front length [mm]"
Y_LABEL = "Res. proliferation per day"

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


def new_main_figure() -> tuple[plt.Figure, plt.Axes]:
    return ft.new_figure(
        width=MAIN_AXES_WIDTH_MM,
        height_mm=MAIN_AXES_HEIGHT_MM,
        margins=MAIN_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )


def new_combined_figure() -> tuple[plt.Figure, plt.Axes]:
    return ft.new_figure(
        width=COMBINED_AXES_WIDTH_MM,
        height_mm=COMBINED_AXES_HEIGHT_MM,
        margins=COMBINED_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )


def strategy_display(strategy: str) -> str:
    return get_display_name(strategy)


def strategy_color(strategy: str) -> str:
    return get_color(strategy, "#000000")


def parse_strategy(sample: str) -> str:
    match = SAMPLE_RE.match(str(sample))
    return match.group(1) if match else "UNKNOWN"


def shared_output_path(local_path: Path) -> Path | None:
    per_strat_prefix = f"{OUT_PDF_PER_STRAT_PREFIX.name}_"
    if local_path.name.startswith(per_strat_prefix):
        return OUT_DIR_SHARED_PER_STRAT / local_path.name
    return OUT_DIR_SHARED / local_path.name


def save_pdf_with_shared_copy(fig: plt.Figure, out_pdf: Path, **savefig_kwargs) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, out_pdf, **savefig_kwargs)
    print(f"[save] {out_pdf}")

    shared_pdf = shared_output_path(out_pdf)
    if shared_pdf == out_pdf:
        return
    shared_pdf.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, shared_pdf, **savefig_kwargs)
    print(f"[save] {shared_pdf}")


def apply_axes_length(ax: plt.Axes, x_max_abs: float, y_min: float, y_max: float) -> None:
    ax.set_xlabel("Escape Front Length [px]")
    ax.set_ylabel("Delta resistant cells / day")
    ax.set_xlim(XMIN_ABS, x_max_abs)
    ax.set_ylim(y_min, y_max)


def apply_axes_fraction(ax: plt.Axes, y_min: float, y_max: float) -> None:
    ax.set_xlabel("Escape Front Fraction")
    ax.set_ylabel("Delta resistant cells / day")
    ax.set_xlim(XMIN_FRAC, XMAX_FRAC)
    ax.set_ylim(y_min, y_max)


def make_combined_split_plots(df_summary: pd.DataFrame, x_max_abs: float, y_min: float, y_max: float) -> None:
    fig, ax = new_main_figure()
    for strat, group in df_summary.groupby("strategy"):
        ax.scatter(
            group["escape_front_length_px"],
            group["delta_res_count_per_day"],
            s=9,
            alpha=0.5,
            color=strategy_color(strat),
            label=strategy_display(strat),
        )
    apply_axes_length(ax, x_max_abs=x_max_abs, y_min=y_min, y_max=y_max)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=6)
    fig.suptitle("All Strategies - Length", y=0.98, fontsize=7)
    save_pdf_with_shared_copy(fig, OUT_PDF_ALL_LENGTH)
    plt.close(fig)

    fig, ax = new_main_figure()
    for strat, group in df_summary.groupby("strategy"):
        ax.scatter(
            group["escape_front_fraction"],
            group["delta_res_count_per_day"],
            s=9,
            alpha=0.5,
            color=strategy_color(strat),
            label=strategy_display(strat),
        )
    apply_axes_fraction(ax, y_min=y_min, y_max=y_max)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=6)
    fig.suptitle("All Strategies - Fraction", y=0.98, fontsize=7)
    save_pdf_with_shared_copy(fig, OUT_PDF_ALL_FRACTION)
    plt.close(fig)


def make_per_strategy_split_plots(df_all: pd.DataFrame, x_max_abs: float, y_min: float, y_max: float) -> None:
    for strategy in sorted(df_all["strategy"].dropna().unique()):
        group = df_all[df_all["strategy"] == strategy].copy()
        if group.empty:
            continue

        fig, ax = new_main_figure()
        ax.scatter(
            group["escape_front_length_px"],
            group["delta_res_count_per_day"],
            s=10,
            alpha=0.55,
            color=strategy_color(strategy),
        )
        apply_axes_length(ax, x_max_abs=x_max_abs, y_min=y_min, y_max=y_max)
        fig.suptitle(f"{strategy_display(strategy)} - Length", y=0.98, fontsize=7)
        save_pdf_with_shared_copy(fig, Path(f"{OUT_PDF_PER_STRAT_PREFIX}_{strategy}_length.pdf"))
        plt.close(fig)

        fig, ax = new_main_figure()
        ax.scatter(
            group["escape_front_fraction"],
            group["delta_res_count_per_day"],
            s=10,
            alpha=0.55,
            color=strategy_color(strategy),
        )
        apply_axes_fraction(ax, y_min=y_min, y_max=y_max)
        fig.suptitle(f"{strategy_display(strategy)} - Fraction", y=0.98, fontsize=7)
        save_pdf_with_shared_copy(fig, Path(f"{OUT_PDF_PER_STRAT_PREFIX}_{strategy}_fraction.pdf"))
        plt.close(fig)


def _binned_median_iqr_by_count(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if data.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    data["bin"] = pd.qcut(data["x"], q=n_bins, labels=False, duplicates="drop")
    data = data.dropna(subset=["bin"]).copy()
    data["bin"] = data["bin"].astype(int)
    grouped = data.groupby("bin", sort=True)
    return (
        grouped["x"].median().to_numpy(dtype=float),
        grouped["y"].median().to_numpy(dtype=float),
        grouped["y"].quantile(0.25).to_numpy(dtype=float),
        grouped["y"].quantile(0.75).to_numpy(dtype=float),
    )


def make_combined_binned_plot(df_plot: pd.DataFrame, x_max_abs: float) -> None:
    data = df_plot.copy()
    data["x_len_raw"] = pd.to_numeric(data["escape_front_length_px"], errors="coerce")
    data["x_frac_norm"] = pd.to_numeric(data["escape_front_fraction"], errors="coerce")
    data["y"] = pd.to_numeric(data["delta_res_count_per_day"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_len_raw", "x_frac_norm", "y"])
    data["x_len_raw"] = data["x_len_raw"].clip(lower=0.0)
    data["x_frac_norm"] = data["x_frac_norm"].clip(0.0, 1.0)
    if data.empty:
        print("[warn] combined binned plot skipped: no valid rows")
        return

    x_len, med_len, q1_len, q3_len = _binned_median_iqr_by_count(
        data["x_len_raw"].to_numpy(float),
        data["y"].to_numpy(float),
        N_BINS_COMBINED,
    )
    x_frac, med_frac, q1_frac, q3_frac = _binned_median_iqr_by_count(
        data["x_frac_norm"].to_numpy(float),
        data["y"].to_numpy(float),
        N_BINS_COMBINED,
    )

    fig, ax = new_combined_figure()
    ax_top = ax.twiny()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax_top.set_facecolor("none")

    color_len = "#222222"
    color_frac = "#1f77b4"
    x_len_k = x_len / 1000.0

    ax.fill_between(x_frac, q1_frac, q3_frac, color=color_frac, alpha=0.18, linewidth=0)
    ax_top.fill_between(x_len_k, q1_len, q3_len, color=color_len, alpha=0.18, linewidth=0)
    ax.plot(x_frac, med_frac, color=color_frac, linewidth=0.9)
    ax_top.plot(x_len_k, med_len, color=color_len, linewidth=0.9)

    ax.set_xlabel("")
    ax.set_ylabel("")
    len_xmax = float(x_len_k[-1]) if x_len_k.size else float(max(x_max_abs / 1000.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax_top.set_xlim(0.0, max(len_xmax, 300.0))
    ax.set_ylim(0.0, 4300.0)

    ax_top.spines["top"].set_position(("axes", 0.0))
    ax_top.spines["bottom"].set_visible(False)
    ax_top.xaxis.set_ticks_position("top")
    ax_top.xaxis.set_label_position("top")
    ax_top.spines["top"].set_visible(False)
    ax.tick_params(axis="x", colors=color_frac, direction="out", pad=2, length=2.5, width=0.8, labelsize=COMBINED_TICK_LABELSIZE)
    ax_top.tick_params(axis="x", colors=color_len, direction="out", pad=2, length=2.5, width=0.8, labelsize=COMBINED_TICK_LABELSIZE)
    ax.tick_params(axis="y", pad=COMBINED_YTICK_PAD, length=2.5, width=0.8, labelsize=COMBINED_TICK_LABELSIZE)
    ax.set_xticks(np.asarray(COMBINED_BOTTOM_XTICKS, dtype=float))
    ax_top.set_xticks(np.asarray(COMBINED_TOP_XTICKS, dtype=float))
    ax.set_yticks([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    fig.text(
        COMBINED_YLABEL_X,
        COMBINED_YLABEL_Y,
        COMBINED_YLABEL_TEXT,
        rotation=90,
        va="center",
        ha="center",
        rotation_mode="anchor",
        fontsize=COMBINED_YLABEL_FONTSIZE,
        color="black",
    )

    label_left = TextArea(
        COMBINED_LENGTH_LABEL_TEXT,
        textprops=dict(color=color_len, fontsize=COMBINED_LABEL_FONTSIZE),
    )
    label_right = TextArea(
        COMBINED_RATIO_LABEL_TEXT,
        textprops=dict(color=color_frac, fontsize=COMBINED_LABEL_FONTSIZE),
    )
    packed = HPacker(children=[label_left, label_right], align="center", pad=0, sep=0)
    ax.add_artist(
        AnnotationBbox(
            packed,
            (0.5, COMBINED_LABEL_Y),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
            annotation_clip=False,
        )
    )
    save_pdf_with_shared_copy(fig, OUT_PDF_COMBINED_BINNED, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def make_length_only_binned_plot(df_plot: pd.DataFrame) -> None:
    data = df_plot.copy()
    data["x_len_raw"] = pd.to_numeric(data["escape_front_length_px"], errors="coerce")
    data["y"] = pd.to_numeric(data["delta_res_count_per_day"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_len_raw", "y"])
    data["x_len_raw"] = data["x_len_raw"].clip(lower=0.0)
    if data.empty:
        print("[warn] length-only combined binned plot skipped: no valid rows")
        return

    x_len, med_len, q1_len, q3_len = _binned_median_iqr_by_count(
        data["x_len_raw"].to_numpy(float),
        data["y"].to_numpy(float),
        N_BINS_COMBINED,
    )
    if not x_len.size:
        print("[warn] length-only combined binned plot skipped: no bins available")
        return

    x_len_mm = x_len * PX_TO_MM

    fig, ax = new_main_figure()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.fill_between(x_len_mm, q1_len, q3_len, color=LINE_COLOR, alpha=0.18, linewidth=0)
    ax.plot(x_len_mm, med_len, color=LINE_COLOR, linewidth=0.9)

    ax.set_xlim(0.0, max(float(x_len_mm[-1]), XTICKS[-1]))
    ax.set_ylim(0.0, 4300.0)
    ax.set_xticks(np.asarray(XTICKS, dtype=float))
    ax.set_yticks(np.asarray(YTICKS, dtype=float))
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.tick_params(axis="x", direction="out", pad=2, length=2.5, width=0.8, labelsize=6)
    ax.tick_params(axis="y", pad=2, length=2.5, width=0.8, labelsize=6)

    save_pdf_with_shared_copy(fig, OUT_PDF_LENGTH_ONLY, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_debug_overlays(frac_gt1_df: pd.DataFrame) -> None:
    if not SAVE_DEBUG_OVERLAYS or frac_gt1_df.empty:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    frac_gt1_df.head(MAX_DEBUG_OVERLAYS).to_csv(DEBUG_DIR / "fraction_gt1_preview.csv", index=False)


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(IN_CSV)

    df = pd.read_csv(IN_CSV)
    if df.empty:
        raise RuntimeError(f"Input CSV is empty: {IN_CSV}")

    if "strategy" not in df.columns and "sample" in df.columns:
        df["strategy"] = df["sample"].map(parse_strategy)

    numeric_cols = [
        "delta_res_count_per_day",
        "escape_front_length_px",
        "escape_front_fraction",
        "boundary_length_px",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["delta_res_count_per_day", "escape_front_length_px", "escape_front_fraction"]
    ).copy()

    x_max_abs = float(max(df["escape_front_length_px"].max(), 1.0))
    y_min = YMIN_FIXED
    y_max = YMAX_FIXED

    neg_df = df[df["delta_res_count_per_day"] < 0].copy().sort_values(
        ["delta_res_count_per_day", "plate", "sample", "t"]
    )
    neg_df.to_csv(OUT_NEG_CSV, index=False)
    print(f"[save] {OUT_NEG_CSV}")

    frac_gt1_df = df[df["escape_front_fraction"] > 1.0].copy().sort_values(
        ["escape_front_fraction", "plate", "sample", "t"],
        ascending=[False, True, True, True],
    )
    frac_gt1_df.to_csv(OUT_FRAC_GT1_CSV, index=False)
    print(f"[save] {OUT_FRAC_GT1_CSV}")

    df_summary = df[df["strategy"].isin(SUMMARY_STRATEGIES)].copy()
    df_plot = df_summary if not df_summary.empty else df

    make_combined_split_plots(df_plot, x_max_abs=x_max_abs, y_min=y_min, y_max=y_max)
    make_combined_binned_plot(df_plot, x_max_abs=x_max_abs)
    make_per_strategy_split_plots(df, x_max_abs=x_max_abs, y_min=y_min, y_max=y_max)
    make_length_only_binned_plot(df_plot)
    save_debug_overlays(frac_gt1_df)


if __name__ == "__main__":
    main()
