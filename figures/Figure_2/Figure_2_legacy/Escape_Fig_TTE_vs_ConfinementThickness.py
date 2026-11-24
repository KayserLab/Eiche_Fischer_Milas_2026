"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script relates time to escape (TTE) to confinement thickness by strategy.

Detailed
--------
Inputs:
- Confinement-thickness summaries from
  `analysis/figure_2/Escape_ConfinementThickness.py`.
- Escape timing information and shared replicate filter rules.

Outputs:
- A binned Figure 2 summary plot connecting confinement thickness to TTE.

How it works:
- Loads confinement-thickness and escape-timing data, then applies the shared
  replicate exclusion/truncation rules.
- Builds common confinement-thickness bins so strategies are compared on the
  same x-axis partition.
- Summarizes TTE within each bin and plots the resulting medians/trends by
  strategy.

Pipeline role:
- Figure 2 plotting script linking local confinement to escape timing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from shared import figtools as ft
from scipy.stats import pearsonr

from shared import experiment_filters


OUT_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset\Figures\Escape_Quantification")
OUT_DIR_SHARED = ft.paper_figure_data_dir(__file__)
CONF_CSV = OUT_DIR / "Escape_ConfinementThickness_all_replicates.csv"
ESC_SUMMARY_CSV = OUT_DIR / "escape_front_batch_summary_all_res_within_sens.csv"

DEBUG_VISUALIZE_C5 = False
DEBUG_SAMPLE = "C5_AT50-C5"
PLOT_STRATEGIES = ["CT", "AT50", "AT100"]
USE_THIN_FIGURE = True

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis
# length. Margins are extra space for tick labels, axis labels, and legends.
FIGURE_AXES_WIDTH_MM = 40.0
FIGURE_AXES_HEIGHT_MM = 40.0
FIGURE_MARGINS_MM = (14.48, 3.86, 20.32, 9.65) if USE_THIN_FIGURE else (36.58, 9.75, 20.32, 9.65)
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
SHOW_FIGURE = DEBUG_VISUALIZE_C5
YLABEL_X = 0.12
YLABEL_Y = 0.584
YLABEL_TEXT = "Time until Escape [days]"
BIN_COUNT = 4

SAMPLE_RE = re.compile(r"^[A-H]\d+_([^-]+)-[A-H]\d+$", re.IGNORECASE)

from shared.strategy_colors import get_color, get_display_name

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    },
)


def strategy_display(s: str) -> str:
    return get_display_name(s)


def strategy_color(s: str) -> str:
    return get_color(s, "#000000")


def parse_strategy(sample: str) -> str:
    m = SAMPLE_RE.match(str(sample))
    return m.group(1) if m else "UNKNOWN"


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    rules = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    keep = np.ones(len(df), dtype=bool)
    rep = df["sample"].astype(str).to_numpy()
    day = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=float)
    for i, s in enumerate(rep):
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


def build_common_quantile_bins(values: pd.Series, n_bins: int) -> np.ndarray:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite confinement_thickness values available for binning.")
    edges = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("Could not construct at least one finite bin for confinement_thickness.")
    return edges


def summarize_binned_medians(df: pd.DataFrame, bin_edges: np.ndarray) -> pd.DataFrame:
    work = df.copy()
    work["bin"] = pd.cut(
        pd.to_numeric(work["confinement_thickness"], errors="coerce"),
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    rows: list[dict] = []
    for strat in PLOT_STRATEGIES:
        g = work[work["strategy"] == strat].copy()
        if g.empty:
            continue
        for bin_idx, (bin_label, sub) in enumerate(g.groupby("bin", observed=False), start=1):
            x = pd.to_numeric(sub["confinement_thickness"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub["remaining_time_until_escape"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size == 0:
                continue
            rows.append(
                {
                    "strategy": strat,
                    "bin_index": bin_idx,
                    "bin_left": float(bin_edges[bin_idx - 1]),
                    "bin_right": float(bin_edges[bin_idx]),
                    "bin_label": str(bin_label),
                    "n": int(x.size),
                    "x_median": float(np.median(x)),
                    "x_q25": float(np.quantile(x, 0.25)),
                    "x_q75": float(np.quantile(x, 0.75)),
                    "y_median": float(np.median(y)),
                    "y_q25": float(np.quantile(y, 0.25)),
                    "y_q75": float(np.quantile(y, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if not CONF_CSV.exists():
        raise FileNotFoundError(CONF_CSV)
    if not ESC_SUMMARY_CSV.exists():
        raise FileNotFoundError(ESC_SUMMARY_CSV)

    conf = pd.read_csv(CONF_CSV).rename(columns={"replicate": "sample"})
    esc = pd.read_csv(ESC_SUMMARY_CSV)
    esc = esc[["plate", "sample", "first_escape_day", "escape_detected"]].copy()
    esc["first_escape_day"] = pd.to_numeric(esc["first_escape_day"], errors="coerce")
    esc = esc[esc["escape_detected"] == True].copy()

    df = conf.merge(esc, on=["plate", "sample"], how="inner")
    df["strategy"] = df["sample"].map(parse_strategy)
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["confinement_thickness"] = pd.to_numeric(df["confinement_thickness"], errors="coerce")
    df["remaining_time_until_escape"] = df["first_escape_day"] - df["day"]
    df = df[df["remaining_time_until_escape"] >= 0].copy()
    df = apply_filters(df)
    if DEBUG_VISUALIZE_C5:
        df = df[df["sample"] == DEBUG_SAMPLE].copy()
    OUT_DIR_SHARED.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "Escape_Fig_TTE_vs_ConfinementThickness_data.csv"
    df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} rows={len(df)}")
    if df.empty:
        return

    stats_by_scale: Dict[str, List[dict]] = {}
    for xscale in ("linear", "log"):
        fig, ax = ft.new_figure(
            width=FIGURE_AXES_WIDTH_MM,
            height_mm=FIGURE_AXES_HEIGHT_MM,
            margins=FIGURE_MARGINS_MM,
            dpi=DPI,
            apply_style=False,
        )
        stats_rows: List[dict] = []
        for strat in PLOT_STRATEGIES:
            g = df[df["strategy"] == strat].copy()
            if g.empty:
                continue
            x = g["confinement_thickness"].to_numpy(dtype=float)
            y = g["remaining_time_until_escape"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if xscale == "log":
                m &= x > 0.0
            x = x[m]
            y = y[m]
            if x.size == 0:
                continue
            c = strategy_color(strat)
            ax.scatter(x, y, s=8, alpha=0.65, color=c, label=strategy_display(strat))
            if x.size >= 3 and np.nanmax(x) > np.nanmin(x):
                r, p = pearsonr(x, y)
                stats_rows.append(
                    {
                        "xscale": xscale,
                        "strategy": strat,
                        "pearson_r": float(r),
                        "pearson_p": float(p),
                        "n": int(x.size),
                    }
                )

        ax.set_xlabel("Confinement Thickness")
        ax.set_ylabel("")
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", left=True, labelleft=True, pad=1)
        ax.set_xscale(xscale)
        if xscale == "log":
            ax.set_xticks([1, 10, 100, 1000, 10000])
            ax.set_xticklabels(["1", "10", "100", "1000", "10000"])
        ax.legend(frameon=False, fontsize=6)
        fig.text(
            YLABEL_X,
            YLABEL_Y,
            YLABEL_TEXT,
            rotation=90,
            va="center",
            ha="center",
            rotation_mode="anchor",
            fontsize=TEXT_SIZE_PT,
            color="black",
        )

        out_pdf = OUT_DIR_SHARED / f"Escape_Fig_TTE_vs_ConfinementThickness_x{xscale}.pdf"
        out_pdf_shared = out_pdf
        ft.savefig(fig, out_pdf)
        ft.savefig(fig, out_pdf_shared)
        print(f"[save] {out_pdf}")
        print(f"[save] {out_pdf_shared}")
        if xscale == "log":
            out_pdf_legacy = OUT_DIR_SHARED / "Escape_Fig_TTE_vs_ConfinementThickness.pdf"
            ft.savefig(fig, out_pdf_legacy)
            print(f"[save] {out_pdf_legacy}")
        if SHOW_FIGURE and plt.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)
        stats_by_scale[xscale] = stats_rows

    bin_edges = build_common_quantile_bins(df["confinement_thickness"], n_bins=BIN_COUNT)
    binned_df = summarize_binned_medians(df, bin_edges=bin_edges)
    out_binned_csv = OUT_DIR / f"Escape_Fig_TTE_vs_ConfinementThickness_binned_medians_{BIN_COUNT}bins.csv"
    binned_df.to_csv(out_binned_csv, index=False)
    print(f"[save] {out_binned_csv} rows={len(binned_df)}")

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    for strat in PLOT_STRATEGIES:
        g = binned_df[binned_df["strategy"] == strat].copy()
        if g.empty:
            continue
        x = g["x_median"].to_numpy(dtype=float)
        y = g["y_median"].to_numpy(dtype=float)
        xerr = np.vstack([
            x - g["x_q25"].to_numpy(dtype=float),
            g["x_q75"].to_numpy(dtype=float) - x,
        ])
        yerr = np.vstack([
            y - g["y_q25"].to_numpy(dtype=float),
            g["y_q75"].to_numpy(dtype=float) - y,
        ])
        c = strategy_color(strat)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o-",
            color=c,
            ecolor=c,
            elinewidth=0.9,
            linewidth=1.0,
            markersize=3.5,
            capsize=0,
            label=strategy_display(strat),
        )

    ax.set_xlabel("Confinement Thickness")
    ax.set_ylabel("")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", left=True, labelleft=True, pad=1)
    ax.legend(frameon=False, fontsize=6)
    fig.text(
        YLABEL_X,
        YLABEL_Y,
        YLABEL_TEXT,
        rotation=90,
        va="center",
        ha="center",
        rotation_mode="anchor",
        fontsize=TEXT_SIZE_PT,
        color="black",
    )

    out_binned_pdf = OUT_DIR_SHARED / f"Escape_Fig_TTE_vs_ConfinementThickness_binned_median_{BIN_COUNT}bins.pdf"
    out_binned_pdf_shared = out_binned_pdf
    ft.savefig(fig, out_binned_pdf)
    ft.savefig(fig, out_binned_pdf_shared)
    print(f"[save] {out_binned_pdf}")
    print(f"[save] {out_binned_pdf_shared}")
    plt.close(fig)

    stats_df = pd.DataFrame(stats_by_scale.get("log", []))
    out_stats = OUT_DIR / "Escape_Fig_TTE_vs_ConfinementThickness_pearson.csv"
    stats_df.to_csv(out_stats, index=False)
    print(f"[save] {out_stats}")


if __name__ == "__main__":
    main()
