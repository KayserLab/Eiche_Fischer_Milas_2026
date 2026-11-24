"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script replots the strategy-wise distribution of additional resistant
cells attributable to escape.

Detailed
--------
Inputs:
- The analysis CSVs written by
  `analysis/legacy/Escape_Fig_Additional_cells_bcs_of_Escape.py`.

Outputs:
- A Figure 2 boxplot-style panel with significance annotations.

How it works:
- Reads the precomputed escape-associated resistant-cell summaries.
- Maps raw sample names to canonical strategy labels and colors.
- Builds the manuscript-style boxplot and adds pairwise significance brackets
  using the precomputed or re-evaluated statistics.

Pipeline role:
- Figure 2 plotting endpoint for the “additional cells because of escape”
  branch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.strategy_colors import get_color, get_display_name


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = DATA_ROOT / r"Figures\Escape_Quantification"
IN_SUMMARY_CSV = OUT_DIR / "Escape_Fig_Additional_cells_bcs_of_Escape_summary.csv"
OUT_DIR_SHARED = ft.paper_figure_data_dir(__file__)
OUT_PDF = OUT_DIR_SHARED / "Escape_Fig_Additional_cells_bcs_of_Escape.pdf"
OUT_PDF_SIG = OUT_DIR_SHARED / "Escape_Fig_Additional_cells_bcs_of_Escape_with_significance.pdf"
OUT_PDF_SHARED = OUT_PDF
OUT_PDF_SIG_SHARED = OUT_PDF_SIG
OUT_TESTS_CSV = OUT_DIR / "Escape_Fig_Additional_cells_bcs_of_Escape_pairwise_tests.csv"

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis
# length. Margins are extra space for tick labels, axis labels, and brackets.
FIGURE_AXES_WIDTH_MM = 20.0
FIGURE_AXES_HEIGHT_MM = 40.0
FIGURE_MARGINS_MM = (11.66, 2.59, 20.12, 2.24)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
YLABEL_X = 0.11
YLABEL_Y = 0.66

PLOT_STRATEGIES = ["CT", "AT50", "AT100"]
SAMPLE_RE = re.compile(r"^[A-H]\d+_([^-]+)-[A-H]\d+$", re.IGNORECASE)

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


def parse_strategy_from_sample(sample: str) -> str:
    m = SAMPLE_RE.match(str(sample))
    return m.group(1) if m else "UNKNOWN"


def p_value_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def build_boxplot_figure(order: list[str], vals: list[np.ndarray]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    bp = ax.boxplot(
        vals,
        tick_labels=[strategy_display(s) for s in order],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(edgecolor="black", linewidth=0.8),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.2),
    )
    for i, s in enumerate(order):
        bp["boxes"][i].set_facecolor(strategy_color(s))

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", left=True, labelleft=True, pad=1)
    ax.tick_params(axis="x", pad=1)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(40)
        lbl.set_horizontalalignment("right")
    fig.text(
        YLABEL_X,
        YLABEL_Y,
        "Fraction of Growth due to Escape",
        rotation=90,
        va="center",
        ha="center",
        rotation_mode="anchor",
        fontsize=TEXT_SIZE_PT,
        color="black",
    )
    return fig, ax


def add_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h - 0.03, text, ha="center", va="bottom", fontsize=7, color="black")


def main() -> None:
    if not IN_SUMMARY_CSV.exists():
        raise FileNotFoundError(IN_SUMMARY_CSV)

    df = pd.read_csv(IN_SUMMARY_CSV)
    if "strategy" not in df.columns and "sample" in df.columns:
        df["strategy"] = df["sample"].map(parse_strategy_from_sample)

    if "strategy" not in df.columns or "escape_growth_fraction_total" not in df.columns:
        raise ValueError("Missing required columns: strategy and/or escape_growth_fraction_total")

    plot_df = df[df["strategy"].isin(PLOT_STRATEGIES)].copy()
    plot_df["escape_growth_fraction_total"] = pd.to_numeric(
        plot_df["escape_growth_fraction_total"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["escape_growth_fraction_total"])
    print(f"[info] rows in summary: {len(df)}")
    print(f"[info] rows used in plot: {len(plot_df)}")

    if plot_df.empty:
        print("[warn] no rows to plot after filtering.")
        return

    order = [s for s in PLOT_STRATEGIES if s in set(plot_df["strategy"])]
    vals = [
        plot_df.loc[plot_df["strategy"] == s, "escape_growth_fraction_total"].to_numpy(dtype=float)
        for s in order
    ]

    # Print replicate-level values per strategy.
    for s in order:
        sub = plot_df.loc[plot_df["strategy"] == s, ["plate", "sample", "escape_growth_fraction_total"]].copy()
        sub = sub.sort_values(["plate", "sample"])
        print(f"[values] {strategy_display(s)} n={len(sub)}")
        for _, rr in sub.iterrows():
            print(
                f"  {rr['plate']}/{rr['sample']}: "
                f"{float(rr['escape_growth_fraction_total']):.6f}"
            )

    fig, ax = build_boxplot_figure(order, vals)
    OUT_DIR_SHARED.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, OUT_PDF)
    ft.savefig(fig, OUT_PDF_SHARED)
    print(f"[save] {OUT_PDF}")
    print(f"[save] {OUT_PDF_SHARED}")
    plt.close(fig)

    # Pairwise tests (all strategy pairs), two-sided Mann-Whitney U.
    test_rows = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a_name = order[i]
            b_name = order[j]
            a = vals[i]
            b = vals[j]
            if len(a) == 0 or len(b) == 0:
                p = np.nan
                u_stat = np.nan
            else:
                res = mannwhitneyu(a, b, alternative="two-sided")
                p = float(res.pvalue)
                u_stat = float(res.statistic)
            test_rows.append(
                {
                    "group_a": a_name,
                    "group_b": b_name,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "median_a": float(np.median(a)) if len(a) else np.nan,
                    "median_b": float(np.median(b)) if len(b) else np.nan,
                    "u_statistic": u_stat,
                    "p_value_two_sided": p,
                }
            )
            print(
                f"[test] {strategy_display(a_name)} vs {strategy_display(b_name)} | "
                f"n=({len(a)},{len(b)}) | U={u_stat:.3f} | p={p:.6g}"
            )

    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(OUT_TESTS_CSV, index=False)
    print(f"[save] {OUT_TESTS_CSV}")

    pair_lookup = {(row["group_a"], row["group_b"]): row for row in test_rows}
    required_pairs = [("CT", "AT50"), ("CT", "AT100")]
    missing_pairs = [pair for pair in required_pairs if pair not in pair_lookup]
    if missing_pairs:
        print(f"[warn] missing required significance pairs for annotated plot: {missing_pairs}")
        return

    fig_sig, ax_sig = build_boxplot_figure(order, vals)
    sig_ymin = -0.05
    sig_ymax = 1.20
    bracket_h = 0.02
    y1 = 1.00
    y2 = 1.08
    pos_lookup = {name: idx + 1 for idx, name in enumerate(order)}
    first_row = pair_lookup[("CT", "AT50")]
    second_row = pair_lookup[("CT", "AT100")]

    add_significance_bracket(
        ax_sig,
        float(pos_lookup["CT"]),
        float(pos_lookup["AT50"]),
        y1,
        bracket_h,
        p_value_to_stars(float(first_row["p_value_two_sided"])),
    )
    add_significance_bracket(
        ax_sig,
        float(pos_lookup["CT"]),
        float(pos_lookup["AT100"]),
        y2,
        bracket_h,
        p_value_to_stars(float(second_row["p_value_two_sided"])),
    )
    ax_sig.set_ylim(sig_ymin, sig_ymax)
    ft.savefig(fig_sig, OUT_PDF_SIG)
    ft.savefig(fig_sig, OUT_PDF_SIG_SHARED)
    plt.close(fig_sig)
    print(
        "[save] "
        f"{OUT_PDF_SIG} | CT-AT50={p_value_to_stars(float(first_row['p_value_two_sided']))}, "
        f"CT-AT100={p_value_to_stars(float(second_row['p_value_two_sided']))}"
    )
    print(f"[save] {OUT_PDF_SIG_SHARED}")


if __name__ == "__main__":
    main()
