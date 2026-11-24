"""Compare experimental and DPD agent decisions in one Figure 4 policy portrait.

Reads the decision CSVs prepared by 02_load_dpd_decisions.py and
02d_load_experimental_decisions.py in policy_distillation_clean. Each point
shows P(treat) versus current normalized burden; colour encodes the last
treatment (previous_action_t), not the action chosen at this decision.
Experiments use circles and DPD simulations use crosses.

Run directly in PyCharm, or pass --experimental-csv, --dpd-csv and
--output-dir to use alternative tables or save locations. Add --show to
display the plot. Exports an editable PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.data_paths import INTERMEDIATE_ANALYSIS_DATA, figure_plot_dir


DATA_DIR = INTERMEDIATE_ANALYSIS_DATA / 'policy_decisions'
EXPERIMENTAL_CSV = DATA_DIR / "experimental_decisions.csv"
DPD_CSV = DATA_DIR / "dpd_decisions.csv"
OUTPUT_DIR = figure_plot_dir("Figure_4", create=False)
OUTPUT_STEM = "Figure_4_Experimental_DPD_Policy_Portrait"

# Match histories ending in 0/1 in Agent_Behaviour_Analysis/
# replot_rl_policy_behavior.py and rl_policy_behavior_analysis.py.
EXPERIMENT_POINT_MARKER = "o"
SIMULATION_POINT_MARKER = "x"
TREATMENT_COLORS = {0: "#4c9f70", 1: "#d97b29"}
FIGURE_AXES_WIDTH_MM = 50.0
FIGURE_AXES_HEIGHT_MM = 20.0
FIGURE_MARGINS_MM = (13.0, 2.0, 11.0, 4.0)
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.0
POINT_SIZE = 10.0
POINT_ALPHA = 0.6
DPI = 300


def load_decisions(path: Path) -> pd.DataFrame:
    """Require usable portrait coordinates and an unambiguous last treatment."""
    if not path.is_file():
        raise FileNotFoundError(f"Decision CSV not found: {path}")
    table = pd.read_csv(path)
    columns = ["burden_t", "p_treat", "previous_action_t"]
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    values = table[columns].apply(pd.to_numeric, errors="coerce")
    if values.empty:
        raise ValueError(f"{path}: no decisions to plot")
    valid = (
        np.isfinite(values).all(axis=1)
        & values["burden_t"].ge(0)
        & values["p_treat"].between(0, 1)
        & values["previous_action_t"].isin([0, 1])
    )
    if not valid.all():
        raise ValueError(f"{path}: {int((~valid).sum())} invalid decision rows")
    return values


def plot_decisions(
    experimental: pd.DataFrame, dpd: pd.DataFrame
) -> tuple[plt.Figure, plt.Axes]:
    """Plot every supplied decision without binning or subsampling."""
    with ft.nature_style(
        text_pt=TEXT_SIZE_PT, tick_pt=TICK_SIZE_PT, legend_pt=LEGEND_SIZE_PT
    ):
        fig, ax = ft.new_figure(
            width=FIGURE_AXES_WIDTH_MM,
            height_mm=FIGURE_AXES_HEIGHT_MM,
            margins=FIGURE_MARGINS_MM,
            dpi=DPI,
            apply_style=False,
        )
        # Draw experiments last so the smaller experimental dataset stays visible.
        for table, marker in (
            (dpd, SIMULATION_POINT_MARKER),
            (experimental, EXPERIMENT_POINT_MARKER),
        ):
            for previous_action, color in TREATMENT_COLORS.items():
                rows = table.loc[table["previous_action_t"] == previous_action]
                ax.scatter(
                    rows["burden_t"], rows["p_treat"],
                    marker=marker, s=POINT_SIZE, color=color,
                    alpha=POINT_ALPHA,
                    edgecolors="white" if marker == "o" else None,
                    linewidths=0.25 if marker == "o" else 0.7,
                )

        ax.axhline(0.5, color="#898781", linewidth=0.6, zorder=0)
        ax.set_xlabel(r"Normalized count, $\eta_{\mathrm{tot}}$")
        ax.set_ylabel("Treatment Probability,\n" + r"$P_{\mathrm{Treat}}$", linespacing=0.95)
        ax.set_xlim(0, 3)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Highlight the shared initial decision and the isolated day-3 group.
        ax.add_patch(Ellipse(
            (1.0, 0.11898816), width=0.16, height=0.10,
            fill=False, edgecolor="#52514e", linewidth=0.7, zorder=4,
        ))
        ax.text(
            0.85, 0.11898816, "day 1", fontsize=TICK_SIZE_PT,
            ha="right", va="center", color="#52514e",
        )
        ax.add_patch(Ellipse(
            (1.34, 0.71), width=0.49, height=0.17, angle=56,
            fill=False, edgecolor="#52514e", linewidth=0.7, zorder=4,
        ))
        ax.annotate(
            "day 3", xy=(1.20, 0.51), xytext=(1.02, 0.39),
            fontsize=TICK_SIZE_PT, ha="right", va="center", color="#52514e",
            arrowprops={"arrowstyle": "-", "color": "#52514e", "linewidth": 0.5},
        )
        handles = [
            Line2D([], [], linestyle="None", marker=marker, color="#52514e",
                   markersize=3, label=label)
            for marker, label in (
                (EXPERIMENT_POINT_MARKER, "Experiment"),
                (SIMULATION_POINT_MARKER, "Simulation"),
            )
        ] + [
            Line2D([], [], linestyle="None", marker="o", color=color,
                   markersize=3, label=f"Previous Treat {'OFF' if action == 0 else 'ON'}")
            for action, color in TREATMENT_COLORS.items()
        ]
        ax.legend(
            handles=handles[2:] + handles[:2], frameon=False, ncol=1,
            loc="upper left",
            bbox_to_anchor=(-0.03, 1.05),
            borderaxespad=0.4, handletextpad=0.3,
        )
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental-csv", type=Path, default=EXPERIMENTAL_CSV)
    parser.add_argument("--dpd-csv", type=Path, default=DPD_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    experimental = load_decisions(args.experimental_csv)
    dpd = load_decisions(args.dpd_csv)
    fig, _ = plot_decisions(experimental, dpd)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{OUTPUT_STEM}.pdf"
    ft.savefig(fig, path, dpi=DPI)
    print(f"[saved] {path}")
    print(f"[points] Experiment: {len(experimental)}, DPD: {len(dpd)}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
