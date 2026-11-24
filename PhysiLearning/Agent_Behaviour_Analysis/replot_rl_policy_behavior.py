"""
Replot grouped RL policy-behavior outputs using the repo's central figure style.

This script reads the grouped CSV outputs produced by
`PhysiLearning/Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py` from
the current `rl_policy_behavior` directory and regenerates clean publication
plots using the repository's `shared/figtools.py` standards.

The grouped CSVs now summarize deterministic chosen actions over binned values
of the colleague-style growth score:
    growth = base + peak_day * peak_addition

The plotted y-axis is therefore a fraction of evaluated decisions choosing
treatment ON, not a policy probability. The source CSVs still carry diagnostic
probability columns, but this replotter intentionally uses the hard-action
metric to match the revised analysis question.

It does not rerun policy probing. It only reuses:
    - experimental_agent_policy_behavior_grouped_curves.csv
    - bootstrap_agents_policy_behavior_grouped_curves.csv

Outputs are written as PDF and PNG into a sibling `plots` directory next to
the current `rl_policy_behavior` folder. The bootstrap policy-behavior panel is
also exported to the canonical external `figure_plots/Figure_4` directory
configured through `EICHE_FISCHER_DATA_ROOT`.

The x-axis is inferred from the available binned growth values. Earlier
versions fixed the axis to [0, 10], but the colleague-style growth definition
can exceed 10 with the current default grid.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.data_paths import figure_plot_dir  # noqa: E402


def infer_paths() -> tuple[Path, Path, Path, Path, Path]:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    analysis_output_dir = this_file.parent / "outputs"
    input_dir = analysis_output_dir / "rl_policy_behavior"
    output_dir = analysis_output_dir / "plots"
    figure4_dir = repo_root / "figures" / "Figure_4"
    paper_figure4_dir = figure_plot_dir("Figure_4", create=False)
    return repo_root, input_dir, output_dir, figure4_dir, paper_figure4_dir


REPO_ROOT, INPUT_DIR, OUTPUT_DIR, FIGURE4_DIR, PAPER_FIGURE4_DIR = infer_paths()

import sys

sys.path.insert(0, str(REPO_ROOT))
from shared import figtools as ft


FIGURE_AXES_WIDTH_MM = 25.0
FIGURE_AXES_HEIGHT_MM = 20.0
FIGURE_MARGINS_MM = (10.0, 24.0, 8.0, 4.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.0
DPI = 300
GROWTH_XLIM = (0.0, 20.0)
GROWTH_XTICKS = [0, 5, 10, 15, 20]

GROUP_ORDER = [
    "[0,0,0,0]",
    "[0,0,0,1]",
    "[0,0,1,1]",
    "[0,1,1,1]",
    "[1,1,1,1]",
]

FIGURE4_GROUP_ORDER = GROUP_ORDER

STYLE_MAP = {
    "[0,0,0,0]": {"color": "#9b2226", "linestyle": "-", "linewidth": 2.6},
    "[0,0,0,1]": {"color": "#4c9f70", "linestyle": "--", "linewidth": 2.0},
    "[0,0,1,1]": {"color": "#d97b29", "linestyle": "--", "linewidth": 2.0},
    "[0,1,1,1]": {"color": "#6a4c93", "linestyle": "--", "linewidth": 2.0},
    "[1,1,1,1]": {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.6},
}

FIGURE4_STYLE_MAP = {
    group: {
        **style,
        "linewidth": 1.0 if group not in {"[0,0,0,0]", "[1,1,1,1]"} else 1.1,
    }
    for group, style in STYLE_MAP.items()
}
FIGURE4_LABEL_MAP = {
    "[0,0,0,0]": "[0,0,0,0]",
    "[0,0,0,1]": "[0,0,0,1]",
    "[0,0,1,1]": "[0,0,1,1]",
    "[0,1,1,1]": "[0,1,1,1]",
    "[1,1,1,1]": "[1,1,1,1]",
}


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    experimental_path = INPUT_DIR / "experimental_agent_policy_behavior_grouped_curves.csv"
    bootstrap_path = INPUT_DIR / "bootstrap_agents_policy_behavior_grouped_curves.csv"

    if not experimental_path.exists():
        raise FileNotFoundError(f"Missing experimental grouped CSV: {experimental_path}")
    if not bootstrap_path.exists():
        raise FileNotFoundError(f"Missing bootstrap grouped CSV: {bootstrap_path}")

    experimental = pd.read_csv(experimental_path)
    bootstrap = pd.read_csv(bootstrap_path)
    return experimental, bootstrap


def save_pdf_png(fig: plt.Figure, stem: str, output_dirs: tuple[Path, ...] = (OUTPUT_DIR,)) -> None:
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        ft.savefig(fig, output_dir / f"{stem}.pdf", dpi=DPI)
        ft.savefig(fig, output_dir / f"{stem}.png", dpi=DPI)


def growth_xlim(*frames: pd.DataFrame) -> tuple[float, float]:
    growth_values = pd.concat(
        [frame["growth"] for frame in frames if not frame.empty],
        ignore_index=True,
    )
    growth_min = float(growth_values.min())
    growth_max = float(growth_values.max())
    if growth_min == growth_max:
        return growth_min - 0.5, growth_max + 0.5
    padding = 0.03 * (growth_max - growth_min)
    return growth_min - padding, growth_max + padding


def base_axes(title: str, *frames: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ft.set_axes_size_mm(ax, width_mm=88.0, height_mm=58.0, left_mm=16.0, right_mm=8.0, bottom_mm=14.0, top_mm=6.0)
    ax.set_title(title)
    ax.set_xlabel("Growth score")
    ax.set_ylabel("Fraction choosing treatment ON")
    ax.set_xlim(*GROWTH_XLIM)
    ax.set_xticks(GROWTH_XTICKS)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def figure4_axes(bootstrap: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    ax.set_xlabel("History growth score")
    ax.set_ylabel("Treatment-on fraction")
    ax.set_xlim(*GROWTH_XLIM)
    ax.set_xticks(GROWTH_XTICKS)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    return fig, ax


def plot_experimental(experimental: pd.DataFrame) -> None:
    fig, ax = base_axes("Experimental Agent Chosen Actions", experimental)
    for group in GROUP_ORDER:
        subset = experimental.loc[experimental["action_history_group"] == group].sort_values("growth")
        if subset.empty:
            continue
        ax.plot(
            subset["growth"],
            subset["fraction_chosen_treatment_on"],
            label=group,
            **STYLE_MAP[group],
        )
    ax.legend(frameon=False, loc="lower right")
    save_pdf_png(fig, "experimental_policy_behavior_replot")
    plt.close(fig)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    fig, ax = figure4_axes(bootstrap)
    for group in FIGURE4_GROUP_ORDER:
        subset = bootstrap.loc[bootstrap["action_history_group"] == group].sort_values("growth")
        if subset.empty:
            continue
        style = FIGURE4_STYLE_MAP[group]
        ax.plot(
            subset["growth"],
            subset["fraction_chosen_treatment_on_mean"],
            label=FIGURE4_LABEL_MAP[group],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax.fill_between(
            subset["growth"],
            subset["fraction_chosen_treatment_on_mean"] - subset["fraction_chosen_treatment_on_sd"],
            subset["fraction_chosen_treatment_on_mean"] + subset["fraction_chosen_treatment_on_sd"],
            color=style["color"],
            alpha=0.16,
            linewidth=0.0,
        )
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=LEGEND_SIZE_PT,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.0,
        labelspacing=0.15,
    )
    save_pdf_png(
        fig,
        "bootstrap_policy_behavior_replot",
        output_dirs=(OUTPUT_DIR, FIGURE4_DIR, PAPER_FIGURE4_DIR),
    )
    plt.close(fig)


def plot_comparison(experimental: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    fig, ax = base_axes("Experimental vs Ensemble Chosen Actions", experimental, bootstrap)
    for group in GROUP_ORDER:
        experimental_subset = experimental.loc[
            experimental["action_history_group"] == group
        ].sort_values("growth")
        bootstrap_subset = bootstrap.loc[
            bootstrap["action_history_group"] == group
        ].sort_values("growth")
        if experimental_subset.empty or bootstrap_subset.empty:
            continue
        color = STYLE_MAP[group]["color"]
        ax.plot(
            experimental_subset["growth"],
            experimental_subset["fraction_chosen_treatment_on"],
            color=color,
            linestyle="-",
            linewidth=2.2,
            label=f"{group} experimental",
        )
        ax.plot(
            bootstrap_subset["growth"],
            bootstrap_subset["fraction_chosen_treatment_on_mean"],
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=f"{group} ensemble mean",
        )
        ax.fill_between(
            bootstrap_subset["growth"],
            bootstrap_subset["fraction_chosen_treatment_on_mean"] - bootstrap_subset["fraction_chosen_treatment_on_sd"],
            bootstrap_subset["fraction_chosen_treatment_on_mean"] + bootstrap_subset["fraction_chosen_treatment_on_sd"],
            color=color,
            alpha=0.15,
            linewidth=0.0,
        )
    ax.legend(frameon=False, loc="lower right", ncol=1)
    save_pdf_png(fig, "experimental_vs_bootstrap_policy_behavior_replot")
    plt.close(fig)


def main() -> None:
    experimental, bootstrap = read_inputs()
    with ft.nature_style(
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
    ):
        plot_experimental(experimental)
        plot_bootstrap(bootstrap)
        plot_comparison(experimental, bootstrap)
    print(f"Replotted figures written to: {OUTPUT_DIR}")
    print(f"Figure 4 bootstrap policy panel written to: {FIGURE4_DIR}")
    print(f"Figure 4 paper-data copy written to: {PAPER_FIGURE4_DIR}")


if __name__ == "__main__":
    main()
