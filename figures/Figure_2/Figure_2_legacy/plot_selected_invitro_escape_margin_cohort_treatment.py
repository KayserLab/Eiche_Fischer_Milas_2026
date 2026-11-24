"""Plot selected in-vitro escape-margin trajectories with treatment backgrounds.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

This is the treatment-schedule companion to
``plot_selected_invitro_escape_margin_three_replicates.py``.  It deliberately
uses only the experimental trajectories: no DPD traces or DPD treatment hatch
are included.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(THIS_FILE.parent)]

from shared import figtools as ft
import plot_selected_invitro_escape_margin_three_replicates as selected
from analysis.legacy import plot_selected_escape_progression_analysis as progression
from shared.strategy_colors import get_color


OUTPUT = (
    ft.PAPER_FIGURE_ROOT / "Figure_2_Data"
    / "CT_AT50_AT100_selected_invitro_escape_margin_until_truncation"
    "_with_cohort_and_treatment_background_invitro_only.pdf"
)


def draw_panel(ax, strategy, plate, sample, margins, burdens, background_margins, audit) -> None:
    """Draw one in-vitro strategy with its cohort and treatment shading."""
    color = get_color(strategy)
    for _, cohort in background_margins[background_margins.strategy.eq(strategy)].groupby(
        ["plate", "sample"], sort=False
    ):
        escaped = cohort[cohort.phase.eq("escaped")].sort_values("day")
        if not escaped.empty:
            ax.plot(escaped.day, escaped.normalized_escape_margin, color=color,
                    alpha=0.14, lw=0.75, zorder=1)

    margin = margins[(margins.strategy == strategy) & (margins.plate == plate) & (margins["sample"] == sample)].sort_values("day")
    burden = burdens[(burdens.strategy == strategy) & (burdens.plate == plate) & (burdens["sample"] == sample)].sort_values("day")
    stop_day = float(audit.loc[audit["sample"].eq(sample), "stop_day"].iloc[0])
    treatment_days = progression.read_invitro_treatment_days(progression.EXPERIMENT_ROOT, plate, sample)
    progression.shade_treatment(ax, treatment_days, stop_day, ymin=0.0, ymax=1.0)

    ax.plot(margin.day, margin.normalized_escape_margin, color=color, lw=1.6, zorder=3)
    ax.plot(burden.day, burden.normalized_resistant, color="#DAA520", lw=1.7, zorder=4)
    ax.plot(burden.day, burden.normalized_resistant, color=color, ls=(0, (5, 4)), lw=1.2, zorder=5)
    escape_day = float(audit.loc[audit["sample"].eq(sample), "escape_day"].iloc[0])
    if escape_day == escape_day:
        ax.axvline(escape_day, color=color, ls=":", lw=0.9, zorder=2)
    ax.axhline(0, color="#111111", lw=0.65, zorder=5)
    ax.set(xlim=selected.X_LIMITS, ylim=selected.Y_LIMITS, ylabel="Escape Margin")
    ax.set_title(strategy, loc="left", pad=2, color=color, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    margins, burdens, background_margins, _background_burdens, audit = selected.load_selected_data()
    fig, axes = ft.new_subplots(
        nrows=3, ncols=1, ax_width_mm=80, ax_height_mm=26,
        margins=(14, 4, 12, 4), hspace_mm=5, sharex=True, sharey=True,
        apply_style=False,
    )
    for ax, (strategy, plate, sample) in zip(axes, selected.SELECTED_REPLICATES):
        draw_panel(ax, strategy, plate, sample, margins, burdens, background_margins, audit)
    axes[-1].set_xlabel("Time [days]")
    fig.legend(
        handles=[
            Line2D([], [], color="#333333", lw=1.0, label="Escape margin"),
            Line2D([], [], color="#333333", lw=1.0, ls="--", label="Total resistant cells"),
            Line2D([], [], color="#333333", lw=0.9, ls=":", label="Time of escape"),
        ],
        loc="lower center", bbox_to_anchor=(0.56, 0.015), frameon=False,
        fontsize=5.2, ncol=3,
    )
    ft.savefig(fig, OUTPUT)
    plt.close(fig)
    print(f"[saved] {OUTPUT}")


if __name__ == "__main__":
    main()
