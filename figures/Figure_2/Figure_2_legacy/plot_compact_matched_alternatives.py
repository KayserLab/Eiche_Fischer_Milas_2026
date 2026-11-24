"""Create two compact Figure 2 in-vitro/DPD comparison alternatives."""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "figures" / "Figure_2"
sys.path[:0] = [str(FIGURE_DIR), str(REPO_ROOT)]

from shared import figtools as ft
from analysis.legacy import plot_selected_escape_progression_analysis as plot
from shared.strategy_colors import FIGURE_COLOR_MAP


Y_LIMITS = (-0.5, 0.6)
FAILURE_COLOR = "#8C510A"
ESCAPE_COLOR = "#8A3FFC"
PANELS = (
    ("AT100", "Plate_5", "A2_AT100-A2", "R20_AT100-R20", 60,
     plot.INDIVIDUAL_SUBDIR),
    ("AT50", "Plate_2", "B2_AT50-B2", "R11_AT50-R11", 50,
     plot.AT50_INDIVIDUAL_SUBDIR),
)


def make_panel(trajectories, burdens, endpoints, args, strategy, plate, sample,
               dpd_sample, xmax, output_subdir):
    fig, ax = ft.new_figure(
        width=30, height_mm=40, margins=(12, 3, 11, 3), apply_style=False,
    )
    ymin, ymax = Y_LIMITS
    zero_fraction = -ymin / (ymax - ymin)
    for dataset_type, source_plate, source_sample, linestyle in (
        ("In vitro", plate, sample, "-"),
        ("DPD", f"Plate_{strategy}", dpd_sample, "--"),
    ):
        trajectory = trajectories[
            (trajectories["dataset_type"] == dataset_type)
            & (trajectories["plate"] == source_plate)
            & (trajectories["sample"] == source_sample)
        ].sort_values("day")
        burden = burdens[
            (burdens["dataset_type"] == dataset_type)
            & (burdens["plate"] == source_plate)
            & (burdens["sample"] == source_sample)
        ].sort_values("day")
        endpoint = endpoints[
            (endpoints["dataset_type"] == dataset_type)
            & (endpoints["plate"] == source_plate)
            & (endpoints["sample"] == source_sample)
        ]
        if trajectory.empty or burden.empty or endpoint.empty:
            raise ValueError(f"Missing data for {dataset_type}: {source_plate}/{source_sample}")
        stop = min(float(xmax), float(endpoint["stop_day"].iloc[0]))
        trajectory = trajectory[trajectory["day"] <= stop].copy()
        burden = burden[burden["day"] <= stop].copy()
        initial_total = float(burden["initial_total"].iloc[0])
        trajectory["normalized_value"] = trajectory["plot_value"] / initial_total
        burden["normalized_resistant"] = burden["resistant"] / initial_total
        treatment_days = (
            plot.read_invitro_treatment_days(args.experiment_root, plate, sample)
            if dataset_type == "In vitro"
            else plot.read_dpd_treatment_days(args.dpd_treatment_status, strategy, dpd_sample)
        )
        plot.shade_treatment(
            ax, treatment_days, stop,
            ymin=zero_fraction if dataset_type == "In vitro" else 0.0,
            ymax=1.0 if dataset_type == "In vitro" else zero_fraction,
            hatch="////////" if dataset_type == "DPD" else None,
        )
        confinement = trajectory[trajectory["phase"] == "confinement"]
        escaped = trajectory[trajectory["phase"] == "escaped"]
        ax.plot(confinement["day"], confinement["normalized_value"], color=FIGURE_COLOR_MAP["blue"], ls=linestyle, lw=1.25, zorder=3)
        ax.plot(escaped["day"], escaped["normalized_value"], color=FIGURE_COLOR_MAP["yellow"], ls=linestyle, lw=1.25, zorder=3)
        ax.plot(burden["day"], burden["normalized_resistant"], color="#111111", ls=linestyle, lw=1.25, zorder=3)
        if not escaped.empty:
            ax.axvline(float(escaped["day"].min()), color=ESCAPE_COLOR, ls=linestyle, lw=0.8, zorder=2)

    ax.axhline(0.0, color="#111111", lw=0.65, zorder=-1)
    ax.axhline(0.5, color=FAILURE_COLOR, ls="--", lw=0.8, zorder=4)
    ax.set(xlim=(0, xmax), ylim=Y_LIMITS, xlabel="Time [days]", ylabel="Escape Margin")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color="#333333", lw=1.0, label="In vitro"),
            Line2D([], [], color="#333333", lw=1.0, ls="--", label="DPD"),
            Line2D([], [], color=FAILURE_COLOR, lw=0.8, ls="--", label="Failure"),
            Line2D([], [], color=ESCAPE_COLOR, lw=0.8, label="Escape"),
        ],
        loc="lower right", frameon=False, fontsize=4.2, handlelength=1.2,
        labelspacing=0.25, borderaxespad=0.25,
    )
    output = args.output_dir / output_subdir / f"{strategy}_{plate}_{sample}_normalized_invitro_DPD_{dpd_sample.split('_')[0]}_compact_alternative.pdf"
    ft.savefig(fig, output)
    plt.close(fig)
    return output


def main():
    args = plot.parse_args()
    quantification = args.simulation_quantification_dir or (args.simulation_root / plot.source.SIMULATION_QUANTIFICATION_SUBDIR)
    experimental = plot.read_experimental_burdens(args.experiment_root)
    dpd = plot.read_dpd_burdens(quantification / plot.source.SIMULATION_ESCAPE_COUNTS_FILENAME)
    burdens = plot.add_progression_fields(plot.pd.concat([experimental, dpd], ignore_index=True), args.progression_fraction)
    endpoints = plot.endpoint_table(burdens, args.include_excluded)
    burdens = plot.trim_to_endpoints(burdens, endpoints)
    carry_forward = plot.load_signed_trajectories(args, carry_forward=True)
    trajectories = plot.trim_to_endpoints(carry_forward[~carry_forward["carried_forward"].astype(bool)].copy(), endpoints)
    for panel in PANELS:
        print(f"[saved] {make_panel(trajectories, burdens, endpoints, args, *panel)}")


if __name__ == "__main__":
    main()
