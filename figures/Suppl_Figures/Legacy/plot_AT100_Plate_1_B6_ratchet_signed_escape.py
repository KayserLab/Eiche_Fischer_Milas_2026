"""Generate Plate 1 B6's ratchet panel with signed confinement/escape axes."""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream/related script reference(s):
# - analysis/figure_2/plot_selected_escape_progression_analysis.py
# - generate_ratchet_effect_all_replicates.py
# - import matplotlib.py
# - plot_all_well_confinement_escape_trajectories.py
# =============================================================================

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Plate 1 B6 Results trajectory and matching experimental/simulation
# confinement and escape-count CSVs.
# Scripts: analysis/figure_2/plot_selected_escape_progression_analysis.py,
# generate_ratchet_effect_all_replicates.py, and
# plot_all_well_confinement_escape_trajectories.py.
# Outputs: the selected Plate 1 B6 signed ratchet PDF.

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIG2 = REPO / "figures" / "Figure_2"
sys.path[:0] = [str(FIG2), str(REPO)]

from shared import figtools as ft
from shared.data_paths import figure_plot_dir
from analysis.legacy import plot_selected_escape_progression_analysis as comparison
from figures.Suppl_Figures.Legacy import plot_all_well_confinement_escape_trajectories as source, \
    generate_ratchet_effect_all_replicates as batch

PLATE, SAMPLE, DPD_PLATE, DPD_SAMPLE = "Plate_1", "B6_AT100-B6", "Plate_AT100", "R24_AT100-R24"
OUTPUT = figure_plot_dir("Suppl_Figures", "ratchet_effect_all_replicates", create=False) / "AT100_Plate_1_B6_AT100-B6_ratchet_2x2.pdf"
CMAP = LinearSegmentedColormap.from_list("blue_orange_time", ["#2C7BB6", "#F28E2B"])
ft.apply_nature_style(text_pt=6.5, tick_pt=5.5, legend_pt=5.5)


def signed_values(burdens):
    args = comparison.parse_args()
    signed = comparison.load_signed_trajectories(args, carry_forward=True)
    signed = signed[~signed["carried_forward"].astype(bool)].copy()
    baselines = burdens.copy()
    baselines["total"] = baselines["sensitive"] + baselines["resistant"]
    baselines = baselines.sort_values("day").groupby(["dataset_type", "plate", "sample"], as_index=False).first()
    baselines = baselines.rename(columns={"total": "initial_total"})
    signed = signed.merge(baselines[["dataset_type", "plate", "sample", "initial_total"]], on=["dataset_type", "plate", "sample"], how="inner")
    signed["signed_norm"] = signed["plot_value"] / signed["initial_total"]
    return signed


def draw_signed(ax, data, xlim, ylim, unavailable=False):
    if unavailable or data.empty:
        ax.text(.5, .5, "DPD unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=6)
    else:
        d = data.dropna(subset=["total_norm", "signed_norm"]).sort_values("day")
        if len(d) >= 2:
            pts = np.column_stack([d["total_norm"], d["signed_norm"]])
            lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap=CMAP, norm=Normalize(0, 1), linewidths=1.15)
            lc.set_array(np.linspace(0, 1, len(d) - 1)); ax.add_collection(lc)
        if len(d):
            p = np.linspace(0, 1, len(d)); ax.scatter(d["total_norm"], d["signed_norm"], c=p, cmap=CMAP, norm=Normalize(0, 1), s=12, edgecolors="black", linewidths=.2, zorder=3)
    ax.axhline(0, color="#222222", lw=.65, zorder=0)
    ax.axhline(.5, color="#8C510A", ls="--", lw=.8, zorder=1)
    ax.set(xlabel="Normalized total cell count", ylabel="Escape Margin", xlim=xlim, ylim=ylim)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    raw = comparison.read_experimental_burdens(comparison.EXPERIMENT_ROOT)
    raw = source.filter_sample_days(raw, sample_column="sample", day_column="day", include_excluded=False)
    qdir = source.DEFAULT_SIMULATION_ROOT / source.SIMULATION_QUANTIFICATION_SUBDIR
    dpd_raw = comparison.read_dpd_burdens(qdir / source.SIMULATION_ESCAPE_COUNTS_FILENAME)
    invitro = batch.normalized(raw[(raw["plate"] == PLATE) & (raw["sample"] == SAMPLE)])
    dpd = batch.normalized(dpd_raw[(dpd_raw["plate"] == DPD_PLATE) & (dpd_raw["sample"] == DPD_SAMPLE)])
    signed = signed_values(pd.concat([raw, dpd_raw], ignore_index=True))
    iv_signed = signed[(signed["dataset_type"] == "In vitro") & (signed["plate"] == PLATE) & (signed["sample"] == SAMPLE)].merge(invitro[["day", "total", "total_norm"]], on="day", how="inner")
    dpd_signed = signed[(signed["dataset_type"] == "DPD") & (signed["plate"] == DPD_PLATE) & (signed["sample"] == DPD_SAMPLE)].merge(dpd[["day", "total", "total_norm"]], on="day", how="inner")
    frames = [invitro, dpd]
    total_xlim = (0.0, 3.5)
    resistance_ylim = batch.limits([f["resistant_norm"] for f in frames], log=True)
    signed_ylim = (-0.8, 0.6)
    fig, axes = ft.new_subplots(2, 2, ax_width_mm=35, ax_height_mm=30, margins=(12, 4, 11, 4), wspace_mm=7, hspace_mm=7, apply_style=False)
    axes = np.asarray(axes)
    batch.draw(axes[0, 0], invitro, "total_norm", "resistant_norm", "Normalized total cell count", "Normalized resistant cell count", total_xlim, resistance_ylim, ylog=True)
    batch.draw(axes[0, 1], dpd, "total_norm", "resistant_norm", "Normalized total cell count", "Normalized resistant cell count", total_xlim, resistance_ylim, ylog=True)
    for ax in axes[0]: ax.axhline(.5, color="#8C510A", ls="--", lw=.8, zorder=1)
    draw_signed(axes[1, 0], iv_signed, total_xlim, signed_ylim)
    draw_signed(axes[1, 1], dpd_signed, total_xlim, signed_ylim)
    axes[0,0].set_title("In vitro", fontsize=6.5); axes[0,1].set_title("DPD", fontsize=6.5)
    ft.savefig(fig, OUTPUT); plt.close(fig); print(f"[saved] {OUTPUT}")


if __name__ == "__main__": main()
