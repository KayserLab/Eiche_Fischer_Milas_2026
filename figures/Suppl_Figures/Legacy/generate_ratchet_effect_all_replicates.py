"""Batch 2x2 ratchet-effect panels for Figure 2."""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream/related script reference(s):
# - analysis/figure_2/plot_selected_escape_progression_analysis.py
# - and shared/figtools.py
# - shared/experiment_filters.py
# - import matplotlib.py
# - plot_all_well_confinement_escape_trajectories.py
# =============================================================================

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental Results/*.txt and experimental/simulation confinement and
# escape-count CSVs.
# Scripts: analysis/figure_2/plot_selected_escape_progression_analysis.py,
# plot_all_well_confinement_escape_trajectories.py, shared/experiment_filters.py,
# and shared/figtools.py.
# Outputs: one 2x2 ratchet panel per replicate and a manifest CSV.

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
from shared.data_paths import figure_data_dir, figure_plot_dir
from figures.Suppl_Figures.Legacy import plot_all_well_confinement_escape_trajectories as source
from analysis.legacy import plot_selected_escape_progression_analysis as comparison

OUTPUT_DIR = figure_plot_dir("Suppl_Figures", "ratchet_effect_all_replicates", create=False)
DATA_OUTPUT_DIR = figure_data_dir("Suppl_Figures", "ratchet_effect_all_replicates", create=False)
CORE = {"CT", "AT50", "AT100"}
CMAP = LinearSegmentedColormap.from_list("blue_orange_time", ["#2C7BB6", "#F28E2B"])

MATCHES = {
    (plate, sample): (strategy, f"Plate_{strategy}", dpd_sample)
    for strategy, cohort in (("CT", comparison.CT_MATCHES), ("AT50", comparison.AT50_MATCHES), ("AT100", comparison.AT100_MATCHES))
    for plate, sample, dpd_sample in cohort
}

ft.apply_nature_style(text_pt=6.5, tick_pt=5.5, legend_pt=5.5)


def normalized(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("day").copy()
    out["total"] = out["sensitive"] + out["resistant"]
    baseline = float(out.iloc[0]["total"])
    out["total_norm"] = out["total"] / baseline
    out["resistant_norm"] = out["resistant"] / baseline
    return out


def limits(values: list[pd.Series], *, log: bool = False) -> tuple[float, float]:
    arr = np.concatenate([v.to_numpy(float) for v in values])
    arr = arr[np.isfinite(arr) & (arr > 0 if log else np.ones(arr.size, dtype=bool))]
    if not arr.size:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo, hi = float(arr.min()), float(arr.max())
    if log:
        if lo == hi:
            return lo / 1.2, hi * 1.2
        pad = (hi / lo) ** 0.06
        return lo / pad, hi * pad
    span = hi - lo
    return min(0.0, lo), hi + (span * 0.06 if span else max(1.0, hi * 0.06))


def draw(ax, data: pd.DataFrame | None, x: str, y: str, xlabel: str, ylabel: str,
         xlim: tuple[float, float], ylim: tuple[float, float], *, xlog=False, ylog=False,
         unavailable=False):
    if unavailable or data is None or data.empty:
        ax.text(0.5, 0.5, "DPD unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=6)
    else:
        d = data.dropna(subset=[x, y]).sort_values("day").copy()
        if xlog: d = d[d[x] > 0]
        if ylog: d = d[d[y] > 0]
        if len(d) >= 2:
            pts = np.column_stack([d[x], d[y]])
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(seg, cmap=CMAP, norm=Normalize(0, 1), linewidths=1.15)
            lc.set_array(np.linspace(0, 1, len(d) - 1))
            ax.add_collection(lc)
        if len(d):
            p = np.linspace(0, 1, len(d))
            ax.scatter(d[x], d[y], c=p, cmap=CMAP, norm=Normalize(0, 1), s=12, edgecolors="black", linewidths=.2, zorder=3)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)


def panel(invitro: pd.DataFrame, dpd: pd.DataFrame | None, output: Path):
    frames = [invitro] + ([] if dpd is None else [dpd])
    total_xlim = limits([f["total"] for f in frames])
    resistant_norm_ylim = limits([f["resistant_norm"] for f in frames])
    confinement_ylim = limits([f["confinement_margin"] for f in frames])
    fig, axes = ft.new_subplots(2, 2, ax_width_mm=35, ax_height_mm=30, margins=(12, 4, 11, 4), wspace_mm=7, hspace_mm=7, apply_style=False)
    axes = np.asarray(axes)
    draw(axes[0, 0], invitro, "total", "resistant_norm", "Total cell count", "Normalized resistant cell count", total_xlim, resistant_norm_ylim)
    draw(axes[0, 1], dpd, "total", "resistant_norm", "Total cell count", "Normalized resistant cell count", total_xlim, resistant_norm_ylim, unavailable=dpd is None)
    for ax in axes[0]:
        ax.axhline(0.5, color="#8C510A", ls="--", lw=0.8, zorder=1)
        ax.text(0.98, 0.5, "Failure", color="#8C510A", fontsize=4.5,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
    draw(axes[1, 0], invitro, "total", "confinement_margin", "Total cell count", "Confinement margin", total_xlim, confinement_ylim)
    draw(axes[1, 1], dpd, "total", "confinement_margin", "Total cell count", "Confinement margin", total_xlim, confinement_ylim, unavailable=dpd is None)
    axes[0, 0].set_title("In vitro", fontsize=6.5); axes[0, 1].set_title("DPD", fontsize=6.5)
    ft.savefig(fig, output); plt.close(fig)


def main():
    raw = comparison.read_experimental_burdens(comparison.EXPERIMENT_ROOT)
    raw = raw[raw["strategy"].isin(CORE)].copy()
    raw = source.filter_sample_days(raw, sample_column="sample", day_column="day", include_excluded=False)
    conf = source.load_confinement(source.DEFAULT_CONFINEMENT_CSV, include_excluded=False).rename(columns={"confinement_thickness": "confinement_margin"})
    raw = raw.merge(conf, on=["plate", "sample", "day"], how="left")
    raw = raw.dropna(subset=["confinement_margin"])

    qdir = source.DEFAULT_SIMULATION_ROOT / source.SIMULATION_QUANTIFICATION_SUBDIR
    dpd_raw = comparison.read_dpd_burdens(qdir / source.SIMULATION_ESCAPE_COUNTS_FILENAME)
    dpd_conf = source.load_confinement(qdir / source.SIMULATION_CONFINEMENT_FILENAME, include_excluded=True).rename(columns={"confinement_thickness": "confinement_margin"})
    dpd_raw = dpd_raw.merge(dpd_conf, on=["plate", "sample", "day"], how="left").dropna(subset=["confinement_margin"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (plate, sample, strategy), group in raw.groupby(["plate", "sample", "strategy"], sort=True):
        invitro = normalized(group)
        match = MATCHES.get((str(plate), str(sample)))
        dpd = None
        if match is not None:
            _, dpd_plate, dpd_sample = match
            candidate = dpd_raw[(dpd_raw["plate"] == dpd_plate) & (dpd_raw["sample"] == dpd_sample)]
            if not candidate.empty:
                dpd = normalized(candidate)
        filename = f"{strategy}_{plate}_{sample}_ratchet_2x2.pdf"
        panel(invitro, dpd, OUTPUT_DIR / filename)
        manifest.append({"plate": plate, "sample": sample, "strategy": strategy, "dpd_available": dpd is not None, "output": filename})
        print(f"[saved] {filename}")
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(DATA_OUTPUT_DIR / "ratchet_effect_2x2_manifest.csv", index=False)
    print(f"[summary] {len(manifest)} PDFs; DPD available for {sum(row['dpd_available'] for row in manifest)}")


if __name__ == "__main__":
    main()
