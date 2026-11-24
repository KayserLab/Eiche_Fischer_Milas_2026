"""
Resistant flux vs local density Ï_tot: simulation (row 1) and raw
experimental data (row 2), on shared cells/1000ÂµmÂ²-per-24h axes. Reads
compute_flux_vs_density.py's and compute_flux_vs_density_experiment.py's
output -- run both first. Standalone: safe to copy out and run on its own.

Run: python fig_flux_vs_density.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis/figure_3/HDF5_analysis")]


import os

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from constants import (
    exp_scaling, px_to_micron, AREA_SCALE,
    prolif_rho_max as rho_max, prolif_n_rho_bins as n_rho_bins,
    rho_homeostatic_res, rho_confluent_res, color_res, color_master, protocol_colors,
    NATURE_FONT_STACK, NATURE_TEXT_PT, NATURE_TICK_PT, NATURE_LEGEND_PT, NATURE_DPI,
    NATURE_AXIS_LINEWIDTH_PT, NATURE_TICK_LENGTH_PT,
    MIN_RUNS_PER_BIN, MIN_PROTOCOLS_PER_BIN, FIG_WIDTH_MM,
    SIMULATION_OUTPUT_DIR, EXPERIMENT_OUTPUT_DIR, FIGURES_OUTPUT_DIR,
)

# ======================================================
# CONFIG -- edit these to point at your own data/output layout
# ======================================================

PRESSURE_EXPERIMENTS = ["AT50", "AT100", "CT"]

# Must match compute_flux_vs_density.py's OUT_DIR.
KERNEL_DENSITIES_DIR = os.fspath(SIMULATION_OUTPUT_DIR)

FIGURES_DIR = os.fspath(FIGURES_OUTPUT_DIR)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Row-2 (experiment) data; must match compute_flux_vs_density_experiment.py's OUT_DIR.
EXP_DIR = os.fspath(EXPERIMENT_OUTPUT_DIR)
# Compare the same treatment strategies in experiment and simulation.
# The experimental master curve is therefore the mean of CT, AT50, and AT100.
EXP_STRATEGIES = list(PRESSURE_EXPERIMENTS)


def flux_vs_density_path(protocol):
    return os.path.join(KERNEL_DENSITIES_DIR, f"flux_vs_density_{protocol}.h5")


def exp_flux_vs_density_path(strategy):
    return os.path.join(EXP_DIR, f"flux_vs_density_{strategy}.h5")


def save_figure(fig, filename):
    """Save fig to FIGURES_DIR/filename, close it, and report the path."""
    out = os.path.join(FIGURES_DIR, filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ======================================================
# DERIVED CONSTANTS
# ======================================================

rho_edges   = np.linspace(0, rho_max, n_rho_bins + 1)
rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])

# Row-2 bins: same rho_max, in exp-px^-2 (AREA_SCALE = exp_scaling**2 converts).
rho_max_exp     = rho_max / AREA_SCALE
rho_edges_exp   = np.linspace(0, rho_max_exp, n_rho_bins + 1)
rho_centers_exp = 0.5 * (rho_edges_exp[:-1] + rho_edges_exp[1:])

matplotlib.rcParams.update({
    "font.family":            "sans-serif",
    "font.sans-serif":        NATURE_FONT_STACK,
    "font.size":              NATURE_TEXT_PT,
    "axes.titlesize":         NATURE_TEXT_PT,
    "axes.labelsize":         NATURE_TEXT_PT,
    "xtick.labelsize":        NATURE_TICK_PT,
    "ytick.labelsize":        NATURE_TICK_PT,
    "legend.fontsize":        NATURE_LEGEND_PT,
    "legend.title_fontsize":  NATURE_LEGEND_PT,
    "figure.titlesize":       NATURE_TEXT_PT,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "text.usetex": False,
    "axes.linewidth":       NATURE_AXIS_LINEWIDTH_PT,
    "xtick.major.width":    NATURE_AXIS_LINEWIDTH_PT,
    "ytick.major.width":    NATURE_AXIS_LINEWIDTH_PT,
    "xtick.major.size":     NATURE_TICK_LENGTH_PT,
    "ytick.major.size":     NATURE_TICK_LENGTH_PT,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "figure.dpi": NATURE_DPI, "savefig.dpi": NATURE_DPI,
})

lw = NATURE_AXIS_LINEWIDTH_PT

UNIT_SCALE    = exp_scaling * px_to_micron   # simulation units -> um
DENSITY_SCALE = 1000 / UNIT_SCALE**2         # sim-unit density -> cells / 1000 um^2

# exp-px -> um directly (no exp_scaling); matches DENSITY_SCALE's x-axis range exactly.
DENSITY_SCALE_exp = 1000 / px_to_micron**2


def fig_size_inches(aspect):
    """(width_in, height_in) for FIG_WIDTH_MM at the given aspect ratio."""
    return FIG_WIDTH_MM / 25.4, FIG_WIDTH_MM * aspect / 25.4


def despine(*axes):
    """Hide the top and right spines on each given axis."""
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ======================================================
# PLOTTING HELPERS (copied from scripts/plotting.py)
# ======================================================

def confluency_split(centers, confluent):
    """Slices splitting `centers` into below/above a confluency threshold;
    they share the boundary index so a line plotted across both stays
    continuous."""
    k = np.searchsorted(centers, confluent)
    k = min(max(k, 1), len(centers) - 1)
    return slice(0, k + 1), slice(k, None)


def median_iqr_band(
    ax, x, median, q25, q75, below, above, color, lw,
    band_alpha=0.25, label=None, linestyle="-",
):
    """Median curve with a shaded IQR band, faded over `below` and solid
    over `above` (see confluency_split)."""
    ax.plot(
        x[below], median[below], color=color, lw=lw,
        linestyle=linestyle, alpha=0.3,
    )
    ax.fill_between(x[below], q25[below], q75[below], color=color, alpha=band_alpha * 0.3, lw=0)
    ax.plot(
        x[above], median[above], color=color, lw=lw,
        linestyle=linestyle, label=label,
    )
    ax.fill_between(x[above], q25[above], q75[above], color=color, alpha=band_alpha, lw=0)


# ======================================================
# LOAD PER-RUN CURVES
# ======================================================

def _load_curves(path_fn, protocols, label):
    median_by, median_rel_by = {}, {}
    for protocol in protocols:
        with h5py.File(path_fn(protocol), "r") as f:
            median_by[protocol]     = f["median"][:]      # (n_runs, n_bins)
            median_rel_by[protocol] = f["median_rel"][:]   # (n_runs, n_bins)
        n_runs = median_by[protocol].shape[0]
        print(f"[{label}] {protocol}: {n_runs} runs, "
              f"{np.isfinite(median_by[protocol]).any(axis=1).sum()} with valid absolute bins, "
              f"{np.isfinite(median_rel_by[protocol]).any(axis=1).sum()} with valid relative bins")
    return median_by, median_rel_by


median_by_protocol, median_rel_by_protocol = _load_curves(flux_vs_density_path, PRESSURE_EXPERIMENTS, "simulation")
exp_median_by, exp_median_rel_by = _load_curves(exp_flux_vs_density_path, EXP_STRATEGIES, "experiment")

# ======================================================
# AVERAGE ACROSS RUNS -> per-protocol curve; AVERAGE ACROSS PROTOCOLS -> master curve
# ======================================================

def _runs_to_protocol_curve(median_2d):
    """median/q25/q75 across a protocol's run curves; NaN where fewer than
    MIN_RUNS_PER_BIN runs have a valid bin."""
    n_runs_valid = np.isfinite(median_2d).sum(axis=0)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(median_2d, axis=0)
        q25    = np.nanpercentile(median_2d, 25, axis=0)
        q75    = np.nanpercentile(median_2d, 75, axis=0)
    enough = n_runs_valid >= MIN_RUNS_PER_BIN
    return np.where(enough, median, np.nan), np.where(enough, q25, np.nan), np.where(enough, q75, np.nan)


def _protocols_to_master_curve(protocol_medians):
    """Mean across the (few) per-protocol median curves, min/max as the band;
    NaN where fewer than MIN_PROTOCOLS_PER_BIN protocols have a valid bin."""
    stacked = np.stack(protocol_medians)   # (n_protocols, n_bins)
    n_valid = np.isfinite(stacked).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
        lo   = np.nanmin(stacked, axis=0)
        hi   = np.nanmax(stacked, axis=0)
    enough = n_valid >= MIN_PROTOCOLS_PER_BIN
    return np.where(enough, mean, np.nan), np.where(enough, lo, np.nan), np.where(enough, hi, np.nan)


def _panel_stats(median_by_protocol, protocols):
    stats = {p: _runs_to_protocol_curve(median_by_protocol[p]) for p in protocols}
    master = _protocols_to_master_curve([stats[p][0] for p in protocols])
    return stats, master

stats_abs, master_abs = _panel_stats(median_by_protocol, PRESSURE_EXPERIMENTS)
stats_rel, master_rel = _panel_stats(median_rel_by_protocol, PRESSURE_EXPERIMENTS)
exp_stats_abs, exp_master_abs = _panel_stats(exp_median_by, EXP_STRATEGIES)
exp_stats_rel, exp_master_rel = _panel_stats(exp_median_rel_by, EXP_STRATEGIES)

# ======================================================
# PLOT
# ======================================================

below, above = confluency_split(rho_centers, rho_confluent_res)
below_exp, above_exp = confluency_split(rho_centers_exp, rho_confluent_res / AREA_SCALE)
rho_centers_scaled = rho_centers * DENSITY_SCALE
rho_centers_scaled_exp = rho_centers_exp * DENSITY_SCALE_exp
# Same physical density in both unit systems, so these lines/limits apply to both rows.
XLABEL_RHO = r"local density $\rho_\mathrm{tot}$ [cells (1000 µm$^2$)$^{-1}$]"
CONFLUENT_SCALED    = rho_confluent_res * DENSITY_SCALE
HOMEOSTATIC_SCALED  = rho_homeostatic_res * DENSITY_SCALE
RHO_MAX_SCALED      = rho_max * DENSITY_SCALE
RELATIVE_Y_LIMITS   = (0.0, 2.0)


def _set_shared_region_ylim(axes, specifications):
    """Scale paired panels from data between the two density references."""
    values = []
    lower_x, upper_x = sorted((CONFLUENT_SCALED, HOMEOSTATIC_SCALED))
    for x, arrays, scale in specifications:
        in_region = (x >= lower_x) & (x <= upper_x)
        for array in arrays:
            data = np.asarray(array, dtype=float)
            selected = data[..., in_region] * scale
            values.extend(selected[np.isfinite(selected)].ravel())
    if not values:
        return
    lower = min(0.0, float(np.min(values)))
    upper = max(0.0, float(np.max(values)))
    span = upper - lower
    padding = 0.08 * span if span > 0 else max(abs(lower), abs(upper), 1.0) * 0.08
    for axis in axes:
        axis.set_ylim(lower - padding, upper + padding)


def _summary_arrays(stats, master, protocols):
    return [
        array
        for protocol in protocols
        for array in stats[protocol]
    ] + list(master)


def _individual_arrays(curves, master, protocols):
    return [curves[protocol] for protocol in protocols] + [master[0]]


def _draw_panel(ax, protocols, stats_by_protocol, master, rho_centers_scaled, below, above,
                y_scale, ylabel, master_label, linestyle=None):
    if linestyle is None:
        linestyle = "--" if master_label == "mean of protocols" else "-"
    ax.axhline(0.0, color="gray", ls="-", lw=lw)

    for protocol in protocols:
        median, q25, q75 = stats_by_protocol[protocol]
        if not np.any(np.isfinite(median)):
            continue
        color = protocol_colors.get(protocol, "gray")
        median_iqr_band(ax, rho_centers_scaled, median * y_scale, q25 * y_scale, q75 * y_scale,
                        below, above, color, lw, label=protocol, linestyle=linestyle)

    median_m, lo_m, hi_m = master
    if np.any(np.isfinite(median_m)):
        median_iqr_band(ax, rho_centers_scaled, median_m * y_scale, lo_m * y_scale, hi_m * y_scale,
                        below, above, color_master, lw * 1.5, label=master_label,
                        linestyle=linestyle)

    ax.axvline(CONFLUENT_SCALED, color=color_res, ls=":", lw=lw)
    ax.axvline(HOMEOSTATIC_SCALED, color=color_res, ls="--", lw=lw)
    ax.set_xlabel(XLABEL_RHO)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, RHO_MAX_SCALED)
    ax.legend()
    despine(ax)

fig, ((ax_abs, ax_rel), (ax_exp_abs, ax_exp_rel)) = plt.subplots(
    2, 2, figsize=fig_size_inches(0.8), sharey="col"
)

_draw_panel(ax_abs, PRESSURE_EXPERIMENTS, stats_abs, master_abs, rho_centers_scaled, below, above,
            DENSITY_SCALE, r"median $\Delta\rho_\mathrm{res}$ [24h, cells (1000 µm$^2$)$^{-1}$]", "mean of protocols")
_draw_panel(ax_rel, PRESSURE_EXPERIMENTS, stats_rel, master_rel, rho_centers_scaled, below, above,
            1.0, r"median $\Delta\rho_\mathrm{res} / \rho_\mathrm{res}(t_0)$ [24h$^{-1}$]", "mean of protocols")
_draw_panel(ax_exp_abs, EXP_STRATEGIES, exp_stats_abs, exp_master_abs, rho_centers_scaled_exp, below_exp, above_exp,
            DENSITY_SCALE_exp, r"median $\Delta\rho_\mathrm{res}$ [24h, cells (1000 µm$^2$)$^{-1}$]", "mean of strategies")
_draw_panel(ax_exp_rel, EXP_STRATEGIES, exp_stats_rel, exp_master_rel, rho_centers_scaled_exp, below_exp, above_exp,
            1.0, r"median $\Delta\rho_\mathrm{res} / \rho_\mathrm{res}(t_0)$ [24h$^{-1}$]", "mean of strategies")

_set_shared_region_ylim(
    (ax_abs, ax_exp_abs),
    (
        (rho_centers_scaled, _summary_arrays(stats_abs, master_abs, PRESSURE_EXPERIMENTS), DENSITY_SCALE),
        (rho_centers_scaled_exp, _summary_arrays(exp_stats_abs, exp_master_abs, EXP_STRATEGIES), DENSITY_SCALE_exp),
    ),
)
for axis in (ax_rel, ax_exp_rel):
    axis.set_ylim(*RELATIVE_Y_LIMITS)

ax_abs.set_title("Simulation", loc="left", fontweight="bold")
ax_exp_abs.set_title("Experiment", loc="left", fontweight="bold")

fig.tight_layout()
save_figure(fig, os.path.join("pressure_analysis", "flux_vs_density.pdf"))

# ======================================================
# VERIFICATION: per-run curves ("spaghetti plot")
# ======================================================

fig2, ((ax2_abs, ax2_rel), (ax2_exp_abs, ax2_exp_rel)) = plt.subplots(
    2, 2, figsize=fig_size_inches(0.8), sharey="col"
)

def _draw_spaghetti(ax, protocols, median_by_protocol, master_median, rho_centers_scaled, y_scale, ylabel):
    linestyle = "--" if ax.get_subplotspec().rowspan.start == 0 else "-"
    ax.axhline(0.0, color="gray", ls="-", lw=lw)
    for protocol in protocols:
        color = protocol_colors.get(protocol, "gray")
        for run_median in median_by_protocol[protocol]:
            ax.plot(
                rho_centers_scaled, run_median * y_scale, color=color,
                ls=linestyle, lw=0.6, alpha=0.25,
            )
        ax.plot(
            [], [], color=color, ls=linestyle, lw=1.5, label=protocol
        )   # legend proxy
    if np.any(np.isfinite(master_median)):
        ax.plot(
            rho_centers_scaled, master_median * y_scale,
            color=color_master, ls=linestyle, lw=lw * 1.5, label="mean",
        )
    ax.set_xlabel(XLABEL_RHO)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, RHO_MAX_SCALED)
    ax.legend()
    despine(ax)

_draw_spaghetti(ax2_abs, PRESSURE_EXPERIMENTS, median_by_protocol, master_abs[0], rho_centers_scaled,
                DENSITY_SCALE, r"$\Delta\rho_\mathrm{res}$ [24h, cells (1000 µm$^2$)$^{-1}$]")
_draw_spaghetti(ax2_rel, PRESSURE_EXPERIMENTS, median_rel_by_protocol, master_rel[0], rho_centers_scaled,
                1.0, r"$\Delta\rho_\mathrm{res} / \rho_\mathrm{res}(t_0)$ [24h$^{-1}$]")
_draw_spaghetti(ax2_exp_abs, EXP_STRATEGIES, exp_median_by, exp_master_abs[0], rho_centers_scaled_exp,
                DENSITY_SCALE_exp, r"$\Delta\rho_\mathrm{res}$ [24h, cells (1000 µm$^2$)$^{-1}$]")
_draw_spaghetti(ax2_exp_rel, EXP_STRATEGIES, exp_median_rel_by, exp_master_rel[0], rho_centers_scaled_exp,
                1.0, r"$\Delta\rho_\mathrm{res} / \rho_\mathrm{res}(t_0)$ [24h$^{-1}$]")
_set_shared_region_ylim(
    (ax2_abs, ax2_exp_abs),
    (
        (rho_centers_scaled, _individual_arrays(median_by_protocol, master_abs, PRESSURE_EXPERIMENTS), DENSITY_SCALE),
        (rho_centers_scaled_exp, _individual_arrays(exp_median_by, exp_master_abs, EXP_STRATEGIES), DENSITY_SCALE_exp),
    ),
)
for axis in (ax2_rel, ax2_exp_rel):
    axis.set_ylim(*RELATIVE_Y_LIMITS)

ax2_abs.set_title("Simulation", loc="left", fontweight="bold")
ax2_exp_abs.set_title("Experiment", loc="left", fontweight="bold")

fig2.tight_layout()
save_figure(fig2, os.path.join("pressure_analysis", "verification", "verification_flux_by_run.pdf"))

# ======================================================
# VERIFICATION: per-run median distributions, one figure per protocol
# ======================================================

N_DIST_BINS = 12
N_HIST_BINS = 10

for protocol in PRESSURE_EXPERIMENTS:
    median_2d = median_by_protocol[protocol]   # (n_runs, n_bins)
    n_valid_per_bin = np.isfinite(median_2d).sum(axis=0)
    valid_bins = np.where(n_valid_per_bin >= MIN_RUNS_PER_BIN)[0]
    if len(valid_bins) == 0:
        print(f"{protocol}: no density bin has enough run curves, skipping distribution plot")
        continue
    dist_bins = np.unique(valid_bins[np.linspace(0, len(valid_bins) - 1,
                                                  min(N_DIST_BINS, len(valid_bins))).astype(int)])

    color = protocol_colors[protocol]

    ncols = 4
    nrows = int(np.ceil(len(dist_bins) / ncols))
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes3 = np.atleast_1d(axes3).reshape(-1)

    median, q25, q75 = stats_abs[protocol]
    for ax3, b in zip(axes3, dist_bins):
        values = median_2d[:, b]
        values = values[np.isfinite(values)] * DENSITY_SCALE

        ax3.hist(values, bins=min(N_HIST_BINS, len(values)), color=color, alpha=0.5)

        ymax = ax3.get_ylim()[1]
        if np.isfinite(median[b]):
            ax3.axvline(median[b] * DENSITY_SCALE, color=color, ls="--", lw=1.5)
            ax3.hlines(0.92 * ymax, q25[b] * DENSITY_SCALE, q75[b] * DENSITY_SCALE, color=color, lw=2)

        lo, hi = rho_edges[b], rho_edges[b + 1]
        ax3.set_title(rf"$\rho_\mathrm{{tot}} \in$ [{lo * DENSITY_SCALE:.2f}, {hi * DENSITY_SCALE:.2f})", fontsize=10)
        ax3.set_xlabel(r"$\Delta\rho_\mathrm{res}$ (per-run median) [cells (1000 µm$^2$)$^{-1}$]")
        ax3.text(0.97, 0.97, f"n={len(values)} runs", transform=ax3.transAxes, ha="right", va="top", fontsize=8)
        despine(ax3)

    for ax3 in axes3[len(dist_bins):]:
        ax3.set_visible(False)

    fig3.suptitle(protocol)
    fig3.tight_layout()
    save_figure(fig3, os.path.join("pressure_analysis", "verification",
                                    f"verification_flux_distributions_{protocol}.pdf"))
