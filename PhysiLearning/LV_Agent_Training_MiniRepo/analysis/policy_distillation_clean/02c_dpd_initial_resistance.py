"""Step 2c: how much do the DPD runs differ in their starting resistant fraction?

Every LV run in step 2b starts from the config's resistant fraction, 0.0005. The
original analysis also tested on LV runs whose starting fraction was drawn
between 0.0001 and 0.003 (its "robustness set"). Here we count the resistant
cells on day 0 of every DPD run, to see where the DPD runs lie.

Runs 0-9 are left out: they are artificial runs that all start from the same
tissue and only place the resistant clone at different distances from the centre.

Output: figures/02c_dpd_initial_resistance.png, a short summary, and
        data/dpd_initial_resistance.csv (one row per run; used by step 2b vary-start).

Run from anywhere, in the `distill` environment:
    python analysis/policy_distillation_clean/02c_dpd_initial_resistance.py
"""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ARTIFICIAL_RUNS, DPD_START_FILE, FIG_DIR, GREY

DPD_DIR = Path("/home/mmilas/Documents/cancer/FINAL/data-DB_Agent/outputs")  # same as step 2
RESISTANT = 1  # CellType.RESISTANT in puls_dpd_tissue/model.py (0 = susceptible)
ACTIVE = 1     # CellFlag.ACTIVE: the cell exists

LV_CONFIG_FRACTION = 0.0005      # starting fraction of every LV run in step 2b
ROBUSTNESS_RANGE = (1e-4, 3e-3)  # the original analysis' robustness set (--mut-fraction-range)
INK, MUTED = "#0b0b0b", "#52514e"


def starting_state(run_dir):
    """Number of cells and of resistant cells in the first snapshot (day 0) of a run."""
    with h5py.File(run_dir / "simulation_000000000.h5") as f:
        data = f["data"]
        assert data['"current_time"'][()] < 0.5, "first snapshot is not day 0"
        active = data['"cell_flags"'][:] == ACTIVE
        resistant = active & (data['"cell_type"'][:] == RESISTANT)
    return int(active.sum()), int(resistant.sum())


# 1. Read day 0 of every run -----------------------------------------------------
rows = []
print("Left out: runs 0-9 (artificial)")
for run_dir in sorted(DPD_DIR.glob("run_*"), key=lambda p: int(p.name.split("_")[1])):
    if run_dir.name in ARTIFICIAL_RUNS:
        continue
    if len(list(run_dir.glob("simulation_*.h5"))) < 2:
        print(f"{run_dir.name}: only a day-0 snapshot, the run never ran -> left out")
        continue
    cells, resistant = starting_state(run_dir)
    rows.append({"run": run_dir.name, "cells": cells, "resistant": resistant,
                 "fraction": resistant / cells})
table = pd.DataFrame(rows)
table.to_csv(DPD_START_FILE, index=False)  # step 2b (vary-start) starts LV episodes from these

f = table["fraction"]
lo, hi = ROBUSTNESS_RANGE
print(f"\n{len(table)} runs. Starting resistant fraction: min {f.min():.5f}, median {f.median():.5f}, "
      f"max {f.max():.5f} (max/min = {f.max() / f.min():.0f}x)")
print(f"Resistant cells at the start: {table['resistant'].min()}-{table['resistant'].max()}")
print(f"Runs below the LV config value {LV_CONFIG_FRACTION}: {(f < LV_CONFIG_FRACTION).sum()}")
print(f"Runs inside the original robustness range {lo}-{hi}: {f.between(lo, hi).sum()}")

# 2. The figure: one dot per run, stacked in bins of the (log) fraction ------------
# Bins are equally wide on a log scale: each bin covers a factor 10**0.1 = 1.26.
log_f = np.log10(f)
bin_width = 0.1
bin_index = np.floor(log_f / bin_width).astype(int)
x = 10 ** ((bin_index + 0.5) * bin_width)  # bin centre
y = table.groupby(bin_index).cumcount() + 1  # 1, 2, 3, ... within each bin

fig, ax = plt.subplots(figsize=(8, 4.2))
# (Labels sit in the empty part right of the dashed line: band label top right,
# LV label lower down, next to its line.)
ax.axvspan(lo, hi, color="#ecebe6", zorder=0)
ax.text(hi * 0.93, 8.3, "original analysis,\nrobustness set: LV runs\ndrawn from 0.0001 to 0.003",
        ha="right", va="top", fontsize=8, color=MUTED)
ax.axvline(LV_CONFIG_FRACTION, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
ax.text(LV_CONFIG_FRACTION * 1.06, 5.5, "every LV run\nin step 2b: 0.0005", ha="left", va="top",
        fontsize=8, color=MUTED)
ax.scatter(x, y, s=70, color="#2a78d6", edgecolors="white", linewidths=0.8, zorder=2)

ax.set_xscale("log")
ticks = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
ax.set_xlim(4e-5, 4e-3)
ax.set_ylim(0, 8.6)
ax.set_yticks(range(0, 9, 2))
ax.set_xlabel("starting resistant fraction  (resistant cells / all cells on day 0, log scale)")
ax.set_ylabel("number of DPD runs")
ax.set_title(f"Starting resistant fraction of the {len(table)} DPD runs (one dot per run)",
             loc="left", fontsize=10)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / "02c_dpd_initial_resistance.png"
fig.savefig(out, dpi=200)
print(f"\nSaved to {out}")
