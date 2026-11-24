#!/usr/bin/env python3
"""
TL;DR
-----
This script computes and plots time to progression (TTP) for CT and AT100
replicates from replicate-level total-burden trajectories.

Detailed
--------
Inputs:
- Corrected total-point-cloud HDF5 outputs for the relevant replicates.
- Replicate exclusion/truncation rules from `experiment_filters.py`.

Outputs:
- A TTP comparison figure and the associated strategy-level summary statistics.

How it works:
- Loads per-day total cell counts, normalizes each replicate to its day-1 total
  burden, and applies the shared replicate exclusion/truncation rules.
- Defines TTP as the first day the normalized total burden crosses the
  progression threshold.
- Compares CT and AT100 replicate TTP distributions and annotates the plot with
  the resulting significance statistics.

Pipeline role:
- Figure 1 plotting script for progression timing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis
# length. Margins are extra space for tick labels, axis labels, brackets,
# and other annotations.
FIGURE_AXES_WIDTH_MM = 20.46
FIGURE_AXES_HEIGHT_MM = 31.98
FIGURE_MARGINS_MM = (9.30, 1.24, 5.74, 3.28)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.0
DPI = 300
SHOW_FIGURE = True

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from scipy.stats import mannwhitneyu

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.experiment_filters import EXCLUDE_SAMPLES, apply_sample_exclusions


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
ROOT_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Data")
PLATES = [f"Plate_{i}" for i in range(1, 9)]

OUT_DIR_EXTRA = ft.paper_figure_data_dir(__file__)
OUT_DIR_EXTRA.mkdir(parents=True, exist_ok=True)
OUT_PDF_EXTRA = OUT_DIR_EXTRA / "Fig. 2 TTP CT & AT100.pdf"
SIMULATION_SUMMARY_CSV = ROOT_DIR / "Maja" / "combined_results.csv"

STRATEGIES_KEEP = ["CT", "AT50", "AT100"]
PROGRESSION_THRESHOLD = 1 # res_norm >= 1.0
EXPERIMENT_END_DAY = 96.0  # colonies without progression get this TTP value
USE_TRUNCATION_DAYS = True  # if False: keep truncated wells; still drop fully excluded wells


# ---------------------------------------------------------------------
# FIGURE STYLE
# ---------------------------------------------------------------------
YLIM = (0, 112)  # adjust if you want

ft.apply_nature_style(
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
)

BRACKET_LW = 0.9
BRACKET_H_FRAC = 0.04
BRACKET_TOP_MARGIN_FRAC = 0.08
BRACKET_BOTTOM_MARGIN_FRAC = 0.06
BRACKET_STEP_FRAC = 0.085
STARS_Y_OFFSET_H_FRAC = 0.45
STARS_FONTSIZE = 5
N_FONTSIZE = 5
GRID_ALPHA = 0.0
POINT_JITTER = 0.08
POINT_SIZE = 8
POINT_ALPHA = 0.85
EXPERIMENT_POINT_MARKER = "o"
SIMULATION_POINT_MARKER = "s"
EXPERIMENT_X_OFFSET = -0.18
SIMULATION_X_OFFSET = 0.18
MEDIAN_COLOR = "#ff7f0e"
MEDIAN_LINE_HALF_WIDTH = 0.08


# ---------------------------------------------------------------------
# IMPORT SHARED STRATEGY COLORS
# ---------------------------------------------------------------------
from shared.strategy_colors import get_color


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def parse_strategy(sample: str) -> str:
    """Expected sample naming like 'A1_AT100-A1' -> strategy between '_' and '-'."""
    try:
        return sample.split("_", 1)[1].split("-", 1)[0]
    except Exception:
        return ""


def numeric_sort_key(s: str) -> int:
    m = re.findall(r"\d+", s)
    return int(m[0]) if m else 0


def count_coords_dataset(ds: np.ndarray) -> int:
    arr = np.asarray(ds)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return int(arr.shape[0])
    flat = arr.ravel()
    if flat.size < 2:
        return 0
    return int(flat.size // 2)


def load_counts_for_group(h5: h5py.File, group_name: str) -> Tuple[np.ndarray, np.ndarray]:
    grp = h5[group_name]
    keys = sorted(list(grp.keys()), key=numeric_sort_key)

    days_raw: List[int] = []
    counts: List[int] = []

    for key in keys:
        m = re.findall(r"\d+", key)
        day_num = int(m[0]) if m else len(days_raw)
        days_raw.append(day_num)

        ds_path = f"{group_name}/{key}/block0_values"
        if ds_path in h5:
            counts.append(count_coords_dataset(h5[ds_path][()]))
            continue

        obj = grp[key]
        if isinstance(obj, h5py.Dataset):
            counts.append(count_coords_dataset(obj[()]))
        elif isinstance(obj, h5py.Group) and "values" in obj:
            counts.append(count_coords_dataset(obj["values"][()]))
        else:
            counts.append(0)

    return np.asarray(days_raw, dtype=float), np.asarray(counts, dtype=float)


def load_counts_from_total_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Reads a single *_total.h5 file and returns a tidy dataframe with:
      sample, strategy, day_tp1, sus, res, total
    day_tp1 is "days since first available timepoint" (baseline day_tp1=0).
    """
    sample = re.sub(r"_total$", "", re.sub(r"^G418_a2a4_", "", path.stem))
    strategy = parse_strategy(sample)
    if strategy not in STRATEGIES_KEEP:
        return None

    with h5py.File(path, "r") as h5:
        if "sus" not in h5 or "res" not in h5:
            return None

        days_s, sus = load_counts_for_group(h5, "sus")
        days_r, res = load_counts_for_group(h5, "res")

        all_days = np.unique(np.concatenate([days_s, days_r]))
        all_days.sort()

        sus_al = np.zeros_like(all_days, dtype=float)
        res_al = np.zeros_like(all_days, dtype=float)

        idx_s = {d: i for i, d in enumerate(days_s)}
        idx_r = {d: i for i, d in enumerate(days_r)}

        for i, d in enumerate(all_days):
            if d in idx_s:
                sus_al[i] = sus[idx_s[d]]
            if d in idx_r:
                res_al[i] = res[idx_r[d]]

        total = np.nan_to_num(sus_al, nan=0) + np.nan_to_num(res_al, nan=0)
        day_tp1 = all_days - np.nanmin(all_days)

        return pd.DataFrame(
            {
                "sample": sample,
                "strategy": strategy,
                "day_tp1": day_tp1,
                "sus": sus_al,
                "res": res_al,
                "total": total,
            }
        )


def collect_all_samples() -> pd.DataFrame:
    all_dfs: List[pd.DataFrame] = []
    seen: set[str] = set()

    for plate in PLATES:
        h5_dir = ROOT_DIR / plate / "hdf5"
        if not h5_dir.exists():
            continue

        for p in sorted(h5_dir.glob("G418_a2a4_*_total.h5")):
            sample = re.sub(r"_total$", "", re.sub(r"^G418_a2a4_", "", p.stem))
            if sample in seen:
                continue

            df = load_counts_from_total_file(p)
            if df is not None and not df.empty:
                seen.add(sample)
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def normalize_to_day1_total(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["day_tp1"] == 0].groupby("sample")["total"].first()
    out = df.copy()
    out["baseline_total"] = out["sample"].map(baseline)
    valid_baseline = np.isfinite(out["baseline_total"]) & (out["baseline_total"] != 0)
    out["res_norm"] = np.nan
    out.loc[valid_baseline, "res_norm"] = (
        out.loc[valid_baseline, "res"] / out.loc[valid_baseline, "baseline_total"]
    )
    return out.drop(columns="baseline_total")


def apply_experiment_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Always apply full exclusions (EXCLUDE_SAMPLES[sample] is None).
    Optionally apply truncation cutoffs when USE_TRUNCATION_DAYS=True.
    """
    if USE_TRUNCATION_DAYS:
        return apply_sample_exclusions(df, column="sample", cutoff_column="day_tp1", strict=True)

    full_excluded = {s for s, v in EXCLUDE_SAMPLES.items() if v is None}
    before = len(df)
    df2 = df[~df["sample"].isin(full_excluded)].copy()
    print(f"[exclude] Full exclusions only (truncation OFF): {len(df2)} / {before} rows kept")
    return df2


def compute_ttp(df: pd.DataFrame, use_truncation_days: bool) -> pd.DataFrame:
    rows = []
    for sample, g in df.groupby("sample"):
        g = g.sort_values("day_tp1")
        hit = g.loc[g["res_norm"] >= PROGRESSION_THRESHOLD, "day_tp1"]
        progressed = not hit.empty

        # Only assign EXPERIMENT_END_DAY if the sample was not truncated by filters.
        cutoff = EXCLUDE_SAMPLES.get(sample, "not_listed")
        was_truncated_by_filter = use_truncation_days and cutoff not in ("not_listed", None)
        censored_ttp = np.nan if was_truncated_by_filter else EXPERIMENT_END_DAY

        rows.append(
            {
                "sample": sample,
                "strategy": g["strategy"].iloc[0],
                "ttp": float(hit.iloc[0]) if progressed else censored_ttp,
                "progressed": progressed,
            }
        )
    return pd.DataFrame(rows)


def load_simulation_ttp() -> pd.DataFrame:
    if not SIMULATION_SUMMARY_CSV.exists():
        print(f"[WARN] Simulation summary CSV not found: {SIMULATION_SUMMARY_CSV}")
        return pd.DataFrame(columns=["strategy", "ttp"])

    df = pd.read_csv(SIMULATION_SUMMARY_CSV)
    required_columns = {"protocol", "run", "TTP"}
    if not required_columns.issubset(df.columns):
        print(
            f"[WARN] Simulation summary CSV is missing required columns "
            f"{sorted(required_columns)}: {SIMULATION_SUMMARY_CSV}"
        )
        return pd.DataFrame(columns=["strategy", "ttp"])

    id_columns = ["protocol", "run"]
    if "well" in df.columns:
        id_columns.insert(1, "well")

    out = df.loc[:, id_columns + ["TTP"]].copy()
    out["strategy"] = out["protocol"].astype(str).str.strip()
    out["run_number"] = pd.to_numeric(out["run"], errors="coerce")
    out["ttp"] = pd.to_numeric(out["TTP"], errors="coerce")
    out = out[
        out["strategy"].isin(STRATEGIES_KEEP)
        & out["run_number"].notna()
        & ~out["run_number"].between(0, 9, inclusive="both")
        & (out["ttp"] != -1)
    ].dropna(subset=["ttp"]).copy()
    unique_columns = ["strategy", "run_number"]
    if "well" in out.columns:
        unique_columns.insert(1, "well")
    out = out.drop_duplicates(subset=unique_columns)
    print(f"[INFO] Simulation TTP: {len(out)} unique strategy/run rows loaded.")
    return out.loc[:, ["strategy", "ttp"]]


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def is_significant(p: float) -> bool:
    return bool(np.isfinite(p) and p < 5e-2)


def add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    stars: str,
    h_frac: float = BRACKET_H_FRAC,
    invert: bool = False,
) -> None:
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if (y1 > y0) else 1.0
    h = h_frac * yr

    xm = (x1 + x2) / 2.0

    if invert:
        ax.plot([x1, x1, x2, x2], [y + h, y, y, y + h],
                lw=BRACKET_LW, color="black", clip_on=False)
        ax.text(xm, y - h * STARS_Y_OFFSET_H_FRAC, stars,
                ha="center", va="top", fontsize=STARS_FONTSIZE, color="black")
    else:
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                lw=BRACKET_LW, color="black", clip_on=False)
        ax.text(xm, y + h * STARS_Y_OFFSET_H_FRAC, stars,
                ha="center", va="bottom", fontsize=STARS_FONTSIZE, color="black")


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------
def plot_boxplot(ttp: pd.DataFrame, simulation_ttp: pd.DataFrame) -> None:
    # Build groups dynamically from STRATEGIES_KEEP
    present_strats = set(ttp["strategy"]) | set(simulation_ttp["strategy"])
    labels = [s for s in STRATEGIES_KEEP if s in present_strats]
    groups = [ttp.loc[ttp["strategy"] == s, "ttp"].dropna() for s in labels]
    simulation_groups = [simulation_ttp.loc[simulation_ttp["strategy"] == s, "ttp"].dropna() for s in labels]
    labels_with_n = [s for s in labels]

    # Print simple sample-size summary for all groups
    print("[STATS] TTP group sizes:")
    for s, vals, sim_vals in zip(labels, groups, simulation_groups):
        print(f"  {s}: experimental n={len(vals)}, simulation n={len(sim_vals)}")

    # Pairwise Mann–Whitney U for all strategy pairs
    pairwise_results = []
    for i, j in combinations(range(len(labels)), 2):
        vals_i = groups[i]
        vals_j = groups[j]

        if len(vals_i) >= 2 and len(vals_j) >= 2:
            p = mannwhitneyu(vals_i.to_numpy(), vals_j.to_numpy(), alternative="two-sided").pvalue
            stars = p_to_stars(float(p)) if np.isfinite(p) else "n/a"
        else:
            p = np.nan
            stars = "n/a"

        n_text = f"n={len(vals_i)}/{len(vals_j)}"

        pairwise_results.append(
            {
                "i": i,
                "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "p": p,
                "stars": stars,
                "n_text": n_text,
            }
        )

    simulation_pairwise_results = []
    for i, j in combinations(range(len(labels)), 2):
        vals_i = simulation_groups[i]
        vals_j = simulation_groups[j]

        if len(vals_i) >= 2 and len(vals_j) >= 2:
            p = mannwhitneyu(vals_i.to_numpy(), vals_j.to_numpy(), alternative="two-sided").pvalue
            stars = p_to_stars(float(p)) if np.isfinite(p) else "n/a"
        else:
            p = np.nan
            stars = "n/a"

        simulation_pairwise_results.append(
            {
                "i": i,
                "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "p": p,
                "stars": stars,
                "n_text": f"n={len(vals_i)}/{len(vals_j)}",
            }
        )

    print("[STATS] Experimental pairwise Mann-Whitney U tests:")
    if pairwise_results:
        for r in pairwise_results:
            p_str = f"{r['p']:.6g}" if np.isfinite(r["p"]) else "n/a"
            print(
                f"  {r['label_i']} vs {r['label_j']}: "
                f"p={p_str}, stars={r['stars']}, {r['n_text']}"
            )
    else:
        print("  No valid pairwise comparisons.")

    print("[STATS] Simulation pairwise Mann-Whitney U tests:")
    if simulation_pairwise_results:
        for r in simulation_pairwise_results:
            p_str = f"{r['p']:.6g}" if np.isfinite(r["p"]) else "n/a"
            print(
                f"  {r['label_i']} vs {r['label_j']}: "
                f"p={p_str}, stars={r['stars']}, {r['n_text']}"
            )
    else:
        print("  No valid pairwise comparisons.")

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    x_positions = np.arange(1, len(labels) + 1, dtype=float)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_with_n)

    ticklabels = ax.get_xticklabels()
    for tl, strat in zip(ticklabels, labels):
        tl.set_color(get_color(strat, "#888888"))

    ax.set_ylabel("Time to progression [days]")
    ax.set_ylim(*YLIM)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", pad=1)

    for xpos, exp_vals, sim_vals in zip(x_positions, groups, simulation_groups):
        ax.text(
            xpos,
            -0.105,
            f"({len(exp_vals)}, {len(sim_vals)})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=N_FONTSIZE,
            clip_on=False,
        )

    # Overlay individual replicate values with slight x-jitter.
    rng = np.random.default_rng(7)
    for idx, vals in enumerate(groups, start=1):
        vals_arr = np.asarray(vals, dtype=float)
        strat = labels[idx - 1]
        col = get_color(strat, "#888888")
        if vals_arr.size > 0:
            x = idx + EXPERIMENT_X_OFFSET + rng.uniform(-POINT_JITTER, POINT_JITTER, size=vals_arr.size)
            ax.scatter(
                x,
                vals_arr,
                s=POINT_SIZE,
                c=col,
                alpha=POINT_ALPHA,
                edgecolors="black",
                linewidths=0.3,
                marker=EXPERIMENT_POINT_MARKER,
                zorder=3,
            )
            med = float(np.median(vals_arr))
            ax.hlines(
                med,
                idx + EXPERIMENT_X_OFFSET - MEDIAN_LINE_HALF_WIDTH,
                idx + EXPERIMENT_X_OFFSET + MEDIAN_LINE_HALF_WIDTH,
                colors=MEDIAN_COLOR,
                linewidth=1.1,
                zorder=4,
            )

        sim_vals_arr = np.asarray(simulation_groups[idx - 1], dtype=float)
        if sim_vals_arr.size > 0:
            x_sim = idx + SIMULATION_X_OFFSET + rng.uniform(-POINT_JITTER, POINT_JITTER, size=sim_vals_arr.size)
            ax.scatter(
                x_sim,
                sim_vals_arr,
                s=POINT_SIZE,
                c=col,
                alpha=POINT_ALPHA,
                edgecolors="black",
                linewidths=0.3,
                marker=SIMULATION_POINT_MARKER,
                zorder=3,
            )
            sim_med = float(np.median(sim_vals_arr))
            ax.hlines(
                sim_med,
                idx + SIMULATION_X_OFFSET - MEDIAN_LINE_HALF_WIDTH,
                idx + SIMULATION_X_OFFSET + MEDIAN_LINE_HALF_WIDTH,
                colors=MEDIAN_COLOR,
                linewidth=1.1,
                zorder=4,
            )

    # Simulation significance brackets: bottom, inverted orientation.
    simulation_significant_results = [
        r for r in simulation_pairwise_results if is_significant(r["p"])
    ]
    if simulation_significant_results:
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        bottom_margin = BRACKET_BOTTOM_MARGIN_FRAC * yr
        step = BRACKET_STEP_FRAC * yr

        for k, r in enumerate(reversed(simulation_significant_results)):
            y_bracket = y0 + bottom_margin + k * step
            add_sig_bracket(
                ax,
                r["i"] + 1 + SIMULATION_X_OFFSET,
                r["j"] + 1 + SIMULATION_X_OFFSET,
                y_bracket,
                stars=r["stars"],
                invert=True,
            )

    # Experimental significance brackets: original top placement.
    significant_results = [r for r in pairwise_results if is_significant(r["p"])]
    if significant_results:
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        top_margin = BRACKET_TOP_MARGIN_FRAC * yr
        step = BRACKET_STEP_FRAC * yr

        for k, r in enumerate(significant_results):
            y_bracket = y1 - top_margin - k * step
            add_sig_bracket(
                ax,
                r["i"] + 1 + EXPERIMENT_X_OFFSET,
                r["j"] + 1 + EXPERIMENT_X_OFFSET,
                y_bracket,
                stars=r["stars"],
            )

    axes_w_mm, axes_h_mm = ft.axes_size_mm(ax)
    print(f"[FIGURE] Axes box: {axes_w_mm:.2f} x {axes_h_mm:.2f} mm")
    ft.savefig(fig, OUT_PDF_EXTRA)
    print(f"[OK] Saved: {OUT_PDF_EXTRA}")

    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    df = collect_all_samples()
    if df.empty:
        print("[ERROR] No samples loaded. Check paths / plate folders.")
        return

    df = normalize_to_day1_total(df)

    # Apply experiment filters according to USE_TRUNCATION_DAYS
    df = apply_experiment_filters(df)

    # keep only requested strategies
    df = df[df["strategy"].isin(STRATEGIES_KEEP)].copy()

    ttp = compute_ttp(df, use_truncation_days=USE_TRUNCATION_DAYS)
    simulation_ttp = load_simulation_ttp()

    for strat in STRATEGIES_KEEP:
        sub = ttp[ttp["strategy"] == strat]
        if len(sub) > 0:
            print(f"{strat}: {int(sub['progressed'].sum())}/{len(sub)} progressed")

    plot_boxplot(ttp, simulation_ttp)


if __name__ == "__main__":
    main()
