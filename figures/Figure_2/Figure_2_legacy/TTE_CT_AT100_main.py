#!/usr/bin/env python3
"""
TL;DR
-----
This script computes and plots time to escape (TTE) for CT and AT100
replicates using the escape-summary branch of the pipeline.

Detailed
--------
Inputs:
- Escape-summary CSV data produced downstream of the escape-front pipeline.
- Replicate exclusion/truncation rules from `experiment_filters.py`.

Outputs:
- A TTE comparison figure and the associated pairwise survival-style
  significance statistics.

How it works:
- Loads escape timing information per replicate and identifies the relevant
  sample/escape columns.
- Applies the shared exclusion and truncation logic so invalid or censored
  replicates are handled consistently.
- Builds replicate-level TTE tables and compares CT versus AT100 with the
  script’s significance logic.

Pipeline role:
- Figure 1 plotting script for escape timing.
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

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, mannwhitneyu

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared import experiment_filters  # we need EXCLUDE_SAMPLES + same logic


# ----------------------------- USER --------------------------------
DATA_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Data")

ESCAPE_SUMMARY_CSV = DATA_DIR / "escape_front_batch_summary_all_res_within_sens.csv"

OUT_DIR_EXTRA = ft.paper_figure_data_dir(__file__)
OUT_DIR_EXTRA.mkdir(parents=True, exist_ok=True)
pdf_path_extra = OUT_DIR_EXTRA / "Fig. 2 TTE CT & AT100.pdf"
SIMULATION_SUMMARY_CSV = DATA_DIR / "Maja" / "combined_results.csv"

STRATEGIES_KEEP = ["CT", "AT50", "AT100"]
EXPERIMENT_END_DAY = 96.0  # non-escapers (not excluded/truncated) get this TTE value

YLIM = (0, 112)

# ---- Typography / strokes (match TTP) ----
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

# Bracket + annotation styling (match TTP)
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
# -------------------------------------------------------------------


# ---------------------- STRATEGY COLORS IMPORT ----------------------
from shared.strategy_colors import get_color
# -------------------------------------------------------------------


def parse_strategy(sample: str) -> str:
    """Expected sample naming like 'A1_AT100-A1' -> strategy between '_' and '-'."""
    try:
        return sample.split("_", 1)[1].split("-", 1)[0]
    except Exception:
        return ""


def find_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Robustly identify the sample-id column and the escape-day column."""
    sample_candidates = ["sample", "replicate", "replicate_name", "well", "id", "sample_id"]
    escape_candidates = ["first_escape_day", "escape_day", "escapeDay", "escape", "tte", "TTE"]

    sample_col = next((c for c in sample_candidates if c in df.columns), None)
    escape_col = next((c for c in escape_candidates if c in df.columns), None)

    if sample_col is None:
        raise KeyError(f"Could not find a sample column. Columns present: {list(df.columns)}")
    if escape_col is None:
        raise KeyError(f"Could not find an escape-day column. Columns present: {list(df.columns)}")

    return sample_col, escape_col


def get_filter_rule(sample: str) -> Tuple[str, Optional[float]]:
    """
    Returns (status, cutoff)
      status: "keep", "exclude", or "truncate"
      cutoff: truncation cutoff day if truncate, else None
    """
    rules: Dict[str, Any] = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    if sample not in rules:
        return "keep", None

    val = rules[sample]
    if val is None:
        return "exclude", None
    try:
        return "truncate", float(val)
    except Exception:
        return "keep", None


def apply_filters_and_build_tte(df: pd.DataFrame, sample_col: str, escape_col: str) -> pd.DataFrame:
    """
    Applies experiment_filters rules and returns a replicate-level table:
      sample, strategy, escape_day, tte, escaped(bool), excluded_reason

    Rules:
    - Fully excluded/truncated-before-event replicates -> tte = NaN
    - Non-escapers not excluded/truncated -> tte = EXPERIMENT_END_DAY
    """
    out_rows = []

    for sample_key, g in df.groupby(sample_col):
        sample = str(sample_key)
        if "strategy" in g.columns:
            strat_vals = g["strategy"].dropna().astype(str)
            strat = strat_vals.iloc[0] if not strat_vals.empty else parse_strategy(sample)
        else:
            strat = parse_strategy(sample)
        if strat not in STRATEGIES_KEEP:
            continue

        escape_series = pd.to_numeric(g[escape_col], errors="coerce")
        finite_escape = escape_series[np.isfinite(escape_series)]
        escape_day = float(finite_escape.min()) if not finite_escape.empty else np.nan
        # For non-escapers, use each sample's latest observed day from this CSV.
        if "day" in g.columns:
            day_series = pd.to_numeric(g["day"], errors="coerce")
            finite_days = day_series[np.isfinite(day_series)]
            last_observed_day = float(finite_days.max()) if not finite_days.empty else EXPERIMENT_END_DAY
        else:
            last_observed_day = EXPERIMENT_END_DAY

        if "escape_detected" in g.columns:
            escape_detected = (
                g["escape_detected"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
                .any()
            )
        else:
            escape_detected = bool(np.isfinite(escape_day))

        include_non_escaper = True
        if not escape_detected and {"n_frames_total", "n_frames_processed"}.issubset(g.columns):
            n_total_series = pd.to_numeric(g["n_frames_total"], errors="coerce")
            n_proc_series = pd.to_numeric(g["n_frames_processed"], errors="coerce")
            n_total_valid = n_total_series[np.isfinite(n_total_series)]
            n_proc_valid = n_proc_series[np.isfinite(n_proc_series)]
            if n_total_valid.empty or n_proc_valid.empty:
                include_non_escaper = False
            else:
                include_non_escaper = bool(float(n_total_valid.iloc[0]) == float(n_proc_valid.iloc[0]))

        status, cutoff = get_filter_rule(sample)

        if status == "exclude":
            out_rows.append(
                {
                    "sample": sample,
                    "strategy": strat,
                    "escape_day": escape_day,
                    "tte": np.nan,
                    "escaped": False,
                    "excluded_reason": "excluded_by_experiment_filters",
                }
            )
            continue

        if status == "truncate" and cutoff is not None:
            if not np.isfinite(escape_day) or escape_day > cutoff:
                out_rows.append(
                    {
                        "sample": sample,
                        "strategy": strat,
                        "escape_day": escape_day,
                        "tte": np.nan,
                        "escaped": False,
                        "excluded_reason": f"no_escape_within_truncation_cutoff_{cutoff:g}",
                    }
                )
                continue

        if escape_detected and np.isfinite(escape_day):
            out_rows.append(
                {
                    "sample": sample,
                    "strategy": strat,
                    "escape_day": escape_day,
                    "tte": float(escape_day),
                    "escaped": True,
                    "excluded_reason": "",
                }
            )
        elif not include_non_escaper:
            out_rows.append(
                {
                    "sample": sample,
                    "strategy": strat,
                    "escape_day": escape_day,
                    "tte": np.nan,
                    "escaped": False,
                    "excluded_reason": "non_escaper_incomplete_frames",
                }
            )
        else:
            out_rows.append(
                {
                    "sample": sample,
                    "strategy": strat,
                    "escape_day": escape_day,
                    "tte": last_observed_day,
                    "escaped": False,
                    "excluded_reason": "no_escape_in_csv",
                }
            )

    return pd.DataFrame(out_rows)


def load_simulation_tte() -> pd.DataFrame:
    if not SIMULATION_SUMMARY_CSV.exists():
        print(f"[WARN] Simulation summary CSV not found: {SIMULATION_SUMMARY_CSV}")
        return pd.DataFrame(columns=["strategy", "tte"])

    df = pd.read_csv(SIMULATION_SUMMARY_CSV)
    required_columns = {"protocol", "run", "TTE"}
    if not required_columns.issubset(df.columns):
        print(
            f"[WARN] Simulation summary CSV is missing required columns "
            f"{sorted(required_columns)}: {SIMULATION_SUMMARY_CSV}"
        )
        return pd.DataFrame(columns=["strategy", "tte"])

    id_columns = ["protocol", "run"]
    if "well" in df.columns:
        id_columns.insert(1, "well")

    out = df.loc[:, id_columns + ["TTE"]].copy()
    out["strategy"] = out["protocol"].astype(str).str.strip()
    out["run_number"] = pd.to_numeric(out["run"], errors="coerce")
    out["tte"] = pd.to_numeric(out["TTE"], errors="coerce")
    out = out[
        out["strategy"].isin(STRATEGIES_KEEP)
        & out["run_number"].notna()
        & ~out["run_number"].between(0, 9, inclusive="both")
        & (out["tte"] != -1)
    ].dropna(subset=["tte"]).copy()
    unique_columns = ["strategy", "run_number"]
    if "well" in out.columns:
        unique_columns.insert(1, "well")
    out = out.drop_duplicates(subset=unique_columns)
    print(f"[INFO] Simulation TTE: {len(out)} unique strategy/run rows loaded.")
    return out.loc[:, ["strategy", "tte"]]


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


def logrank_pvalue(
    durations_a: np.ndarray,
    events_a: np.ndarray,
    durations_b: np.ndarray,
    events_b: np.ndarray,
) -> float:
    """
    Two-sample log-rank test p-value (chi-square with 1 df).
    durations_*: observed/censor times
    events_*: 1=event occurred, 0=right-censored
    """
    t_a = np.asarray(durations_a, dtype=float)
    t_b = np.asarray(durations_b, dtype=float)
    e_a = np.asarray(events_a, dtype=bool)
    e_b = np.asarray(events_b, dtype=bool)

    valid_a = np.isfinite(t_a)
    valid_b = np.isfinite(t_b)
    t_a = t_a[valid_a]
    t_b = t_b[valid_b]
    e_a = e_a[valid_a]
    e_b = e_b[valid_b]

    if t_a.size < 2 or t_b.size < 2:
        return np.nan

    event_times = np.unique(np.concatenate([t_a[e_a], t_b[e_b]]))
    if event_times.size == 0:
        return np.nan

    oe_sum = 0.0
    var_sum = 0.0

    for t in event_times:
        n1 = int(np.sum(t_a >= t))
        n2 = int(np.sum(t_b >= t))
        n = n1 + n2
        if n <= 1:
            continue

        d1 = int(np.sum((t_a == t) & e_a))
        d2 = int(np.sum((t_b == t) & e_b))
        d = d1 + d2
        if d == 0:
            continue

        e1 = d * (n1 / n)
        var = (n1 * n2 * d * (n - d)) / (n * n * (n - 1))
        oe_sum += (d1 - e1)
        var_sum += var

    if var_sum <= 0:
        return np.nan

    chisq = (oe_sum * oe_sum) / var_sum
    return float(chi2.sf(chisq, df=1))


def add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    stars: str,
    h_frac: float = BRACKET_H_FRAC,
    invert: bool = False,
) -> None:
    """Bracket with stars above and n below (match TTP small-panel styling)."""
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


def plot_tte_boxplot(tte_df: pd.DataFrame, simulation_tte: pd.DataFrame) -> None:
    """Boxplot of TTE for selected strategies (including end-censored non-escapers)."""
    present_strats = set(tte_df["strategy"]) | set(simulation_tte["strategy"])
    labels = [s for s in STRATEGIES_KEEP if s in present_strats]
    group_frames = [
        tte_df.loc[tte_df["strategy"] == s, ["tte", "escaped"]].dropna(subset=["tte"]).copy()
        for s in labels
    ]
    groups = [g["tte"].to_numpy(dtype=float) for g in group_frames]
    simulation_groups = [simulation_tte.loc[simulation_tte["strategy"] == s, "tte"].dropna() for s in labels]
    labels_with_n = [s for s in labels]

    if len(labels) < 2:
        print("[WARN] Need at least two strategies in STRATEGIES_KEEP with data.")
        return

    # Log rows excluded from boxplot (tte is NaN after exclusions/truncations)
    excluded_from_plot = tte_df[tte_df["tte"].isna()].copy()
    if not excluded_from_plot.empty:
        print("\n[INFO] Replicates excluded from boxplot by experiment filters:")
        for strat in labels:
            sub = excluded_from_plot[excluded_from_plot["strategy"] == strat]
            if sub.empty:
                continue
            print(f"  {strat}:")
            for _, r in sub.iterrows():
                reason = r.get("excluded_reason", "")
                print(f"    - {r['sample']}  ({reason})")
        print("")

    # Pairwise log-rank tests (handles right-censoring) for all strategy pairs.
    pairwise_results = []
    for i, j in combinations(range(len(labels)), 2):
        g_i = group_frames[i]
        g_j = group_frames[j]

        if len(g_i) >= 2 and len(g_j) >= 2:
            p = logrank_pvalue(
                durations_a=g_i["tte"].to_numpy(dtype=float),
                events_a=g_i["escaped"].to_numpy(dtype=bool),
                durations_b=g_j["tte"].to_numpy(dtype=float),
                events_b=g_j["escaped"].to_numpy(dtype=bool),
            )
            stars = p_to_stars(float(p)) if np.isfinite(p) else "n/a"
        else:
            p = np.nan
            stars = "n/a"

        pairwise_results.append(
            {
                "i": i,
                "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "p": p,
                "stars": stars,
                "n_text": f"n={len(g_i)}/{len(g_j)}",
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

    print("[STATS] Pairwise log-rank tests:")
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

    # Summary counts
    for strat in labels:
        sub = tte_df[tte_df["strategy"] == strat]
        escaped = int(sub["escaped"].sum())
        total = int(len(sub))
        in_plot = int(sub["tte"].notna().sum())
        print(f"[INFO] {strat}: escaped {escaped}/{total}, in_boxplot {in_plot}/{total}.")

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

    # Color x tick labels to match
    ticklabels = ax.get_xticklabels()
    for tl, strat in zip(ticklabels, labels):
        tl.set_color(get_color(strat, "#888888"))

    ax.set_ylabel("Time to escape [days]")
    ax.set_ylim(*YLIM)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", pad=1)

    for xpos, group_frame, sim_vals in zip(x_positions, group_frames, simulation_groups):
        ax.text(
            xpos,
            -0.105,
            f"({len(group_frame)}, {len(sim_vals)})",
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
    ft.savefig(fig, pdf_path_extra)

    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"[OK] Saved: {pdf_path_extra}")


def main() -> None:
    if not ESCAPE_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Escape summary CSV not found:\n  {ESCAPE_SUMMARY_CSV}\n"
            "Expected: Data\\escape_front_batch_summary_all_res_within_sens.csv"
        )

    df = pd.read_csv(ESCAPE_SUMMARY_CSV)
    sample_col, escape_col = find_columns(df)

    tte_df = apply_filters_and_build_tte(df, sample_col=sample_col, escape_col=escape_col)
    simulation_tte = load_simulation_tte()

    if tte_df.empty:
        print(f"[WARN] No replicates found for STRATEGIES_KEEP={STRATEGIES_KEEP}.")
        return

    plot_tte_boxplot(tte_df, simulation_tte)


if __name__ == "__main__":
    main()
