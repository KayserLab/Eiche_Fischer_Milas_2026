"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - \\]+?
# - D:\2025_08_Co
# Upstream/related script reference(s):
# - analysis/supplementary/Figure_5_Escape_Day.py
# - and shared/figtools.py
# - shared/experiment_filters.py
# - import matplotlib.py
# - shared/strategy_colors.py
# =============================================================================
TL;DR
-----
This script compares the normalized resistant-cell burden at escape across
strategies.

Detailed
--------
Inputs:
- Raw `Results/*.txt` replicate trajectories.
- Escape-summary data and shared replicate exclusion/truncation rules.

Outputs:
- A supplementary boxplot-style comparison of resistant burden at escape.
- Printed/statistical summaries for the pairwise comparisons.

How it works:
- Loads normalized resistant trajectories and joins them to replicate escape
  timing.
- Extracts the normalized resistant count at the escape event for each valid
  replicate.
- Compares those values across strategies and visualizes the resulting
  distributions.

Pipeline role:
- Supplementary plotting endpoint for resistant burden at the moment of escape.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Plate_*/Results/*.txt and the escape-day summary used by
# analysis/supplementary/Figure_5_Escape_Day.py.
# Scripts: analysis/supplementary/Figure_5_Escape_Day.py, shared/experiment_filters.py, shared/strategy_colors.py,
# and shared/figtools.py.
# Outputs: normalized resistant-at-escape comparison PDF and statistics.

import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared import experiment_filters
from shared.data_paths import EXP_RAW_DATA
from analysis.legacy import Figure_5_Escape_Day as figure4_escape
from shared.strategy_colors import COLOR_MAP


DPI = 300
STRATEGIES = ["CT", "AT50", "AT100"]
DATA_ROOT = EXP_RAW_DATA
PLATES = [f"Plate_{i}" for i in range(1, 6)]
OUT_DIR = ft.paper_figure_data_dir(__file__)
OUT_PDF = OUT_DIR / "Normalized_Resistant_Count_At_Escape_Strategy_Comparison.pdf"
RESULTS_RE = re.compile(
    r"^Results_G418_a2a4_(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$",
    re.IGNORECASE,
)
TOKEN_DAY_RE = re.compile(r"_d(?P<day>\d{1,3})\b", re.IGNORECASE)
IGNORE_FILENAME_PARTS = ("overexposed", "seg")



# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 62.0
FIGURE_AXES_HEIGHT_MM = 45.0
FIGURE_MARGINS_MM = (15.0, 8.0, 12.0, 8.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0

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


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


@dataclass(frozen=True)
class ResultRow:
    plate: str
    sample: str
    strategy: str
    day: int
    sensitive: int
    resistant: int


def parse_day(token: str) -> int:
    match = TOKEN_DAY_RE.search(token)
    if not match:
        raise ValueError(f"Cannot parse day from token: {token}")
    return int(match.group("day"))


def iter_result_rows() -> Iterable[ResultRow]:
    for plate in PLATES:
        results_dir = DATA_ROOT / plate / "Results"
        if not results_dir.is_dir():
            continue

        for path in sorted(results_dir.glob("*.txt")):
            if any(tag in path.name.lower() for tag in IGNORE_FILENAME_PARTS):
                continue

            match = RESULTS_RE.match(path.name)
            if not match:
                continue

            strategy = str(match.group("strategy"))
            if strategy not in STRATEGIES:
                continue

            well = match.group("well").upper()
            sample = f"{well}_{strategy}-{well}"

            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    try:
                        yield ResultRow(
                            plate=plate,
                            sample=sample,
                            strategy=strategy,
                            day=parse_day(parts[0]),
                            sensitive=int(parts[1]),
                            resistant=int(parts[2]),
                        )
                    except ValueError:
                        continue


def normalize_to_baseline_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["total"] > 0].copy()

    base0 = (
        df[df["day"] == 0]
        .groupby(["plate", "sample"], as_index=False)["total"]
        .first()
        .rename(columns={"total": "base0"})
    )
    base_earliest = (
        df.sort_values(["plate", "sample", "day"])
        .groupby(["plate", "sample"], as_index=False)
        .first()[["plate", "sample", "total"]]
        .rename(columns={"total": "base_earliest"})
    )

    out = (
        df.merge(base0, on=["plate", "sample"], how="left")
        .merge(base_earliest, on=["plate", "sample"], how="left")
    )
    out["base_total"] = out["base0"].fillna(out["base_earliest"])
    out = out[out["base_total"] > 0].copy()
    out["resistant_norm"] = out["resistant"] / out["base_total"]
    return out


def load_results() -> pd.DataFrame:
    df = pd.DataFrame([row.__dict__ for row in iter_result_rows()])
    if df.empty:
        raise RuntimeError("No rows loaded from Results/*.txt.")

    df["total"] = df["sensitive"] + df["resistant"]
    df = experiment_filters.apply_sample_exclusions(
        df,
        exclude_map=experiment_filters.EXCLUDE_SAMPLES,
        column="sample",
        cutoff_column="day",
        strict=False,
    )
    df = normalize_to_baseline_total(df)
    df = df.sort_values(["strategy", "plate", "sample", "day"]).reset_index(drop=True)
    return df


def load_escape_summary() -> pd.DataFrame:
    summary = pd.read_csv(figure4_escape.SUM_CSV)
    if "replicate" not in summary.columns and "sample" in summary.columns:
        summary["replicate"] = summary["sample"]
    if "status" in summary.columns:
        summary = summary[summary["status"].astype(str).str.lower() == "ok"].copy()

    summary["strategy"] = summary["replicate"].astype(str).map(figure4_escape.parse_strategy)
    if "escape_day" in summary.columns:
        summary["escape_day"] = figure4_escape.safe_numeric(summary["escape_day"])
    elif "first_escape_day" in summary.columns:
        summary["escape_day"] = figure4_escape.safe_numeric(summary["first_escape_day"])
    else:
        raise ValueError("Escape summary is missing escape_day/first_escape_day.")

    keep_cols = [
        col
        for col in [
            "plate",
            "replicate",
            "strategy",
            "escape_day",
            "escape_detected",
            "n_frames_total",
            "n_frames_processed",
        ]
        if col in summary.columns
    ]
    return summary[keep_cols].copy()


def build_resistant_norm_at_escape_table(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in summary_df.iterrows():
        plate = str(row["plate"])
        sample = str(row["replicate"])
        strategy = str(row["strategy"])
        status, cutoff = figure4_escape.get_filter_rule(sample)
        escape_day = pd.to_numeric(pd.Series([row.get("escape_day")]), errors="coerce").iloc[0]

        if "escape_detected" in row.index:
            escaped = str(row["escape_detected"]).strip().lower() in {"true", "1", "yes"}
        else:
            escaped = bool(np.isfinite(escape_day))

        if status == "exclude":
            res_norm_at_escape = np.nan
        elif status == "truncate" and cutoff is not None and (not np.isfinite(escape_day) or float(escape_day) > float(cutoff)):
            res_norm_at_escape = np.nan
        elif not (escaped and np.isfinite(escape_day)):
            res_norm_at_escape = np.nan
        else:
            g = results_df[
                (results_df["plate"].astype(str) == plate)
                & (results_df["sample"].astype(str) == sample)
                & (pd.to_numeric(results_df["day"], errors="coerce") == float(escape_day))
            ].copy()
            if g.empty:
                res_norm_at_escape = np.nan
            else:
                res_norm_at_escape = float(pd.to_numeric(g["resistant_norm"], errors="coerce").iloc[0])

        rows.append(
            {
                "plate": plate,
                "sample": sample,
                "strategy": strategy,
                "resistant_norm_at_escape": res_norm_at_escape,
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = float(pvals[idx])
        holm = (m - rank) * raw
        running_max = max(running_max, holm)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def prepare_escape_values(results_df: pd.DataFrame, escape_summary: pd.DataFrame) -> pd.DataFrame:
    df = build_resistant_norm_at_escape_table(results_df, escape_summary).copy()
    df["resistant_norm_at_escape"] = pd.to_numeric(df["resistant_norm_at_escape"], errors="coerce")
    df = df[np.isfinite(df["resistant_norm_at_escape"])].copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGIES, ordered=True)
    df = df.sort_values(["strategy", "plate", "sample"]).reset_index(drop=True)
    return df


def make_figure(plot_df: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    positions = np.arange(1, len(STRATEGIES) + 1, dtype=float)
    data = [
        plot_df.loc[plot_df["strategy"] == strategy, "resistant_norm_at_escape"].to_numpy(dtype=float)
        for strategy in STRATEGIES
    ]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"edgecolor": "black", "linewidth": 1.0},
    )

    for patch, strategy in zip(bp["boxes"], STRATEGIES):
        patch.set_facecolor(COLOR_MAP[strategy])
        patch.set_alpha(0.45)

    for pos, strategy, vals in zip(positions, STRATEGIES, data):
        if vals.size == 0:
            continue
        offsets = np.linspace(-0.12, 0.12, vals.size) if vals.size > 1 else np.array([0.0])
        ax.scatter(
            pos + offsets,
            vals,
            s=16,
            color=COLOR_MAP[strategy],
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Normalized resistant count at escape")
    ax.set_title("Resistant count at escape", fontsize=7)
    ax.set_xlim(0.5, len(STRATEGIES) + 0.5)
    ax.set_ylim(bottom=0.0)
    style_axis(ax)

    ft.savefig(fig, out_pdf)
    plt.close(fig)
    print(f"[save] {out_pdf}")


def print_stats(plot_df: pd.DataFrame) -> None:
    print("[stats] Normalized resistant count at escape across CT, AT50, and AT100")
    print(
        "        Each replicate contributes exactly one value: the normalized resistant count at its actual Figure 4 escape day."
    )
    print(
        "        Replicates without a true escape day, or without a matching resistant-count value at that day, are excluded."
    )

    groups: list[np.ndarray] = []
    by_strategy: dict[str, np.ndarray] = {}
    valid_for_test = True
    for strategy in STRATEGIES:
        vals = (
            plot_df.loc[plot_df["strategy"] == strategy, "resistant_norm_at_escape"]
            .to_numpy(dtype=float)
        )
        vals = vals[np.isfinite(vals)]
        by_strategy[strategy] = vals
        groups.append(vals)
        if vals.size == 0:
            valid_for_test = False
            print(f"  {strategy}: n=0, mean=nan, median=nan, SD=nan")
        else:
            mean = float(np.mean(vals))
            median = float(np.median(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            print(f"  {strategy}: n={vals.size}, mean={mean:.4f}, median={median:.4f}, SD={sd:.4f}")

    if not valid_for_test:
        print("  Kruskal-Wallis result: not computed because at least one strategy has no finite values.")
        return

    stat, pval = kruskal(*groups, nan_policy="omit")
    print(f"  Kruskal-Wallis result: H={float(stat):.4f}, p={float(pval):.4g}")
    print(
        "  Interpretation: this tests whether the distribution of normalized resistant count at escape differs between the three strategies."
    )

    if float(pval) < 0.05:
        pair_labels: list[str] = []
        pair_pvals: list[float] = []
        for a, b in itertools.combinations(STRATEGIES, 2):
            _, raw_p = mannwhitneyu(by_strategy[a], by_strategy[b], alternative="two-sided")
            pair_labels.append(f"{a} vs {b}")
            pair_pvals.append(float(raw_p))
        adj_pvals = holm_adjust(pair_pvals)
        print("  Post hoc pairwise Mann-Whitney U tests with Holm correction:")
        for pair_label, raw_p, adj_p in zip(pair_labels, pair_pvals, adj_pvals):
            print(f"    {pair_label}: raw p={raw_p:.4g}, Holm-adjusted p={adj_p:.4g}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = load_results()
    escape_summary = load_escape_summary()
    plot_df = prepare_escape_values(results_df, escape_summary)

    make_figure(plot_df, OUT_PDF)
    print("[counts] replicates with finite normalized resistant count at true escape day:")
    for strategy in STRATEGIES:
        n_rep = int(plot_df.loc[plot_df["strategy"] == strategy, "sample"].nunique())
        print(f"  {strategy}: {n_rep}")
    print_stats(plot_df)


if __name__ == "__main__":
    main()
