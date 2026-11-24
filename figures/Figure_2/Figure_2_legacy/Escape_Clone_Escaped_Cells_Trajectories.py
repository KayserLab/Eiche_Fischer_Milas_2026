"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
Plot clone- or well-level escaped-resistant-cell trajectories over time.

Detailed
--------
Inputs:
- Resistant clone mask stacks.
- Escape-front point CSVs from the all_res_within_sens branch.
- Resistant-cell HDF5 point clouds.
- Shared replicate exclusion/truncation rules.

Outputs:
- A strategy-coloured median trajectory plot for escaped resistant cells per
  clone or per well.
- A matching plot for the fraction of resistant cells in each clone that are
  escaped.
- Detailed clone/timepoint and summary CSVs used to make the plots.

How it works:
- Uses the saved escape-front point CSVs as escaped resistant-cell coordinates.
- Counts total and escaped resistant-cell coordinates per clone or per well.
- Aggregates values by strategy and day using the median and a
  bootstrap 95% confidence interval.

Pipeline role:
- Figure 2 support script for time-resolved clone-level escape burden.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "legacy"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from shared import figtools as ft
from shared import experiment_filters
from analysis.legacy import Escape_Fig_Build_escape_areas as escape_area
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name, normalize_strategy


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
ESCAPE_SUMMARY_CSV = (
    DATA_ROOT
    / "Figures"
    / "Escape_Quantification"
    / "escape_front_batch_summary_all_res_within_sens.csv"
)
OUT_DIR = ft.paper_figure_data_dir(__file__) / "Escape_Clone_Escaped_Cells_Trajectories"
EXCLUDED_PLOT_STRATEGIES = {"KPP-Agent", "Pulse"}

DEFINITION = "all_res_within_sens"
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 17
DPI = 300

FIGURE_AXES_WIDTH_MM = 62.0
FIGURE_AXES_HEIGHT_MM = 42.0
FIGURE_MARGINS_MM = (15.0, 5.0, 13.0, 5.0)
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
LINEWIDTH = 1.1
CI_ALPHA = 0.18

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


@dataclass(frozen=True)
class CloneTrajectoryRow:
    plate: str
    sample: str
    strategy: str
    clone_id: int
    t: int
    day: int
    days_since_escape: float
    resistant_count: int
    escaped_count: int
    escaped_fraction: float
    escape_detected: bool
    escape_day: float


@dataclass(frozen=True)
class WellTrajectoryRow:
    plate: str
    sample: str
    strategy: str
    t: int
    day: int
    days_since_escape: float
    resistant_count: int
    escaped_count: int
    escaped_fraction: float
    escape_detected: bool
    escape_day: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot median clone-level escaped-cell trajectories by strategy. "
            "Experiment filters from shared/experiment_filters.py are applied by default."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Dataset root containing Plate_* folders. Default: {DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory. Default: {OUT_DIR}",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=(
            "Optional canonical or raw strategy names to include. "
            "Default: all strategies available in the data."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=["clone", "well"],
        default="clone",
        help="Aggregate escaped cells per clone or per well. Default: clone.",
    )
    parser.add_argument(
        "--time-axis",
        choices=["day", "since-escape"],
        default="day",
        help="Use absolute experiment day or days since replicate escape. Default: day.",
    )
    parser.add_argument(
        "--include-zero-resistant",
        action="store_true",
        help=(
            "Keep clone/timepoints with resistant_count == 0 as escaped_count=0. "
            "By default they are excluded from both summaries."
        ),
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Ignore shared/experiment_filters.py full exclusions and truncation days.",
    )
    parser.add_argument(
        "--carry-forward-censored",
        action="store_true",
        help=(
            "For well-wise summaries, add extra plots where each well's last valid "
            "value is carried forward to later timepoints after censoring."
        ),
    )
    parser.add_argument(
        "--max-replicates",
        type=int,
        default=None,
        help="Optional debug limit on processed replicates after filtering.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=BOOTSTRAP_REPEATS,
        help=f"Bootstrap repeats for the median CI. Default: {BOOTSTRAP_REPEATS}",
    )
    return parser.parse_args()


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)
    ax.grid(False)


def load_escape_summary(summary_csv: Path) -> dict[str, dict[str, float | bool]]:
    if not summary_csv.exists():
        print(f"[warn] escape summary not found: {summary_csv}")
        return {}

    df = pd.read_csv(summary_csv)
    if "replicate" not in df.columns and "sample" in df.columns:
        df["replicate"] = df["sample"]
    if "replicate" not in df.columns:
        return {}

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    if "escape_day" in df.columns:
        escape_day = pd.to_numeric(df["escape_day"], errors="coerce")
    elif "first_escape_day" in df.columns:
        escape_day = pd.to_numeric(df["first_escape_day"], errors="coerce")
    else:
        escape_day = pd.Series(np.nan, index=df.index)

    if "escape_detected" in df.columns:
        escaped = (
            df["escape_detected"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    else:
        escaped = np.isfinite(escape_day)

    out: dict[str, dict[str, float | bool]] = {}
    for sample, day, esc in zip(df["replicate"].astype(str), escape_day, escaped):
        out[str(sample)] = {
            "escape_day": float(day) if np.isfinite(day) else np.nan,
            "escape_detected": bool(esc),
        }
    return out


def filter_rule(sample: str, include_excluded: bool) -> tuple[str, float | None]:
    if include_excluded:
        return "keep", None
    if sample not in experiment_filters.EXCLUDE_SAMPLES:
        return "keep", None
    cutoff = experiment_filters.EXCLUDE_SAMPLES[sample]
    if cutoff is None:
        return "drop", None
    return "truncate", float(cutoff)


def valid_clone_ids(mask_stack: np.ndarray) -> np.ndarray:
    if mask_stack.shape[0] > 1:
        labels = np.unique(mask_stack[1])
    else:
        labels = np.unique(mask_stack)
    labels = labels[labels >= 2]
    return labels.astype(int)


def count_resistant_by_clone(
    frame: np.ndarray,
    xy: np.ndarray,
    clone_ids: np.ndarray,
) -> dict[int, int]:
    counts = {int(cid): 0 for cid in clone_ids}
    if xy.size == 0 or clone_ids.size == 0:
        return counts

    h, w = frame.shape
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(inb):
        return counts

    labels = frame[yi[inb], xi[inb]].astype(int)
    keep = labels >= 2
    labels = labels[keep]

    if labels.size == 0:
        return counts

    ids_set = set(int(cid) for cid in clone_ids)
    for label, count in zip(*np.unique(labels, return_counts=True)):
        label_i = int(label)
        if label_i in ids_set:
            counts[label_i] = int(count)

    return counts


def iter_clone_trajectory_rows(
    data_root: Path,
    escape_lookup: dict[str, dict[str, float | bool]],
    *,
    include_excluded: bool,
    include_zero_resistant: bool,
    strategy_filter: set[str] | None = None,
    max_replicates: int | None = None,
) -> Iterable[CloneTrajectoryRow]:
    escape_area.DATA_ROOT = data_root
    replicates = escape_area.discover_replicates()
    print(f"[info] discovered replicates with escape inputs: {len(replicates)}", flush=True)

    processed_replicates = 0
    for plate, sample, h5_path, mask_path, points_csv in replicates:
        raw_strategy = escape_area.parse_strategy(sample)
        strategy = normalize_strategy(raw_strategy)
        if strategy_filter is not None and strategy not in strategy_filter:
            continue
        if max_replicates is not None and processed_replicates >= max_replicates:
            print(f"[info] stopped after --max-replicates={max_replicates}", flush=True)
            break

        status, cutoff_day = filter_rule(sample, include_excluded=include_excluded)
        if status == "drop":
            print(f"[skip] {plate}/{sample}: excluded by shared/experiment_filters.py", flush=True)
            continue

        try:
            mask_stack = escape_area.load_stack(mask_path)
        except Exception as exc:
            print(f"[skip] {plate}/{sample}: cannot load mask stack ({exc})", flush=True)
            continue
        if mask_stack.ndim != 3:
            print(f"[skip] {plate}/{sample}: invalid mask shape {mask_stack.shape}", flush=True)
            continue

        clone_ids = valid_clone_ids(mask_stack)
        if clone_ids.size == 0:
            print(f"[skip] {plate}/{sample}: no clone labels", flush=True)
            continue

        points_by_t = escape_area.load_points_by_t(points_csv)
        res_by_day = escape_area.preload_res_points_by_day(h5_path, n_frames=mask_stack.shape[0])
        esc_info = escape_lookup.get(sample, {})
        escape_day = float(esc_info.get("escape_day", np.nan))
        escape_detected = bool(esc_info.get("escape_detected", np.isfinite(escape_day)))

        kept_frame_count = 0
        kept_clone_rows = 0
        for t in range(mask_stack.shape[0]):
            day = int(escape_area.day_from_tidx(t))
            if cutoff_day is not None and day > cutoff_day:
                continue

            frame = mask_stack[t]
            total_counts = count_resistant_by_clone(
                frame=frame,
                xy=res_by_day.get(day, np.empty((0, 2), dtype=float)),
                clone_ids=clone_ids,
            )
            escaped_counts = count_resistant_by_clone(
                frame=frame,
                xy=points_by_t.get(t, np.empty((0, 2), dtype=float)),
                clone_ids=clone_ids,
            )

            kept_frame_count += 1
            for clone_id in clone_ids:
                clone_i = int(clone_id)
                total = int(total_counts[clone_i])
                escaped = int(escaped_counts[clone_i])
                if total <= 0 and not include_zero_resistant:
                    continue
                fraction = float(escaped / total) if total > 0 else np.nan
                kept_clone_rows += 1
                yield CloneTrajectoryRow(
                    plate=plate,
                    sample=sample,
                    strategy=strategy,
                    clone_id=clone_i,
                    t=int(t),
                    day=day,
                    days_since_escape=(
                        float(day - escape_day) if np.isfinite(escape_day) else np.nan
                    ),
                    resistant_count=total,
                    escaped_count=escaped,
                    escaped_fraction=fraction,
                    escape_detected=escape_detected,
                    escape_day=escape_day,
                )

        print(
            f"[ok] {plate}/{sample}: frames={kept_frame_count}, "
            f"clones={clone_ids.size}, rows={kept_clone_rows}"
            ,
            flush=True,
        )
        processed_replicates += 1


def iter_well_trajectory_rows(
    data_root: Path,
    escape_lookup: dict[str, dict[str, float | bool]],
    *,
    include_excluded: bool,
    include_zero_resistant: bool,
    strategy_filter: set[str] | None = None,
    max_replicates: int | None = None,
) -> Iterable[WellTrajectoryRow]:
    escape_area.DATA_ROOT = data_root
    replicates = escape_area.discover_replicates()
    print(f"[info] discovered replicates with escape inputs: {len(replicates)}", flush=True)

    processed_replicates = 0
    for plate, sample, h5_path, mask_path, points_csv in replicates:
        raw_strategy = escape_area.parse_strategy(sample)
        strategy = normalize_strategy(raw_strategy)
        if strategy_filter is not None and strategy not in strategy_filter:
            continue
        if max_replicates is not None and processed_replicates >= max_replicates:
            print(f"[info] stopped after --max-replicates={max_replicates}", flush=True)
            break

        status, cutoff_day = filter_rule(sample, include_excluded=include_excluded)
        if status == "drop":
            print(f"[skip] {plate}/{sample}: excluded by shared/experiment_filters.py", flush=True)
            continue

        try:
            mask_stack = escape_area.load_stack(mask_path)
        except Exception as exc:
            print(f"[skip] {plate}/{sample}: cannot load mask stack ({exc})", flush=True)
            continue
        if mask_stack.ndim != 3:
            print(f"[skip] {plate}/{sample}: invalid mask shape {mask_stack.shape}", flush=True)
            continue

        points_by_t = escape_area.load_points_by_t(points_csv)
        res_by_day = escape_area.preload_res_points_by_day(h5_path, n_frames=mask_stack.shape[0])
        esc_info = escape_lookup.get(sample, {})
        escape_day = float(esc_info.get("escape_day", np.nan))
        escape_detected = bool(esc_info.get("escape_detected", np.isfinite(escape_day)))

        kept_frame_count = 0
        kept_rows = 0
        for t in range(mask_stack.shape[0]):
            day = int(escape_area.day_from_tidx(t))
            if cutoff_day is not None and day > cutoff_day:
                continue

            resistant_count = int(len(res_by_day.get(day, np.empty((0, 2), dtype=float))))
            escaped_count = int(len(points_by_t.get(t, np.empty((0, 2), dtype=float))))
            kept_frame_count += 1

            if resistant_count <= 0 and not include_zero_resistant:
                continue

            fraction = float(escaped_count / resistant_count) if resistant_count > 0 else np.nan
            kept_rows += 1
            yield WellTrajectoryRow(
                plate=plate,
                sample=sample,
                strategy=strategy,
                t=int(t),
                day=day,
                days_since_escape=(
                    float(day - escape_day) if np.isfinite(escape_day) else np.nan
                ),
                resistant_count=resistant_count,
                escaped_count=escaped_count,
                escaped_fraction=fraction,
                escape_detected=escape_detected,
                escape_day=escape_day,
            )

        print(
            f"[ok] {plate}/{sample}: frames={kept_frame_count}, rows={kept_rows}",
            flush=True,
        )
        processed_replicates += 1


def bootstrap_median_ci(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    if values.size == 1 or repeats <= 0:
        return median, median, median
    draws = rng.choice(values, size=(int(repeats), values.size), replace=True)
    medians = np.median(draws, axis=1)
    return (
        median,
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def summarize_by_strategy_time(
    df: pd.DataFrame,
    *,
    value_col: str,
    time_col: str,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped = df.groupby(["strategy", time_col], sort=True)
    for (strategy, time_value), group in grouped:
        values = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        median, ci_low, ci_high = bootstrap_median_ci(values, repeats=repeats, rng=rng)
        rows.append(
            {
                "strategy": strategy,
                time_col: float(time_value),
                "median": median,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_timepoints": int(values.size),
                "n_replicates": int(group["sample"].nunique()),
                "n_clones": (
                    int(group[["sample", "clone_id"]].drop_duplicates().shape[0])
                    if "clone_id" in group.columns
                    else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["strategy", time_col]).reset_index(drop=True)


def carry_forward_well_values(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Hold each well's last valid value constant after its final observed row.

    The fill grid is the set of timepoints observed for the same strategy.
    Rows before a well's first valid timepoint are not created.
    """
    if df.empty:
        return df.copy()
    if "clone_id" in df.columns:
        raise ValueError("Carry-forward censor handling is only defined for well-wise rows.")

    required = {"strategy", "sample", time_col, "escaped_count", "escaped_fraction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for carry-forward handling: {sorted(missing)}")

    out_rows: list[dict] = []
    for strategy, strategy_df in df.groupby("strategy", sort=False):
        grid = np.sort(pd.to_numeric(strategy_df[time_col], errors="coerce").dropna().unique())
        if grid.size == 0:
            continue

        for sample, sample_df in strategy_df.groupby("sample", sort=False):
            sample_df = sample_df.copy()
            sample_df[time_col] = pd.to_numeric(sample_df[time_col], errors="coerce")
            sample_df = sample_df[np.isfinite(sample_df[time_col])].sort_values(time_col)
            if sample_df.empty:
                continue

            first_time = float(sample_df[time_col].iloc[0])
            last_time = float(sample_df[time_col].iloc[-1])
            fill_grid = grid[grid >= first_time]
            sample_by_time = sample_df.drop_duplicates(subset=[time_col], keep="last").set_index(time_col)
            last_row: dict | None = None

            for time_value in fill_grid:
                if time_value in sample_by_time.index:
                    row = sample_by_time.loc[time_value].to_dict()
                    row[time_col] = float(time_value)
                    row["carried_forward"] = False
                    last_row = row.copy()
                elif last_row is not None:
                    row = last_row.copy()
                    row[time_col] = float(time_value)
                    row["carried_forward"] = bool(float(time_value) > last_time)
                else:
                    continue
                out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    sort_cols = ["strategy", "sample", time_col]
    return out.sort_values(sort_cols).reset_index(drop=True)


def ordered_strategies(strategies: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(strategies))
    ordered = [s for s in STRATEGY_ORDER if s in present]
    ordered.extend(sorted(s for s in present if s not in ordered))
    return ordered


def make_trajectory_plot(
    summary: pd.DataFrame,
    *,
    time_col: str,
    ylabel: str,
    out_pdf: Path,
    y_max: float | None = None,
    y_scale: str = "linear",
) -> None:
    if summary.empty:
        print(f"[warn] no data for {out_pdf.name}")
        return

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    for strategy in ordered_strategies(summary["strategy"].dropna().astype(str).unique()):
        group = summary[summary["strategy"] == strategy].copy()
        if group.empty:
            continue
        group = group.sort_values(time_col)
        x = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float)
        med = pd.to_numeric(group["median"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(group["ci_low"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(group["ci_high"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(med)
        if not np.any(valid):
            continue

        color = get_color(strategy, "#666666")
        ax.fill_between(x[valid], lo[valid], hi[valid], color=color, alpha=CI_ALPHA, linewidth=0)
        ax.plot(
            x[valid],
            med[valid],
            color=color,
            linewidth=LINEWIDTH,
            label=get_display_name(strategy),
        )

    ax.set_xlabel("Time [days]" if time_col == "day" else "Days since escape")
    ax.set_ylabel(ylabel)
    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6, base=10)
        ax.set_ylim(bottom=0.0)
    elif y_max is not None:
        ax.set_ylim(0.0, y_max)
    else:
        ax.set_ylim(bottom=0.0)
    if time_col == "days_since_escape":
        ax.axvline(0.0, color="#444444", linewidth=0.7, linestyle="--", zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="best", fontsize=LEGEND_SIZE_PT)
    style_axis(ax)
    ft.savefig(fig, out_pdf)
    plt.close(fig)
    print(f"[save] {out_pdf}")


def save_summaries_and_plots(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    aggregation: str,
    time_axis: str,
    time_col: str,
    bootstrap_repeats: int,
    filename_suffix: str,
    plot_label_suffix: str,
) -> None:
    count_summary = summarize_by_strategy_time(
        df,
        value_col="escaped_count",
        time_col=time_col,
        repeats=bootstrap_repeats,
    )
    fraction_summary = summarize_by_strategy_time(
        df,
        value_col="escaped_fraction",
        time_col=time_col,
        repeats=bootstrap_repeats,
    )

    count_summary_csv = output_dir / (
        f"{aggregation}_escaped_count_median_ci_by_strategy_{time_axis}{filename_suffix}.csv"
    )
    fraction_summary_csv = output_dir / (
        f"{aggregation}_escaped_fraction_median_ci_by_strategy_{time_axis}{filename_suffix}.csv"
    )
    count_summary.to_csv(count_summary_csv, index=False)
    fraction_summary.to_csv(fraction_summary_csv, index=False)
    print(f"[save] {count_summary_csv}")
    print(f"[save] {fraction_summary_csv}")

    suffix = f"{time_axis.replace('-', '_')}{filename_suffix}"
    unit_label = "clone" if aggregation == "clone" else "well"
    title_unit = "Clone" if aggregation == "clone" else "Well"
    make_trajectory_plot(
        count_summary,
        time_col=time_col,
        ylabel=f"Escaped resistant cells per {unit_label}",
        out_pdf=output_dir / f"Escaped_Cells_Per_{title_unit}_Median_CI_{suffix}.pdf",
    )
    make_trajectory_plot(
        count_summary,
        time_col=time_col,
        ylabel=f"Escaped resistant cells per {unit_label}",
        out_pdf=output_dir / f"Escaped_Cells_Per_{title_unit}_Median_CI_{suffix}_log_y.pdf",
        y_scale="symlog",
    )
    make_trajectory_plot(
        fraction_summary,
        time_col=time_col,
        ylabel=f"Fraction escaped resistant cells{plot_label_suffix}",
        out_pdf=output_dir / f"Escaped_Fraction_Per_{title_unit}_Median_CI_{suffix}.pdf",
        y_max=1.0,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    escape_summary_csv = (
        args.data_root
        / "Figures"
        / "Escape_Quantification"
        / "escape_front_batch_summary_all_res_within_sens.csv"
    )
    escape_lookup = load_escape_summary(escape_summary_csv)
    strategy_filter = (
        {normalize_strategy(strategy) for strategy in args.strategies}
        if args.strategies
        else None
    )
    if strategy_filter is None:
        strategy_filter = {
            normalize_strategy(strategy)
            for strategy in STRATEGY_ORDER
            if normalize_strategy(strategy) not in EXCLUDED_PLOT_STRATEGIES
        }
    else:
        strategy_filter = strategy_filter.difference(EXCLUDED_PLOT_STRATEGIES)
    if not strategy_filter:
        raise ValueError(
            "No strategies remain after excluding KPP-Agent and Pulse from escape-cell plots."
        )
    print(
        "[info] plotting strategies: "
        + ", ".join(get_display_name(strategy) for strategy in ordered_strategies(strategy_filter)),
        flush=True,
    )
    row_iter = (
        iter_clone_trajectory_rows(
            args.data_root,
            escape_lookup,
            include_excluded=args.include_excluded,
            include_zero_resistant=args.include_zero_resistant,
            strategy_filter=strategy_filter,
            max_replicates=args.max_replicates,
        )
        if args.aggregation == "clone"
        else iter_well_trajectory_rows(
            args.data_root,
            escape_lookup,
            include_excluded=args.include_excluded,
            include_zero_resistant=args.include_zero_resistant,
            strategy_filter=strategy_filter,
            max_replicates=args.max_replicates,
        )
    )
    rows = [row.__dict__ for row in row_iter]
    if not rows:
        raise RuntimeError(f"No {args.aggregation} trajectory rows were generated.")

    df = pd.DataFrame(rows)
    time_col = "day" if args.time_axis == "day" else "days_since_escape"
    if time_col == "days_since_escape":
        df = df[np.isfinite(pd.to_numeric(df[time_col], errors="coerce"))].copy()
        if df.empty:
            raise RuntimeError("No rows have finite days_since_escape values.")

    detail_csv = output_dir / f"{args.aggregation}_escaped_cell_trajectories_{args.time_axis}.csv"
    df.to_csv(detail_csv, index=False)
    print(f"[save] {detail_csv}")

    save_summaries_and_plots(
        df,
        output_dir=output_dir,
        aggregation=args.aggregation,
        time_axis=args.time_axis,
        time_col=time_col,
        bootstrap_repeats=args.bootstrap_repeats,
        filename_suffix="",
        plot_label_suffix="",
    )

    if args.carry_forward_censored:
        if args.aggregation != "well":
            raise ValueError("--carry-forward-censored is only available with --aggregation well.")
        df_cf = carry_forward_well_values(df, time_col=time_col)
        detail_cf_csv = output_dir / (
            f"{args.aggregation}_escaped_cell_trajectories_{args.time_axis}_carry_forward_censored.csv"
        )
        df_cf.to_csv(detail_cf_csv, index=False)
        print(f"[save] {detail_cf_csv}")
        save_summaries_and_plots(
            df_cf,
            output_dir=output_dir,
            aggregation=args.aggregation,
            time_axis=args.time_axis,
            time_col=time_col,
            bootstrap_repeats=args.bootstrap_repeats,
            filename_suffix="_carry_forward_censored",
            plot_label_suffix=" (carry-forward)",
        )

    print(f"[counts] {args.aggregation}/timepoint rows by strategy:")
    for strategy in ordered_strategies(df["strategy"].dropna().astype(str).unique()):
        sub = df[df["strategy"] == strategy]
        if "clone_id" in sub.columns:
            extra = f", clones={sub[['sample', 'clone_id']].drop_duplicates().shape[0]}"
        else:
            extra = ""
        print(
            f"  {get_display_name(strategy)}: rows={len(sub)}, "
            f"replicates={sub['sample'].nunique()}{extra}"
        )


if __name__ == "__main__":
    main()
