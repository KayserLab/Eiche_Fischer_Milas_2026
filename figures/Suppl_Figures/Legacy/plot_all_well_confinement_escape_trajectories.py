"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - D:\2025_08_Co
# - F:\2025_08_Co
# Upstream/related script reference(s):
# - analysis/figure_2/Escape_ConfinementThickness.py
# - analysis/figure_2/Escape_Simulation_Confinement_And_Escape.py
# - and shared/figtools.py
# - shared/experiment_filters.py
# - Ignore full exclusions and cutoff days from shared/experiment_filters.py
# - import matplotlib.py
# - shared/strategy_colors.py
# =============================================================================
Plot one signed confinement/escape trajectory for every eligible well.

Confinement thickness is positive before a detected escape. From the recorded
escape day onward, the same line shows the negative number of escaped resistant
cells. Wells without a detected escape remain on the positive side for their
entire valid observation window. Experimental mode applies shared exclusions
and cutoff days from ``shared/experiment_filters.py``. Simulation mode reads the
aggregate outputs from ``analysis/figure_2/Escape_Simulation_Confinement_And_Escape.py``
and does not apply experimental filters.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental or simulation confinement-thickness and escape-count/
# escape-summary CSVs.
# Scripts: analysis/figure_2/Escape_ConfinementThickness.py and
# analysis/figure_2/Escape_Simulation_Confinement_And_Escape.py (upstream),
# shared/experiment_filters.py, shared/strategy_colors.py, and shared/figtools.py.
# Outputs: per-well and strategy-summary signed trajectory PDFs/CSVs.

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, intermediate_dir
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import (
    STRATEGY_ORDER,
    get_color,
    get_display_name,
    normalize_strategy,
)


DATA_ROOT = EXP_RAW_DATA
ESCAPE_DIR = intermediate_dir("experimental", "figure_2", "escape_quantification", create=False)
DEFAULT_CONFINEMENT_CSV = ESCAPE_DIR / "Escape_ConfinementThickness_all_replicates.csv"
DEFAULT_ESCAPE_SUMMARY_CSV = ESCAPE_DIR / "escape_front_batch_summary_all_res_within_sens.csv"
DEFAULT_OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "All_Well_Confinement_Escape_Trajectories"
DEFAULT_DATA_OUTPUT_DIR = figure_data_dir("Suppl_Figures", "all_well_confinement_escape_trajectories", create=False)
DEFAULT_SIMULATION_ROOT = intermediate_dir("simulation", "basic", "density_maps", create=False)
SIMULATION_QUANTIFICATION_SUBDIR = Path("Escape_Quantification")
SIMULATION_CONFINEMENT_FILENAME = "Escape_ConfinementThickness_all_simulations.csv"
SIMULATION_ESCAPE_COUNTS_FILENAME = "escape_cell_counts_all_simulations.csv"
SIMULATION_MIN_RUN_INDEX = 10
SIMULATION_RUN_PATTERN = r"^R(?P<run_index>\d+)_"
DEFAULT_SIMULATION_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "Simulation"
DEFAULT_SIMULATION_DATA_OUTPUT_DIR = DEFAULT_DATA_OUTPUT_DIR / "Simulation"
OUTPUT_PDF = "all_well_confinement_escape_trajectories.pdf"
OUTPUT_CSV = "all_well_confinement_escape_trajectories.csv"
AVERAGE_OUTPUT_PDF = "strategy_average_confinement_escape_trajectories.pdf"
AVERAGE_OUTPUT_CSV = "strategy_average_confinement_escape_trajectories.csv"
MEDIAN_OUTPUT_PDF = "strategy_median_confinement_escape_trajectories.pdf"
MEDIAN_OUTPUT_CSV = "strategy_median_confinement_escape_trajectories.csv"
CARRY_FORWARD_OUTPUT_PDF = "all_well_confinement_escape_trajectories_carry_forward.pdf"
CARRY_FORWARD_OUTPUT_CSV = "all_well_confinement_escape_trajectories_carry_forward.csv"
CARRY_FORWARD_AVERAGE_OUTPUT_PDF = "strategy_average_confinement_escape_trajectories_carry_forward.pdf"
CARRY_FORWARD_AVERAGE_OUTPUT_CSV = "strategy_average_confinement_escape_trajectories_carry_forward.csv"
CARRY_FORWARD_MEDIAN_OUTPUT_PDF = "strategy_median_confinement_escape_trajectories_carry_forward.pdf"
CARRY_FORWARD_MEDIAN_OUTPUT_CSV = "strategy_median_confinement_escape_trajectories_carry_forward.csv"
EXCLUDED_PLOT_STRATEGIES = {"KPP-Agent"}

FIGURE_AXES_WIDTH_MM = 85.0
FIGURE_AXES_HEIGHT_MM = 55.0
FIGURE_MARGINS_MM = (18.0, 5.0, 13.0, 5.0)
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
LINEWIDTH = 0.75
LINE_ALPHA = 0.42
ZERO_LINEWIDTH = 0.8
AVERAGE_LINEWIDTH = 1.5
UNCERTAINTY_ALPHA = 0.16
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 17
MEDIAN_CONFIDENCE_LEVEL = 0.95
POSITIVE_Y_MAX = 20000.0

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot positive confinement thickness before escape and negative escaped-cell "
            "counts from the escape day onward, with one line per well."
        )
    )
    parser.add_argument(
        "--dataset-type",
        choices=("experimental", "simulation", "both"),
        default="both",
        help="Input dataset to plot. Default: both experimental and simulation.",
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=DEFAULT_SIMULATION_ROOT,
        help=f"Simulation density run. Default: {DEFAULT_SIMULATION_ROOT}",
    )
    parser.add_argument(
        "--simulation-quantification-dir",
        type=Path,
        default=None,
        help=(
            "Simulation aggregate CSV directory. Default: "
            "<simulation-root>/Escape_Quantification."
        ),
    )
    parser.add_argument(
        "--confinement-csv",
        type=Path,
        default=DEFAULT_CONFINEMENT_CSV,
        help=f"Per-well confinement table. Default: {DEFAULT_CONFINEMENT_CSV}",
    )
    parser.add_argument(
        "--escape-summary-csv",
        type=Path,
        default=DEFAULT_ESCAPE_SUMMARY_CSV,
        help=f"Escape batch summary with metrics paths. Default: {DEFAULT_ESCAPE_SUMMARY_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Figure and plot-data output directory. Defaults to the experimental "
            "paper directory or its Simulation subdirectory, depending on --dataset-type."
        ),
    )
    parser.add_argument("--data-output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Ignore full exclusions and cutoff days from shared/experiment_filters.py.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def parse_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def filter_sample_days(
    df: pd.DataFrame,
    *,
    sample_column: str,
    day_column: str,
    include_excluded: bool,
) -> pd.DataFrame:
    if include_excluded:
        return df.copy()

    keep = np.ones(len(df), dtype=bool)
    samples = df[sample_column].astype(str).to_numpy()
    days = pd.to_numeric(df[day_column], errors="coerce").to_numpy(dtype=float)
    for idx, (sample, day) in enumerate(zip(samples, days)):
        if sample not in EXCLUDE_SAMPLES:
            continue
        cutoff = EXCLUDE_SAMPLES[sample]
        if cutoff is None or (np.isfinite(day) and day > float(cutoff)):
            keep[idx] = False
    return df.loc[keep].copy()


def load_confinement(path: Path, include_excluded: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Confinement table not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, {"plate", "replicate", "day", "confinement_thickness"}, path)
    df = df.rename(columns={"replicate": "sample"})
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["confinement_thickness"] = pd.to_numeric(
        df["confinement_thickness"], errors="coerce"
    )
    df = df[np.isfinite(df["day"]) & np.isfinite(df["confinement_thickness"])].copy()
    df = filter_sample_days(
        df,
        sample_column="sample",
        day_column="day",
        include_excluded=include_excluded,
    )
    return (
        df[["plate", "sample", "day", "confinement_thickness"]]
        .drop_duplicates(subset=["plate", "sample", "day"], keep="last")
        .sort_values(["plate", "sample", "day"])
    )


def load_escape_data(path: Path, include_excluded: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Escape summary not found: {path}")
    summary = pd.read_csv(path)
    require_columns(
        summary,
        {"plate", "sample", "strategy", "status", "escape_detected", "out_metrics_csv"},
        path,
    )
    summary = summary[summary["status"].astype(str).str.lower().eq("ok")].copy()
    summary["escape_detected"] = parse_bool(summary["escape_detected"])
    escape_day_column = "escape_day" if "escape_day" in summary else "first_escape_day"
    if escape_day_column not in summary:
        raise ValueError(f"{path} has neither escape_day nor first_escape_day")
    summary["escape_day"] = pd.to_numeric(summary[escape_day_column], errors="coerce")
    summary["strategy"] = summary["strategy"].astype(str).map(normalize_strategy)

    if not include_excluded:
        fully_excluded = {
            sample for sample, cutoff in EXCLUDE_SAMPLES.items() if cutoff is None
        }
        summary = summary[~summary["sample"].astype(str).isin(fully_excluded)].copy()
        for idx, row in summary.iterrows():
            cutoff = EXCLUDE_SAMPLES.get(str(row["sample"]))
            if cutoff is not None and row["escape_detected"]:
                if not np.isfinite(row["escape_day"]) or row["escape_day"] > float(cutoff):
                    summary.at[idx, "escape_detected"] = False
                    summary.at[idx, "escape_day"] = np.nan

    metric_frames: list[pd.DataFrame] = []
    for row in summary.itertuples(index=False):
        metrics_path = Path(str(row.out_metrics_csv))
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Escape metrics table not found: {metrics_path}")
        metrics = pd.read_csv(metrics_path)
        require_columns(metrics, {"day", "n_front_points"}, metrics_path)
        metrics["day"] = pd.to_numeric(metrics["day"], errors="coerce")
        metrics["escaped_count"] = pd.to_numeric(metrics["n_front_points"], errors="coerce")
        metrics = metrics[np.isfinite(metrics["day"]) & np.isfinite(metrics["escaped_count"])].copy()
        metrics["plate"] = str(row.plate)
        metrics["sample"] = str(row.sample)
        metric_frames.append(metrics[["plate", "sample", "day", "escaped_count"]])

    if not metric_frames:
        raise ValueError(f"No usable per-well escape metrics were listed in {path}")
    escaped = pd.concat(metric_frames, ignore_index=True)
    escaped = filter_sample_days(
        escaped,
        sample_column="sample",
        day_column="day",
        include_excluded=include_excluded,
    )
    escaped = escaped.drop_duplicates(subset=["plate", "sample", "day"], keep="last")
    metadata = summary[
        ["plate", "sample", "strategy", "escape_detected", "escape_day"]
    ].drop_duplicates(subset=["plate", "sample"], keep="last")
    return metadata, escaped


def load_simulation_escape_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Simulation escape-count table not found: {path}")
    counts = pd.read_csv(path)
    require_columns(
        counts,
        {"plate", "sample", "day", "n_resistant_escaped"},
        path,
    )
    strategy_column = "strategy" if "strategy" in counts.columns else "treatment"
    if strategy_column not in counts.columns:
        raise ValueError(f"{path} has neither strategy nor treatment")

    counts["day"] = pd.to_numeric(counts["day"], errors="coerce")
    counts["escaped_count"] = pd.to_numeric(
        counts["n_resistant_escaped"], errors="coerce"
    )
    counts["strategy"] = counts[strategy_column].astype(str).map(normalize_strategy)
    counts = counts[
        np.isfinite(counts["day"]) & np.isfinite(counts["escaped_count"])
    ].copy()
    counts = counts.drop_duplicates(
        subset=["plate", "sample", "day"], keep="last"
    )

    metadata_rows: list[dict[str, object]] = []
    for (plate, sample), group in counts.groupby(["plate", "sample"], sort=True):
        group = group.sort_values("day")
        escaped_days = group.loc[group["escaped_count"] > 0, "day"]
        escape_detected = not escaped_days.empty
        metadata_rows.append(
            {
                "plate": plate,
                "sample": sample,
                "strategy": str(group["strategy"].iloc[0]),
                "escape_detected": bool(escape_detected),
                "escape_day": (
                    float(escaped_days.iloc[0]) if escape_detected else np.nan
                ),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    escaped = counts[["plate", "sample", "day", "escaped_count"]].copy()
    return metadata, escaped


def filter_simulation_initial_condition_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep genuine simulation runs R10 and above."""
    require_columns(df, {"sample"}, Path("simulation table"))
    run_indices = pd.to_numeric(
        df["sample"].astype(str).str.extract(SIMULATION_RUN_PATTERN, expand=False),
        errors="coerce",
    )
    keep = run_indices.isna() | (run_indices >= SIMULATION_MIN_RUN_INDEX)
    return df.loc[keep].copy()


def build_signed_trajectories(
    confinement: pd.DataFrame,
    metadata: pd.DataFrame,
    escaped: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    confinement_keys = confinement[["plate", "sample"]].drop_duplicates()
    metadata = metadata.merge(confinement_keys, on=["plate", "sample"], how="inner")

    for meta in metadata.itertuples(index=False):
        conf = confinement[
            (confinement["plate"] == meta.plate) & (confinement["sample"] == meta.sample)
        ].copy()
        esc = escaped[
            (escaped["plate"] == meta.plate) & (escaped["sample"] == meta.sample)
        ].copy()

        if bool(meta.escape_detected) and np.isfinite(meta.escape_day):
            conf = conf[conf["day"] < float(meta.escape_day)].copy()
            esc = esc[esc["day"] >= float(meta.escape_day)].copy()
        else:
            esc = esc.iloc[0:0].copy()

        conf["phase"] = "confinement"
        conf["signed_value"] = conf["confinement_thickness"]
        esc["phase"] = "escaped"
        esc["signed_value"] = -esc["escaped_count"]
        combined = pd.concat(
            [
                conf[["plate", "sample", "day", "phase", "signed_value"]],
                esc[["plate", "sample", "day", "phase", "signed_value"]],
            ],
            ignore_index=True,
        ).sort_values(["day", "phase"])
        if combined.empty:
            continue
        combined["strategy"] = str(meta.strategy)
        combined["escape_detected"] = bool(meta.escape_detected)
        combined["escape_day"] = float(meta.escape_day) if np.isfinite(meta.escape_day) else np.nan
        rows.append(combined)

    if not rows:
        raise ValueError("No signed trajectories remain after filtering and phase selection")
    return pd.concat(rows, ignore_index=True)[
        [
            "plate",
            "sample",
            "strategy",
            "day",
            "phase",
            "signed_value",
            "escape_detected",
            "escape_day",
        ]
    ].sort_values(["strategy", "plate", "sample", "day", "phase"])


def carry_forward_truncated_wells(
    df: pd.DataFrame,
    *,
    enabled: bool,
) -> pd.DataFrame:
    """Extend every ending trajectory with its last observed value."""
    carried = df.copy()
    carried["carried_forward"] = False
    carried["carry_forward_method"] = "observed"
    carried["source_day"] = carried["day"]
    if not enabled:
        return carried

    strategy_days = {
        str(strategy): np.sort(group["day"].dropna().unique())
        for strategy, group in carried.groupby("strategy", sort=False)
    }
    additions: list[pd.Series] = []
    for _, group in carried.groupby(["plate", "sample"], sort=False):
        group = group.sort_values(["day", "phase"])
        last = group.iloc[-1].copy()
        last_day = float(last["day"])
        future_days = strategy_days[str(last["strategy"])]
        future_days = future_days[future_days > last_day]
        for day in future_days:
            row = last.copy()
            row["day"] = day
            row["carried_forward"] = True
            row["carry_forward_method"] = "last_value"
            row["source_day"] = last_day
            additions.append(row)

    if additions:
        carried = pd.concat([carried, pd.DataFrame(additions)], ignore_index=True)
    return carried.sort_values(["strategy", "plate", "sample", "day", "phase"])


def rank_carry_forward_truncated_wells(
    df: pd.DataFrame,
    *,
    enabled: bool,
    rank_filter_cutoffs: bool,
) -> pd.DataFrame:
    """Use rank for filter cutoffs and last values for natural endings."""
    carried = df.copy()
    carried["carried_forward"] = False
    carried["carry_forward_method"] = "observed"
    carried["source_day"] = carried["day"]
    carried["source_signed_value"] = carried["signed_value"]
    carried["rank_percentile"] = np.nan
    if not enabled:
        return carried

    strategy_days = {
        str(strategy): np.sort(group["day"].dropna().unique())
        for strategy, group in df.groupby("strategy", sort=False)
    }
    additions: list[pd.Series] = []
    for (_, sample), group in df.groupby(["plate", "sample"], sort=False):
        sample = str(sample)
        group = group.sort_values(["day", "phase"])
        last = group.iloc[-1].copy()
        strategy = str(last["strategy"])
        last_day = float(last["day"])
        last_value = float(last["signed_value"])
        future_days = strategy_days[strategy]
        future_days = future_days[future_days > last_day]
        if future_days.size == 0:
            continue

        use_rank = (
            rank_filter_cutoffs
            and sample in EXCLUDE_SAMPLES
            and EXCLUDE_SAMPLES[sample] is not None
        )
        percentile = np.nan
        if use_rank:
            source_values = df.loc[
                (df["strategy"].astype(str) == strategy) & (df["day"] == last_day),
                "signed_value",
            ].dropna().to_numpy(dtype=float)
            if source_values.size == 0:
                continue
            below = int(np.count_nonzero(source_values < last_value))
            equal = int(np.count_nonzero(source_values == last_value))
            percentile = (below + 0.5 * equal) / source_values.size

        for day in future_days:
            row = last.copy()
            row["day"] = day
            if use_rank:
                future_values = df.loc[
                    (df["strategy"].astype(str) == strategy) & (df["day"] == day),
                    "signed_value",
                ].dropna().to_numpy(dtype=float)
                if future_values.size == 0:
                    continue
                row["signed_value"] = float(np.quantile(future_values, percentile))
                row["phase"] = (
                    "confinement" if row["signed_value"] > 0 else "escaped"
                )
                row["carry_forward_method"] = "filter_rank"
                row["rank_percentile"] = percentile
            else:
                row["carry_forward_method"] = "last_value"
            row["carried_forward"] = True
            row["source_day"] = last_day
            row["source_signed_value"] = last_value
            additions.append(row)

    if additions:
        carried = pd.concat([carried, pd.DataFrame(additions)], ignore_index=True)
    return carried.sort_values(["strategy", "plate", "sample", "day", "phase"])


def ordered_strategies(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    ordered = [strategy for strategy in STRATEGY_ORDER if strategy in present]
    return ordered + sorted(present.difference(ordered))


def summarize_strategy_averages(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["strategy", "day"], sort=True)
        .agg(
            mean_signed_value=("signed_value", "mean"),
            std_signed_value=("signed_value", "std"),
            n_wells=("signed_value", "size"),
            n_confinement=("phase", lambda values: int((values == "confinement").sum())),
            n_escaped=("phase", lambda values: int((values == "escaped").sum())),
        )
        .reset_index()
    )
    summary["sem_signed_value"] = summary["std_signed_value"] / np.sqrt(summary["n_wells"])
    summary["mean_minus_std"] = summary["mean_signed_value"] - summary["std_signed_value"]
    summary["mean_plus_std"] = summary["mean_signed_value"] + summary["std_signed_value"]
    return summary


def bootstrap_median_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    samples = rng.choice(finite, size=(BOOTSTRAP_REPEATS, finite.size), replace=True)
    bootstrap_medians = np.median(samples, axis=1)
    tail = (1.0 - MEDIAN_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(bootstrap_medians, [tail, 1.0 - tail])
    return float(lower), float(upper)


def summarize_strategy_medians(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["strategy", "day"], sort=True)
        .agg(
            median_signed_value=("signed_value", "median"),
            q25_signed_value=("signed_value", lambda values: values.quantile(0.25)),
            q75_signed_value=("signed_value", lambda values: values.quantile(0.75)),
            n_wells=("signed_value", "size"),
            n_confinement=("phase", lambda values: int((values == "confinement").sum())),
            n_escaped=("phase", lambda values: int((values == "escaped").sum())),
        )
        .reset_index()
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    intervals: list[dict[str, float | str]] = []
    for (strategy, day), group in df.groupby(["strategy", "day"], sort=True):
        lower, upper = bootstrap_median_ci(
            group["signed_value"].to_numpy(dtype=float),
            rng,
        )
        intervals.append(
            {
                "strategy": str(strategy),
                "day": float(day),
                "median_ci_lower": lower,
                "median_ci_upper": upper,
            }
        )
    return summary.merge(pd.DataFrame(intervals), on=["strategy", "day"], how="left")


def plot_trajectories(
    df: pd.DataFrame,
    output_path: Path,
    show: bool,
    title: str,
) -> None:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        apply_style=False,
    )

    ax.set_title(title, loc="left", pad=4.0, fontsize=TEXT_SIZE_PT)
    strategies = ordered_strategies(df["strategy"])
    for (_, _), group in df.groupby(["plate", "sample"], sort=True):
        group = group.sort_values(["day", "phase"])
        strategy = str(group["strategy"].iloc[0])
        ax.plot(
            group["day"].to_numpy(dtype=float),
            group["signed_value"].to_numpy(dtype=float),
            color=get_color(strategy, "#666666"),
            linewidth=LINEWIDTH,
            alpha=LINE_ALPHA,
            solid_capstyle="round",
            zorder=2,
        )

    ax.axhline(0.0, color="#222222", linewidth=ZERO_LINEWIDTH, zorder=3)
    negative_y_min, _ = ax.get_ylim()
    ax.set_ylim(negative_y_min, POSITIVE_Y_MAX)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Confinement thickness (+) / escaped cells (-)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(direction="out", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    handles = [
        Line2D(
            [],
            [],
            color=get_color(strategy, "#666666"),
            linewidth=1.4,
            label=get_display_name(strategy),
        )
        for strategy in strategies
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
    )

    ft.savefig(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_strategy_summary(
    df: pd.DataFrame,
    value_column: str,
    lower_column: str,
    upper_column: str,
    ylabel: str,
    output_path: Path,
    show: bool,
    title: str,
) -> None:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        apply_style=False,
    )

    ax.set_title(title, loc="left", pad=4.0, fontsize=TEXT_SIZE_PT)
    for strategy in ordered_strategies(df["strategy"]):
        group = df[df["strategy"] == strategy].sort_values("day")
        x = group["day"].to_numpy(dtype=float)
        color = get_color(strategy, "#666666")
        ax.fill_between(
            x,
            group[lower_column].to_numpy(dtype=float),
            group[upper_column].to_numpy(dtype=float),
            color=color,
            alpha=UNCERTAINTY_ALPHA,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            group[value_column].to_numpy(dtype=float),
            color=color,
            linewidth=AVERAGE_LINEWIDTH,
            label=get_display_name(strategy),
            solid_capstyle="round",
            zorder=2,
        )

    ax.axhline(0.0, color="#222222", linewidth=ZERO_LINEWIDTH, zorder=3)
    negative_y_min, _ = ax.get_ylim()
    ax.set_ylim(negative_y_min, POSITIVE_Y_MAX)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(direction="out", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
    )

    ft.savefig(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_dataset(args: argparse.Namespace, dataset_type: str) -> None:
    if dataset_type == "simulation":
        quantification_dir = args.simulation_quantification_dir or (
            args.simulation_root / SIMULATION_QUANTIFICATION_SUBDIR
        )
        confinement = load_confinement(
            quantification_dir / SIMULATION_CONFINEMENT_FILENAME,
            include_excluded=True,
        )
        metadata, escaped = load_simulation_escape_data(
            quantification_dir / SIMULATION_ESCAPE_COUNTS_FILENAME
        )
        confinement = filter_simulation_initial_condition_runs(confinement)
        metadata = filter_simulation_initial_condition_runs(metadata)
        escaped = filter_simulation_initial_condition_runs(escaped)
        if args.output_dir is not None and args.dataset_type == "both":
            output_dir = args.output_dir / "Simulation"
        else:
            output_dir = args.output_dir or DEFAULT_SIMULATION_OUTPUT_DIR
        if args.data_output_dir is not None and args.dataset_type == "both":
            data_output_dir = args.data_output_dir / "Simulation"
        else:
            data_output_dir = args.data_output_dir or DEFAULT_SIMULATION_DATA_OUTPUT_DIR
        carry_forward_enabled = True
        rank_filter_cutoffs = False
    else:
        confinement = load_confinement(args.confinement_csv, args.include_excluded)
        metadata, escaped = load_escape_data(
            args.escape_summary_csv,
            args.include_excluded,
        )
        if args.output_dir is not None and args.dataset_type == "both":
            output_dir = args.output_dir / "Experimental"
        else:
            output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        if args.data_output_dir is not None and args.dataset_type == "both":
            data_output_dir = args.data_output_dir / "Experimental"
        else:
            data_output_dir = args.data_output_dir or DEFAULT_DATA_OUTPUT_DIR
        carry_forward_enabled = True
        rank_filter_cutoffs = not args.include_excluded

    trajectories = build_signed_trajectories(confinement, metadata, escaped)
    trajectories = trajectories[
        ~trajectories["strategy"].isin(EXCLUDED_PLOT_STRATEGIES)
    ].copy()
    strategy_averages = summarize_strategy_averages(trajectories)
    strategy_medians = summarize_strategy_medians(trajectories)
    carry_forward_trajectories = carry_forward_truncated_wells(
        trajectories,
        enabled=carry_forward_enabled,
    )
    rank_carry_forward_trajectories = rank_carry_forward_truncated_wells(
        trajectories,
        enabled=carry_forward_enabled,
        rank_filter_cutoffs=rank_filter_cutoffs,
    )
    carry_forward_averages = summarize_strategy_averages(carry_forward_trajectories)
    carry_forward_medians = summarize_strategy_medians(
        rank_carry_forward_trajectories
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = data_output_dir / OUTPUT_CSV
    output_pdf = output_dir / OUTPUT_PDF
    average_output_csv = data_output_dir / AVERAGE_OUTPUT_CSV
    average_output_pdf = output_dir / AVERAGE_OUTPUT_PDF
    median_output_csv = data_output_dir / MEDIAN_OUTPUT_CSV
    median_output_pdf = output_dir / MEDIAN_OUTPUT_PDF
    carry_forward_output_csv = data_output_dir / CARRY_FORWARD_OUTPUT_CSV
    carry_forward_output_pdf = output_dir / CARRY_FORWARD_OUTPUT_PDF
    carry_forward_average_output_csv = data_output_dir / CARRY_FORWARD_AVERAGE_OUTPUT_CSV
    carry_forward_average_output_pdf = output_dir / CARRY_FORWARD_AVERAGE_OUTPUT_PDF
    carry_forward_median_output_csv = data_output_dir / CARRY_FORWARD_MEDIAN_OUTPUT_CSV
    carry_forward_median_output_pdf = output_dir / CARRY_FORWARD_MEDIAN_OUTPUT_PDF
    dataset_title = (
        f"Simulation data (R{SIMULATION_MIN_RUN_INDEX} onward)"
        if dataset_type == "simulation"
        else "Experimental data"
    )
    trajectories.to_csv(output_csv, index=False)
    strategy_averages.to_csv(average_output_csv, index=False)
    strategy_medians.to_csv(median_output_csv, index=False)
    carry_forward_trajectories.to_csv(carry_forward_output_csv, index=False)
    carry_forward_averages.to_csv(carry_forward_average_output_csv, index=False)
    carry_forward_medians.to_csv(carry_forward_median_output_csv, index=False)
    plot_trajectories(
        trajectories,
        output_pdf,
        show=args.show,
        title=dataset_title,
    )
    plot_strategy_summary(
        strategy_averages,
        value_column="mean_signed_value",
        lower_column="mean_minus_std",
        upper_column="mean_plus_std",
        ylabel="Mean confinement thickness (+) / escaped cells (-)",
        output_path=average_output_pdf,
        show=args.show,
        title=dataset_title,
    )
    plot_strategy_summary(
        strategy_medians,
        value_column="median_signed_value",
        lower_column="median_ci_lower",
        upper_column="median_ci_upper",
        ylabel="Median confinement thickness (+) / escaped cells (-)",
        output_path=median_output_pdf,
        show=args.show,
        title=dataset_title,
    )

    plot_trajectories(
        carry_forward_trajectories,
        carry_forward_output_pdf,
        show=args.show,
        title=dataset_title,
    )
    plot_strategy_summary(
        carry_forward_averages,
        value_column="mean_signed_value",
        lower_column="mean_minus_std",
        upper_column="mean_plus_std",
        ylabel="Mean confinement thickness (+) / escaped cells (-)",
        output_path=carry_forward_average_output_pdf,
        show=args.show,
        title=dataset_title,
    )
    plot_strategy_summary(
        carry_forward_medians,
        value_column="median_signed_value",
        lower_column="median_ci_lower",
        upper_column="median_ci_upper",
        ylabel="Median confinement thickness (+) / escaped cells (-)",
        output_path=carry_forward_median_output_pdf,
        show=args.show,
        title=dataset_title,
    )

    wells = trajectories[["plate", "sample"]].drop_duplicates().shape[0]
    escaping_wells = trajectories.loc[
        trajectories["escape_detected"], ["plate", "sample"]
    ].drop_duplicates().shape[0]
    added_rows = int(carry_forward_trajectories["carried_forward"].sum())
    carried_wells = carry_forward_trajectories.loc[
        carry_forward_trajectories["carried_forward"], ["plate", "sample"]
    ].drop_duplicates().shape[0]
    rank_added_rows = int(rank_carry_forward_trajectories["carried_forward"].sum())
    print(f"[dataset] {dataset_type}")
    print(f"[counts] plotted wells={wells}, escaping={escaping_wells}, non-escaping={wells - escaping_wells}")
    print(f"[counts] value carry-forward wells={carried_wells}, added rows={added_rows}")
    print(f"[counts] median carry-forward added rows={rank_added_rows}")
    print(f"[saved] {output_pdf}")
    print(f"[saved] {output_csv}")
    print(f"[saved] {average_output_pdf}")
    print(f"[saved] {average_output_csv}")
    print(f"[saved] {median_output_pdf}")
    print(f"[saved] {median_output_csv}")
    print(f"[saved] {carry_forward_output_pdf}")
    print(f"[saved] {carry_forward_output_csv}")
    print(f"[saved] {carry_forward_average_output_pdf}")
    print(f"[saved] {carry_forward_average_output_csv}")
    print(f"[saved] {carry_forward_median_output_pdf}")
    print(f"[saved] {carry_forward_median_output_csv}")


def main() -> None:
    args = parse_args()
    dataset_types = (
        ("experimental", "simulation")
        if args.dataset_type == "both"
        else (args.dataset_type,)
    )
    for dataset_type in dataset_types:
        run_dataset(args, dataset_type)


if __name__ == "__main__":
    main()
