"""Focused confinement/escape and progression illustration for Figure 2.

Outputs one carry-forward strategy-average PDF for CT, AT50, and AT100, plus
one overlaid in-vitro/DPD diagnostic PDF for every mapped AT100 replicate.

Every trajectory stops at the earliest applicable endpoint: manual truncation,
progression threshold, or its final available observation.  A progression that
occurs after a manual truncation is therefore not treated as an observed event.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental Results/*.txt, simulation escape-cell counts, and the
# confinement/escape CSVs configured by the supplementary trajectory module.
# Scripts: figures/Suppl_Figures/plot_all_well_confinement_escape_trajectories.py,
# shared/experiment_filters.py, shared/strategy_colors.py, and shared/figtools.py.
# Outputs: strategy summaries and matched in-vitro/DPD diagnostic plots/CSVs.

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import (
    FIGURE_COLOR_MAP,
    STRATEGY_ORDER,
    get_color,
    get_display_name,
    normalize_strategy,
)
from figures.Suppl_Figures.Legacy import plot_all_well_confinement_escape_trajectories as source
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, figure_plot_dir

OUTPUT_DIR = figure_plot_dir("Figure_2", "selected_escape_progression", create=False)
DATA_OUTPUT_DIR = figure_data_dir("Figure_2", "selected_escape_progression", create=False)
AVERAGE_PDF = "selected_CT_AT50_AT100_strategy_average_invitro_DPD_carry_forward_escape_positive.pdf"
AVERAGE_CSV = "selected_CT_AT50_AT100_strategy_average_invitro_DPD_carry_forward_escape_positive.csv"
RAW_PROGRESSION_AVERAGE_PDF = "01_strategy_average_raw_cut_at_progression.pdf"
RAW_PROGRESSION_AVERAGE_CSV = "01_strategy_average_raw_cut_at_progression.csv"
NORMALIZED_AVERAGE_PDF = "02_strategy_average_normalized_no_carry_forward.pdf"
NORMALIZED_AVERAGE_CSV = "02_strategy_average_normalized_no_carry_forward.csv"
NORMALIZED_POSITIVE_CARRY_AVERAGE_PDF = "03_strategy_average_normalized_positive_carry_forward.pdf"
NORMALIZED_POSITIVE_CARRY_AVERAGE_CSV = "03_strategy_average_normalized_positive_carry_forward.csv"
EXPANDED_NORMALIZED_POSITIVE_CARRY_AVERAGE_PDF = "03_strategy_average_normalized_positive_carry_forward_with_LV-Agent_AT100plus.pdf"
EXPANDED_NORMALIZED_POSITIVE_CARRY_AVERAGE_CSV = "03_strategy_average_normalized_positive_carry_forward_with_LV-Agent_AT100plus.csv"
RAW_PROGRESSION_MEDIAN_PDF = "01_strategy_median_IQR_raw_cut_at_progression.pdf"
RAW_PROGRESSION_MEDIAN_CSV = "01_strategy_median_IQR_raw_cut_at_progression.csv"
NORMALIZED_POSITIVE_CARRY_MEDIAN_PDF = "03_strategy_median_IQR_normalized_positive_carry_forward.pdf"
NORMALIZED_POSITIVE_CARRY_MEDIAN_CSV = "03_strategy_median_IQR_normalized_positive_carry_forward.csv"
INDIVIDUAL_SUBDIR = "AT100_InVitro_DPD_Individual_Progression"
INDIVIDUAL_VALUES_CSV = "AT100_invitro_DPD_individual_values.csv"
AT50_INDIVIDUAL_SUBDIR = "AT50_InVitro_DPD_Individual_Progression"
AT50_INDIVIDUAL_VALUES_CSV = "AT50_invitro_DPD_individual_values.csv"
CT_INDIVIDUAL_SUBDIR = "CT_InVitro_DPD_Individual_Progression"
CT_INDIVIDUAL_VALUES_CSV = "CT_invitro_DPD_individual_values.csv"

EXPERIMENT_ROOT = EXP_RAW_DATA
EXAMPLE_STRATEGY = "AT100"
PROGRESSION_FRACTION = 0.5
CORE_STRATEGIES = {"CT", "AT50", "AT100"}
EXPANDED_STRATEGIES = CORE_STRATEGIES | {"LV-Agent", "AT100+"}

# Transcribed DPD initial-condition mapping used by the existing Figure 4
# comparison scripts.  These are all ten AT100 initial conditions.
AT100_MATCHES = (
    ("Plate_5", "A2_AT100-A2", "R20_AT100-R20"),
    ("Plate_5", "A5_AT100-A5", "R21_AT100-R21"),
    ("Plate_1", "B3_AT100-B3", "R22_AT100-R22"),
    ("Plate_1", "B5_AT100-B5", "R23_AT100-R23"),
    ("Plate_1", "B6_AT100-B6", "R24_AT100-R24"),
    ("Plate_2", "C1_AT100-C1", "R25_AT100-R25"),
    ("Plate_2", "C2_AT100-C2", "R26_AT100-R26"),
    ("Plate_2", "C4_AT100-C4", "R27_AT100-R27"),
    ("Plate_2", "C6_AT100-C6", "R28_AT100-R28"),
    ("Plate_3", "D4_AT100-D4", "R29_AT100-R29"),
)
AT50_MATCHES = (
    ("Plate_1", "A1_AT50-A1", "R10_AT50-R10"),
    ("Plate_2", "B2_AT50-B2", "R11_AT50-R11"),
    ("Plate_2", "B6_AT50-B6", "R12_AT50-R12"),
    ("Plate_3", "C2_AT50-C2", "R13_AT50-R13"),
    ("Plate_3", "C5_AT50-C5", "R14_AT50-R14"),
    ("Plate_4", "D1_AT50-D1", "R15_AT50-R15"),
    ("Plate_4", "D3_AT50-D3", "R16_AT50-R16"),
    ("Plate_4", "D4_AT50-D4", "R17_AT50-R17"),
    ("Plate_4", "D5_AT50-D5", "R18_AT50-R18"),
    ("Plate_4", "D6_AT50-D6", "R19_AT50-R19"),
)
CT_MATCHES = (
    ("Plate_2", "A1_CT-A1", "R30_CT-R30"),
    ("Plate_2", "A3_CT-A3", "R31_CT-R31"),
    ("Plate_3", "B2_CT-B2", "R32_CT-R32"),
    ("Plate_3", "B3_CT-B3", "R33_CT-R33"),
    ("Plate_4", "C1_CT-C1", "R34_CT-R34"),
    ("Plate_4", "C2_CT-C2", "R35_CT-R35"),
    ("Plate_4", "C4_CT-C4", "R36_CT-R36"),
    ("Plate_4", "C5_CT-C5", "R37_CT-R37"),
    ("Plate_5", "D2_CT-D2", "R38_CT-R38"),
    ("Plate_5", "D3_CT-D3", "R39_CT-R39"),
    ("Plate_5", "D4_CT-D4", "R40_CT-R40"),
    ("Plate_5", "D5_CT-D5", "R41_CT-R41"),
)

DPD_TREATMENT_STATUS = (
    source.DEFAULT_SIMULATION_ROOT
    / "summary_plots"
    / "requested_variations_20260716"
    / "simulation_treatment_status.tsv"
)
TREATMENT_COLOR = "#C8C8C8"
SIMULATION_TREATMENT_COLOR = "#BFBFBF"
BURDEN_COLOR = "#111111"
ESCAPE_COLOR = "#8A3FFC"
THRESHOLD_COLOR = "#D55E00"
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
FINAL_AVERAGE_PDF = "03_strategy_average_normalized_positive_carry_forward.pdf"
FINAL_EXAMPLE_PDF = "AT100_Plate_2_C2_normalized_invitro_DPD_R26.pdf"
FINAL_EXAMPLE_EVENT_TICKS = {15: "A", 24: "B", 29: "C"}
LINEWIDTH = 0.7
LINE_ALPHA = 0.34

ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--confinement-csv", type=Path, default=source.DEFAULT_CONFINEMENT_CSV)
    parser.add_argument("--escape-summary-csv", type=Path, default=source.DEFAULT_ESCAPE_SUMMARY_CSV)
    parser.add_argument("--simulation-root", type=Path, default=source.DEFAULT_SIMULATION_ROOT)
    parser.add_argument("--simulation-quantification-dir", type=Path, default=None)
    parser.add_argument("--dpd-treatment-status", type=Path, default=DPD_TREATMENT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-output-dir", type=Path, default=DATA_OUTPUT_DIR)
    parser.add_argument("--progression-fraction", type=float, default=PROGRESSION_FRACTION)
    parser.add_argument("--review-png-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def read_experimental_burdens(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pattern = re.compile(r"Results_G418_a2a4_(?P<sample>[A-H]\d+_(?P<strategy>[^-]+)-[A-H]\d+)\.txt$")
    for plate_dir in sorted(root.glob("Plate_*")):
        for path in sorted((plate_dir / "Results").glob("Results_G418_a2a4_*.txt")):
            match = pattern.match(path.name)
            if not match:
                continue
            strategy = normalize_strategy(match.group("strategy"))
            if strategy not in EXPANDED_STRATEGIES:
                continue
            table = pd.read_csv(path, sep=r"\s+", header=None, names=["token", "sensitive", "resistant"])
            table["day"] = pd.to_numeric(table["token"].str.extract(r"_d(\d+)$", expand=False), errors="coerce")
            table["plate"] = plate_dir.name
            table["sample"] = match.group("sample")
            table["strategy"] = strategy
            table["dataset_type"] = "In vitro"
            frames.append(table[["dataset_type", "plate", "sample", "strategy", "day", "sensitive", "resistant"]])
    if not frames:
        raise ValueError(f"No CT/AT50/AT100 result files found below {root}")
    data = pd.concat(frames, ignore_index=True).dropna(subset=["day", "sensitive", "resistant"])
    return data.sort_values(["plate", "sample", "day"])


def read_dpd_burdens(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    strategy_col = "strategy" if "strategy" in data else "treatment"
    data["strategy"] = data[strategy_col].astype(str).map(normalize_strategy)
    data = data[data["strategy"].isin(EXPANDED_STRATEGIES)].copy()
    data = source.filter_simulation_initial_condition_runs(data)
    data["dataset_type"] = "DPD"
    data["sensitive"] = pd.to_numeric(data["n_sensitive_clean"], errors="coerce")
    data["resistant"] = pd.to_numeric(data["n_resistant_total"], errors="coerce")
    data["day"] = pd.to_numeric(data["day"], errors="coerce")
    return data.dropna(subset=["day", "sensitive", "resistant"])


def add_progression_fields(burdens: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction:
        raise ValueError("--progression-fraction must be positive")
    frames: list[pd.DataFrame] = []
    for _, group in burdens.groupby(["dataset_type", "plate", "sample"], sort=False):
        group = group.sort_values("day").copy()
        initial_total = float(group.iloc[0]["sensitive"] + group.iloc[0]["resistant"])
        threshold = fraction * initial_total
        hits = group[group["resistant"] >= threshold]
        failure_day = float(hits.iloc[0]["day"]) if not hits.empty else np.nan
        group["initial_total"] = initial_total
        group["progression_threshold"] = threshold
        group["failure_day_unfiltered"] = failure_day
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def endpoint_table(burdens: pd.DataFrame, include_excluded: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in burdens.groupby(["dataset_type", "plate", "sample"], sort=False):
        dataset_type, plate, sample = keys
        last_day = float(group["day"].max())
        failure_day = float(group["failure_day_unfiltered"].iloc[0])
        truncation = np.nan
        if dataset_type == "In vitro" and not include_excluded:
            cutoff = EXCLUDE_SAMPLES.get(str(sample), np.inf)
            if cutoff is None:
                continue
            truncation = float(cutoff)
        candidates = [last_day]
        if np.isfinite(truncation):
            candidates.append(truncation)
        if np.isfinite(failure_day):
            candidates.append(failure_day)
        stop_day = min(candidates)
        observed_failure = bool(np.isfinite(failure_day) and failure_day <= stop_day)
        rows.append({"dataset_type": dataset_type, "plate": plate, "sample": sample,
                     "stop_day": stop_day, "truncation_day": truncation,
                     "failure_day": failure_day if observed_failure else np.nan,
                     "failure_day_unfiltered": failure_day,
                     "endpoint_reason": "progression" if observed_failure else
                         ("truncation" if np.isfinite(truncation) and truncation <= last_day else "last observation")})
    return pd.DataFrame(rows)


def trim_to_endpoints(data: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    dataset_labels = {"Simulation": "DPD", "In vitro": "In vitro", "DPD": "DPD"}
    trimmed = data.copy()
    trimmed["dataset_type"] = trimmed["dataset_type"].map(dataset_labels).fillna(trimmed["dataset_type"])
    trimmed = trimmed.merge(endpoints, on=["dataset_type", "plate", "sample"], how="inner")
    return trimmed[trimmed["day"] <= trimmed["stop_day"]].copy()


def load_signed_trajectories(args: argparse.Namespace, *, carry_forward: bool) -> pd.DataFrame:
    confinement = source.load_confinement(args.confinement_csv, args.include_excluded)
    metadata, escaped = source.load_escape_data(
        args.escape_summary_csv, args.include_excluded,
    )
    invitro = source.build_signed_trajectories(confinement, metadata, escaped)
    invitro = source.carry_forward_truncated_wells(invitro, enabled=True)
    invitro["dataset_type"] = "In vitro"

    quantification = args.simulation_quantification_dir or (
        args.simulation_root / source.SIMULATION_QUANTIFICATION_SUBDIR
    )
    confinement = source.load_confinement(
        quantification / source.SIMULATION_CONFINEMENT_FILENAME,
        include_excluded=True,
    )
    metadata, escaped = source.load_simulation_escape_data(
        quantification / source.SIMULATION_ESCAPE_COUNTS_FILENAME
    )
    confinement = source.filter_simulation_initial_condition_runs(confinement)
    metadata = source.filter_simulation_initial_condition_runs(metadata)
    escaped = source.filter_simulation_initial_condition_runs(escaped)
    dpd = source.build_signed_trajectories(confinement, metadata, escaped)
    dpd = source.carry_forward_truncated_wells(dpd, enabled=True)
    dpd["dataset_type"] = "DPD"
    trajectories = pd.concat([invitro, dpd], ignore_index=True)
    trajectories = trajectories[
        trajectories["strategy"].isin(EXPANDED_STRATEGIES)
    ].copy()
    if not carry_forward:
        trajectories = trajectories[~trajectories["carried_forward"].astype(bool)].copy()
    trajectories["plot_value"] = -pd.to_numeric(trajectories["signed_value"], errors="coerce")
    return trajectories


def read_invitro_treatment_days(root: Path, plate: str, sample: str) -> set[int]:
    schedule = root / plate / "treatment_schedule.txt"
    for line in schedule.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t", 1)
        if parts and parts[0] == sample:
            return {int(match.group(1)) for token in parts[1].split()
                    if (match := re.match(r"(\d+)", token))}
    raise ValueError(f"Treatment schedule entry not found for {plate}/{sample}")


def read_dpd_treatment_days(path: Path, strategy: str, dpd_sample: str) -> set[int]:
    status = pd.read_csv(path, sep="\t")
    rows = status[(status["plate"] == f"Plate_{strategy}") & (status["well"] == dpd_sample)].copy()
    treated = rows["is_treatment"] if rows["is_treatment"].dtype == bool else rows["is_treatment"].astype(str).str.lower().eq("true")
    return set(pd.to_numeric(rows.loc[treated, "first_day"], errors="coerce").dropna().astype(int))


def shade_treatment(ax: plt.Axes, days: set[int], stop_day: float, *, ymin: float,
                    ymax: float, hatch: str | None = None,
                    dashed: bool = False) -> None:
    valid = sorted(day for day in days if day <= stop_day)
    if not valid:
        return
    start = previous = valid[0]
    for day in valid[1:] + [10**9]:
        if day != previous + 1:
            is_simulation_hatch = hatch is not None
            ax.axvspan(
                start, min(previous + 1, stop_day), ymin=ymin, ymax=ymax,
                facecolor="none" if dashed or is_simulation_hatch else TREATMENT_COLOR,
                edgecolor=SIMULATION_TREATMENT_COLOR if is_simulation_hatch else TREATMENT_COLOR,
                hatch=hatch,
                alpha=1.0 if is_simulation_hatch else 0.52,
                linestyle="--" if dashed else "-",
                lw=0.8 if dashed else 0,
                zorder=0,
            )
            start = day
        previous = day


def summarize_carry_forward(trajectories: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_type, group in trajectories.groupby("dataset_type", sort=True):
        values = group.copy()
        values["signed_value"] = values["plot_value"]
        summary = source.summarize_strategy_averages(values)
        summary["dataset_type"] = dataset_type
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def summarize_median_iqr(trajectories: pd.DataFrame) -> pd.DataFrame:
    return (
        trajectories.groupby(["dataset_type", "strategy", "day"], sort=True)
        .agg(
            median_signed_value=("plot_value", "median"),
            q25_signed_value=("plot_value", lambda values: values.quantile(0.25)),
            q75_signed_value=("plot_value", lambda values: values.quantile(0.75)),
            n_wells=("plot_value", "size"),
            n_confinement=("phase", lambda values: int((values == "confinement").sum())),
            n_escaped=("phase", lambda values: int((values == "escaped").sum())),
        )
        .reset_index()
    )


def attach_initial_totals(trajectories: pd.DataFrame, burdens: pd.DataFrame) -> pd.DataFrame:
    baselines = burdens[
        ["dataset_type", "plate", "sample", "initial_total"]
    ].drop_duplicates(["dataset_type", "plate", "sample"])
    result = trajectories.merge(
        baselines, on=["dataset_type", "plate", "sample"], how="left",
        validate="many_to_one",
    )
    if result["initial_total"].isna().any() or (result["initial_total"] <= 0).any():
        missing = result.loc[
            result["initial_total"].isna() | (result["initial_total"] <= 0),
            ["dataset_type", "plate", "sample"],
        ].drop_duplicates()
        raise ValueError(f"Missing/invalid initial totals for:\n{missing.to_string(index=False)}")
    result["plot_value"] = result["plot_value"] / result["initial_total"]
    result["normalization"] = "initial_total_cell_count"
    return result


def positive_only_carry_forward(df: pd.DataFrame) -> pd.DataFrame:
    """LOCF only after a replicate has a positive escaped-cell endpoint."""
    carried = df.copy()
    carried["carried_forward"] = False
    carried["carry_forward_method"] = "observed"
    carried["source_day"] = carried["day"]
    strategy_days = {
        (str(dataset), str(strategy)): np.sort(group["day"].dropna().unique())
        for (dataset, strategy), group in carried.groupby(
            ["dataset_type", "strategy"], sort=False,
        )
    }
    additions: list[pd.Series] = []
    for _, group in carried.groupby(
        ["dataset_type", "plate", "sample"], sort=False,
    ):
        group = group.sort_values(["day", "phase"])
        last = group.iloc[-1]
        if float(last["plot_value"]) <= 0:
            continue
        future_days = strategy_days[(str(last["dataset_type"]), str(last["strategy"]))]
        future_days = future_days[future_days > float(last["day"])]
        for day in future_days:
            row = last.copy()
            row["day"] = float(day)
            row["carried_forward"] = True
            row["carry_forward_method"] = "positive_escaped_endpoint_locf"
            row["source_day"] = float(last["day"])
            additions.append(row)
    if additions:
        carried = pd.concat([carried, pd.DataFrame(additions)], ignore_index=True)
    return carried.sort_values(
        ["dataset_type", "strategy", "plate", "sample", "day", "phase"]
    )


def all_endpoint_carry_forward(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_type, group in df.groupby("dataset_type", sort=True):
        carried = source.carry_forward_truncated_wells(group, enabled=True)
        carried["dataset_type"] = dataset_type
        frames.append(carried)
    return pd.concat(frames, ignore_index=True)


def shared_y_limits(summary: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate([
        summary["mean_minus_std"].to_numpy(dtype=float),
        summary["mean_plus_std"].to_numpy(dtype=float),
        np.asarray([0.0, 0.5]),
    ])
    values = values[np.isfinite(values)]
    span = float(values.max() - values.min())
    padding = max(0.03, span * 0.05)
    return float(values.min() - padding), float(values.max() + padding)


def plot_strategy_averages(summary: pd.DataFrame, output: Path, show: bool,
                           review_png: Path | None = None, *, title: str = "",
                           ylabel: str = "Mean confinement margin (-) / escaped cells (+)",
                           value_column: str = "mean_signed_value",
                           lower_column: str = "mean_minus_std",
                           upper_column: str = "mean_plus_std",
                           show_progression_threshold: bool = False,
                           y_limits: tuple[float, float] | None = None,
                           x_limits: tuple[float, float] | None = None,
                           split_y_labels: bool = False,
                           width_mm: float = 50,
                           height_mm: float = 40) -> None:
    fig, ax = ft.new_figure(
        width=width_mm, height_mm=height_mm,
        margins=(15, 4, 11, 4), apply_style=False,
    )
    for (dataset_type, strategy), group in summary.groupby(["dataset_type", "strategy"], sort=True):
        group = group.sort_values("day")
        color = get_color(str(strategy), "#666666")
        linestyle = "--" if dataset_type == "DPD" else "-"
        ax.fill_between(group["day"], group[lower_column], group[upper_column],
                        color=color, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(group["day"], group[value_column], color=color,
                ls=linestyle, lw=1.5, zorder=2)
    ax.axhline(0, color="#222222", lw=0.8)
    if show_progression_threshold:
        ax.axhline(0.5, color="#222222", ls="--", lw=0.8, zorder=4)
    ax.set(xlabel="Time [days]", ylabel="" if split_y_labels else ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if title:
        ax.set_title(title, loc="center", fontsize=6.5, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    present_strategies = set(summary["strategy"].astype(str))
    strategy_handles = [Line2D([], [], color=get_color(s), lw=1.4, label=get_display_name(s))
                        for s in STRATEGY_ORDER if s in present_strategies]
    source_handles = [
        Line2D([], [], color="#333", lw=1.2, label="In vitro"),
        Line2D([], [], color="#333", lw=1.2, ls="--", label="DPD"),
    ]
    compact_legend = width_mm <= 30
    plot_legend = ax.legend(
        handles=strategy_handles + source_handles,
        frameon=True, framealpha=0.82, facecolor="white", edgecolor="none",
        loc="lower right", ncol=1,
        fontsize=4.5 if compact_legend else None,
        handlelength=1.5 if compact_legend else None,
        labelspacing=0.25 if compact_legend else None,
        borderpad=0.25 if compact_legend else None,
    )
    plot_legend.set_zorder(20)
    if split_y_labels:
        ax.text(-0.22, 0.76, "Escaped cells", transform=ax.transAxes,
                rotation=90, ha="center", va="center", fontsize=6)
        ax.text(-0.22, 0.24, "Confinement margin", transform=ax.transAxes,
                rotation=90, ha="center", va="center", fontsize=6)
    ft.savefig(fig, output)
    if review_png is not None:
        fig.savefig(review_png, dpi=200)
    plt.show() if show else plt.close(fig)


def plot_matched_strategy(
    trajectories: pd.DataFrame,
    burdens: pd.DataFrame,
    endpoints: pd.DataFrame,
    experiment_root: Path,
    treatment_status: Path,
    plate: str,
    sample: str,
    dpd_sample: str,
    strategy: str,
    output: Path,
    show: bool,
    common_xlim: tuple[float, float],
    common_ylim: tuple[float, float],
    review_png: Path | None = None,
) -> None:
    selections = (("In vitro", plate, sample, "-"),
                  ("DPD", f"Plate_{strategy}", dpd_sample, "--"))
    fig, ax = ft.new_figure(width=85, height_mm=62, margins=(14, 5, 11, 6), apply_style=False)
    all_values: list[np.ndarray] = []
    xmax = 0.0
    threshold_values: list[float] = []
    for dataset_type, source_plate, source_sample, linestyle in selections:
        trajectory = trajectories[(trajectories["dataset_type"] == dataset_type) &
                                  (trajectories["plate"] == source_plate) &
                                  (trajectories["sample"] == source_sample)].sort_values("day")
        burden = burdens[(burdens["dataset_type"] == dataset_type) &
                         (burdens["plate"] == source_plate) &
                         (burdens["sample"] == source_sample)].sort_values("day")
        endpoint_rows = endpoints[(endpoints["dataset_type"] == dataset_type) &
                                  (endpoints["plate"] == source_plate) &
                                  (endpoints["sample"] == source_sample)]
        if burden.empty or endpoint_rows.empty:
            message = (f"{dpd_sample.split('_')[0]} DPD output unavailable"
                       if dataset_type == "DPD" else "In-vitro well excluded")
            ax.text(0.99, 0.03, message, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.2, color=ESCAPE_COLOR)
            continue
        endpoint = endpoint_rows.iloc[0]
        stop = float(endpoint["stop_day"])
        trajectory = trajectory[trajectory["day"] <= stop]
        burden = burden[burden["day"] <= stop]
        days = (read_invitro_treatment_days(experiment_root, plate, sample)
                if dataset_type == "In vitro" else read_dpd_treatment_days(treatment_status, strategy, dpd_sample))
        shade_treatment(ax, days, stop, ymin=0.5 if dataset_type == "In vitro" else 0.0,
                        ymax=1.0 if dataset_type == "In vitro" else 0.5,
                        hatch=None if dataset_type == "In vitro" else "////")
        if not trajectory.empty:
            ax.plot(trajectory["day"], trajectory["plot_value"], color=ESCAPE_COLOR,
                    ls=linestyle, lw=1.35, zorder=3)
        elif dataset_type == "DPD":
            ax.text(0.99, 0.03, "DPD confinement unavailable", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.2, color=ESCAPE_COLOR)
        ax.plot(burden["day"], burden["resistant"], color=BURDEN_COLOR,
                ls=linestyle, lw=1.35, zorder=3)
        threshold = float(burden["progression_threshold"].iloc[0])
        threshold_values.append(threshold)
        escaped = trajectory[trajectory["phase"] == "escaped"]
        if not escaped.empty:
            ax.axvline(float(escaped["day"].min()), color=ESCAPE_COLOR,
                       ls=linestyle, lw=0.9, zorder=2)
        failure = endpoint["failure_day"]
        if np.isfinite(failure):
            ax.plot(float(failure), threshold, marker="o" if dataset_type == "In vitro" else "s",
                    ms=3.2, color=THRESHOLD_COLOR, zorder=4)
        xmax = max(xmax, stop)
        if not trajectory.empty:
            all_values.append(trajectory["plot_value"].to_numpy(float))
        all_values.append(burden["resistant"].to_numpy(float))
    # The matched initial conditions have the same initial total; draw one threshold.
    ax.axhline(float(np.mean(threshold_values)), color=THRESHOLD_COLOR, ls="--", lw=0.9, zorder=2)
    ax.axhline(0.0, color="#111111", lw=0.65, zorder=4)
    ax.set(xlim=common_xlim, ylim=common_ylim,
           xlabel="Time [days]", ylabel="Cell count / confinement margin")
    ax.set_title(f"{plate} / {sample} and matched DPD {dpd_sample.split('_')[0]}", loc="left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=ESCAPE_COLOR, lw=1.35, label="Confinement / escaped cells"),
        Line2D([], [], color=BURDEN_COLOR, lw=1.35, label="Total resistant cells"),
        Line2D([], [], color="#333", lw=1.2, label="In vitro"),
        Line2D([], [], color="#333", lw=1.2, ls="--", label="DPD"),
        Line2D([], [], color=THRESHOLD_COLOR, lw=0.9, ls="--", label="50% initial total"),
        Patch(facecolor=TREATMENT_COLOR, alpha=0.38, label="Treatment: in vitro (top) / DPD (bottom)"),
    ], frameon=False, fontsize=5.2, loc="upper left", ncol=2)
    ft.savefig(fig, output)
    if review_png is not None:
        fig.savefig(review_png, dpi=200)
    plt.show() if show else plt.close(fig)


def plot_final_example(
    trajectories: pd.DataFrame,
    burdens: pd.DataFrame,
    endpoints: pd.DataFrame,
    experiment_root: Path,
    treatment_status: Path,
    output: Path,
    show: bool,
    y_limits: tuple[float, float],
    review_png: Path | None = None,
) -> None:
    plate, sample, dpd_sample, strategy = (
        "Plate_2", "C2_AT100-C2", "R26_AT100-R26", "AT100"
    )
    selections = (
        ("In vitro", plate, sample, "-"),
        ("DPD", "Plate_AT100", dpd_sample, "--"),
    )
    prepared: list[tuple[str, pd.DataFrame, pd.DataFrame, str, set[int], float]] = []
    values: list[np.ndarray] = [np.asarray([0.0, 0.5])]
    for dataset_type, source_plate, source_sample, linestyle in selections:
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
            raise ValueError(f"Missing example data for {dataset_type}: {source_plate}/{source_sample}")
        stop = min(85.0, float(endpoint["stop_day"].iloc[0]))
        trajectory = trajectory[trajectory["day"] <= stop].copy()
        burden = burden[burden["day"] <= stop].copy()
        initial_total = float(burden["initial_total"].iloc[0])
        trajectory["normalized_value"] = trajectory["plot_value"] / initial_total
        burden["normalized_resistant"] = burden["resistant"] / initial_total
        treatment_days = (
            read_invitro_treatment_days(experiment_root, plate, sample)
            if dataset_type == "In vitro"
            else read_dpd_treatment_days(treatment_status, strategy, dpd_sample)
        )
        prepared.append(
            (dataset_type, trajectory, burden, linestyle, treatment_days, stop)
        )
        values.extend([
            trajectory["normalized_value"].to_numpy(float),
            burden["normalized_resistant"].to_numpy(float),
        ])

    ymin, ymax = y_limits
    zero_fraction = (0.0 - ymin) / (ymax - ymin)

    fig, ax = ft.new_figure(
        width=40, height_mm=40, margins=(15, 4, 15, 4), apply_style=False,
    )
    for dataset_type, trajectory, burden, linestyle, treatment_days, stop in prepared:
        shade_treatment(
            ax, treatment_days, stop,
            ymin=zero_fraction if dataset_type == "In vitro" else 0.0,
            ymax=1.0 if dataset_type == "In vitro" else zero_fraction,
            hatch="////////" if dataset_type == "DPD" else None,
        )
        confinement = trajectory[trajectory["phase"] == "confinement"]
        escaped = trajectory[trajectory["phase"] == "escaped"]
        ax.plot(
            confinement["day"], confinement["normalized_value"],
            color=FIGURE_COLOR_MAP["blue"], ls=linestyle,
            lw=1.25, zorder=3,
        )
        ax.plot(
            escaped["day"], escaped["normalized_value"],
            color=FIGURE_COLOR_MAP["yellow"], ls=linestyle,
            lw=1.25, zorder=3,
        )
        ax.plot(
            burden["day"], burden["normalized_resistant"],
            color="#111111", ls=linestyle, lw=1.25, zorder=3,
        )
        if not escaped.empty:
            ax.axvline(
                float(escaped["day"].min()), color=FIGURE_COLOR_MAP["yellow"],
                ls=linestyle, lw=0.75, zorder=2,
            )

    ax.axhline(0.0, color="#111111", lw=0.65, zorder=-1)
    ax.axhline(0.5, color="#111111", ls="--", lw=0.75, zorder=4)
    ax.set_xlim(0, 85)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("")
    ax.set_xticks(list(FINAL_EXAMPLE_EVENT_TICKS), minor=True)
    ax.set_xticklabels(list(FINAL_EXAMPLE_EVENT_TICKS.values()), minor=True)
    ax.tick_params(axis="x", which="minor", labelbottom=True, length=2.5, pad=1)
    ax.text(
        -0.22, 0.76, "Escaped cells",
        transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6,
    )
    ax.text(
        -0.22, 0.24, "Confinement margin",
        transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color="#333333", lw=0.85, label="In vitro"),
            Line2D([], [], color="#333333", lw=0.85, ls="--", label="DPD"),
        ],
        loc="lower right", frameon=False, fontsize=5.2,
    )
    ft.savefig(fig, output)
    if review_png is not None:
        fig.savefig(review_png, dpi=220)
    plt.show() if show else plt.close(fig)


def save_matched_cohort(
    *, args: argparse.Namespace, trajectories: pd.DataFrame, burdens: pd.DataFrame,
    endpoints: pd.DataFrame, strategy: str,
    matches: tuple[tuple[str, str, str], ...], subdir: str, values_filename: str,
    common_xlim: tuple[float, float], common_ylim: tuple[float, float],
) -> list[Path]:
    output_dir = args.output_dir / subdir
    data_output_dir = args.data_output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)
    invitro_keys = {(plate, sample) for plate, sample, _ in matches}
    dpd_samples = {dpd_sample for _, _, dpd_sample in matches}
    value_mask = burdens.apply(
        lambda row: (row["dataset_type"] == "DPD" and row["sample"] in dpd_samples)
        or (row["dataset_type"] == "In vitro" and (row["plate"], row["sample"]) in invitro_keys),
        axis=1,
    )
    burdens.loc[value_mask].to_csv(data_output_dir / values_filename, index=False)
    saved: list[Path] = []
    for plate, sample, dpd_sample in matches:
        filename = f"{plate}_{sample}_matched_{dpd_sample.split('_')[0]}.pdf"
        review_png = None
        if args.review_png_dir is not None:
            review_png = args.review_png_dir / filename.replace(".pdf", ".png")
        plot_matched_strategy(
            trajectories, burdens, endpoints, args.experiment_root,
            args.dpd_treatment_status, plate, sample, dpd_sample, strategy,
            output_dir / filename, args.show, common_xlim, common_ylim, review_png,
        )
        saved.append(output_dir / filename)
    return saved


def cohort_limits(
    trajectories: pd.DataFrame, burdens: pd.DataFrame, endpoints: pd.DataFrame,
    strategy: str, matches: tuple[tuple[str, str, str], ...],
) -> tuple[float, float, float]:
    values: list[np.ndarray] = []
    endpoint_days: list[float] = []
    for plate, sample, dpd_sample in matches:
        for dataset_type, source_plate, source_sample in (
            ("In vitro", plate, sample), ("DPD", f"Plate_{strategy}", dpd_sample)
        ):
            trajectory = trajectories[
                (trajectories["dataset_type"] == dataset_type)
                & (trajectories["plate"] == source_plate)
                & (trajectories["sample"] == source_sample)
            ]
            burden = burdens[
                (burdens["dataset_type"] == dataset_type)
                & (burdens["plate"] == source_plate)
                & (burdens["sample"] == source_sample)
            ]
            endpoint = endpoints[
                (endpoints["dataset_type"] == dataset_type)
                & (endpoints["plate"] == source_plate)
                & (endpoints["sample"] == source_sample)
            ]
            if not trajectory.empty:
                values.append(trajectory["plot_value"].to_numpy(float))
            if not burden.empty:
                values.extend([
                    burden["resistant"].to_numpy(float),
                    burden["progression_threshold"].to_numpy(float),
                ])
            if not endpoint.empty:
                endpoint_days.append(float(endpoint["stop_day"].iloc[0]))
    if not values or not endpoint_days:
        raise ValueError(f"No plottable values found for {strategy}")
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    span = float(finite.max() - finite.min())
    pad = max(1.0, span * 0.07)
    return float(finite.min() - pad), float(finite.max() + pad), max(endpoint_days)


def main() -> None:
    args = parse_args()
    quantification = args.simulation_quantification_dir or (
        args.simulation_root / source.SIMULATION_QUANTIFICATION_SUBDIR
    )
    experimental = read_experimental_burdens(args.experiment_root)
    dpd = read_dpd_burdens(quantification / source.SIMULATION_ESCAPE_COUNTS_FILENAME)
    burdens = add_progression_fields(
        pd.concat([experimental, dpd], ignore_index=True),
        args.progression_fraction,
    )
    endpoints = endpoint_table(burdens, args.include_excluded)
    burdens = trim_to_endpoints(burdens, endpoints)
    carry_forward_trajectories = load_signed_trajectories(args, carry_forward=True)
    raw_trajectories = carry_forward_trajectories[
        ~carry_forward_trajectories["carried_forward"].astype(bool)
    ].copy()
    trajectories = trim_to_endpoints(raw_trajectories, endpoints)
    core_trajectories = trajectories[
        trajectories["strategy"].isin(CORE_STRATEGIES)
    ].copy()
    normalized = attach_initial_totals(core_trajectories, burdens)
    normalized_positive_carried = positive_only_carry_forward(normalized)
    final_average = summarize_carry_forward(normalized_positive_carried)
    final_median = summarize_median_iqr(normalized_positive_carried)
    y_limits = shared_y_limits(final_average)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_output_dir.mkdir(parents=True, exist_ok=True)
    if args.review_png_dir is not None:
        args.review_png_dir.mkdir(parents=True, exist_ok=True)
    plot_strategy_averages(
        final_average,
        args.output_dir / FINAL_AVERAGE_PDF,
        args.show,
        None if args.review_png_dir is None else args.review_png_dir / "final_average.png",
        ylabel="Escape Margin",
        y_limits=(-0.4, 0.3),
        width_mm=50,
        height_mm=35,
    )
    final_median.to_csv(args.data_output_dir / NORMALIZED_POSITIVE_CARRY_MEDIAN_CSV, index=False)
    plot_strategy_averages(
        final_median,
        args.output_dir / NORMALIZED_POSITIVE_CARRY_MEDIAN_PDF,
        args.show,
        ylabel="Escape Margin",
        value_column="median_signed_value",
        lower_column="q25_signed_value",
        upper_column="q75_signed_value",
        y_limits=(-0.4, 0.3),
        width_mm=40,
    )
    plot_final_example(
        trajectories, burdens, endpoints, args.experiment_root,
        args.dpd_treatment_status, args.output_dir / FINAL_EXAMPLE_PDF, args.show,
        y_limits,
        None if args.review_png_dir is None else args.review_png_dir / "final_example.png",
    )
    print(f"[saved] {args.output_dir / FINAL_AVERAGE_PDF}")
    print(f"[saved] {args.output_dir / FINAL_EXAMPLE_PDF}")


if __name__ == "__main__":
    main()
