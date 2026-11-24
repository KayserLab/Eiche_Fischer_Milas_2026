#!/usr/bin/env python3
"""Export two pages of observed-failure escape-margin examples, 25 per source.

Five rows (strategies) by five columns (replicates), with no cohort summaries.
The examples are fixed. The AT100+ simulation row uses the new Dens dataset.
Experimental manual cutoffs apply. Simulation selections are independent,
non-pilot runs; columns are not necessarily matched experimental/simulation pairs.

Population curves use canonical HDF5 counts. All five components share the
initial canonical total as denominator. Black total is exactly blue sensitive
plus yellow resistant. In silico curves are dashed, matching Figure 1;
in vitro curves are solid. Escape is the first positive signed margin; failure is
the first resistant count >= 50% of the initial total. Events use acquisition
days (baseline day 1), without interpolating event times. Every example must
have observed failure; unobserved escape remains censored in the selection CSV.
Strategy-coloured bars above each panel show treatment-on intervals from the
experimental schedules or recorded simulation treatment states, clipped to
that example's retained observation window.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from analysis.figure_2.Event_Time_Analysis import (
    EXPERIMENT_ROOT, SIMULATION_ROOT, _read_population_file,
)
from shared.data_paths import COMBINED_RESULTS_CSV, figure_data_dir, figure_plot_dir, intermediate_dir
from shared.experiment_filters import EXCLUDE_SAMPLES, apply_sample_exclusions
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color, normalize_strategy

INPUT_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
DATA_DIR = figure_data_dir("Suppl_Figures", "escape_margin_over_time", create=False)
PLOT_DIR = figure_plot_dir("Suppl_Figures", "escape_margin_over_time", create=False)
VALUES_NAME = "escape_margin_25_examples_per_source_values.csv"
SELECTION_NAME = "escape_margin_25_examples_per_source_selection.csv"
TREATMENT_NAME = "escape_margin_25_examples_per_source_treatment_intervals.csv"
PDF_NAMES = {
    "experimental": "Escape_Margin_Over_Time_By_Strategy_InVitro_25_Examples_Rescaled_0p8_3.pdf",
    "simulation": "Escape_Margin_Over_Time_By_Strategy_InSilico_25_Examples_Rescaled_0p8_3.pdf",
}
STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")
EXPERIMENTAL_EXAMPLES = {
    "CT": (("Plate_4", "C5_CT-C5"), ("Plate_3", "B2_CT-B2"),
           ("Plate_4", "C2_CT-C2"), ("Plate_4", "C1_CT-C1"),
           ("Plate_2", "A3_CT-A3")),
    "AT50": (("Plate_2", "B6_AT50-B6"), ("Plate_2", "B2_AT50-B2"),
             ("Plate_4", "D5_AT50-D5"), ("Plate_4", "D6_AT50-D6"),
             ("Plate_3", "C4_AT50-C4")),
    "AT100": (("Plate_2", "C1_AT100-C1"), ("Plate_5", "A5_AT100-A5"),
              ("Plate_1", "B6_AT100-B6"), ("Plate_2", "C4_AT100-C4"),
              ("Plate_5", "A4_AT100-A4")),
    "LV-Agent": (("Plate_7", "C5_ATAgent2-C5"), ("Plate_7", "C4_ATAgent2-C4"),
                 ("Plate_7", "C2_ATAgent2-C2"), ("Plate_6", "B3_ATAgent2-B3"),
                 ("Plate_7", "A4_ATAgent2-A4")),
    "AT100+": (("Plate_8", "C5_Dens-C5"), ("Plate_8", "C2_Dens-C2"),
               ("Plate_8", "C3_Dens-C3"), ("Plate_8", "D5_Dens-D5"),
               ("Plate_8", "D3_Dens-D3")),
}
SIMULATION_RUNS = {
    "CT": (11, 39, 14, 34, 36),
    "AT50": (12, 11, 19, 15, 27),
    "AT100": (20, 26, 34, 42, 13),
    "LV-Agent": (20, 27, 16, 26, 28),
    # Keep 48 and 33. Replace unavailable runs with the real-seeded runs
    # closest to the cohort's median failure day (escape day breaks ties).
    "AT100+": (23, 48, 39, 20, 33),
}
SIMULATION_PROTOCOLS = {
    "CT": "CT", "AT50": "AT50", "AT100": "AT100",
    "LV-Agent": "Agent", "AT100+": "Dens",
}
X_LIMITS = {"experimental": (0.0, 100.0), "simulation": (0.0, 160.0)}
X_TICKS = {"experimental": (0, 25, 50, 75, 100), "simulation": (0, 40, 80, 120, 160)}
MARGIN_LIMITS, COUNT_LIMITS = (0.0, 0.8), (0.0, 3.0)
MARGIN_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8)
COUNT_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PAGE_WIDTH_MM, PAGE_HEIGHT_MM = ft.COL_WIDTHS_MM["nature_double"], 185.0
MARGINS_MM = (20.0, 14.0, 17.0, 29.0)
COLUMN_GAP_MM, ROW_GAP_MM = 6.0, 12.0
AXES_WIDTH_MM = (PAGE_WIDTH_MM - sum(MARGINS_MM[:2]) - 4 * COLUMN_GAP_MM) / 5
AXES_HEIGHT_MM = (PAGE_HEIGHT_MM - sum(MARGINS_MM[2:]) - 4 * ROW_GAP_MM) / 5
CONFINEMENT_COLOR, ESCAPE_COLOR = "#c8a2c8", "#f15a29"
SENSITIVE_COLOR, RESISTANT_COLOR = FIGURE_COLOR_MAP["blue"], FIGURE_COLOR_MAP["yellow"]
FAILURE_THRESHOLD, EVENT_MARKER_SIZE = 0.5, 3.4
EVENT_DOWN_PT = 2.0
FAILURE_MARKER = (8, 0, 22.5)
TREATMENT_BAR_Y, TREATMENT_LINEWIDTH = 1.045, 1.5
METRICS = (
    ("normalized_confinement_margin", CONFINEMENT_COLOR, "Confinement"),
    ("normalized_escape_margin", ESCAPE_COLOR, "Escape"),
    ("normalized_sensitive", SENSITIVE_COLOR, "Sensitive"),
    ("normalized_resistant", RESISTANT_COLOR, "Resistant"),
    ("normalized_total", "#000000", "Total"),
)


def selection_records() -> list[dict[str, object]]:
    """Resolve the approved choices without automatic matching or reselection."""
    records = []
    for strategy in STRATEGIES:
        for example, (plate, sample) in enumerate(EXPERIMENTAL_EXAMPLES[strategy], 1):
            records.append(dict(dataset_type="experimental", strategy=strategy,
                                example=example, plate=plate, sample=sample))
        protocol = SIMULATION_PROTOCOLS[strategy]
        for example, run in enumerate(SIMULATION_RUNS[strategy], 1):
            records.append(dict(dataset_type="simulation", strategy=strategy,
                                example=example, plate=f"Plate_{protocol}",
                                sample=f"R{run}_{protocol}-R{run}", simulation_run=run))
    return records


def event_day(group: pd.DataFrame, column: str, threshold: float) -> float:
    hit = group.loc[group[column].ge(threshold), "day"]
    return float(hit.min()) if not hit.empty else np.nan


def build_audit(values: pd.DataFrame) -> pd.DataFrame:
    records = []
    metadata = pd.read_csv(COMBINED_RESULTS_CSV, usecols=["protocol", "run", "initial_condition"])
    metadata["run"] = pd.to_numeric(metadata["run"], errors="coerce")
    for record in selection_records():
        group = values.loc[
            values.dataset_type.eq(record["dataset_type"]) & values.plate.eq(record["plate"])
            & values["sample"].eq(record["sample"])
        ].sort_values("day")
        if group.empty or not group.strategy.eq(record["strategy"]).all():
            raise ValueError(f"Missing or incorrect approved trajectory: {record}")
        last = float(group.day.max())
        tte = event_day(group, "signed_escape_margin", np.nextafter(0.0, 1.0))
        ttf = event_day(group, "normalized_resistant", FAILURE_THRESHOLD)
        cutoff = EXCLUDE_SAMPLES.get(str(record["sample"]))
        row = dict(record, n_points=len(group), first_day=float(group.day.min()),
                   last_day=last, initial_total=float(group.initial_total.iloc[0]),
                   tte_days=tte, ttf_days=ttf, escape_observed=np.isfinite(tte),
                   failure_observed=np.isfinite(ttf),
                   escape_censor_day=last if not np.isfinite(tte) else np.nan,
                   failure_censor_day=last if not np.isfinite(ttf) else np.nan,
                   manual_cutoff_day=cutoff,
                   selection_method="fixed observed failures with full-cohort dynamic replacements",
                   endpoint_reason=("manual quality cutoff" if cutoff is not None else
                                    "last available simulation observation" if record["dataset_type"] == "simulation"
                                    else "natural experimental endpoint"))
        if record["dataset_type"] == "simulation":
            if record["strategy"] == "AT100+":
                row["selection_method"] = "fixed observed failures from the new Dens cohort"
            matches = metadata.loc[
                metadata.protocol.eq(SIMULATION_PROTOCOLS[str(record["strategy"])])
                & metadata.run.eq(record["simulation_run"]), "initial_condition"
            ].dropna().unique()
            if len(matches) != 1:
                raise ValueError(f"Ambiguous simulation seed: {record}")
            row["simulation_initial_condition"] = matches[0]
            row["seed_matches_selected_experiment"] = matches[0] in {
                sample for _, sample in EXPERIMENTAL_EXAMPLES[str(record["strategy"])]
            }
        records.append(row)
    return pd.DataFrame(records)


def validate_values(values: pd.DataFrame) -> None:
    """Check scientific relationships and the exact approved set before export."""
    identities = ["dataset_type", "strategy", "example", "plate", "sample"]
    expected = {tuple(r[k] for k in identities) for r in selection_records()}
    actual = set(values[identities].drop_duplicates().itertuples(index=False, name=None))
    if actual != expected or values.duplicated(identities + ["day"]).any():
        raise ValueError("Values do not contain exactly the 50 approved unique trajectories")
    if not np.allclose(values.normalized_total.to_numpy(),
                       (values.normalized_sensitive + values.normalized_resistant).to_numpy(),
                       rtol=0, atol=1e-14):
        raise ValueError("Black total must equal blue plus yellow exactly")
    if not np.isfinite(values[[m for m, _, _ in METRICS]]).all().all():
        raise ValueError("A plotted normalized value is nonfinite")
    for key, group in values.groupby(identities):
        group = group.sort_values("day")
        if group.day.iloc[0] != 1 or not np.isclose(group.normalized_total.iloc[0], 1):
            raise ValueError(f"Missing or invalid day-1 baseline: {key}")
        if group.dataset_type.iloc[0] == "simulation" and not group.day.diff().dropna().eq(1).all():
            raise ValueError(f"Incomplete daily simulation record: {key}")
        if group.dataset_type.iloc[0] == "experimental":
            cutoff = EXCLUDE_SAMPLES.get(str(group['sample'].iloc[0]), "not_listed")
            if cutoff is None or (cutoff != "not_listed" and group.day.max() > cutoff):
                raise ValueError(f"Experimental exclusion/cutoff not respected: {key}")
        if group.day.max() > X_LIMITS[str(key[0])][1]:
            raise ValueError(f"Time window would clip a selected record: {key}")


def load_trajectories(path: Path = INPUT_CSV) -> pd.DataFrame:
    """Read margins and only the 50 selected canonical population HDF5s."""
    table = pd.read_csv(path)
    table["strategy"] = table.strategy.map(normalize_strategy)
    table = table.drop_duplicates(["dataset_type", "plate", "sample", "day"], keep="last")
    frames = []
    for record in selection_records():
        source, plate, sample = (str(record[k]) for k in ("dataset_type", "plate", "sample"))
        margin = table.loc[table.dataset_type.eq(source) & table.plate.eq(plate)
                           & table['sample'].eq(sample)].sort_values("day").copy()
        if source == "experimental":
            rules = {sample: EXCLUDE_SAMPLES[sample]} if sample in EXCLUDE_SAMPLES else {}
            margin = apply_sample_exclusions(margin, exclude_map=rules,
                                             column="sample", cutoff_column="day", strict=False)
            paths = list((EXPERIMENT_ROOT / plate / "hdf5").glob(f"*_{sample}_total_poisson_padded.h5"))
        else:
            paths = [SIMULATION_ROOT / plate / "hdf5" / f"{sample}_simulation.h5"]
        if margin.empty or len(paths) != 1 or not paths[0].is_file():
            raise ValueError(f"Missing/ambiguous selected inputs: {record}")
        counts = _read_population_file(paths[0], dataset_type=source, plate=plate,
                                       sample=sample, strategy=str(record["strategy"]))
        if source == "experimental":
            counts = apply_sample_exclusions(counts, exclude_map=rules,
                                             column="sample", cutoff_column="day", strict=False)
        if set(margin.day) != set(counts.day):
            raise ValueError(f"Margin/population acquisition days differ: {record}")
        group = margin.merge(counts[["day", "sus", "res", "total", "input_file"]].rename(
            columns={"input_file": "population_input_file"}), on="day", validate="one_to_one")
        initial = float(group.sort_values("day").total.iloc[0])
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError(f"Invalid initial population: {record}")
        group["initial_total"] = initial
        for raw, normalized in (("confinement_count", "normalized_confinement_margin"),
                                ("escaped_resistant_count", "normalized_escape_margin")):
            group[normalized] = group[raw].fillna(0) / initial
        group["normalized_sensitive"] = group.sus / initial
        group["normalized_resistant"] = group.res / initial
        group["normalized_total"] = group.normalized_sensitive + group.normalized_resistant
        group["example"] = record["example"]
        frames.append(group)
        print(f"[loaded] {source} {plate}/{sample}", flush=True)
    values = pd.concat(frames, ignore_index=True)
    validate_values(values)
    return values


def _event_marker(axis: plt.Axes, day: float, marker: object, color: str,
                  extra_down_pt: float = 0) -> None:
    if not np.isfinite(day):
        return
    offset = ScaledTranslation(0, -(EVENT_MARKER_SIZE * .52 + EVENT_DOWN_PT + extra_down_pt) / 72,
                               axis.figure.dpi_scale_trans)
    axis.plot([day], [0], marker=marker, markersize=EVENT_MARKER_SIZE,
              color=color, markeredgewidth=.4, linestyle="None",
              transform=axis.get_xaxis_transform() + offset, clip_on=False, zorder=7)


def load_treatment_intervals(audit: pd.DataFrame) -> pd.DataFrame:
    """Resolve recorded treatment days using the same day-1 calendar as the curves.

    Each treated day covers [day, day + 1); consecutive days are merged.
    Schedule suffix ``s`` marks stopping, not dosing; ``d`` is a treated day.
    """
    simulation = pd.read_csv(COMBINED_RESULTS_CSV, usecols=["protocol", "run", "day", "treatment"])
    simulation["run"] = pd.to_numeric(simulation["run"], errors="coerce")
    schedules: dict[str, dict[str, list[str]]] = {}
    rows = []
    for record in audit.itertuples(index=False):
        if record.dataset_type == "experimental":
            path = EXPERIMENT_ROOT / record.plate / "treatment_schedule.txt"
            if record.plate not in schedules:
                entries = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        parts = line.strip().split("\t", 1)
                        entries[parts[0]] = parts[1].split() if len(parts) > 1 else []
                schedules[record.plate] = entries
            if record.sample not in schedules[record.plate]:
                raise ValueError(f"Missing treatment schedule: {record.plate}/{record.sample}")
            days = []
            for token in schedules[record.plate][record.sample]:
                match = re.fullmatch(r"(\d+)([ds]?)", token)
                if match is None:
                    raise ValueError(f"Invalid treatment token {token!r} in {path}")
                if match.group(2) != "s":
                    days.append(int(match.group(1)))
        else:
            path = COMBINED_RESULTS_CSV
            group = simulation.loc[
                simulation.protocol.eq(SIMULATION_PROTOCOLS[record.strategy])
                & simulation.run.eq(record.simulation_run)
                & simulation.day.between(record.first_day, record.last_day)
            ].sort_values("day")
            expected_days = np.arange(record.first_day, record.last_day + 1)
            if (not np.array_equal(group.day.to_numpy(), expected_days)
                    or not group.treatment.isin([0, 1]).all()):
                raise ValueError(f"Missing/invalid daily treatment states: {record.plate}/{record.sample}")
            days = group.loc[group.treatment.eq(1), "day"].astype(int).tolist()
        days = sorted({day for day in days if record.first_day <= day < record.last_day})
        if not days:
            continue
        start = previous = days[0]
        for day in days[1:] + [None]:
            if day is None or day != previous + 1:
                rows.append(dict(
                    dataset_type=record.dataset_type, strategy=record.strategy,
                    example=record.example, plate=record.plate, sample=record.sample,
                    start_day=float(start), stop_day=min(float(previous + 1), record.last_day),
                    treatment_input_file=str(path),
                ))
                if day is None:
                    break
                start = day
            previous = day
    return pd.DataFrame(rows, columns=[
        "dataset_type", "strategy", "example", "plate", "sample",
        "start_day", "stop_day", "treatment_input_file",
    ])


def draw_treatment_timing(ax: plt.Axes, intervals: pd.DataFrame, strategy: str) -> None:
    """Use the Figure 4b treatment-bar placement immediately above the axes."""
    for interval in intervals.itertuples(index=False):
        ax.plot([interval.start_day, interval.stop_day], [TREATMENT_BAR_Y] * 2,
                transform=ax.get_xaxis_transform(), color=get_color(strategy),
                lw=TREATMENT_LINEWIDTH, solid_capstyle="butt", clip_on=False,
                zorder=8, gid="treatment-on")


def _strategy_label(fig: plt.Figure, y: float, strategy: str) -> None:
    """Keep superscript glyphs at a real 5 pt (no automatic font shrinkage)."""
    base = "CT" if strategy == "CT" else "AT"
    text = fig.text(2.5 / PAGE_WIDTH_MM, y, base, fontsize=7,
                    color=get_color(strategy), ha="left", va="center")
    superscript = {"AT50": "50", "AT100": "100", "LV-Agent": "agent", "AT100+": "+"}.get(strategy)
    if superscript:
        width = text.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
        fig.text(2.5 / PAGE_WIDTH_MM + width, y + 2.3 / (PAGE_HEIGHT_MM / 25.4 * 72),
                 superscript, fontsize=5, color=get_color(strategy), ha="left", va="center")


def _row_ylabels(fig: plt.Figure, center: float) -> None:
    """Repeat the colour-coded margin and population labels for each row."""
    for offset_mm, label, color in (
        (-4.5, r"$\eta^{\mathrm{conf}}_{\mathrm{sen}}$", CONFINEMENT_COLOR),
        (4.5, r"$\eta^{\mathrm{esc}}_{\mathrm{res}}$", ESCAPE_COLOR),
    ):
        fig.text(11 / PAGE_WIDTH_MM, center + offset_mm / PAGE_HEIGHT_MM,
                 label, fontsize=7, rotation=90, color=color, ha="center", va="center")
    for offset_mm, label, color in (
        (-6, r"$\eta_{\mathrm{sen}}$", SENSITIVE_COLOR),
        (0, r"$\eta_{\mathrm{res}}$", RESISTANT_COLOR),
        (6, r"$\eta_{\mathrm{tot}}$", "black"),
    ):
        fig.text(177 / PAGE_WIDTH_MM, center + offset_mm / PAGE_HEIGHT_MM,
                 label, fontsize=7, rotation=270, color=color, ha="center", va="center")


def plot_examples(values: pd.DataFrame, audit: pd.DataFrame, source: str, output: Path,
                  treatment_intervals: pd.DataFrame) -> None:
    """One publication-sized page containing all 25 approved source examples."""
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    linestyle = "--" if source == "simulation" else "-"
    page = values.loc[values.dataset_type.eq(source)]
    treatments = treatment_intervals.loc[treatment_intervals.dataset_type.eq(source)]
    fig, axes = ft.new_subplots(5, 5, ax_width_mm=AXES_WIDTH_MM, ax_height_mm=AXES_HEIGHT_MM,
                                margins=MARGINS_MM, wspace_mm=COLUMN_GAP_MM,
                                hspace_mm=ROW_GAP_MM, sharex=True, sharey=True,
                                squeeze=False, apply_style=False)
    for row, strategy in enumerate(STRATEGIES):
        center = (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2
        _strategy_label(fig, center, strategy)
        _row_ylabels(fig, center)
        for col in range(5):
            ax = axes[row, col]
            g = page.loc[page.strategy.eq(strategy) & page.example.eq(col + 1)].sort_values("day")
            record = audit.loc[audit.dataset_type.eq(source) & audit.strategy.eq(strategy)
                               & audit.example.eq(col + 1)].iloc[0]
            draw_treatment_timing(ax, treatments.loc[
                treatments.strategy.eq(strategy) & treatments.example.eq(col + 1)
            ], strategy)
            counts_ax = ax.twinx()
            counts_ax.set_position(ax.get_position())
            counts_ax.set_zorder(0)
            ax.patch.set_visible(False)
            for metric, color, _ in METRICS:
                target = ax if "margin" in metric else counts_ax
                target.plot(g.day, g[metric], color=color, lw=.85 if metric == "normalized_total" else .8,
                            linestyle=linestyle, zorder=2 if metric == "normalized_total" else 3)
            if np.isfinite(record.ttf_days):
                counts_ax.plot([record.ttf_days, X_LIMITS[source][1]], [.5, .5],
                               color="#bfbfbf", lw=.65, ls=(0, (4, 5)), zorder=0)
            ax.set(xlim=X_LIMITS[source], ylim=MARGIN_LIMITS, yticks=MARGIN_TICKS, xticks=X_TICKS[source])
            counts_ax.set(ylim=COUNT_LIMITS, yticks=COUNT_TICKS)
            ax.spines[["top", "right"]].set_visible(False)
            counts_ax.spines[["top", "left", "bottom"]].set_visible(False)
            ax.tick_params(axis="y", left=col == 0, labelleft=col == 0)
            counts_ax.tick_params(axis="y", right=col == 4, labelright=col == 4, labelsize=6)
            counts_ax.spines["right"].set_visible(col == 4)
            ax.tick_params(axis="x", labelbottom=True, pad=8)
            # Keep end labels inside each narrow axes box, away from neighbours.
            tick_labels = ax.get_xticklabels()
            tick_labels[0].set_ha("left")
            tick_labels[-1].set_ha("right")
            if row == 4:
                ax.set_xlabel("Time [d]", fontsize=7, labelpad=3)
            if source == "experimental":
                title = f"P{str(record.plate).split('_')[-1]} {str(record['sample']).rsplit('-', 1)[-1]}"
            else:
                title = f"R{int(record.simulation_run)}"
            ax.set_title(title, fontsize=6, pad=7)
            overlap = np.isfinite(record.tte_days) and np.isfinite(record.ttf_days) and record.tte_days == record.ttf_days
            _event_marker(ax, record.tte_days, "^", get_color(strategy), EVENT_MARKER_SIZE if overlap else 0)
            _event_marker(ax, record.ttf_days, FAILURE_MARKER, get_color(strategy))
    fig.legend(handles=[Line2D([], [], color=color, lw=.9, linestyle=linestyle, label=label)
                        for _, color, label in METRICS],
               loc="upper center", bbox_to_anchor=(.53, 1 - 5 / PAGE_HEIGHT_MM),
               ncol=5, frameon=False, fontsize=6, handlelength=1.5, columnspacing=1.3)
    event_handles = [Line2D([], [], color="black", marker=marker, lw=0, markersize=EVENT_MARKER_SIZE,
                            label=label) for marker, label in (("^", "Escape"), (FAILURE_MARKER, "Failure"))]
    event_handles.append(Line2D([], [], color="black", lw=TREATMENT_LINEWIDTH, label="Treatment on"))
    fig.legend(handles=event_handles,
               loc="upper center", bbox_to_anchor=(.53, 1 - 13 / PAGE_HEIGHT_MM),
               ncol=3, frameon=False, fontsize=6, handlelength=1, columnspacing=1.5)
    fig.canvas.draw()
    offenders = ft.audit_text_sizes(fig)
    if offenders:
        raise ValueError(f"Text outside the repository's 5-7 pt range: {offenders}")
    if not np.allclose(fig.get_size_inches() * 25.4, (PAGE_WIDTH_MM, PAGE_HEIGHT_MM)):
        raise ValueError("Incorrect publication page dimensions")
    for ax in fig.axes:
        if not np.allclose(ft.axes_size_mm(ax), (AXES_WIDTH_MM, AXES_HEIGHT_MM)):
            raise ValueError("Physical axes dimensions changed")
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=plt.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                raise ValueError(f"Text clipped by page boundary: {text.get_text()}")
    ft.savefig(fig, output)
    plt.close(fig)
    print(f"[saved] {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--replot-from", type=Path, help="Replot a previously exported, validated values CSV.")
    args = parser.parse_args()
    values = pd.read_csv(args.replot_from) if args.replot_from else load_trajectories()
    validate_values(values)
    # Re-form the exact sum after a CSV roundtrip before plotting/exporting.
    values["normalized_total"] = values.normalized_sensitive + values.normalized_resistant
    audit = build_audit(values)
    if not audit.failure_observed.all():
        missing = audit.loc[~audit.failure_observed, ["dataset_type", "plate", "sample"]]
        raise ValueError(f"Every selected example must reach the resistant failure threshold:\n{missing}")
    treatment_intervals = load_treatment_intervals(audit)
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.output_data_dir / VALUES_NAME, index=False)
    audit.to_csv(args.output_data_dir / SELECTION_NAME, index=False)
    treatment_intervals.to_csv(args.output_data_dir / TREATMENT_NAME, index=False)
    for source in PDF_NAMES:
        plot_examples(values, audit, source, args.output_plot_dir / PDF_NAMES[source], treatment_intervals)
    print(f"[saved] {args.output_data_dir / VALUES_NAME}")
    print(f"[saved] {args.output_data_dir / SELECTION_NAME}")
    print(f"[saved] {args.output_data_dir / TREATMENT_NAME}")


if __name__ == "__main__":
    main()
