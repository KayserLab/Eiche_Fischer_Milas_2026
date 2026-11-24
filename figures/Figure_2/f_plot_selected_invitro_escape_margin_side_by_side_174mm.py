"""Sidequest: three selected in-vitro trajectories side by side on 174 mm.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
Treatment exposure is shown as a compact strategy-coloured timing line above
each selected trajectory. Each trajectory is retained up to its manual
truncation (or last available observation), irrespective of its
progression/failure time.

This duplicate exports horizontal versions, with and without cell counts,
to a separate Sidequest_174mm folder. The 174 mm canvas leaves 3 mm at each
edge when centred on a 180 mm page, with wider gaps and narrower panels than
the 170 mm revision. The original stacked script is independent.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: all_escape_margin.csv plus experimental Plate_*/Results/*.txt
# trajectories.
# Scripts: analysis/figure_2/Escape_Margin_Analysis.py,
# analysis/figure_2/plot_selected_escape_progression_analysis.py,
# shared/experiment_filters.py, shared/strategy_colors.py, and shared/figtools.py.
# Outputs: selected-replicate margin PDFs, plotted-values CSV, and audit CSV.

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.data_paths import EXP_RAW_DATA, figure_data_dir
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color, get_display_name, normalize_strategy
from analysis.figure_2 import experimental_burden as progression
from analysis.figure_2.Event_Time_Analysis import ESCAPE_MARGIN_CSV

OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "Sidequest_174mm"
DATA_OUTPUT_DIR = figure_data_dir("Figure_2", "selected_escape_margin", "side_by_side_174mm", create=False)
OUTPUT_PDF = OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_escape_margin_until_truncation.pdf"
OUTPUT_WITH_COHORT_PDF = OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_escape_margin_until_truncation_with_cohort_background.pdf"
OUTPUT_SPLIT_SIGNED_PDF = OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_escape_margin_split_positive_negative.pdf"
OUTPUT_SEPARATE_MARGINS_PDF = OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_separate_confinement_escape_margins_side_by_side_174mm.pdf"
OUTPUT_SEPARATE_MARGINS_WITH_CELLS_PDF = OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_separate_confinement_escape_margins_with_cell_counts_side_by_side_174mm.pdf"
OUTPUT_CSV = DATA_OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_escape_margin_until_truncation_values.csv"
OUTPUT_AUDIT_CSV = DATA_OUTPUT_DIR / "CT_AT50_AT100_selected_invitro_escape_margin_until_truncation_audit.csv"
# (strategy, plate, sample)
SELECTED_REPLICATES = (
    ("CT", "Plate_3", "B2_CT-B2"),
    ("AT50", "Plate_2", "B2_AT50-B2"),
    ("AT100", "Plate_5", "A5_AT100-A5"),
)
Y_LIMITS = (-0.4, 0.4)
X_LIMITS = (0.0, 60.0)
CORE_STRATEGIES = {"CT", "AT50", "AT100"}
# Slightly brighter background-only colours retain cohort visibility without
# changing the canonical colours of the three representative trajectories.
BACKGROUND_COLORS = {"AT50": "#39BDF8", "AT100": "#E95A70"}
CONFINEMENT_COLOR = "#c8a2c8"
ESCAPE_COLOR = "#f15a29"
SENSITIVE_CELL_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_CELL_COLOR = FIGURE_COLOR_MAP["yellow"]
FAILURE_THRESHOLD = 0.5
FAILURE_REFERENCE_COLOR = "#bfbfbf"
FAILURE_REFERENCE_LINEWIDTH = 0.75
FAILURE_REFERENCE_LINESTYLE = (0.0, (4.0, 5.0))
AXIS_LINEWIDTH = 0.8
EVENT_MARKER_SIZE = 3.4
EVENT_MARKER_AXIS_CLEARANCE_PT = 0.5
TREATMENT_LINEWIDTH = 1.5
FIGURE_WIDTH_MM = 174.0
SIDE_BY_SIDE_AXES_HEIGHT_MM = 20.0
SIDE_BY_SIDE_PANEL_GAP_MM = 11.0

# Embed TrueType fonts (PDF Type 42) rather than Type 3 glyphs.  This keeps
# axis labels editable as text when the PDF is opened in Adobe Illustrator.
ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)


def _manual_stop_day(sample: str, last_available_day: float) -> tuple[float, str]:
    """Return the manual truncation endpoint, never a failure endpoint."""
    cutoff = EXCLUDE_SAMPLES.get(sample, np.inf)
    if cutoff is not None and np.isfinite(cutoff):
        return min(float(cutoff), last_available_day), "manual truncation"
    return last_available_day, "last observation"


def _treatment_intervals(plate: str, sample: str) -> list[tuple[float, float]]:
    """Return consecutive treatment-day intervals for one selected replicate."""
    schedule_path = EXP_RAW_DATA / plate / "treatment_schedule.txt"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Treatment schedule not found: {schedule_path}")
    treatment_days: list[int] = []
    for raw_line in schedule_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split("\t", maxsplit=1)
        if not parts or parts[0].strip() != sample:
            continue
        for token in parts[1].split() if len(parts) > 1 else []:
            token = token.strip()
            if not token or token.endswith("s"):
                continue
            if token.endswith("d"):
                token = token[:-1]
            try:
                treatment_days.append(int(float(token)))
            except ValueError:
                continue
        break
    if not treatment_days:
        return []
    intervals: list[tuple[float, float]] = []
    start = end = treatment_days[0]
    for day in treatment_days[1:]:
        if day == end + 1:
            end = day
        else:
            intervals.append((float(start), float(end + 1)))
            start = end = day
    intervals.append((float(start), float(end + 1)))
    return intervals


def _draw_treatment_timing(
    ax: plt.Axes, plate: str, sample: str, color: str, final_day: float,
) -> None:
    """Draw treatment exposure as line segments immediately above the axis."""
    for start, stop in _treatment_intervals(plate, sample):
        start = max(start, X_LIMITS[0])
        stop = min(stop, X_LIMITS[1], final_day)
        if stop <= start:
            continue
        ax.plot(
            [start, stop], [1.045, 1.045], transform=ax.get_xaxis_transform(),
            color=color, lw=TREATMENT_LINEWIDTH, solid_capstyle="butt",
            clip_on=False, zorder=8,
        )


def load_escape_margins(path: Path) -> pd.DataFrame:
    """Load canonical experimental escape margins and validate their schema."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical escape-margin table not found: {path}. Run "
            "analysis/figure_2/Escape_Margin_Analysis.py first."
        )
    margins = pd.read_csv(path)
    required = {
        "dataset_type",
        "plate",
        "sample",
        "strategy",
        "day",
        "signed_escape_margin",
    }
    missing = required.difference(margins.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    margins = margins[
        margins["dataset_type"].astype(str).str.lower().eq("experimental")
    ].copy()
    margins["strategy"] = margins["strategy"].astype(str).map(normalize_strategy)
    margins["day"] = pd.to_numeric(margins["day"], errors="coerce")
    margins["escape_margin"] = pd.to_numeric(
        margins["signed_escape_margin"], errors="coerce"
    )
    margins = margins[
        np.isfinite(margins["day"]) & np.isfinite(margins["escape_margin"])
    ].copy()
    return margins.drop_duplicates(
        subset=["plate", "sample", "day"], keep="last"
    ).sort_values(["strategy", "plate", "sample", "day"])


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_selected_data(
    escape_margin_csv: Path = ESCAPE_MARGIN_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load selected and cohort trajectories through their manual truncations."""
    margins = load_escape_margins(escape_margin_csv)

    # Reuse the experimental result reader for its filename parsing and count fields.
    figure_dir = THIS_FILE.parent
    if str(figure_dir) not in sys.path:
        sys.path.insert(0, str(figure_dir))

    burdens = progression.read_experimental_burdens(progression.EXPERIMENT_ROOT).copy()
    selected_keys = {(strategy, plate, sample) for strategy, plate, sample in SELECTED_REPLICATES}
    selected_margin_frames: list[pd.DataFrame] = []
    selected_burden_frames: list[pd.DataFrame] = []
    background_margin_frames: list[pd.DataFrame] = []
    background_burden_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for strategy, plate, sample in burdens[["strategy", "plate", "sample"]].drop_duplicates().itertuples(index=False):
        if strategy not in CORE_STRATEGIES or EXCLUDE_SAMPLES.get(sample, np.inf) is None:
            continue
        margin = margins[(margins["strategy"] == strategy) & (margins["plate"] == plate) & (margins["sample"] == sample)].copy()
        burden = burdens[(burdens["strategy"] == strategy) & (burdens["plate"] == plate) & (burdens["sample"] == sample)].copy()
        if margin.empty or burden.empty:
            if (strategy, plate, sample) in selected_keys:
                raise ValueError(f"Missing selected replicate: {strategy} {plate}/{sample}")
            continue

        last_available_day = min(float(margin["day"].max()), float(burden["day"].max()))
        stop_day, endpoint_reason = _manual_stop_day(sample, last_available_day)
        margin = margin[margin["day"] <= stop_day].sort_values("day").copy()
        burden = burden[burden["day"] <= stop_day].sort_values("day").copy()
        initial_total = float(burden.iloc[0]["sensitive"] + burden.iloc[0]["resistant"])
        burden["normalized_sensitive"] = burden["sensitive"] / initial_total
        burden["normalized_resistant"] = burden["resistant"] / initial_total
        margin["normalized_escape_margin"] = margin["escape_margin"] / initial_total
        for frame in (margin, burden):
            frame["strategy"] = strategy
            frame["plate"] = plate
            frame["sample"] = sample

        escaped_rows = margin[margin["normalized_escape_margin"] > 0]
        escape_day = float(escaped_rows["day"].min()) if not escaped_rows.empty else np.nan
        if (strategy, plate, sample) in selected_keys:
            audit_rows.append({
                "strategy": strategy,
                "plate": plate,
                "sample": sample,
                "initial_total_cells": initial_total,
                "last_common_available_day": last_available_day,
                "stop_day": stop_day,
                "endpoint_reason": endpoint_reason,
                "escape_day": escape_day,
            })
            selected_margin_frames.append(margin)
            selected_burden_frames.append(burden)
        else:
            background_margin_frames.append(margin)
            background_burden_frames.append(burden)

    return (
        _concat_or_empty(selected_margin_frames),
        _concat_or_empty(selected_burden_frames),
        _concat_or_empty(background_margin_frames),
        _concat_or_empty(background_burden_frames),
        pd.DataFrame(audit_rows),
    )


def draw_signed_margin(ax: plt.Axes, margin: pd.DataFrame, *, fill_color: str) -> None:
    """Draw a solid treatment-strategy-coloured escape margin."""
    x = margin["day"].to_numpy(float)
    y = margin["normalized_escape_margin"].to_numpy(float)
    ax.plot(x, y, color=fill_color, lw=1.2, zorder=3)


def _selected_margin(
    margins: pd.DataFrame, strategy: str, plate: str, sample: str
) -> pd.DataFrame:
    """Return one selected margin trajectory in time order."""
    key = (
        (margins["strategy"] == strategy)
        & (margins["plate"] == plate)
        & (margins["sample"] == sample)
    )
    margin = margins[key].sort_values("day")
    if margin.empty:
        raise ValueError(f"Missing selected margin: {strategy} {plate}/{sample}")
    return margin


def _selected_burden(
    burdens: pd.DataFrame, strategy: str, plate: str, sample: str
) -> pd.DataFrame:
    """Return one selected measured cell-count trajectory in time order."""
    key = (
        (burdens["strategy"] == strategy)
        & (burdens["plate"] == plate)
        & (burdens["sample"] == sample)
    )
    burden = burdens[key].sort_values("day")
    if burden.empty:
        raise ValueError(f"Missing selected burden: {strategy} {plate}/{sample}")
    return burden


def _separate_margins(margin: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the signed metric into non-negative confinement and escape margins."""
    day = margin["day"].to_numpy(float)
    signed = margin["normalized_escape_margin"].to_numpy(float)
    confinement = np.maximum(-signed, 0.0)
    escape = np.maximum(signed, 0.0)
    return day, confinement, escape


def _add_component_margin_ylabels(
    fig: plt.Figure, axes: np.ndarray, *, with_cell_counts: bool,
) -> None:
    """Place shared component labels just outside the first and last panels."""
    canvas_width_mm = fig.get_figwidth() * 25.4
    position = axes[0].get_position()
    left_x = position.x0 - (7.0 if with_cell_counts else 8.0) / canvas_width_mm
    right_x = axes[-1].get_position().x1 + 6.5 / canvas_width_mm
    lower_y = position.y0 + 0.34 * position.height
    upper_y = position.y0 + 0.66 * position.height
    middle_y = position.y0 + 0.50 * position.height

    label_parts = [
        (left_x, lower_y, r"$\eta^{\mathrm{conf}}_{\mathrm{sen}}$", CONFINEMENT_COLOR, 90),
        (left_x, upper_y, r"$\eta^{\mathrm{esc}}_{\mathrm{res}}$", ESCAPE_COLOR, 90),
    ]
    for label_x, label_y, label, color, rotation in label_parts:
        fig.text(
            label_x, label_y, label, rotation=rotation, ha="center", va="center",
            color=color, fontsize=6.5,
        )

    # A short, heavier horizontal rule reads as the rotated vertical bar.
    for label_x in (left_x,):
        fig.add_artist(Line2D(
            [label_x - 0.6 / canvas_width_mm, label_x + 0.6 / canvas_width_mm],
            [middle_y, middle_y],
            color="#000000", lw=1.0, solid_capstyle="butt",
            transform=fig.transFigure, clip_on=False,
        ))

    if with_cell_counts:
        # Align coloured text runs along one downward-reading, editable title.
        title_parts = [
            ("Normalized count, ", "#000000"),
            (r"$\eta_{\mathrm{res}}$", RESISTANT_CELL_COLOR),
            (" | ", "#000000"),
            (r"$\eta_{\mathrm{sen}}$", SENSITIVE_CELL_COLOR),
        ]
        texts = [
            fig.text(
                right_x, middle_y, text, color=color, fontsize=6.5,
                rotation=270, rotation_mode="anchor", ha="left", va="baseline",
            )
            for text, color in title_parts
        ]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lengths = [text.get_window_extent(renderer).height / fig.bbox.height for text in texts]
        top = middle_y + sum(lengths) / 2.0
        for text, length in zip(texts, lengths):
            text.set_position((right_x, top))
            top -= length

def _event_day(burden: pd.DataFrame, *, column: str) -> float:
    """Return the first observed day satisfying an event condition."""
    event = burden.loc[burden[column], "day"]
    return float(event.iloc[0]) if not event.empty else np.nan


def _threshold_crossing_day(
    days: np.ndarray, values: np.ndarray, threshold: float,
) -> float:
    """Return the first upward threshold crossing, linearly interpolated."""
    for index, value in enumerate(values):
        if value < threshold:
            continue
        if index == 0:
            return float(days[0])
        previous_value = values[index - 1]
        previous_day = days[index - 1]
        if previous_value >= threshold or np.isclose(value, previous_value):
            return float(days[index])
        fraction = (threshold - previous_value) / (value - previous_value)
        return float(previous_day + fraction * (days[index] - previous_day))
    return np.nan


def _draw_event_marker(
    ax: plt.Axes,
    day: float,
    *,
    marker: object,
    top_extent: float,
    color: str,
    additional_down_pt: float = 0.0,
) -> None:
    """Place a marker directly below the x-axis, with its edge touching it."""
    if not np.isfinite(day) or not X_LIMITS[0] <= day <= X_LIMITS[1]:
        return
    # Offset by the marker's actual upper extent: 0.5 for the triangle tip and
    # cos(22.5 deg) / 2 for the flat-topped octagon.
    offset = ScaledTranslation(
        0.0,
        -(
            EVENT_MARKER_SIZE * top_extent
            + EVENT_MARKER_AXIS_CLEARANCE_PT
            + additional_down_pt
        ) / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.plot(
        [day], [0.0], marker=marker, markersize=EVENT_MARKER_SIZE,
        markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.4,
        linestyle="None", transform=ax.get_xaxis_transform() + offset,
        clip_on=False, zorder=7,
    )


def make_split_signed_plot(margins: pd.DataFrame, output: Path) -> None:
    """Plot positive and negative halves of the signed margin on separate axes."""
    fig, axes = ft.new_subplots(
        nrows=2,
        ncols=1,
        ax_width_mm=30,
        ax_height_mm=16,
        margins=(15, 4, 10, 4),
        hspace_mm=4,
        sharex=True,
        apply_style=False,
    )
    upper, lower = np.asarray(axes).reshape(-1)
    for strategy, plate, sample in SELECTED_REPLICATES:
        margin = _selected_margin(margins, strategy, plate, sample)
        day = margin["day"].to_numpy(float)
        signed = margin["normalized_escape_margin"].to_numpy(float)
        color = get_color(strategy)
        upper.plot(
            day, np.where(signed >= 0.0, signed, np.nan),
            color=color, lw=1.2, zorder=3,
        )
        lower.plot(
            day, np.where(signed <= 0.0, signed, np.nan),
            color=color, lw=1.2, zorder=3,
        )

    upper.set(xlim=X_LIMITS, ylim=(0.0, Y_LIMITS[1]))
    lower.set(xlim=X_LIMITS, ylim=(Y_LIMITS[0], 0.0))
    upper.tick_params(axis="x", which="both", labelbottom=True)
    upper.spines[["top", "right"]].set_visible(False)
    lower.spines[["bottom", "right"]].set_visible(False)
    lower.spines["top"].set_visible(True)
    lower.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    fig.supylabel(r"$\eta_{\mathrm{esc}}-\eta_{\mathrm{conf}}$", x=0.025)
    fig.supxlabel("Time [d]", y=0.02)
    ft.savefig(fig, output)
    plt.close(fig)


def make_separate_margin_plot(
    margins: pd.DataFrame,
    burdens: pd.DataFrame,
    output: Path,
    *,
    with_cell_counts: bool = False,
) -> None:
    """Plot the three examples in one row on an exactly 174 mm-wide canvas."""
    n_panels = len(SELECTED_REPLICATES)
    margins_mm = (10.0 if with_cell_counts else 12.0, 10.0 if with_cell_counts else 3.5, 11.0, 8.0)
    axes_width_mm = (
        FIGURE_WIDTH_MM - margins_mm[0] - margins_mm[1]
        - (n_panels - 1) * SIDE_BY_SIDE_PANEL_GAP_MM
    ) / n_panels
    fig, axes = ft.new_subplots(
        nrows=1,
        ncols=n_panels,
        ax_width_mm=axes_width_mm,
        ax_height_mm=SIDE_BY_SIDE_AXES_HEIGHT_MM,
        margins=margins_mm,
        wspace_mm=SIDE_BY_SIDE_PANEL_GAP_MM,
        sharex=True,
        sharey=with_cell_counts,
        apply_style=False,
    )
    axes = np.asarray(axes).reshape(-1)
    count_axes: list[plt.Axes] = []
    for index, (ax, (strategy, plate, sample)) in enumerate(zip(axes, SELECTED_REPLICATES)):
        margin = _selected_margin(margins, strategy, plate, sample)
        day, confinement, escape = _separate_margins(margin)
        escape_day = _event_day(
            margin.assign(_escaped=margin["normalized_escape_margin"].gt(0.0)),
            column="_escaped",
        )
        visible = (day >= X_LIMITS[0]) & (day <= X_LIMITS[1])
        largest_value = max(
            float(confinement[visible].max(initial=0.0)),
            float(escape[visible].max(initial=0.0)),
        )

        if with_cell_counts:
            burden = _selected_burden(burdens, strategy, plate, sample)
            failure_day = _event_day(
                burden.assign(_failed=burden["normalized_resistant"].ge(FAILURE_THRESHOLD)),
                column="_failed",
            )
            count_ax = ax.twinx()
            count_ax.set_position(ax.get_position())
            count_ax.plot(
                burden["day"], burden["normalized_sensitive"],
                color=SENSITIVE_CELL_COLOR, ls="-",
                lw=1.0, zorder=1,
            )
            count_ax.plot(
                burden["day"], burden["normalized_resistant"],
                color=RESISTANT_CELL_COLOR, ls="-",
                lw=1.0, zorder=1,
            )
            resistant_crossing_day = _threshold_crossing_day(
                burden["day"].to_numpy(float),
                burden["normalized_resistant"].to_numpy(float),
                FAILURE_THRESHOLD,
            )
            if np.isfinite(resistant_crossing_day) and X_LIMITS[0] <= resistant_crossing_day <= X_LIMITS[1]:
                count_ax.plot(
                    [resistant_crossing_day, X_LIMITS[1]],
                    [FAILURE_THRESHOLD, FAILURE_THRESHOLD],
                    color=FAILURE_REFERENCE_COLOR, linewidth=FAILURE_REFERENCE_LINEWIDTH,
                    linestyle=FAILURE_REFERENCE_LINESTYLE, dash_capstyle="butt", zorder=0,
                )
            count_ax.set_ylim(0.0, 1.5)
            count_ax.spines[["left", "bottom", "top"]].set_visible(False)
            count_ax.spines["right"].set_linestyle("-")
            count_ax.spines["right"].set_linewidth(AXIS_LINEWIDTH)
            count_ax.spines["right"].set_color("#000000")
            count_ax.spines["right"].set_visible(True)
            count_ax.tick_params(
                axis="y", colors="#000000", labelsize=5,
                right=True, labelright=True,
            )
            count_ax.set_zorder(0)
            ax.patch.set_visible(False)
            count_axes.append(count_ax)

        ax.plot(
            day, confinement, color=CONFINEMENT_COLOR,
            ls="-", lw=1.2, zorder=3,
        )
        ax.plot(
            day, escape, color=ESCAPE_COLOR,
            ls="-", lw=1.2, zorder=3,
        )
        margin_ylim = (0.0, 0.4) if with_cell_counts else (
            0.0, max(Y_LIMITS[1], 1.05 * largest_value)
        )
        ax.set(xlim=X_LIMITS, ylim=margin_ylim)
        ax.set_xlabel("Time [d]", fontsize=7)
        ax.text(
            0.02, 0.94, get_display_name(strategy, superscript=True), transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color=get_color(strategy, "#000000"),
        )
        _draw_treatment_timing(
            ax, plate, sample, get_color(strategy, "#000000"),
            final_day=float(day[visible].max()),
        )
        ax.spines[["top", "right"]].set_visible(False)
        if with_cell_counts:
            ax.tick_params(axis="x", pad=6)
            if index:
                ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
            strategy_color = get_color(strategy, "#000000")
            events_overlap = (
                np.isfinite(escape_day) and np.isfinite(failure_day)
                and np.isclose(escape_day, failure_day)
            )
            _draw_event_marker(
                ax, escape_day, marker="^", top_extent=0.52,
                color=strategy_color,
                additional_down_pt=EVENT_MARKER_SIZE if events_overlap else 0.0,
            )
            _draw_event_marker(
                ax,
                failure_day,
                marker=(8, 0, 22.5),
                top_extent=0.49,
                color=strategy_color,
            )

    _add_component_margin_ylabels(fig, axes, with_cell_counts=with_cell_counts)
    legend_handles = (
        [
            Line2D(
                [], [], color="#000000", marker="^", markersize=EVENT_MARKER_SIZE,
                markerfacecolor="#000000", markeredgewidth=0.4, linestyle="None",
                label="Escape",
            ),
            Line2D(
                [], [], color="#000000", marker=(8, 0, 22.5), markersize=EVENT_MARKER_SIZE,
                markerfacecolor="#000000", markeredgewidth=0.4, linestyle="None",
                label="Failure",
            ),
        ]
        if with_cell_counts else [
            Line2D([], [], color=CONFINEMENT_COLOR, lw=1.2, label=r"$\eta_{\mathrm{conf}}$"),
            Line2D([], [], color=ESCAPE_COLOR, lw=1.2, label=r"$\eta_{\mathrm{esc}}$"),
        ]
    )
    axes[n_panels // 2].legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        borderaxespad=0.0,
        fontsize=5,
        handlelength=1.0,
        labelspacing=0.25,
        ncol=2,
        columnspacing=0.8,
    )
    ft.savefig(fig, output)
    plt.close(fig)


def make_plot(
    margins: pd.DataFrame,
    burdens: pd.DataFrame,
    audit: pd.DataFrame,
    output: Path,
    background_margins: pd.DataFrame | None = None,
    background_burdens: pd.DataFrame | None = None,
) -> None:
    fig, ax = ft.new_figure(
        width=30,
        height_mm=40,
        margins=(14, 4, 12, 4),
        apply_style=False,
    )
    if (
        background_margins is not None
        and background_burdens is not None
        and not background_margins.empty
    ):
        for (strategy, plate, sample), margin in background_margins.groupby(["strategy", "plate", "sample"], sort=False):
            color = BACKGROUND_COLORS.get(strategy, get_color(strategy))
            cohort_margin = margin.sort_values("day")
            if not cohort_margin.empty:
                ax.plot(
                    cohort_margin["day"], cohort_margin["normalized_escape_margin"],
                    color=color, lw=0.75, alpha=0.18, zorder=1,
                )
    for strategy, plate, sample in SELECTED_REPLICATES:
        key = (margins["strategy"] == strategy) & (margins["plate"] == plate) & (margins["sample"] == sample)
        margin = margins[key].sort_values("day")
        color = get_color(strategy)
        draw_signed_margin(ax, margin, fill_color=color)

    ax.axhline(0.0, color="#111111", lw=0.65, zorder=5)
    ax.set(
        xlim=X_LIMITS,
        ylim=Y_LIMITS,
        xlabel="Time [d]",
        # Keep this as one plain-text run for Illustrator.  MathText subscripts
        # make Matplotlib emit every character as a separate PDF text object.
        ylabel="Escape margin, Mₑ",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ft.savefig(fig, output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected in-vitro trajectories from all_escape_margin.csv."
    )
    parser.add_argument(
        "--escape-margin-csv", type=Path, default=ESCAPE_MARGIN_CSV
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    margins, burdens, background_margins, background_burdens, audit = load_selected_data(
        args.escape_margin_csv
    )
    exported = margins.merge(
        burdens[["strategy", "plate", "sample", "day", "normalized_resistant"]],
        on=["strategy", "plate", "sample", "day"], how="outer",
    ).sort_values(["strategy", "plate", "sample", "day"])
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exported.to_csv(OUTPUT_CSV, index=False)
    audit.to_csv(OUTPUT_AUDIT_CSV, index=False)
    make_separate_margin_plot(margins, burdens, OUTPUT_SEPARATE_MARGINS_PDF)
    make_separate_margin_plot(
        margins,
        burdens,
        OUTPUT_SEPARATE_MARGINS_WITH_CELLS_PDF,
        with_cell_counts=True,
    )
    print(f"[saved] {OUTPUT_SEPARATE_MARGINS_PDF}")
    print(f"[saved] {OUTPUT_SEPARATE_MARGINS_WITH_CELLS_PDF}")
    print(f"[saved] {OUTPUT_CSV}")
    print(f"[saved] {OUTPUT_AUDIT_CSV}")


if __name__ == "__main__":
    main()
