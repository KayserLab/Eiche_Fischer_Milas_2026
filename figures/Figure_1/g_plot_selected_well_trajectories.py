"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script plots selected replicate trajectories directly from raw result
files, together with treatment thresholds and truncation-aware stopping points.

Detailed
--------
Inputs:
- `Results/*.txt` replicate trajectories.
- `treatment_schedule.txt` files for the relevant plates.
- Optional sample-selection arguments and exclusion overrides.

Outputs:
- One PDF-style trajectory plot per requested sample selection.
- Additionally, one stacked multi-panel PDF containing all selected wells.

How it works:
- Resolves user-specified sample requests to concrete result files.
- Reads the raw time series, treatment schedule, and any applicable truncation
  day from `shared/experiment_filters.py`.
- Trims each series to the allowed observation window and overlays treatment
  threshold and reference lines on the trajectory.
- Saves both individual panels and a stacked panel figure.

Pipeline role:
- Figure 1 support script for showing representative replicate trajectories.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Plate_*/Results/*.txt, treatment_schedule.txt, and selected raw
# simulation HDF5 snapshots below SIMULATION_DATABASE.
# Scripts: shared/experiment_filters.py, shared/strategy_colors.py, and shared/figtools.py.
# Outputs: individual and stacked selected-well trajectory PDFs, plus derived
# simulation trajectory CSVs below figure_data/Figure_1/selected_well_trajectories.

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import h5py
import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color, normalize_strategy
from shared.data_paths import EXP_RAW_DATA, SIMULATION_DATABASE, figure_data_dir


RESULTS_SUBDIR = "Results"
RESULTS_RE = re.compile(
    r"^Results_G418_a2a4_(?P<well>[A-H]\d{1,2})_(?P<treatment>[^-]+)-(?P=well)\.txt$"
)
DEFAULT_BASE_PATH = EXP_RAW_DATA
DEFAULT_OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "Selected_Well_Trajectories"
DEFAULT_SIMULATION_CSV_DIR = figure_data_dir("Figure_1", "selected_well_trajectories", create=False)
DEFAULT_SIMULATION_RAW_ROOT = SIMULATION_DATABASE
SIMULATION_SNAPSHOT_RE = re.compile(r"^simulation_(?P<timestep>\d+)\.h5$")
SIMULATION_DUMP_INTERVAL = 100_000

# Each displayed in-silico trajectory is tied to one raw simulation run.  Keep
# this mapping here, beside the Figure 1 selection, rather than relying on
# manually copied CSV files.  The raw snapshots start at timestep 0, which is
# displayed as day 1 to match the experimental trajectories.
SIMULATION_SOURCES = {
    ("Plate_2", "A1", "CT"): ("data-DB_CT", "run_30"),
    ("Plate_4", "D3", "AT50"): ("data-DB_AT50", "run_16"),
    ("Plate_5", "A2", "AT100"): ("data-DB_AT100", "run_20"),
}

# Define the wells to plot here.
# Accepted formats:
#   "WELL:TREATMENT"
#   "PLATE:WELL:TREATMENT"
SELECTED_SAMPLES = [
    "A1:CT",
    "D3:AT50",
    "A2:AT100",
    # "C5:AT100",
    # "Plate_4:D2:AT50",
]

# Set to a number to force a fixed y-maximum, or leave as None for auto-scaling.
MAX_Y = 2

TOTAL_COLOR = "black"
SENSITIVE_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
TREATMENT_COLOR = "#bfbfbf"
DOUBLE_TREATMENT_COLOR = "#A8DADC"
STOP_COLOR = "red"
TREATMENT_FACTOR = 1.001
SIMULATION_TREATMENT_HATCH = "////////"
SERIES_LINEWIDTH = 1.3
TOTAL_LINEWIDTH = 1.6
EXPERIMENTAL_MARKER_SIZE = 3.4
EXPERIMENTAL_LINESTYLE = "-"
SIMULATION_ALPHA = 0.95
SIMULATION_LINESTYLE = "--"
TREATMENT_DASHED_THRESHOLDS = {
    "CT": 1.0,
    "AT50": 1.0,
    "AT100": 1.0,
}
TREATMENT_DOTTED_THRESHOLDS = {
    "AT50": 0.5,
    "AT100": 1.0,
}
TREATMENT_HIGHLIGHT_TICKS = {
    "AT50": 1.0,
    "AT100": 1.0,
}
# Treatment-on thresholds are dashed and treatment-off thresholds are dotted.
THRESHOLD_LINEWIDTH = 1.5
REFERENCE_LINE_VALUE = 0.5
REFERENCE_LINEWIDTH = 0.75
DAY_REFERENCE_DAYS = (1, 10, 19, 33)
DAY_REFERENCE_LINEWIDTH = 0.5
DAY_REFERENCE_COLOR = "gray"
THRESHOLD_DASH_POINTS = 4.0
THRESHOLD_GAP_POINTS = 5.0
THRESHOLD_DOT_POINTS = 0.8
THRESHOLD_DASH_STYLE = (0.0, (THRESHOLD_DASH_POINTS, THRESHOLD_GAP_POINTS))
THRESHOLD_DOT_STYLE = (
    -(THRESHOLD_DASH_POINTS + (THRESHOLD_GAP_POINTS - THRESHOLD_DOT_POINTS) / 2.0),
    (
        THRESHOLD_DOT_POINTS,
        THRESHOLD_DASH_POINTS + THRESHOLD_GAP_POINTS - THRESHOLD_DOT_POINTS,
    ),
)

SIZE_SCALE = 1.15
HEIGHT_SCALE = 1.20

# Double the trajectory-panel width (relative to the previous 2.4 cm setting).
PLOT_BOX_WIDTH_CM = 4.8 * SIZE_SCALE
PLOT_BOX_HEIGHT_CM = 1.6 * SIZE_SCALE * HEIGHT_SCALE
LEFT_MARGIN_CM = 1.3 * SIZE_SCALE
RIGHT_MARGIN_CM = 0.3 * SIZE_SCALE
BOTTOM_MARGIN_CM = 1.0 * SIZE_SCALE * HEIGHT_SCALE
TOP_MARGIN_CM = 0.6 * SIZE_SCALE * HEIGHT_SCALE
FIGURE_AXES_WIDTH_MM = PLOT_BOX_WIDTH_CM * 10.0
FIGURE_AXES_HEIGHT_MM = PLOT_BOX_HEIGHT_CM * 10.0
FIGURE_MARGINS_MM = (
    LEFT_MARGIN_CM * 10.0,
    RIGHT_MARGIN_CM * 10.0,
    BOTTOM_MARGIN_CM * 10.0,
    TOP_MARGIN_CM * 10.0,
)

# Additional settings for the stacked multi-panel export.
# The single panel axes width/height stay unchanged.
STACKED_PANEL_GAP_MM = 6
STACKED_OUTPUT_NAME = "Selected_Well_Trajectories_Stacked.pdf"

AXIS_FONTSIZE = 7
TITLE_FONTSIZE = 7
TICK_FONTSIZE = 6
TICK_LABEL_PAD = 1
TICK_LENGTH = 2
XLABEL_PAD = 2
YLABEL_PAD = 2
LEGEND_FONTSIZE = 5

ft.apply_nature_style(
    text_pt=AXIS_FONTSIZE,
    tick_pt=TICK_FONTSIZE,
    legend_pt=LEGEND_FONTSIZE,
    overrides={
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
    },
)


@dataclass(frozen=True)
class SampleRequest:
    well: str
    treatment: str
    plate: str | None = None

    @property
    def replicate_key(self) -> str:
        return f"{self.well}_{self.treatment}-{self.well}"


@dataclass(frozen=True)
class ResolvedSample:
    plate: str
    well: str
    treatment: str
    result_path: Path

    @property
    def replicate_key(self) -> str:
        return f"{self.well}_{self.treatment}-{self.well}"


@dataclass(frozen=True)
class TimeSeries:
    days: list[int]
    sensitive: list[int]
    resistant: list[int]

    @property
    def total(self) -> list[int]:
        return [s + r for s, r in zip(self.sensitive, self.resistant)]


@dataclass(frozen=True)
class ProcessedSample:
    resolved: ResolvedSample
    series: TimeSeries
    simulation_series: TimeSeries | None
    simulation_treatment_days: list[int]
    truncation_day: float | None
    failure_day: int | None
    plot_stop_day: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot normalized single-well trajectories for selected (well, treatment) pairs. "
            "Each plot contains total, sensitive, and resistant counts on one axis."
        )
    )
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        default=None,
        help=(
            "Optional override for requested samples in the form WELL:TREATMENT or PLATE:WELL:TREATMENT. "
            "If omitted, the script uses SELECTED_SAMPLES defined at the top of the file."
        ),
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help=f"Dataset root containing Plate_* folders. Default: {DEFAULT_BASE_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for exported plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--simulation-csv-dir",
        type=Path,
        default=DEFAULT_SIMULATION_CSV_DIR,
        help=(
            "Directory in which this script saves its derived per-sample simulation CSVs. "
            f"Default: {DEFAULT_SIMULATION_CSV_DIR}"
        ),
    )
    parser.add_argument(
        "--simulation-raw-root",
        type=Path,
        default=DEFAULT_SIMULATION_RAW_ROOT,
        help=(
            "Raw simulation database containing data-DB_*/outputs/run_* snapshots. "
            f"Default: {DEFAULT_SIMULATION_RAW_ROOT}"
        ),
    )
    parser.add_argument(
        "--normalization-index",
        type=int,
        default=0,
        help="Timepoint index used as denominator. Default: 0 (first available timepoint).",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Ignore shared/experiment_filters.py exclusions and truncations for the requested samples.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def parse_sample_spec(spec: str) -> SampleRequest:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) == 2:
        well, treatment = parts
        plate = None
    elif len(parts) == 3:
        plate, well, treatment = parts
    else:
        raise ValueError(
            f"Invalid sample specification '{spec}'. Use WELL:TREATMENT or PLATE:WELL:TREATMENT."
        )

    well = well.upper()
    if not re.fullmatch(r"[A-H]\d{1,2}", well):
        raise ValueError(f"Invalid well '{well}' in sample specification '{spec}'.")

    if plate is not None and not re.fullmatch(r"Plate_\d+", plate):
        raise ValueError(f"Invalid plate '{plate}' in sample specification '{spec}'.")

    if not treatment:
        raise ValueError(f"Missing treatment in sample specification '{spec}'.")

    return SampleRequest(well=well, treatment=treatment, plate=plate)


def discover_result_files(base_path: Path) -> dict[tuple[str, str, str], Path]:
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    discovered: dict[tuple[str, str, str], Path] = {}
    for plate_dir in sorted(p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("Plate_")):
        results_dir = plate_dir / RESULTS_SUBDIR
        if not results_dir.is_dir():
            continue
        for result_path in sorted(results_dir.glob("Results_G418_a2a4_*_*-*.txt")):
            match = RESULTS_RE.match(result_path.name)
            if not match:
                continue
            key = (plate_dir.name, match.group("well"), match.group("treatment"))
            discovered[key] = result_path
    if not discovered:
        raise FileNotFoundError(f"No matching result files found under: {base_path}")
    return discovered


def resolve_requests(
    requests: Iterable[SampleRequest],
    available_results: dict[tuple[str, str, str], Path],
) -> list[ResolvedSample]:
    resolved: list[ResolvedSample] = []
    for request in requests:
        matches = [
            ResolvedSample(plate=plate, well=well, treatment=treatment, result_path=path)
            for (plate, well, treatment), path in available_results.items()
            if well == request.well
            and treatment == request.treatment
            and (request.plate is None or plate == request.plate)
        ]

        if not matches:
            target = request.replicate_key if request.plate is None else f"{request.plate}:{request.replicate_key}"
            raise FileNotFoundError(f"No result file found for requested sample '{target}'.")
        if len(matches) > 1:
            plates = ", ".join(match.plate for match in matches)
            raise ValueError(
                f"Request '{request.replicate_key}' is ambiguous across plates: {plates}. "
                "Use PLATE:WELL:TREATMENT."
            )
        resolved.append(matches[0])
    return resolved


def parse_treatment_schedule(schedule_path: Path) -> dict[str, list[str]]:
    if not schedule_path.is_file():
        return {}

    schedule: dict[str, list[str]] = {}
    with schedule_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            key = parts[0].strip()
            days = parts[1].split() if len(parts) > 1 and parts[1].strip() else []
            schedule[key] = days
    return schedule


def read_raw_timeseries(result_path: Path) -> TimeSeries:
    days: list[int] = []
    sensitive: list[int] = []
    resistant: list[int] = []
    with result_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 3:
                continue
            match = re.search(r"d(\d+)", parts[0])
            if not match:
                continue
            day = int(match.group(1))
            try:
                sens_value = int(parts[1])
                res_value = int(parts[2])
            except ValueError:
                continue
            days.append(day)
            sensitive.append(sens_value)
            resistant.append(res_value)

    if not days:
        raise ValueError(f"No usable timepoints found in {result_path}.")

    return TimeSeries(days=days, sensitive=sensitive, resistant=resistant)


def _h5_dataset(data_group: h5py.Group, name: str) -> np.ndarray:
    """Read a raw snapshot dataset, accepting both quoted and unquoted keys."""
    for key in (name, f'"{name}"'):
        if key in data_group:
            return np.asarray(data_group[key])
    raise KeyError(f"Raw simulation snapshot is missing data/{name}.")


def extract_simulation_csv(
    resolved: ResolvedSample,
    raw_root: Path,
    simulation_csv_dir: Path,
) -> Path | None:
    """Derive the selected simulation trajectory from raw HDF5 snapshots.

    The output is deliberately a small Figure 1 data product, not an input:
    every normal invocation recreates it from the raw simulation database.
    """
    source = SIMULATION_SOURCES.get((resolved.plate, resolved.well, resolved.treatment))
    if source is None:
        return None

    database, run = source
    snapshot_dir = raw_root / database / "outputs" / run
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(
            f"Selected raw simulation directory for {resolved.replicate_key} does not exist: {snapshot_dir}"
        )

    snapshots: list[tuple[int, Path]] = []
    for path in snapshot_dir.glob("simulation_*.h5"):
        match = SIMULATION_SNAPSHOT_RE.match(path.name)
        if match:
            snapshots.append((int(match.group("timestep")), path))
    snapshots.sort(key=lambda item: item[0])
    if not snapshots:
        raise FileNotFoundError(f"No raw simulation snapshots found in {snapshot_dir}.")

    rows: list[dict[str, int | str]] = []
    for timestep, snapshot_path in snapshots:
        if timestep % SIMULATION_DUMP_INTERVAL != 0:
            continue
        with h5py.File(snapshot_path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"Raw simulation snapshot has no data group: {snapshot_path}")
            data = handle["data"]
            active = _h5_dataset(data, "cell_flags") == 1
            cell_type = _h5_dataset(data, "cell_type")
            if active.shape != cell_type.shape:
                raise ValueError(f"cell_flags and cell_type have incompatible shapes in {snapshot_path}")
            susceptible = int(np.count_nonzero(cell_type[active] == 0))
            resistant = int(np.count_nonzero(cell_type[active] == 1))
            therapy_switch = int(np.asarray(_h5_dataset(data, "therapy_switch")).reshape(-1)[0])

        rows.append(
            {
                "day": timestep // SIMULATION_DUMP_INTERVAL + 1,
                "susceptible": susceptible,
                "resistant": resistant,
                "total": susceptible + resistant,
                "therapy on/off": therapy_switch,
                "source_database": database,
                "source_run": run,
            }
        )

    if not rows:
        raise ValueError(f"No day-aligned snapshots found in {snapshot_dir}.")

    sample_dir = simulation_csv_dir / f"{resolved.treatment}_{resolved.well}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{resolved.treatment}_{resolved.well}_simulation_trajectory.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"[saved] simulation trajectory: {output_path} "
        f"({len(rows)} days from {database}/{run})"
    )
    return output_path


def read_simulation_timeseries(simulation_csv_path: Path) -> tuple[TimeSeries, list[int]]:
    days: list[int] = []
    sensitive: list[int] = []
    resistant: list[int] = []
    treatment_days: list[int] = []

    with simulation_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"day", "susceptible", "resistant"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Simulation CSV '{simulation_csv_path}' is missing one of the required columns: "
                f"{', '.join(sorted(required_columns))}."
            )

        for row in reader:
            try:
                day = int(float(str(row["day"]).strip()))
                sens_value = int(round(float(str(row["susceptible"]).strip())))
                res_value = int(round(float(str(row["resistant"]).strip())))
            except (TypeError, ValueError):
                continue
            days.append(day)
            sensitive.append(sens_value)
            resistant.append(res_value)
            therapy_value = str(row.get("therapy on/off", "")).strip()
            if therapy_value:
                try:
                    if int(round(float(therapy_value))) != 0:
                        treatment_days.append(day)
                except ValueError:
                    continue

    if not days:
        raise ValueError(f"No usable timepoints found in simulation CSV {simulation_csv_path}.")

    return TimeSeries(days=days, sensitive=sensitive, resistant=resistant), treatment_days


def get_truncation_day(replicate_key: str, include_excluded: bool) -> float | None:
    if include_excluded or replicate_key not in EXCLUDE_SAMPLES:
        return None

    cutoff = EXCLUDE_SAMPLES[replicate_key]
    if cutoff is None:
        raise ValueError(
            f"Requested sample '{replicate_key}' is marked as excluded in shared/experiment_filters.py. "
            "Use --include-excluded to override."
        )

    try:
        return float(cutoff)
    except (TypeError, ValueError):
        return None


def compute_failure_day(series: TimeSeries) -> int | None:
    if not series.days:
        return None

    initial_total = series.total[0]
    if initial_total <= 0:
        return None

    for day, resistant_count in zip(series.days, series.resistant):
        if resistant_count > initial_total:
            return day
    return None


def trim_series(series: TimeSeries, stop_day: float) -> TimeSeries:
    keep_idx = [idx for idx, day in enumerate(series.days) if day <= stop_day]
    if not keep_idx:
        raise ValueError(f"No timepoints remain after applying stop day {stop_day:g}.")

    return TimeSeries(
        days=[series.days[idx] for idx in keep_idx],
        sensitive=[series.sensitive[idx] for idx in keep_idx],
        resistant=[series.resistant[idx] for idx in keep_idx],
    )


def prepare_sample(
    resolved: ResolvedSample,
    include_excluded: bool,
    simulation_raw_root: Path | None,
    simulation_csv_dir: Path | None,
) -> ProcessedSample:
    raw_series = read_raw_timeseries(resolved.result_path)
    truncation_day = get_truncation_day(resolved.replicate_key, include_excluded=include_excluded)
    failure_day = compute_failure_day(raw_series)

    candidate_stop_days = [raw_series.days[-1]]
    if truncation_day is not None:
        candidate_stop_days.append(truncation_day)
    if failure_day is not None:
        candidate_stop_days.append(float(failure_day))
    plot_stop_day = min(candidate_stop_days)

    trimmed = trim_series(raw_series, stop_day=plot_stop_day)
    simulation_series = None
    simulation_treatment_days: list[int] = []
    if simulation_raw_root is not None and simulation_csv_dir is not None:
        simulation_csv = extract_simulation_csv(resolved, simulation_raw_root, simulation_csv_dir)
        if simulation_csv is not None:
            raw_simulation_series, raw_simulation_treatment_days = read_simulation_timeseries(simulation_csv)
            simulation_series = trim_series(raw_simulation_series, stop_day=plot_stop_day)
            simulation_treatment_days = raw_simulation_treatment_days
    return ProcessedSample(
        resolved=resolved,
        series=trimmed,
        simulation_series=simulation_series,
        simulation_treatment_days=simulation_treatment_days,
        truncation_day=truncation_day,
        failure_day=failure_day,
        plot_stop_day=plot_stop_day,
    )


def group_consecutive_days(days: Iterable[int]) -> list[tuple[int, int]]:
    sorted_days = sorted(set(days))
    if not sorted_days:
        return []

    groups: list[tuple[int, int]] = []
    start = sorted_days[0]
    end = start
    for day in sorted_days[1:]:
        if day == end + 1:
            end = day
        else:
            groups.append((start, end))
            start = end = day
    groups.append((start, end))
    return groups


def add_treatment_spans(
    ax: plt.Axes,
    treatment_days: Iterable[str],
    max_day: float,
    ymin: float,
    ymax: float,
) -> None:
    regular_days: list[int] = []
    double_days: list[int] = []
    max_day_int = int(math.floor(max_day))

    for day in treatment_days:
        token = str(day).strip()
        if not token:
            continue
        try:
            if token.endswith("s"):
                continue
            elif token.endswith("d"):
                parsed_day = int(float(token[:-1]))
                if parsed_day <= max_day_int:
                    double_days.append(parsed_day)
            else:
                parsed_day = int(float(token))
                if parsed_day <= max_day_int:
                    regular_days.append(parsed_day)
        except ValueError:
            continue

    for start, end in group_consecutive_days(regular_days):
        ax.axvspan(
            start,
            min(end + TREATMENT_FACTOR, max_day),
            ymin=ymin,
            ymax=ymax,
            facecolor=TREATMENT_COLOR,
            lw=0,
            alpha=1.0,
            zorder=0,
        )
    for start, end in group_consecutive_days(double_days):
        ax.axvspan(
            start,
            min(end + TREATMENT_FACTOR, max_day),
            ymin=ymin,
            ymax=ymax,
            facecolor=DOUBLE_TREATMENT_COLOR,
            lw=0,
            alpha=1.0,
            zorder=0,
        )


def add_simulation_treatment_spans(
    ax: plt.Axes,
    treatment_days: Iterable[int],
    max_day: float,
    ymin: float,
    ymax: float,
) -> None:
    valid_days = [day for day in treatment_days if day <= int(math.floor(max_day))]
    for start, end in group_consecutive_days(valid_days):
        ax.axvspan(
            start,
            min(end + TREATMENT_FACTOR, max_day),
            ymin=ymin,
            ymax=ymax,
            facecolor="none",
            edgecolor=TREATMENT_COLOR,
            hatch=SIMULATION_TREATMENT_HATCH,
            lw=0,
            alpha=1.0,
            zorder=0,
        )


def treatment_schedule_tokens_to_days(
    treatment_days: Iterable[str],
    max_day: float,
) -> list[int]:
    parsed_days: list[int] = []
    max_day_int = int(math.floor(max_day))

    for day in treatment_days:
        token = str(day).strip()
        if not token or token.endswith("s"):
            continue
        if token.endswith("d"):
            token = token[:-1]
        try:
            parsed_day = int(float(token))
        except ValueError:
            continue
        if parsed_day <= max_day_int:
            parsed_days.append(parsed_day)

    return parsed_days


def add_treatment_threshold_line(ax: plt.Axes, treatment: str) -> None:
    strategy = normalize_strategy(treatment)
    dashed_threshold = TREATMENT_DASHED_THRESHOLDS.get(strategy)
    dotted_threshold = TREATMENT_DOTTED_THRESHOLDS.get(strategy)
    if dashed_threshold is None and dotted_threshold is None:
        return

    color = get_color(strategy, "grey")
    if dashed_threshold is not None:
        ax.axhline(
            dashed_threshold,
            color=color,
            linestyle=THRESHOLD_DASH_STYLE,
            linewidth=THRESHOLD_LINEWIDTH,
            dash_capstyle="butt",
            zorder=1,
        )
    if dotted_threshold is not None:
        ax.axhline(
            dotted_threshold,
            color=color,
            linestyle=THRESHOLD_DOT_STYLE,
            linewidth=THRESHOLD_LINEWIDTH,
            dash_capstyle="round",
            zorder=2,
        )


def add_half_reference_line(ax: plt.Axes) -> None:
    """Add the shared, neutral 0.5 reference line behind treatment thresholds."""
    ax.axhline(
        REFERENCE_LINE_VALUE,
        color=TREATMENT_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        linestyle="-",
        zorder=0,
    )


def style_treatment_threshold_tick(ax: plt.Axes, treatment: str) -> None:
    strategy = normalize_strategy(treatment)
    tick_value = TREATMENT_HIGHLIGHT_TICKS.get(strategy)
    if tick_value is None:
        return

    ymin, ymax = ax.get_ylim()
    if tick_value < ymin or tick_value > ymax:
        return

    ticks = list(ax.get_yticks())
    if not any(math.isclose(tick, tick_value, rel_tol=0.0, abs_tol=1e-9) for tick in ticks):
        ticks.append(tick_value)
        ticks = sorted(tick for tick in ticks if ymin <= tick <= ymax)
        ax.set_yticks(ticks)

    ax.figure.canvas.draw()
    color = get_color(strategy, "grey")
    for tick in ax.yaxis.get_major_ticks():
        location = tick.get_loc()
        if math.isclose(location, tick_value, rel_tol=0.0, abs_tol=1e-9):
            tick.label1.set_color(color)
            tick.tick1line.set_color(color)
            tick.tick2line.set_color(color)


def save_pdf(fig: plt.Figure, output_base: Path) -> None:
    ft.savefig(fig, output_base.with_suffix(".pdf"))


def get_normalized_series(
    processed: ProcessedSample,
    normalization_index: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    float,
]:
    resolved = processed.resolved
    series = processed.series
    totals = series.total

    if normalization_index < 0 or normalization_index >= len(totals):
        raise IndexError(
            f"Normalization index {normalization_index} is out of range for '{resolved.replicate_key}'. "
            f"Available indices: 0..{len(totals) - 1}."
        )

    denominator = totals[normalization_index]
    if denominator <= 0:
        raise ValueError(
            f"Normalization denominator is non-positive for '{resolved.replicate_key}' at index {normalization_index}."
        )

    total_norm = [value / denominator for value in totals]
    sensitive_norm = [value / denominator for value in series.sensitive]
    resistant_norm = [value / denominator for value in series.resistant]

    simulation_total_norm: list[float] = []
    simulation_sensitive_norm: list[float] = []
    simulation_resistant_norm: list[float] = []

    if processed.simulation_series is not None:
        simulation_total_norm = [value / denominator for value in processed.simulation_series.total]
        simulation_sensitive_norm = [value / denominator for value in processed.simulation_series.sensitive]
        simulation_resistant_norm = [value / denominator for value in processed.simulation_series.resistant]

    max_y = max(
        [1.1]
        + total_norm
        + sensitive_norm
        + resistant_norm
        + simulation_total_norm
        + simulation_sensitive_norm
        + simulation_resistant_norm
    )
    upper_y = float(MAX_Y) if MAX_Y is not None else max(1.25, max_y * 1.05)

    return (
        total_norm,
        sensitive_norm,
        resistant_norm,
        simulation_total_norm,
        simulation_sensitive_norm,
        simulation_resistant_norm,
        upper_y,
    )


def plot_sample_on_axis(
    ax: plt.Axes,
    processed: ProcessedSample,
    treatment_days: Iterable[str],
    normalization_index: int,
    x_axis_max: float,
    *,
    show_title: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    resolved = processed.resolved
    series = processed.series

    (
        total_norm,
        sensitive_norm,
        resistant_norm,
        simulation_total_norm,
        simulation_sensitive_norm,
        simulation_resistant_norm,
        upper_y,
    ) = get_normalized_series(processed, normalization_index)

    ax.plot(
        series.days,
        total_norm,
        color=TOTAL_COLOR,
        linewidth=TOTAL_LINEWIDTH,
        linestyle=EXPERIMENTAL_LINESTYLE,
        label="Total",
    )
    ax.plot(
        series.days,
        sensitive_norm,
        color=SENSITIVE_COLOR,
        linewidth=SERIES_LINEWIDTH,
        linestyle=EXPERIMENTAL_LINESTYLE,
        label="Sensitive",
    )
    ax.plot(
        series.days,
        resistant_norm,
        color=RESISTANT_COLOR,
        linewidth=SERIES_LINEWIDTH,
        linestyle=EXPERIMENTAL_LINESTYLE,
        label="Resistant",
    )

    if processed.simulation_series is not None:
        ax.plot(
            processed.simulation_series.days,
            simulation_total_norm,
            color=TOTAL_COLOR,
            linestyle=SIMULATION_LINESTYLE,
            linewidth=TOTAL_LINEWIDTH,
            alpha=SIMULATION_ALPHA,
            label="Total (sim)",
        )
        ax.plot(
            processed.simulation_series.days,
            simulation_sensitive_norm,
            color=SENSITIVE_COLOR,
            linestyle=SIMULATION_LINESTYLE,
            linewidth=SERIES_LINEWIDTH,
            alpha=SIMULATION_ALPHA,
            label="Sensitive (sim)",
        )
        ax.plot(
            processed.simulation_series.days,
            simulation_resistant_norm,
            color=RESISTANT_COLOR,
            linestyle=SIMULATION_LINESTYLE,
            linewidth=SERIES_LINEWIDTH,
            alpha=SIMULATION_ALPHA,
            label="Resistant (sim)",
        )

    add_half_reference_line(ax)
    for day in DAY_REFERENCE_DAYS:
        ax.axvline(
            day,
            color=DAY_REFERENCE_COLOR,
            linewidth=DAY_REFERENCE_LINEWIDTH,
            linestyle="--",
            zorder=0.5,
        )
    add_treatment_threshold_line(ax, resolved.treatment)

    ax.set_xlim(min(series.days), x_axis_max)
    ax.set_ylim(0, upper_y)

    if show_xlabel:
        ax.set_xlabel("Time [d]", fontsize=AXIS_FONTSIZE, labelpad=XLABEL_PAD)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(
            r"Normalized count, $\eta$",
            fontsize=AXIS_FONTSIZE,
            labelpad=YLABEL_PAD,
        )
    else:
        ax.set_ylabel("")

    if show_title:
        ax.set_title(f"{resolved.plate} | {resolved.replicate_key}", fontsize=TITLE_FONTSIZE)
    else:
        ax.set_title("")

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE, pad=TICK_LABEL_PAD, length=TICK_LENGTH)
    ax.tick_params(axis="x", which="minor", labelsize=0, length=TICK_LENGTH)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, pad=TICK_LABEL_PAD, length=TICK_LENGTH)
    style_treatment_threshold_tick(ax, resolved.treatment)
    ax.grid(False)


def plot_single_sample(
    processed: ProcessedSample,
    treatment_days: Iterable[str],
    output_dir: Path,
    normalization_index: int,
    x_axis_max: float,
    show: bool,
) -> Path:
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        apply_style=False,
    )

    plot_sample_on_axis(
        ax=ax,
        processed=processed,
        treatment_days=treatment_days,
        normalization_index=normalization_index,
        x_axis_max=x_axis_max,
        show_title=False,
        show_xlabel=processed.resolved.treatment.upper() == "AT100",
        show_ylabel=processed.resolved.treatment.upper() == "AT50",
    )
    if processed.resolved.treatment.upper() in {"AT50", "CT"}:
        ax.tick_params(axis="x", labelbottom=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / f"{processed.resolved.plate}__{processed.resolved.replicate_key}_norm"
    save_pdf(fig, output_base)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_base


def plot_stacked_samples(
    processed_samples: list[ProcessedSample],
    treatment_schedule_by_replicate: dict[str, list[str]],
    output_dir: Path,
    normalization_index: int,
    x_axis_max: float,
    show: bool,
) -> Path:
    """
    Create one stacked multi-panel figure.

    The individual panel axes size is unchanged:
    FIGURE_AXES_WIDTH_MM x FIGURE_AXES_HEIGHT_MM.
    """
    if not processed_samples:
        raise ValueError("No processed samples available for stacked plot.")

    nrows = len(processed_samples)

    fig, axes = ft.new_subplots(
        nrows,
        1,
        ax_width_mm=FIGURE_AXES_WIDTH_MM,
        ax_height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        hspace_mm=STACKED_PANEL_GAP_MM,
        dpi=300,
        sharex=True,
        apply_style=False,
    )

    if nrows == 1:
        axes = [axes]

    for idx, (ax, processed) in enumerate(zip(axes, processed_samples)):
        treatment_days = treatment_schedule_by_replicate.get(processed.resolved.replicate_key, [])

        plot_sample_on_axis(
            ax=ax,
            processed=processed,
            treatment_days=treatment_days,
            normalization_index=normalization_index,
            x_axis_max=x_axis_max,
            show_title=False,
            show_xlabel=False,
            show_ylabel=False,
        )

        if idx < nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time [d]", fontsize=AXIS_FONTSIZE, labelpad=XLABEL_PAD)

    # One y-label only, attached to the middle panel.
    # This preserves the original sideward distance via YLABEL_PAD and avoids
    # manually shifting the label up/down.
    middle_ax = axes[nrows // 2]
    middle_ax.set_ylabel(
        r"Normalized count, $\eta$",
        fontsize=AXIS_FONTSIZE,
        labelpad=YLABEL_PAD,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / STACKED_OUTPUT_NAME
    ft.savefig(fig, output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()
    sample_specs = args.samples if args.samples else SELECTED_SAMPLES
    if not sample_specs:
        raise ValueError(
            "No samples defined. Populate SELECTED_SAMPLES at the top of the script "
            "or pass --sample on the command line."
        )

    requests = [parse_sample_spec(spec) for spec in sample_specs]
    base_path = args.base_path
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    simulation_csv_dir = args.simulation_csv_dir
    simulation_raw_root = args.simulation_raw_root

    available_results = discover_result_files(base_path)
    resolved_samples = resolve_requests(requests, available_results)

    processed_samples = [
        prepare_sample(
            resolved,
            include_excluded=args.include_excluded,
            simulation_raw_root=simulation_raw_root,
            simulation_csv_dir=simulation_csv_dir,
        )
        for resolved in resolved_samples
    ]

    x_axis_max = max(sample.series.days[-1] for sample in processed_samples)

    treatment_schedule_by_replicate: dict[str, list[str]] = {}

    for processed in processed_samples:
        plate_dir = base_path / processed.resolved.plate
        schedule = parse_treatment_schedule(plate_dir / "treatment_schedule.txt")
        treatment_days = schedule.get(processed.resolved.replicate_key, [])
        treatment_schedule_by_replicate[processed.resolved.replicate_key] = treatment_days

        output_base = plot_single_sample(
            processed=processed,
            treatment_days=treatment_days,
            output_dir=output_dir,
            normalization_index=args.normalization_index,
            x_axis_max=x_axis_max,
            show=args.show,
        )

        failure_txt = "none" if processed.failure_day is None else f"{processed.failure_day:g}"
        trunc_txt = "none" if processed.truncation_day is None else f"{processed.truncation_day:g}"
        print(
            f"[info] {processed.resolved.replicate_key}: "
            f"failure_day={failure_txt}, truncation_day={trunc_txt}, plot_stop_day={processed.plot_stop_day:g}"
        )
        if processed.simulation_series is None:
            print(
                f"[info] {processed.resolved.replicate_key}: no raw simulation mapping is configured "
                "for this selected sample."
            )
        print(f"[saved] {output_base.with_suffix('.pdf')}")

    stacked_output = plot_stacked_samples(
        processed_samples=processed_samples,
        treatment_schedule_by_replicate=treatment_schedule_by_replicate,
        output_dir=output_dir,
        normalization_index=args.normalization_index,
        x_axis_max=x_axis_max,
        show=args.show,
    )
    print(f"[saved] {stacked_output}")


if __name__ == "__main__":
    main()
