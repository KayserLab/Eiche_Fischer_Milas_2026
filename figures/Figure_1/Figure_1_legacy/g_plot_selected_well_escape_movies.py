"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
Render experimental or simulation HDF5 cell positions with escaped cells highlighted.

The HDF5 point clouds provide sensitive and resistant cell coordinates. Resistant
cells belonging to the dominant clone are classified with the canonical
``all_res_within_sens`` definition from
``analysis/basic/escape/Escape_Front_Write_Tiffs.py``:
for each external reference point, every resistant cell closer than the nearest
cleaned sensitive cell is escaped.

Each output frame contains cleaned sensitive cells in blue, non-escaped
resistant cells in orange, and escaped resistant cells in dark orange. The
script also writes an MP4, per-timepoint counts, and escaped-cell coordinates.
By default it renders the same selected wells as ``g_plot_selected_well_trajectories.py``;
batch flags can render all compatible experimental and simulation HDF5 files.

Simulation point clouds do not contain resistant-clone identities. For those
files, all resistant cells are therefore treated as one resistant clone and all
filtered simulation-sensitive points are used as the competing population.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental *_total_poisson_padded.h5 / clone-mask TIFFs, or
# simulation *_simulation.h5 point clouds; optional treatment schedules.
# Scripts: analysis/basic/escape/Escape_Front_Write_Tiffs.py (escape classification),
# shared/experiment_filters.py, shared/figtools.py, and the selected-trajectory conventions.
# Outputs: per-timepoint PNGs, MP4 movies, and escaped-cell CSV tables.

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import numpy as np
import tifffile


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.legacy.Escape_Front_Write_Tiffs import (  # noqa: E402
    assign_resistant_cells_to_clone_labels,
    build_day_to_frame,
    escaped_front_points_all_res_within_sens,
    filter_points_by_mask,
    find_dominant_clone_label_from_tiff,
)
from shared.experiment_filters import EXCLUDE_SAMPLES  # noqa: E402
from shared.strategy_colors import FIGURE_COLOR_MAP  # noqa: E402


DEFAULT_BASE_PATH = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
DEFAULT_OUTPUT_SUBDIR = Path("Figures") / "Escape_Cell_Visualizations"
DEFAULT_SAMPLES = (
    "Plate_2:A1:CT",
    "Plate_4:D3:AT50",
    "Plate_3:D4:AT100",
)

H5_RE = re.compile(
    r"^.+_(?P<well>[A-H]\d{1,2})_(?P<treatment>[^-]+)-(?P=well)_total_poisson"
    r"(?P<padded>_padded)?\.h5$",
    re.IGNORECASE,
)
SIMULATION_H5_RE = re.compile(r"^(?P<sample>.+)_simulation\.h5$", re.IGNORECASE)
COARSENING_RE = re.compile(r"(?:^|;)\s*coarsening_factor=(?P<factor>\d+)(?:;|$)")

BACKGROUND_COLOR = "#FFFFFF"
SENSITIVE_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
ESCAPED_COLOR = "#B65313"
TEXT_COLOR = "#202020"


@dataclass(frozen=True)
class SampleRequest:
    well: str
    treatment: str
    plate: str | None = None


@dataclass(frozen=True)
class SampleInputs:
    plate: str
    well: str
    treatment: str
    sample: str
    h5_path: Path
    clone_mask_path: Path
    sensitive_mask_path: Path


@dataclass(frozen=True)
class SimulationInputs:
    plate: str
    well: str
    treatment: str
    sample: str
    h5_path: Path
    density_maps_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render HDF5 cell positions with escaped resistant cells in dark orange, "
            "other resistant cells in orange, and sensitive cells in blue."
        )
    )
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        default=None,
        help=(
            "Sample to render as WELL:TREATMENT or PLATE:WELL:TREATMENT. "
            "Repeat for multiple wells. Defaults to the Figure 1 selected wells."
        ),
    )
    parser.add_argument(
        "--all-experimental",
        action="store_true",
        help=(
            "Render every experimental HDF5 well with the clone and cleaned-sensitive "
            "mask stacks required by the escape definition."
        ),
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help=f"Dataset root containing Plate_* directories. Default: {DEFAULT_BASE_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root. Default: <base-path>/Figures/Escape_Cell_Visualizations."
        ),
    )
    parser.add_argument(
        "--simulation-base-path",
        type=Path,
        default=None,
        help=(
            "Simulation density run containing Plate_*/hdf5/*_simulation.h5. "
            "When supplied, simulation outputs are rendered in addition to experimental outputs."
        ),
    )
    parser.add_argument(
        "--simulation-output-dir",
        type=Path,
        default=None,
        help="Simulation output root. Default: <simulation-base-path>/Escape_Cell_Visualizations.",
    )
    parser.add_argument(
        "--simulation-sample",
        dest="simulation_samples",
        action="append",
        default=None,
        help=(
            "Optional simulation selector SAMPLE or PLATE:SAMPLE. Repeat for multiple runs. "
            "If omitted, all simulation HDF5 files are rendered."
        ),
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Skip experimental rendering and process only --simulation-base-path.",
    )
    parser.add_argument(
        "--max-simulation-files",
        type=int,
        default=0,
        help="Optional simulation test limit. Default: 0 (all matching files).",
    )
    parser.add_argument("--fps", type=float, default=3.0, help="Movie frames per second. Default: 3.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution. Default: 160.")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.55,
        help="Scatter marker area in points squared. Default: 0.55.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Ignore exclusions and truncation days from shared/experiment_filters.py.",
    )
    parser.add_argument(
        "--skip-movie",
        action="store_true",
        help="Write PNG frames and counts CSV only; do not call ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace existing frames, movies, and count CSVs from this renderer. "
            "Enabled by default; use --no-overwrite to fail if outputs already exist."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose count CSV, escaped-cell CSV, frames, and requested movie already exist.",
    )
    return parser.parse_args()


def parse_sample_spec(spec: str) -> SampleRequest:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) == 2:
        plate = None
        well, treatment = parts
    elif len(parts) == 3:
        plate, well, treatment = parts
    else:
        raise ValueError(
            f"Invalid sample specification {spec!r}; use WELL:TREATMENT or PLATE:WELL:TREATMENT."
        )

    well = well.upper()
    if not re.fullmatch(r"[A-H]\d{1,2}", well):
        raise ValueError(f"Invalid well {well!r} in sample specification {spec!r}.")
    if plate is not None and not re.fullmatch(r"Plate_\d+", plate):
        raise ValueError(f"Invalid plate {plate!r} in sample specification {spec!r}.")
    if not treatment:
        raise ValueError(f"Missing treatment in sample specification {spec!r}.")
    return SampleRequest(well=well, treatment=treatment, plate=plate)


def discover_hdf5_files(base_path: Path) -> dict[tuple[str, str, str], Path]:
    if not base_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {base_path}")

    discovered: dict[tuple[str, str, str], Path] = {}
    for plate_dir in sorted(base_path.glob("Plate_*")):
        h5_dir = plate_dir / "hdf5"
        if not h5_dir.is_dir():
            continue
        for h5_path in sorted(h5_dir.glob("*.h5")):
            match = H5_RE.match(h5_path.name)
            if match is None:
                continue
            key = (plate_dir.name, match.group("well").upper(), match.group("treatment"))
            previous = discovered.get(key)
            if previous is None or ("_padded.h5" in h5_path.name and "_padded.h5" not in previous.name):
                discovered[key] = h5_path
    if not discovered:
        raise FileNotFoundError(f"No *_total_poisson*.h5 files found below {base_path}")
    return discovered


def resolve_requests(
    requests: Iterable[SampleRequest],
    available: dict[tuple[str, str, str], Path],
    base_path: Path,
) -> list[SampleInputs]:
    resolved: list[SampleInputs] = []
    for request in requests:
        matches = [
            (plate, well, treatment, h5_path)
            for (plate, well, treatment), h5_path in available.items()
            if well == request.well
            and treatment == request.treatment
            and (request.plate is None or request.plate == plate)
        ]
        if not matches:
            raise FileNotFoundError(
                f"No HDF5 point cloud found for {request.plate or '*'}:"
                f"{request.well}:{request.treatment}."
            )
        if len(matches) > 1:
            plates = ", ".join(match[0] for match in matches)
            raise ValueError(
                f"{request.well}:{request.treatment} is present on multiple plates ({plates}); "
                "use PLATE:WELL:TREATMENT."
            )

        plate, well, treatment, h5_path = matches[0]
        sample = f"{well}_{treatment}-{well}"
        clone_mask_path = base_path / plate / "clone_masks_stacks" / f"{sample}_res_clone_masks.tif"
        sensitive_mask_path = (
            base_path / plate / "sensitive_clean_masks_stacks" / f"{sample}_sus_clean_masks.tif"
        )
        for required_path in (clone_mask_path, sensitive_mask_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Required escape-definition input is missing: {required_path}")
        resolved.append(
            SampleInputs(
                plate=plate,
                well=well,
                treatment=treatment,
                sample=sample,
                h5_path=h5_path,
                clone_mask_path=clone_mask_path,
                sensitive_mask_path=sensitive_mask_path,
            )
        )
    return resolved


def resolve_all_experimental(
    available: dict[tuple[str, str, str], Path],
    base_path: Path,
    include_excluded: bool,
) -> list[SampleInputs]:
    resolved: list[SampleInputs] = []
    for plate, well, treatment in sorted(available):
        sample = f"{well}_{treatment}-{well}"
        if not include_excluded and EXCLUDE_SAMPLES.get(sample, False) is None:
            continue
        clone_mask_path = base_path / plate / "clone_masks_stacks" / f"{sample}_res_clone_masks.tif"
        sensitive_mask_path = (
            base_path / plate / "sensitive_clean_masks_stacks" / f"{sample}_sus_clean_masks.tif"
        )
        if not clone_mask_path.is_file() or not sensitive_mask_path.is_file():
            print(f"Skipping {plate}/{sample}: required escape-definition masks are missing")
            continue
        resolved.append(
            SampleInputs(
                plate=plate,
                well=well,
                treatment=treatment,
                sample=sample,
                h5_path=available[(plate, well, treatment)],
                clone_mask_path=clone_mask_path,
                sensitive_mask_path=sensitive_mask_path,
            )
        )
    if not resolved:
        raise FileNotFoundError("No compatible experimental wells were found.")
    return resolved


def experimental_input_failures(
    available: dict[tuple[str, str, str], Path],
    base_path: Path,
    include_excluded: bool,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for plate, well, treatment in sorted(available):
        sample = f"{well}_{treatment}-{well}"
        error = ""
        if not include_excluded and EXCLUDE_SAMPLES.get(sample, False) is None:
            error = "Fully excluded by shared/experiment_filters.py."
        else:
            clone_mask_path = base_path / plate / "clone_masks_stacks" / f"{sample}_res_clone_masks.tif"
            sensitive_mask_path = (
                base_path / plate / "sensitive_clean_masks_stacks" / f"{sample}_sus_clean_masks.tif"
            )
            missing = [
                label
                for label, path in (
                    ("resistant clone mask", clone_mask_path),
                    ("cleaned sensitive mask", sensitive_mask_path),
                )
                if not path.is_file()
            ]
            if missing:
                error = f"Missing escape-definition input: {', '.join(missing)}."
        if error:
            failures.append(
                {
                    "dataset_type": "experimental",
                    "plate": plate,
                    "sample": sample,
                    "error": error,
                }
            )
    return failures


def parse_simulation_selectors(specs: Iterable[str] | None) -> set[tuple[str | None, str]]:
    selectors: set[tuple[str | None, str]] = set()
    for spec in specs or ():
        parts = [part.strip() for part in spec.split(":", maxsplit=1)]
        if len(parts) == 1:
            plate, sample = None, parts[0]
        else:
            plate, sample = parts
        if plate is not None and not plate.startswith("Plate_"):
            raise ValueError(f"Invalid simulation plate in {spec!r}; expected Plate_*.")
        if not sample:
            raise ValueError(f"Missing simulation sample in {spec!r}.")
        selectors.add((plate, sample))
    return selectors


def discover_simulation_inputs(
    base_path: Path,
    selectors: Iterable[str] | None,
    max_files: int,
) -> list[SimulationInputs]:
    if not base_path.is_dir():
        raise FileNotFoundError(f"Simulation run does not exist: {base_path}")
    requested = parse_simulation_selectors(selectors)
    discovered: list[SimulationInputs] = []
    matched: set[tuple[str | None, str]] = set()
    for plate_dir in sorted(base_path.glob("Plate_*")):
        hdf5_dir = plate_dir / "hdf5"
        if not hdf5_dir.is_dir():
            continue
        treatment = plate_dir.name.removeprefix("Plate_")
        for h5_path in sorted(hdf5_dir.glob("*_simulation.h5")):
            match = SIMULATION_H5_RE.match(h5_path.name)
            if match is None:
                continue
            sample = match.group("sample")
            matching_selectors = {
                selector
                for selector in requested
                if selector[1] == sample and (selector[0] is None or selector[0] == plate_dir.name)
            }
            if requested and not matching_selectors:
                continue
            matched.update(matching_selectors)
            prefix = sample.rsplit("-", maxsplit=1)[0]
            treatment_suffix = f"_{treatment}"
            well = prefix[: -len(treatment_suffix)] if prefix.endswith(treatment_suffix) else prefix
            discovered.append(
                SimulationInputs(
                    plate=plate_dir.name,
                    well=well,
                    treatment=treatment,
                    sample=sample,
                    h5_path=h5_path,
                    density_maps_path=base_path / plate_dir.name / sample / "density_maps",
                )
            )
    unmatched = requested - matched
    if unmatched:
        formatted = ", ".join(
            f"{plate}:{sample}" if plate else sample for plate, sample in sorted(unmatched, key=str)
        )
        raise FileNotFoundError(f"Simulation selectors were not found: {formatted}")
    if max_files > 0:
        discovered = discovered[:max_files]
    if not discovered:
        raise FileNotFoundError(f"No simulation HDF5 files found below {base_path}")
    return discovered


def sample_cutoff_day(sample: str, include_excluded: bool) -> float | None:
    if include_excluded or sample not in EXCLUDE_SAMPLES:
        return None
    cutoff = EXCLUDE_SAMPLES[sample]
    if cutoff is None:
        raise ValueError(
            f"Sample {sample} is excluded by shared/experiment_filters.py; use --include-excluded to override."
        )
    return float(cutoff)


def timepoint_sort_key(timepoint: str) -> tuple[int, int | str]:
    return (0, int(timepoint)) if timepoint.isdigit() else (1, timepoint)


def list_common_timepoints(handle: h5py.File) -> list[str]:
    if "sus" not in handle or "res" not in handle:
        return []
    common = set(handle["sus"]).intersection(handle["res"])
    timepoints = [key.removeprefix("d_") for key in common if key.startswith("d_")]
    return sorted(timepoints, key=timepoint_sort_key)


def read_pointcloud_positions(handle: h5py.File, timepoint: str) -> tuple[np.ndarray, np.ndarray]:
    def read_channel(channel: str) -> np.ndarray:
        path = f"{channel}/d_{timepoint}/block0_values"
        if path not in handle:
            return np.empty((0, 2), dtype=float)
        values = np.asarray(handle[path])
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"Unexpected HDF5 point-cloud shape at {path}: {values.shape}")
        return np.column_stack((values[:, 1], values[:, 0])).astype(float, copy=False)

    return read_channel("res"), read_channel("sus")


def resolve_simulation_canvas(inputs: SimulationInputs) -> tuple[int, int]:
    candidates = sorted(inputs.density_maps_path.glob(f"{inputs.sample}_*_total_density_epanechnikov.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"No simulation density TIFF found for canvas dimensions: {inputs.density_maps_path}"
        )
    with tifffile.TiffFile(candidates[0]) as density_tif:
        coarse_height, coarse_width = density_tif.pages[0].shape
        description = density_tif.pages[0].description or ""
    match = COARSENING_RE.search(description)
    if match is None:
        raise ValueError(f"Missing coarsening_factor metadata in {candidates[0]}")
    factor = int(match.group("factor"))
    return int(coarse_height * factor), int(coarse_width * factor)


def escaped_membership(res_xy: np.ndarray, escaped_xy: np.ndarray) -> np.ndarray:
    if res_xy.size == 0 or escaped_xy.size == 0:
        return np.zeros(len(res_xy), dtype=bool)
    escaped_coordinates = {(float(x), float(y)) for x, y in escaped_xy}
    return np.fromiter(
        ((float(x), float(y)) in escaped_coordinates for x, y in res_xy),
        dtype=bool,
        count=len(res_xy),
    )


ESCAPED_CELL_FIELDS = (
    "dataset_type",
    "plate",
    "sample",
    "treatment",
    "frame",
    "timepoint",
    "day",
    "resistant_index",
    "x",
    "y",
    "escape_definition",
    "dominant_clone_label",
)


def write_escaped_cell_rows(
    writer: csv.DictWriter,
    *,
    inputs: SampleInputs | SimulationInputs,
    dataset_type: str,
    frame: int,
    timepoint: str,
    day: int,
    resistant_xy: np.ndarray,
    escaped_mask: np.ndarray,
    dominant_clone_label: int | str,
) -> None:
    for resistant_index in np.flatnonzero(escaped_mask):
        x, y = resistant_xy[resistant_index]
        writer.writerow(
            {
                "dataset_type": dataset_type,
                "plate": inputs.plate,
                "sample": inputs.sample,
                "treatment": inputs.treatment,
                "frame": frame,
                "timepoint": timepoint,
                "day": day,
                "resistant_index": int(resistant_index),
                "x": float(x),
                "y": float(y),
                "escape_definition": "all_res_within_sens",
                "dominant_clone_label": dominant_clone_label,
            }
        )


def configure_axes(ax: plt.Axes, width: int, height: int) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def add_frame_text(
    ax: plt.Axes,
    inputs: SampleInputs | SimulationInputs,
    day: int,
    n_sensitive: int,
    n_resistant: int,
    n_escaped: int,
) -> None:
    ax.text(
        0.02,
        0.975,
        f"{inputs.plate}  {inputs.well}  {inputs.treatment}  |  Day {day}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.94,
        f"Sensitive: {n_sensitive:,}    Resistant: {n_resistant:,}    Escaped: {n_escaped:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=8,
    )


def add_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=SENSITIVE_COLOR, label="Sensitive"),
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=RESISTANT_COLOR, label="Resistant, not escaped"),
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=ESCAPED_COLOR, label="Resistant, escaped"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.015),
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor="#B5B5B5",
        framealpha=0.9,
        fontsize=7,
        handletextpad=0.4,
        borderpad=0.5,
    )


def render_frame(
    *,
    inputs: SampleInputs | SimulationInputs,
    day: int,
    frame_path: Path,
    width: int,
    height: int,
    sensitive_xy: np.ndarray,
    resistant_xy: np.ndarray,
    escaped_mask: np.ndarray,
    marker_size: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor=BACKGROUND_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BACKGROUND_COLOR)

    ax.scatter(
        sensitive_xy[:, 0],
        sensitive_xy[:, 1],
        s=marker_size,
        c=SENSITIVE_COLOR,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    not_escaped_mask = ~escaped_mask
    ax.scatter(
        resistant_xy[not_escaped_mask, 0],
        resistant_xy[not_escaped_mask, 1],
        s=marker_size,
        c=RESISTANT_COLOR,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        resistant_xy[escaped_mask, 0],
        resistant_xy[escaped_mask, 1],
        s=marker_size * 1.35,
        c=ESCAPED_COLOR,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    configure_axes(ax, width=width, height=height)
    add_frame_text(
        ax,
        inputs=inputs,
        day=day,
        n_sensitive=len(sensitive_xy),
        n_resistant=len(resistant_xy),
        n_escaped=int(np.count_nonzero(escaped_mask)),
    )
    add_legend(ax)
    fig.savefig(frame_path, facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_COLOR)
    plt.close(fig)


def write_movie(frame_dir: Path, movie_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP4 output but was not found on PATH.")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie_path),
        ],
        check=True,
    )


def prepare_output_dir(sample_output_dir: Path, overwrite: bool) -> Path:
    if sample_output_dir.exists():
        existing_outputs = list(sample_output_dir.glob("frame_*.png"))
        existing_outputs.extend(sample_output_dir.glob("*.mp4"))
        existing_outputs.extend(sample_output_dir.glob("*.csv"))
        if existing_outputs and not overwrite:
            raise FileExistsError(
                f"Output already exists in {sample_output_dir}; rerun without "
                "--no-overwrite to replace it."
            )
        if overwrite:
            if existing_outputs:
                print(f"Replacing {len(existing_outputs)} existing output files in {sample_output_dir}")
            for output_path in existing_outputs:
                output_path.unlink()
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    return sample_output_dir


def sample_outputs_complete(
    inputs: SampleInputs | SimulationInputs,
    output_root: Path,
    skip_movie: bool,
) -> bool:
    sample_output_dir = output_root / inputs.plate / inputs.sample
    required = [
        sample_output_dir / f"{inputs.sample}_escape_cell_counts.csv",
        sample_output_dir / f"{inputs.sample}_escaped_cells.csv",
    ]
    if not skip_movie:
        required.append(sample_output_dir / f"{inputs.sample}_escape_cells.mp4")
    return all(path.is_file() for path in required) and any(sample_output_dir.glob("frame_*.png"))


def remove_partial_sample_outputs(
    inputs: SampleInputs | SimulationInputs,
    output_root: Path,
) -> None:
    sample_output_dir = output_root / inputs.plate / inputs.sample
    if not sample_output_dir.exists():
        return
    partial_outputs = list(sample_output_dir.glob("frame_*.png"))
    partial_outputs.extend(sample_output_dir.glob("*.mp4"))
    partial_outputs.extend(sample_output_dir.glob("*.csv"))
    for output_path in partial_outputs:
        output_path.unlink()


def write_batch_failures(output_root: Path, filename: str, failures: list[dict[str, str]]) -> None:
    failure_path = output_root / filename
    if not failures:
        if failure_path.exists():
            failure_path.unlink()
        return
    output_root.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset_type", "plate", "sample", "error"))
        writer.writeheader()
        writer.writerows(failures)
    print(f"Recorded {len(failures)} skipped samples in {failure_path}")


def render_sample(inputs: SampleInputs, args: argparse.Namespace, output_root: Path) -> None:
    cutoff_day = sample_cutoff_day(inputs.sample, include_excluded=args.include_excluded)
    dominant = find_dominant_clone_label_from_tiff(str(inputs.clone_mask_path))
    if dominant is None:
        raise ValueError(f"No dominant resistant clone found for {inputs.plate}/{inputs.sample}.")

    sample_output_dir = prepare_output_dir(
        output_root / inputs.plate / inputs.sample,
        overwrite=args.overwrite,
    )
    counts_path = sample_output_dir / f"{inputs.sample}_escape_cell_counts.csv"
    escaped_cells_path = sample_output_dir / f"{inputs.sample}_escaped_cells.csv"
    movie_path = sample_output_dir / f"{inputs.sample}_escape_cells.mp4"

    rows: list[dict[str, int | str]] = []
    with (
        h5py.File(inputs.h5_path, mode="r") as pointcloud,
        tifffile.TiffFile(inputs.clone_mask_path) as clone_tif,
        tifffile.TiffFile(inputs.sensitive_mask_path) as sensitive_tif,
        escaped_cells_path.open("w", encoding="utf-8", newline="") as escaped_handle,
    ):
        escaped_writer = csv.DictWriter(escaped_handle, fieldnames=ESCAPED_CELL_FIELDS)
        escaped_writer.writeheader()
        n_frames = len(clone_tif.pages)
        if len(sensitive_tif.pages) != n_frames:
            raise ValueError(
                f"Mask frame mismatch for {inputs.sample}: clone={n_frames}, "
                f"sensitive={len(sensitive_tif.pages)}."
            )
        day_to_frame = build_day_to_frame(n_frames)
        timepoint_by_day = {
            int(timepoint): timepoint
            for timepoint in list_common_timepoints(pointcloud)
            if timepoint.isdigit()
        }
        days = [
            day
            for day in sorted(day_to_frame)
            if day in timepoint_by_day
        ]
        if cutoff_day is not None:
            days = [day for day in days if day <= cutoff_day]
        if not days:
            raise ValueError(f"No usable HDF5 timepoints remain for {inputs.plate}/{inputs.sample}.")

        first_frame = clone_tif.pages[0].asarray()
        height, width = first_frame.shape
        print(
            f"Rendering {inputs.plate}/{inputs.sample}: {len(days)} frames, "
            f"dominant clone={dominant}, HDF5={inputs.h5_path.name}"
        )

        for frame_number, day in enumerate(days):
            frame_index = day_to_frame[day]
            timepoint = timepoint_by_day[day]
            clone_frame = clone_tif.pages[frame_index].asarray()
            sensitive_mask = sensitive_tif.pages[frame_index].asarray().astype(bool, copy=False)
            resistant_xy, sensitive_xy_raw = read_pointcloud_positions(pointcloud, timepoint)
            sensitive_xy, _ = filter_points_by_mask(sensitive_xy_raw, sensitive_mask)

            labels = assign_resistant_cells_to_clone_labels(resistant_xy, clone_frame)
            dominant_xy = resistant_xy[labels == dominant]
            escaped_xy = escaped_front_points_all_res_within_sens(
                clone_points=dominant_xy,
                sus_points=sensitive_xy,
                h=height,
                w=width,
            )
            escaped_mask = escaped_membership(resistant_xy, escaped_xy)
            write_escaped_cell_rows(
                escaped_writer,
                inputs=inputs,
                dataset_type="experimental",
                frame=frame_index,
                timepoint=timepoint,
                day=day,
                resistant_xy=resistant_xy,
                escaped_mask=escaped_mask,
                dominant_clone_label=dominant,
            )

            frame_path = sample_output_dir / f"frame_{frame_number:03d}.png"
            render_frame(
                inputs=inputs,
                day=day,
                frame_path=frame_path,
                width=width,
                height=height,
                sensitive_xy=sensitive_xy,
                resistant_xy=resistant_xy,
                escaped_mask=escaped_mask,
                marker_size=args.marker_size,
                dpi=args.dpi,
            )
            n_escaped = int(np.count_nonzero(escaped_mask))
            rows.append(
                {
                    "dataset_type": "experimental",
                    "plate": inputs.plate,
                    "sample": inputs.sample,
                    "treatment": inputs.treatment,
                    "frame": frame_index,
                    "timepoint": timepoint,
                    "day": day,
                    "escape_definition": "all_res_within_sens",
                    "dominant_clone_label": dominant,
                    "n_sensitive_clean": len(sensitive_xy),
                    "n_resistant_total": len(resistant_xy),
                    "n_resistant_escaped": n_escaped,
                    "n_resistant_not_escaped": len(resistant_xy) - n_escaped,
                }
            )
            print(
                f"  {frame_number + 1:02d}/{len(days):02d} day {day:02d}: "
                f"sensitive={len(sensitive_xy):,}, resistant={len(resistant_xy):,}, "
                f"escaped={n_escaped:,}"
            )

    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if not args.skip_movie:
        write_movie(sample_output_dir, movie_path, fps=args.fps)
        print(f"  Saved movie: {movie_path}")
    print(f"  Saved {len(rows)} PNG frames, counts, and escaped-cell CSV: {sample_output_dir}")


def render_simulation_sample(
    inputs: SimulationInputs,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    height, width = resolve_simulation_canvas(inputs)
    sample_output_dir = prepare_output_dir(
        output_root / inputs.plate / inputs.sample,
        overwrite=args.overwrite,
    )
    counts_path = sample_output_dir / f"{inputs.sample}_escape_cell_counts.csv"
    escaped_cells_path = sample_output_dir / f"{inputs.sample}_escaped_cells.csv"
    movie_path = sample_output_dir / f"{inputs.sample}_escape_cells.mp4"

    rows: list[dict[str, int | str]] = []
    with (
        h5py.File(inputs.h5_path, mode="r") as pointcloud,
        escaped_cells_path.open("w", encoding="utf-8", newline="") as escaped_handle,
    ):
        escaped_writer = csv.DictWriter(escaped_handle, fieldnames=ESCAPED_CELL_FIELDS)
        escaped_writer.writeheader()
        timepoints = list_common_timepoints(pointcloud)
        if not timepoints:
            raise ValueError(f"No common sensitive/resistant timepoints in {inputs.h5_path}")
        print(
            f"Rendering simulation {inputs.plate}/{inputs.sample}: {len(timepoints)} frames, "
            f"all resistant cells treated as one clone, HDF5={inputs.h5_path.name}"
        )

        for frame_number, timepoint in enumerate(timepoints):
            day = int(timepoint) if timepoint.isdigit() else frame_number + 1
            resistant_xy, sensitive_xy = read_pointcloud_positions(pointcloud, timepoint)
            escaped_xy = escaped_front_points_all_res_within_sens(
                clone_points=resistant_xy,
                sus_points=sensitive_xy,
                h=height,
                w=width,
            )
            escaped_mask = escaped_membership(resistant_xy, escaped_xy)
            write_escaped_cell_rows(
                escaped_writer,
                inputs=inputs,
                dataset_type="simulation",
                frame=frame_number,
                timepoint=timepoint,
                day=day,
                resistant_xy=resistant_xy,
                escaped_mask=escaped_mask,
                dominant_clone_label="all_resistant",
            )

            frame_path = sample_output_dir / f"frame_{frame_number:03d}.png"
            render_frame(
                inputs=inputs,
                day=day,
                frame_path=frame_path,
                width=width,
                height=height,
                sensitive_xy=sensitive_xy,
                resistant_xy=resistant_xy,
                escaped_mask=escaped_mask,
                marker_size=args.marker_size,
                dpi=args.dpi,
            )
            n_escaped = int(np.count_nonzero(escaped_mask))
            rows.append(
                {
                    "dataset_type": "simulation",
                    "plate": inputs.plate,
                    "sample": inputs.sample,
                    "treatment": inputs.treatment,
                    "frame": frame_number,
                    "timepoint": timepoint,
                    "day": day,
                    "escape_definition": "all_res_within_sens",
                    "dominant_clone_label": "all_resistant",
                    "n_sensitive_clean": len(sensitive_xy),
                    "n_resistant_total": len(resistant_xy),
                    "n_resistant_escaped": n_escaped,
                    "n_resistant_not_escaped": len(resistant_xy) - n_escaped,
                }
            )
            print(
                f"  {frame_number + 1:03d}/{len(timepoints):03d} day {day:03d}: "
                f"sensitive={len(sensitive_xy):,}, resistant={len(resistant_xy):,}, "
                f"escaped={n_escaped:,}"
            )

    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if not args.skip_movie:
        write_movie(sample_output_dir, movie_path, fps=args.fps)
        print(f"  Saved movie: {movie_path}")
    print(f"  Saved {len(rows)} PNG frames, counts, and escaped-cell CSV: {sample_output_dir}")


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.marker_size <= 0:
        raise ValueError("--marker-size must be positive.")
    if args.max_simulation_files < 0:
        raise ValueError("--max-simulation-files cannot be negative.")
    if args.all_experimental and args.samples:
        raise ValueError("Use either --all-experimental or --sample, not both.")
    if args.simulation_only and args.simulation_base_path is None:
        raise ValueError("--simulation-only requires --simulation-base-path.")

    if not args.simulation_only:
        available = discover_hdf5_files(args.base_path)
        if args.all_experimental:
            experimental_inputs = resolve_all_experimental(
                available,
                base_path=args.base_path,
                include_excluded=args.include_excluded,
            )
        else:
            requests = [parse_sample_spec(spec) for spec in (args.samples or DEFAULT_SAMPLES)]
            experimental_inputs = resolve_requests(
                requests,
                available=available,
                base_path=args.base_path,
            )
        output_root = args.output_dir or (args.base_path / DEFAULT_OUTPUT_SUBDIR)
        print(f"Processing {len(experimental_inputs)} experimental HDF5 files.")
        experimental_failures = (
            experimental_input_failures(
                available,
                base_path=args.base_path,
                include_excluded=args.include_excluded,
            )
            if args.all_experimental
            else []
        )
        for sample_inputs in experimental_inputs:
            if args.resume and sample_outputs_complete(sample_inputs, output_root, args.skip_movie):
                print(f"Resume: skipping completed experimental sample {sample_inputs.plate}/{sample_inputs.sample}")
                continue
            try:
                render_sample(sample_inputs, args=args, output_root=output_root)
            except Exception as exc:
                if not args.all_experimental:
                    raise
                remove_partial_sample_outputs(sample_inputs, output_root)
                print(f"Skipping experimental {sample_inputs.plate}/{sample_inputs.sample}: {exc}")
                experimental_failures.append(
                    {
                        "dataset_type": "experimental",
                        "plate": sample_inputs.plate,
                        "sample": sample_inputs.sample,
                        "error": str(exc),
                    }
                )
        write_batch_failures(
            output_root,
            "experimental_escape_cell_batch_failures.csv",
            experimental_failures,
        )

    if args.simulation_base_path is not None:
        simulation_inputs = discover_simulation_inputs(
            args.simulation_base_path,
            selectors=args.simulation_samples,
            max_files=args.max_simulation_files,
        )
        simulation_output_root = args.simulation_output_dir or (
            args.simulation_base_path / "Escape_Cell_Visualizations"
        )
        print(f"Processing {len(simulation_inputs)} simulation HDF5 files.")
        simulation_failures: list[dict[str, str]] = []
        simulation_batch_mode = not args.simulation_samples or len(simulation_inputs) > 1
        for simulation_input in simulation_inputs:
            if args.resume and sample_outputs_complete(
                simulation_input,
                simulation_output_root,
                args.skip_movie,
            ):
                print(f"Resume: skipping completed simulation {simulation_input.plate}/{simulation_input.sample}")
                continue
            try:
                render_simulation_sample(
                    simulation_input,
                    args=args,
                    output_root=simulation_output_root,
                )
            except Exception as exc:
                if not simulation_batch_mode:
                    raise
                remove_partial_sample_outputs(simulation_input, simulation_output_root)
                print(f"Skipping simulation {simulation_input.plate}/{simulation_input.sample}: {exc}")
                simulation_failures.append(
                    {
                        "dataset_type": "simulation",
                        "plate": simulation_input.plate,
                        "sample": simulation_input.sample,
                        "error": str(exc),
                    }
                )
        write_batch_failures(
            simulation_output_root,
            "simulation_escape_cell_batch_failures.csv",
            simulation_failures,
        )


if __name__ == "__main__":
    main()
