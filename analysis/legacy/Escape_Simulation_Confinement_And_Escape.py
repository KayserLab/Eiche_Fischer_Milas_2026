"""
Quantify confinement thickness and escaped resistant cells for simulations.

The input is a simulation density run containing
``Plate_*/hdf5/*_simulation.h5`` point clouds and matching density-map TIFFs.
All resistant cells in a simulation are treated as one clone, matching the
simulation path in ``figures/Figure_1/g_plot_selected_well_escape_movies.py``.

For each replicate and day, this script computes:

- confinement thickness with the same 360 external reference points used by
  ``analysis/figure_2/Escape_ConfinementThickness.py``;
- escaped resistant cells with the canonical ``all_res_within_sens`` rule;
- the first day with at least one escaped resistant cell.

Per-replicate CSVs are checkpoints. Aggregate outputs are rebuilt on every run,
which makes ``--resume`` safe after an interrupted batch.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import tifffile


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.legacy.Escape_ConfinementThickness import confinement_for_clone, make_probe_angles
from analysis.legacy.Escape_Front_Write_Tiffs import escaped_front_points_all_res_within_sens
from shared.strategy_colors import normalize_strategy
from shared.data_paths import intermediate_dir


DEFAULT_SIMULATION_ROOT = intermediate_dir("simulation", "basic", "density_maps", create=False)
DEFAULT_OUTPUT_SUBDIR = Path("Escape_Quantification")
PER_REPLICATE_SUBDIR = Path("Simulation_Per_Replicate")
CONFINEMENT_FILENAME = "Escape_ConfinementThickness_all_simulations.csv"
ESCAPE_COUNTS_FILENAME = "escape_cell_counts_all_simulations.csv"
ESCAPE_SUMMARY_FILENAME = "escape_summary_all_simulations.csv"
FAILURES_FILENAME = "simulation_quantification_failures.csv"

SIMULATION_H5_RE = re.compile(r"^(?P<sample>.+)_simulation\.h5$")
COARSENING_RE = re.compile(r"(?:^|;\s*)coarsening_factor=(?P<factor>\d+)(?:;|$)")
ESCAPE_DEFINITION = "all_res_within_sens"
# Version 2: confinement thickness now uses exact tangent half-planes rather
# than the legacy external-circle approximation.  This invalidates v1 resume
# checkpoints so a normal run recomputes them automatically.
QUANTIFICATION_VERSION = 2


@dataclass(frozen=True)
class SimulationInput:
    plate: str
    sample: str
    strategy: str
    h5_path: Path
    density_maps_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute simulation confinement thickness and escaped-cell counts "
            "from Plate_*/hdf5/*_simulation.h5 point clouds."
        )
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=DEFAULT_SIMULATION_ROOT,
        help=f"Simulation density run. Default: {DEFAULT_SIMULATION_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <simulation-root>/Escape_Quantification.",
    )
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        default=None,
        help=(
            "Optional SAMPLE or PLATE:SAMPLE selector. Repeat to select multiple runs. "
            "By default every simulation HDF5 file is processed."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional deterministic debug limit after discovery. Zero means all files.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed per-replicate CSV checkpoints. Enabled by default.",
    )
    return parser.parse_args()


def parse_selectors(specs: Iterable[str] | None) -> set[tuple[str | None, str]]:
    selectors: set[tuple[str | None, str]] = set()
    for spec in specs or ():
        parts = [part.strip() for part in str(spec).split(":", maxsplit=1)]
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


def discover_simulations(
    root: Path,
    selectors: Iterable[str] | None,
    max_files: int,
) -> list[SimulationInput]:
    if not root.is_dir():
        raise FileNotFoundError(f"Simulation root does not exist: {root}")

    requested = parse_selectors(selectors)
    matched: set[tuple[str | None, str]] = set()
    discovered: list[SimulationInput] = []
    for plate_dir in sorted(root.glob("Plate_*")):
        hdf5_dir = plate_dir / "hdf5"
        if not hdf5_dir.is_dir():
            continue
        raw_strategy = plate_dir.name.removeprefix("Plate_")
        strategy = normalize_strategy(raw_strategy)
        for h5_path in sorted(hdf5_dir.glob("*_simulation.h5")):
            match = SIMULATION_H5_RE.match(h5_path.name)
            if match is None:
                continue
            sample = match.group("sample")
            sample_matches = {
                selector
                for selector in requested
                if selector[1] == sample
                and (selector[0] is None or selector[0] == plate_dir.name)
            }
            if requested and not sample_matches:
                continue
            matched.update(sample_matches)
            discovered.append(
                SimulationInput(
                    plate=plate_dir.name,
                    sample=sample,
                    strategy=strategy,
                    h5_path=h5_path,
                    density_maps_path=plate_dir / sample / "density_maps",
                )
            )

    unmatched = requested.difference(matched)
    if unmatched:
        formatted = ", ".join(
            f"{plate}:{sample}" if plate else sample
            for plate, sample in sorted(unmatched, key=str)
        )
        raise FileNotFoundError(f"Simulation selectors were not found: {formatted}")
    if max_files > 0:
        discovered = discovered[:max_files]
    if not discovered:
        raise FileNotFoundError(f"No simulation HDF5 files found below {root}")
    return discovered


def timepoint_sort_key(timepoint: str) -> tuple[int, int | str]:
    return (0, int(timepoint)) if timepoint.isdigit() else (1, timepoint)


def list_common_timepoints(handle: h5py.File) -> list[str]:
    if "sus" not in handle or "res" not in handle:
        return []
    common = set(handle["sus"]).intersection(handle["res"])
    timepoints = [key.removeprefix("d_") for key in common if key.startswith("d_")]
    return sorted(timepoints, key=timepoint_sort_key)


def read_pointcloud_positions(
    handle: h5py.File,
    timepoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    def read_channel(channel: str) -> np.ndarray:
        dataset_path = f"{channel}/d_{timepoint}/block0_values"
        if dataset_path not in handle:
            return np.empty((0, 2), dtype=float)
        values = np.asarray(handle[dataset_path])
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                f"Unexpected simulation point-cloud shape at {dataset_path}: {values.shape}"
            )
        return np.column_stack((values[:, 1], values[:, 0])).astype(float, copy=False)

    return read_channel("res"), read_channel("sus")


def resolve_simulation_canvas(inputs: SimulationInput) -> tuple[int, int]:
    candidates = sorted(
        inputs.density_maps_path.glob(
            f"{inputs.sample}_*_total_density_epanechnikov.tif"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No total-density TIFF found for canvas dimensions: {inputs.density_maps_path}"
        )
    with tifffile.TiffFile(candidates[0]) as density_tif:
        coarse_height, coarse_width = density_tif.pages[0].shape
        description = density_tif.pages[0].description or ""
    match = COARSENING_RE.search(description)
    if match is None:
        raise ValueError(f"Missing coarsening_factor metadata in {candidates[0]}")
    factor = int(match.group("factor"))
    return int(coarse_height * factor), int(coarse_width * factor)


def escaped_membership_count(resistant_xy: np.ndarray, escaped_xy: np.ndarray) -> int:
    if resistant_xy.size == 0 or escaped_xy.size == 0:
        return 0
    escaped_coordinates = {(float(x), float(y)) for x, y in escaped_xy}
    return sum(
        (float(x), float(y)) in escaped_coordinates
        for x, y in resistant_xy
    )


def quantify_simulation(inputs: SimulationInput) -> pd.DataFrame:
    height, width = resolve_simulation_canvas(inputs)
    center_x = float(width) / 2.0
    center_y = float(height) / 2.0
    angles_deg = make_probe_angles()

    rows: list[dict[str, object]] = []
    with h5py.File(inputs.h5_path, mode="r") as pointcloud:
        timepoints = list_common_timepoints(pointcloud)
        if not timepoints:
            raise ValueError(f"No common resistant/sensitive timepoints in {inputs.h5_path}")

        for frame, timepoint in enumerate(timepoints):
            day = int(timepoint) if timepoint.isdigit() else frame + 1
            resistant_xy, sensitive_xy = read_pointcloud_positions(pointcloud, timepoint)

            confinement: dict[str, float | int] | None = None
            if resistant_xy.shape[0] > 0:
                confinement = confinement_for_clone(
                    resistant_xy,
                    sensitive_xy,
                    angles_deg,
                )

            escaped_xy = escaped_front_points_all_res_within_sens(
                clone_points=resistant_xy,
                sus_points=sensitive_xy,
                h=height,
                w=width,
            )
            escaped_count = escaped_membership_count(resistant_xy, escaped_xy)
            rows.append(
                {
                    "quantification_version": QUANTIFICATION_VERSION,
                    "dataset_type": "simulation",
                    "plate": inputs.plate,
                    "sample": inputs.sample,
                    "replicate": inputs.sample,
                    "strategy": inputs.strategy,
                    "treatment": inputs.strategy,
                    "frame": frame,
                    "frame_idx": frame,
                    "timepoint": timepoint,
                    "day": day,
                    "dominant_clone_id": "all_resistant",
                    "confinement_thickness": (
                        np.nan if confinement is None else confinement["confinement_thickness"]
                    ),
                    "min_angle_deg": (
                        np.nan if confinement is None else confinement["min_angle_deg"]
                    ),
                    "distance_px_at_min": (
                        np.nan if confinement is None else confinement["distance_px_at_min"]
                    ),
                    "n_resistant_cells_in_clone": int(resistant_xy.shape[0]),
                    "n_sensitive_cells_total": int(sensitive_xy.shape[0]),
                    "reference_center_x": center_x,
                    "reference_center_y": center_y,
                    "has_dominant_clone_cells": bool(resistant_xy.shape[0] > 0),
                    "escape_definition": ESCAPE_DEFINITION,
                    "n_sensitive_clean": int(sensitive_xy.shape[0]),
                    "n_resistant_total": int(resistant_xy.shape[0]),
                    "n_resistant_escaped": int(escaped_count),
                    "n_resistant_not_escaped": int(resistant_xy.shape[0] - escaped_count),
                    "h5_path": str(inputs.h5_path),
                    "clone_mask_path": "",
                }
            )
    return pd.DataFrame(rows)


def checkpoint_path(output_dir: Path, inputs: SimulationInput) -> Path:
    return output_dir / PER_REPLICATE_SUBDIR / inputs.plate / f"{inputs.sample}.csv"


def checkpoint_is_complete(path: Path, inputs: SimulationInput) -> bool:
    if not path.is_file():
        return False
    try:
        table = pd.read_csv(path)
        with h5py.File(inputs.h5_path, mode="r") as pointcloud:
            expected_timepoints = list_common_timepoints(pointcloud)
    except Exception:
        return False
    required = {
        "quantification_version",
        "plate",
        "sample",
        "strategy",
        "timepoint",
        "day",
        "confinement_thickness",
        "n_resistant_escaped",
    }
    if table.empty or not required.issubset(table.columns):
        return False
    versions = pd.to_numeric(table["quantification_version"], errors="coerce")
    observed_timepoints = table["timepoint"].astype(str).str.zfill(3)
    return (
        versions.eq(QUANTIFICATION_VERSION).all()
        and len(table) == len(expected_timepoints)
        and observed_timepoints.nunique() == len(expected_timepoints)
        and set(observed_timepoints) == set(expected_timepoints)
    )


def build_escape_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (plate, sample), group in table.groupby(["plate", "sample"], sort=True):
        group = group.sort_values("day")
        escaped = group[pd.to_numeric(group["n_resistant_escaped"], errors="coerce") > 0]
        escape_detected = not escaped.empty
        rows.append(
            {
                "quantification_version": QUANTIFICATION_VERSION,
                "dataset_type": "simulation",
                "plate": plate,
                "sample": sample,
                "strategy": str(group["strategy"].iloc[0]),
                "status": "ok",
                "n_timepoints": int(len(group)),
                "first_escape_day": (
                    int(escaped["day"].iloc[0]) if escape_detected else np.nan
                ),
                "escape_day": int(escaped["day"].iloc[0]) if escape_detected else np.nan,
                "escape_detected": bool(escape_detected),
                "h5_path": str(group["h5_path"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_aggregate_outputs(table: pd.DataFrame, output_dir: Path) -> None:
    confinement_columns = [
        "quantification_version",
        "plate",
        "replicate",
        "day",
        "frame_idx",
        "dominant_clone_id",
        "confinement_thickness",
        "min_angle_deg",
        "distance_px_at_min",
        "n_resistant_cells_in_clone",
        "n_sensitive_cells_total",
        "reference_center_x",
        "reference_center_y",
        "has_dominant_clone_cells",
        "h5_path",
        "clone_mask_path",
    ]
    escape_count_columns = [
        "quantification_version",
        "dataset_type",
        "plate",
        "sample",
        "treatment",
        "strategy",
        "frame",
        "timepoint",
        "day",
        "escape_definition",
        "dominant_clone_id",
        "n_sensitive_clean",
        "n_resistant_total",
        "n_resistant_escaped",
        "n_resistant_not_escaped",
    ]
    table[confinement_columns].to_csv(
        output_dir / CONFINEMENT_FILENAME,
        index=False,
    )
    table[escape_count_columns].to_csv(
        output_dir / ESCAPE_COUNTS_FILENAME,
        index=False,
    )
    build_escape_summary(table).to_csv(
        output_dir / ESCAPE_SUMMARY_FILENAME,
        index=False,
    )


def main() -> None:
    args = parse_args()
    if args.max_files < 0:
        raise ValueError("--max-files must be zero or positive")

    output_dir = args.output_dir or (args.simulation_root / DEFAULT_OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    simulations = discover_simulations(
        args.simulation_root,
        selectors=args.samples,
        max_files=args.max_files,
    )
    print(f"[info] discovered simulation replicates: {len(simulations)}")

    tables: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, inputs in enumerate(simulations, start=1):
        per_replicate_path = checkpoint_path(output_dir, inputs)
        try:
            if args.resume and checkpoint_is_complete(per_replicate_path, inputs):
                table = pd.read_csv(per_replicate_path)
                print(
                    f"[resume {index}/{len(simulations)}] "
                    f"{inputs.plate}/{inputs.sample}: rows={len(table)}"
                )
            else:
                table = quantify_simulation(inputs)
                per_replicate_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = per_replicate_path.with_suffix(".csv.tmp")
                table.to_csv(temporary_path, index=False)
                temporary_path.replace(per_replicate_path)
                print(
                    f"[ok {index}/{len(simulations)}] "
                    f"{inputs.plate}/{inputs.sample}: rows={len(table)}"
                )
            tables.append(table)
        except Exception as exc:
            print(f"[error] {inputs.plate}/{inputs.sample}: {exc}")
            failures.append(
                {
                    "plate": inputs.plate,
                    "sample": inputs.sample,
                    "strategy": inputs.strategy,
                    "h5_path": str(inputs.h5_path),
                    "error": str(exc),
                }
            )

    pd.DataFrame(
        failures,
        columns=["plate", "sample", "strategy", "h5_path", "error"],
    ).to_csv(output_dir / FAILURES_FILENAME, index=False)
    if not tables:
        raise RuntimeError("No simulation replicate was quantified successfully")

    aggregate = pd.concat(tables, ignore_index=True).sort_values(
        ["strategy", "plate", "sample", "day"]
    )
    write_aggregate_outputs(aggregate, output_dir)
    print(f"[saved] {output_dir / CONFINEMENT_FILENAME}")
    print(f"[saved] {output_dir / ESCAPE_COUNTS_FILENAME}")
    print(f"[saved] {output_dir / ESCAPE_SUMMARY_FILENAME}")
    print(f"[saved] {output_dir / FAILURES_FILENAME}")


if __name__ == "__main__":
    main()
