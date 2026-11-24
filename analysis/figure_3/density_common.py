"""Shared Figure 3 density, input, and escape-time helpers.

The active HDF5 workflow and map plotters use these helpers directly rather
than importing a historical aggregation entry point. Importing is read-only.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as density_base
from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (
    strategy_name_from_well_identifier,
)
from shared.strategy_colors import normalize_strategy
from shared.data_paths import COMBINED_RESULTS_CSV, SIMULATION_DATABASE, intermediate_dir


X_DISPLAY_SCALE = 1000.0
REL_CHANGE_RE = re.compile(
    r"^(?P<well>.+)_(?P<first_tp>[^_]+)_to_(?P<next_tp>[^_]+)_"
    r"res_density_epanechnikov_rel_diff\.tif$"
)
CHANGE_SPECS = {
    "rel": {
        "glob": "*_res_density_epanechnikov_rel_diff.tif",
    },
}

# Raw simulation-source folder -> Figure 3 presentation label.  These legacy
# labels are canonicalized by the active aggregation script before output.
SOURCE_TO_DISPLAY = {
    "CT": "CT",
    "NT": "NT",
    "AT50": "AT50",
    "AT100": "AT100",
    "Agent": "ATagent",
    "Dens": r"AT$^{+}$",
}


def strategy_from_plate_and_well(plate: str, well_identifier: str) -> str:
    strategy = strategy_name_from_well_identifier(well_identifier) or ""
    if not strategy and str(plate).startswith("Plate_"):
        strategy = str(plate).split("Plate_", 1)[1]
    return normalize_strategy(strategy or "unknown")


def timepoint_to_day(timepoint: str) -> float:
    try:
        return float(timepoint)
    except (TypeError, ValueError):
        return float("nan")


def density_map_path(density_maps_dir: Path, well_identifier: str, timepoint: str) -> Path:
    return density_maps_dir / f"{well_identifier}_{timepoint}_total_density_epanechnikov.tif"


def resistant_density_map_path(
    density_maps_dir: Path,
    well_identifier: str,
    timepoint: str,
) -> Path:
    return density_maps_dir / f"{well_identifier}_{timepoint}_res_density_epanechnikov.tif"


def load_escape_cutoffs(path: Path) -> pd.DataFrame:
    """Load legacy experimental escape cutoffs without altering their logic."""
    if not path.exists():
        raise FileNotFoundError(f"Escape summary CSV does not exist: {path}")
    summary = pd.read_csv(path)
    sample_col = next(
        (column for column in ("replicate", "sample", "well") if column in summary.columns),
        None,
    )
    escape_col = next(
        (column for column in ("escape_day", "first_escape_day") if column in summary.columns),
        None,
    )
    if sample_col is None or escape_col is None or "plate" not in summary.columns:
        raise ValueError(
            "Escape summary must contain plate, a replicate/sample/well column, "
            "and escape_day or first_escape_day."
        )
    escape = summary[["plate", sample_col, escape_col]].copy()
    escape = escape.rename(columns={sample_col: "well", escape_col: "escape_day"})
    escape["escape_day"] = pd.to_numeric(escape["escape_day"], errors="coerce")
    escape = escape[np.isfinite(escape["escape_day"])].copy()
    if escape.empty:
        raise RuntimeError(f"No finite escape days found in {path}")
    if "last_confined_day" in summary.columns:
        cutoff = summary[["plate", sample_col, "last_confined_day"]].copy()
        cutoff = cutoff.rename(columns={sample_col: "well"})
        cutoff["last_confined_day"] = pd.to_numeric(cutoff["last_confined_day"], errors="coerce")
        escape = escape.merge(cutoff, on=["plate", "well"], how="left")
        escape["pre_escape_cutoff_day"] = escape["last_confined_day"].where(
            np.isfinite(escape["last_confined_day"]), escape["escape_day"]
        )
    else:
        escape["pre_escape_cutoff_day"] = escape["escape_day"]
    return (
        escape[["plate", "well", "escape_day", "pre_escape_cutoff_day"]]
        .groupby(["plate", "well"], as_index=False)
        .min()
    )


def normalize_simulation_run_id(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:run[_-]?)?0*(\d+)", text, flags=re.IGNORECASE)
    return text.lower() if match is None else str(int(match.group(1)))


def load_positive_tte(path: Path) -> dict[tuple[str, str], float]:
    """Return the positive simulation TTE values used by the legacy fallback."""
    summary = pd.read_csv(path, dtype={"protocol": str, "run": str})
    required = {"protocol", "run", "TTE"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    data = summary[["protocol", "run", "TTE"]].copy()
    data["strategy_key"] = data["protocol"].astype(str).map(normalize_strategy)
    data["run_key"] = data["run"].map(normalize_simulation_run_id)
    data["escape_day"] = pd.to_numeric(data["TTE"], errors="coerce")
    finite = data[np.isfinite(data["escape_day"]) & (data["escape_day"] > 0)]
    conflicts = finite.groupby(["strategy_key", "run_key"])["escape_day"].nunique()
    if not conflicts[conflicts > 1].empty:
        raise ValueError(f"Conflicting positive TTE values in {path}")
    finite = finite.groupby(["strategy_key", "run_key"], as_index=False)["escape_day"].first()
    return {
        (str(row.strategy_key), str(row.run_key)): float(row.escape_day)
        for row in finite.itertuples(index=False)
    }


def simulation_args(
    input_root: str,
    strategy: str,
    run_name: str | None = None,
    max_timepoints: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_root=input_root,
        strategies=[strategy],
        runs=[run_name] if run_name else None,
        max_runs_per_strategy=0,
        max_timepoints=int(max_timepoints),
        time_dump=sim.DEFAULT_TIME_DUMP,
        day_offset=sim.DEFAULT_DAY_OFFSET,
        active_flag=sim.DEFAULT_ACTIVE_FLAG,
        type_dataset=sim.DEFAULT_TYPE_DATASET,
        sensitive_value=sim.DEFAULT_SENSITIVE_VALUE,
        resistant_value=sim.DEFAULT_RESISTANT_VALUE,
        position_scale_px=sim.DEFAULT_POSITION_SCALE_PX,
        position_scale_um=None,
        no_outlier_filter=False,
        filter_center=list(sim.DEFAULT_FILTER_CENTER),
        filter_outlier_boundary=sim.DEFAULT_FILTER_OUTLIER_BOUNDARY,
        filter_neighbour_radius=sim.DEFAULT_FILTER_NEIGHBOUR_RADIUS,
        filter_min_neighbours=sim.DEFAULT_FILTER_MIN_NEIGHBOURS,
    )


def canvas_from_loaded(
    loaded: list[tuple[np.ndarray, np.ndarray]], radius_um: float
) -> sim.CanvasSpec:
    nonempty = [positions for positions, _types in loaded if positions.size]
    if not nonempty:
        raise RuntimeError("No filtered cells in simulation run")
    min_xy = np.nanmin(np.vstack([np.nanmin(item, axis=0) for item in nonempty]), axis=0)
    max_xy = np.nanmax(np.vstack([np.nanmax(item, axis=0) for item in nonempty]), axis=0)
    min_xy -= radius_um
    max_xy += radius_um
    span = np.maximum(max_xy - min_xy, sim.PIXEL_SIZE_UM)
    return sim.CanvasSpec(
        min_x_um=float(min_xy[0]), min_y_um=float(min_xy[1]),
        width_px=int(math.ceil(span[0] / sim.PIXEL_SIZE_UM)),
        height_px=int(math.ceil(span[1] / sim.PIXEL_SIZE_UM)),
    )


def resistant_and_total_density(
    susceptible, resistant, *, radius_um: float, canvas: sim.CanvasSpec, coarsening_factor: int
) -> tuple[np.ndarray, np.ndarray]:
    resistant_image = density_base.rasterize_points_to_coarse_grid(
        resistant, canvas.height_px, canvas.width_px, coarsening_factor
    )
    susceptible_image = density_base.rasterize_points_to_coarse_grid(
        susceptible, canvas.height_px, canvas.width_px, coarsening_factor
    )
    coarse_radius = density_base.radius_um_to_coarse_px(radius_um, coarsening_factor)
    resistant_density = density_base.get_local_density_fft(resistant_image, coarse_radius)
    total_density = density_base.get_local_density_fft(resistant_image + susceptible_image, coarse_radius)
    coarse_pixel_area_um2 = (sim.PIXEL_SIZE_UM * coarsening_factor) ** 2
    resistant_density = density_base.zero_density_fft_noise(resistant_density / coarse_pixel_area_um2)
    total_density = density_base.zero_density_fft_noise(total_density / coarse_pixel_area_um2)
    return resistant_density, total_density


# TIFF-map settings retained for the existing Figure 3 map overlays.
KERNEL_UM = 160
COARSENING_FACTOR = 16
DEFAULT_INVITRO_RUN = intermediate_dir(
    "experimental", "figure_3", "figure2_mask_filtered_mastercurve",
    "run_kernel_160_coarse_16_figure2_masks_all_clones", create=False,
)
DEFAULT_CONFIG = Path(__file__).with_name("density_maps_config_160_figure2_masks_all_clones.yaml")
DEFAULT_EXPERIMENT_ESCAPE = intermediate_dir("experimental", "figure_2", create=False) / "frontcontact_summary_ALL.csv"
DEFAULT_ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
DEFAULT_SIM_ROOT = SIMULATION_DATABASE
DEFAULT_SIM_ESCAPE = COMBINED_RESULTS_CSV
_LEGACY_STRATEGY_LABELS = {"ATagent": "LV-Agent", r"AT$^{+}$": "AT+"}

def canonical_strategy_label(strategy: str) -> str:
    """Return the Figure 3 strategy label, including legacy cache aliases."""
    label = _LEGACY_STRATEGY_LABELS.get(str(strategy).strip(), str(strategy).strip())
    canonical = normalize_strategy(label)
    return "AT+" if canonical == "AT100+" else canonical



def _margin_rows(path: Path, dataset_type: str) -> pd.DataFrame:
    """Return canonical escape-margin rows for one dataset, if available."""
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path)
    required = {"dataset_type", "plate", "sample", "day", "signed_escape_margin"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing escape-margin columns: {sorted(missing)}")
    rows = table.loc[table["dataset_type"].astype(str) == dataset_type].copy()
    rows["day"] = pd.to_numeric(rows["day"], errors="coerce")
    rows["signed_escape_margin"] = pd.to_numeric(rows["signed_escape_margin"], errors="coerce")
    return rows[np.isfinite(rows["day"]) & np.isfinite(rows["signed_escape_margin"])].copy()



def experimental_escape_cutoffs(legacy_path: Path, margin_path: Path) -> pd.DataFrame:
    """Prefer canonical margin-derived escape limits; retain legacy fallback."""
    rows = _margin_rows(margin_path, "experimental")
    if rows.empty:
        return load_escape_cutoffs(legacy_path)
    records: list[dict[str, object]] = []
    for (plate, well), group in rows.groupby(["plate", "sample"], sort=False):
        escaped = group.loc[group["signed_escape_margin"] > 0, "day"]
        if escaped.empty:
            continue
        escape_day = float(escaped.min())
        confined = group.loc[group["day"] < escape_day, "day"]
        if confined.empty:
            continue
        records.append({
            "plate": str(plate), "well": str(well),
            "escape_day": escape_day,
            "pre_escape_cutoff_day": float(confined.max()),
        })
    if not records:
        raise RuntimeError(f"No escaped experimental replicates found in {margin_path}")
    return pd.DataFrame(records)



def _margin_run_id(sample: object) -> str:
    """Normalize ``run_18`` and Figure-2-style ``R18_*`` identifiers alike."""
    text = str(sample).strip()
    match = re.match(r"(?:run[_-]?)?0*(\d+)$", text, flags=re.IGNORECASE)
    if match is None:
        match = re.match(r"R0*(\d+)(?:_|$)", text, flags=re.IGNORECASE)
    return str(int(match.group(1))) if match is not None else normalize_simulation_run_id(text)



def simulation_escape_days(legacy_path: Path, margin_path: Path) -> dict[tuple[str, str], float]:
    """Prefer canonical simulation margins; retain legacy TTE fallback."""
    rows = _margin_rows(margin_path, "simulation")
    if rows.empty:
        return load_positive_tte(legacy_path)
    result: dict[tuple[str, str], float] = {}
    for (strategy, sample), group in rows.groupby(["strategy", "sample"], sort=False):
        escaped = group.loc[group["signed_escape_margin"] > 0, "day"]
        if not escaped.empty:
            result[(canonical_strategy_label(strategy), _margin_run_id(sample))] = float(escaped.min())
    if not result:
        raise RuntimeError(f"No escaped simulation runs found in {margin_path}")
    return result



def simulation_density(
    positions_um: np.ndarray,
    type_values: np.ndarray,
    canvas: sim.CanvasSpec,
) -> tuple[np.ndarray, np.ndarray]:
    susceptible, resistant = sim.positions_to_density_inputs(
        positions_um,
        type_values,
        canvas,
    )
    return resistant_and_total_density(
        susceptible,
        resistant,
        radius_um=float(KERNEL_UM),
        canvas=canvas,
        coarsening_factor=COARSENING_FACTOR,
    )



def run_kernel_radii(script_path, default_radii):
    """Run an analysis entry point once per radius, or configure one child run.

    Separate processes keep radius-derived constants independent. Importing an
    analysis module never starts a computation.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kernel-radius-um", type=float, default=None)
    parser.add_argument("--kernel-radii-um", nargs="+", type=float, default=None)
    args, remaining = parser.parse_known_args()
    radii = ((args.kernel_radius_um,) if args.kernel_radius_um is not None
             else tuple(args.kernel_radii_um or default_radii))
    if not radii or any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("Kernel radii must be finite positive values")
    if args.kernel_radius_um is not None:
        os.environ["HDF5_KERNEL_RADIUS_UM"] = f"{args.kernel_radius_um:g}"
        return
    for radius in radii:
        environment = os.environ.copy()
        environment["HDF5_KERNEL_RADIUS_UM"] = f"{radius:g}"
        print(f"[kernel] {Path(script_path).name}: {radius:g} um", flush=True)
        subprocess.run(
            [sys.executable, os.fspath(script_path), "--kernel-radius-um", f"{radius:g}", *remaining],
            env=environment, check=True,
        )
    raise SystemExit(0)
