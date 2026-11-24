"""Compute the canonical signed escape margin for experiments and simulations.

Experimental inputs are the non-overwriting HDF5 files produced by
``analysis/basic/escape/Prepare_Escape_Analysis_Inputs.py``. Simulation inputs
are the ``*_simulation.h5`` point clouds produced by
``analysis/basic/density_maps/Prepare_Simulation_Pointclouds.py``. They contain
upstream-filtered active sensitive and resistant cells in experimental pixel
coordinates; density-map TIFFs are not used by this analysis.

For each timepoint and direction, the analysis uses exact tangent half-planes:

* confinement is the minimum number of sensitive cells beyond the outermost
  resistant-cell tangent;
* escaped resistant cells are the union of resistant row IDs beyond the
  outermost sensitive-cell tangent in any direction.

The signed value is negative confinement until escape and the positive number
of unique escaped resistant cells after escape. Only ``all_escape_margin.csv``
is written; TTE remains a downstream derivation (first row with a positive
signed margin).

Experimental resistant points have already passed the documented clone
eligibility policy in the preparation step: a tracked clone is retained only
when its final actual mapped appearance contains at least 10 resistant cells.
This removes transient isolated detections before geometric escape assessment;
the exact applied rule is propagated from the prepared-file metadata to every
output row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.strategy_colors import normalize_strategy  # noqa: E402
from shared.data_paths import intermediate_dir  # noqa: E402


DEFAULT_EXPERIMENTAL_ROOT = intermediate_dir(
    "experimental", "basic", "escape", create=False
)
DEFAULT_EXPERIMENTAL_PREPARED_SUBDIR = "escape_analysis_inputs"
DEFAULT_SIMULATION_ROOT = intermediate_dir(
    "simulation", "basic", "prepared_pointclouds", create=False
)
DEFAULT_OUTPUT_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
# Set True to replace the canonical escape-margin CSV when running from
# PyCharm's play button.
OVERWRITE = True
PREPARED_H5_SUFFIX = "_escape_analysis_inputs.h5"
SIMULATION_H5_SUFFIX = "_simulation.h5"
ANALYSIS_VERSION = 2
DEFAULT_ANGLE_COUNT = 360
DEFAULT_DIRECTION_CHUNK_SIZE = 32
PROJECTION_TOLERANCE = 1e-9
SIMULATION_TIMEPOINT_RE = re.compile(r"^d_(\d+)$")


def outward_normals(angles_deg: np.ndarray) -> np.ndarray:
    """Return unit outward normals for angles measured clockwise from the top."""
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.column_stack((np.sin(theta), -np.cos(theta)))


@dataclass(frozen=True)
class MarginMetrics:
    confinement_count: int | None
    escaped_resistant_count: int
    signed_escape_margin: float
    phase: str
    min_confinement_angle_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one canonical signed escape-margin table from prepared "
            "experimental and/or simulation point-cloud HDF5 files."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("experimental", "simulation", "both"),
        default="both",
    )
    parser.add_argument(
        "--experimental-prepared-root",
        type=Path,
        default=DEFAULT_EXPERIMENTAL_ROOT,
        help=(
            "Root containing Plate_*/escape_analysis_inputs, or any directory "
            "below which prepared HDF5 files can be found."
        ),
    )
    parser.add_argument(
        "--experimental-prepared-subdir",
        default=DEFAULT_EXPERIMENTAL_PREPARED_SUBDIR,
    )
    parser.add_argument(
        "--simulation-root", type=Path, default=DEFAULT_SIMULATION_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        default=None,
        help="Optional sample selector. Repeat for multiple samples.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Deterministic validation limit per selected dataset; zero means all.",
    )
    parser.add_argument(
        "--angle-count", type=int, default=DEFAULT_ANGLE_COUNT
    )
    parser.add_argument(
        "--direction-chunk-size",
        type=int,
        default=DEFAULT_DIRECTION_CHUNK_SIZE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help=(
            "Replace this analysis's existing output CSV (also controlled by OVERWRITE)."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail rather than replace an existing output CSV.",
    )
    return parser.parse_args()


def probe_angles(angle_count: int) -> np.ndarray:
    if angle_count < 4:
        raise ValueError("angle_count must be at least 4")
    return np.arange(angle_count, dtype=float) * (360.0 / float(angle_count))


def _as_xy(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def signed_escape_metrics(
    resistant_xy: np.ndarray,
    sensitive_xy: np.ndarray,
    *,
    angles_deg: np.ndarray,
    direction_chunk_size: int = DEFAULT_DIRECTION_CHUNK_SIZE,
) -> MarginMetrics:
    """Return the exact directional confinement/escape metric for one frame."""
    resistant = _as_xy(resistant_xy, "resistant_xy")
    sensitive = _as_xy(sensitive_xy, "sensitive_xy")
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1 or not len(angles):
        raise ValueError("angles_deg must be a non-empty one-dimensional array")
    if direction_chunk_size < 1:
        raise ValueError("direction_chunk_size must be positive")

    if not len(resistant):
        return MarginMetrics(
            confinement_count=None,
            escaped_resistant_count=0,
            signed_escape_margin=np.nan,
            phase="no_resistant_cells",
            min_confinement_angle_deg=np.nan,
        )
    if not len(sensitive):
        return MarginMetrics(
            confinement_count=0,
            escaped_resistant_count=int(len(resistant)),
            signed_escape_margin=float(len(resistant)),
            phase="escaped",
            min_confinement_angle_deg=np.nan,
        )

    normals = outward_normals(angles)
    confinement_counts = np.empty(len(normals), dtype=np.int64)
    escaped = np.zeros(len(resistant), dtype=bool)

    for start in range(0, len(normals), direction_chunk_size):
        stop = min(start + direction_chunk_size, len(normals))
        chunk = normals[start:stop]
        resistant_projection = resistant @ chunk.T
        sensitive_projection = sensitive @ chunk.T

        resistant_front = np.max(resistant_projection, axis=0)
        confinement_counts[start:stop] = np.count_nonzero(
            sensitive_projection
            >= resistant_front[np.newaxis, :] - PROJECTION_TOLERANCE,
            axis=0,
        )

        sensitive_front = np.max(sensitive_projection, axis=0)
        escaped |= np.any(
            resistant_projection
            > sensitive_front[np.newaxis, :] + PROJECTION_TOLERANCE,
            axis=1,
        )

    minimum_idx = int(np.argmin(confinement_counts))
    confinement_count = int(confinement_counts[minimum_idx])
    escaped_count = int(np.count_nonzero(escaped))
    if escaped_count > 0:
        signed_margin = float(escaped_count)
        phase = "escaped"
    else:
        signed_margin = float(-confinement_count)
        phase = "confined" if confinement_count > 0 else "boundary"

    return MarginMetrics(
        confinement_count=confinement_count,
        escaped_resistant_count=escaped_count,
        signed_escape_margin=signed_margin,
        phase=phase,
        min_confinement_angle_deg=float(angles[minimum_idx]),
    )


def _limit_paths(paths: list[Path], max_files: int) -> list[Path]:
    return paths if max_files <= 0 else paths[:max_files]


def discover_prepared_experiments(
    root: Path,
    *,
    prepared_subdir: str,
    samples: Iterable[str] | None,
    max_files: int,
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared experimental root does not exist: {root}")
    requested = set(samples or ())
    direct = sorted(
        root.glob(f"Plate_*/{prepared_subdir}/*{PREPARED_H5_SUFFIX}")
    )
    paths = direct or sorted(root.rglob(f"*{PREPARED_H5_SUFFIX}"))
    if requested:
        paths = [
            path
            for path in paths
            if path.name.removesuffix(PREPARED_H5_SUFFIX) in requested
        ]
    paths = _limit_paths(paths, max_files)
    if not paths:
        raise FileNotFoundError(f"No prepared experimental HDF5 files found below {root}")
    return paths


def discover_simulations(
    root: Path,
    *,
    samples: Iterable[str] | None,
    max_files: int,
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Simulation root does not exist: {root}")
    requested = set(samples or ())
    paths = sorted(root.glob(f"Plate_*/hdf5/*{SIMULATION_H5_SUFFIX}"))
    if requested:
        paths = [
            path
            for path in paths
            if path.name.removesuffix(SIMULATION_H5_SUFFIX) in requested
        ]
    paths = _limit_paths(paths, max_files)
    if not paths:
        raise FileNotFoundError(f"No simulation HDF5 files found below {root}")
    return paths


def _prepared_xy(store: pd.HDFStore, channel: str, day_key: str) -> np.ndarray:
    key = f"/{channel}/{day_key}"
    if key not in store.keys():
        return np.empty((0, 2), dtype=float)
    frame = store.get(key)
    missing = {"x", "y"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{store.filename}:{key} is missing columns {sorted(missing)}")
    return frame[["x", "y"]].to_numpy(dtype=float)


def quantify_prepared_experiment(
    path: Path,
    *,
    angles_deg: np.ndarray,
    direction_chunk_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with pd.HDFStore(path, mode="r") as store:
        required = {"/metadata/run", "/qc/daily", "/qc/clone_eligibility"}
        missing = required.difference(store.keys())
        if missing:
            raise ValueError(f"{path} is missing prepared keys {sorted(missing)}")
        metadata = store.get("/metadata/run")
        qc = store.get("/qc/daily").sort_values("day")
        if len(metadata) != 1:
            raise ValueError(f"{path} must contain exactly one /metadata/run row")
        meta = metadata.iloc[0]

        for record in qc.itertuples(index=False):
            timepoint = str(record.timepoint)
            day_key = f"d_{timepoint.zfill(2)}"
            resistant = _prepared_xy(store, "res", day_key)
            sensitive = _prepared_xy(store, "sus", day_key)
            metrics = signed_escape_metrics(
                resistant,
                sensitive,
                angles_deg=angles_deg,
                direction_chunk_size=direction_chunk_size,
            )
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "dataset_type": "experimental",
                    "plate": str(meta["plate"]),
                    "sample": str(meta["sample"]),
                    "replicate": str(meta["sample"]),
                    "strategy": normalize_strategy(str(meta["strategy"])),
                    "timepoint": timepoint,
                    "day": int(record.day),
                    "frame_idx": int(record.frame_idx),
                    "n_resistant_raw": int(record.n_resistant_raw),
                    "n_resistant_retained": int(record.n_resistant_retained),
                    "n_sensitive_raw": int(record.n_sensitive_raw),
                    "n_sensitive_retained": int(record.n_sensitive_retained),
                    "confinement_count": metrics.confinement_count,
                    "escaped_resistant_count": metrics.escaped_resistant_count,
                    "signed_escape_margin": metrics.signed_escape_margin,
                    "phase": metrics.phase,
                    "min_confinement_angle_deg": metrics.min_confinement_angle_deg,
                    "angle_count": int(len(angles_deg)),
                    "resistant_filter": str(meta["resistant_filter"]),
                    "sensitive_cleanup": bool(record.sensitive_cleanup),
                    "input_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def _simulation_xy(group: h5py.Group, day_key: str) -> np.ndarray:
    dataset_path = f"{day_key}/block0_values"
    if dataset_path not in group:
        return np.empty((0, 2), dtype=float)
    values = np.asarray(group[dataset_path])
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"Unexpected simulation point-cloud shape at {group.name}/{dataset_path}: "
            f"{values.shape}"
        )
    # Simulation output intentionally follows the experimental HDF convention:
    # block0_values[:, 0] is y and block0_values[:, 1] is x.
    return np.column_stack((values[:, 1], values[:, 0])).astype(float, copy=False)


def quantify_simulation(
    path: Path,
    *,
    angles_deg: np.ndarray,
    direction_chunk_size: int,
) -> pd.DataFrame:
    plate = path.parent.parent.name
    sample = path.name.removesuffix(SIMULATION_H5_SUFFIX)
    strategy = normalize_strategy(plate.removeprefix("Plate_"))
    rows: list[dict[str, object]] = []
    with h5py.File(path, mode="r") as handle:
        if "res" not in handle or "sus" not in handle:
            raise ValueError(f"Simulation HDF5 lacks /res or /sus: {path}")
        day_keys = sorted(
            set(handle["res"]).intersection(handle["sus"]),
            key=lambda value: int(value.removeprefix("d_")),
        )
        if not day_keys:
            raise ValueError(f"No common /res and /sus timepoints in {path}")

        for frame_idx, day_key in enumerate(day_keys):
            match = SIMULATION_TIMEPOINT_RE.match(day_key)
            if match is None:
                raise ValueError(f"Unexpected simulation timepoint {day_key!r} in {path}")
            timepoint = match.group(1)
            resistant = _simulation_xy(handle["res"], day_key)
            sensitive = _simulation_xy(handle["sus"], day_key)
            metrics = signed_escape_metrics(
                resistant,
                sensitive,
                angles_deg=angles_deg,
                direction_chunk_size=direction_chunk_size,
            )
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "dataset_type": "simulation",
                    "plate": plate,
                    "sample": sample,
                    "replicate": sample,
                    "strategy": strategy,
                    "timepoint": timepoint,
                    "day": int(timepoint),
                    "frame_idx": int(frame_idx),
                    "n_resistant_raw": int(len(resistant)),
                    "n_resistant_retained": int(len(resistant)),
                    "n_sensitive_raw": int(len(sensitive)),
                    "n_sensitive_retained": int(len(sensitive)),
                    "confinement_count": metrics.confinement_count,
                    "escaped_resistant_count": metrics.escaped_resistant_count,
                    "signed_escape_margin": metrics.signed_escape_margin,
                    "phase": metrics.phase,
                    "min_confinement_angle_deg": metrics.min_confinement_angle_deg,
                    "angle_count": int(len(angles_deg)),
                    "resistant_filter": "simulation_upstream_filtered_all_resistant",
                    "sensitive_cleanup": False,
                    "input_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def _atomic_write_csv(table: pd.DataFrame, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        table.to_csv(temporary, index=False)
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to replace an output created concurrently: {output}"
            )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.output.exists() and not args.overwrite:
        from reproduction.stage_checks import check_escape_margins
        try:
            check_escape_margins(args.output.parent, module=sys.modules[__name__], args=args)
            print(f'[skip] verified complete escape-margin table: {args.output}')
            return pd.DataFrame()
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f'[rebuild] incomplete or stale escape-margin table: {error}')
            args.overwrite = True
    angles = probe_angles(int(args.angle_count))
    frames: list[pd.DataFrame] = []

    if args.dataset in {"experimental", "both"}:
        paths = discover_prepared_experiments(
            args.experimental_prepared_root,
            prepared_subdir=str(args.experimental_prepared_subdir),
            samples=args.samples,
            max_files=int(args.max_files),
        )
        for index, path in enumerate(paths, start=1):
            print(f"[experimental {index}/{len(paths)}] {path}")
            frames.append(
                quantify_prepared_experiment(
                    path,
                    angles_deg=angles,
                    direction_chunk_size=int(args.direction_chunk_size),
                )
            )

    if args.dataset in {"simulation", "both"}:
        paths = discover_simulations(
            args.simulation_root,
            samples=args.samples,
            max_files=int(args.max_files),
        )
        for index, path in enumerate(paths, start=1):
            print(f"[simulation {index}/{len(paths)}] {path}")
            frames.append(
                quantify_simulation(
                    path,
                    angles_deg=angles,
                    direction_chunk_size=int(args.direction_chunk_size),
                )
            )

    if not frames:
        raise RuntimeError("No escape-margin inputs were quantified")
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(
        ["dataset_type", "strategy", "plate", "sample", "day"]
    ).reset_index(drop=True)
    _atomic_write_csv(table, args.output, overwrite=bool(args.overwrite))
    from reproduction.stage_checks import record_escape_margins
    record_escape_margins(sys.modules[__name__], args)
    print(
        f"[saved] {args.output} ({len(table)} rows, "
        f"{table[['dataset_type', 'plate', 'sample']].drop_duplicates().shape[0]} replicates)"
    )
    return table


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
