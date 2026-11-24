"""Prepare QC-filtered experimental point clouds for escape-margin analysis.

Inputs (read-only)
------------------
* ``exp_raw_data/Plate_*/hdf5/*_total_poisson_padded.h5``
* ``exp_raw_data/Plate_*/clone_masks_stacks/*_res_clone_masks.tif``
* ``intermediate_analysis_data/experimental/basic/escape/``
  ``sensitive_clean_masks_stacks/Plate_*/*_sus_clean_masks.tif``

For every day, susceptible points are retained where the cleaned susceptible
mask is non-zero. Resistant points must first map to a tracked clone (mask
label >= 2). A tracked clone is then retained only if it has at least 10 mapped
resistant cells at its final actual appearance in the replicate.

This terminal-clone criterion removes isolated detections, dust, and transient
lineages that never establish a resistant population, while retaining every
established clone rather than only the historical single dominant clone. The
criterion is evaluated from mapped point counts, not clone-mask area: a clone's
``last actual appearance`` is its final frame containing one or more mapped
resistant-cell coordinates. Original floating-point coordinates and row order
are preserved for all retained cells.

Outputs
-------
* ``intermediate_analysis_data/experimental/basic/escape/Plate_*/``
  ``escape_analysis_inputs/*_escape_analysis_inputs.h5``
* A batch summary CSV and, while ``QC`` is true, representative visual QC PNGs.

Raw inputs are never modified. Existing prepared HDF5 files are skipped unless
``OVERWRITE = True`` (or ``--overwrite``) is supplied. New files are validated
before atomic placement.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tifffile

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_paths import EXP_RAW_DATA, intermediate_dir  # noqa: E402


# Visual QC is intentionally enabled for the first production run. The CLI
# also exposes --qc/--no-qc without changing this source-level default.
QC = True
# Set True to regenerate prepared HDF5 inputs, the summary CSV, and QC output
# when launching this file directly from PyCharm.
OVERWRITE = True
QC_MAX_REPRESENTATIVE_WELLS = 8
QC_FRAMES_PER_WELL = 3
QC_MAX_POINTS_PER_CLASS = 25_000
QC_MAX_IMAGE_DIMENSION = 1_600

PLATES = [f"Plate_{index}" for index in range(1, 9)]
ESCAPE_ROOT = intermediate_dir("experimental", "basic", "escape", create=False)
CLEAN_SENSITIVE_ROOT = ESCAPE_ROOT / "sensitive_clean_masks_stacks"
PREPARED_SUBDIR = "escape_analysis_inputs"
PREPARED_SUFFIX = "_escape_analysis_inputs.h5"
SUMMARY_CSV = "Prepare_Escape_Analysis_Inputs_summary.csv"
VISUAL_QC_SUBDIR = "prepare_escape_analysis_inputs_visual_qc"
PREPARATION_VERSION = 2
MIN_TERMINAL_CLONE_CELLS = 10
RESISTANT_FILTER_DESCRIPTION = (
    "tracked_clone_labels_ge_2_with_terminal_mapped_count_ge_10"
)

CLONE_SUFFIX = "_res_clone_masks.tif"
CLEAN_SENSITIVE_SUFFIX = "_sus_clean_masks.tif"
H5_SAMPLE_RE = re.compile(r"(.+)_total_poisson(?:_padded)?\.h5$", re.IGNORECASE)
WELL_SAMPLE_RE = re.compile(
    r"(?:^|_)([A-H]\d{1,2}_.+?-[A-H]\d{1,2})$", re.IGNORECASE
)
SAMPLE_STRATEGY_RE = re.compile(
    r"^[A-H]\d{1,2}_(?P<strategy>.+)-[A-H]\d{1,2}$", re.IGNORECASE
)
DAY_KEY_RE = re.compile(r"^/(res|sus)/d_(\d+)$")


@dataclass(frozen=True)
class ReplicateInputs:
    plate: str
    sample: str
    strategy: str
    raw_h5_path: Path
    clone_mask_path: Path
    clean_sensitive_path: Path
    output_path: Path
    visual_qc: bool = False


def day_from_frame(frame_idx: int) -> int:
    offsets = (1, 3, 5)
    return (int(frame_idx) // 3) * 7 + offsets[int(frame_idx) % 3]


def parse_sample_from_h5(path: Path) -> Optional[str]:
    match = H5_SAMPLE_RE.match(path.name)
    if match is None:
        return None
    raw_name = match.group(1)
    sample_match = WELL_SAMPLE_RE.search(raw_name)
    return sample_match.group(1) if sample_match else raw_name


def parse_strategy(sample: str) -> str:
    match = SAMPLE_STRATEGY_RE.match(sample)
    return match.group("strategy") if match else "UNKNOWN"


def discover_replicates(
    plates: Iterable[str], samples: Optional[Iterable[str]] = None
) -> list[ReplicateInputs]:
    requested_samples = set(samples or ())
    replicates: list[ReplicateInputs] = []

    for plate in plates:
        plate_raw = EXP_RAW_DATA / plate
        h5_dir = plate_raw / "hdf5"
        selected_by_sample: dict[str, Path] = {}

        for pattern in ("*_total_poisson_padded.h5", "*_total_poisson.h5"):
            for raw_path in sorted(h5_dir.glob(pattern)):
                sample = parse_sample_from_h5(raw_path)
                if sample is None or sample in selected_by_sample:
                    continue
                selected_by_sample[sample] = raw_path

        for sample, raw_h5_path in sorted(selected_by_sample.items()):
            if requested_samples and sample not in requested_samples:
                continue
            output_path = (
                ESCAPE_ROOT / plate / PREPARED_SUBDIR / f"{sample}{PREPARED_SUFFIX}"
            )
            replicates.append(
                ReplicateInputs(
                    plate=plate,
                    sample=sample,
                    strategy=parse_strategy(sample),
                    raw_h5_path=raw_h5_path,
                    clone_mask_path=(
                        plate_raw / "clone_masks_stacks" / f"{sample}{CLONE_SUFFIX}"
                    ),
                    clean_sensitive_path=(
                        CLEAN_SENSITIVE_ROOT
                        / plate
                        / f"{sample}{CLEAN_SENSITIVE_SUFFIX}"
                    ),
                    output_path=output_path,
                )
            )
    return replicates


def select_visual_qc_replicates(
    replicates: list[ReplicateInputs], enabled: bool
) -> list[ReplicateInputs]:
    if not enabled:
        return replicates

    complete = [
        rep
        for rep in replicates
        if rep.clone_mask_path.is_file() and rep.clean_sensitive_path.is_file()
    ]
    by_strategy: dict[str, list[ReplicateInputs]] = {}
    for rep in complete:
        by_strategy.setdefault(rep.strategy, []).append(rep)

    selected: set[tuple[str, str]] = set()
    for strategy in sorted(by_strategy):
        candidates = sorted(by_strategy[strategy], key=lambda rep: (rep.plate, rep.sample))
        selected_rep = candidates[len(candidates) // 2]
        selected.add((selected_rep.plate, selected_rep.sample))
        if len(selected) >= QC_MAX_REPRESENTATIVE_WELLS:
            break

    return [
        ReplicateInputs(**{**rep.__dict__, "visual_qc": (rep.plate, rep.sample) in selected})
        for rep in replicates
    ]


def raw_days(store: pd.HDFStore, channel: str) -> list[int]:
    days: list[int] = []
    for key in store.keys():
        match = DAY_KEY_RE.match(key)
        if match is not None and match.group(1) == channel:
            days.append(int(match.group(2)))
    return sorted(days)


def load_xy(store: pd.HDFStore, channel: str, day: int) -> np.ndarray:
    key = f"/{channel}/d_{int(day):02d}"
    frame = store.get(key)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{store.filename}:{key} is not a pandas DataFrame")
    missing = {"x", "y"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{store.filename}:{key} missing columns {sorted(missing)}")
    return frame[["x", "y"]].to_numpy(dtype=float)


def mask_membership(
    xy: np.ndarray, mask: np.ndarray, *, resistant: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return keep, finite, in-bounds, and sampled pixel values."""
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Point coordinates must have shape (n, 2), got {xy.shape}")

    h, w = mask.shape
    finite = np.isfinite(xy).all(axis=1)
    xi = np.zeros(len(xy), dtype=np.int64)
    yi = np.zeros(len(xy), dtype=np.int64)
    if np.any(finite):
        xi[finite] = np.rint(xy[finite, 0]).astype(np.int64)
        yi[finite] = np.rint(xy[finite, 1]).astype(np.int64)
    in_bounds = finite & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    sampled = np.zeros(len(xy), dtype=mask.dtype)
    if np.any(in_bounds):
        sampled[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]
    keep = in_bounds & (sampled >= 2 if resistant else sampled != 0)
    return keep, finite, in_bounds, sampled


def determine_clone_eligibility(
    raw_store: pd.HDFStore,
    clone_tiff: tifffile.TiffFile,
    days: list[int],
) -> pd.DataFrame:
    """Return one auditable terminal-count decision per tracked clone.

    Counts are based on raw resistant point coordinates that map to labels >= 2
    in each corresponding clone-mask frame. The final observed non-zero mapped
    count is the terminal count; a clone is eligible iff that count is at least
    ``MIN_TERMINAL_CLONE_CELLS``. This deliberately filters transient/dust-like
    lineages without imposing the historical single-dominant-clone restriction.
    """
    records: dict[int, dict[str, int]] = {}
    for frame_idx, day in enumerate(days):
        clone_mask = clone_tiff.pages[frame_idx].asarray()
        resistant_raw = load_xy(raw_store, "res", day)
        tracked, _finite, _in_bounds, labels = mask_membership(
            resistant_raw, clone_mask, resistant=True
        )
        observed_labels, counts = np.unique(labels[tracked], return_counts=True)
        for label, count in zip(observed_labels, counts):
            clone_label = int(label)
            cell_count = int(count)
            record = records.setdefault(
                clone_label,
                {
                    "clone_label": clone_label,
                    "first_frame_idx": int(frame_idx),
                    "first_day": int(day),
                    "last_frame_idx": int(frame_idx),
                    "last_day": int(day),
                    "terminal_mapped_cell_count": cell_count,
                    "max_mapped_cell_count": cell_count,
                    "n_observed_frames": 0,
                },
            )
            record["last_frame_idx"] = int(frame_idx)
            record["last_day"] = int(day)
            record["terminal_mapped_cell_count"] = cell_count
            record["max_mapped_cell_count"] = max(
                int(record["max_mapped_cell_count"]), cell_count
            )
            record["n_observed_frames"] = int(record["n_observed_frames"]) + 1

    columns = [
        "clone_label", "first_frame_idx", "first_day", "last_frame_idx",
        "last_day", "terminal_mapped_cell_count", "max_mapped_cell_count",
        "n_observed_frames", "minimum_terminal_cell_count", "eligible",
        "decision_reason",
    ]
    rows: list[dict[str, object]] = []
    for clone_label in sorted(records):
        record = records[clone_label]
        terminal_count = int(record["terminal_mapped_cell_count"])
        eligible = terminal_count >= MIN_TERMINAL_CLONE_CELLS
        rows.append(
            {
                **record,
                "minimum_terminal_cell_count": MIN_TERMINAL_CLONE_CELLS,
                "eligible": eligible,
                "decision_reason": (
                    "kept_terminal_count_meets_threshold"
                    if eligible
                    else "dropped_terminal_count_below_threshold"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def representative_frame_indices(n_frames: int) -> set[int]:
    if n_frames <= 0:
        return set()
    candidates = np.linspace(
        0, n_frames - 1, num=min(QC_FRAMES_PER_WELL, n_frames), dtype=int
    )
    return {int(index) for index in candidates}


def _plot_points(ax, xy: np.ndarray, keep: np.ndarray) -> None:
    def limited(points: np.ndarray) -> np.ndarray:
        if len(points) <= QC_MAX_POINTS_PER_CLASS:
            return points
        indices = np.linspace(
            0, len(points) - 1, QC_MAX_POINTS_PER_CLASS, dtype=np.int64
        )
        return points[indices]

    excluded = limited(xy[(~keep) & np.isfinite(xy).all(axis=1)])
    retained = limited(xy[keep])
    if len(excluded):
        ax.scatter(
            excluded[:, 0], excluded[:, 1], s=1.0, c="#f28e2b", alpha=0.35,
            linewidths=0, label=f"excluded ({np.count_nonzero(~keep)})",
        )
    if len(retained):
        ax.scatter(
            retained[:, 0], retained[:, 1], s=1.0, c="#00e5ff", alpha=0.65,
            linewidths=0, label=f"retained ({np.count_nonzero(keep)})",
        )


def write_visual_qc(
    rep: ReplicateInputs,
    frame_idx: int,
    day: int,
    clone_mask: np.ndarray,
    clean_sensitive_mask: np.ndarray,
    resistant_xy: np.ndarray,
    sensitive_xy: np.ndarray,
    resistant_keep: np.ndarray,
    sensitive_keep: np.ndarray,
    qc_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    h, w = clone_mask.shape
    step = max(1, int(np.ceil(max(h, w) / QC_MAX_IMAGE_DIMENSION)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    panels = (
        (axes[0], clone_mask >= 2, resistant_xy, resistant_keep, "Resistant: clone label ≥ 2"),
        (axes[1], clean_sensitive_mask != 0, sensitive_xy, sensitive_keep, "Sensitive: cleaned mask"),
    )
    for axis, accepted_mask, xy, keep, title in panels:
        axis.imshow(
            accepted_mask[::step, ::step],
            cmap="gray",
            interpolation="nearest",
            extent=(0, w, h, 0),
            vmin=0,
            vmax=1,
        )
        _plot_points(axis, xy, keep)
        axis.set_xlim(0, w)
        axis.set_ylim(h, 0)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("x [px]")
        axis.set_ylabel("y [px]")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="upper right", fontsize=7, markerscale=4)

    figure.suptitle(
        f"{rep.plate} / {rep.sample} / day {day:02d} (frame {frame_idx})"
    )
    out_dir = qc_root / rep.plate / rep.sample
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rep.sample}_d_{day:02d}_filter_qc.png"
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def validate_source_alignment(
    store: pd.HDFStore,
    clone_tiff: tifffile.TiffFile,
    clean_tiff: tifffile.TiffFile,
) -> list[int]:
    resistant_days = raw_days(store, "res")
    sensitive_days = raw_days(store, "sus")
    if resistant_days != sensitive_days:
        raise ValueError(
            f"Raw resistant/sensitive day keys differ: res={resistant_days}, "
            f"sus={sensitive_days}"
        )
    if not resistant_days:
        raise ValueError("Raw HDF5 contains no paired resistant/sensitive days")

    clone_frames = len(clone_tiff.pages)
    clean_frames = len(clean_tiff.pages)
    if clone_frames != clean_frames:
        raise ValueError(
            f"TIFF frame-count mismatch: clone={clone_frames}, clean_sensitive={clean_frames}"
        )
    if clone_frames != len(resistant_days):
        raise ValueError(
            f"HDF5/TIFF frame-count mismatch: hdf_days={len(resistant_days)}, "
            f"tiff_frames={clone_frames}"
        )

    expected_days = [day_from_frame(index) for index in range(clone_frames)]
    if resistant_days != expected_days:
        raise ValueError(
            f"Unexpected day/frame schedule: hdf_days={resistant_days}, "
            f"expected={expected_days}"
        )

    clone_shape = tuple(clone_tiff.pages[0].shape)
    clean_shape = tuple(clean_tiff.pages[0].shape)
    if clone_shape != clean_shape:
        raise ValueError(
            f"TIFF canvas mismatch: clone={clone_shape}, clean_sensitive={clean_shape}"
        )
    if len(clone_shape) != 2:
        raise ValueError(f"Expected 2D TIFF frames, got {clone_shape}")
    return resistant_days


def validate_prepared_h5(path: Path, days: list[int]) -> None:
    with pd.HDFStore(os.fspath(path), mode="r") as store:
        required = {"/metadata/run", "/qc/daily", "/qc/clone_eligibility"}
        for day in days:
            required.update({f"/res/d_{day:02d}", f"/sus/d_{day:02d}"})
        missing = required.difference(store.keys())
        if missing:
            raise ValueError(f"Prepared HDF5 missing keys: {sorted(missing)}")
        metadata = store.get("/metadata/run")
        daily = store.get("/qc/daily")
        clone_eligibility = store.get("/qc/clone_eligibility")
        if not all(
            isinstance(table, pd.DataFrame)
            for table in (metadata, daily, clone_eligibility)
        ):
            raise TypeError("Prepared metadata and daily QC keys must be DataFrames")
        if len(metadata) != 1:
            raise ValueError("Prepared /metadata/run must contain exactly one row")
        if len(daily) != len(days):
            raise ValueError(
                f"Prepared /qc/daily has {len(daily)} rows, expected {len(days)}"
            )


def prepare_replicate(
    rep: ReplicateInputs, overwrite: bool, qc_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    result: dict[str, object] = {
        "plate": rep.plate,
        "sample": rep.sample,
        "strategy": rep.strategy,
        "raw_h5_path": str(rep.raw_h5_path),
        "clone_mask_path": str(rep.clone_mask_path),
        "clean_sensitive_path": str(rep.clean_sensitive_path),
        "output_path": str(rep.output_path),
        "visual_qc": bool(rep.visual_qc),
        "status": "ok",
        "n_days": 0,
        "n_resistant_raw": 0,
        "n_resistant_retained": 0,
        "n_sensitive_raw": 0,
        "n_sensitive_retained": 0,
        "elapsed_s": np.nan,
        "error": "",
    }

    try:
        replacing_incomplete = False
        if rep.output_path.exists() and not overwrite:
            try:
                with pd.HDFStore(os.fspath(rep.raw_h5_path), 'r') as source:
                    days = raw_days(source, 'res')
                validate_prepared_h5(rep.output_path, days)
                from reproduction.file_checks import fresh_outputs
                if not fresh_outputs([rep.output_path], [rep.raw_h5_path, rep.clone_mask_path, rep.clean_sensitive_path]):
                    raise ValueError('source files are newer than prepared coordinates')
                with pd.HDFStore(os.fspath(rep.output_path), 'r') as prepared:
                    if int(prepared['/metadata/run']['preparation_version'].iloc[0]) != PREPARATION_VERSION:
                        raise ValueError('incompatible preparation version')
                result["status"] = "skipped_exists"
                result["elapsed_s"] = 0.0
                return result
            except Exception as error:
                replacing_incomplete = True
                print(f'[rebuild] {rep.plate}/{rep.sample}: {error}', flush=True)
        for status, path in (
            ("missing_raw_h5", rep.raw_h5_path),
            ("missing_clone_mask", rep.clone_mask_path),
            ("missing_clean_sensitive", rep.clean_sensitive_path),
        ):
            if not path.is_file():
                result["status"] = status
                result["elapsed_s"] = 0.0
                return result

        rep.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = rep.output_path.with_name(
            f".{rep.output_path.name}.tmp-{os.getpid()}"
        )
        if temporary_path.exists():
            temporary_path.unlink()

        try:
            with (
                pd.HDFStore(os.fspath(rep.raw_h5_path), mode="r") as raw_store,
                tifffile.TiffFile(rep.clone_mask_path) as clone_tiff,
                tifffile.TiffFile(rep.clean_sensitive_path) as clean_tiff,
            ):
                days = validate_source_alignment(raw_store, clone_tiff, clean_tiff)
                result["n_days"] = len(days)
                clone_eligibility = determine_clone_eligibility(
                    raw_store, clone_tiff, days
                )
                eligible_clone_labels = clone_eligibility.loc[
                    clone_eligibility["eligible"], "clone_label"
                ].to_numpy(dtype=np.int64)
                result["n_tracked_clones"] = int(len(clone_eligibility))
                result["n_eligible_clones"] = int(len(eligible_clone_labels))
                visual_frames = (
                    representative_frame_indices(len(days)) if rep.visual_qc else set()
                )
                daily_rows: list[dict[str, object]] = []

                with pd.HDFStore(
                    os.fspath(temporary_path), mode="w", complevel=5, complib="zlib"
                ) as output_store:
                    for frame_idx, day in enumerate(days):
                        clone_mask = clone_tiff.pages[frame_idx].asarray()
                        clean_mask = clean_tiff.pages[frame_idx].asarray()
                        if clone_mask.ndim != 2 or clean_mask.ndim != 2:
                            raise ValueError(
                                f"Non-2D frame at index {frame_idx}: "
                                f"clone={clone_mask.shape}, clean={clean_mask.shape}"
                            )
                        if clone_mask.shape != clean_mask.shape:
                            raise ValueError(
                                f"Frame canvas mismatch at index {frame_idx}: "
                                f"clone={clone_mask.shape}, clean={clean_mask.shape}"
                            )

                        resistant_raw = load_xy(raw_store, "res", day)
                        sensitive_raw = load_xy(raw_store, "sus", day)
                        res_keep, res_finite, res_inb, res_labels = mask_membership(
                            resistant_raw, clone_mask, resistant=True
                        )
                        tracked_res_keep = res_keep.copy()
                        res_keep &= np.isin(res_labels, eligible_clone_labels)
                        sus_keep, sus_finite, sus_inb, _ = mask_membership(
                            sensitive_raw, clean_mask, resistant=False
                        )

                        resistant_retained = pd.DataFrame(
                            resistant_raw[res_keep], columns=["x", "y"]
                        )
                        sensitive_retained = pd.DataFrame(
                            sensitive_raw[sus_keep], columns=["x", "y"]
                        )
                        output_store.put(
                            f"/res/d_{day:02d}", resistant_retained, format="fixed"
                        )
                        output_store.put(
                            f"/sus/d_{day:02d}", sensitive_retained, format="fixed"
                        )

                        row = {
                            "timepoint": f"{day:02d}",
                            "day": int(day),
                            "frame_idx": int(frame_idx),
                            "height_px": int(clone_mask.shape[0]),
                            "width_px": int(clone_mask.shape[1]),
                            "n_resistant_raw": int(len(resistant_raw)),
                            "n_resistant_retained": int(np.count_nonzero(res_keep)),
                            "n_resistant_nonfinite": int(np.count_nonzero(~res_finite)),
                            "n_resistant_out_of_bounds": int(
                                np.count_nonzero(res_finite & ~res_inb)
                            ),
                            "n_resistant_label_0": int(
                                np.count_nonzero(res_inb & (res_labels == 0))
                            ),
                            "n_resistant_label_1": int(
                                np.count_nonzero(res_inb & (res_labels == 1))
                            ),
                            "n_resistant_label_ge_2": int(
                                np.count_nonzero(res_inb & (res_labels >= 2))
                            ),
                            "n_resistant_ineligible_clone": int(
                                np.count_nonzero(tracked_res_keep & ~res_keep)
                            ),
                            "n_eligible_clone_labels_present": int(
                                np.unique(res_labels[res_keep]).size
                            ),
                            "resistant_retained_fraction": (
                                float(np.count_nonzero(res_keep) / len(resistant_raw))
                                if len(resistant_raw) else np.nan
                            ),
                            "n_sensitive_raw": int(len(sensitive_raw)),
                            "n_sensitive_retained": int(np.count_nonzero(sus_keep)),
                            "n_sensitive_nonfinite": int(np.count_nonzero(~sus_finite)),
                            "n_sensitive_out_of_bounds": int(
                                np.count_nonzero(sus_finite & ~sus_inb)
                            ),
                            "n_sensitive_outside_clean_mask": int(
                                np.count_nonzero(sus_inb & ~sus_keep)
                            ),
                            "sensitive_retained_fraction": (
                                float(np.count_nonzero(sus_keep) / len(sensitive_raw))
                                if len(sensitive_raw) else np.nan
                            ),
                            "resistant_retained_membership_pct": (
                                100.0 if np.any(res_keep) else np.nan
                            ),
                            "sensitive_retained_membership_pct": (
                                100.0 if np.any(sus_keep) else np.nan
                            ),
                            "sensitive_cleanup": True,
                        }
                        if (
                            row["n_resistant_retained"]
                            + row["n_resistant_ineligible_clone"]
                            != row["n_resistant_label_ge_2"]
                        ):
                            raise AssertionError("Resistant clone-eligibility QC failed")
                        if row["n_sensitive_retained"] != len(sensitive_retained):
                            raise AssertionError("Sensitive membership QC failed")
                        daily_rows.append(row)

                        result["n_resistant_raw"] += row["n_resistant_raw"]
                        result["n_resistant_retained"] += row["n_resistant_retained"]
                        result["n_sensitive_raw"] += row["n_sensitive_raw"]
                        result["n_sensitive_retained"] += row["n_sensitive_retained"]

                        if frame_idx in visual_frames:
                            write_visual_qc(
                                rep=rep,
                                frame_idx=frame_idx,
                                day=day,
                                clone_mask=clone_mask,
                                clean_sensitive_mask=clean_mask,
                                resistant_xy=resistant_raw,
                                sensitive_xy=sensitive_raw,
                                resistant_keep=res_keep,
                                sensitive_keep=sus_keep,
                                qc_root=qc_root,
                            )

                    metadata = pd.DataFrame(
                        [
                            {
                                "plate": rep.plate,
                                "sample": rep.sample,
                                "strategy": rep.strategy,
                                "raw_h5_path": str(rep.raw_h5_path),
                                "clone_mask_path": str(rep.clone_mask_path),
                                "sensitive_clean_mask_path": str(rep.clean_sensitive_path),
                                "preparation_version": PREPARATION_VERSION,
                                "resistant_filter": RESISTANT_FILTER_DESCRIPTION,
                                "minimum_terminal_clone_cells": MIN_TERMINAL_CLONE_CELLS,
                                "clone_eligibility_rule": (
                                    "retain clone iff its final actual mapped appearance "
                                    "contains at least minimum_terminal_clone_cells"
                                ),
                                "sensitive_filter": "inside_clean_sensitive_mask",
                                "coordinate_rule": "round_x_y_to_nearest_pixel",
                                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        ]
                    )
                    output_store.put("/metadata/run", metadata, format="fixed")
                    output_store.put(
                        "/qc/daily", pd.DataFrame(daily_rows), format="fixed"
                    )
                    output_store.put(
                        "/qc/clone_eligibility", clone_eligibility, format="fixed"
                    )

            validate_prepared_h5(temporary_path, days)
            if rep.output_path.exists() and not overwrite and not replacing_incomplete:
                raise FileExistsError(
                    f"Output appeared during processing: {rep.output_path}"
                )
            os.replace(temporary_path, rep.output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        result["elapsed_s"] = float(time.perf_counter() - started)
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = float(time.perf_counter() - started)
        return result


def write_batch_summary(rows: list[dict[str, object]], output_path: Path) -> Path:
    new_table = pd.DataFrame(rows)
    if output_path.is_file():
        existing = pd.read_csv(os.fspath(output_path))
        table = pd.concat([existing, new_table], ignore_index=True)
        if {"plate", "sample"}.issubset(table.columns):
            table = table.drop_duplicates(["plate", "sample"], keep="last")
    else:
        table = new_table
    if len(table) and {"plate", "sample"}.issubset(table.columns):
        table = table.sort_values(["plate", "sample"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    table.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    return output_path


def run_all(
    plates: Optional[list[str]],
    samples: Optional[list[str]],
    workers: int,
    overwrite: bool,
    visual_qc: bool,
) -> Path:
    target_plates = plates or PLATES
    reps = discover_replicates(target_plates, samples=samples)
    reps = select_visual_qc_replicates(reps, enabled=visual_qc)
    selected_qc = [f"{rep.plate}/{rep.sample}" for rep in reps if rep.visual_qc]
    qc_root = ESCAPE_ROOT / VISUAL_QC_SUBDIR
    print(
        f"[info] targets={len(reps)}, plates={target_plates}, workers={workers}, "
        f"overwrite={overwrite}, QC={visual_qc}"
    )
    if selected_qc:
        print(f"[info] representative visual QC wells: {selected_qc}")

    rows: list[dict[str, object]] = []
    if workers <= 1:
        iterator = reps
        if tqdm is not None:
            iterator = tqdm(reps, total=len(reps), desc="Prepare escape inputs", unit="rep")
        for rep in iterator:
            row = prepare_replicate(rep, overwrite=overwrite, qc_root=qc_root)
            rows.append(row)
            print(
                f"[rep] {rep.plate}/{rep.sample} status={row['status']} "
                f"days={row['n_days']} elapsed={float(row['elapsed_s']):.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(prepare_replicate, rep, overwrite, qc_root): rep
                for rep in reps
            }
            completed = as_completed(future_map)
            if tqdm is not None:
                completed = tqdm(
                    completed, total=len(future_map), desc="Prepare escape inputs", unit="rep"
                )
            for future in completed:
                rep = future_map[future]
                row = future.result()
                rows.append(row)
                print(
                    f"[rep] {rep.plate}/{rep.sample} status={row['status']} "
                    f"days={row['n_days']} elapsed={float(row['elapsed_s']):.1f}s"
                )

    summary_path = write_batch_summary(rows, ESCAPE_ROOT / SUMMARY_CSV)
    status_counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    print(f"[save] batch summary: {summary_path}")
    print(f"[done] status counts: {status_counts}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare mask-filtered experimental point clouds for escape analysis."
    )
    parser.add_argument(
        "--plates", nargs="+", choices=PLATES, default=None,
        help="Optional plate subset; omit to process all eight plates.",
    )
    parser.add_argument(
        "--samples", nargs="+", default=None,
        help="Optional exact sample-name subset.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                        help="Replace existing prepared outputs.")
    parser.add_argument(
        "--qc", action=argparse.BooleanOptionalAction, default=QC,
        help="Write representative visual QC overlays (default follows QC constant).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_all(
        plates=arguments.plates,
        samples=arguments.samples,
        workers=max(1, int(arguments.workers)),
        overwrite=bool(arguments.overwrite),
        visual_qc=bool(arguments.qc),
    )
