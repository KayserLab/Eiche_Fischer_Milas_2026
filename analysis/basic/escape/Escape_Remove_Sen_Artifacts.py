"""
TL;DR
-----
This script cleans susceptible mask stacks by removing artifact-like regions
that would otherwise distort the escape analysis. It writes cached cleaned
susceptible masks so later scripts do not need to repeat the cleanup.

Detailed
--------
Inputs:
- Susceptible point clouds and/or mask-derived geometry for each replicate.
- Plate/sample selection and cleanup parameters.

Outputs:
- `intermediate_analysis_data/experimental/basic/escape/`
  `sensitive_clean_masks_stacks/<plate>/<sample>_sus_clean_masks.tif`
- A summary table describing processed replicates.

How it works:
- Discovers all susceptible replicate inputs for the requested plates.
- Reconstructs susceptible occupancy by day/frame.
- Applies morphological cleanup and artifact splitting logic to separate real
  colony structure from small tracking or segmentation artifacts.
- Writes one cleaned susceptible mask stack per replicate for reuse downstream.

Pipeline role:
- Quality-control preprocessing stage immediately upstream of
  `Prepare_Escape_Analysis_Inputs.py` (and the legacy escape-front writer).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label as cc_label
from skimage.morphology import closing, disk
from skimage.segmentation import expand_labels
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_paths import EXP_RAW_DATA, intermediate_dir  # noqa: E402
from analysis.basic.microscopy.mask_processing import fill_holes_2d


DATA_ROOT = os.fspath(EXP_RAW_DATA)
DEFAULT_OUT_DIR = os.fspath(intermediate_dir("experimental", "basic", "escape", create=False))

PLATES = [f"Plate_{i}" for i in range(1, 9)]
# Default for the PyCharm play button: two independent, contiguous plate lanes
# (Plate_1..Plate_4 and Plate_5..Plate_8). Set to 1 to run sequentially.
PLATE_WORKERS = 2
# PyCharm play-button setting. Change to True to regenerate every cleaned mask,
# including masks already present in ``sensitive_clean_masks_stacks``.
# The ``--overwrite`` command-line option also enables this behaviour.
OVERWRITE = False
FILE_SUFFIX = "_res_clone_masks.tif"
from reproduction.file_checks import tiff_layout, tiff_matches
SENS_OUT_SUBDIR = "sensitive_clean_masks_stacks"
SENS_OUT_SUFFIX = "_sus_clean_masks.tif"
SUMMARY_CSV = "Sensitive_Deartifacting_summary.csv"

SENS_SELEM_RAD = 3
SENS_LINK_DIST_PER_DAY = 150
IMAGING_DAY_OFFSETS = (1, 3, 5)

H5_DAY_RE = re.compile(r"^d_(\d+)$")
H5_SAMPLE_RE = re.compile(r"(.+)_total_poisson(?:_padded)?\.h5$", re.IGNORECASE)
WELL_SAMPLE_RE = re.compile(r"(?:^|_)([A-H]\d{1,2}_.+?-[A-H]\d{1,2})$", re.IGNORECASE)


@dataclass
class SensitiveReplicate:
    plate: str
    sample_name: str
    clone_mask_path: str
    h5_path: str
    out_path: str


def day_from_tidx(t_idx: int) -> int:
    return (t_idx // 3) * 7 + IMAGING_DAY_OFFSETS[t_idx % 3]


def sensitive_link_distance(previous_day: int, current_day: int) -> int:
    """Return the temporal-link radius at 150 pixels for each elapsed day."""
    day_gap = int(current_day) - int(previous_day)
    if day_gap <= 0:
        raise ValueError(f"Days must be chronological; got {previous_day} then {current_day}.")
    return SENS_LINK_DIST_PER_DAY * day_gap


def build_day_to_frame(n_frames: int) -> Dict[int, int]:
    return {day_from_tidx(t): t for t in range(int(n_frames))}


def parse_sample_from_h5(path: str) -> Optional[str]:
    name = os.path.basename(path)
    m = H5_SAMPLE_RE.match(name)
    if not m:
        return None
    raw = m.group(1)
    # H5 files often contain experiment prefix (e.g. G418_a2a4_A1_AT50-A1).
    # Clone masks are keyed only by well+strategy (e.g. A1_AT50-A1).
    m2 = WELL_SAMPLE_RE.search(raw)
    if m2:
        return m2.group(1)
    return raw


def discover_sensitive_replicates(
    plates: Iterable[str],
    clean_masks_root: str,
    samples: Optional[Iterable[str]] = None,
    skip_existing: bool = False,
) -> List[SensitiveReplicate]:
    """Find cleanup targets, optionally restricted to explicit sample names.

    When ``skip_existing`` is true, final cleaned-mask TIFFs are excluded here
    rather than being queued and later reported one by one as skipped.
    """
    requested_samples = set(samples) if samples else None
    reps: List[SensitiveReplicate] = []
    for plate in plates:
        plate_dir = os.path.join(DATA_ROOT, plate)
        h5_dir = os.path.join(plate_dir, "hdf5")
        mask_dir = os.path.join(plate_dir, "clone_masks_stacks")
        # Raw data is input-only. Keep generated masks in intermediate analysis data.
        out_dir = os.path.join(clean_masks_root, plate)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(h5_dir):
            continue

        selected_by_sample: Dict[str, str] = {}
        padded_hits = sorted(glob.glob(os.path.join(h5_dir, "*_total_poisson_padded.h5")))
        for hp in padded_hits:
            sample = parse_sample_from_h5(hp)
            if sample:
                selected_by_sample[sample] = hp

        legacy_hits = sorted(glob.glob(os.path.join(h5_dir, "*_total_poisson.h5")))
        for hp in legacy_hits:
            sample = parse_sample_from_h5(hp)
            if sample and sample not in selected_by_sample:
                selected_by_sample[sample] = hp

        for sample, h5_path in sorted(selected_by_sample.items()):
            if requested_samples is not None and sample not in requested_samples:
                continue
            clone_mask_path = os.path.join(mask_dir, f"{sample}{FILE_SUFFIX}")
            out_path = os.path.join(out_dir, f"{sample}{SENS_OUT_SUFFIX}")
            if skip_existing and tiff_matches(out_path, clone_mask_path, [h5_path]):
                continue
            if not os.path.exists(clone_mask_path):
                print(f"[warn] clone mask missing for parsed sample='{sample}' from h5='{os.path.basename(h5_path)}'")
            reps.append(
                SensitiveReplicate(
                    plate=plate,
                    sample_name=sample,
                    clone_mask_path=clone_mask_path,
                    h5_path=h5_path,
                    out_path=out_path,
                )
            )
    return reps


def extract_sus_days(store: pd.HDFStore) -> List[int]:
    out: List[int] = []
    for key in store.keys():
        parts = key.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "sus":
            continue
        m = H5_DAY_RE.match(parts[1])
        if not m:
            continue
        out.append(int(m.group(1)))
    out.sort()
    return out


def load_sus_xy(store: pd.HDFStore, day: int) -> np.ndarray:
    key = f"/sus/d_{int(day):02d}"
    df = store.get(key)
    return df[["x", "y"]].to_numpy(dtype=float)


def preprocess_binary(mask_2d: np.ndarray, selem_radius: int) -> np.ndarray:
    m = mask_2d.astype(bool, copy=False)
    m = closing(m, disk(int(selem_radius)))
    m = fill_holes_2d(m)
    return m.astype(bool)


def split_sensitive_points_by_artifact(
    sus_xy: np.ndarray,
    h: int,
    w: int,
    prev_assigned_lbl: Optional[np.ndarray],
    link_distance: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sus_xy.shape[0] == 0:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.zeros((h, w), dtype=np.uint16)

    xi = np.rint(sus_xy[:, 0]).astype(np.int32)
    yi = np.rint(sus_xy[:, 1]).astype(np.int32)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(inb):
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.zeros((h, w), dtype=np.uint16)

    xi_inb = xi[inb]
    yi_inb = yi[inb]

    if prev_assigned_lbl is not None and np.any(prev_assigned_lbl):
        if link_distance is None or link_distance <= 0:
            raise ValueError("A positive link_distance is required when a previous sensitive mask exists.")
        effective_link_distance = int(link_distance)
    else:
        effective_link_distance = 0

    margin = int(effective_link_distance + SENS_SELEM_RAD + 1)
    x0 = max(0, int(np.min(xi_inb)) - margin)
    x1 = min(w, int(np.max(xi_inb)) + margin + 1)
    y0 = max(0, int(np.min(yi_inb)) - margin)
    y1 = min(h, int(np.max(yi_inb)) + margin + 1)

    bh = y1 - y0
    bw = x1 - x0
    occ_local = np.zeros((bh, bw), dtype=bool)
    occ_local[yi_inb - y0, xi_inb - x0] = True
    sens_mask = preprocess_binary(occ_local, SENS_SELEM_RAD)
    raw_lbl = cc_label(sens_mask)
    if raw_lbl.max() == 0:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.zeros((h, w), dtype=np.uint16)

    curr_assigned_local = np.zeros((bh, bw), dtype=bool)
    if prev_assigned_lbl is None or not np.any(prev_assigned_lbl):
        curr_assigned_local = raw_lbl > 0
    else:
        prev_local = prev_assigned_lbl[y0:y1, x0:x1].astype(np.uint16, copy=False)
        dilated_prev = expand_labels(prev_local, distance=effective_link_distance)
        overlap_prev_ids = np.unique(raw_lbl[(prev_local > 0) & (raw_lbl > 0)])
        linked_prev_ids = np.unique(raw_lbl[(dilated_prev > 0) & (raw_lbl > 0)])
        keep_ids = np.union1d(overlap_prev_ids, linked_prev_ids)
        if keep_ids.size > 0:
            curr_assigned_local = np.isin(raw_lbl, keep_ids)

    keep = np.zeros(sus_xy.shape[0], dtype=bool)
    keep[inb] = curr_assigned_local[yi_inb - y0, xi_inb - x0]
    excl = inb & (~keep)

    kept_xy = sus_xy[keep]
    excl_xy = sus_xy[excl]
    curr_lbl = np.zeros((h, w), dtype=np.uint16)
    curr_lbl[y0:y1, x0:x1] = curr_assigned_local.astype(np.uint16)
    return kept_xy, excl_xy, curr_lbl


def process_sensitive_replicate(rep: SensitiveReplicate, overwrite: bool) -> dict:
    t_start = time.perf_counter()
    row = {
        "plate": rep.plate,
        "sample": rep.sample_name,
        "clone_mask_path": rep.clone_mask_path,
        "h5_path": rep.h5_path,
        "out_path": rep.out_path,
        "status": "ok",
        "n_frames": 0,
        "n_days_with_sus": 0,
        "n_nonempty_frames": 0,
        "mean_kept_points": np.nan,
        "mean_excluded_points": np.nan,
        "elapsed_s": np.nan,
        "error": "",
    }

    try:
        replacing_incomplete = os.path.exists(rep.out_path)
        if (not overwrite) and tiff_matches(rep.out_path, rep.clone_mask_path, [rep.h5_path]):
            row["status"] = "skipped_exists"
            row["elapsed_s"] = 0.0
            return row
        if not os.path.exists(rep.clone_mask_path):
            row["status"] = "missing_clone_mask"
            row["elapsed_s"] = 0.0
            return row
        if not os.path.exists(rep.h5_path):
            row["status"] = "missing_h5"
            row["elapsed_s"] = 0.0
            return row

        t_frames, h, w = tiff_layout(rep.clone_mask_path)
        row["n_frames"] = t_frames

        day_to_frame = build_day_to_frame(t_frames)
        kept_counts: List[int] = []
        excl_counts: List[int] = []
        nonempty_frames = 0
        prev_sens_assigned_lbl: Optional[np.ndarray] = None
        prev_sensitive_day: Optional[int] = None

        os.makedirs(os.path.dirname(rep.out_path), exist_ok=True)
        out_dir = os.path.dirname(rep.out_path)
        temporary_path = os.path.join(
            out_dir,
            f".{os.path.basename(rep.out_path)}.tmp-{os.getpid()}.tif",
        )
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        with pd.HDFStore(rep.h5_path, mode="r") as store:
            sus_days = set(extract_sus_days(store))
            row["n_days_with_sus"] = len(sus_days)

            # Write to a temporary TIFF first. This prevents an interrupted
            # process from leaving a final-looking but incomplete output file.
            with tifffile.TiffWriter(temporary_path, bigtiff=True) as tw:
                for day, _t_idx in sorted(day_to_frame.items()):
                    frame_mask = np.zeros((h, w), dtype=np.uint8)
                    if day in sus_days:
                        sus_xy = load_sus_xy(store, day)
                        link_distance = (
                            sensitive_link_distance(prev_sensitive_day, day)
                            if prev_sens_assigned_lbl is not None and np.any(prev_sens_assigned_lbl)
                            else None
                        )
                        kept_xy, excl_xy, prev_sens_assigned_lbl = split_sensitive_points_by_artifact(
                            sus_xy=sus_xy,
                            h=h,
                            w=w,
                            prev_assigned_lbl=prev_sens_assigned_lbl,
                            link_distance=link_distance,
                        )
                        prev_sensitive_day = day
                        kept_counts.append(int(kept_xy.shape[0]))
                        excl_counts.append(int(excl_xy.shape[0]))
                        if prev_sens_assigned_lbl is not None and np.any(prev_sens_assigned_lbl):
                            frame_mask = (prev_sens_assigned_lbl > 0).astype(np.uint8)
                            nonempty_frames += 1
                    tw.write(frame_mask, compression="deflate")

        if (not overwrite) and not replacing_incomplete and os.path.exists(rep.out_path):
            raise FileExistsError(f"Output appeared during processing: {rep.out_path}")
        os.replace(temporary_path, rep.out_path)

        row["n_nonempty_frames"] = int(nonempty_frames)
        if kept_counts:
            row["mean_kept_points"] = float(np.mean(kept_counts))
            row["mean_excluded_points"] = float(np.mean(excl_counts))
        row["elapsed_s"] = float(time.perf_counter() - t_start)
        return row
    except Exception as e:
        if 'temporary_path' in locals() and os.path.exists(temporary_path):
            os.remove(temporary_path)
        row["status"] = "error"
        row["error"] = str(e)
        row["elapsed_s"] = float(time.perf_counter() - t_start)
        return row


def process_plate_group(
    plates: List[str],
    overwrite: bool,
    clean_masks_root: str,
    samples: Optional[List[str]],
) -> List[dict]:
    """Process one contiguous group of plates sequentially in one process."""
    reps = discover_sensitive_replicates(
        plates,
        clean_masks_root=clean_masks_root,
        samples=samples,
        skip_existing=not overwrite,
    )
    lane_name = ", ".join(plates)
    print(f"[lane start] {lane_name}: {len(reps)} replicate(s)")
    rows: List[dict] = []
    for rep in reps:
        row = process_sensitive_replicate(rep, overwrite=overwrite)
        rows.append(row)
        print(
            f"[rep] {rep.plate}/{rep.sample_name} status={row['status']} "
            f"frames={row['n_frames']} elapsed={row['elapsed_s']:.1f}s"
        )
    print(f"[lane done] {lane_name}")
    return rows


def split_contiguous_plate_groups(plates: List[str], plate_workers: int) -> List[List[str]]:
    lane_count = min(max(1, int(plate_workers)), len(plates))
    base_size, remainder = divmod(len(plates), lane_count)
    groups: List[List[str]] = []
    start = 0
    for lane_index in range(lane_count):
        stop = start + base_size + (1 if lane_index < remainder else 0)
        groups.append(plates[start:stop])
        start = stop
    return groups


def run_all(
    out_dir: str,
    plate_workers: int,
    overwrite: bool,
    plates: Optional[List[str]],
    clean_masks_root: str,
    samples: Optional[List[str]] = None,
) -> str:
    target_plates = plates if plates else PLATES
    groups = split_contiguous_plate_groups(target_plates, plate_workers)
    print(
        f"[info] Plate-level lanes={len(groups)}, overwrite={overwrite}, "
        f"samples={samples or 'all'}, clean_masks_root={clean_masks_root}"
    )
    print(f"[info] Plate groups: {groups}")

    if len(groups) == 1:
        rows = process_plate_group(
            groups[0], overwrite=overwrite, clean_masks_root=clean_masks_root, samples=samples
        )
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(
                    process_plate_group, group, overwrite, clean_masks_root, samples
                ): group
                for group in groups
            }
            for future in as_completed(futures):
                group = futures[future]
                lane_rows = future.result()
                rows.extend(lane_rows)
                print(f"[lane collected] {group}: {len(lane_rows)} replicate(s)")

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["plate", "sample"]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, SUMMARY_CSV)
    df.to_csv(out_path, index=False)

    ok = int(np.count_nonzero(df["status"].astype(str) == "ok")) if len(df) else 0
    skipped = int(np.count_nonzero(df["status"].astype(str) == "skipped_exists")) if len(df) else 0
    err = int(np.count_nonzero(df["status"].astype(str) == "error")) if len(df) else 0
    print(f"[save] Sensitive summary CSV: {out_path} ({len(df)} rows; ok={ok}, skipped={skipped}, errors={err})")
    print("[done] Sensitive cleanup finished.")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute cleaned sensitive mask stacks.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for summary CSV")
    p.add_argument(
        "--clean-masks-root",
        default=os.path.join(DEFAULT_OUT_DIR, SENS_OUT_SUBDIR),
        help="Intermediate-data root for cleaned masks (one subdirectory per plate).",
    )
    p.add_argument(
        "--plate-workers",
        type=int,
        default=PLATE_WORKERS,
        help="Number of contiguous plate groups to run concurrently (default: 2).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help="Overwrite existing cleaned masks (also controlled by OVERWRITE).",
    )
    p.add_argument("--plates", nargs="*", default=None, help="Optional plate subset: Plate_1 Plate_2 ...")
    p.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample subset, e.g. D3_NT-D3; useful for smoke tests.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        out_dir=args.out_dir,
        plate_workers=max(1, int(args.plate_workers)),
        overwrite=bool(args.overwrite),
        plates=args.plates,
        clean_masks_root=args.clean_masks_root,
        samples=args.samples,
    )
