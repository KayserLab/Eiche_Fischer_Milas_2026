"""
TL;DR
-----
This script builds the Figure 2h local-density analysis tables around tracked
resistant clones. It is the main source of ring-density and sector-density data
used by Figure 2, Figure 3, and part of Figure 4.

Detailed
--------
Inputs:
- Clone mask stacks, total-density stacks, and raw/derived clone summaries for
  each replicate.

Outputs:
- Figure 2h analysis CSVs with clone-level and timepoint-level local density
  summaries, now including both timepoint index and real experimental day.
- Optional overlay and sector-map TIFF outputs for debugging or inspection.

How it works:
- Loads the required clone and density stacks without fallback to alternate
  file patterns.
- Isolates each clone over time and constructs a ring around the clone
  boundary at a fixed radius/thickness.
- Measures local density around that ring, including sector-resolved summaries
  and clone-centered overlays.
- Writes stable CSV outputs reused by several downstream figure scripts,
  preserving the real day associated with each timepoint.

Pipeline role:
- Shared density-context analysis stage for Figure 2h, Figure 3 density plots,
  and the Figure 4 TTP-versus-density branch.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import tifffile as tiff
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    binary_dilation,
    binary_fill_holes,
)
import pandas as pd
import csv

# =========================
# Basic paths & dataset
# =========================

# Root of your dataset
BASE_DIR: str = r"D:\2025_08_CoreSeed_Large_Testdataset"

# Plates to process (order matters only for logging)
PLATES = [f"Plate_{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8]]

# Subfolder names (inside each plate folder)
DENSITY_DIR_NAME: str   = "density_stacks_epanechnikov_poisson"          # total & (optionally) resistant density live here
CHANGES_DIR_NAME: str   = "density_changes_stacks_epanechnikov_poisson"  # precomputed Δres stacks live here
LABELS_DIR_NAME: str    = "clone_masks_stacks"    # ONLY this folder; no fallbacks

# Output roots (to be created later)
VIS_ROOT: str       = r"Figures\Figure_2\Visualization\2h_debug"   # under BASE_DIR
CSV_OUT_DIR: str    = r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data"
PLOTS_OUT_ROOT: str = r"Figures\Figure_2"                          # under BASE_DIR

# =========================
# Debug / filtering
# =========================

MAX_TIMEPOINTS: int | None = None   # e.g. 3 to use only first 3 paired timepoints; None = all

# Choose the analysis mode before running the script in PyCharm:
# - RUN_MODE = "resume"
#   -> interrupted-run continuation: calculate only clone/change blocks that are
#      not yet present in the output CSV.
# - RUN_MODE = "repair_partial"
#   -> partial-write repair: recalculate clone/change blocks, but append only
#      missing timepoint/sector rows that are not already present in the CSV.
# - RUN_MODE = "recalculate_all"
#   -> full rerun: move the existing CSV to a backup file and calculate every
#      clone/change block again into a fresh CSV.
RUN_MODE: str = "recalculate_all"

# If True, only process EXAMPLES; if False, process all detected samples
DEBUG_EXAMPLES_ONLY: bool = False

# Optional strategy filter for full-scan mode.
# Set to None (or an empty set) to process all strategies.
INCLUDE_STRATS: set[str] | None = None   # e.g., {"AT50", "AT100"} to restrict

# Examples to process when DEBUG_EXAMPLES_ONLY=True (strict: must exist or will be skipped)
EXAMPLES = {
    "NT":    {"plate": "Plate_4", "sample": "C1_NT-C1",    "clone_label": 3},
    "CT":    {"plate": "Plate_3", "sample": "B2_CT-B2",    "clone_label": 6},
    "AT50":  {"plate": "Plate_3", "sample": "C1_AT50-C1",  "clone_label": 2},
    "AT100": {"plate": "Plate_1", "sample": "B2_AT100-B2", "clone_label": 2},    # Add more for debugging if needed
}

# =========================
# Stack usage & alignment
# =========================

# Use ONLY precomputed change stacks in CHANGES_DIR_NAME (strict). If False, expect resistant density in DENSITY_DIR_NAME.
USE_PRECOMPUTED_CHANGES: bool = True

# Which resistant changes to write. Default resume skips existing
# (plate, sample, clone_label, change_type) blocks; --repair-partial can append
# only missing timepoint/sector rows inside a partially written block.
CHANGE_KINDS: tuple[str, ...] = ("abs", "rel")  # options: "abs", "rel"
DENSITY_VALUE_AREA_UM2: float = 19000.0
TOTAL_DENSITY_COL: str = "Mean_Total_Density_cells_per_um2"
RESISTANT_CHANGE_COL: str = "Mean_Resistant_Density_Change"

# Trailing blank-frame trimming for total density stacks
TRIM_PERCENTILE: float = 99.0
TRIM_THRESHOLD: float  = 1e-6  # p99 <= threshold → consider frame blank and drop from end

# Time pairing: analyze total density at t with resistant-density change at t+1
PAIR_DENS_T_WITH_CHANGE_TPLUS1: bool = True

# How many last frames of the total-density stack to print p99 diagnostics for
PRINT_LAST_K_FRAMES: int = 3   # set to 0 to disable the tail p99 printout

# =========================
# Sectorization & geometry
# =========================

CENTER_METHOD: str   = "clone_centroid"  # strict: sectors radiate from the clone centroid
N_SECTORS: int       = 12
RING_THICKNESS_PX: int = 40
RING_METHOD: str     = "distance"        # "distance" only; no fallback morphology
MIN_PIXELS_PER_SECTOR: int = 5
MIN_CLONE_AREA_PX: int     = 10

# =========================
# Visualization toggles
# =========================

SAVE_LOADED_STACKS: bool      = False
SAVE_CLONE_LABELS_STACK: bool = False
SAVE_OVERLAYS: bool           = False
SAVE_SECTOR_MAPS: bool        = False

VIZ_FILENAMES = {
    "total_density":      "01_total_density_stack.tif",
    "res_change_abs":     "02_res_change_abs_stack.tif",
    "res_change_rel":     "03_res_change_rel_stack.tif",
    "clone_labels":       "04_clone_labels_stack.tif",
    "overlay_boundaries": "05_overlay_total_boundaries.tif",
    "sectors_total":      "06_sectors_colored_by_total_density.tif",
    "sectors_change":     "07_sectors_colored_by_res_change.tif",
    "sector_index": "08_sector_index_stack.tif",
    "ring_mask": "09_ring_mask_stack.tif",
}

# =========================
# Output tabular names
# =========================

CSV_PER_SECTOR_TIME: str = (
    f"Figure2h_ring_change_vs_ring_total_density_timeseries_epanechnikov_{RING_THICKNESS_PX}pxboundary.csv"
)

# =========================
# Exact filenames (STRICT)
# =========================
# We will only look for these exact names (no wildcards, no alternates).

FNAME_TOTAL_DENSITY     = "{sample}_total_density_epanechnikov.tif"        # in DENSITY_DIR_NAME
FNAME_RESISTANT_DENSITY = "{sample}_res_density_epanechnikov.tif"          # in DENSITY_DIR_NAME (only if USE_PRECOMPUTED_CHANGES is False)
FNAME_RES_CHANGE_ABS    = "{sample}_res_density_epanechnikov_abs_diff.tif" # in CHANGES_DIR_NAME
FNAME_RES_CHANGE_REL    = "{sample}_res_density_epanechnikov_rel_diff.tif" # in CHANGES_DIR_NAME
FNAME_CLONE_LABELS      = "{sample}_res_clone_masks.tif"      # in LABELS_DIR_NAME (no contour/label fallbacks)

# =========================
# Sample/strategy parsing
# =========================

SAMPLE_STRATEGY_SPLIT_LEFT: str  = "_"   # "B2_CT-B2" -> split once on "_"
SAMPLE_STRATEGY_SPLIT_RIGHT: str = "-"   # then split once on "-"

def normalize_run_mode(run_mode: str) -> str:
    valid_modes = {"resume", "repair_partial", "recalculate_all"}
    mode = str(run_mode).strip().lower().replace("-", "_")
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported RUN_MODE={run_mode!r}; choose one of {sorted(valid_modes)}."
        )
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Figure 2h local-density CSV. By default, existing "
            "clone/change blocks in the output CSV are skipped according to "
            "the editable RUN_MODE setting."
        )
    )
    parser.add_argument(
        "--recalculate-all",
        action="store_true",
        help=(
            "Ignore the existing output CSV and calculate all clone/change "
            "blocks again. The existing CSV is moved to a backup path first."
        ),
    )
    parser.add_argument(
        "--repair-partial",
        action="store_true",
        help=(
            "Recompute blocks even when they already have rows in the CSV, "
            "then append only missing timepoint/sector rows. Use this if the "
            "script was interrupted during CSV writing."
        ),
    )
    args = parser.parse_args()
    if args.recalculate_all and args.repair_partial:
        parser.error("--recalculate-all and --repair-partial are mutually exclusive.")

    run_mode = normalize_run_mode(RUN_MODE)
    if args.recalculate_all:
        run_mode = "recalculate_all"
    elif args.repair_partial:
        run_mode = "repair_partial"

    args.run_mode = run_mode
    args.recalculate_all = run_mode == "recalculate_all"
    args.repair_partial = run_mode == "repair_partial"
    return args


def backup_existing_csv_for_recalculate_all(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    backup_path = csv_path.with_name(
        f"{csv_path.stem}.backup_before_recalculate_all{csv_path.suffix}"
    )
    i = 1
    while backup_path.exists():
        backup_path = csv_path.with_name(
            f"{csv_path.stem}.backup_before_recalculate_all_{i}{csv_path.suffix}"
        )
        i += 1

    csv_path.replace(backup_path)
    print(f"[checkpoint] --recalculate-all active; moved existing CSV to: {backup_path}")


def make_row_key(
    plate: str,
    sample: str,
    clone_label: int,
    change_type: str,
    timepoint: int,
    sector: int,
) -> tuple[str, str, int, str, int, int]:
    return (
        str(plate),
        str(sample),
        int(clone_label),
        str(change_type),
        int(timepoint),
        int(sector),
    )


def make_row_key_from_dict(row: dict) -> tuple[str, str, int, str, int, int]:
    return make_row_key(
        row["plate"],
        row["sample"],
        int(row["clone_label"]),
        row["change_type"],
        int(row["timepoint"]),
        int(row["sector"]),
    )


def load_existing_output_keys_from_csv(
    csv_path: Path,
) -> tuple[set[tuple[str, str, int, str]], set[tuple[str, str, int, str, int, int]]]:
    if not csv_path.exists():
        return set(), set()

    try:
        df = pd.read_csv(
            csv_path,
            usecols=["plate", "sample", "clone_label", "change_type", "timepoint", "sector"],
        )
    except Exception as e:
        print(f"[checkpoint] Could not read existing CSV for checkpointing: {csv_path} ({e})")
        return set(), set()

    block_keys: set[tuple[str, str, int, str]] = set()
    row_keys: set[tuple[str, str, int, str, int, int]] = set()
    for r in df.itertuples(index=False):
        try:
            block_keys.add((str(r.plate), str(r.sample), int(r.clone_label), str(r.change_type)))
            row_keys.add(
                make_row_key(
                    str(r.plate),
                    str(r.sample),
                    int(r.clone_label),
                    str(r.change_type),
                    int(r.timepoint),
                    int(r.sector),
                )
            )
        except Exception:
            continue

    print(
        f"[checkpoint] loaded {len(block_keys)} done block key(s) "
        f"and {len(row_keys)} done row key(s) from CSV."
    )
    return block_keys, row_keys


def ensure_existing_csv_schema(csv_path: Path, fieldnames: list[str]) -> None:
    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration:
            return

    if header != fieldnames:
        header_set = set(header)
        field_set = set(fieldnames)
        if not header_set.issubset(field_set):
            raise ValueError(
                "Existing output CSV schema cannot be upgraded safely.\n"
                f"CSV: {csv_path}\n"
                f"Existing header: {header}\n"
                f"Expected header: {fieldnames}"
            )

        tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with open(csv_path, "r", newline="") as src, open(tmp_path, "w", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                out = {name: row.get(name, "") for name in fieldnames}
                if "day" in fieldnames and not out.get("day"):
                    try:
                        out["day"] = int(timepoint_to_day(int(float(row.get("timepoint", "")))))
                    except Exception:
                        out["day"] = ""
                writer.writerow(out)
        tmp_path.replace(csv_path)

def append_rows_to_csv_incremental(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    ensure_existing_csv_schema(csv_path, fieldnames)

    with open(csv_path, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)
        fp.flush()

def make_done_key(plate: str, sample: str, clone_label: int, change_type: str) -> tuple[str, str, int, str]:
    return (str(plate), str(sample), int(clone_label), str(change_type))


def ring_sector_means_from_ring_pixels(
    ring2d: np.ndarray,
    cy: float,
    cx: float,
    frame_values: np.ndarray,
    n_sectors: int,
    min_pixels: int,
) -> np.ndarray:
    means = np.full(n_sectors, np.nan, dtype=np.float32)

    if not np.any(ring2d) or np.isnan(cy) or np.isnan(cx):
        return means

    ys, xs = np.nonzero(ring2d)
    if ys.size == 0:
        return means

    dy = ys.astype(np.float32) - np.float32(cy)
    dx = xs.astype(np.float32) - np.float32(cx)

    ang = np.arctan2(dy, dx)
    ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)

    sid = np.floor(ang / (2.0 * np.pi / n_sectors)).astype(np.int16, copy=False)
    vals = frame_values[ys, xs].astype(np.float32, copy=False)

    sums = np.bincount(sid, weights=vals, minlength=n_sectors).astype(np.float32, copy=False)
    cnts = np.bincount(sid, minlength=n_sectors).astype(np.int32, copy=False)

    ok = cnts >= int(min_pixels)
    means[ok] = sums[ok] / cnts[ok].astype(np.float32, copy=False)
    return means

def p99(frame: np.ndarray) -> float:
    return float(np.percentile(frame.astype(np.float64), TRIM_PERCENTILE))

def trim_trailing_blank_frames(total_stack: np.ndarray) -> Tuple[np.ndarray, int]:
    T = total_stack.shape[0]
    cut = T
    while cut > 0 and p99(total_stack[cut - 1]) <= TRIM_THRESHOLD:
        cut -= 1
    n_trimmed = T - cut
    return total_stack[:cut], n_trimmed

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def sample_strategy(sample: str) -> str:
    left = sample.split(SAMPLE_STRATEGY_SPLIT_LEFT, 1)
    if len(left) < 2:
        return "UNK"
    right = left[1].split(SAMPLE_STRATEGY_SPLIT_RIGHT, 1)
    return right[0]


def timepoint_to_day(tp: int) -> int:
    """
    Map 1-based timepoint index to the real experimental day:
    1, 3, 5, 8, 10, 12, ...
    """
    if tp < 1:
        raise ValueError(f"timepoint_to_day expects tp >= 1, got {tp}")
    day = 1
    for k in range(2, tp + 1):
        step = 2 if ((k - 2) % 3) in (0, 1) else 3
        day += step
    return day


def load_required_stacks(
    base: Path, plate: str, sample: str
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray]]:

    plate_dir   = base / plate
    dens_dir    = plate_dir / DENSITY_DIR_NAME
    chg_dir     = plate_dir / CHANGES_DIR_NAME
    labels_dir  = plate_dir / LABELS_DIR_NAME

    f_total = dens_dir / FNAME_TOTAL_DENSITY.format(sample=sample)
    f_abs   = chg_dir  / FNAME_RES_CHANGE_ABS.format(sample=sample)
    f_rel   = chg_dir  / FNAME_RES_CHANGE_REL.format(sample=sample)
    f_lbls  = labels_dir / FNAME_CLONE_LABELS.format(sample=sample)

    print(f"\n[load] plate={plate} sample={sample} strategy={sample_strategy(sample)}")
    print(f"  total_density:      {f_total}")
    print(f"  res_change_abs:     {f_abs}")
    print(f"  res_change_rel:     {f_rel}")
    print(f"  clone_labels:       {f_lbls}")

    if not f_total.exists() or not f_lbls.exists():
        print("  ⚠ Missing essential stacks (total or clone labels). Skipping.")
        return None, {}, None

    change_paths = {"abs": f_abs, "rel": f_rel}
    requested_change_paths = {
        kind: change_paths[kind]
        for kind in CHANGE_KINDS
        if kind in change_paths
    }
    if not requested_change_paths:
        print(f"  [WARN] No valid CHANGE_KINDS requested: {CHANGE_KINDS}. Skipping.")
        return None, {}, None

    missing_change_kinds = [
        kind for kind, path in requested_change_paths.items() if not path.exists()
    ]
    for kind in missing_change_kinds:
        print(f"  [WARN] Missing change stack for change_type='{kind}': {requested_change_paths[kind]}")

    requested_change_paths = {
        kind: path for kind, path in requested_change_paths.items() if path.exists()
    }
    if not requested_change_paths:
        print("  [WARN] No requested change stacks are available. Skipping.")
        return None, {}, None

    def _stat_line(p: Path, tag: str) -> None:
        try:
            st = p.stat()
            print(f"  [DEBUG] {tag}: size={st.st_size/1e6:.1f} MB  mtime={st.st_mtime:.0f}")
        except Exception as e:
            print(f"  [DEBUG] {tag}: stat failed: {e}")

    def debug_tiff(path: Path, tag: str) -> None:
        try:
            with tiff.TiffFile(str(path)) as tf:
                n_pages = len(tf.pages)
                n_series = len(tf.series)
                print(f"  [TIFF DEBUG] {tag}: is_bigtiff={tf.is_bigtiff} is_imagej={getattr(tf,'is_imagej',False)} "
                      f"pages={n_pages} series={n_series}")
                if n_pages > 0:
                    p0 = tf.pages[0]
                    print(f"    page0.shape={p0.shape} dtype={p0.dtype}")
                if n_series > 0:
                    s0 = tf.series[0]
                    print(f"    series0.shape={s0.shape} axes={s0.axes} dtype={s0.dtype}")
        except Exception as e:
            print(f"  [TIFF DEBUG] {tag}: failed to open: {e}")

    _stat_line(f_total, "TOTAL")
    for kind, path in requested_change_paths.items():
        _stat_line(path, f"CHANGE_{kind.upper()}")
    _stat_line(f_lbls, "LABELS")

    debug_tiff(f_total, "TOTAL")
    for kind, path in requested_change_paths.items():
        debug_tiff(path, f"CHANGE_{kind.upper()}")
    debug_tiff(f_lbls, "LABELS")

    try:
        total_stack  = tiff.imread(str(f_total))
        labels_stack = tiff.imread(str(f_lbls))
    except Exception as e:
        print(f"  [WARN] Error reading total or labels stack with imread(): {e}")
        return None, {}, None

    change_stacks: Dict[str, np.ndarray] = {}
    for kind, path in requested_change_paths.items():
        try:
            change_stacks[kind] = tiff.imread(str(path))
        except Exception as e:
            print(f"  [WARN] Error reading {kind} change stack with imread(): {e}")

    print(f"  [imread] total:  shape={getattr(total_stack,'shape',None)}  ndim={total_stack.ndim}  dtype={total_stack.dtype}")
    for kind, stack in change_stacks.items():
        print(f"  [imread] change_{kind}: shape={getattr(stack,'shape',None)} ndim={stack.ndim} dtype={stack.dtype}")
    print(f"  [imread] labels: shape={getattr(labels_stack,'shape',None)} ndim={labels_stack.ndim} dtype={labels_stack.dtype}")
    def diagnose_non3d(path: Path, tag: str) -> None:
        try:
            with tiff.TiffFile(str(path)) as tf:
                n_pages = len(tf.pages)
                n_series = len(tf.series)
                print(f"  ⚠ {tag} stack not 3D by imread(). Investigating with TiffFile...")
                print(f"    [TiffFile] pages={n_pages} series={n_series} "
                      f"series0.shape={tf.series[0].shape if n_series else None} "
                      f"axes={tf.series[0].axes if n_series else None}")
                if n_pages > 0:
                    a0 = tf.pages[0].asarray()
                    print(f"    [TiffFile] page0: shape={a0.shape} dtype={a0.dtype} min/max={np.nanmin(a0)}/{np.nanmax(a0)}")
                if n_pages > 1:
                    a1 = tf.pages[1].asarray()
                    print(f"    [TiffFile] page1: shape={a1.shape} dtype={a1.dtype} min/max={np.nanmin(a1)}/{np.nanmax(a1)}")
        except Exception as e:
            print(f"    [TiffFile] diagnose failed for {tag}: {e}")

    if total_stack.ndim != 3:
        diagnose_non3d(f_total, "TOTAL")
        print("  [WARN] Total stack is not 3D. Skipping.")
        return None, {}, None

    for kind in list(change_stacks):
        if change_stacks[kind].ndim != 3:
            diagnose_non3d(requested_change_paths[kind], f"CHANGE_{kind.upper()}")
            print(f"  [WARN] Skipping change_type='{kind}' because it is not a 3D stack.")
            del change_stacks[kind]
    if not change_stacks:
        print("  [WARN] No valid 3D change stacks remain. Skipping.")
        return None, {}, None

    if labels_stack.ndim != 3:
        diagnose_non3d(f_lbls, "LABELS")
        print("  [WARN] Labels stack is not 3D. Skipping.")
        return None, {}, None

    change_shapes = ", ".join(f"{kind}={stack.shape}" for kind, stack in change_stacks.items())
    print(f"  shapes (before trim): total={total_stack.shape}, changes=({change_shapes}), labels={labels_stack.shape}")

    T_tot = total_stack.shape[0]
    tail_k = min(PRINT_LAST_K_FRAMES, T_tot)
    if tail_k > 0:
        tail_p99 = [p99(total_stack[T_tot - 1 - i]) for i in range(tail_k)]
        tail_p99_str = ", ".join(f"{v:.3g}" for v in tail_p99)
        print(f"  total tail p99 (last {tail_k} frames, newest→oldest): [{tail_p99_str}]")

    trimmed_total, n_trimmed = trim_trailing_blank_frames(total_stack)
    if n_trimmed > 0:
        print(f"  ⨯ trimmed {n_trimmed} trailing blank frame(s) from total_density; new total T={trimmed_total.shape[0]}")
    else:
        print("  ✓ no trailing blank frames detected in total_density")

    if trimmed_total.shape[0] < 2:
        print("  ⚠ Not enough total frames after trimming to pair with changes. Skipping.")
        return None, {}, None

    print(f"  shapes (after trim):  total={trimmed_total.shape}, changes=({change_shapes}), labels={labels_stack.shape}")
    return trimmed_total, change_stacks, labels_stack


def save_visual_stacks(
    out_dir: Path,
    total_stack: Optional[np.ndarray],
    change_abs: Optional[np.ndarray],
    change_rel: Optional[np.ndarray],
    labels_stack: Optional[np.ndarray],
) -> None:
    ensure_dir(out_dir)

    def _save(name: str, arr: Optional[np.ndarray]):
        if arr is None:
            return
        tiff.imwrite(str(out_dir / name), arr, imagej=False)

    if SAVE_LOADED_STACKS:
        if total_stack is not None:
            _save(VIZ_FILENAMES["total_density"], total_stack)
        _save(VIZ_FILENAMES["res_change_abs"], change_abs)
        _save(VIZ_FILENAMES["res_change_rel"], change_rel)

    if SAVE_CLONE_LABELS_STACK and labels_stack is not None:
        _save(VIZ_FILENAMES["clone_labels"], labels_stack)

def isolate_clone_mask(labels_stack: np.ndarray, clone_label: int) -> np.ndarray:
    return (labels_stack == int(clone_label))

def frame_centroid(mask2d: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(ys.mean()), float(xs.mean())

def centroids_over_time(mask_stack: np.ndarray) -> np.ndarray:
    T = mask_stack.shape[0]
    cc = np.full((T, 2), np.nan, dtype=np.float64)
    last_ok = None
    for t in range(T):
        cy, cx = frame_centroid(mask_stack[t])
        if np.isnan(cy) or np.isnan(cx):
            if last_ok is not None:
                cc[t] = last_ok
        else:
            cc[t] = (cy, cx)
            last_ok = cc[t]
    return cc


def build_ring_mask_from_distance(clone_mask_stack: np.ndarray, labels_stack: np.ndarray, ring_thick: int) -> np.ndarray:
    """
    Build an annulus just OUTSIDE the clone boundary, of thickness=ring_thick px.

    ✅ Updated to match the "fill holes before boundary" convention used in your
    standard contour pipeline:

        m_filled = binary_fill_holes(m)

    Then we compute outside-distance on the FILLED mask:
        dist_out = distance_transform_edt(~m_filled)
        ring     = 0 < dist_out <= ring_thick

    Excludes other clones' pixels (unchanged in spirit).
    """
    T, H, W = clone_mask_stack.shape
    ring = np.zeros((T, H, W), dtype=bool)

    for t in range(T):
        # ensure boolean
        m = clone_mask_stack[t].astype(bool, copy=False)

        # ✅ hole-fill to align with your boundary convention
        m_filled = binary_fill_holes(m)

        # distance OUTSIDE the clone: distance to nearest clone pixel for background only
        dist_out = distance_transform_edt(~m_filled)
        outer = (dist_out > 0) & (dist_out <= int(ring_thick))

        # exclude other clones (anything labeled that is NOT this clone)
        others = (labels_stack[t] > 0) & (~m_filled)
        ring[t] = outer & (~others)

    return ring


def sector_index_map(ring_mask_2d: np.ndarray, cy: float, cx: float, n_sectors: int) -> np.ndarray:
    H, W = ring_mask_2d.shape
    sid2d = -np.ones((H, W), dtype=np.int16)

    if not np.any(ring_mask_2d) or np.isnan(cy) or np.isnan(cx):
        return sid2d

    ys, xs = np.nonzero(ring_mask_2d)
    dy = ys.astype(np.float32) - np.float32(cy)
    dx = xs.astype(np.float32) - np.float32(cx)

    ang = np.arctan2(dy, dx)
    ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)

    sid = np.floor(ang / (2.0 * np.pi / n_sectors)).astype(np.int16, copy=False)
    sid2d[ys, xs] = sid
    return sid2d


def overlay_boundaries_on_total(total_stack: np.ndarray,
                                clone_mask_stack: np.ndarray,
                                edge_value: float | None = None,
                                out_dtype: np.dtype | None = None) -> np.ndarray:
    """
    Overlay thin clone boundaries onto total density frames.

    Minor QC improvement:
    ✅ derive the edge from the hole-filled clone mask so holes don't create internal edges.
    """
    out = total_stack.astype(out_dtype, copy=True) if out_dtype is not None else total_stack.copy()
    T = out.shape[0]
    for t in range(T):
        m = clone_mask_stack[t].astype(bool, copy=False)
        if not np.any(m):
            continue

        m_filled = binary_fill_holes(m)
        edge = binary_dilation(m_filled) & (~binary_erosion(m_filled))

        ev = p99(total_stack[t]) if edge_value is None else edge_value
        if np.issubdtype(out.dtype, np.integer):
            ev = np.clip(ev, np.iinfo(out.dtype).min, np.iinfo(out.dtype).max).astype(out.dtype)
        else:
            ev = out.dtype.type(ev)
        out[t][edge] = ev
    return out


def write_overlays_streaming(total_stack: np.ndarray,
                             clone_mask_stack: np.ndarray,
                             out_path: Path) -> None:
    """
    Stream overlays frame-by-frame to disk.

    ✅ edge computed from hole-filled clone mask for consistency with above.
    """
    with tiff.TiffWriter(str(out_path), bigtiff=True) as tw:
        for t in range(total_stack.shape[0]):
            frame = total_stack[t].copy()
            m = clone_mask_stack[t].astype(bool, copy=False)
            if np.any(m):
                m_filled = binary_fill_holes(m)
                edge = binary_dilation(m_filled) & (~binary_erosion(m_filled))
                ev = p99(frame)
                if np.issubdtype(frame.dtype, np.integer):
                    ev = np.clip(ev, np.iinfo(frame.dtype).min, np.iinfo(frame.dtype).max).astype(frame.dtype)
                else:
                    ev = frame.dtype.type(ev)
                frame[edge] = ev
            tw.write(frame, contiguous=True)

def colorize_by_sector_mean_frame(sector_ids_2d: np.ndarray,
                                  sector_means_1d: np.ndarray,
                                  dtype_out=np.float32) -> np.ndarray:
    out = np.zeros(sector_ids_2d.shape, dtype=dtype_out)
    valid = sector_ids_2d >= 0
    if np.any(valid):
        idx = sector_ids_2d[valid]
        out[valid] = sector_means_1d[idx].astype(dtype_out, copy=False)
    return out

def write_sector_maps_streaming(
    per_frame_sector_total: np.ndarray,
    per_frame_sector_change: np.ndarray,
    ring_t: np.ndarray,
    centroids: np.ndarray,
    n_sectors: int,
    out_total_path: Path,
    out_change_path: Path,
    out_index_path: Path,
    out_ring_path: Path,
) -> None:
    T = ring_t.shape[0]
    with tiff.TiffWriter(str(out_total_path), bigtiff=True) as tw_total, \
         tiff.TiffWriter(str(out_change_path), bigtiff=True) as tw_change, \
         tiff.TiffWriter(str(out_index_path), bigtiff=True) as tw_index, \
         tiff.TiffWriter(str(out_ring_path), bigtiff=True) as tw_ring:

        for t in range(T):
            cy, cx = centroids[t]
            sid2d = sector_index_map(ring_t[t], cy, cx, n_sectors)

            tot_frame = colorize_by_sector_mean_frame(sid2d, per_frame_sector_total[t], dtype_out=np.float32)
            chg_frame = colorize_by_sector_mean_frame(sid2d, per_frame_sector_change[t], dtype_out=np.float32)

            idx_frame  = sid2d.astype(np.int16, copy=False)
            ring_frame = (ring_t[t].astype(np.uint8, copy=False) * 255)

            tw_total.write(tot_frame, contiguous=True)
            tw_change.write(chg_frame, contiguous=True)
            tw_index.write(idx_frame, contiguous=True)
            tw_ring.write(ring_frame, contiguous=True)


# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()
    base = Path(BASE_DIR)

    csv_dir = Path(CSV_OUT_DIR)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / CSV_PER_SECTOR_TIME

    FIELDNAMES = [
        "plate", "sample", "strategy", "clone_label", "sector",
        "change_type", "timepoint", "day",
        TOTAL_DENSITY_COL, RESISTANT_CHANGE_COL
    ]

    if args.recalculate_all:
        backup_existing_csv_for_recalculate_all(out_csv)
        done_keys: set[tuple[str, str, int, str]] = set()
        done_row_keys: set[tuple[str, str, int, str, int, int]] = set()
        print("[checkpoint] mode: recalculate_all; calculate all clone/change blocks from scratch.")
    else:
        ensure_existing_csv_schema(out_csv, FIELDNAMES)
        done_keys, done_row_keys = load_existing_output_keys_from_csv(out_csv)
        if args.repair_partial:
            print("[checkpoint] mode: repair_partial; append only missing rows.")
        else:
            print("[checkpoint] mode: resume; skip clone/change blocks already present in CSV.")

    if DEBUG_EXAMPLES_ONLY:
        work: Dict[str, Tuple[str, str, int]] = {}
        for key, meta in EXAMPLES.items():
            work[key] = (meta["plate"], meta["sample"], int(meta["clone_label"]))
        print(f"▶ DEBUG mode: {len(work)} example(s) → {list(work.keys())}")
    else:
        work = {}
        for plate in PLATES:
            dens_dir = base / plate / DENSITY_DIR_NAME
            if not dens_dir.exists():
                continue
            for p in dens_dir.glob("*_total_density_epanechnikov.tif"):
                sample = p.stem.replace("_total_density_epanechnikov", "")
                strat = sample_strategy(sample)
                if INCLUDE_STRATS and strat not in INCLUDE_STRATS:
                    continue
                work[f"{plate}:{sample}"] = (plate, sample, -1)
        print(f"▶ Full scan: {len(work)} sample(s) discovered.")

    for key, (plate, sample, clone_label) in work.items():

        vis_dir = base / VIS_ROOT / plate / sample
        ensure_dir(vis_dir)

        total_stack, change_stacks, labels_stack = load_required_stacks(base, plate, sample)
        if total_stack is None or labels_stack is None or not change_stacks:
            continue

        save_visual_stacks(
            vis_dir,
            total_stack=total_stack,
            change_abs=change_stacks.get("abs"),
            change_rel=change_stacks.get("rel"),
            labels_stack=labels_stack,
        )

        T_tot = total_stack.shape[0]
        T_chg = min(stack.shape[0] for stack in change_stacks.values())
        paired_T = min(T_tot - 1, T_chg)
        print(
            f"  pairing preview: total T={T_tot} -> usable pairs={paired_T} "
            f"(expects change planes T-1={T_tot-1}; changes={list(change_stacks)})"
        )

        if MAX_TIMEPOINTS is not None:
            paired_T = max(0, min(paired_T, int(MAX_TIMEPOINTS)))
        if paired_T <= 0:
            print("  [WARN] Nothing to pair after trimming/alignment. Skipping Step 2.")
            continue

        labels_t = labels_stack[:paired_T]
        labels_for_sample: list[int]
        if clone_label < 0:
            first_nonempty = None
            for k in range(labels_t.shape[0]):
                if np.any(labels_t[k] > 0):
                    first_nonempty = k
                    break
            if first_nonempty is None:
                print("  [WARN] No clone pixels found in paired window. Skipping Step 2.")
                continue

            all_labels = np.unique(labels_t[first_nonempty])
            labels_for_sample = [int(x) for x in all_labels if int(x) != 0]
            if not labels_for_sample:
                print("  [WARN] No clone labels discovered. Skipping Step 2.")
                continue

            print(f"  -> discovered {len(labels_for_sample)} clone(s): {labels_for_sample}")
        else:
            labels_for_sample = [int(clone_label)]

        total_t = total_stack[:paired_T]
        change_t_by_kind = {kind: stack[:paired_T] for kind, stack in change_stacks.items()}

        for clabel in labels_for_sample:
            pending_change_kinds = []
            for kind in CHANGE_KINDS:
                if kind not in change_t_by_kind:
                    continue
                done_key = make_done_key(plate, sample, int(clabel), kind)
                if not args.repair_partial and done_key in done_keys:
                    continue
                pending_change_kinds.append(kind)

            if not pending_change_kinds:
                print(f"  skip (already in CSV): plate={plate} sample={sample} clone={clabel} changes={list(change_t_by_kind)}")
                continue

            clone_mask_t = isolate_clone_mask(labels_t, clabel)

            areas = [int(clone_mask_t[t].sum()) for t in range(paired_T)]
            if max(areas) < MIN_CLONE_AREA_PX:
                print(
                    f"  [WARN] Clone {clabel} never exceeds MIN_CLONE_AREA_PX={MIN_CLONE_AREA_PX} px "
                    f"(max={max(areas)}). Skipping this clone."
                )
                continue

            cc = centroids_over_time(clone_mask_t)
            ring_t = build_ring_mask_from_distance(clone_mask_t, labels_t, RING_THICKNESS_PX)

            per_frame_sector_total = np.full((paired_T, N_SECTORS), np.nan, dtype=np.float32)
            for t in range(paired_T):
                cy, cx = cc[t]
                ring2d = ring_t[t]
                if not np.any(ring2d):
                    continue
                per_frame_sector_total[t] = ring_sector_means_from_ring_pixels(
                    ring2d=ring2d, cy=cy, cx=cx,
                    frame_values=total_t[t],
                    n_sectors=N_SECTORS,
                    min_pixels=MIN_PIXELS_PER_SECTOR,
                )

            valid_secs = np.isfinite(per_frame_sector_total).sum()
            print(f"  sectors with valid total means:  {valid_secs}/{paired_T*N_SECTORS}")

            per_frame_sector_change_by_kind: Dict[str, np.ndarray] = {}
            for change_kind in pending_change_kinds:
                change_t = change_t_by_kind[change_kind]
                per_frame_sector_change = np.full((paired_T, N_SECTORS), np.nan, dtype=np.float32)
                for t in range(paired_T):
                    cy, cx = cc[t]
                    ring2d = ring_t[t]
                    if not np.any(ring2d):
                        continue
                    per_frame_sector_change[t] = ring_sector_means_from_ring_pixels(
                        ring2d=ring2d, cy=cy, cx=cx,
                        frame_values=change_t[t],
                        n_sectors=N_SECTORS,
                        min_pixels=MIN_PIXELS_PER_SECTOR,
                    )
                per_frame_sector_change_by_kind[change_kind] = per_frame_sector_change
                valid_secs_ch = np.isfinite(per_frame_sector_change).sum()
                print(
                    f"  sectors with valid {change_kind} change means: "
                    f"{valid_secs_ch}/{paired_T*N_SECTORS}"
                )

            if SAVE_SECTOR_MAPS and per_frame_sector_change_by_kind:
                first_change_kind = pending_change_kinds[0]
                write_sector_maps_streaming(
                    per_frame_sector_total=per_frame_sector_total,
                    per_frame_sector_change=per_frame_sector_change_by_kind[first_change_kind],
                    ring_t=ring_t,
                    centroids=cc,
                    n_sectors=N_SECTORS,
                    out_total_path=vis_dir / VIZ_FILENAMES["sectors_total"],
                    out_change_path=vis_dir / VIZ_FILENAMES["sectors_change"],
                    out_index_path=vis_dir / VIZ_FILENAMES["sector_index"],
                    out_ring_path=vis_dir / VIZ_FILENAMES["ring_mask"],
                )

            if SAVE_OVERLAYS:
                write_overlays_streaming(
                    total_stack=total_t,
                    clone_mask_stack=clone_mask_t,
                    out_path=vis_dir / VIZ_FILENAMES["overlay_boundaries"],
                )

            for change_kind, per_frame_sector_change in per_frame_sector_change_by_kind.items():
                rows_this_clone = []
                for t in range(paired_T):
                    tp = int(t + 1)
                    day = int(timepoint_to_day(tp))
                    for s in range(N_SECTORS):
                        val_total = per_frame_sector_total[t, s]
                        val_change = per_frame_sector_change[t, s]
                        if np.isfinite(val_total) and np.isfinite(val_change):
                            total_density_cells_per_um2 = float(val_total) / DENSITY_VALUE_AREA_UM2
                            if change_kind == "abs":
                                resistant_change_value = float(val_change) / DENSITY_VALUE_AREA_UM2
                            else:
                                resistant_change_value = float(val_change)
                            rows_this_clone.append({
                                "plate": plate,
                                "sample": sample,
                                "strategy": sample_strategy(sample),
                                "clone_label": int(clabel),
                                "sector": int(s),
                                "change_type": change_kind,
                                "timepoint": tp,
                                "day": day,
                                TOTAL_DENSITY_COL: total_density_cells_per_um2,
                                RESISTANT_CHANGE_COL: resistant_change_value,
                            })
                if rows_this_clone:
                    if args.recalculate_all:
                        rows_to_append = rows_this_clone
                    else:
                        rows_to_append = [
                            row for row in rows_this_clone
                            if make_row_key_from_dict(row) not in done_row_keys
                        ]

                    skipped_existing = len(rows_this_clone) - len(rows_to_append)
                    if rows_to_append:
                        append_rows_to_csv_incremental(out_csv, rows_to_append, FIELDNAMES)
                        for row in rows_to_append:
                            done_row_keys.add(make_row_key_from_dict(row))
                        done_keys.add(make_done_key(plate, sample, int(clabel), change_kind))
                        if skipped_existing:
                            print(
                                f"  appended {len(rows_to_append)} missing row(s) for "
                                f"clone={clabel}, change={change_kind}; "
                                f"kept {skipped_existing} existing row(s)."
                            )
                    else:
                        print(
                            f"  skip rows (already complete in CSV): "
                            f"clone={clabel}, change={change_kind}"
                        )
                else:
                    print(
                        f"  [WARN] No valid sector rows for clone={clabel}, change={change_kind} "
                        "(nothing appended; will re-run next time)."
                    )
    print("\n[done] Finished Figure 2h analysis (incremental CSV + checkpointing).")

