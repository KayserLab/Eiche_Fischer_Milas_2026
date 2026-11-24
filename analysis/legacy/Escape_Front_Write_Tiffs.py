"""
TL;DR
-----
This script defines the escape front for each replicate and writes the result
as mask stacks plus point/summary CSVs. It is the central bridge between clone
geometry and all later Figure 2 / Figure 4 escape timing analyses.

Detailed
--------
Inputs:
- Cleaned susceptible mask stacks from `Escape_Remove_Sen_Artifacts.py`.
- Resistant clone masks and resistant point clouds by day.
- Plate/sample discovery information and optional filter/truncation rules.

Outputs:
- Escape-front TIFF mask stacks.
- Per-timepoint escape-front point CSVs.
- Batch summary CSVs such as
  `escape_front_batch_summary_all_res_within_sens.csv`.

How it works:
- Identifies the dominant resistant clone in each replicate and frame.
- Compares resistant clone geometry against the surrounding susceptible region
  using a dot/radial reference construction.
- Marks resistant points as belonging to the escape front under a chosen escape
  definition, then rasterizes those points into masks and exports point tables.
- Records replicate-level summary information for downstream time-to-escape and
  escape-area analyses.

Pipeline role:
- Canonical escape-front generator used by both Figure 2 and Figure 4 branches.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "Arial"
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["axes.labelsize"] = 7
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from skimage.measure import label as cc_label, regionprops
from skimage.draw import disk as draw_disk, line as draw_line
from skimage.morphology import closing, disk
from skimage.segmentation import expand_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import experiment_filters
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DATA_ROOT = r"Y:\Members\MaxE\Publications\STEM Paper\Data\exp_raw_data"
MASK_SUFFIX = "_res_clone_masks.tif"
OUT_SUBDIR = "escape_front_masks_stacks"
SENS_CLEAN_SUBDIR = "sensitive_clean_masks_stacks"
SENS_CLEAN_SUFFIX = "_sus_clean_masks.tif"

DEFAULT_TEST_PLATE = "Plate_3"
DEFAULT_TEST_SAMPLE = "C5_AT50-C5"
DEFAULT_DEFINITION = "all_res_within_sens"
DEFAULT_T_START = None
DEFAULT_T_END = None
DEFAULT_WORKERS = 4
SUMMARY_DIR = Path(DATA_ROOT) / r"Figures\Escape_Quantification"

N_DOTS = 360
REF_DIAMETER_PX = 50_000.0
ESCAPE_DISK_RADIUS_PX = 10  # 10 um @ 1 px/um
POLYLINE_WIDTH_PX = 5       # 5 um @ 1 px/um
VIS_RING_RADIUS_PX = ESCAPE_DISK_RADIUS_PX * 5
VIS_RING_WIDTH_PX = 8
DEBUG_POINTS_PANEL = True
SENS_SELEM_RAD = 3
SENS_LINK_DIST = 300


@dataclass
class ReplicateIO:
    plate: str
    sample: str
    clone_mask_path: str
    h5_path: str


@dataclass
class RunResult:
    plate: str
    sample: str
    strategy: str
    status: str
    n_frames_total: int
    n_frames_processed: int
    first_escape_frame: int
    first_escape_day: int
    escape_detected: bool
    out_tif: str
    out_metrics_csv: str
    out_points_csv: str
    elapsed_s: float
    error: str


SAMPLE_STRAT_RE = re.compile(r"^[A-H]\d+_([^-]+)-[A-H]\d+$", re.IGNORECASE)


def day_from_tidx(t_idx: int) -> int:
    offsets = (1, 3, 5)  # Mon/Wed/Fri schedule
    return (t_idx // 3) * 7 + offsets[t_idx % 3]


def build_day_to_frame(n_frames: int) -> Dict[int, int]:
    return {day_from_tidx(t): t for t in range(n_frames)}


def extract_days_in_h5(h5_path: str) -> list[int]:
    days = []
    with h5py.File(h5_path, "r") as f:
        if "res" not in f or "sus" not in f:
            return days
        common = sorted(set(f["res"].keys()).intersection(set(f["sus"].keys())))
        for k in common:
            m = re.match(r"^d_(\d+)$", k)
            if m:
                days.append(int(m.group(1)))
    return days


def extract_res_days_in_h5(h5_path: str) -> list[int]:
    days = []
    with h5py.File(h5_path, "r") as f:
        if "res" not in f:
            return days
        for k in sorted(f["res"].keys()):
            m = re.match(r"^d_(\d+)$", k)
            if m:
                days.append(int(m.group(1)))
    return days


def load_day_positions(h5_path: str, day: int) -> tuple[np.ndarray, np.ndarray]:
    key = f"d_{day:02d}"
    res_df = pd.read_hdf(h5_path, key=f"/res/{key}")
    sus_df = pd.read_hdf(h5_path, key=f"/sus/{key}")
    res_xy = res_df[["x", "y"]].to_numpy(dtype=float)
    sus_xy = sus_df[["x", "y"]].to_numpy(dtype=float)
    return res_xy, sus_xy


def load_day_positions_from_store(store: pd.HDFStore, day: int) -> tuple[np.ndarray, np.ndarray]:
    key = f"d_{day:02d}"
    res_df = store.get(f"/res/{key}")
    sus_df = store.get(f"/sus/{key}")
    res_xy = res_df[["x", "y"]].to_numpy(dtype=float)
    sus_xy = sus_df[["x", "y"]].to_numpy(dtype=float)
    return res_xy, sus_xy


def load_res_positions_from_store(store: pd.HDFStore, day: int) -> np.ndarray:
    key = f"d_{day:02d}"
    res_df = store.get(f"/res/{key}")
    return res_df[["x", "y"]].to_numpy(dtype=float)


def assign_resistant_cells_to_clone_labels(res_xy: np.ndarray, clone_mask_2d: np.ndarray) -> np.ndarray:
    h, w = clone_mask_2d.shape
    xi = np.rint(res_xy[:, 0]).astype(int)
    yi = np.rint(res_xy[:, 1]).astype(int)
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.zeros(res_xy.shape[0], dtype=np.int32)
    labels[in_bounds] = clone_mask_2d[yi[in_bounds], xi[in_bounds]].astype(np.int32)
    return labels


def find_dominant_clone_label(mask_stack: np.ndarray) -> Optional[int]:
    if mask_stack.ndim != 3 or mask_stack.shape[0] == 0:
        return None
    n_frames = int(mask_stack.shape[0])
    last = mask_stack[-1]
    final_labels, final_counts = np.unique(last[last >= 2], return_counts=True)
    if final_labels.size == 0:
        return None
    final_size = {int(lbl): int(cnt) for lbl, cnt in zip(final_labels, final_counts)}

    first_seen: Dict[int, int] = {}
    for t in range(n_frames):
        labs = np.unique(mask_stack[t])
        labs = labs[labs >= 2]
        for lbl in labs:
            li = int(lbl)
            if li not in first_seen:
                first_seen[li] = t

    best_label = None
    best_key = None
    for lbl in final_labels:
        li = int(lbl)
        first_t = first_seen.get(li, n_frames - 1)
        duration = n_frames - first_t
        key = (duration, final_size.get(li, 0), -li)
        if best_key is None or key > best_key:
            best_key = key
            best_label = li
    return best_label


def make_reference_dots(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    cx = float(w) / 2.0
    cy = float(h) / 2.0
    radius = REF_DIAMETER_PX / 2.0
    angles_deg = np.arange(N_DOTS, dtype=float)
    theta = np.deg2rad(angles_deg)
    x = cx + radius * np.sin(theta)   # 0 deg at top, clockwise
    y = cy - radius * np.cos(theta)
    dots = np.column_stack((x, y))
    return angles_deg, dots


def rasterize_polyline(mask: np.ndarray, points_xy: np.ndarray, width_px: int) -> None:
    if points_xy.shape[0] < 2:
        return
    h, w = mask.shape
    n = points_xy.shape[0]
    half = max(0, int(width_px) // 2)

    for i in range(n):
        p0 = points_xy[i]
        p1 = points_xy[(i + 1) % n]
        rr, cc = draw_line(int(round(p0[1])), int(round(p0[0])), int(round(p1[1])), int(round(p1[0])))
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr = rr[valid]
        cc = cc[valid]
        for r, c in zip(rr, cc):
            r0 = max(0, r - half)
            r1 = min(h, r + half + 1)
            c0 = max(0, c - half)
            c1 = min(w, c + half + 1)
            mask[r0:r1, c0:c1] = True


def front_points_from_clone(clone_points: np.ndarray, h: int, w: int) -> np.ndarray:
    if clone_points.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    _, dots = make_reference_dots(h, w)
    tree = cKDTree(clone_points)
    _, idx = tree.query(dots, k=1)
    return clone_points[np.asarray(idx, dtype=int)]


def dedupe_points_by_pixel(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    seen = set()
    out = []
    for p in points_xy:
        k = (int(round(p[0])), int(round(p[1])))
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    if not out:
        return np.empty((0, 2), dtype=float)
    return np.asarray(out, dtype=float)


def escaped_front_points_legacy_nearest(
    clone_points: np.ndarray, sus_points: np.ndarray, h: int, w: int
) -> np.ndarray:
    """
    Keep only front points where confinement thickness equals zero:
    - 360 reference dots on big circle
    - nearest dominant-clone resistant point per dot => distance d
    - count sensitive points within dot-centered circle radius d
    - keep dot/point if count == 0
    """
    if clone_points.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    _, dots = make_reference_dots(h, w)
    res_tree = cKDTree(clone_points)
    dists, idx = res_tree.query(dots, k=1)
    dists = np.asarray(dists, dtype=float)
    idx = np.asarray(idx, dtype=int)

    if sus_points.shape[0] == 0:
        keep = np.ones(dots.shape[0], dtype=bool)
    else:
        sus_tree = cKDTree(sus_points)
        counts = sus_tree.query_ball_point(dots, r=dists, return_length=True)
        keep = np.asarray(counts, dtype=int) == 0

    if not np.any(keep):
        return np.empty((0, 2), dtype=float)

    pts = clone_points[idx[keep]]
    return dedupe_points_by_pixel(pts)


def escaped_front_points_all_res_within_sens(
    clone_points: np.ndarray, sus_points: np.ndarray, h: int, w: int
) -> np.ndarray:
    """
    New definition under fixed dot-centered confinement rule:
    For each dot, all resistant cells with d_res < d_nearest_sensitive are escaped.
    """
    if clone_points.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    if sus_points.shape[0] == 0:
        # No sensitive competition anywhere -> all resistant cells escaped.
        return dedupe_points_by_pixel(clone_points)

    _, dots = make_reference_dots(h, w)
    res_tree = cKDTree(clone_points)
    sus_tree = cKDTree(sus_points)

    d_sens, _ = sus_tree.query(dots, k=1)
    d_sens = np.asarray(d_sens, dtype=float)
    radii = np.maximum(0.0, d_sens - 1e-6)  # strict inequality d_res < d_sens

    # Query candidate resistant indices for all dots in one call.
    idx_lists = res_tree.query_ball_point(dots, r=radii)
    if idx_lists is None or len(idx_lists) == 0:
        return np.empty((0, 2), dtype=float)

    idx_iter = idx_lists.tolist() if isinstance(idx_lists, np.ndarray) else idx_lists
    idx_all = [int(i) for lst in idx_iter for i in lst]
    if not idx_all:
        return np.empty((0, 2), dtype=float)
    unique_idx = np.unique(np.asarray(idx_all, dtype=np.int64))
    pts = clone_points[unique_idx]
    return dedupe_points_by_pixel(pts)


def escaped_front_points(
    clone_points: np.ndarray,
    sus_points: np.ndarray,
    h: int,
    w: int,
    definition: str,
) -> np.ndarray:
    if definition == "legacy_nearest":
        return escaped_front_points_legacy_nearest(clone_points=clone_points, sus_points=sus_points, h=h, w=w)
    if definition == "all_res_within_sens":
        return escaped_front_points_all_res_within_sens(clone_points=clone_points, sus_points=sus_points, h=h, w=w)
    raise ValueError(f"Unknown definition: {definition}")


def build_escape_front_mask(
    front_points: np.ndarray,
    h: int,
    w: int,
    circle_radius_px: int = ESCAPE_DISK_RADIUS_PX,
    add_polyline: bool = True,
    polyline_width_px: int = POLYLINE_WIDTH_PX,
) -> np.ndarray:
    esc = np.zeros((h, w), dtype=bool)
    for x, y in front_points:
        rr, cc = draw_disk((float(y), float(x)), radius=int(circle_radius_px), shape=(h, w))
        esc[rr, cc] = True

    if add_polyline:
        rasterize_polyline(esc, front_points, width_px=int(polyline_width_px))
    return esc


def build_escape_front_ring_mask(
    front_points: np.ndarray,
    h: int,
    w: int,
    ring_radius_px: int = VIS_RING_RADIUS_PX,
    ring_width_px: int = VIS_RING_WIDTH_PX,
) -> np.ndarray:
    ring = np.zeros((h, w), dtype=bool)
    outer_r = int(max(1, ring_radius_px))
    inner_r = int(max(0, outer_r - max(1, int(ring_width_px))))
    for x, y in front_points:
        rr_o, cc_o = draw_disk((float(y), float(x)), radius=outer_r, shape=(h, w))
        ring[rr_o, cc_o] = True
        if inner_r > 0:
            rr_i, cc_i = draw_disk((float(y), float(x)), radius=inner_r, shape=(h, w))
            ring[rr_i, cc_i] = False
    return ring


def discover_h5_path(h5_dir: str, sample: str) -> str:
    hits_padded = sorted(glob.glob(os.path.join(h5_dir, f"*_{sample}_total_poisson_padded.h5")))
    if hits_padded:
        return hits_padded[0]
    hits_legacy = sorted(glob.glob(os.path.join(h5_dir, f"*_{sample}_total_poisson.h5")))
    if hits_legacy:
        return hits_legacy[0]
    raise FileNotFoundError(f"No padded/legacy poisson h5 for sample={sample} in {h5_dir}")


def build_test_replicate(plate: str, sample: str) -> ReplicateIO:
    plate_dir = os.path.join(DATA_ROOT, plate)
    mask_path = os.path.join(plate_dir, "clone_masks_stacks", f"{sample}{MASK_SUFFIX}")
    h5_path = discover_h5_path(os.path.join(plate_dir, "hdf5"), sample)
    return ReplicateIO(plate=plate, sample=sample, clone_mask_path=mask_path, h5_path=h5_path)


def parse_strategy_from_sample(sample: str) -> str:
    m = SAMPLE_STRAT_RE.match(str(sample))
    if not m:
        return "UNKNOWN"
    return m.group(1)


def sample_filter_rule(sample: str) -> tuple[str, float | None]:
    rules = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    if sample not in rules:
        return "keep", None
    v = rules[sample]
    if v is None:
        return "exclude", None
    try:
        return "truncate", float(v)
    except Exception:
        return "keep", None


def find_dominant_clone_label_from_tiff(mask_tif_path: str) -> Optional[int]:
    with tifffile.TiffFile(mask_tif_path) as tf:
        n_frames = len(tf.pages)
        if n_frames == 0:
            return None
        last = tf.pages[n_frames - 1].asarray()
        final_labels, final_counts = np.unique(last[last >= 2], return_counts=True)
        if final_labels.size == 0:
            return None
        final_size = {int(lbl): int(cnt) for lbl, cnt in zip(final_labels, final_counts)}

        first_seen: Dict[int, int] = {}
        for t in range(n_frames):
            frame = tf.pages[t].asarray()
            labs = np.unique(frame)
            labs = labs[labs >= 2]
            for lbl in labs:
                li = int(lbl)
                if li not in first_seen:
                    first_seen[li] = t

    best_label = None
    best_key = None
    for lbl in final_labels:
        li = int(lbl)
        first_t = first_seen.get(li, n_frames - 1)
        duration = n_frames - first_t
        key = (duration, final_size.get(li, 0), -li)
        if best_key is None or key > best_key:
            best_key = key
            best_label = li
    return best_label


def discover_all_replicates() -> List[ReplicateIO]:
    reps: List[ReplicateIO] = []
    data_root = Path(DATA_ROOT)
    plate_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("Plate_")])
    for plate_dir in plate_dirs:
        plate = plate_dir.name
        mask_dir = plate_dir / "clone_masks_stacks"
        h5_dir = plate_dir / "hdf5"
        if not mask_dir.is_dir() or not h5_dir.is_dir():
            continue
        for mask_path in sorted(mask_dir.glob(f"*{MASK_SUFFIX}")):
            sample = mask_path.name[: -len(MASK_SUFFIX)]
            status, _ = sample_filter_rule(sample)
            if status == "exclude":
                continue
            try:
                h5_path = discover_h5_path(str(h5_dir), sample)
            except FileNotFoundError:
                continue
            reps.append(
                ReplicateIO(
                    plate=plate,
                    sample=sample,
                    clone_mask_path=str(mask_path),
                    h5_path=str(h5_path),
                )
            )
    return reps


def get_sensitive_clean_path(plate: str, sample: str) -> str:
    return os.path.join(DATA_ROOT, plate, SENS_CLEAN_SUBDIR, f"{sample}{SENS_CLEAN_SUFFIX}")


def stack_points_from_mask(mask_2d: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask_2d.astype(bool, copy=False))
    if yy.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((xx.astype(float), yy.astype(float)))


def filter_points_by_mask(xy: np.ndarray, mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep raw cell coordinates whose rounded pixel lands inside a binary mask.
    Returns kept_xy, excluded_xy.
    """
    if xy.shape[0] == 0:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty
    h, w = mask_2d.shape
    xi = np.rint(xy[:, 0]).astype(np.int32)
    yi = np.rint(xy[:, 1]).astype(np.int32)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    keep = np.zeros(xy.shape[0], dtype=bool)
    if np.any(inb):
        keep[inb] = mask_2d[yi[inb], xi[inb]].astype(bool, copy=False)
    return xy[keep], xy[~keep]


def load_mask_stack(path: str) -> np.ndarray:
    """
    Robust TIFF stack loader.
    Some multi-page TIFFs are written as multiple series where tifffile.imread
    returns only the first 2D page. In that case, stack all pages explicitly.
    """
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        with tifffile.TiffFile(path) as tf:
            n_pages = len(tf.pages)
            if n_pages > 1:
                return tf.asarray(key=range(n_pages))
        return arr[None, ...]
    raise ValueError(f"Unexpected TIFF dimensionality for {path}: {arr.shape}")


def show_overlay(colony_like: np.ndarray, esc_mask: np.ndarray, title: str) -> None:
    raise RuntimeError("Old show_overlay signature should not be used.")


def build_point_mask(xy: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if xy.shape[0] == 0:
        return m
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    m[yi[inb], xi[inb]] = True
    return m


def preprocess_binary(mask_2d: np.ndarray, selem_radius: int) -> np.ndarray:
    m = mask_2d.astype(bool, copy=False)
    m = closing(m, disk(int(selem_radius)))
    m = binary_fill_holes(m)
    return m.astype(bool)


def split_sensitive_points_by_artifact(
    sus_xy: np.ndarray,
    h: int,
    w: int,
    prev_assigned_lbl: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirror single_clone_tracking artifact logic for sensitive points:
    - Build connected regions from preprocessed sensitive occupancy.
    - Keep regions that overlap previous assigned region OR overlap its dilated map.
    - Mark other regions as artifacts (excluded).

    Returns:
      kept_xy, excluded_xy, current_assigned_lbl
    """
    if sus_xy.shape[0] == 0:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.zeros((h, w), dtype=np.uint16)

    xi = np.rint(sus_xy[:, 0]).astype(int)
    yi = np.rint(sus_xy[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(inb):
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.zeros((h, w), dtype=np.uint16)

    xi_inb = xi[inb]
    yi_inb = yi[inb]

    # Local ROI around sensitive cells (+ margins) keeps logic identical but avoids full-image morphology.
    margin = int(SENS_LINK_DIST + SENS_SELEM_RAD + 1)
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
        # Seed step: keep all sensitive regions.
        curr_assigned_local = raw_lbl > 0
    else:
        prev_local = prev_assigned_lbl[y0:y1, x0:x1].astype(np.uint16, copy=False)
        dilated_prev = expand_labels(prev_local, distance=SENS_LINK_DIST)
        for reg in regionprops(raw_lbl):
            rmask = raw_lbl == reg.label
            overlap_prev = np.any(prev_local[rmask] > 0)
            linked_prev = np.any(dilated_prev[rmask] > 0)
            if overlap_prev or linked_prev:
                curr_assigned_local[rmask] = True

    keep = np.zeros(sus_xy.shape[0], dtype=bool)
    keep[inb] = curr_assigned_local[yi_inb - y0, xi_inb - x0]
    excl = inb & (~keep)

    kept_xy = sus_xy[keep]
    excl_xy = sus_xy[excl]
    curr_lbl = np.zeros((h, w), dtype=np.uint16)
    curr_lbl[y0:y1, x0:x1] = curr_assigned_local.astype(np.uint16)
    return kept_xy, excl_xy, curr_lbl


def show_overlay_debug(
    clone_mask: np.ndarray,
    sus_kept_xy: np.ndarray,
    sus_excluded_xy: np.ndarray,
    esc_mask: np.ndarray,
    front_points: Optional[np.ndarray],
    title: str,
) -> None:
    h, w = clone_mask.shape

    rgb = np.ones((h, w, 3), dtype=float)  # white background
    rgb[clone_mask] = np.array([1.0, 0.9, 0.2])   # yellow
    rgb[esc_mask] = np.array([1.0, 0.0, 0.0])      # red

    if DEBUG_POINTS_PANEL:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax0, ax1 = axs
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))
        ax1 = None

    ax0.imshow(rgb)
    if sus_excluded_xy is not None and sus_excluded_xy.shape[0] > 0:
        ax0.scatter(sus_excluded_xy[:, 0], sus_excluded_xy[:, 1], s=3, c="#8a8a8a", alpha=0.7, linewidths=0)
    if sus_kept_xy is not None and sus_kept_xy.shape[0] > 0:
        ax0.scatter(sus_kept_xy[:, 0], sus_kept_xy[:, 1], s=3, c="#2a55ff", alpha=0.7, linewidths=0)
    if front_points is not None and front_points.shape[0] > 0:
        ax0.scatter(
            front_points[:, 0],
            front_points[:, 1],
            s=36,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
        )
    ax0.set_title(title)
    ax0.axis("off")

    if ax1 is not None:
        # Dedicated panel: only escape-point positions for unambiguous debugging.
        base = np.zeros((h, w), dtype=float)
        base[clone_mask] = 0.2
        ax1.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
        if front_points is not None and front_points.shape[0] > 0:
            ax1.scatter(front_points[:, 0], front_points[:, 1], s=20, c="red", linewidths=0)
        nfp = 0 if front_points is None else int(front_points.shape[0])
        ax1.set_title(f"Escape points only (n={nfp})")
        ax1.axis("off")

    fig.tight_layout()
    plt.show()


def run_test(
    plate: str,
    sample: str,
    definition: str,
    visualize: bool = False,
    require_clean_sensitive: bool = True,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> RunResult:
    print(f"[start] run_test plate={plate}, sample={sample}, definition={definition}")
    t_run_start = time.perf_counter()
    status, cutoff_day = sample_filter_rule(sample)
    if status == "exclude":
        raise RuntimeError(f"Sample excluded by experiment_filters: {sample}")
    io = build_test_replicate(plate, sample)
    if not os.path.isfile(io.clone_mask_path):
        raise FileNotFoundError(io.clone_mask_path)
    with tifffile.TiffFile(io.clone_mask_path) as tf_mask:
        T = len(tf_mask.pages)
        if T == 0:
            raise RuntimeError(f"Empty clone mask stack: {io.clone_mask_path}")
        first_frame = tf_mask.pages[0].asarray()
        if first_frame.ndim != 2:
            raise ValueError(f"Unexpected frame shape in clone mask tif: {first_frame.shape}")
        H, W = int(first_frame.shape[0]), int(first_frame.shape[1])

    dominant = find_dominant_clone_label_from_tiff(io.clone_mask_path)
    if dominant is None:
        raise RuntimeError(f"No dominant clone found in final frame for {plate}/{sample}")
    print(f"[info] dominant clone label: {dominant}")
    print(f"[info] h5 source: {io.h5_path}")
    print(f"[info] definition: {definition}")

    day_to_frame = build_day_to_frame(T)
    days_h5 = set(extract_res_days_in_h5(io.h5_path))

    clean_sens_path = get_sensitive_clean_path(plate=plate, sample=sample)
    use_clean_sensitive = os.path.isfile(clean_sens_path)
    if use_clean_sensitive:
        with tifffile.TiffFile(clean_sens_path) as tf_sens_check:
            if len(tf_sens_check.pages) != T:
                raise ValueError(
                    f"Sensitive clean stack frame count mismatch: got {len(tf_sens_check.pages)}, expected {T}"
                )
        print(f"[info] using precomputed sensitive masks: {clean_sens_path}")
    else:
        msg = f"Precomputed sensitive masks missing: {clean_sens_path}"
        if require_clean_sensitive:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}; fallback to on-the-fly filtering.")

    day_frame_pairs = sorted(day_to_frame.items())
    if t_start is not None or t_end is not None:
        lo = 0 if t_start is None else int(t_start)
        hi = (T - 1) if t_end is None else int(t_end)
        if lo < 0 or hi < 0 or lo > hi:
            raise ValueError(f"Invalid frame range: t_start={t_start}, t_end={t_end}")
        day_frame_pairs = [(day, t) for (day, t) in day_frame_pairs if lo <= t <= hi]
        print(f"[info] frame subset enabled: t={lo}..{hi} ({len(day_frame_pairs)} frames)")
    if cutoff_day is not None:
        day_frame_pairs = [(day, t) for (day, t) in day_frame_pairs if day <= cutoff_day]
        print(f"[info] sample truncation applied via experiment_filters: day <= {cutoff_day} ({len(day_frame_pairs)} frames)")

    pair_by_t = {t: day for day, t in day_frame_pairs}

    timings = {
        "h5_read": 0.0,
        "artifact_filter": 0.0,
        "front_calc": 0.0,
        "mask_raster": 0.0,
        "plot": 0.0,
    }
    t_total_start = time.perf_counter()
    plotted_frames = 0
    frame_rows: List[dict] = []
    point_rows: List[dict] = []
    prev_sens_assigned_lbl: Optional[np.ndarray] = None

    out_dir = os.path.join(DATA_ROOT, plate, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    range_suffix = ""
    if t_start is not None or t_end is not None:
        lo = 0 if t_start is None else int(t_start)
        hi = (T - 1) if t_end is None else int(t_end)
        range_suffix = f"_t{lo:02d}-{hi:02d}"
    out_path = os.path.join(out_dir, f"{sample}_escape_front_masks_{definition}{range_suffix}.tif")

    with pd.HDFStore(io.h5_path, mode="r") as store, tifffile.TiffFile(io.clone_mask_path) as tf_mask:
        tf_sens = tifffile.TiffFile(clean_sens_path) if use_clean_sensitive else None
        frame_iter = range(T)
        pbar = tqdm(total=len(day_frame_pairs), desc=f"{sample} {definition}", unit="frame") if tqdm is not None else None
        try:
            with tifffile.TiffWriter(out_path, bigtiff=True) as tw:
                for t in frame_iter:
                    frame = tf_mask.pages[t].asarray()
                    clone_mask = (frame == dominant)
                    out_frame = np.zeros((H, W), dtype=np.uint16)
                    out_frame[clone_mask] = 1

                    day = pair_by_t.get(t)
                    if day is not None and day in days_h5:
                        t0 = time.perf_counter()
                        res_xy, sus_xy = load_day_positions_from_store(store, day)
                        timings["h5_read"] += time.perf_counter() - t0

                        t0 = time.perf_counter()
                        if use_clean_sensitive and tf_sens is not None:
                            sus_mask = tf_sens.pages[t].asarray().astype(bool, copy=False)
                            sus_kept_xy, sus_excl_xy = filter_points_by_mask(sus_xy, sus_mask)
                        else:
                            sus_kept_xy, sus_excl_xy, prev_sens_assigned_lbl = split_sensitive_points_by_artifact(
                                sus_xy=sus_xy, h=H, w=W, prev_assigned_lbl=prev_sens_assigned_lbl
                            )
                        timings["artifact_filter"] += time.perf_counter() - t0

                        t0 = time.perf_counter()
                        labels = assign_resistant_cells_to_clone_labels(res_xy, frame) if res_xy.shape[0] else np.array([], dtype=int)
                        clone_points = res_xy[labels == dominant] if labels.size else np.empty((0, 2), dtype=float)
                        if clone_points.shape[0] > 0:
                            front_points = escaped_front_points(
                                clone_points=clone_points,
                                sus_points=sus_kept_xy,
                                h=H,
                                w=W,
                                definition=definition,
                            )
                        else:
                            front_points = np.empty((0, 2), dtype=float)
                        timings["front_calc"] += time.perf_counter() - t0

                        t0 = time.perf_counter()
                        if front_points.shape[0] > 0:
                            esc = build_escape_front_mask(
                                front_points, H, W, circle_radius_px=ESCAPE_DISK_RADIUS_PX, add_polyline=True, polyline_width_px=POLYLINE_WIDTH_PX
                            )
                            esc_vis = build_escape_front_ring_mask(
                                front_points, H, W, ring_radius_px=VIS_RING_RADIUS_PX, ring_width_px=VIS_RING_WIDTH_PX
                            )
                        else:
                            esc = np.zeros((H, W), dtype=bool)
                            esc_vis = np.zeros((H, W), dtype=bool)
                        timings["mask_raster"] += time.perf_counter() - t0
                    else:
                        day = -1 if day is None else int(day)
                        esc = np.zeros((H, W), dtype=bool)
                        esc_vis = np.zeros((H, W), dtype=bool)
                        sus_kept_xy = np.empty((0, 2), dtype=float)
                        sus_excl_xy = np.empty((0, 2), dtype=float)
                        front_points = np.empty((0, 2), dtype=float)

                    out_frame[esc] = 2
                    tw.write(out_frame, compression="deflate")

                    if t in pair_by_t:
                        escaped_px = int(np.count_nonzero(esc))
                        n_front_points = int(front_points.shape[0]) if front_points is not None else 0
                        clone_px = int(np.count_nonzero(clone_mask))
                        frame_rows.append(
                            {
                                "plate": plate,
                                "sample": sample,
                                "strategy": parse_strategy_from_sample(sample),
                                "definition": definition,
                                "t": int(t),
                                "day": int(pair_by_t[t]),
                                "n_front_points": n_front_points,
                                "esc_px": escaped_px,
                                "clone_px": clone_px,
                            }
                        )
                        if front_points is not None and front_points.shape[0] > 0:
                            for i, pxy in enumerate(front_points):
                                point_rows.append(
                                    {
                                        "plate": plate,
                                        "sample": sample,
                                        "strategy": parse_strategy_from_sample(sample),
                                        "definition": definition,
                                        "t": int(t),
                                        "day": int(pair_by_t[t]),
                                        "point_idx": int(i),
                                        "x": float(pxy[0]),
                                        "y": float(pxy[1]),
                                    }
                                )
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(day=int(pair_by_t[t]), esc_px=escaped_px, n_front=n_front_points, refresh=False)
                        else:
                            print(
                                f"[frame] def={definition} t={t:02d} day={int(pair_by_t[t]):02d} "
                                f"front_points={n_front_points} esc_px={escaped_px}"
                            )

                        if visualize:
                            t0 = time.perf_counter()
                            show_overlay_debug(
                                clone_mask=clone_mask,
                                sus_kept_xy=sus_kept_xy,
                                sus_excluded_xy=sus_excl_xy,
                                esc_mask=esc_vis,
                                front_points=front_points,
                                title=f"{plate} {sample} day {pair_by_t[t]} (t={t}) dominant={dominant} def={definition}",
                            )
                            timings["plot"] += time.perf_counter() - t0
                            plotted_frames += 1
        finally:
            if tf_sens is not None:
                tf_sens.close()
            if pbar is not None:
                pbar.close()

    metrics_path = os.path.join(out_dir, f"{sample}_escape_front_metrics_{definition}{range_suffix}.csv")
    points_path = os.path.join(out_dir, f"{sample}_escape_front_points_{definition}{range_suffix}.csv")
    pd.DataFrame(frame_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(point_rows).to_csv(points_path, index=False)
    total_elapsed = time.perf_counter() - t_total_start
    measured = sum(timings.values())
    other = max(0.0, total_elapsed - measured)
    print(f"[save] {out_path} n_frames={T}, shape=({H},{W})")
    print(f"[save] {metrics_path}")
    print(f"[save] {points_path}")
    print(f"[timing] frames_processed={len(day_frame_pairs)}, frames_plotted={plotted_frames}, total={total_elapsed:.2f}s")
    for k in ["h5_read", "artifact_filter", "front_calc", "mask_raster", "plot"]:
        v = timings[k]
        pct = (100.0 * v / total_elapsed) if total_elapsed > 0 else 0.0
        print(f"[timing] {k:>15}: {v:8.2f}s ({pct:5.1f}%)")
    pct_other = (100.0 * other / total_elapsed) if total_elapsed > 0 else 0.0
    print(f"[timing] {'other':>15}: {other:8.2f}s ({pct_other:5.1f}%)")
    print(f"[done] run_test finished definition={definition}")
    esc_frames = [r["t"] for r in frame_rows if int(r["n_front_points"]) > 0]
    first_escape_frame = int(min(esc_frames)) if esc_frames else -1
    first_escape_day = int(day_from_tidx(first_escape_frame)) if first_escape_frame >= 0 else -1
    return RunResult(
        plate=plate,
        sample=sample,
        strategy=parse_strategy_from_sample(sample),
        status="ok",
        n_frames_total=int(T),
        n_frames_processed=int(len(day_frame_pairs)),
        first_escape_frame=first_escape_frame,
        first_escape_day=first_escape_day,
        escape_detected=bool(first_escape_frame >= 0),
        out_tif=out_path,
        out_metrics_csv=metrics_path,
        out_points_csv=points_path,
        elapsed_s=float(time.perf_counter() - t_run_start),
        error="",
    )


def _run_one_replicate(
    rep: ReplicateIO,
    definition: str,
    visualize: bool,
    require_clean_sensitive: bool,
    t_start: Optional[int],
    t_end: Optional[int],
) -> RunResult:
    try:
        return run_test(
            plate=rep.plate,
            sample=rep.sample,
            definition=definition,
            visualize=visualize,
            require_clean_sensitive=require_clean_sensitive,
            t_start=t_start,
            t_end=t_end,
        )
    except Exception as e:
        return RunResult(
            plate=rep.plate,
            sample=rep.sample,
            strategy=parse_strategy_from_sample(rep.sample),
            status="error",
            n_frames_total=0,
            n_frames_processed=0,
            first_escape_frame=-1,
            first_escape_day=-1,
            escape_detected=False,
            out_tif="",
            out_metrics_csv="",
            out_points_csv="",
            elapsed_s=0.0,
            error=str(e),
        )


def save_batch_summary(results: List[RunResult], definition: str, t_start: Optional[int], t_end: Optional[int]) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if t_start is not None or t_end is not None:
        lo = 0 if t_start is None else int(t_start)
        hi = -1 if t_end is None else int(t_end)
        suffix = f"_t{lo:02d}-{hi:02d}"
    out_path = SUMMARY_DIR / f"escape_front_batch_summary_{definition}{suffix}.csv"
    rows = [r.__dict__ for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log_path = SUMMARY_DIR / f"escape_front_batch_summary_{definition}{suffix}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"summary_csv={out_path}\n")
        f.write(f"n_rows={len(df)}\n")
        if len(df):
            f.write("status_counts:\n")
            f.write(df["status"].value_counts(dropna=False).to_string())
            f.write("\n")
            err = df[df["status"] != "ok"][["plate", "sample", "error"]]
            f.write(f"n_errors={len(err)}\n")
            if len(err):
                f.write(err.to_string(index=False))
                f.write("\n")
    print(f"[save] batch summary: {out_path}")
    print(f"[save] batch log: {log_path}")
    return str(out_path)


def run_batch(
    definition: str,
    workers: int,
    visualize: bool,
    require_clean_sensitive: bool,
    t_start: Optional[int],
    t_end: Optional[int],
) -> str:
    reps = discover_all_replicates()
    print(f"[start] batch mode: replicates={len(reps)}, workers={workers}, definition={definition}")
    if not reps:
        raise RuntimeError("No replicates discovered.")

    results: List[RunResult] = []
    if workers <= 1:
        seq = reps
        if tqdm is not None:
            seq = tqdm(reps, total=len(reps), desc="Escape batch", unit="rep")
        for rep in seq:
            rr = _run_one_replicate(
                rep=rep,
                definition=definition,
                visualize=visualize,
                require_clean_sensitive=require_clean_sensitive,
                t_start=t_start,
                t_end=t_end,
            )
            results.append(rr)
            print(f"[rep] {rep.plate}/{rep.sample} status={rr.status} first_escape_day={rr.first_escape_day}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {
                ex.submit(
                    _run_one_replicate,
                    rep,
                    definition,
                    visualize,
                    require_clean_sensitive,
                    t_start,
                    t_end,
                ): rep
                for rep in reps
            }
            fut_iter = as_completed(fut_map)
            if tqdm is not None:
                fut_iter = tqdm(fut_iter, total=len(fut_map), desc="Escape batch", unit="rep")
            for fut in fut_iter:
                rep = fut_map[fut]
                rr = fut.result()
                results.append(rr)
                print(f"[rep] {rep.plate}/{rep.sample} status={rr.status} first_escape_day={rr.first_escape_day}")

    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status != "ok")
    print(f"[done] batch finished: ok={ok}, errors={err}")
    return save_batch_summary(results, definition=definition, t_start=t_start, t_end=t_end)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write escaped-front TIFF masks and batch summaries.")
    p.add_argument("--mode", choices=["test", "batch"], default="batch")
    p.add_argument("--plate", default=DEFAULT_TEST_PLATE)
    p.add_argument("--sample", default=DEFAULT_TEST_SAMPLE)
    p.add_argument(
        "--definition",
        choices=["legacy_nearest", "all_res_within_sens"],
        default=DEFAULT_DEFINITION,
        help="Escape definition (batch should use all_res_within_sens)",
    )
    p.add_argument("--t-start", type=int, default=DEFAULT_T_START, help="Start frame index (0-based)")
    p.add_argument("--t-end", type=int, default=DEFAULT_T_END, help="End frame index (0-based, inclusive)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers for batch mode")
    p.add_argument("--visualize", action="store_true", help="Show per-frame plots (slow)")
    p.add_argument(
        "--allow-missing-clean-sensitive",
        action="store_true",
        help="Allow fallback if precomputed sensitive masks are missing",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "batch":
        run_batch(
            definition=args.definition,
            workers=max(1, int(args.workers)),
            visualize=bool(args.visualize),
            require_clean_sensitive=not bool(args.allow_missing_clean_sensitive),
            t_start=args.t_start,
            t_end=args.t_end,
        )
    else:
        run_test(
            args.plate,
            args.sample,
            definition=args.definition,
            visualize=bool(args.visualize),
            require_clean_sensitive=not bool(args.allow_missing_clean_sensitive),
            t_start=args.t_start,
            t_end=args.t_end,
        )
