"""
TL;DR
-----
This script converts escape-front point data into time-resolved escape-area
metrics. It is the original branch for quantifying how much spatial territory
the dominant resistant clone gains as it escapes.

Detailed
--------
Inputs:
- Escape-front point CSVs and summary outputs from `Escape_Front_Write_Tiffs.py`.
- Clone-mask geometry and resistant point clouds used to localize the dominant
  resistant clone.

Outputs:
- Timeseries and summary CSVs describing escape area, front length, and
  associated resistant-cell counts.

How it works:
- Discovers valid replicate inputs and applies replicate-level filters.
- Converts escape-front points into clustered geometric regions on the image
  canvas.
- Builds escape-effect zones and measures quantities such as area and front
  length relative to the dominant clone.
- Appends those per-timepoint measurements into figure-ready CSV tables.

Pipeline role:
- Original Figure 2 escape-area quantification branch; still informative, but
  generally superseded by the newer `_new` branch for canonical use.
"""

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile
from pandas.errors import EmptyDataError
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree
from skimage.draw import polygon
from skimage.measure import perimeter
from skimage.morphology import disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import experiment_filters

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = DATA_ROOT / r"Figures\Escape_Quantification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_SUFFIX = "_res_clone_masks.tif"
H5_RE = re.compile(r".+_total_poisson(?:_padded)?\.h5$", re.IGNORECASE)
SAMPLE_RE = re.compile(r"^[A-H]\d+_([^-]+)-[A-H]\d+$", re.IGNORECASE)

# Shared geometry parameters.
ANGLE_BINS = 1800
CLUSTER_DIST_PX = 300.0
POINT_DILATE_RADIUS_PX = 24
OUTWARD_PERCENTILE = 40.0
RAY_GAP_BRIDGE_PX = 80.0
RAY_ROI_MARGIN_PX = 64


def day_from_tidx(t: int) -> int:
    offsets = (1, 3, 5)
    return (t // 3) * 7 + offsets[t % 3]


def parse_strategy(sample: str) -> str:
    m = SAMPLE_RE.match(str(sample))
    return m.group(1) if m else "UNKNOWN"


def parse_sample_from_h5_name(h5_name: str) -> str | None:
    stem = h5_name.replace("_total_poisson_padded.h5", "").replace("_total_poisson.h5", "")
    m = re.search(r"(?:^|_)([A-H]\d{1,2}_.+?-[A-H]\d{1,2})$", stem, re.IGNORECASE)
    return m.group(1) if m else None


def load_stack(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        with tifffile.TiffFile(str(path)) as tf:
            n_pages = len(tf.pages)
            if n_pages > 1:
                return tf.asarray(key=range(n_pages))
        return arr[None, ...]
    raise ValueError(f"Unexpected TIFF dimensionality for {path}: {arr.shape}")


def sample_filter_rule(sample: str) -> Tuple[str, float | None]:
    rules = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    if sample in rules:
        cutoff = rules[sample]
        if cutoff is None:
            return "drop", None
        return "truncate", float(cutoff)
    return "keep", None


def discover_replicates() -> List[Tuple[str, str, Path, Path, Path]]:
    reps: List[Tuple[str, str, Path, Path, Path]] = []
    for plate_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("Plate_")]):
        plate = plate_dir.name
        h5_dir = plate_dir / "hdf5"
        mask_dir = plate_dir / "clone_masks_stacks"
        esc_dir = plate_dir / "escape_front_masks_stacks"
        if not (h5_dir.is_dir() and mask_dir.is_dir() and esc_dir.is_dir()):
            continue
        h5_files = sorted([p for p in h5_dir.iterdir() if p.is_file() and H5_RE.match(p.name)])
        by_sample: Dict[str, Path] = {}
        for hp in h5_files:
            s = parse_sample_from_h5_name(hp.name)
            if not s:
                continue
            old = by_sample.get(s)
            if old is None or ("_padded" in hp.name and "_padded" not in old.name):
                by_sample[s] = hp
        for sample, h5p in by_sample.items():
            mask_path = mask_dir / f"{sample}{MASK_SUFFIX}"
            points_csv = esc_dir / f"{sample}_escape_front_points_all_res_within_sens.csv"
            if mask_path.exists() and points_csv.exists():
                reps.append((plate, sample, h5p, mask_path, points_csv))
    return reps


def find_dominant_clone_label(mask_stack: np.ndarray) -> int | None:
    if mask_stack.ndim != 3 or mask_stack.shape[0] == 0:
        return None
    labels, counts = np.unique(mask_stack[-1][mask_stack[-1] >= 2], return_counts=True)
    if labels.size == 0:
        return None
    return int(labels[np.argmax(counts)])


def load_points_by_t(points_csv: Path) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not points_csv.exists():
        return out
    try:
        pdf = pd.read_csv(points_csv)
    except EmptyDataError:
        return out
    except Exception:
        return out
    if len(pdf) == 0 or not {"t", "x", "y"}.issubset(pdf.columns):
        return out
    for tt, grp in pdf.groupby("t"):
        out[int(tt)] = grp[["x", "y"]].to_numpy(dtype=float)
    return out


def preload_res_points_by_day(h5_path: Path, n_frames: int) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for t in range(n_frames):
        day = day_from_tidx(t)
        if day in out:
            continue
        try:
            df = pd.read_hdf(str(h5_path), key=f"/res/d_{int(day):02d}")
        except Exception:
            out[day] = np.empty((0, 2), dtype=float)
            continue
        if len(df) == 0:
            out[day] = np.empty((0, 2), dtype=float)
        else:
            out[day] = df[["x", "y"]].to_numpy(dtype=float)
    return out


def dominant_res_count_all(frame: np.ndarray, dominant: int, xy: np.ndarray) -> int:
    if xy.size == 0:
        return 0
    h, w = frame.shape
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.zeros(len(xy), dtype=np.int32)
    labels[inb] = frame[yi[inb], xi[inb]].astype(np.int32)
    return int(np.count_nonzero(labels == dominant))


def dominant_res_count_in_region(frame: np.ndarray, dominant: int, xy: np.ndarray, region_mask: np.ndarray) -> int:
    if xy.size == 0:
        return 0
    h, w = frame.shape
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.zeros(len(xy), dtype=np.int32)
    labels[inb] = frame[yi[inb], xi[inb]].astype(np.int32)
    in_region = np.zeros(len(xy), dtype=bool)
    in_region[inb] = region_mask[yi[inb], xi[inb]]
    return int(np.count_nonzero((labels == dominant) & in_region))


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.p[rb] = ra


def points_in_bounds(x: np.ndarray, y: np.ndarray, w: int, h: int) -> np.ndarray:
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)


def cluster_points(points_xy: np.ndarray, dist_px: float) -> List[np.ndarray]:
    if points_xy.size == 0:
        return []
    if len(points_xy) == 1:
        return [points_xy.copy()]
    tree = cKDTree(points_xy)
    pairs = tree.query_pairs(r=dist_px)
    uf = _UF(len(points_xy))
    for i, j in pairs:
        uf.union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(points_xy)):
        groups.setdefault(uf.find(i), []).append(i)
    return [points_xy[idxs] for idxs in groups.values()]


def cluster_area_mask(cluster_xy: np.ndarray, shape_hw: Tuple[int, int], dilate_radius_px: int) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=bool)
    if cluster_xy.size == 0:
        return m
    if cluster_xy.shape[0] >= 3:
        try:
            hull = ConvexHull(cluster_xy)
            hp = cluster_xy[hull.vertices]
            rr, cc = polygon(hp[:, 1], hp[:, 0], shape=(h, w))
            m[rr, cc] = True
        except Exception:
            xi = np.rint(cluster_xy[:, 0]).astype(int)
            yi = np.rint(cluster_xy[:, 1]).astype(int)
            inb = points_in_bounds(xi, yi, w, h)
            m[yi[inb], xi[inb]] = True
    else:
        xi = np.rint(cluster_xy[:, 0]).astype(int)
        yi = np.rint(cluster_xy[:, 1]).astype(int)
        inb = points_in_bounds(xi, yi, w, h)
        m[yi[inb], xi[inb]] = True
    if dilate_radius_px > 0:
        m = ndi.binary_dilation(m, structure=disk(dilate_radius_px))
    m = ndi.binary_fill_holes(m)
    return m


def escape_area_from_points(points_xy: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape_hw, dtype=bool)
    if points_xy.size == 0:
        return union
    for c in cluster_points(points_xy, dist_px=CLUSTER_DIST_PX):
        union |= cluster_area_mask(c, shape_hw=shape_hw, dilate_radius_px=POINT_DILATE_RADIUS_PX)
    return union


def _draw_line_mask(shape_hw: Tuple[int, int], p0: Tuple[float, float], p1: Tuple[float, float]) -> np.ndarray:
    h, w = shape_hw
    x0, y0 = p0
    x1, y1 = p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    out = np.zeros((h, w), dtype=bool)
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    out[ys[inb], xs[inb]] = True
    return out


def build_escape_effect_zone(esc_prev: np.ndarray, esc_now: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    union = esc_prev | esc_now
    bridge = np.zeros_like(union, dtype=bool)
    if np.count_nonzero(esc_prev) == 0 or np.count_nonzero(esc_now) == 0:
        return union, bridge
    if np.count_nonzero(esc_prev & esc_now) > 0:
        return union, bridge
    l_prev, n_prev = ndi.label(esc_prev)
    l_now, n_now = ndi.label(esc_now)
    if n_prev == 0 or n_now == 0:
        return union, bridge

    prev_pts: Dict[int, np.ndarray] = {}
    prev_trees: Dict[int, cKDTree] = {}
    for pid in range(1, n_prev + 1):
        yy, xx = np.nonzero(l_prev == pid)
        if xx.size == 0:
            continue
        pts = np.column_stack([xx.astype(float), yy.astype(float)])
        prev_pts[pid] = pts
        prev_trees[pid] = cKDTree(pts)

    for cid in range(1, n_now + 1):
        yy, xx = np.nonzero(l_now == cid)
        if xx.size == 0:
            continue
        now_pts = np.column_stack([xx.astype(float), yy.astype(float)])
        best_dist = np.inf
        best_prev = None
        best_i_prev = None
        best_i_now = None
        for pid, tree in prev_trees.items():
            d, idx_prev = tree.query(now_pts, k=1)
            j = int(np.argmin(d))
            dj = float(d[j])
            if dj < best_dist:
                best_dist = dj
                best_prev = pid
                best_i_prev = int(idx_prev[j])
                best_i_now = j
        if best_prev is None:
            continue
        p_prev = prev_pts[best_prev][best_i_prev]
        p_now = now_pts[best_i_now]
        line = _draw_line_mask(union.shape, (p_prev[0], p_prev[1]), (p_now[0], p_now[1]))
        bridge |= ndi.binary_dilation(line, structure=disk(2))

    out = ndi.binary_fill_holes(union | bridge)
    return out, bridge


def true_ray_outer_front_length_from_area(area_mask: np.ndarray, center_xy: Tuple[float, float]) -> float:
    if not np.any(area_mask):
        return 0.0
    h, w = area_mask.shape
    yy, xx = np.nonzero(area_mask)
    x0 = max(0, int(np.min(xx) - RAY_ROI_MARGIN_PX))
    x1 = min(w, int(np.max(xx) + RAY_ROI_MARGIN_PX + 1))
    y0 = max(0, int(np.min(yy) - RAY_ROI_MARGIN_PX))
    y1 = min(h, int(np.max(yy) + RAY_ROI_MARGIN_PX + 1))
    roi = area_mask[y0:y1, x0:x1]

    cx = center_xy[0] - x0
    cy = center_xy[1] - y0
    rh, rw = roi.shape
    max_t = int(np.ceil(np.hypot(max(cx, rw - cx), max(cy, rh - cy)))) + 2
    ts = np.arange(0, max_t, 1.0, dtype=float)
    ang = np.linspace(0.0, 2.0 * np.pi, ANGLE_BINS, endpoint=False, dtype=float)
    px = np.full(ANGLE_BINS, np.nan, dtype=float)
    py = np.full(ANGLE_BINS, np.nan, dtype=float)
    for i, a in enumerate(ang):
        xs = np.rint(cx + ts * np.cos(a)).astype(int)
        ys = np.rint(cy + ts * np.sin(a)).astype(int)
        inb = (xs >= 0) & (xs < rw) & (ys >= 0) & (ys < rh)
        if not np.any(inb):
            continue
        xs = xs[inb]
        ys = ys[inb]
        inside = roi[ys, xs]
        if not np.any(inside):
            continue
        j = int(np.flatnonzero(inside)[-1])
        px[i] = float(xs[j] + x0)
        py[i] = float(ys[j] + y0)

    valid = np.isfinite(px) & np.isfinite(py)
    if np.count_nonzero(valid) < 2:
        return 0.0
    rr = np.hypot(px[valid] - center_xy[0], py[valid] - center_xy[1])
    thr = float(np.nanpercentile(rr, OUTWARD_PERCENTILE))
    keep = valid & (np.hypot(px - center_xy[0], py - center_xy[1]) >= thr)
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        return 0.0

    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends = np.r_[splits + 1, idx.size]
    segs: List[np.ndarray] = []
    for s, e in zip(starts, ends):
        ii = idx[s:e]
        if ii.size >= 2:
            segs.append(np.column_stack([px[ii], py[ii]]))
    if not segs:
        return 0.0

    merged: List[np.ndarray] = [segs[0]]
    for seg in segs[1:]:
        d = float(np.hypot(*(merged[-1][-1] - seg[0])))
        if d <= RAY_GAP_BRIDGE_PX:
            merged[-1] = np.vstack([merged[-1], seg])
        else:
            merged.append(seg)
    return float(max(np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))) for seg in merged))


def front_length_from_points(points_xy: np.ndarray, center_xy: Tuple[float, float], shape_hw: Tuple[int, int]) -> float:
    clusters = cluster_points(points_xy, dist_px=CLUSTER_DIST_PX)
    if not clusters:
        return 0.0
    total = 0.0
    for c in clusters:
        m = cluster_area_mask(c, shape_hw=shape_hw, dilate_radius_px=POINT_DILATE_RADIUS_PX)
        total += true_ray_outer_front_length_from_area(m, center_xy=center_xy)
    return float(total)


def append_rows_csv(path: Path, rows: List[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    write_header = (not path.exists()) or path.stat().st_size == 0
    df.to_csv(path, mode="a", index=False, header=write_header)
    return int(len(df))


def load_already_processed(path: Path) -> set[Tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["plate", "sample"])
    except Exception:
        return set()
    if len(df) == 0:
        return set()
    return set((str(r["plate"]), str(r["sample"])) for _, r in df.iterrows())


def main() -> None:
    reps = discover_replicates()
    print(f"[info] discovered replicates: {len(reps)}")
    print(
        f"[info] params angle_bins={ANGLE_BINS}, cluster_dist={CLUSTER_DIST_PX}, "
        f"dilate={POINT_DILATE_RADIUS_PX}, outward_p={OUTWARD_PERCENTILE}"
    )

    out_frame = OUT_DIR / "Escape_Fig_Build_escape_areas_timeseries.csv"
    out_summary = OUT_DIR / "Escape_Fig_Build_escape_areas_summary.csv"
    already_done = load_already_processed(out_summary)
    if already_done:
        print(f"[info] resume mode: found {len(already_done)} already processed wells in {out_summary.name}")

    total_frame_rows_written = 0
    total_summary_rows_written = 0
    n_ok = 0
    n_skip = 0

    rep_iter = reps
    if tqdm is not None:
        rep_iter = tqdm(reps, total=len(reps), desc="Build escape areas", unit="rep")

    for plate, sample, h5_path, mask_path, points_csv in rep_iter:
        if (plate, sample) in already_done:
            n_skip += 1
            print(f"[skip] {plate}/{sample}: already present in summary CSV")
            continue

        status, cutoff_day = sample_filter_rule(sample)
        if status == "drop":
            n_skip += 1
            print(f"[skip] {plate}/{sample}: excluded")
            continue

        mask_stack = load_stack(mask_path)
        if mask_stack.ndim != 3:
            n_skip += 1
            print(f"[skip] {plate}/{sample}: invalid mask shape {mask_stack.shape}")
            continue
        dom = find_dominant_clone_label(mask_stack)
        if dom is None:
            n_skip += 1
            print(f"[skip] {plate}/{sample}: no dominant clone")
            continue

        h, w = mask_stack.shape[1], mask_stack.shape[2]
        center = (w / 2.0, h / 2.0)
        points_by_t = load_points_by_t(points_csv)
        res_by_day = preload_res_points_by_day(h5_path, n_frames=mask_stack.shape[0])

        esc_area_by_t: Dict[int, np.ndarray] = {}
        for t in range(mask_stack.shape[0]):
            esc_area_by_t[t] = escape_area_from_points(points_by_t.get(t, np.empty((0, 2), dtype=float)), shape_hw=mask_stack[t].shape)

        prev_included_t = None
        sum_escape = 0.0
        sum_non_escape = 0.0
        all_escaped_flag = False
        n_frames_used = 0
        first_escape_day = np.nan
        sample_frame_rows: List[dict] = []

        for t in range(mask_stack.shape[0]):
            day = day_from_tidx(t)
            if cutoff_day is not None and day > cutoff_day:
                continue

            frame = mask_stack[t]
            clone_mask = frame == dom
            boundary_len = float(perimeter(clone_mask, neighborhood=8))
            p = points_by_t.get(t, np.empty((0, 2), dtype=float))
            esc_len = front_length_from_points(p, center_xy=center, shape_hw=(h, w))
            esc_frac = float(esc_len / boundary_len) if boundary_len > 0 else 0.0
            res_xy_now = res_by_day.get(day, np.empty((0, 2), dtype=float))
            res_count = dominant_res_count_all(frame, dom, res_xy_now)
            if np.isnan(first_escape_day) and len(p) > 0:
                first_escape_day = float(day)

            if prev_included_t is None:
                delta_per_day = np.nan
                escape_growth = np.nan
                non_escape_growth = np.nan
                cum_escape_frac = np.nan
                overlap_px = 0
                escape_zone_px = int(np.count_nonzero(esc_area_by_t[t]))
                non_escape_px = int(np.count_nonzero(clone_mask & (~esc_area_by_t[t])))
            else:
                prev_day = day_from_tidx(prev_included_t)
                days_delta = float(day - prev_day)
                frame_prev = mask_stack[prev_included_t]
                res_xy_prev = res_by_day.get(prev_day, np.empty((0, 2), dtype=float))
                prev_count = dominant_res_count_all(frame_prev, dom, res_xy_prev)
                delta_per_day = float((res_count - prev_count) / days_delta) if days_delta > 0 else np.nan

                esc_prev = esc_area_by_t[prev_included_t]
                esc_now = esc_area_by_t[t]
                overlap_esc = esc_prev & esc_now
                escape_zone, _ = build_escape_effect_zone(esc_prev, esc_now)
                clone_prev = frame_prev == dom
                clone_now = frame == dom
                clone_union = clone_prev | clone_now
                non_escape_zone = clone_union & (~escape_zone)

                if all_escaped_flag or np.count_nonzero(non_escape_zone) == 0:
                    all_escaped_flag = True
                    non_escape_growth = 0.0
                else:
                    c0n = dominant_res_count_in_region(frame_prev, dom, res_xy_prev, non_escape_zone)
                    c1n = dominant_res_count_in_region(frame, dom, res_xy_now, non_escape_zone)
                    non_escape_growth = float(max(0, c1n - c0n))

                c0e = dominant_res_count_in_region(frame_prev, dom, res_xy_prev, escape_zone)
                c1e = dominant_res_count_in_region(frame, dom, res_xy_now, escape_zone)
                escape_growth = float(max(0, c1e - c0e))
                if all_escaped_flag:
                    c0all = dominant_res_count_in_region(frame_prev, dom, res_xy_prev, clone_union)
                    c1all = dominant_res_count_in_region(frame, dom, res_xy_now, clone_union)
                    escape_growth = float(max(0, c1all - c0all))

                sum_escape += float(escape_growth)
                sum_non_escape += float(non_escape_growth)
                denom = sum_escape + sum_non_escape
                cum_escape_frac = float(sum_escape / denom) if denom > 0 else np.nan
                overlap_px = int(np.count_nonzero(overlap_esc))
                escape_zone_px = int(np.count_nonzero(escape_zone))
                non_escape_px = int(np.count_nonzero(non_escape_zone))
                n_frames_used += 1

            sample_frame_rows.append(
                {
                    "plate": plate,
                    "sample": sample,
                    "strategy": parse_strategy(sample),
                    "dom": int(dom),
                    "t": int(t),
                    "day": int(day),
                    "res_count": int(res_count),
                    "delta_res_count_per_day": delta_per_day,
                    "escape_front_length_px": float(esc_len),
                    "escape_front_fraction": float(esc_frac),
                    "boundary_length_px": float(boundary_len),
                    "overlap_px": int(overlap_px),
                    "escape_zone_px": int(escape_zone_px),
                    "non_escape_px": int(non_escape_px),
                    "escape_growth_cells": escape_growth,
                    "non_escape_growth_cells": non_escape_growth,
                    "cumulative_escape_fraction": cum_escape_frac,
                }
            )
            prev_included_t = t

        frac_total = float(sum_escape / (sum_escape + sum_non_escape)) if (sum_escape + sum_non_escape) > 0 else np.nan
        summary_row = {
            "plate": plate,
            "sample": sample,
            "strategy": parse_strategy(sample),
            "dom": int(dom),
            "n_frames_total": int(mask_stack.shape[0]),
            "n_frames_used": int(n_frames_used),
            "first_escape_day": first_escape_day,
            "sum_escape_growth_cells": float(sum_escape),
            "sum_non_escape_growth_cells": float(sum_non_escape),
            "escape_growth_fraction_total": frac_total,
        }

        written_frames = append_rows_csv(out_frame, sample_frame_rows)
        written_summary = append_rows_csv(out_summary, [summary_row])
        total_frame_rows_written += written_frames
        total_summary_rows_written += written_summary
        already_done.add((plate, sample))
        n_ok += 1
        print(f"[save] {plate}/{sample}: appended frames={written_frames} summary={written_summary}")
        print(
            f"[done] {plate}/{sample} dom={dom} used={n_frames_used} "
            f"sum_escape={sum_escape:.1f} sum_non_escape={sum_non_escape:.1f} "
            f"frac={frac_total if np.isfinite(frac_total) else float('nan'):.3f}"
        )

    print(f"[save] {out_frame} appended_rows_this_run={total_frame_rows_written}")
    print(f"[save] {out_summary} appended_rows_this_run={total_summary_rows_written}")
    print(f"[done] build finished: ok={n_ok}, skipped={n_skip}")


if __name__ == "__main__":
    main()
