"""
TL;DR
-----
Legacy circular-query implementation of confinement thickness around the dominant resistant clone.
for each replicate over time. The key summary is the minimum surrounding
sensitive-cell support measured across many radial directions.

Detailed
--------
Inputs:
- Dominant-clone masks, resistant point clouds, and susceptible point clouds
  for each replicate/day.

Outputs:
- A per-replicate/per-day confinement-thickness CSV in the escape
  quantification area.
- Optional debug/test visualizations when run in test mode.

How it works:
- Builds a large reference circle of angular probe points around the colony.
- For each angle, finds the nearest resistant cell belonging to the dominant
  clone and uses that distance as a local search radius.
- Counts susceptible cells within that radius and defines confinement
  thickness as the minimum count observed over all probe angles.
- Also records the angle and radius where the minimum occurs.

Pipeline role:
- Figure 2 analysis branch that quantifies how strongly the dominant resistant
  clone is locally surrounded by susceptible cells.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["text.usetex"] = False
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["font.family"] = "Arial"
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["axes.labelsize"] = 7
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from tqdm import tqdm


DATA_ROOT = r"Y:\Members\MaxE\Publications\STEM Paper\Data\exp_raw_data"
OUT_DIR = r"Y:\Members\MaxE\Publications\STEM Paper\Data\intermediate_analysis_data\experimental\figure_2\escape_quantification"

PLATES = [f"Plate_{i}" for i in range(1, 9)]
MASK_SUFFIX = "_res_clone_masks.tif"

DEFAULT_TEST_PLATE = "Plate_3"
DEFAULT_TEST_SAMPLE = "C5_AT50-C5"

CIRCLE_DIAMETER_PX = 50_000.0
N_DOTS = 360

DAY_RE = re.compile(r"^d_(\d+)$")


@dataclass
class ReplicateIO:
    plate: str
    sample: str
    clone_mask_path: str
    h5_path: str


def day_from_tidx(t_idx: int) -> int:
    # Mon/Wed/Fri schedule: 1,3,5,8,10,12,...
    offsets = (1, 3, 5)
    return (t_idx // 3) * 7 + offsets[t_idx % 3]


def build_day_to_frame(n_frames: int) -> Dict[int, int]:
    return {day_from_tidx(t): t for t in range(n_frames)}


def make_reference_dots(center_x: float, center_y: float) -> tuple[np.ndarray, np.ndarray]:
    cx = float(center_x)
    cy = float(center_y)
    radius = CIRCLE_DIAMETER_PX / 2.0
    angles_deg = np.arange(N_DOTS, dtype=float)  # 0..359
    theta = np.deg2rad(angles_deg)
    # 0 deg at top, clockwise
    x = cx + radius * np.sin(theta)
    y = cy - radius * np.cos(theta)
    dots = np.column_stack((x, y))
    return angles_deg, dots


def find_dominant_clone_label(mask_stack: np.ndarray) -> Optional[int]:
    """Dominant clone = present at final frame and with maximal survival duration."""
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

    best_label: Optional[int] = None
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


def extract_days_in_h5(h5_path: str) -> List[int]:
    days = []
    with h5py.File(h5_path, "r") as f:
        if "res" not in f or "sus" not in f:
            return days
        res_keys = set(f["res"].keys())
        sus_keys = set(f["sus"].keys())
        common = sorted(res_keys.intersection(sus_keys))
        for k in common:
            m = DAY_RE.match(k)
            if m:
                days.append(int(m.group(1)))
    return days


def load_day_positions(h5_path: str, day: int) -> tuple[np.ndarray, np.ndarray]:
    key = f"d_{day:02d}"
    res_df = pd.read_hdf(h5_path, key=f"/res/{key}")
    sus_df = pd.read_hdf(h5_path, key=f"/sus/{key}")
    if "x" not in res_df.columns or "y" not in res_df.columns:
        raise ValueError(f"Missing x/y columns in /res/{key} of {h5_path}")
    if "x" not in sus_df.columns or "y" not in sus_df.columns:
        raise ValueError(f"Missing x/y columns in /sus/{key} of {h5_path}")
    res_xy = res_df[["x", "y"]].to_numpy(dtype=float)
    sus_xy = sus_df[["x", "y"]].to_numpy(dtype=float)
    return res_xy, sus_xy


def assign_resistant_cells_to_clone_labels(res_xy: np.ndarray, clone_mask_2d: np.ndarray) -> np.ndarray:
    h, w = clone_mask_2d.shape
    xi = np.rint(res_xy[:, 0]).astype(int)
    yi = np.rint(res_xy[:, 1]).astype(int)
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.zeros(res_xy.shape[0], dtype=np.int32)
    labels[in_bounds] = clone_mask_2d[yi[in_bounds], xi[in_bounds]].astype(np.int32)
    return labels


def count_sensitive_within_variable_radii(
    dots_xy: np.ndarray, radii: np.ndarray, sus_xy: np.ndarray
) -> np.ndarray:
    if sus_xy.shape[0] == 0:
        return np.zeros(dots_xy.shape[0], dtype=np.int64)
    tree = cKDTree(sus_xy)
    try:
        counts = tree.query_ball_point(dots_xy, r=radii, return_length=True)
        return np.asarray(counts, dtype=np.int64)
    except TypeError:
        vals = [len(tree.query_ball_point(dots_xy[i], r=float(radii[i]))) for i in range(dots_xy.shape[0])]
        return np.asarray(vals, dtype=np.int64)


def confinement_for_clone(
    clone_res_xy: np.ndarray, sus_xy: np.ndarray, angles_deg: np.ndarray, dots_xy: np.ndarray
) -> dict:
    res_tree = cKDTree(clone_res_xy)
    dists, _ = res_tree.query(dots_xy, k=1)
    dists = np.asarray(dists, dtype=float)
    counts = count_sensitive_within_variable_radii(dots_xy, dists, sus_xy)
    min_idx = int(np.argmin(counts))
    return {
        "confinement_thickness": int(counts[min_idx]),
        "min_angle_deg": float(angles_deg[min_idx]),
        "distance_px_at_min": float(dists[min_idx]),
        "min_dot_x": float(dots_xy[min_idx, 0]),
        "min_dot_y": float(dots_xy[min_idx, 1]),
    }


def discover_replicates_for_plate(plate: str) -> List[ReplicateIO]:
    plate_dir = os.path.join(DATA_ROOT, plate)
    mask_dir = os.path.join(plate_dir, "clone_masks_stacks")
    h5_dir = os.path.join(plate_dir, "hdf5")
    if not os.path.isdir(mask_dir) or not os.path.isdir(h5_dir):
        return []

    reps: List[ReplicateIO] = []
    for fn in sorted(os.listdir(mask_dir)):
        if not fn.endswith(MASK_SUFFIX):
            continue
        sample = fn[: -len(MASK_SUFFIX)]
        mask_path = os.path.join(mask_dir, fn)
        pattern_padded = os.path.join(h5_dir, f"*_{sample}_total_poisson_padded.h5")
        pattern_legacy = os.path.join(h5_dir, f"*_{sample}_total_poisson.h5")
        hits_padded = sorted(glob.glob(pattern_padded))
        hits_legacy = sorted(glob.glob(pattern_legacy))
        if hits_padded:
            h5_path = hits_padded[0]
        elif hits_legacy:
            h5_path = hits_legacy[0]
        else:
            continue
        reps.append(ReplicateIO(plate=plate, sample=sample, clone_mask_path=mask_path, h5_path=h5_path))
    return reps


def build_test_replicate(plate: str, sample: str) -> ReplicateIO:
    mask_path = os.path.join(DATA_ROOT, plate, "clone_masks_stacks", f"{sample}{MASK_SUFFIX}")
    h5_dir = os.path.join(DATA_ROOT, plate, "hdf5")
    hits_padded = sorted(glob.glob(os.path.join(h5_dir, f"*_{sample}_total_poisson_padded.h5")))
    hits_legacy = sorted(glob.glob(os.path.join(h5_dir, f"*_{sample}_total_poisson.h5")))
    hits = hits_padded if hits_padded else hits_legacy
    if not hits:
        raise FileNotFoundError(f"No h5 match for sample {sample} in {h5_dir}")
    return ReplicateIO(plate=plate, sample=sample, clone_mask_path=mask_path, h5_path=hits[0])


def analyze_replicate(io: ReplicateIO) -> tuple[pd.DataFrame, Optional[dict]]:
    if not os.path.isfile(io.clone_mask_path):
        raise FileNotFoundError(f"Missing clone mask: {io.clone_mask_path}")
    if not os.path.isfile(io.h5_path):
        raise FileNotFoundError(f"Missing h5: {io.h5_path}")

    mask_stack = tifffile.imread(io.clone_mask_path)
    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]
    if mask_stack.ndim != 3:
        raise ValueError(f"Unexpected mask stack shape: {mask_stack.shape} ({io.clone_mask_path})")

    n_frames = int(mask_stack.shape[0])
    day_to_frame = build_day_to_frame(n_frames)
    days_in_h5 = set(extract_days_in_h5(io.h5_path))
    h, w = int(mask_stack.shape[1]), int(mask_stack.shape[2])
    center_x = float(w) / 2.0
    center_y = float(h) / 2.0
    dominant_clone_id = find_dominant_clone_label(mask_stack)

    rows: List[dict] = []
    viz_payload: Optional[dict] = None

    for day in sorted(day_to_frame.keys()):
        t = day_to_frame[day]
        clone_mask = mask_stack[t]

        if day not in days_in_h5:
            rows.append(
                {
                    "plate": io.plate,
                    "replicate": io.sample,
                    "day": int(day),
                    "frame_idx": int(t),
                    "dominant_clone_id": float("nan"),
                    "confinement_thickness": np.nan,
                    "min_angle_deg": np.nan,
                    "distance_px_at_min": np.nan,
                    "n_resistant_cells_in_clone": 0,
                    "n_sensitive_cells_total": np.nan,
                    "reference_center_x": float(center_x),
                    "reference_center_y": float(center_y),
                    "has_dominant_clone_cells": False,
                    "h5_path": io.h5_path,
                    "clone_mask_path": io.clone_mask_path,
                }
            )
            continue

        res_xy, sus_xy = load_day_positions(io.h5_path, day)
        if dominant_clone_id is None:
            rows.append(
                {
                    "plate": io.plate,
                    "replicate": io.sample,
                    "day": int(day),
                    "frame_idx": int(t),
                    "dominant_clone_id": np.nan,
                    "confinement_thickness": np.nan,
                    "min_angle_deg": np.nan,
                    "distance_px_at_min": np.nan,
                    "n_resistant_cells_in_clone": 0,
                    "n_sensitive_cells_total": int(sus_xy.shape[0]),
                    "reference_center_x": float(center_x),
                    "reference_center_y": float(center_y),
                    "has_dominant_clone_cells": False,
                    "h5_path": io.h5_path,
                    "clone_mask_path": io.clone_mask_path,
                }
            )
            continue

        angles_deg, dots_xy = make_reference_dots(center_x=center_x, center_y=center_y)

        labels = assign_resistant_cells_to_clone_labels(res_xy, clone_mask)
        clone_points = res_xy[labels == dominant_clone_id]
        if clone_points.shape[0] == 0:
            rows.append(
                {
                    "plate": io.plate,
                    "replicate": io.sample,
                    "day": int(day),
                    "frame_idx": int(t),
                    "dominant_clone_id": int(dominant_clone_id),
                    "confinement_thickness": np.nan,
                    "min_angle_deg": np.nan,
                    "distance_px_at_min": np.nan,
                    "n_resistant_cells_in_clone": 0,
                    "n_sensitive_cells_total": int(sus_xy.shape[0]),
                    "reference_center_x": float(center_x),
                    "reference_center_y": float(center_y),
                    "has_dominant_clone_cells": False,
                    "h5_path": io.h5_path,
                    "clone_mask_path": io.clone_mask_path,
                }
            )
            continue

        result = confinement_for_clone(clone_points, sus_xy, angles_deg, dots_xy)
        row = {
            "plate": io.plate,
            "replicate": io.sample,
            "day": int(day),
            "frame_idx": int(t),
            "dominant_clone_id": int(dominant_clone_id),
            "confinement_thickness": int(result["confinement_thickness"]),
            "min_angle_deg": float(result["min_angle_deg"]),
            "distance_px_at_min": float(result["distance_px_at_min"]),
            "n_resistant_cells_in_clone": int(clone_points.shape[0]),
            "n_sensitive_cells_total": int(sus_xy.shape[0]),
            "reference_center_x": float(center_x),
            "reference_center_y": float(center_y),
            "has_dominant_clone_cells": True,
            "h5_path": io.h5_path,
            "clone_mask_path": io.clone_mask_path,
        }
        rows.append(row)

        if viz_payload is None or row["n_resistant_cells_in_clone"] > viz_payload["n_resistant_cells_in_clone"]:
            viz_payload = {
                "plate": io.plate,
                "replicate": io.sample,
                "day": day,
                "clone_id": int(dominant_clone_id),
                "clone_points": clone_points.copy(),
                "sus_points": sus_xy.copy(),
                "angles_deg": angles_deg.copy(),
                "dots_xy": dots_xy.copy(),
                "min_angle_deg": float(result["min_angle_deg"]),
                "distance_px_at_min": float(result["distance_px_at_min"]),
                "min_dot_x": float(result["min_dot_x"]),
                "min_dot_y": float(result["min_dot_y"]),
                "reference_center_x": float(center_x),
                "reference_center_y": float(center_y),
                "n_resistant_cells_in_clone": int(clone_points.shape[0]),
            }

    return pd.DataFrame(rows), viz_payload


def debug_dominant_clone_mapping(io: ReplicateIO) -> None:
    """Verbose day-by-day diagnostics for dominant-clone cell mapping."""
    print(f"[debug] replicate: {io.plate}/{io.sample}")
    print(f"[debug] clone mask: {io.clone_mask_path}")
    print(f"[debug] h5: {io.h5_path}")

    mask_stack = tifffile.imread(io.clone_mask_path)
    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]
    n_frames, h, w = int(mask_stack.shape[0]), int(mask_stack.shape[1]), int(mask_stack.shape[2])
    day_to_frame = build_day_to_frame(n_frames)
    days_in_h5 = set(extract_days_in_h5(io.h5_path))
    dominant = find_dominant_clone_label(mask_stack)

    print(f"[debug] n_frames={n_frames}, image_shape=({h},{w})")
    print(f"[debug] mapped day range: {min(day_to_frame.keys())}..{max(day_to_frame.keys())} ({len(day_to_frame)} days)")
    print(f"[debug] h5 day range: {min(days_in_h5)}..{max(days_in_h5)} ({len(days_in_h5)} days)")
    print(f"[debug] dominant_clone_id={dominant}")
    if dominant is None:
        print("[debug] No dominant clone found in final frame.")
        return

    all_days = sorted(day_to_frame.keys())
    for day in all_days:
        t = day_to_frame[day]
        frame = mask_stack[t]
        mask_dom_px = int(np.count_nonzero(frame == dominant))
        mask_any_clone_px = int(np.count_nonzero(frame >= 2))

        if day not in days_in_h5:
            print(
                f"[debug][day {day:02d}] no h5 group | "
                f"mask_dom_px={mask_dom_px}, mask_any_clone_px={mask_any_clone_px}"
            )
            continue

        res_xy, _ = load_day_positions(io.h5_path, day)
        if res_xy.shape[0] == 0:
            print(
                f"[debug][day {day:02d}] res_n=0 | "
                f"mask_dom_px={mask_dom_px}, mask_any_clone_px={mask_any_clone_px}"
            )
            continue

        xi = np.rint(res_xy[:, 0]).astype(int)
        yi = np.rint(res_xy[:, 1]).astype(int)
        inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        inb_n = int(np.count_nonzero(inb))

        labels = np.zeros(res_xy.shape[0], dtype=np.int32)
        labels[inb] = frame[yi[inb], xi[inb]].astype(np.int32)

        mapped_any_clone = int(np.count_nonzero(labels >= 2))
        mapped_dom = int(np.count_nonzero(labels == dominant))

        if mapped_dom == 0 and mask_dom_px > 0:
            dom_yx = np.argwhere(frame == dominant)
            sample_dom = dom_yx[:3].tolist()
            sample_res = np.column_stack((xi[:3], yi[:3])).tolist()
            print(
                f"[debug][day {day:02d}] res_n={res_xy.shape[0]}, inb_n={inb_n}, "
                f"mapped_any_clone={mapped_any_clone}, mapped_dom=0 | "
                f"mask_dom_px={mask_dom_px}, mask_any_clone_px={mask_any_clone_px} | "
                f"sample_mask_dom_yx={sample_dom}, sample_res_xy_rounded={sample_res}"
            )
        else:
            print(
                f"[debug][day {day:02d}] res_n={res_xy.shape[0]}, inb_n={inb_n}, "
                f"mapped_any_clone={mapped_any_clone}, mapped_dom={mapped_dom} | "
                f"mask_dom_px={mask_dom_px}, mask_any_clone_px={mask_any_clone_px}"
            )


def show_test_visualization(viz: dict, df: pd.DataFrame) -> None:
    angles = viz["angles_deg"]
    dots = viz["dots_xy"]
    clone_points = viz["clone_points"]
    sus_points = viz["sus_points"]
    min_angle = viz["min_angle_deg"]
    min_dist = viz["distance_px_at_min"]
    dot_x = viz["min_dot_x"]
    dot_y = viz["min_dot_y"]
    center_x = viz["reference_center_x"]
    center_y = viz["reference_center_y"]
    ref_radius = CIRCLE_DIAMETER_PX / 2.0

    res_tree = cKDTree(clone_points)
    dists, _ = res_tree.query(dots, k=1)
    counts = count_sensitive_within_variable_radii(dots, np.asarray(dists), sus_points)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(angles, counts, lw=1.6)
    ax1.scatter([min_angle], [np.min(counts)], s=40)
    ax1.set_xlabel("Angle (deg), 0=top clockwise")
    ax1.set_ylabel("Sensitive cells in dot-centered circle")
    ax1.set_title(
        f"{viz['plate']} {viz['replicate']} day {viz['day']} clone {viz['clone_id']}\n"
        f"minimum={int(np.min(counts))}, angle={min_angle:.1f} deg, d={min_dist:.1f}px"
    )
    ax1.grid(alpha=0.25)

    if sus_points.shape[0] > 12000:
        idx = np.random.default_rng(0).choice(sus_points.shape[0], size=12000, replace=False)
        sus_plot = sus_points[idx]
    else:
        sus_plot = sus_points
    ax2.scatter(sus_plot[:, 0], sus_plot[:, 1], s=1, alpha=0.15, label="sensitive")
    ax2.scatter(clone_points[:, 0], clone_points[:, 1], s=4, alpha=0.8, label="resistant clone")
    ref_circ = plt.Circle((center_x, center_y), ref_radius, fill=False, linewidth=1.0, linestyle="--")
    ax2.add_patch(ref_circ)
    ax2.scatter([center_x], [center_y], s=30, label="big-circle center")
    circ = plt.Circle((dot_x, dot_y), min_dist, fill=False, linewidth=1.5)
    ax2.add_patch(circ)
    ax2.scatter([dot_x], [dot_y], s=40, label="min dot")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [px]")
    ax2.set_ylabel("y [px]")
    ax2.legend(loc="best")
    ax2.set_title("Geometry sanity-check view (big reference + local min circle)")

    ts_full = df.sort_values("day").reset_index(drop=True)
    y = ts_full["confinement_thickness"].fillna(0.0).to_numpy()
    day0 = int(ts_full["day"].min())
    x_rel = ts_full["day"].to_numpy() - day0
    ax3.plot(x_rel, y, marker="o", lw=1.4)
    ax3.set_xlabel("Relative day (start=0)")
    ax3.set_ylabel("Confinement thickness")
    ax3.set_title(f"Confinement thickness over time\ndominant clone {viz['clone_id']}")
    ax3.set_xlim(left=0)
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def show_test_visualization_per_timepoint(io: ReplicateIO, df: pd.DataFrame) -> None:
    if df.empty:
        return

    mask_stack = tifffile.imread(io.clone_mask_path)
    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]

    n_frames = int(mask_stack.shape[0])
    day_to_frame = build_day_to_frame(n_frames)
    h, w = int(mask_stack.shape[1]), int(mask_stack.shape[2])
    center_x = float(w) / 2.0
    center_y = float(h) / 2.0
    ref_radius = CIRCLE_DIAMETER_PX / 2.0
    angles_deg, dots_xy = make_reference_dots(center_x=center_x, center_y=center_y)

    dom_vals = df["dominant_clone_id"].dropna().to_numpy()
    dominant_clone_id = int(dom_vals[0]) if dom_vals.size > 0 else None
    days_in_h5 = set(extract_days_in_h5(io.h5_path))
    df_by_day = {int(r["day"]): r for _, r in df.iterrows()}

    all_days = sorted(day_to_frame.keys())
    for day in all_days:
        t = day_to_frame[day]
        row = df_by_day.get(day, None)
        if day in days_in_h5:
            res_xy, sus_xy = load_day_positions(io.h5_path, day)
            labels = assign_resistant_cells_to_clone_labels(res_xy, mask_stack[t]) if res_xy.shape[0] else np.array([])
        else:
            res_xy = np.empty((0, 2), dtype=float)
            sus_xy = np.empty((0, 2), dtype=float)
            labels = np.array([], dtype=int)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        clone_points = np.empty((0, 2), dtype=float)
        if dominant_clone_id is not None and labels.size > 0:
            clone_points = res_xy[labels == dominant_clone_id]

        if dominant_clone_id is not None and clone_points.shape[0] > 0:
            res_tree = cKDTree(clone_points)
            dists, _ = res_tree.query(dots_xy, k=1)
            dists = np.asarray(dists, dtype=float)
            counts = count_sensitive_within_variable_radii(dots_xy, dists, sus_xy)
            min_idx = int(np.argmin(counts))
            min_angle = float(angles_deg[min_idx])
            min_dist = float(dists[min_idx])
            dot_x = float(dots_xy[min_idx, 0])
            dot_y = float(dots_xy[min_idx, 1])

            ax1.plot(angles_deg, counts, lw=1.6)
            ax1.scatter([min_angle], [int(counts[min_idx])], s=40)
            ax1.set_title(
                f"{io.plate} {io.sample} day {day} dominant clone {dominant_clone_id}\n"
                f"minimum={int(counts[min_idx])}, angle={min_angle:.1f} deg, d={min_dist:.1f}px"
            )
        else:
            mapped = int((labels >= 2).sum()) if labels.size > 0 else 0
            conf_val = np.nan if row is None else row["confinement_thickness"]
            ax1.text(
                0.5,
                0.5,
                f"No dominant-clone cells mapped\nfor day {day}\n(mapped resistant cells: {mapped})\n"
                f"confinement={conf_val}",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title(f"{io.plate} {io.sample} day {day}")
        ax1.set_xlabel("Angle (deg), 0=top clockwise")
        ax1.set_ylabel("Sensitive cells in dot-centered circle")
        ax1.grid(alpha=0.25)

        if sus_xy.shape[0] > 12000:
            idx = np.random.default_rng(0).choice(sus_xy.shape[0], size=12000, replace=False)
            sus_plot = sus_xy[idx]
        else:
            sus_plot = sus_xy
        ax2.scatter(sus_plot[:, 0], sus_plot[:, 1], s=1, alpha=0.15, label="sensitive")
        if clone_points.shape[0] > 0:
            ax2.scatter(clone_points[:, 0], clone_points[:, 1], s=4, alpha=0.8, label="resistant clone")
        ref_circ = plt.Circle((center_x, center_y), ref_radius, fill=False, linewidth=1.0, linestyle="--")
        ax2.add_patch(ref_circ)
        ax2.scatter([center_x], [center_y], s=30, label="big-circle center")
        if dominant_clone_id is not None and clone_points.shape[0] > 0:
            local_circ = plt.Circle((dot_x, dot_y), min_dist, fill=False, linewidth=1.5)
            ax2.add_patch(local_circ)
            ax2.scatter([dot_x], [dot_y], s=40, label="min dot")
        ax2.set_aspect("equal")
        ax2.set_xlabel("x [px]")
        ax2.set_ylabel("y [px]")
        ax2.legend(loc="best")
        ax2.set_title("Geometry sanity-check view (big reference + local min circle)")

        plt.tight_layout()
        plt.show()


def show_test_min_confinement_over_time(df: pd.DataFrame, plate: str, sample: str) -> None:
    """Single summary plot for test replicate: confinement thickness over all timepoints."""
    if df.empty:
        return

    ts = df.sort_values("day").reset_index(drop=True)
    y = pd.to_numeric(ts["confinement_thickness"], errors="coerce").fillna(0.0).to_numpy()
    x_day = ts["day"].to_numpy(dtype=int)

    if y.size == 0:
        return

    imin = int(np.argmin(y))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x_day, y, marker="o", linewidth=1.5)
    ax.scatter([x_day[imin]], [y[imin]], s=45)
    ax.set_xlabel("Day")
    ax.set_ylabel("Confinement thickness")
    ax.set_title(
        f"Min-confinement summary over time\n{plate} {sample} "
        f"(global min={int(y[imin])} at day {int(x_day[imin])})"
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def run_test(plate: str, sample: str) -> str:
    io = build_test_replicate(plate, sample)
    debug_dominant_clone_mapping(io)
    df, viz = analyze_replicate(io)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"Escape_ConfinementThickness_test_{plate}_{sample}.csv")
    df.to_csv(out_path, index=False)
    print(f"[save] Test CSV: {out_path} ({len(df)} rows)")
    if viz is not None:
        show_test_visualization(viz, df)
        show_test_visualization_per_timepoint(io, df)
        show_test_min_confinement_over_time(df, plate=plate, sample=sample)
    else:
        print("[warn] No visualization payload available (no valid clone/day found).")
    return out_path


def run_all() -> str:
    all_rows = []
    for plate in tqdm(PLATES, desc="Plates"):
        reps = discover_replicates_for_plate(plate)
        for io in tqdm(reps, desc=f"{plate} replicates", leave=False):
            try:
                df, _ = analyze_replicate(io)
            except Exception as e:
                print(f"[warn] {io.plate}/{io.sample} skipped: {e}")
                continue
            if not df.empty:
                all_rows.append(df)

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame(
            columns=[
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
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "Escape_ConfinementThickness_all_replicates.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[save] Full CSV: {out_path} ({len(out_df)} rows)")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute confinement thickness for resistant clones.")
    p.add_argument("--mode", choices=["test", "all"], default="all")
    p.add_argument("--plate", default=DEFAULT_TEST_PLATE, help="Used in test mode")
    p.add_argument("--sample", default=DEFAULT_TEST_SAMPLE, help="Used in test mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "test":
        run_test(args.plate, args.sample)
    else:
        run_all()
