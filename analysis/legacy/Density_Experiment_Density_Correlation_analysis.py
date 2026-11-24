"""
TL;DR
-----
This script quantifies the relationship between local density and resistant
clone expansion in the density experiment branch. It exports both trajectory-
level and clone-summary tables.

Detailed
--------
Inputs:
- Clone mask stacks and raw clone count tables for the density experiment.
- Local density measurements around each clone over time.

Outputs:
- A per-timepoint trajectory CSV relating local total density to absolute clone
  growth.
- A per-clone summary CSV relating initial local density to eventual clone
  expansion.

How it works:
- Identifies clones and builds ring masks around their boundaries.
- Measures local total density within those ring neighborhoods, optionally by
  angular sector.
- Computes absolute clone growth trajectories from raw clone count tables.
- Writes one time-resolved table and one clone-summary table for plotting.

Pipeline role:
- Analysis-side producer for the density-experiment panels combined in
  `figures/Figure_3/Density_Experiment_Density_Correlation_plot_both.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial.qhull import QhullError
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


def _log(msg: str) -> None:
    try:
        tqdm.write(msg)
    except Exception:
        print(msg)


# =========================
# Paths / dataset
# =========================
BASE_DIR = Path(r"F:\2026_01_Density_Experiment")
PLATES = ["Plate_1", "Plate_2"]

DENSITY_DIR_NAME = "density_stacks_parabolic_poisson"
LABELS_DIR_NAME = "clone_masks_stacks"
CLONE_COUNTS_DIR_NAME = "plots_clones"

OUT_TIME_CSV = BASE_DIR / r"Figures\Figure_2\Data\Density_Experiment_ring_total_density_vs_abs_clone_growth.csv"
OUT_CLONE_CSV = BASE_DIR / r"Figures\Figure_2\Data\Density_Experiment_initial_density_vs_max_clone_radius.csv"


# =========================
# Geometry / sectorization
# =========================
N_SECTORS = 12
RING_THICKNESSES_PX = [40, 200]
# Additional expensive ring to compute only for first timepoint.
RING_FIRST_TIMEPOINT_ONLY_PX = [500]
MIN_PIXELS_PER_SECTOR = 5
MIN_CLONE_AREA_PX = 10
LEGACY_RING_PX = 40


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_condition(sample: str) -> Optional[float]:
    m = re.search(r"^[A-Z]\d+_(\d+\.\d+)_\d+-[A-Z]\d+$", sample)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def centroids_over_time(mask_stack: np.ndarray) -> np.ndarray:
    tmax = mask_stack.shape[0]
    cc = np.full((tmax, 2), np.nan, dtype=np.float64)
    last_ok = None
    for t in range(tmax):
        ys, xs = np.nonzero(mask_stack[t])
        if ys.size == 0:
            if last_ok is not None:
                cc[t] = last_ok
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        cc[t] = (cy, cx)
        last_ok = cc[t]
    return cc


def build_ring_mask_from_distance(clone_mask_stack: np.ndarray, labels_stack: np.ndarray, ring_thick: int) -> np.ndarray:
    """
    Build annulus outside clone, excluding other clone pixels.
    """
    tmax, h, w = clone_mask_stack.shape
    ring = np.zeros((tmax, h, w), dtype=bool)
    for t in range(tmax):
        m = clone_mask_stack[t].astype(bool, copy=False)
        m_filled = binary_fill_holes(m)
        dist_out = distance_transform_edt(~m_filled)
        outer = (dist_out > 0) & (dist_out <= int(ring_thick))
        others = (labels_stack[t] > 0) & (~m_filled)
        ring[t] = outer & (~others)
    return ring


def build_ring_mask_single_timepoint(clone_mask_2d: np.ndarray, labels_2d: np.ndarray, ring_thick: int) -> np.ndarray:
    """
    Build annulus outside clone for one 2D frame, excluding other clone pixels.
    Used for expensive first-timepoint-only rings (e.g. 500 px).
    """
    m = clone_mask_2d.astype(bool, copy=False)
    m_filled = binary_fill_holes(m)
    dist_out = distance_transform_edt(~m_filled)
    outer = (dist_out > 0) & (dist_out <= int(ring_thick))
    others = (labels_2d > 0) & (~m_filled)
    return outer & (~others)


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


def clone_furthest_distance_px(mask_2d: np.ndarray) -> float:
    """
    Furthest in-clone pixel distance (diameter-like metric) for one timepoint.
    Uses convex hull points for speed, then exact max pairwise distance on hull.
    """
    ys, xs = np.nonzero(mask_2d)
    n = ys.size
    if n <= 1:
        return 0.0
    pts = np.column_stack([ys.astype(np.float64), xs.astype(np.float64)])
    if n == 2:
        d = pts[1] - pts[0]
        return float(np.hypot(d[0], d[1]))
    try:
        hull = ConvexHull(pts)
        hpts = pts[hull.vertices]
    except (QhullError, ValueError):
        # Degenerate point set (e.g., collinear) fallback to all points.
        hpts = pts
    if hpts.shape[0] <= 1:
        return 0.0
    return float(np.max(pdist(hpts, metric="euclidean")))


def load_clone_counts_raw(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    return df


def clone_growth_abs_series(counts_df: pd.DataFrame, clone_label: int) -> Optional[np.ndarray]:
    col = f"clone_{int(clone_label)}"
    if col not in counts_df.columns:
        return None
    vals = pd.to_numeric(counts_df[col], errors="coerce").to_numpy(dtype=float)
    if vals.size < 2:
        return None
    return vals[1:] - vals[:-1]


def iter_samples_for_plate(plate_dir: Path) -> list[str]:
    labels_dir = plate_dir / LABELS_DIR_NAME
    if not labels_dir.exists():
        return []
    samples = []
    for p in labels_dir.glob("*_res_clone_masks.tif"):
        sample = p.stem.replace("_res_clone_masks", "")
        samples.append(sample)
    return sorted(samples)


def process_sample(
    base: Path,
    plate: str,
    sample: str,
    compute_base_rings: bool,
    compute_first_timepoint_only_rings: bool,
) -> tuple[list[dict], list[dict]]:
    plate_dir = base / plate
    density_path = plate_dir / DENSITY_DIR_NAME / f"{sample}_total_density.tif"
    labels_path = plate_dir / LABELS_DIR_NAME / f"{sample}_res_clone_masks.tif"
    counts_csv = plate_dir / CLONE_COUNTS_DIR_NAME / f"{sample}_clone_counts_raw.csv"

    if not compute_base_rings and not compute_first_timepoint_only_rings:
        return [], []

    if not density_path.exists() or not labels_path.exists() or not counts_csv.exists():
        return [], []

    try:
        total_stack = tiff.imread(str(density_path))
        labels_stack = tiff.imread(str(labels_path))
    except Exception:
        return [], []

    if total_stack.ndim != 3 or labels_stack.ndim != 3:
        return [], []

    counts_df = load_clone_counts_raw(counts_csv)
    if counts_df is None:
        return [], []

    t_density = total_stack.shape[0]
    t_labels = labels_stack.shape[0]
    t_counts = len(counts_df.index)

    # Density and masks used at t; growth is computed from count(t+1)-count(t).
    paired_t = min(t_density, t_labels, t_counts - 1)
    if paired_t <= 0:
        return [], []

    total_t = total_stack[:paired_t]
    labels_t = labels_stack[:paired_t]

    labels_all = np.unique(labels_t)
    clone_labels = [int(v) for v in labels_all if int(v) >= 2]
    if not clone_labels:
        return [], []

    cond = sample_condition(sample)
    out_time_rows = []
    out_clone_rows = []

    for clabel in tqdm(
        clone_labels,
        desc=f"{plate}/{sample} clones",
        unit="clone",
        leave=False,
    ):
        growth_abs = clone_growth_abs_series(counts_df, clabel)
        if growth_abs is None and compute_base_rings:
            continue

        clone_mask_t = labels_t == int(clabel)
        per_t_area = np.sum(clone_mask_t, axis=(1, 2)).astype(float)
        per_t_furthest = np.asarray(
            [clone_furthest_distance_px(clone_mask_t[t]) for t in range(paired_t)],
            dtype=float,
        )
        max_area = int(np.max(per_t_area))
        if max_area < MIN_CLONE_AREA_PX:
            continue

        max_radius_px = float(np.sqrt(float(max_area) / np.pi))
        max_furthest_px = float(np.nanmax(per_t_furthest)) if per_t_furthest.size else np.nan

        cc = centroids_over_time(clone_mask_t)
        ring_by_thick = {}
        if compute_base_rings:
            ring_by_thick = {
                int(rt): build_ring_mask_from_distance(clone_mask_t, labels_t, int(rt)) for rt in RING_THICKNESSES_PX
            }
        ring500_t0 = None
        if compute_first_timepoint_only_rings and paired_t > 0:
            ring500_t0 = build_ring_mask_single_timepoint(clone_mask_t[0], labels_t[0], int(RING_FIRST_TIMEPOINT_ONLY_PX[0]))

        initial_median_by_ring: dict[int, float] = {
            int(rt): np.nan for rt in (RING_THICKNESSES_PX + RING_FIRST_TIMEPOINT_ONLY_PX)
        }
        ring200_density_series = np.full(paired_t, np.nan, dtype=float)

        for t in range(paired_t):
            cy, cx = cc[t]
            row = {
                "plate": plate,
                "sample": sample,
                "clone_label": int(clabel),
                "timepoint": int(t + 1),
            }
            if compute_base_rings:
                y_growth = float(growth_abs[t]) if (growth_abs is not None and t < len(growth_abs)) else np.nan
                row["condition_value"] = cond
                row["abs_resistant_growth"] = y_growth
                row["clone_area_px"] = float(per_t_area[t]) if t < len(per_t_area) else np.nan
                row["max_clone_area_px"] = float(max_area)
                row["max_clone_radius_px"] = max_radius_px

            if compute_base_rings:
                rings_to_compute = list(RING_THICKNESSES_PX)
            else:
                rings_to_compute = []
            if compute_first_timepoint_only_rings:
                rings_to_compute.extend(RING_FIRST_TIMEPOINT_ONLY_PX)

            for rt in rings_to_compute:
                if int(rt) in RING_FIRST_TIMEPOINT_ONLY_PX:
                    # Explicitly compute only at first timepoint to keep runtime bounded.
                    if t != 0:
                        row[f"ring{int(rt)}_total_density_mean_12sectors"] = np.nan
                        row[f"ring{int(rt)}_total_density_median_12sectors"] = np.nan
                        row[f"ring{int(rt)}_total_density_n_valid_sectors"] = 0
                        continue
                    ring2d = ring500_t0
                else:
                    ring2d = ring_by_thick[int(rt)][t]
                if not np.any(ring2d):
                    row[f"ring{int(rt)}_total_density_mean_12sectors"] = np.nan
                    row[f"ring{int(rt)}_total_density_median_12sectors"] = np.nan
                    row[f"ring{int(rt)}_total_density_n_valid_sectors"] = 0
                    continue

                sector_means = ring_sector_means_from_ring_pixels(
                    ring2d=ring2d,
                    cy=cy,
                    cx=cx,
                    frame_values=total_t[t],
                    n_sectors=N_SECTORS,
                    min_pixels=MIN_PIXELS_PER_SECTOR,
                )
                finite = np.isfinite(sector_means)
                if not np.any(finite):
                    row[f"ring{int(rt)}_total_density_mean_12sectors"] = np.nan
                    row[f"ring{int(rt)}_total_density_median_12sectors"] = np.nan
                    row[f"ring{int(rt)}_total_density_n_valid_sectors"] = 0
                else:
                    dens_mean = float(np.nanmean(sector_means))
                    dens_median = float(np.nanmedian(sector_means))
                    n_valid = int(np.sum(finite))
                    row[f"ring{int(rt)}_total_density_mean_12sectors"] = dens_mean
                    row[f"ring{int(rt)}_total_density_median_12sectors"] = dens_median
                    row[f"ring{int(rt)}_total_density_n_valid_sectors"] = n_valid
                    if int(rt) == 200:
                        ring200_density_series[t] = dens_median
                    if t == 0:
                        initial_median_by_ring[int(rt)] = dens_median

            if compute_base_rings:
                # Backward-compatible legacy columns (use ring 40).
                row["ring_total_density_mean_12sectors"] = row.get(
                    f"ring{LEGACY_RING_PX}_total_density_mean_12sectors", np.nan
                )
                row["ring_total_density_median_12sectors"] = row.get(
                    f"ring{LEGACY_RING_PX}_total_density_median_12sectors", np.nan
                )
                row["ring_total_density_n_valid_sectors"] = row.get(
                    f"ring{LEGACY_RING_PX}_total_density_n_valid_sectors", 0
                )
                row["clone_furthest_distance_px"] = float(per_t_furthest[t]) if t < len(per_t_furthest) else np.nan

            out_time_rows.append(row)

        # Per-clone threshold selection on ring200 local density:
        # first timepoint where ring200 density >= 90% of that clone's ring200 max.
        dens_valid = np.isfinite(ring200_density_series)
        if np.any(dens_valid):
            dens_max = float(np.nanmax(ring200_density_series))
            dens_thr90 = 0.9 * dens_max
            hit = np.where(np.isfinite(ring200_density_series) & (ring200_density_series >= dens_thr90))[0]
            t_sel = int(hit[0]) if hit.size > 0 else int(paired_t - 1)
        else:
            dens_max = np.nan
            dens_thr90 = np.nan
            t_sel = int(paired_t - 1)
        t_max_area = int(np.nanargmax(per_t_area)) if per_t_area.size else 0
        t_max_furthest = int(np.nanargmax(per_t_furthest)) if per_t_furthest.size else 0

        clone_row = {
            "plate": plate,
            "sample": sample,
            "clone_label": int(clabel),
            "initial_density_median_ring500": float(initial_median_by_ring.get(500, np.nan)),
            "ring500_first_timepoint_computed": 1 if compute_first_timepoint_only_rings else np.nan,
            "ring200_max_local_density_median_12sectors": dens_max,
            "ring200_threshold90_local_density": dens_thr90,
            "timepoint_first_density90_ring200": int(t_sel + 1),
            "area_at_first_density90_ring200_px": float(per_t_area[t_sel]) if per_t_area.size else np.nan,
            "furthest_distance_at_first_density90_ring200_px": float(per_t_furthest[t_sel]) if per_t_furthest.size else np.nan,
            "timepoint_max_clone_area": int(t_max_area + 1),
            "timepoint_max_clone_furthest_distance": int(t_max_furthest + 1),
            "max_clone_furthest_distance_px": max_furthest_px,
        }
        if compute_base_rings:
            clone_row["condition_value"] = cond
            clone_row["initial_density_median_ring40"] = float(initial_median_by_ring.get(40, np.nan))
            clone_row["initial_density_median_ring200"] = float(initial_median_by_ring.get(200, np.nan))
            clone_row["max_clone_area_px"] = float(max_area)
            clone_row["max_clone_radius_px"] = max_radius_px
        out_clone_rows.append(clone_row)

    return out_time_rows, out_clone_rows


def _upsert_by_key(existing: pd.DataFrame, new: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if new is None or new.empty:
        return existing.copy()
    if existing is None or existing.empty:
        out = new.copy()
        return out.reset_index(drop=True)

    ex = existing.copy().set_index(key_cols)
    nw = new.copy().set_index(key_cols)
    out = ex.combine_first(nw)
    out = out.combine_first(nw)
    return out.reset_index()


def main() -> None:
    ensure_dir(OUT_TIME_CSV.parent)
    _log(
        "[INFO] Requested rings: base="
        f"{RING_THICKNESSES_PX}, first-timepoint-only={RING_FIRST_TIMEPOINT_ONLY_PX}"
    )

    existing_time_df = pd.DataFrame()
    existing_clone_df = pd.DataFrame()
    if OUT_TIME_CSV.exists():
        try:
            existing_time_df = pd.read_csv(OUT_TIME_CSV)
        except Exception:
            existing_time_df = pd.DataFrame()
    if OUT_CLONE_CSV.exists():
        try:
            existing_clone_df = pd.read_csv(OUT_CLONE_CSV)
        except Exception:
            existing_clone_df = pd.DataFrame()
    _log(
        "[INFO] Existing outputs: "
        f"time_rows={len(existing_time_df)}, clone_rows={len(existing_clone_df)}"
    )

    time_rows = []
    clone_rows = []
    n_samples = 0
    n_skipped = 0
    for plate in tqdm(PLATES, desc="Plates", unit="plate"):
        plate_dir = BASE_DIR / plate
        samples = iter_samples_for_plate(plate_dir)
        _log(f"[INFO] {plate}: discovered {len(samples)} samples in {LABELS_DIR_NAME}")
        for sample in tqdm(samples, desc=f"{plate} samples", unit="sample", leave=False):
            n_samples += 1
            ex_time_sample = pd.DataFrame()
            if not existing_time_df.empty and {"plate", "sample"}.issubset(existing_time_df.columns):
                ex_time_sample = existing_time_df[
                    (existing_time_df["plate"] == plate) & (existing_time_df["sample"] == sample)
                ]
            ex_clone_sample = pd.DataFrame()
            if not existing_clone_df.empty and {"plate", "sample"}.issubset(existing_clone_df.columns):
                ex_clone_sample = existing_clone_df[
                    (existing_clone_df["plate"] == plate) & (existing_clone_df["sample"] == sample)
                ]
            has_base_for_sample = (
                (not ex_time_sample.empty)
                and (not ex_clone_sample.empty)
                and {
                    "ring40_total_density_median_12sectors",
                    "ring200_total_density_median_12sectors",
                    "abs_resistant_growth",
                    "clone_area_px",
                    "clone_furthest_distance_px",
                }.issubset(
                    set(ex_time_sample.columns)
                )
                and {"initial_density_median_ring40", "initial_density_median_ring200", "max_clone_radius_px"}.issubset(
                    set(ex_clone_sample.columns)
                )
            )
            has_500_for_sample = (
                (not ex_clone_sample.empty)
                and ("ring500_first_timepoint_computed" in ex_clone_sample.columns)
                and ex_clone_sample["ring500_first_timepoint_computed"].fillna(0).astype(float).gt(0).all()
            )

            compute_base = not has_base_for_sample
            compute_500 = not has_500_for_sample
            if not compute_base and not compute_500:
                n_skipped += 1
                _log(f"[SKIP] {plate}/{sample}: base(40/200)=done, ring500@t1=done")
                continue
            _log(
                f"[RUN] {plate}/{sample}: "
                f"compute_base(40/200)={'yes' if compute_base else 'no'}, "
                f"compute_ring500_t1={'yes' if compute_500 else 'no'}"
            )

            sample_time_rows, sample_clone_rows = process_sample(
                BASE_DIR,
                plate,
                sample,
                compute_base_rings=compute_base,
                compute_first_timepoint_only_rings=compute_500,
            )
            if sample_time_rows:
                time_rows.extend(sample_time_rows)
            if sample_clone_rows:
                clone_rows.extend(sample_clone_rows)
            _log(
                f"[DONE] {plate}/{sample}: "
                f"new_time_rows={len(sample_time_rows)}, new_clone_rows={len(sample_clone_rows)}"
            )

    new_time_df = pd.DataFrame(time_rows)
    new_clone_df = pd.DataFrame(clone_rows)

    time_df = _upsert_by_key(existing_time_df, new_time_df, ["plate", "sample", "clone_label", "timepoint"])
    clone_df = _upsert_by_key(existing_clone_df, new_clone_df, ["plate", "sample", "clone_label"])

    if time_df.empty:
        raise RuntimeError("No rows available after merge. Check paths and prerequisite outputs.")

    # Enforce global ring200 threshold logic:
    # threshold = 0.9 * global maximum ring200 local density across all clone-timepoints.
    needed = {
        "plate",
        "sample",
        "clone_label",
        "timepoint",
        "ring200_total_density_median_12sectors",
        "clone_area_px",
        "clone_furthest_distance_px",
    }
    if needed.issubset(set(time_df.columns)):
        tsel = time_df.copy()
        tsel["ring200_total_density_median_12sectors"] = pd.to_numeric(
            tsel["ring200_total_density_median_12sectors"], errors="coerce"
        )
        tsel["clone_area_px"] = pd.to_numeric(tsel["clone_area_px"], errors="coerce")
        tsel["clone_furthest_distance_px"] = pd.to_numeric(tsel["clone_furthest_distance_px"], errors="coerce")
        tsel["timepoint"] = pd.to_numeric(tsel["timepoint"], errors="coerce")
        tsel = tsel.dropna(
            subset=[
                "plate",
                "sample",
                "clone_label",
                "timepoint",
                "ring200_total_density_median_12sectors",
                "clone_area_px",
                "clone_furthest_distance_px",
            ]
        ).copy()
        if not tsel.empty:
            global_max = float(np.nanmax(tsel["ring200_total_density_median_12sectors"].to_numpy(dtype=float)))
            global_thr90 = 0.9 * global_max

            rows = []
            for (plate, sample, clabel), sub in tsel.groupby(["plate", "sample", "clone_label"], sort=False):
                sub = sub.sort_values("timepoint")
                dens = sub["ring200_total_density_median_12sectors"].to_numpy(dtype=float)
                area = sub["clone_area_px"].to_numpy(dtype=float)
                fur = sub["clone_furthest_distance_px"].to_numpy(dtype=float)
                tpt = sub["timepoint"].to_numpy(dtype=float)
                hit = np.where(np.isfinite(dens) & (dens >= global_thr90))[0]
                idx = int(hit[0]) if hit.size > 0 else int(len(sub) - 1)
                rows.append(
                    {
                        "plate": plate,
                        "sample": sample,
                        "clone_label": int(clabel),
                        "ring200_threshold90_local_density": float(global_thr90),
                        "ring200_global_max_local_density_median_12sectors": float(global_max),
                        "timepoint_first_density90_ring200": int(tpt[idx]),
                        "area_at_first_density90_ring200_px": float(area[idx]),
                        "furthest_distance_at_first_density90_ring200_px": float(fur[idx]),
                    }
                )
            gdf = pd.DataFrame(rows)
            if not gdf.empty:
                clone_df = _upsert_by_key(clone_df, gdf, ["plate", "sample", "clone_label"])
                _log(
                    f"[INFO] Global ring200 threshold applied: "
                    f"0.9*max={global_thr90:.4f} (max={global_max:.4f})"
                )
    else:
        _log("[WARN] Missing columns for global ring200 threshold selection; keeping existing per-clone values.")

    time_df = time_df.sort_values(["plate", "sample", "clone_label", "timepoint"]).reset_index(drop=True)
    clone_df = clone_df.sort_values(["plate", "sample", "clone_label"]).reset_index(drop=True)
    time_df.to_csv(OUT_TIME_CSV, index=False)
    clone_df.to_csv(OUT_CLONE_CSV, index=False)

    print(f"[OK] samples scanned: {n_samples}")
    print(f"[OK] samples skipped (already complete): {n_skipped}")
    print(f"[OK] time rows written: {len(time_df)}")
    print(f"[OK] clone rows written: {len(clone_df)}")
    print(f"[OK] saved: {OUT_TIME_CSV}")
    print(f"[OK] saved: {OUT_CLONE_CSV}")


if __name__ == "__main__":
    main()
