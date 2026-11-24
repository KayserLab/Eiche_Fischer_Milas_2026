"""
TL;DR
-----
This script reconstructs total cell-position point clouds in overexposed
regions by adding synthetic points back into the measured data. It is the base
overexposure-correction branch used before density and clone-level analyses.

Detailed
--------
Inputs:
- HDF5 susceptible/resistant point clouds from the raw analysis pipeline.
- Overexposed-region masks and overexposed-cell count estimates.
- Parameters controlling Poisson-disc or FPS-style point placement.

Outputs:
- Corrected total point-cloud HDF5 files.
- Optional Poisson-only point additions saved separately for inspection.

How it works:
- Reads observed susceptible and resistant point clouds for each day.
- Reads the binary overexposed regions and the estimated number of missing
  cells in those regions.
- Reconstructs missing points inside overexposed areas, either by Poisson-disc
  sampling or a related placement strategy, while respecting existing geometry.
- Saves corrected total point clouds for downstream density and colony-mask
  workflows.

Pipeline role:
- Base point-cloud correction step that repairs the raw positional data before
  most spatial analyses begin.
"""

import os
import re
import numpy as np
import pandas as pd
import tifffile as tiff

from scipy.ndimage import (
    label as cc_label,
    binary_opening,
    binary_closing,
    generate_binary_structure,
)
from scipy.spatial import cKDTree


# ============================================================
# I/O helpers
# ============================================================

def pointcloud_read(filename: str, time: str, unit: str = "d"):
    """Reads y,x positions for susceptible and resistant cells from HDF5."""
    try:
        df_sus = pd.read_hdf(filename, key=f"sus/{unit}_{time}")
    except Exception:
        df_sus = None
    try:
        df_res = pd.read_hdf(filename, key=f"res/{unit}_{time}")
    except Exception:
        df_res = None
    return df_sus, df_res


def overexposed_cellcount_read(filename: str, time: str, unit: str = "d"):
    """Reads overexposed cell counts (sus,res) from *_overexposed.h5."""
    try:
        df_sus = pd.read_hdf(filename, key=f"sus/{unit}_{time}")
        sus_count = int(df_sus.iloc[0, 0])
    except Exception:
        sus_count = 0

    try:
        df_res = pd.read_hdf(filename, key=f"res/{unit}_{time}")
        res_count = int(df_res.iloc[0, 0])
    except Exception:
        res_count = 0

    print(f"Overexposed cell count (sus,res): {sus_count} {res_count}")
    return int(sus_count), int(res_count)


def load_overexposed_regions(filename: str):
    return tiff.imread(filename)


def total_pointcloud_save(
    df_sus: pd.DataFrame,
    df_res: pd.DataFrame,
    filename: str,
    time: str,
    unit: str = "d",
    suffix: str = "_total",
):
    """
    Saves combined pointclouds to <basename><suffix>.h5 with keys sus/d_<time>, res/d_<time>.
      suffix="_total"         -> *_total.h5
      suffix="_total_poisson" -> *_total_poisson.h5
    """
    out_h5 = os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1]
    df_sus.to_hdf(out_h5, key=f"sus/{unit}_{time}")
    df_res.to_hdf(out_h5, key=f"res/{unit}_{time}")


def poisson_overexp_save(
    poisson_sus_new: np.ndarray,
    poisson_res_new: np.ndarray,
    filename: str,
    time: str,
    unit: str = "d",
    suffix: str = "_poisson_overexp",
):
    """
    Saves ONLY the newly generated Poisson points to <basename>_poisson_overexp.h5
    under keys sus/d_<time>, res/d_<time>.
    """
    out_h5 = os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1]
    pd.DataFrame(poisson_sus_new, columns=["y", "x"]).to_hdf(out_h5, key=f"sus/{unit}_{time}")
    pd.DataFrame(poisson_res_new, columns=["y", "x"]).to_hdf(out_h5, key=f"res/{unit}_{time}")


# ============================================================
# Small utilities
# ============================================================

def _collapse_mask_to_2d(overexposed_regions):
    if overexposed_regions is None:
        return None
    return overexposed_regions.max(axis=0) if overexposed_regions.ndim > 2 else overexposed_regions


def _to_existing_array(masks_positions):
    """
    Return:
      existing: (E,2) float array in (y,x) if available
      masks_df: DataFrame with columns ['y','x']
    """
    if masks_positions is None:
        return np.empty((0, 2), dtype=float), pd.DataFrame(columns=["y", "x"])

    if isinstance(masks_positions, np.ndarray):
        arr = masks_positions.astype(float)
        df = pd.DataFrame(arr, columns=["y", "x"])
        return arr, df

    if isinstance(masks_positions, pd.DataFrame):
        if {"y", "x"}.issubset(set(masks_positions.columns)):
            arr = masks_positions[["y", "x"]].to_numpy(dtype=float)
            return arr, masks_positions[["y", "x"]].copy()

        arr = masks_positions.iloc[:, :2].to_numpy(dtype=float)
        df = masks_positions.copy()
        df.columns = ["y", "x"] + list(df.columns[2:])
        return arr, df[["y", "x"]].copy()

    return np.empty((0, 2), dtype=float), pd.DataFrame(columns=["y", "x"])


def remove_small_components(mask_bool: np.ndarray, min_size: int, connectivity: int = 2) -> np.ndarray:
    """Remove connected components smaller than min_size (in pixels)."""
    if min_size <= 1:
        return mask_bool

    struct = generate_binary_structure(2, 2 if connectivity == 2 else 1)
    lab, n = cc_label(mask_bool, structure=struct)
    if n == 0:
        return mask_bool

    counts = np.bincount(lab.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[counts >= min_size] = True
    keep[0] = False

    return keep[lab]


# ============================================================
# Allocation: Option B (min 1 per island, then remainder by area)
# ============================================================

def allocate_counts_by_area(areas: np.ndarray, N: int) -> np.ndarray:
    """Largest remainder allocation; guarantees sum == N."""
    areas = np.asarray(areas, dtype=float)
    if N <= 0 or areas.size == 0 or areas.sum() <= 0:
        return np.zeros_like(areas, dtype=int)

    raw = areas / areas.sum() * N
    base = np.floor(raw).astype(int)
    rem = raw - base

    missing = int(N - base.sum())
    if missing > 0:
        idx = np.argsort(-rem)
        base[idx[:missing]] += 1
    elif missing < 0:
        idx = np.argsort(rem)
        for i in idx:
            if missing == 0:
                break
            if base[i] > 0:
                base[i] -= 1
                missing += 1

    base[0] += int(N - base.sum())
    return base


def allocate_counts_min1_then_area(areas: np.ndarray, N: int) -> np.ndarray:
    """
    Option B allocator:
      - If N >= n_islands: give every island 1 first, then allocate remaining by area.
      - If N <  n_islands: give 1 to the N largest islands, 0 to the rest.
    Guarantees sum == N.
    """
    areas = np.asarray(areas, dtype=float)
    n = int(areas.size)

    if N <= 0 or n == 0:
        return np.zeros(n, dtype=int)

    if N < n:
        counts = np.zeros(n, dtype=int)
        idx = np.argsort(-areas)
        counts[idx[:N]] = 1
        return counts

    counts = np.ones(n, dtype=int)
    remaining = int(N - n)
    if remaining > 0:
        counts += allocate_counts_by_area(areas, remaining)

    diff = int(N - counts.sum())
    if diff != 0:
        counts[0] += diff
    return counts


# ============================================================
# Continuous Poisson-like sampling inside a set of pixels
# ============================================================

def sample_continuous_in_pixels(pixels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    if pixels is None or pixels.size == 0:
        return np.zeros((n, 2), dtype=float)

    idx = rng.integers(0, len(pixels), size=n, endpoint=False)
    base = pixels[idx].astype(float)
    jitter = rng.random((n, 2))
    return base + jitter


def greedy_poisson_filter(
    pts: np.ndarray,
    r: float,
    existing_tree: cKDTree | None = None,
    r_existing: float = 0.0,
) -> np.ndarray:
    if pts.size == 0:
        return pts

    if r <= 0:
        if existing_tree is not None and r_existing > 0:
            d, _ = existing_tree.query(pts, k=1)
            return pts[d >= r_existing]
        return pts

    cell = r / np.sqrt(2.0)
    inv = 1.0 / cell

    grid = {}
    accepted_pts = []

    def grid_key(p):
        return (int(p[0] * inv), int(p[1] * inv))

    r2 = r * r

    for p in pts:
        if existing_tree is not None and r_existing > 0:
            d, _ = existing_tree.query(p, k=1)
            if d < r_existing:
                continue

        gy, gx = grid_key(p)
        ok = True
        for ny in range(gy - 2, gy + 3):
            for nx in range(gx - 2, gx + 3):
                key = (ny, nx)
                if key not in grid:
                    continue
                for j in grid[key]:
                    q = accepted_pts[j]
                    dy = p[0] - q[0]
                    dx = p[1] - q[1]
                    if dy * dy + dx * dx < r2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            j = len(accepted_pts)
            accepted_pts.append(p)
            grid.setdefault((gy, gx), []).append(j)

    return np.asarray(accepted_pts, dtype=float)


def poisson_exact_continuous(
    pixels: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    r_start: float,
    r_min: float,
    relax_factor: float,
    oversample_factor: int,
    max_rounds: int,
    existing_tree: cKDTree | None = None,
    r_existing: float = 0.0,
) -> np.ndarray:
    if n_target <= 0:
        return np.empty((0, 2), dtype=float)

    r = float(r_start)
    best = np.empty((0, 2), dtype=float)

    for _round in range(int(max_rounds)):
        pool_n = int(max(n_target * oversample_factor, n_target + 50))
        pts = sample_continuous_in_pixels(pixels, pool_n, rng)
        rng.shuffle(pts, axis=0)

        acc = greedy_poisson_filter(pts, r=r, existing_tree=existing_tree, r_existing=r_existing)
        if acc.shape[0] > best.shape[0]:
            best = acc

        if acc.shape[0] >= n_target:
            return acc[:n_target]

        r *= float(relax_factor)
        if r < r_min:
            break

    out = best.copy()
    need = int(n_target - out.shape[0])
    if need > 0:
        fill = sample_continuous_in_pixels(pixels, need, rng)

        if existing_tree is not None and r_existing > 0:
            tries = 0
            kept = []
            while True:
                kept_arr = np.vstack(kept) if kept else np.empty((0, 2), dtype=float)
                if kept_arr.shape[0] >= need:
                    fill = kept_arr[:need]
                    break
                if tries >= 30:
                    break
                cand = sample_continuous_in_pixels(pixels, need, rng)
                d, _ = existing_tree.query(cand, k=1)
                good = cand[d >= r_existing]
                kept.append(good)
                tries += 1

        out = np.vstack([out, fill])

    return out[:n_target]


# ============================================================
# Poisson creator (GLOBAL vs LOCAL)
# ============================================================

def create_total_positions_poisson_continuous_exact(
    masks_positions,
    overexposed_regions,
    cell_number_overexposed,
    # Poisson among NEW points
    r_start=8.0,
    r_min=0.5,
    relax_factor=0.85,
    oversample_factor=20,
    max_rounds=12,
    seed=0,
    connectivity=2,
    debug_print=False,
    # Mask cleanup
    do_mask_cleanup=True,
    cleanup_open_iters=1,
    cleanup_close_iters=1,
    min_island_size=5,
    # NEW: sampling mode
    sampling_mode: str = "global",   # "global" or "local"
    # Optional blocking vs existing
    block_against_existing=False,
    r_existing=0.0,
):
    """
    sampling_mode:
      - "global": sample over the entire (cleaned) mask as one candidate set
      - "local" : sample per connected component (region) with Option-B allocation
    """
    sampling_mode = str(sampling_mode).lower().strip()
    if sampling_mode not in {"global", "local"}:
        raise ValueError("sampling_mode must be 'global' or 'local'")

    existing, masks_df = _to_existing_array(masks_positions)
    N = int(cell_number_overexposed)

    mask2d = _collapse_mask_to_2d(overexposed_regions)
    if mask2d is None or N <= 0:
        return masks_df, np.empty((0, 2), dtype=float)

    mask_bool = (mask2d > 0)

    if do_mask_cleanup:
        struct = generate_binary_structure(2, 2 if connectivity == 2 else 1)
        for _ in range(int(cleanup_open_iters)):
            mask_bool = binary_opening(mask_bool, structure=struct)
        for _ in range(int(cleanup_close_iters)):
            mask_bool = binary_closing(mask_bool, structure=struct)
        mask_bool = remove_small_components(mask_bool, min_size=int(min_island_size), connectivity=connectivity)

    if mask_bool.sum() == 0:
        new_pts = np.zeros((N, 2), dtype=float)
        combined = pd.concat([masks_df, pd.DataFrame(new_pts, columns=["y", "x"])], ignore_index=True)
        return combined, new_pts

    rng = np.random.default_rng(int(seed))

    existing_tree = None
    if block_against_existing and existing.shape[0] > 0 and float(r_existing) > 0:
        existing_tree = cKDTree(existing)

    # -----------------------
    # GLOBAL sampling
    # -----------------------
    if sampling_mode == "global":
        pixels = np.argwhere(mask_bool).astype(int)

        if debug_print:
            print(f"    [GLOBAL] N={N} pixels={int(pixels.shape[0])} "
                  f"cleanup={do_mask_cleanup} open={cleanup_open_iters} close={cleanup_close_iters} "
                  f"min_island_size={min_island_size}")

        new_pts = poisson_exact_continuous(
            pixels=pixels,
            n_target=N,
            rng=rng,
            r_start=float(r_start),
            r_min=float(r_min),
            relax_factor=float(relax_factor),
            oversample_factor=int(oversample_factor),
            max_rounds=int(max_rounds),
            existing_tree=existing_tree,
            r_existing=float(r_existing) if existing_tree is not None else 0.0,
        )

        if new_pts.shape[0] != N:
            raise RuntimeError(f"[GLOBAL] Expected N={N}, got {new_pts.shape[0]}")

        combined = pd.concat([masks_df, pd.DataFrame(new_pts, columns=["y", "x"])], ignore_index=True)
        return combined, new_pts

    # -----------------------
    # LOCAL sampling (per region)
    # -----------------------
    struct = generate_binary_structure(2, 2 if connectivity == 2 else 1)
    labeled, num = cc_label(mask_bool, structure=struct)

    islands = []
    areas = []
    for lab in range(1, num + 1):
        coords = np.argwhere(labeled == lab).astype(int)
        if coords.size == 0:
            continue
        islands.append(coords)
        areas.append(coords.shape[0])

    if not islands:
        new_pts = np.zeros((N, 2), dtype=float)
        combined = pd.concat([masks_df, pd.DataFrame(new_pts, columns=["y", "x"])], ignore_index=True)
        return combined, new_pts

    areas = np.asarray(areas, dtype=float)
    counts = allocate_counts_min1_then_area(areas, N)

    if debug_print:
        print(f"    [LOCAL] n_regions={len(islands)} N={N} allocated_sum={int(counts.sum())} "
              f"(min={counts.min()}, max={counts.max()})")

    all_new = []
    for i, (cand_pixels, ni) in enumerate(zip(islands, counts)):
        ni = int(ni)
        if ni <= 0:
            continue

        local_rng = np.random.default_rng(int(seed) + 1000 * i)

        pts_i = poisson_exact_continuous(
            pixels=cand_pixels,
            n_target=ni,
            rng=local_rng,
            r_start=float(r_start),
            r_min=float(r_min),
            relax_factor=float(relax_factor),
            oversample_factor=int(oversample_factor),
            max_rounds=int(max_rounds),
            existing_tree=existing_tree,
            r_existing=float(r_existing) if existing_tree is not None else 0.0,
        )
        all_new.append(pts_i)

    new_pts = np.vstack(all_new) if all_new else np.empty((0, 2), dtype=float)
    if new_pts.shape[0] != N:
        raise RuntimeError(f"[LOCAL] Expected total N={N}, got {new_pts.shape[0]}")

    combined = pd.concat([masks_df, pd.DataFrame(new_pts, columns=["y", "x"])], ignore_index=True)
    return combined, new_pts


# ============================================================
# Optional old FPS method (unchanged)
# ============================================================

def create_total_positions_fps(masks_positions, overexposed_regions, cell_number_overexposed):
    mask2d = _collapse_mask_to_2d(overexposed_regions)
    existing, masks_df = _to_existing_array(masks_positions)

    if mask2d is None:
        return masks_df

    candidates = np.argwhere(mask2d > 0)
    N = int(cell_number_overexposed)

    if candidates.size == 0 or N <= 0:
        return masks_df

    if existing.size:
        tree = cKDTree(existing)
        distances, _ = tree.query(candidates, k=1)
    else:
        idx0 = np.random.randint(len(candidates))
        distances = np.hypot(
            candidates[:, 0] - candidates[idx0, 0],
            candidates[:, 1] - candidates[idx0, 1],
        )

    chosen_idxs = []
    for _ in range(N):
        idx = int(np.argmax(distances))
        chosen_idxs.append(idx)
        dist_new = np.hypot(
            candidates[:, 0] - candidates[idx, 0],
            candidates[:, 1] - candidates[idx, 1],
        )
        distances = np.minimum(distances, dist_new)

    new_points = candidates[chosen_idxs]
    overexp_df = pd.DataFrame(new_points, columns=["y", "x"])
    return pd.concat([masks_df, overexp_df], ignore_index=True)


# ============================================================
# Plate discovery helpers
# ============================================================

def discover_plates(root_dir: str) -> list[str]:
    plates = []
    if not os.path.isdir(root_dir):
        return plates
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.lower().startswith("plate"):
            plates.append(name)
    return plates


def normalize_plates(plates) -> list[str]:
    if plates is None:
        return []
    if isinstance(plates, str):
        return [plates]
    return list(plates)


# ============================================================
# Main correction routine (single plate)
# ============================================================

def total_positions_correction(
    plate: str,
    directory_path: str,
    days=None,
    mode: str = "poisson",
    # Poisson params
    poisson_r_start: float = 8.0,
    poisson_r_min: float = 0.5,
    poisson_relax_factor: float = 0.85,
    poisson_oversample_factor: int = 20,
    poisson_max_rounds: int = 12,
    poisson_seed: int = 0,
    poisson_connectivity: int = 2,
    poisson_debug_print: bool = True,
    # Mask cleanup
    do_mask_cleanup: bool = True,
    cleanup_open_iters: int = 1,
    cleanup_close_iters: int = 1,
    min_island_size: int = 5,
    # NEW: global vs local sampling
    poisson_sampling_mode: str = "global",  # "global" or "local"
    # Existing blocking
    block_against_existing: bool = False,
    r_existing: float = 0.0,
):
    mode = mode.lower().strip()
    if mode not in {"fps", "poisson", "both"}:
        raise ValueError("mode must be one of: 'fps', 'poisson', 'both'")

    directory_path_to_use = os.path.join(directory_path, plate)
    hdf5_dir = os.path.join(directory_path_to_use, "hdf5")

    if not os.path.isdir(hdf5_dir):
        print(f"[SKIP] Plate '{plate}': missing folder: {hdf5_dir}")
        return

    filenames = [
        fn for fn in os.listdir(hdf5_dir)
        if fn.endswith(".h5")
        and "overexposed" not in fn
        and "_total" not in fn
        and "_total_poisson" not in fn
        and "_poisson_overexp" not in fn
    ]

    for file in filenames:
        print("\n=== Plate:", plate, "| File:", file)
        pointcloud_filepath = os.path.join(hdf5_dir, file)
        overexposed_cell_count_filepath = os.path.join(
            hdf5_dir, os.path.splitext(file)[0] + "_overexposed.h5"
        )

        well_identifier = "_".join(file.split("_")[2:]).split(".")[0]

        try:
            with pd.HDFStore(pointcloud_filepath, mode="r") as store:
                keys = store.keys()
        except Exception as e:
            print(f"Could not open {pointcloud_filepath}: {e}")
            continue

        present = set()
        for kkey in keys:
            m = re.match(r"^/(sus|res)/([A-Za-z]+)_(\d+)$", kkey)
            if m:
                present.add(m.group(3))

        if not present:
            print("  [SKIP] No timepoints found.")
            continue

        if days is None:
            days_to_process = sorted(present)
        else:
            requested = {str(d).zfill(2) for d in days}
            days_to_process = sorted(present & requested)

        for timepoint in days_to_process:
            print("day:", timepoint)

            overexp_img_dir = os.path.join(directory_path_to_use, "overexposed_images")
            sus_mask_path = os.path.join(overexp_img_dir, f"{well_identifier}_{timepoint}_sus_overexposed_img.tif")
            res_mask_path = os.path.join(overexp_img_dir, f"{well_identifier}_{timepoint}_res_overexposed_img.tif")

            if not os.path.exists(sus_mask_path):
                print(f"  [SKIP] Missing sus mask: {sus_mask_path}")
                continue

            overexposed_regions_sus = load_overexposed_regions(sus_mask_path)
            overexposed_regions_res = load_overexposed_regions(res_mask_path) if os.path.exists(res_mask_path) else None

            N_sus, N_res = overexposed_cellcount_read(overexposed_cell_count_filepath, timepoint, unit="d")
            masks_positions_sus, masks_positions_res = pointcloud_read(pointcloud_filepath, timepoint, unit="d")

            if mode in {"fps", "both"}:
                total_fps_sus = create_total_positions_fps(masks_positions_sus, overexposed_regions_sus, N_sus)
                total_fps_res = create_total_positions_fps(masks_positions_res, overexposed_regions_res, N_res)
                total_pointcloud_save(total_fps_sus, total_fps_res, pointcloud_filepath, time=timepoint, unit="d", suffix="_total")
                print(f"  [SAVED] FPS total -> {os.path.splitext(pointcloud_filepath)[0]}_total.h5")

            if mode in {"poisson", "both"}:
                total_poi_sus, sus_new = create_total_positions_poisson_continuous_exact(
                    masks_positions_sus, overexposed_regions_sus, N_sus,
                    r_start=poisson_r_start,
                    r_min=poisson_r_min,
                    relax_factor=poisson_relax_factor,
                    oversample_factor=poisson_oversample_factor,
                    max_rounds=poisson_max_rounds,
                    seed=poisson_seed,
                    connectivity=poisson_connectivity,
                    debug_print=poisson_debug_print,
                    do_mask_cleanup=do_mask_cleanup,
                    cleanup_open_iters=cleanup_open_iters,
                    cleanup_close_iters=cleanup_close_iters,
                    min_island_size=min_island_size,
                    sampling_mode=poisson_sampling_mode,   # <<< HERE
                    block_against_existing=block_against_existing,
                    r_existing=r_existing,
                )

                total_poi_res, res_new = create_total_positions_poisson_continuous_exact(
                    masks_positions_res, overexposed_regions_res, N_res,
                    r_start=poisson_r_start,
                    r_min=poisson_r_min,
                    relax_factor=poisson_relax_factor,
                    oversample_factor=poisson_oversample_factor,
                    max_rounds=poisson_max_rounds,
                    seed=poisson_seed + 77,
                    connectivity=poisson_connectivity,
                    debug_print=False,
                    do_mask_cleanup=do_mask_cleanup,
                    cleanup_open_iters=cleanup_open_iters,
                    cleanup_close_iters=cleanup_close_iters,
                    min_island_size=min_island_size,
                    sampling_mode=poisson_sampling_mode,   # <<< HERE
                    block_against_existing=block_against_existing,
                    r_existing=r_existing,
                )

                total_pointcloud_save(total_poi_sus, total_poi_res, pointcloud_filepath, time=timepoint, unit="d", suffix="_total_poisson")
                poisson_overexp_save(sus_new, res_new, pointcloud_filepath, time=timepoint, unit="d", suffix="_poisson_overexp")

                print(f"  [SAVED] Poisson total    -> {os.path.splitext(pointcloud_filepath)[0]}_total_poisson.h5")
                print(f"  [SAVED] Poisson new-only -> {os.path.splitext(pointcloud_filepath)[0]}_poisson_overexp.h5")
                print(f"  [CHECK] N_sus={N_sus} added_poisson_sus={len(sus_new)} | N_res={N_res} added_poisson_res={len(res_new)}")


# ============================================================
# Multi-plate wrapper
# ============================================================

def total_positions_correction_multi(plates, directory_path: str, days=None, **kwargs):
    if plates is None:
        plates_list = discover_plates(directory_path)
        if not plates_list:
            raise RuntimeError(f"No Plate* directories found in: {directory_path}")
        print("[INFO] Auto-detected plates:", plates_list)
    else:
        plates_list = normalize_plates(plates)

    for pl in plates_list:
        total_positions_correction(
            plate=pl,
            directory_path=directory_path,
            days=days,
            **kwargs,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    ROOT_DIR = r"D:\2025_08_CoreSeed_Large_Testdataset"

    # Plates:
    #   None -> auto-detect Plate* in ROOT_DIR
    #   or ["Plate_2","Plate_3"]
    PLATES = None

    # Days:
    #   None -> all days present per file
    #   or ["05","08"]
    DAYS = None

    MODE = "poisson"  # "poisson", "fps", "both"

    total_positions_correction_multi(
        plates=PLATES,
        directory_path=ROOT_DIR,
        days=DAYS,
        mode=MODE,

        # --- Poisson params
        poisson_r_start=15,
        poisson_r_min=4,
        poisson_relax_factor=0.85,
        poisson_oversample_factor=40,
        poisson_max_rounds=20,
        poisson_seed=0,
        poisson_connectivity=2,
        poisson_debug_print=True,

        # --- Mask cleanup
        do_mask_cleanup=True,
        cleanup_open_iters=1,
        cleanup_close_iters=1,
        min_island_size=5,

        # --- NEW: choose sampling mode
        poisson_sampling_mode="global",   # "global" or "local"

        # --- block vs existing
        block_against_existing=False,
        r_existing=2.0,
    )


if __name__ == "__main__":
    main()
