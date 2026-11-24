"""
TL;DR
-----
This script is the padding-aware variant of total-position correction. It
produces corrected point clouds both in the legacy geometry and in the
center-padded geometry used by later clone and escape analyses.

Detailed
--------
Inputs:
- The same HDF5 point clouds and overexposure information used by
  `total_positions_correction_comparison.py`.
- Colony-mask dimensions used to determine the required padding offsets.

Outputs:
- Legacy corrected total point clouds.
- Padded corrected total point clouds aligned to stacked colony/clone masks.
- Matching Poisson-overexposure helper files in both coordinate systems.

How it works:
- Imports the base correction logic from
  `total_positions_correction_comparison.py`.
- Computes per-well/per-day center-padding offsets from source and target mask
  geometries.
- Applies the base correction and then writes both unshifted and shifted
  coordinate outputs so older and newer downstream scripts can coexist.

Pipeline role:
- Canonical corrected-point-cloud branch for analyses that need all geometry to
  live on the same padded image canvas.
"""

import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff

from analysis.basic.microscopy import total_positions_correction_comparison as base

# Save both variants in one run:
# - legacy (no padding shift): keeps old downstream scripts working
# - padded: for clone-mask-aligned analyses (escape/confinement)
SUFFIX_TOTAL_LEGACY = "_total"
SUFFIX_TOTAL_PADDED = "_total_padded"
SUFFIX_TOTAL_POISSON_LEGACY = "_total_poisson"
SUFFIX_TOTAL_POISSON_PADDED = "_total_poisson_padded"
SUFFIX_POISSON_OVEREXP_LEGACY = "_poisson_overexp"
SUFFIX_POISSON_OVEREXP_PADDED = "_poisson_overexp_padded"


def _read_hw(path: str) -> Optional[Tuple[int, int]]:
    if not path or not os.path.exists(path):
        return None
    arr = tiff.imread(path)
    if arr.ndim >= 2:
        return int(arr.shape[-2]), int(arr.shape[-1])
    return None


def _padding_offsets_for_well_day(plate_dir: str, well: str, day: str) -> Tuple[int, int]:
    """
    Compute the exact center-padding offsets used by Make_susrestotal_Colony_Masks.py.
    top = (Ht - h) // 2, left = (Wt - w) // 2
    """
    stacks_dir = os.path.join(plate_dir, "colony_masks_stacks")
    perday_dir = os.path.join(plate_dir, "colony_masks")

    # Target canvas from stacked colony masks (same geometry used for clone tracking output).
    target_path = os.path.join(stacks_dir, f"{well}_res_colony_masks.tif")
    target = _read_hw(target_path)
    if target is None:
        target_path = os.path.join(stacks_dir, f"{well}_total_colony_masks.tif")
        target = _read_hw(target_path)
    if target is None:
        return 0, 0
    ht, wt = target

    # Source per-day mask shape before stacking.
    src_path = os.path.join(perday_dir, f"{well}_{day}_res_colony_mask.tif")
    src = _read_hw(src_path)
    if src is None:
        src_path = os.path.join(perday_dir, f"{well}_{day}_total_colony_mask.tif")
        src = _read_hw(src_path)
    if src is None:
        return 0, 0
    h, w = src

    top = (ht - h) // 2
    left = (wt - w) // 2
    return int(top), int(left)


def _shift_df_yx(df: Optional[pd.DataFrame], top: int, left: int) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    out = df.copy()
    if {"y", "x"}.issubset(out.columns):
        out["y"] = pd.to_numeric(out["y"], errors="coerce") + float(top)
        out["x"] = pd.to_numeric(out["x"], errors="coerce") + float(left)
    return out


def _shift_points_yx(arr: np.ndarray, top: int, left: int) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.empty((0, 2), dtype=float)
    out = np.asarray(arr, dtype=float).copy()
    out[:, 0] += float(top)   # y
    out[:, 1] += float(left)  # x
    return out


def total_positions_correction(
    plate: str,
    directory_path: str,
    days=None,
    mode: str = "poisson",
    poisson_r_start: float = 8.0,
    poisson_r_min: float = 0.5,
    poisson_relax_factor: float = 0.85,
    poisson_oversample_factor: int = 20,
    poisson_max_rounds: int = 12,
    poisson_seed: int = 0,
    poisson_connectivity: int = 2,
    poisson_debug_print: bool = True,
    do_mask_cleanup: bool = True,
    cleanup_open_iters: int = 1,
    cleanup_close_iters: int = 1,
    min_island_size: int = 5,
    poisson_sampling_mode: str = "global",
    block_against_existing: bool = False,
    r_existing: float = 0.0,
):
    mode = mode.lower().strip()
    if mode not in {"fps", "poisson", "both"}:
        raise ValueError("mode must be one of: 'fps', 'poisson', 'both'")

    plate_dir = os.path.join(directory_path, plate)
    hdf5_dir = os.path.join(plate_dir, "hdf5")
    if not os.path.isdir(hdf5_dir):
        print(f"[SKIP] Plate '{plate}': missing folder: {hdf5_dir}")
        return

    filenames = [
        fn
        for fn in os.listdir(hdf5_dir)
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
            overexp_img_dir = os.path.join(plate_dir, "overexposed_images")
            sus_mask_path = os.path.join(overexp_img_dir, f"{well_identifier}_{timepoint}_sus_overexposed_img.tif")
            res_mask_path = os.path.join(overexp_img_dir, f"{well_identifier}_{timepoint}_res_overexposed_img.tif")
            if not os.path.exists(sus_mask_path):
                print(f"  [SKIP] Missing sus mask: {sus_mask_path}")
                continue

            overexposed_regions_sus = base.load_overexposed_regions(sus_mask_path)
            overexposed_regions_res = base.load_overexposed_regions(res_mask_path) if os.path.exists(res_mask_path) else None

            n_sus, n_res = base.overexposed_cellcount_read(overexposed_cell_count_filepath, timepoint, unit="d")
            masks_positions_sus, masks_positions_res = base.pointcloud_read(pointcloud_filepath, timepoint, unit="d")

            top, left = _padding_offsets_for_well_day(plate_dir, well_identifier, timepoint)
            print(f"  [PAD] well={well_identifier} day={timepoint} -> top={top}, left={left}")

            if mode in {"fps", "both"}:
                total_fps_sus = base.create_total_positions_fps(masks_positions_sus, overexposed_regions_sus, n_sus)
                total_fps_res = base.create_total_positions_fps(masks_positions_res, overexposed_regions_res, n_res)
                total_fps_sus_pad = _shift_df_yx(total_fps_sus, top, left)
                total_fps_res_pad = _shift_df_yx(total_fps_res, top, left)

                base.total_pointcloud_save(
                    total_fps_sus, total_fps_res, pointcloud_filepath, time=timepoint, unit="d", suffix=SUFFIX_TOTAL_LEGACY
                )
                base.total_pointcloud_save(
                    total_fps_sus_pad, total_fps_res_pad, pointcloud_filepath, time=timepoint, unit="d", suffix=SUFFIX_TOTAL_PADDED
                )
                print(f"  [SAVED] FPS total legacy -> {os.path.splitext(pointcloud_filepath)[0]}{SUFFIX_TOTAL_LEGACY}.h5")
                print(f"  [SAVED] FPS total padded -> {os.path.splitext(pointcloud_filepath)[0]}{SUFFIX_TOTAL_PADDED}.h5")

            if mode in {"poisson", "both"}:
                total_poi_sus, sus_new = base.create_total_positions_poisson_continuous_exact(
                    masks_positions_sus,
                    overexposed_regions_sus,
                    n_sus,
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
                    sampling_mode=poisson_sampling_mode,
                    block_against_existing=block_against_existing,
                    r_existing=r_existing,
                )
                total_poi_res, res_new = base.create_total_positions_poisson_continuous_exact(
                    masks_positions_res,
                    overexposed_regions_res,
                    n_res,
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
                    sampling_mode=poisson_sampling_mode,
                    block_against_existing=block_against_existing,
                    r_existing=r_existing,
                )

                total_poi_sus_pad = _shift_df_yx(total_poi_sus, top, left)
                total_poi_res_pad = _shift_df_yx(total_poi_res, top, left)
                sus_new_pad = _shift_points_yx(sus_new, top, left)
                res_new_pad = _shift_points_yx(res_new, top, left)

                base.total_pointcloud_save(
                    total_poi_sus, total_poi_res, pointcloud_filepath, time=timepoint, unit="d", suffix=SUFFIX_TOTAL_POISSON_LEGACY
                )
                base.poisson_overexp_save(
                    sus_new, res_new, pointcloud_filepath, time=timepoint, unit="d", suffix=SUFFIX_POISSON_OVEREXP_LEGACY
                )
                base.total_pointcloud_save(
                    total_poi_sus_pad, total_poi_res_pad, pointcloud_filepath, time=timepoint, unit="d", suffix=SUFFIX_TOTAL_POISSON_PADDED
                )
                base.poisson_overexp_save(
                    sus_new_pad, res_new_pad, pointcloud_filepath, time=timepoint, unit="d", suffix=SUFFIX_POISSON_OVEREXP_PADDED
                )

                print(f"  [SAVED] Poisson total legacy     -> {os.path.splitext(pointcloud_filepath)[0]}{SUFFIX_TOTAL_POISSON_LEGACY}.h5")
                print(f"  [SAVED] Poisson total padded     -> {os.path.splitext(pointcloud_filepath)[0]}{SUFFIX_TOTAL_POISSON_PADDED}.h5")
                print(f"  [SAVED] Poisson new-only legacy  -> {os.path.splitext(pointcloud_filepath)[0]}{SUFFIX_POISSON_OVEREXP_LEGACY}.h5")
                print(f"  [SAVED] Poisson new-only padded  -> {os.path.splitext(pointcloud_filepath)[0]}{SUFFIX_POISSON_OVEREXP_PADDED}.h5")
                print(f"  [CHECK] N_sus={n_sus} added_poisson_sus={len(sus_new)} | N_res={n_res} added_poisson_res={len(res_new)}")


def total_positions_correction_multi(plates, directory_path: str, days=None, **kwargs):
    if plates is None:
        plates_list = base.discover_plates(directory_path)
        if not plates_list:
            raise RuntimeError(f"No Plate* directories found in: {directory_path}")
        print("[INFO] Auto-detected plates:", plates_list)
    else:
        plates_list = base.normalize_plates(plates)

    for pl in plates_list:
        total_positions_correction(
            plate=pl,
            directory_path=directory_path,
            days=days,
            **kwargs,
        )


def main():
    root_dir = r"D:\2025_08_CoreSeed_Large_Testdataset"
    plates = None
    days = None
    mode = "poisson"

    total_positions_correction_multi(
        plates=plates,
        directory_path=root_dir,
        days=days,
        mode=mode,
        poisson_r_start=15,
        poisson_r_min=4,
        poisson_relax_factor=0.85,
        poisson_oversample_factor=40,
        poisson_max_rounds=20,
        poisson_seed=0,
        poisson_connectivity=2,
        poisson_debug_print=True,
        do_mask_cleanup=True,
        cleanup_open_iters=1,
        cleanup_close_iters=1,
        min_island_size=5,
        poisson_sampling_mode="global",
        block_against_existing=False,
        r_existing=2.0,
    )


if __name__ == "__main__":
    main()
