#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TL;DR
-----
This script converts filled colony mask stacks into contour-only TIFF stacks.
The resulting outlines are lighter-weight geometric summaries of colony shape
than the original filled masks.

Detailed
--------
Inputs:
- Stacked susceptible/resistant/total colony mask TIFFs.

Outputs:
- Contour-only TIFF stacks in which only mask boundaries are retained.

How it works:
- Loads each colony mask stack frame by frame.
- Converts filled masks to binary occupancy.
- Uses boundary detection to extract the outer contour for each frame.
- Writes a new TIFF stack with the same time structure but contour-only
  geometry.

Pipeline role:
- Auxiliary geometry-conversion step for downstream visualization and contour-
  based inspection of colony shape.
"""

"""
get_contour_from_susrestotal_colony_masks.py

Boundary encoding:
  1 on boundary pixels, 0 elsewhere (uint8)

Per frame:
  - binarize (>0)
  - fill holes
  - find outer boundaries
"""

import os
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries
from tqdm import tqdm


# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------
BASE_DIR = r"D:\2025_08_CoreSeed_Large_Testdataset"

# Plates to process
PLATE_NAMES = ["Plate_1", "Plate_2", "Plate_3", "Plate_4", "Plate_5", "Plate_6", "Plate_7", "Plate_8"]

INPUT_SUBFOLDER  = "colony_masks_stacks"
OUTPUT_SUBFOLDER = "colony_masks_boundaries"

KINDS = ["res", "sus", "total"]
# ----------------------------------------------------------------------


def process_file(in_path: Path, out_path: Path) -> None:
    """Load a colony mask stack and write a boundary-only stack."""
    arr = tiff.imread(str(in_path))

    # Allow single-frame 2D files
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        print(f"    [skip] Unexpected shape={arr.shape} : {in_path.name}")
        return

    T, H, W = arr.shape
    out = np.zeros((T, H, W), dtype=np.uint8)

    # Frame-level progress bar
    for t in tqdm(
        range(T),
        desc=f"      frames ({in_path.stem})",
        leave=False,
        unit="frame",
    ):
        mask = arr[t] > 0
        mask = binary_fill_holes(mask)
        boundary = find_boundaries(mask, mode="outer")
        out[t, boundary] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(out_path),
        out,
        compression="deflate",
        photometric="minisblack",
    )


def main() -> None:
    base = Path(BASE_DIR)

    # Plate-level progress bar
    for plate in tqdm(PLATE_NAMES, desc="Plates", unit="plate"):
        plate_dir = base / plate
        in_dir = plate_dir / INPUT_SUBFOLDER
        out_dir = plate_dir / OUTPUT_SUBFOLDER
        out_dir.mkdir(parents=True, exist_ok=True)

        if not plate_dir.is_dir():
            print(f"[warn] Plate folder not found, skipping: {plate_dir}")
            continue
        if not in_dir.is_dir():
            print(f"[warn] Input folder not found, skipping: {in_dir}")
            continue

        print(f"\n=== {plate} ===")
        print(f"Reading : {in_dir}")
        print(f"Writing : {out_dir}")

        for kind in KINDS:
            pattern = f"*_{kind}_colony_masks.tif"
            files = sorted(in_dir.glob(pattern))

            if not files:
                print(f"  [info] No files for kind='{kind}'")
                continue

            print(f"  kind='{kind}'  n_files={len(files)}")

            # File-level progress bar
            for in_path in tqdm(
                files,
                desc=f"    {kind} files",
                unit="file",
                leave=False,
            ):
                out_name = in_path.stem + "_boundaries.tif"
                out_path = out_dir / out_name
                process_file(in_path, out_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
