"""
TL;DR
-----
This script builds susceptible, resistant, and total colony masks from the
per-day CellPose mask images. It also stacks those masks over time and aligns
them to the density-map canvas when needed.

Detailed
--------
Inputs:
- Per-day CellPose mask TIFFs for susceptible and resistant channels.
- Optional density maps used to define a common spatial canvas size.

Outputs:
- Per-day colony masks for susceptible, resistant, and total cell occupancy.
- Time-stacked colony mask TIFFs for each well and mask type.

How it works:
- Reads per-day mask TIFFs, binarizes them by channel, and pairs susceptible
  and resistant masks by well/day.
- Center-pads masks to the density-map canvas when that canvas exists so later
  scripts share one geometry.
- Builds total colony masks as the union of susceptible and resistant masks.
- Stacks all days for each well into consistent 3D TIFF volumes.

Pipeline role:
- Produces the colony-geometry masks used by contour extraction, clone
  tracking alignment, and several escape-related analyses.
"""

import argparse
import json
from functools import lru_cache
import os, glob, re, sys
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_paths import DATA_ROOT, EXP_RAW_DATA  # noqa: E402
from reproduction.progress import run_plates
from reproduction.file_checks import tiff_layout, fresh_outputs

PLATES = [f"Plate_{index}" for index in range(1, 9)]

# Set to True to regenerate existing mask TIFFs when running via PyCharm.
OVERWRITE = False


@lru_cache(maxsize=1)
def recorded_geometry():
    path = DATA_ROOT / 'Geometry' / 'experimental_canvases.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

# ---- small helper ----
def center_pad(img: np.ndarray, Ht: int, Wt: int):
    h, w = img.shape
    if h > Ht or w > Wt:
        raise ValueError(f"src {(h,w)} larger than target {(Ht,Wt)}")
    top  = (Ht - h) // 2
    left = (Wt - w) // 2
    out = np.zeros((Ht, Wt), dtype=img.dtype)
    out[top:top+h, left:left+w] = img
    return out

def make_colony_masks(experiment_path, plate,
                      mask_dir="czi_NPY",
                      output_dir="colony_masks",
                      density_kind="total"):
    """
    Build per-day SUS/RES/TOTAL colony masks from CellPose *_masks.tif,
    but FIRST align each day's mask to that day's density canvas size
    (Option B).
    """
    print(f"\n=== make_colony_masks: {plate} ===")
    plate_dir     = os.path.join(experiment_path, plate)
    input_folder  = os.path.join(plate_dir, mask_dir)
    output_folder = os.path.join(plate_dir, output_dir)
    dens_folder   = os.path.join(plate_dir, "density_maps")
    os.makedirs(output_folder, exist_ok=True)

    # original regex: day, well, channel (2=sus, 3=res)
    mask_regex = re.compile(r".*d(\d+)_.*_MIPs_(.+)_\[(2|3),\s*0\]_masks\.tif$")
    all_files  = glob.glob(os.path.join(input_folder, "*_masks.tif"))

    sus_map, res_map = {}, {}
    for fp in all_files:
        fn = os.path.basename(fp)
        m = mask_regex.match(fn)
        if not m:
            continue
        day = m.group(1).zfill(2)
        well = m.group(2)
        ch = m.group(3)
        key = (well, day)
        if ch == '2': sus_map[key] = fp
        else:         res_map[key] = fp

    keys = sorted(set(sus_map) | set(res_map))
    for (well, day) in keys:
        # load labels → binarize
        sus = (imread(sus_map[(well, day)]) > 0).astype(np.uint8) if (well, day) in sus_map else None
        res = (imread(res_map[(well, day)]) > 0).astype(np.uint8) if (well, day) in res_map else None
        if sus is None and res is None:
            continue
        if sus is None: sus = np.zeros_like(res, dtype=np.uint8)
        if res is None: res = np.zeros_like(sus, dtype=np.uint8)

        # --- NEW: align per-day to the density canvas of this day ---
        # --- OPTIONAL align per-day to density canvas size ---
        dens_name = f"{well}_{day}_{density_kind}_density.tif"
        dens_path = os.path.join(dens_folder, dens_name)

        if not os.path.exists(dens_path):
            # fallback: try any *_density for this well/day
            cand = glob.glob(os.path.join(dens_folder, f"{well}_{day}_*_density.tif"))
            dens_path = cand[0] if cand else None

        geometry = recorded_geometry()
        if geometry is not None:
            key = f'{plate}/{well}/{day}'
            if key not in geometry:
                raise ValueError(f'Missing declared canvas geometry: {key}')
            entry = geometry[key]
            for channel, image in (('2', sus), ('3', res)):
                expected = entry['channels'].get(channel)
                if expected is not None and tuple(expected['source_shape']) != image.shape:
                    raise ValueError(f'Source geometry differs from manifest: {key}, channel {channel}')
            Ht, Wt = entry['canvas_shape']
            sus = center_pad(sus, Ht, Wt)
            res = center_pad(res, Ht, Wt)
        elif dens_path is not None and os.path.exists(dens_path):
            Ht, Wt = imread(dens_path).shape
            sus = center_pad(sus, Ht, Wt)
            res = center_pad(res, Ht, Wt)
        else:
            # If density map is missing, keep mask size as-is (do NOT fail)
            # This keeps colony mask generation independent of density pipeline.
            pass

        total = ((sus + res) > 0).astype(np.uint8)

        base = f"{well}_{day}"
        outputs = [
            os.path.join(output_folder, f"{base}_{kind}_colony_mask.tif")
            for kind in ("sus", "res", "total")
        ]
        try:
            complete = (all(tiff_layout(path) == (1, *sus.shape) for path in outputs)
                        and fresh_outputs(outputs, [p for p in (sus_map.get((well, day)), res_map.get((well, day)), dens_path) if p]))
        except (OSError, ValueError):
            complete = False
        if not OVERWRITE and complete:
            print(f"  [skip] {well} d{day}: colony masks already exist (set OVERWRITE = True to replace).")
            continue
        imwrite(outputs[0], sus, compression="deflate")
        imwrite(outputs[1], res, compression="deflate")
        imwrite(outputs[2], total, compression="deflate")

def stack_colony_masks(experiment_path, plate,
                       input_dir="colony_masks",
                       output_dir="colony_masks_stacks"):
    """
    Stack per-day RES masks to <well>_res_colony_masks.tif.
    If sizes differ across days, CENTER-pad to the group max size
    (mirrors Density_Maps stacking).
    """
    plate_path = os.path.join(experiment_path, plate)
    in_folder  = os.path.join(plate_path, input_dir)
    out_folder = os.path.join(plate_path, output_dir)
    os.makedirs(out_folder, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_folder, "*_colony_mask.tif")))
    rgx = re.compile(r"(.+?)_(\d{2})_(res|sus|total)_colony_mask\.tif$", re.IGNORECASE)
    # Discover wells present
    wells = set()
    for f in files:
        m = rgx.match(os.path.basename(f))
        if m:
            wells.add(m.group(1))
    mask_kinds = ["res", "sus", "total"]

    for well in sorted(wells):
        kind_to_files = {}
        for kind in mask_kinds:
            tp_files = sorted(glob.glob(os.path.join(in_folder, f"{well}_??_{kind}_colony_mask.tif")))
            kind_to_files[kind] = tp_files

        # If nothing exists for this well, skip
        if all(len(v) == 0 for v in kind_to_files.values()):
            continue
        # Compute per-well target H/W across ALL kinds and ALL timepoints
        shapes = [tiff_layout(fp)[1:] for paths in kind_to_files.values() for fp in paths]
        Ht = max(shape[0] for shape in shapes)
        Wt = max(shape[1] for shape in shapes)

        # Write one stack per kind (only if that kind exists)
        for kind in mask_kinds:
            tp_files = kind_to_files[kind]
            if not tp_files:
                continue

            shape = (len(tp_files), Ht, Wt)
            out_path = os.path.join(out_folder, f"{well}_{kind}_colony_masks.tif")
            try:
                complete = tiff_layout(out_path) == shape and fresh_outputs([out_path], tp_files)
            except (OSError, ValueError):
                complete = False
            if not OVERWRITE and complete:
                print(f"  [skip] {os.path.basename(out_path)} already exists (set OVERWRITE = True to replace).")
                continue
            temporary = out_path + f'.tmp-{os.getpid()}.tif'
            try:
                frames = (center_pad(imread(fp), Ht, Wt).astype(np.uint8) for fp in tp_files)
                imwrite(temporary, frames, shape=shape, dtype=np.uint8,
                        compression="deflate", photometric='minisblack', maxworkers=1)
                if tiff_layout(temporary) != shape:
                    raise ValueError(f'Incomplete written stack: {temporary}')
                os.replace(temporary, out_path)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
            print(f"  [{kind}] Stacked {len(tp_files)} → {os.path.basename(out_path)}  shape={shape}")


def process_plate(plate, overwrite=False):
    global OVERWRITE
    OVERWRITE = overwrite
    exp = os.fspath(EXP_RAW_DATA)
    print(f"Processing {plate}")
    make_colony_masks(exp, plate)
    stack_colony_masks(exp, plate)
    print(f"Finished {plate}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plates", nargs="+", choices=PLATES, default=PLATES)
    parser.add_argument("--plate-workers", type=int,
                        default=int(os.environ.get("EICHE_FISCHER_PLATE_WORKERS", "1")))
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    args = parser.parse_args()
    if args.plate_workers < 1:
        parser.error("--plate-workers must be positive")
    run_plates(process_plate, args.plates, args.plate_workers,
               label="Colony preparation", overwrite=args.overwrite)


if __name__ == "__main__":
    main()
