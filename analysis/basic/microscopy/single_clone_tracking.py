"""
TL;DR
-----
This script assigns persistent clone IDs to resistant mask regions across time.
It turns frame-wise resistant occupancy into time-resolved clone trajectories
that later analyses can quantify.

Detailed
--------
Inputs:
- Resistant mask stacks for each well.
- Tracking hyperparameters such as DBSCAN seed radius, morphological cleanup,
  and maximum inter-frame linking distance.

Outputs:
- Time-stacked clone-label TIFFs.
- Optional merge logs recording clone-absorption events.

How it works:
- Cleans each frame morphologically and seeds initial clone identities in the
  first frame using connected components plus DBSCAN centroid grouping.
- Propagates clone IDs forward by overlap and spatial proximity after
  dilation-based linking.
- Resolves merge events by absorbing the smaller clone into the larger one and
  preserving a redirect map for lineage consistency.

Pipeline role:
- Core lineage-style preprocessing step that creates the clone masks required
  by clone-count, density-correlation, and escape analyses.
"""

# ----------------------------------------------------------------------
# USER  SETTINGS
# ----------------------------------------------------------------------
PLATES     = ["Plate_1", "Plate_2", "Plate_3", "Plate_4", "Plate_5", "Plate_6" ,"Plate_7", "Plate_8"]  # list of plate subdirectories to process
OVERWRITE = False  # Set True to replace existing clone-mask TIFFs and merge logs.

EPS        = 200         # DBSCAN grouping radius for seeds (frame 0)
SELEM_RAD  = 3           # morphological closing radius
MIN_SIZE   = 50          # remove objects smaller than this (px²) --> not used right now
LINK_DIST_PER_DAY = 150  # max pixel jump allowed per elapsed imaging day
IMAGING_DAY_OFFSETS = (1, 3, 5)

UNASSIGNED_ID = 1     # label for non-clone resistant regions
CLONE_START   = 2     # first genuine clone ID (2, 3, 4, …)
# ----------------------------------------------------------------------

import argparse
import os, time, csv, sys
from tqdm import tqdm
from skimage.morphology import closing, disk, remove_small_objects
from skimage.morphology import dilation       # kept for completeness
from skimage.segmentation import expand_labels
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
from pathlib import Path
import numpy as np
import tifffile as tiff

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_paths import EXP_RAW_DATA  # noqa: E402
from analysis.basic.microscopy.mask_processing import fill_holes_2d
from reproduction.progress import run_plates


ROOT_DIR = os.fspath(EXP_RAW_DATA)  # parent directory of all plates

# ---------------------  helper functions from previous script --------
def preprocess(slice_: np.ndarray, selem_radius: int, min_size: int) -> np.ndarray:
    mask = slice_ > 0
    mask = closing(mask, disk(selem_radius))
    mask = fill_holes_2d(mask)
    #mask = remove_small_objects(mask, min_size=min_size)
    return mask.astype(bool)


def day_from_frame_index(frame_index: int) -> int:
    """Return the experimental day for the 1, 3, 5, 8, ... imaging schedule."""
    return (int(frame_index) // 3) * 7 + IMAGING_DAY_OFFSETS[int(frame_index) % 3]


def link_distance_for_frame_pair(previous_frame: int, current_frame: int) -> int:
    """Scale the allowed link distance by the true elapsed time between frames."""
    day_gap = day_from_frame_index(current_frame) - day_from_frame_index(previous_frame)
    if day_gap <= 0:
        raise ValueError(f"Frames must be chronological; got {previous_frame} then {current_frame}.")
    return LINK_DIST_PER_DAY * day_gap


def build_initial_clones(mask0: np.ndarray, eps: float):
    """
    Seed clones in frame 0 with IDs 2, 3, … and return:
        labelled_img, {clone_id: centroid}
    """
    lbl0 = label(mask0)
    props0 = regionprops(lbl0)
    if not props0:
        return np.zeros_like(lbl0, np.uint16), {}

    centroids  = np.array([p.centroid for p in props0])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centroids)

    labelled   = np.zeros_like(lbl0, np.uint16)
    centroids_per_clone = {}
    next_id = CLONE_START

    for cl_id in np.unique(clustering.labels_):
        members = np.where(clustering.labels_ == cl_id)[0]
        for mi in members:
            labelled[lbl0 == props0[mi].label] = next_id
        centroids_per_clone[next_id] = centroids[members].mean(axis=0)
        next_id += 1

    return labelled, centroids_per_clone


def track(stack: np.ndarray,
          eps: float = EPS,
          selem_radius: int = SELEM_RAD,
          min_size: int = MIN_SIZE,
          storage=None):
    """
    Track clones; smaller clone is absorbed into the larger on merge.

    Labels
    ------
      0 background
      1 new / unassigned resistant regions
      2+ living clones (IDs redirected on absorption)
    """
    n_frames = stack.shape[0]
    if storage is None:
        labelled_stack = np.zeros(stack.shape, dtype=np.uint16)
    elif isinstance(storage, np.ndarray):
        if storage.shape != stack.shape or storage.dtype != np.uint16:
            raise ValueError('Tracking storage must match the input shape and uint16 labels')
        labelled_stack = storage
    else:
        labelled_stack = np.memmap(storage, mode='w+', dtype=np.uint16, shape=stack.shape)

    # ---------- initialise (frame 0) ----------
    mask0 = preprocess(stack[0], selem_radius, min_size)
    lbl0, centroids = build_initial_clones(mask0, eps)
    labelled_stack[0] = lbl0

    active_ids   = set(centroids.keys())          # living clones
    redirect     = {}                             # old_id ➔ new_id
    prev_lbl     = lbl0
    prev_area    = {cid: (lbl0 == cid).sum() for cid in active_ids}
    merge_events = []                             # rows for CSV

    t0 = time.perf_counter()

    # ---------- iterate frames ----------
    for t in tqdm(range(1, n_frames), desc="    frames", unit="frame", leave=False):
        # if t > 19:
        #     continue
        mask       = preprocess(stack[t], selem_radius, min_size)
        raw_lbl    = label(mask)
        frame_out  = np.zeros_like(raw_lbl, np.uint16)
        curr_area  = {cid: 0 for cid in active_ids}   # updated this frame

        # --- helper to resolve redirects ----------------------------------
        def root(cid):
            while cid in redirect:
                cid = redirect[cid]
            return cid
        # ------------------------------------------------------------------

        # dilate ONLY living clones (after redirect)
        tmp_prev = prev_lbl.copy()
        for old, new in redirect.items():
            tmp_prev[tmp_prev == old] = new
        clone_mask = np.isin(tmp_prev, list(active_ids))
        link_distance = link_distance_for_frame_pair(t - 1, t)
        dilated_prev = expand_labels(clone_mask.astype(np.uint16) * tmp_prev,
                                     distance=link_distance)

        for region in regionprops(raw_lbl):
            rmask = (raw_lbl == region.label)

            # ---- find overlaps (after redirect) ----
            overlap_ids = np.unique(prev_lbl[rmask])
            overlap_clones = {root(cid) for cid in overlap_ids if root(cid) >= CLONE_START}

            # ---- case 1: unique overlap with one clone -------------------
            if len(overlap_clones) == 1:
                cid = next(iter(overlap_clones))
                frame_out[rmask] = cid
                curr_area[cid] += region.area
                continue

            # ---- case 2: try dilated map ---------------------------------
            candidates = np.unique(dilated_prev[rmask])
            candidates = [root(c) for c in candidates if c >= CLONE_START]
            candidates = list(set(candidates))     # unique roots

            if not overlap_clones and len(candidates) == 1:
                cid = candidates[0]
                frame_out[rmask] = cid
                curr_area[cid] += region.area
                continue

            # ---- case 3: no clone contact → unassigned (1) ---------------
            if not overlap_clones and not candidates:
                frame_out[rmask] = UNASSIGNED_ID
                continue

            # ---- case 4: merge of ≥2 clones ------------------------------
            clones_involved = overlap_clones.union(candidates)
            if len(clones_involved) >= 2:
                # choose largest by previous-frame area
                sizes = {cid: prev_area.get(cid, 0) for cid in clones_involved}
                keep_id = max(sizes, key=sizes.get)
                lose_ids = [cid for cid in clones_involved if cid != keep_id]

                # redirect losers and log events
                for loser in lose_ids:
                    redirect[loser] = keep_id
                    if loser in active_ids:
                        active_ids.remove(loser)
                    merge_events.append((loser, keep_id, t))

                frame_out[rmask] = keep_id
                curr_area[keep_id] += region.area
                continue

            # ---- fallback (shouldn't happen) -----------------------------
            frame_out[rmask] = UNASSIGNED_ID

        # -------- rewrite any redirected labels in the frame -------------
        if redirect:
            for old, new in redirect.items():
                frame_out[frame_out == old] = new

        # -------- fill remaining resistant pixels as UNASSIGNED ----------
        frame_out[(raw_lbl > 0) & (frame_out == 0)] = UNASSIGNED_ID

        labelled_stack[t] = frame_out
        prev_lbl   = frame_out
        prev_area  = {cid: curr_area.get(cid, 0) or prev_area.get(cid, 0)
                      for cid in active_ids}

    print(f"      ⏱️  done in {(time.perf_counter()-t0)/60:.1f} min")

    return labelled_stack, merge_events


class TiffFrames:
    """Read one frame at a time; support old packed 3/4-frame TIFFs as well."""
    def __init__(self, path):
        from reproduction.file_checks import tiff_layout
        self.handle = tiff.TiffFile(path)
        self.shape = tiff_layout(path)
        self.packed = None

    def __getitem__(self, index):
        if len(self.handle.pages) == self.shape[0] and len(self.handle.pages[0].shape) == 2:
            return self.handle.pages[index].asarray()
        if self.packed is None:
            self.packed = self.handle.series[0].asarray().reshape(self.shape)
        return self.packed[index]

    def close(self):
        self.packed = None
        self.handle.close()

def save_merge_log(csv_path: Path, merge_events: list[tuple[int, int, int]]):
    """
    Save rows of (loser_id, winner_id, frame) to CSV.
    """
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["clone_id", "merged_into", "merge_frame"])
        w.writerows(merge_events)

# ---------------------------  batch loop over plates -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create resistant clone-mask stacks for one or more plates."
    )
    parser.add_argument(
        "--plates",
        nargs="+",
        choices=PLATES,
        help="Plates to process. Omit to process all plates.",
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                        help="Replace existing clone-mask outputs.")
    parser.add_argument("--plate-workers", type=int,
                        default=int(os.environ.get("EICHE_FISCHER_PLATE_WORKERS", "1")))
    args = parser.parse_args()
    if args.plate_workers < 1:
        parser.error("--plate-workers must be positive")
    return args


def process_plate(plate, overwrite=OVERWRITE):
    print(f"\n=== {plate} ===")
    plate_dir = Path(ROOT_DIR) / plate
    in_dir    = plate_dir / 'colony_masks_stacks'
    out_dir   = plate_dir / 'clone_masks_stacks'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"⚠️  {in_dir} not found – skipping.")
        return

    stacks = sorted(p for p in in_dir.iterdir()
                    if p.name.endswith('_res_colony_masks.tif'))

    if not stacks:
        print("   no *_res_colony_masks.tif found.")
        return

    for in_path in stacks:
        base_name = in_path.stem.replace('_colony_masks', '_clone_masks')
        out_stack = out_dir / f"{base_name}.tif"
        out_csv   = out_stack.with_suffix('.csv')

        print(f" [{plate}, PID {os.getpid()}] → {in_path.name}", flush=True)
        from reproduction.file_checks import clone_outputs_complete
        if not overwrite and clone_outputs_complete(in_path, out_stack, out_csv):
            print("      [skip] output already exists (set OVERWRITE = True or pass --overwrite to replace).")
            continue

        # save results (deflate compression keeps size small)
        temporary_stack = out_stack.with_name(f'.{out_stack.stem}.tmp-{os.getpid()}.tif')
        temporary_csv = out_csv.with_name(f'.{out_csv.name}.tmp-{os.getpid()}')
        scratch = out_stack.with_name(f'.{out_stack.stem}.frames-{os.getpid()}.dat')
        stack = TiffFrames(in_path)
        labelled = None
        try:
            labelled = np.memmap(scratch, mode='w+', dtype=np.uint16, shape=stack.shape)
            labelled, merge_info = track(stack, storage=labelled)
            tiff.imwrite(temporary_stack, labelled,
                         photometric='minisblack', compression='deflate', maxworkers=1)
            save_merge_log(temporary_csv, merge_info)
            os.replace(temporary_stack, out_stack)
            os.replace(temporary_csv, out_csv)
        finally:
            stack.close()
            if isinstance(labelled, np.memmap):
                labelled.flush()
                labelled._mmap.close()
            scratch.unlink(missing_ok=True)
            temporary_stack.unlink(missing_ok=True)
            temporary_csv.unlink(missing_ok=True)

        print(f"      💾 clone masks  → {out_stack.name}")
        print(f"      📄 merge log    → {out_csv.name}")

def main(plates=None, overwrite=OVERWRITE, plate_workers=1):
    run_plates(process_plate, plates or PLATES, plate_workers,
               label="Clone tracking", overwrite=overwrite)


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args.plates, overwrite=args.overwrite, plate_workers=args.plate_workers)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
