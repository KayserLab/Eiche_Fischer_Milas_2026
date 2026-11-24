#!/usr/bin/env python3
"""
TL;DR
-----
This script turns tracked clone masks and contour metadata into clone-level
trajectory summaries. Its paper-critical output is the raw clone count CSV used
by later density/growth analyses.

Detailed
--------
Inputs:
- Clone mask stacks, clone contour stacks, and per-clone tracking metadata
  produced by earlier preprocessing scripts.

Outputs:
- `*_clone_counts_raw.csv` tables used by downstream analysis.
- Additional convenience outputs such as truncated clone count tables,
  metadata CSVs, and optional SVG summaries.

How it works:
- Matches the relevant clone-mask, contour, and summary files for each well.
- Builds per-clone trajectories across frames/timepoints.
- Truncates trajectories after clone disappearance where needed and records
  both raw and processed views.
- Writes clone-level count summaries in a format later scripts can join with
  density, escape, and treatment information.

Pipeline role:
- Bridge between image-derived clone tracking and later figure-facing clone
  growth analyses.
"""

"""
Batch-Analyse: Wachstum resistenter Klone
----------------------------------------

Pro Well entstehen:
  * Plot  ........ <Plate>/plots_clones/<Well>_clones.svg
  * CSV   ........ <Well>_clone_counts.csv        (gekappt)
  * CSV raw ...... <Well>_clone_counts_raw.csv    (roh)
  * CSV meta ..... <Well>_clone_meta.csv          (Front-Distanz)

Zusätzlich pro Strategie ein Sammel-Plot.

IMPORTANT FIX (winner-survives merge logic)
-------------------------------------------
- Previously, the merge cutoff was applied to BOTH clone_id and merged_into.
  That truncates the "winner" clone even if other clones merge into it.
- Now we apply the cutoff ONLY to clone_id (the absorbed/disappearing clone),
  so merged_into keeps its identity after merges.
"""

# ----------------------------- USER --------------------------------
PLATES   = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6', 'Plate_7', 'Plate_8']
#PLATES   = ['Plate_3']


CBAR_MIN, CBAR_MAX = 0, 1200           # Farbskala 0-1200 px
# -------------------------------------------------------------------

import re, sys, warnings
from pathlib import Path
import numpy as np
import tifffile, h5py, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.data_paths import EXP_RAW_DATA, intermediate_dir

ROOT_DIR = EXP_RAW_DATA
OUTPUT_ROOT = intermediate_dir("experimental", "basic", "clone_tracking", create=False)


def find_matching_files(plate_dir: Path):
    cd, cntd, h5d, dens_maps_dir, dens_stacks_dir = (
        plate_dir / 'clone_masks_stacks',
        plate_dir / 'colony_masks_boundaries',
        plate_dir / 'hdf5',
        plate_dir / 'density_maps',
        plate_dir / 'density_stacks',
    )
    wells = {}
    for tif in cd.glob('*_res_clone_masks.tif'):
        well  = tif.stem.split('_res_clone_masks')[0]
        strat = well.split('_')[1].split('-')[0]

        # try to locate a matching stack like "A1_AT50-A1_total_density.tif"
        dens_stack = None
        if dens_stacks_dir.exists():
            # pattern: {WELL}_{STRAT}-{WELL}_total_density.tif
            for cand in dens_stacks_dir.glob(f"{well}_*-{well}_total_density.tif"):
                dens_stack = cand
                break

        files = dict(
            clone_stack = tif,
            contour     = cntd / f"{well}_total_colony_masks_boundaries.tif",
            h5_total    = h5d  / f"G418_a2a4_{well}_total_poisson.h5",
            merge_csv   = cd   / f"{well}_res_clone_masks.csv",
            dens_dir    = dens_maps_dir if dens_maps_dir.exists() else None,
            dens_stack  = dens_stack,
            strategy    = strat,
            well        = well
        )
        # keep the original strict checks for the 4 core inputs
        if all(files[k].exists() for k in ('clone_stack','contour','h5_total','merge_csv')):
            wells[well] = files
    return wells

def load_group_and_time_keys(h5_path: Path):
    with h5py.File(h5_path) as h5:
        grp = next((g for g in ('total','res','sus') if g in h5), next(iter(h5.keys())))
        keys = sorted(h5[grp].keys(), key=lambda k: int(re.findall(r'\d+', k)[0]))
    return grp, keys


def density_offsets(shape, paths, prefix, tkeys):
    """
    Compute (y,x) offsets per frame.
    Priority:
      1) If density_maps per-day files exist -> use them.
      2) Else if a multi-page density_stack exists -> use each page.
      3) Else -> return zeros.
    """
    H, W = shape
    offs = []

    # Case 1: per-day single images in density_maps
    if paths.get('dens_dir'):
        ok = True
        tmp = []
        for key in tkeys:
            day = re.findall(r'\d+', key)[0].zfill(2)
            fname = paths['dens_dir'] / f"{prefix}_{day}_res_density.tif"
            if not fname.exists():
                ok = False
                break
            img = tifffile.imread(fname)
            h, w = img.shape
            tmp.append(((H - h)//2, (W - w)//2))
        if ok:
            return tmp

    # Case 2: multi-page stack in density_stacks
    if paths.get('dens_stack') and Path(paths['dens_stack']).exists():
        stack = tifffile.imread(paths['dens_stack'])
        # If stack is (T,Hs,Ws) use per-frame sizes; else assume same as clone
        if stack.ndim == 3:
            T = min(len(tkeys), stack.shape[0])
            for t in range(T):
                h, w = stack[t].shape
                offs.append(((H - h)//2, (W - w)//2))
            # If more timepoints than pages, pad with last known offset
            while len(offs) < len(tkeys):
                offs.append(offs[-1] if offs else (0,0))
            return offs

    # Fallback: zeros
    return [(0,0)] * len(tkeys)


def first_merge_dict(csv_path: Path):
    """
    Build merge cutoff dictionary.

    NEW (winner-survives):
    - We only apply the first merge frame cutoff to r['clone_id'] (the absorbed clone).
    - We DO NOT apply cutoffs to r['merged_into'] (the surviving/winner clone).

    The returned values are "merge_frame" indices (int).
    """
    df = pd.read_csv(csv_path)
    d  = {}
    for _, r in df.iterrows():
        if 'merge_frame' not in r or 'clone_id' not in r:
            continue

        # Merge frame
        try:
            mf = int(r['merge_frame'])
        except Exception:
            continue

        # Absorbed clone id (only this one gets truncated)
        cid = r.get('clone_id', np.nan)
        if pd.notna(cid):
            try:
                cid = int(cid)
            except Exception:
                continue
            d[cid] = min(d.get(cid, mf), mf)

    return d


def truncate_after_two_zeros(arr: np.ndarray):
    z = np.flatnonzero((arr[:-1]==0)&(arr[1:]==0))
    return (z[0]+1) if z.size else len(arr)

# ---------------- Einzel-Analyse ----------------
def analyse_single_well(paths: dict, svg_out: Path, csv_trunc: Path, csv_raw: Path, csv_meta: Path):
    # Load clone mask and contour
    clone_stack   = tifffile.imread(paths['clone_stack'])
    contour_stack = (tifffile.imread(paths['contour']) == 1)
    H, W          = clone_stack.shape[1:]

    # HDF5 groups and time keys
    grp, tkeys = load_group_and_time_keys(paths['h5_total'])
    n_time     = len(tkeys)
    prefix     = "_".join(paths['well'].split('_')[:2])
    offs       = density_offsets((H, W), paths, prefix, tkeys)
    merge_lim  = first_merge_dict(paths['merge_csv'])

    # Distance map on day 1
    dist_map  = distance_transform_edt(~contour_stack[1])
    ids_all   = np.unique(clone_stack)
    ids_all   = ids_all[ids_all>=2]
    dist_front = {cid: np.median(dist_map[clone_stack[1]==cid]) for cid in ids_all if (clone_stack[1]==cid).any()}
    valid_ids = list(dist_front.keys())

    counts_raw = {cid: np.zeros(n_time, int) for cid in valid_ids}

    # Read HDF5 coords safely
    with h5py.File(paths['h5_total']) as h5:
        for t, key in enumerate(tkeys):
            raw = h5[f"{grp}/{key}/block0_values"][()]
            arr = np.asarray(raw)
            # If already shape (N,2), good
            if arr.ndim == 2 and arr.shape[1] == 2:
                coords = arr
            else:
                coords = arr.ravel()
                if coords.size < 2:
                    continue
                coords = coords.reshape(-1, 2)

            yy, xx = coords[:,0].astype(int), coords[:,1].astype(int)
            yy += offs[t][0];  xx += offs[t][1]
            inb = (yy>=0)&(yy<H)&(xx>=0)&(xx<W)
            lbl = clone_stack[t, yy[inb], xx[inb]]
            for cid in valid_ids:
                counts_raw[cid][t] += np.sum(lbl==cid)

    # ---------- Plot ----------
    cmap, norm = plt.cm.viridis_r, plt.Normalize(CBAR_MIN, CBAR_MAX)
    fig, ax = plt.subplots(figsize=(6,4), dpi=120)
    counts_trunc = {}
    for cid in valid_ids:
        vec_full = counts_raw[cid]
        cut      = truncate_after_two_zeros(vec_full)

        # Apply merge cutoff ONLY for absorbed clones (per first_merge_dict fix).
        # Important: treat cut as a LENGTH. If merge_frame==4, keep indices 0..3 (length 4).
        if cid in merge_lim:
            mf = int(merge_lim[cid])
            mf = max(0, min(mf, n_time))  # clamp to [0, n_time]
            cut = min(cut, mf)

        if cut == 0:
            continue

        vec_trunc = vec_full[:cut]
        ax.plot(range(cut), vec_trunc, lw=2.2, color=cmap(norm(dist_front[cid])))
        counts_trunc[cid] = vec_trunc

    print(f"\n[DEBUG] Well {paths['well']} — truncated counts check")
    for clabel, series in counts_raw.items():
        nz = np.count_nonzero(series)
        print(f"   Clone {clabel}: nonzero frames = {nz}")
        if nz <= 1:
            print("     -> Skipped (<=1 nonzero frame)")
        else:
            print("     -> Will be kept")

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('# Resistant cells')
    ax.set_title(paths['well'])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.04,
                 label='Initial distance to front (px)')
    fig.tight_layout()
    fig.savefig(svg_out)
    plt.close(fig)

    # ---------- CSVs ----------
    if counts_raw:
        df_raw  = pd.DataFrame({f"clone_{cid}": pd.Series(v) for cid,v in counts_raw.items()})
        df_raw.to_csv(csv_raw, index_label='day')

    if counts_trunc:
        df_t = pd.DataFrame({f"clone_{cid}": pd.Series(v) for cid,v in counts_trunc.items()})
        df_t.to_csv(csv_trunc, index_label='day')

    if not counts_trunc:
        print(f"[DEBUG] No clones passed filter for {paths['well']}")
    else:
        print(f"[DEBUG] {len(counts_trunc)} clones passed filter for {paths['well']}")

    meta = pd.DataFrame({
        "clone_id": list(dist_front.keys()),
        "distance_t1_px": [dist_front[c] for c in dist_front]
    })
    meta.to_csv(csv_meta, index=False)

    return {cid: (counts_trunc[cid], dist_front[cid]) for cid in counts_trunc}

# ---------------- Batch-Durchlauf ----------------
strategy_bucket = {}
for plate in PLATES:
    print(f"\n=== {plate} ===")
    pdir  = Path(ROOT_DIR)/plate
    wells = find_matching_files(pdir)
    if not wells:
        print("   keine Wells.")
        continue
    plate_output_dir = OUTPUT_ROOT / plate
    plate_output_dir.mkdir(parents=True, exist_ok=True)

    for well, p in wells.items():
        plot_dir = plate_output_dir
        print(f" → {well}")
        svg   = plot_dir/f"{well}_clones.svg"
        csv_t = plot_dir/f"{well}_clone_counts.csv"
        csv_r = plot_dir/f"{well}_clone_counts_raw.csv"
        csv_m = plot_dir/f"{well}_clone_meta.csv"
        try:
            data = analyse_single_well(p, svg, csv_t, csv_r, csv_m)
        except Exception as e:
            warnings.warn(f"{well}: skipped due to {e}")
            continue
        print("      Dateien gespeichert")
        bucket = strategy_bucket.setdefault(p['strategy'], [])
        for vec,dist in data.values():
            bucket.append((vec, dist))

# ---------------- Strategie-Plots ----------------
sdir = OUTPUT_ROOT / 'strategy_plots'; sdir.mkdir(parents=True, exist_ok=True)
cmap, norm = plt.cm.viridis_r, plt.Normalize(CBAR_MIN, CBAR_MAX)
for strat, items in strategy_bucket.items():
    print(f"\n*** Strategie {strat}: {len(items)} Klone ***")
    fig, ax = plt.subplots(figsize=(6,4), dpi=120)
    for vec, dist in items:
        ax.plot(range(len(vec)), vec, alpha=0.6, color=cmap(norm(dist)))
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('# Resistant cells')
    ax.set_title(f'{strat}: alle Klone aller Wells')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.04,
                 label='Initial distance to front (px)')
    fig.tight_layout()
    svg = sdir/f"{strat}_clones.svg"
    fig.savefig(svg)
    plt.close(fig)
    print(f"      Sammel-Plot gespeichert: {svg.name}")
