"""
TL;DR
-----
This script is the newer canonical escape-area builder. It reuses the original
escape-area machinery but feeds it boundary-aware inputs derived from the
cleaned susceptible masks.

Detailed
--------
Inputs:
- Cleaned susceptible mask stacks.
- Escape-front summaries/points and the base escape-area builder logic.

Outputs:
- `Escape_Fig_Build_escape_areas_timeseries_new.csv`
- `Escape_Fig_Build_escape_areas_summary_new.csv`

How it works:
- Loads the cleaned susceptible stacks for each replicate.
- Delegates most geometric processing to the original
  `Escape_Fig_Build_escape_areas.py` helpers.
- Rebuilds the escape-area timeseries using the newer, boundary-aware mask
  inputs so downstream Figure 2 analyses use the revised scientific branch.

Pipeline role:
- Canonical upstream source for the current Figure 2 escape-area analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from skimage.segmentation import find_boundaries

import Escape_Fig_Build_escape_areas as base


SENS_SUFFIX = "_sus_clean_masks.tif"


def _load_sensitive_stack(plate: str, sample: str, hw: tuple[int, int]) -> np.ndarray | None:
    path = base.DATA_ROOT / plate / "sensitive_clean_masks_stacks" / f"{sample}{SENS_SUFFIX}"
    if not path.exists():
        return None
    try:
        arr = base.load_stack(path)
    except Exception:
        return None
    if arr.ndim != 3:
        return None
    if tuple(arr.shape[1:]) != tuple(hw):
        return None
    return arr > 0


def main() -> None:
    reps = base.discover_replicates()
    print(f"[info] discovered replicates: {len(reps)}")
    print(
        f"[info] params angle_bins={base.ANGLE_BINS}, cluster_dist={base.CLUSTER_DIST_PX}, "
        f"dilate={base.POINT_DILATE_RADIUS_PX}, outward_p={base.OUTWARD_PERCENTILE}"
    )
    print("[info] NEW boundary-based method: colony boundary = resistant OR sensitive(clean), mode='inner'")

    out_frame = base.OUT_DIR / "Escape_Fig_Build_escape_areas_timeseries_new.csv"
    out_summary = base.OUT_DIR / "Escape_Fig_Build_escape_areas_summary_new.csv"
    already_done = base.load_already_processed(out_summary)
    if already_done:
        print(f"[info] resume mode: found {len(already_done)} already processed wells in {out_summary.name}")

    total_frame_rows_written = 0
    total_summary_rows_written = 0
    n_ok = 0
    n_skip = 0

    rep_iter = reps
    if base.tqdm is not None:
        rep_iter = base.tqdm(reps, total=len(reps), desc="Build escape areas NEW", unit="rep")

    for plate, sample, h5_path, mask_path, points_csv in rep_iter:
        if (plate, sample) in already_done:
            n_skip += 1
            print(f"[skip] {plate}/{sample}: already present in summary CSV")
            continue

        status, cutoff_day = base.sample_filter_rule(sample)
        if status == "drop":
            n_skip += 1
            print(f"[skip] {plate}/{sample}: excluded")
            continue

        mask_stack = base.load_stack(mask_path)
        if mask_stack.ndim != 3:
            n_skip += 1
            print(f"[skip] {plate}/{sample}: invalid mask shape {mask_stack.shape}")
            continue
        dom = base.find_dominant_clone_label(mask_stack)
        if dom is None:
            n_skip += 1
            print(f"[skip] {plate}/{sample}: no dominant clone")
            continue

        h, w = mask_stack.shape[1], mask_stack.shape[2]
        center = (w / 2.0, h / 2.0)
        points_by_t = base.load_points_by_t(points_csv)
        res_by_day = base.preload_res_points_by_day(h5_path, n_frames=mask_stack.shape[0])
        sens_stack = _load_sensitive_stack(plate=plate, sample=sample, hw=(h, w))

        esc_area_by_t: Dict[int, np.ndarray] = {}
        for t in range(mask_stack.shape[0]):
            esc_area_by_t[t] = base.escape_area_from_points(
                points_by_t.get(t, np.empty((0, 2), dtype=float)),
                shape_hw=mask_stack[t].shape,
            )

        prev_included_t = None
        sum_escape = 0.0
        sum_non_escape = 0.0
        all_escaped_flag = False
        n_frames_used = 0
        first_escape_day = np.nan
        sample_frame_rows: list[dict] = []

        for t in range(mask_stack.shape[0]):
            day = base.day_from_tidx(t)
            if cutoff_day is not None and day > cutoff_day:
                continue

            frame = mask_stack[t]
            res_mask = frame >= 2
            if sens_stack is not None and t < sens_stack.shape[0]:
                sen_mask = sens_stack[t]
            else:
                sen_mask = np.zeros_like(res_mask, dtype=bool)
            colony_mask = res_mask | sen_mask
            boundary_mask = find_boundaries(colony_mask, mode="inner")
            boundary_len = float(np.count_nonzero(boundary_mask))

            esc_area = esc_area_by_t[t]
            esc_len = float(np.count_nonzero(boundary_mask & esc_area))
            esc_frac = float(esc_len / boundary_len) if boundary_len > 0 else 0.0

            p = points_by_t.get(t, np.empty((0, 2), dtype=float))
            res_xy_now = res_by_day.get(day, np.empty((0, 2), dtype=float))
            res_count = base.dominant_res_count_all(frame, dom, res_xy_now)
            if np.isnan(first_escape_day) and len(p) > 0:
                first_escape_day = float(day)

            if prev_included_t is None:
                delta_per_day = np.nan
                escape_growth = np.nan
                non_escape_growth = np.nan
                cum_escape_frac = np.nan
                overlap_px = 0
                escape_zone_px = int(np.count_nonzero(esc_area))
                non_escape_px = int(np.count_nonzero((frame == dom) & (~esc_area)))
            else:
                prev_day = base.day_from_tidx(prev_included_t)
                days_delta = float(day - prev_day)
                frame_prev = mask_stack[prev_included_t]
                res_xy_prev = res_by_day.get(prev_day, np.empty((0, 2), dtype=float))
                prev_count = base.dominant_res_count_all(frame_prev, dom, res_xy_prev)
                delta_per_day = float((res_count - prev_count) / days_delta) if days_delta > 0 else np.nan

                esc_prev = esc_area_by_t[prev_included_t]
                esc_now = esc_area
                overlap_esc = esc_prev & esc_now
                escape_zone, _ = base.build_escape_effect_zone(esc_prev, esc_now)
                clone_prev = frame_prev == dom
                clone_now = frame == dom
                clone_union = clone_prev | clone_now
                non_escape_zone = clone_union & (~escape_zone)

                if all_escaped_flag or np.count_nonzero(non_escape_zone) == 0:
                    all_escaped_flag = True
                    non_escape_growth = 0.0
                else:
                    c0n = base.dominant_res_count_in_region(frame_prev, dom, res_xy_prev, non_escape_zone)
                    c1n = base.dominant_res_count_in_region(frame, dom, res_xy_now, non_escape_zone)
                    non_escape_growth = float(max(0, c1n - c0n))

                c0e = base.dominant_res_count_in_region(frame_prev, dom, res_xy_prev, escape_zone)
                c1e = base.dominant_res_count_in_region(frame, dom, res_xy_now, escape_zone)
                escape_growth = float(max(0, c1e - c0e))
                if all_escaped_flag:
                    c0all = base.dominant_res_count_in_region(frame_prev, dom, res_xy_prev, clone_union)
                    c1all = base.dominant_res_count_in_region(frame, dom, res_xy_now, clone_union)
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
                    "strategy": base.parse_strategy(sample),
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

        frac_total = (
            float(sum_escape / (sum_escape + sum_non_escape))
            if (sum_escape + sum_non_escape) > 0
            else np.nan
        )
        summary_row = {
            "plate": plate,
            "sample": sample,
            "strategy": base.parse_strategy(sample),
            "dom": int(dom),
            "n_frames_total": int(mask_stack.shape[0]),
            "n_frames_used": int(n_frames_used),
            "first_escape_day": first_escape_day,
            "sum_escape_growth_cells": float(sum_escape),
            "sum_non_escape_growth_cells": float(sum_non_escape),
            "escape_growth_fraction_total": frac_total,
        }

        written_frames = base.append_rows_csv(out_frame, sample_frame_rows)
        written_summary = base.append_rows_csv(out_summary, [summary_row])
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
    print(f"[done] build NEW finished: ok={n_ok}, skipped={n_skip}")


if __name__ == "__main__":
    main()
