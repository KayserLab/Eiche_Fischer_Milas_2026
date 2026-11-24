#!/usr/bin/env python3
"""
TL;DR
-----
This script converts density-map stacks into day-to-day change stacks. It
computes both absolute and relative density changes so later scripts can work
with local growth or decline rather than only absolute density.

Detailed
--------
Inputs:
- Time-ordered density TIFF stacks produced by `Density_Maps_Epanechnikov.py`.

Outputs:
- Absolute change stacks (`*_abs_diff.tif`).
- Relative change stacks (`*_rel_diff.tif`).

How it works:
- Iterates over density TIFF stacks plate by plate.
- Computes consecutive-frame differences in a streaming manner so the whole
  stack does not need to be duplicated in memory.
- Writes separate outputs for absolute difference and relative difference,
  preserving the time ordering of the input density stack.

Pipeline role:
- Derivative preprocessing stage that exposes density dynamics for later
  analysis and plotting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import tifffile as tiff

# =====================
# CONFIG (edit here)
# =====================
BASE_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Data\exp_raw_data")
PLATES = ["Plate_1", "Plate_2", "Plate_3", "Plate_4", "Plate_5", "Plate_6", "Plate_7", "Plate_8"]

DO_RELATIVE = True
DO_ABSOLUTE = True

# Restrict to specific channels in filenames; set to None to process all.
# Valid values: {"sus", "res", "total"}
ONLY_CHANNELS: Optional[Iterable[str]] = None

ABS_DTYPE = np.int16
TIF_COMPRESSION = "deflate"
# =====================


_chan_pat = re.compile(r"_(sus|res|total)_")


def _list_tiffs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"})


def _keep_file(name: str, only_channels: Optional[Iterable[str]]) -> bool:
    if not only_channels:
        return True
    m = _chan_pat.search(name)
    return bool(m and m.group(1) in set(only_channels))


def _write_diffs_streaming(in_path: Path, out_path: Path, relative: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    with tiff.TiffFile(str(in_path)) as tf:
        if len(tf.series) > 0 and getattr(tf.series[0], "ndim", None) != 3:
            raise ValueError(
                f"Input stack is not 3D after load: {in_path} "
                f"(ndim={tf.series[0].ndim}, shape={getattr(tf.series[0], 'shape', None)})"
            )

        if len(tf.pages) < 2:
            raise ValueError(f"Need >=2 frames, got {len(tf.pages)} in {in_path}")

        first = tf.pages[0].asarray()
        if first.ndim != 2:
            raise ValueError(
                f"Expected 2D pages in {in_path}, got first page shape={getattr(first, 'shape', None)}"
            )

        t_count = len(tf.pages)
        height, width = first.shape
        out_dtype = np.float16 if relative else ABS_DTYPE

        def frame_generator():
            prev = first
            for page_idx in range(1, t_count):
                nxt = tf.pages[page_idx].asarray()
                if nxt.shape != (height, width):
                    raise ValueError(
                        f"Inconsistent page shape in {in_path}: "
                        f"page0={(height, width)} page{page_idx}={getattr(nxt, 'shape', None)}"
                    )

                if relative:
                    prev_f = prev.astype(np.float32, copy=False)
                    nxt_f = nxt.astype(np.float32, copy=False)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        diff = (nxt_f - prev_f) / np.maximum(prev_f, 1.0)
                    yield diff.astype(np.float16, copy=False)
                else:
                    prev_i = prev.astype(np.int32, copy=False)
                    nxt_i = nxt.astype(np.int32, copy=False)
                    diff = np.clip(nxt_i - prev_i, -32768, 32767)
                    yield diff.astype(ABS_DTYPE, copy=False)

                prev = nxt

        tiff.imwrite(
            str(out_path),
            frame_generator(),
            shape=(t_count - 1, height, width),
            dtype=out_dtype,
            compression=TIF_COMPRESSION,
        )

    with tiff.TiffFile(str(out_path)) as tf_out:
        print(
            f"[VERIFY] {out_path.name}: pages={len(tf_out.pages)} series={len(tf_out.series)} "
            f"series0.shape={tf_out.series[0].shape} axes={tf_out.series[0].axes}"
        )


def process_plate(
    base_dir: Path,
    plate: str | Path,
    do_relative: bool,
    do_absolute: bool,
    only_channels: Optional[Iterable[str]] = None,
) -> None:
    plate_dir = base_dir / str(plate)
    in_dir = plate_dir / "density_stacks_epanechnikov_poisson"
    out_dir = plate_dir / "density_changes_stacks_epanechnikov_poisson"

    tiffs = _list_tiffs(in_dir)
    if not tiffs:
        print(f"[{plate}] No TIFF stacks in {in_dir}")
        return

    for tif in tiffs:
        if not _keep_file(tif.name, only_channels):
            continue
        try:
            stem = tif.stem

            if do_relative:
                out = out_dir / f"{stem}_rel_diff.tif"
                _write_diffs_streaming(tif, out, relative=True)
                print(f"[{plate}] Wrote {out}")

            if do_absolute:
                out = out_dir / f"{stem}_abs_diff.tif"
                _write_diffs_streaming(tif, out, relative=False)
                print(f"[{plate}] Wrote {out}")

        except Exception as e:
            print(f"[{plate}] ERROR processing {tif.name}: {e}")


def run_all() -> None:
    for plate in PLATES:
        process_plate(BASE_DIR, plate, DO_RELATIVE, DO_ABSOLUTE, ONLY_CHANNELS)


if __name__ == "__main__":
    run_all()
