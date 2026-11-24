"""Stack structured per-timepoint density and density-change TIFF outputs."""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


MAP_RE = re.compile(r"^(?P<well>.+)_(?P<tp>\d+)_(?P<channel>sus|res|total)_density_epanechnikov\.tif$")
CHANGE_RE = re.compile(r"^(?P<well>.+)_(?P<t0>\d+)_to_(?P<t1>\d+)_res_density_epanechnikov_(?P<kind>abs_diff|rel_diff)\.tif$")
CHANNELS = ("sus", "res", "total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def stack_well(well_dir: Path) -> tuple[int, int]:
    well = well_dir.name
    maps_dir = well_dir / "density_maps"
    by_time: dict[int, dict[str, Path]] = {}
    for path in maps_dir.glob("*.tif"):
        match = MAP_RE.match(path.name)
        if match and match.group("well") == well:
            by_time.setdefault(int(match.group("tp")), {})[match.group("channel")] = path
    times = sorted(tp for tp, paths in by_time.items() if all(ch in paths for ch in CHANNELS))
    if not times:
        raise RuntimeError(f"No complete density frames for {well_dir}")
    first = tiff.imread(by_time[times[0]][CHANNELS[0]])
    shape = (len(times), len(CHANNELS), *first.shape)
    density_dir = well_dir / "density_stacks"
    density_dir.mkdir(exist_ok=True)

    def density_frames():
        for tp in times:
            for channel in CHANNELS:
                frame = tiff.imread(by_time[tp][channel]).astype(np.float32, copy=False)
                if frame.shape != shape[2:]:
                    raise ValueError(f"Density shape mismatch for {well} timepoint {tp}, channel {channel}: {frame.shape}")
                yield frame

    tiff.imwrite(
        density_dir / f"{well}_density_epanechnikov.tif",
        density_frames(), shape=(shape[0], 1, shape[1], shape[2], shape[3]),
        dtype=np.float32, compression="deflate",
        metadata={"axes": "TZCYX", "unit": "um"},
        imagej=True, photometric="minisblack", bigtiff=False,
    )
    pd.DataFrame({"frame": range(len(times)), "timepoint": [f"{tp:02d}" for tp in times]}).to_csv(
        density_dir / "stack_index.tsv", sep="\t", index=False
    )

    changes_dir = well_dir / "density_changes"
    by_kind: dict[str, list[tuple[int, int, Path]]] = {"abs_diff": [], "rel_diff": []}
    for path in changes_dir.glob("*.tif"):
        match = CHANGE_RE.match(path.name)
        if match and match.group("well") == well:
            by_kind[match.group("kind")].append((int(match.group("t0")), int(match.group("t1")), path))
    change_stack_dir = well_dir / "density_change_stacks"
    change_stack_dir.mkdir(exist_ok=True)
    reference_intervals = None
    for kind in ("abs_diff", "rel_diff"):
        entries = sorted(by_kind[kind], key=lambda item: (item[0], item[1]))
        if not entries:
            raise RuntimeError(f"No {kind} changes for {well_dir}")
        intervals = [(t0, t1) for t0, t1, _path in entries]
        if reference_intervals is None:
            reference_intervals = intervals
        elif intervals != reference_intervals:
            raise ValueError(f"Change interval mismatch for {well_dir}")
        first_change = tiff.imread(entries[0][2])

        def change_frames(entries=entries, expected=first_change.shape):
            for _t0, _t1, path in entries:
                frame = tiff.imread(path).astype(np.float32, copy=False)
                if frame.shape != expected:
                    raise ValueError(f"Change shape mismatch: {path}")
                yield frame

        tiff.imwrite(
            change_stack_dir / f"{well}_res_density_epanechnikov_{kind}.tif",
            change_frames(), shape=(len(entries), 1, 1, *first_change.shape), dtype=np.float32,
            compression="deflate", metadata={"axes": "TZCYX", "unit": "um"},
            imagej=True, photometric="minisblack", bigtiff=False,
        )
    pd.DataFrame({
        "frame": range(len(reference_intervals)),
        "first_timepoint": [f"{t0:02d}" for t0, _t1 in reference_intervals],
        "next_timepoint": [f"{t1:02d}" for _t0, t1 in reference_intervals],
    }).to_csv(change_stack_dir / "stack_index.tsv", sep="\t", index=False)
    return len(times), len(reference_intervals)


def main() -> None:
    root = parse_args().run_root.resolve()
    wells = sorted(path.parent for path in root.glob("Plate_*/*/density_maps"))
    if not wells:
        raise FileNotFoundError(f"No structured density-map wells under {root}")
    rows = []
    for index, well_dir in enumerate(wells, 1):
        density_frames, change_frames = stack_well(well_dir)
        rows.append({"plate": well_dir.parent.name, "well": well_dir.name, "density_frames": density_frames, "change_frames": change_frames})
        print(f"[{index}/{len(wells)}] {well_dir.parent.name}/{well_dir.name}: {density_frames} density, {change_frames} change", flush=True)
    pd.DataFrame(rows).to_csv(root / "stack_manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
