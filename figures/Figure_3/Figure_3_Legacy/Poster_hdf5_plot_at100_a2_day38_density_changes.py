#!/usr/bin/env python3
"""Export rotated, absolute experimental AT100-A2 d38-to-d39 change-rate maps.

The experimental acquisitions are d38 and d40.  Their two-day difference is
divided by two, yielding the one-day rate represented by the d38-to-d39 map.

Each requested quantity is written as an image-only PNG (no axes, labels, or
colour bars), with a matching float TIFF for lossless rescaling in FIJI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.data_paths import FIGURE_PLOTS


DENSITY_MAPS = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\intermediate_analysis_data"
    r"\experimental\basic\density_maps\filtered_160\r160_c16_filtereq"
    r"\Plate_5\A2_AT100-A2\density_maps"
)
OUTPUT_DIRECTORY = FIGURE_PLOTS / "Figure_3" / "density_maps"
WELL = "A2_AT100-A2"
START_DAY = 38
MEASURED_END_DAY = 40
OUTPUT_END_DAY = 39


def density(cell_type: str, day: int) -> np.ndarray:
    path = DENSITY_MAPS / f"{WELL}_{day:02d}_{cell_type}_density_epanechnikov.tif"
    return tifffile.imread(path).astype(np.float32)


def main() -> None:
    elapsed_days = MEASURED_END_DAY - START_DAY
    # This is the absolute (rather than relative/fold) density change per day,
    # consistent with ``resistant_change(..., relative=False)`` in the AT50
    # plotting script.
    delta_res = (density("res", MEASURED_END_DAY) - density("res", START_DAY)) / elapsed_days
    delta_sus = (density("sus", MEASURED_END_DAY) - density("sus", START_DAY)) / elapsed_days
    delta_difference = delta_res - delta_sus
    maps = (delta_res, delta_sus, delta_difference)
    suffixes = ("delta_rho_res", "delta_rho_sus", "delta_rho_res_minus_sus")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tag = f"AT100_A2_experiment_d{START_DAY:02d}_to_d{OUTPUT_END_DAY:02d}"
    maps = tuple(np.rot90(array, 2) for array in maps)
    for suffix, array in zip(suffixes, maps):
        tifffile.imwrite(OUTPUT_DIRECTORY / f"{tag}_{suffix}.tif", array)

    finite = np.concatenate([np.abs(array[np.isfinite(array)]).ravel() for array in maps])
    limit = float(np.quantile(finite, 0.998))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    for suffix, array in zip(suffixes, maps):
        rgba = cmap(norm(np.ma.masked_invalid(array)))
        plt.imsave(
            OUTPUT_DIRECTORY / f"{tag}_{suffix}.png",
            rgba, origin="lower",
        )


if __name__ == "__main__":
    main()
