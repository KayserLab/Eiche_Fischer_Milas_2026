"""Create the period-specific PDFs for the 120 µm-radius analysis."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis/figure_3/HDF5_analysis")]


import os
import runpy
from pathlib import Path


os.environ["HDF5_KERNEL_RADIUS_UM"] = "120"
runpy.run_path(
    Path(__file__).with_name("fig_flux_vs_density_periods.py"),
    run_name="__main__",
)
