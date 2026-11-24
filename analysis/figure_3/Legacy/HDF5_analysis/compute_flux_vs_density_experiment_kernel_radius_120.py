"""Archived single-radius entry point; use the active base script for both radii."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
ACTIVE = ROOT / "analysis/figure_3/HDF5_analysis"
sys.path[:0] = [os.fspath(ROOT), os.fspath(ACTIVE)]
target = ACTIVE / "compute_flux_vs_density_experiment.py"
sys.argv = [os.fspath(target), "--kernel-radius-um", "120", *sys.argv[1:]]
os.environ["HDF5_KERNEL_RADIUS_UM"] = "120"
runpy.run_path(target, run_name="__main__")
