#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Generate the 320-um simulation gate matrix using pre-escape intervals only.

Both the first-frame total/resistant densities and the resistant-density
growth maps come from the standard 320-um, coarse-32 simulation run.  An
interval is included only when its end day is no later than the run's positive
time-to-escape (TTE) in the configured simulation results table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import plot_density_change_datapoints_coarsegrained as base
import plot_binned_density_change_strategy_lines_coarsegrained as plotter


DEFAULT_RUN_DIR = Path(
    r"F:\2025_08_CoreSeed_Simulation_Density_Output"
    r"\run_sim_20260709_145640_320_32"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\CoreSeed_InSilico_PreEscape_320\run_kernel_320_coarse_32"
    r"\summary_plots\gate_matrix_with_n"
)
DEFAULT_ESCAPE_CSV = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja"
    r"\combined_results.csv"
)
GATES = (
    (2.0e-5, 4.375e-4),
    (2.0e-5, 0.001),
    (2.0e-5, 0.002),
    (1.0e-4, 4.375e-4),
    (1.0e-4, 0.001),
    (1.0e-4, 0.002),
    (5.0e-4, 0.001),
    (5.0e-4, 0.002),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the simulation 320-um density/growth gate matrix using "
            "only intervals ending before a positive TTE."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--simulation-escape-csv",
        type=Path,
        default=DEFAULT_ESCAPE_CSV,
    )
    parser.add_argument("--x-bin-size", type=float, default=plotter.DEFAULT_X_BIN_SIZE)
    parser.add_argument("--y-hist-bins", type=int, default=plotter.DEFAULT_Y_HIST_BINS)
    parser.add_argument(
        "--min-bin-pixels",
        type=int,
        default=plotter.DEFAULT_MIN_BIN_PIXELS,
    )
    parser.add_argument(
        "--no-annotate-n-pixels",
        action="store_true",
        help="Do not label strategy-panel points with qualifying pixel counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    escape_csv = args.simulation_escape_csv.expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not escape_csv.is_file():
        raise FileNotFoundError(f"Escape CSV does not exist: {escape_csv}")
    if args.x_bin_size <= 0:
        raise ValueError("--x-bin-size must be positive")
    if args.y_hist_bins < 32:
        raise ValueError("--y-hist-bins must be at least 32")
    if args.min_bin_pixels < 1:
        raise ValueError("--min-bin-pixels must be at least 1")

    config = base.load_config(args.config)
    radius_um = float(config.get("radius_um", 0))
    coarse = int(config.get("coarsening_factor", 0))
    if radius_um != 320.0 or coarse != 32:
        raise ValueError(
            "This entry point requires radius_um=320 and coarsening_factor=32; "
            f"got radius_um={radius_um:g}, coarsening_factor={coarse}."
        )

    plotter.DEFAULT_GATES = GATES
    plotter.CHANGE_TYPES = ("rel",)

    shared_args = argparse.Namespace(
        experimental_escape_summary=None,
        simulation_escape_csv=str(escape_csv),
        x_bin_size=float(args.x_bin_size),
        y_hist_bins=int(args.y_hist_bins),
        min_bin_pixels=int(args.min_bin_pixels),
        annotate_n_pixels=not bool(args.no_annotate_n_pixels),
    )
    n_pdfs = plotter.process_run(
        run_dir=run_dir,
        output_root=output_dir,
        config=config,
        args=shared_args,
        selected_phases=("before_escape",),
    )
    print(f"[done] {n_pdfs} PDFs: {output_dir}")


if __name__ == "__main__":
    main()
