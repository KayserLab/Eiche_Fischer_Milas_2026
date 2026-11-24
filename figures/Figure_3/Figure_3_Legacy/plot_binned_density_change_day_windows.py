#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Generate relative density-change gate matrices in fixed experiment-day windows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

import plot_binned_density_change_strategy_lines_coarsegrained as producer


DEFAULT_RUN_DIR = Path(
    r"F:\CoreSeed_Kernel_Sweep_CT_AT50_AT100\run_kernel_320_coarse_32"
)
DEFAULT_CONFIG = Path(
    r"F:\CoreSeed_Kernel_Sweep_CT_AT50_AT100\configs\kernel_320_coarse_32.yaml"
)
GATE_MATRIX = (
    (2.0e-5, 4.375e-4),
    (2.0e-5, 1.0e-3),
    (2.0e-5, 2.0e-3),
    (1.0e-4, 4.375e-4),
    (1.0e-4, 1.0e-3),
    (1.0e-4, 2.0e-3),
    (5.0e-4, 1.0e-3),
    (5.0e-4, 2.0e-3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate n-annotated relative-change plots in fixed day windows."
    )
    parser.add_argument("--run-dir", default=os.fspath(DEFAULT_RUN_DIR))
    parser.add_argument("--config", default=os.fspath(DEFAULT_CONFIG))
    parser.add_argument("--window-days", type=int, default=10)
    parser.add_argument("--x-bin-size", type=float, default=0.2)
    parser.add_argument("--y-hist-bins", type=int, default=4096)
    parser.add_argument("--min-bin-pixels", type=int, default=100)
    parser.add_argument(
        "--cumulative-starts",
        action="store_true",
        help="Use cumulative first-frame-day subsets: 0-end, 10-end, ...",
    )
    parser.add_argument(
        "--ct-only",
        action="store_true",
        help="Include CT wells only.",
    )
    parser.add_argument(
        "--single-gate",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        help="Generate only one resistant-density gate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run>/summary_plots/gate_matrix_with_n/by_10_day",
    )
    return parser.parse_args()


def observed_first_days(run_dir: Path) -> list[float]:
    days: list[float] = []
    pattern = producer.base.CHANGE_SPECS["rel"]["glob"]
    for path in run_dir.glob(f"Plate_*/*/density_changes/{pattern}"):
        match = producer.base.REL_CHANGE_RE.match(path.name)
        if match is not None:
            days.append(producer.base.timepoint_to_day(match.group("first_tp")))
    return [day for day in days if producer.np.isfinite(day)]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if args.window_days <= 0:
        raise ValueError("--window-days must be positive")

    days = observed_first_days(run_dir)
    if not days:
        raise RuntimeError(f"No relative density-change intervals found in {run_dir}")
    last_start = int(max(days) // args.window_days) * args.window_days
    starts = range(0, last_start + 1, args.window_days)
    phase_by_start = {
        start: (
            f"from_day_{start:03d}_to_end"
            if args.cumulative_starts
            else f"day_{start:03d}_{start + args.window_days - 1:03d}"
        )
        for start in starts
    }
    selected_phases = tuple(phase_by_start.values())

    def interval_day_window(
        *,
        selected_phases: tuple[str, ...],
        first_day: float,
        **_kwargs,
    ) -> tuple[str, ...]:
        if not producer.np.isfinite(first_day) or first_day < 0:
            return ()
        if args.cumulative_starts:
            return tuple(
                phase
                for start, phase in phase_by_start.items()
                if first_day >= start and phase in selected_phases
            )
        start = int(first_day // args.window_days) * args.window_days
        phase = phase_by_start.get(start)
        return (phase,) if phase in selected_phases else ()

    # Reuse the established pooled-pixel accumulator and plot layout, changing
    # only the requested gate matrix, relative-change output, and time grouping.
    producer.DEFAULT_GATES = (
        (tuple(float(value) for value in args.single_gate),)
        if args.single_gate
        else GATE_MATRIX
    )
    producer.CHANGE_TYPES = ("rel",)
    if args.ct_only:
        producer.strategies_in_run = lambda _run_dir: ["CT"]
    producer.interval_phases = interval_day_window
    producer.resolve_escape_cutoffs = lambda *_args, **_kwargs: ({}, False, None)

    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else run_dir / "summary_plots" / "gate_matrix_with_n" / "by_10_day"
    )
    config = producer.base.load_config(args.config)
    process_args = argparse.Namespace(
        x_bin_size=float(args.x_bin_size),
        y_hist_bins=int(args.y_hist_bins),
        min_bin_pixels=int(args.min_bin_pixels),
        annotate_n_pixels=True,
    )
    count = producer.process_run(
        run_dir=run_dir,
        output_root=output_root,
        config=config,
        args=process_args,
        selected_phases=selected_phases,
    )
    print(f"[done] {count} PDFs in {output_root}")


if __name__ == "__main__":
    main()
