#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Replot existing binned strategy tables with compact pixel-count labels.

This companion avoids rescanning the density TIFFs. It reads the statistics
tables produced by ``plot_binned_density_change_strategy_lines_coarsegrained``
and writes a separate annotated output tree. The plotted statistics are
therefore identical to the source figures; only the point labels differ.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

import plot_binned_density_change_strategy_lines_coarsegrained as producer
from shared.strategy_colors import STRATEGY_ORDER


DEFAULT_OUTPUT_SUBDIR = Path(
    "summary_plots/requested_variations_20260716/binned_strategy_lines_with_n"
)
CHANGE_TYPES = {
    "relative": "rel",
    "absolute_per_day": "abs_per_day",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot existing pooled-pixel binned strategy statistics with "
            "compact n-pixel labels in the individual-strategy panels."
        )
    )
    parser.add_argument(
        "--config",
        default=os.fspath(producer.base.DEFAULT_CONFIG_PATH),
        help=(
            "Density-map YAML configuration. Default: "
            f"{producer.base.DEFAULT_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        default=None,
        help=(
            "Run directory. Repeat for multiple runs. By default the standard "
            "experimental and simulation runs are replotted."
        ),
    )
    parser.add_argument(
        "--source-subdir",
        default=os.fspath(producer.DEFAULT_OUTPUT_SUBDIR),
        help=(
            "Existing statistics root, relative to each run directory unless "
            "absolute. Default: "
            f"{producer.DEFAULT_OUTPUT_SUBDIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=os.fspath(DEFAULT_OUTPUT_SUBDIR),
        help=(
            "Annotated output root, relative to each run directory unless "
            f"absolute. Default: {DEFAULT_OUTPUT_SUBDIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--min-bin-pixels",
        type=int,
        default=producer.DEFAULT_MIN_BIN_PIXELS,
        help=(
            "Primary-plot pixel threshold. The all_bins mirror always uses 1. "
            f"Default: {producer.DEFAULT_MIN_BIN_PIXELS}"
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=producer.PHASES,
        default=["overall"],
        help="Phases to replot. Default: overall.",
    )
    parser.add_argument(
        "--change-types",
        nargs="+",
        choices=tuple(CHANGE_TYPES),
        default=["relative"],
        help="Change types to replot. Default: relative.",
    )
    return parser.parse_args()


def resolve_root(run_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else run_dir / path


def phase_root(root: Path, phase: str) -> Path:
    return root if phase == "overall" else root / phase


def ordered_strategies(summary: pd.DataFrame) -> list[str]:
    present = set(summary["strategy"].astype(str))
    ordered = [strategy for strategy in STRATEGY_ORDER if strategy in present]
    return ordered + sorted(present.difference(ordered))


def replot_run(
    *,
    run_dir: Path,
    source_root: Path,
    output_root: Path,
    config: dict,
    phases: tuple[str, ...],
    change_names: tuple[str, ...],
    min_bin_pixels: int,
) -> int:
    variants = (
        ("minimum_bin_pixels", output_root, int(min_bin_pixels)),
        ("all_bins", output_root / "all_bins", 1),
    )
    inventory: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        for gate in producer.DEFAULT_GATES:
            for change_label in change_names:
                kind = CHANGE_TYPES[change_label]
                relative_dir = (
                    Path(producer.change_name(kind)) / producer.gate_name(gate)
                )
                table_path = (
                    phase_root(source_root, phase)
                    / relative_dir
                    / "binned_strategy_statistics.tsv"
                )
                if not table_path.is_file():
                    raise FileNotFoundError(f"Statistics table not found: {table_path}")
                summary = pd.read_csv(table_path, sep="\t")
                missing = set(producer.SUMMARY_COLUMNS).difference(summary.columns)
                if missing:
                    raise ValueError(
                        f"Missing columns in {table_path}: {sorted(missing)}"
                    )
                strategies = ordered_strategies(summary)

                for variant, variant_root, threshold in variants:
                    output_dir = phase_root(variant_root, phase) / relative_dir
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_table = output_dir / "binned_strategy_statistics.tsv"
                    summary.to_csv(output_table, sep="\t", index=False)
                    pdf_path = (
                        output_dir / "binned_mean_median_all_strategies.pdf"
                    )
                    producer.plot_summary(
                        summary,
                        pdf_path,
                        producer.plot_settings(config, kind),
                        gate,
                        phase,
                        strategies,
                        threshold,
                        annotate_n_pixels=True,
                    )
                    inventory.append(
                        {
                            "variant": variant,
                            "minimum_bin_pixels": threshold,
                            "phase": phase,
                            "change_type": change_label,
                            "gate": producer.gate_name(gate),
                            "source_table": os.fspath(table_path),
                            "pdf": os.fspath(pdf_path.relative_to(output_root)),
                            "bytes": pdf_path.stat().st_size,
                        }
                    )
                    print(f"[saved] {pdf_path}")

    pd.DataFrame(inventory).to_csv(
        output_root / "plot_inventory.tsv",
        sep="\t",
        index=False,
    )
    run_type = (
        "simulation"
        if producer.base.is_simulation_density_run(run_dir)
        else "experimental"
    )
    (output_root / "README.txt").write_text(
        f"{run_type.title()} binned strategy lines with n-pixel labels.\n\n"
        f"Source statistics: {source_root}\n"
        "No density TIFFs were rescanned and no statistics were recomputed.\n"
        "Individual-strategy mean and median points are labeled with the same "
        "qualifying n_pixels stored in each source TSV row.\n"
        "Compact units use k = 10^3 and M = 10^6.\n"
        f"Primary plots omit bins with fewer than {min_bin_pixels} pixels.\n"
        "The all_bins folder shows every nonempty bin.\n",
        encoding="ascii",
    )
    (output_root / "all_bins" / "README.txt").write_text(
        "Annotated plots of all nonempty bins.\n"
        "See ../README.txt for provenance and count-label definitions.\n",
        encoding="ascii",
    )
    return len(inventory)


def main() -> None:
    args = parse_args()
    if args.min_bin_pixels < 1:
        raise ValueError("--min-bin-pixels must be at least 1")
    run_dirs = [
        Path(value).expanduser()
        for value in (args.run_dirs or producer.DEFAULT_RUN_DIRS)
    ]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config = producer.base.load_config(args.config)
    phases = tuple(dict.fromkeys(args.phases))
    change_names = tuple(dict.fromkeys(args.change_types))
    total = 0
    for run_dir in run_dirs:
        total += replot_run(
            run_dir=run_dir,
            source_root=resolve_root(run_dir, args.source_subdir),
            output_root=resolve_root(run_dir, args.output_subdir),
            config=config,
            phases=phases,
            change_names=change_names,
            min_bin_pixels=int(args.min_bin_pixels),
        )
    print(f"[done] {total} annotated PDFs across {len(run_dirs)} run(s)")


if __name__ == "__main__":
    main()
