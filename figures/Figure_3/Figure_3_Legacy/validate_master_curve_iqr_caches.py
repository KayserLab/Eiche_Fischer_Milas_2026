#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Validate the reusable kernel/phase/gate IQR caches and write diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.figure_3 import master_curve_iqr_common as common


DEFAULT_CACHE_ROOT = Path(r"F:\CoreSeed_Master_Curve_Linear_IQR_Cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-bin-pixels", type=int, default=100)
    return parser.parse_args()


def cache_paths(cache_root: Path, dataset: str, kernel_um: int) -> list[Path]:
    root = cache_root / dataset
    if dataset == "in_vitro" and kernel_um == 40:
        return [
            root / "kernel_40um_ct_at50_at100.npz",
            root / "kernel_40um_streamed_other_strategies.npz",
        ]
    return [root / f"kernel_{kernel_um}um_all_phases_iqr.npz"]


def validate_bundle(
    dataset: str,
    kernel_um: int,
    bundle: common.AggregationBundle,
    min_bin_pixels: int,
) -> list[dict[str, object]]:
    expected = list(common.ALLOWED_STRATEGIES)
    if bundle.strategies != expected:
        raise AssertionError(
            f"{dataset}/{kernel_um}: strategies={bundle.strategies}, expected={expected}"
        )
    rows: list[dict[str, object]] = []
    for gate_name, _gate in common.GATES:
        for phase in common.PHASES:
            aggregator = bundle.aggregators[(gate_name, phase)]
            histogram_count = aggregator.histogram.sum(axis=2)
            if not np.array_equal(histogram_count, aggregator.count):
                raise AssertionError(
                    f"{dataset}/{kernel_um}/{gate_name}/{phase}: "
                    "histogram counts do not equal exact counts"
                )
            nonempty = aggregator.count > 0
            if np.any(~np.isfinite(aggregator.total[nonempty])):
                raise AssertionError("Non-finite totals in nonempty bins")
            summary = common.summarize_iqr(
                aggregator,
                bundle.strategies,
                gate_name=gate_name,
                phase=phase,
            )
            if not summary.empty:
                ordered = (
                    (summary["q1"] <= summary["median"])
                    & (summary["median"] <= summary["q3"])
                )
                if not bool(ordered.all()):
                    raise AssertionError(
                        f"{dataset}/{kernel_um}/{gate_name}/{phase}: unordered quantiles"
                    )
            plotted_summary = summary[
                summary["n_pixels"] >= min_bin_pixels
            ]
            with np.errstate(divide="ignore", invalid="ignore"):
                low_fraction = np.divide(
                    aggregator.histogram[:, :, 0],
                    aggregator.count,
                    out=np.zeros_like(aggregator.count, dtype=float),
                    where=aggregator.count > 0,
                )
                high_fraction = np.divide(
                    aggregator.histogram[:, :, -1],
                    aggregator.count,
                    out=np.zeros_like(aggregator.count, dtype=float),
                    where=aggregator.count > 0,
                )
            plotted = aggregator.count >= min_bin_pixels
            pooled = common.pooled_aggregator(aggregator)
            with np.errstate(divide="ignore", invalid="ignore"):
                pooled_high_fraction = np.divide(
                    pooled.histogram[0, :, -1],
                    pooled.count[0],
                    out=np.zeros_like(pooled.count[0], dtype=float),
                    where=pooled.count[0] > 0,
                )
                pooled_low_fraction = np.divide(
                    pooled.histogram[0, :, 0],
                    pooled.count[0],
                    out=np.zeros_like(pooled.count[0], dtype=float),
                    where=pooled.count[0] > 0,
                )
            pooled_plotted = pooled.count[0] >= min_bin_pixels
            rows.append(
                {
                    "dataset": dataset,
                    "kernel_um": kernel_um,
                    "gate": gate_name,
                    "phase": phase,
                    "strategies": ",".join(bundle.strategies),
                    "total_n_pixels": int(aggregator.count.sum()),
                    "nonempty_strategy_x_bins": int(np.count_nonzero(nonempty)),
                    "max_low_edge_fraction_plotted_strategy_bin": float(
                        low_fraction[plotted].max(initial=0.0)
                    ),
                    "max_high_edge_fraction_plotted_strategy_bin": float(
                        high_fraction[plotted].max(initial=0.0)
                    ),
                    "strategy_bins_high_edge_fraction_ge_0p25": int(
                        np.count_nonzero((high_fraction >= 0.25) & plotted)
                    ),
                    "strategy_bins_low_edge_fraction_ge_0p25": int(
                        np.count_nonzero((low_fraction >= 0.25) & plotted)
                    ),
                    "max_high_edge_fraction_plotted_pooled_bin": float(
                        pooled_high_fraction[pooled_plotted].max(initial=0.0)
                    ),
                    "max_low_edge_fraction_plotted_pooled_bin": float(
                        pooled_low_fraction[pooled_plotted].max(initial=0.0)
                    ),
                    "minimum_q1": (
                        float(plotted_summary["q1"].min())
                        if not plotted_summary.empty
                        else np.nan
                    ),
                    "maximum_q3": (
                        float(plotted_summary["q3"].max())
                        if not plotted_summary.empty
                        else np.nan
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.expanduser()
    rows: list[dict[str, object]] = []
    for kernel_um in (320, 160, 40):
        for dataset in ("in_vitro", "in_silico"):
            paths = cache_paths(cache_root, dataset, kernel_um)
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Missing cache(s): " + ", ".join(str(path) for path in missing)
                )
            bundle = common.merge_bundles(paths)
            rows.extend(
                validate_bundle(
                    dataset,
                    kernel_um,
                    bundle,
                    int(args.min_bin_pixels),
                )
            )
            print(
                f"[valid] {dataset} kernel={kernel_um}; "
                f"strategies={bundle.strategies}",
                flush=True,
            )
            del bundle
    frame = pd.DataFrame(rows)
    if (frame["strategy_bins_high_edge_fraction_ge_0p25"] > 0).any():
        problem = frame[
            frame["strategy_bins_high_edge_fraction_ge_0p25"] > 0
        ]
        raise AssertionError(
            "At least one IQR could be clipped by the upper histogram edge:\n"
            + problem.to_string(index=False)
        )
    if (frame["strategy_bins_low_edge_fraction_ge_0p25"] > 0).any():
        problem = frame[
            frame["strategy_bins_low_edge_fraction_ge_0p25"] > 0
        ]
        raise AssertionError(
            "At least one IQR could be clipped by the lower histogram edge:\n"
            + problem.to_string(index=False)
        )
    if (frame["max_high_edge_fraction_plotted_pooled_bin"] >= 0.25).any():
        raise AssertionError("At least one plotted pooled Q3 could be clipped")
    if (frame["max_low_edge_fraction_plotted_pooled_bin"] >= 0.25).any():
        raise AssertionError("At least one plotted pooled Q1 could be clipped")
    output = (
        args.output.expanduser()
        if args.output is not None
        else cache_root / "validation_summary.tsv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)
    print(frame.to_string(index=False))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
