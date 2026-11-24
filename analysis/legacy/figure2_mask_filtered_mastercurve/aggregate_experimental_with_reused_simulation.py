#!/usr/bin/env python3
"""Aggregate mask-filtered experimental maps and reuse established simulation tables.

This avoids recalculating simulation data when testing a new experimental cell
filter.  It writes a self-contained table/cache directory and never modifies
the source simulation tables or caches.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-vitro-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-simulation-tables", type=Path, required=True)
    parser.add_argument("--source-simulation-cache", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.expanduser()
    output_root = args.output_root.expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    strategies, aggregators, metadata = aggregate.extract_in_vitro(
        args.in_vitro_run.expanduser(),
        args.config.expanduser(),
        aggregate.DEFAULT_EXPERIMENT_ESCAPE,
    )
    metadata.update(filter="figure2_clean_sensitive_and_all_clone_resistant")
    aggregate.save_bundle(cache_root / "in_vitro_all_conditions.npz", strategies, aggregators, metadata)

    source_sim_cache = args.source_simulation_cache.expanduser() / "in_silico_all_conditions.npz"
    if not source_sim_cache.is_file():
        raise FileNotFoundError(source_sim_cache)
    shutil.copy2(source_sim_cache, cache_root / source_sim_cache.name)

    manifest_rows: list[dict[str, object]] = []
    for condition in aggregate.CONDITIONS:
        output_dir = output_root / condition.output_relative
        output_dir.mkdir(parents=True, exist_ok=True)
        vitro_strategy = aggregate.summarize(aggregators[condition.slug], strategies, condition)
        vitro_pooled = aggregate.summarize(
            aggregate.pooled(aggregators[condition.slug], strategies),
            ["CT, AT50, and AT100 pooled"], condition,
        )
        vitro_strategy.to_csv(output_dir / "in_vitro_strategy_binned_statistics.tsv", sep="\t", index=False)
        vitro_pooled.to_csv(output_dir / "in_vitro_pooled_master_binned_statistics.tsv", sep="\t", index=False)

        source_dir = args.source_simulation_tables.expanduser() / condition.output_relative
        for name in ("in_silico_strategy_binned_statistics.tsv", "in_silico_pooled_master_binned_statistics.tsv"):
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, output_dir / name)
        manifest_rows.append({
            "condition_directory": os.fspath(output_dir),
            "in_vitro_pixels": int(vitro_strategy["n_pixels"].sum()),
            "simulation": "reused_unchanged",
        })
        print(f"[experimental aggregation] {condition.slug}", flush=True)
    pd.DataFrame(manifest_rows).to_csv(output_root / "output_manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
