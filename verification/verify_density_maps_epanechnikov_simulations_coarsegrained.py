"""Verify simulation Epanechnikov density-map TIFFs against prepared point clouds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.density_maps.Density_Maps_Epanechnikov_simulations_coarsegrained import DEFAULT_OUTPUT_ROOT, DENSITY_MANIFEST
from shared.data_paths import VERIFICATION_DATA
from verification.selection import balanced_strategy_sample
from verification.verify_density_maps_common import DensitySample, audit_samples

DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def discover(output_root: Path) -> list[DensitySample]:
    manifest = pd.read_csv(output_root / DENSITY_MANIFEST, sep="\t")
    samples: list[DensitySample] = []
    for row in manifest.itertuples(index=False):
        if str(row.status) not in {"ok", "skipped_exists"} or str(row.strategy) == "ATAgent1":
            continue
        maps_dir = output_root / str(row.plate) / str(row.well_identifier) / "density_maps"
        source = Path(str(row.prepared_hdf5_path))
        if source.is_file() and maps_dir.is_dir():
            samples.append(DensitySample(str(row.plate), str(row.well_identifier), str(row.strategy), source, maps_dir, simulation=True))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    selected = balanced_strategy_sample(discover(args.density_root), lambda item: item.strategy, args.seed)
    if not selected:
        raise RuntimeError(f"No valid simulation density-map runs found under {args.density_root}")
    review_dir = args.verification_root / "Density_Maps_Epanechnikov_simulations_coarsegrained" / f"sample_n{len(selected)}_seed{args.seed}"
    audit_samples(selected, review_dir, "Verifying simulation density maps", "analysis/basic/density_maps/Density_Maps_Epanechnikov_simulations_coarsegrained.py")


if __name__ == "__main__":
    main()
