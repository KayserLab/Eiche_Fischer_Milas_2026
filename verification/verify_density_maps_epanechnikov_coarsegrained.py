"""Verify experimental Epanechnikov density-map TIFFs against their point clouds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import DEFAULT_OUTPUT_ROOT, load_config, DEFAULT_CONFIG_PATH, strategy_name_from_well_identifier
from shared.data_paths import VERIFICATION_DATA
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample
from verification.verify_density_maps_common import DensitySample, audit_samples

DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def latest_run(root: Path) -> Path:
    runs = [path for path in root.glob("run_*") if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else root


def discover(run_root: Path, experiment_root: Path) -> list[DensitySample]:
    samples: list[DensitySample] = []
    for maps_dir in run_root.rglob("density_maps"):
        try:
            plate_path = next(parent for parent in maps_dir.parents if parent.name.startswith("Plate_"))
        except StopIteration:
            continue
        plate, identifier = plate_path.name, maps_dir.parent.name
        cutoff = EXCLUDE_SAMPLES.get(identifier)
        if identifier in EXCLUDE_SAMPLES and cutoff is None:
            continue
        hdf5_dir = experiment_root / plate / "hdf5"
        matches = sorted(hdf5_dir.glob(f"*{identifier}_total_poisson_padded.h5")) or sorted(hdf5_dir.glob(f"*{identifier}_total*.h5"))
        if len(matches) == 1:
            samples.append(DensitySample(plate, identifier, strategy_name_from_well_identifier(identifier), matches[0], maps_dir, last_valid_day=cutoff))
    return samples


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--density-run", type=Path, default=None, help="Specific run_* folder; by default the newest run is used.")
    parser.add_argument("--experiment-root", type=Path, default=Path(config["experiment_path"]))
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    run_root = args.density_run or latest_run(args.density_root)
    available = discover(run_root, args.experiment_root)
    selected = balanced_strategy_sample(available, lambda item: item.strategy, args.seed)
    if not selected:
        raise RuntimeError(f"No valid experimental density-map replicates found under {run_root}")
    review_dir = args.verification_root / "Density_Maps_Epanechnikov_coarsegrained" / f"sample_n{len(selected)}_seed{args.seed}"
    audit_samples(selected, review_dir, "Verifying experimental density maps", "analysis/basic/density_maps/Density_Maps_Epanechnikov_coarsegrained.py")


if __name__ == "__main__":
    main()
