"""Create a visual sample audit for Make_susrestotal_Colony_Masks.py.

Selects valid well/day samples using experiment_filters, then compares the
CellPose inputs with all three colony-mask outputs. This is read-only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from tqdm import tqdm

# Make production modules importable when this file is run as a script.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.microscopy.Make_susrestotal_Colony_Masks import center_pad
from shared.data_paths import EXP_RAW_DATA, VERIFICATION_DATA
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample, strategy_from_sample


DEFAULT_DATA_ROOT = EXP_RAW_DATA
DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA
OUTPUT_PATTERN = re.compile(r"(?P<well>.+)_(?P<day>\d{2})_total_colony_mask\.tif$")


def channel_overlay(susceptible: np.ndarray, resistant: np.ndarray) -> np.ndarray:
    """Render sensitive blue, resistant yellow, and overlap white."""
    output = np.zeros((*susceptible.shape, 3), dtype=float)
    susceptible_only = (susceptible > 0) & (resistant == 0)
    resistant_only = (resistant > 0) & (susceptible == 0)
    overlap = (susceptible > 0) & (resistant > 0)
    output[susceptible_only] = (0.15, 0.45, 1.0)
    output[resistant_only] = (1.0, 0.82, 0.05)
    output[overlap] = (1.0, 1.0, 1.0)
    return output


def valid_well_days(data_root: Path) -> dict[tuple[str, str], list[str]]:
    """Return output days for wells retained by the canonical filter rules."""
    eligible: dict[tuple[str, str], list[str]] = {}
    for plate_dir in sorted(data_root.glob("Plate_*")):
        for path in sorted((plate_dir / "colony_masks").glob("*_total_colony_mask.tif")):
            match = OUTPUT_PATTERN.match(path.name)
            if not match:
                continue
            well, day = match["well"], match["day"]
            cutoff = EXCLUDE_SAMPLES.get(well, float("inf"))
            if cutoff is None or int(day) > cutoff:
                continue
            eligible.setdefault((plate_dir.name, well), []).append(day)
    return eligible


def find_input_mask(input_dir: Path, well: str, day: str, channel: int) -> Path:
    pattern = re.compile(
        rf".*d{int(day):02d}_.*_MIPs_{re.escape(well)}_\[{channel},\s*0\]_masks\.tif$",
        re.IGNORECASE,
    )
    matches = sorted(path for path in input_dir.glob("*_masks.tif") if pattern.match(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one channel-{channel} input for {well}, day {day}; found {len(matches)}.")
    return matches[0]


def load_sample(data_root: Path, plate: str, well: str, day: str) -> tuple[list[tuple[str, np.ndarray, str | None]], dict[str, bool]]:
    """Load one comparison, using the original script's centre-padding function."""
    plate_dir = data_root / plate
    input_dir, output_dir = plate_dir / "czi_NPY", plate_dir / "colony_masks"
    sus_input = (imread(find_input_mask(input_dir, well, day, 2)) > 0).astype(np.uint8)
    res_input = (imread(find_input_mask(input_dir, well, day, 3)) > 0).astype(np.uint8)
    sus_output = imread(output_dir / f"{well}_{day}_sus_colony_mask.tif").astype(np.uint8)
    res_output = imread(output_dir / f"{well}_{day}_res_colony_mask.tif").astype(np.uint8)
    total_output = imread(output_dir / f"{well}_{day}_total_colony_mask.tif").astype(np.uint8)
    expected_total = ((sus_output > 0) | (res_output > 0)).astype(np.uint8)
    checks = {
        "sus_matches_input_after_padding": bool(np.array_equal(sus_output, center_pad(sus_input, *sus_output.shape))),
        "res_matches_input_after_padding": bool(np.array_equal(res_output, center_pad(res_input, *res_output.shape))),
        "total_is_union_of_generated_channels": bool(np.array_equal(total_output, expected_total)),
    }
    panels = [
        ("Input sensitive (blue)", sus_input, "Blues"),
        ("Input RES (yellow)", res_input, "Wistia"),
        ("Input channels combined", channel_overlay(sus_input, res_input), None),
        ("Output sensitive (blue)", sus_output, "Blues"),
        ("Output RES (yellow)", res_output, "Wistia"),
        ("Output channels combined", channel_overlay(sus_output, res_output), None),
    ]
    return panels, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831, help="Fixed by default for reproducibility.")
    args = parser.parse_args()

    progress = tqdm(total=3, desc="Verifying colony masks", unit="step")
    progress.set_postfix_str("discovering valid wells")
    available = valid_well_days(args.data_root)
    progress.update()
    selected_wells = balanced_strategy_sample(available, lambda item: strategy_from_sample(item[1]), args.seed)
    rng = random.Random(args.seed + 1)
    samples = [(plate, well, rng.choice(sorted(available[(plate, well)]))) for plate, well in selected_wells]
    args.sample_count = len(samples)
    progress.total = args.sample_count + 3
    progress.refresh()
    progress.update()

    review_dir = args.verification_root / "Make_susrestotal_Colony_Masks" / f"sample_n{args.sample_count}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Colony-mask verification\n\n"
        "This review checks whether the colony-mask step faithfully converts the two CellPose inputs into simple masks used later in the analysis.\n\n"
        "Each PNG represents one valid replicate and day from the strategy-balanced review cohort. The first row shows the original sensitive and resistant cell regions. The second row shows the corresponding generated masks. Sensitive regions are blue, resistant regions are yellow, and white marks where the two channels overlap.\n\n"
        "A panel is marked PASS only when the generated sensitive and resistant masks match their inputs after the expected canvas alignment, and their combined occupancy agrees with the total-mask rule. `metadata.json` records the selected samples and these checks.\n",
        encoding="utf-8",
    )
    all_checks: list[dict[str, object]] = []
    for plate, well, day in samples:
        progress.set_postfix_str(f"{plate}: {well}, day {day}")
        panels, checks = load_sample(args.data_root, plate, well, day)
        all_checks.append({"plate": plate, "well": well, "day": day, "checks": checks})
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for axis, (title, image, cmap) in zip(axes.flat, panels):
            axis.imshow(image, cmap=cmap, interpolation="nearest")
            axis.set_title(title, fontsize=9)
            axis.set_axis_off()
        passed_sample = all(checks.values())
        fig.suptitle(f"{'PASS' if passed_sample else 'FAIL'}: {plate}, {well}, day {day}", fontsize=14)
        figure_path = review_dir / f"{plate}_{well}_day_{day}.png"
        fig.savefig(figure_path, dpi=400)
        plt.close(fig)
        progress.update()

    passed = all(all(result["checks"].values()) for result in all_checks)
    metadata = {
        "script_under_test": "analysis/basic/microscopy/Make_susrestotal_Colony_Masks.py",
        "selection": "Strategy-balanced cohort after EXCLUDE_SAMPLES exclusions and day cutoffs; ATAgent1 excluded.",
        "sample_count": args.sample_count,
        "seed": args.seed,
        "passed": passed,
        "samples": all_checks,
    }
    (review_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
