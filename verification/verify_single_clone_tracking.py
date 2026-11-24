"""Create per-replicate visual audits for single_clone_tracking.py.

This is read-only. It uses the production preprocessing function and settings
to check clone-label stacks against their resistant-mask-stack inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from tifffile import imread
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.microscopy.single_clone_tracking import (
    MIN_SIZE,
    SELEM_RAD,
    day_from_frame_index,
    link_distance_for_frame_pair,
    preprocess,
)
from shared.data_paths import EXP_RAW_DATA, VERIFICATION_DATA
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample, strategy_from_sample


DEFAULT_DATA_ROOT = EXP_RAW_DATA
DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA
DAY_PATTERN = re.compile(r".+_(?P<day>\d{2})_res_colony_mask\.tif$")


def retained_frames(plate_dir: Path, well: str) -> list[int]:
    """Find stack-frame indices retained by the canonical exclusion rules."""
    paths = sorted((plate_dir / "colony_masks").glob(f"{well}_??_res_colony_mask.tif"))
    cutoff = EXCLUDE_SAMPLES.get(well, float("inf"))
    if cutoff is None:
        return []
    return [index for index, path in enumerate(paths) if (match := DAY_PATTERN.match(path.name)) and int(match["day"]) <= cutoff]


def eligible_wells(data_root: Path) -> dict[tuple[str, str], list[int]]:
    """Return non-excluded clone stacks that have at least three valid frames."""
    result: dict[tuple[str, str], list[int]] = {}
    for plate_dir in sorted(data_root.glob("Plate_*")):
        for clone_path in (plate_dir / "clone_masks_stacks").glob("*_res_clone_masks.tif"):
            well = clone_path.name.removesuffix("_res_clone_masks.tif")
            frames = retained_frames(plate_dir, well)
            if len(frames) >= 3 and (plate_dir / "colony_masks_stacks" / f"{well}_res_colony_masks.tif").exists():
                result[(plate_dir.name, well)] = frames
    return result


def merge_count(plate_dir: Path, well: str) -> int:
    path = plate_dir / "clone_masks_stacks" / f"{well}_res_clone_masks.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def colony_overview(sensitive: np.ndarray, resistant: np.ndarray) -> np.ndarray:
    """Render sensitive blue, resistant yellow, and overlap white."""
    image = np.zeros((*sensitive.shape, 3), dtype=float)
    image[(sensitive > 0) & (resistant == 0)] = (0.15, 0.45, 1.0)
    image[(resistant > 0) & (sensitive == 0)] = (1.0, 0.82, 0.05)
    image[(sensitive > 0) & (resistant > 0)] = (1.0, 1.0, 1.0)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    progress = tqdm(total=3, desc="Verifying clone tracking", unit="step")
    progress.set_postfix_str("discovering valid wells")
    available = eligible_wells(args.data_root)
    progress.update()
    selected = balanced_strategy_sample(available, lambda item: strategy_from_sample(item[1]), args.seed)
    args.sample_count = len(selected)
    progress.total = args.sample_count + 3
    progress.refresh()
    progress.update()

    review_dir = args.verification_root / "single_clone_tracking" / f"sample_n{args.sample_count}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Clone-tracking verification\n\nThis review checks whether resistant cell regions keep a consistent clone identity as time passes.\n\nEach PNG shows one strategy-balanced review replicate at an early, middle, and late timepoint. The first image shows the whole colony (sensitive blue, resistant yellow), followed by the resistant-cell input, clone labels, and a clone-colour overlay. The colourbar identifies clone IDs.\n\nA PASS means that every cleaned resistant region has a clone label in the checked frames and the output agrees with the production tracking preparation. The metadata file records the checked frames and clone merges.\n",
        encoding="utf-8",
    )
    results: list[dict[str, object]] = []
    for plate, well in selected:
        progress.set_postfix_str(f"{plate}: {well}")
        plate_dir = args.data_root / plate
        raw_stack = imread(plate_dir / "colony_masks_stacks" / f"{well}_res_colony_masks.tif")
        sensitive_stack = imread(plate_dir / "colony_masks_stacks" / f"{well}_sus_colony_masks.tif")
        clone_stack = imread(plate_dir / "clone_masks_stacks" / f"{well}_res_clone_masks.tif")
        if raw_stack.ndim == 2:
            raw_stack = raw_stack[None, ...]
            sensitive_stack = sensitive_stack[None, ...]
            clone_stack = clone_stack[None, ...]
        valid = available[(plate, well)]
        frame_indices = [valid[0], valid[len(valid) // 2], valid[-1]]
        fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
        frame_checks: list[dict[str, object]] = []
        labels_present = np.unique(clone_stack[frame_indices])
        max_label = int(labels_present.max()) if len(labels_present) else 0
        clone_cmap = plt.get_cmap("turbo", max_label + 1)
        clone_norm = BoundaryNorm(np.arange(-0.5, max_label + 1.5), clone_cmap.N)
        clone_image = None
        for row, frame in enumerate(frame_indices):
            raw = raw_stack[frame]
            sensitive = sensitive_stack[frame]
            clone_labels = clone_stack[frame]
            processed = preprocess(raw, SELEM_RAD, MIN_SIZE)
            matches = bool(np.array_equal(clone_labels > 0, processed))
            incoming_distance = (
                None if frame == 0 else link_distance_for_frame_pair(frame - 1, frame)
            )
            frame_checks.append({
                "frame": frame,
                "day": day_from_frame_index(frame),
                "incoming_link_distance_px": incoming_distance,
                "processed_mask_matches_labels": matches,
            })
            axes[row, 0].imshow(colony_overview(sensitive, raw), interpolation="nearest")
            distance_text = "initial frame" if incoming_distance is None else f"link from prior frame: {incoming_distance} px"
            axes[row, 0].set_title(f"Full colony\nday {day_from_frame_index(frame)} ({distance_text})")
            axes[row, 1].imshow(raw, cmap="gray", interpolation="nearest")
            axes[row, 1].set_title("Input resistant mask")
            clone_image = axes[row, 2].imshow(clone_labels, cmap=clone_cmap, norm=clone_norm, interpolation="nearest")
            axes[row, 2].set_title("Clone labels")
            axes[row, 3].imshow(colony_overview(sensitive, raw), interpolation="nearest")
            axes[row, 3].imshow(np.ma.masked_equal(clone_labels, 0), cmap=clone_cmap, norm=clone_norm, interpolation="nearest", alpha=0.75)
            axes[row, 3].set_title("Clone overlay")
            for axis in axes[row]:
                axis.set_axis_off()
        if clone_image is not None:
            colorbar = fig.colorbar(clone_image, ax=axes[:, 2], ticks=labels_present, shrink=0.8)
            colorbar.set_label("Clone ID (1 = unassigned; 2+ = tracked clone)")
        sample_passed = raw_stack.shape == clone_stack.shape and all(item["processed_mask_matches_labels"] for item in frame_checks)
        fig.suptitle(f"{'PASS' if sample_passed else 'FAIL'}: {plate}, {well} | merge events: {merge_count(plate_dir, well)}", fontsize=14)
        figure_path = review_dir / f"{plate}_{well}.png"
        fig.savefig(figure_path, dpi=400)
        plt.close(fig)
        results.append({"plate": plate, "well": well, "frames": frame_checks, "merge_events": merge_count(plate_dir, well), "passed": sample_passed})
        progress.update()

    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({
        "script_under_test": "analysis/basic/microscopy/single_clone_tracking.py",
        "production_imports": ["preprocess", "SELEM_RAD", "MIN_SIZE", "day_from_frame_index", "link_distance_for_frame_pair"],
        "selection": "Strategy-balanced cohort after EXCLUDE_SAMPLES exclusions and day cutoffs; ATAgent1 excluded.",
        "sample_count": args.sample_count,
        "seed": args.seed,
        "passed": passed,
        "samples": results,
    }, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
