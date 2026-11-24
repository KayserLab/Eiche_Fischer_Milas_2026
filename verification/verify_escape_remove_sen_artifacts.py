"""Create per-replicate visual audits for Escape_Remove_Sen_Artifacts.py.

The verifier is read-only and uses the production reconstruction and artifact-
splitting functions to compare HDF5 sensitive points with cleaned mask stacks.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import TiffFile, imread
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.escape.Escape_Remove_Sen_Artifacts import (
    DEFAULT_OUT_DIR,
    SENS_OUT_SUBDIR,
    build_day_to_frame,
    extract_sus_days,
    load_sus_xy,
    sensitive_link_distance,
    split_sensitive_points_by_artifact,
)
from shared.data_paths import EXP_RAW_DATA, VERIFICATION_DATA
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample, strategy_from_sample


DEFAULT_DATA_ROOT = EXP_RAW_DATA
DEFAULT_CLEAN_ROOT = Path(DEFAULT_OUT_DIR) / SENS_OUT_SUBDIR
DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def eligible_replicates(data_root: Path, clean_root: Path) -> dict[tuple[str, str], Path]:
    """Find non-excluded outputs without loading their large TIFF/HDF5 inputs."""
    eligible: dict[tuple[str, str], Path] = {}
    for plate_dir in sorted(data_root.glob("Plate_*")):
        for cleaned_path in (clean_root / plate_dir.name).glob("*_sus_clean_masks.tif"):
            sample = cleaned_path.name.removesuffix("_sus_clean_masks.tif")
            cutoff = EXCLUDE_SAMPLES.get(sample, float("inf"))
            if cutoff is None:
                continue
            h5_matches = sorted((plate_dir / "hdf5").glob(f"*_{sample}_total_poisson*.h5"))
            if not h5_matches:
                continue
            eligible[(plate_dir.name, sample)] = h5_matches[0]
    return eligible


def reconstruct_selected_frames(h5_path: Path, shape: tuple[int, int], n_frames: int, selected: list[int]) -> dict[int, dict[str, np.ndarray | int]]:
    """Recreate the production sensitive-cleaning state up to selected frames."""
    height, width = shape
    wanted = set(selected)
    output: dict[int, dict[str, np.ndarray | int]] = {}
    previous_labels: np.ndarray | None = None
    previous_sensitive_day: int | None = None
    day_to_frame = build_day_to_frame(n_frames)
    with pd.HDFStore(h5_path, mode="r") as store:
        sensitive_days = set(extract_sus_days(store))
        for day, frame in sorted(day_to_frame.items()):
            if day in sensitive_days:
                raw_xy = load_sus_xy(store, day)
                link_distance = (
                    sensitive_link_distance(previous_sensitive_day, day)
                    if previous_labels is not None and np.any(previous_labels)
                    else None
                )
                kept_xy, removed_xy, previous_labels = split_sensitive_points_by_artifact(
                    sus_xy=raw_xy,
                    h=height,
                    w=width,
                    prev_assigned_lbl=previous_labels,
                    link_distance=link_distance,
                )
                previous_sensitive_day = day
                if frame in wanted:
                    output[frame] = {
                        "day": day,
                        "incoming_link_distance_px": link_distance,
                        "raw_xy": raw_xy,
                        "kept_xy": kept_xy,
                        "removed_xy": removed_xy,
                        "expected_mask": (previous_labels > 0).astype(np.uint8),
                    }
    return output


def plot_points(axis: plt.Axes, points: np.ndarray, shape: tuple[int, int], color: str, title: str) -> None:
    axis.set_facecolor("black")
    if len(points):
        axis.scatter(points[:, 0], points[:, 1], s=2, c=color, linewidths=0)
    axis.set_xlim(0, shape[1])
    axis.set_ylim(shape[0], 0)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_axis_off()


def colony_image(sensitive: np.ndarray, resistant: np.ndarray) -> np.ndarray:
    """Render sensitive blue, resistant yellow, and overlap white."""
    image = np.zeros((*sensitive.shape, 3), dtype=float)
    image[(sensitive > 0) & (resistant == 0)] = (0.15, 0.45, 1.0)
    image[(resistant > 0) & (sensitive == 0)] = (1.0, 0.82, 0.05)
    image[(sensitive > 0) & (resistant > 0)] = (1.0, 1.0, 1.0)
    return image


def plot_before_colony(axis: plt.Axes, raw_sensitive: np.ndarray, resistant: np.ndarray, shape: tuple[int, int], title: str) -> None:
    axis.imshow(colony_image(np.zeros(shape, dtype=np.uint8), resistant), interpolation="nearest")
    if len(raw_sensitive):
        axis.scatter(raw_sensitive[:, 0], raw_sensitive[:, 1], s=2, c="#268bd2", linewidths=0)
    axis.set_xlim(0, shape[1])
    axis.set_ylim(shape[0], 0)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_axis_off()


def verify_replicate(plate: str, sample: str, h5_path: Path, data_root: Path, clean_root: Path, review_dir: Path) -> dict[str, object]:
    """Verify one replicate; safe to run independently in a worker process."""
    clean_path = clean_root / plate / f"{sample}_sus_clean_masks.tif"
    with TiffFile(clean_path) as clean_tiff:
        n_frames = len(clean_tiff.pages)
        frame_shape = clean_tiff.pages[0].shape
        with pd.HDFStore(h5_path, mode="r") as store:
            sensitive_days = set(extract_sus_days(store))
        cutoff = EXCLUDE_SAMPLES.get(sample, float("inf"))
        valid_frames = [frame for day, frame in build_day_to_frame(n_frames).items() if day in sensitive_days and day <= cutoff]
        if len(valid_frames) < 4:
            raise ValueError(f"{plate}/{sample} has fewer than four valid sensitive frames.")
        last_pair_start = len(valid_frames) - 2
        frames = valid_frames[round(last_pair_start / 3) : round(last_pair_start / 3) + 2] + valid_frames[round(2 * last_pair_start / 3) : round(2 * last_pair_start / 3) + 2]
        clean_frames = {frame: clean_tiff.pages[frame].asarray() for frame in frames}
    clone_stack = imread(data_root / plate / "clone_masks_stacks" / f"{sample}_res_clone_masks.tif")
    if clone_stack.ndim == 2:
        clone_stack = clone_stack[None, ...]
    expected = reconstruct_selected_frames(h5_path, frame_shape, n_frames, frames)
    fig, axes = plt.subplots(4, 3, figsize=(14, 17), constrained_layout=True)
    frame_checks: list[dict[str, object]] = []
    for row, frame in enumerate(frames):
        data, output_mask, resistant_mask = expected[frame], clean_frames[frame], clone_stack[frame] > 0
        matches = bool(np.array_equal(output_mask > 0, data["expected_mask"] > 0))
        frame_checks.append({"frame": frame, "day": data["day"], "incoming_link_distance_px": data["incoming_link_distance_px"], "raw_points": int(len(data["raw_xy"])), "retained_points": int(len(data["kept_xy"])), "removed_points": int(len(data["removed_xy"])), "output_matches_production_reconstruction": matches})
        plot_before_colony(axes[row, 0], data["raw_xy"], resistant_mask, output_mask.shape, f"Before cleaning\nday {data['day']}")
        for column in (1, 2):
            axes[row, column].imshow(colony_image(output_mask, resistant_mask), interpolation="nearest")
            if len(data["kept_xy"]):
                axes[row, column].scatter(data["kept_xy"][:, 0], data["kept_xy"][:, 1], s=2, c="#268bd2", linewidths=0)
            axes[row, column].set_axis_off()
        distance_text = "initial sensitive frame" if data["incoming_link_distance_px"] is None else f"link to prior sensitive frame: {data['incoming_link_distance_px']} px"
        axes[row, 1].set_title(f"After cleaning ({distance_text})\n{int(output_mask.sum()):,} retained mask pixels")
        if len(data["removed_xy"]):
            axes[row, 2].scatter(data["removed_xy"][:, 0], data["removed_xy"][:, 1], s=2, c="#e74c3c", linewidths=0)
        axes[row, 2].set_title(f"Removed sensitive points (red)\n{len(data['removed_xy']):,} points")
    passed = all(item["output_matches_production_reconstruction"] for item in frame_checks)
    fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {plate}, {sample}", fontsize=14)
    fig.savefig(review_dir / f"{plate}_{sample}.png", dpi=400)
    plt.close(fig)
    return {"plate": plate, "sample": sample, "frames": frame_checks, "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--workers", type=int, default=2, help="Independent replicates processed in parallel.")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    progress = tqdm(total=3, desc="Verifying sensitive artifact removal", unit="step")
    progress.set_postfix_str("discovering valid replicates")
    available = eligible_replicates(args.data_root, args.clean_root)
    progress.update()
    selected = balanced_strategy_sample(available, lambda item: strategy_from_sample(item[1]), args.seed)
    args.sample_count = len(selected)
    progress.total = args.sample_count + 3
    progress.refresh()
    progress.update()

    review_dir = args.verification_root / "Escape_Remove_Sen_Artifacts" / f"sample_n{args.sample_count}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Sensitive-artifact removal verification\n\nThis review checks the step that removes isolated or implausible sensitive-cell detections before escape analysis.\n\nEach PNG shows one strategy-balanced review replicate at two consecutive early/middle timepoints and two consecutive later timepoints. The first panel is the full colony before cleaning, the second is the full colony after cleaning, and the third highlights removed sensitive points in red. Sensitive cells are blue and resistant cells are yellow. The after-cleaning panel states the applied temporal linking distance: 300 px after two days and 450 px after three days.\n\nA PASS means the saved cleaned mask agrees with the same filtering procedure used by the production pipeline. The metadata file records how many points were retained and removed.\n",
        encoding="utf-8",
    )
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(verify_replicate, plate, sample, available[(plate, sample)], args.data_root, args.clean_root, review_dir): (plate, sample)
            for plate, sample in selected
        }
        for future in as_completed(futures):
            plate, sample = futures[future]
            progress.set_postfix_str(f"{plate}: {sample}")
            results.append(future.result())
            progress.update()

    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({
        "script_under_test": "analysis/basic/escape/Escape_Remove_Sen_Artifacts.py",
        "production_imports": ["build_day_to_frame", "extract_sus_days", "load_sus_xy", "split_sensitive_points_by_artifact"],
        "sample_count": args.sample_count, "seed": args.seed, "passed": passed, "samples": results,
    }, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
