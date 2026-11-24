"""Create per-replicate visual audits for Prepare_Escape_Analysis_Inputs.py."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
import tifffile
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.escape.Prepare_Escape_Analysis_Inputs import (
    PLATES,
    determine_clone_eligibility,
    discover_replicates,
    load_xy,
    mask_membership,
    validate_source_alignment,
)
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample
from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import PIXEL_SIZE_UM
from shared.data_paths import VERIFICATION_DATA


DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def eligible_replicates() -> list[object]:
    """Return existing prepared outputs retained by the canonical filters."""
    output = []
    for rep in discover_replicates(PLATES):
        cutoff = EXCLUDE_SAMPLES.get(rep.sample, float("inf"))
        if cutoff is not None and rep.output_path.is_file():
            output.append(rep)
    return output


def plot_points(axis: plt.Axes, xy: np.ndarray, keep: np.ndarray, title: str, color: str) -> None:
    excluded = xy[(~keep) & np.isfinite(xy).all(axis=1)]
    retained = xy[keep]
    if len(excluded):
        axis.scatter(excluded[:, 0], excluded[:, 1], s=1, c="#808080", alpha=0.25, linewidths=0)
    if len(retained):
        axis.scatter(retained[:, 0], retained[:, 1], s=1, c=color, alpha=0.7, linewidths=0)
    axis.set_title(title)
    axis.set_axis_off()


def add_link_distance_scale_bar(axis: plt.Axes, shape: tuple[int, int]) -> None:
    """Show the 300-px sensitive-region linking distance on a spatial panel."""
    height, width = shape
    x0, y0 = 0.05 * width, 0.93 * height
    label = f"300 px = {300 * PIXEL_SIZE_UM:.0f} µm"
    outline = [patheffects.withStroke(linewidth=3, foreground="black")]
    axis.plot([x0, x0 + 300], [y0, y0], color="white", linewidth=3, path_effects=outline)
    axis.text(x0 + 150, y0 - 0.02 * height, label, color="white", ha="center", va="bottom", fontsize=8, path_effects=outline)


def same_points(expected: np.ndarray, stored: pd.DataFrame) -> bool:
    return np.array_equal(expected, stored[["x", "y"]].to_numpy(dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    progress = tqdm(total=3, desc="Verifying escape inputs", unit="step")
    progress.set_postfix_str("discovering prepared replicates")
    available = eligible_replicates()
    progress.update()
    selected = balanced_strategy_sample(available, lambda rep: rep.strategy, args.seed)
    args.sample_count = len(selected)
    progress.total = args.sample_count + 3
    progress.refresh()
    progress.update()

    review_dir = args.verification_root / "Prepare_Escape_Analysis_Inputs" / f"sample_n{args.sample_count}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Escape-analysis input verification\n\nThis review checks the final selection of sensitive and resistant cells that is passed into the escape analysis.\n\nEach PNG shows one strategy-balanced review replicate at three timepoints. Sensitive cells are checked against their cleaned blue region. Resistant cells are checked against eligible tracked-clone regions in yellow; grey points are not retained.\n\nA PASS means the saved HDF5 point lists exactly match the production selection rules. The metadata file also records which resistant clones were eligible.\n",
        encoding="utf-8",
    )
    results: list[dict[str, object]] = []
    for rep in selected:
        progress.set_postfix_str(f"{rep.plate}: {rep.sample}")
        cutoff = EXCLUDE_SAMPLES.get(rep.sample, float("inf"))
        with (
            pd.HDFStore(rep.raw_h5_path, mode="r") as raw_store,
            pd.HDFStore(rep.output_path, mode="r") as prepared_store,
            tifffile.TiffFile(rep.clone_mask_path) as clone_tiff,
            tifffile.TiffFile(rep.clean_sensitive_path) as sensitive_tiff,
        ):
            days = validate_source_alignment(raw_store, clone_tiff, sensitive_tiff)
            clone_qc = determine_clone_eligibility(raw_store, clone_tiff, days)
            eligible_labels = clone_qc.loc[clone_qc["eligible"], "clone_label"].to_numpy(dtype=np.int64)
            frame_indices = [i for i, day in enumerate(days) if day <= cutoff]
            frames = [frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1]]
            fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
            frame_checks: list[dict[str, object]] = []
            for row, frame in enumerate(frames):
                day = days[frame]
                clone_mask = clone_tiff.pages[frame].asarray()
                sensitive_mask = sensitive_tiff.pages[frame].asarray()
                resistant_raw = load_xy(raw_store, "res", day)
                sensitive_raw = load_xy(raw_store, "sus", day)
                resistant_keep, _, _, labels = mask_membership(resistant_raw, clone_mask, resistant=True)
                resistant_keep &= np.isin(labels, eligible_labels)
                sensitive_keep, _, _, _ = mask_membership(sensitive_raw, sensitive_mask, resistant=False)
                stored_resistant = prepared_store.get(f"/res/d_{day:02d}")
                stored_sensitive = prepared_store.get(f"/sus/d_{day:02d}")
                checks = {
                    "resistant_rows_match": same_points(resistant_raw[resistant_keep], stored_resistant),
                    "sensitive_rows_match": same_points(sensitive_raw[sensitive_keep], stored_sensitive),
                }
                frame_checks.append({
                    "frame": frame, "day": day, "checks": checks,
                    "resistant_raw": int(len(resistant_raw)), "resistant_retained": int(resistant_keep.sum()),
                    "sensitive_raw": int(len(sensitive_raw)), "sensitive_retained": int(sensitive_keep.sum()),
                })

                axes[row, 0].imshow(sensitive_mask, cmap="Blues", interpolation="nearest")
                plot_points(axes[row, 0], sensitive_raw, sensitive_keep, f"Sensitive: day {day}", "#268bd2")
                add_link_distance_scale_bar(axes[row, 0], sensitive_mask.shape)
                eligible_mask = np.isin(clone_mask, eligible_labels)
                axes[row, 1].imshow(eligible_mask, cmap="Wistia", interpolation="nearest")
                plot_points(axes[row, 1], resistant_raw, resistant_keep, f"Resistant: day {day}", "#f1c40f")

        sample_passed = all(all(item["checks"].values()) for item in frame_checks)
        fig.suptitle(f"{'PASS' if sample_passed else 'FAIL'}: {rep.plate}, {rep.sample}", fontsize=14)
        fig.savefig(review_dir / f"{rep.plate}_{rep.sample}.png", dpi=400)
        plt.close(fig)
        results.append({
            "plate": rep.plate, "sample": rep.sample, "passed": sample_passed,
            "frames": frame_checks,
            "clone_eligibility": clone_qc.to_dict(orient="records"),
        })
        progress.update()

    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({
        "script_under_test": "analysis/basic/escape/Prepare_Escape_Analysis_Inputs.py",
        "production_imports": ["validate_source_alignment", "determine_clone_eligibility", "mask_membership", "load_xy"],
        "sample_count": args.sample_count, "seed": args.seed, "passed": passed, "samples": results,
    }, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
