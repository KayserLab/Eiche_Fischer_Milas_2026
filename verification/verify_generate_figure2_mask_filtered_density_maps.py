"""Verify the Figure-2-mask filtering applied before Figure 3 density maps."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as maps
from analysis.figure_3 import generate_figure2_mask_filtered_density_maps as filtered_maps
from shared.data_paths import EXP_RAW_DATA, VERIFICATION_DATA, intermediate_dir
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample
from verification.verify_density_maps_common import map_settings


DEFAULT_MAP_ROOT = intermediate_dir(
    "experimental", "figure_3", "figure2_mask_filtered_mastercurve", create=False
)
DEFAULT_EXPERIMENT_ROOT = EXP_RAW_DATA
DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA
RUN_NAME = filtered_maps.RUN_NAME


@dataclass(frozen=True)
class Sample:
    plate: str
    well: str
    strategy: str
    pointcloud: Path
    maps_dir: Path


def discover(map_root: Path, experiment_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for maps_dir in (map_root / RUN_NAME).rglob("density_maps"):
        well = maps_dir.parent.name
        plate = maps_dir.parent.parent.name
        if well in EXCLUDE_SAMPLES and EXCLUDE_SAMPLES[well] is None:
            continue
        matches = sorted((experiment_root / plate / "hdf5").glob(f"*{well}_total_poisson_padded.h5"))
        if len(matches) == 1:
            samples.append(Sample(plate, well, maps.strategy_name_from_well_identifier(well), matches[0], maps_dir))
    return samples


def days(sample: Sample) -> list[str]:
    suffix = "_total_density_epanechnikov.tif"
    values = [path.name[len(sample.well) + 1:-len(suffix)] for path in sample.maps_dir.glob(f"{sample.well}_*{suffix}")]
    cutoff = EXCLUDE_SAMPLES.get(sample.well)
    return [day for day in sorted(set(values), key=int) if cutoff is None or int(day) <= cutoff]


def visual_days(available: list[str]) -> list[str]:
    return [available[index] for index in dict.fromkeys((0, len(available) // 2, len(available) - 1))]


def inside(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype=bool)
    xy = frame[["x", "y"]].to_numpy(dtype=float, copy=False)
    x, y = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    result = np.zeros(len(frame), dtype=bool)
    result[valid] = mask[y[valid], x[valid]]
    return result


def plot_points(axis: plt.Axes, frame: pd.DataFrame, colour: str, label: str, coarsening: int, alpha: float = 0.8) -> None:
    if not frame.empty:
        axis.scatter(frame.x / coarsening, frame.y / coarsening, s=1.5, c=colour, linewidths=0, alpha=alpha, label=label)


def inspect_day(sample: Sample, day: str) -> tuple[dict[str, object], tuple]:
    # The Figure 3 production script deliberately writes these TIFFs through
    # the mapped Y: drive.  For Plate 6+ the equivalent UNC spelling can list
    # the directory but cannot reopen its files on this workstation.
    def map_path(channel: str) -> Path:
        mapped = filtered_maps.mapped_y_path(
            sample.maps_dir / f"{sample.well}_{day}_{channel}_density_epanechnikov.tif"
        )
        return Path(filtered_maps.extended_windows_path(mapped))

    total_path = map_path("total")
    radius_um, coarsening, _ = map_settings(total_path)
    def read_tiff(path: Path) -> np.ndarray:
        # Do not pass the pathname to tifffile: it converts mapped Y: paths
        # back to UNC paths on this workstation (see map_settings above).
        with path.open("rb") as stream:
            return tifffile.imread(stream)

    saved = tuple(read_tiff(map_path(channel)) for channel in ("sus", "res", "total"))
    sensitive, resistant = maps.pointcloud_read(sample.pointcloud, day, require_timepoint=True)
    sensitive_mask, clone_mask = filtered_maps.mask_frames(sample.pointcloud, sample.well, day)
    keep_sensitive = inside(sensitive, sensitive_mask)
    keep_resistant = inside(resistant, clone_mask)
    # Use the filtering function from the script under test, rather than a
    # duplicate implementation, for the density recalculation.
    filtered_sensitive = filtered_maps.positions_in_mask(sensitive, sensitive_mask)
    filtered_resistant = filtered_maps.positions_in_mask(resistant, clone_mask)
    image_shape = (saved[2].shape[0] * coarsening, saved[2].shape[1] * coarsening)
    expected = maps.density(filtered_sensitive, filtered_resistant, radius_um, image_shape, coarsening)
    raw_total = maps.density(sensitive, resistant, radius_um, image_shape, coarsening)[2]
    audit = {
        "day": int(day),
        "sensitive_before": len(sensitive), "sensitive_retained": len(filtered_sensitive),
        "resistant_before": len(resistant), "resistant_retained": len(filtered_resistant),
        # Check the points that survived the production filter.  The previous
        # version accidentally tested every raw point, so any correctly removed
        # point made this verification PNG report FAIL.
        "all_retained_sensitive_inside_clean_mask": bool(inside(filtered_sensitive, sensitive_mask).all()),
        "all_retained_resistant_inside_clone_mask": bool(inside(filtered_resistant, clone_mask).all()),
        "saved_maps_match_density_from_retained_points": bool(all(np.allclose(a, b, rtol=1e-6, atol=1e-12) for a, b in zip(expected, saved, strict=True))),
        "total_map_additive": bool(np.allclose(saved[2], saved[0] + saved[1], rtol=1e-5, atol=1e-9)),
    }
    return audit, (sensitive, resistant, filtered_sensitive, filtered_resistant, sensitive_mask, clone_mask, raw_total, saved[2], coarsening)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    selected = balanced_strategy_sample(discover(args.map_root, args.experiment_root), lambda item: item.strategy, args.seed)
    if not selected:
        raise RuntimeError(f"No valid Figure-2-mask-filtered density-map replicates found in {args.map_root / RUN_NAME}")
    review_dir = args.verification_root / "generate_figure2_mask_filtered_density_maps" / f"sample_n{len(selected)}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Figure 2 mask-filtered density-map verification\n\nThis review checks the extra filtering performed before these density maps are made. It starts with all measured sensitive and resistant cell positions, then retains sensitive cells only inside the cleaned sensitive mask and resistant cells only inside any labelled clone mask. Orange points were excluded by that rule.\n\nEach row is one acquisition day. The first panel overlays the two masks on all raw positions. The second shows exactly which points are retained or excluded. The final panels compare the density calculated from all raw points with the saved density after filtering; their difference shows the density contribution removed by the masks.\n\nA PASS means every retained point lies in its required mask, and the saved TIFFs match a fresh production density calculation from precisely those retained points.\n",
        encoding="utf-8",
    )

    progress = tqdm(total=len(selected) + 2, desc="Verifying Figure 2 mask-filtered density maps", unit="replicate")
    results = []
    for sample in selected:
        progress.set_postfix_str(f"{sample.plate}: {sample.well}")
        available = days(sample)
        if not available:
            raise ValueError(f"No saved maps for {sample.plate} {sample.well}")
        inspected = [inspect_day(sample, day) for day in visual_days(available)]
        checks = [item[0] for item in inspected]
        passed = all(all(value for key, value in check.items() if isinstance(value, bool)) for check in checks)
        fig, axes = plt.subplots(len(inspected), 4, figsize=(15, 3.7 * len(inspected)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, (check, payload) in enumerate(inspected):
            sensitive, resistant, kept_sens, kept_res, sensitive_mask, clone_mask, raw_total, filtered_total, coarsening = payload
            mask_rgb = np.zeros((*sensitive_mask.shape, 3), dtype=float)
            mask_rgb[..., 2] = sensitive_mask * 0.55
            mask_rgb[..., 0] = clone_mask * 0.75
            mask_rgb[..., 1] = clone_mask * 0.6
            axes[row, 0].imshow(mask_rgb[::coarsening, ::coarsening], origin="upper")
            plot_points(axes[row, 0], sensitive, "#268bd2", "raw sensitive", coarsening, 0.45)
            plot_points(axes[row, 0], resistant, "#f1c40f", "raw resistant", coarsening, 0.45)
            plot_points(axes[row, 1], sensitive.loc[~inside(sensitive, sensitive_mask)], "#ff6f00", "excluded", coarsening)
            plot_points(axes[row, 1], resistant.loc[~inside(resistant, clone_mask)], "#ff6f00", "excluded", coarsening)
            plot_points(axes[row, 1], kept_sens, "#268bd2", "retained sensitive", coarsening)
            plot_points(axes[row, 1], kept_res, "#f1c40f", "retained resistant", coarsening)
            vmax = max(float(raw_total.max()), float(filtered_total.max()), np.finfo(float).eps)
            axes[row, 2].imshow(raw_total, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            axes[row, 3].imshow(raw_total - filtered_total, cmap="Oranges", vmin=0, vmax=vmax, interpolation="nearest")
            for column, title in enumerate(("raw points + masks", "retained / excluded points", "density from all raw points", "density removed by masks")):
                axes[row, column].set_title(f"Day {check['day']}: {title}", fontsize=9)
                axes[row, column].set_axis_off()
        handles, labels = axes[0, 1].get_legend_handles_labels()
        axes[0, 1].legend(handles, labels, fontsize=7, markerscale=4, loc="upper right")
        fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {sample.plate}, {sample.well}", fontsize=13)
        fig.savefig(review_dir / f"{sample.plate}_{sample.well}_{sample.strategy}.png", dpi=400)
        plt.close(fig)
        results.append({"plate": sample.plate, "sample": sample.well, "strategy": sample.strategy, "passed": passed, "frames": checks})
        progress.update()
    passed = all(result["passed"] for result in results)
    (review_dir / "metadata.json").write_text(json.dumps({"script_under_test": "analysis/figure_3/generate_figure2_mask_filtered_density_maps.py", "production_imports": ["mask_frames", "positions_in_mask", "pointcloud_read", "density"], "selection": "Strategy-balanced cohort; experiment_filters exclusions and cutoffs applied.", "passed": passed, "samples": results}, indent=2), encoding="utf-8")
    progress.update(2)
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
