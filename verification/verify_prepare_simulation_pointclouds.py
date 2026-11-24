"""Create per-run visual audits for Prepare_Simulation_Pointclouds.py."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.basic.density_maps.Prepare_Simulation_Pointclouds import (
    DEFAULT_OUTPUT_ROOT,
    PREPARED_SUFFIX,
    PreparationSettings,
    _pointcloud_values,
    load_active_cells,
    outlier_mask,
)
from shared.data_paths import VERIFICATION_DATA
from verification.selection import balanced_strategy_sample


DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def text_attr(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def settings_from_output(handle: h5py.File) -> PreparationSettings:
    attrs = handle.attrs
    return PreparationSettings(
        active_flag=int(attrs["active_flag"]), type_dataset=text_attr(attrs["type_dataset"]),
        sensitive_value=int(attrs["sensitive_value"]), resistant_value=int(attrs["resistant_value"]),
        position_scale_px=float(attrs["position_scale_px"]), outlier_filter=bool(attrs["outlier_filter"]),
        filter_center=tuple(float(value) for value in attrs["filter_center"]),
        filter_outlier_boundary=float(attrs["filter_outlier_boundary"]),
        filter_neighbour_radius=float(attrs["filter_neighbour_radius"]),
        filter_min_neighbours=int(attrs["filter_min_neighbours"]), overwrite=False,
    )


def plot_frame(axis: plt.Axes, raw_xy: np.ndarray, classes: np.ndarray, keep: np.ndarray, title: str, center_px: np.ndarray, half_span_px: float) -> None:
    removed = ~keep
    if np.any(removed):
        axis.scatter(raw_xy[removed, 0], raw_xy[removed, 1], s=6, c="#ff6b00", alpha=0.9, linewidths=0, label="removed")
    for class_id, color, label in ((0, "#268bd2", "sensitive"), (1, "#f1c40f", "resistant")):
        selected = keep & (classes == class_id)
        if np.any(selected):
            axis.scatter(raw_xy[selected, 0], raw_xy[selected, 1], s=3, c=color, alpha=0.7, linewidths=0, label=label)
    axis.set_xlim(center_px[0] - half_span_px, center_px[0] + half_span_px)
    axis.set_ylim(center_px[1] - half_span_px, center_px[1] + half_span_px)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_axis_off()
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="upper right", fontsize=7, markerscale=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    progress = tqdm(total=3, desc="Verifying simulation point clouds", unit="step")
    progress.set_postfix_str("discovering prepared runs")
    available = sorted(args.output_root.glob(f"Plate_*/hdf5/*{PREPARED_SUFFIX}"))
    progress.update()
    selected = balanced_strategy_sample(available, lambda path: path.parent.parent.name.removeprefix("Plate_"), args.seed)
    args.sample_count = len(selected)
    progress.total = args.sample_count + 3
    progress.refresh()
    progress.update()

    review_dir = args.verification_root / "Prepare_Simulation_Pointclouds" / f"sample_n{args.sample_count}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Simulation point-cloud verification\n\nThis review checks how raw simulation cells are converted into filtered point clouds for later analyses.\n\nEach PNG shows one strategy-balanced simulation run at three timepoints. The left panels show all active known cells after coordinate scaling; the right panels show the saved filtered result. Sensitive cells are blue, resistant cells are yellow, and removed cells are bright orange.\n\nA PASS means the stored simulation coordinates exactly match the production filtering and coordinate conversion. The metadata file records the checked timepoints.\n",
        encoding="utf-8",
    )
    results: list[dict[str, object]] = []
    for output_path in selected:
        progress.set_postfix_str(output_path.name)
        with h5py.File(output_path, mode="r") as output:
            settings = settings_from_output(output)
            center_px = np.asarray(settings.filter_center) * settings.position_scale_px
            half_span_px = settings.filter_outlier_boundary * settings.position_scale_px
            timepoints = sorted(output["res"], key=lambda key: int(key.removeprefix("d_")))
            chosen = [timepoints[0], timepoints[len(timepoints) // 2], timepoints[-1]]
            fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
            checks: list[dict[str, object]] = []
            for row, timepoint in enumerate(chosen):
                source_path = Path(text_attr(output[f"res/{timepoint}"].attrs["source_file"]))
                positions, classes = load_active_cells(source_path, settings)
                keep = outlier_mask(
                    positions, center=settings.filter_center, boundary=settings.filter_outlier_boundary,
                    neighbour_radius=settings.filter_neighbour_radius, min_neighbours=settings.filter_min_neighbours,
                ) if settings.outlier_filter else np.ones(len(positions), dtype=bool)
                raw_px = positions * settings.position_scale_px
                filtered_px = raw_px[keep]
                filtered_classes = classes[keep]
                expected_sensitive = _pointcloud_values(filtered_px[filtered_classes == 0])
                expected_resistant = _pointcloud_values(filtered_px[filtered_classes == 1])
                stored_sensitive = np.asarray(output[f"sus/{timepoint}/block0_values"])
                stored_resistant = np.asarray(output[f"res/{timepoint}/block0_values"])
                checks.append({
                    "timepoint": timepoint,
                    "sensitive_coordinates_match": bool(np.array_equal(expected_sensitive, stored_sensitive)),
                    "resistant_coordinates_match": bool(np.array_equal(expected_resistant, stored_resistant)),
                    "raw_cells": int(len(raw_px)),
                    "retained_cells": int(len(filtered_px)),
                })
                plot_frame(axes[row, 0], raw_px, classes, np.ones(len(raw_px), dtype=bool), f"Raw scaled input: {timepoint}", center_px, half_span_px)
                plot_frame(axes[row, 1], raw_px, classes, keep, "Saved filtered output (shared scale)", center_px, half_span_px)
            passed = all(item["sensitive_coordinates_match"] and item["resistant_coordinates_match"] for item in checks)
            plate = text_attr(output.attrs["plate"])
            run_id = text_attr(output.attrs["run_id"])
        fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {plate}, {run_id}", fontsize=14)
        fig.savefig(review_dir / f"{plate}_{run_id}.png", dpi=400)
        plt.close(fig)
        results.append({"plate": plate, "run_id": run_id, "output": str(output_path), "passed": passed, "frames": checks})
        progress.update()

    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({
        "script_under_test": "analysis/basic/density_maps/Prepare_Simulation_Pointclouds.py",
        "production_imports": ["load_active_cells", "outlier_mask", "_pointcloud_values"],
        "sample_count": args.sample_count, "seed": args.seed, "passed": passed, "samples": results,
    }, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
