"""Shared visual audit helpers for the experimental and simulation density maps."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (
    PIXEL_SIZE_UM,
    density,
    pointcloud_read,
)


DESCRIPTION_RADIUS_RE = re.compile(r"radius_um=([0-9.]+)")
DESCRIPTION_COARSENING_RE = re.compile(r"coarsening_factor=(\d+)")
DESCRIPTION_ORIGIN_RE = re.compile(r"origin_([xy])_px=([-+0-9.eE]+)")


@dataclass(frozen=True)
class DensitySample:
    plate: str
    identifier: str
    strategy: str
    source_hdf5: Path
    maps_dir: Path
    simulation: bool = False
    last_valid_day: float | None = None


def density_timepoints(sample: DensitySample) -> list[str]:
    suffix = "_total_density_epanechnikov.tif"
    days = []
    for path in sample.maps_dir.glob(f"*{suffix}"):
        base = path.name.removesuffix(suffix)
        if "_" in base:
            _, day = base.rsplit("_", 1)
            if day.isdigit():
                days.append(day)
    valid = sorted(set(days), key=int)
    if sample.last_valid_day is not None:
        valid = [day for day in valid if int(day) <= sample.last_valid_day]
    return valid


def map_path(sample: DensitySample, day: str, channel: str) -> Path:
    return sample.maps_dir / f"{sample.identifier}_{day}_{channel}_density_epanechnikov.tif"


def map_settings(path: Path) -> tuple[float, int, tuple[float, float]]:
    # Passing a mapped network-drive pathname to tifffile can make it resolve
    # the drive back to its UNC form.  On this workstation some newer Plate
    # output folders can be enumerated but not reopened through that UNC form.
    # Opening the mapped path ourselves keeps the exact, working namespace.
    with Path(path).open("rb") as stream, tifffile.TiffFile(stream) as handle:
        description = handle.pages[0].description or ""
    radius = DESCRIPTION_RADIUS_RE.search(description)
    coarsening = DESCRIPTION_COARSENING_RE.search(description)
    if radius is None or coarsening is None:
        raise ValueError(f"Density TIFF metadata is incomplete: {path}")
    origins = {axis: float(value) for axis, value in DESCRIPTION_ORIGIN_RE.findall(description)}
    return float(radius.group(1)), int(coarsening.group(1)), (origins.get("x", 0.0), origins.get("y", 0.0))


def load_input(sample: DensitySample, day: str, origin_px: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not sample.simulation:
        return pointcloud_read(str(sample.source_hdf5), day, require_timepoint=True)
    with h5py.File(sample.source_hdf5, "r") as handle:
        def read(channel: str) -> pd.DataFrame:
            values = np.asarray(handle[f"{channel}/d_{day}/block0_values"])
            points = values[:, [1, 0]].astype(float, copy=False)
            return pd.DataFrame({"x": points[:, 0] - origin_px[0], "y": points[:, 1] - origin_px[1]})
        return read("sus"), read("res")


def choose_visual_days(days: list[str]) -> list[str]:
    indices = (0, len(days) // 2, len(days) - 1)
    return [days[index] for index in indices]


def scatter_points(axis: plt.Axes, sensitive: pd.DataFrame, resistant: pd.DataFrame, coarsening: int) -> None:
    if not sensitive.empty:
        axis.scatter(sensitive.x / coarsening, sensitive.y / coarsening, s=1.2, c="#268bd2", linewidths=0, alpha=0.7, label="sensitive")
    if not resistant.empty:
        axis.scatter(resistant.x / coarsening, resistant.y / coarsening, s=1.2, c="#f1c40f", linewidths=0, alpha=0.75, label="resistant")


def draw_scale_bar(axis: plt.Axes, width: int, height: int, coarsening: int) -> None:
    length = 100.0 / (PIXEL_SIZE_UM * coarsening)
    x, y = width * 0.05, height * 0.92
    axis.plot((x, x + length), (y, y), color="white", linewidth=2.5, solid_capstyle="butt")
    axis.text(x, y - height * 0.035, "100 µm", color="white", fontsize=7, va="top")


def draw_density(axis: plt.Axes, image: np.ndarray, coarsening: int, cmap: str, title: str, vmax: float) -> None:
    axis.imshow(image, cmap=cmap, vmin=0, vmax=max(vmax, np.finfo(float).eps), interpolation="nearest", origin="upper")
    draw_scale_bar(axis, image.shape[1], image.shape[0], coarsening)
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def check_day(sample: DensitySample, day: str) -> tuple[dict[str, object], tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, int, float]]:
    total_path = map_path(sample, day, "total")
    radius_um, coarsening, origin_px = map_settings(total_path)
    stored_sus, stored_res, stored_total = (tifffile.imread(map_path(sample, day, channel)) for channel in ("sus", "res", "total"))
    sensitive, resistant = load_input(sample, day, origin_px)
    image_shape = (int(stored_total.shape[0]) * coarsening, int(stored_total.shape[1]) * coarsening)
    expected_sus, expected_res, expected_total = density(sensitive, resistant, radius_um, image_shape, coarsening)
    recomputed = all(np.allclose(expected, stored, rtol=1e-6, atol=1e-12) for expected, stored in zip((expected_sus, expected_res, expected_total), (stored_sus, stored_res, stored_total), strict=True))
    # The three FFTs are evaluated independently in float32, so their final
    # rounding differs by about 1e-9 cells/µm² even though convolution is linear.
    additive = bool(np.allclose(stored_total, stored_sus + stored_res, rtol=1e-5, atol=1e-9))
    check = {
        "day": int(day), "radius_um": radius_um, "coarsening_factor": coarsening,
        "sensitive_points": len(sensitive), "resistant_points": len(resistant),
        "saved_maps_match_production": recomputed, "total_equals_sensitive_plus_resistant": additive,
        "max_recalculation_abs_error": float(max(np.max(np.abs(expected - stored)) for expected, stored in zip((expected_sus, expected_res, expected_total), (stored_sus, stored_res, stored_total), strict=True))),
    }
    return check, (sensitive, resistant, stored_sus, stored_res, stored_total, coarsening, radius_um)


def audit_samples(samples: list[DensitySample], review_dir: Path, description: str, script_under_test: str) -> None:
    progress = tqdm(total=len(samples) + 3, desc=description, unit="replicate")
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Density-map verification\n\nThis review checks that cell positions are converted into the saved local-density maps. The calculation spreads each cell's contribution smoothly over a circular neighbourhood; the kernel radius and 100 µm scale bar are shown in each PNG.\n\nEach PNG contains early, middle, and late timepoints for one strategy-balanced replicate. The first column shows the original sensitive (blue) and resistant (yellow) cell positions. The next three columns show the saved sensitive, resistant, and total density maps without point overlays, so the smooth density field is visible. Dense clusters should create strong, centred density regions.\n\nA PASS means the saved TIFF maps exactly match a fresh calculation using the production density function, and that the total map equals the sensitive plus resistant maps within floating-point precision.\n",
        encoding="utf-8",
    )
    progress.update()
    results: list[dict[str, object]] = []
    for sample in samples:
        progress.set_postfix_str(f"{sample.plate}: {sample.identifier}")
        days = density_timepoints(sample)
        if len(days) < 1:
            raise ValueError(f"No total density maps found in {sample.maps_dir}")
        selected_days = choose_visual_days(days)
        loaded = [check_day(sample, day) for day in selected_days]
        checks = [item[0] for item in loaded]
        passed = all(item["saved_maps_match_production"] and item["total_equals_sensitive_plus_resistant"] for item in checks)
        fig, axes = plt.subplots(len(loaded), 4, figsize=(14, 3.6 * len(loaded)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, (check, payload) in enumerate(loaded):
            sensitive, resistant, sus_map, res_map, total_map, coarsening, radius_um = payload
            scatter_points(axes[row, 0], sensitive, resistant, coarsening)
            axes[row, 0].add_patch(Circle((total_map.shape[1] * 0.88, total_map.shape[0] * 0.12), radius_um / (PIXEL_SIZE_UM * coarsening), edgecolor="black", facecolor="none", linewidth=1.0))
            draw_scale_bar(axes[row, 0], total_map.shape[1], total_map.shape[0], coarsening)
            axes[row, 0].set_xlim(0, total_map.shape[1]); axes[row, 0].set_ylim(total_map.shape[0], 0)
            axes[row, 0].set_aspect("equal"); axes[row, 0].set_axis_off()
            axes[row, 0].set_title(f"Day {check['day']}: raw positions\nkernel radius {radius_um:g} µm", fontsize=9)
            draw_density(axes[row, 1], sus_map, coarsening, "Blues", "sensitive density", float(sus_map.max()))
            draw_density(axes[row, 2], res_map, coarsening, "YlOrBr", "resistant density", float(res_map.max()))
            draw_density(axes[row, 3], total_map, coarsening, "magma", "total density", float(total_map.max()))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, loc="upper left", fontsize=7, markerscale=4)
        fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {sample.plate}, {sample.identifier}", fontsize=13)
        fig.savefig(review_dir / f"{sample.plate}_{sample.identifier}.png", dpi=400)
        plt.close(fig)
        results.append({"plate": sample.plate, "sample": sample.identifier, "strategy": sample.strategy, "source_hdf5": str(sample.source_hdf5), "maps_dir": str(sample.maps_dir), "passed": passed, "frames": checks})
        progress.update()
    passed = all(result["passed"] for result in results)
    (review_dir / "metadata.json").write_text(json.dumps({"script_under_test": script_under_test, "production_imports": ["density", "pointcloud_read"], "selection": "Strategy-balanced cohort; ATAgent1 excluded.", "passed": passed, "samples": results}, indent=2), encoding="utf-8")
    progress.update(2)
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")
