#!/usr/bin/env python3
"""Generate Figure 3 density maps using the Figure 2 mask-based cell filter.

This is an independent pipeline.  It never modifies source HDF5 files, the
existing simulation-equivalent-filter density-map run, or any Figure 3 script.

For every requested timepoint it applies the masks used by the Figure 2 escape
pipeline:
* sensitive positions must lie in ``sensitive_clean_masks_stacks``;
* resistant positions must lie in *any* clone mask (label >= 2), not just the
  dominant clone.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as maps
from shared.data_paths import intermediate_dir


OUTPUT_ROOT = intermediate_dir(
    "experimental", "figure_3", "figure2_mask_filtered_mastercurve", create=False
)
RUN_NAME = "run_kernel_160_coarse_16_figure2_masks_all_clones"
# Set True to replace the filtered density-map TIFFs and audit table when this
# file is run from PyCharm.
OVERWRITE = True
CONFIG = Path(__file__).with_name("density_maps_config_160_figure2_masks_all_clones.yaml")
CLONE_LABEL_MIN = 2
SENS_CLEAN_SUBDIR = "sensitive_clean_masks_stacks"
SENS_CLEAN_SUFFIX = "_sus_clean_masks.tif"
CLONE_SUBDIR = "clone_masks_stacks"
CLONE_SUFFIX = "_res_clone_masks.tif"
# Figure 2's cleaned sensitive masks are an intermediate Y: output, organized
# one folder per plate.  Clone masks remain alongside the Y: HDF5 snapshots.
SENSITIVE_MASK_ROOT = intermediate_dir(
    "experimental", "basic", "escape", "sensitive_clean_masks_stacks", create=False
)
_Y_UNC_PREFIX = r"\\syn03.frontend.storage.int.mpl.mpg.de\guck_division2"


def day_to_frame(day: int) -> int | None:
    """Inverse of Figure 2's Mon/Wed/Fri acquisition-day schedule."""
    if day < 1:
        return None
    week, offset = divmod(day - 1, 7)
    offset_to_frame = {0: 0, 2: 1, 4: 2}
    if offset not in offset_to_frame:
        return None
    return 3 * week + offset_to_frame[offset]


def mask_stack_path(pointcloud: Path, well: str, subdir: str, suffix: str) -> Path:
    plate_dir = pointcloud.parent.parent
    return plate_dir / subdir / f"{well}{suffix}"


def positions_in_mask(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    height, width = mask.shape
    xy = frame[["x", "y"]].to_numpy(dtype=float, copy=False)
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    keep = np.zeros(len(frame), dtype=bool)
    keep[inside] = mask[yi[inside], xi[inside]]
    return frame.loc[keep].reset_index(drop=True)


def mapped_y_path(path: str | os.PathLike[str]) -> str:
    """Keep legacy laboratory writes on the mapped-drive view of the share.

    The density-map implementation resolves Y: to a UNC path.  On this
    workstation that UNC view does not expose newly created Plate-6+ output
    folders, while the mapped Y: drive does.  Inputs and outputs remain on the
    same shared drive; this only avoids the inconsistent path namespace. Paths
    below a reviewer-configured local data root pass through unchanged.
    """
    text = os.fspath(path)
    if text.lower().startswith(_Y_UNC_PREFIX.lower()):
        return "Y:\\" + text[len(_Y_UNC_PREFIX):].lstrip("\\/")
    return text


def extended_windows_path(path: str | os.PathLike[str]) -> str:
    """Return a Windows extended-length path when a mapped path needs it.

    Plate 6 and later have descriptive well and TIFF names which make the
    complete path longer than Windows' legacy 260-character limit.  Directory
    enumeration can still succeed in that form, while ``os.path.exists`` used
    by the change writer cannot.  Use one canonical path form for both reads
    and writes instead.
    """
    text = mapped_y_path(path)
    if text.startswith("\\\\?\\"):
        return text
    # Prefix every mapped-drive path, rather than only an already-long
    # directory.  ``write_test_density_changes`` appends a long TIFF filename
    # after receiving this directory, so a seemingly short parent can still
    # cross MAX_PATH at its first existence check.
    if os.name == "nt" and len(text) >= 3 and text[1:3] == ":\\":
        return "\\\\?\\" + text
    return text


def write_tiff_on_mapped_y(path: str | os.PathLike[str], *args, **kwargs):
    """Write through a Y: file handle, avoiding tifffile's UNC conversion."""
    target = Path(os.fspath(path))
    target_text = str(target)
    is_mapped_y = target_text.lower().startswith("y:")
    is_extended_mapped_y = target_text.lower().startswith("\\\\?\\y:")
    if not is_mapped_y and not is_extended_mapped_y:
        return _ORIGINAL_TIFF_WRITE(path, *args, **kwargs)
    # Plate 6+ plus the descriptive TIFF names exceed Windows' legacy
    # MAX_PATH limit.  The extended-length form keeps the exact same Y: file.
    write_target = Path(extended_windows_path(target))
    write_target.parent.mkdir(parents=True, exist_ok=True)
    with write_target.open("wb") as handle:
        return _ORIGINAL_TIFF_WRITE(handle, *args, **kwargs)


_ORIGINAL_TIFF_WRITE = tifffile.imwrite


def mask_frames(pointcloud: Path, well: str, timepoint: str) -> tuple[np.ndarray, np.ndarray]:
    day = int(maps.normalize_timepoint(timepoint))
    frame_index = day_to_frame(day)
    if frame_index is None:
        raise ValueError(f"No Figure 2 mask-stack frame exists for day {day}: {pointcloud}")
    plate_name = pointcloud.parent.parent.name
    sensitive_path = SENSITIVE_MASK_ROOT / plate_name / f"{well}{SENS_CLEAN_SUFFIX}"
    clone_path = mask_stack_path(pointcloud, well, CLONE_SUBDIR, CLONE_SUFFIX)
    if not sensitive_path.is_file() or not clone_path.is_file():
        raise FileNotFoundError(
            f"Figure 2 masks required for {well}, d{day:02d}: "
            f"sensitive={sensitive_path}, clone={clone_path}"
        )
    with tifffile.TiffFile(sensitive_path) as sensitive_tiff, tifffile.TiffFile(clone_path) as clone_tiff:
        if frame_index >= len(sensitive_tiff.pages) or frame_index >= len(clone_tiff.pages):
            raise IndexError(f"Mask frame {frame_index} is unavailable for {well}, d{day:02d}")
        sensitive = sensitive_tiff.pages[frame_index].asarray().astype(bool, copy=False)
        clone = clone_tiff.pages[frame_index].asarray() >= CLONE_LABEL_MIN
    if sensitive.shape != clone.shape:
        raise ValueError(f"Mask-shape mismatch for {well}, d{day:02d}: {sensitive.shape} vs {clone.shape}")
    return sensitive, clone


def density_map_directories(plates: tuple[str, ...] | None = None) -> list[Path]:
    """Discover existing per-well density-map directories on the mapped Y: drive."""
    run_root = OUTPUT_ROOT / RUN_NAME
    selected_plates = plates or tuple(
        path.name for path in run_root.glob("Plate_*") if path.is_dir()
    )
    result: list[Path] = []
    for plate in selected_plates:
        plate_dir = run_root / plate
        if not plate_dir.is_dir():
            raise FileNotFoundError(f"Missing Figure 2 density-map plate directory: {plate_dir}")
        result.extend(
            path / "density_maps"
            for path in plate_dir.iterdir()
            if path.is_dir() and (path / "density_maps").is_dir()
        )
    return sorted(result, key=lambda path: str(path).lower())


def write_relative_density_changes(map_dirs: list[Path]) -> None:
    """Write resistant-density changes from the supplied filtered map folders."""
    suffix = "_res_density_epanechnikov.tif"
    for maps_dir in map_dirs:
        changes_dir = maps_dir.parent / "density_changes"
        if not OVERWRITE and changes_dir.exists() and any(changes_dir.iterdir()):
            print(f"[skip] density changes already exist: {changes_dir} (set OVERWRITE = True to replace).")
            continue
        well = maps_dir.parent.name
        prefix = f"{well}_"
        timepoints = sorted(
            path.name[len(prefix):-len(suffix)]
            for path in maps_dir.glob(f"{well}_*{suffix}")
        )
        maps.write_test_density_changes(
            directory_path="",
            well_identifier=well,
            timepoints=timepoints,
            density_maps_folder=extended_windows_path(maps_dir),
            density_changes_folder=extended_windows_path(maps_dir.parent / "density_changes"),
            channels=("res",),
            write_zero_initial_growth_channel=False,
            zero_initial_density_threshold=1.0e-10,
            measurable_growth_threshold=1.0e-12,
        )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--write-existing-density-changes",
        action="store_true",
        help="Create change TIFFs from existing filtered density maps without regenerating maps.",
    )
    parser.add_argument(
        "--plates",
        nargs="+",
        metavar="PLATE",
        help="Limit --write-existing-density-changes to specific plates.",
    )
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_ROOT / "point_filter_counts.tsv"

    if args.write_existing_density_changes:
        tifffile.imwrite = write_tiff_on_mapped_y
        write_relative_density_changes(density_map_directories(tuple(args.plates) if args.plates else None))
        return
    if args.plates:
        parser.error("--plates requires --write-existing-density-changes")
    if audit_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output already exists: {audit_path}. Set OVERWRITE = True to replace it."
        )

    audit_rows: list[dict[str, object]] = []
    created_map_dirs: set[Path] = set()
    original_create = maps.create_density_map_for_timepoint
    tifffile.imwrite = write_tiff_on_mapped_y

    def figure2_mask_filtered_create(**kwargs):
        pointcloud = Path(kwargs["pointcloud_filepath"])
        timepoint = maps.normalize_timepoint(kwargs["timepoint"])
        well = maps.well_identifier_from_pointcloud_filename(pointcloud)
        density_maps_dir = Path(kwargs["directory_path"]) / kwargs["density_maps_folder"]
        expected = [
            density_maps_dir / f"{well}_{timepoint}_{channel}_density_epanechnikov.tif"
            for channel in ("sus", "res", "total")
        ]
        if not OVERWRITE and any(path.exists() for path in expected):
            print(f"[skip] {well}, d{int(timepoint):02d}: density map exists (set OVERWRITE = True to replace).")
            return False
        susceptible, resistant = maps.pointcloud_read(
            pointcloud, timepoint, require_timepoint=kwargs.get("require_timepoint", False)
        )
        try:
            sensitive_mask, all_clone_mask = mask_frames(pointcloud, well, timepoint)
        except (FileNotFoundError, IndexError) as error:
            # Figure 2 only evaluates frames for which its mask stacks exist.
            # Do not fall back to raw HDF5 positions: omit the unmatched day
            # so every density value in this separate run has the same filter.
            print(f"[skip] {well}, d{int(timepoint):02d}: {error}")
            audit_rows.append({
                "pointcloud": str(pointcloud), "timepoint": timepoint,
                "filter": "skipped_no_matching_figure2_mask_frame",
                "clone_label_min": CLONE_LABEL_MIN,
                "sus_before": len(susceptible), "sus_after": np.nan,
                "res_before": len(resistant), "res_after": np.nan,
            })
            pd.DataFrame(audit_rows).to_csv(
                audit_path, sep="\t", index=False
            )
            return False
        susceptible_filtered = positions_in_mask(susceptible, sensitive_mask)
        resistant_filtered = positions_in_mask(resistant, all_clone_mask)

        saved_reader = maps.pointcloud_read
        maps.pointcloud_read = lambda *_args, **_kwargs: (susceptible_filtered, resistant_filtered)
        try:
            kwargs = dict(kwargs)
            kwargs["directory_path"] = mapped_y_path(kwargs["directory_path"])
            kwargs["density_maps_folder"] = mapped_y_path(kwargs["density_maps_folder"])
            print(f"  → output directory: {kwargs['density_maps_folder']}")
            result = original_create(**kwargs)
            if result:
                # Keep the exact mapped Y: output location. Re-discovering
                # Plate-6+ folders through a UNC view can omit directories
                # that were successfully written during this same run.
                created_map_dirs.add(Path(kwargs["density_maps_folder"]))
        finally:
            maps.pointcloud_read = saved_reader

        audit_rows.append({
            "pointcloud": str(pointcloud),
            "timepoint": timepoint,
            "filter": "figure2_clean_sensitive_and_all_clone_resistant",
            "clone_label_min": CLONE_LABEL_MIN,
            "sus_before": len(susceptible), "sus_after": len(susceptible_filtered),
            "res_before": len(resistant), "res_after": len(resistant_filtered),
        })
        pd.DataFrame(audit_rows).to_csv(audit_path, sep="\t", index=False)
        return result

    maps.create_density_map_for_timepoint = figure2_mask_filtered_create
    maps.OVERWRITE = OVERWRITE
    os.environ["CORESEED_OUTPUT_RUN_NAME"] = RUN_NAME
    sys.argv = [sys.argv[0], "--config", str(CONFIG)]
    maps.main()
    pd.DataFrame(audit_rows).to_csv(audit_path, sep="\t", index=False)

    # The master-curve aggregator discovers valid experimental intervals from
    # these relative-change TIFF names.  They are derived only from the newly
    # written, mask-filtered density maps.
    write_relative_density_changes(sorted(created_map_dirs, key=lambda path: str(path).lower()))


if __name__ == "__main__":
    main()
