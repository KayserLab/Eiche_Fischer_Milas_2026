#!/usr/bin/env python3
"""
Render HDF5 cell positions as per-timepoint scatter PNGs and an MP4 movie.

The script reads the active Density_Maps_Epanechnikov_coarsegrained.py config, resolves the
configured HDF5 wells/files, plots all available HDF5 timepoints, and writes the
frames and movies under the configured Plate_*/test_output folder.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(
    parents=True,
    exist_ok=True,
)

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TEST_OUTPUT_FOLDER,
    PIXEL_SIZE_UM,
    density_canvas_shape,
    load_config,
    normalize_plates,
    plate_well_designators,
    pointcloud_read,
    pointcloud_filenames,
    resolve_output_run_dir,
    resolve_experiment_path,
    resolve_pointcloud_file,
    strategy_name_from_well_identifier,
    use_structured_output_layout,
    well_output_type_dir,
    well_designator_from_well_identifier,
    well_identifier_from_pointcloud_filename,
)

BACKGROUND_COLOR = "#05070A"
TEXT_COLOR = "#F5F7FA"
SCALE_BAR_COLOR = "#F5F7FA"
RESISTANT_COLOR = "#FFD166"
SENSITIVE_COLOR = "#5DA9FF"
POINTCLOUD_SUFFIX_PRIORITY = (
    "_total_poisson_padded.h5",
    "_total_poisson.h5",
    "_total.h5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all sensitive/resistant cell positions from the configured "
            "HDF5 wells/files as PNG frames and combine them into MP4 movies."
        )
    )
    parser.add_argument(
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help=f"Density-map config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Frames per second for the MP4 movie. Default: 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG render DPI. Default: 180.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.35,
        help="Scatter marker size. Default: 0.35.",
    )
    parser.add_argument(
        "--scale-bar-um",
        type=float,
        default=None,
        help=(
            "Scale-bar length in micrometers. By default, the script chooses a "
            "nice length from the field of view."
        ),
    )
    parser.add_argument(
        "--output-subfolder",
        default="hdf5_cell_position_scatter",
        help="Subfolder inside test_output for PNG frames and MP4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing PNG frames before rendering.",
    )
    return parser.parse_args()


def normalize_config_list(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError as exc:
        raise TypeError(f"Config key {key!r} must be a scalar or list.") from exc


def first_config_list(config: dict, keys: tuple[str, ...]) -> tuple[list[str], str | None]:
    for key in keys:
        values = normalize_config_list(config.get(key), key)
        if values:
            return values, key
    return [], None


def pointcloud_suffix_priority(path: Path) -> tuple[int, str]:
    for index, suffix in enumerate(POINTCLOUD_SUFFIX_PRIORITY):
        if path.name.endswith(suffix):
            return index, path.name
    return len(POINTCLOUD_SUFFIX_PRIORITY), path.name


def pointcloud_suffixes(poisson: bool) -> tuple[str, ...]:
    if poisson:
        return ("_total_poisson_padded.h5", "_total_poisson.h5")
    return ("_total.h5",)


def hdf5_directory(experiment_path: Path, plate: str, hdf5_path: str | None) -> Path:
    if hdf5_path:
        return Path(hdf5_path).expanduser()
    return experiment_path / plate / "hdf5"


def looks_like_hdf5_file(selector: str) -> bool:
    lowered = selector.lower()
    return (
        lowered.endswith((".h5", ".hdf5"))
        or "/" in selector
        or "\\" in selector
    )


def resolve_well_pointcloud(
    *,
    experiment_path: Path,
    plate: str,
    well_identifier: str,
    hdf5_path: str | None,
    poisson: bool,
) -> Path:
    directory = hdf5_directory(experiment_path, plate, hdf5_path)
    if not directory.exists():
        raise FileNotFoundError(f"Missing HDF5 directory: {directory}")

    matches = [
        directory / filename
        for filename in pointcloud_filenames(
            os.fspath(directory),
            poisson,
            well_designators=[well_identifier],
        )
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a point-cloud HDF5 for well {well_identifier!r} "
            f"in {directory}"
        )
    return sorted(matches, key=pointcloud_suffix_priority)[0]


def resolve_configured_pointclouds(
    *,
    config: dict,
    experiment_path: Path,
    plate: str,
) -> tuple[list[Path], str]:
    hdf5_path = config.get("hdf5_path")
    poisson = bool(config.get("poisson", True))

    plate_wells = plate_well_designators(config, plate)
    well_selectors, source_key = first_config_list(
        config,
        ("scatter_movie_wells",),
    )
    if not well_selectors and plate_wells:
        well_selectors = plate_wells
        source_key = "plates.wells"

    if well_selectors:
        pointcloud_paths = []
        for selector in well_selectors:
            if looks_like_hdf5_file(selector):
                pointcloud_paths.append(
                    Path(
                        resolve_pointcloud_file(
                            experiment_path=os.fspath(experiment_path),
                            plate=plate,
                            hdf5_filename=selector,
                            hdf5_path=hdf5_path,
                        )
                    )
                )
            else:
                pointcloud_paths.append(
                    resolve_well_pointcloud(
                        experiment_path=experiment_path,
                        plate=plate,
                        well_identifier=selector,
                        hdf5_path=hdf5_path,
                        poisson=poisson,
                    )
                )
        for pointcloud_path in pointcloud_paths:
            well_identifier = well_identifier_from_pointcloud_filename(pointcloud_path)
            strategy_name = strategy_name_from_well_identifier(well_identifier)
            strategy_text = f", strategy={strategy_name}" if strategy_name else ""
            print(
                "Selected movie well "
                f"{well_designator_from_well_identifier(well_identifier)} "
                f"as {well_identifier}{strategy_text}"
            )
        return dedupe_paths(pointcloud_paths), source_key or "scatter_movie_wells"

    file_selectors, source_key = first_config_list(
        config,
        ("scatter_movie_hdf5_files", "test_hdf5_files", "test_hdf5_file"),
    )
    if not file_selectors:
        raise ValueError(
            "Config must define scatter_movie_wells, scatter_movie_hdf5_files, "
            "test_hdf5_files, or test_hdf5_file."
        )

    pointcloud_paths = [
        Path(
            resolve_pointcloud_file(
                experiment_path=os.fspath(experiment_path),
                plate=plate,
                hdf5_filename=selector,
                hdf5_path=hdf5_path,
            )
        )
        for selector in file_selectors
    ]
    return dedupe_paths(pointcloud_paths), source_key or "test_hdf5_file"


def dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser()
        key = resolved.resolve() if resolved.exists() else resolved
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def timepoint_sort_key(timepoint: str) -> tuple[int, int | str]:
    return (0, int(timepoint)) if timepoint.isdigit() else (1, timepoint)


def format_day_label(timepoint: str) -> str:
    return f"Day {int(timepoint)}" if timepoint.isdigit() else f"Day {timepoint}"


def list_hdf5_timepoints(hdf5_path: Path, unit: str = "d") -> list[str]:
    prefix = f"{unit}_"
    timepoints: set[str] = set()
    with h5py.File(hdf5_path, "r") as h5:
        for group_name in ("sus", "res"):
            if group_name not in h5:
                continue
            for key in h5[group_name].keys():
                if key.startswith(prefix):
                    timepoints.add(key[len(prefix) :])
    return sorted(timepoints, key=timepoint_sort_key)


def resolve_canvas_limits(
    plate_dir: Path,
    well_identifier: str,
    first_timepoint: str,
    pointcloud_path: Path,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    canvas_shape = density_canvas_shape(
        directory_path=os.fspath(plate_dir),
        well_identifier=well_identifier,
        timepoint=first_timepoint,
        pointcloud_filepath=os.fspath(pointcloud_path),
    )
    if canvas_shape is None:
        return None

    height, width = canvas_shape
    return (0.0, float(width)), (float(height), 0.0)


def nice_scale_bar_length_um(visible_width_um: float) -> float | None:
    if not np.isfinite(visible_width_um) or visible_width_um <= 0:
        return None

    target = visible_width_um * 0.18
    exponent = math.floor(math.log10(target))
    candidates = [
        multiplier * (10 ** exp)
        for exp in range(exponent - 1, exponent + 2)
        for multiplier in (1, 2, 5)
    ]
    viable = [candidate for candidate in candidates if candidate <= target]
    return float(max(viable) if viable else min(candidates))


def format_scale_bar_label(length_um: float) -> str:
    if length_um >= 1000:
        return f"{length_um / 1000:g} mm"
    return f"{length_um:g} um"


def add_day_and_scale_bar(
    ax: plt.Axes,
    *,
    timepoint: str,
    scale_bar_um: float | None,
) -> None:
    ax.text(
        0.045,
        0.945,
        format_day_label(timepoint),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=18,
        fontweight="bold",
    )

    xlim = ax.get_xlim()
    visible_width_px = abs(xlim[1] - xlim[0])
    if visible_width_px <= 0:
        return

    chosen_scale_bar_um = scale_bar_um or nice_scale_bar_length_um(
        visible_width_px * PIXEL_SIZE_UM
    )
    if chosen_scale_bar_um is None or chosen_scale_bar_um <= 0:
        return

    scale_bar_px = chosen_scale_bar_um / PIXEL_SIZE_UM
    x_min, x_max = sorted(xlim)
    x_end = x_max - (visible_width_px * 0.06)
    x_start = x_end - scale_bar_px
    if x_start < x_min:
        x_start = x_min + (visible_width_px * 0.06)
        x_end = x_start + scale_bar_px

    transform = ax.get_xaxis_transform()
    ax.plot(
        [x_start, x_end],
        [0.07, 0.07],
        color=SCALE_BAR_COLOR,
        linewidth=4,
        solid_capstyle="butt",
        transform=transform,
        clip_on=False,
    )
    ax.text(
        (x_start + x_end) / 2,
        0.095,
        format_scale_bar_label(chosen_scale_bar_um),
        transform=transform,
        ha="center",
        va="bottom",
        color=SCALE_BAR_COLOR,
        fontsize=11,
        fontweight="bold",
    )


def plot_frame(
    *,
    pointcloud_path: Path,
    timepoint: str,
    frame_path: Path,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    marker_size: float,
    dpi: int,
    scale_bar_um: float | None,
) -> tuple[int, int]:
    sus_positions, res_positions = pointcloud_read(
        os.fspath(pointcloud_path),
        timepoint,
        require_timepoint=True,
    )

    fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor=BACKGROUND_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BACKGROUND_COLOR)
    ax.scatter(
        sus_positions["x"],
        sus_positions["y"],
        s=marker_size,
        c=SENSITIVE_COLOR,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        res_positions["x"],
        res_positions["y"],
        s=marker_size,
        c=RESISTANT_COLOR,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    add_day_and_scale_bar(
        ax,
        timepoint=timepoint,
        scale_bar_um=scale_bar_um,
    )

    fig.savefig(frame_path, facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_COLOR)
    plt.close(fig)
    return len(sus_positions), len(res_positions)


def write_movie(frame_dir: Path, movie_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to write the MP4 movie, but it was not found on PATH."
        )

    frame_pattern = frame_dir / "frame_%03d.png"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:g}",
        "-i",
        os.fspath(frame_pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        os.fspath(movie_path),
    ]
    subprocess.run(cmd, check=True)


def render_pointcloud_movie(
    *,
    args: argparse.Namespace,
    config: dict,
    experiment_path: Path,
    plate: str,
    pointcloud_path: Path,
) -> None:
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"Missing HDF5 point-cloud file: {pointcloud_path}")

    timepoints = list_hdf5_timepoints(pointcloud_path)
    if not timepoints:
        raise ValueError(f"No HDF5 timepoints found in {pointcloud_path}")

    well_identifier = well_identifier_from_pointcloud_filename(pointcloud_path)
    plate_dir = experiment_path / plate
    if use_structured_output_layout(config):
        output_dir = well_output_type_dir(
            config,
            experiment_path,
            plate,
            well_identifier,
            "scatter_movie_output_type",
            args.output_subfolder,
        )
        frame_dir = output_dir / "frames"
    else:
        test_output_folder = config.get("test_output_folder", DEFAULT_TEST_OUTPUT_FOLDER)
        output_dir = plate_dir / test_output_folder / args.output_subfolder
        frame_dir = output_dir / well_identifier / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for frame_path in frame_dir.glob("frame_*.png"):
            frame_path.unlink()

    canvas_limits = resolve_canvas_limits(
        plate_dir=plate_dir,
        well_identifier=well_identifier,
        first_timepoint=timepoints[0],
        pointcloud_path=pointcloud_path,
    )
    xlim, ylim = canvas_limits if canvas_limits is not None else (None, None)

    print(
        f"Rendering {plate} {well_identifier}: "
        f"{len(timepoints)} timepoints from {pointcloud_path.name}"
    )
    counts: list[tuple[str, int, int]] = []
    for frame_idx, timepoint in enumerate(timepoints):
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        n_sus, n_res = plot_frame(
            pointcloud_path=pointcloud_path,
            timepoint=timepoint,
            frame_path=frame_path,
            xlim=xlim,
            ylim=ylim,
            marker_size=args.marker_size,
            dpi=args.dpi,
            scale_bar_um=args.scale_bar_um,
        )
        counts.append((timepoint, n_sus, n_res))
        print(
            f"  Saved frame {frame_idx + 1:02d}/{len(timepoints):02d}: "
            f"day {timepoint} ({n_sus:,} sensitive, {n_res:,} resistant)"
        )

    movie_path = output_dir / f"{well_identifier}_cell_positions.mp4"
    write_movie(frame_dir=frame_dir, movie_path=movie_path, fps=args.fps)

    print(f"  Saved {len(timepoints)} PNG frames to: {frame_dir}")
    print(f"  Saved MP4 movie to: {movie_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_path = Path(
        resolve_experiment_path(
            experiment_path=config.get("experiment_path"),
            mount_root=config.get("mount_root"),
            hdf5_path=config.get("hdf5_path"),
        )
    )
    plates = normalize_plates(config.get("plates"))
    if not plates:
        raise ValueError("Config must define at least one plate.")

    if use_structured_output_layout(config):
        print(f"Using output run directory: {resolve_output_run_dir(config, experiment_path)}")

    for plate in plates:
        pointcloud_paths, source_key = resolve_configured_pointclouds(
            config=config,
            experiment_path=experiment_path,
            plate=plate,
        )
        print(
            f"Processing {len(pointcloud_paths)} HDF5 point-cloud file(s) for "
            f"{plate} from config key {source_key}."
        )
        for pointcloud_path in pointcloud_paths:
            render_pointcloud_movie(
                args=args,
                config=config,
                experiment_path=experiment_path,
                plate=plate,
                pointcloud_path=pointcloud_path,
            )


if __name__ == "__main__":
    main()
