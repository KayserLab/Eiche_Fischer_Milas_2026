#!/usr/bin/env python3
"""
Plot resistant density change and growth against first-frame density.

The script uses the active Density_Maps_Epanechnikov_coarsegrained.py config, creates missing
resistant density-change maps when needed, loops over every consecutive
timepoint pair, and writes absolute-change, relative-growth, and zero-initial
growth PNG scatterplots.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(
    parents=True,
    exist_ok=True,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import tifffile as tiff

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TEST_OUTPUT_FOLDER,
    DENSITY_ZERO_FLOOR_UM2,
    load_config,
    normalize_timepoint,
    normalize_plates,
    normalize_timepoints,
    plate_well_designators,
    pointcloud_filenames,
    radius_um_folder_token,
    resolve_coarsening_factor,
    resolve_experiment_path,
    resolve_radius_um,
    resolve_output_run_dir,
    resolve_test_output_subfolder,
    strategy_name_from_well_identifier,
    use_structured_output_layout,
    well_output_type_dir,
    well_designator_from_well_identifier,
    well_identifier_from_pointcloud_filename,
    write_test_density_changes,
)

DEFAULT_SCATTER_PLOT_TYPES = ("absolute", "relative", "zero_initial_growth")
SCATTER_PLOT_TYPE_ALIASES = {
    "absolute": "absolute",
    "abs": "absolute",
    "relative": "relative",
    "rel": "relative",
    "zero_initial": "zero_initial_growth",
    "zero-initial": "zero_initial_growth",
    "zero_initial_growth": "zero_initial_growth",
    "zero-initial-growth": "zero_initial_growth",
}
PIPELINE_OUTPUT_ALIASES = {
    "density_maps": "density_maps",
    "maps": "density_maps",
    "density_stacks": "density_stacks",
    "stacks": "density_stacks",
    "density_changes": "density_changes",
    "changes": "density_changes",
    "change_scatter_plots": "change_scatter_plots",
    "scatter_plots": "change_scatter_plots",
    "scatter": "change_scatter_plots",
    "hdf5_scatter_movie": "hdf5_scatter_movie",
    "scatter_movie": "hdf5_scatter_movie",
    "movie": "hdf5_scatter_movie",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter-plot resistant absolute density change against local total density "
            "in the first test frame, plus zero-initial-resistant pixels against "
            "local sensitive density."
        )
    )
    parser.add_argument(
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help=f"Density-map config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional PNG output path for a single configured timepoint pair. "
            "Defaults to one output per pair in the configured scatter folder."
        ),
    )
    parser.add_argument(
        "--scatter-folder",
        default="scatter_plots",
        help=(
            "Subfolder inside the density-change output folder for scatter PNGs. "
            "Default: scatter_plots."
        ),
    )
    return parser.parse_args()


def plot_change_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    marker_color: np.ndarray,
    color_norm: Normalize,
    color_map: LinearSegmentedColormap,
    first_tp: str,
    ylabel: str,
    title: str,
    zero_initial_density_threshold: float,
    measurable_growth_threshold: float,
    axis_xlim: tuple[float, float] | None,
    axis_ylim: tuple[float, float] | None,
) -> tuple[int, plt.Collection]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(marker_color)
    zero_initial_growth = (
        finite
        & (marker_color <= zero_initial_density_threshold)
        & (y > measurable_growth_threshold)
    )
    positive_initial = finite & ~zero_initial_growth
    positive_idx = np.flatnonzero(positive_initial)
    order = positive_idx[np.argsort(marker_color[positive_idx])]
    scatter = ax.scatter(
        x[order],
        y[order],
        c=marker_color[order],
        cmap=color_map,
        norm=color_norm,
        s=3,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )
    if np.any(zero_initial_growth):
        ax.scatter(
            x[zero_initial_growth],
            y[zero_initial_growth],
            color="#D62728",
            s=3,
            alpha=1.0,
            linewidths=0,
            rasterized=True,
            label="Near-zero initial resistant density with growth",
            zorder=3,
        )
        ax.legend(loc="best", frameon=False, markerscale=3)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_xlabel(f"Total density at day {first_tp} (cells / um2)")
    if axis_xlim is not None:
        ax.set_xlim(*axis_xlim)
    if axis_ylim is not None:
        ax.set_ylim(*axis_ylim)
    return int(finite.sum()), scatter


def plot_relative_change_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    marker_color: np.ndarray,
    valid_growth: np.ndarray,
    color_norm: Normalize,
    color_map: LinearSegmentedColormap,
    first_tp: str,
    next_tp: str,
    zero_initial_density_threshold: float,
    axis_xlim: tuple[float, float] | None,
    axis_ylim: tuple[float, float] | None,
    xscale: str,
    yscale: str,
) -> tuple[int, plt.Collection]:
    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(marker_color)
        & valid_growth
    )
    if xscale == "log":
        finite &= x > 0
    if yscale == "log":
        finite &= y > 0
    valid_idx = np.flatnonzero(finite)
    order = valid_idx[np.argsort(marker_color[valid_idx])]
    scatter = ax.scatter(
        x[order],
        y[order],
        c=marker_color[order],
        cmap=color_map,
        norm=color_norm,
        s=3,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if yscale != "log":
        ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel(f"Total density at day {first_tp} (cells / um2)")
    ax.set_ylabel(
        f"Relative resistant growth {first_tp}->{next_tp} "
        "(ln density ratio / day)"
    )
    ax.set_title("Relative resistant growth per day")
    ax.grid(True, color="0.9", linewidth=0.6)
    set_scaled_axis_limits(
        ax,
        axis="x",
        limits=axis_xlim,
        scale=xscale,
        values=x[finite],
    )
    set_scaled_axis_limits(
        ax,
        axis="y",
        limits=axis_ylim,
        scale=yscale,
        values=y[finite],
    )
    ax.text(
        0.02,
        0.98,
        f"plotted where initial and next res > {zero_initial_density_threshold:g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )
    return int(finite.sum()), scatter


def plot_zero_initial_sensitive_scatter(
    ax: plt.Axes,
    sensitive_density: np.ndarray,
    abs_res_change: np.ndarray,
    initial_res_density: np.ndarray,
    first_tp: str,
    next_tp: str,
    zero_initial_density_threshold: float,
    measurable_growth_threshold: float,
    axis_xlim: tuple[float, float] | None,
    axis_ylim: tuple[float, float] | None,
) -> int:
    finite = (
        np.isfinite(sensitive_density)
        & np.isfinite(abs_res_change)
        & np.isfinite(initial_res_density)
    )
    zero_initial_growth = (
        finite
        & (initial_res_density <= zero_initial_density_threshold)
        & (abs_res_change > measurable_growth_threshold)
    )
    ax.scatter(
        sensitive_density[zero_initial_growth],
        abs_res_change[zero_initial_growth],
        color="#D62728",
        s=3,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
        label="Near-zero initial resistant density with growth",
    )
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel(f"Sensitive density at day {first_tp} (cells / um2)")
    ax.set_ylabel(
        f"Absolute resistant density change {first_tp}->{next_tp} (cells / um2)"
    )
    ax.set_title("Zero-initial-resistant growth pixels")
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.legend(loc="best", frameon=False, markerscale=3)
    if axis_xlim is not None:
        ax.set_xlim(*axis_xlim)
    if axis_ylim is not None:
        ax.set_ylim(*axis_ylim)
    return int(zero_initial_growth.sum())


def companion_output_path(out_path: Path, suffix: str) -> Path:
    output_suffix = out_path.suffix or ".png"
    return out_path.with_name(f"{out_path.stem}{suffix}{output_suffix}")


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    output_format = out_path.suffix.lstrip(".") or "png"
    buffer = BytesIO()
    fig.savefig(buffer, format=output_format, bbox_inches="tight")
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_bytes(buffer.getvalue())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def normalize_config_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return [str(item) for item in value]


def config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def selected_scatter_plot_types(config: dict) -> set[str]:
    values = normalize_config_list(
        config.get("scatter_plot_types", list(DEFAULT_SCATTER_PLOT_TYPES))
    )
    if not values:
        return set()

    selected: set[str] = set()
    for value in values:
        key = value.strip().lower()
        if key == "all":
            return set(DEFAULT_SCATTER_PLOT_TYPES)
        if key not in SCATTER_PLOT_TYPE_ALIASES:
            valid = ", ".join(DEFAULT_SCATTER_PLOT_TYPES)
            raise ValueError(
                f"Unsupported scatter_plot_types entry {value!r}. "
                f"Use one of: {valid}."
            )
        selected.add(SCATTER_PLOT_TYPE_ALIASES[key])
    return selected


def selected_pipeline_outputs(config: dict) -> set[str] | None:
    if "pipeline_outputs" not in config:
        return None

    outputs: set[str] = set()
    for value in normalize_config_list(config.get("pipeline_outputs")):
        key = value.strip().lower()
        if key == "all":
            return set(PIPELINE_OUTPUT_ALIASES.values())
        if key not in PIPELINE_OUTPUT_ALIASES:
            valid = ", ".join(sorted(set(PIPELINE_OUTPUT_ALIASES.values())))
            raise ValueError(
                f"Unsupported pipeline_outputs entry {value!r}. "
                f"Use one of: {valid}."
            )
        outputs.add(PIPELINE_OUTPUT_ALIASES[key])
    return outputs


def pipeline_output_enabled(
    config: dict,
    output_key: str,
    legacy_bool_key: str,
    default: bool,
) -> bool:
    selected_outputs = selected_pipeline_outputs(config)
    if selected_outputs is not None:
        return output_key in selected_outputs
    return config_bool(config, legacy_bool_key, default)


def timepoint_sort_key(timepoint: str) -> tuple[int, int | str]:
    return (0, int(timepoint)) if timepoint.isdigit() else (1, timepoint)


def discover_density_timepoints(
    density_maps_dir: Path,
    well_identifier: str,
    channels: tuple[str, ...] = ("sus", "res", "total"),
) -> list[str]:
    timepoint_sets: list[set[str]] = []
    for channel in channels:
        prefix = f"{well_identifier}_"
        suffix = f"_{channel}_density_epanechnikov.tif"
        channel_timepoints = {
            path.name[len(prefix) : -len(suffix)]
            for path in density_maps_dir.glob(f"{well_identifier}_*{suffix}")
            if path.name.startswith(prefix) and path.name.endswith(suffix)
        }
        timepoint_sets.append(channel_timepoints)

    common_timepoints = set.intersection(*timepoint_sets) if timepoint_sets else set()
    return sorted(common_timepoints, key=timepoint_sort_key)


def selected_batch_wells(config: dict, experiment_path: Path, plate: str) -> list[str]:
    explicit_wells = normalize_config_list(config.get("change_vs_density_wells"))
    if explicit_wells:
        return explicit_wells

    well_designators = plate_well_designators(config, plate)
    hdf5_path = config.get("hdf5_path")
    hdf5_dir = Path(hdf5_path).expanduser() if hdf5_path else experiment_path / plate / "hdf5"
    filenames = pointcloud_filenames(
        os.fspath(hdf5_dir),
        bool(config.get("poisson", True)),
        include_patterns=config.get("hdf5_include_patterns"),
        exclude_patterns=config.get("hdf5_exclude_patterns"),
        well_designators=well_designators,
    )
    if filenames:
        wells = [well_identifier_from_pointcloud_filename(filename) for filename in filenames]
        for well_identifier in wells:
            strategy_name = strategy_name_from_well_identifier(well_identifier)
            strategy_text = f", strategy={strategy_name}" if strategy_name else ""
            print(
                "Selected change/scatter well "
                f"{well_designator_from_well_identifier(well_identifier)} "
                f"as {well_identifier}{strategy_text}"
            )
        return wells

    return normalize_config_list(config.get("scatter_movie_wells"))


def ensure_res_change_channel(channels: tuple[str, ...]) -> tuple[str, ...]:
    if "res" in channels:
        return channels
    return channels + ("res",)


def zero_initial_density_threshold(config: dict) -> float:
    return float(
        config.get(
            "zero_initial_res_density_threshold_um2",
            DENSITY_ZERO_FLOOR_UM2,
        )
    )


def measurable_growth_threshold(config: dict) -> float:
    return float(config.get("res_growth_threshold_um2", DENSITY_ZERO_FLOOR_UM2))


def axis_limits(config: dict, key: str) -> tuple[float, float] | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().strip("[]")
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = list(value)
    if len(values) != 2:
        raise ValueError(f"Config key {key!r} must contain exactly two values.")
    return float(values[0]), float(values[1])


def positive_float_or_none(config: dict, key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError(f"Config key {key!r} must be > 0 when set.")
    return value


def axis_scale(config: dict, key: str, default: str = "linear") -> str:
    value = str(config.get(key, default)).strip().lower()
    if value not in {"linear", "log"}:
        raise ValueError(f"Config key {key!r} must be 'linear' or 'log'.")
    return value


def positive_log_limit(
    limit: float,
    values: np.ndarray,
    fallback: float | None,
) -> float | None:
    if limit > 0:
        return limit
    positive_values = values[np.isfinite(values) & (values > 0)]
    if positive_values.size:
        return float(np.nanmin(positive_values))
    return fallback


def set_scaled_axis_limits(
    ax: plt.Axes,
    *,
    axis: str,
    limits: tuple[float, float] | None,
    scale: str,
    values: np.ndarray,
) -> None:
    if limits is None:
        return

    lower, upper = limits
    if scale == "log":
        lower = positive_log_limit(lower, values, None)
        upper = positive_log_limit(upper, values, None)
        if lower is None or upper is None or lower >= upper:
            return

    if axis == "x":
        ax.set_xlim(lower, upper)
    else:
        ax.set_ylim(lower, upper)


def plot_density_change_pair(
    *,
    plate: str,
    plate_dir: Path,
    density_maps_folder: str,
    density_changes_dir: Path,
    scatter_dir: Path,
    well_identifier: str,
    first_tp: str,
    next_tp: str,
    radius_um: float,
    coarsening_factor: int,
    zero_initial_threshold: float,
    growth_threshold: float,
    axis_xlim: tuple[float, float] | None,
    axis_ylim: tuple[float, float] | None,
    relative_axis_xlim: tuple[float, float] | None,
    relative_axis_ylim: tuple[float, float] | None,
    relative_xscale: str,
    relative_yscale: str,
    colorbar_vmax: float | None,
    scatter_plot_types: set[str],
    output_path: Path | None = None,
) -> list[Path]:
    total_density_path = (
        plate_dir
        / density_maps_folder
        / f"{well_identifier}_{first_tp}_total_density_epanechnikov.tif"
    )
    res_density_path = (
        plate_dir
        / density_maps_folder
        / f"{well_identifier}_{first_tp}_res_density_epanechnikov.tif"
    )
    next_res_density_path = (
        plate_dir
        / density_maps_folder
        / f"{well_identifier}_{next_tp}_res_density_epanechnikov.tif"
    )
    sus_density_path = (
        plate_dir
        / density_maps_folder
        / f"{well_identifier}_{first_tp}_sus_density_epanechnikov.tif"
    )
    abs_change_path = (
        density_changes_dir
        / f"{well_identifier}_{first_tp}_to_{next_tp}_res_density_epanechnikov_abs_diff.tif"
    )
    rel_change_path = (
        density_changes_dir
        / f"{well_identifier}_{first_tp}_to_{next_tp}_res_density_epanechnikov_rel_diff.tif"
    )
    needs_total_density = bool({"absolute", "relative"} & scatter_plot_types)
    needs_sensitive_density = "zero_initial_growth" in scatter_plot_types
    needs_absolute_change = bool({"absolute", "zero_initial_growth"} & scatter_plot_types)
    needs_relative_change = "relative" in scatter_plot_types
    needs_next_res_density = "relative" in scatter_plot_types

    if needs_total_density and not total_density_path.exists():
        raise FileNotFoundError(f"Missing total density map: {total_density_path}")
    if not res_density_path.exists():
        raise FileNotFoundError(f"Missing resistant density map: {res_density_path}")
    if needs_next_res_density and not next_res_density_path.exists():
        raise FileNotFoundError(f"Missing next resistant density map: {next_res_density_path}")
    if needs_sensitive_density and not sus_density_path.exists():
        raise FileNotFoundError(f"Missing sensitive density map: {sus_density_path}")
    if needs_absolute_change and not abs_change_path.exists():
        raise FileNotFoundError(f"Missing resistant absolute change map: {abs_change_path}")
    if needs_relative_change and not rel_change_path.exists():
        raise FileNotFoundError(f"Missing resistant relative growth map: {rel_change_path}")

    total_density = (
        tiff.imread(total_density_path).astype(np.float32, copy=False)
        if needs_total_density
        else None
    )
    res_density = tiff.imread(res_density_path).astype(np.float32, copy=False)
    next_res_density = (
        tiff.imread(next_res_density_path).astype(np.float32, copy=False)
        if needs_next_res_density
        else None
    )
    sus_density = (
        tiff.imread(sus_density_path).astype(np.float32, copy=False)
        if needs_sensitive_density
        else None
    )
    abs_res_change = (
        tiff.imread(abs_change_path).astype(np.float32, copy=False)
        if needs_absolute_change
        else None
    )
    rel_res_change = (
        tiff.imread(rel_change_path).astype(np.float32, copy=False)
        if needs_relative_change
        else None
    )
    reference_shape = res_density.shape
    for map_name, map_array in (
        ("sensitive density", sus_density),
        ("next resistant density", next_res_density),
        ("absolute change", abs_res_change),
        ("relative growth", rel_res_change),
    ):
        if map_array is not None and reference_shape != map_array.shape:
            raise ValueError(
                f"Resistant density and {map_name} maps must have the same shape: "
                f"{reference_shape} vs {map_array.shape}"
            )
    if total_density is not None and reference_shape != total_density.shape:
        raise ValueError(
            f"Resistant density and total density maps must have the same shape: "
            f"{reference_shape} vs {total_density.shape}"
        )

    x = total_density.ravel() if total_density is not None else None
    color_values = res_density.ravel()
    next_res_values = next_res_density.ravel() if next_res_density is not None else None
    y_abs = abs_res_change.ravel() if abs_res_change is not None else None
    y_rel = rel_res_change.ravel() if rel_res_change is not None else None
    sensitive_x = sus_density.ravel() if sus_density is not None else None
    finite_color_values = color_values[np.isfinite(color_values)]
    color_max = (
        colorbar_vmax
        if colorbar_vmax is not None
        else float(np.nanmax(finite_color_values)) if finite_color_values.size else 1.0
    )
    if color_max <= 0:
        color_max = 1.0
    color_norm = Normalize(vmin=0, vmax=color_max, clip=True)
    color_map = LinearSegmentedColormap.from_list(
        "black_to_goldenrod",
        ["#000000", "#DAA520"],
    )
    saved_paths: list[Path] = []

    if "absolute" in scatter_plot_types:
        if x is None or y_abs is None:
            raise RuntimeError("Absolute scatter requires total density and absolute change maps.")
        if output_path is not None:
            out_path = output_path
        else:
            out_path = scatter_dir / (
                f"absolute_res_change_vs_total_density_"
                f"{well_identifier}_{first_tp}_to_{next_tp}_"
                f"{radius_um_folder_token(radius_um)}_{coarsening_factor}x.png"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6.4, 5.0),
            dpi=200,
            constrained_layout=True,
        )
        n_abs, scatter = plot_change_scatter(
            ax,
            x,
            y_abs,
            marker_color=color_values,
            color_norm=color_norm,
            color_map=color_map,
            first_tp=first_tp,
            ylabel=f"Absolute resistant density change {first_tp}->{next_tp} (cells / um2)",
            title="Absolute resistant density change",
            zero_initial_density_threshold=zero_initial_threshold,
            measurable_growth_threshold=growth_threshold,
            axis_xlim=axis_xlim,
            axis_ylim=axis_ylim,
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label(
            f"Resistant density at day {first_tp} (cells / um2); "
            f"vmax={color_max:.3g}"
        )
        fig.suptitle(
            f"{plate} {well_identifier}: resistant absolute change vs total density\n"
                f"radius={radius_um:g} um, coarsening={coarsening_factor}x, "
                f"red=initial res <= {zero_initial_threshold:g} and change > {growth_threshold:g}, "
                f"n={n_abs:,} pixels"
        )
        save_figure(fig, out_path)
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"Saved absolute-change scatter plot: {out_path}")

    if "relative" in scatter_plot_types:
        if x is None or y_rel is None or next_res_values is None:
            raise RuntimeError("Relative scatter requires total, next-res, and relative change maps.")
        if output_path is not None:
            relative_out_path = companion_output_path(
                output_path,
                "_relative_growth_per_day",
            )
        else:
            relative_out_path = scatter_dir / (
                f"relative_res_growth_per_day_vs_total_density_"
                f"{well_identifier}_{first_tp}_to_{next_tp}_"
                f"{radius_um_folder_token(radius_um)}_{coarsening_factor}x.png"
            )
        relative_out_path.parent.mkdir(parents=True, exist_ok=True)

        valid_relative_growth = (
            np.isfinite(color_values)
            & np.isfinite(next_res_values)
            & (color_values > zero_initial_threshold)
            & (next_res_values > zero_initial_threshold)
        )
        relative_color_norm = Normalize(vmin=0, vmax=color_max, clip=True)

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6.4, 5.0),
            dpi=200,
            constrained_layout=True,
        )
        n_rel, rel_scatter = plot_relative_change_scatter(
            ax,
            x,
            y_rel,
            marker_color=color_values,
            valid_growth=valid_relative_growth,
            color_norm=relative_color_norm,
            color_map=color_map,
            first_tp=first_tp,
            next_tp=next_tp,
            zero_initial_density_threshold=zero_initial_threshold,
            axis_xlim=relative_axis_xlim,
            axis_ylim=relative_axis_ylim,
            xscale=relative_xscale,
            yscale=relative_yscale,
        )
        colorbar = fig.colorbar(rel_scatter, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label(
            f"Resistant density at day {first_tp} (cells / um2); "
            f"vmax={color_max:.3g}"
        )
        fig.suptitle(
            f"{plate} {well_identifier}: relative resistant growth vs total density\n"
            f"radius={radius_um:g} um, coarsening={coarsening_factor}x, "
            f"n={n_rel:,} valid pixels"
        )
        save_figure(fig, relative_out_path)
        plt.close(fig)
        saved_paths.append(relative_out_path)
        print(f"Saved relative growth scatter plot: {relative_out_path}")

    if "zero_initial_growth" in scatter_plot_types:
        if sensitive_x is None or y_abs is None:
            raise RuntimeError("Zero-initial scatter requires sensitive density and absolute change maps.")
        if output_path is not None:
            zero_initial_out_path = companion_output_path(
                output_path,
                "_zero_initial_res_vs_sensitive_density",
            )
        else:
            zero_initial_out_path = scatter_dir / (
                f"absolute_res_change_vs_sensitive_density_zero_initial_growth_"
                f"{well_identifier}_{first_tp}_to_{next_tp}_"
                f"{radius_um_folder_token(radius_um)}_{coarsening_factor}x.png"
            )
        zero_initial_out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6.4, 5.0),
            dpi=200,
            constrained_layout=True,
        )
        n_zero_initial = plot_zero_initial_sensitive_scatter(
            ax,
            sensitive_x,
            y_abs,
            color_values,
            first_tp=first_tp,
            next_tp=next_tp,
            zero_initial_density_threshold=zero_initial_threshold,
            measurable_growth_threshold=growth_threshold,
            axis_xlim=axis_xlim,
            axis_ylim=axis_ylim,
        )
        fig.suptitle(
            f"{plate} {well_identifier}: zero-initial-resistant growth pixels vs sensitive density\n"
            f"radius={radius_um:g} um, coarsening={coarsening_factor}x, "
            f"initial res <= {zero_initial_threshold:g}, change > {growth_threshold:g}, "
            f"n={n_zero_initial:,} pixels"
        )
        save_figure(fig, zero_initial_out_path)
        plt.close(fig)
        saved_paths.append(zero_initial_out_path)
        print(
            "Saved zero-initial-resistant growth sensitive-density scatter plot: "
            f"{zero_initial_out_path}"
        )

    if total_density is not None:
        print(f"Loaded total density: {total_density_path}")
    if sus_density is not None:
        print(f"Loaded sensitive density: {sus_density_path}")
    print(f"Loaded resistant density: {res_density_path}")
    if next_res_density is not None:
        print(f"Loaded next resistant density: {next_res_density_path}")
    if abs_res_change is not None:
        print(f"Loaded resistant absolute change: {abs_change_path}")
    if rel_res_change is not None:
        print(f"Loaded resistant relative growth: {rel_change_path}")
    return saved_paths


def run_test_mode(
    *,
    args: argparse.Namespace,
    config: dict,
    experiment_path: Path,
    plate: str,
    radius_um: float,
    coarsening_factor: int,
    test_parameter_suffix: str,
) -> list[Path]:
    test_timepoints = normalize_timepoints(config)
    if len(test_timepoints) < 2:
        raise ValueError("Need at least two test_timepoints to plot a change map.")

    test_hdf5_file = config.get("test_hdf5_file")
    if not test_hdf5_file:
        raise ValueError("Config must define test_hdf5_file.")

    plate_dir = experiment_path / plate
    test_output_folder = config.get("test_output_folder", DEFAULT_TEST_OUTPUT_FOLDER)
    well_identifier = well_identifier_from_pointcloud_filename(test_hdf5_file)

    if use_structured_output_layout(config):
        density_maps_folder = os.fspath(
            well_output_type_dir(
                config,
                experiment_path,
                plate,
                well_identifier,
                "density_maps_output_type",
                "density_maps",
            )
        )
        density_changes_folder = os.fspath(
            well_output_type_dir(
                config,
                experiment_path,
                plate,
                well_identifier,
                "density_changes_output_type",
                "density_changes",
            )
        )
        scatter_dir = well_output_type_dir(
            config,
            experiment_path,
            plate,
            well_identifier,
            "density_change_scatter_output_type",
            args.scatter_folder,
        )
    else:
        density_maps_folder = resolve_test_output_subfolder(
            config.get(
                "test_density_maps_folder",
                f"density_maps_epanechnikov{test_parameter_suffix}_poisson_test",
            ),
            test_output_folder,
        )
        density_changes_folder = resolve_test_output_subfolder(
            config.get(
                "test_density_changes_folder",
                f"density_changes_epanechnikov{test_parameter_suffix}_poisson_test",
            ),
            test_output_folder,
        )
        scatter_dir = plate_dir / density_changes_folder / args.scatter_folder
    density_changes_dir = plate_dir / density_changes_folder
    density_changes_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    change_channels = ensure_res_change_channel(
        tuple(config.get("test_change_channels", ["res"]))
    )
    zero_initial_threshold = zero_initial_density_threshold(config)
    growth_threshold = measurable_growth_threshold(config)
    create_density_changes = pipeline_output_enabled(
        config,
        "density_changes",
        "create_density_changes",
        bool(config.get("test_write_changes", True)),
    )
    create_change_scatter_plots = pipeline_output_enabled(
        config,
        "change_scatter_plots",
        "create_change_scatter_plots",
        True,
    )
    scatter_plot_types = selected_scatter_plot_types(config)
    scatter_xlim = axis_limits(config, "density_change_scatter_xlim")
    scatter_ylim = axis_limits(config, "density_change_scatter_ylim")
    relative_scatter_xlim = axis_limits(
        config,
        "relative_density_change_scatter_xlim",
    )
    if relative_scatter_xlim is None:
        relative_scatter_xlim = scatter_xlim
    relative_scatter_ylim = axis_limits(
        config,
        "relative_density_change_scatter_ylim",
    )
    relative_xscale = axis_scale(config, "relative_density_change_scatter_xscale")
    relative_yscale = axis_scale(config, "relative_density_change_scatter_yscale")
    scatter_colorbar_vmax = positive_float_or_none(
        config,
        "density_change_scatter_colorbar_vmax",
    )
    if create_density_changes:
        write_test_density_changes(
            directory_path=os.fspath(plate_dir),
            well_identifier=well_identifier,
            timepoints=test_timepoints,
            density_maps_folder=density_maps_folder,
            density_changes_folder=density_changes_folder,
            channels=change_channels,
            write_zero_initial_growth_channel=config_bool(
                config,
                "write_zero_initial_growth_channel",
                True,
            ),
            zero_initial_density_threshold=zero_initial_threshold,
            measurable_growth_threshold=growth_threshold,
        )

    if not create_change_scatter_plots or not scatter_plot_types:
        print("Skipping change-vs-density scatter plots by config.")
        return []

    if args.output and len(test_timepoints) > 2:
        raise ValueError("--output can only be used when plotting a single timepoint pair.")

    saved_paths = []
    for pair_index, (first_tp, next_tp) in enumerate(
        zip(test_timepoints[:-1], test_timepoints[1:]),
        start=1,
    ):
        output_path = Path(args.output) if args.output else None
        print(
            f"Plotting pair {pair_index}/{len(test_timepoints) - 1}: "
            f"{first_tp}->{next_tp}"
        )
        saved_paths.extend(
            plot_density_change_pair(
                plate=plate,
                plate_dir=plate_dir,
                density_maps_folder=density_maps_folder,
                density_changes_dir=density_changes_dir,
                scatter_dir=scatter_dir,
                well_identifier=well_identifier,
                first_tp=first_tp,
                next_tp=next_tp,
                radius_um=radius_um,
                coarsening_factor=coarsening_factor,
                zero_initial_threshold=zero_initial_threshold,
                growth_threshold=growth_threshold,
                axis_xlim=scatter_xlim,
                axis_ylim=scatter_ylim,
                relative_axis_xlim=relative_scatter_xlim,
                relative_axis_ylim=relative_scatter_ylim,
                relative_xscale=relative_xscale,
                relative_yscale=relative_yscale,
                colorbar_vmax=scatter_colorbar_vmax,
                scatter_plot_types=scatter_plot_types,
                output_path=output_path,
            )
        )

    return saved_paths


def run_batch_mode(
    *,
    args: argparse.Namespace,
    config: dict,
    experiment_path: Path,
    plates: list[str],
    radius_um: float,
    coarsening_factor: int,
) -> list[Path]:
    if args.output:
        raise ValueError("--output can only be used in single-well test mode.")

    coarsening_suffix = "" if coarsening_factor == 1 else f"_{coarsening_factor}x"
    legacy_density_maps_folder = config.get(
        "density_maps_folder",
        f"density_maps_epanechnikov{coarsening_suffix}_poisson",
    )
    legacy_density_changes_folder = config.get(
        "density_changes_folder",
        f"density_changes_epanechnikov_{radius_um_folder_token(radius_um)}_"
        f"{coarsening_factor}x_poisson_batch",
    )
    change_channels = ensure_res_change_channel(
        tuple(config.get("density_change_channels", config.get("test_change_channels", ["res"])))
    )
    zero_initial_threshold = zero_initial_density_threshold(config)
    growth_threshold = measurable_growth_threshold(config)
    create_density_changes = pipeline_output_enabled(
        config,
        "density_changes",
        "create_density_changes",
        True,
    )
    create_change_scatter_plots = pipeline_output_enabled(
        config,
        "change_scatter_plots",
        "create_change_scatter_plots",
        True,
    )
    scatter_plot_types = selected_scatter_plot_types(config)
    should_plot_scatters = create_change_scatter_plots and bool(scatter_plot_types)
    scatter_xlim = axis_limits(config, "density_change_scatter_xlim")
    scatter_ylim = axis_limits(config, "density_change_scatter_ylim")
    relative_scatter_xlim = axis_limits(
        config,
        "relative_density_change_scatter_xlim",
    )
    if relative_scatter_xlim is None:
        relative_scatter_xlim = scatter_xlim
    relative_scatter_ylim = axis_limits(
        config,
        "relative_density_change_scatter_ylim",
    )
    relative_xscale = axis_scale(config, "relative_density_change_scatter_xscale")
    relative_yscale = axis_scale(config, "relative_density_change_scatter_yscale")
    scatter_colorbar_vmax = positive_float_or_none(
        config,
        "density_change_scatter_colorbar_vmax",
    )
    write_zero_initial_growth = config_bool(
        config,
        "write_zero_initial_growth_channel",
        True,
    )
    structured_output = use_structured_output_layout(config)

    saved_paths: list[Path] = []
    if not create_density_changes and not should_plot_scatters:
        print(
            "Skipping density changes and change-vs-density scatter plots by config."
        )
        return saved_paths

    for plate in plates:
        plate_dir = experiment_path / plate

        wells = selected_batch_wells(config, experiment_path, plate)
        if not wells:
            raise ValueError(
                "No batch wells found. Configure change_vs_density_wells, "
                "hdf5_include_patterns, or scatter_movie_wells."
            )

        for well_identifier in wells:
            if structured_output:
                density_maps_folder = os.fspath(
                    well_output_type_dir(
                        config,
                        experiment_path,
                        plate,
                        well_identifier,
                        "density_maps_output_type",
                        "density_maps",
                    )
                )
                density_changes_folder = os.fspath(
                    well_output_type_dir(
                        config,
                        experiment_path,
                        plate,
                        well_identifier,
                        "density_changes_output_type",
                        "density_changes",
                    )
                )
                scatter_dir = well_output_type_dir(
                    config,
                    experiment_path,
                    plate,
                    well_identifier,
                    "density_change_scatter_output_type",
                    args.scatter_folder,
                )
            else:
                density_maps_folder = legacy_density_maps_folder
                density_changes_folder = legacy_density_changes_folder
                scatter_dir = plate_dir / density_changes_folder / args.scatter_folder

            density_maps_dir = plate_dir / density_maps_folder
            density_changes_dir = plate_dir / density_changes_folder
            density_changes_dir.mkdir(parents=True, exist_ok=True)
            if should_plot_scatters:
                scatter_dir.mkdir(parents=True, exist_ok=True)

            timepoints = discover_density_timepoints(density_maps_dir, well_identifier)
            if len(timepoints) < 2:
                print(
                    f"Skipping {plate} {well_identifier}; found fewer than two "
                    f"complete density-map timepoints in {density_maps_dir}."
                )
                continue

            actions = []
            if create_density_changes:
                actions.append("density changes")
            if should_plot_scatters:
                actions.append("scatterplots")
            print(
                f"Writing {' and '.join(actions)} for {plate} "
                f"{well_identifier}: {len(timepoints)} timepoints"
            )
            if create_density_changes:
                density_changes_dir.mkdir(parents=True, exist_ok=True)
                write_test_density_changes(
                    directory_path=os.fspath(plate_dir),
                    well_identifier=well_identifier,
                    timepoints=[normalize_timepoint(timepoint) for timepoint in timepoints],
                    density_maps_folder=density_maps_folder,
                    density_changes_folder=density_changes_folder,
                    channels=change_channels,
                    write_zero_initial_growth_channel=write_zero_initial_growth,
                    zero_initial_density_threshold=zero_initial_threshold,
                    measurable_growth_threshold=growth_threshold,
                )

            if not should_plot_scatters:
                continue

            for pair_index, (first_tp, next_tp) in enumerate(
                zip(timepoints[:-1], timepoints[1:]),
                start=1,
            ):
                print(
                    f"Plotting {plate} {well_identifier} pair "
                    f"{pair_index}/{len(timepoints) - 1}: {first_tp}->{next_tp}"
                )
                saved_paths.extend(
                    plot_density_change_pair(
                        plate=plate,
                        plate_dir=plate_dir,
                        density_maps_folder=density_maps_folder,
                        density_changes_dir=density_changes_dir,
                        scatter_dir=scatter_dir,
                        well_identifier=well_identifier,
                        first_tp=first_tp,
                        next_tp=next_tp,
                        radius_um=radius_um,
                        coarsening_factor=coarsening_factor,
                        zero_initial_threshold=zero_initial_threshold,
                        growth_threshold=growth_threshold,
                        axis_xlim=scatter_xlim,
                        axis_ylim=scatter_ylim,
                        relative_axis_xlim=relative_scatter_xlim,
                        relative_axis_ylim=relative_scatter_ylim,
                        relative_xscale=relative_xscale,
                        relative_yscale=relative_yscale,
                        colorbar_vmax=scatter_colorbar_vmax,
                        scatter_plot_types=scatter_plot_types,
                    )
                )

    return saved_paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    radius_um = resolve_radius_um(config)
    coarsening_factor = resolve_coarsening_factor(config, radius_um)
    test_parameter_suffix = f"_{radius_um_folder_token(radius_um)}_{coarsening_factor}x"

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

    if config.get("test_mode", False):
        if len(plates) != 1:
            raise ValueError("test_mode expects exactly one configured plate.")
        saved_paths = run_test_mode(
            args=args,
            config=config,
            experiment_path=experiment_path,
            plate=plates[0],
            radius_um=radius_um,
            coarsening_factor=coarsening_factor,
            test_parameter_suffix=test_parameter_suffix,
        )
    else:
        saved_paths = run_batch_mode(
            args=args,
            config=config,
            experiment_path=experiment_path,
            plates=plates,
            radius_um=radius_um,
            coarsening_factor=coarsening_factor,
        )

    print(f"Saved {len(saved_paths)} scatter PNGs.")


if __name__ == "__main__":
    main()
