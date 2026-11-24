#!/usr/bin/env python3
"""
Retired exploratory density-change relationship plots.

Plot sampled density-change datapoints from Epanechnikov pipeline outputs.

The plot mirrors the visual grammar of figures/Figure_3/Density_Correlation.py:
strategy colors, paper-style axes, faint raw datapoints, and filled KDE
contours per strategy.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")).mkdir(
    parents=True,
    exist_ok=True,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import tifffile as tiff

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from shared import figtools as ft
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name, normalize_strategy

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (
    DEFAULT_CONFIG_PATH,
    load_config,
    output_type_name,
    resolve_experiment_path,
    resolve_output_root,
    resolve_output_run_dir,
    strategy_name_from_well_identifier,
)


REL_CHANGE_RE = re.compile(
    r"^(?P<well>.+)_(?P<first_tp>[^_]+)_to_(?P<next_tp>[^_]+)_"
    r"res_density_epanechnikov_rel_diff\.tif$"
)
ABS_CHANGE_RE = re.compile(
    r"^(?P<well>.+)_(?P<first_tp>[^_]+)_to_(?P<next_tp>[^_]+)_"
    r"res_density_epanechnikov_abs_diff\.tif$"
)

X_DISPLAY_SCALE = 1000.0
X_LABEL = "Total density [cells (1000 µm$^2$)$^{-1}$]"
DEFAULT_XLIM = (0.0, 3.0)
DEFAULT_REL_YLIM = (0.0, 8.0)
DEFAULT_ABS_PER_DAY_YLIM = (-0.00025, 0.0008)
DEFAULT_INCLUDE_STRATEGIES = ("AT50", "AT100", "CT", "NT")
DEFAULT_RES_DENSITY_WINDOW = ((2.0e-5, 4.375e-4), (2.0e-5, 0.001), (2.0e-5, 0.002))
# To plot several gates in one PyCharm run, use e.g.:
# DEFAULT_RES_DENSITY_WINDOW = ((2.0e-5, 4.375e-4), (2.0e-5, 0.001))
#DEFAULT_RES_DENSITY_WINDOW = (2.0e-5, 4.375e-4)
INCLUDE_UNGATED_PLOTS = True

# Set this to the processed density output run you want to plot when running from PyCharm.
# It should contain Plate_* folders, and simulation runs also contain simulation_density_run_manifest.tsv.
#INPUT_RUN_DIR = r"D:\2025_08_CoreSeed_Large_Testdataset\output\run_20260630_145305_320_32"
INPUT_RUN_DIR = r"F:\2025_08_CoreSeed_Simulation_Density_Output\run_sim_20260709_145640_320_32"
DEFAULT_RUN_DIR: str | None = INPUT_RUN_DIR
DEFAULT_OUTPUT_DIR: str | None = None
DEFAULT_ESCAPE_SUMMARY_REL_PATH = (
    "Figures/Distance_to_front_analysis/frontcontact_summary_ALL.csv"
)
DEFAULT_SIMULATION_ESCAPE_CSV = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja\combined_results.csv"
)
GRID_N = 140
CONTOUR_QUANTILES = (0.80, 0.90, 0.95)
POINT_ALPHA = 0.12
POINT_SIZE = 1.0
MAX_CONTOUR_POINTS = 6000
FILL_ALPHA_MIN = 0.10
FILL_ALPHA_MAX = 0.65
OUTLINE_LW = 0.8
OUTLINE_ALPHA = 0.9
MARGINAL_HIST_BINS = 24
DPI = 300

CHANGE_SPECS: dict[str, dict[str, Any]] = {
    "rel": {
        "display_name": "relative resistant growth",
        "glob": "*_res_density_epanechnikov_rel_diff.tif",
        "regex": REL_CHANGE_RE,
        "ylabel": "Relative resistant growth",
        "ylim_key": "relative_density_change_scatter_ylim",
        "yscale_key": "relative_density_change_scatter_yscale",
        "default_ylim": DEFAULT_REL_YLIM,
        "combined_stem": "rel_contour",
        "panel_stem": "rel_panel_contour",
        "before_stem": "rel_before_contour",
        "after_stem": "rel_after_contour",
    },
    "abs_per_day": {
        "display_name": "absolute resistant density change per day",
        "glob": "*_res_density_epanechnikov_abs_diff.tif",
        "regex": ABS_CHANGE_RE,
        "ylabel": "Resistant density change per day [cells µm$^{-2}$ day$^{-1}$]",
        "ylim_key": "absolute_density_change_per_day_scatter_ylim",
        "yscale_key": "absolute_density_change_per_day_scatter_yscale",
        "default_ylim": DEFAULT_ABS_PER_DAY_YLIM,
        "combined_stem": "abs_contour",
        "panel_stem": "abs_panel_contour",
        "before_stem": "abs_before_contour",
        "after_stem": "abs_after_contour",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample pixel-level total-density/relative-growth datapoints from "
            "pipeline output and plot them with Density_Correlation-style "
            "strategy colors and contours."
        )
    )
    parser.add_argument(
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help=f"YAML config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help=(
            "Pipeline run directory to read. Defaults to the configured run dir "
            "if DEFAULT_RUN_DIR is None and it exists, otherwise the newest run_* "
            "folder under output_root."
        ),
    )
    parser.add_argument(
        "--max-points-per-map",
        type=int,
        default=200,
        help="Maximum valid pixels sampled from each well/time interval. 0 keeps all. Default: 200.",
    )
    parser.add_argument(
        "--max-points-per-strategy",
        type=int,
        default=30000,
        help="Maximum plotted points retained per strategy. 0 keeps all. Default: 30000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible pixel sampling. Default: 7.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Defaults to <run-dir>/summary_plots.",
    )
    parser.add_argument(
        "--include-strategies",
        nargs="+",
        default=list(DEFAULT_INCLUDE_STRATEGIES),
        help=(
            "Canonical strategies to include in the combined plot. "
            f"Default: {' '.join(DEFAULT_INCLUDE_STRATEGIES)}."
        ),
    )
    parser.add_argument(
        "--change-types",
        nargs="+",
        choices=tuple(CHANGE_SPECS) + ("abs",),
        default=list(CHANGE_SPECS),
        help=(
            "Density-change outputs to plot. "
            f"Default: {' '.join(CHANGE_SPECS)}. Use 'abs' as an alias for 'abs_per_day'."
        ),
    )
    parser.add_argument(
        "--res-density-window",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=None,
        help=(
            "Override with a single first-frame resistant-density gate in cells µm$^{-2}$. "
            "By default, the script uses DEFAULT_RES_DENSITY_WINDOW."
        ),
    )
    parser.add_argument(
        "--no-res-density-gate",
        action="store_true",
        help="Use all finite pixels; ignore the first-frame resistant-density gate.",
    )
    parser.add_argument(
        "--skip-escape-panels",
        action="store_true",
        help="Skip before/after-escape panel exports.",
    )
    parser.add_argument(
        "--simulation-escape-csv",
        default=None,
        help=(
            "CSV containing protocol, run, and TTE for simulation runs. "
            "Defaults to simulation_escape_csv in the config, then "
            f"{DEFAULT_SIMULATION_ESCAPE_CSV}."
        ),
    )
    return parser.parse_args()


def latest_run_dir(config: dict, experiment_path: str) -> Path:
    root = resolve_output_root(config, experiment_path)
    if not root.exists():
        raise FileNotFoundError(f"Configured output_root does not exist: {root}")

    runs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {root}")
    return runs[0]


def resolve_input_run_dir(args: argparse.Namespace, config: dict, experiment_path: str) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.exists():
            raise FileNotFoundError(f"Requested --run-dir does not exist: {run_dir}")
        return run_dir

    configured_run_dir = resolve_output_run_dir(config, experiment_path)
    if configured_run_dir.exists():
        return configured_run_dir
    return latest_run_dir(config, experiment_path)


def parse_float_limits(config: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    return (float(value[0]) * X_DISPLAY_SCALE if key.endswith("_xlim") else float(value[0]),
            float(value[1]) * X_DISPLAY_SCALE if key.endswith("_xlim") else float(value[1]))


def parse_axis_scale(config: dict, key: str, default: str = "linear") -> str:
    value = str(config.get(key, default)).strip().lower()
    if value not in {"linear", "log"}:
        raise ValueError(f"{key} must be 'linear' or 'log', got {value!r}.")
    return value


def timepoint_to_day(timepoint: str) -> float:
    try:
        return float(timepoint)
    except (TypeError, ValueError):
        return float("nan")


def change_values_for_plot(
    change_values: np.ndarray,
    *,
    first_tp: str,
    next_tp: str,
    change_type: str,
) -> np.ndarray:
    if change_type != "abs_per_day":
        return change_values

    first_day = timepoint_to_day(first_tp)
    next_day = timepoint_to_day(next_tp)
    delta_days = next_day - first_day
    if not np.isfinite(delta_days) or delta_days <= 0:
        raise ValueError(
            f"Cannot normalize absolute change by interval length for {first_tp}->{next_tp}"
        )
    return (change_values / np.float32(delta_days)).astype(np.float32, copy=False)


def density_map_path(density_maps_dir: Path, well_identifier: str, first_tp: str) -> Path:
    return density_maps_dir / f"{well_identifier}_{first_tp}_total_density_epanechnikov.tif"


def resistant_density_map_path(
    density_maps_dir: Path,
    well_identifier: str,
    first_tp: str,
) -> Path:
    return density_maps_dir / f"{well_identifier}_{first_tp}_res_density_epanechnikov.tif"


def density_window_token(res_density_window: tuple[float, float] | None) -> str:
    if res_density_window is None:
        return ""

    def fmt(value: float) -> str:
        return f"{value:.6g}".replace(".", "p")

    lower, upper = res_density_window
    return f"_gate_{fmt(lower)}_{fmt(upper)}"


def normalize_res_density_windows(
    value: object,
) -> list[tuple[float, float] | None]:
    if value is None:
        return [None]
    if not isinstance(value, (list, tuple)):
        raise ValueError("Resistant-density window must be None, one pair, or a list of pairs.")
    if len(value) == 0:
        return [None]
    if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        windows: list[tuple[float, float] | None] = [
            (float(value[0]), float(value[1]))
        ]
    else:
        windows = []
        for item in value:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not all(isinstance(v, (int, float)) for v in item)
            ):
                raise ValueError(
                    "Each resistant-density window must be a pair like "
                    "(2.0e-5, 4.375e-4)."
                )
            windows.append((float(item[0]), float(item[1])))

    for window in windows:
        if window is not None and window[0] > window[1]:
            raise ValueError("Resistant-density window lower bound must be <= upper bound")
    return windows


def strategy_from_plate_and_well(plate: str, well_identifier: str) -> str:
    strategy = strategy_name_from_well_identifier(well_identifier) or ""
    if not strategy and str(plate).startswith("Plate_"):
        strategy = str(plate).split("Plate_", 1)[1]
    return normalize_strategy(strategy or "unknown")


def is_simulation_density_run(run_dir: Path) -> bool:
    return (run_dir / "simulation_density_run_manifest.tsv").exists()


def sample_pair_points(
    *,
    total_density_path: Path,
    change_path: Path,
    first_tp: str,
    next_tp: str,
    change_type: str,
    res_density_path: Path | None,
    res_density_window: tuple[float, float] | None,
    rng: np.random.Generator,
    max_points: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
) -> pd.DataFrame:
    total_density = tiff.imread(total_density_path).astype(np.float32, copy=False).ravel()
    change_values = tiff.imread(change_path).astype(np.float32, copy=False).ravel()
    if total_density.shape != change_values.shape:
        raise ValueError(
            f"Shape mismatch for {total_density_path.name} and {change_path.name}: "
            f"{total_density.shape} vs {change_values.shape}"
        )
    res_density = None
    if res_density_window is not None:
        if res_density_path is None:
            raise ValueError("res_density_path is required when res_density_window is set.")
        res_density = tiff.imread(res_density_path).astype(np.float32, copy=False).ravel()
        if total_density.shape != res_density.shape:
            raise ValueError(
                f"Shape mismatch for {total_density_path.name} and {res_density_path.name}: "
                f"{total_density.shape} vs {res_density.shape}"
            )

    x = total_density * X_DISPLAY_SCALE
    y = change_values_for_plot(
        change_values,
        first_tp=first_tp,
        next_tp=next_tp,
        change_type=change_type,
    )
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xlim[0])
        & (x <= xlim[1])
        & (y >= ylim[0])
        & (y <= ylim[1])
    )
    if res_density is not None and res_density_window is not None:
        lower_res, upper_res = res_density_window
        valid &= (
            np.isfinite(res_density)
            & (res_density >= lower_res)
            & (res_density <= upper_res)
        )
    if xscale == "log":
        valid &= x > 0
    if yscale == "log":
        valid &= y > 0
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return pd.DataFrame({"x": [], "y": []})
    if max_points > 0 and idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return pd.DataFrame(
        {
            "x": x[idx].astype(np.float32, copy=False),
            "y": y[idx].astype(np.float32, copy=False),
        }
    )


def sample_pair_arrays(
    *,
    total_density_path: Path,
    change_path: Path,
    first_tp: str,
    next_tp: str,
    change_type: str,
    res_density_path: Path | None,
    res_density_window: tuple[float, float] | None,
    rng: np.random.Generator,
    max_points: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
) -> tuple[np.ndarray, np.ndarray]:
    total_density = tiff.imread(total_density_path).astype(np.float32, copy=False).ravel()
    change_values = tiff.imread(change_path).astype(np.float32, copy=False).ravel()
    if total_density.shape != change_values.shape:
        raise ValueError(
            f"Shape mismatch for {total_density_path.name} and {change_path.name}: "
            f"{total_density.shape} vs {change_values.shape}"
        )

    x = total_density * X_DISPLAY_SCALE
    y = change_values_for_plot(
        change_values,
        first_tp=first_tp,
        next_tp=next_tp,
        change_type=change_type,
    )
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xlim[0])
        & (x <= xlim[1])
        & (y >= ylim[0])
        & (y <= ylim[1])
    )
    if res_density_window is not None:
        if res_density_path is None:
            raise ValueError("res_density_path is required when res_density_window is set.")
        res_density = tiff.imread(res_density_path).astype(np.float32, copy=False).ravel()
        if total_density.shape != res_density.shape:
            raise ValueError(
                f"Shape mismatch for {total_density_path.name} and {res_density_path.name}: "
                f"{total_density.shape} vs {res_density.shape}"
            )
        lower_res, upper_res = res_density_window
        valid &= (
            np.isfinite(res_density)
            & (res_density >= lower_res)
            & (res_density <= upper_res)
        )
    if xscale == "log":
        valid &= x > 0
    if yscale == "log":
        valid &= y > 0
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    if max_points > 0 and idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return x[idx].astype(np.float32, copy=False), y[idx].astype(np.float32, copy=False)


def concatenate_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.empty(0, dtype=np.float32)
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks)


def collect_strategy_arrays(
    *,
    run_dir: Path,
    config: dict,
    change_type: str,
    rng: np.random.Generator,
    max_points_per_map: int,
    max_points_per_strategy: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    res_density_window: tuple[float, float] | None,
) -> dict[str, dict[str, Any]]:
    density_maps_name = output_type_name(config, "density_maps_output_type", "density_maps")
    density_changes_name = output_type_name(config, "density_changes_output_type", "density_changes")
    spec = CHANGE_SPECS[change_type]
    change_regex = spec["regex"]
    change_glob = spec["glob"]

    data: dict[str, dict[str, Any]] = {}
    for changes_dir in run_dir.glob(f"Plate_*/*/{density_changes_name}"):
        well_identifier = changes_dir.parent.name
        plate = changes_dir.parent.parent.name
        strategy = strategy_from_plate_and_well(plate, well_identifier)
        maps_dir = changes_dir.parent / density_maps_name

        for change_path in sorted(changes_dir.glob(change_glob)):
            match = change_regex.match(change_path.name)
            if match is None:
                continue
            first_tp = match.group("first_tp")
            next_tp = match.group("next_tp")
            total_path = density_map_path(maps_dir, well_identifier, first_tp)
            if not total_path.exists():
                print(f"[WARN] Missing total-density map for {change_path}: {total_path}")
                continue
            res_path = None
            if res_density_window is not None:
                res_path = resistant_density_map_path(maps_dir, well_identifier, first_tp)
                if not res_path.exists():
                    print(f"[WARN] Missing resistant-density map for {change_path}: {res_path}")
                    continue
            x, y = sample_pair_arrays(
                total_density_path=total_path,
                change_path=change_path,
                first_tp=first_tp,
                next_tp=next_tp,
                change_type=change_type,
                res_density_path=res_path,
                res_density_window=res_density_window,
                rng=rng,
                max_points=max_points_per_map,
                xlim=xlim,
                ylim=ylim,
                xscale=xscale,
                yscale=yscale,
            )
            if x.size == 0:
                continue
            entry = data.setdefault(strategy, {"x_blocks": [], "y_blocks": [], "count": 0})
            entry["x_blocks"].append(x)
            entry["y_blocks"].append(y)
            entry["count"] += int(x.size)

    if not data:
        raise RuntimeError(
            f"No plottable {spec['display_name']} datapoints found under {run_dir}"
        )

    if max_points_per_strategy > 0:
        for entry in data.values():
            x = concatenate_blocks(entry["x_blocks"])
            y = concatenate_blocks(entry["y_blocks"])
            if x.size > max_points_per_strategy:
                idx = rng.choice(np.arange(x.size), size=max_points_per_strategy, replace=False)
                x = x[idx]
                y = y[idx]
            entry["x_blocks"] = [x]
            entry["y_blocks"] = [y]
            entry["count"] = int(x.size)
    return data


def collect_datapoints(
    *,
    run_dir: Path,
    config: dict,
    change_type: str,
    rng: np.random.Generator,
    max_points_per_map: int,
    max_points_per_strategy: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    res_density_window: tuple[float, float] | None,
    include_interval_metadata: bool,
) -> pd.DataFrame:
    density_maps_name = output_type_name(config, "density_maps_output_type", "density_maps")
    density_changes_name = output_type_name(config, "density_changes_output_type", "density_changes")
    spec = CHANGE_SPECS[change_type]
    change_regex = spec["regex"]
    change_glob = spec["glob"]

    frames: list[pd.DataFrame] = []
    for changes_dir in run_dir.glob(f"Plate_*/*/{density_changes_name}"):
        well_identifier = changes_dir.parent.name
        plate = changes_dir.parent.parent.name
        strategy = strategy_from_plate_and_well(plate, well_identifier)
        maps_dir = changes_dir.parent / density_maps_name

        for change_path in sorted(changes_dir.glob(change_glob)):
            match = change_regex.match(change_path.name)
            if match is None:
                continue
            first_tp = match.group("first_tp")
            next_tp = match.group("next_tp")
            total_path = density_map_path(maps_dir, well_identifier, first_tp)
            if not total_path.exists():
                print(f"[WARN] Missing total-density map for {change_path}: {total_path}")
                continue
            res_path = None
            if res_density_window is not None:
                res_path = resistant_density_map_path(maps_dir, well_identifier, first_tp)
                if not res_path.exists():
                    print(f"[WARN] Missing resistant-density map for {change_path}: {res_path}")
                    continue
            points = sample_pair_points(
                total_density_path=total_path,
                change_path=change_path,
                first_tp=first_tp,
                next_tp=next_tp,
                change_type=change_type,
                res_density_path=res_path,
                res_density_window=res_density_window,
                rng=rng,
                max_points=max_points_per_map,
                xlim=xlim,
                ylim=ylim,
                xscale=xscale,
                yscale=yscale,
            )
            if points.empty:
                continue
            points["strategy"] = strategy
            if include_interval_metadata:
                points["plate"] = plate
                points["well"] = well_identifier
                points["first_tp"] = first_tp
                points["next_tp"] = next_tp
                points["first_day"] = timepoint_to_day(first_tp)
                points["next_day"] = timepoint_to_day(next_tp)
            frames.append(points)

    if not frames:
        raise RuntimeError(
            f"No plottable {spec['display_name']} datapoints found under {run_dir}"
        )

    data = pd.concat(frames, ignore_index=True)
    data["strategy"] = data["strategy"].astype("category")
    sampled_frames = []
    for strategy, sub in data.groupby("strategy", sort=False):
        if max_points_per_strategy > 0 and len(sub) > max_points_per_strategy:
            sampled_frames.append(
                sub.sample(n=max_points_per_strategy, random_state=int(rng.integers(0, 2**31 - 1)))
            )
        else:
            sampled_frames.append(sub)
    return pd.concat(sampled_frames, ignore_index=True)


def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if x.size < 10:
        return None
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception as exc:
        print(f"[WARN] Skipping KDE contour: {exc}")
        return None

    xi = np.linspace(xlim[0], xlim[1], GRID_N)
    yi = np.linspace(ylim[0], ylim[1], GRID_N)
    xg, yg = np.meshgrid(xi, yi)
    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    return xg, yg, z


def filled_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    zorder: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> bool:
    if x.size > MAX_CONTOUR_POINTS:
        rng = np.random.default_rng(12345)
        idx = rng.choice(np.arange(x.size), size=MAX_CONTOUR_POINTS, replace=False)
        x = x[idx]
        y = y[idx]
    grid = compute_density_grid(x, y, xlim, ylim)
    if grid is None:
        return False
    xg, yg, z = grid
    vals = z[np.isfinite(z) & (z > 0)]
    if vals.size == 0:
        return False

    core_levels = np.quantile(vals, CONTOUR_QUANTILES)
    levels = np.concatenate((core_levels, [vals.max() * 1.001]))
    base_rgba = np.array(mcolors.to_rgba(color))
    fill_colors = []
    for i in range(len(levels) - 1):
        alpha = FILL_ALPHA_MIN + (FILL_ALPHA_MAX - FILL_ALPHA_MIN) * (
            i / max(1, len(levels) - 2)
        )
        rgba = base_rgba.copy()
        rgba[-1] = alpha
        fill_colors.append(tuple(rgba))

    ax.contourf(xg, yg, z, levels=levels, colors=fill_colors, zorder=zorder)
    ax.contour(
        xg,
        yg,
        z,
        levels=core_levels,
        colors=[color],
        linewidths=OUTLINE_LW,
        alpha=OUTLINE_ALPHA,
        zorder=zorder + 1,
    )
    return True


def sorted_strategy_keys(strategies: set[str]) -> list[str]:
    ordered = [strategy for strategy in STRATEGY_ORDER if strategy in strategies]
    ordered.extend(sorted(strategy for strategy in strategies if strategy not in ordered))
    return ordered


def strategy_arrays(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return concatenate_blocks(entry["x_blocks"]), concatenate_blocks(entry["y_blocks"])


def histogram_bin_edges(
    limits: tuple[float, float],
    scale: str,
    n_bins: int = MARGINAL_HIST_BINS,
) -> np.ndarray:
    if scale == "log":
        if limits[0] <= 0:
            raise ValueError("Log-scaled histogram limits must be positive.")
        return np.geomspace(limits[0], limits[1], n_bins + 1)
    return np.linspace(limits[0], limits[1], n_bins + 1)


def plot_strategy_panel_grid(
    strategy_data: list[tuple[str, np.ndarray, np.ndarray]],
    out_base: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    y_label: str,
) -> None:
    if not strategy_data:
        return

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=6,
        overrides={
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        },
    )
    ncols = min(4, len(strategy_data))
    nrows = int(math.ceil(len(strategy_data) / ncols))
    fig = plt.figure(
        figsize=(1.55 * ncols, 1.55 * nrows),
        dpi=DPI,
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(nrows, ncols)
    x_bins = histogram_bin_edges(xlim, xscale)
    y_bins = histogram_bin_edges(ylim, yscale)
    shared_ax: plt.Axes | None = None

    for idx, (strategy, x, y) in enumerate(strategy_data):
        row = idx // ncols
        col = idx % ncols
        panel_grid = outer_grid[row, col].subgridspec(
            2,
            2,
            width_ratios=(5.0, 1.15),
            height_ratios=(1.15, 5.0),
            wspace=0.05,
            hspace=0.05,
        )
        if shared_ax is None:
            ax = fig.add_subplot(panel_grid[1, 0])
            shared_ax = ax
        else:
            ax = fig.add_subplot(panel_grid[1, 0], sharex=shared_ax, sharey=shared_ax)
        ax_hist_x = fig.add_subplot(panel_grid[0, 0], sharex=ax)
        ax_hist_y = fig.add_subplot(panel_grid[1, 1], sharey=ax)

        color = get_color(strategy, "#000000")
        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            c=color,
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        filled_contours(
            ax,
            x,
            y,
            color,
            zorder=2,
            xlim=xlim,
            ylim=ylim,
        )
        ax_hist_x.hist(
            x,
            bins=x_bins,
            color=color,
            alpha=0.55,
            edgecolor=color,
            linewidth=0.35,
        )
        ax_hist_y.hist(
            y,
            bins=y_bins,
            orientation="horizontal",
            color=color,
            alpha=0.55,
            edgecolor=color,
            linewidth=0.35,
        )

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax_hist_x.set_title(get_display_name(strategy), fontsize=7, pad=2)

        if row != nrows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if col != 0:
            ax.tick_params(axis="y", which="both", labelleft=False)
        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", pad=1)

        ax_hist_x.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        ax_hist_y.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        for marginal_ax in (ax_hist_x, ax_hist_y):
            for spine in marginal_ax.spines.values():
                spine.set_visible(False)

    fig.supxlabel(X_LABEL, fontsize=7)
    fig.supylabel(y_label, fontsize=7)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_datapoints_arrays(
    data: dict[str, dict[str, Any]],
    out_base: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    y_label: str,
    include_strategies: list[str],
) -> None:
    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=6,
        overrides={
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        },
    )
    fig, ax = ft.new_figure(
        width=58.0,
        height_mm=46.0,
        margins=(12.0, 5.0, 12.0, 4.0),
        dpi=DPI,
        apply_style=False,
    )

    include = {normalize_strategy(strategy) for strategy in include_strategies}
    strategies = [strategy for strategy in sorted_strategy_keys(set(data)) if strategy in include]
    legend_handles = []
    for i, strategy in enumerate(strategies):
        x, y = strategy_arrays(data[strategy])
        color = get_color(strategy, "#000000")
        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            c=color,
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=1 + 5 * i,
        )
        if filled_contours(
            ax,
            x,
            y,
            color,
            zorder=2 + 5 * i,
            xlim=xlim,
            ylim=ylim,
        ):
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=2.2, label=get_display_name(strategy))
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            handletextpad=0.4,
            columnspacing=0.8,
            borderaxespad=0.2,
            fontsize=5,
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_strategy_panels_arrays(
    data: dict[str, dict[str, Any]],
    out_base: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    y_label: str,
) -> None:
    strategies = sorted_strategy_keys(set(data))
    strategy_data = [
        (strategy, *strategy_arrays(data[strategy]))
        for strategy in strategies
    ]
    plot_strategy_panel_grid(
        strategy_data,
        out_base,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        y_label=y_label,
    )


def plot_datapoints(
    data: pd.DataFrame,
    out_base: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    y_label: str,
) -> None:
    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=6,
        overrides={
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        },
    )
    fig, ax = ft.new_figure(
        width=58.0,
        height_mm=46.0,
        margins=(12.0, 5.0, 12.0, 4.0),
        dpi=DPI,
        apply_style=False,
    )

    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if strategy in set(data["strategy"])
    ]
    strategies.extend(
        sorted(strategy for strategy in set(data["strategy"]) if strategy not in strategies)
    )

    legend_handles = []
    for i, strategy in enumerate(strategies):
        sub = data[data["strategy"] == strategy]
        color = get_color(strategy, "#000000")
        ax.scatter(
            sub["x"],
            sub["y"],
            s=POINT_SIZE,
            c=color,
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=1 + 5 * i,
        )
        if filled_contours(
            ax,
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            color,
            zorder=2 + 5 * i,
            xlim=xlim,
            ylim=ylim,
        ):
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=2.2, label=get_display_name(strategy))
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            handletextpad=0.4,
            columnspacing=0.8,
            borderaxespad=0.2,
            fontsize=5,
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, out_base.with_suffix(".pdf"))
    plt.close(fig)


def filter_strategies(data: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    include_strategies = {normalize_strategy(strategy) for strategy in strategies}
    filtered = data[data["strategy"].isin(include_strategies)].copy()
    if filtered.empty:
        raise RuntimeError(
            "No datapoints remain after filtering to strategies: "
            f"{', '.join(sorted(include_strategies))}"
        )
    return filtered


def resolve_escape_summary_path(config: dict, experiment_path: str) -> Path:
    configured = config.get("relative_density_change_escape_summary_path")
    if configured:
        path = Path(os.fspath(configured)).expanduser()
        if path.is_absolute():
            return path
        return Path(experiment_path).expanduser().resolve() / path
    return Path(experiment_path).expanduser().resolve() / DEFAULT_ESCAPE_SUMMARY_REL_PATH


def resolve_simulation_escape_path(args: argparse.Namespace, config: dict) -> Path:
    configured = args.simulation_escape_csv or config.get("simulation_escape_csv")
    if configured:
        return Path(os.fspath(configured)).expanduser()
    return DEFAULT_SIMULATION_ESCAPE_CSV


def load_escape_cutoffs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Escape summary CSV does not exist: {path}")

    summary = pd.read_csv(path)
    sample_col = next(
        (column for column in ("replicate", "sample", "well") if column in summary.columns),
        None,
    )
    escape_col = next(
        (column for column in ("escape_day", "first_escape_day") if column in summary.columns),
        None,
    )
    if sample_col is None or escape_col is None or "plate" not in summary.columns:
        raise ValueError(
            "Escape summary must contain plate, a replicate/sample/well column, "
            "and escape_day or first_escape_day."
        )

    escape = summary[["plate", sample_col, escape_col]].copy()
    escape = escape.rename(columns={sample_col: "well", escape_col: "escape_day"})
    escape["escape_day"] = pd.to_numeric(escape["escape_day"], errors="coerce")
    escape = escape[np.isfinite(escape["escape_day"])].copy()
    if escape.empty:
        raise RuntimeError(f"No finite escape days found in {path}")

    if "last_confined_day" in summary.columns:
        cutoff = summary[["plate", sample_col, "last_confined_day"]].copy()
        cutoff = cutoff.rename(columns={sample_col: "well"})
        cutoff["last_confined_day"] = pd.to_numeric(
            cutoff["last_confined_day"],
            errors="coerce",
        )
        escape = escape.merge(cutoff, on=["plate", "well"], how="left")
        escape["pre_escape_cutoff_day"] = escape["last_confined_day"].where(
            np.isfinite(escape["last_confined_day"]),
            escape["escape_day"],
        )
    else:
        escape["pre_escape_cutoff_day"] = escape["escape_day"]

    return (
        escape[["plate", "well", "escape_day", "pre_escape_cutoff_day"]]
        .groupby(["plate", "well"], as_index=False)
        .min()
    )


def normalize_simulation_run_id(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:run[_-]?)?0*(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        return text.lower()
    return str(int(match.group(1)))


def load_simulation_escape_cutoffs(run_dir: Path, path: Path) -> pd.DataFrame:
    manifest_path = run_dir / "simulation_density_run_manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Simulation run manifest does not exist: {manifest_path}")
    if not path.exists():
        raise FileNotFoundError(f"Simulation escape CSV does not exist: {path}")

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    required_manifest = {"plate", "well_identifier", "strategy", "run_id"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"{manifest_path} is missing columns: {', '.join(sorted(missing_manifest))}"
        )

    summary = pd.read_csv(path, dtype={"protocol": str, "run": str})
    required_summary = {"protocol", "run", "TTE"}
    missing_summary = required_summary - set(summary.columns)
    if missing_summary:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing_summary))}")

    escape = summary[["protocol", "run", "TTE"]].copy()
    escape["strategy_key"] = escape["protocol"].astype(str).map(normalize_strategy)
    escape["run_key"] = escape["run"].map(normalize_simulation_run_id)
    escape["escape_day"] = pd.to_numeric(escape["TTE"], errors="coerce")
    escape = escape[["strategy_key", "run_key", "escape_day"]].drop_duplicates()

    conflicts = (
        escape[np.isfinite(escape["escape_day"])]
        .groupby(["strategy_key", "run_key"])["escape_day"]
        .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        examples = ", ".join(f"{strategy}/run_{run}" for strategy, run in conflicts.index[:5])
        raise ValueError(f"Conflicting TTE values in {path}: {examples}")

    escape = (
        escape[np.isfinite(escape["escape_day"]) & (escape["escape_day"] > 0)]
        .groupby(["strategy_key", "run_key"], as_index=False)["escape_day"]
        .first()
    )
    manifest["strategy_key"] = manifest["strategy"].astype(str).map(normalize_strategy)
    manifest["run_key"] = manifest["run_id"].map(normalize_simulation_run_id)
    joined = manifest.merge(
        escape,
        on=["strategy_key", "run_key"],
        how="left",
        validate="many_to_one",
    )

    matched = joined[np.isfinite(joined["escape_day"])].copy()
    if matched.empty:
        raise RuntimeError(
            f"No simulation runs in {manifest_path} matched a positive TTE in {path}"
        )
    unmatched = joined[~np.isfinite(joined["escape_day"])]
    if not unmatched.empty:
        counts = unmatched.groupby("strategy_key").size().sort_index()
        detail = ", ".join(f"{strategy}: {count}" for strategy, count in counts.items())
        print(
            f"[WARN] {len(unmatched)} simulation manifest runs have no positive TTE "
            f"in {path} ({detail}); excluding them from escape panels."
        )

    matched = matched.rename(columns={"well_identifier": "well"})
    matched["pre_escape_cutoff_day"] = matched["escape_day"]
    return matched[["plate", "well", "escape_day", "pre_escape_cutoff_day"]]


def filter_before_escape(data: pd.DataFrame, escape_cutoffs: pd.DataFrame) -> pd.DataFrame:
    merged = data.merge(escape_cutoffs, on=["plate", "well"], how="inner")
    valid = (
        np.isfinite(merged["next_day"])
        & np.isfinite(merged["pre_escape_cutoff_day"])
        & (merged["next_day"] <= merged["pre_escape_cutoff_day"])
    )
    filtered = merged[valid].copy()
    if filtered.empty:
        raise RuntimeError("No datapoints remain after filtering to intervals before escape.")
    return filtered


def filter_after_escape(data: pd.DataFrame, escape_cutoffs: pd.DataFrame) -> pd.DataFrame:
    merged = data.merge(escape_cutoffs, on=["plate", "well"], how="inner")
    valid = (
        np.isfinite(merged["first_day"])
        & np.isfinite(merged["escape_day"])
        & (merged["first_day"] >= merged["escape_day"])
    )
    filtered = merged[valid].copy()
    if filtered.empty:
        raise RuntimeError("No datapoints remain after filtering to intervals after escape.")
    return filtered


def apply_experiment_filters(data: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    for sample, cutoff_day in EXCLUDE_SAMPLES.items():
        sample_mask = data["well"] == sample
        if not sample_mask.any():
            continue
        if cutoff_day is None:
            mask &= ~sample_mask
        else:
            mask &= ~(sample_mask & (data["next_day"] > float(cutoff_day)))

    filtered = data[mask].copy()
    if filtered.empty:
        raise RuntimeError("No datapoints remain after applying experiment filters.")
    return filtered


def plot_strategy_panels(
    data: pd.DataFrame,
    out_base: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xscale: str,
    yscale: str,
    y_label: str,
) -> None:
    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if strategy in set(data["strategy"])
    ]
    strategies.extend(
        sorted(strategy for strategy in set(data["strategy"]) if strategy not in strategies)
    )
    if not strategies:
        return
    strategy_data = []
    for strategy in strategies:
        sub = data[data["strategy"] == strategy]
        strategy_data.append(
            (strategy, sub["x"].to_numpy(), sub["y"].to_numpy())
        )
    plot_strategy_panel_grid(
        strategy_data,
        out_base,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        y_label=y_label,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_path = resolve_experiment_path(
        experiment_path=config.get("experiment_path"),
        mount_root=config.get("mount_root"),
        hdf5_path=config.get("hdf5_path"),
    )
    run_dir = resolve_input_run_dir(args, config, experiment_path)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir / "summary_plots"

    rng = np.random.default_rng(args.seed)
    if args.no_res_density_gate:
        res_density_windows = [None]
    elif args.res_density_window is not None:
        res_density_windows = normalize_res_density_windows(tuple(args.res_density_window))
    else:
        res_density_windows = normalize_res_density_windows(DEFAULT_RES_DENSITY_WINDOW)
        if INCLUDE_UNGATED_PLOTS:
            res_density_windows = [None] + [
                window for window in res_density_windows if window is not None
            ]

    print(f"Reading pipeline output from: {run_dir}")
    skip_escape_panels = bool(args.skip_escape_panels)
    simulation_run = is_simulation_density_run(run_dir)
    if simulation_run:
        if skip_escape_panels:
            print("Detected simulation density run; before/after-escape panels were disabled.")
        else:
            print("Detected simulation density run; enabling TTE-based escape panels.")

    selected_change_types = [
        "abs_per_day" if change_type == "abs" else change_type
        for change_type in args.change_types
    ]
    selected_change_types = list(dict.fromkeys(selected_change_types))

    for res_density_window in res_density_windows:
        output_suffix = density_window_token(res_density_window)
        if res_density_window is not None:
            print(
                "Keeping pixels with first-frame resistant density "
                f"{res_density_window[0]:g} to {res_density_window[1]:g} cells µm$^{{-2}}$."
            )
        else:
            print("No first-frame resistant-density gate.")

        for change_type in selected_change_types:
            spec = CHANGE_SPECS[change_type]
            xlim = parse_float_limits(config, "relative_density_change_scatter_xlim", DEFAULT_XLIM)
            ylim = parse_float_limits(config, spec["ylim_key"], spec["default_ylim"])
            xscale = parse_axis_scale(config, "relative_density_change_scatter_xscale")
            yscale = parse_axis_scale(config, spec["yscale_key"])
            y_label = str(spec["ylabel"])

            print(f"Plotting {spec['display_name']}.")

            if skip_escape_panels:
                data_arrays = collect_strategy_arrays(
                    run_dir=run_dir,
                    config=config,
                    change_type=change_type,
                    rng=rng,
                    max_points_per_map=args.max_points_per_map,
                    max_points_per_strategy=args.max_points_per_strategy,
                    xlim=xlim,
                    ylim=ylim,
                    xscale=xscale,
                    yscale=yscale,
                    res_density_window=res_density_window,
                )
                out_base = output_dir / f"{spec['combined_stem']}{output_suffix}"
                panel_out_base = output_dir / f"{spec['panel_stem']}{output_suffix}"
                plot_datapoints_arrays(
                    data_arrays,
                    out_base,
                    xlim=xlim,
                    ylim=ylim,
                    xscale=xscale,
                    yscale=yscale,
                    y_label=y_label,
                    include_strategies=args.include_strategies,
                )
                plot_strategy_panels_arrays(
                    data_arrays,
                    panel_out_base,
                    xlim=xlim,
                    ylim=ylim,
                    xscale=xscale,
                    yscale=yscale,
                    y_label=y_label,
                )
                include = {normalize_strategy(strategy) for strategy in args.include_strategies}
                combined_total = sum(
                    int(entry["count"]) for strategy, entry in data_arrays.items() if strategy in include
                )
                print(
                    f"Plotted {combined_total:,} {spec['display_name']} datapoints "
                    "in the combined plot."
                )
                for strategy in sorted_strategy_keys(set(data_arrays)):
                    if strategy in include:
                        print(f"- {strategy}: {int(data_arrays[strategy]['count']):,}")
                panel_total = sum(int(entry["count"]) for entry in data_arrays.values())
                print(
                    f"Plotted {panel_total:,} {spec['display_name']} datapoints "
                    "in the all-strategy panel plot."
                )
                for strategy in sorted_strategy_keys(set(data_arrays)):
                    print(f"- {strategy}: {int(data_arrays[strategy]['count']):,}")
                print(f"Saved: {out_base.with_suffix('.pdf')}")
                print(f"Saved: {panel_out_base.with_suffix('.pdf')}")
                continue

            data = collect_datapoints(
                run_dir=run_dir,
                config=config,
                change_type=change_type,
                rng=rng,
                max_points_per_map=args.max_points_per_map,
                max_points_per_strategy=args.max_points_per_strategy,
                xlim=xlim,
                ylim=ylim,
                xscale=xscale,
                yscale=yscale,
                res_density_window=res_density_window,
                include_interval_metadata=not skip_escape_panels,
            )
            out_base = output_dir / f"{spec['combined_stem']}{output_suffix}"
            combined_data = filter_strategies(data, args.include_strategies)
            plot_datapoints(
                combined_data,
                out_base,
                xlim=xlim,
                ylim=ylim,
                xscale=xscale,
                yscale=yscale,
                y_label=y_label,
            )
            panel_out_base = output_dir / f"{spec['panel_stem']}{output_suffix}"
            plot_strategy_panels(
                data,
                panel_out_base,
                xlim=xlim,
                ylim=ylim,
                xscale=xscale,
                yscale=yscale,
                y_label=y_label,
            )
            before_escape_counts = None
            after_escape_counts = None
            escape_summary_path = None
            if not skip_escape_panels:
                if simulation_run:
                    escape_summary_path = resolve_simulation_escape_path(args, config)
                    escape_cutoffs = load_simulation_escape_cutoffs(
                        run_dir,
                        escape_summary_path,
                    )
                    before_escape_data = filter_before_escape(data, escape_cutoffs)
                else:
                    escape_summary_path = resolve_escape_summary_path(config, experiment_path)
                    escape_cutoffs = load_escape_cutoffs(escape_summary_path)
                    before_escape_data = apply_experiment_filters(
                        filter_before_escape(data, escape_cutoffs)
                    )
                before_escape_panel_out_base = (
                    output_dir / f"{spec['before_stem']}{output_suffix}"
                )
                plot_strategy_panels(
                    before_escape_data,
                    before_escape_panel_out_base,
                    xlim=xlim,
                    ylim=ylim,
                    xscale=xscale,
                    yscale=yscale,
                    y_label=y_label,
                )
                after_escape_data = filter_after_escape(data, escape_cutoffs)
                if not simulation_run:
                    after_escape_data = apply_experiment_filters(after_escape_data)
                after_escape_panel_out_base = (
                    output_dir / f"{spec['after_stem']}{output_suffix}"
                )
                plot_strategy_panels(
                    after_escape_data,
                    after_escape_panel_out_base,
                    xlim=xlim,
                    ylim=ylim,
                    xscale=xscale,
                    yscale=yscale,
                    y_label=y_label,
                )
                before_escape_counts = before_escape_data.groupby("strategy").size().sort_index()
                after_escape_counts = after_escape_data.groupby("strategy").size().sort_index()

            combined_counts = combined_data.groupby("strategy").size().sort_index()
            panel_counts = data.groupby("strategy").size().sort_index()
            print(
                f"Plotted {len(combined_data):,} sampled {spec['display_name']} datapoints "
                "in the combined plot."
            )
            for strategy, count in combined_counts.items():
                print(f"- {strategy}: {count:,}")
            print(
                f"Plotted {len(data):,} sampled {spec['display_name']} datapoints "
                "in the all-strategy panel plot."
            )
            for strategy, count in panel_counts.items():
                print(f"- {strategy}: {count:,}")
            if before_escape_counts is not None and after_escape_counts is not None:
                print(
                    "Plotted "
                    f"{len(before_escape_data):,} sampled {spec['display_name']} datapoints "
                    "in the before-escape panel plot."
                )
                print(f"Before-escape filtering used: {escape_summary_path}")
                for strategy, count in before_escape_counts.items():
                    print(f"- {strategy}: {count:,}")
                print(
                    "Plotted "
                    f"{len(after_escape_data):,} sampled {spec['display_name']} datapoints "
                    "in the after-escape panel plot."
                )
                print(f"After-escape filtering used: {escape_summary_path}")
                for strategy, count in after_escape_counts.items():
                    print(f"- {strategy}: {count:,}")
            print(f"Saved: {out_base.with_suffix('.pdf')}")
            print(f"Saved: {panel_out_base.with_suffix('.pdf')}")
            if not skip_escape_panels:
                print(f"Saved: {before_escape_panel_out_base.with_suffix('.pdf')}")
                print(f"Saved: {after_escape_panel_out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
