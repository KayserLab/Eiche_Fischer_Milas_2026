#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot non-overlapping resistant-density bands with equal well weighting."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

import plot_binned_density_change_strategy_lines_coarsegrained as pooled
from shared.strategy_colors import get_color, get_display_name

DEFAULT_OUTPUT_SUBDIR = Path(
    "summary_plots/requested_variations_20260716/"
    "nonoverlap_bands_equal_well"
)
BANDS = (
    ("low", 2.0e-5, 4.375e-4, True),
    ("middle", 4.375e-4, 1.0e-3, False),
    ("high", 1.0e-3, 2.0e-3, False),
)
BAND_TITLES = {
    "low": "Low resistant density\n[2e-5, 4.375e-4] cells µm$^{-2}$",
    "middle": "Middle resistant density\n(4.375e-4, 1e-3] cells µm$^{-2}$",
    "high": "High resistant density\n(1e-3, 2e-3] cells µm$^{-2}$",
}
PER_WELL_COLUMNS = (
    "run_type", "plate", "well", "strategy", "band",
    "res_density_lower_cells_per_um2", "res_density_upper_cells_per_um2",
    "lower_bound_inclusive", "x_bin_left_cells_per_1000_um2",
    "x_bin_right_cells_per_1000_um2", "x_bin_center_cells_per_1000_um2",
    "n_pixels", "pixel_mean", "pixel_median",
)
SUMMARY_COLUMNS = (
    "run_type", "strategy", "band", "res_density_lower_cells_per_um2",
    "res_density_upper_cells_per_um2", "x_bin_left_cells_per_1000_um2",
    "x_bin_right_cells_per_1000_um2", "x_bin_center_cells_per_1000_um2",
    "n_wells", "n_pixels_total", "mean_of_well_pixel_means",
    "sd_between_well_pixel_means", "median_of_well_pixel_medians",
    "median_well_ci95_low", "median_well_ci95_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.fspath(pooled.base.DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-dir", dest="run_dirs", action="append")
    parser.add_argument("--output-dir")
    parser.add_argument("--x-bin-size", type=float, default=pooled.DEFAULT_X_BIN_SIZE)
    parser.add_argument("--y-hist-bins", type=int, default=pooled.DEFAULT_Y_HIST_BINS)
    parser.add_argument("--min-well-bin-pixels", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def summarize_well(task: tuple[object, ...]) -> tuple[list[dict[str, object]], int]:
    (
        run_dir_text, plate, well, strategy, x_edges_tuple, y_limits,
        y_hist_bins, run_type,
    ) = task
    run_dir = Path(str(run_dir_text))
    well_dir = run_dir / str(plate) / str(well)
    maps_dir = well_dir / "density_maps"
    changes_dir = well_dir / "density_changes"
    x_edges = np.asarray(x_edges_tuple, dtype=np.float64)
    n_x = len(x_edges) - 1
    x_width = float(x_edges[1] - x_edges[0])
    y_edges = np.linspace(*y_limits, int(y_hist_bins) + 1)
    counts = np.zeros((3, n_x), dtype=np.int64)
    totals = np.zeros((3, n_x), dtype=np.float64)
    hist = np.zeros((3, n_x, int(y_hist_bins)), dtype=np.int64)
    intervals = 0

    for rel_path in sorted(changes_dir.glob(pooled.base.CHANGE_SPECS["rel"]["glob"])):
        match = pooled.base.REL_CHANGE_RE.match(rel_path.name)
        if match is None:
            continue
        first_tp = match.group("first_tp")
        total_path = pooled.base.density_map_path(maps_dir, str(well), first_tp)
        res_path = pooled.base.resistant_density_map_path(
            maps_dir, str(well), first_tp
        )
        if not (total_path.is_file() and res_path.is_file()):
            continue
        total = tiff.imread(total_path).astype(np.float32, copy=False).ravel()
        res = tiff.imread(res_path).astype(np.float32, copy=False).ravel()
        change = tiff.imread(rel_path).astype(np.float32, copy=False).ravel()
        if len({total.shape, res.shape, change.shape}) != 1:
            raise ValueError(f"Shape mismatch for {rel_path}")
        x = total * pooled.base.X_DISPLAY_SCALE
        common = (
            np.isfinite(x) & np.isfinite(res) & np.isfinite(change)
            & (x >= x_edges[0]) & (x <= x_edges[-1])
            & (change >= y_limits[0]) & (change <= y_limits[1])
        )
        x_bins = np.zeros(x.shape, dtype=np.int64)
        finite_x = np.isfinite(x)
        x_bins[finite_x] = np.floor(
            (x[finite_x] - x_edges[0]) / x_width
        ).astype(np.int64)
        np.clip(x_bins, 0, n_x - 1, out=x_bins)
        for band_index, (_, lower, upper, include_lower) in enumerate(BANDS):
            lower_mask = res >= lower if include_lower else res > lower
            valid = common & lower_mask & (res <= upper)
            if not np.any(valid):
                continue
            bx = x_bins[valid]
            by = change[valid].astype(np.float64, copy=False)
            counts[band_index] += np.bincount(bx, minlength=n_x)
            totals[band_index] += np.bincount(bx, weights=by, minlength=n_x)
            hy = np.floor(
                (by - y_edges[0]) * int(y_hist_bins)
                / (y_edges[-1] - y_edges[0])
            ).astype(np.int64)
            np.clip(hy, 0, int(y_hist_bins) - 1, out=hy)
            flat = bx * int(y_hist_bins) + hy
            hist[band_index] += np.bincount(
                flat, minlength=n_x * int(y_hist_bins)
            ).reshape(n_x, int(y_hist_bins))
        intervals += 1

    rows = []
    for band_index, (band, lower, upper, include_lower) in enumerate(BANDS):
        for x_index in range(n_x):
            n_pixels = int(counts[band_index, x_index])
            if not n_pixels:
                continue
            rows.append({
                "run_type": run_type, "plate": plate, "well": well,
                "strategy": strategy, "band": band,
                "res_density_lower_cells_per_um2": lower,
                "res_density_upper_cells_per_um2": upper,
                "lower_bound_inclusive": include_lower,
                "x_bin_left_cells_per_1000_um2": x_edges[x_index],
                "x_bin_right_cells_per_1000_um2": x_edges[x_index + 1],
                "x_bin_center_cells_per_1000_um2": 0.5 * (
                    x_edges[x_index] + x_edges[x_index + 1]
                ),
                "n_pixels": n_pixels,
                "pixel_mean": totals[band_index, x_index] / n_pixels,
                "pixel_median": pooled.histogram_quantile(
                    hist[band_index, x_index], y_edges, 0.5
                ),
            })
    return rows, intervals


def bootstrap_ci(
    values: np.ndarray, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))]
    return tuple(float(value) for value in np.quantile(np.median(samples, axis=1), (.025, .975)))


def aggregate_wells(
    frame: pd.DataFrame, min_pixels: int, replicates: int
) -> pd.DataFrame:
    eligible = frame[frame["n_pixels"] >= min_pixels].copy()
    keys = [
        "run_type", "strategy", "band",
        "res_density_lower_cells_per_um2", "res_density_upper_cells_per_um2",
        "x_bin_left_cells_per_1000_um2", "x_bin_right_cells_per_1000_um2",
        "x_bin_center_cells_per_1000_um2",
    ]
    rng = np.random.default_rng(20260730)
    rows = []
    for values, group in eligible.sort_values(keys).groupby(keys, sort=False):
        well_means = group["pixel_mean"].to_numpy(float)
        well_medians = group["pixel_median"].to_numpy(float)
        low, high = bootstrap_ci(well_medians, replicates, rng)
        row = dict(zip(keys, values))
        row.update({
            "n_wells": len(group),
            "n_pixels_total": int(group["n_pixels"].sum()),
            "mean_of_well_pixel_means": np.mean(well_means),
            "sd_between_well_pixel_means": (
                np.std(well_means, ddof=1) if len(group) > 1 else np.nan
            ),
            "median_of_well_pixel_medians": np.median(well_medians),
            "median_well_ci95_low": low,
            "median_well_ci95_high": high,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def style_axes(axes: np.ndarray, settings: dict[str, object]) -> None:
    for ax in axes.ravel():
        ax.set_xlim(*settings["xlim"])
        ax.set_ylim(*pooled.RELATIVE_DISPLAY_YLIM)
        ax.axhline(0, color="#777777", linewidth=.45, alpha=.45, zorder=0)


def plot_statistics(
    summary: pd.DataFrame,
    path: Path,
    settings: dict[str, object],
    run_type: str,
    min_pixels: int,
) -> None:
    pooled.base.ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    handles, labels = [], []
    for column, (band, *_rest) in enumerate(BANDS):
        axes[0, column].set_title(BAND_TITLES[band], fontsize=6.5, pad=3)
        for strategy in pooled.OVERVIEW_STRATEGIES:
            sub = summary[
                (summary["band"] == band) & (summary["strategy"] == strategy)
            ].sort_values("x_bin_center_cells_per_1000_um2")
            if sub.empty:
                continue
            x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(float)
            mean = sub["mean_of_well_pixel_means"].to_numpy(float)
            sd = sub["sd_between_well_pixel_means"].to_numpy(float)
            median = sub["median_of_well_pixel_medians"].to_numpy(float)
            low = sub["median_well_ci95_low"].to_numpy(float)
            high = sub["median_well_ci95_high"].to_numpy(float)
            color = get_color(strategy)
            axes[0, column].fill_between(
                x, mean - sd, mean + sd, color=color, alpha=.10, linewidth=0
            )
            line, = axes[0, column].plot(
                x, mean, color=color, marker="o", markersize=2.4, linewidth=1
            )
            axes[1, column].fill_between(
                x, low, high, color=color, alpha=.16, linewidth=0
            )
            axes[1, column].plot(
                x, median, color=color, marker="o", markersize=2.4, linewidth=1
            )
            if column == 0:
                handles.append(line)
                labels.append(get_display_name(strategy))
    style_axes(axes, settings)
    for ax in axes[1]:
        ax.set_xlabel(pooled.base.X_LABEL)
    axes[0, 0].set_ylabel("Mean relative growth\nacross wells")
    axes[1, 0].set_ylabel("Median relative growth\nacross wells")
    fig.suptitle(
        f"{run_type.title()} | Non-overlapping bands | Equal well weighting",
        fontsize=8, y=.985,
    )
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, .935),
        ncol=5, frameon=False, handlelength=1.4, columnspacing=.9,
        handletextpad=.3,
    )
    fig.text(
        .995, .008,
        "Top: mean +/- between-well SD. Bottom: median of well medians with "
        f"95% well-bootstrap CI. Well/bin requires n_px >= {min_pixels}.",
        ha="right", va="bottom", fontsize=5, color="#444444",
    )
    fig.subplots_adjust(
        left=.10, right=.985, bottom=.12, top=.84, wspace=.20, hspace=.24
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pooled.base.ft.savefig(fig, path)
    plt.close(fig)


def plot_support(
    summary: pd.DataFrame,
    path: Path,
    settings: dict[str, object],
    run_type: str,
) -> None:
    pooled.base.ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharex=True, sharey=True)
    handles, labels = [], []
    for column, (band, *_rest) in enumerate(BANDS):
        axes[column].set_title(BAND_TITLES[band], fontsize=6.5, pad=3)
        for strategy in pooled.OVERVIEW_STRATEGIES:
            sub = summary[
                (summary["band"] == band) & (summary["strategy"] == strategy)
            ].sort_values("x_bin_center_cells_per_1000_um2")
            if sub.empty:
                continue
            line, = axes[column].plot(
                sub["x_bin_center_cells_per_1000_um2"], sub["n_wells"],
                color=get_color(strategy), marker="o", markersize=2.6, linewidth=1,
            )
            if column == 0:
                handles.append(line)
                labels.append(get_display_name(strategy))
        axes[column].set_xlim(*settings["xlim"])
        axes[column].set_xlabel(pooled.base.X_LABEL)
    axes[0].set_ylabel("Contributing wells")
    fig.suptitle(f"{run_type.title()} | Equal-well analysis support", fontsize=8, y=.98)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, .885),
        ncol=5, frameon=False, handlelength=1.4, columnspacing=.9,
        handletextpad=.3,
    )
    fig.subplots_adjust(left=.085, right=.985, bottom=.21, top=.72, wspace=.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    pooled.base.ft.savefig(fig, path)
    plt.close(fig)


def process_run(
    run_dir: Path, output_root: Path, config: dict, args: argparse.Namespace
) -> None:
    settings = pooled.plot_settings(config, "rel")
    n_x = int(np.ceil(float(settings["xlim"][1]) / args.x_bin_size))
    x_edges = np.arange(n_x + 1, dtype=float) * args.x_bin_size
    y_limits = tuple(float(value) for value in settings["ylim"])
    run_type = (
        "simulation" if pooled.base.is_simulation_density_run(run_dir)
        else "experimental"
    )
    tasks = []
    for changes_dir in sorted(run_dir.glob("Plate_*/*/density_changes")):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        tasks.append((
            os.fspath(run_dir), plate, well,
            pooled.base.strategy_from_plate_and_well(plate, well),
            tuple(x_edges), y_limits, args.y_hist_bins, run_type,
        ))
    if not tasks:
        raise RuntimeError(f"No wells found in {run_dir}")
    print(f"[run] {run_type}: {run_dir}")
    print(f"[wells] {len(tasks)} using {args.workers} workers")
    rows, interval_count = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(summarize_well, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                well_rows, intervals = future.result()
            except Exception as exc:
                task = futures[future]
                raise RuntimeError(f"Failed at {task[1]}/{task[2]}") from exc
            rows.extend(well_rows)
            interval_count += intervals
            if done % 10 == 0 or done == len(tasks):
                print(f"[scan] {done}/{len(tasks)} wells; {interval_count} intervals")
    per_well = pd.DataFrame(rows, columns=PER_WELL_COLUMNS)
    summary = aggregate_wells(
        per_well, args.min_well_bin_pixels, args.bootstrap_replicates
    )
    output_root.mkdir(parents=True, exist_ok=True)
    per_well.to_csv(
        output_root / "per_well_band_bin_statistics.tsv", sep="\t", index=False
    )
    summary.to_csv(
        output_root / "equal_well_strategy_statistics.tsv", sep="\t", index=False
    )
    plot_statistics(
        summary, output_root / "nonoverlap_bands_equal_well.pdf",
        settings, run_type, args.min_well_bin_pixels,
    )
    plot_support(
        summary, output_root / "nonoverlap_bands_n_wells.pdf", settings, run_type
    )
    (output_root / "README.txt").write_text(
        f"{run_type.title()} non-overlapping resistant-density bands.\n\n"
        "Bands: [2e-5, 4.375e-4], (4.375e-4, 1e-3], and "
        "(1e-3, 2e-3] cells µm$^{-2}$.\n"
        "Pixels are pooled within each well/band/total-density bin. Each "
        "eligible well then contributes one pixel mean and one pixel median, "
        "so wells receive equal weight.\n"
        f"Eligibility: at least {args.min_well_bin_pixels} pixels per "
        "well/band/bin.\n"
        "Mean panels: mean +/- sample SD of well-level pixel means.\n"
        "Median panels: median of well-level pixel medians with a 95% "
        f"percentile CI from {args.bootstrap_replicates} well bootstraps.\n"
        f"Scanned intervals: {interval_count}. Per-well rows: {len(per_well)}. "
        f"Summary rows: {len(summary)}.\n",
        encoding="ascii",
    )
    print(f"[saved] {output_root}")


def main() -> None:
    args = parse_args()
    if args.x_bin_size <= 0 or args.min_well_bin_pixels < 1:
        raise ValueError("Bin sizes and count thresholds must be positive")
    if args.y_hist_bins < 32 or args.bootstrap_replicates < 100:
        raise ValueError("Use >=32 histogram bins and >=100 bootstraps")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    run_dirs = [
        Path(value).expanduser()
        for value in (args.run_dirs or pooled.DEFAULT_RUN_DIRS)
    ]
    if args.output_dir and len(run_dirs) != 1:
        raise ValueError("--output-dir requires exactly one --run-dir")
    config = pooled.base.load_config(args.config)
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        output_root = (
            Path(args.output_dir).expanduser()
            if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
        )
        process_run(run_dir, output_root, config, args)


if __name__ == "__main__":
    main()
