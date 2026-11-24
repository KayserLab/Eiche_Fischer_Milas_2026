#!/usr/bin/env python3
"""Build one density-binned median curve per experimental well/simulation run.

Pixels from all eligible intervals of a well/run are accumulated by local
density. No time averaging is performed. The output retains one curve per
replicate so the plotting script can average curves with equal replicate weight.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as core
from shared.data_paths import figure_data_dir

PIXEL_STORE = figure_data_dir(
    "Figure_3", "master_curve_160_pixel_store", create=False,
) / "master_curve_pixels.h5"
STRATEGIES = ("CT", "AT50", "AT100")
DATASETS = ("simulation", "experiment")
INTERVAL_MODES = {
    "simulation": "daily",
    "experiment": "experimental_schedule",
}
PERIOD = ("after_escape" if "--after-escape" in sys.argv else "all_timepoints" if "--all-timepoints" in sys.argv else os.environ.get("CORESEED_WELL_CURVE_PERIOD", "before_escape")).strip().lower()
if PERIOD not in {"before_escape", "after_escape", "all_timepoints"}:
    raise ValueError(f"Invalid CORESEED_WELL_CURVE_PERIOD={PERIOD!r}")
# The one-cell option matches the local 160-um Epanechnikov-kernel density
# used in plot_at50_density_maps.py. The legacy option preserves prior plots.
SUPPORT_MODE = ("one_cell_t0" if "--one-cell-threshold" in sys.argv else os.environ.get("CORESEED_WELL_CURVE_SUPPORT", "legacy")).strip().lower()
if SUPPORT_MODE not in {"legacy", "one_cell_t0"}:
    raise ValueError(f"Invalid CORESEED_WELL_CURVE_SUPPORT={SUPPORT_MODE!r}")
ONE_CELL_DENSITY_THRESHOLD = 2.0 / (np.pi * core.KERNEL_UM ** 2)
SUPPORT_THRESHOLD = (
    ONE_CELL_DENSITY_THRESHOLD if SUPPORT_MODE == "one_cell_t0" else 1.0e-8
)
X_BIN_WIDTH = float(os.environ.get("CORESEED_WELL_CURVE_X_BIN_WIDTH", "0.10"))
if not np.isfinite(X_BIN_WIDTH) or X_BIN_WIDTH <= 0:
    raise ValueError(f"Invalid CORESEED_WELL_CURVE_X_BIN_WIDTH={X_BIN_WIDTH!r}")
X_BIN_TOKEN = f"{X_BIN_WIDTH:g}".replace(".", "p")
_output_name = "master_curve_160_well_weighted"
if SUPPORT_MODE == "one_cell_t0":
    _output_name += "_one_cell_t0"
if PERIOD == "all_timepoints":
    _output_name += "_all_timepoints"
elif PERIOD == "after_escape":
    _output_name += "_after_escape"
if not np.isclose(X_BIN_WIDTH, 0.10):
    _output_name += f"_xbin_{X_BIN_TOKEN}"
OUTPUT_DIR = figure_data_dir(
    "Figure_3",
    _output_name,
    create=False,
)
OUTPUT_TABLE = OUTPUT_DIR / "well_curve_medians.tsv"
OUTPUT_METADATA = OUTPUT_DIR / "well_curve_medians.metadata.json"

X_EDGES = np.arange(0.0, 3.0 + 0.5 * X_BIN_WIDTH, X_BIN_WIDTH)
Y_EDGES = {
    # A t0-referenced fractional change is not bounded by +/-2 like the
    # previous symmetric definition, so retain a wider finite range.
    "relative": np.linspace(-10.01, 10.01, 8193),
    "absolute": np.linspace(-4.01, 4.01, 8193),
}
READ_CHUNK_ROWS = 500_000
CACHE_VERSION = 7
RELATIVE_DEFINITION = "change_over_t0_per_day"
AGGREGATION_WORKERS = max(
    1,
    int(
        os.environ.get(
            "CORESEED_WELL_CURVE_AGGREGATION_WORKERS",
            min(3, os.cpu_count() or 1),
        )
    ),
)


@dataclass
class MedianAccumulator:
    """Histogram-backed medians for every density bin of one well/run."""

    count: np.ndarray
    histogram: np.ndarray
    y_edges: np.ndarray

    @classmethod
    def create(cls, growth_mode: str) -> "MedianAccumulator":
        return cls(
            count=np.zeros(len(X_EDGES) - 1, dtype=np.int64),
            histogram=np.zeros(
                (len(X_EDGES) - 1, len(Y_EDGES[growth_mode]) - 1),
                dtype=np.int32,
            ),
            y_edges=Y_EDGES[growth_mode],
        )

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        valid = (
                np.isfinite(x)
                & np.isfinite(y)
                & (x >= X_EDGES[0])
                & (x <= X_EDGES[-1])
                & (y >= self.y_edges[0])
                & (y <= self.y_edges[-1])
        )
        if not np.any(valid):
            return

        x_values = x[valid]
        y_values = y[valid]
        x_bin = np.floor(x_values / X_BIN_WIDTH).astype(np.int64)
        np.clip(x_bin, 0, len(X_EDGES) - 2, out=x_bin)
        y_bin = np.searchsorted(self.y_edges, y_values, side="right") - 1
        np.clip(y_bin, 0, len(self.y_edges) - 2, out=y_bin)
        self.count += np.bincount(x_bin, minlength=len(self.count))
        np.add.at(self.histogram, (x_bin, y_bin), 1)

    def median(self, x_bin: int) -> float:
        histogram = self.histogram[x_bin]
        count = int(histogram.sum())
        if count == 0:
            return float("nan")
        cumulative = np.cumsum(histogram)
        lower_rank = (count - 1) // 2
        upper_rank = count // 2
        lower_index = int(
            np.searchsorted(cumulative, lower_rank, side="right")
        )
        upper_index = int(
            np.searchsorted(cumulative, upper_rank, side="right")
        )
        lower_center = 0.5 * (
                self.y_edges[lower_index] + self.y_edges[lower_index + 1]
        )
        upper_center = 0.5 * (
                self.y_edges[upper_index] + self.y_edges[upper_index + 1]
        )
        return float(0.5 * (lower_center + upper_center))


def _source_signature() -> dict[str, object]:
    stat = PIXEL_STORE.stat()
    return {
        "cache_version": CACHE_VERSION,
        "pixel_store": os.fspath(PIXEL_STORE),
        "pixel_store_size": stat.st_size,
        "pixel_store_mtime_ns": stat.st_mtime_ns,
        "interval_modes": INTERVAL_MODES,
        "strategies": list(STRATEGIES),
        "support_mode": SUPPORT_MODE,
        "support_threshold": SUPPORT_THRESHOLD,
        "x_bin_width": X_BIN_WIDTH,
        "relative_definition": RELATIVE_DEFINITION,
        "relative_support": "resistant t0 >= threshold",
        "absolute_support": (
            "resistant t0 >= threshold"
            if SUPPORT_MODE == "one_cell_t0"
            else "average resistant endpoint density >= threshold"
        ),
        "period": f"{PERIOD} and analysis allowed",
    }


def cache_is_current() -> bool:
    if (
            not PIXEL_STORE.is_file()
            or not OUTPUT_TABLE.is_file()
            or not OUTPUT_METADATA.is_file()
    ):
        return False
    try:
        metadata = json.loads(OUTPUT_METADATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata == _source_signature()


def _rows_for_accumulators(
        dataset: str,
        strategy: str,
        interval_mode: str,
        accumulators: dict[tuple[str, str], MedianAccumulator],
) -> list[dict[str, object]]:
    centers = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
    rows: list[dict[str, object]] = []
    for (replicate, growth_mode), accumulator in sorted(accumulators.items()):
        for x_bin, center in enumerate(centers):
            count = int(accumulator.count[x_bin])
            if count == 0:
                continue
            rows.append({
                "dataset": dataset,
                "strategy": strategy,
                "replicate": replicate,
                "interval_mode": interval_mode,
                "relative_definition": RELATIVE_DEFINITION,
                "growth_mode": growth_mode,
                "x_bin_center_cells_per_1000_um2": float(center),
                "n_pixels": count,
                "median": accumulator.median(x_bin),
            })
    return rows


def build_strategy_well_curve_rows(
        pixel_store: Path,
        dataset: str,
        strategy: str,
) -> list[dict[str, object]]:
    """Reduce one independent dataset/strategy HDF5 subset."""
    interval_mode = INTERVAL_MODES[dataset]
    columns = [
        "dataset",
        "strategy",
        "replicate",
        "interval_mode",
        "delta_days",
        "before_escape",
        "analysis_allowed",
        "total_density_average",
        "resistant_density_t0",
        "resistant_density_t1",
        "resistant_density_average",
        "non_isolated_change",
    ]
    accumulators: dict[tuple[str, str], MedianAccumulator] = {}
    where = (
        f'dataset == "{dataset}" & strategy == "{strategy}" '
        f'& interval_mode == "{interval_mode}"'
    )
    iterator = pd.read_hdf(
        pixel_store,
        key="pixels",
        where=where,
        columns=columns,
        chunksize=READ_CHUNK_ROWS,
    )
    for chunk in iterator:
        before_escape = chunk["before_escape"].to_numpy(bool)
        period_valid = (
            before_escape
            if PERIOD == "before_escape"
            else (~before_escape if PERIOD == "after_escape"
                  else np.ones(len(chunk), dtype=bool))
        )
        base = chunk[
            period_valid
            & chunk["analysis_allowed"].to_numpy(bool)
            & chunk["non_isolated_change"].to_numpy(bool)
        ]
        if base.empty:
            continue
        x_all = base["total_density_average"].to_numpy(float) * 1000.0
        resistant0 = base["resistant_density_t0"].to_numpy(float)
        resistant1 = base["resistant_density_t1"].to_numpy(float)
        relative_support = resistant0 >= SUPPORT_THRESHOLD
        absolute_support = (
            resistant0 >= SUPPORT_THRESHOLD
            if SUPPORT_MODE == "one_cell_t0"
            else base["resistant_density_average"].to_numpy(float)
            >= SUPPORT_THRESHOLD
        )
        absolute_growth = (
                (resistant1 - resistant0)
                / base["delta_days"].to_numpy(float)
                * 1000.0
        )
        relative_growth = np.full(resistant0.shape, np.nan, dtype=float)
        np.divide(
            resistant1 - resistant0,
            resistant0,
            out=relative_growth,
            where=relative_support,
        )
        relative_growth /= base["delta_days"].to_numpy(float)

        replicate_values = base["replicate"].astype(str).to_numpy()
        for replicate in pd.unique(replicate_values):
            in_replicate = replicate_values == replicate
            for growth_mode, support, values in (
                    ("relative", relative_support, relative_growth),
                    ("absolute", absolute_support, absolute_growth),
            ):
                selected = in_replicate & support
                if not np.any(selected):
                    continue
                key = (replicate, growth_mode)
                accumulator = accumulators.setdefault(
                    key,
                    MedianAccumulator.create(growth_mode),
                )
                accumulator.add(x_all[selected], values[selected])

    return _rows_for_accumulators(
        dataset, strategy, interval_mode, accumulators
    )


def build_well_curve_table(
        datasets: tuple[str, ...] = DATASETS,
        pixel_store: Path | None = None,
) -> pd.DataFrame:
    """Build curve rows serially; useful for a single HDF5 reader."""
    source = PIXEL_STORE if pixel_store is None else pixel_store
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run build_master_curve_pixel_store.py first."
        )
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for strategy in STRATEGIES:
            rows.extend(build_strategy_well_curve_rows(source, dataset, strategy))
    return pd.DataFrame(rows)


def build_well_curve_table_parallel(
        pixel_store: Path,
        datasets: tuple[str, ...] = DATASETS,
        workers: int = AGGREGATION_WORKERS,
) -> pd.DataFrame:
    """Reduce independent strategy subsets concurrently from a closed store."""
    if not pixel_store.is_file():
        raise FileNotFoundError(f"Missing {pixel_store}.")
    tasks = [
        (pixel_store, dataset, strategy)
        for dataset in datasets
        for strategy in STRATEGIES
    ]
    if workers <= 1 or len(tasks) <= 1:
        return build_well_curve_table(datasets, pixel_store)

    rows: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=context,
    ) as executor:
        futures = [
            executor.submit(build_strategy_well_curve_rows, *task)
            for task in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    return pd.DataFrame(rows)


def read_precomputed_well_curves(pixel_store: Path) -> pd.DataFrame | None:
    """Return compatible build-time curves and rebuild incompatible subsets."""
    # The current build stores compact curves at the standard 0.10-bin width.
    # Finer bins are reconstructed once from the retained pixel-level table.
    if (
            SUPPORT_MODE != "legacy"
            or not np.isclose(X_BIN_WIDTH, 0.10)
            or PERIOD != "before_escape"
    ):
        return None
    with pd.HDFStore(pixel_store, mode="r") as store:
        if "/well_curves" not in store.keys():
            return None
        stored = store["well_curves"]

    pieces: list[pd.DataFrame] = []
    missing: list[str] = []
    for dataset in DATASETS:
        subset = stored[stored["dataset"] == dataset].copy()
        expected_mode = INTERVAL_MODES[dataset]
        if "interval_mode" in subset.columns:
            subset = subset[subset["interval_mode"] == expected_mode]
        elif dataset == "experiment":
            # Legacy build-time curves always used experimental acquisition
            # intervals, which remain correct for the experiment subset.
            subset["interval_mode"] = expected_mode
        else:
            # Legacy simulation curves used experimental-schedule endpoints and
            # must not be reused for the requested daily simulation analysis.
            subset = subset.iloc[0:0]

        if "relative_definition" in subset.columns:
            subset = subset[
                subset["relative_definition"] == RELATIVE_DEFINITION
            ]
        else:
            # Earlier compact curves contain symmetric relative changes and
            # cannot be converted without returning to the endpoint pixels.
            subset = subset.iloc[0:0]

        available_strategies = set(subset["strategy"].astype(str))
        if subset.empty or not set(STRATEGIES).issubset(available_strategies):
            missing.append(dataset)
        else:
            pieces.append(subset)

    if missing:
        print(
            "[cache] Rebuilding incompatible compact subsets from raw pixels: "
            + ", ".join(
                f"{dataset} ({INTERVAL_MODES[dataset]})" for dataset in missing
            )
        )
        pieces.append(
            build_well_curve_table_parallel(pixel_store, tuple(missing))
        )
    return pd.concat(pieces, ignore_index=True) if pieces else None


def ensure_well_curve_table(*, force: bool = False) -> Path:
    """Build the reusable TSV cache when it is absent or out of date."""
    if not force and cache_is_current():
        print(f"[cache] Reusing {OUTPUT_TABLE}")
        return OUTPUT_TABLE

    result = read_precomputed_well_curves(PIXEL_STORE)
    if result is None:
        print("[cache] No build-time well curves; reducing raw pixel rows.")
        result = build_well_curve_table_parallel(PIXEL_STORE)
    else:
        print("[cache] Reading build-time compact well curves.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_table = OUTPUT_TABLE.with_suffix(".building.tsv")
    temporary_metadata = OUTPUT_METADATA.with_suffix(".building.json")
    result.to_csv(temporary_table, sep="\t", index=False)
    temporary_metadata.write_text(
        json.dumps(_source_signature(), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_table.replace(OUTPUT_TABLE)
    temporary_metadata.replace(OUTPUT_METADATA)
    print(
        f"[saved] {OUTPUT_TABLE} "
        f"({result['replicate'].nunique()} well/run curves, {len(result)} bins)"
    )
    return OUTPUT_TABLE


def main() -> None:
    ensure_well_curve_table()


if __name__ == "__main__":
    main()
