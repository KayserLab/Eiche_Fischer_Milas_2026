"""Retired shared aggregation utilities for linear-growth master curves with IQR."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.figure_3 import density_change_data_utils as base
from analysis.figure_3 import master_curve_binning as plotter
from shared.strategy_colors import STRATEGY_ORDER


PHASES = ("overall", "before_escape", "after_escape")
GATES = (
    ("ungated", None),
    ("gate_2e-05_0p0004375", (2.0e-5, 4.375e-4)),
)
ALLOWED_STRATEGIES = tuple(
    strategy for strategy in STRATEGY_ORDER if strategy not in {"Pulse", "KPP-Agent"}
)
CHANGE_TYPE = "linear_rel_per_day"
DEFAULT_X_EDGES = np.arange(16, dtype=float) * 0.2
DEFAULT_Y_RANGE = (-1.05, 100.0)
DEFAULT_Y_HIST_BINS = 65536
ZERO_FLOOR = 1.0e-10


@dataclass
class AggregationBundle:
    strategies: list[str]
    aggregators: dict[tuple[str, str], plotter.Aggregator]
    metadata: dict[str, object]


def create_aggregators(
    strategies: list[str],
    *,
    x_edges: np.ndarray = DEFAULT_X_EDGES,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    y_hist_bins: int = DEFAULT_Y_HIST_BINS,
) -> dict[tuple[str, str], plotter.Aggregator]:
    return {
        (gate_name, phase): plotter.Aggregator.create(
            n_strategies=len(strategies),
            x_edges=np.asarray(x_edges, dtype=float),
            ylim=y_range,
            n_y_bins=int(y_hist_bins),
        )
        for gate_name, _gate in GATES
        for phase in PHASES
    }


def add_linear_interval(
    aggregators: dict[tuple[str, str], plotter.Aggregator],
    strategy_index: int,
    previous_total: np.ndarray,
    previous_res: np.ndarray,
    next_res: np.ndarray,
    delta_days: float,
    active_phases: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    if not np.isfinite(delta_days) or delta_days <= 0:
        raise ValueError(f"Invalid interval length: {delta_days}")
    template = next(iter(aggregators.values()))
    x_edges = template.x_edges
    x_bin_size = float(x_edges[1] - x_edges[0])
    n_x_bins = template.n_x_bins

    x = previous_total.ravel().astype(np.float64, copy=False) * base.X_DISPLAY_SCALE
    previous = previous_res.ravel()
    nxt = next_res.ravel()
    if len({x.shape, previous.shape, nxt.shape}) != 1:
        raise ValueError("Density-array shape mismatch")
    valid_growth = (
        np.isfinite(previous)
        & np.isfinite(nxt)
        & (previous > ZERO_FLOOR)
    )
    y = np.full(previous.shape, np.nan, dtype=np.float32)
    y[valid_growth] = (
        (nxt[valid_growth] - previous[valid_growth])
        / previous[valid_growth]
        / np.float32(delta_days)
    )
    x_valid = np.isfinite(x) & (x >= x_edges[0]) & (x <= x_edges[-1])
    x_bins = np.zeros(x.shape, dtype=np.int64)
    finite_x = np.isfinite(x)
    x_bins[finite_x] = np.floor(
        (x[finite_x] - x_edges[0]) / x_bin_size
    ).astype(np.int64)
    np.clip(x_bins, 0, n_x_bins - 1, out=x_bins)
    common = x_valid & valid_growth & np.isfinite(y)

    tails: dict[str, tuple[int, int]] = {}
    for gate_name, gate in GATES:
        valid = common.copy()
        if gate is not None:
            valid &= (previous >= gate[0]) & (previous <= gate[1])
        values = y[valid].astype(np.float64, copy=False)
        tails[gate_name] = (
            int(np.count_nonzero(values < template.y_edges[0])),
            int(np.count_nonzero(values > template.y_edges[-1])),
        )
        valid_bins = x_bins[valid]
        for phase in active_phases:
            aggregators[(gate_name, phase)].add(
                strategy_index,
                valid_bins,
                values,
            )
    return tails


def save_bundle(path: Path, bundle: AggregationBundle) -> None:
    arrays: dict[str, np.ndarray] = {
        "strategies": np.asarray(bundle.strategies, dtype="U32"),
        "metadata_json": np.asarray([json.dumps(bundle.metadata, sort_keys=True)]),
    }
    template = next(iter(bundle.aggregators.values()))
    arrays["x_edges"] = template.x_edges
    arrays["y_edges"] = template.y_edges
    for gate_index, (gate_name, _gate) in enumerate(GATES):
        for phase_index, phase in enumerate(PHASES):
            prefix = f"g{gate_index}_p{phase_index}"
            aggregator = bundle.aggregators[(gate_name, phase)]
            arrays[f"{prefix}_count"] = aggregator.count
            arrays[f"{prefix}_total"] = aggregator.total
            arrays[f"{prefix}_total_sq"] = aggregator.total_sq
            arrays[f"{prefix}_histogram"] = aggregator.histogram
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_bundle(path: Path) -> AggregationBundle:
    with np.load(path, allow_pickle=False) as data:
        strategies = [str(value) for value in data["strategies"].tolist()]
        metadata = json.loads(str(data["metadata_json"][0]))
        x_edges = data["x_edges"].copy()
        y_edges = data["y_edges"].copy()
        aggregators = create_aggregators(
            strategies,
            x_edges=x_edges,
            y_range=(float(y_edges[0]), float(y_edges[-1])),
            y_hist_bins=len(y_edges) - 1,
        )
        for gate_index, (gate_name, _gate) in enumerate(GATES):
            for phase_index, phase in enumerate(PHASES):
                prefix = f"g{gate_index}_p{phase_index}"
                aggregator = aggregators[(gate_name, phase)]
                aggregator.count[:] = data[f"{prefix}_count"]
                aggregator.total[:] = data[f"{prefix}_total"]
                aggregator.total_sq[:] = data[f"{prefix}_total_sq"]
                aggregator.histogram[:] = data[f"{prefix}_histogram"]
    return AggregationBundle(strategies, aggregators, metadata)


def merge_bundles(paths: list[Path]) -> AggregationBundle:
    if not paths:
        raise ValueError("At least one cache path is required")
    loaded = load_bundle(paths[0])
    present = set(loaded.strategies)
    for path in paths[1:]:
        with np.load(path, allow_pickle=False) as data:
            present.update(str(value) for value in data["strategies"].tolist())
    strategies = [name for name in ALLOWED_STRATEGIES if name in present]
    template = next(iter(loaded.aggregators.values()))
    target = create_aggregators(
        strategies,
        x_edges=template.x_edges,
        y_range=(float(template.y_edges[0]), float(template.y_edges[-1])),
        y_hist_bins=template.n_y_bins,
    )
    metadata = {"source_caches": [str(path) for path in paths]}

    def merge_one(source: AggregationBundle) -> None:
        if not np.array_equal(template.x_edges, next(iter(source.aggregators.values())).x_edges):
            raise ValueError("Cannot merge caches with different X bins")
        if not np.array_equal(template.y_edges, next(iter(source.aggregators.values())).y_edges):
            raise ValueError("Cannot merge caches with different Y bins")
        for source_index, strategy in enumerate(source.strategies):
            if strategy not in strategies:
                continue
            target_index = strategies.index(strategy)
            for key, target_aggregator in target.items():
                source_aggregator = source.aggregators[key]
                target_aggregator.count[target_index] += source_aggregator.count[source_index]
                target_aggregator.total[target_index] += source_aggregator.total[source_index]
                target_aggregator.total_sq[target_index] += source_aggregator.total_sq[source_index]
                target_aggregator.histogram[target_index] += source_aggregator.histogram[source_index]

    merge_one(loaded)
    del loaded
    for path in paths[1:]:
        source = load_bundle(path)
        merge_one(source)
        del source
    return AggregationBundle(strategies, target, metadata)


def summarize_iqr(
    aggregator: plotter.Aggregator,
    strategies: list[str],
    *,
    gate_name: str,
    phase: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    x_centers = 0.5 * (aggregator.x_edges[:-1] + aggregator.x_edges[1:])
    for strategy_index, strategy in enumerate(strategies):
        for x_index, x_center in enumerate(x_centers):
            n = int(aggregator.count[strategy_index, x_index])
            if n == 0:
                continue
            total = float(aggregator.total[strategy_index, x_index])
            total_sq = float(aggregator.total_sq[strategy_index, x_index])
            mean = total / n
            variance = np.nan
            if n > 1:
                variance = max(0.0, (total_sq - total * total / n) / (n - 1))
            histogram = aggregator.histogram[strategy_index, x_index]
            q1 = plotter.histogram_quantile(histogram, aggregator.y_edges, 0.25)
            median = plotter.histogram_quantile(histogram, aggregator.y_edges, 0.50)
            q3 = plotter.histogram_quantile(histogram, aggregator.y_edges, 0.75)
            rows.append(
                {
                    "phase": phase,
                    "gate_mode": gate_name,
                    "change_type": CHANGE_TYPE,
                    "strategy": strategy,
                    "x_bin_left_cells_per_1000_um2": aggregator.x_edges[x_index],
                    "x_bin_right_cells_per_1000_um2": aggregator.x_edges[x_index + 1],
                    "x_bin_center_cells_per_1000_um2": x_center,
                    "n_pixels": n,
                    "mean": mean,
                    "sd": float(np.sqrt(variance)) if np.isfinite(variance) else np.nan,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )
    return pd.DataFrame(rows)


def pooled_aggregator(source: plotter.Aggregator) -> plotter.Aggregator:
    pooled = plotter.Aggregator.create(
        n_strategies=1,
        x_edges=source.x_edges,
        ylim=(float(source.y_edges[0]), float(source.y_edges[-1])),
        n_y_bins=source.n_y_bins,
    )
    pooled.count[0] = source.count.sum(axis=0)
    pooled.total[0] = source.total.sum(axis=0)
    pooled.total_sq[0] = source.total_sq.sum(axis=0)
    pooled.histogram[0] = source.histogram.sum(axis=0)
    return pooled


def compact_count(value: int) -> str:
    count = int(value)
    for scale, suffix in ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "k")):
        if count < scale:
            continue
        scaled = count / scale
        precision = 1 if scaled < 10 else 0
        return f"{scaled:.{precision}f}{suffix}"
    return str(count)
