#!/usr/bin/env python3
"""Aggregate and save reusable 160-um master-curve plotting tables."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import ndimage, stats
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = PROJECT_ROOT / "figures" / "Figure_3"
ANALYSIS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, FIGURE_DIR, ANALYSIS_DIR):
    if os.fspath(import_path) not in sys.path:
        sys.path.insert(0, os.fspath(import_path))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_simulations_coarsegrained as sim
from types import SimpleNamespace

@dataclass
class Aggregator:
    x_edges: np.ndarray
    y_edges: np.ndarray
    count: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray
    histogram: np.ndarray

    @classmethod
    def create(
        cls, *, n_strategies: int, x_edges: np.ndarray,
        ylim: tuple[float, float], n_y_bins: int,
    ) -> "Aggregator":
        n_x_bins = len(x_edges) - 1
        return cls(
            x_edges=x_edges,
            y_edges=np.linspace(float(ylim[0]), float(ylim[1]), n_y_bins + 1),
            count=np.zeros((n_strategies, n_x_bins), dtype=np.int64),
            total=np.zeros((n_strategies, n_x_bins), dtype=np.float64),
            total_sq=np.zeros((n_strategies, n_x_bins), dtype=np.float64),
            histogram=np.zeros((n_strategies, n_x_bins, n_y_bins), dtype=np.int64),
        )

    @property
    def n_x_bins(self) -> int:
        return len(self.x_edges) - 1

    @property
    def n_y_bins(self) -> int:
        return len(self.y_edges) - 1

    def add(self, strategy_index: int, x_bins: np.ndarray, y_values: np.ndarray) -> None:
        if y_values.size == 0:
            return
        self.count[strategy_index] += np.bincount(x_bins, minlength=self.n_x_bins)
        self.total[strategy_index] += np.bincount(x_bins, weights=y_values, minlength=self.n_x_bins)
        self.total_sq[strategy_index] += np.bincount(
            x_bins, weights=y_values * y_values, minlength=self.n_x_bins
        )
        y_min, y_max = self.y_edges[0], self.y_edges[-1]
        y_bins = np.floor((y_values - y_min) * self.n_y_bins / (y_max - y_min)).astype(np.int64)
        np.clip(y_bins, 0, self.n_y_bins - 1, out=y_bins)
        flat_bins = x_bins * self.n_y_bins + y_bins
        histogram = np.bincount(flat_bins, minlength=self.n_x_bins * self.n_y_bins)
        self.histogram[strategy_index] += histogram.reshape(self.n_x_bins, self.n_y_bins)



def histogram_quantile(counts: np.ndarray, edges: np.ndarray, quantile: float) -> float:
    n = int(counts.sum())
    if n == 0 or not np.isfinite(quantile):
        return np.nan
    q = float(np.clip(quantile, 0.0, 1.0))
    cumulative = np.cumsum(counts)
    target = q * max(n - 1, 0) + 1.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(counts) - 1)
    previous = int(cumulative[index - 1]) if index else 0
    within = int(counts[index])
    fraction = (
        0.5
        if within <= 0
        else float(np.clip((target - previous) / within, 0.0, 1.0))
    )
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


plotter = SimpleNamespace(Aggregator=Aggregator, histogram_quantile=histogram_quantile)
from analysis.figure_3 import density_common as base

# These aliases preserve the established internal call sites while the helpers
# are now grouped into the two explicitly internal Figure 3 modules above.
sweep = base
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import normalize_strategy
from shared.data_paths import COMBINED_RESULTS_CSV, SIMULATION_DATABASE, figure_data_dir, intermediate_dir


# =============================================================================
# USER SETTINGS -- edit these values, then run this file normally.
# =============================================================================
USER_PERIOD = "before_escape"       # before_escape | after_escape | all_timepoints
USER_GROWTH_MODE = os.environ.get("CORESEED_GROWTH_MODE", "relative")  # relative | absolute
USER_SIMULATION_COHORT = os.environ.get(
    "CORESEED_SIMULATION_COHORT", "escaping"
)  # escaping | all
USER_SIMULATION_INTERVAL_MODE = os.environ.get(
    "CORESEED_SIMULATION_INTERVAL_MODE", "daily"
)  # daily | experimental_schedule
USER_TARGET_POPULATION = os.environ.get(
    "CORESEED_TARGET_POPULATION", "resistant"
)  # resistant | sensitive
USER_SUPPORT_POLICY = os.environ.get("CORESEED_SUPPORT_POLICY", "both_endpoints")
# The master curve uses 0.2-wide local-density bins.  The KDE marginal uses a
# separate dedicated 0.05-bin cache (see plot_master_curve_sim_exp.py).
USER_X_BIN_WIDTH = float(os.environ.get("CORESEED_X_BIN_WIDTH", "0.10"))
USER_BIN_STATISTIC = "mean"         # median (with IQR) | mean (with SD)
USER_FOCUSED_GATE_MATRIX = True     # True: requested 8 gates + 3 ungated plots

# Experimental positions use the Figure 2 escape-analysis masks: susceptible
# cells must be inside the cleaned susceptible mask and resistant cells inside
# any resistant-clone mask. The corresponding density maps are generated by
# generate_figure2_mask_filtered_density_maps.py in this active Figure 3
# analysis directory.
# Y:-only Figure 2-mask-filtered density-map run.
USER_INVITRO_RUN = intermediate_dir(
    "experimental", "figure_3", "figure2_mask_filtered_mastercurve",
    "run_kernel_160_coarse_16_figure2_masks_all_clones", create=False,
)
USER_RESULTS_BASE = intermediate_dir("combined", "figure_3", "master_curve_160", create=False)
USER_SIMULATION_ROOT = SIMULATION_DATABASE
USER_SIMULATION_ESCAPE = COMBINED_RESULTS_CSV
USER_WORKERS = 6
USER_OVERWRITE_CACHE = True        # True forces aggregation to be recalculated
USER_SEN_ALSO = True              # Set True to build the optional NT-sensitive variants.
# PyCharm Play runs all four Figure 3 master-curve variants.  Keep the CLI
# switch as well, so command-line runs can explicitly request the same work.
USER_BUILD_ALL_VARIANTS = True

# Exclude isolated nonzero coarse-grid change pixels from the master-curve
# aggregation, matching the Figure 3 density-change-map display filter.
MIN_CHANGE_COMPONENT_PIXELS = 2


KERNEL_UM = 160
COARSENING_FACTOR = 16
GATE = (2.0e-5, 4.375e-4)
FOCUSED_GATE_MATRIX = bool(USER_FOCUSED_GATE_MATRIX)
FOCUSED_PERIOD = USER_PERIOD.strip().lower()
GROWTH_MODE = USER_GROWTH_MODE.strip().lower()
SIMULATION_COHORT = USER_SIMULATION_COHORT.strip().lower()
SIMULATION_INTERVAL_MODE = USER_SIMULATION_INTERVAL_MODE.strip().lower()
TARGET_POPULATION = USER_TARGET_POPULATION.strip().lower()
SUPPORT_POLICY = USER_SUPPORT_POLICY.strip().lower()
BIN_STATISTIC = USER_BIN_STATISTIC.strip().lower()

if FOCUSED_PERIOD not in {"before_escape", "after_escape", "all_timepoints"}:
    raise ValueError(f"Unsupported USER_PERIOD={FOCUSED_PERIOD!r}")
if SUPPORT_POLICY not in {"both_endpoints", "t0_positive"}:
    raise ValueError(f"Unsupported USER_SUPPORT_POLICY={SUPPORT_POLICY!r}")
if GROWTH_MODE not in {"relative", "absolute"}:
    raise ValueError(f"Unsupported USER_GROWTH_MODE={GROWTH_MODE!r}")
if SIMULATION_COHORT not in {"escaping", "all"}:
    raise ValueError(
        f"Unsupported USER_SIMULATION_COHORT={SIMULATION_COHORT!r}"
    )
if SIMULATION_INTERVAL_MODE not in {"daily", "experimental_schedule"}:
    raise ValueError(
        "Unsupported USER_SIMULATION_INTERVAL_MODE="
        f"{SIMULATION_INTERVAL_MODE!r}"
    )
if TARGET_POPULATION not in {"resistant", "sensitive"}:
    raise ValueError(
        f"Unsupported USER_TARGET_POPULATION={TARGET_POPULATION!r}"
    )
if BIN_STATISTIC not in {"median", "mean"}:
    raise ValueError(f"Unsupported USER_BIN_STATISTIC={BIN_STATISTIC!r}")
X_BIN_WIDTH = float(USER_X_BIN_WIDTH)
if not np.isfinite(X_BIN_WIDTH) or X_BIN_WIDTH <= 0:
    raise ValueError(f"Invalid USER_X_BIN_WIDTH={X_BIN_WIDTH!r}")
X_EDGES = np.arange(0.0, 3.0 + 0.5 * X_BIN_WIDTH, X_BIN_WIDTH)
Y_RANGE = (-2.01, 2.01) if GROWTH_MODE == "relative" else (-4.01, 4.01)
Y_HIST_BINS = 8192
MIN_BIN_PIXELS = 100
# Independent-unit inference for the Figure 3 mean-curve density contrast.
# The contrast is high-density mean minus low-density mean; a negative value
# therefore supports reduced resistant expansion at high local density.
TREND_CONTRAST_LOW_DENSITY = 0.45
TREND_CONTRAST_HIGH_DENSITY = 1.55
TREND_BOOTSTRAP_SAMPLES = 20_000
TREND_BOOTSTRAP_SEED = 20260828
STRATEGIES = ("CT", "NT", "AT50", "AT100", "LV-Agent", "AT+")
POOLED_STRATEGIES = ("CT", "AT50", "AT100")
SIMULATION_SOURCES = ("CT", "NT", "AT50", "AT100", "Agent", "Dens")
# Consecutive acquisition days in the experimental master-curve density maps.
# The optional simulation variant evaluates only these matching endpoints, e.g.
# d1-to-d3, instead of separately using d1-to-d2 and d2-to-d3.
EXPERIMENTAL_INTERVAL_DAY_PAIRS = tuple(zip(
    (1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38, 40, 43, 45, 47,
     50, 52, 54, 57, 59, 61, 64, 66, 68, 71, 73, 75, 78, 80, 82, 85, 87, 89, 92, 94),
    (3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38, 40, 43, 45, 47, 50,
     52, 54, 57, 59, 61, 64, 66, 68, 71, 73, 75, 78, 80, 82, 85, 87, 89, 92, 94, 96),
))

# Older per-run simulation caches use presentation labels from
# simulation_density_utils.py.  Normalize those alongside raw experimental
# and simulation names before combining data into the Figure 3 categories.
_LEGACY_STRATEGY_LABELS = {
    "ATagent": "LV-Agent",
    r"AT$^{+}$": "AT+",
}


def canonical_strategy_label(strategy: str) -> str:
    """Return the Figure 3 strategy label, including legacy cache aliases."""
    label = _LEGACY_STRATEGY_LABELS.get(str(strategy).strip(), str(strategy).strip())
    canonical = normalize_strategy(label)
    return "AT+" if canonical == "AT100+" else canonical

DEFAULT_INVITRO_RUN = USER_INVITRO_RUN
DEFAULT_CONFIG = (
    PROJECT_ROOT / "analysis" / "figure_3"
    / "density_maps_config_160_figure2_masks_all_clones.yaml"
)
DEFAULT_EXPERIMENT_ESCAPE = intermediate_dir("experimental", "figure_2", create=False) / "frontcontact_summary_ALL.csv"
DEFAULT_ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
DEFAULT_SIM_ROOT = USER_SIMULATION_ROOT
DEFAULT_SIM_ESCAPE = USER_SIMULATION_ESCAPE
_mode_token = "absolute_" if GROWTH_MODE == "absolute" else ""
_support_token = (
    "average_endpoints" if GROWTH_MODE == "absolute"
    else ("all_both_endpoints" if SUPPORT_POLICY == "both_endpoints" else "t0_positive")
)
_x_bin_token = f"{X_BIN_WIDTH:g}".replace(".", "p")
_population_token = "" if TARGET_POPULATION == "resistant" else "_sensitive"
_cohort_token = "" if SIMULATION_COHORT == "escaping" else "_all_runs"
_variant_token = _population_token + _cohort_token
_interval_token = (
    "" if SIMULATION_INTERVAL_MODE == "daily" else "_experimental_schedule"
)
DEFAULT_CACHE_ROOT = USER_RESULTS_BASE / (
    f"aggregation_cache_figure2_masks_{_mode_token}{FOCUSED_PERIOD}_"
    f"{_support_token}_xbin_{_x_bin_token}{_variant_token}{_interval_token}"
)
_data_prefix = "plots_absolute" if GROWTH_MODE == "absolute" else "plots"
_output_suffix = "gate_matrix" if SUPPORT_POLICY == "both_endpoints" else "t0_positive"
DEFAULT_OUTPUT_ROOT = figure_data_dir(
    "Figure_3", "master_curve_160",
    f"{_data_prefix}_{FOCUSED_PERIOD}_{_output_suffix}{_variant_token}{_interval_token}",
    create=False,
)


@dataclass(frozen=True)
class Condition:
    cutoff: str
    support_mode: str
    threshold: float
    gate_lower: float | None = GATE[0]
    gate_upper: float | None = GATE[1]

    @property
    def threshold_token(self) -> str:
        return f"{self.threshold:.0e}"

    @property
    def slug(self) -> str:
        if FOCUSED_GATE_MATRIX and self.gate_lower is not None and self.gate_upper is not None:
            return f"{self.cutoff}__gate_{self.gate_lower:.6g}_to_{self.gate_upper:.6g}".replace("+", "")
        return (
            f"{self.cutoff}__{self.support_mode}_gt_{self.threshold_token}"
            .replace("+", "")
        )

    @property
    def output_relative(self) -> Path:
        if FOCUSED_GATE_MATRIX and self.gate_lower is not None and self.gate_upper is not None:
            lower = f"{self.gate_lower:.6g}".replace(".", "p")
            upper = f"{self.gate_upper:.6g}".replace(".", "p")
            return Path("gated") / f"gate_{lower}_to_{upper}"
        if FOCUSED_GATE_MATRIX:
            support = {
                "average": "average_t0_t1", "both": "both_t0_t1", "t0": "t0"
            }[self.support_mode]
            return Path("no_gating") / f"{support}_gt_{self.threshold_token}"
        cutoff_folder = {
            "three_days_before_escape": "cutoff_3_days_before_escape",
            "three_timepoints_before_escape": "cutoff_3_timepoints_before_escape",
        }[self.cutoff]
        support_folder = (
            f"average_t0_t1_gt_{self.threshold_token}"
            if self.support_mode == "average"
            else f"both_t0_t1_gt_{self.threshold_token}"
        )
        return Path(cutoff_folder) / support_folder


CONDITIONS = tuple(
    Condition(cutoff, support_mode, threshold)
    for cutoff in ("three_days_before_escape", "three_timepoints_before_escape")
    for support_mode in ("average", "both")
    for threshold in (1.0e-10, 1.0e-8, 2.0e-5)
)
if FOCUSED_GATE_MATRIX and GROWTH_MODE == "absolute":
    CONDITIONS = tuple(
        Condition(FOCUSED_PERIOD, "average", 0.0, lower, upper)
        for lower in (2.0e-5, 1.0e-4, 1.0e-3)
        for upper in (4.375e-4, 1.1e-3, 5.0e-3)
        if lower <= upper
    ) + tuple(
        Condition(FOCUSED_PERIOD, "average", threshold, None, None)
        for threshold in (1.0e-10, 1.0e-8, 2.0e-5)
    )
elif FOCUSED_GATE_MATRIX and SUPPORT_POLICY == "t0_positive":
    CONDITIONS = (Condition(FOCUSED_PERIOD, "t0", 0.0, None, None),)
elif FOCUSED_GATE_MATRIX:
    CONDITIONS = tuple(
        Condition(FOCUSED_PERIOD, "both", 0.0, lower, upper)
        for lower in (2.0e-5, 1.0e-4, 1.0e-3)
        for upper in (4.375e-4, 1.1e-3, 5.0e-3)
        if lower <= upper
    ) + tuple(
        Condition(FOCUSED_PERIOD, "both", threshold, None, None)
        for threshold in (1.0e-10, 1.0e-8, 2.0e-5)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-vitro-run", type=Path, default=DEFAULT_INVITRO_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experimental-escape", type=Path, default=DEFAULT_EXPERIMENT_ESCAPE)
    parser.add_argument(
        "--escape-margin-csv", type=Path, default=DEFAULT_ESCAPE_MARGIN_CSV,
        help=(
            "Canonical output of Escape_Margin_Analysis.py. When present, it "
            "defines escape and last-confined days for both experimental and "
            "simulation intervals."
        ),
    )
    parser.add_argument("--simulation-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--simulation-escape", type=Path, default=DEFAULT_SIM_ESCAPE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=USER_WORKERS)
    parser.add_argument("--overwrite-cache", action="store_true", default=USER_OVERWRITE_CACHE)
    parser.add_argument(
        "--all-variants", action="store_true",
        help="Build relative/absolute data for escaping-only and all-run cohorts.",
    )
    parser.add_argument("--single-variant", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def create_aggregators(n_strategies: int) -> dict[str, plotter.Aggregator]:
    return {
        condition.slug: plotter.Aggregator.create(
            n_strategies=n_strategies,
            x_edges=X_EDGES,
            ylim=Y_RANGE,
            n_y_bins=Y_HIST_BINS,
        )
        for condition in CONDITIONS
    }


def non_isolated_change_pixels(change: np.ndarray) -> np.ndarray:
    """Return a mask excluding isolated nonzero change-map pixels."""
    valid = np.isfinite(change) & ~np.isclose(change, 0)
    labels, count = ndimage.label(valid)
    if count == 0:
        return np.ones(change.shape, dtype=bool)
    component_sizes = np.bincount(labels.ravel())
    isolated = (
        (labels > 0)
        & (component_sizes[labels] < MIN_CHANGE_COMPONENT_PIXELS)
    )
    return ~isolated


def add_interval(
    aggregators: dict[str, plotter.Aggregator],
    strategy_index: int,
    total_t0: np.ndarray,
    resistant_t0: np.ndarray,
    total_t1: np.ndarray,
    resistant_t1: np.ndarray,
    delta_days: float,
    active_cutoffs: set[str],
) -> None:
    if not np.isfinite(delta_days) or delta_days <= 0:
        raise ValueError(f"Invalid interval length: {delta_days}")
    if len({total_t0.shape, resistant_t0.shape, total_t1.shape, resistant_t1.shape}) != 1:
        raise ValueError("Density-array shape mismatch")

    total_average = 0.5 * (
        total_t0.astype(np.float64, copy=False)
        + total_t1.astype(np.float64, copy=False)
    )
    resistant_average = 0.5 * (
        resistant_t0.astype(np.float64, copy=False)
        + resistant_t1.astype(np.float64, copy=False)
    )
    x = total_average.ravel() * base.X_DISPLAY_SCALE
    average_res = resistant_average.ravel()
    res_t0 = resistant_t0.ravel().astype(np.float64, copy=False)
    res_t1 = resistant_t1.ravel().astype(np.float64, copy=False)
    y = np.full(average_res.shape, np.nan, dtype=np.float64)
    if GROWTH_MODE == "relative":
        nonzero_average = np.isfinite(average_res) & (average_res > 0)
        y[nonzero_average] = (
            (res_t1[nonzero_average] - res_t0[nonzero_average])
            / average_res[nonzero_average]
            / float(delta_days)
        )
    else:
        finite_endpoints = np.isfinite(res_t0) & np.isfinite(res_t1)
        y[finite_endpoints] = (
            (res_t1[finite_endpoints] - res_t0[finite_endpoints])
            / float(delta_days)
            * base.X_DISPLAY_SCALE
        )

    x_bins = np.zeros(x.shape, dtype=np.int64)
    finite_x = np.isfinite(x)
    x_bins[finite_x] = np.floor(x[finite_x] / X_BIN_WIDTH).astype(np.int64)
    np.clip(x_bins, 0, len(X_EDGES) - 2, out=x_bins)
    common_valid = (
        finite_x
        & (x >= X_EDGES[0])
        & (x <= X_EDGES[-1])
        & np.isfinite(y)
        & np.isfinite(res_t0)
        & np.isfinite(res_t1)
    )
    common_valid &= non_isolated_change_pixels(y.reshape(total_t0.shape)).ravel()

    for condition in CONDITIONS:
        if condition.cutoff not in active_cutoffs:
            continue
        if FOCUSED_GATE_MATRIX and condition.gate_lower is not None and condition.gate_upper is not None:
            if condition.support_mode == "average":
                support = (
                    (average_res >= condition.gate_lower)
                    & (average_res <= condition.gate_upper)
                )
            else:
                support = (
                    (res_t0 >= condition.gate_lower)
                    & (res_t0 <= condition.gate_upper)
                    & (res_t1 >= condition.gate_lower)
                    & (res_t1 <= condition.gate_upper)
                )
        elif condition.support_mode == "average":
            support = average_res >= condition.threshold
        elif condition.support_mode == "t0":
            support = res_t0 > condition.threshold
        else:
            support = (res_t0 >= condition.threshold) & (res_t1 >= condition.threshold)
        if not FOCUSED_GATE_MATRIX:
            support &= (average_res >= GATE[0]) & (average_res <= GATE[1])
        valid = common_valid & support
        aggregators[condition.slug].add(
            strategy_index,
            x_bins[valid],
            y[valid],
        )


def target_density(total: np.ndarray, resistant: np.ndarray) -> np.ndarray:
    """Return the density field whose change is being aggregated."""
    if TARGET_POPULATION == "resistant":
        return resistant
    return np.maximum(total - resistant, 0.0)


def timepoints_from_maps(maps_dir: Path, well: str) -> list[tuple[str, float]]:
    pattern = re.compile(
        rf"^{re.escape(well)}_(?P<tp>[^_]+)_total_density_epanechnikov\.tif$"
    )
    result: list[tuple[str, float]] = []
    for path in maps_dir.glob("*_total_density_epanechnikov.tif"):
        match = pattern.match(path.name)
        if match is None:
            continue
        token = match.group("tp")
        day = base.timepoint_to_day(token)
        if np.isfinite(day):
            result.append((token, float(day)))
    return sorted(result, key=lambda item: item[1])


def cutoff_limits(
    timepoint_days: list[float],
    escape_day: float,
    confined_day: float,
    sample_cutoff: float | None | object,
) -> dict[str, float]:
    if FOCUSED_GATE_MATRIX and FOCUSED_PERIOD in {"after_escape", "all_timepoints"}:
        if not np.isfinite(escape_day) or sample_cutoff is None:
            return {}
        upper = np.inf if sample_cutoff is ... else float(sample_cutoff)
        return {FOCUSED_PERIOD: upper}
    if not np.isfinite(escape_day) or not np.isfinite(confined_day):
        return {}
    if sample_cutoff is None:
        return {}
    upper_confined = float(confined_day)
    if sample_cutoff is not ...:
        upper_confined = min(upper_confined, float(sample_cutoff))

    three_day_limit = min(upper_confined, float(escape_day) - 3.0)
    confined_timepoints = [day for day in timepoint_days if day <= upper_confined]
    if len(confined_timepoints) < 4:
        three_timepoint_limit = -np.inf
    else:
        three_timepoint_limit = float(confined_timepoints[-4])
    return {
        "three_days_before_escape": three_day_limit,
        "three_timepoints_before_escape": three_timepoint_limit,
        "before_escape": upper_confined,
    }


def _margin_rows(path: Path, dataset_type: str) -> pd.DataFrame:
    """Return canonical escape-margin rows for one dataset, if available."""
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path)
    required = {"dataset_type", "plate", "sample", "day", "signed_escape_margin"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing escape-margin columns: {sorted(missing)}")
    rows = table.loc[table["dataset_type"].astype(str) == dataset_type].copy()
    rows["day"] = pd.to_numeric(rows["day"], errors="coerce")
    rows["signed_escape_margin"] = pd.to_numeric(rows["signed_escape_margin"], errors="coerce")
    return rows[np.isfinite(rows["day"]) & np.isfinite(rows["signed_escape_margin"])].copy()


def experimental_escape_cutoffs(legacy_path: Path, margin_path: Path) -> pd.DataFrame:
    """Prefer canonical margin-derived escape limits; retain legacy fallback."""
    rows = _margin_rows(margin_path, "experimental")
    if rows.empty:
        return base.load_escape_cutoffs(legacy_path)
    records: list[dict[str, object]] = []
    for (plate, well), group in rows.groupby(["plate", "sample"], sort=False):
        escaped = group.loc[group["signed_escape_margin"] > 0, "day"]
        if escaped.empty:
            continue
        escape_day = float(escaped.min())
        confined = group.loc[group["day"] < escape_day, "day"]
        if confined.empty:
            continue
        records.append({
            "plate": str(plate), "well": str(well),
            "escape_day": escape_day,
            "pre_escape_cutoff_day": float(confined.max()),
        })
    if not records:
        raise RuntimeError(f"No escaped experimental replicates found in {margin_path}")
    return pd.DataFrame(records)


def _margin_run_id(sample: object) -> str:
    """Normalize ``run_18`` and Figure-2-style ``R18_*`` identifiers alike."""
    text = str(sample).strip()
    match = re.match(r"(?:run[_-]?)?0*(\d+)$", text, flags=re.IGNORECASE)
    if match is None:
        match = re.match(r"R0*(\d+)(?:_|$)", text, flags=re.IGNORECASE)
    return str(int(match.group(1))) if match is not None else base.normalize_simulation_run_id(text)


def simulation_escape_days(legacy_path: Path, margin_path: Path) -> dict[tuple[str, str], float]:
    """Prefer canonical simulation margins; retain legacy TTE fallback."""
    rows = _margin_rows(margin_path, "simulation")
    if rows.empty:
        return base.load_positive_tte(legacy_path)
    result: dict[tuple[str, str], float] = {}
    for (strategy, sample), group in rows.groupby(["strategy", "sample"], sort=False):
        escaped = group.loc[group["signed_escape_margin"] > 0, "day"]
        if not escaped.empty:
            result[(canonical_strategy_label(strategy), _margin_run_id(sample))] = float(escaped.min())
    if not result:
        raise RuntimeError(f"No escaped simulation runs found in {margin_path}")
    return result


def extract_in_vitro(
    run_dir: Path,
    config_path: Path,
    escape_path: Path,
    escape_margin_path: Path = DEFAULT_ESCAPE_MARGIN_CSV,
    well_cache_root: Path | None = None,
) -> tuple[list[str], dict[str, plotter.Aggregator], dict[str, object]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cutoffs = experimental_escape_cutoffs(escape_path, escape_margin_path)
    cutoff_lookup = {
        (str(row.plate), str(row.well)): (
            float(row.escape_day),
            float(row.pre_escape_cutoff_day),
        )
        for row in cutoffs.itertuples(index=False)
    }
    strategies = list(STRATEGIES)
    strategy_index = {strategy: index for index, strategy in enumerate(strategies)}
    aggregators = create_aggregators(len(strategies))
    scanned = 0
    used = {condition.slug: 0 for condition in CONDITIONS}

    for changes_dir in sorted(run_dir.glob("Plate_*/*/density_changes")):
        plate = changes_dir.parent.parent.name
        well = changes_dir.parent.name
        strategy = canonical_strategy_label(base.strategy_from_plate_and_well(plate, well))
        if strategy not in strategy_index:
            continue
        cutoff_pair = cutoff_lookup.get((plate, well))
        if cutoff_pair is None:
            continue
        maps_dir = changes_dir.parent / "density_maps"
        timepoints = timepoints_from_maps(maps_dir, well)
        sample_cutoff: float | None | object = ...
        if well in EXCLUDE_SAMPLES:
            configured = EXCLUDE_SAMPLES[well]
            sample_cutoff = None if configured is None else float(configured)
        limits = cutoff_limits(
            [day for _token, day in timepoints],
            cutoff_pair[0],
            cutoff_pair[1],
            sample_cutoff,
        )
        if not limits:
            continue

        # Preserve each well as an independent unit for the Figure 3
        # replicate-level contrast analysis.  The aggregate cache below is
        # retained unchanged for the plotting workflow.
        well_aggregators = create_aggregators(1)
        well_intervals = 0

        for rel_path in sorted(changes_dir.glob(base.CHANGE_SPECS["rel"]["glob"])):
            match = base.REL_CHANGE_RE.match(rel_path.name)
            if match is None:
                continue
            first_tp = match.group("first_tp")
            next_tp = match.group("next_tp")
            first_day = base.timepoint_to_day(first_tp)
            next_day = base.timepoint_to_day(next_tp)
            active = {
                cutoff_name
                for cutoff_name, limit in limits.items()
                if np.isfinite(next_day)
                and next_day <= limit
                and (cutoff_name != "after_escape" or first_day >= cutoff_pair[0])
            }
            if not active:
                continue
            paths = (
                base.density_map_path(maps_dir, well, first_tp),
                base.resistant_density_map_path(maps_dir, well, first_tp),
                base.density_map_path(maps_dir, well, next_tp),
                base.resistant_density_map_path(maps_dir, well, next_tp),
            )
            if not all(path.is_file() for path in paths):
                print(f"[WARN] Missing density pair for {plate}/{well} {first_tp}->{next_tp}")
                continue
            arrays = [tiff.imread(path).astype(np.float32, copy=False) for path in paths]
            target_t0 = target_density(arrays[0], arrays[1])
            target_t1 = target_density(arrays[2], arrays[3])
            add_interval(
                aggregators,
                strategy_index[strategy],
                arrays[0],
                target_t0,
                arrays[2],
                target_t1,
                float(next_day - first_day),
                active,
            )
            add_interval(
                well_aggregators,
                0,
                arrays[0],
                target_t0,
                arrays[2],
                target_t1,
                float(next_day - first_day),
                active,
            )
            scanned += 1
            well_intervals += 1
            for condition in CONDITIONS:
                if condition.cutoff in active:
                    used[condition.slug] += 1
            if scanned % 200 == 0:
                print(f"[in vitro] scanned {scanned} intervals", flush=True)

        if well_cache_root is not None and well_intervals:
            save_bundle(
                well_cache_root / plate / f"{well}.npz",
                [strategy],
                well_aggregators,
                {
                    "dataset": "in_vitro_well",
                    "plate": plate,
                    "well": well,
                    "strategy": strategy,
                    "intervals": well_intervals,
                },
            )

    metadata = {
        "dataset": "in_vitro",
        "target_population": TARGET_POPULATION,
        "run_dir": os.fspath(run_dir),
        "config": os.fspath(config_path),
        "input_filter": "figure2_clean_sensitive_and_all_clone_resistant",
        "input_filter_detail": (
            "susceptible positions inside sensitive_clean_masks_stacks; "
            "resistant positions inside clone_masks_stacks with label >= 2"
        ),
        "escape_source": os.fspath(escape_path),
        "escape_margin_source": os.fspath(escape_margin_path) if escape_margin_path.is_file() else "legacy fallback",
        "kernel_um": KERNEL_UM,
        "coarsening_factor": COARSENING_FACTOR,
        "intervals_scanned": scanned,
        "condition_interval_counts": used,
    }
    return strategies, aggregators, metadata


def simulation_density(
    positions_um: np.ndarray,
    type_values: np.ndarray,
    canvas: sim.CanvasSpec,
) -> tuple[np.ndarray, np.ndarray]:
    susceptible, resistant = sim.positions_to_density_inputs(
        positions_um,
        type_values,
        canvas,
    )
    return sweep.resistant_and_total_density(
        susceptible,
        resistant,
        radius_um=float(KERNEL_UM),
        canvas=canvas,
        coarsening_factor=COARSENING_FACTOR,
    )


def save_bundle(
    path: Path,
    strategies: list[str],
    aggregators: dict[str, plotter.Aggregator],
    metadata: dict[str, object],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "strategies": np.asarray(strategies, dtype="U32"),
        "condition_slugs": np.asarray(
            [condition.slug for condition in CONDITIONS], dtype="U128"
        ),
        "metadata_json": np.asarray([json.dumps(metadata, sort_keys=True)]),
        "x_edges": X_EDGES,
        "y_edges": next(iter(aggregators.values())).y_edges,
    }
    for index, condition in enumerate(CONDITIONS):
        aggregator = aggregators[condition.slug]
        prefix = f"c{index}"
        arrays[f"{prefix}_count"] = aggregator.count
        arrays[f"{prefix}_total"] = aggregator.total
        arrays[f"{prefix}_total_sq"] = aggregator.total_sq
        arrays[f"{prefix}_histogram"] = aggregator.histogram
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_bundle(
    path: Path,
) -> tuple[list[str], dict[str, plotter.Aggregator], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        strategies = [canonical_strategy_label(str(value)) for value in data["strategies"].tolist()]
        metadata = json.loads(str(data["metadata_json"][0]))
        x_edges = data["x_edges"].copy()
        y_edges = data["y_edges"].copy()
        aggregators = {
            condition.slug: plotter.Aggregator.create(
                n_strategies=len(strategies),
                x_edges=x_edges,
                ylim=(float(y_edges[0]), float(y_edges[-1])),
                n_y_bins=len(y_edges) - 1,
            )
            for condition in CONDITIONS
        }
        for index, condition in enumerate(CONDITIONS):
            aggregator = aggregators[condition.slug]
            prefix = f"c{index}"
            aggregator.count[:] = data[f"{prefix}_count"]
            aggregator.total[:] = data[f"{prefix}_total"]
            aggregator.total_sq[:] = data[f"{prefix}_total_sq"]
            aggregator.histogram[:] = data[f"{prefix}_histogram"]
    return strategies, aggregators, metadata


def _mean_at_density(
    aggregator: plotter.Aggregator,
    density: float,
) -> tuple[float, int] | None:
    """Return a one-unit mean at a plotted density bin, if sufficiently sampled."""
    centers = 0.5 * (aggregator.x_edges[:-1] + aggregator.x_edges[1:])
    matches = np.flatnonzero(np.isclose(centers, density))
    if matches.size != 1:
        raise RuntimeError(f"No unique x bin at density={density}")
    index = int(matches[0])
    n_pixels = int(aggregator.count[0, index])
    if n_pixels < MIN_BIN_PIXELS:
        return None
    return float(aggregator.total[0, index] / n_pixels), n_pixels


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust finite p-values while preserving their input order."""
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    ordered = finite[np.argsort(p_values[finite])]
    running_max = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        value = min(1.0, (total - rank) * float(p_values[index]))
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def write_mean_curve_contrast_statistics(
    cache_root: Path,
    output_dir: Path,
    condition: Condition,
) -> Path:
    """Write well/run-level mean-curve contrasts and one-sided test results.

    Each experimental well and simulation run is one independent observation.
    Only units with at least ``MIN_BIN_PIXELS`` in both prespecified density
    bins contribute.  The one-sided Wilcoxon signed-rank test evaluates whether
    the median high-minus-low contrast is below zero; confidence intervals are
    percentile bootstrap intervals for the mean contrast across independent
    units.  Holm correction covers CT, AT50, and AT100 in each dataset.
    """
    cache_sets = {
        "in_vitro": cache_root / "in_vitro_wells",
        "in_silico": cache_root / "in_silico_runs",
    }
    records: list[dict[str, object]] = []

    for dataset, root in cache_sets.items():
        paths = sorted(root.rglob("*.npz")) if root.is_dir() else []
        if not paths:
            raise FileNotFoundError(
                f"No independent-unit caches found for {dataset}: {root}. "
                "Re-run this aggregation with --overwrite-cache."
            )

        values_by_strategy: dict[str, list[float]] = {
            strategy: [] for strategy in POOLED_STRATEGIES
        }
        for path in paths:
            strategies, aggregators, metadata = load_bundle(path)
            strategy = canonical_strategy_label(metadata.get("strategy", strategies[0]))
            if strategy not in values_by_strategy:
                continue
            aggregator = aggregators[condition.slug]
            low_result = _mean_at_density(aggregator, TREND_CONTRAST_LOW_DENSITY)
            high_result = _mean_at_density(aggregator, TREND_CONTRAST_HIGH_DENSITY)
            if low_result is None or high_result is None:
                continue
            values_by_strategy[strategy].append(high_result[0] - low_result[0])

        for strategy, values in values_by_strategy.items():
            contrasts = np.asarray(values, dtype=float)
            n_units = len(contrasts)
            if n_units == 0:
                raw_p = np.nan
                mean_contrast = ci_low = ci_high = np.nan
            else:
                mean_contrast = float(contrasts.mean())
                rng = np.random.default_rng(
                    TREND_BOOTSTRAP_SEED
                    + sum(ord(character) for character in f"{dataset}:{strategy}")
                )
                samples = rng.integers(
                    0,
                    n_units,
                    size=(TREND_BOOTSTRAP_SAMPLES, n_units),
                )
                bootstrap_means = contrasts[samples].mean(axis=1)
                ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
                raw_p = (
                    1.0
                    if np.allclose(contrasts, 0.0)
                    else float(stats.wilcoxon(contrasts, alternative="less").pvalue)
                )

            records.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "low_density_cells_per_1000_um2": TREND_CONTRAST_LOW_DENSITY,
                    "high_density_cells_per_1000_um2": TREND_CONTRAST_HIGH_DENSITY,
                    "contrast_definition": "mean(high_density)-mean(low_density)",
                    "n_independent_units": n_units,
                    "mean_contrast_day_minus_1": mean_contrast,
                    "bootstrap_ci_95_low_day_minus_1": ci_low,
                    "bootstrap_ci_95_high_day_minus_1": ci_high,
                    "wilcoxon_one_sided_p": raw_p,
                }
            )

    result = pd.DataFrame(records)
    result["wilcoxon_one_sided_p_holm"] = np.nan
    for dataset, indices in result.groupby("dataset").groups.items():
        result.loc[indices, "wilcoxon_one_sided_p_holm"] = _holm_adjust(
            result.loc[indices, "wilcoxon_one_sided_p"].to_numpy(float)
        )
    output_path = output_dir / "mean_curve_contrast_statistics.tsv"
    result.to_csv(output_path, sep="\t", index=False)
    return output_path


def process_simulation_run(
    source: str,
    run_name: str,
    escape_day: float | None,
    input_root: str,
    cache_path: str,
) -> str:
    display_strategy = canonical_strategy_label(sweep.SOURCE_TO_DISPLAY[source])
    sim_args = sweep.simulation_args(input_root, source, run_name)
    plans = sim.discover_run_plans(sim_args)
    if len(plans) != 1:
        raise RuntimeError(f"Expected one plan for {source}/{run_name}, found {len(plans)}")
    plan = plans[0]
    loaded = [sim.load_filtered_cells(item.path, sim_args) for item in plan.files]
    snapshots = list(zip(plan.files, loaded))
    canvas = sweep.canvas_from_loaded(loaded, float(KERNEL_UM))
    aggregators = create_aggregators(1)
    if escape_day is None:
        eligible_days = [float(item.day) for item in plan.files]
        if not eligible_days:
            raise RuntimeError(f"No simulation snapshots found for {source}/{run_name}")
        limits = {FOCUSED_PERIOD: max(eligible_days)}
    else:
        eligible_days = [
            float(item.day) for item in plan.files
            if float(item.day) <= escape_day
        ]
        confined_days = [day for day in eligible_days if day < escape_day]
        # Match the experimental rule: the first escape snapshot is not a
        # confined endpoint.  Thus no interval may end at, or start from, the
        # escape-day snapshot in a before-escape master curve.
        if FOCUSED_PERIOD == "before_escape" and not confined_days:
            limits = {FOCUSED_PERIOD: -np.inf}
        else:
            confined_day = max(confined_days) if confined_days else float(escape_day)
            limits = cutoff_limits(eligible_days, escape_day, confined_day, ...)
    max_limit = max(limits.values())

    if SIMULATION_INTERVAL_MODE == "experimental_schedule":
        def snapshot_at(day: int) -> tuple[object, tuple[np.ndarray, np.ndarray]]:
            matches = [
                snapshot for snapshot in snapshots
                if np.isclose(float(snapshot[0].day), float(day))
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"{source}/{run_name} has no unique simulation snapshot at day {day}, "
                    "required for the experimental interval schedule"
                )
            return matches[0]

        interval_pairs = [
            (snapshot_at(start_day), snapshot_at(end_day))
            for start_day, end_day in EXPERIMENTAL_INTERVAL_DAY_PAIRS
            if end_day <= max_limit
        ]
    else:
        interval_pairs = list(zip(snapshots, snapshots[1:]))

    intervals = {condition.slug: 0 for condition in CONDITIONS}
    for ((previous_file, (previous_positions, previous_types)),
         (current_file, (positions_um, type_values))) in interval_pairs:
        previous_day = float(previous_file.day)
        current_day = float(current_file.day)
        if current_day > max_limit:
            continue
        previous_resistant, previous_total = simulation_density(
            previous_positions, previous_types, canvas,
        )
        resistant, total = simulation_density(positions_um, type_values, canvas)
        previous_target = target_density(previous_total, previous_resistant)
        target = target_density(total, resistant)
        active = {
            cutoff_name
            for cutoff_name, limit in limits.items()
            if current_day <= limit
            and (
                cutoff_name != "after_escape"
                or (escape_day is not None and previous_day >= escape_day)
            )
        }
        if not active:
            continue
        add_interval(
            aggregators,
            0,
            previous_total,
            previous_target,
            total,
            target,
            current_day - previous_day,
            active,
        )
        for condition in CONDITIONS:
            if condition.cutoff in active:
                intervals[condition.slug] += 1

    save_bundle(
        Path(cache_path),
        [display_strategy],
        aggregators,
        {
            "dataset": "in_silico_run",
            "source": source,
            "strategy": display_strategy,
            "run": run_name,
            "escape_day": escape_day,
            "escaped": escape_day is not None,
            "target_population": TARGET_POPULATION,
            "simulation_interval_mode": SIMULATION_INTERVAL_MODE,
            "intervals": intervals,
        },
    )
    return cache_path
def merge_bundles(
    paths: list[Path],
) -> tuple[list[str], dict[str, plotter.Aggregator], dict[str, object]]:
    target_strategies = list(STRATEGIES)
    target = create_aggregators(len(target_strategies))
    for path in paths:
        strategies, source, _metadata = load_bundle(path)
        for source_index, strategy in enumerate(strategies):
            canonical = canonical_strategy_label(strategy)
            if canonical not in target_strategies:
                raise ValueError(
                    f"Unsupported strategy label {strategy!r} in cache {path}; "
                    f"normalized to {canonical!r}"
                )
            target_index = target_strategies.index(canonical)
            for condition in CONDITIONS:
                destination = target[condition.slug]
                origin = source[condition.slug]
                destination.count[target_index] += origin.count[source_index]
                destination.total[target_index] += origin.total[source_index]
                destination.total_sq[target_index] += origin.total_sq[source_index]
                destination.histogram[target_index] += origin.histogram[source_index]
    return target_strategies, target, {"source_caches": [os.fspath(path) for path in paths]}


def aggregate_simulations(
    input_root: Path,
    escape_path: Path,
    escape_margin_path: Path,
    cache_root: Path,
    workers: int,
    overwrite: bool,
) -> tuple[list[str], dict[str, plotter.Aggregator], dict[str, object]]:
    tte = simulation_escape_days(escape_path, escape_margin_path)
    tasks: list[tuple[str, str, float | None, Path]] = []
    for source in SIMULATION_SOURCES:
        plans = sim.discover_run_plans(sweep.simulation_args(os.fspath(input_root), source))
        for plan in plans:
            key = (
                canonical_strategy_label(plan.strategy),
                base.normalize_simulation_run_id(plan.run_id),
            )
            escape_day = tte.get(key)
            if escape_day is None and SIMULATION_COHORT == "escaping":
                continue
            cache_path = cache_root / "in_silico_runs" / source / f"{plan.run_id}.npz"
            tasks.append((
                source,
                plan.run_id,
                float(escape_day) if escape_day is not None else None,
                cache_path,
            ))
    if not tasks:
        raise RuntimeError("No simulation runs matched the selected cohort")

    pending = [task for task in tasks if overwrite or not task[3].is_file()]
    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1, workers), len(pending))) as pool:
            futures = {
                pool.submit(
                    process_simulation_run,
                    source,
                    run_name,
                    escape_day,
                    os.fspath(input_root),
                    os.fspath(cache_path),
                ): (source, run_name)
                for source, run_name, escape_day, cache_path in pending
            }
            completed = 0
            for future in as_completed(futures):
                source, run_name = futures[future]
                future.result()
                completed += 1
                print(
                    f"[in silico] {completed}/{len(pending)} {source}/{run_name}",
                    flush=True,
                )
    paths = [task[3] for task in tasks]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} simulation run caches")
    strategies, aggregators, metadata = merge_bundles(paths)
    metadata.update(
        dataset="in_silico",
        input_root=os.fspath(input_root),
        escape_source=os.fspath(escape_path),
        escape_margin_source=(os.fspath(escape_margin_path) if escape_margin_path.is_file() else "legacy fallback"),
        growth_mode=GROWTH_MODE,
        target_population=TARGET_POPULATION,
        simulation_cohort=SIMULATION_COHORT,
        n_runs=len(paths),
    )
    return strategies, aggregators, metadata


def summarize(
    aggregator: plotter.Aggregator,
    strategies: list[str],
    condition: Condition,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    centers = 0.5 * (aggregator.x_edges[:-1] + aggregator.x_edges[1:])
    for strategy_index, strategy in enumerate(strategies):
        for x_index, center in enumerate(centers):
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
                    "cutoff": condition.cutoff,
                    "support_mode": condition.support_mode,
                    "support_threshold": condition.threshold,
                    "gate_on": f"both {TARGET_POPULATION}-density endpoints t0 and t1",
                    "gate_lower_cells_per_um2": condition.gate_lower,
                    "gate_upper_cells_per_um2": condition.gate_upper,
                    "x_definition": "average total density (t0+t1)/2",
                    "growth_definition": (
                        f"({TARGET_POPULATION}_t1-{TARGET_POPULATION}_t0)/"
                        f"average({TARGET_POPULATION}_t0,{TARGET_POPULATION}_t1)/delta_days"
                        if GROWTH_MODE == "relative"
                        else f"({TARGET_POPULATION}_t1-{TARGET_POPULATION}_t0)/"
                        "delta_days; displayed per 1000 um2"
                    ),
                    "strategy": strategy,
                    "x_bin_left_cells_per_1000_um2": aggregator.x_edges[x_index],
                    "x_bin_right_cells_per_1000_um2": aggregator.x_edges[x_index + 1],
                    "x_bin_center_cells_per_1000_um2": center,
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


def pooled(
    aggregator: plotter.Aggregator,
    available_strategies: tuple[str, ...],
) -> plotter.Aggregator:
    """Pool only the strategies selected for the publication master curve."""
    missing = [name for name in POOLED_STRATEGIES if name not in available_strategies]
    if missing:
        raise RuntimeError(f"Missing pooled strategies: {missing}")
    selected = [available_strategies.index(name) for name in POOLED_STRATEGIES]
    result = plotter.Aggregator.create(
        n_strategies=1,
        x_edges=aggregator.x_edges,
        ylim=(float(aggregator.y_edges[0]), float(aggregator.y_edges[-1])),
        n_y_bins=aggregator.n_y_bins,
    )
    result.count[0] = aggregator.count[selected].sum(axis=0)
    result.total[0] = aggregator.total[selected].sum(axis=0)
    result.total_sq[0] = aggregator.total_sq[selected].sum(axis=0)
    result.histogram[0] = aggregator.histogram[selected].sum(axis=0)
    return result


def main() -> None:
    args = parse_args()
    if (args.all_variants or USER_BUILD_ALL_VARIANTS) and not args.single_variant:
        common_args = [
            sys.executable,
            os.fspath(Path(__file__).resolve()),
            "--in-vitro-run", os.fspath(args.in_vitro_run),
            "--config", os.fspath(args.config),
            "--experimental-escape", os.fspath(args.experimental_escape),
            "--escape-margin-csv", os.fspath(args.escape_margin_csv),
            "--simulation-root", os.fspath(args.simulation_root),
            "--simulation-escape", os.fspath(args.simulation_escape),
            "--workers", str(max(1, int(args.workers))),
            "--single-variant",
        ]
        if args.overwrite_cache:
            common_args.append("--overwrite-cache")
        target_populations = (
            ("resistant", "sensitive") if USER_SEN_ALSO else ("resistant",)
        )
        for target_population in target_populations:
            for growth_mode in ("relative", "absolute"):
                for cohort in ("escaping", "all"):
                    child_environment = os.environ.copy()
                    child_environment["CORESEED_GROWTH_MODE"] = growth_mode
                    child_environment["CORESEED_SIMULATION_COHORT"] = cohort
                    child_environment["CORESEED_TARGET_POPULATION"] = target_population
                    print(
                        f"[variant] target={target_population}; growth={growth_mode}; "
                        f"simulation_cohort={cohort}",
                        flush=True,
                    )
                    subprocess.run(
                        common_args,
                        env=child_environment,
                        check=True,
                    )
        return
    cache_root = args.cache_root.expanduser()
    output_root = args.output_root.expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "[settings] "
        f"period={FOCUSED_PERIOD}, growth={GROWTH_MODE}, "
        f"target={TARGET_POPULATION}, "
        f"simulation_cohort={SIMULATION_COHORT}, statistic={BIN_STATISTIC}, "
        f"overwrite_cache={bool(args.overwrite_cache)}"
    )
    print(f"[settings] in vitro: {args.in_vitro_run.expanduser()}")
    print(f"[settings] cache:    {cache_root}")
    print(f"[settings] output:   {output_root}")
    if FOCUSED_PERIOD != "before_escape" or GROWTH_MODE != "relative":
        print(
            "[settings] standalone Figure 3 panels skipped: they require "
            "period=before_escape and growth=relative"
        )

    vitro_cache = cache_root / "in_vitro_all_conditions.npz"
    if args.overwrite_cache or not vitro_cache.is_file():
        vitro_strategies, vitro_aggregators, vitro_metadata = extract_in_vitro(
            args.in_vitro_run.expanduser(),
            args.config.expanduser(),
            args.experimental_escape.expanduser(),
            args.escape_margin_csv.expanduser(),
            cache_root / "in_vitro_wells",
        )
        save_bundle(vitro_cache, vitro_strategies, vitro_aggregators, vitro_metadata)
    else:
        vitro_strategies, vitro_aggregators, vitro_metadata = load_bundle(vitro_cache)
        print(f"[cache] {vitro_cache}")

    silico_cache = cache_root / "in_silico_all_conditions.npz"
    if args.overwrite_cache or not silico_cache.is_file():
        silico_strategies, silico_aggregators, silico_metadata = aggregate_simulations(
            args.simulation_root.expanduser(),
            args.simulation_escape.expanduser(),
            args.escape_margin_csv.expanduser(),
            cache_root,
            max(1, int(args.workers)),
            bool(args.overwrite_cache),
        )
        save_bundle(
            silico_cache,
            silico_strategies,
            silico_aggregators,
            silico_metadata,
        )
    else:
        silico_strategies, silico_aggregators, silico_metadata = load_bundle(silico_cache)
        print(f"[cache] {silico_cache}")

    manifest_rows: list[dict[str, object]] = []
    average_tables: dict[tuple[str, float], pd.DataFrame] = {}
    for condition in CONDITIONS:
        output_dir = output_root / condition.output_relative
        vitro_strategy = summarize(
            vitro_aggregators[condition.slug],
            vitro_strategies,
            condition,
        )
        vitro_pooled = summarize(
            pooled(vitro_aggregators[condition.slug], vitro_strategies),
            ["CT, AT50, and AT100 pooled"],
            condition,
        )
        silico_strategy = summarize(
            silico_aggregators[condition.slug],
            silico_strategies,
            condition,
        )
        silico_pooled = summarize(
            pooled(silico_aggregators[condition.slug], silico_strategies),
            ["CT, AT50, and AT100 pooled"],
            condition,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        vitro_strategy.to_csv(
            output_dir / "in_vitro_strategy_binned_statistics.tsv", sep="\t", index=False
        )
        vitro_pooled.to_csv(
            output_dir / "in_vitro_pooled_master_binned_statistics.tsv", sep="\t", index=False
        )
        silico_strategy.to_csv(
            output_dir / "in_silico_strategy_binned_statistics.tsv", sep="\t", index=False
        )
        silico_pooled.to_csv(
            output_dir / "in_silico_pooled_master_binned_statistics.tsv", sep="\t", index=False
        )
        if (
            GROWTH_MODE == "relative"
            and condition.cutoff == "before_escape"
            and condition.support_mode == "both"
            and condition.gate_lower is None
            and condition.gate_upper is None
            and math.isclose(condition.threshold, 1.0e-8)
        ):
            statistics_path = write_mean_curve_contrast_statistics(
                cache_root,
                output_dir,
                condition,
            )
            print(f"[saved] {statistics_path}")
        manifest_rows.append(
            {
                "cutoff": condition.cutoff,
                "support_mode": condition.support_mode,
                "support_threshold": condition.threshold,
                "condition_directory": os.fspath(output_dir),
                "in_vitro_pixels": int(vitro_strategy["n_pixels"].sum()),
                "in_silico_pixels": int(silico_strategy["n_pixels"].sum()),
            }
        )
        if condition.support_mode == "average":
            average_tables[(condition.cutoff, condition.threshold)] = pd.concat(
                [
                    vitro_strategy.assign(dataset="in_vitro"),
                    silico_strategy.assign(dataset="in_silico"),
                ],
                ignore_index=True,
            )
        print(f"[saved plotting data] {output_dir}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "output_manifest.tsv", sep="\t", index=False)
    # Keep a machine-readable record alongside the binned tables.  This makes
    # it impossible to silently pair a master curve with a cache produced from
    # a different experimental filter, simulation database, or bin width.
    provenance = {
        "x_bin_width": X_BIN_WIDTH,
        "period": FOCUSED_PERIOD,
        "growth_mode": GROWTH_MODE,
        "target_population": TARGET_POPULATION,
        "simulation_cohort": SIMULATION_COHORT,
        "support_policy": SUPPORT_POLICY,
        "in_vitro_cache": vitro_metadata,
        "in_silico_cache": silico_metadata,
    }
    (output_root / "master_curve_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    if FOCUSED_GATE_MATRIX:
        readme = (
            f"160-um average-t0/t1 {TARGET_POPULATION} {GROWTH_MODE}-growth "
            f"{FOCUSED_PERIOD} gate matrix\n"
            + ({
                "before_escape": "All confined intervals through the last pre-escape timepoint are included; the escape timepoint is excluded.\n",
                "after_escape": "Intervals starting at the escape timepoint or later are included.\n",
                "all_timepoints": "All available intervals are included.\n",
            }[FOCUSED_PERIOD])
            + (
                "Simulation cohort: escaping runs only.\n"
                if SIMULATION_COHORT == "escaping"
                else "Simulation cohort: all runs; non-escaping runs use all available intervals.\n"
            )
            + "Strategies: CT, AT50, AT100, LV-Agent, AT+.\n"
            f"Eight valid lower/upper gates and three lower thresholds; every condition is applied independently to both {TARGET_POPULATION}-density endpoints.\n"
            "The invalid 1e-3 to 4.375e-4 combination was omitted.\n"
        )
        (output_root / "README.txt").write_text(readme, encoding="utf-8")
        print(f"[done] {len(CONDITIONS)} plotting-data directories -> {output_root}")
        return

    equivalence_rows = []
    for cutoff in ("three_days_before_escape", "three_timepoints_before_escape"):
        reference = average_tables[(cutoff, 1.0e-10)].drop(
            columns=["support_threshold"]
        ).reset_index(drop=True)
        for threshold in (1.0e-8, 2.0e-5):
            candidate = average_tables[(cutoff, threshold)].drop(
                columns=["support_threshold"]
            ).reset_index(drop=True)
            equivalent = reference.equals(candidate)
            equivalence_rows.append(
                {
                    "cutoff": cutoff,
                    "reference_threshold": 1.0e-10,
                    "comparison_threshold": threshold,
                    "identical_binned_data": equivalent,
                }
            )
            if not equivalent:
                raise AssertionError(
                    f"Average-support variants unexpectedly differ for {cutoff}: {threshold}"
                )
    pd.DataFrame(equivalence_rows).to_csv(
        output_root / "average_support_equivalence_validation.tsv",
        sep="\t",
        index=False,
    )

    readme = (
        "160-um average-t0/t1 symmetric-growth pre-escape condition matrix\n"
        "=================================================================\n\n"
        "Datasets: in vitro and in silico. Included strategies: CT, AT50, "
        "AT100, LV-Agent, AT+.\n"
        "X: average total density, (total_t0 + total_t1)/2.\n"
        "Gate: average resistant density from 2e-5 to 4.375e-4 cells/um2.\n"
        "Y: (res_t1-res_t0)/average(res_t0,res_t1)/delta_days.\n"
        "Uncertainty: median and pixel-level IQR. Minimum plotted bin n=100.\n"
        "Y limits: -2.05 to 2.05 fraction/day, the natural one-day symmetric range.\n\n"
        "Cutoff variants:\n"
        "- cutoff_3_days_before_escape: t1 <= min(last confined day, escape day - 3).\n"
        "- cutoff_3_timepoints_before_escape: remove the final three confined "
        "measurement timepoints for each well/run.\n\n"
        "Support variants:\n"
        "- average_t0_t1_gt_*: average resistant density exceeds the threshold.\n"
        "- both_t0_t1_gt_*: resistant density at both endpoints exceeds the threshold.\n\n"
        "The three average-support thresholds are expected and validated to have "
        "identical binned data because the average gate already requires >=2e-5.\n"
        f"Aggregation caches: {cache_root}\n"
        "No experimental density maps were recalculated. Simulation density maps "
        "were reconstructed in memory at 160 um.\n"
    )
    (output_root / "README.txt").write_text(readme, encoding="utf-8")
    print(f"[done] 12 PDFs -> {output_root}")


if __name__ == "__main__":
    main()
