"""Visual audit of the pixel transformation and gates used for Figure 3 master curves."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as production
from analysis.figure_3 import generate_figure2_mask_filtered_density_maps as mapped_paths
from shared.data_paths import VERIFICATION_DATA


DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA
STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT+")


@dataclass(frozen=True)
class Interval:
    plate: str
    well: str
    strategy: str
    maps_dir: Path
    first_token: str
    next_token: str
    first_day: float
    next_day: float
    active_cutoffs: frozenset[str]


def usable_path(path: Path) -> Path:
    """Use the mapped extended-length path, as in the Figure 3 map writer."""
    return Path(mapped_paths.extended_windows_path(mapped_paths.mapped_y_path(path)))


def read_tiff(path: Path) -> np.ndarray:
    with usable_path(path).open("rb") as stream:
        return tifffile.imread(stream).astype(np.float32, copy=False)


def find_intervals() -> list[Interval]:
    cutoffs = production.experimental_escape_cutoffs(
        production.DEFAULT_EXPERIMENT_ESCAPE, production.DEFAULT_ESCAPE_MARGIN_CSV
    )
    cutoff_lookup = {
        (str(row.plate), str(row.well)): (float(row.escape_day), float(row.pre_escape_cutoff_day))
        for row in cutoffs.itertuples(index=False)
    }
    candidates: list[Interval] = []
    for changes_dir in sorted(production.DEFAULT_INVITRO_RUN.glob("Plate_*/*/density_changes")):
        plate, well = changes_dir.parent.parent.name, changes_dir.parent.name
        strategy = production.canonical_strategy_label(production.base.strategy_from_plate_and_well(plate, well))
        if strategy not in STRATEGIES or (plate, well) not in cutoff_lookup:
            continue
        configured = production.EXCLUDE_SAMPLES.get(well, ...)
        if configured is None:
            continue
        maps_dir = changes_dir.parent / "density_maps"
        timepoints = production.timepoints_from_maps(maps_dir, well)
        limits = production.cutoff_limits([day for _, day in timepoints], *cutoff_lookup[(plate, well)], configured)
        if not limits:
            continue
        for change_path in sorted(changes_dir.glob(production.base.CHANGE_SPECS["rel"]["glob"])):
            match = production.base.REL_CHANGE_RE.match(change_path.name)
            if match is None:
                continue
            first_token, next_token = match.group("first_tp"), match.group("next_tp")
            first_day, next_day = production.base.timepoint_to_day(first_token), production.base.timepoint_to_day(next_token)
            active = frozenset(name for name, limit in limits.items() if next_day <= limit and (name != "after_escape" or first_day >= cutoff_lookup[(plate, well)][0]))
            if active:
                candidates.append(Interval(plate, well, strategy, maps_dir, first_token, next_token, first_day, next_day, active))
    # One interval from each selected well: choose its middle valid interval.
    result: list[Interval] = []
    rng = random.Random(20260831)
    for strategy in STRATEGIES:
        pool = sorted((entry for entry in candidates if entry.strategy == strategy), key=str)
        if not pool:
            continue
        item = rng.choice(pool)
        same_well = [entry for entry in candidates if (entry.plate, entry.well) == (item.plate, item.well)]
        result.append(same_well[len(same_well) // 2])
    return result


def selected_pixels(total0: np.ndarray, res0: np.ndarray, total1: np.ndarray, res1: np.ndarray, delta_days: float, condition: production.Condition) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total0_float, total1_float = total0.astype(np.float64), total1.astype(np.float64)
    res0_float, res1_float = res0.astype(np.float64), res1.astype(np.float64)
    average_total = 0.5 * (total0_float + total1_float)
    average_res = 0.5 * (res0_float + res1_float)
    x = average_total.ravel() * production.base.X_DISPLAY_SCALE
    y = np.full(x.shape, np.nan)
    valid_average = average_res.ravel() > 0
    res0_flat, res1_flat = res0_float.ravel(), res1_float.ravel()
    y[valid_average] = (res1_flat[valid_average] - res0_flat[valid_average]) / average_res.ravel()[valid_average] / delta_days
    if condition.gate_lower is None or condition.gate_upper is None:
        # This is the publication condition: a lower support threshold at both
        # endpoints, with no upper density gate.
        support = (res0_flat >= condition.threshold) & (res1_flat >= condition.threshold)
    else:
        support = (res0_flat >= condition.gate_lower) & (res0_flat <= condition.gate_upper) & (res1_flat >= condition.gate_lower) & (res1_flat <= condition.gate_upper)
    valid = np.isfinite(x) & (x >= production.X_EDGES[0]) & (x <= production.X_EDGES[-1]) & np.isfinite(y) & support
    bins = np.clip(np.floor(x / production.X_BIN_WIDTH).astype(int), 0, len(production.X_EDGES) - 2)
    return x, y, bins, valid


def audit_interval(interval: Interval) -> tuple[dict[str, object], tuple]:
    total0 = read_tiff(production.base.density_map_path(interval.maps_dir, interval.well, interval.first_token))
    res0 = read_tiff(production.base.resistant_density_map_path(interval.maps_dir, interval.well, interval.first_token))
    total1 = read_tiff(production.base.density_map_path(interval.maps_dir, interval.well, interval.next_token))
    res1 = read_tiff(production.base.resistant_density_map_path(interval.maps_dir, interval.well, interval.next_token))
    condition = next(
        item for item in production.CONDITIONS
        if item.cutoff == "before_escape"
        and item.support_mode == "both"
        and item.gate_lower is None
        and item.gate_upper is None
        and item.threshold == 1.0e-8
    )
    delta_days = interval.next_day - interval.first_day
    x, y, bins, valid = selected_pixels(total0, res0, total1, res1, delta_days, condition)
    aggregators = production.create_aggregators(1)
    production.add_interval(aggregators, 0, total0, res0, total1, res1, delta_days, set(interval.active_cutoffs))
    aggregate = aggregators[condition.slug]
    expected_count = np.bincount(bins[valid], minlength=len(production.X_EDGES) - 1)
    expected_total = np.bincount(bins[valid], weights=y[valid], minlength=len(production.X_EDGES) - 1)
    audit = {
        "interval": f"d{interval.first_day:g} to d{interval.next_day:g}",
        "condition": condition.slug,
        "pixels_passing_publication_threshold": int(valid.sum()),
        "production_aggregator_matches_selected_pixels": bool(np.array_equal(aggregate.count[0], expected_count) and np.allclose(aggregate.total[0], expected_total, rtol=1e-12, atol=1e-12)),
        "all_selected_pixels_pass_both_endpoint_threshold": bool(((res0.ravel()[valid] >= condition.threshold) & (res1.ravel()[valid] >= condition.threshold)).all()),
    }
    return audit, (total0, total1, res0, res1, valid.reshape(total0.shape), x, y, bins, aggregate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    args = parser.parse_args()
    selected = find_intervals()
    if not selected:
        raise RuntimeError("No eligible pre-escape Figure 3 intervals found.")
    review_dir = args.verification_root / "aggregate_average_symmetric_condition_matrix_160" / "sample_seed20260831"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Figure 3 master-curve aggregation verification\n\nThis review checks how one density-map interval becomes the points used in the published Figure 3 master curve. For every map pixel, the script uses the average total density as the horizontal value and the symmetric resistant-density change per day as the vertical value. A pixel is retained when resistant density is at least 1e-8 cells/µm² at both endpoints; there is no upper density gate.\n\nThe first four panels show the input density maps at the beginning and end of an interval. The threshold panel marks retained pixels in white. The final panel shows their calculated values and the resulting one-interval bin means. A PASS means the production aggregator has exactly the same bin counts and sums as these displayed pixels.\n",
        encoding="utf-8",
    )
    progress = tqdm(total=len(selected) + 2, desc="Verifying Figure 3 master-curve aggregation", unit="interval")
    results = []
    for interval in selected:
        progress.set_postfix_str(f"{interval.plate}: {interval.well}")
        audit, payload = audit_interval(interval)
        total0, total1, res0, res1, gate_mask, x, y, bins, aggregate = payload
        passed = all(value for value in audit.values() if isinstance(value, bool))
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        vmax_total, vmax_res = max(total0.max(), total1.max()), max(res0.max(), res1.max())
        for index, (axis, image, title, vmax, cmap) in enumerate(zip(axes.flat[:4], (total0, total1, res0, res1), (f"total: day {interval.first_day:g}", f"total: day {interval.next_day:g}", f"resistant: day {interval.first_day:g}", f"resistant: day {interval.next_day:g}"), (vmax_total, vmax_total, vmax_res, vmax_res), ("magma", "magma", "YlOrBr", "YlOrBr"), strict=True)):
            axis.imshow(image, cmap=cmap, vmin=0, vmax=max(float(vmax), np.finfo(float).eps), interpolation="nearest")
            # The same publication-threshold mask shown as white in panel five, overlaid
            # here to make its location relative to each density map explicit.
            axis.contour(gate_mask, levels=[0.5], colors="black" if index >= 2 else "white", linewidths=0.4)
            axis.set_title(title); axis.set_axis_off()
        axes[1, 1].imshow(gate_mask, cmap="gray", interpolation="nearest")
        axes[1, 1].set_title(f"white: resistant density ≥1e-8 at both endpoints\n{audit['pixels_passing_publication_threshold']:,} pixels")
        axes[1, 1].set_axis_off()
        sample = np.flatnonzero(gate_mask.ravel())
        if len(sample) > 80_000:
            sample = sample[::max(1, len(sample) // 80_000)]
        axes[1, 2].scatter(x[sample], y[sample], s=0.3, alpha=0.2, color="#4c72b0", rasterized=True)
        nonzero = aggregate.count[0] > 0
        centers = 0.5 * (production.X_EDGES[:-1] + production.X_EDGES[1:])
        axes[1, 2].plot(centers[nonzero], (aggregate.total[0][nonzero] / aggregate.count[0][nonzero]), color="#d55e00", linewidth=2, label="one-interval bin mean")
        axes[1, 2].set(xlabel="average total density (display units)", ylabel="symmetric resistant growth / day", ylim=production.Y_RANGE)
        axes[1, 2].legend(fontsize=8)
        fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {interval.plate}, {interval.well} ({interval.strategy}); {audit['interval']}", fontsize=13)
        fig.savefig(review_dir / f"{interval.plate}_{interval.well}_{interval.strategy}.png", dpi=400)
        plt.close(fig)
        results.append({"plate": interval.plate, "sample": interval.well, "strategy": interval.strategy, "passed": passed, **audit})
        progress.update()
    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({"script_under_test": "analysis/figure_3/Legacy/aggregate_average_symmetric_condition_matrix_160.py", "production_imports": ["timepoints_from_maps", "cutoff_limits", "add_interval", "create_aggregators"], "passed": passed, "samples": results}, indent=2), encoding="utf-8")
    progress.update(2)
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
