"""Create per-run visual audits for build_simulation_overview.py."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.figure_2.build_simulation_overview import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT,
    FILTER_OUTLIER_BOUNDARY,
    TIME_DUMP,
    dataset,
    discover_runs,
    rows_for_run,
    spatial_filter_mask,
)
from shared.data_paths import VERIFICATION_DATA
from verification.selection import balanced_strategy_sample


DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def fields_match(expected: dict[str, object], stored: pd.Series) -> bool:
    for field, value in expected.items():
        observed = stored[field]
        if isinstance(value, float):
            if not np.isclose(float(value), float(observed), rtol=0, atol=1e-9, equal_nan=True):
                return False
        elif str(value) != str(observed):
            return False
    return True


def snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read active known-type cells and the production spatial-filter mask."""
    with h5py.File(path, mode="r") as handle:
        data = handle["data"]
        flags = np.asarray(dataset(data, "cell_flags"))
        types = np.asarray(dataset(data, "cell_type"))
        active = flags == 1
        positions = np.asarray(dataset(data, "cell_nucleus"))[active, :2]
        types = types[active]
    known = (types == 0) | (types == 1)
    return positions[known], types[known], spatial_filter_mask(positions[known])


def plot_snapshot(axis: plt.Axes, path: Path, day: int) -> None:
    positions, types, kept = snapshot(path)
    if np.any(~kept):
        axis.scatter(positions[~kept, 0], positions[~kept, 1], s=3, c="#808080", alpha=0.2, linewidths=0, label="excluded")
    for value, color, label in ((0, "#268bd2", "sensitive"), (1, "#f1c40f", "resistant")):
        selection = kept & (types == value)
        if np.any(selection):
            axis.scatter(positions[selection, 0], positions[selection, 1], s=4, c=color, alpha=0.7, linewidths=0, label=label)
    axis.set_xlim(-FILTER_OUTLIER_BOUNDARY, FILTER_OUTLIER_BOUNDARY)
    axis.set_ylim(-FILTER_OUTLIER_BOUNDARY, FILTER_OUTLIER_BOUNDARY)
    axis.set_aspect("equal")
    axis.set_title(f"Day {day} (shared spatial scale)")
    axis.set_axis_off()
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="upper right", fontsize=7, markerscale=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    progress = tqdm(total=3, desc="Verifying simulation overview", unit="step")
    progress.set_postfix_str("discovering simulation runs")
    available = discover_runs(args.input_root, protocol_filter=None)
    progress.update()
    selected = balanced_strategy_sample(available, lambda plan: plan.protocol, args.seed)
    args.sample_count = len(selected)
    progress.total = args.sample_count + 3
    progress.refresh()
    progress.update()
    stored_csv = pd.read_csv(args.output_csv)

    review_dir = args.verification_root / "build_simulation_overview" / f"sample_n{args.sample_count}_seed{args.seed}"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Simulation-overview verification\n\nThis review checks the daily simulation-count table used for downstream comparisons.\n\nEach PNG shows one strategy-balanced simulation run. Six panels show filtered spatial cell snapshots on the same fixed scale, with sensitive cells in blue, resistant cells in yellow, and excluded cells in grey. Small arrows in the trajectory panel mark the six displayed days.\n\nA PASS means every daily CSV row for that run matches the production summary calculation. The metadata file lists the checked runs.\n",
        encoding="utf-8",
    )
    results: list[dict[str, object]] = []
    for plan in selected:
        progress.set_postfix_str(f"{plan.protocol}: run {plan.run}")
        expected_rows = rows_for_run(plan, day_offset=1)
        expected = pd.DataFrame(expected_rows)
        stored = stored_csv[(stored_csv["protocol"].astype(str) == plan.protocol) & (stored_csv["run"].astype(str) == str(plan.run))].sort_values("day")
        row_checks = [
            fields_match(row, stored.iloc[index]) if index < len(stored) else False
            for index, row in enumerate(expected_rows)
        ]
        indices = np.unique(np.linspace(0, len(plan.dumps) - 1, num=min(6, len(plan.dumps)), dtype=int))
        fig = plt.figure(figsize=(15, 16), constrained_layout=True)
        grid = fig.add_gridspec(3, 3)
        snapshot_axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
        for axis, index in zip(snapshot_axes, indices):
            dump = plan.dumps[index]
            plot_snapshot(axis, dump.path, dump.timestep // TIME_DUMP + 1)
        for axis in snapshot_axes[len(indices):]:
            axis.set_axis_off()
        trajectory = fig.add_subplot(grid[2, :])
        for column, color, label in (
            ("sen_cell_count", "#268bd2", "raw sensitive"),
            ("res_cell_count", "#f1c40f", "raw resistant"),
            ("filtered_sen_cell_count", "#268bd2", "filtered sensitive"),
            ("filtered_res_cell_count", "#f1c40f", "filtered resistant"),
        ):
            style = "--" if column.startswith("filtered") else "-"
            trajectory.plot(expected["day"], expected[column], style, color=color, label=label)
            if len(stored):
                trajectory.scatter(stored["day"], stored[column], s=8, color=color)
        trajectory.set_title("Counts: line=recomputed, dots=CSV")
        trajectory.set_xlabel("day")
        trajectory.set_ylabel("cells")
        trajectory.legend(fontsize=7)
        upper = trajectory.get_ylim()[1]
        for index in indices:
            day = plan.dumps[index].timestep // TIME_DUMP + 1
            trajectory.annotate(
                "", xy=(day, upper * 0.94), xytext=(day, upper * 1.04),
                arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 0.8},
                annotation_clip=False,
            )
        passed = len(stored) == len(expected) and all(row_checks)
        fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {plan.protocol}, run {plan.run}", fontsize=14)
        fig.savefig(review_dir / f"{plan.protocol}_run_{plan.run}.png", dpi=400)
        plt.close(fig)
        results.append({"protocol": plan.protocol, "run": plan.run, "passed": passed, "daily_rows_match": int(sum(row_checks)), "daily_rows": len(expected_rows)})
        progress.update()

    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({
        "script_under_test": "analysis/figure_2/build_simulation_overview.py",
        "production_imports": ["discover_runs", "rows_for_run", "spatial_filter_mask"],
        "sample_count": args.sample_count, "seed": args.seed, "passed": passed, "samples": results,
    }, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
