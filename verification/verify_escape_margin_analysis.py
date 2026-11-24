"""Create visual and row-level audits for Escape_Margin_Analysis.py."""

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

from analysis.figure_2.Escape_Margin_Analysis import (
    DEFAULT_ANGLE_COUNT,
    DEFAULT_DIRECTION_CHUNK_SIZE,
    DEFAULT_EXPERIMENTAL_PREPARED_SUBDIR,
    DEFAULT_EXPERIMENTAL_ROOT,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SIMULATION_ROOT,
    PROJECTION_TOLERANCE,
    PREPARED_H5_SUFFIX,
    SIMULATION_H5_SUFFIX,
    _prepared_xy,
    _simulation_xy,
    discover_prepared_experiments,
    discover_simulations,
    outward_normals,
    probe_angles,
    quantify_prepared_experiment,
    quantify_simulation,
    signed_escape_metrics,
)
from shared.data_paths import VERIFICATION_DATA
from shared.experiment_filters import EXCLUDE_SAMPLES
from verification.selection import balanced_strategy_sample, strategy_from_sample


DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA


def comparable(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-9))
    return left == right


def row_matches(expected: pd.Series, stored: pd.Series) -> bool:
    columns = ["confinement_count", "escaped_resistant_count", "signed_escape_margin", "phase", "min_confinement_angle_deg"]
    return all(comparable(expected[column], stored[column]) for column in columns)


def escape_flags(resistant: np.ndarray, sensitive: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Visual-only trace of the production escaped-cell rule."""
    if not len(resistant) or not len(sensitive):
        return np.ones(len(resistant), dtype=bool) if not len(sensitive) else np.zeros(len(resistant), dtype=bool)
    normals = outward_normals(angles)
    return np.any((resistant @ normals.T) > np.max(sensitive @ normals.T, axis=0), axis=1)


def point_limits(*point_sets: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return compact, equal-aspect limits based only on the displayed cells."""
    populated = [np.asarray(points) for points in point_sets if len(points)]
    if not populated:
        return (-1.0, 1.0), (-1.0, 1.0)
    points = np.vstack(populated)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = max(float(np.max(upper - lower)), 1.0)
    padding = 0.06 * span
    return (
        (float(lower[0] - padding), float(upper[0] + padding)),
        (float(lower[1] - padding), float(upper[1] + padding)),
    )


def draw_frame(axis: plt.Axes, resistant: np.ndarray, sensitive: np.ndarray, metrics, day: int, angles: np.ndarray) -> None:
    escaped = escape_flags(resistant, sensitive, angles)
    if len(sensitive):
        axis.scatter(sensitive[:, 0], sensitive[:, 1], s=1, c="#268bd2", alpha=0.45, linewidths=0, label="sensitive")
    if len(resistant):
        axis.scatter(resistant[~escaped, 0], resistant[~escaped, 1], s=1, c="#f1c40f", alpha=0.55, linewidths=0, label="resistant")
        if np.any(escaped):
            axis.scatter(resistant[escaped, 0], resistant[escaped, 1], s=2, c="#e74c3c", alpha=0.85, linewidths=0, label="escaped")
    x_limits, y_limits = point_limits(resistant, sensitive)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    if (
        metrics.phase != "escaped"
        and len(resistant)
        and len(sensitive)
        and np.isfinite(metrics.min_confinement_angle_deg)
    ):
        normal = outward_normals(np.array([metrics.min_confinement_angle_deg]))[0]
        resistant_front = float(np.max(resistant @ normal))
        confining_sensitive = (sensitive @ normal) >= resistant_front - PROJECTION_TOLERANCE
        if np.any(confining_sensitive):
            axis.scatter(
                sensitive[confining_sensitive, 0], sensitive[confining_sensitive, 1],
                s=1.5, c="#005f9e", alpha=0.8, linewidths=0,
                label="sensitive at/beyond tangent",
            )
        centre = normal * resistant_front
        direction = np.array([-normal[1], normal[0]])
        span = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
        line = np.vstack([centre - direction * span, centre + direction * span])
        axis.plot(
            line[:, 0], line[:, 1], color="#00bcd4", linewidth=1.2,
            label="least-confining resistant-front tangent", clip_on=True,
        )
    axis.set_aspect("equal")
    axis.set_title(f"Day {day}: {metrics.phase}, margin {metrics.signed_escape_margin:g}")
    axis.set_axis_off()


def load_experimental(path: Path, day: int) -> tuple[np.ndarray, np.ndarray]:
    with pd.HDFStore(path, mode="r") as store:
        day_key = f"d_{day:02d}"
        return _prepared_xy(store, "res", day_key), _prepared_xy(store, "sus", day_key)


def load_simulation(path: Path, day: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, mode="r") as handle:
        key = next(key for key in handle["res"] if int(key.removeprefix("d_")) == day)
        return _simulation_xy(handle["res"], key), _simulation_xy(handle["sus"], key)


def audit_dataset(dataset: str, paths: list[Path], stored: pd.DataFrame, review_dir: Path, angles: np.ndarray, cutoff_map: dict[str, float | None], progress) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in paths:
        progress.set_postfix_str(f"{dataset}: {path.name}")
        if dataset == "experimental":
            sample = path.name.removesuffix(PREPARED_H5_SUFFIX)
            expected = quantify_prepared_experiment(path, angles_deg=angles, direction_chunk_size=DEFAULT_DIRECTION_CHUNK_SIZE)
            loader = load_experimental
        else:
            sample = path.name.removesuffix(SIMULATION_H5_SUFFIX)
            expected = quantify_simulation(path, angles_deg=angles, direction_chunk_size=DEFAULT_DIRECTION_CHUNK_SIZE)
            loader = load_simulation
        plate = str(expected.iloc[0]["plate"])
        cutoff = cutoff_map.get(sample, float("inf"))
        expected = expected[expected["day"] <= cutoff].copy()
        visual = expected.iloc[[0, len(expected) // 2, len(expected) - 1]]
        fig, axes = plt.subplots(3, 1, figsize=(7, 15), constrained_layout=True)
        checks: list[dict[str, object]] = []
        for axis, (_, expected_row) in zip(axes, visual.iterrows()):
            day = int(expected_row.day)
            resistant, sensitive = loader(path, day)
            metrics = signed_escape_metrics(resistant, sensitive, angles_deg=angles, direction_chunk_size=DEFAULT_DIRECTION_CHUNK_SIZE)
            stored_rows = stored[
                (stored["dataset_type"] == dataset)
                & (stored["plate"] == plate)
                & (stored["sample"] == sample)
                & (stored["day"] == day)
            ]
            matched = len(stored_rows) == 1 and row_matches(expected_row, stored_rows.iloc[0])
            checks.append({"day": day, "csv_row_matches_production": matched})
            draw_frame(axis, resistant, sensitive, metrics, day, angles)
        passed = all(item["csv_row_matches_production"] for item in checks)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right", fontsize=7, markerscale=4)
        fig.suptitle(f"{'PASS' if passed else 'FAIL'}: {dataset}, {plate}, {sample}", fontsize=14)
        fig.savefig(review_dir / dataset / f"{plate}_{sample}.png", dpi=400)
        plt.close(fig)
        results.append({"dataset": dataset, "plate": plate, "sample": sample, "passed": passed, "visual_frames": checks})
        progress.update()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    progress = tqdm(total=5, desc="Verifying escape margins", unit="step")
    progress.set_postfix_str("discovering inputs")
    experimental = [
        path for path in discover_prepared_experiments(DEFAULT_EXPERIMENTAL_ROOT, prepared_subdir=DEFAULT_EXPERIMENTAL_PREPARED_SUBDIR, samples=None, max_files=0)
        if EXCLUDE_SAMPLES.get(path.name.removesuffix(PREPARED_H5_SUFFIX), float("inf")) is not None
    ]
    simulation = discover_simulations(DEFAULT_SIMULATION_ROOT, samples=None, max_files=0)
    progress.update(2)
    selected_experimental = balanced_strategy_sample(experimental, lambda path: strategy_from_sample(path.name.removesuffix(PREPARED_H5_SUFFIX)), args.seed)
    selected_simulation = balanced_strategy_sample(simulation, lambda path: path.parent.parent.name.removeprefix("Plate_"), args.seed)
    progress.total = len(selected_experimental) + len(selected_simulation) + 5
    progress.refresh()
    progress.update(2)

    review_dir = args.verification_root / "Escape_Margin_Analysis" / f"strategy_balanced_seed{args.seed}"
    (review_dir / "experimental").mkdir(parents=True, exist_ok=True)
    (review_dir / "simulation").mkdir(parents=True, exist_ok=True)
    (review_dir / "README.md").write_text(
        "# Escape-margin verification\n\nThis review checks the geometric calculation that decides whether resistant cells remain enclosed or have escaped beyond the sensitive population.\n\nEach PNG shows an early, middle, and late timepoint for one strategy-balanced experimental or simulation replicate. Sensitive cells are blue, with the cells at or beyond the tangent shown in darker blue. Resistant cells are yellow and escaped resistant cells are red. Before escape, the cyan line is the resistant-front tangent at the least-confining sampled direction; it is a visual aid for that directional test, not a physical colony boundary. No tangent is shown once escape has occurred.\n\nA PASS means the signed escape-margin values in the analysis CSV match the production calculation for the displayed frames. The metadata file identifies every checked replicate.\n",
        encoding="utf-8",
    )
    stored = pd.read_csv(args.output_csv)
    angles = probe_angles(DEFAULT_ANGLE_COUNT)
    experimental_results = audit_dataset("experimental", selected_experimental, stored, review_dir, angles, EXCLUDE_SAMPLES, progress)
    simulation_results = audit_dataset("simulation", selected_simulation, stored, review_dir, angles, {}, progress)
    results = experimental_results + simulation_results
    passed = all(item["passed"] for item in results)
    (review_dir / "metadata.json").write_text(json.dumps({
        "script_under_test": "analysis/figure_2/Escape_Margin_Analysis.py",
        "production_imports": ["quantify_prepared_experiment", "quantify_simulation", "signed_escape_metrics"],
        "selection": "Strategy-balanced cohort; ATAgent1 excluded.", "seed": args.seed, "passed": passed, "samples": results,
    }, indent=2), encoding="utf-8")
    progress.update()
    progress.close()
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")


if __name__ == "__main__":
    main()
