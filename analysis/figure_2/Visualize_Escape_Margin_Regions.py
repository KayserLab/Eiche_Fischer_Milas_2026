"""Render the geometric regions used by the experimental escape-margin metric.

The blue containment envelope is the intersection of all 360 sensitive-cell
tangent half-planes.  A resistant cell is counted as escaped when it lies
outside that envelope (the red region).  In a confined frame, the cyan
half-plane is the least-confining resistant-front test: dark-blue sensitive
cells within it are the cells counted for the signed negative margin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.figure_2.Escape_Margin_Analysis import (  # noqa: E402
    DEFAULT_ANGLE_COUNT,
    DEFAULT_DIRECTION_CHUNK_SIZE,
    DEFAULT_EXPERIMENTAL_PREPARED_SUBDIR,
    DEFAULT_EXPERIMENTAL_ROOT,
    PROJECTION_TOLERANCE,
    _prepared_xy,
    discover_prepared_experiments,
    outward_normals,
    probe_angles,
    signed_escape_metrics,
)


SAMPLE = "A5_AT100-A5"
PLATE = "Plate_5"
# One-based recorded timepoint indices 2, 19, and 24 correspond to days 3, 45,
# and 54 for this experiment.
DAYS = (3, 45, 54)


def point_limits(*point_sets: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a shared compact, equal-aspect display extent."""
    points = np.vstack([np.asarray(item) for item in point_sets if len(item)])
    low, high = points.min(axis=0), points.max(axis=0)
    span = max(float(np.max(high - low)), 1.0)
    padding = 0.06 * span
    return (float(low[0] - padding), float(high[0] + padding)), (float(low[1] - padding), float(high[1] + padding))


def clip_to_half_plane(polygon: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray:
    """Clip a convex polygon to ``normal dot point <= bound``."""
    if not len(polygon):
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(previous @ normal - bound)
    previous_inside = previous_value <= PROJECTION_TOLERANCE
    for current in polygon:
        current_value = float(current @ normal - bound)
        current_inside = current_value <= PROJECTION_TOLERANCE
        if current_inside != previous_inside:
            fraction = previous_value / (previous_value - current_value)
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous, previous_value, previous_inside = current, current_value, current_inside
    return np.asarray(output, dtype=float)


def sensitive_containment_envelope(sensitive: np.ndarray, normals: np.ndarray, limits: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Intersect the sampled sensitive tangent half-planes within the plot."""
    (xmin, xmax), (ymin, ymax) = limits
    polygon = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)
    fronts = np.max(sensitive @ normals.T, axis=0)
    for normal, front in zip(normals, fronts):
        polygon = clip_to_half_plane(polygon, normal, float(front))
        if not len(polygon):
            break
    return polygon


def escaped_flags(resistant: np.ndarray, sensitive: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Apply the production union-of-half-planes escape rule point by point."""
    sensitive_fronts = np.max(sensitive @ normals.T, axis=0)
    return np.any(resistant @ normals.T > sensitive_fronts + PROJECTION_TOLERANCE, axis=1)


def load_frame(path: Path, day: int) -> tuple[np.ndarray, np.ndarray]:
    with pd.HDFStore(path, mode="r") as store:
        day_key = f"d_{day:02d}"
        resistant = _prepared_xy(store, "res", day_key)
        sensitive = _prepared_xy(store, "sus", day_key)
    if not len(resistant) or not len(sensitive):
        raise ValueError(f"Day {day} lacks resistant or sensitive cells in {path}")
    return resistant, sensitive


def draw_frame(axis: plt.Axes, *, day: int, resistant: np.ndarray, sensitive: np.ndarray, normals: np.ndarray, limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    metrics = signed_escape_metrics(resistant, sensitive, angles_deg=probe_angles(DEFAULT_ANGLE_COUNT), direction_chunk_size=DEFAULT_DIRECTION_CHUNK_SIZE)
    escaped = escaped_flags(resistant, sensitive, normals)
    envelope = sensitive_containment_envelope(sensitive, normals, limits)
    (xmin, xmax), (ymin, ymax) = limits

    # Outside the sensitive-tangent envelope is precisely the union of the
    # direction-specific escape half-planes, within this displayed rectangle.
    axis.fill([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], color="#f9d9df", zorder=0)
    if len(envelope):
        axis.fill(envelope[:, 0], envelope[:, 1], color="#dbeef7", zorder=1)
        axis.plot(*np.vstack([envelope, envelope[0]]).T, color="#377f9c", lw=0.75, zorder=2)

    axis.scatter(sensitive[:, 0], sensitive[:, 1], s=1.2, color="#268bd2", alpha=0.48, linewidths=0, zorder=3)
    axis.scatter(resistant[~escaped, 0], resistant[~escaped, 1], s=1.5, color="#f1c40f", alpha=0.62, linewidths=0, zorder=4)
    if np.any(escaped):
        axis.scatter(resistant[escaped, 0], resistant[escaped, 1], s=4, color="#d7304f", alpha=0.9, linewidths=0, zorder=5)

    if metrics.phase != "escaped" and np.isfinite(metrics.min_confinement_angle_deg):
        normal = outward_normals(np.array([metrics.min_confinement_angle_deg]))[0]
        resistant_front = float(np.max(resistant @ normal))
        counted = (sensitive @ normal) >= resistant_front - PROJECTION_TOLERANCE
        # Shade the half-plane beyond the resistant tangent, clipped to view.
        half_plane = clip_to_half_plane(
            np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float),
            -normal, -resistant_front,
        )
        if len(half_plane):
            axis.fill(half_plane[:, 0], half_plane[:, 1], color="#55c8d4", alpha=0.22, zorder=2)
        tangent_direction = np.array([-normal[1], normal[0]])
        centre = normal * resistant_front
        span = max(xmax - xmin, ymax - ymin)
        tangent = np.vstack([centre - tangent_direction * span, centre + tangent_direction * span])
        axis.plot(tangent[:, 0], tangent[:, 1], color="#00a6b9", lw=1.0, zorder=6)
        axis.scatter(sensitive[counted, 0], sensitive[counted, 1], s=3.5, color="#005f9e", alpha=0.9, linewidths=0, zorder=7)

    detail = (
        f"escaped: {int(np.count_nonzero(escaped))} resistant cells"
        if metrics.phase == "escaped"
        else f"confined: {metrics.confinement_count:,} sensitive cells counted"
    )
    axis.set_title(f"Day {day}\n{detail}", fontsize=10, loc="left", pad=6)
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    axis.set_aspect("equal")
    axis.invert_yaxis()  # Experimental pixel coordinates: preserve image orientation.
    axis.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "Desktop" / "Plate_5_A5_AT100_escape_margin_regions_timepoints_2_19_24.png")
    args = parser.parse_args()

    paths = discover_prepared_experiments(DEFAULT_EXPERIMENTAL_ROOT, prepared_subdir=DEFAULT_EXPERIMENTAL_PREPARED_SUBDIR, samples=[SAMPLE], max_files=1)
    path = next(path for path in paths if path.parent.parent.name == PLATE)
    frames = {day: load_frame(path, day) for day in DAYS}
    limits = point_limits(*(points for frame in frames.values() for points in frame))
    angles = probe_angles(DEFAULT_ANGLE_COUNT)
    normals = outward_normals(angles)

    figure, axes = plt.subplots(1, len(DAYS), figsize=(15, 6.8))
    for axis, day in zip(axes, DAYS):
        resistant, sensitive = frames[day]
        draw_frame(axis, day=day, resistant=resistant, sensitive=sensitive, normals=normals, limits=limits)
    figure.legend(
        handles=[
            Patch(facecolor="#dbeef7", edgecolor="#377f9c", label="sensitive containment envelope (not escaped)"),
            Patch(facecolor="#f9d9df", edgecolor="none", label="escape region (outside ≥1 sensitive tangent)"),
            plt.Line2D([], [], marker="o", linestyle="", color="#268bd2", markersize=4, label="sensitive cell"),
            plt.Line2D([], [], marker="o", linestyle="", color="#f1c40f", markersize=4, label="resistant cell, not escaped"),
            plt.Line2D([], [], marker="o", linestyle="", color="#d7304f", markersize=4, label="escaped resistant cell"),
            plt.Line2D([], [], color="#00a6b9", lw=1.2, label="least-confining resistant tangent (confined frame only)"),
        ],
        loc="lower center", ncol=2, fontsize=8, frameon=False,
    )
    figure.suptitle("Escape-margin regions — Plate 5, A5_AT100-A5", fontsize=14, y=0.97)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.17, wspace=0.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"[saved] {args.output}")
    print(f"[saved] {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
