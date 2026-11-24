from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

try:
    from analysis.legacy import escape_confinement_circular_core as legacy
except ImportError:
    import escape_confinement_circular_core as legacy

N_DOTS = legacy.N_DOTS


def make_probe_angles() -> np.ndarray:
    return np.arange(N_DOTS, dtype=float)


def outward_normals(angles_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.column_stack((np.sin(theta), -np.cos(theta)))


def tangent_counts(clone_res_xy, sus_xy, angles_deg):
    clone = np.asarray(clone_res_xy, dtype=float)
    sensitive = np.asarray(sus_xy, dtype=float)
    if clone.ndim != 2 or clone.shape[1:] != (2,) or not len(clone):
        raise ValueError('clone_res_xy must have shape (n, 2), with n > 0')
    if sensitive.ndim != 2 or (sensitive.size and sensitive.shape[1:] != (2,)):
        raise ValueError('sus_xy must have shape (n, 2)')
    normals = outward_normals(angles_deg)
    counts = np.zeros(len(normals), dtype=np.int64)
    points = np.empty((len(normals), 2), dtype=float)
    offsets = np.empty(len(normals), dtype=float)
    for i, normal in enumerate(normals):
        projection = clone @ normal
        outermost = int(np.argmax(projection))
        offset = float(projection[outermost])
        points[i], offsets[i] = clone[outermost], offset
        if len(sensitive):
            counts[i] = int(np.count_nonzero((sensitive @ normal) >= offset - 1e-9))
    return counts, points, offsets


def confinement_for_clone(clone_res_xy, sus_xy, angles_deg, dots_xy: Optional[np.ndarray] = None) -> dict:
    # dots_xy is deliberately ignored: this calculation has no reference circle.
    counts, points, offsets = tangent_counts(clone_res_xy, sus_xy, angles_deg)
    minimum = int(np.argmin(counts))
    return {
        'confinement_thickness': int(counts[minimum]),
        'min_angle_deg': float(angles_deg[minimum]),
        # Kept for compatible CSVs; it now means tangent offset x dot n.
        'distance_px_at_min': float(offsets[minimum]),
        'min_dot_x': float(points[minimum, 0]),
        'min_dot_y': float(points[minimum, 1]),
    }


# The retained I/O pipeline resolves this global when it calls analyze_replicate.
legacy.confinement_for_clone = confinement_for_clone


def run_test(plate: str, sample: str) -> str:
    io = legacy.build_test_replicate(plate, sample)
    legacy.debug_dominant_clone_mapping(io)
    table, _ = legacy.analyze_replicate(io)
    legacy.os.makedirs(legacy.OUT_DIR, exist_ok=True)
    path = legacy.os.path.join(legacy.OUT_DIR, f'Escape_ConfinementThickness_test_{plate}_{sample}.csv')
    table.to_csv(path, index=False)
    print(f'[save] Test CSV: {path} ({len(table)} rows)')
    print('[info] Exact tangent-half-plane metric; no full pipeline was run.')
    return path


def run_all() -> str:
    # legacy.run_all retains its existing nested tqdm progress bars.
    return legacy.run_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute exact tangent-line confinement.')
    parser.add_argument('--mode', choices=['test', 'all'], default='all')
    parser.add_argument('--plate', default=legacy.DEFAULT_TEST_PLATE)
    parser.add_argument('--sample', default=legacy.DEFAULT_TEST_SAMPLE)
    args = parser.parse_args()
    run_test(args.plate, args.sample) if args.mode == 'test' else run_all()
