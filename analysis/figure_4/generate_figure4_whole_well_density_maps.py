"""Generate Figure 4 whole-well density maps after the fixed point filter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analysis.basic.density_maps import Density_Maps_Epanechnikov_coarsegrained as maps
from shared.data_paths import intermediate_dir


FILTER_RADIUS_PX = float(os.environ.get("CORESEED_FILTER_RADIUS_PX", "237.8"))
FILTER_MIN_NEIGHBOURS = 3
FILTER_RADIAL_BOUNDARY_PX = float(os.environ.get("CORESEED_FILTER_BOUNDARY_PX", "5000"))
OUTPUT_ROOT = Path(os.environ.get(
    "CORESEED_FILTER_OUTPUT_ROOT",
    str(intermediate_dir("experimental", "basic", "density_maps", "filtered_160", create=False)),
))
RUN_NAME = os.environ.get(
    "CORESEED_FILTER_RUN_NAME",
    "r160_c16_filtereq",
)
OVERWRITE = False  # Set True to replace maps and the audit TSV when run from PyCharm.
CONFIG = Path(os.environ.get(
    "CORESEED_FILTER_CONFIG",
    str(ANALYSIS / "density_maps_config_160_outlier_filtered_sim_equivalent.yaml"),
))


def filter_frames(
    susceptible: pd.DataFrame,
    resistant: pd.DataFrame,
    image_shape: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    sus = susceptible[["x", "y"]].to_numpy(dtype=float, copy=False)
    res = resistant[["x", "y"]].to_numpy(dtype=float, copy=False)
    points = np.vstack([sus, res]) if len(sus) + len(res) else np.empty((0, 2))
    mask = np.zeros(len(points), dtype=bool)
    if len(points):
        height, width = image_shape
        center = np.asarray([width / 2.0, height / 2.0])
        radial = np.linalg.norm(points - center, axis=1) <= FILTER_RADIAL_BOUNDARY_PX
        indices = np.flatnonzero(radial)
        if len(indices):
            neighbour_counts = cKDTree(points[indices]).query_ball_point(
                points[indices], r=FILTER_RADIUS_PX, return_length=True
            )
            dense = np.asarray(neighbour_counts >= FILTER_MIN_NEIGHBOURS, dtype=bool)
            mask[indices[dense]] = True
    sus_mask = mask[: len(sus)]
    res_mask = mask[len(sus) :]
    return (
        susceptible.loc[sus_mask].reset_index(drop=True),
        resistant.loc[res_mask].reset_index(drop=True),
        {
            "sus_before": len(sus), "sus_after": int(sus_mask.sum()),
            "res_before": len(res), "res_after": int(res_mask.sum()),
        },
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_ROOT / "point_filter_counts.tsv"
    if audit_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output already exists: {audit_path}. Set OVERWRITE = True to replace it."
        )
    rows: list[dict[str, object]] = []
    original = maps.create_density_map_for_timepoint

    def filtered_create(**kwargs):
        pointcloud = kwargs["pointcloud_filepath"]
        timepoint = maps.normalize_timepoint(kwargs["timepoint"])
        well = maps.well_identifier_from_pointcloud_filename(pointcloud)
        density_maps_dir = Path(kwargs["directory_path"]) / kwargs["density_maps_folder"]
        expected = [
            density_maps_dir / f"{well}_{timepoint}_{channel}_density_epanechnikov.tif"
            for channel in ("sus", "res", "total")
        ]
        if not OVERWRITE and any(path.exists() for path in expected):
            print(f"[skip] {well}, d{int(timepoint):02d}: density map exists (set OVERWRITE = True to replace).")
            return False
        sus, res = maps.pointcloud_read(
            pointcloud, timepoint, require_timepoint=kwargs.get("require_timepoint", False)
        )
        shape = maps.density_canvas_shape(
            kwargs["directory_path"],
            maps.well_identifier_from_pointcloud_filename(pointcloud),
            timepoint,
            pointcloud,
        )
        if shape is None:
            return False
        sus_filtered, res_filtered, counts = filter_frames(sus, res, shape)
        saved_reader = maps.pointcloud_read
        maps.pointcloud_read = lambda *_args, **_kwargs: (sus_filtered, res_filtered)
        try:
            result = original(**kwargs)
        finally:
            maps.pointcloud_read = saved_reader
        rows.append({
            "pointcloud": str(pointcloud), "timepoint": timepoint,
            "radius_px": FILTER_RADIUS_PX,
            "radius_um": FILTER_RADIUS_PX * maps.PIXEL_SIZE_UM,
            "min_neighbours_including_self": FILTER_MIN_NEIGHBOURS,
            "radial_boundary_px": FILTER_RADIAL_BOUNDARY_PX,
            **counts,
        })
        pd.DataFrame(rows).to_csv(audit_path, sep="\t", index=False)
        return result

    maps.create_density_map_for_timepoint = filtered_create
    maps.OVERWRITE = OVERWRITE
    sys.argv = [sys.argv[0], "--config", str(CONFIG)]
    maps.main()
    pd.DataFrame(rows).to_csv(audit_path, sep="\t", index=False)

    run_dir = OUTPUT_ROOT / RUN_NAME
    for maps_dir in sorted(run_dir.glob("Plate_*/*/density_maps")):
        well = maps_dir.parent.name
        prefix = f"{well}_"
        suffix = "_res_density_epanechnikov.tif"
        timepoints = sorted(
            path.name[len(prefix):-len(suffix)]
            for path in maps_dir.glob(f"{well}_*{suffix}")
        )
        maps.write_test_density_changes(
            directory_path=str(maps_dir.parent), well_identifier=well,
            timepoints=timepoints, density_maps_folder="density_maps",
            density_changes_folder="density_changes", channels=("res",),
            write_zero_initial_growth_channel=False,
            zero_initial_density_threshold=1.0e-10, measurable_growth_threshold=1.0e-12,
        )


if __name__ == "__main__":
    main()
