"""Direct verification of the two internal Figure 3 master-curve helpers."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.figure_3.Legacy import aggregate_average_symmetric_condition_matrix_160 as statistics
from analysis.figure_3 import density_common as density_io
from shared.data_paths import VERIFICATION_DATA


DEFAULT_VERIFICATION_ROOT = VERIFICATION_DATA
OUTPUT_FOLDER = "_master_curve_helpers"
SEED_LABEL = "sample_seed20260831"


def check_naming_and_normalisation() -> None:
    maps_dir = Path("density_maps")
    assert density_io.strategy_from_plate_and_well("Plate_4", "D5_AT50-D5") == "AT50"
    assert density_io.strategy_from_plate_and_well("Plate_CT", "unrecognised") == "CT"
    assert density_io.timepoint_to_day("12") == 12.0
    assert math.isnan(density_io.timepoint_to_day("day_12"))
    assert density_io.density_map_path(maps_dir, "D5_AT50-D5", "12").name == "D5_AT50-D5_12_total_density_epanechnikov.tif"
    assert density_io.resistant_density_map_path(maps_dir, "D5_AT50-D5", "12").name == "D5_AT50-D5_12_res_density_epanechnikov.tif"
    assert density_io.normalize_simulation_run_id("run_0007") == "7"
    assert density_io.normalize_simulation_run_id("Run-08") == "8"
    assert density_io.normalize_simulation_run_id("custom_run") == "custom_run"


def check_csv_loaders() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        escape_path = root / "escape.csv"
        pd.DataFrame(
            {
                "plate": ["Plate_1", "Plate_1", "Plate_2"],
                "replicate": ["A1_CT-A1", "A1_CT-A1", "B2_AT50-B2"],
                "escape_day": [10, 12, 8],
                "last_confined_day": [8, 11, np.nan],
            }
        ).to_csv(escape_path, index=False)
        cutoffs = density_io.load_escape_cutoffs(escape_path).set_index(["plate", "well"])
        assert cutoffs.loc[("Plate_1", "A1_CT-A1"), "escape_day"] == 10
        assert cutoffs.loc[("Plate_1", "A1_CT-A1"), "pre_escape_cutoff_day"] == 8
        assert cutoffs.loc[("Plate_2", "B2_AT50-B2"), "pre_escape_cutoff_day"] == 8

        tte_path = root / "tte.csv"
        pd.DataFrame(
            {
                "protocol": ["CT", "CT", "AT50", "AT50"],
                "run": ["run_0007", "7", "run_0002", "run_0003"],
                "TTE": [14, 14, 0, np.nan],
            }
        ).to_csv(tte_path, index=False)
        positive_tte = density_io.load_positive_tte(tte_path)
        assert positive_tte == {("CT", "7"): 14.0}


def check_simulation_density_helpers() -> None:
    positions_a = np.array([[10.0, 20.0], [30.0, 40.0]])
    positions_b = np.array([[15.0, 25.0]])
    canvas = density_io.canvas_from_loaded([(positions_a, np.array([0, 1])), (positions_b, np.array([0]))], 10.0)
    assert canvas.min_x_um == 0.0 and canvas.min_y_um == 10.0
    assert canvas.width_px >= density_io.sim.PIXEL_SIZE_UM
    assert canvas.height_px >= density_io.sim.PIXEL_SIZE_UM

    local_canvas = density_io.sim.CanvasSpec(min_x_um=0.0, min_y_um=0.0, width_px=32, height_px=32)
    susceptible = pd.DataFrame({"x": [8.0], "y": [8.0]})
    resistant = pd.DataFrame({"x": [16.0], "y": [16.0]})
    resistant_density, total_density = density_io.resistant_and_total_density(
        susceptible, resistant, radius_um=8.0, canvas=local_canvas, coarsening_factor=4
    )
    assert resistant_density.shape == total_density.shape == (8, 8)
    assert np.isfinite(resistant_density).all() and np.isfinite(total_density).all()
    # The two FFT convolutions are evaluated separately in float32, so allow
    # their sub-nanodensity rounding difference while requiring total >= resistant.
    assert np.min(total_density - resistant_density) >= -1e-8


def check_binned_accumulator() -> None:
    accumulator = statistics.Aggregator.create(
        n_strategies=1,
        x_edges=np.array([0.0, 1.0, 2.0]),
        ylim=(-1.0, 1.0),
        n_y_bins=4,
    )
    accumulator.add(0, np.array([0, 0, 1]), np.array([-0.9, 0.0, 0.9]))
    assert np.array_equal(accumulator.count[0], np.array([2, 1]))
    assert np.allclose(accumulator.total[0], np.array([-0.9, 0.9]))
    assert np.allclose(accumulator.total_sq[0], np.array([0.81, 0.81]))
    assert np.array_equal(accumulator.histogram[0].sum(axis=1), accumulator.count[0])


def check_histogram_quantiles() -> None:
    counts = np.array([1, 2, 1])
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    expected = {0.0: 1.0, 0.25: 1.375, 0.5: 1.75, 0.75: 2.25, 1.0: 3.0}
    for quantile, value in expected.items():
        assert np.isclose(statistics.histogram_quantile(counts, edges, quantile), value)
    assert math.isnan(statistics.histogram_quantile(np.zeros(3, dtype=int), edges, 0.5))


def main() -> None:
    review_dir = DEFAULT_VERIFICATION_ROOT / OUTPUT_FOLDER / SEED_LABEL
    review_dir.mkdir(parents=True, exist_ok=True)
    checks = {
        "naming_and_normalisation": check_naming_and_normalisation,
        "csv_loaders": check_csv_loaders,
        "simulation_density_helpers": check_simulation_density_helpers,
        "binned_accumulator": check_binned_accumulator,
        "histogram_quantiles": check_histogram_quantiles,
    }
    results: dict[str, dict[str, object]] = {}
    with tqdm(total=len(checks), desc="Verifying Figure 3 master-curve helpers", unit="check") as progress:
        for name, check in checks.items():
            progress.set_postfix_str(name)
            try:
                check()
            except Exception as error:  # Keep all check outcomes in one audit file.
                results[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            else:
                results[name] = {"passed": True}
            progress.update()

    passed = all(result["passed"] for result in results.values())
    (review_dir / "README.md").write_text(
        "# Figure 3 master-curve helper verification\n\n"
        "This check verifies the small shared helper modules used by the Figure 3 master-curve aggregation. "
        "It checks that file names, strategy names, days, escape cutoffs, and simulation run labels are interpreted consistently. "
        "It also checks that the binned summaries count values correctly and calculate the displayed median and quartiles from their histogram.\n\n"
        "The checks use small known examples, so every expected result is exact. A PASS means the helper returned those expected values. "
        "This complements the aggregate verification, which checks the full transformation of real density-map intervals.\n",
        encoding="utf-8",
    )
    (review_dir / "metadata.json").write_text(
        json.dumps(
            {
                "scripts_under_test": [
                    "analysis/figure_3/density_common.py",
                    "analysis/figure_3/Legacy/aggregate_average_symmetric_condition_matrix_160.py",
                ],
                "passed": passed,
                "checks": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{'PASS' if passed else 'FAIL'}: {review_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
