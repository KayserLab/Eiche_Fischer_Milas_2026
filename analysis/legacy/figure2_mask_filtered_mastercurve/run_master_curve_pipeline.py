#!/usr/bin/env python3
r"""Build a separate Figure 3 master curve with the Figure 2 mask filter.

Outputs are deliberately isolated under
``F:\CoreSeed_InVitro_160_Figure2MaskFiltered_AllClones``.  Existing Figure 3
maps, caches, scripts, and figures are neither replaced nor edited.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = Path(r"F:\CoreSeed_InVitro_160_Figure2MaskFiltered_AllClones")
RUN_NAME = "run_kernel_160_coarse_16_figure2_masks_all_clones"
RUN_DIR = OUTPUT_ROOT / RUN_NAME
CONFIG = Path(__file__).with_name("density_maps_config_160_figure2_masks_all_clones.yaml")
MAP_GENERATOR = Path(__file__).with_name("generate_density_maps.py")
EXPERIMENTAL_AGGREGATOR = Path(__file__).with_name("aggregate_experimental_with_reused_simulation.py")
MASTER_PLOT = ROOT / "figures" / "Figure_3" / "plot_master_curve_sim_exp.py"
SOURCE_SIMULATION_TABLES = Path(
    r"F:\CoreSeed_InVitro_160_Experimental_Outlier_Filter_SimEquivalent_r237p8px_min3"
    r"\plots_before_escape_gate_matrix"
)
SOURCE_SIMULATION_HISTOGRAM_CACHE = Path(
    r"F:\CoreSeed_InVitro_160_Experimental_Outlier_Filter_SimEquivalent_r237p8px_min3"
    r"\aggregation_cache_before_escape_histogram_xbin_0p05"
)


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment, cwd=ROOT)


def aggregate_experimental_only(*, bin_width: str, cache_root: Path, table_root: Path) -> None:
    environment = os.environ.copy()
    environment["CORESEED_X_BIN_WIDTH"] = bin_width
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), environment.get("PYTHONPATH", "")))
    run([
        sys.executable, str(EXPERIMENTAL_AGGREGATOR),
        "--in-vitro-run", str(RUN_DIR),
        "--config", str(CONFIG),
        "--cache-root", str(cache_root),
        "--output-root", str(table_root),
        "--source-simulation-tables", str(SOURCE_SIMULATION_TABLES),
        "--source-simulation-cache", str(SOURCE_SIMULATION_HISTOGRAM_CACHE),
    ], environment=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-density-maps", action="store_true",
        help="Reuse the already completed mask-filtered density-map run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The upstream density-map writer prints Unicode progress markers.  Force
    # UTF-8 so redirected background logs work on Windows' cp1252 consoles.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.skip_density_maps:
        if not RUN_DIR.is_dir():
            raise FileNotFoundError(f"Cannot skip density maps; run is missing: {RUN_DIR}")
        print(f"[reuse] completed mask-filtered density maps: {RUN_DIR}", flush=True)
    else:
        run([sys.executable, str(MAP_GENERATOR)])

    table_root = OUTPUT_ROOT / "master_curve_tables_xbin_0p2"
    histogram_table_root = OUTPUT_ROOT / "master_curve_tables_xbin_0p05"
    histogram_cache_root = OUTPUT_ROOT / "master_curve_histograms_xbin_0p05"
    aggregate_experimental_only(
        bin_width="0.2",
        cache_root=OUTPUT_ROOT / "master_curve_cache_xbin_0p2",
        table_root=table_root,
    )
    aggregate_experimental_only(
        bin_width="0.05",
        cache_root=histogram_cache_root,
        table_root=histogram_table_root,
    )

    # Reuse the existing Figure 3 renderer without changing it; point it to
    # this independent set of tables and histogram caches at runtime.
    from figures.Figure_3.Figure_3_Legacy import plot_master_curve_sim_exp as master_plot

    master_plot.USER_HISTOGRAM_CACHE_ROOT = histogram_cache_root
    sys.argv = [
        str(MASTER_PLOT),
        "--data-root", str(table_root),
        "--output-root", str(OUTPUT_ROOT / "master_curve_plot_outputs"),
        "--figure3-output", str(OUTPUT_ROOT / "master_curve_figures"),
    ]
    master_plot.main()


if __name__ == "__main__":
    main()
