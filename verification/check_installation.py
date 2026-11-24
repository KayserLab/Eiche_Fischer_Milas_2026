#!/usr/bin/env python3
"""Check the root environment and canonical data-root wiring without analysis."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared import data_paths  # noqa: E402
from shared import figtools  # noqa: E402
from analysis.basic.density_maps import (  # noqa: E402
    Density_Maps_Epanechnikov_coarsegrained as experimental_density,
)


REQUIRED_IMPORTS = (
    "cv2",
    "h5py",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "sklearn",
    "tables",
    "tifffile",
    "tqdm",
    "yaml",
)


PIPELINE_MODULES = (
    "analysis.basic.microscopy.Make_susrestotal_Colony_Masks",
    "analysis.basic.microscopy.single_clone_tracking",
    "analysis.basic.escape.Escape_Remove_Sen_Artifacts",
    "analysis.basic.escape.Prepare_Escape_Analysis_Inputs",
    "analysis.basic.density_maps.Prepare_Simulation_Pointclouds",
    "analysis.basic.density_maps.Density_Maps_Epanechnikov_simulations_coarsegrained",
    "analysis.figure_2.build_simulation_overview",
    "analysis.figure_2.Escape_Margin_Analysis",
    "analysis.figure_3.generate_figure2_mask_filtered_density_maps",
    "analysis.figure_3.density_common",
    "analysis.figure_4.generate_figure4_whole_well_density_maps",
    "analysis.figure_4.run_rl_policy_behavior_analysis",
)


def check_archive_topology() -> None:
    """Check representative deposited inputs without opening scientific data."""
    print("[preflight] checking deposited input topology", flush=True)
    required_directories = [
        data_paths.EXP_RAW_DATA / "alamar_blue",
        data_paths.EXP_RAW_DATA / "Plate_4" / "Tifs",
        data_paths.SIMULATION_DATABASE,
    ]
    for plate_number in range(1, 9):
        plate = data_paths.EXP_RAW_DATA / f"Plate_{plate_number}"
        required_directories.extend((plate / "hdf5", plate / "czi_NPY", plate / "Results"))
        schedule = plate / "treatment_schedule.txt"
        if not schedule.is_file():
            raise FileNotFoundError(f"Treatment schedule missing: {schedule}")

    missing_directories = [path for path in required_directories if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError(
            "Required deposited input directories are missing: "
            + ", ".join(os.fspath(path) for path in missing_directories)
        )

    for strategy in ("Agent", "AT100", "AT50", "CT", "Dens"):
        outputs = data_paths.SIMULATION_DATABASE / f"data-DB_{strategy}" / "outputs"
        if not outputs.is_dir():
            raise FileNotFoundError(f"Simulation outputs directory missing: {outputs}")

    experimental_agent = (
        REPOSITORY_ROOT
        / "PhysiLearning"
        / "Agent_used_for_experiment"
        / "2025_10_01_lv_dont_treat_1_load_best_reward"
    )
    for name in ("data", "policy.pth", "pytorch_variables.pth"):
        if not (experimental_agent / name).is_file():
            raise FileNotFoundError(f"Experimental PPO artifact missing: {experimental_agent / name}")
    bootstrap_models = (
        REPOSITORY_ROOT
        / "PhysiLearning"
        / "Bootstrapping"
        / "outputs"
        / "bootstrap_n20_methods_seed_maxiter1000"
        / "agent_training"
    )
    if not bootstrap_models.is_dir():
        raise FileNotFoundError(f"Bootstrap agent-training directory missing: {bootstrap_models}")


def check_pipeline_entrypoints() -> None:
    """Import active modules without invoking their analysis entry points."""
    for module_name in PIPELINE_MODULES:
        print(f"[preflight] importing {module_name}", flush=True)
        importlib.import_module(module_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Also require the configured experimental and simulation input directories.",
    )
    parser.add_argument(
        "--pipeline-preflight",
        action="store_true",
        help=(
            "Read-only check of deposited input topology, active module imports, and "
            "path wiring. It does not run analysis or write outputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)

    if figtools.PAPER_FIGURE_ROOT != data_paths.FIGURE_PLOTS:
        raise RuntimeError(
            "Figure output root is not connected to the canonical data root: "
            f"{figtools.PAPER_FIGURE_ROOT} != {data_paths.FIGURE_PLOTS}"
        )

    config_paths = (
        REPOSITORY_ROOT
        / "analysis/basic/density_maps/density_maps_epanechnikov_config_160_all_strategies_all_plates.yaml",
        REPOSITORY_ROOT
        / "analysis/figure_3/density_maps_config_160_figure2_masks_all_clones.yaml",
        REPOSITORY_ROOT
        / "analysis/figure_4/density_maps_config_160_outlier_filtered_sim_equivalent.yaml",
    )
    for config_path in config_paths:
        config = experimental_density.load_config(config_path)
        configured_experiment = Path(config["experiment_path"])
        configured_output = Path(config["output_root"])
        if configured_experiment != data_paths.EXP_RAW_DATA:
            raise RuntimeError(
                "Density-map YAML did not rebase to the canonical input root: "
                f"{configured_experiment} != {data_paths.EXP_RAW_DATA}"
            )
        if not configured_output.is_relative_to(data_paths.DATA_ROOT):
            raise RuntimeError(
                "Density-map YAML did not rebase below the canonical output root: "
                f"{configured_output} is outside {data_paths.DATA_ROOT}"
            )

    if args.require_data:
        missing = [
            path
            for path in (data_paths.EXP_RAW_DATA, data_paths.SIMULATION_DATABASE)
            if not path.is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "Required Figshare input directories are missing: "
                + ", ".join(os.fspath(path) for path in missing)
            )

    if args.pipeline_preflight:
        check_archive_topology()
        check_pipeline_entrypoints()

    source = "environment variable" if os.environ.get(data_paths.DATA_ROOT_ENV) else "shared/data_paths.py"
    print("PASS: root dependencies imported and canonical path wiring is consistent.")
    if args.pipeline_preflight:
        print("PASS: deposited input topology and active module imports passed preflight.")
    print(f"Data root ({source}): {data_paths.DATA_ROOT}")


if __name__ == "__main__":
    main()
