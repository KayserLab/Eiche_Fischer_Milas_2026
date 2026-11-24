#!/usr/bin/env python3
"""
Run the Epanechnikov analysis stages selected in the YAML config.

The stage scripts keep the actual analysis logic. This runner only reads the
same config, decides which stages are enabled, runs them serially, and reports
elapsed time per stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_RUN_NAME_FORMAT,
    OUTPUT_RUN_NAME_ENV,
    load_config,
)

PIPELINE_OUTPUT_ALIASES = {
    "density_maps": "density_maps",
    "maps": "density_maps",
    "density_stacks": "density_stacks",
    "stacks": "density_stacks",
    "density_changes": "density_changes",
    "changes": "density_changes",
    "change_scatter_plots": "change_scatter_plots",
    "scatter_plots": "change_scatter_plots",
    "scatter": "change_scatter_plots",
    "hdf5_scatter_movie": "hdf5_scatter_movie",
    "scatter_movie": "hdf5_scatter_movie",
    "movie": "hdf5_scatter_movie",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected Epanechnikov pipeline stages from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help=f"YAML config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without running them.",
    )
    return parser.parse_args()


def config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_config_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return [str(item) for item in value]


def selected_pipeline_outputs(config: dict) -> set[str] | None:
    if "pipeline_outputs" not in config:
        return None

    outputs: set[str] = set()
    for value in normalize_config_list(config.get("pipeline_outputs")):
        key = value.strip().lower()
        if key == "all":
            return set(PIPELINE_OUTPUT_ALIASES.values())
        if key not in PIPELINE_OUTPUT_ALIASES:
            valid = ", ".join(sorted(set(PIPELINE_OUTPUT_ALIASES.values())))
            raise ValueError(
                f"Unsupported pipeline_outputs entry {value!r}. "
                f"Use one of: {valid}."
            )
        outputs.add(PIPELINE_OUTPUT_ALIASES[key])
    return outputs


def pipeline_output_enabled(
    config: dict,
    output_key: str,
    legacy_bool_key: str,
    default: bool,
) -> bool:
    selected_outputs = selected_pipeline_outputs(config)
    if selected_outputs is not None:
        return output_key in selected_outputs
    return config_bool(config, legacy_bool_key, default)


def add_optional_arg(
    command: list[str],
    *,
    config: dict,
    config_key: str,
    arg_name: str,
) -> None:
    value = config.get(config_key)
    if value is not None:
        command.extend([arg_name, str(value)])


def resolve_pipeline_run_name(config: dict) -> str:
    configured = (
        os.environ.get(OUTPUT_RUN_NAME_ENV)
        or config.get("output_run_name")
        or config.get("run_name")
    )
    if configured is not None:
        configured = str(configured).strip()
        if configured and configured.lower() not in {"auto", "datetime", "timestamp"}:
            return configured
    return datetime.now().strftime(DEFAULT_OUTPUT_RUN_NAME_FORMAT)


def selected_stages(config: dict, config_path: Path) -> list[tuple[str, list[str]]]:
    script_dir = Path(__file__).resolve().parent
    analysis_dir = script_dir.parent
    python = sys.executable
    stages: list[tuple[str, list[str]]] = []

    create_maps = pipeline_output_enabled(config, "density_maps", "create_maps", True)
    stack_maps = pipeline_output_enabled(
        config,
        "density_stacks",
        "stack_maps",
        True,
    )
    if create_maps or stack_maps:
        stages.append(
            (
                "density maps/stacks",
                [
                    python,
                    os.fspath(analysis_dir / "Density_Maps_Epanechnikov_coarsegrained.py"),
                    "--config",
                    os.fspath(config_path),
                ],
            )
        )

    create_density_changes = pipeline_output_enabled(
        config,
        "density_changes",
        "create_density_changes",
        True,
    )
    create_change_scatter_plots = pipeline_output_enabled(
        config,
        "change_scatter_plots",
        "create_change_scatter_plots",
        True,
    )
    if create_density_changes or create_change_scatter_plots:
        stages.append(
            (
                "density changes/scatter plots",
                [
                    python,
                    os.fspath(script_dir / "abolute_res_change_vs_total_density_coarsegrained.py"),
                    "--config",
                    os.fspath(config_path),
                ],
            )
        )

    if pipeline_output_enabled(
        config,
        "hdf5_scatter_movie",
        "create_hdf5_scatter_movie",
        False,
    ):
        movie_command = [
            python,
            os.fspath(script_dir / "hdf5_cell_position_scatter_movie_coarsegrained.py"),
            "--config",
            os.fspath(config_path),
        ]
        if config_bool(config, "hdf5_scatter_movie_overwrite", False):
            movie_command.append("--overwrite")
        add_optional_arg(
            movie_command,
            config=config,
            config_key="hdf5_scatter_movie_fps",
            arg_name="--fps",
        )
        add_optional_arg(
            movie_command,
            config=config,
            config_key="hdf5_scatter_movie_dpi",
            arg_name="--dpi",
        )
        add_optional_arg(
            movie_command,
            config=config,
            config_key="hdf5_scatter_movie_marker_size",
            arg_name="--marker-size",
        )
        add_optional_arg(
            movie_command,
            config=config,
            config_key="hdf5_scatter_movie_scale_bar_um",
            arg_name="--scale-bar-um",
        )
        stages.append(("HDF5 scatter movie", movie_command))

    return stages


def format_seconds(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {remainder:.1f}s"
    return f"{seconds:.2f}s"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    stages = selected_stages(config, config_path)

    if not stages:
        print("No pipeline stages enabled in the config.")
        return

    output_run_name = resolve_pipeline_run_name(config)
    print(f"Using config: {config_path}")
    print(f"Using output run name: {output_run_name}")
    if args.dry_run:
        print("Dry run; commands selected from YAML:")
        for stage_name, command in stages:
            print(f"- {stage_name}: {shlex.join(command)}")
        return

    timings: list[tuple[str, float]] = []
    total_start = time.perf_counter()
    repo_dir = Path(__file__).resolve().parent.parent
    stage_env = os.environ.copy()
    stage_env[OUTPUT_RUN_NAME_ENV] = output_run_name

    for stage_name, command in stages:
        print(f"\n=== Running {stage_name} ===")
        stage_start = time.perf_counter()
        subprocess.run(command, cwd=repo_dir, env=stage_env, check=True)
        elapsed = time.perf_counter() - stage_start
        timings.append((stage_name, elapsed))
        print(f"=== Finished {stage_name} in {format_seconds(elapsed)} ===")

    total_elapsed = time.perf_counter() - total_start
    print("\nPipeline timing summary:")
    for stage_name, elapsed in timings:
        print(f"- {stage_name}: {format_seconds(elapsed)}")
    print(f"- total: {format_seconds(total_elapsed)}")


if __name__ == "__main__":
    main()
