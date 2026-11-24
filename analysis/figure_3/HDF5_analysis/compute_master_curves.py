"""Compute relative- and absolute-growth variants for selection diagnostics.

The two variants differ only in the density used on the x axis. Kernel
densities are calculated once per snapshot and reused for both variants.

Defaults:
* 80, 100, 120, and 160 um Epanechnikov-kernel radii
* all-timepoint and before-escape experimental intervals
* daily simulation intervals
* CT, AT50, and AT100 only
* simulation run numbers greater than 9
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if os.fspath(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPO_ROOT))

# These must be set before importing the established HDF5-analysis modules.
_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument("--kernel-radius-um", type=float, default=None)
_bootstrap_parser.add_argument("--kernel-radii-um", nargs="+", type=float, default=None)
_bootstrap_parser.add_argument("--force", action="store_true")
_bootstrap_args, _unknown_args = _bootstrap_parser.parse_known_args()
if _bootstrap_args.kernel_radius_um is not None:
    if not math.isfinite(_bootstrap_args.kernel_radius_um) or _bootstrap_args.kernel_radius_um <= 0:
        raise ValueError("--kernel-radius-um must be positive")
    os.environ["HDF5_KERNEL_RADIUS_UM"] = f"{_bootstrap_args.kernel_radius_um:g}"
else:
    os.environ.setdefault("HDF5_KERNEL_RADIUS_UM", "120")
os.environ["HDF5_STRATEGIES"] = "CT,AT50,AT100"

import h5py
from joblib import Parallel, delayed
import numpy as np

from analysis.figure_3.HDF5_analysis import compute_flux_vs_density as simulation
from analysis.figure_3.HDF5_analysis import compute_flux_vs_density_experiment as experiment
from analysis.figure_3.HDF5_analysis.constants import (
    EXPERIMENT_N_JOBS,
    N_JOBS,
    prolif_radius_um,
    is_non_artificial_run, resistant_support, resistant_change_rate,
)
from shared.data_paths import figure_data_dir
from analysis.figure_3.HDF5_analysis.constants import (
    kernel_settings,
    validate_kernel_settings,
)


STRATEGIES = ("CT", "AT50", "AT100")
PERIODS = ("all_timepoints", "before_tte")
DEFAULT_KERNEL_RADII_UM = (120.0, 160.0)
VARIANTS = {
    "current": {
        "both_endpoint_threshold": False,
        "x_at_t0": False,
        "description": "threshold at t0; x=(rho_total(t0)+rho_total(t1))/2",
    },
    "t0_density": {
        "both_endpoint_threshold": False,
        "x_at_t0": True,
        "description": "threshold at t0; x=rho_total(t0)",
    },
}

OUTPUT_PATH = figure_data_dir(
    "Figure_3",
    "HDF5_analysis",
    "flux_vs_density",
    f"kernel_radius_{prolif_radius_um:g}um",
    "selection_diagnostics",
) / f"master_curve_selection_diagnostics_radius_{prolif_radius_um:g}um.h5"


_SNAPSHOT_LOG_DISABLED = False


def _record_snapshot(dataset, run_name, source, timepoint):
    """Leave the exact input of a native worker crash on disk, without console noise."""
    global _SNAPSHOT_LOG_DISABLED
    if _SNAPSHOT_LOG_DISABLED:
        return
    folder = OUTPUT_PATH.parent / 'run_checkpoints'
    path = folder / f'worker_{os.getpid()}_last_snapshot.json'
    temporary = path.with_suffix('.tmp')
    try:
        folder.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({
            'dataset': dataset, 'run': run_name, 'source': str(source),
            'timepoint': timepoint, 'kernel_radius_um': prolif_radius_um,
            'updated_unix': time.time(),
        }), encoding='utf-8')
        os.replace(temporary, path)
    except OSError as error:
        _SNAPSHOT_LOG_DISABLED = True
        print(f'[warn] Snapshot diagnostic disabled for worker {os.getpid()}: {error}', flush=True)


def _empty_curve(edges: np.ndarray) -> tuple[np.ndarray, ...]:
    size = len(edges) - 1
    return (
        np.full(size, np.nan),
        np.full(size, np.nan),
        np.full(size, np.nan),
        np.zeros(size),
    )


def _period_variant_curves(
    rho_total_0_parts: list[np.ndarray],
    rho_total_1_parts: list[np.ndarray],
    rho_resistant_1_parts: list[np.ndarray],
    absolute_change_parts: list[np.ndarray],
    relative_change_parts: list[np.ndarray],
    before_tte_parts: list[np.ndarray],
    threshold: float,
    edges: np.ndarray,
) -> dict[str, dict[str, dict[str, tuple[np.ndarray, ...]]]]:
    if not rho_total_0_parts:
        return {
            period: {
                name: {
                    mode: _empty_curve(edges)
                    for mode in ("absolute", "relative")
                }
                for name in VARIANTS
            }
            for period in PERIODS
        }

    rho_total_0 = np.concatenate(rho_total_0_parts)
    rho_total_1 = np.concatenate(rho_total_1_parts)
    rho_resistant_1 = np.concatenate(rho_resistant_1_parts)
    absolute_change = np.concatenate(absolute_change_parts)
    relative_change = np.concatenate(relative_change_parts)
    before_tte = np.concatenate(before_tte_parts)

    curves = {period: {} for period in PERIODS}
    period_masks = {
        "all_timepoints": np.ones(len(before_tte), dtype=bool),
        "before_tte": before_tte,
    }
    for period, period_mask in period_masks.items():
        for name, settings in VARIANTS.items():
            selected = (
                period_mask
                & np.isfinite(absolute_change)
                & np.isfinite(relative_change)
            )
            if settings["both_endpoint_threshold"]:
                selected &= rho_resistant_1 >= threshold
            x_values = (
                rho_total_0
                if settings["x_at_t0"]
                else 0.5 * (rho_total_0 + rho_total_1)
            )
            selected &= np.isfinite(x_values)
            if not np.any(selected):
                curves[period][name] = {
                    mode: _empty_curve(edges)
                    for mode in ("absolute", "relative")
                }
            else:
                curves[period][name] = {
                    "absolute": simulation._bin_curve(
                        x_values[selected], absolute_change[selected], edges
                    ),
                    "relative": simulation._bin_curve(
                        x_values[selected], relative_change[selected], edges
                    ),
                }
    return curves


def _experiment_run(
    path: str,
    run_name: str,
    escape_info: tuple[float, float] | None,
) -> dict[str, dict[str, dict[str, tuple[np.ndarray, ...]]]]:
    _, rule = experiment._sample_and_rule(path)
    cutoff = None if rule is ... else float(rule)
    rho_total_0_parts = []
    rho_total_1_parts = []
    rho_resistant_1_parts = []
    absolute_change_parts = []
    relative_change_parts = []
    before_tte_parts = []
    escape_day, pre_escape_day = (
        (None, None) if escape_info is None else escape_info
    )

    with h5py.File(path, "r") as handle:
        grid_points, grid_shape = experiment._fixed_well_grid(handle)
        days = experiment._sorted_days(handle)
        if cutoff is not None:
            days = [(day, key) for day, key in days if day <= cutoff]
        previous = None
        previous_valid = None
        previous_day = None
        for day, day_key in days:
            _record_snapshot('experiment', run_name, path, day_key)
            rho_total_1, _, rho_resistant_1, valid_1 = experiment.local_densities(
                handle, day_key, grid_points, grid_shape
            )
            if previous is not None:
                rho_total_0, rho_resistant_0 = previous
                base = resistant_support(
                    rho_total_0, rho_resistant_0, rho_total_1, rho_resistant_1,
                    previous_valid, valid_1, threshold=experiment.MIN_RES_DENSITY,
                )
                if np.any(base):
                    delta_days = day - previous_day
                    rho_total_0_parts.append(rho_total_0[base])
                    rho_total_1_parts.append(rho_total_1[base])
                    rho_resistant_1_parts.append(rho_resistant_1[base])
                    absolute_change = resistant_change_rate(
                        rho_resistant_0[base], rho_resistant_1[base], delta_days, mode="absolute",
                    )
                    absolute_change_parts.append(absolute_change)
                    relative_change_parts.append(resistant_change_rate(
                        rho_resistant_0[base], rho_resistant_1[base], delta_days, mode="relative",
                    ))
                    before_tte_parts.append(
                        np.full(
                            np.count_nonzero(base),
                            escape_day is None or day <= pre_escape_day,
                            dtype=bool,
                        )
                    )
            previous = (rho_total_1, rho_resistant_1)
            previous_valid = valid_1
            previous_day = day

    print(f"[experiment] {run_name}: {max(len(days) - 1, 0)} intervals", flush=True)
    return _period_variant_curves(
        rho_total_0_parts,
        rho_total_1_parts,
        rho_resistant_1_parts,
        absolute_change_parts,
        relative_change_parts,
        before_tte_parts,
        experiment.MIN_RES_DENSITY,
        experiment.rho_edges,
    )


def _simulation_run(
    protocol: str,
    run: str,
    tte: float | None,
) -> dict[str, dict[str, dict[str, tuple[np.ndarray, ...]]]]:
    run_dir = simulation.protocol_outputs_dir(protocol, run)
    timesteps = simulation._selected_timesteps(run_dir)
    rho_total_0_parts = []
    rho_total_1_parts = []
    rho_resistant_1_parts = []
    absolute_change_parts = []
    relative_change_parts = []
    before_tte_parts = []
    previous = None
    previous_valid = None

    for timestep in timesteps:
        _record_snapshot('simulation', f'{protocol}/{run}',
                         Path(run_dir) / f'simulation_{timestep:09d}.h5', timestep)
        rho_total_1, _, rho_resistant_1, valid_1 = simulation.local_densities(
            run_dir, timestep, use_spatial_bins=True
        )
        if previous is not None:
            rho_total_0, rho_resistant_0 = previous
            base = resistant_support(
                rho_total_0, rho_resistant_0, rho_total_1, rho_resistant_1,
                previous_valid, valid_1, threshold=simulation.MIN_RES_DENSITY,
            )
            if np.any(base):
                rho_total_0_parts.append(rho_total_0[base])
                rho_total_1_parts.append(rho_total_1[base])
                rho_resistant_1_parts.append(rho_resistant_1[base])
                absolute_change = resistant_change_rate(
                    rho_resistant_0[base], rho_resistant_1[base], 1.0, mode="absolute",
                )
                absolute_change_parts.append(absolute_change)
                relative_change_parts.append(resistant_change_rate(
                    rho_resistant_0[base], rho_resistant_1[base], 1.0, mode="relative",
                ))
                day = (
                    timestep / simulation.time_dump
                    + simulation.escape_core.sim.DEFAULT_DAY_OFFSET
                )
                before_tte_parts.append(
                    np.full(
                        np.count_nonzero(base),
                        tte is None or day < tte,
                        dtype=bool,
                    )
                )
        previous = (rho_total_1, rho_resistant_1)
        previous_valid = valid_1

    print(f"[simulation] {protocol}/{run}: {max(len(timesteps) - 1, 0)} intervals", flush=True)
    return _period_variant_curves(
        rho_total_0_parts,
        rho_total_1_parts,
        rho_resistant_1_parts,
        absolute_change_parts,
        relative_change_parts,
        before_tte_parts,
        simulation.MIN_RES_DENSITY,
        simulation.rho_edges,
    )


def _simulation_run_number(run: str) -> int | None:
    match = re.fullmatch(r"run_(\d+)", run)
    return None if match is None else int(match.group(1))


def _cached_run(dataset, strategy, run_name, arguments):
    """Checkpoint completed wells/runs atomically, before the strategy finishes."""
    if dataset == 'experiment':
        sources = [Path(arguments[0])]
        compute = _experiment_run
    else:
        sources = sorted(Path(simulation.protocol_outputs_dir(arguments[0], arguments[1])).glob('simulation_*.h5'))
        compute = _simulation_run
    source_state = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in sources]
    # Changing analysis code, raw input bytes/mtime, event timing, or radius
    # invalidates a checkpoint rather than silently using incompatible curves.
    code = hashlib.sha256()
    for path in (Path(__file__), Path(simulation.__file__), Path(experiment.__file__),
                 _REPO_ROOT / 'analysis/figure_3/HDF5_analysis/constants.py',
                 _REPO_ROOT / 'shared/experiment_filters.py'):
        code.update(path.read_bytes())
    signature = hashlib.sha256(json.dumps(
        [source_state, arguments, kernel_settings(prolif_radius_um), code.hexdigest()],
        default=str, sort_keys=True,
    ).encode()).hexdigest()
    token = hashlib.sha256(f'{dataset}/{strategy}/{run_name}'.encode()).hexdigest()[:20]
    path = OUTPUT_PATH.parent / 'run_checkpoints' / f'{dataset}_{token}.h5'
    if path.is_file() and not _bootstrap_args.force:
        try:
            with h5py.File(path, 'r') as handle:
                if handle.attrs.get('signature') == signature:
                    result = {
                        period: {variant: {mode: tuple(
                            handle[f'{period}/{variant}/{mode}/{index}'][:]
                            for index in range(4)
                        ) for mode in ('absolute', 'relative')} for variant in VARIANTS}
                        for period in PERIODS
                    }
                    if all(array.shape == (len(simulation.rho_edges) - 1,)
                           for variants in result.values() for modes in variants.values()
                           for arrays in modes.values() for array in arrays):
                        return result
        except (OSError, KeyError, ValueError):
            pass
    result = compute(*arguments)
    if source_state != [(str(item.resolve()), item.stat().st_size, item.stat().st_mtime_ns)
                        for item in sources]:
        raise RuntimeError(f'Inputs changed during {dataset}/{run_name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        with h5py.File(temporary, 'w') as handle:
            handle.attrs['signature'] = signature
            handle.attrs['run_name'] = run_name
            for period, variants in result.items():
                for variant, modes in variants.items():
                    for mode, arrays in modes.items():
                        group = handle.require_group(f'{period}/{variant}/{mode}')
                        for index, array in enumerate(arrays):
                            group.create_dataset(str(index), data=array)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


def _complete_strategy(output, dataset, strategy, run_names=None):
    try:
        group = output[f'{dataset}/{strategy}']
        names = group['run_names'].asstr()[:].tolist()
        if not names or (run_names is not None and names != run_names):
            return False
        for period in PERIODS:
            for variant in VARIANTS:
                for name in ('median', 'q25', 'q75', 'count',
                             'median_rel', 'q25_rel', 'q75_rel', 'count_rel'):
                    if group[f'{period}/{variant}/{name}'].shape != (len(names), len(simulation.rho_edges) - 1):
                        return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _write_strategy(
    output: h5py.File,
    dataset: str,
    strategy: str,
    run_names: list[str],
    results: list[
        dict[str, dict[str, dict[str, tuple[np.ndarray, ...]]]]
    ],
) -> None:
    parent = output.require_group(dataset)
    if strategy in parent:
        del parent[strategy]
    strategy_group = parent.create_group(strategy)
    strategy_group.create_dataset("run_names", data=np.asarray(run_names, dtype="S"))
    for period in PERIODS:
        period_group = strategy_group.create_group(period)
        for variant in VARIANTS:
            variant_group = period_group.create_group(variant)
            for array_name, mode, index in (
                ("median", "absolute", 0),
                ("q25", "absolute", 1),
                ("q75", "absolute", 2),
                ("count", "absolute", 3),
                ("median_rel", "relative", 0),
                ("q25_rel", "relative", 1),
                ("q75_rel", "relative", 2),
                ("count_rel", "relative", 3),
            ):
                variant_group.create_dataset(
                    array_name,
                    data=np.stack([
                        result[period][variant][mode][index]
                        for result in results
                    ]),
                    compression="gzip",
                )
    output.flush()


def _compute_experiments(output: h5py.File) -> None:
    escape_table = experiment.escape_core.experimental_escape_cutoffs(
        experiment.escape_core.DEFAULT_EXPERIMENT_ESCAPE,
        experiment.escape_core.DEFAULT_ESCAPE_MARGIN_CSV,
    )
    escape_lookup = {
        (str(row.plate), str(row.well)): (
            float(row.escape_day),
            float(row.pre_escape_cutoff_day),
        )
        for row in escape_table.itertuples(index=False)
    }
    runs_by_strategy = experiment.discover_runs()
    for strategy in STRATEGIES:
        runs = runs_by_strategy.get(strategy, {})
        run_names = sorted(runs)
        if not run_names:
            raise RuntimeError(f"No experimental runs found for {strategy}")
        if not os.environ.get('PIPELINE_REFRESH_MASTER_CURVES') and _complete_strategy(output, 'experiment', strategy, run_names):
            print(f'[reuse] experiment/{strategy}: {len(run_names)} completed wells', flush=True)
            continue
        print(f"[experiment] {strategy}: {len(run_names)} wells", flush=True)
        results = Parallel(n_jobs=EXPERIMENT_N_JOBS, backend="loky")(
            delayed(_cached_run)('experiment', strategy, name, (
                runs[name],
                name,
                escape_lookup.get(
                    (
                        experiment._plate_name(runs[name]),
                        experiment._sample_and_rule(runs[name])[0],
                    )
                ),
            ))
            for name in run_names
        )
        _write_strategy(output, "experiment", strategy, run_names, results)


def _compute_simulations(output: h5py.File) -> None:
    escape_days = simulation.escape_core.simulation_escape_days(
        simulation.escape_core.DEFAULT_SIM_ESCAPE,
        simulation.escape_core.DEFAULT_ESCAPE_MARGIN_CSV,
    )
    available = {name.upper(): name for name in simulation.discover_simulation_protocols()}
    for strategy in STRATEGIES:
        protocol = available.get(strategy)
        if protocol is None:
            raise RuntimeError(f"No simulation protocol found for {strategy}")
        outputs_dir = Path(simulation.protocol_outputs_dir(protocol, ""))
        run_names = []
        for path in sorted(outputs_dir.iterdir()):
            run_number = _simulation_run_number(path.name)
            if path.is_dir() and is_non_artificial_run(path.name):
                if len(simulation._selected_timesteps(os.fspath(path))) >= 2:
                    run_names.append(path.name)
        if not run_names:
            raise RuntimeError(f"No non-artificial simulation runs found for {strategy}")
        if not os.environ.get('PIPELINE_REFRESH_MASTER_CURVES') and _complete_strategy(output, 'simulation', strategy, run_names):
            print(f'[reuse] simulation/{strategy}: {len(run_names)} completed runs', flush=True)
            continue
        strategy_key = simulation.escape_core.canonical_strategy_label(
            simulation.escape_core.SOURCE_TO_DISPLAY.get(protocol, protocol)
        )
        print(f"[simulation] {strategy}: {len(run_names)} runs (>9)", flush=True)
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_cached_run)('simulation', strategy, run, (
                protocol,
                run,
                escape_days.get(
                    (
                        strategy_key,
                        simulation.escape_core.normalize_simulation_run_id(run),
                    )
                ),
            ))
            for run in run_names
        )
        _write_strategy(output, "simulation", strategy, run_names, results)


def _is_complete_output(path: Path) -> bool:
    """Return whether *path* contains every requested diagnostic curve."""
    try:
        with h5py.File(path, "r") as handle:
            validate_kernel_settings(handle.attrs, prolif_radius_um, source=path)
            for dataset in ("experiment", "simulation"):
                for strategy in STRATEGIES:
                    strategy_group = handle[dataset][strategy]
                    run_count = len(strategy_group["run_names"])
                    if run_count == 0:
                        return False
                    for period in PERIODS:
                        for variant in VARIANTS:
                            variant_group = strategy_group[period][variant]
                            shapes = [
                                variant_group[name].shape
                                for name in (
                                    "median", "q25", "q75", "count",
                                    "median_rel", "q25_rel", "q75_rel", "count_rel",
                                )
                            ]
                            if any(
                                len(shape) != 2 or shape[0] != run_count
                                for shape in shapes
                            ):
                                return False
                            if len(set(shapes)) != 1:
                                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _finalize_output(temporary_path: Path) -> None:
    """Promote a complete building file despite transient Windows/SMB locks."""
    if not _is_complete_output(temporary_path):
        raise RuntimeError(f"Incomplete diagnostic building file: {temporary_path}")

    gc.collect()
    for attempt in range(1, 11):
        try:
            temporary_path.replace(OUTPUT_PATH)
            print(f"[saved] {OUTPUT_PATH}", flush=True)
            return
        except PermissionError:
            print(
                f"[finalize] file is temporarily locked; retry {attempt}/10",
                flush=True,
            )
            time.sleep(1.0)

    # Some SMB clients allow reading a recently closed HDF5 file before they
    # allow renaming it.  Copying preserves the completed analysis and avoids
    # forcing an expensive recomputation.
    print("[finalize] rename remained locked; copying completed HDF5 instead", flush=True)
    try:
        shutil.copyfile(temporary_path, OUTPUT_PATH)
    except OSError as error:
        raise PermissionError(
            "The completed .building.h5 could neither be renamed nor copied. "
            "Close programs that have the file open and rerun this script; it "
            "will reuse the completed building file."
        ) from error
    if not _is_complete_output(OUTPUT_PATH):
        raise RuntimeError(f"Copied diagnostic output failed validation: {OUTPUT_PATH}")
    try:
        temporary_path.unlink()
    except PermissionError:
        print(
            f"[finalize] kept locked building file as a backup: {temporary_path}",
            flush=True,
        )
    print(f"[saved] {OUTPUT_PATH}", flush=True)


def _run_all_kernel_radii() -> None:
    radii = (
        DEFAULT_KERNEL_RADII_UM
        if _bootstrap_args.kernel_radii_um is None
        else tuple(_bootstrap_args.kernel_radii_um)
    )
    if not radii or any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("--kernel-radii-um values must all be positive")
    for radius in radii:
        print(f"\n[kernel] starting radius {radius:g} um", flush=True)
        environment = os.environ.copy()
        environment["HDF5_KERNEL_RADIUS_UM"] = f"{radius:g}"
        command = [
            sys.executable,
            os.fspath(Path(__file__).resolve()),
            "--kernel-radius-um",
            f"{radius:g}",
        ]
        if _bootstrap_args.force:
            command.append("--force")
        subprocess.run(
            command,
            check=True,
            env=environment,
        )
        print(f"[kernel] completed radius {radius:g} um", flush=True)


def _compute_one_radius() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if (
        not _bootstrap_args.force
        and not os.environ.get('PIPELINE_REFRESH_MASTER_CURVES')
        and OUTPUT_PATH.exists()
        and _is_complete_output(OUTPUT_PATH)
    ):
        print(f"[reuse] complete output already exists: {OUTPUT_PATH}", flush=True)
        return
    temporary_path = OUTPUT_PATH.with_suffix(".building.h5")
    if temporary_path.exists():
        if not _bootstrap_args.force and not os.environ.get('PIPELINE_REFRESH_MASTER_CURVES') and _is_complete_output(temporary_path):
            print(
                "[resume] found a complete building file; finalizing it without "
                "recomputing densities",
                flush=True,
            )
            _finalize_output(temporary_path)
            return
        if _bootstrap_args.force:
            temporary_path.unlink()
        else:
            # A worker failure normally leaves a readable HDF5 containing
            # completed strategies. Preserve those instead of truncating it.
            with h5py.File(temporary_path, 'r') as previous:
                validate_kernel_settings(previous.attrs, prolif_radius_um, source=temporary_path)
            print('[resume] retaining completed strategies in the building file', flush=True)
    with h5py.File(temporary_path, "a") as output:
        output.attrs.update(kernel_settings(prolif_radius_um))
        output.attrs["strategies"] = ",".join(STRATEGIES)
        output.attrs["periods"] = ",".join(PERIODS)
        output.attrs["growth_modes"] = "absolute,relative"
        output.attrs["variants"] = ",".join(VARIANTS)
        output.attrs["simulation_run_selection"] = "run number > 9"
        for name, settings in VARIANTS.items():
            output.attrs[f"variant_{name}"] = settings["description"]
        _compute_experiments(output)
        _compute_simulations(output)
        output.flush()
    _finalize_output(temporary_path)


def main() -> None:
    if _bootstrap_args.kernel_radius_um is None:
        _run_all_kernel_radii()
    else:
        _compute_one_radius()


if __name__ == "__main__":
    main()
