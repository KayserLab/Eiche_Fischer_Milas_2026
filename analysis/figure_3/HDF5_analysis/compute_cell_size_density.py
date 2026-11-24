"""Recalculate only cell-local densities for the Figure 3 expansion plot.

Uses positions already filtered by the original pressure pipeline. Preserves
the original cell types, particle-distance changes, runs, and 12-hour windows.
The shared Epanechnikov kernel is evaluated at cell centres, rather than the
master curve's fixed grid. No resistant-growth support gate is imposed on this
single-cell expansion outcome. Original pressure files are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from analysis.figure_3.HDF5_analysis import constants as shared
from analysis.figure_3.HDF5_analysis.compute_flux_vs_density import _kernel_density_field
from shared.data_paths import SIM_RAW_DATA, figure_data_dir

PROTOCOLS = ("AT50", "AT100", "CT")
DEFAULT_INPUT_DIR = SIM_RAW_DATA / "pressure_analysis" / "pressure_data" / "pressure_data"
DEFAULT_RADIUS_UM = 160.0


def output_directory(radius_um=DEFAULT_RADIUS_UM):
    token = f"{radius_um:g}".replace(".", "p")
    return figure_data_dir("Figure_3", "cell_size_vs_density", f"kernel_radius_{token}um", create=False)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recompute_file(source, destination, *, radius_um=DEFAULT_RADIUS_UM, force=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Density regeneration must not overwrite the original pressure file")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("Kernel radius must be finite and positive")
    print(f"[source] {source}", flush=True)
    digest = file_digest(source)
    if destination.exists() and not force:
        try:
            with h5py.File(destination, "r") as existing:
                shared.validate_cell_density_settings(existing.attrs, radius_um, source=destination)
                if existing.attrs.get("source_sha256") != digest or not existing.attrs.get("complete", False):
                    raise ValueError('incomplete or different source')
            verify_preserved(source, destination)
            print(f"[reuse] {destination}", flush=True)
            return destination
        except (OSError, ValueError, KeyError, AssertionError) as error:
            print(f'[rebuild] {destination}: {error}', flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".h5.partial")
    radius = radius_um / (shared.exp_scaling * shared.px_to_micron)
    norm = 2.0 / (np.pi * radius**2)
    windows = cells = 0
    with h5py.File(source, "r") as old, h5py.File(temporary, "w") as new:
        new.attrs.update({
            "kernel_radius_um": float(radius_um),
            "kernel_min_neighbors": shared.MIN_KERNEL_NEIGHBORS,
            "density_method": "epanechnikov",
            "density_evaluation": "cell_centres_at_t0",
            "density_units": "cells/simulation_pixel_squared",
            "source_path": str(source.resolve()), "source_sha256": digest,
            "calculation_version": 1, "observation_hours": 12.0,
            "source_position_filter": "original_pressure_pipeline_active_non_outlier_cells",
            "scope": "cell_size_plot_only; old kernel-dependent p_div/p_death not copied",
            "expansion_definition": "original delta_d; surviving non-dividing cells over 12 hours",
            "resistant_support_gate_applied": False,
            "exp_scaling": shared.exp_scaling, "px_to_micron": shared.px_to_micron,
            "complete": False,
        })
        for run_name, run in old.items():
            if "cells" not in run:
                raise ValueError(f"{source}:{run_name} has no cells group")
            target_run = new.create_group(run_name)
            target_run.attrs.update(dict(run.attrs))
            target_cells = target_run.create_group("cells")
            for window_name, group in run["cells"].items():
                positions = np.asarray(group["positions"][:])
                cell_type = np.asarray(group["cell_type"][:])
                delta_d = np.asarray(group["delta_d"][:])
                size = len(cell_type)
                if positions.shape != (size, 2) or delta_d.shape != (size,) or not np.isfinite(positions).all():
                    raise ValueError(f"Invalid cell arrays: {source}:{group.name}")
                evaluation_points = positions.astype(np.float64)
                total, susceptible, resistant, valid = _kernel_density_field(
                    evaluation_points, cell_type, evaluation_points, radius, norm,
                    (size,), shared.MIN_KERNEL_NEIGHBORS)
                target = target_cells.create_group(window_name)
                target.attrs.update(dict(group.attrs))
                for name, values in (("positions", positions), ("cell_type", cell_type), ("delta_d", delta_d)):
                    target.create_dataset(name, data=values, compression="gzip")
                for name, values in (("rho_tot", total), ("rho_sus", susceptible), ("rho_res", resistant)):
                    target.create_dataset(name, data=np.where(valid, values, np.nan), compression="gzip")
                target.create_dataset("kernel_valid", data=valid, compression="gzip")
                windows += 1
                cells += size
                print(f"[window {windows}] {run_name}/{window_name}: {size:,} cells, {valid.sum():,} supported", flush=True)
        new.attrs.update({"n_windows": windows, "n_cell_observations": cells, "complete": True})
    if file_digest(source) != digest:
        raise RuntimeError(f"Source changed during density calculation: {source}")
    os.replace(temporary, destination)
    print(f"[saved] {destination}: {windows} windows, {cells:,} observations", flush=True)
    return destination


def verify_preserved(source, destination):
    """Require all original runs/windows and non-density observations unchanged."""
    with h5py.File(source, "r") as old, h5py.File(destination, "r") as new:
        if set(old) != set(new):
            raise AssertionError("Run selection changed")
        for run in old:
            if set(old[run]["cells"]) != set(new[run]["cells"]):
                raise AssertionError(f"Window selection changed: {run}")
            for name, group in old[run]["cells"].items():
                target = new[run]["cells"][name]
                for field in ("positions", "cell_type", "delta_d"):
                    a, b = group[field][:], target[field][:]
                    if a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                        raise AssertionError(f"Changed original values: {run}/{name}/{field}")
                if dict(group.attrs) != dict(target.attrs):
                    raise AssertionError(f"Changed window attributes: {run}/{name}")
        if file_digest(source) != new.attrs["source_sha256"]:
            raise AssertionError("Original pressure-file fingerprint changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--kernel-radius-um", type=float, default=None,
                        help="One kernel radius; omitted: prepare both 120 and 160 um.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    radii = (args.kernel_radius_um,) if args.kernel_radius_um is not None else shared.MASTER_KERNEL_RADII_UM
    for radius in radii:
        destination = output_directory(radius)
        if args.output_dir is not None:
            destination = args.output_dir
            if len(radii) > 1:
                destination /= output_directory(radius).name
        for protocol in PROTOCOLS:
            source, target = args.input_dir/f"{protocol}.h5", destination/f"{protocol}.h5"
            recompute_file(source, target, radius_um=radius, force=args.force)
            verify_preserved(source, target)
            print(f"[verified] {protocol}, {radius:g} um: original expansion data, runs, and windows unchanged", flush=True)


if __name__ == "__main__":
    main()
