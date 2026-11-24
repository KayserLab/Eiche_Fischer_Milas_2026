"""
Shared Figure 3 density parameters, plotting defaults, support/growth rules,
and input validation. Used by the master curves, HDF5 maps, and trajectories.
"""

import hashlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


# -- project data locations --
# Keep these scripts runnable directly while using the same portable data-root
# configuration as the rest of the repository.
REPO_ROOT = Path(__file__).resolve().parents[3]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from shared.data_paths import EXP_RAW_DATA, SIMULATION_DATABASE, figure_data_dir, figure_plot_dir
MIN_KERNEL_NEIGHBORS = 14
MIN_RESISTANT_NUMERATOR = 2.25
MASTER_KERNEL_RADIUS_UM = 160.0
MASTER_KERNEL_RADII_UM = (120.0, 160.0)
MASTER_GROWTH_MODE = "relative"
MASTER_DENSITY_COORDINATE = "first_endpoint"
MASTER_PERIOD = "all_timepoints"
MASTER_SUMMARY_STATISTIC = "median"


def is_non_artificial_run(name: str) -> bool:
    """Apply the master-curve cohort rule to canonical or protocol-prefixed IDs."""
    match = re.fullmatch(r"(?:[A-Za-z0-9+_-]+_)?run_(\d+)", str(name))
    return match is not None and int(match.group(1)) > 9


def resistant_support(total0, resistant0, total1, resistant1, valid0, valid1, *, threshold):
    """The master curve's exact finite-data and inclusive t0 support mask."""
    return (
        np.asarray(valid0, dtype=bool) & np.asarray(valid1, dtype=bool)
        & np.isfinite(total0) & np.isfinite(total1)
        & np.isfinite(resistant0) & np.isfinite(resistant1)
        & (np.asarray(resistant0) >= threshold)
    )


def resistant_change_rate(resistant0, resistant1, elapsed_days, *, mode=MASTER_GROWTH_MODE):
    """Per-pixel growth before any aggregation; relative growth uses rho_res(t0)."""
    if not np.isfinite(elapsed_days) or elapsed_days <= 0:
        raise ValueError("An interval must have a finite positive duration")
    first, second = np.asarray(resistant0, dtype=float), np.asarray(resistant1, dtype=float)
    change = (second - first) / float(elapsed_days)
    if mode == "relative":
        return np.divide(change, first, out=np.full_like(change, np.nan), where=first > 0)
    if mode == "absolute":
        return change
    raise ValueError(f"Unsupported growth mode: {mode}")

# Exact legacy inputs verified against raw-data spot checks on 2026-10-29.
# The 120-um flux tables also match the diagnostic current-variant curves for
# every retained replicate. These hashes identify the audited bytes, not a
# blanket exception for files that omit parameter metadata.
VERIFIED_LEGACY_INPUTS = {
    120.0: frozenset({
        "40110b6924d4a5f91c2d1ec9732161132b84972226eeaabcf0140cf6e957a33e",
        "762c03d72080a93ca39e528b8297a9b15cf3792d45d8f22e1331181ae58aa94c",
        "2767a624b471a78ed446a0ac4309bc0450a49425b61d855231b3ac076490137b",
        "f8632e1664a50d67168e1420c7c7c9867d9575a7d96dfed48dc9319854d2737d",
        "3ab70db7cd2bc3ab4e8509c7b2db9137d2abf783979b614c78767e4f20092f71",
        "6ad221885958c70c90aecaad2621fac48e0e893616bb10cf9361e56533b24a9a",
        "0bdf65351bef9ea8c056494eecd1c357c6ba1b6b6f065af7ba1edd2bf8396c34",
    }),
    160.0: frozenset({
        "82c732d6bb6777164d60eaa28ba6129312fd9cb4b175bd07432fde24084602c1",
    }),
}


def kernel_settings(radius_um: float) -> dict[str, float | int]:
    """Return the parameters required to identify a reusable density result."""
    return {
        "kernel_radius_um": float(radius_um),
        "kernel_min_neighbors": MIN_KERNEL_NEIGHBORS,
        "minimum_resistant_numerator": MIN_RESISTANT_NUMERATOR,
    }


def validate_cell_density_settings(attributes, radius_um, *, source="input"):
    """Check the kernel itself; single-cell expansion has no resistant-growth gate."""
    expected = {"kernel_radius_um": float(radius_um), "kernel_min_neighbors": MIN_KERNEL_NEIGHBORS}
    missing = [name for name in (*expected, "density_method") if name not in attributes]
    if missing:
        raise ValueError(f"{source}: missing parameter metadata {', '.join(missing)}. "
                         "Run analysis/figure_3/HDF5_analysis/compute_cell_size_density.py.")
    for name, value in expected.items():
        if not math.isclose(float(attributes[name]), value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{source}: {name}={attributes[name]}, expected {value}")
    if attributes["density_method"] != "epanechnikov":
        raise ValueError(f"{source}: expansion-plot density must use the Epanechnikov kernel")



def validate_kernel_settings(attributes, radius_um: float, *, source="input") -> None:
    """Require matching metadata or an exact previously audited legacy input."""
    missing = []
    for name, expected in kernel_settings(radius_um).items():
        if name not in attributes:
            missing.append(name)
            continue
        try:
            actual = float(attributes[name])
        except (KeyError, TypeError, ValueError):
            actual = math.nan
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{source}: {name}={actual:g}, expected {expected:g}. "
                "Recompute this density result with the current analysis settings."
            )
    if not missing:
        return
    if (
        set(missing) <= {"kernel_min_neighbors", "minimum_resistant_numerator"}
        and MIN_KERNEL_NEIGHBORS == 14
        and MIN_RESISTANT_NUMERATOR == 2.25
    ):
        try:
            with Path(source).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest in VERIFIED_LEGACY_INPUTS.get(float(radius_um), ()):
                return
        except (OSError, TypeError):
            pass
    raise ValueError(
        f"{source}: missing parameter metadata {', '.join(missing)}. "
        "This file is not an exact audited legacy input; verify its provenance "
        "or recompute it with the current analysis settings."
    )



def _environment_value(name, legacy_name, default):
    """Read the HDF5 setting while accepting the historical name."""
    return os.environ.get(name, os.environ.get(legacy_name, default))


prolif_radius_um = float(
    _environment_value("HDF5_KERNEL_RADIUS_UM", "MAJA_KERNEL_RADIUS_UM", "160")
)
if not np.isfinite(prolif_radius_um) or prolif_radius_um <= 0:
    raise ValueError(
        f"HDF5_KERNEL_RADIUS_UM must be positive, got {prolif_radius_um!r}"
    )


def _radius_token(value):
    numeric = float(value)
    text = f"{numeric:g}".replace(".", "p")
    return f"kernel_radius_{text}um"


KERNEL_VARIANT_DIRECTORY = (
    None if np.isclose(prolif_radius_um, 160.0)
    else _radius_token(prolif_radius_um)
)
_default_analysis_data_dir = figure_data_dir(
    "Figure_3", "HDF5_analysis", "flux_vs_density"
)
_default_figures_output_dir = figure_plot_dir(
    "Figure_3", "HDF5_analysis", "flux_vs_density"
)
if KERNEL_VARIANT_DIRECTORY is not None:
    _default_analysis_data_dir /= KERNEL_VARIANT_DIRECTORY
    _default_figures_output_dir /= KERNEL_VARIANT_DIRECTORY

HDF5_ANALYSIS_DATA_DIR = Path(
    _environment_value(
        "HDF5_ANALYSIS_DATA_DIR",
        "MAJA_ANALYSIS_DATA_DIR",
        _default_analysis_data_dir,
    )
).expanduser()
SIMULATION_INPUT_DIR = Path(
    _environment_value(
        "HDF5_SIMULATION_INPUT_DIR", "MAJA_SIMULATION_INPUT_DIR", SIMULATION_DATABASE
    )
).expanduser()
EXPERIMENT_INPUT_DIR = Path(
    _environment_value(
        "HDF5_EXPERIMENT_INPUT_DIR", "MAJA_EXPERIMENT_INPUT_DIR", EXP_RAW_DATA
    )
).expanduser()
SIMULATION_OUTPUT_DIR = Path(
    os.environ.get("FLUX_OUTPUT_DIR", HDF5_ANALYSIS_DATA_DIR / "simulation")
).expanduser()
EXPERIMENT_OUTPUT_DIR = Path(
    os.environ.get("EXP_FLUX_OUTPUT_DIR", HDF5_ANALYSIS_DATA_DIR / "experiment")
).expanduser()
FIGURES_OUTPUT_DIR = Path(
    os.environ.get(
        "FLUX_FIGURES_DIR",
        _default_figures_output_dir,
    )
).expanduser()

# -- unit conversion (scripts/metadata.py) --
A_max_exp = 2591        # px² (experimental typical cell area)
A_max_sim = 18.34       # simulation typical cell area
exp_scaling = np.sqrt(A_max_exp / A_max_sim)
px_to_micron = 1.095    # 1 experimental px = 1.095 um
AREA_SCALE = A_max_exp / A_max_sim   # = exp_scaling**2; cells/sim-px^2 -> cells/exp-px^2 by division

time_dump = 100000   # simulation steps per "day"

# -- outlier filtering, simulation only (scripts/analysis.py: remove_outliers) --
filter_robust_center    = np.array([0.0, 0.0])
filter_outlier_boundary = 6000   # px
filter_neighbour_radius = 360    # px
filter_min_neighbours   = 5

# -- proliferation/kernel-density binning (scripts/metadata.py) --
prolif_min_neighbors = MIN_KERNEL_NEIGHBORS  # same support rule for every radius
prolif_rho_max       = 0.32   # sim-px^-2
prolif_n_rho_bins    = 30
min_prolif_bin_count = 20     # min datapoints per 1D rho bin; below this -> NaN

# -- cell-type/flag codes (puls_dpd_tissue/model.py) --
CELL_FLAG_ACTIVE      = 1
CELL_TYPE_SUSCEPTIBLE = 0
CELL_TYPE_RESISTANT   = 1

N_JOBS = max(1, int(_environment_value("HDF5_N_JOBS", "MAJA_N_JOBS", "28")))
EXPERIMENT_N_JOBS = max(
    1,
    int(
        _environment_value(
            "HDF5_EXPERIMENT_N_JOBS", "MAJA_EXPERIMENT_N_JOBS", "28"
        )
    ),
)
KERNEL_QUERY_CHUNK_SIZE = max(
    1,
    int(
        _environment_value(
            "HDF5_KERNEL_QUERY_CHUNK_SIZE",
            "MAJA_KERNEL_QUERY_CHUNK_SIZE",
            "50000",
        )
    ),
)

# -- figure reference densities, sim-px^-2 (scripts/metadata.py) --
rho_homeostatic_res = 0.199453
rho_confluent_res   = 0.064895

# -- figure colors --
color_res    = "#DAA520"
color_master = "black"
protocol_colors = {
    "AT50": "#88CCEE",
    "AT100": "#CC6677",
    "CT": "#117733",
}

# -- Nature house style --
NATURE_FONT_STACK        = ["Helvetica", "Arial", "DejaVu Sans"]
NATURE_TEXT_PT           = 7.0
NATURE_TICK_PT           = 6.0
NATURE_LEGEND_PT         = 6.0
NATURE_DPI               = 300
NATURE_AXIS_LINEWIDTH_PT = 0.8
NATURE_TICK_LENGTH_PT    = 2.5

# -- figure averaging thresholds --
MIN_RUNS_PER_BIN      = 3     # within a protocol: bins backed by fewer run curves are too noisy to show
MIN_PROTOCOLS_PER_BIN = 2     # master: require at least this many protocol curves valid in a bin
FIG_WIDTH_MM = 180.0          # Nature two-column full-page width
