"""
TL;DR
-----
This script converts corrected cell-center point clouds into spatial density
maps using an Epanechnikov kernel. It also stacks those per-day maps into
time-ordered TIFF volumes for downstream analyses.

Detailed
--------
Inputs:
- HDF5 point clouds for susceptible/resistant cells, typically after total
  position correction.
- Plate/day selection and kernel radius in physical units.

Outputs:
- Per-day density TIFFs for susceptible, resistant, or total populations.
- Time-stacked density TIFF volumes per well.

How it works:
- Reads cell-center coordinates from HDF5 point clouds.
- Rasterizes cell positions onto the image canvas or a configurable coarser grid.
- Applies a circular Epanechnikov smoothing kernel to estimate local cell
  density in physical space.
- Encodes density maps and optionally stacks all available days for each well
  into a single time-ordered TIFF.

Pipeline role:
- Core spatial summarization step that turns point clouds into the density
  fields later used for confinement, correlation, and TTP-vs-density analyses.
"""

import h5py
import argparse
import os
import sys
import fnmatch
import math
from datetime import datetime
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.signal import fftconvolve
import glob, re
from pathlib import Path
from tifffile import imread, imwrite
import skimage

from skimage.morphology import remove_small_holes, binary_closing, disk
from skimage.segmentation import find_boundaries

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_paths import EXP_RAW_DATA, intermediate_dir, rebase_legacy_data_path  # noqa: E402

PIXEL_SIZE_UM = 1.095
PIXEL_AREA_UM2 = PIXEL_SIZE_UM ** 2
DEFAULT_COARSENING_FACTOR = 4
DEFAULT_RADIUS_UM = 100.0
DEFAULT_COARSE_PIXELS_PER_RADIUS = 10.0
DEFAULT_TEST_OUTPUT_FOLDER = 'test_output'
DEFAULT_DATASET_NAME = '2025_08_CoreSeed_Large_Testdataset'
DEFAULT_CONFIG_PATH = Path(__file__).with_name('density_maps_epanechnikov_config_coarsegrained.yaml')
CONFIG_PATH_ENV = 'CORESEED_DENSITY_CONFIG'
EXPERIMENT_PATH_ENV = 'CORESEED_EXPERIMENT_PATH'
MOUNT_ROOT_ENV = 'CORESEED_MOUNT_ROOT'
OUTPUT_RUN_NAME_ENV = 'CORESEED_OUTPUT_RUN_NAME'
DEFAULT_PLATES = ['Plate_1','Plate_2','Plate_3', 'Plate_4', 'Plate_5', 'Plate_6', 'Plate_7', 'Plate_8']
DEFAULT_OUTPUT_ROOT = os.fspath(intermediate_dir("experimental", "basic", "density_maps", create=False))
DEFAULT_OUTPUT_RUN_NAME_FORMAT = 'run_%Y%m%d_%H%M%S'
# Set True when running this script with PyCharm's play button to replace
# existing density-map TIFFs. The --overwrite option provides the same switch.
OVERWRITE = False
DENSITY_ZERO_FLOOR_UM2 = 1e-12
DENSITY_STACK_CHANNELS = ('sus', 'res', 'total')
PIPELINE_OUTPUT_ALIASES = {
    'density_maps': 'density_maps',
    'maps': 'density_maps',
    'density_stacks': 'density_stacks',
    'stacks': 'density_stacks',
    'density_changes': 'density_changes',
    'changes': 'density_changes',
    'change_scatter_plots': 'change_scatter_plots',
    'scatter_plots': 'change_scatter_plots',
    'scatter': 'change_scatter_plots',
    'hdf5_scatter_movie': 'hdf5_scatter_movie',
    'scatter_movie': 'hdf5_scatter_movie',
    'movie': 'hdf5_scatter_movie',
}

def config_bool(config, key, default):
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def normalize_config_list(value):
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return [str(item) for item in value]


def selected_pipeline_outputs(config):
    if 'pipeline_outputs' not in config:
        return None

    outputs = set()
    for value in normalize_config_list(config.get('pipeline_outputs')):
        key = value.strip().lower()
        if key == 'all':
            return set(PIPELINE_OUTPUT_ALIASES.values())
        if key not in PIPELINE_OUTPUT_ALIASES:
            valid = ', '.join(sorted(set(PIPELINE_OUTPUT_ALIASES.values())))
            raise ValueError(
                f"Unsupported pipeline_outputs entry {value!r}. "
                f"Use one of: {valid}."
            )
        outputs.add(PIPELINE_OUTPUT_ALIASES[key])
    return outputs


def pipeline_output_enabled(config, output_key, legacy_bool_key, default):
    selected_outputs = selected_pipeline_outputs(config)
    if selected_outputs is not None:
        return output_key in selected_outputs
    return config_bool(config, legacy_bool_key, default)


def density_stack_modes(config):
    mode = str(config.get('density_stack_mode', 'multichannel')).strip().lower()
    aliases = {
        'separate': 'separate_channels',
        'separate_channel': 'separate_channels',
        'separate_channels': 'separate_channels',
        'per_channel': 'separate_channels',
        'multichannel': 'multichannel',
        'multi_channel': 'multichannel',
        'hyperstack': 'multichannel',
        'both': 'both',
    }
    if mode not in aliases:
        valid = ', '.join(('multichannel', 'separate_channels', 'both'))
        raise ValueError(f"density_stack_mode must be one of: {valid}. Got {mode!r}.")
    resolved = aliases[mode]
    if resolved == 'both':
        return {'multichannel', 'separate_channels'}
    return {resolved}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create and stack Epanechnikov density maps for CoreSeed plates.'
    )
    parser.add_argument(
        '--config',
        default=os.environ.get(CONFIG_PATH_ENV, os.fspath(DEFAULT_CONFIG_PATH)),
        help=(
            'YAML config path. Use an empty string to skip config loading. '
            f'Default: {DEFAULT_CONFIG_PATH}. Can also be set with ${CONFIG_PATH_ENV}.'
        ),
    )
    parser.add_argument(
        '--experiment-path',
        default=None,
        help=(
            'Path to the dataset folder containing Plate_* folders. '
            f'Overrides config and can also be set with ${EXPERIMENT_PATH_ENV}.'
        ),
    )
    parser.add_argument(
        '--mount-root',
        default=None,
        help=(
            'Optional root of a locally mounted share. The resolver checks both '
            f'this folder and this folder/{DEFAULT_DATASET_NAME}. '
            f'Overrides config and can also be set with ${MOUNT_ROOT_ENV}.'
        ),
    )
    parser.add_argument(
        '--plates',
        nargs='+',
        help='Plate folders to process, for example: --plates Plate_2 Plate_3.',
    )
    parser.add_argument('--overwrite', action='store_true', default=OVERWRITE,
                        help='Replace existing density-map TIFFs.')
    return parser.parse_args()


def parse_simple_yaml_value(value):
    value = value.strip()
    if not value:
        return ''
    if value[0] in {'"', "'"} and value[-1:] == value[0]:
        return value[1:-1]
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_simple_yaml_value(part.strip()) for part in inner.split(',')]
    if value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    if value.lower() in {'none', 'null'}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path):
    data = {}
    current_key = None

    with open(path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.split('#', 1)[0].rstrip()
            if not line.strip():
                continue

            stripped = line.strip()
            if stripped.startswith('- '):
                if current_key is None:
                    raise ValueError(f'List item without a key in {path}: {raw_line.rstrip()}')
                data.setdefault(current_key, []).append(parse_simple_yaml_value(stripped[2:]))
                continue

            if ':' not in line:
                raise ValueError(f'Unsupported YAML line in {path}: {raw_line.rstrip()}')

            key, value = line.split(':', 1)
            current_key = key.strip()
            value = value.strip()
            data[current_key] = [] if value == '' else parse_simple_yaml_value(value)

    return data


def load_config(config_path):
    if not config_path:
        return {}

    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        config = load_simple_yaml(config_path)
    else:
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle) or {}

    for key in (
        'experiment_path',
        'hdf5_path',
        'output_root',
        'relative_density_change_escape_summary_path',
        'test_output_folder',
    ):
        if isinstance(config.get(key), str):
            config[key] = os.fspath(rebase_legacy_data_path(config[key]))
    return config


def default_experiment_path_candidates():
    yield EXP_RAW_DATA

    volumes_dir = Path('/Volumes')
    yield volumes_dir / DEFAULT_DATASET_NAME

    if volumes_dir.is_dir():
        try:
            mounts = sorted(volumes_dir.iterdir())
        except OSError:
            mounts = []
        for mount in mounts:
            yield mount / DEFAULT_DATASET_NAME


def looks_like_experiment_path(path):
    try:
        return path.is_dir() and any(path.glob('Plate_*'))
    except OSError:
        return False


def infer_experiment_path_from_hdf5_path(hdf5_path):
    hdf5_path = Path(hdf5_path).expanduser()
    if hdf5_path.name.lower() == 'hdf5' and hdf5_path.parent.name.startswith('Plate_'):
        return hdf5_path.parent.parent
    return None


def infer_plate_from_hdf5_path(hdf5_path):
    hdf5_path = Path(hdf5_path).expanduser()
    if hdf5_path.name.lower() == 'hdf5' and hdf5_path.parent.name.startswith('Plate_'):
        return hdf5_path.parent.name
    return None


def normalize_plates(plates):
    if plates is None:
        return None
    if isinstance(plates, str):
        return [plates]
    if isinstance(plates, dict):
        return list(plates.keys())

    normalized = []
    for plate in plates:
        if isinstance(plate, dict):
            plate_name = plate.get('name') or plate.get('plate')
            if not plate_name and len(plate) == 1:
                plate_name = next(iter(plate.keys()))
            if not plate_name:
                raise ValueError(
                    "Plate entries must be strings or mappings with a 'name' key."
                )
            normalized.append(str(plate_name))
        else:
            normalized.append(str(plate))
    return normalized


def plate_well_designators(config, plate):
    """Return short well selectors such as C1/C2 for a configured plate."""
    plate_text = str(plate)

    plate_wells = config.get('plate_wells')
    if isinstance(plate_wells, dict) and plate_text in plate_wells:
        return normalize_config_list(plate_wells[plate_text])

    plates = config.get('plates')
    if isinstance(plates, dict):
        plate_entry = plates.get(plate_text)
        if isinstance(plate_entry, dict):
            return normalize_config_list(plate_entry.get('wells'))
        return normalize_config_list(plate_entry)

    if isinstance(plates, list):
        for plate_entry in plates:
            if isinstance(plate_entry, dict):
                plate_name = plate_entry.get('name') or plate_entry.get('plate')
                wells = plate_entry.get('wells')
                if plate_name is None and len(plate_entry) == 1:
                    plate_name, nested = next(iter(plate_entry.items()))
                    wells = nested.get('wells') if isinstance(nested, dict) else nested
                if str(plate_name) == plate_text:
                    return normalize_config_list(wells)

    return []


def normalize_timepoint(timepoint):
    if isinstance(timepoint, int):
        return f'{timepoint:02d}'
    timepoint = str(timepoint).strip()
    return timepoint.zfill(2) if timepoint.isdigit() else timepoint


def timepoint_delta_days(prev_tp, next_tp):
    try:
        delta_days = float(next_tp) - float(prev_tp)
    except (TypeError, ValueError):
        return 1.0

    if delta_days <= 0:
        raise ValueError(
            f"Timepoints must increase to compute per-day growth, got "
            f"{prev_tp!r}->{next_tp!r}."
        )
    return delta_days


def normalize_timepoints(config):
    timepoints = config.get('test_timepoints')
    if timepoints is None:
        timepoint = config.get('test_timepoint')
        return [normalize_timepoint(timepoint)] if timepoint is not None else []
    if isinstance(timepoints, (str, int)):
        return [normalize_timepoint(timepoints)]
    return [normalize_timepoint(timepoint) for timepoint in timepoints]


def resolve_test_output_subfolder(folder, test_output_folder=DEFAULT_TEST_OUTPUT_FOLDER):
    folder_path = Path(os.fspath(folder))
    if folder_path.is_absolute():
        return os.fspath(folder_path)

    test_output_path = Path(os.fspath(test_output_folder))
    if os.fspath(test_output_path) in {'', '.'}:
        return os.fspath(folder_path)

    if folder_path.parts[:len(test_output_path.parts)] == test_output_path.parts:
        return os.fspath(folder_path)

    return os.fspath(test_output_path / folder_path)


def use_structured_output_layout(config):
    layout = str(config.get('output_layout', 'legacy')).strip().lower()
    return layout in {
        'run_plate_well_type',
        'run/plate/well/type',
        'run_plate_well_output_type',
    }


def resolve_output_root(config, experiment_path):
    root = Path(os.fspath(config.get('output_root', DEFAULT_OUTPUT_ROOT))).expanduser()
    if root.is_absolute():
        return root
    return Path(experiment_path).expanduser().resolve() / root


def resolve_output_run_name(config):
    cached = config.get('_resolved_output_run_name')
    if cached:
        return cached

    configured = (
        os.environ.get(OUTPUT_RUN_NAME_ENV)
        or config.get('output_run_name')
        or config.get('run_name')
    )
    if configured is not None:
        configured = str(configured).strip()
        if configured and configured.lower() not in {'auto', 'datetime', 'timestamp'}:
            config['_resolved_output_run_name'] = configured
            return configured

    generated = datetime.now().strftime(DEFAULT_OUTPUT_RUN_NAME_FORMAT)
    config['_resolved_output_run_name'] = generated
    return generated


def resolve_output_run_dir(config, experiment_path):
    explicit_path = config.get('output_run_path')
    if explicit_path:
        run_path = Path(os.fspath(explicit_path)).expanduser()
        if run_path.is_absolute():
            return run_path
        return Path(experiment_path).expanduser().resolve() / run_path

    return resolve_output_root(config, experiment_path) / resolve_output_run_name(config)


def output_type_name(config, config_key, default_name):
    return str(config.get(config_key, default_name)).strip()


def well_output_type_dir(config, experiment_path, plate, well_identifier, output_type_config_key, default_output_type):
    return (
        resolve_output_run_dir(config, experiment_path)
        / str(plate)
        / str(well_identifier)
        / output_type_name(config, output_type_config_key, default_output_type)
    )


def resolve_pointcloud_file(experiment_path, plate, hdf5_filename, hdf5_path=None):
    hdf5_filename = Path(hdf5_filename).expanduser()
    if hdf5_filename.is_absolute():
        return os.fspath(hdf5_filename)

    if hdf5_path:
        return os.fspath(Path(hdf5_path).expanduser() / hdf5_filename)

    return os.path.join(experiment_path, plate, 'hdf5', os.fspath(hdf5_filename))


def resolve_experiment_path(experiment_path=None, mount_root=None, hdf5_path=None):
    candidates = []

    if experiment_path:
        candidates.append(Path(experiment_path).expanduser())

    if hdf5_path:
        inferred = infer_experiment_path_from_hdf5_path(hdf5_path)
        if inferred is not None:
            candidates.append(inferred)

    if mount_root:
        mount_root = Path(mount_root).expanduser()
        candidates.extend([mount_root, mount_root / DEFAULT_DATASET_NAME])

    candidates.extend(default_experiment_path_candidates())

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        candidate_key = os.fspath(candidate)
        if candidate_key not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate_key)

    for candidate in unique_candidates:
        if looks_like_experiment_path(candidate):
            return os.fspath(candidate)

    checked = '\n  - '.join(os.fspath(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(
        'Could not find the experiment folder.\n'
        'Pass it explicitly, for example:\n'
        '  python analysis/basic/density_maps/Density_Maps_Epanechnikov_coarsegrained.py '
        '--experiment-path /Volumes/YourShare/2025_08_CoreSeed_Large_Testdataset\n'
        'If your config uses hdf5_path, point it at a Plate_*/hdf5 folder.\n'
        f'Or set ${EXPERIMENT_PATH_ENV} to the dataset folder.\n'
        f'Checked:\n  - {checked}'
    )


def pointcloud_timepoint_candidates(time) -> list[str]:
    text = str(time).strip()
    candidates = [text]
    if text.isdigit():
        value = int(text)
        for width in (len(text), 2, 3):
            candidates.append(f'{value:0{width}d}')
    return list(dict.fromkeys(candidates))


def resolve_pointcloud_timepoint_path(handle, group: str, unit: str, time) -> str | None:
    if group not in handle:
        return None
    for candidate in pointcloud_timepoint_candidates(time):
        key = f'{unit}_{candidate}'
        if key in handle[group]:
            return f'{group}/{key}'
    return None


def pointcloud_read(filename: str, time: str, unit: str = 'd', require_timepoint: bool = False):
    """
    Reads x,y positions for sus/res cells from a plain HDF5 (not a PyTables store).
    If the dataset is missing or has the wrong shape, returns empty DataFrames.

    :param filename: .h5 file
    :param time:    timepoint, e.g. '00', '01', or simulation-style '001'
    :param unit:    prefix in the HDF5 keys, e.g. 'd' → groups like 'sus/d_00'
    :param require_timepoint: If True, raise when neither sus nor res contains the requested timepoint.
    :return:        (df_sus, df_res) where each is a pd.DataFrame(columns=['x','y'])
    """
    df_sus = pd.DataFrame(columns=['x', 'y'])
    df_res = pd.DataFrame(columns=['x', 'y'])

    with h5py.File(filename, 'r') as f:
        sus_path = resolve_pointcloud_timepoint_path(f, 'sus', unit, time)
        res_path = resolve_pointcloud_timepoint_path(f, 'res', unit, time)

        if require_timepoint and sus_path is None and res_path is None:
            available = sorted(
                {
                    key
                    for grp in ('sus', 'res')
                    if grp in f
                    for key in f[grp].keys()
                }
            )
            tried = [f'{unit}_{candidate}' for candidate in pointcloud_timepoint_candidates(time)]
            raise KeyError(
                f"Timepoint {unit}_{time} is missing in {filename}. Tried: {tried}. "
                f"Available timepoints: {available}"
            )

        if sus_path is not None:
            arr = f[sus_path]['block0_values'][:]
            if arr.ndim == 2 and arr.shape[1] == 2:
                df_sus = pd.DataFrame({'x': arr[:, 1], 'y': arr[:, 0]})

        if res_path is not None:
            arr = f[res_path]['block0_values'][:]
            if arr.ndim == 2 and arr.shape[1] == 2:
                df_res = pd.DataFrame({'x': arr[:, 1], 'y': arr[:, 0]})

    return df_sus, df_res


def circular_kernel(radius):
    """
    Create a circular averaging kernel with the given radius.
    (Kept for reference; not used in current pipeline.)
    """
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    mask = (X ** 2 + Y ** 2) <= radius ** 2
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    kernel[mask] = 1
    kernel /= kernel.sum()
    return kernel


def radius_um_to_original_px(radius_um: int | float) -> float:
    return float(radius_um) / PIXEL_SIZE_UM


def radius_um_to_coarse_px(radius_um: int | float, coarsening_factor: int) -> float:
    return float(radius_um) / (PIXEL_SIZE_UM * coarsening_factor)


def radius_um_folder_token(radius_um: int | float) -> str:
    radius_text = f'{float(radius_um):g}'.replace('-', 'm').replace('.', 'p')
    return f'r{radius_text}um'


def resolve_radius_um(config):
    if 'radius_um' in config:
        return float(config['radius_um'])

    if 'radius_px' in config:
        radius_um = float(config['radius_px']) * PIXEL_SIZE_UM
        print(
            f"Using legacy radius_px={config['radius_px']} "
            f"({radius_um:.3f} µm). Prefer radius_um in new configs."
        )
        return radius_um

    if 'radius' in config:
        radius_um = float(config['radius']) * PIXEL_SIZE_UM
        print(
            f"Using legacy radius={config['radius']} original pixels "
            f"({radius_um:.3f} µm). Prefer radius_um in new configs."
        )
        return radius_um

    return DEFAULT_RADIUS_UM


def is_auto_coarsening(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in {'auto', 'automatic'}


def resolve_manual_coarsening_factor(value) -> int:
    try:
        coarsening_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"coarsening_factor must be an integer or 'auto', got {value!r}"
        ) from None

    if not coarsening_float.is_integer():
        raise ValueError(
            f"coarsening_factor must be an integer or 'auto', got {value!r}"
        )

    return int(coarsening_float)


def resolve_coarsening_factor(config, radius_um: int | float) -> int:
    coarsening = config.get('coarsening_factor', DEFAULT_COARSENING_FACTOR)
    if not is_auto_coarsening(coarsening):
        return resolve_manual_coarsening_factor(coarsening)

    coarse_pixels_per_radius = float(
        config.get('coarse_pixels_per_radius', DEFAULT_COARSE_PIXELS_PER_RADIUS)
    )
    if coarse_pixels_per_radius <= 0:
        raise ValueError(
            f"coarse_pixels_per_radius must be > 0, got {coarse_pixels_per_radius}"
        )

    radius_original_px = radius_um_to_original_px(radius_um)
    coarsening_factor = max(
        1,
        int(math.floor(radius_original_px / coarse_pixels_per_radius)),
    )
    radius_coarse_px = radius_um_to_coarse_px(radius_um, coarsening_factor)
    print(
        f"Auto coarsening: target={coarse_pixels_per_radius:g} coarse px/radius, "
        f"resolved coarsening_factor={coarsening_factor} "
        f"({radius_coarse_px:.3f} coarse px/radius)"
    )
    if radius_coarse_px < coarse_pixels_per_radius:
        print(
            "  Requested coarse pixels per radius exceeds the original-pixel "
            "radius; using coarsening_factor=1."
        )

    return coarsening_factor


def validate_coarsening_configuration(radius_um: int | float, coarsening_factor: int):
    if coarsening_factor < 1:
        raise ValueError(f"coarsening_factor must be >= 1, got {coarsening_factor}")

    coarse_radius = radius_um_to_coarse_px(radius_um, coarsening_factor)
    if coarse_radius < 1.0:
        raise ValueError(
            f"coarsening_factor={coarsening_factor} is too large for radius_um={radius_um}. "
            f"The kernel radius would be {coarse_radius:.3f} coarse pixels."
        )


def epanechnikov_circular_kernel(radius: int | float) -> np.ndarray:
    """
    Normalized Epanechnikov kernel with compact support (disk of radius R).
    Kernel:
        K(r) = (2 / (pi * R^2)) * (1 - r^2 / R^2)   for r <= R
             = 0                                    for r > R

    Returns a float32 kernel with support wide enough to cover radius R.
    """
    R = float(radius)
    if R <= 0:
        raise ValueError(f"Kernel radius must be > 0, got {radius}")

    kernel_extent = int(math.ceil(R))
    L = np.arange(-kernel_extent, kernel_extent + 1, dtype=np.float32)
    X, Y = np.meshgrid(L, L)
    r2 = X * X + Y * Y

    kernel = np.zeros((2 * kernel_extent + 1, 2 * kernel_extent + 1), dtype=np.float32)
    mask = r2 <= (R * R)

    kernel[mask] = (2.0 / (math.pi * R * R)) * (1.0 - (r2[mask] / (R * R)))
    return kernel


def load_overexposed_regions(filename: str):
    return tiff.imread(filename)


def tiff_xy_shape(filename: str | Path) -> tuple[int, int]:
    with tiff.TiffFile(str(filename)) as tf:
        if tf.series:
            shape = tf.series[0].shape
        else:
            shape = tf.pages[0].shape
    return int(shape[-2]), int(shape[-1])


def get_local_density_fft(results_array: np.ndarray, radius: int | float) -> np.ndarray:
    """
    Convolve an occupancy/count image with the normalized Epanechnikov kernel
    to get a local density map.
    """
    # ----------------------------
    # OLD UNIFORM DISK KERNEL (COUNT CELLS IN RADIUS) — kept for reference
    # y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    # kernel = (x**2 + y**2 <= radius**2).astype(float)
    # local_density = fftconvolve(results_array, kernel, mode='same')
    # return np.rint(local_density).astype(int)
    # ----------------------------

    kernel = epanechnikov_circular_kernel(radius=radius).astype(np.float32)

    # FFT convolution
    local_density = fftconvolve(results_array.astype(np.float32), kernel, mode='same')
    np.maximum(local_density, 0, out=local_density)

    return local_density.astype(np.float32)


def zero_density_fft_noise(density_um2: np.ndarray) -> np.ndarray:
    density_um2 = density_um2.astype(np.float32, copy=False)
    density_um2[density_um2 <= DENSITY_ZERO_FLOOR_UM2] = 0.0
    return density_um2


def rasterize_points_to_coarse_grid(
    df: pd.DataFrame,
    height: int,
    width: int,
    coarsening_factor: int,
) -> np.ndarray:
    coarse_height = math.ceil(height / coarsening_factor)
    coarse_width = math.ceil(width / coarsening_factor)
    image = np.zeros((coarse_height, coarse_width), dtype=np.float32)

    if df.empty:
        return image

    x = df['x'].to_numpy(dtype=np.float64, copy=False)
    y = df['y'].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(x) & np.isfinite(y) & (0 <= x) & (x < width) & (0 <= y) & (y < height)

    if not np.any(valid):
        return image

    xi = (x[valid].astype(np.int64) // coarsening_factor).clip(0, coarse_width - 1)
    yi = (y[valid].astype(np.int64) // coarsening_factor).clip(0, coarse_height - 1)

    if coarsening_factor == 1:
        image[yi, xi] = 1.0
    else:
        np.add.at(image, (yi, xi), 1.0)

    return image


def density(
    df_sus: pd.DataFrame,
    df_res: pd.DataFrame,
    radius_um: int | float,
    image_shape: tuple[int, int],
    coarsening_factor: int = DEFAULT_COARSENING_FACTOR,
):
    validate_coarsening_configuration(radius_um, coarsening_factor)

    height, width = image_shape

    image_resistant = rasterize_points_to_coarse_grid(df_res, height, width, coarsening_factor)
    image_susceptible = rasterize_points_to_coarse_grid(df_sus, height, width, coarsening_factor)

    image_total = image_resistant + image_susceptible
    coarse_radius = radius_um_to_coarse_px(radius_um, coarsening_factor)

    density_resistant = get_local_density_fft(image_resistant, coarse_radius)
    density_susceptible = get_local_density_fft(image_susceptible, coarse_radius)
    density_total = get_local_density_fft(image_total, coarse_radius)

    # Convert from cells / coarse-pixel² to cells / µm².
    coarse_pixel_area_um2 = (PIXEL_SIZE_UM * coarsening_factor) ** 2
    density_resistant = density_resistant / coarse_pixel_area_um2
    density_susceptible = density_susceptible / coarse_pixel_area_um2
    density_total = density_total / coarse_pixel_area_um2

    density_resistant = zero_density_fft_noise(density_resistant)
    density_susceptible = zero_density_fft_noise(density_susceptible)
    density_total = zero_density_fft_noise(density_total)

    return density_susceptible, density_resistant, density_total


def normalize_patterns(patterns):
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return [str(pattern) for pattern in patterns]


def pointcloud_filenames(
    directory,
    poisson,
    include_patterns=None,
    exclude_patterns=None,
    well_designators=None,
):
    simulation_suffixes = ('_simulation.h5',)
    if poisson:
        suffixes = simulation_suffixes + ('_total_poisson_padded.h5', '_total_poisson.h5')
    else:
        suffixes = simulation_suffixes + ('_total.h5',)
    filenames = [f for f in os.listdir(directory) if f.endswith(suffixes)]

    include_patterns = normalize_patterns(include_patterns)
    exclude_patterns = normalize_patterns(exclude_patterns)
    if include_patterns:
        filenames = [
            filename
            for filename in filenames
            if any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns)
        ]
    if exclude_patterns:
        filenames = [
            filename
            for filename in filenames
            if not any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns)
        ]

    well_designators = normalize_config_list(well_designators)
    if well_designators:
        filenames = [
            filename
            for filename in filenames
            if any(
                pointcloud_filename_matches_well_designator(filename, well_designator)
                for well_designator in well_designators
            )
        ]
        missing = [
            well_designator
            for well_designator in well_designators
            if not any(
                pointcloud_filename_matches_well_designator(filename, well_designator)
                for filename in filenames
            )
        ]
        if missing:
            available = sorted(
                well_identifier_from_pointcloud_filename(filename)
                for filename in os.listdir(directory)
                if filename.endswith(suffixes)
            )
            raise FileNotFoundError(
                f"No point-cloud HDF5 found for well designator(s) {missing} "
                f"in {directory}. Available wells/identifiers: {available}"
            )
    return sorted(filenames)


def well_identifier_from_pointcloud_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    for suffix in ('_simulation', '_total_poisson_padded', '_total_poisson', '_total'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    parts = stem.split('_')
    if len(parts) > 2 and parts[0] == 'G418' and parts[1] == 'a2a4':
        parts = parts[2:]

    return '_'.join(parts)


def well_designator_from_well_identifier(well_identifier):
    return str(well_identifier).split('_', 1)[0]


def strategy_name_from_well_identifier(well_identifier):
    well_identifier = str(well_identifier)
    well_designator = well_designator_from_well_identifier(well_identifier)
    prefix = f'{well_designator}_'
    suffix = f'-{well_designator}'
    if well_identifier.startswith(prefix) and well_identifier.endswith(suffix):
        return well_identifier[len(prefix):-len(suffix)]
    return ''


def strategy_name_from_pointcloud_filename(filename):
    return strategy_name_from_well_identifier(
        well_identifier_from_pointcloud_filename(filename)
    )


def pointcloud_filename_matches_well_designator(filename, well_designator):
    well_identifier = well_identifier_from_pointcloud_filename(filename)
    well_designator = str(well_designator)
    if well_identifier == well_designator:
        return True
    return well_designator_from_well_identifier(well_identifier) == well_designator


def is_padded_pointcloud(filename):
    return os.path.basename(filename).endswith('_total_poisson_padded.h5')


def density_canvas_shape(
    directory_path,
    well_identifier,
    timepoint,
    pointcloud_filepath,
):
    if is_padded_pointcloud(pointcloud_filepath):
        stack_candidates = [
            os.path.join(directory_path, 'colony_masks_stacks', f'{well_identifier}_total_colony_masks.tif'),
            os.path.join(directory_path, 'colony_masks_stacks', f'{well_identifier}_sus_colony_masks.tif'),
            os.path.join(directory_path, 'colony_masks_stacks', f'{well_identifier}_res_colony_masks.tif'),
            os.path.join(directory_path, 'clone_masks_stacks', f'{well_identifier}_res_clone_masks.tif'),
        ]
        for candidate in stack_candidates:
            if os.path.exists(candidate):
                shape = tiff_xy_shape(candidate)
                print(f"Using padded canvas from {candidate}: shape={shape}")
                return shape

        print(
            f"  ⚠ Padded point cloud detected, but no padded mask stack was found for "
            f"{well_identifier}. Falling back to the per-day overexposed image shape."
        )

    overexposed_regions_path = os.path.join(
        directory_path,
        'overexposed_images',
        f'{well_identifier}_{timepoint}_sus_overexposed_img.tif'
    )
    if not os.path.exists(overexposed_regions_path):
        print(f'File {overexposed_regions_path} does not exist')
        return None

    return tiff_xy_shape(overexposed_regions_path)


def density_tiff_description(radius_um, coarsening_factor):
    radius_original_px = radius_um_to_original_px(radius_um)
    radius_coarse_px = radius_um_to_coarse_px(radius_um, coarsening_factor)
    return (
        f"kernel=epanechnikov_normalized; radius_um={radius_um}; "
        f"radius_original_px={radius_original_px}; "
        f"radius_coarse_px={radius_coarse_px}; "
        f"coarsening_factor={coarsening_factor}; "
        f"coarse_pixel_size_um={PIXEL_SIZE_UM * coarsening_factor}; "
        f"units=cells_per_um2; stored_as=float32; "
        f"pixel_size_um={PIXEL_SIZE_UM}"
    )


def create_density_map_for_timepoint(
    pointcloud_filepath,
    directory_path,
    timepoint,
    radius_um,
    density_maps_folder,
    coarsening_factor,
    require_timepoint=False,
    overwrite=None,
):
    if overwrite is None:
        overwrite = OVERWRITE
    timepoint = normalize_timepoint(timepoint)
    well_identifier = well_identifier_from_pointcloud_filename(pointcloud_filepath)
    density_maps_directory = os.path.join(directory_path, density_maps_folder)
    expected = [
        os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_{channel}_density_epanechnikov.tif')
        for channel in ('sus', 'res', 'total')
    ]
    if not overwrite and any(os.path.exists(path) for path in expected):
        print(f"Skipping {well_identifier} timepoint {timepoint}; density map exists (set OVERWRITE = True to replace).")
        return False

    masks_positions_sus, masks_positions_res = pointcloud_read(
        pointcloud_filepath,
        timepoint,
        require_timepoint=require_timepoint,
    )

    canvas_shape = density_canvas_shape(
        directory_path=directory_path,
        well_identifier=well_identifier,
        timepoint=timepoint,
        pointcloud_filepath=pointcloud_filepath,
    )
    if canvas_shape is None:
        return False

    sus_density, res_density, total_density = density(
        masks_positions_sus, masks_positions_res,
        radius_um=radius_um,
        image_shape=canvas_shape,
        coarsening_factor=coarsening_factor
    )

    sus_density = sus_density.astype(np.float32, copy=False)
    res_density = res_density.astype(np.float32, copy=False)
    total_density = total_density.astype(np.float32, copy=False)

    os.makedirs(density_maps_directory, exist_ok=True)
    description = density_tiff_description(radius_um, coarsening_factor)

    tiff.imwrite(
        os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_sus_density_epanechnikov.tif'),
        sus_density,
        description=description,
        compression='deflate'
    )
    tiff.imwrite(
        os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_res_density_epanechnikov.tif'),
        res_density,
        description=description,
        compression='deflate'
    )
    tiff.imwrite(
        os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_total_density_epanechnikov.tif'),
        total_density,
        description=description,
        compression='deflate'
    )

    print(f"Saved density maps for {well_identifier} timepoint {timepoint} to {density_maps_directory}")
    return True


def write_test_density_changes(
    directory_path,
    well_identifier,
    timepoints,
    density_maps_folder,
    density_changes_folder,
    channels=('res',),
    write_zero_initial_growth_channel=False,
    zero_initial_density_threshold=DENSITY_ZERO_FLOOR_UM2,
    measurable_growth_threshold=DENSITY_ZERO_FLOOR_UM2,
):
    if len(timepoints) < 2:
        return

    density_maps_directory = os.path.join(directory_path, density_maps_folder)
    density_changes_directory = os.path.join(directory_path, density_changes_folder)
    os.makedirs(density_changes_directory, exist_ok=True)

    for prev_tp, next_tp in zip(timepoints[:-1], timepoints[1:]):
        for channel in channels:
            prev_path = os.path.join(
                density_maps_directory,
                f'{well_identifier}_{prev_tp}_{channel}_density_epanechnikov.tif'
            )
            next_path = os.path.join(
                density_maps_directory,
                f'{well_identifier}_{next_tp}_{channel}_density_epanechnikov.tif'
            )
            if not os.path.exists(prev_path) or not os.path.exists(next_path):
                print(f"Skipping {channel} density change {prev_tp}->{next_tp}; missing density map.")
                continue

            prev = tiff.imread(prev_path).astype(np.float32, copy=False)
            nxt = tiff.imread(next_path).astype(np.float32, copy=False)
            if prev.shape != nxt.shape:
                raise ValueError(
                    f"Cannot compute {channel} density change {prev_tp}->{next_tp}: "
                    f"shape mismatch {prev.shape} vs {nxt.shape}"
                )

            abs_diff = (nxt - prev).astype(np.float32, copy=False)
            delta_days = timepoint_delta_days(prev_tp, next_tp)
            valid_log_growth = (
                np.isfinite(prev)
                & np.isfinite(nxt)
                & (prev > zero_initial_density_threshold)
                & (nxt > zero_initial_density_threshold)
            )
            rel_diff = np.zeros_like(abs_diff, dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff[valid_log_growth] = (
                    np.log(nxt[valid_log_growth] / prev[valid_log_growth])
                    / delta_days
                ).astype(np.float32, copy=False)

            base = f'{well_identifier}_{prev_tp}_to_{next_tp}_{channel}_density_epanechnikov'
            output_paths = [
                os.path.join(density_changes_directory, f'{base}_abs_diff.tif'),
                os.path.join(density_changes_directory, f'{base}_rel_diff.tif'),
            ]
            if not OVERWRITE and any(os.path.exists(path) for path in output_paths):
                print(f"Skipping {channel} density change {prev_tp}->{next_tp}; output exists (set OVERWRITE = True to replace).")
                continue
            tiff.imwrite(
                output_paths[0],
                abs_diff,
                dtype='float32',
                description=(
                    f"change=next-prev; channel={channel}; from_timepoint={prev_tp}; "
                    f"to_timepoint={next_tp}; units=cells_per_um2; stored_as=float32"
                ),
                compression='deflate',
            )
            tiff.imwrite(
                output_paths[1],
                rel_diff,
                dtype='float32',
                description=(
                    f"change=ln(next/prev)/delta_days where prev and next are >"
                    f"{zero_initial_density_threshold}, else 0; channel={channel}; "
                    f"from_timepoint={prev_tp}; to_timepoint={next_tp}; "
                    f"delta_days={delta_days}; units=per_day; stored_as=float32"
                ),
                compression='deflate',
            )
            if write_zero_initial_growth_channel and channel == 'res':
                zero_initial_growth = np.where(
                    (prev <= zero_initial_density_threshold)
                    & (abs_diff > measurable_growth_threshold),
                    abs_diff,
                    0.0,
                ).astype(np.float32, copy=False)
                zero_initial_growth_base = (
                    f'{well_identifier}_{prev_tp}_to_{next_tp}_'
                    f'{channel}_zero_initial_growth_density_epanechnikov'
                )
                tiff.imwrite(
                    os.path.join(
                        density_changes_directory,
                        f'{zero_initial_growth_base}_abs_diff.tif',
                    ),
                    zero_initial_growth,
                    dtype='float32',
                    description=(
                        f"change=next-prev where previous_res_density<="
                        f"{zero_initial_density_threshold} and change>"
                        f"{measurable_growth_threshold}; channel={channel}; "
                        f"from_timepoint={prev_tp}; to_timepoint={next_tp}; "
                        f"units=cells_per_um2; stored_as=float32"
                    ),
                    compression='deflate',
                )
            print(
                f"Saved {channel} density changes for {well_identifier} "
                f"{prev_tp}->{next_tp} to {density_changes_directory}"
            )


def create_density_maps(
    experiment_path,
    plate,
    days=None,
    radius_um=DEFAULT_RADIUS_UM,
    density_maps_folder='density_maps_epanechnikov',
    poisson=False,
    coarsening_factor: int = DEFAULT_COARSENING_FACTOR,
    hdf5_include_patterns=None,
    hdf5_exclude_patterns=None,
    well_designators=None,
    output_config=None,
):
    validate_coarsening_configuration(radius_um, coarsening_factor)
    directory_path = os.path.join(experiment_path, plate)
    directory = os.path.join(directory_path, 'hdf5')

    filenames = pointcloud_filenames(
        directory,
        poisson,
        include_patterns=hdf5_include_patterns,
        exclude_patterns=hdf5_exclude_patterns,
        well_designators=well_designators,
    )

    for file in filenames:
        pointcloud_filepath = os.path.join(directory, file)
        well_identifier = well_identifier_from_pointcloud_filename(file)
        strategy_name = strategy_name_from_well_identifier(well_identifier)
        strategy_text = f", strategy={strategy_name}" if strategy_name else ""
        print(f"{file} -> well={well_designator_from_well_identifier(well_identifier)}{strategy_text}")
        density_maps_folder_for_well = density_maps_folder
        if output_config and use_structured_output_layout(output_config):
            density_maps_folder_for_well = os.fspath(
                well_output_type_dir(
                    output_config,
                    experiment_path,
                    plate,
                    well_identifier,
                    'density_maps_output_type',
                    'density_maps',
                )
            )

        # Auto-discover days per well if not provided
        if days is None:
            discovered = set()
            try:
                with h5py.File(pointcloud_filepath, 'r') as fh:
                    for grp in ('sus', 'res'):
                        if grp in fh:
                            for key in fh[grp].keys():
                                m = re.match(r'^d_(\d+)$', key)
                                if m:
                                    discovered.add(m.group(1))
            except Exception as e:
                print(f"  ⚠ Failed to inspect HDF5 for {well_identifier}: {e}")

            days_for_file = sorted(discovered)
            print(f"  → auto-days for {well_identifier}: {days_for_file}")
        else:
            days_for_file = [normalize_timepoint(day) for day in days]

        for day in days_for_file:
            create_density_map_for_timepoint(
                pointcloud_filepath=pointcloud_filepath,
                directory_path=directory_path,
                timepoint=day,
                radius_um=radius_um,
                density_maps_folder=density_maps_folder_for_well,
                coarsening_factor=coarsening_factor
            )


def stack_density_maps(experiment_path, plate, scaling_factor=1,
                      loading='density_maps_epanechnikov',
                      saving='density_stacks_epanechnikov',
                      addon='density_epanechnikov',
                      coarsening_factor: int = DEFAULT_COARSENING_FACTOR,
                      radius_um: int | float | None = None,
                      stack_mode='multichannel',
                      stack_channels_order=DENSITY_STACK_CHANNELS):
    """
    Reads all density tif files in a given directory and merges them to stack TIFFs.
    Stored maps and output stacks use float32 values in cells / µm².
    """
    loading_path = os.path.join(experiment_path, plate, loading)
    saving_path = os.path.join(experiment_path, plate, saving)
    os.makedirs(saving_path, exist_ok=True)

    image_timesteps = {}
    tiffiles = [f for f in os.listdir(loading_path) if f.endswith('.tif')]

    for filename in tiffiles:
        scene = '_'.join(filename.split('_')[0:2])
        timestep = re.findall(r'_(\d\d)_', filename)[0]
        channel = re.findall(r'_(res|sus|total)_', filename)[0]

        image_timesteps.setdefault(scene, {}).setdefault(channel, {})[timestep] = filename

    if isinstance(stack_mode, str):
        stack_modes = {stack_mode}
    else:
        stack_modes = set(stack_mode)
    stack_channels_order = tuple(stack_channels_order)

    radius_description = ''
    if radius_um is not None:
        radius_description = (
            f"radius_um={radius_um}; "
            f"radius_original_px={radius_um_to_original_px(radius_um)}; "
            f"radius_coarse_px={radius_um_to_coarse_px(radius_um, coarsening_factor)}; "
        )

    description = (
        f"kernel=epanechnikov_normalized; units=cells_per_um2; "
        f"{radius_description}"
        f"coarsening_factor={coarsening_factor}; "
        f"coarse_pixel_size_um={PIXEL_SIZE_UM * coarsening_factor}; "
        f"stored_as=float32; "
        f"pixel_size_um={PIXEL_SIZE_UM}"
    )

    def padded_image(filename, total_height, max_width):
        img = tiff.imread(os.path.join(loading_path, filename))

        x_offset = round(((max_width - img.shape[1]) / 2) * scaling_factor)
        y_offset = round(((total_height - img.shape[0]) / 2) * scaling_factor)

        concatenated_image = np.zeros(
            (round(total_height * scaling_factor), round(max_width * scaling_factor)),
            dtype=np.float32
        )

        if scaling_factor != 1:
            resized_img = skimage.transform.rescale(
                img.astype(np.float32, copy=False),
                scaling_factor,
                anti_aliasing=True,
                preserve_range=True,
            )
        else:
            resized_img = img

        resized_img = resized_img.astype(np.float32, copy=False)

        concatenated_image[
            y_offset:y_offset + resized_img.shape[0],
            x_offset:x_offset + resized_img.shape[1]
        ] = resized_img
        return concatenated_image

    for scene, scene_channels in image_timesteps.items():
        if 'separate_channels' in stack_modes:
            for channel, timesteps in scene_channels.items():
                print(f"Processing scene {scene} channel {channel} with {len(timesteps)} timepoints")
                sorted_timesteps = sorted(timesteps.keys())

                max_width = max(tiff.imread(os.path.join(loading_path, timesteps[t])).shape[1] for t in sorted_timesteps)
                total_height = max(tiff.imread(os.path.join(loading_path, timesteps[t])).shape[0] for t in sorted_timesteps)
                print(f"Max width: {max_width}, Total height: {total_height}")

                concatenated_images = []

                for t in sorted_timesteps:
                    print(f"Processing timepoint {t}")
                    concatenated_images.append(
                        padded_image(timesteps[t], total_height, max_width)
                    )

                merged_filename = scene + '_' + channel + '_' + addon + '.tif'
                merged_filepath = os.path.join(saving_path, merged_filename)
                if not OVERWRITE and os.path.exists(merged_filepath):
                    print(f"Skipping existing density stack: {merged_filepath} (set OVERWRITE = True to replace).")
                    continue

                channel_description = f"{description}; axes=TYX; channel={channel}"
                tiff.imwrite(
                    merged_filepath,
                    concatenated_images,
                    dtype='float32',
                    description=channel_description,
                    compression='deflate',
                )
                print(f"Saved merged image stack for scene {scene} channel {channel} to {merged_filepath}")

        if 'multichannel' in stack_modes:
            stack_channels = tuple(
                channel for channel in stack_channels_order if channel in scene_channels
            )
            if not stack_channels:
                print(f"Skipping scene {scene}; no density channels found.")
                continue
            missing_channels = [
                channel for channel in stack_channels_order if channel not in scene_channels
            ]
            if missing_channels:
                print(
                    f"Scene {scene} is missing channels {missing_channels}; "
                    f"writing available channels {list(stack_channels)}."
                )

            common_timesteps = set(scene_channels[stack_channels[0]].keys())
            for channel in stack_channels[1:]:
                common_timesteps &= set(scene_channels[channel].keys())
            sorted_timesteps = sorted(common_timesteps)
            if not sorted_timesteps:
                print(f"Skipping scene {scene}; no complete multichannel timepoints.")
                continue

            for channel in stack_channels:
                missing_timesteps = sorted(
                    set().union(*(set(scene_channels[c].keys()) for c in stack_channels))
                    - set(scene_channels[channel].keys())
                )
                if missing_timesteps:
                    print(
                        f"Scene {scene} channel {channel} is missing timepoints "
                        f"{missing_timesteps}; multichannel stack uses complete "
                        f"timepoints {sorted_timesteps}."
                    )

            max_width = 0
            total_height = 0
            for channel in stack_channels:
                for timestep in sorted_timesteps:
                    shape = tiff.imread(os.path.join(loading_path, scene_channels[channel][timestep])).shape
                    total_height = max(total_height, shape[0])
                    max_width = max(max_width, shape[1])
            print(
                f"Processing scene {scene} multichannel stack with "
                f"{len(sorted_timesteps)} timepoints, channels={list(stack_channels)}"
            )
            print(f"Max width: {max_width}, Total height: {total_height}")

            multichannel_images = np.zeros(
                (
                    len(sorted_timesteps),
                    len(stack_channels),
                    round(total_height * scaling_factor),
                    round(max_width * scaling_factor),
                ),
                dtype=np.float32,
            )
            for time_index, timestep in enumerate(sorted_timesteps):
                print(f"Processing timepoint {timestep}")
                for channel_index, channel in enumerate(stack_channels):
                    multichannel_images[time_index, channel_index] = padded_image(
                        scene_channels[channel][timestep],
                        total_height,
                        max_width,
                    )

            merged_filename = scene + '_' + addon + '.tif'
            merged_filepath = os.path.join(saving_path, merged_filename)
            if not OVERWRITE and os.path.exists(merged_filepath):
                print(f"Skipping existing density stack: {merged_filepath} (set OVERWRITE = True to replace).")
                continue
            stack_description = (
                f"{description}; axes=TCYX; "
                f"channels={','.join(stack_channels)}; "
                f"timepoints={','.join(sorted_timesteps)}"
            )
            tiff.imwrite(
                merged_filepath,
                multichannel_images,
                dtype='float32',
                imagej=True,
                metadata={
                    'axes': 'TCYX',
                    'channels': len(stack_channels),
                    'frames': len(sorted_timesteps),
                    'Labels': list(stack_channels),
                    'Info': stack_description,
                },
                compression='deflate',
            )
            print(f"Saved multichannel density stack for scene {scene} to {merged_filepath}")

    return


def main():
    global OVERWRITE
    args = parse_args()
    OVERWRITE = bool(args.overwrite)
    config = load_config(args.config)

    config_experiment_path = config.get('experiment_path')
    config_mount_root = config.get('mount_root')
    config_hdf5_path = config.get('hdf5_path')

    experiment_path = resolve_experiment_path(
        experiment_path=args.experiment_path or os.environ.get(EXPERIMENT_PATH_ENV) or config_experiment_path,
        mount_root=args.mount_root or os.environ.get(MOUNT_ROOT_ENV) or config_mount_root,
        hdf5_path=config_hdf5_path,
    )
    print(f"Using experiment path: {experiment_path}")

    # Epanechnikov kernel setting: support radius / bandwidth in physical units.
    radius_um = resolve_radius_um(config)
    coarsening_factor = resolve_coarsening_factor(config, radius_um)

    poisson = config_bool(config, 'poisson', True)
    create_maps = pipeline_output_enabled(config, 'density_maps', 'create_maps', True)
    stack_maps = pipeline_output_enabled(config, 'density_stacks', 'stack_maps', True)
    test_mode = config_bool(config, 'test_mode', False)

    plates = normalize_plates(args.plates or config.get('plates'))
    if not plates and config_hdf5_path:
        inferred_plate = infer_plate_from_hdf5_path(config_hdf5_path)
        plates = [inferred_plate] if inferred_plate else None
    if not plates:
        plates = DEFAULT_PLATES

    # Output folders (Epanechnikov)
    coarsening_suffix = '' if coarsening_factor == 1 else f'_{coarsening_factor}x'
    test_parameter_suffix = f'_{radius_um_folder_token(radius_um)}_{coarsening_factor}x'
    density_maps_folder = config.get(
        'density_maps_folder',
        f'density_maps_epanechnikov{coarsening_suffix}_poisson'
    )
    density_stacks_folder = config.get(
        'density_stacks_folder',
        f'density_stacks_epanechnikov{coarsening_suffix}_poisson'
    )
    structured_output = use_structured_output_layout(config)
    stack_modes = density_stack_modes(config)
    print(f"Processing plates: {', '.join(plates)}")
    if structured_output:
        print(f"Using output run directory: {resolve_output_run_dir(config, experiment_path)}")
    print(
        f"Using kernel radius: {radius_um:g} µm "
        f"({radius_um_to_original_px(radius_um):.3f} original px, "
        f"{radius_um_to_coarse_px(radius_um, coarsening_factor):.3f} coarse px)"
    )

    if test_mode:
        if len(plates) != 1:
            raise ValueError('test_mode expects exactly one plate in config or --plates.')
        test_hdf5_file = config.get('test_hdf5_file')
        test_timepoints = normalize_timepoints(config)
        if not test_hdf5_file or not test_timepoints:
            raise ValueError('test_mode requires test_hdf5_file and test_timepoint/test_timepoints in the config.')

        test_output_folder = config.get('test_output_folder', DEFAULT_TEST_OUTPUT_FOLDER)
        test_change_channels = tuple(config.get('test_change_channels', ['res']))
        pointcloud_filepath = resolve_pointcloud_file(
            experiment_path=experiment_path,
            plate=plates[0],
            hdf5_filename=test_hdf5_file,
            hdf5_path=config_hdf5_path,
        )
        well_identifier = well_identifier_from_pointcloud_filename(pointcloud_filepath)
        if structured_output:
            test_density_maps_folder = os.fspath(
                well_output_type_dir(
                    config,
                    experiment_path,
                    plates[0],
                    well_identifier,
                    'density_maps_output_type',
                    'density_maps',
                )
            )
            test_density_changes_folder = os.fspath(
                well_output_type_dir(
                    config,
                    experiment_path,
                    plates[0],
                    well_identifier,
                    'density_changes_output_type',
                    'density_changes',
                )
            )
        else:
            test_density_maps_folder = resolve_test_output_subfolder(
                config.get(
                    'test_density_maps_folder',
                    f'density_maps_epanechnikov{test_parameter_suffix}_poisson_test',
                ),
                test_output_folder,
            )
            test_density_changes_folder = resolve_test_output_subfolder(
                config.get(
                    'test_density_changes_folder',
                    f'density_changes_epanechnikov{test_parameter_suffix}_poisson_test'
                ),
                test_output_folder,
            )
        print(
            "Running test mode: "
            f"plate={plates[0]}, file={os.path.basename(pointcloud_filepath)}, "
            f"timepoints={', '.join(test_timepoints)}"
        )
        if not os.path.exists(pointcloud_filepath):
            raise FileNotFoundError(f'Test HDF5 file does not exist: {pointcloud_filepath}')

        directory_path = os.path.join(experiment_path, plates[0])
        for test_timepoint in test_timepoints:
            create_density_map_for_timepoint(
                pointcloud_filepath=pointcloud_filepath,
                directory_path=directory_path,
                timepoint=test_timepoint,
                radius_um=radius_um,
                density_maps_folder=test_density_maps_folder,
                coarsening_factor=coarsening_factor,
                require_timepoint=True,
            )

        write_test_changes = pipeline_output_enabled(
            config,
            'density_changes',
            'create_density_changes',
            config_bool(config, 'test_write_changes', True),
        )
        if write_test_changes:
            write_test_density_changes(
                directory_path=directory_path,
                well_identifier=well_identifier,
                timepoints=test_timepoints,
                density_maps_folder=test_density_maps_folder,
                density_changes_folder=test_density_changes_folder,
                channels=test_change_channels,
                write_zero_initial_growth_channel=config_bool(
                    config,
                    'write_zero_initial_growth_channel',
                    False,
                ),
                zero_initial_density_threshold=config.get(
                    'zero_initial_res_density_threshold_um2',
                    DENSITY_ZERO_FLOOR_UM2,
                ),
                measurable_growth_threshold=config.get(
                    'res_growth_threshold_um2',
                    DENSITY_ZERO_FLOOR_UM2,
                ),
            )
        return

    # ---- 1) CREATE per-timepoint density maps (auto-discover days per well) ----
    if create_maps:
        for plate in plates:
            well_designators = plate_well_designators(config, plate)
            create_density_maps(
                experiment_path=experiment_path,
                plate=plate,
                days=None,
                radius_um=radius_um,
                density_maps_folder=density_maps_folder,
                poisson=poisson,
                coarsening_factor=coarsening_factor,
                hdf5_include_patterns=config.get('hdf5_include_patterns'),
                hdf5_exclude_patterns=config.get('hdf5_exclude_patterns'),
                well_designators=well_designators,
                output_config=config,
            )


    # ---- 2) STACK density maps into timestacks (per scene & channel) ----
    if stack_maps:
        for plate in plates:
            well_designators = plate_well_designators(config, plate)
            if structured_output:
                hdf5_directory = os.path.join(experiment_path, plate, 'hdf5')
                for filename in pointcloud_filenames(
                    hdf5_directory,
                    poisson,
                    include_patterns=config.get('hdf5_include_patterns'),
                    exclude_patterns=config.get('hdf5_exclude_patterns'),
                    well_designators=well_designators,
                ):
                    well_identifier = well_identifier_from_pointcloud_filename(filename)
                    stack_density_maps(
                        experiment_path,
                        plate,
                        loading=os.fspath(
                            well_output_type_dir(
                                config,
                                experiment_path,
                                plate,
                                well_identifier,
                                'density_maps_output_type',
                                'density_maps',
                            )
                        ),
                        saving=os.fspath(
                            well_output_type_dir(
                                config,
                                experiment_path,
                                plate,
                                well_identifier,
                                'density_stacks_output_type',
                                'density_stacks',
                            )
                        ),
                        addon='density_epanechnikov',
                        coarsening_factor=coarsening_factor,
                        radius_um=radius_um,
                        stack_mode=stack_modes,
                    )
            else:
                stack_density_maps(
                    experiment_path,
                    plate,
                    loading=density_maps_folder,
                    saving=density_stacks_folder,
                    addon='density_epanechnikov',
                    coarsening_factor=coarsening_factor,
                    radius_um=radius_um,
                    stack_mode=stack_modes,
                )

    pass

if __name__ == "__main__":
    main()
