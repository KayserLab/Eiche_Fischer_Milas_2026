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
- Plate/day selection and kernel radius parameters.

Outputs:
- Per-day density TIFFs for susceptible, resistant, or total populations.
- Time-stacked density TIFF volumes per well.

How it works:
- Reads cell-center coordinates from HDF5 point clouds.
- Rasterizes cell positions onto the image canvas.
- Applies a circular Epanechnikov smoothing kernel to estimate local cell
  density in physical space.
- Encodes density maps and optionally stacks all available days for each well
  into a single time-ordered TIFF.

Pipeline role:
- Core spatial summarization step that turns point clouds into the density
  fields later used for confinement, correlation, and TTP-vs-density analyses.
"""

import h5py
import os
import math
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.signal import fftconvolve
import glob, re
from tifffile import imread, imwrite
import skimage

from skimage.morphology import remove_small_holes, binary_closing, disk
from skimage.segmentation import find_boundaries

PIXEL_SIZE_UM = 1.095
PIXEL_AREA_UM2 = PIXEL_SIZE_UM ** 2

# Stored TIFF values are scaled integers. Recover cells / µm² in analysis as:
#   density_um2 = stored_int16 / DENSITY_SCALE_FACTOR
# Chosen so total-density maps remain safely inside int16 even if sus and res
# overlap on the same pixel (upper bound: 2 / PIXEL_AREA_UM2 cells / µm²).
DENSITY_SCALE_FACTOR = 19000
INT16_MAX = np.iinfo(np.int16).max


def validate_scale_factor_configuration():
    """
    Check the configured scale factor against a conservative theoretical upper
    bound for the normalized total-density map in cells / um^2.
    """
    theoretical_total_max_um2 = 2.0 / PIXEL_AREA_UM2
    theoretical_total_max_scaled = theoretical_total_max_um2 * DENSITY_SCALE_FACTOR

    if theoretical_total_max_scaled > INT16_MAX:
        raise ValueError("Unsafe DENSITY_SCALE_FACTOR configuration.")


def pointcloud_read(filename: str, time: str, unit: str = 'd'):
    """
    Reads x,y positions for sus/res cells from a plain HDF5 (not a PyTables store).
    If the dataset is missing or has the wrong shape, returns empty DataFrames.

    :param filename: .h5 file
    :param time:    zero-padded timepoint, e.g. '00'
    :param unit:    prefix in the HDF5 keys, e.g. 'd' → groups like 'sus/d_00'
    :return:        (df_sus, df_res) where each is a pd.DataFrame(columns=['x','y'])
    """
    df_sus = pd.DataFrame(columns=['x', 'y'])
    df_res = pd.DataFrame(columns=['x', 'y'])

    with h5py.File(filename, 'r') as f:
        sus_path = f'sus/{unit}_{time}'
        res_path = f'res/{unit}_{time}'

        if sus_path in f:
            arr = f[sus_path]['block0_values'][:]
            if arr.ndim == 2 and arr.shape[1] == 2:
                df_sus = pd.DataFrame({'x': arr[:, 1], 'y': arr[:, 0]})

        if res_path in f:
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


def epanechnikov_circular_kernel(radius: int) -> np.ndarray:
    """
    Normalized Epanechnikov kernel with compact support (disk of radius R).
    Kernel:
        K(r) = (2 / (pi * R^2)) * (1 - r^2 / R^2)   for r <= R
             = 0                                    for r > R

    Returns a float32 kernel of shape (2R+1, 2R+1).
    """
    R = float(radius)
    L = np.arange(-radius, radius + 1, dtype=np.float32)
    X, Y = np.meshgrid(L, L)
    r2 = X * X + Y * Y

    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    mask = r2 <= (R * R)

    kernel[mask] = (2.0 / (math.pi * R * R)) * (1.0 - (r2[mask] / (R * R)))
    return kernel


def load_overexposed_regions(filename: str):
    return tiff.imread(filename)


def encode_density_to_int16(density_um2: np.ndarray) -> np.ndarray:
    """
    Convert cells / µm² to scaled int16 for compact TIFF storage.
    Analysis scripts must divide stored values by DENSITY_SCALE_FACTOR to
    recover physical units.
    """
    scaled = np.rint(density_um2 * DENSITY_SCALE_FACTOR)

    if scaled.min() < np.iinfo(np.int16).min or scaled.max() > np.iinfo(np.int16).max:
        raise ValueError(
            f"Density values exceed int16 range after scaling by {DENSITY_SCALE_FACTOR}. "
            "Choose a smaller scale factor."
        )

    return scaled.astype(np.int16)


def get_local_density_fft(results_array: np.ndarray, radius: int) -> np.ndarray:
    """
    Convolve a binary occupancy image with the normalized Epanechnikov kernel
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

    return local_density.astype(np.float32)


def density(df_sus: pd.DataFrame, df_res: pd.DataFrame, radius: int, image: np.ndarray):
    # Determine image size by using the dimensions of image
    height, width = image.shape[-2], image.shape[-1]

    # Initialize blank images for each cell type
    image_resistant = np.zeros((height, width), dtype=np.uint8)
    image_susceptible = np.zeros((height, width), dtype=np.uint8)

    # Fill in each image (binary occupancy)
    for _, row in df_res.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= x < width and 0 <= y < height:
            image_resistant[y, x] = 1

    for _, row in df_sus.iterrows():
        x, y = int(row['x']), int(row['y'])
        if 0 <= x < width and 0 <= y < height:
            image_susceptible[y, x] = 1

    image_total = image_resistant + image_susceptible

    density_resistant = get_local_density_fft(image_resistant, radius)
    density_susceptible = get_local_density_fft(image_susceptible, radius)
    density_total = get_local_density_fft(image_total, radius)

    # Convert from cells / pixel² to cells / µm² using 1 px = 1.095 µm.
    density_resistant = density_resistant / PIXEL_AREA_UM2
    density_susceptible = density_susceptible / PIXEL_AREA_UM2
    density_total = density_total / PIXEL_AREA_UM2

    return density_susceptible, density_resistant, density_total


def create_density_maps(experiment_path, plate, days=None, radius=100, density_maps_folder='density_maps_epanechnikov', poisson=False):
    validate_scale_factor_configuration()
    directory_path = os.path.join(experiment_path, plate)
    directory = os.path.join(directory_path, 'hdf5')

    if poisson:
        filenames = [f for f in os.listdir(directory) if f.endswith('_total_poisson.h5')]
    else:
        filenames = [f for f in os.listdir(directory) if f.endswith('_total.h5')]

    for file in filenames:
        print(file)
        pointcloud_filepath = os.path.join(directory, file)
        if poisson == True:
            well_identifier = file.split('_')[2:-2]
        else:
            well_identifier = file.split('_')[2:-1]
        well_identifier = '_'.join(well_identifier).split('.')[0]

        # Auto-discover days per well if not provided
        if days is None:
            discovered = set()
            try:
                with h5py.File(pointcloud_filepath, 'r') as fh:
                    for grp in ('sus', 'res'):
                        if grp in fh:
                            for key in fh[grp].keys():
                                m = re.match(r'^d_(\d{2})$', key)
                                if m:
                                    discovered.add(m.group(1))
            except Exception as e:
                print(f"  ⚠ Failed to inspect HDF5 for {well_identifier}: {e}")

            days_for_file = sorted(discovered)
            print(f"  → auto-days for {well_identifier}: {days_for_file}")
        else:
            days_for_file = list(days)

        for day in days_for_file:
            timepoint = day

            masks_positions_sus, masks_positions_res = pointcloud_read(pointcloud_filepath, timepoint)

            overexposed_regions_paths = os.path.join(directory_path, 'overexposed_images')
            overexposed_regions_sus_filepath = os.path.join(
                overexposed_regions_paths, f'{well_identifier}_{timepoint}_sus_overexposed_img.tif'
            )

            if os.path.exists(overexposed_regions_sus_filepath):
                overexposed_regions_sus = load_overexposed_regions(overexposed_regions_sus_filepath)
            else:
                print(f'File {overexposed_regions_sus_filepath} does not exist')
                continue

            sus_density, res_density, total_density = density(
                masks_positions_sus, masks_positions_res,
                radius=radius,
                image=overexposed_regions_sus
            )

            sus_density_int16 = encode_density_to_int16(sus_density)
            res_density_int16 = encode_density_to_int16(res_density)
            total_density_int16 = encode_density_to_int16(total_density)

            density_maps_directory = os.path.join(directory_path, density_maps_folder)
            os.makedirs(density_maps_directory, exist_ok=True)
            description = (
                f"kernel=epanechnikov_normalized; radius_px={radius}; "
                f"units=cells_per_um2; stored_as=int16_scaled; "
                f"scale_factor={DENSITY_SCALE_FACTOR}; "
                f"recover_by=stored_value/{DENSITY_SCALE_FACTOR}; "
                f"pixel_size_um={PIXEL_SIZE_UM}"
            )

            tiff.imwrite(
                os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_sus_density_epanechnikov.tif'),
                sus_density_int16,
                description=description,
                compression='deflate'
            )
            tiff.imwrite(
                os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_res_density_epanechnikov.tif'),
                res_density_int16,
                description=description,
                compression='deflate'
            )
            tiff.imwrite(
                os.path.join(density_maps_directory, f'{well_identifier}_{timepoint}_total_density_epanechnikov.tif'),
                total_density_int16,
                description=description,
                compression='deflate'
            )


def stack_density_maps(experiment_path, plate, scaling_factor=1,
                      loading='density_maps_epanechnikov',
                      saving='density_stacks_epanechnikov',
                      addon='density_epanechnikov'):
    """
    Reads all density tif files in a given directory and merges them to a timestack.
    Stored maps use scaled int16 values. Downstream analysis must divide by
    DENSITY_SCALE_FACTOR to recover cells / µm².
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

    for scene, channels in image_timesteps.items():
        for channel, timesteps in channels.items():
            print(f"Processing scene {scene} channel {channel} with {len(timesteps)} timepoints")
            sorted_timesteps = sorted(timesteps.keys())

            max_width = max(tiff.imread(os.path.join(loading_path, timesteps[t])).shape[1] for t in sorted_timesteps)
            total_height = max(tiff.imread(os.path.join(loading_path, timesteps[t])).shape[0] for t in sorted_timesteps)
            print(f"Max width: {max_width}, Total height: {total_height}")

            concatenated_images = []

            for t in sorted_timesteps:
                print(f"Processing timepoint {t}")
                img = tiff.imread(os.path.join(loading_path, timesteps[t]))

                x_offset = round(((max_width - img.shape[1]) / 2) * scaling_factor)
                y_offset = round(((total_height - img.shape[0]) / 2) * scaling_factor)

                concatenated_image = np.zeros(
                    (round(total_height * scaling_factor), round(max_width * scaling_factor)),
                    dtype=np.int16
                )

                if scaling_factor != 1:
                    resized_img = skimage.transform.rescale(img, scaling_factor, anti_aliasing=True)
                    resized_img = np.rint(resized_img)
                else:
                    resized_img = img

                resized_img = resized_img.astype(np.int16)

                concatenated_image[
                    y_offset:y_offset + resized_img.shape[0],
                    x_offset:x_offset + resized_img.shape[1]
                ] = resized_img

                concatenated_images.append(concatenated_image)

            merged_filename = scene + '_' + channel + '_' + addon + '.tif'
            merged_filepath = os.path.join(saving_path, merged_filename)

            description = (
                f"kernel=epanechnikov_normalized; units=cells_per_um2; "
                f"stored_as=int16_scaled; scale_factor={DENSITY_SCALE_FACTOR}; "
                f"recover_by=stored_value/{DENSITY_SCALE_FACTOR}; "
                f"pixel_size_um={PIXEL_SIZE_UM}"
            )
            tiff.imwrite(merged_filepath, concatenated_images, dtype='int16', description=description, compression='deflate')
            print(f"Saved merged image stack for scene {scene} channel {channel} to {merged_filepath}")

    return


def main():
    experiment_path = r'D:\2025_08_CoreSeed_Large_Testdataset'

    # Epanechnikov kernel setting: support radius / bandwidth
    radius = 40

    poisson = True

    # Output folders (Epanechnikov)
    density_maps_folder = 'density_maps_epanechnikov_poisson'
    density_stacks_folder = 'density_stacks_epanechnikov_poisson'

    # ---- 1) CREATE per-timepoint density maps (auto-discover days per well) ----
    # Uncomment plates you want to process

    #create_density_maps(experiment_path=experiment_path, plate='Plate_1', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    #create_density_maps(experiment_path=experiment_path, plate='Plate_2', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    create_density_maps(experiment_path=experiment_path, plate='Plate_3', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    create_density_maps(experiment_path=experiment_path, plate='Plate_4', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    create_density_maps(experiment_path=experiment_path, plate='Plate_5', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    create_density_maps(experiment_path=experiment_path, plate='Plate_6', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    create_density_maps(experiment_path=experiment_path, plate='Plate_7', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)
    create_density_maps(experiment_path=experiment_path, plate='Plate_8', days=None, radius=radius, density_maps_folder=density_maps_folder, poisson=poisson)


    # ---- 2) STACK density maps into timestacks (per scene & channel) ----
    # Uncomment plates you want to stack

    #stack_density_maps(experiment_path, 'Plate_1', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    #stack_density_maps(experiment_path, 'Plate_2', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    stack_density_maps(experiment_path, 'Plate_3', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    stack_density_maps(experiment_path, 'Plate_4', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    stack_density_maps(experiment_path, 'Plate_5', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    stack_density_maps(experiment_path, 'Plate_6', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    stack_density_maps(experiment_path, 'Plate_7', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')
    stack_density_maps(experiment_path, 'Plate_8', loading=density_maps_folder, saving=density_stacks_folder, addon='density_epanechnikov')

    pass

if __name__ == "__main__":
    main()
