"""Export snapshot-density arrays and diagnostic overlays for the raw demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicspylibczi import CziFile
import tifffile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shared.data_paths import DATA_ROOT, figure_plot_dir
from analysis.figure_3.HDF5_analysis import compute_flux_vs_density_experiment as density
from analysis.figure_3.HDF5_analysis.constants import px_to_micron, prolif_min_neighbors

NAME = 'Plate3_D4_AT100_tp018_day040'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-radius-um', type=float, default=120)
    args = parser.parse_args()
    raw = DATA_ROOT / 'Raw_Demo' / NAME
    generated = DATA_ROOT / 'intermediate_analysis_data/raw_demo' / NAME
    plots = figure_plot_dir('Raw_Demo', NAME)
    report = json.loads((generated / 'demo_report.json').read_text(encoding='utf-8'))
    if report['status'] != 'computed':
        raise ValueError('Raw demo measurement stage is incomplete')
    with h5py.File(generated / 'corrected_coordinates.h5', 'r') as handle:
        grid, shape = density._fixed_well_grid(handle, kernel_radius_um=args.kernel_radius_um)
        total, sus, res, valid = density.local_densities(handle, 'd_40', grid, shape,
                                                        kernel_radius_um=args.kernel_radius_um)
    with h5py.File(generated / f'snapshot_density_{args.kernel_radius_um:g}um.h5', 'w') as handle:
        for name, values in [('grid_xy_pixels', grid), ('total_cells_per_pixel2', total),
                             ('sensitive_cells_per_pixel2', sus), ('resistant_cells_per_pixel2', res),
                             ('valid_support', valid)]:
            handle.create_dataset(name, data=values, compression='gzip')
        handle.attrs['kernel_radius_um'] = args.kernel_radius_um
        handle.attrs['min_neighbors'] = prolif_min_neighbors
        handle.attrs['analysis_pixel_size_um'] = px_to_micron
        handle.attrs['day'] = 40
        handle.attrs['coordinate_order'] = 'grid x,y; input point columns y,x'
        handle.attrs['native_pixel_size_um'] = 1.0952380952380954
    image = CziFile(raw / '20251010_d40_Plate3_MIPs.czi')
    box = image.get_mosaic_scene_bounding_box(index=0)
    display = []
    display_limits = []
    for channel in (1, 2):
        plane = np.squeeze(image.read_mosaic(C=channel, scale_factor=1.0,
                           region=(box.x, box.y, box.w, box.h)))[::8, ::8].astype(np.float32)
        lo, hi = np.percentile(plane, [1, 99.5])
        display.append(np.clip((plane-lo)/max(1, hi-lo), 0, 1)**0.5)
        display_limits.append([float(lo), float(hi)])
    blue = np.array([.15,.4,1.])
    yellow = np.array([1.,.8,.05])
    rgb = np.clip(display[0][...,None]*blue + display[1][...,None]*yellow, 0, 1)
    offsets = np.asarray(report['settings']['padding_offset_yx'])
    extent = (offsets[1], offsets[1]+box.w, offsets[0]+box.h, offsets[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(rgb, extent=extent)
    axes[0].set_title('Native image: sensitive / resistant')
    axes[1].imshow(rgb, extent=extent, alpha=.4)
    for channel, color in [('sus', 'deepskyblue'), ('res', 'gold')]:
        points = pd.read_csv(generated / f'{channel}_detected_yx.csv')
        axes[1].scatter(points['x']+offsets[1], points['y']+offsets[0], s=.2, color=color, rasterized=True)
    axes[1].set_title('Fresh detected cell centers')
    xx = grid[:,0].reshape(shape)
    yy = grid[:,1].reshape(shape)
    field = np.where(valid, total * 1000 / px_to_micron**2, np.nan)
    mesh = axes[2].pcolormesh(xx, yy, field, shading='auto', cmap='viridis')
    axes[2].invert_yaxis()
    fig.colorbar(mesh, ax=axes[2], label='cells / 1,000 µm²')
    axes[2].set_title(f'Snapshot density, R={args.kernel_radius_um:g} µm')
    for axis in axes:
        axis.set_aspect('equal')
        axis.set_xlabel('x (image pixels, padded)')
        axis.set_ylabel('y (image pixels, padded)')
    fig.savefig(plots / 'raw_demo_diagnostics.pdf', dpi=180)
    fig.savefig(plots / 'raw_demo_diagnostics.png', dpi=180)
    plt.close(fig)
    (generated / 'display_settings.json').write_text(json.dumps({
        'purpose': 'diagnostic display only; scientific input pixels remain unchanged',
        'preview_stride': 8, 'percentiles': [1,99.5], 'gamma': .5,
        'channel_limits': display_limits, 'colors_sensitive_resistant': [blue.tolist(), yellow.tolist()],
        'native_pixel_size_um': 1.0952380952380954,
        'analysis_pixel_size_um': px_to_micron, 'density_kernel_radius_um': args.kernel_radius_um,
        'single_frame_scope': 'no temporal tracking, growth rate, or time-to-escape inferred'},
        indent=2) + '\n', encoding='utf-8')
    print(f'Demo snapshot density and diagnostics saved: {plots}', flush=True)


if __name__ == '__main__':
    main()
