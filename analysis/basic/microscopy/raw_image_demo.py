"""Run the selected full-frame, native-CZI image demonstration on CPU.

Segmentation is freshly computed with the exact custom Cellpose 3 weights.
Historical masks/counts/coordinates are comparisons, never substituted outputs.
The main cohort continues to use the deposited corrected observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import pandas as pd
import tifffile
import torch
from aicspylibczi import CziFile
from cellpose import models

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shared.data_paths import DATA_ROOT, MODELS
from analysis.basic.microscopy.demo_measurements import calculate_centers_2d, saturation_measurement
from analysis.basic.microscopy.total_positions_correction_comparison import create_total_positions_poisson_continuous_exact

DEMO_NAME = 'Plate3_D4_AT100_tp018_day040'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-root', type=Path, default=DATA_ROOT / 'Raw_Demo' / DEMO_NAME)
    parser.add_argument('--output-dir', type=Path, default=DATA_ROOT / 'intermediate_analysis_data/raw_demo' / DEMO_NAME)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if metadata.version('cellpose') != '3.0.11':
        parser.error('Use the declared Cellpose 3.0.11 environment')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    settings = json.loads((args.demo_root / 'settings.json').read_text(encoding='utf-8'))
    image = CziFile(args.demo_root / '20251010_d40_Plate3_MIPs.czi')
    box = image.get_mosaic_scene_bounding_box(index=0)
    planes = [np.squeeze(image.read_mosaic(C=c, scale_factor=1.0,
              region=(box.x, box.y, box.w, box.h))) for c in range(3)]
    pixels = np.stack(planes, axis=-1)  # BF/GFP/RFP -> YXC; production selectors remain [2,0]/[3,0].
    del planes, image
    segmentations, rows, detected = {}, [], {}
    for channel, selector, weight in (
        ('res', [3, 0], 'a4_cyto2_PhD_1000_new'),
        ('sus', [2, 0], 'a2_cyto3_Epi_1000_PhD')):
        weights = MODELS / 'Cellpose' / weight
        print(f'Fresh CPU segmentation: {channel}; full frame {pixels.shape}; {weights}', flush=True)
        model = models.CellposeModel(gpu=False, pretrained_model=str(weights))
        masks, flows, styles = model.eval(pixels, channel_axis=-1, diameter=None,
            channels=selector, min_size=250, flow_threshold=0,
            cellprob_threshold=-1, stitch_threshold=0, do_3D=False)
        tifffile.imwrite(args.output_dir / f'{channel}_instance_masks.tif', masks,
                         compression='deflate', photometric='minisblack')
        segmentations[channel] = masks
        print(f'{channel}: {int(masks.max())} labels', flush=True)
        del model, flows, styles
        gc.collect()
    sensitive, resistant = calculate_centers_2d(segmentations['sus'], segmentations['res'])
    detected.update(sus=sensitive, res=resistant)
    corrected = {}
    for channel, plane_index, expected_detected, expected_added in (
        ('sus', 1, 30529, 2471), ('res', 2, 931, 735)):
        mask = segmentations[channel]
        additions, saturated = saturation_measurement(pixels[..., plane_index], mask)
        tifffile.imwrite(args.output_dir / f'{channel}_saturation_region.tif', saturated.astype(np.uint8),
                         compression='deflate', photometric='minisblack')
        total, new = create_total_positions_poisson_continuous_exact(detected[channel], saturated, additions,
            r_start=15, r_min=4, relax_factor=0.85, oversample_factor=40, max_rounds=20,
            seed=0 if channel == 'sus' else 77, connectivity=2, do_mask_cleanup=True,
            cleanup_open_iters=1, cleanup_close_iters=1, min_island_size=5,
            sampling_mode='global', block_against_existing=False, r_existing=2.0)
        offset = np.asarray(settings['padding_offset_yx'], dtype=float)
        total = total.copy()
        total[['y', 'x']] = total[['y', 'x']].to_numpy() + offset
        corrected[channel] = total
        detected[channel].to_csv(args.output_dir / f'{channel}_detected_yx.csv', index=False)
        total.to_csv(args.output_dir / f'{channel}_corrected_padded_yx.csv', index=False)
        historical_file = next(p for p in (args.demo_root / 'Reference').glob('*_masks.tif')
                               if f'[{2 if channel == "sus" else 3}, 0]' in p.name)
        historical = tifffile.imread(historical_file)
        with h5py.File(args.demo_root / 'Reference/corrected_padded.h5', 'r') as h5:
            historical_points = h5[f'{channel}/d_40/block0_values'][()]
        row = {'population': channel, 'detected': len(detected[channel]), 'saturation_added': additions,
            'corrected': len(total), 'historical_detected': expected_detected,
            'historical_saturation_added': expected_added,
            'historical_corrected': len(historical_points),
            'instance_labels_exactly_equal': bool(np.array_equal(mask, historical)),
            'foreground_disagreement_pixels': int(np.count_nonzero((mask > 0) != (historical > 0))),
            'corrected_coordinates_exactly_equal': bool(np.array_equal(total.to_numpy(), historical_points))}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del historical, historical_points, saturated
    pd.DataFrame(rows).to_csv(args.output_dir / 'historical_comparison.csv', index=False)
    with h5py.File(args.output_dir / 'corrected_coordinates.h5', 'w') as handle:
        for channel, table in corrected.items():
            handle.create_dataset(f'{channel}/d_40/block0_values', data=table.to_numpy(), compression='gzip')
        handle.attrs['coordinate_order'] = 'y,x'
        handle.attrs['day'] = 40
        handle.attrs['poisson_seed_sus'] = 0
        handle.attrs['poisson_seed_res'] = 77
    report = {'status': 'computed', 'historical_agreement': rows,
        'historical_random_state_claim': 'released seeds fixed; historical equality is measured, not assumed',
        'cellpose': metadata.version('cellpose'), 'torch': metadata.version('torch'),
        'python': sys.version, 'platform': sys.platform, 'device': 'cpu',
        'threads': args.threads, 'seconds': time.monotonic()-started,
        'settings': settings, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    (args.output_dir / 'demo_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Full-frame raw demo completed in {report["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
