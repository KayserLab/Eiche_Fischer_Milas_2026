"""Inspect the full simulation example using the existing production reader."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    data = args.data_root.resolve()
    os.environ['EICHE_FISCHER_DATA_ROOT'] = str(data)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared.data_paths import DATA_ROOT
    data = DATA_ROOT

    import h5py
    import numpy as np
    from analysis.basic.density_maps.Prepare_Simulation_Pointclouds import (
        default_settings, load_filtered_points_px,
    )

    folder = data / 'Raw_Demo/Simulation_Full_State/data-DB_Dens/run_25'
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    full = folder / 'simulation_004000000.h5'
    reduced = data / metadata['reduced_relative_path']
    with full.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != metadata['source_sha256']:
        raise ValueError('Complete simulation example differs from its original source hash')
    inventory = {}
    for label, path in (('full', full), ('reduced', reduced)):
        fields = []
        with h5py.File(path, 'r') as handle:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    fields.append({'name': name, 'shape': None if obj.shape is None else list(obj.shape),
                                   'dtype': str(obj.dtype)})
            handle.visititems(visit)
        inventory[label] = fields
    if len(inventory['full']) != metadata['full_dataset_count']:
        raise ValueError('Unexpected full-snapshot dataset count')
    full_names = {row['name'].rsplit('/', 1)[-1].strip('"') for row in inventory['full']}
    reduced_names = {row['name'].rsplit('/', 1)[-1].strip('"') for row in inventory['reduced']}
    if not set(metadata['fields_omitted_from_reduced']).issubset(full_names):
        raise ValueError('Complete example is missing declared original fields')
    if set(metadata['fields_omitted_from_reduced']) & reduced_names:
        raise ValueError('Reduced example unexpectedly contains omitted fields')

    settings = default_settings()
    full_points, full_classes, full_counts = load_filtered_points_px(full, settings, with_overview=True)
    reduced_points, reduced_classes, reduced_counts = load_filtered_points_px(reduced, settings, with_overview=True)
    np.testing.assert_array_equal(full_points, reduced_points)
    np.testing.assert_array_equal(full_classes, reduced_classes)
    if full_counts != reduced_counts:
        raise ValueError('Production overview counts differ between full and reduced snapshots')

    output = args.output_dir or data / 'Reproduction_Report/simulation_snapshot_demo'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'prepared_points.npz', positions_px=full_points, classes=full_classes)
    report = {'status': 'passed', 'source_sha256': digest, 'datasets': inventory,
              'production_reader': 'Prepare_Simulation_Pointclouds.load_filtered_points_px',
              'settings': asdict(settings), 'prepared_cells': len(full_points),
              'overview_counts': full_counts, 'full_and_reduced_results_identical': True,
              'scope': 'one unchanged snapshot and the existing production preprocessing; no simulator restart'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Simulation snapshot demonstration passed: {len(full_points)} prepared cells; report: {output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
