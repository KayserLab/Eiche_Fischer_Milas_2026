"""Fingerprint regenerated scientific outputs and compare with frozen results.

Checks logical table/array contents, decoded TIFF pixels and rendered PDF pages.
File-container timestamps, absolute build paths and narrowly listed execution
metadata are excluded. Numerical comparisons are exact in the pinned Windows
reference environment; a mismatch fails rather than increasing a tolerance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import pandas as pd
import pymupdf
import tifffile
from shared.data_paths import DATA_ROOT

POLICY = {
    'numerical_agreement': 'exact canonical values in the pinned Windows reference environments',
    'float_normalization': 'NaN payload and signed zero normalized; other finite values retain their full precision',
    'paths': 'only the declared code and Data prefixes replaced with <CODE> and <DATA>',
    'csv_execution_columns': {'*_summary.csv': ['elapsed_s']},
    'hdf5_provenance_attributes': ['source_signature', 'source_sha256', 'signature'],
    'pandas_hdf5_execution_columns': {'/metadata/run': ['created_at_utc']},
    'overview_cache_json': ['sources', 'coordinates'],
    'demo_report_json': ['seconds', 'finished_utc'],
    'statistics_manifest_json': 'additional table byte hashes omitted; each table itself is checked',
    'pdf': 'all pages rendered on white at 150 dpi; page bounds and every RGB pixel compared',
    'excluded': ['starting inputs from input_manifest.json', 'tmp', 'Reference', 'Models', 'Fitting',
        'Raw_Demo', 'Geometry', 'run_checkpoints', 'Reproduction_Report', '*.provenance.json cache receipts',
        'SVG containers (PDF and numerical drawing data checked)'],
}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


class Fingerprinter:
    def __init__(self, data, code):
        self.data, self.code = Path(data).resolve(), Path(code).resolve()

    def text(self, value):
        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode('utf-8')
        if not isinstance(value, str):
            return value
        for path, replacement in sorted([(self.data, '<DATA>'), (self.code, '<CODE>')], key=lambda x: -len(str(x[0]))):
            canonical = str(path).replace('\\\\?\\', '')
            forms = {canonical, canonical.replace('\\', '/'), '\\\\?\\' + canonical}
            for form in sorted(forms, key=len, reverse=True):
                if re.search(re.escape(form), value, flags=re.IGNORECASE):
                    value = re.sub(re.escape(form), lambda _: replacement, value, flags=re.IGNORECASE)
                    value = value.replace('\\', '/')
        return value

    def clean(self, value):
        if isinstance(value, dict):
            return {str(k): self.clean(v) for k, v in value.items()}
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return self.clean(value.item())
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self.clean(v) for v in value]
        if isinstance(value, (str, bytes, np.bytes_)):
            return self.text(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if np.isnan(value):
                return {'nonfinite': 'nan'}
            if np.isinf(value):
                return {'nonfinite': '+inf' if value > 0 else '-inf'}
            return float(value) if value != 0 else 0.0
        if value is None:
            return None
        raise TypeError(f'Unsupported scientific metadata: {type(value)}')

    def json_hash(self, value):
        return digest_bytes(json.dumps(self.clean(value), sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':'), allow_nan=False).encode('utf-8'))

    def array(self, value):
        value = np.asarray(value)
        if value.dtype.fields or value.dtype.kind in {'O', 'U', 'S'}:
            content = self.json_hash(value.tolist())
        else:
            array = value.astype(value.dtype.newbyteorder('<'), copy=True)
            if array.dtype.kind in {'f', 'c'}:
                array[array == 0] = 0
                array[np.isnan(array)] = np.nan
            content = digest_bytes(array.tobytes(order='C'))
        return {'shape': list(value.shape), 'dtype': str(value.dtype.newbyteorder('=')), 'sha256': content}

    def hdf5(self, path):
        result = {}
        pandas_groups = []
        with h5py.File(path, 'r') as handle:
            handle.visititems(lambda name, obj: pandas_groups.append('/' + name)
                              if isinstance(obj, h5py.Group) and 'pandas_type' in obj.attrs else None)
            def visit(name, obj):
                absolute = '/' + name.strip('/')
                if any(absolute == key or absolute.startswith(key + '/') for key in pandas_groups):
                    return
                attrs = {key: val for key, val in obj.attrs.items()
                         if key not in POLICY['hdf5_provenance_attributes']}
                record = {'attributes': self.json_hash(attrs)}
                if isinstance(obj, h5py.Dataset):
                    if h5py.check_dtype(ref=obj.dtype):
                        raise TypeError(f'Unexpected object reference in generated HDF5 {path}:{name}')
                    record['values'] = self.array(obj[()])
                result[name or '/'] = record
            visit('/', handle)
            handle.visititems(visit)
        if pandas_groups:
            with pd.HDFStore(str(path), mode='r') as store:
                for key in sorted(pandas_groups):
                    frame = store[key]
                    ignored = POLICY['pandas_hdf5_execution_columns'].get(key, [])
                    frame = frame.drop(columns=[c for c in ignored if c in frame])
                    result[key] = {'pandas_table': self.frame(frame), 'index': self.array(frame.index.to_numpy())}
        return {'kind': 'hdf5_logical', 'objects': result}

    def frame(self, frame):
        return {'rows': len(frame), 'columns': list(frame.columns),
                'values': {str(name): self.array(frame[name].to_numpy()) for name in frame.columns}}

    def table(self, path):
        sep = '\t' if path.suffix.lower() == '.tsv' else ','
        # Preserve the exported numeric precision. Empty fields and NaNs retain
        # their parsed table meaning; row/column order remains part of the check.
        frame = pd.read_csv(path, sep=sep, float_precision='round_trip')
        ignored = []
        if path.name.lower().endswith('_summary.csv'):
            ignored = [c for c in ('elapsed_s',) if c in frame]
            frame = frame.drop(columns=ignored)
        return {'kind': 'table', **self.frame(frame), 'ignored_execution_columns': ignored}

    def json(self, path):
        record = json.loads(path.read_text(encoding='utf-8'))
        if path.name.endswith('.overview.json'):
            record = {key: value for key, value in record.items()
                      if key not in POLICY['overview_cache_json']}
        if path.name == 'demo_report.json':
            record = {key: value for key, value in record.items()
                      if key not in POLICY['demo_report_json']}
        if path.name == 'statistics_manifest.json':
            for item in record.get('additional_generated_test_tables', []):
                item.pop('sha256', None)
        return {'kind': 'json', 'sha256': self.json_hash(record)}

    def tiff(self, path):
        from reproduction.file_checks import tiff_layout
        frames = []
        with tifffile.TiffFile(path) as handle:
            shape = tiff_layout(path)
            if len(handle.pages) == shape[0] and len(handle.pages[0].shape) == 2:
                for page in handle.pages:
                    frames.append(self.array(page.asarray()))
            else:
                array = handle.series[0].asarray().reshape(shape)
                frames = [self.array(frame) for frame in array]
        return {'kind': 'decoded_tiff', 'shape': list(shape), 'frames': frames}

    def pdf(self, path):
        pages = []
        with pymupdf.open(str(path)) as document:
            for page in document:
                pixmap = page.get_pixmap(dpi=150, alpha=False)
                pages.append({'bounds_pt': list(page.rect), 'size_px': [pixmap.width, pixmap.height],
                              'rgb_sha256': digest_bytes(pixmap.samples)})
        return {'kind': 'rendered_pdf', 'dpi': 150, 'pages': pages}

    def artifact(self, path):
        suffix = path.suffix.lower()
        if suffix in {'.h5', '.hdf5'}:
            return self.hdf5(path)
        if suffix in {'.csv', '.tsv'}:
            return self.table(path)
        if suffix == '.json':
            return self.json(path)
        if suffix == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                return {'kind': 'numerical_drawing_data',
                        'arrays': {name: self.array(arrays[name]) for name in sorted(arrays.files)}}
        if suffix in {'.tif', '.tiff'}:
            return self.tiff(path)
        if suffix == '.pdf':
            return self.pdf(path)
        if suffix in {'.png', '.jpg', '.jpeg'}:
            from PIL import Image
            with Image.open(path) as image:
                return {'kind': 'decoded_image', 'values': self.array(np.asarray(image.convert('RGBA')))}
        raise ValueError(f'Unsupported output type: {path}')


def starting_inputs(code):
    manifest = code / 'docs/input_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'An explicit starting-input manifest is required: {manifest}')
    return {row['path'].removeprefix('Data/') for row in json.loads(manifest.read_text(encoding='utf-8'))
            if row['path'].startswith('Data/')}


def discover(data, inputs):
    roots = ['figure_data', 'figure_plots', 'Source_Data', 'Statistics', 'intermediate_analysis_data']
    result = []
    for name in roots:
        folder = data / name
        if folder.is_dir():
            result.extend(path for path in folder.rglob('*') if path.is_file())
    for plate in (data / 'exp_raw_data').glob('Plate_*'):
        for name in ('colony_masks', 'colony_masks_stacks', 'clone_masks_stacks', 'sensitive_clean_masks_stacks'):
            folder = plate / name
            if folder.is_dir():
                result.extend(path for path in folder.rglob('*') if path.is_file())
        result.extend(plate.glob('*_summary.csv'))
    supported = {'.h5', '.hdf5', '.csv', '.tsv', '.json', '.npz', '.tif', '.tiff', '.pdf', '.png', '.jpg', '.jpeg'}
    return sorted({p for p in result if p.relative_to(data).as_posix() not in inputs
                   and p.suffix.lower() in supported and 'run_checkpoints' not in p.parts
                   and not p.name.startswith('.') and '.partial' not in p.name
                   and not p.name.endswith('.provenance.json')}, key=lambda p: p.relative_to(data).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--code-root', type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--write-reference', type=Path)
    action.add_argument('--reference', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    data, code = args.data_root.resolve(), args.code_root.resolve()
    fingerprinter = Fingerprinter(data, code)
    inputs = starting_inputs(code)
    paths = discover(data, inputs)
    if not paths:
        raise ValueError('No generated scientific outputs found')
    started = time.monotonic()
    artifacts = {}
    for index, path in enumerate(paths, 1):
        key = path.relative_to(data).as_posix()
        artifacts[key] = fingerprinter.artifact(path)
        if index % 50 == 0 or index == len(paths):
            print(f'Checked {index}/{len(paths)} scientific artifacts: {key}', flush=True)
    result = {'schema_version': 1, 'comparison_policy': POLICY, 'artifacts': artifacts}
    if args.write_reference:
        args.write_reference.parent.mkdir(parents=True, exist_ok=True)
        args.write_reference.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Frozen {len(artifacts)} logical output fingerprints: {args.write_reference}', flush=True)
        return
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if reference['comparison_policy'] != POLICY:
        raise ValueError('Reference and current comparison policies differ')
    expected = reference['artifacts']
    missing, unexpected = sorted(set(expected)-set(artifacts)), sorted(set(artifacts)-set(expected))
    different = [key for key in sorted(set(expected)&set(artifacts)) if expected[key] != artifacts[key]]
    report = {'status': 'passed' if not (missing or unexpected or different) else 'failed',
        'reference': fingerprinter.text(str(args.reference.resolve())), 'seconds': time.monotonic()-started,
        'expected_artifacts': len(expected), 'observed_artifacts': len(artifacts),
        'missing': missing, 'unexpected': unexpected, 'different': different,
        'comparison_policy': POLICY}
    report_path = args.report or data / 'Reproduction_Report/comparison.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'comparison_policy'}, indent=2), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
