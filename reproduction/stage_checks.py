"""Provenance checks for a generated escape-margin table.

An old CSV alone never counts as complete. Reuse requires identical source HDF5
bytes, calculation code, arguments, and output bytes from a successful run.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(module, args):
    paths = []
    if args.dataset in {'experimental', 'both'}:
        paths.extend(module.discover_prepared_experiments(args.experimental_prepared_root,
            prepared_subdir=str(args.experimental_prepared_subdir), samples=args.samples,
            max_files=int(args.max_files)))
    if args.dataset in {'simulation', 'both'}:
        paths.extend(module.discover_simulations(args.simulation_root,
            samples=args.samples, max_files=int(args.max_files)))
    root = Path(module.REPO_ROOT)
    code = sorted(set([*root.joinpath('analysis/basic/escape').glob('*.py'),
                       root / 'analysis/figure_2/Escape_Margin_Analysis.py',
                       root / 'shared/experiment_filters.py', root / 'shared/data_paths.py']))
    payload = {'arguments': {k: str(v) for k,v in vars(args).items() if k not in {'overwrite', 'output'}},
        'packages': {name: metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'h5py')},
        'inputs': [{'path': str(p), 'sha256': file_hash(p)} for p in paths],
        'code': [{'path': p.relative_to(root).as_posix(), 'sha256': file_hash(p)} for p in code]}
    return payload


def receipt_path(output):
    return Path(output).with_suffix('.provenance.json')


def record_escape_margins(module, args):
    value = {'signature': signature(module, args), 'output_sha256': file_hash(args.output)}
    target = receipt_path(args.output)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)


def check_escape_margin_provenance(parent, *, module, args):
    import pandas as pd
    target = receipt_path(args.output)
    if not target.is_file():
        raise ValueError('No successful input/code/output provenance receipt')
    recorded = json.loads(target.read_text(encoding='utf-8'))
    if recorded['signature'] != signature(module, args):
        raise ValueError('Escape input data, code or calculation arguments changed')
    if recorded['output_sha256'] != file_hash(args.output):
        raise ValueError('Escape-margin output bytes changed')
    table = pd.read_csv(args.output)
    required = {'dataset_type', 'plate', 'sample', 'day', 'signed_escape_margin'}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError('Escape table is empty or missing required columns')
    if table.duplicated(['dataset_type', 'plate', 'sample', 'day']).any():
        raise ValueError('Duplicate replicate/day entries in escape table')
    return True


from pathlib import Path
import argparse
import contextlib
import io
import json
import os
import re
import sys

from reproduction.file_checks import clone_outputs_complete, fresh_outputs, tiff_layout, tiff_matches

SPECIAL_STAGES = {'colony_masks', 'clone_tracking', 'clean_sensitive', 'escape_inputs',
                  'simulation_overview', 'simulation_points', 'escape_margins',
                  'master_curves', 'cell_size_density'}


def _result(outputs, inputs):
    if not outputs:
        raise ValueError('No expected outputs were discovered')
    return {'complete': True, 'reason': f'verified {len(outputs)} output files',
            'artifacts': sorted(set(map(str, outputs))), 'inputs': sorted(set(map(str, inputs)))}


def check_colony_masks(root):
    from analysis.basic.microscopy import Make_susrestotal_Colony_Masks as masks
    inputs, outputs = [], []
    pattern = re.compile(r'.*d(\d+)_.*_MIPs_(.+)_\[(2|3),\s*0\]_masks\.tif$')
    for name in masks.PLATES:
        plate = root / 'exp_raw_data' / name
        days = {}
        for path in (plate / 'czi_NPY').glob('*_masks.tif'):
            match = pattern.match(path.name)
            if match:
                days.setdefault((match[2], match[1].zfill(2)), []).append(path)
        wells = {}
        for (well, day), sources in days.items():
            geometry = tiff_layout(sources[0])[1:]
            density = list((plate / 'density_maps').glob(f'{well}_{day}_*_density.tif'))
            if density:
                preferred = plate / 'density_maps' / f'{well}_{day}_total_density.tif'
                source = preferred if preferred.is_file() else sorted(density)[0]
                geometry = tiff_layout(source)[1:]
                sources = [*sources, source]
            for kind in ('sus', 'res', 'total'):
                output = plate / 'colony_masks' / f'{well}_{day}_{kind}_colony_mask.tif'
                if tiff_layout(output) != (1, *geometry) or not fresh_outputs([output], sources):
                    raise ValueError(f'Incomplete or stale daily mask: {output}')
                outputs.append(output)
                wells.setdefault((well, kind), []).append(output)
            inputs.extend(sources)
        for (well, kind), daily in wells.items():
            shapes = [tiff_layout(path) for path in daily]
            expected = (len(daily), max(s[1] for s in shapes), max(s[2] for s in shapes))
            output = plate / 'colony_masks_stacks' / f'{well}_{kind}_colony_masks.tif'
            if tiff_layout(output) != expected or not fresh_outputs([output], daily):
                raise ValueError(f'Incomplete or stale colony stack: {output}')
            outputs.append(output)
    return _result(outputs, inputs)


def check_clone_tracking(root):
    from analysis.basic.microscopy import single_clone_tracking as tracking
    sources = sorted(path for name in tracking.PLATES for path in
                     (root / 'exp_raw_data' / name / 'colony_masks_stacks').glob('*_res_colony_masks.tif'))
    outputs = []
    for source in sources:
        stack = source.parent.parent / 'clone_masks_stacks' / source.name.replace('_colony_masks', '_clone_masks')
        merge = stack.with_suffix('.csv')
        if not clone_outputs_complete(source, stack, merge):
            raise ValueError(f'Incomplete or stale clone TIFF/CSV pair: {stack}')
        outputs.extend((stack, merge))
    return _result(outputs, sources)


def check_clean_sensitive(root):
    from analysis.basic.escape import Escape_Remove_Sen_Artifacts as cleanup
    clean_root = root / 'intermediate_analysis_data/experimental/basic/escape/sensitive_clean_masks_stacks'
    reps = cleanup.discover_sensitive_replicates(cleanup.PLATES, str(clean_root), skip_existing=False)
    outputs, inputs = [], []
    for rep in reps:
        if not tiff_matches(rep.out_path, rep.clone_mask_path, [rep.h5_path]):
            raise ValueError(f'Incomplete or stale sensitive stack: {rep.out_path}')
        outputs.append(rep.out_path)
        inputs.extend((rep.clone_mask_path, rep.h5_path))
    return _result(outputs, inputs)


def check_escape_inputs(root):
    import pandas as pd
    import tifffile
    from analysis.basic.escape import Prepare_Escape_Analysis_Inputs as preparation
    outputs, inputs = [], []
    for rep in preparation.discover_replicates(preparation.PLATES):
        sources = [rep.raw_h5_path, rep.clone_mask_path, rep.clean_sensitive_path]
        with pd.HDFStore(str(rep.raw_h5_path), 'r') as raw, tifffile.TiffFile(rep.clone_mask_path) as clone, tifffile.TiffFile(rep.clean_sensitive_path) as clean:
            days = preparation.validate_source_alignment(raw, clone, clean)
        preparation.validate_prepared_h5(rep.output_path, days)
        with pd.HDFStore(str(rep.output_path), 'r') as handle:
            metadata = handle['/metadata/run']
            if int(metadata['preparation_version'].iloc[0]) != preparation.PREPARATION_VERSION:
                raise ValueError(f'Outdated escape preparation: {rep.output_path}')
        if not fresh_outputs([rep.output_path], sources):
            raise ValueError(f'Stale escape preparation: {rep.output_path}')
        outputs.append(rep.output_path)
        inputs.extend(sources)
    return _result(outputs, inputs)


def check_simulation_overview(root):
    from analysis.figure_2 import build_simulation_overview as overview
    plans = overview.discover_runs(root / 'sim_raw_data/data-DB', None)
    output = root / 'intermediate_analysis_data/simulation/basic/combined_results.csv'
    overview.validate_summary(output, plans)
    return _result([output], [dump.path for plan in plans for dump in plan.dumps])


def check_simulation_points(root):
    import pandas as pd
    from analysis.basic.density_maps import Prepare_Simulation_Pointclouds as preparation
    arguments = preparation.parse_args()
    plans = preparation.discover_run_plans(arguments)
    outputs, inputs = [], []
    for plan in plans:
        output = preparation.output_path_for_plan(plan, arguments.output_root)
        preparation.validate_run_cache(output, plan, preparation.default_settings())
        outputs.append(output)
        inputs.extend(item.path for item in plan.files)
    manifest = arguments.output_root / preparation.POINTCLOUD_MANIFEST
    table = pd.read_csv(manifest, sep='\t')
    expected = {(plan.plate, plan.well_identifier) for plan in plans}
    observed = set(zip(table.plate, table.well_identifier))
    if expected != observed or len(table) != len(expected) or not table.status.isin(['ok', 'skipped_exists']).all():
        raise ValueError('Simulation manifest does not cover every completed run')
    outputs.extend([manifest, arguments.output_root / preparation.SUMMARY_CSV])
    if not outputs[-1].is_file():
        raise ValueError('Simulation summary is missing')
    return _result(outputs, inputs)


def check_escape_margins(root, module=None, args=None):
    import h5py
    import pandas as pd
    if module is None:
        from analysis.figure_2 import Escape_Margin_Analysis as module
    args = args or module.parse_args()
    check_escape_margin_provenance(root, module=module, args=args)
    expected, inputs = set(), []
    if args.dataset in ('experimental', 'both'):
        paths = module.discover_prepared_experiments(args.experimental_prepared_root,
                    prepared_subdir=args.experimental_prepared_subdir, samples=args.samples,
                    max_files=args.max_files)
        for path in paths:
            with pd.HDFStore(str(path), 'r') as handle:
                for key in handle.keys():
                    match = re.fullmatch(r'/res/d_(\d+)', key)
                    if match:
                        expected.add(('experimental', path.parent.parent.name,
                                      path.name.removesuffix(module.PREPARED_H5_SUFFIX), int(match[1])))
            inputs.append(path)
    if args.dataset in ('simulation', 'both'):
        paths = module.discover_simulations(args.simulation_root, samples=args.samples, max_files=args.max_files)
        for path in paths:
            with h5py.File(path, 'r') as handle:
                for key in set(handle['res']) & set(handle['sus']):
                    expected.add(('simulation', path.parent.parent.name,
                                  path.name.removesuffix(module.SIMULATION_H5_SUFFIX), int(key.removeprefix('d_'))))
            inputs.append(path)
    table = pd.read_csv(args.output)
    required = {'dataset_type', 'plate', 'sample', 'day', 'analysis_version', 'angle_count',
                'signed_escape_margin', 'n_resistant_retained', 'n_sensitive_retained'}
    if required - set(table):
        raise ValueError('Escape-margin columns are missing')
    observed = set(zip(table.dataset_type, table.plate, table['sample'], table.day.astype(int)))
    if observed != expected or len(table) != len(expected) or not expected:
        raise ValueError(f'Escape-margin cohort/day coverage is incomplete: {len(table)}/{len(expected)}')
    if not (table.analysis_version.eq(module.ANALYSIS_VERSION).all()
            and table.angle_count.eq(args.angle_count).all()):
        raise ValueError('Escape-margin analysis settings differ')
    if not fresh_outputs([args.output], inputs):
        raise ValueError('Prepared inputs are newer than the escape-margin table')
    return _result([args.output], inputs)


def check_master_curves(root):
    import h5py
    from analysis.figure_3.HDF5_analysis import compute_master_curves as curves
    from analysis.figure_3.HDF5_analysis.constants import validate_kernel_settings
    experiments = curves.experiment.discover_runs()
    outputs, inputs = [], []
    for path in (curves.experiment.escape_core.DEFAULT_ESCAPE_MARGIN_CSV,
                 curves.simulation.escape_core.DEFAULT_ESCAPE_MARGIN_CSV):
        if Path(path).is_file():
            inputs.append(Path(path))
    for radius in curves.DEFAULT_KERNEL_RADII_UM:
        path = root / f'figure_data/Figure_3/HDF5_analysis/flux_vs_density/kernel_radius_{radius:g}um/selection_diagnostics/master_curve_selection_diagnostics_radius_{radius:g}um.h5'
        with h5py.File(path, 'r') as handle:
            validate_kernel_settings(handle.attrs, radius, source=path)
            for strategy in curves.STRATEGIES:
                runs = experiments.get(strategy, {})
                if not curves._complete_strategy(handle, 'experiment', strategy, sorted(runs)):
                    raise ValueError(f'Incomplete experimental curves: {radius}/{strategy}')
                inputs.extend(runs.values())
                folder = Path(curves.simulation.protocol_outputs_dir(strategy, ''))
                names = [p.name for p in sorted(folder.iterdir()) if p.is_dir()
                         and curves.is_non_artificial_run(p.name)
                         and len(curves.simulation._selected_timesteps(str(p))) >= 2]
                if not curves._complete_strategy(handle, 'simulation', strategy, names):
                    raise ValueError(f'Incomplete simulation curves: {radius}/{strategy}')
                inputs.extend(source for name in names for source in (folder/name).glob('simulation_*.h5'))
        if not fresh_outputs([path], inputs):
            raise ValueError(f'Raw inputs are newer than master curves: {radius}')
        outputs.append(path)
    return _result(outputs, inputs)


def check_cell_size_density(root):
    import h5py
    from analysis.figure_3.HDF5_analysis import compute_cell_size_density as density
    outputs, inputs = [], []
    for radius in density.shared.MASTER_KERNEL_RADII_UM:
        for protocol in density.PROTOCOLS:
            source = density.DEFAULT_INPUT_DIR / f'{protocol}.h5'
            output = density.output_directory(radius) / f'{protocol}.h5'
            with h5py.File(source, 'r') as raw, h5py.File(output, 'r') as prepared:
                density.shared.validate_cell_density_settings(prepared.attrs, radius, source=output)
                if not prepared.attrs.get('complete') or prepared.attrs.get('source_sha256') != density.file_digest(source):
                    raise ValueError(f'Incomplete or outdated cell densities: {output}')
                if set(raw) != set(prepared):
                    raise ValueError(f'Cell-density runs are missing: {output}')
                windows = cells = 0
                for run in raw:
                    if set(raw[run]['cells']) != set(prepared[run]['cells']):
                        raise ValueError(f'Cell-density windows are missing: {output}/{run}')
                    for name, group in raw[run]['cells'].items():
                        target = prepared[run]['cells'][name]
                        n = len(group['cell_type'])
                        for field in ('cell_type', 'delta_d', 'rho_tot', 'rho_sus', 'rho_res', 'kernel_valid'):
                            if target[field].shape != (n,):
                                raise ValueError(f'Incomplete cell observations: {output}/{run}/{name}')
                        windows += 1
                        cells += n
                if prepared.attrs.get('n_windows') != windows or prepared.attrs.get('n_cell_observations') != cells:
                    raise ValueError(f'Wrong cell-density totals: {output}')
            outputs.append(output)
            inputs.append(source)
    return _result(outputs, inputs)


def probe(stage, root):
    return globals()[f'check_{stage}'](Path(root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=sorted(SPECIAL_STAGES))
    parser.add_argument('data_root', type=Path)
    arguments = parser.parse_args()
    os.environ['EICHE_FISCHER_DATA_ROOT'] = str(arguments.data_root)
    sys.argv = [sys.argv[0]]  # Imported script defaults must not see probe arguments.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = probe(arguments.stage, arguments.data_root)
    except Exception as error:
        result = {'complete': False, 'reason': f'{type(error).__name__}: {error}'}
    print(json.dumps(result))
