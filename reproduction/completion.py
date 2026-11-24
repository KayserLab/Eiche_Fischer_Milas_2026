"""Validated stage reuse; an exit code or filename alone is never a checkpoint."""
from pathlib import Path
import ast
import csv
from datetime import datetime
import hashlib
import json
import os
import re
import subprocess
import threading

from reproduction.stage_checks import SPECIAL_STAGES

ARTIFACT_SUFFIXES = {'.csv', '.tsv', '.h5', '.hdf5', '.tif', '.tiff', '.pdf',
                     '.png', '.jpg', '.jpeg', '.svg', '.json', '.xlsx'}


def file_state(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def validate_artifact(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or empty output: {path}')
    suffix = path.suffix.lower()
    if suffix in ('.csv', '.tsv'):
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = csv.reader(handle, delimiter='\t' if suffix == '.tsv' else ',')
            header = next(rows, None)
            if not header or any(len(row) != len(header) for row in rows if row):
                raise ValueError(f'Truncated or malformed table: {path}')
    elif suffix in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as handle:
            if not len(handle) or ('complete' in handle.attrs and not handle.attrs['complete']):
                raise ValueError(f'Incomplete HDF5: {path}')
            # Opening every dataset's metadata also detects missing/damaged links.
            def inspect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    _ = obj.shape, obj.dtype
            handle.visititems(inspect)
    elif suffix == '.pdf':
        with path.open('rb') as handle:
            if not handle.read(8).startswith(b'%PDF-'):
                raise ValueError(f'Invalid PDF: {path}')
            handle.seek(max(0, path.stat().st_size - 2048))
            if b'%%EOF' not in handle.read():
                raise ValueError(f'Unfinished PDF: {path}')
    elif suffix in ('.png', '.jpg', '.jpeg'):
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    elif suffix in ('.tif', '.tiff'):
        import tifffile
        size = path.stat().st_size
        with tifffile.TiffFile(path) as handle:
            if not len(handle.pages):
                raise ValueError(f'Empty TIFF: {path}')
            for page in handle.pages:
                if any(offset <= 0 or count <= 0 or offset + count > size
                       for offset, count in zip(page.dataoffsets, page.databytecounts)):
                    raise ValueError(f'Truncated TIFF: {path}')
    elif suffix == '.json':
        json.loads(path.read_text(encoding='utf-8-sig'))
    elif suffix == '.svg':
        import xml.etree.ElementTree as ET
        ET.parse(path)
    elif suffix == '.xlsx':
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None or 'xl/workbook.xml' not in archive.namelist():
                raise ValueError(f'Invalid workbook: {path}')


class CompletionTracker:
    def __init__(self, data_root, repo_root, interpreter, env, *, state_root=None, history_root=None):
        self.data_root, self.repo_root = Path(data_root), Path(repo_root)
        self.interpreter, self.env = str(interpreter), dict(env)
        self.state_root = Path(state_root or self.data_root / 'tmp/pipeline_completion')
        self.history_root = Path(history_root or self.data_root / 'tmp/pipeline_runs')
        self.before = {}
        self.lock = threading.RLock()
        self.raw_cache = {}
        self.refresh = set()

    def _receipt_path(self, name):
        return self.state_root / f'{name}.json'

    def _scopes(self, step):
        figure = {'fig1': 'Figure_1', 'fig2': 'Figure_2', 'fig3': 'Figure_3',
                  'fig4': 'Figure_4', 'suppl': 'Suppl_Figures'}.get(step.group)
        paths = []
        if figure:
            paths += [self.data_root / folder / figure for folder in ('figure_data', 'figure_plots')]
        else:
            paths += [self.data_root / 'intermediate_analysis_data']
        if step.name in {'policy_behavior', 'dpd_decisions', 'experimental_decisions'}:
            paths.append(self.data_root / 'intermediate_analysis_data/policy_decisions')
        return paths

    def _inventory(self, step):
        result = {}
        for folder in self._scopes(step):
            if not folder.is_dir():
                continue
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES and not any(
                    marker in path.name.lower() for marker in ('.tmp', '.partial', '.building')):
                    result[str(path)] = file_state(path)
        return result

    def _code_files(self, entry):
        visited, pending = set(), [self.repo_root / entry]
        while pending:
            path = pending.pop()
            if path in visited or not path.is_file():
                continue
            visited.add(path)
            try:
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            except (SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = path.parent
                        for _ in range(node.level - 1):
                            base = base.parent
                        prefix = '.'.join(base.relative_to(self.repo_root).parts)
                        module = '.'.join(filter(None, (prefix, node.module)))
                    else:
                        module = node.module or ''
                    names = [module, *(f'{module}.{item.name}' for item in node.names)]
                for name in names:
                    candidate = self.repo_root.joinpath(*name.split('.'))
                    for source in (candidate.with_suffix('.py'), candidate / '__init__.py'):
                        if source.is_file() and source not in visited:
                            pending.append(source)
        return sorted(visited)

    def _raw_state(self, step):
        roots = ['exp_raw_data'] if step.group == 'fig1' else ['exp_raw_data', 'sim_raw_data']
        key = tuple(roots)
        if key not in self.raw_cache:
            records = []
            excluded = {'colony_masks', 'colony_masks_stacks', 'clone_masks_stacks',
                        'density_maps', 'density_changes', 'sensitive_clean_masks_stacks', '__pycache__'}
            for root in roots:
                for folder, dirs, files in os.walk(self.data_root / root):
                    dirs[:] = [name for name in dirs if name not in excluded]
                    for name in files:
                        path = Path(folder) / name
                        if path.suffix.lower() in ARTIFACT_SUFFIXES | {'.txt', '.yaml', '.npy', '.pth'}:
                            records.append((str(path), file_state(path)))
            self.raw_cache[key] = sorted(records)
        return self.raw_cache[key]

    def _signature(self, step, command):
        code = [(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in self._code_files(step.script)]
        # Numerical worker counts do not change the scientific output.
        args = list(command)
        for option in ('--workers', '--plate-workers'):
            if option in args:
                index = args.index(option)
                args[index + 1] = '<workers>'
        dependencies = []
        for name in step.requires:
            path = self._receipt_path(name)
            try:
                receipt = json.loads(path.read_text(encoding='utf-8'))
                dependencies.append([name, receipt.get('signature'),
                                     [(artifact, file_state(artifact)) for artifact in sorted(receipt['artifacts'])]])
            except (OSError, ValueError, KeyError):
                dependencies.append([name, 'unverified'])
        settings = {key: value for key, value in self.env.items()
                    if key.startswith(('HDF5_', 'MAJA_')) and not key.endswith('_N_JOBS')}
        state = [args, code, self._raw_state(step), dependencies, settings]
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def _probe(self, step):
        result = subprocess.run([self.interpreter, '-m', 'reproduction.stage_checks',
                                 step.name, str(self.data_root)], cwd=self.repo_root, env=self.env,
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {'complete': False, 'reason': (result.stderr or result.stdout)[-800:] or 'Completion probe failed'}

    def _output_paths(self, log):
        paths = set()
        if not Path(log).is_file():
            return paths
        expression = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\r\n'\"]+?\.(?:hdf5|h5|csv|tsv|tiff|tif|pdf|png|jpe?g|svg|xlsx|json)(?=$|[\s,;:)'\"])", re.I)
        for line in Path(log).read_text(encoding='utf-8', errors='replace').splitlines()[1:]:
            if not re.search(r'sav(?:e|ed|ing)|wrote|written|export|output|\[cache\]', line, re.I):
                continue
            for match in expression.finditer(line):
                path = Path(match.group().strip())
                # Missing declared products must remain in the manifest: dropping
                # them here would incorrectly accept the remaining output files.
                if self.data_root in path.parents or self.repo_root in path.parents:
                    paths.add(str(path))
        return paths

    def _history(self, step):
        # Never adopt outputs from an attempt that subsequently failed.
        for report_path in sorted(self.history_root.glob('*/run.json'), reverse=True):
            try:
                report = json.loads(report_path.read_text(encoding='utf-8'))
                if Path(report.get('data_root', '')) != self.data_root:
                    continue
                attempt = next((item for item in report['steps'] if item['step'] == step.name), None)
                if attempt is None or attempt.get('status') in ('skipped_complete',):
                    continue
                if attempt.get('status') != 'complete' or attempt.get('exit_code') != 0:
                    return None
                finished = datetime.fromisoformat(attempt['started_at']).timestamp() + attempt['seconds']
                if any(path.stat().st_mtime > finished + 2 for path in self._code_files(step.script)):
                    return None
                if any(state[1] / 1e9 > finished + 2 for _, state in self._raw_state(step)):
                    return None
                for name in step.requires:
                    receipt_path = self._receipt_path(name)
                    if receipt_path.is_file():
                        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                        if any(Path(path).stat().st_mtime > finished + 2 for path in receipt['artifacts']):
                            return None
                logs = list(report_path.parent.glob(f'*_{step.name}.log'))
                paths = self._output_paths(logs[0]) if len(logs) == 1 else set()
                began = datetime.fromisoformat(attempt['started_at']).timestamp()
                paths.update(path for path, state in self._inventory(step).items()
                             if began - 2 <= state[1] / 1e9 <= finished + 2)
                if not paths or any(Path(path).stat().st_mtime > finished + 2 for path in paths):
                    return None
                return paths
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def _write_receipt(self, step, command, paths, inputs=()):
        states = {}
        for path in sorted(paths):
            validate_artifact(path)
            states[str(path)] = file_state(path)
        if not states:
            return
        receipt = {'version': 1, 'step': step.name, 'signature': self._signature(step, command),
                   'artifacts': states, 'input_state': {str(path): file_state(path) for path in inputs}}
        self.state_root.mkdir(parents=True, exist_ok=True)
        path = self._receipt_path(step.name)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        os.replace(temporary, path)

    def check(self, step, command):
        with self.lock:
            if '--overwrite' in command or '--force' in command:
                return False, 'explicit rebuild requested'
            if step.name in SPECIAL_STAGES:
                result = self._probe(step)
                if step.name == 'master_curves':
                    # Aggregated curve shapes do not reveal a newly added day
                    # within an existing run. Compare the source inventory too.
                    previous = self._receipt_path(step.name)
                    if result['complete'] and previous.is_file():
                        try:
                            old = json.loads(previous.read_text(encoding='utf-8'))
                            current = {path: file_state(path) for path in result.get('inputs', [])}
                            if old.get('input_state') and old['input_state'] != current:
                                self.refresh.add(step.name)
                                return False, 'master-curve input inventory changed'
                        except (OSError, ValueError, KeyError):
                            pass
                    elif not result['complete'] and all((self.data_root / (
                        f'figure_data/Figure_3/HDF5_analysis/flux_vs_density/kernel_radius_{radius}um/'
                        f'selection_diagnostics/master_curve_selection_diagnostics_radius_{radius}um.h5'
                    )).is_file() for radius in (120, 160)):
                        self.refresh.add(step.name)
                if result['complete']:
                    self._write_receipt(step, command, result['artifacts'], result.get('inputs', []))
                    return True, result['reason']
                return False, result['reason']
            receipt_path = self._receipt_path(step.name)
            if receipt_path.is_file():
                try:
                    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                    if receipt.get('version') != 1 or receipt.get('step') != step.name:
                        return False, 'incompatible completion record'
                    if receipt.get('signature') != self._signature(step, command):
                        return False, 'inputs or analysis code changed'
                    for path, state in receipt['artifacts'].items():
                        if file_state(path) != state:
                            return False, 'an output changed'
                        validate_artifact(path)
                    if receipt['artifacts']:
                        return True, f"verified {len(receipt['artifacts'])} saved outputs"
                except (OSError, ValueError, KeyError, TypeError):
                    return False, 'saved outputs are missing or invalid'
            paths = self._history(step)
            if paths:
                try:
                    self._write_receipt(step, command, paths)
                    return True, f'verified {len(paths)} outputs from a successful run'
                except (OSError, ValueError, KeyError):
                    pass
            return False, 'no verified completion record'

    def begin(self, step, command):
        with self.lock:
            self.before[step.name] = {} if step.name in SPECIAL_STAGES else self._inventory(step)
            # A failed retry must not leave an earlier completion receipt usable.
            self._receipt_path(step.name).unlink(missing_ok=True)

    def environment(self, step):
        return {'PIPELINE_REFRESH_MASTER_CURVES': '1'} if step.name in self.refresh else {}

    def finish(self, step, command, log):
        with self.lock:
            if step.name in SPECIAL_STAGES:
                result = self._probe(step)
                if not result['complete']:
                    raise ValueError(f"Completion verification failed: {result['reason']}")
                paths = set(result['artifacts'])
            else:
                before = self.before.pop(step.name, {})
                paths = self._output_paths(log)
                paths.update(path for path, state in self._inventory(step).items() if before.get(path) != state)
            self._write_receipt(step, command, paths,
                                result.get('inputs', []) if step.name in SPECIAL_STAGES else [])
            return bool(paths)
