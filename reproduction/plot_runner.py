"""Public reviewer entry point: original renderers, frozen figure inputs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time


def main(argv=None):
    import reproduce_figures as runner
    from reproduction.plot_profiles import plotting_steps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=False, type=Path, default=runner.DATA_DIRECTORY)
    parser.add_argument('--only', nargs='+', choices=('fig1', 'fig2', 'fig3', 'fig4', 'suppl'))
    parser.add_argument('--demo', choices=('czi', 'simulation'), help='Run one optional example instead of the plots')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true', help='Verify matching code and every frozen input hash')
    args = parser.parse_args(argv)
    if args.only and args.demo:
        parser.error('--only and --demo are separate routes')
    data = args.data_root.expanduser().absolute()
    root = runner.ROOT
    installed_file = root / 'environments/.local/installed.json'
    installed = json.loads(installed_file.read_text(encoding='utf-8')) if installed_file.exists() else {}
    interpreters = {}
    for role, label in (('analysis', 'root'), ('policy', 'policy'), ('cellpose', 'cellpose')):
        default = root / 'environments/.local/envs' / role / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        interpreters[label] = str(root / installed[role]['python'] if role in installed else default)
    if args.demo == 'czi':
        steps = [s for s in runner.build_steps() if s.name in ('raw_image_demo', 'raw_demo_report')]
        commands = [(s.name, s.environment, runner.command_for(s, interpreters, data, 1, 1)) for s in steps]
    elif args.demo == 'simulation':
        commands = [('simulation_demo', 'root', [interpreters['root'], '-u', '-B',
            str(root / 'verification/simulation_snapshot_demo.py'), '--data-root', str(data),
            '--output-dir', str(data / 'Results/demos/simulation')])]
    else:
        steps = [s for s in plotting_steps(runner.build_steps()) if not args.only or s.group in args.only]
        commands = [(s.name, s.environment, [interpreters[s.environment], '-u', '-B',
            str(root / 'reproduction/plot_runtime.py'), s.name]) for s in steps]
    print(f'Data: {data}')
    print('Route: ' + (f'optional {args.demo} example' if args.demo else 'plots from frozen figure inputs'))
    for i, (name, role, command) in enumerate(commands, 1):
        print(f'{i:02d}. {name} [{role}]')
        if args.dry_run:
            print('    ' + shlex.join(command))
    if args.list or args.dry_run:
        return 0
    try:
        for role in {role for _, role, _ in commands}:
            if not Path(interpreters[role]).is_file():
                raise ValueError(f'Missing {role} environment. Run setup_environments.py first.')
        if not args.demo:
            verify_inputs(data, root, [name for name, _, _ in commands])
        elif args.demo == 'czi':
            runner.preflight(steps, interpreters, data, None)
        else:
            if not (data / 'Raw_Demo/Simulation_Full_State/data-DB_Dens/run_25/metadata.json').is_file():
                raise ValueError('Missing full simulation snapshot demo')
        if args.check:
            print('Matching code and required frozen inputs verified.' if not args.demo else 'Demo input paths verified.')
            return 0
        if not args.demo:
            # Inline original renderers may export tables. Give them independent
            # working copies; the deposited input tables remain immutable.
            for src in (data / 'figure_data').rglob('*'):
                if src.is_file() and src.suffix.lower() in ('.csv', '.tsv', '.json'):
                    dst = data / 'Results/figure_data' / src.relative_to(data / 'figure_data')
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
        log_dir = data / 'Results/logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log_dir.mkdir(parents=True, exist_ok=False)
        report = {'route': args.demo or 'plots', 'status': 'running', 'data_root': str(data), 'stages': []}
        def save():
            (log_dir / 'run.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        save()
        env = runner.child_environment(data, workers=1)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        print(f'Logs: {log_dir}', flush=True)
        for name, _, command in commands:
            print(f'Running {name}', flush=True)
            start = time.monotonic()
            code = runner.run_step(command, env, log_dir / (name + '.log'))
            report['stages'].append({'name': name, 'exit_code': code, 'seconds': time.monotonic() - start})
            report['status'] = 'failed' if code else 'running'
            save()
            if code:
                print(f'Stopped at {name}. See {log_dir / (name + ".log")}', file=sys.stderr)
                return code
        report['status'] = 'complete'
        save()
        print(f'Complete. Outputs: {data / "Results"}')
        return 0
    except ValueError as exc:
        parser.error(str(exc))


def verify_inputs(data, root, stages):
    """Fail before plotting; never fall back to costly recomputation."""
    from reproduction.figure_sources import script_sha256
    manifest = data / 'Plot_Source/manifest.json'
    if not manifest.is_file():
        raise ValueError('Missing Plot_Source/manifest.json. Use the compact plot-source data release.')
    record = json.loads(manifest.read_text(encoding='utf-8'))
    files = record['files']
    for stage in stages:
        source_path = data / 'Plot_Source' / stage / 'source.json'
        if not source_path.is_file():
            raise ValueError(f'Missing frozen figure source: {stage}')
        source = json.loads(source_path.read_text(encoding='utf-8'))
        script = root / source['script']
        if script_sha256(script) != source['script_sha256']:
            raise ValueError(f'Code/data release mismatch: {stage}')
    for relative, expected in files.items():
        parts = Path(relative).parts
        if Path(relative).is_absolute() or '..' in parts:
            raise ValueError(f'Invalid frozen input path: {relative}')
        path = data / 'Plot_Source' / relative
        if not path.is_file():
            raise ValueError(f'Missing frozen input: {relative}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen input hash mismatch: {relative}')
