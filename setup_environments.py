"""Create the declared isolated environments and check their installations.

Requires a working Python 3.10+ with venv and pip, plus internet access for
software installation. Scientific inputs and fixed weights are bundled.
Reviewer installation uses exact, resolved requirements; --resolve is a release
maintainer command for preparing those locks and is recorded in the build log.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'environments' / '.local'


def python_in(folder):
    return folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run(command, env, *, capture=False):
    print(' '.join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                           text=True, encoding='utf-8', capture_output=capture)
    if result.returncode:
        if capture:
            print(result.stdout[-4000:], file=sys.stderr)
            print(result.stderr[-8000:], file=sys.stderr)
        result.check_returncode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', nargs='+', choices=('analysis', 'cellpose', 'policy'),
                        default=('analysis', 'policy'), help='Default: analysis policy; add cellpose for the optional CZI demo')
    parser.add_argument('--resolve', action='store_true', help='Maintainer: resolve versioned input requirements into locks')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'environments/manifest.json').read_text(encoding='utf-8'))
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(LOCAL / 'cache/uv'),
               UV_PYTHON_INSTALL_DIR=str(LOCAL / 'python'),
               UV_PYTHON_INSTALL_REGISTRY='0', PYTHONIOENCODING='utf-8',
               PIP_CACHE_DIR=str(LOCAL / 'cache/pip'),
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE=os.devnull,
               UV_NO_CONFIG='1')
    LOCAL.mkdir(parents=True, exist_ok=True)
    bootstrap = LOCAL / 'bootstrap'
    bootstrap_python = python_in(bootstrap)
    if not bootstrap_python.is_file():
        run([sys.executable, '-m', 'venv', bootstrap], env)
    uv_version = manifest['bootstrap_uv_version']
    probe = subprocess.run([str(bootstrap_python), '-c',
        'import importlib.metadata as m; print(m.version("uv"))'], capture_output=True, text=True)
    if probe.returncode or probe.stdout.strip() != uv_version:
        run([bootstrap_python, '-m', 'pip', 'install', f'uv=={uv_version}'], env)
    uv = [bootstrap_python, '-m', 'uv']
    installed_path = LOCAL / 'installed.json'
    installed = json.loads(installed_path.read_text(encoding='utf-8')) if installed_path.exists() else {}
    for role in args.roles or manifest['environments']:
        spec = manifest['environments'][role]
        folder = LOCAL / 'envs' / role
        interpreter = python_in(folder)
        if not interpreter.is_file():
            run([*uv, 'venv', '--python', spec['python'], '--python-preference', 'only-managed', folder], env)
        lock = ROOT / spec['requirements']
        if args.resolve:
            source = lock.with_name('requirements.in')
            run([*uv, 'pip', 'compile', source.relative_to(ROOT), '--python', interpreter,
                 '--generate-hashes', '--no-header', '--output-file', lock.relative_to(ROOT)], env, capture=True)
        if not lock.is_file():
            raise FileNotFoundError(f'Release is incomplete: missing resolved dependency lock {lock}')
        run([*uv, 'pip', 'sync', lock, '--python', interpreter, '--require-hashes'], env)
        run([*uv, 'pip', 'check', '--python', interpreter], env)
        version = run([interpreter, '-c', 'import platform; print(platform.python_version())'], env, capture=True).stdout.strip()
        if version != spec['python']:
            raise RuntimeError(f'Python mismatch for {role}: {version} != {spec["python"]}')
        freeze = run([*uv, 'pip', 'freeze', '--python', interpreter], env, capture=True).stdout
        installed[role] = {'python': str(interpreter.relative_to(ROOT)),
            'python_version': version, 'requirements': spec['requirements'],
            'installed_utc': datetime.now(timezone.utc).isoformat(), 'packages': freeze.splitlines()}
    # Publish interpreter paths only after every requested installation passes.
    temporary = installed_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(installed, indent=2) + '\n', encoding='utf-8')
    temporary.replace(installed_path)
    print('Environment installation and dependency checks passed.', flush=True)


if __name__ == '__main__':
    main()
