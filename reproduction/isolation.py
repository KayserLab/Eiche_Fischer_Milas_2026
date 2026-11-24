"""Audit scientific-stage file access against declared bundle/runtime roots.

This is a reproducibility check, not an operating-system security sandbox.
Python file access and the native HDF5 entry points used by this workflow are
checked; software installation is a separate, network-enabled setup operation.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

_state = None


def normalized(path):
    text = os.fsdecode(os.fspath(path))
    if text.startswith('\\\\?\\UNC\\'):
        text = '\\\\' + text[8:]
    elif text.startswith('\\\\?\\'):
        text = text[4:]
    return os.path.normcase(os.path.abspath(text))


def inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def check_path(path, *, write=False):
    if _state is None or not isinstance(path, (str, bytes, os.PathLike)):
        return
    text = os.fsdecode(os.fspath(path))
    if text.lower() in {'nul', 'nul:', os.devnull.lower()} or text.startswith('\\\\.\\pipe\\'):
        return
    canonical = normalized(path)
    roots = _state['write_roots'] if write else _state['read_roots']
    if not any(inside(canonical, root) for root in roots):
        raise PermissionError(f'Reproduction isolation rejected {"write" if write else "read"} outside declared roots: {text}')
    if write and (canonical in _state['immutable_files'] or
                  any(inside(canonical, root) for root in _state['immutable_roots'])):
        raise PermissionError(f'Reproduction isolation rejected a write to a starting input/reference: {text}')


def writing(mode):
    return isinstance(mode, str) and any(flag in mode for flag in ('w', 'a', 'x', '+'))


def patch_hdf5(module):
    if module.__name__ == 'h5py':
        original = module.File.__init__
        if getattr(original, '_reproduction_guard', False):
            return
        def initialize(self, name, mode='r', *args, **kwargs):
            check_path(name, write=writing(mode))
            return original(self, name, mode, *args, **kwargs)
        initialize._reproduction_guard = True
        module.File.__init__ = initialize
    elif module.__name__ == 'tables':
        original = module.open_file
        if getattr(original, '_reproduction_guard', False):
            return
        def open_file(filename, mode='r', *args, **kwargs):
            check_path(filename, write=writing(mode))
            return original(filename, mode, *args, **kwargs)
        open_file._reproduction_guard = True
        module.open_file = open_file


class GuardedLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        method = getattr(self.loader, 'create_module', None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        patch_hdf5(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class GuardedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in {'h5py', 'tables'}:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = GuardedLoader(spec.loader)
        return spec


def install(code_root, data_root):
    global _state
    if _state is not None or os.environ.get('EICHE_FISCHER_STRICT_IO') != '1':
        return
    code_root, data_root = Path(code_root).absolute(), Path(data_root).absolute()
    read_roots = [normalized(root) for root in {code_root, data_root, Path(sys.prefix), Path(sys.base_prefix)}]
    # Standard operating-system resources are declared separately from study
    # observations, model caches and developer/user-site packages.
    if os.name == 'nt':
        system = Path(os.environ.get('SystemRoot', r'C:\Windows'))
        read_roots.extend(normalized(system / name) for name in ('Fonts', 'System32', 'SysWOW64', 'WinSxS'))
    else:
        read_roots.extend(normalized(root) for root in ('/usr/share/fonts', '/usr/local/share/fonts',
            '/usr/lib', '/lib', '/etc/fonts', '/etc/ssl', '/proc', '/sys', '/dev'))
    immutable_files = set()
    manifest = code_root / 'docs/input_manifest.json'
    if manifest.is_file():
        for row in json.loads(manifest.read_text(encoding='utf-8')):
            immutable_files.add(normalized(code_root / row['path']))
    _state = {'read_roots': read_roots, 'write_roots': [normalized(code_root), normalized(data_root)],
        'immutable_files': immutable_files, 'immutable_roots': [normalized(data_root / root) for root in
            ('Models', 'Fitting', 'Reference', 'Raw_Demo')]}

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            write = writing(mode) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            check_path(path, write=write)
        elif event in {'os.listdir', 'os.scandir'}:
            check_path(args[0] if args else os.curdir)
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir'}:
            check_path(args[0], write=True)
        elif event in {'os.rename', 'os.replace'}:
            check_path(args[0], write=True)
            check_path(args[1], write=True)
        elif event in {'socket.connect', 'socket.getaddrinfo'}:
            raise PermissionError(f'Reproduction isolation rejected network access during a scientific stage: {event}')
    sys.addaudithook(audit)
    sys.meta_path.insert(0, GuardedFinder())
    for name in ('h5py', 'tables'):
        if name in sys.modules:
            patch_hdf5(sys.modules[name])


def description():
    if _state is None:
        return {'enabled': False}
    return {'enabled': True, 'scope': 'Python file access and workflow native HDF5 readers; network calls rejected',
            'read_roots': _state['read_roots'], 'write_roots': _state['write_roots'],
            'immutable_files': len(_state['immutable_files']), 'immutable_roots': _state['immutable_roots']}
