"""Regenerate figures with their original renderers and frozen source values."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_figure_module(stage, script, arguments):
    name = '_reviewer_' + stage
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.argv = [str(script), *arguments]
    sys.path.insert(0, str(script.parent))
    spec.loader.exec_module(module)
    return module


class SelectedTiffPages:
    """Read exactly the original indexed movie pages from a lossless NPZ subset.

    Unselected pages are absent and raise an error if requested. No artificial
    or empty frames are supplied. Source indices and TCYX shape are recorded.
    """
    def __init__(self, path, *, root):
        import numpy as np
        self.archive = np.load(Path(root) / (Path(path).stem + '.npz'), allow_pickle=False)
        self.axes = str(self.archive['axes'].item())
        self.shape = tuple(map(int, self.archive['source_shape']))
        self.series, self.pages = [self], self

    def __getitem__(self, page):
        values = self.archive[f'page_{int(page)}']
        from types import SimpleNamespace
        return SimpleNamespace(asarray=lambda: values)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()


def run(stage):
    import matplotlib.pyplot as plt
    import numpy as np
    from reproduction.figure_sources import FigureSources, read_json, script_sha256
    from shared import data_paths
    DATA_ROOT = data_paths.DATA_ROOT
    output_root = DATA_ROOT / 'Results'
    data_paths.FIGURE_PLOTS = output_root / 'figure_plots'
    data_paths.FIGURE_DATA = output_root / 'figure_data'
    from reproduction import isolation
    if isolation._state is not None:
        isolation._state['write_roots'] = [isolation.normalized(output_root), isolation.normalized(DATA_ROOT / 'tmp')]
    folder = DATA_ROOT / 'Plot_Source' / stage
    record = read_json(folder / 'source.json')
    for relative in record['outputs']:
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Invalid figure output path: {relative}')
        (output_root / path).parent.mkdir(parents=True, exist_ok=True)
    script = ROOT / record['script']
    actual = script_sha256(script)
    if record['script_sha256'] != actual:
        raise ValueError(f'Figure source/code mismatch for {stage}; use the matching release.')
    source = FigureSources(folder, data_root=DATA_ROOT, code_root=ROOT, output_root=output_root)
    module = load_figure_module(stage, script, [source.decode(a) for a in record['arguments']])
    positions = {}
    for name, values in record.get('loaders', {}).items():
        if not hasattr(module, name):
            raise ValueError(f'Missing original figure input function: {stage}/{name}')
        positions[name] = 0
        def frozen(*args, _name=name, _values=values, **kwargs):
            index = positions[_name]
            if index >= len(_values):
                raise ValueError(f'Unrecorded figure input requested: {stage}/{_name}')
            positions[_name] += 1
            return source.decode(_values[index])
        setattr(module, name, frozen)
    if record['entry'] == 'movie':
        module.tifffile.TiffFile = lambda path: SelectedTiffPages(path, root=folder)
        module.OUTPUT_DIRECTORY = output_root / 'figure_plots/Figure_1/Movies/fluorescence_composites_days_01_10_19_33'
        module.main()
    elif record['entry'] == 'main':
        plt.rcParams.update(source.decode(record['rcparams']))
        np.random.set_state(source.decode(record['numpy_random']))
        random.setstate(source.decode(record['python_random']))
        module.main()
    elif record['entry'] == 'calls':
        for call in record['calls']:
            plt.rcParams.update(source.decode(call['rcparams']))
            np.random.set_state(source.decode(call['numpy_random']))
            random.setstate(source.decode(call['python_random']))
            function = getattr(module, call['function'])
            function(*source.decode(call['args']), **source.decode(call['kwargs']))
            # Matplotlib figures contain cycles; release closed image panels
            # before decoding the next large original drawing call.
            gc.collect()
    else:
        raise ValueError(f'Unknown frozen-figure entry: {record["entry"]}')
    for name, values in record.get('loaders', {}).items():
        if positions[name] != len(values):
            raise ValueError(f'Unused figure-source values: {stage}/{name}')
    for relative in record['outputs']:
        path = output_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Expected figure was not regenerated: {path}')
    print(f'[replot complete] {stage}: {len(record["outputs"])} original figure outputs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage')
    args = parser.parse_args()
    run(args.stage)


if __name__ == '__main__':
    main()
