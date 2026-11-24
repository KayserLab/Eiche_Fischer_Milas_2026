"""Run a scientific stage with execution metadata and plot-data export.

Plot export observes the existing Matplotlib artists after savefig. It does not
change the analysis, rendering settings, scientific tables or plotted arrays.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


def _json_value(value):
    import numpy as np
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k,v in value.items()}
    return str(value)


def export_plot_data(figure, output):
    import numpy as np
    from shared.data_paths import DATA_ROOT
    output = Path(output).resolve()
    try:
        relative = output.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f'Generated figure is outside the declared Data root: {output}') from exc
    folder = DATA_ROOT / 'Source_Data' / 'Rendered' / relative.parent
    folder.mkdir(parents=True, exist_ok=True)
    arrays, axes = {}, []

    def save_array(key, value):
        array = np.ma.asarray(value)
        if array.dtype.hasobject or array.dtype.kind in {'U', 'S'}:
            # Matplotlib's converted numerical positions are exported below;
            # textual category/axis labels are preserved as metadata.
            return {'values': [_json_value(v) for v in array.tolist()]}
        if np.ma.is_masked(array):
            arrays[key + '_mask'] = np.ma.getmaskarray(array)
        arrays[key] = np.ma.getdata(array)
        return {'array': key, 'shape': list(array.shape), 'dtype': str(array.dtype)}

    for axis_index, axis in enumerate(figure.axes):
        record = {'axis': axis_index, 'xlabel': axis.get_xlabel(), 'ylabel': axis.get_ylabel(),
            'title': axis.get_title(), 'xlim': list(axis.get_xlim()), 'ylim': list(axis.get_ylim()),
            'xscale': axis.get_xscale(), 'yscale': axis.get_yscale(),
            'xtick_labels': [text.get_text() for text in axis.get_xticklabels()],
            'ytick_labels': [text.get_text() for text in axis.get_yticklabels()],
            'lines': [], 'collections': [], 'images': [], 'annotations': []}
        for index, line in enumerate(axis.lines):
            prefix = f'axis{axis_index}_line{index}'
            record['lines'].append({'label': line.get_label(),
                'xy': save_array(prefix + '_xy', line.get_xydata()),
                'color': _json_value(line.get_color()), 'linestyle': str(line.get_linestyle()),
                'marker': str(line.get_marker())})
        for index, collection in enumerate(axis.collections):
            prefix = f'axis{axis_index}_collection{index}'
            item = {'label': collection.get_label(), 'type': type(collection).__name__,
                    'offsets': save_array(prefix + '_offsets', collection.get_offsets())}
            if collection.get_array() is not None:
                item['values'] = save_array(prefix + '_values', collection.get_array())
            if callable(getattr(collection, 'get_segments', None)):
                item['segments'] = [save_array(prefix + f'_segment{j}', segment)
                                    for j, segment in enumerate(collection.get_segments())]
            elif callable(getattr(collection, 'get_coordinates', None)):
                item['mesh_coordinates'] = save_array(prefix + '_mesh_coordinates', collection.get_coordinates())
            else:
                paths = collection.get_paths()
                if paths:
                    vertices = [path.vertices for path in paths]
                    item['path_vertices'] = save_array(prefix + '_path_vertices', np.concatenate(vertices))
                    item['path_offsets'] = save_array(prefix + '_path_offsets',
                        np.concatenate(([0], np.cumsum([len(v) for v in vertices]))))
            record['collections'].append(item)
        record['patches'] = []
        for patch in axis.patches:
            if all(callable(getattr(patch, method, None)) for method in ('get_x', 'get_y', 'get_width', 'get_height')):
                record['patches'].append({'type': type(patch).__name__, 'label': patch.get_label(),
                    'rectangle_xywh': [patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()]})
        for index, image in enumerate(axis.images):
            record['images'].append({'values': save_array(f'axis{axis_index}_image{index}', image.get_array()),
                'extent': list(image.get_extent()), 'clim': list(image.get_clim()),
                'origin': image.origin, 'colormap': image.get_cmap().name})
        for text in axis.texts:
            record['annotations'].append({'text': text.get_text(), 'position': list(text.get_position())})
        axes.append(record)
    np.savez_compressed(folder / f'{output.stem}.npz', **arrays)
    record = {'figure': relative.as_posix(), 'stage': os.environ.get('EICHE_FISCHER_STAGE'),
        'source_script': os.environ.get('EICHE_FISCHER_SOURCE_SCRIPT'),
        'note': 'Numerical drawing data; biological identities and test inputs are in figure_data and Statistics/Test_Inputs.',
        'axes': axes}
    (folder / f'{output.stem}.json').write_text(json.dumps(_json_value(record), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def install_plot_capture():
    if os.environ.get('EICHE_FISCHER_CAPTURE_PLOTS') != '1':
        return
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, '_release_capture', False):
        return
    original = Figure.savefig

    def savefig(self, filename, *args, **kwargs):
        if isinstance(filename, (str, os.PathLike)):
            from shared.data_paths import DATA_ROOT
            try:
                Path(filename).resolve().relative_to(DATA_ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f'Generated figure is outside the declared Data root: {filename}') from exc
        result = original(self, filename, *args, **kwargs)
        if isinstance(filename, (str, os.PathLike)):
            export_plot_data(self, filename)
        return result

    savefig._release_capture = True
    Figure.savefig = savefig


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: reproduction/runtime.py STAGE SCRIPT [arguments...]')
    stage, script = sys.argv[1], Path(sys.argv[2]).resolve()
    arguments = sys.argv[3:]
    root = Path(__file__).resolve().parents[1]
    from shared.data_paths import DATA_ROOT
    from reproduction.isolation import description as isolation_description
    report_dir = DATA_ROOT / 'Reproduction_Report/executions'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'{stage}_{os.getpid()}.json'
    os.environ['EICHE_FISCHER_STAGE'] = stage
    os.environ['EICHE_FISCHER_SOURCE_SCRIPT'] = script.relative_to(root).as_posix()
    record = {'stage': stage, 'script': script.relative_to(root).as_posix(), 'arguments': arguments,
        'python': sys.version, 'interpreter': sys.executable, 'platform': sys.platform,
        'code_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'isolation': isolation_description(),
        'packages': {dist.metadata['Name']: dist.version for dist in metadata.distributions()},
        'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    report_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    install_plot_capture()
    started = time.monotonic()
    sys.argv = [str(script), *arguments]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name='__main__')
        record.update(status='complete', exit_code=0)
    except SystemExit as exc:
        record.update(status='complete' if exc.code in (None, 0) else 'failed', exit_code=exc.code)
        raise
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        report_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
