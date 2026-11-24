"""Portable, non-executable figure-source tables and arrays.

CSV stores numerical tables; NPZ stores numerical arrays (never pickle).
The small JSON schema preserves labels, dtypes and plotting arguments. The
original figure functions still draw every plot; no scientific formula lives here.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
import importlib
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


class FigureSources:
    def __init__(self, folder, *, data_root, code_root, source_roots=(), output_root=None):
        self.folder = Path(folder)
        self.data_root, self.code_root = Path(data_root), Path(code_root)
        self.source_roots = [self.data_root, *map(Path, source_roots)]
        self.output_root = Path(output_root) if output_root is not None else None
        self.serial = 0
        self.array_files = {}

    def _file(self, suffix):
        self.serial += 1
        name = f"v{self.serial:04d}{suffix}"
        self.folder.mkdir(parents=True, exist_ok=True)
        return name

    def _path(self, value):
        original = os.fspath(value)
        text = original.replace('\\', '/')
        for token, roots in (('$DATA/', self.source_roots), ('$CODE/', [self.code_root])):
            for root in roots:
                prefix = str(root).replace('\\', '/').rstrip('/')
                if text.casefold() == prefix.casefold():
                    return token.rstrip('/')
                if text.casefold().startswith(prefix.casefold() + '/'):
                    return token + text[len(prefix) + 1:]
        return original

    def _restore_path(self, text):
        if self.output_root is not None:
            for directory in ('figure_plots', 'figure_data'):
                prefix = '$DATA/' + directory
                if text == prefix or text.startswith(prefix + '/'):
                    suffix = text[len('$DATA/'):].split('/')
                    if '..' in suffix:
                        raise ValueError('Parent traversal in output path')
                    return str(self.output_root.joinpath(*suffix))
        for token, root in (('$DATA', self.data_root), ('$CODE', self.code_root)):
            if text == token:
                return str(root)
            if text.startswith(token + '/'):
                parts = text[len(token) + 1:].split('/')
                if '..' in parts:
                    raise ValueError('Parent traversal in figure-source path')
                return str(root.joinpath(*parts))
        return text

    def encode(self, value):
        import numpy as np
        import pandas as pd
        from cycler import Cycler
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, Enum):
            return {'type': 'enum', 'class': self._class(value), 'name': value.name}
        if isinstance(value, str):
            return self._path(value)
        if isinstance(value, Path):
            return {'type': 'path', 'value': self._path(value)}
        if isinstance(value, np.generic):
            return {'type': 'scalar', 'dtype': str(value.dtype), 'value': self.encode(value.item())}
        if isinstance(value, np.ma.MaskedArray):
            return {'type': 'masked', 'data': self.encode(value.data), 'mask': self.encode(np.ma.getmaskarray(value))}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {'type': 'object_array', 'shape': list(value.shape), 'items': self.encode(value.tolist())}
            contiguous = np.ascontiguousarray(value)
            content = memoryview(contiguous).cast('B') if contiguous.size else b''
            key = (str(value.dtype), value.shape, hashlib.sha256(content).hexdigest())
            name = self.array_files.get(key)
            if name is None:
                name = self._file('.npz')
                np.savez_compressed(self.folder / name, values=value)
                self.array_files[key] = name
            return {'type': 'array', 'file': name, 'dtype': str(value.dtype), 'shape': list(value.shape)}
        if isinstance(value, pd.DataFrame):
            name = self._file('.csv')
            exported = value.copy()
            labels = list(value.columns)
            header = labels if value.columns.is_unique and all(isinstance(c, str) for c in labels) else [f'column_{i}' for i in range(len(labels))]
            exported.columns = header
            schema = []
            for i, col in enumerate(header):
                series = value.iloc[:, i]
                special = str(series.dtype) in ('object', 'string', 'str') or isinstance(series.dtype, (pd.CategoricalDtype, pd.StringDtype))
                if special:
                    exported[col] = [self._path(x) if isinstance(x, (str, Path)) else x for x in series]
                schema.append({'dtype': str(series.dtype), 'header': col,
                    'items': self.encode(list(series)) if special else None,
                    'categories': self.encode(list(series.cat.categories)) if isinstance(series.dtype, pd.CategoricalDtype) else None,
                    'ordered': bool(series.cat.ordered) if isinstance(series.dtype, pd.CategoricalDtype) else False})
            exported.to_csv(self.folder / name, index=False, float_format='%.17g', na_rep='__EF_MISSING__')
            return {'type': 'frame', 'file': name, 'columns': self.encode(labels),
                'column_names': self.encode(list(value.columns.names)), 'multi_columns': isinstance(value.columns, pd.MultiIndex),
                'index': self.encode(list(value.index)), 'index_names': self.encode(list(value.index.names)),
                'multi_index': isinstance(value.index, pd.MultiIndex), 'schema': schema}
        if isinstance(value, pd.Series):
            return {'type': 'series', 'frame': self.encode(value.to_frame(name='value')), 'name': self.encode(value.name)}
        if value is pd.NA:
            return {'type': 'pandas_na'}
        if isinstance(value, pd.Timestamp):
            return {'type': 'timestamp', 'value': value.isoformat()}
        if isinstance(value, Cycler):
            return {'type': 'cycler', 'values': self.encode(value.by_key())}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {'type': 'dataclass', 'class': self._class(value),
                    'fields': self.encode({f.name: getattr(value, f.name) for f in dataclasses.fields(value)})}
        if isinstance(value, SimpleNamespace):
            return {'type': 'namespace', 'fields': self.encode(vars(value))}
        if isinstance(value, dict):
            return {'type': 'dict', 'items': [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return {'type': type(value).__name__, 'items': [self.encode(x) for x in value]}
        raise TypeError(f'Unsupported figure-source value: {type(value)!r}')

    @staticmethod
    def _class(value):
        return type(value).__module__ + ':' + type(value).__qualname__

    @staticmethod
    def _resolve_class(name):
        module, qualified = name.split(':', 1)
        if not (module.startswith(('_reviewer_', 'analysis.', 'figures.', 'shared.', 'PhysiLearning.')) or module == 'matplotlib._enums'):
            raise ValueError(f'Class outside the declared figure code: {name}')
        value = importlib.import_module(module)
        for part in qualified.split('.'):
            if part.startswith('_'):
                raise ValueError(f'Private class member in figure-source schema: {name}')
            value = getattr(value, part)
        return value

    def decode(self, value):
        import numpy as np
        import pandas as pd
        if not isinstance(value, dict):
            return self._restore_path(value) if isinstance(value, str) else value
        kind = value['type']
        if 'file' in value and Path(value['file']).name != value['file']:
            raise ValueError('Figure-source file must be beside its schema')
        if kind == 'path':
            return Path(self._restore_path(value['value']))
        if kind == 'scalar':
            return np.asarray(self.decode(value['value']), dtype=value['dtype'])[()]
        if kind == 'array':
            with np.load(self.folder / value['file'], allow_pickle=False) as archive:
                result = archive['values']
            if list(result.shape) != value['shape'] or str(result.dtype) != value['dtype']:
                raise ValueError(f'Array schema mismatch: {value["file"]}')
            return result
        if kind == 'masked':
            return np.ma.array(self.decode(value['data']), mask=self.decode(value['mask']))
        if kind == 'object_array':
            return np.asarray(self.decode(value['items']), dtype=object).reshape(value['shape'])
        if kind == 'frame':
            raw = pd.read_csv(self.folder / value['file'], dtype=str, keep_default_na=False)
            result = pd.DataFrame(index=range(len(raw)))
            for i, schema in enumerate(value['schema']):
                if schema['items'] is not None:
                    values = self.decode(schema['items'])
                    dtype = schema['dtype']
                    if dtype == 'category' and schema.get('categories') is not None:
                        dtype = pd.CategoricalDtype(self.decode(schema['categories']), ordered=schema.get('ordered', False))
                    result[i] = pd.Series(values, dtype=dtype)
                else:
                    column = raw[schema['header']].replace('__EF_MISSING__', np.nan)
                    if schema['dtype'] in ('bool', 'boolean'):
                        column = column.map({'True': True, 'False': False})
                    result[i] = column.astype(schema['dtype'])
            columns, index = self.decode(value['columns']), self.decode(value['index'])
            result.columns = pd.MultiIndex.from_tuples(columns, names=self.decode(value['column_names'])) if value['multi_columns'] else pd.Index(columns, name=self.decode(value['column_names'])[0])
            result.index = pd.MultiIndex.from_tuples(index, names=self.decode(value['index_names'])) if value['multi_index'] else pd.Index(index, name=self.decode(value['index_names'])[0])
            return result
        if kind == 'series':
            result = self.decode(value['frame']).iloc[:, 0]
            result.name = self.decode(value['name'])
            return result
        if kind == 'pandas_na':
            return pd.NA
        if kind == 'timestamp':
            return pd.Timestamp(value['value'])
        if kind == 'cycler':
            from cycler import cycler
            parts = [cycler(k, v) for k, v in self.decode(value['values']).items()]
            result = parts[0]
            for part in parts[1:]:
                result += part
            return result
        if kind == 'dataclass':
            cls = self._resolve_class(value['class'])
            result = object.__new__(cls)
            for k, v in self.decode(value['fields']).items():
                object.__setattr__(result, k, v)
            return result
        if kind == 'enum':
            return self._resolve_class(value['class'])[value['name']]
        if kind == 'namespace':
            return SimpleNamespace(**self.decode(value['fields']))
        if kind == 'dict':
            return {self.decode(k): self.decode(v) for k, v in value['items']}
        if kind in ('list', 'tuple', 'set', 'frozenset'):
            return {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}[kind](self.decode(x) for x in value['items'])
        raise ValueError(f'Unknown figure-source type: {kind}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def script_sha256(path):
    """Git checkouts may change line endings; all other bytes must match."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
