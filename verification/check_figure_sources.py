"""Check exact float, label, missing-value and style round trips for replot I/O."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    from reproduction.figure_sources import FigureSources
    rng = np.random.default_rng(26)
    floats = np.concatenate([rng.normal(size=4096), [np.nextafter(0., 1.), np.finfo(float).max, -0., np.nan, np.inf, -np.inf]])
    frame = pd.DataFrame({'value': floats, 'label': ['01', None, '__EF_MISSING__', ''] + ['x'] * (len(floats)-4)})
    frame.index = pd.MultiIndex.from_arrays([np.arange(len(frame)) // 2, np.arange(len(frame)) % 2], names=['well', 'time'])
    with tempfile.TemporaryDirectory() as tmp:
        codec = FigureSources(tmp, data_root=Path(tmp)/'Data', code_root=ROOT)
        decoded = codec.decode(codec.encode(frame))
        pd.testing.assert_frame_equal(frame, decoded, check_exact=True)
        finite = np.isfinite(floats)
        np.testing.assert_array_equal(frame.value.to_numpy()[finite].view('uint64'), decoded.value.to_numpy()[finite].view('uint64'))
        nullable = pd.DataFrame({'b': pd.Series([True, pd.NA, False], dtype='boolean'), 'i': pd.Series([1, pd.NA, 3], dtype='Int64')})
        pd.testing.assert_frame_equal(nullable, codec.decode(codec.encode(nullable)), check_exact=True)
        style = codec.decode(codec.encode(dict(mpl.rcParams)))
        validated = mpl.RcParams(style)
        assert dict(validated) == dict(mpl.rcParams)
        array = np.arange(24, dtype=np.int16).reshape(2,3,4)
        np.testing.assert_array_equal(array, codec.decode(codec.encode(array)))
        empty = np.empty((0,3), dtype=float)
        np.testing.assert_array_equal(empty, codec.decode(codec.encode(empty)))
        label = r'Normalized count, $\eta$; $\Delta\rho_{\mathrm{res}}$'
        assert codec.decode(codec.encode(label)) == label
    print('Passed: exact numerical values and float bits, table schema, missing values, arrays and Matplotlib styles.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
