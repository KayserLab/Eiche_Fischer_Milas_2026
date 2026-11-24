"""Small, read-only checks shared by completion probes and microscopy writers."""
from pathlib import Path
import csv


def tiff_layout(path):
    """Check every frame's geometry and encoded byte ranges without loading pixels."""
    import tifffile
    path = Path(path)
    size = path.stat().st_size
    with tifffile.TiffFile(path) as handle:
        if not len(handle.pages):
            raise ValueError(f'Empty TIFF: {path}')
        shape = tuple(handle.pages[0].shape)
        if len(shape) not in (2, 3):
            raise ValueError(f'Expected a 2D mask or 3D mask stack: {path}')
        for page in handle.pages:
            if tuple(page.shape) != shape:
                raise ValueError(f'Inconsistent TIFF frames: {path}')
            if not page.dataoffsets or len(page.dataoffsets) != len(page.databytecounts):
                raise ValueError(f'Missing TIFF image data: {path}')
            if any(offset <= 0 or count <= 0 or offset + count > size
                   for offset, count in zip(page.dataoffsets, page.databytecounts)):
                raise ValueError(f'Truncated TIFF: {path}')
        # Historical 3/4-frame stacks were written as one planar TIFF page
        # (SYX), while newer stacks use one page per frame (QYX).
        return ((len(handle.pages), *shape) if len(shape) == 2
                else (len(handle.pages) * shape[0], *shape[1:]))


def fresh_outputs(outputs, inputs):
    return min(Path(path).stat().st_mtime_ns for path in outputs) >= max(
        Path(path).stat().st_mtime_ns for path in inputs)


def tiff_matches(output, source, extra_inputs=()):
    try:
        return (tiff_layout(output) == tiff_layout(source)
                and fresh_outputs([output], [source, *extra_inputs]))
    except Exception:
        return False


def clone_outputs_complete(source, stack, merge_csv):
    try:
        shape = tiff_layout(source)
        if tiff_layout(stack) != shape or not fresh_outputs([stack, merge_csv], [source]):
            return False
        with Path(merge_csv).open(newline='', encoding='utf-8-sig') as handle:
            rows = csv.reader(handle)
            if next(rows, None) != ['clone_id', 'merged_into', 'merge_frame']:
                return False
            for row in rows:
                if len(row) != 3:
                    return False
                clone, target, frame = map(int, row)
                if clone < 0 or target < 0 or not 0 <= frame < shape[0]:
                    return False
        return True
    except Exception:
        return False
