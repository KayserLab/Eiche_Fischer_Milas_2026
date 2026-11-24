"""Shared mask operations that avoid iterative SciPy hole propagation."""
from __future__ import annotations

import numpy as np
from skimage.measure import label


def fill_holes_2d(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background using the default four-neighbour hole rule.

    Background connected to any image edge stays background. All other
    background components are holes. This matches binary_fill_holes with its
    default structure, without entering the native binary-dilation code that
    crashed during clone tracking on Windows.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    if not mask.size:
        return mask.copy()
    # Integer input selects scikit-image's Cython label implementation; bool
    # input would instead dispatch to SciPy. Connectivity 1 means four neighbours.
    components, count = label(
        np.logical_not(mask).astype(np.uint8),
        background=0, connectivity=1, return_num=True,
    )
    border_labels = np.unique(np.concatenate((
        components[0, :], components[-1, :],
        components[:, 0], components[:, -1],
    )))
    exterior = np.zeros(count + 1, dtype=bool)
    exterior[border_labels] = True
    exterior[0] = False  # Label zero is the original foreground, never a hole.
    return ~exterior[components]
