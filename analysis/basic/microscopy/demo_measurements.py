"""Noninteractive measurement helpers from the production microscopy pipeline.

The centroid calculation deliberately preserves the original labelled-intensity
center of mass within each object's bounding box, including its rounding and
(y, x) column convention. It is not replaced by a different centroid estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage


def calculate_centers_2d(mask_sus, mask_res, x_scale=1, y_scale=1):
    slices_sus = scipy.ndimage.find_objects(mask_sus)
    centers_sus = []
    for i, si in enumerate(slices_sus):
        if si is not None:
            sr, sc = si
            mask = mask_sus[sr, sc]
            center = scipy.ndimage.center_of_mass(mask)
            center_rounded = list([round((center[0] + sr.start) * x_scale), round((center[1] + sc.start) * y_scale)])
            centers_sus.append(center_rounded)
    slices_res = scipy.ndimage.find_objects(mask_res)
    centers_res = []
    for i, si in enumerate(slices_res):
        if si is not None:
            sr, sc = si
            mask = mask_res[sr, sc]
            center = scipy.ndimage.center_of_mass(mask)
            center_rounded = list([round((center[0] + sr.start) * x_scale), round((center[1] + sc.start) * y_scale)])
            centers_res.append(center_rounded)
    return (pd.DataFrame(centers_sus, columns=['y', 'x']),
            pd.DataFrame(centers_res, columns=['y', 'x']))


def saturation_measurement(image, masks, threshold=12000, max_density=0.00286):
    """Production threshold/count rule, with explicit returned region geometry."""
    outside = (np.asarray(image) > threshold) & ~(masks > 0)
    return int(outside.sum() * max_density), outside
