# Frozen Cellpose segmentation models

This directory contains the two frozen custom Cellpose weight files used by the
production microscopy-segmentation pipeline in
`analysis/basic/microscopy/analysis_pipeline_2D_corr.py`. The files are model
weights, not the Cellpose package. Their byte identities are recorded in
`manifest.csv`.

## Software and model mapping

Production segmentation used Cellpose 3.0.11. Install the pinned version from
the repository root with:

```console
pip install -r requirements.txt
```

or install Cellpose alone with `pip install cellpose==3.0.11`.

| Fluorescence population/channel | Frozen model file | Cellpose `channels` argument |
| --- | --- | --- |
| Resistant | `a4_cyto2_PhD_1000_new` | `[3, 0]` |
| Sensitive | `a2_cyto3_Epi_1000_PhD` | `[2, 0]` |

The production call used `models.CellposeModel` with `min_size=250`,
`flow_threshold=0`, `cellprob_threshold=-1`, `stitch_threshold=0`, and 2D
segmentation. `diameter=None` was passed. In Cellpose 3.0.11,
`CellposeModel.eval` does **not** run the per-image size model in this case; it
uses the mean training-mask diameter stored in the custom weights. Those stored
values are 31.244544982910156 pixels for `a4_cyto2_PhD_1000_new` and
23.872602462768555 pixels for `a2_cyto3_Epi_1000_PhD`. Thus, describing this
production call as "automatic diameter estimation" would be inaccurate.

## Using the files

Load a frozen file directly, avoiding dependence on the contents of a user's
Cellpose model cache:

```python
from pathlib import Path
from cellpose import models

model_path = Path("analysis/cellpose/models/a4_cyto2_PhD_1000_new")
model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
masks, flows, styles = model.eval(
    image,
    diameter=None,
    channels=[3, 0],
    min_size=250,
    flow_threshold=0,
    cellprob_threshold=-1,
    stitch_threshold=0,
    do_3D=False,
)
```

Use the second file and `channels=[2, 0]` for the sensitive channel. The
production script instead selected these models by name from the Cellpose user
model cache. The cached production files are byte-for-byte identical to the
training-output files identified in `manifest.csv`.

## Scope of the manuscript release and training provenance

The manuscript analysis release begins from the provided HDF5-derived cell
position data. Raw CZI inputs are not included. These frozen weights document
and enable reproduction of the upstream segmentation step when the corresponding
CZI inputs are available.

The retained training records establish the following:

- `a4_cyto2_PhD_1000_new` used 39 TIFF/mask pairs made from maximum-intensity
  projections or crops from G418 b2b4 and a2a4 experiments acquired from
  November 2023 through April 2025. Thirty-eight of the 39 retained Cellpose
  `_seg.npy` records contain manual-edit history. The retained masks contain
  44,187 labelled objects in total.
- `a2_cyto3_Epi_1000_PhD` used 19 TIFF/mask pairs made from epifluorescence
  maximum-intensity projections or crops from G418 a2a4 and b2b4 experiments
  acquired from February 2024 through March 2025. All 19 retained `_seg.npy`
  records contain manual-edit history. The retained masks contain 55,825
  labelled objects in total.

The model-embedded mean training diameters exactly match the means calculated
from all 39 and all 19 retained masks (with the zero-mask training image handled
as Cellpose 3.0.11 does), independently confirming the training-set counts.
No separate validation set, validation metrics, or documented post-training
manual-versus-model QC sample was found in the available repositories and local
training records. Whether such an assessment occurred, and whether the training
TIFFs/masks may be released or should instead be described as available from
the authors upon reasonable request, requires author confirmation. The training
TIFFs and masks are therefore not included here.
