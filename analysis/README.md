# Active analysis

This directory contains the active transformations between the deposited inputs and plot-ready manuscript data. The ordered reviewer commands are maintained in [`../docs/FULL_REPRODUCTION.md`](../docs/FULL_REPRODUCTION.md).

## Hierarchy

| Directory | Scope |
| --- | --- |
| `basic/microscopy/` | construct colony masks and track resistant clones from deposited image-derived inputs |
| `basic/escape/` | clean sensitive artifacts and prepare escape-analysis point clouds |
| `basic/density_maps/` | prepare simulation point clouds and compute experimental/simulation density maps |
| `figure_2/` | simulation overview, event-time helpers, and escape-margin analysis |
| `figure_3/` | Figure 2 mask filtering and master-curve aggregation |
| `figure_4/` | filtered whole-well density generation, LV/RL-derived preparation, and launchers for Figure 4 |
| `legacy/` | development history; excluded from reviewer reproduction |

Scripts in the active directories must not import from `legacy/`. Shared input and output locations come from [`../shared/data_paths.py`](../shared/data_paths.py), controlled by `EICHE_FISCHER_DATA_ROOT`.

Figure plotting is kept separately in [`../figures/`](../figures/): analysis scripts generate derived data, while figure scripts render manuscript panels from those products.
