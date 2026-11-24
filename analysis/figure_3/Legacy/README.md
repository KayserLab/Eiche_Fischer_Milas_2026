# Archived Figure 3 analysis

- `aggregate_average_symmetric_condition_matrix_160.py`: historical symmetric
  growth aggregation. Its small histogram-statistics helper is now included in
  this file.
- `build_master_curve_pixel_store.py` and
  `aggregate_well_weighted_master_curve.py`: historical pixel-store and
  well-weighted workflow.
- `HDF5_analysis/fig_flux_vs_density*.py`: original plot layouts and period
  comparisons.
- `HDF5_analysis/compute_flux_vs_density*_kernel_radius_120.py`: compatibility
  wrappers for the active computations at 120 micrometres.

The current workflow is documented in `../README.md`. Shared density and
escape-time helpers now live in `../density_common.py`.
