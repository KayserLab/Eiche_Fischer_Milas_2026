# Figure 3 analysis

The active workflow has five runnable scripts. Run them directly in PyCharm.
Shared helpers live in `density_common.py`; HDF5 parameters and validation live
in `HDF5_analysis/constants.py`.

| Script | Purpose |
|---|---|
| `HDF5_analysis/compute_master_curves.py` | Main master-curve data, including density at t0, for experiments and simulations |
| `HDF5_analysis/compute_flux_vs_density.py` | Separate simulation endpoint-mean flux tables |
| `HDF5_analysis/compute_flux_vs_density_experiment.py` | Separate experimental endpoint-mean flux tables |
| `HDF5_analysis/compute_cell_size_density.py` | Recalculate 160-um cell-local kernel densities for the expansion plot, preserving original expansion measurements |
| `generate_figure2_mask_filtered_density_maps.py` | Optional historical mask-filtered TIFF pipeline; current HDF5 plots do not require it |

The three HDF5 entry points default to **120 and 160 micrometres**. Each has an
editable `DEFAULT_KERNEL_RADII_UM = (120.0, 160.0)` near the top. Their functions
can also be imported without starting an analysis.

## Current plotting scripts

The plotters remain in `figures/Figure_3/`:

- `plot_hdf5_master_curves.py` reads the main `compute_master_curves.py` outputs
  by default (`first_endpoint`). The alternative `endpoint_mean` coordinate
  reads the two separate flux-table outputs.
- The former standalone 120-micrometre simulation master-curve plotter is
  archived in `figures/Figure_3/Figure_3_Legacy/`. The current master plotter
  exports both simulation and experimental panels for either radius.
- `plot_at50_density_maps.py` and
  `plot_at50_hdf5_absolute_density_trajectories.py` use their own caches and the
  shared density functions. Both support 120 and 160 micrometres.
- `b_plot_cell_size_vs_density.py` reads precomputed pressure-analysis tables.
  Its original minimum of 100 observations per bin and pressure-run pooling
  are preserved. It now defaults to the regenerated 160-um data in
  `figure_data/Figure_3/cell_size_vs_density/kernel_radius_160um/`.
  Pressure-run IDs are not treated as master-curve database run IDs.

`compute_cell_size_density.py` reads the original `AT50.h5`, `AT100.h5`, and
`CT.h5` pressure tables without modifying them. It evaluates the same
Epanechnikov kernel and 14-neighbour density-validity rule at each saved cell
position, replacing the original 87.6-um count/area estimator. Positions were
already filtered by the recovered pressure pipeline. Cell types, surviving
cells' `delta_d`, all six pressure runs, and their 12-hour window attributes
are copied exactly and verified against the originals. The radius is 160 um.
The new files include source SHA256 fingerprints and explicit kernel metadata.

Only density is recalculated: the original 12-hour expansion, 100-cell bin
minimum, pooling, units, and plotting style are unchanged. This per-cell
expansion outcome does not use the resistant-density-change support gate from
the growth maps; the separate metadata validator checks the actual density
kernel and neighbour minimum. Old kernel-dependent division/death estimates
are not copied into these plot-specific derived files.

The master-curve default is **160 micrometres**, controlled by
`MASTER_KERNEL_RADIUS_UM` in `HDF5_analysis/constants.py`. For a 120-micrometre
master curve with the main plotter's t0 definition, set its radius in the PyCharm
  run configuration. The master runner generates both radii automatically.

## Shared plotting definitions

Default plot folders under `figure_plots/Figure_3/` are:

- `density_mastercurve/`: experimental and simulation master curves.
- `density_maps/`: experimental and simulation density maps.
- `density_trajectories/`: experimental and simulation density trajectories.
- `cell_size_vs_density/`: cell-size plots and their optional verification output.

The 120-um and 160-um plots share the same folder for each plot type; their
filenames identify the radius. These defaults apply on the next plotting run.

The current master curve, HDF5 maps, and HDF5 trajectories use the same raw
point-cloud inputs, fixed grids, Epanechnikov kernel, unit conversions,
experimental exclusions, simulation outlier filter, and support mask:
14 neighbours at both endpoints; finite fields; resistant density at t0
greater than or equal to `2.25/(pi*radius^2)` in native coordinates. Simulations
use run numbers greater than 9. Maps/trajectories retain the D5/run_18 examples.

Master curves default to relative resistant growth per day, normalized by resistant
density at t0. Maps and trajectories show **absolute** resistant-density change
per day and absolute total density at t0. They share the density calculation and
support mask, while displaying different growth definitions. Trajectories retain
their original spatial mean and sample SD, centered four-observation running
average with propagated SD, and 0-40-day view. Master curves retain their
density bins and aggregation across replicates/protocols; maps show individual
pixels and total-density context. They are different views of the same fields.
Maps retain their original crop/zoom behavior, raw-image padding, colourbar label,
zero-change masking, and isolated-pixel display filter. The native kernel-support
decision is retained through conversion to physical units, including equality at
the threshold; display filters are applied afterwards as in the original plot.

Running `plot_at50_density_maps.py` now produces only the matched HDF5 panels
for both radii. TIFF analysis and unfiltered comparison panels are available
through historical helpers, outside the default run. Map and trajectory outputs
are explicitly named `absolute`; trajectory caches identify `mean_sd`.

Outdated map caches and incompatible relative or median/IQR trajectory caches are rejected.
Maps can reuse current endpoint-density caches because the absolute rate is
calculated while plotting. Trajectories rebuild matching absolute mean/SD caches from
raw HDF5 as needed. This does
not require recomputing the existing verified master-curve diagnostics.

## Existing data and validation

Source cleanup does not change the data directory layout or numerical rules:
both radii use 14 minimum neighbours and resistant support numerator 2.25.
The verified October 2026 diagnostic files and the verified 120-micrometre
endpoint-mean files can be reused despite missing newer parameter metadata.
`constants.py` checks their exact SHA256 fingerprints against the audit.
Other files require matching parameter metadata; explicit conflicting settings
are always rejected. The older standalone 160-micrometre flux tables use the
old threshold and need recomputing if that plotting route is required.

## Legacy

`Legacy/` preserves older aggregation workflows and the original HDF5 plotting
layouts. Its single-radius wrappers target the active base computations with
an explicit 120-micrometre radius. Current Figure 3 plotters import only active
analysis modules, not historical aggregation entry points.

The former `compute_master_curve_selection_diagnostics.py` entry point is now
`HDF5_analysis/compute_master_curves.py`. Its stored HDF5 filenames retain
`master_curve_selection_diagnostics_radius_...` for data compatibility.

## Verification

- `verification/verify_figure3_kernel_settings.py`: support rules, stale caches,
  parameter metadata, and fingerprint-based legacy acceptance.
- `verification/verify_figure3_plotting.py`: local synthetic inputs through all
  four current plotters, with PDFs and raster fingerprints.
- `verification/verify_figure3_cleanup.py`: shared-function equivalence, imports,
  radius entry points, and comparison of pre/post-cleanup plot fingerprints.

These checks use local fixtures and do not regenerate laboratory analysis data.
