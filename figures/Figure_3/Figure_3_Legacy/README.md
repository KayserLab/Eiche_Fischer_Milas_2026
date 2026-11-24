# Figure 3 master-curve scripts

The current workflow is split into two entry points:

- `../../../analysis/figure_3/aggregate_average_symmetric_condition_matrix_160.py` performs the expensive in-vitro and in-silico aggregation and saves reusable TSV plotting tables.
- `plot_master_curve_sim_exp.py` ? reads those TSV tables and generates the combined master curves and standalone Figure 3 panels without repeating the analysis.

Supporting modules retained in `..` because the current entry point imports them directly or transitively:

- `aggregate_simulation_master_curve_iqr_direct.py` â€” supplies simulation escape-time loading and simulation aggregation utilities.
- `master_curve_iqr_common.py` â€” shared aggregation bundle and cache utilities used by the simulation aggregator.
- `plot_binned_density_change_strategy_lines_coarsegrained.py` â€” binned aggregation data structures and plotting utilities.
- `plot_simulation_kernel_sweep_direct.py` â€” simulation density-map and interval helpers.

## Legacy scripts

- `aggregate_invitro_40um_master_curve_iqr_direct.py` â€” older direct 40 Âµm in-vitro master-curve aggregation.
- `aggregate_master_curve_iqr_tiff.py` â€” older aggregation from precomputed density-map TIFF stacks.
- `aggregate_preescape_linear_tiff.py` â€” older pre-escape linear-growth aggregation from TIFF stacks.
- `aggregate_simulation_preescape_linear_direct.py` â€” older direct simulation pre-escape linear-growth aggregation.
- `analyze_preescape_low_density_peak_160.py` â€” diagnostic analysis of the low-density pre-escape peak at 160 Âµm.
- `plot_master_curve_iqr_kernel_phase_matrix.py` â€” older multi-kernel, gate, and phase master-curve plotter.
- `plot_preescape_linear_master_curve_invitro_vs_insilico.py` â€” older linear pre-escape comparison plotter.
- `plot_preescape_master_curve_invitro_vs_insilico.py` â€” older relative-growth pre-escape comparison plotter.
- `validate_master_curve_iqr_caches.py` â€” validation utility for the older reusable master-curve caches.

These files are preserved for provenance. They are no longer the recommended plotting entry points, and their relative imports may need adjustment if they are run from this archive directory.
