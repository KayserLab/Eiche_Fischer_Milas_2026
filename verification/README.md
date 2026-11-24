# Optional verification workflows

For the compact public plot release, run
`python verification/compare_plot_outputs.py --data-root PATH` with the installed
analysis interpreter after the default plot run. It checks every output against
the matching release renderer reference using the existing exact comparison
policy. `check_figure_sources.py` tests numerical/schema precision of the new
input loader. Author-baseline comparison findings are supplied separately.

The cohort checks below belong to the full-analysis route and require the
additional full cohort data; the compact public ZIP alone cannot run them.

This directory contains read-only, function-first spot checks for selected manuscript transformations. They inspect real inputs and outputs and write human-reviewable diagnostics; they are not a reduced demo pipeline and are not required to reproduce all figures.

## Output location

All generated checks are written beneath:

```text
<EICHE_FISCHER_DATA_ROOT>/verification/<script_name>/sample_n10_seed<seed>/
```

The checks never overwrite deposited inputs or production analysis outputs. Each cohort-based verifier uses a fixed seed and the shared strategy-balanced selection in `selection.py` where applicable.

`check_installation.py` is the lightweight exception: it imports the root dependencies and active pipeline modules, checks canonical path/YAML wiring and deposited input topology, without running scientific analysis or generating output. After configuring the Figshare archive, run:

```powershell
python verification/check_installation.py --require-data --pipeline-preflight
```

## Available checks

| Verifier | Production stage checked |
| --- | --- |
| `verify_colony_masks.py` | sensitive, resistant, and total colony-mask construction |
| `verify_single_clone_tracking.py` | persistent clone labels across resistant-mask stacks |
| `verify_escape_remove_sen_artifacts.py` | removal of sensitive-channel artifacts |
| `verify_prepare_escape_analysis_inputs.py` | preparation of final experimental escape point clouds |
| `verify_prepare_simulation_pointclouds.py` | simulation snapshot to spatial point-cloud conversion |
| `verify_build_simulation_overview.py` | raw simulations to `combined_results.csv` |
| `verify_escape_margin_analysis.py` | signed escape-margin calculation |
| `verify_density_maps_epanechnikov_coarsegrained.py` | experimental Epanechnikov density maps |
| `verify_density_maps_epanechnikov_simulations_coarsegrained.py` | simulation Epanechnikov density maps |
| `verify_generate_figure2_mask_filtered_density_maps.py` | extra Figure 2 mask filter used for Figure 3 |
| `verify_aggregate_average_symmetric_condition_matrix_160.py` | experimental master-curve aggregation and binning |
| `verify_master_curve_helpers.py` | exact small-input checks for internal density and binning helpers |

`verify_density_maps_common.py` is shared support code rather than a standalone command.

## Run

First configure `EICHE_FISCHER_DATA_ROOT` as described in the root README. Then run a selected verifier from the repository root, for example:

```powershell
python verification/verify_colony_masks.py
python verification/verify_build_simulation_overview.py
python verification/verify_density_maps_epanechnikov_coarsegrained.py
```

Use `python verification/<name>.py --help` to see stage-specific options. Leave `--seed` at its default when present so that review selections remain reproducible.

## Interpretation

Most spatial checks write one high-resolution PNG per selected replicate plus `metadata.json` and a short README in the output folder. A PASS means the production transformation reproduced the displayed and recorded checks for that selection. It does not by itself validate every cohort-level result; the full workflow remains the authoritative reproduction.
