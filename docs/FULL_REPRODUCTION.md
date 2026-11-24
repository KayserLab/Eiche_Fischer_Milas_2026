# Analysis guide

Start with the [README](../README.md) to regenerate plots from the deposited
data and run the two processing examples. Commands below are run from the
repository root. Set `$dataRoot` to the full path of your extracted `Data`
folder (the path below is an example; any drive/location can be used):

```powershell
$dataRoot = 'D:\StudyData\Data'
```

## Compare regenerated plots

After generating the plots, compare them with the deposited references:

```powershell
.\environments\.local\envs\analysis\Scripts\python.exe verification\compare_plot_outputs.py --data-root "$dataRoot"
```

The comparison uses the numerical/image policy implemented in
`verification/compare_release_outputs.py`. Its report is saved to
`Data/Results/comparison.json`. The reference plots are under
`Data/Reference/Plot_Renderer_Outputs/`.

## Individual scripts

| Directory | Contents |
| --- | --- |
| `analysis/` | Microscopy and simulation data processing, derived measurements and statistics |
| `figures/` | Main and supplementary figure scripts |
| `PhysiLearning/` | Model fitting and treatment-policy analysis |
| `shared/` | Common paths and plotting utilities |
| `reproduction/` | Pipeline execution and figure-input loading |
| `verification/` | Output comparisons and processing examples |

The runner selects the installed interpreter automatically. When running an
individual script, use its environment from
[`environments/README.md`](../environments/README.md) and set the data root
before starting Python:

```powershell
$env:EICHE_FISCHER_DATA_ROOT = $dataRoot
```

## Full-cohort analysis

The original analysis pipeline is available with `--mode full`. It requires
additional raw and intermediate cohort data that are not included in the
compact Figshare archive. Some steps, especially clone tracking, can take days.
With those inputs available in the configured data directory:

```powershell
py -3.12 setup_environments.py --roles analysis policy cellpose
py -3.12 reproduce_figures.py --mode full --data-root "$dataRoot" --check
py -3.12 reproduce_figures.py --mode full --data-root "$dataRoot"
```

| Option | Purpose |
| --- | --- |
| `--list` | List stages in execution order |
| `--dry-run` | Show commands without running them |
| `--only prepare`, `fig1`, `fig2`, `fig3`, `fig4`, `suppl`, `demo`, `stats` | Select a group and its prerequisites |
| `--from-step NAME` | Resume after earlier required outputs have been generated |
| `--log-dir PATH` | Select a log directory |
| `--plate-workers N`, `--workers N`, `--stage-workers N` | Set concurrency |

These full-pipeline options apply with `--mode full`. Per-stage logs and status
are saved under `Data/tmp/pipeline_runs/`; execution records are under
`Data/Reproduction_Report/`. Raw-data inputs and archived references should be
preserved when comparing newly generated results.

The CZI example writes measurements to
`Data/intermediate_analysis_data/raw_demo/` and plots to
`Data/figure_plots/Raw_Demo/`. Fresh segmentations are reported separately from
historical measurements. A single image/day does not support time-dependent
clone tracking. Simulation generation, policy retraining, schematics and
assembled figure layouts are separate from the downstream pipeline.
