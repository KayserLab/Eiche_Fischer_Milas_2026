# Advanced reproduction guide

Start with the [repository README](../README.md) for the default plot run and
two separate examples. This guide describes the original full analysis,
selected explicitly with `--mode full`. It requires additional cohort data;
the compact public ZIP alone cannot run it. Clone tracking can take days.
This guide does not change any
scientific definitions, exclusions, thresholds, statistical tests or plotting
parameters in the source scripts.

## Data and environments

Point `--data-root` at the extracted `Data` directory.
The runner passes it to every stage. For a standalone scientific script, set
`EICHE_FISCHER_DATA_ROOT` to that directory before starting the interpreter.
The demonstrations accept `--data-root` explicitly. Use a separate extraction
for testing; keep the original laboratory data unchanged.

`setup_environments.py` installs the exact Python/package versions declared in
`environments/manifest.json` and the three resolved lock files. The runner
selects the interpreter for each stage. Installation needs internet access;
inputs, fixed weights and fitting resources come from the data ZIP. Windows is
the reference platform. On other systems, interpreter paths use
`environments/.local/envs/analysis/bin/python` and installation needs independent checking.
`--resolve` is a maintainer option for generating replacement dependency locks;
do not use it for reviewer installation.

## Commands

Use the installed analysis interpreter and prefix every option below with
`reproduce_figures.py --mode full`. Install all three environments with
`python setup_environments.py --roles analysis policy cellpose` first.

| Option | Meaning |
| --- | --- |
| `--list` | Show stages and their order without running analyses |
| `--dry-run` | Show child commands without running analyses |
| `--check` | Check interpreter and required input paths |
| `--only prepare`, `fig1`, `fig2`, `fig3`, `fig4`, `suppl`, `demo`, `stats` | Select groups; declared prerequisites are included |
| `--from-step NAME` | Resume; earlier required outputs must already exist |
| `--log-dir PATH` | Choose the parent directory for per-run logs |
| `--plate-workers N`, `--workers N`, `--stage-workers N` | Control concurrency |

A clean complete full-analysis run uses `--mode full` and no selection,
resume or skip arguments. These options apply to the full route only.
Defaults are one plate, one analysis worker and one stage at a time. Keep
them until the complete workflow and resource use have been validated.

The full-analysis route includes the native CZI example. Run the additional full
simulation example with:

```powershell
.\environments\.local\envs\analysis\Scripts\python.exe verification\simulation_snapshot_demo.py --data-root "C:\EF26\Data"
```

It compares the unchanged full-state example with the matching reduced file
using the existing production point-cloud reader and its default settings.
Reports and arrays go to `Reproduction_Report/simulation_snapshot_demo/`.
It does not restart a simulator or modify the main cohort. The microscopy
example recomputes segmentation; historical references remain labelled separately.

## Outputs and validation

`Data/tmp/pipeline_runs/<run>/run.json` records stage status, with logs alongside.
Execution receipts under `Reproduction_Report/` record commands, interpreters
and code identities. A successful exit alone does not certify manuscript agreement.

Compare regenerated tables, source data, statistics and plots with the original
frozen outputs. Author-supplied references are in
`Data/Reference/Author_Supplied_Outputs/`; their provenance is in the data
manifest. They must not be copied into live output directories to skip producer
stages. Differences are findings for review and do not authorize changing
scientific methods, references or comparison tolerances.

The local test uses a frozen code copy, fresh ZIP extraction and separate logs.
It stops on the first error; proposed fixes are reviewed by the author before
application. A colleague then tests the exact same code, ZIP and README.

Full acquisitions, training/restarts, the Figure 1f schematic and assembled Illustrator figures are
outside this downstream reproduction boundary. The data archive contains
full-cohort observations and geometry, reduced simulation snapshots,
pressure/expansion observations, fixed resources, selected images, a native
CZI example and a complete original simulation snapshot. Its data guide and
manifest define the exact boundaries.

## Integrated upstream sources

The supplementary AT+/AT+100 comparison includes the `EF_dev` update from
`a923333`: Kaplan–Meier survival, failure/escape times and shared-initial-condition
statistics. Its stage requires simulation point clouds, escape margins and the
simulation overview.

Older sources from `main` are retained in the existing `analysis/legacy/` and
figure legacy folders, including the earlier TTE/TTP and trajectory plots.
They do not replace the later publication implementations or join the main
execution plan. The Figure 1f generation script is omitted because the panel
is a schematic.

Upstream validation helpers live in `reproduction/`. The standalone
`completion.py` reuse API is retained; the publication runner does not invoke
it and launches every selected stage. Escape-table reuse additionally requires
the existing input/code/output provenance checks.
