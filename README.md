# Eiche, Fischer et al. (2026): analysis and figures

Code for *Tissue physics governs therapy dynamics in a 2D melanoma model
system*. The default reviewer run redraws the main and supplementary plots
from frozen figure-source tables and arrays. It uses the original drawing
functions, including their existing statistical annotations. It does not rerun
cohort segmentation, clone tracking, density calculations or agent inference.

The custom dissipative particle dynamics (DPD) simulation code used to generate
the in silico data is available from the authors upon request.

## Regenerate the plots

On Windows, use Python 3.12 and the matching code/data release. Extract into a
short, new folder using the helper beside the ZIPs (not PowerShell's extractor):

```powershell
py -3.12 extract_release.py --code-zip downloads\Eiche_Fischer_2026_Code.zip --data-zip downloads\Eiche_Fischer_2026_Data.zip --destination C:\EF26
cd C:\EF26\Code
py -3.12 setup_environments.py
py -3.12 reproduce_figures.py --data-root C:\EF26\Data --check
py -3.12 reproduce_figures.py --data-root C:\EF26\Data
.\environments\.local\envs\analysis\Scripts\python.exe verification\compare_plot_outputs.py --data-root C:\EF26\Data
```

Installation needs internet access and creates pinned analysis/policy
environments under `environments/.local/`. The runner selects them automatically
and configures UTF-8 output. Generated plots, table copies and logs go to
`Data/Results/`; deposited source files stay unchanged.

Use `--only fig2` to run one figure group (`fig1`, `fig2`, `fig3`, `fig4`,
`suppl`), or `--list` to see all stages. Missing or mismatched frozen inputs
cause an error; there is no fallback to expensive analysis. CSV holds numerical
tables, NPZ holds numerical/image arrays, and JSON records the drawing inputs.
The Figure 1f schematic and assembled Illustrator layouts are outside this run.

## Optional examples

These are separate from the plot run:

```powershell
py -3.12 reproduce_figures.py --data-root C:\EF26\Data --demo simulation
py -3.12 setup_environments.py --roles cellpose
py -3.12 reproduce_figures.py --data-root C:\EF26\Data --demo czi
```

The simulation example compares one complete snapshot (all original fields)
with its matching reduced snapshot using the production reader. It does not
restart the simulator. The CZI example recomputes full-frame segmentation,
coordinate correction and a 120 micrometre density map for one image/day;
it is slower than plotting. One day cannot reproduce time-dependent clone
tracking. Its generated measurements go to
`Data/intermediate_analysis_data/raw_demo/`, diagnostic plots to
`Data/figure_plots/Raw_Demo/`. Historical agreement is reported, not assumed.
No full microscopy timeline is bundled.

## Further information

The [advanced guide](docs/FULL_REPRODUCTION.md) describes the original full
analysis (`--mode full`), which requires additional cohort data and can take
days. The compact public ZIP alone cannot run that route. The data ZIP's
`README.md`, `Documentation/DATA_GUIDE.md` and `FILE_MANIFEST.tsv` describe the
deposited sources, examples and references.

Scientific code lives in `analysis/`, `figures/` and `PhysiLearning/`; shared
paths and formatting in `shared/`; execution helpers in `reproduction/`;
checks and examples in `verification/`. Use a fresh extraction for testing and
compare results with the matching release references. Successful execution
alone does not establish agreement with the manuscript.

The [release validation record](docs/RELEASE_VALIDATION.md) describes the
completed plot test and main-figure comparison, including the corrected
Figure 4j artwork and the remaining optional CZI demonstration limitation.

Release identifiers/DOIs and the final licence will be added before publication.
