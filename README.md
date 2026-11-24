# Eiche_Fischer_Milas_2026

Code accompanying **A melanoma colony model reveals how physical cell–cell
interactions sustain resistance confinement during adaptive therapy**
by Eiche, Fischer, Milas et al. (2026).

The repository contains microscopy processing, simulation analysis, model
fitting and figure-generation scripts. The main and supplementary data plots
can be regenerated directly from the processed data deposited on [Figshare](https://doi.org/10.6084/m9.figshare.35106367.v1).

## Repository structure

| Folder | Contents |
| --- | --- |
| [`analysis/`](analysis/) | Microscopy and simulation data processing, including segmentation, clone tracking, density and escape measurements, and statistics. |
| [`figures/`](figures/) | Plotting scripts organised by main figure (`Figure_1` to `Figure_4`) and supplementary figures (`Suppl_Figures`). |
| [`PhysiLearning/`](PhysiLearning/README.md) | Lotka-Volterra model fitting, reinforcement-learning policy training, bootstrap uncertainty and treatment-policy analysis. Its README explains the individual components and saved agents. |
| [`shared/`](shared/) | Common data paths, colours and plotting helpers. |
| [`reproduction/`](reproduction/) | Workflow execution and loading the deposited figure inputs. |
| [`verification/`](verification/README.md) | Output comparisons, processing examples and checks of individual analysis steps. |
| [`environments/`](environments/README.md) | Python and dependency definitions used by the installation script. |
| [`docs/`](docs/) | Detailed analysis instructions and data documentation. |

The root scripts `setup_environments.py`, `extract_release.py` and
`reproduce_figures.py` install dependencies, unpack the data and run the
workflow, respectively. The commands below provide the usual starting point.

## Requirements

Windows and Python 3.12 are required for the instructions below. Internet access
is needed for installation. All package dependencies and the required Python
versions are installed automatically in `environments/.local/`.

Tested on Windows 11 Enterprise 64-bit (version 10.0.26200), using Python 3.12.7
for analysis and Python 3.10.9 for policy inference. Exact dependency versions
are recorded in the [analysis](environments/analysis/requirements.lock.txt),
[policy](environments/policy/requirements.lock.txt) and
[Cellpose](environments/cellpose/requirements.lock.txt) lock files. No specialised
hardware or GPU is required for plot reconstruction or the simulation-snapshot
example.

## Installation

Download or clone this repository, then install the dependencies:

```powershell
git clone https://github.com/KayserLab/Eiche_Fischer_Milas_2026.git
cd Eiche_Fischer_Milas_2026
py -3.12 setup_environments.py
```

Installation of the default environments took approximately 40 seconds on the
test desktop (Intel Core i9-14900K, 128 GB RAM). These are the test-machine
specifications, not minimum hardware requirements. This timing excludes the
data download; installation time depends on the computer and internet connection.

## Data

Download `Eiche_Fischer_Milas_2026_Data.zip` from [Figshare](https://doi.org/10.6084/m9.figshare.35106367.v1) to any location on
your computer. **Edit the first two lines below to use your own paths**:

- `$dataZip`: the full path to the downloaded ZIP file.
- `$extractTo`: a new, short folder for extraction, on any drive and with any
  name. This folder must not already exist; the helper creates it for you.

The `D:` paths below are examples only. Run these commands from the repository
folder in PowerShell:

```powershell
$dataZip = 'D:\Downloads\Eiche_Fischer_Milas_2026_Data.zip'
$extractTo = 'D:\StudyData'
py -3.12 extract_release.py --data-zip "$dataZip" --checksums docs\DATA_ARCHIVE.sha256 --destination "$extractTo"
$dataRoot = Join-Path $extractTo 'Data'
```

The helper verifies the checksum and creates a `Data` subfolder inside your
chosen extraction folder. The archive contains processed plot inputs, reference
outputs and two processing examples, rather than the full raw microscopy and
simulation cohorts.

## Reproduce the figures

In the same PowerShell window, check the inputs and generate the plots:

```powershell
py -3.12 reproduce_figures.py --data-root "$dataRoot" --check
py -3.12 reproduce_figures.py --data-root "$dataRoot"
```

Plots, table copies and logs are saved in the `Results/` subfolder of your
extracted `Data` directory.
The plot run took approximately 10 minutes on the test system. Use `--only fig2`
to select a figure (`fig1`, `fig2`, `fig3`, `fig4`, `suppl`), or `--list` to see
all steps. Schematics and assembled figure layouts are prepared separately.

## Processing examples

The data archive includes one complete simulation snapshot and one native CZI
microscopy image/day. Run either example separately:

```powershell
py -3.12 reproduce_figures.py --data-root "$dataRoot" --demo simulation
py -3.12 setup_environments.py --roles cellpose
py -3.12 reproduce_figures.py --data-root "$dataRoot" --demo czi
```

The simulation-snapshot example is the quick demonstration and took approximately
3 seconds on the test desktop. It compares complete and reduced snapshots and
writes `report.json` and `prepared_points.npz` under `Results/demos/simulation/`
in your data directory. The report should contain `status: passed`,
`full_and_reduced_results_identical: true` and `prepared_cells: 35603`.

The optional CZI example runs segmentation and density estimation and is more
computationally demanding.
The custom dissipative particle dynamics (DPD) simulation code used to generate
the in silico data is available from the authors upon request.

## Further use

See the [analysis guide](docs/FULL_REPRODUCTION.md) for output comparison,
individual scripts and full-cohort analysis, and the
[data guide](docs/DATA_MANIFEST.md) for archive contents.
For other datasets, follow the individual-script and full-analysis instructions
in the analysis guide, using compatible input formats and setting your own data
root. The default plot command uses the deposited figure-source inputs.
This repository is distributed under the [MIT licence](LICENSE).
