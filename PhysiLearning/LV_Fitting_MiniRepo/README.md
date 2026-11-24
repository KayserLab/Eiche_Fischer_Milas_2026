# LV Fitting Mini-Repo

This is a self-contained mini-repo for the paper-facing LV refit comparison.

It does one focused procedure:

1. evaluate the saved reference vector, `original_saved`
2. refit the current `new_env` LV model from `original_seed`
3. refit the same current `new_env` LV model from `provided_seed`
4. compare fitted parameters and objectives

This repo intentionally excludes historic parent-repo paths, old environment variants, plotting branches, and methods-adjusted follow-up scripts. The runnable source of truth is local to this directory.

## Terminology

- `original_saved`: saved reference vector in `reference/original_saved.npy`
- `original_seed`: historical initial optimizer guess, `[0.07, 0.14, 2.0, 0.01, 0.06, 2.0, 1.0, 4.0, 6.5, 1.5, 0.01]`
- `provided_seed`: the `original_saved` vector reused as an optimizer start

The parameter order is documented in `reference/seed_metadata.json` and in `scripts/refit_core.py`.

## Layout

- `scripts/run_refit_comparison.py`: main refit comparison entrypoint
- `scripts/refit_core.py`: fitting/evaluation implementation
- `src/physilearning/`: local runtime required by the fitter
- `config.yaml`: local LV environment configuration
- `data/experimental_data_elias/2025_stem_paper_final_final/`: experimental data used by the comparison
- `reference/`: canonical reference vector and comparison metadata
- `outputs/`: generated local validation/run outputs, ignored by git

## Environment

Use a separate virtual environment in this mini-repo. The reference environment was:

- Python 3.10.9
- dependencies pinned in `requirements.txt`

The `.python-version` file records the intended Python version. On this Windows machine, prefer:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3.10 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

After a successful validation install, capture the resolved environment:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```

```bash
./.venv/bin/python -m pip freeze > requirements.lock.txt
```

## Main Commands

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\run_refit_comparison.py --maxiter 1000 --outdir outputs\refit_compare_full
```

Linux/macOS:

```bash
./.venv/bin/python scripts/run_refit_comparison.py --maxiter 1000 --outdir outputs/refit_compare_full
```

The full `--maxiter 1000` run can be slow. It is not required for portability validation.

## Validation

Run the bundled validation script before changing scientific code:

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\validate_smoke.py
```

Linux/macOS:

```bash
./.venv/bin/python scripts/validate_smoke.py
```

The script runs:

- a `--maxiter 1` smoke check
- a `--maxiter 3` capped deterministic validation check
- output-file existence checks
- objective-value comparison against the reference values below

Expected capped validation objectives:

- `original_saved`: `0.10378561340793453`
- `fit_from_original_seed`: `0.3821832095141169`
- `fit_from_provided_seed`: `0.10363965367398319`

Treat validation as passed only if the script exits successfully and reported objective deltas are within floating-point tolerance.

## Outputs

Each run writes a folder under `outputs/` unless `--outdir` is given. Main files:

- `parameter_table.csv`
- `objective_summary.csv`
- `summary.json`
- `report.txt`
- `per_experiment_original_saved.csv`
- `per_experiment_fit_from_original_seed.csv`
- `per_experiment_fit_from_provided_seed.csv`
- `progress/fit_from_original_seed.*`
- `progress/fit_from_provided_seed.*`

Generated outputs are not canonical. Canonical copied artifacts live under `reference/`.

## Integrity

`checksums.sha256` records SHA-256 hashes for reference artifacts and experimental data files. Regenerate it only after deliberately changing canonical data/reference files.

## Methodological Note

This mini-repo reproduces the current `new_env` refit procedure used for the paper-facing comparison. The objective is:

- per experiment: `mean(|S_sim - S_data|) + mean(|R_sim - R_data|)`
- then averaged across the 32 experiments

This repo is for deterministic validation of the capped refit comparison under the pinned environment. It is not a containerized bitwise archive of every possible platform/runtime dependency.
