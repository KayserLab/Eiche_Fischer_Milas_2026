# LV Fitting Bootstrap

This directory contains a standalone bootstrap workflow for the validated LV fitting mini-repo.

It does not modify the fitting or agent-training pipelines. It reads the fitting implementation, config, data, and optional full-refit summary from `../LV_Fitting_MiniRepo/` and writes generated results under `outputs/`.

## Bootstrap Unit

The default bootstrap unit is the experimental replicate. The script samples the 32 LV fitting experiments with replacement, stratified by treatment type (`NT`, `CT`, `Pulse`, `AT100`).

This preserves the original treatment composition while estimating parameter variability across experimental replicates. That is more defensible for paper uncertainty intervals than resampling individual timepoints, because timepoints within a replicate are correlated.

## Environment

Use Python 3.10 and the same dependencies as `LV_Fitting_MiniRepo`.

Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can also run this with the fitting mini-repo venv because dependencies are identical:

```powershell
..\LV_Fitting_MiniRepo\.venv\Scripts\python.exe scripts\bootstrap_lv_refits.py --n-bootstrap 1 --maxiter 1 --outdir outputs\smoke
```

## Smoke Test

```powershell
..\LV_Fitting_MiniRepo\.venv\Scripts\python.exe scripts\bootstrap_lv_refits.py --n-bootstrap 1 --maxiter 1 --outdir outputs\smoke
```

## Recommended Runs

Pilot run:

```powershell
..\LV_Fitting_MiniRepo\.venv\Scripts\python.exe scripts\bootstrap_lv_refits.py --n-bootstrap 10 --maxiter 100 --outdir outputs\pilot_n10_maxiter100
```

Paper-facing run, computationally expensive:

```powershell
..\LV_Fitting_MiniRepo\.venv\Scripts\python.exe scripts\bootstrap_lv_refits.py --n-bootstrap 100 --maxiter 1000 --outdir outputs\bootstrap_n100_maxiter1000
```

Parallel run initialized from the methods / agent-training LV parameters:

```powershell
..\LV_Fitting_MiniRepo\.venv\Scripts\python.exe scripts\bootstrap_lv_refits_from_methods_params.py --n-bootstrap 20 --maxiter 1000 --outdir outputs\bootstrap_n20_methods_seed_maxiter1000
```

## Seed Modes

Default: `full_provided_best`, which starts every bootstrap fit from `../LV_Fitting_MiniRepo/outputs/refit_compare_full/summary.json`, using the completed `fit_from_provided_seed` parameter vector.

Other options:

- `provided_saved`: starts from `reference/original_saved.npy`
- `original_seed`: starts from the historical initial guess
- `full_original_best`: starts from the completed `fit_from_original_seed` vector
- `agent_training_frozen`: starts from `../LV_Agent_Training_MiniRepo/reference/fitted_lv_params.json`, mapped into the LV fitting parameter order

For uncertainty around the best current LV fit, use the default.
For checking whether the methods / agent-training parameter basin is stable under bootstrap resampling, use `scripts/bootstrap_lv_refits_from_methods_params.py` or pass `--seed-mode agent_training_frozen`.

## Outputs

Main files:

- `metadata.json`: run settings and seed vector
- `bootstrap_results.csv`: one row per bootstrap fit, including all fitted parameters
- `bootstrap_sampled_experiments.csv`: sampled replicate composition for every bootstrap
- `parameter_intervals.csv`: parameter summary ranges and percentile intervals
- `report.txt`: short text summary
- `runs/bootstrap_*/`: per-fit optimizer progress and sampled-experiment details

Use `parameter_intervals.csv` to check whether methods-section parameters fall inside the bootstrap ranges.

## Agent Training On Bootstrap LV Fits

After the kept methods-seed bootstrap run, prepare the agreed agent ensemble with:

```powershell
..\LV_Agent_Training_MiniRepo\.venv\Scripts\python.exe scripts\train_agent_bootstrap_ensemble.py --dry-run --resume --outdir outputs\bootstrap_n20_methods_seed_maxiter1000\agent_training\dryrun_verify
```

Full training uses the same script without `--dry-run`. The script defaults to `--parallel-workers 8 --torch-num-threads 2` on this workstation:

```powershell
..\LV_Agent_Training_MiniRepo\.venv\Scripts\python.exe scripts\train_agent_bootstrap_ensemble.py
```

Use `--parallel-workers 4` if the machine becomes sluggish. Use `--parallel-workers 12 --torch-num-threads 1` if you want to push CPU utilization harder. The current venv uses CPU-only PyTorch (`torch 2.1.2+cpu`), so GPU training is not available in this environment.

Default ensemble:

- `20` agents trained on the kept bootstrap LV parameter vectors.
- `5` repeated-seed agents trained on the methods-section LV parameters.
- `1` current-config control agent trained on the methods-section LV parameters.
- Stage 1: `100000` timesteps with the canonical `dont_treat` config.
- Stage 2: `160000` timesteps with the canonical `ttp` config, loading the matching stage-1 `best_reward` model.

Outputs are written under:

- `outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/<timestamp>/`

Each agent gets an isolated training root under:

- `outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/agt_<timestamp>/r/<agent_code>/`

The short `r/<agent_code>/` layout and short model prefixes are intentional on Windows: TensorBoard event-file paths otherwise exceed `MAX_PATH` during parallel runs.

Model zips are copied into:

- `outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/agt_<timestamp>/models/<agent_code>/`

See `AGENT_BOOTSTRAPPING_PLAN.md` for the rationale, seed scheme, and output structure.

Legacy/simple bootstrap-only training is still available with:

```powershell
..\LV_Agent_Training_MiniRepo\.venv\Scripts\python.exe scripts\train_agents_on_bootstrap_lvs.py --bootstrap-results outputs\bootstrap_n20_methods_seed_maxiter1000\bootstrap_results.csv --n-agents 20 --mode two_stage --outdir outputs\agents_n20_two_stage
```

The script consumes `bootstrap_results.csv`, injects each LV parameter vector into the canonical agent configs, and writes runtime configs plus a manifest under the chosen `outputs/` folder.

Supported modes:

- `two_stage`: canonical procedure; stage 1 pretraining from scratch, then stage 2 fine-tuning from that bootstrap-specific stage-1 model.
- `stage2_reference`: cheaper sensitivity mode; stage 2 fine-tuning from the preserved reference pretrain model only.

Smoke test using the one-row bootstrap smoke output:

```powershell
..\LV_Agent_Training_MiniRepo\.venv\Scripts\python.exe scripts\train_agents_on_bootstrap_lvs.py --bootstrap-results outputs\smoke\bootstrap_results.csv --n-agents 1 --stage1-timesteps 64 --stage2-timesteps 64 --outdir outputs\agent_smoke
```

Dry run without training:

```powershell
..\LV_Agent_Training_MiniRepo\.venv\Scripts\python.exe scripts\train_agents_on_bootstrap_lvs.py --bootstrap-results outputs\bootstrap_n20_methods_seed_maxiter1000\bootstrap_results.csv --n-agents 20 --mode two_stage --dry-run --outdir outputs\agents_n20_two_stage_dryrun
```

The legacy script writes trained model zips to:

- `../LV_Agent_Training_MiniRepo/Training/SavedModels/`

The bootstrap orchestration manifest records the expected model paths:

- `agent_training_manifest.csv`

Parameter mapping from LV bootstrap fit to agent config:

- `growth_rate_sus` -> `env.growth_rate_wt`
- `growth_rate_res` -> `env.growth_rate_mut`
- `random_death` -> `env.death_rate_wt` and `env.death_rate_mut`
- `death_treat_sus` -> `env.treat_death_rate_wt`
- `competition` -> `env.LvEnv.competition_wt`
- `ramp_time_up` -> `env.LvEnv.ramp_time_up`
- `ramp_time_down` -> `env.LvEnv.ramp_time_down`
- `carrying_capacity` -> `env.LvEnv.carrying_capacity`
- `on_treat_threshold` -> `env.LvEnv.on_treat_threshold`
- `off_treat_threshold` -> `env.LvEnv.off_treat_threshold`
- `min_treat` -> `env.LvEnv.min_death_during_treat`
