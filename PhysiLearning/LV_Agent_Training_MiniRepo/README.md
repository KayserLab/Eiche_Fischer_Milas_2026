# LV Agent Training Mini-Repo

This is a self-contained mini-repo containing an archived example of the
two-stage LV PPO training lineage. Its preserved models document the training
procedure; they are not used to generate paper figures.

It does one focused procedure:

1. run stage-1 PPO pretraining on the fitted LV system
2. run stage-2 PPO fine-tuning on the same fitted LV system
3. evaluate the preserved stage-2 example agent

This repo intentionally excludes historic parent-repo paths, unrelated KPP branches, SLURM plumbing, and old training variants.

## Relation To Fitting

This package does not rerun fitting. It uses the fitted LV system as a fixed input.

The sibling repo `../LV_Fitting_MiniRepo/` documents and validates the LV refit comparison. The frozen LV values used here are recorded in `reference/fitted_lv_params.json` and embedded in the copied training configs.

## Canonical Training Lineage

Configs:

- `configs/2025_10_lv_real_mela_1.yaml`
- `configs/2025_10_lv_real_mela_1_load.yaml`

Interpretation:

- stage 1: PPO pretraining with `reward_shaping: dont_treat` for `1.0e+5` timesteps
- stage 2: PPO fine-tuning with `reward_shaping: ttp` for `1.6e+5` timesteps, loading the best stage-1 model

Preserved reference models:

- `reference/saved_models/2025_10_lv_real_mela_1_best_reward.zip`
- `reference/saved_models/2025_10_lv_real_mela_1_load_best_reward.zip`

Archived final stage-2 example agent:

- `reference/saved_models/2025_10_lv_real_mela_1_load_best_reward.zip`

For the experimental-policy and bootstrap-policy provenance, see
[`../AGENT_PROVENANCE.md`](../AGENT_PROVENANCE.md).

## Layout

- `configs/`: canonical two-stage training configs
- `scripts/training_common.py`: shared local helper for config rewriting, training, and evaluation
- `scripts/run_stage1_training.py`: stage-1 pretraining entrypoint
- `scripts/run_stage2_finetune.py`: stage-2 fine-tuning entrypoint
- `scripts/run_two_stage_pipeline.py`: combined two-stage runner
- `scripts/evaluate_saved_agent.py`: deterministic evaluation of a saved PPO model
- `src/physilearning/`: local minimal runtime for `LvEnv`, training, reward, and evaluation
- `reference/`: frozen fitted LV parameters, provenance note, and preserved candidate models
- `Training/`: generated training outputs, ignored except `Training/Logs/empty_monitor.csv`
- `outputs/`: generated runtime configs and evaluation outputs, ignored by git

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
.\.venv\Scripts\python.exe scripts\run_stage1_training.py
.\.venv\Scripts\python.exe scripts\run_stage2_finetune.py
.\.venv\Scripts\python.exe scripts\run_stage2_finetune.py --use-reference-pretrain
.\.venv\Scripts\python.exe scripts\run_two_stage_pipeline.py
.\.venv\Scripts\python.exe scripts\evaluate_saved_agent.py --episodes 5
```

Linux/macOS:

```bash
./.venv/bin/python scripts/run_stage1_training.py
./.venv/bin/python scripts/run_stage2_finetune.py
./.venv/bin/python scripts/run_stage2_finetune.py --use-reference-pretrain
./.venv/bin/python scripts/run_two_stage_pipeline.py
./.venv/bin/python scripts/evaluate_saved_agent.py --episodes 5
```

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

The script checks:

- preserved final agent loads and writes an `.h5` trajectory
- short stage-1 smoke training completes and writes config/model outputs
- short stage-2 smoke fine-tuning completes and writes config/model outputs

This validates functional portability. PPO training is procedural rather than bitwise reproducible because seed handling is preserved as originally implemented.

## Generated Outputs

Generated files belong under `outputs/` or `Training/`. They are not canonical reference artifacts.

Canonical copied artifacts belong under `reference/`.

## Integrity

`checksums.sha256` records SHA-256 hashes for reference artifacts and canonical configs. Regenerate it only after deliberately changing canonical files.

## Methodological Note

This package preserves the runnable artifact lineage rather than rewriting the runtime to match prose exactly.

Accepted mismatches:

- the written noise description is not forced back into runtime if the preserved artifact lineage does not use it
- `dont_treat` pretraining reward is preserved as implemented in code
- seed handling is left as-is; this package is practical/procedural reproducibility, not strict bitwise replay

## Supporting Analysis: Policy Distillation

`scripts/distill_policy.py` distils a frozen PPO checkpoint into a shallow CART regression tree on P(treatment ON) (supporting analysis; it does not retrain or modify any model or config). Optional dependency: `requirements-analysis.txt`. Results, method and limitations: `analysis/policy_distillation/ANALYSIS_NOTE.md`.
