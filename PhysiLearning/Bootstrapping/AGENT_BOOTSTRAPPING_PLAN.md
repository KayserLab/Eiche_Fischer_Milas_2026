# Agent Bootstrapping Plan

This document records the rationale and implementation plan for training an agent ensemble from the completed methods-seed LV bootstrap.

## Goal

The LV bootstrap estimates how much the fitted LV dynamics vary when the experimental replicates are resampled. The next question is whether the adaptive-therapy policy learned by PPO is stable across those LV dynamics.

The agent-bootstrap ensemble therefore has two roles:

- estimate sensitivity to LV parameter uncertainty;
- estimate baseline PPO training variability from repeated seeds on the methods-section LV parameters.

## Fixed LV Bootstrap Input

Use the kept LV bootstrap result:

```text
outputs/bootstrap_n20_methods_seed_maxiter1000/bootstrap_results.csv
```

No additional LV refitting is performed for this stage.

## Agent Training Design

Train:

- `20` `bootstrap_lv` agents, one per row of the kept bootstrap LV result;
- `5` `methods_repeat` agents, all on the methods-section LV parameter vector with different seeds;
- `1` `methods_control` agent, also on the methods-section LV parameter vector, as a current-config control.

All 20 bootstrap LV rows are used, including rows where the optimizer reached `maxiter`. The decision is deliberate: the kept LV bootstrap result set contains all 20 rows, and downstream sensitivity should reflect the full kept set.

## Training Procedure

The orchestration script uses the existing agent mini-repo training path:

```text
../LV_Agent_Training_MiniRepo/src/physilearning/train.py
```

Base configs:

```text
../LV_Agent_Training_MiniRepo/configs/2025_10_lv_real_mela_1.yaml
../LV_Agent_Training_MiniRepo/configs/2025_10_lv_real_mela_1_load.yaml
```

Stage 1:

- PPO, `MlpPolicy`;
- canonical `dont_treat` reward;
- `100000` timesteps by default.

Stage 2:

- loads the matching stage-1 `best_reward` model;
- canonical `ttp` reward;
- `160000` timesteps by default.

The script changes only:

- LV parameters;
- model save prefix;
- explicit random seed;
- total timesteps when command-line overrides are used.

## Observation And Noise Decision

The current agent config does not expose resistant cells separately:

```yaml
see_resistance: False
see_prev_action: True
wrapper: VecFrameStack
n_stack: 4
```

Thus the agent sees a four-decision history of total burden and previous treatment decisions.

The current config also does not add Gaussian observation noise. We intentionally keep the current config unchanged to reproduce the training regime of the reference/methods agent. The methods manuscript text should later be revised to avoid claiming Gaussian observation noise unless that training regime is explicitly reintroduced and validated.

## Reproducibility

The training code previously defaulted to time-based seeds. A backwards-compatible seed hook was added:

- if `learning.model.seed` is absent or zero, old behavior remains;
- if `learning.model.seed` is present, that seed is passed through the environment factory and PPO constructor.

The ensemble script writes explicit seeds into every runtime config.

Default seed scheme:

```text
bootstrap_lv:    20260600 + bootstrap_index
methods_repeat:  20260700 + repeat_index
methods_control: 20260800 + control_index
```

This improves traceability. It does not guarantee perfect bitwise determinism across hardware and PyTorch/SB3 internals.

## Outputs

The ensemble script writes a timestamped folder under:

```text
outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/
```

Each run contains:

- `AGENT_BOOTSTRAPPING_PLAN.txt`;
- `metadata.json`;
- `agent_training_manifest.csv`;
- `cfg/*.yaml`;
- `r/<agent_code>/`;
- `models/<agent_code>/*.zip` copied from the isolated training root after each successful stage.

For this ensemble script, the preserved trainer writes primary model files to per-agent isolated roots:

```text
r/<agent_code>/SavedModels/
```

The short `r/<agent_code>/` runtime layout and short model prefixes such as `b0000_s1` are intentional. On Windows, longer TensorBoard event-file paths can exceed `MAX_PATH` and fail even when the parent directory exists.

The copied models inside the bootstrap output folder are the reproducibility archive for this specific ensemble run.

## Execution Strategy

First run a dry-run to verify config generation:

```powershell
..\LV_Agent_Training_MiniRepo\.venv\Scripts\python.exe scripts\train_agent_bootstrap_ensemble.py --dry-run --resume --outdir outputs\bootstrap_n20_methods_seed_maxiter1000\agent_training\dryrun_verify
```

Then run full training without `--dry-run`. The script defaults to `--parallel-workers 8 --torch-num-threads 2` on this workstation. Use `--parallel-workers N` to train a different number of agents concurrently. This is safe because logs, monitor files, copied configs, TensorBoard files, and model zips are isolated per agent.

Machine check on 2026-06-05:

- CPU: Intel Core i9-14900K, 24 physical cores / 32 logical CPUs.
- RAM: about 128 GB total, about 108 GB free at check time.
- PyTorch in the PhysiLearning venvs: `2.1.2+cpu`.
- CUDA: unavailable in the current venv.

The machine does have an NVIDIA RTX 4080 Super visible to Windows via `nvidia-smi`, but the current PhysiLearning venvs use CPU-only PyTorch. GPU training is therefore not available without changing the environment. Other repository scripts use GPU mainly for Cellpose microscopy segmentation, not for the RL training path.

For this PPO ensemble, CPU parallelism is the recommended route:

- default: `8` workers x `2` Torch threads = 16 Torch CPU threads plus environment overhead;
- conservative: `4` workers x `2` Torch threads;
- aggressive: `12` workers x `1` Torch thread.

Behavioural analysis is intentionally separate. The user will copy in the existing behavioural-analysis script later, and metrics will be defined then.
