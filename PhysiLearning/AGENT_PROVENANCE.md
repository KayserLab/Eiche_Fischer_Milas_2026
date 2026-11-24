# Agent provenance and manuscript use

This page is the top-level map from PhysiLearning model artifacts to their roles in the manuscript. It prevents the archived training example from being mistaken for a figure-generating policy.

## Artifact map

| Artifact set | Location | Role |
| --- | --- | --- |
| LV fit inputs and reference outputs | `LV_Fitting_MiniRepo/` | documents and validates the fitted LV system |
| archived two-stage PPO example | `LV_Agent_Training_MiniRepo/reference/saved_models/` | training-method provenance only; not used for paper figures |
| experimental policy | `Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward/` | policy deployed in the in-vitro experiment and used in experimental-policy analysis |
| bootstrap policies | `Bootstrapping/outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/*/models/` | independently fitted/trained policy ensemble used for bootstrap comparisons |

## Experimental-policy checkpoint

The experimental policy is stored as an extracted Stable-Baselines3 save directory containing `data`, `policy.pth`, `policy.optimizer.pth`, `pytorch_variables.pth`, and version/system metadata. Downstream analysis loads this frozen artifact; it does not retrain or alter it.

## Training example versus figure agents

The two-stage configuration and checkpoints in `LV_Agent_Training_MiniRepo/` preserve a runnable example of the training lineage. They establish how PPO pretraining and fine-tuning were performed, but their archived final checkpoint is not substituted for either the experimental policy or the bootstrap policies.

The detailed local training record is preserved in [`LV_Agent_Training_MiniRepo/reference/TRAINING_PROVENANCE.md`](LV_Agent_Training_MiniRepo/reference/TRAINING_PROVENANCE.md).

## Downstream use

[`Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py`](Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py) reads the experimental checkpoint and the bootstrap ensemble directly. It generates policy-behaviour tables and plots without changing or retraining any policy. See the local [`Agent_Behaviour_Analysis/README.md`](Agent_Behaviour_Analysis/README.md) for commands and outputs.

## Integrity for the archival release

Before release, frozen checkpoints and their configuration/provenance files should be covered by a checked-in SHA-256 manifest. Existing local checksum files remain authoritative where present; the final repository-level release record should identify exactly which artifact set each checksum covers.
