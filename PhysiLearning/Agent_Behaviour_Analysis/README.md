# RL policy-behaviour analysis

This directory probes the frozen experimental and bootstrap PPO policies with defined synthetic four-frame observation histories. It does not train agents and does not modify checkpoints.

## Inputs

- experimental policy: `../Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward/`;
- bootstrap policy ensemble: the latest valid model directory beneath `../Bootstrapping/outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/`; and
- agent roles and distinctions: [`../AGENT_PROVENANCE.md`](../AGENT_PROVENANCE.md).

The analysis requires Stable-Baselines3/PyTorch dependencies compatible with the saved policies. Use the PhysiLearning Python 3.10.9 environment described by the training mini-repository rather than assuming the root plotting environment can load these models.

## Run

From the repository root:

```powershell
python PhysiLearning/Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py --mode both
```

The Figure 4 launcher is equivalent and forwards any additional command-line arguments:

```powershell
python analysis/figure_4/run_rl_policy_behavior_analysis.py --mode both
```

Use `--experimental-agent-path` or `--bootstrap-agent-dir` only when reproducing from an explicitly selected archival artifact set.

## Outputs

Canonical manuscript outputs are written below the configured external data root:

- `figure_data/Figure_4/rl_policy_behavior/` for CSV source data;
- `figure_plots/Figure_4/rl_policy_behavior/` for PDF, PNG, and SVG plots.

The checked-in `outputs/` directory records existing analysis artifacts and is useful for provenance. New manuscript outputs should follow the canonical external-data locations above.

The replotter can regenerate styled panels from grouped CSVs:

```powershell
python PhysiLearning/Agent_Behaviour_Analysis/replot_rl_policy_behavior.py
```
