# PhysiLearning branch

This directory documents the manuscript's model-fitting and reinforcement-learning branch. It is related to the main experimental pipeline but is intentionally self-contained because it has distinct environments, model artifacts, and provenance requirements.

## Which component should I use?

| Component | Purpose | Manuscript role |
| --- | --- | --- |
| [`LV_Fitting_MiniRepo/`](LV_Fitting_MiniRepo/README.md) | reproduce and validate the focused LV fit comparison | establishes the fitted LV system |
| [`LV_Agent_Training_MiniRepo/`](LV_Agent_Training_MiniRepo/README.md) | reproduce the preserved two-stage PPO training procedure | training-method provenance; archived example checkpoints are not figure agents |
| [`Agent_used_for_experiment/`](AGENT_PROVENANCE.md) | frozen PPO checkpoint deployed in the in-vitro experiment | experimental-policy source |
| [`Bootstrapping/`](Bootstrapping/AGENT_BOOTSTRAPPING_PLAN.md) | bootstrap fitting and independently trained policy ensemble | uncertainty and bootstrap-policy source |
| [`Agent_Behaviour_Analysis/`](Agent_Behaviour_Analysis/README.md) | probe frozen policies and prepare Figure 4 policy-behaviour outputs | Figure 4 analysis |

The authoritative mapping between checkpoints and manuscript uses is in [`AGENT_PROVENANCE.md`](AGENT_PROVENANCE.md).

## Environments and order

Use Python 3.10.9 and a separate local environment inside each mini-repository. Their dependency files and exact commands are in their local READMEs.

For a complete provenance-oriented reproduction:

1. Run the fitting validation in `LV_Fitting_MiniRepo/`.
2. Run the training validation or full two-stage training in `LV_Agent_Training_MiniRepo/` if training-method replay is required.
3. Consult `AGENT_PROVENANCE.md` before selecting any checkpoint; the mini-repository's archived checkpoints are not the policies used in manuscript figures.
4. Run the policy-behaviour analysis using the frozen experimental and bootstrap agents.
5. Return to the repository-root environment for the remaining Figure 4 preparation and plotting scripts.

Generated fitting or training runs are procedural reproductions and need not be bitwise identical across hardware. The deposited policy artifacts and their provenance records are the fixed inputs to downstream manuscript analysis.
