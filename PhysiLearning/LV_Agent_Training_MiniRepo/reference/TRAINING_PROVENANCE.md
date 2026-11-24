# Training Provenance

This mini-repo freezes an archived example of the two-stage LV-agent training
lineage found in the parent repository. It is retained for training-method
provenance and reproducibility examples; its preserved checkpoints are not the
agents used in the paper figures.

Canonical local configs copied here:
- `configs/2025_10_lv_real_mela_1.yaml`
- `configs/2025_10_lv_real_mela_1_load.yaml`

Canonical preserved models copied here:
- `reference/saved_models/2025_10_lv_real_mela_1_best_reward.zip`
- `reference/saved_models/2025_10_lv_real_mela_1_load_best_reward.zip`

Interpretation carried forward:
- stage 1: PPO pretraining on the fitted LV system with `reward_shaping: dont_treat` for `1.0e+5` timesteps
- stage 2: PPO fine-tuning on the same fitted LV system with `reward_shaping: ttp` for `1.6e+5` timesteps
- archived final stage-2 example: `2025_10_lv_real_mela_1_load_best_reward.zip`

Paper-figure agent policy
-------------------------

- The preserved checkpoints in this mini-repo are **not** used to generate
  paper figures.
- The PPO agent used for the in-vitro experiment is stored separately at
  `../../Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward/`.
- Bootstrap-policy comparisons use the independently trained checkpoints in
  `../../Bootstrapping/outputs/.../agent_training/.../models/`.

This package intentionally preserves the runnable artifact lineage rather than trying to rewrite the code to match the prose exactly. Known prose-to-code mismatches were accepted and documented rather than corrected in the runtime.

Relation to the sibling fitting package:
- `../LV_Fitting_MiniRepo/` is the provenance package for the LV refit comparison
- this package uses the frozen fitted LV values as fixed training inputs
