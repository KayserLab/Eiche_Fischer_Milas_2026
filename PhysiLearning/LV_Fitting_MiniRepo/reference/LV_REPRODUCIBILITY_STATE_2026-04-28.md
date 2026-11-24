# LV Reproducibility State

Date: 2026-04-28

This note freezes the current interpretation of the LV refit/evaluation work so a future chat can resume without reconstructing the context from scratch.

## Goal

We are trying to decide which `new_env` LV fit is the most plausible reproducibility candidate for a paper submission, given that the exact historical script/seed/init combination that produced the original methods-section values is not fully known.

## Naming Clarification

The word "original" has been overloaded. Use these terms instead:

- `original_saved`
  - Meaning: the saved 11-parameter vector currently used as the reference/original values.
  - File:
    - `scripts/kpp/recreate_29_sim_fits_res_comp_delays.npy`
    - `scripts/kpp/recreate_2_29_sim_fits_res_comp_delays.npy`
  - In practice, both files currently contain the same vector.

- `original_seed`
  - Meaning: the `initial_guess` parsed from the historical script at a given commit.
  - This is not the same thing as `original_saved`.
  - It depends on the commit:
    - `old` commit `576b440`: historical script initial guess
    - `new` commit `d960cb5cf6745d25b1f4979bdbf5af9e09dc6872`: historical script initial guess

- `provided_seed`
  - Meaning: the externally supplied seed vector passed into the refit helper.
  - In the April 27 run, for the `new` commit, this is the saved reference vector from `original_saved`.
  - For the `old` commit, the helper also used the historical 11-value comment vector as the provided seed for that lineage.

So for the April 27 `new_env` comparison:

- `new / original_seed` = rerun of the `new` commit starting from the `new` script's own historical `initial_guess`
- `new / provided_seed` = rerun of the `new` commit starting from the saved reference values (`original_saved`)

This is the key distinction that was getting mixed under the word "original".

## Commits / Lineages

- `old` = `576b440`
- `new` = `d960cb5cf6745d25b1f4979bdbf5af9e09dc6872`

Current working assumption for the paper is to ignore `old_env` and focus on `new_env`.

## April 27 Artifacts To Trust

Refit batch:

- `logs/lv_refits/20260427_095440/plan.json`
- `logs/lv_refits/20260427_095440/summary.csv`
- `logs/lv_refits/20260427_095440/analysis_report.txt`

Evaluation in envs:

- `logs/kpp/env_score_matrix_20260427_095440/reproduction_report.txt`
- `logs/kpp/env_score_matrix_20260427_095440/score_matrix.csv`
- `logs/kpp/env_score_matrix_20260427_095440/score_matrix.json`

Representative trajectory comparison:

- `logs/kpp/trajectory_comparison_20260427/lv_threeway_new_env_comparison.png`
- `logs/kpp/trajectory_comparison_20260427/lv_threeway_new_env_comparison.pdf`
- `logs/kpp/trajectory_comparison_20260427/selected_experiment_metrics.csv`
- `logs/kpp/trajectory_comparison_20260427/selected_experiments.json`

Helper script for the trajectory plots:

- `scripts/kpp/plot_lv_threeway_new_env_comparison.py`

## April 27 `new_env` Headline Result

From `logs/kpp/env_score_matrix_20260427_095440/score_matrix.csv`:

- `original_saved`
  - `new_env = 0.103785613408`
- `new_original_seed_best`
  - `new_env = 0.10492073126`
  - parameter `L2` to `original_saved` = `3.50161856979`
- `new_provided_seed_best`
  - `new_env = 0.0944254617938`
  - parameter `L2` to `original_saved` = `19.1262128322`

Interpretation:

- `new_provided_seed_best` is the best fit by the `new_env` objective.
- `new_original_seed_best` is the closest fit to `original_saved` in parameter space.

## Practical Decision Framing

If the paper needs the best reproducibility candidate under the `new_env` objective:

- prefer `new_provided_seed_best`

If the paper needs the closest parameter-level reproduction of the saved reference values:

- prefer `new_original_seed_best`

This is the current unresolved tradeoff.

## Parameter Table In Current Discussion

Parameter order in the LV script:

1. `growth_rate_sus`
2. `growth_rate_res`
3. `competition`
4. `random_death`
5. `death_treat_sus`
6. `ramp_time_up`
7. `ramp_time_down`
8. `carrying_capacity`
9. `on_treat_threshold`
10. `off_treat_threshold`
11. `min_treat`

Values:

| Parameter | `original_saved` | `new_original_seed_best` | `new_provided_seed_best` |
|---|---:|---:|---:|
| `growth_rate_sus` | 0.07379598 | 0.07114355 | 0.07950108 |
| `growth_rate_res` | 0.24973997 | 0.18167295 | 0.23790300 |
| `competition` | 13.20513944 | 15.66310810 | 7.14936706 |
| `random_death` | 0.00001649 | 0.00006409 | 0.00002308 |
| `death_treat_sus` | 0.06923471 | 0.06670238 | 0.07297282 |
| `ramp_time_up` | 19.72000975 | 20.03425503 | 35.96000692 |
| `ramp_time_down` | 3.83497901 | 4.78963484 | 3.64954247 |
| `carrying_capacity` | 10.94968342 | 13.09296543 | 8.53194426 |
| `on_treat_threshold` | 4.63742697 | 3.86419257 | 4.45749596 |
| `off_treat_threshold` | 7.81793413 | 7.70264755 | 0.10500395 |
| `min_treat` | 0.08366815 | 0.07280188 | 0.10432430 |
| `new_env` score | 0.10378561 | 0.10492073 | 0.09442546 |

## Representative Trajectory Plot Status

There were no saved April 27 trajectory visualizations in the evaluation pipeline itself.

A new helper was added and run to generate a three-way comparison:

- observed fitted-data trajectories
- `original_saved`
- `new_original_seed_best`
- `new_provided_seed_best`

The default representative experiments selected were:

- `Plate 4/Results/Results_G418_a2a4_B6_NT-B6.txt`
- `Plate 3/Results/Results_G418_a2a4_B2_CT-B2.txt`
- `Plate 5/Results/Results_G418_a2a4_B3_Pulse-B3.txt`
- `Plate 2/Results/Results_G418_a2a4_C1_AT100-C1.txt`

In the generated selected-experiment metrics, `new_provided_seed_best` was best on all four representatives.

## Recommendation For Next Chat

If continuing this work in a fresh chat, start from:

1. this note
2. `logs/kpp/env_score_matrix_20260427_095440/score_matrix.csv`
3. `logs/kpp/trajectory_comparison_20260427/lv_threeway_new_env_comparison.png`

Then decide whether the paper should prioritize:

- best `new_env` objective (`new_provided_seed_best`)
- or closest reproduction of `original_saved` (`new_original_seed_best`)

