# Decision-tree distillation of the LV PPO treatment policy

Date: 2026-10-22. Branch `policy-distillation`. Supporting analysis only: no PPO
agent was retrained, and no checkpoint, config or `reference/` file was modified.

**Method in one sentence.** Ordinary CART policy distillation: a
`DecisionTreeRegressor(max_depth=3, min_samples_leaf=0.02)` regressed onto the
frozen PPO actor's P(treatment ON). The inputs are the eight raw stacked
observations, sampled from LV trajectories that were generated dynamically.
This is **not** VIPER (no iteration, no Q-value weighting).

## 1. Short answer

* **Tree recovers a history rule.** The depth-3 tree was not given any
  heuristic. For both checkpoints it chose the current burden `B_t` as its root
  split (100% of 100 trajectory-bootstrap refits) and the latest previous action
  `a_t` as the next split on the high-burden branch (100% of refits).
* **Treatment history *raises* the burden needed to treat.** After an OFF
  interval, treatment starts at about `B_t ≥ 1.5` (× initial burden). After an
  ON interval, it continues only if `B_t > 2.2–2.3`. Otherwise the policy
  switches OFF.
* **Dominant behavior is short pulses.** Mostly a single ON interval, then about
  4 OFF intervals, timed by burden. The policy does not favor continuing
  treatment. This matches the sign in `Agent_Behaviour_Analysis/policy_behavior_summary.txt`
  (higher P(ON) after OFF histories).
* **Two thresholds capture most of it.** A rule using only `B_t` and `a_t`
  ("start if `B_t ≥ s` when previously OFF, continue if `B_t > c` when
  previously ON", fitted `s≈1.5`, `c≈2.3`) reproduces 90–96% of held-out PPO
  decisions. It is the only student that never progressed when deployed in
  closed loop (§5).
* **History is needed beyond burden, with a different effect size per checkpoint.**
  * For the experimental-agent checkpoint, the gain is clear: burden-only
    agreement 0.88 vs 0.93 with history, and switch-ON recall 0.43 vs 0.75.
  * For the paper-candidate checkpoint, the gain is small but significant: MAE
    0.100 → 0.082, CI of the difference [0.011, 0.024].
* **Closed-loop deployment needs a caveat.** Per-decision fidelity is high
  (≈93%). But in this deterministic LV model a single deviating decision often
  leads to progression, and PPO itself is equally fragile. The depth-3 tree is
  therefore **not** a drop-in replacement controller. See §5.

## 2. Checkpoints and provenance (must be resolved before publication)

The two candidate checkpoints are **different policies**.

| label | path | policy.pth SHA-256 | PPO hyper-parameters (from saved `data`) |
|---|---|---|---|
| `paper_candidate` | `LV_Agent_Training_MiniRepo/reference/saved_models/2025_10_lv_real_mela_1_load_best_reward.zip` (zip SHA-256 `172593d7…dd60`) | `7c29673e1d9b…` | n_steps 64, batch 32, n_epochs 4; 110 720 timesteps |
| `experimental_agent` | `Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward/` | `b136b65433d5…` | n_steps 128, batch 64, n_epochs 8; 119 936 timesteps |

* Both share the same architecture (pi/vf [64, 64]), the same observation space
  (8-dim, bounds [0, 3]) and the same ent_coef, learning rate and γ.
* The experimental agent's training config is not in the repository, so both
  were rolled out in the mini-repo stage-2 config
  `configs/2025_10_lv_real_mela_1_load.yaml` (SHA-256 `66672a53…84de7`).
* **Output agreement** (`teacher_comparison.json`): on about 5 700 unique
  observations visited under either policy, hard actions agree on 95.6–95.9%.
  Mean |ΔP(ON)| is 0.043–0.046. The experimental agent treats slightly more
  often (ON fraction 0.22–0.23 vs 0.19–0.21).
* Qualitatively the two policies are the same rule (§3). Quantitatively they
  are not interchangeable.
* **Open item:** confirm which checkpoint generated the tested experimental
  schedules. The folder name suggests `experimental_agent`, but no provenance
  file states it.

## 3. Learned trees (depth 3, fit on the training split)

Leaf value = mean teacher P(ON) in the leaf. Hard decision: P(ON) > 0.5 is ON.

**`paper_candidate`**
```
B_t <= 1.483
|  B_t-1 <= 0.029          -> 0.728   (first decision of an episode; zero-padded history)
|  B_t-1 >  0.029
|  |  B_t <= 1.318         -> 0.022
|  |  B_t >  1.318         -> 0.155
B_t >  1.483
|  a_t = OFF
|  |  B_t <= 1.870         -> 0.580   (ON)
|  |  B_t >  1.870         -> 0.900   (ON)
|  a_t = ON
|  |  B_t <= 2.196         -> 0.096   (switch OFF)
|  |  B_t >  2.196         -> 0.642   (continue ON)
```

**`experimental_agent`**
```
B_t <= 1.316
|  B_t-1 <= 0.046          -> 0.119   (first decision of an episode)
|  B_t-1 >  0.046
|  |  B_t <= 1.189         -> 0.006
|  |  B_t >  1.189         -> 0.047
B_t >  1.316
|  a_t = OFF
|  |  B_t <= 1.528         -> 0.388   (OFF)
|  |  B_t >  1.528         -> 0.817   (ON)
|  a_t = ON
|  |  B_t <= 2.251         -> 0.048   (switch OFF)
|  |  B_t >  2.251         -> 0.672   (continue ON)
```

**Reading the trees.**
* Both reduce to "start at `B_t ≈ 1.5` when previously OFF; continue only above
  `B_t ≈ 2.2` when previously ON".
* The `B_t-1 ≈ 0` split is an artifact. It isolates the first decision of every
  episode, whose stacked history is zero-padded by `VecFrameStack`.
* Re-running with the first three (padded) decisions excluded
  (`*_no_padded`): the tree spends that split on `B_t`/`B_t-k` instead. Root
  and second-level structure are unchanged, and test agreement is 0.925
  (paper_candidate) and 0.953 (experimental_agent).
* No split encodes an explicit growth rate (`B_t − B_t-1`). An axis-aligned
  depth-3 tree cannot represent a difference cheaply, so this is **not**
  evidence that the policy ignores growth. Deeper trees do pick up older burden
  frames, e.g. `B_t-3`, `B_t-2` at depth 4.

**Comparison with `effective` in `evaluate.fixed_at`.**
* That heuristic starts when `B ≥ high·B0` (previously OFF) and continues while
  `B > low·B0` (previously ON).
* The PPO-fitted values correspond to `high ≈ 1.5`, `low ≈ 2.3`, i.e. **low > high**.
* The continuation threshold is therefore higher than the start threshold:
  history discourages continuation. This is the same ordering as the
  commented-out `effective_high_1.10_low_1.20` example in the configs, but the
  PPO's thresholds are larger and much further apart.

## 4. Fidelity on held-out trajectories

**Protocol**
* **Primary data:** 250 episodes from the config initial condition (resistant
  fraction 5e-4). The executed action is flipped with ε ∈ {0, 0.05, 0.15, 0.30}
  (cycled by episode). Every visited observation is labeled with the unchanged
  PPO P(ON).
* **De-duplication:** the LV model is deterministic, so all ε = 0 episodes are
  identical. Identical trajectories (same executed-action sequence) were
  collapsed to one copy: 76 (paper_candidate) and 72 (experimental_agent)
  duplicate episodes removed. This leaves 174 / 178 unique trajectories.
* **Split:** by trajectory into train/validation/test (60/20/20%). There is no
  overlap of trajectories between splits; this is asserted.
* **Robustness test (never used for fitting):** 100 episodes with the same ε
  schedule but a genuinely different initial resistant fraction, drawn
  log-uniformly from [1e-4, 3e-3]. All 100 trajectories are unique.
* **Depth:** fixed a priori at 3. The validation split was used only to inspect
  depths 1–6; it selects depth 4 among 2–4. Depth-4 results are reported as a
  secondary row.
* **Uncertainty:** 95% intervals come from a paired bootstrap over test
  trajectories (2 000 resamples; `uncertainty.json`).

| | data | rows / unique obs | teacher ON frac | teacher switch frac |
|---|---|---|---|---|
| paper_candidate | test | 927 / 578 (35 traj) | 0.213 | 0.407 |
| | robustness | 3552 / 3418 (100 traj) | 0.224 | 0.394 |
| experimental_agent | test | 1026 / 685 (36 traj) | 0.207 | 0.412 |
| | robustness | 3506 / 3372 (100 traj) | 0.215 | 0.409 |

A constant "always OFF" student already scores about 0.78–0.79 agreement,
so balanced accuracy and switching F1 are the informative metrics.

**paper_candidate** (test / robustness test):

| student | P(ON) MAE | hard agreement | balanced acc. | switching F1 | switch-ON recall | switch-OFF recall |
|---|---|---|---|---|---|---|
| tree, all 8 inputs, depth 3 | 0.082 / 0.076 | 0.927 / 0.931 | 0.926 / 0.937 | 0.911 / 0.914 | 0.92 / 0.94 | 0.92 / 0.93 |
| tree, burden only (4), depth 3 | 0.100 / 0.099 | 0.916 / 0.917 | 0.904 / 0.911 | 0.893 / 0.893 | 0.86 / 0.88 | 0.86 / 0.88 |
| tree, actions only (4), depth 3 | 0.180 / 0.167 | 0.787 / 0.776 | 0.500 / 0.500 | 0.683 / 0.654 | 0 / 0 | 1 / 1 |
| two-threshold rule (`B_t`, `a_t`) | – | 0.902 / 0.924 | 0.821 / 0.872 | 0.873 / 0.901 | 0.65 / 0.73 | 0.97 / 1.00 |
| tree, all inputs, depth 4 (val.-selected) | 0.062 / 0.057 | 0.944 / 0.945 | 0.885 / 0.885 | 0.930 / 0.929 | 0.82 / 0.79 | 0.98 / 1.00 |
| always OFF | – | 0.787 / 0.776 | 0.500 | 0.683 / 0.654 | 0 | 1 |

**experimental_agent** (test / robustness test):

| student | P(ON) MAE | hard agreement | balanced acc. | switching F1 | switch-ON recall | switch-OFF recall |
|---|---|---|---|---|---|---|
| tree, all 8 inputs, depth 3 | 0.078 / 0.085 | 0.928 / 0.930 | 0.873 / 0.880 | 0.908 / 0.910 | 0.75 / 0.77 | 0.95 / 0.96 |
| tree, burden only (4), depth 3 | 0.120 / 0.129 | 0.880 / 0.873 | 0.738 / 0.728 | 0.830 / 0.816 | 0.43 / 0.42 | 0.93 / 0.94 |
| tree, actions only (4), depth 3 | 0.205 / 0.193 | 0.793 / 0.785 | 0.500 / 0.500 | 0.690 / 0.668 | 0 / 0 | 1 / 1 |
| two-threshold rule (`B_t`, `a_t`) | – | 0.938 / 0.942 | 0.893 / 0.905 | 0.922 / 0.928 | 0.79 / 0.83 | 0.98 / 0.99 |
| tree, all inputs, depth 4 (val.-selected) | 0.048 / 0.045 | 0.962 / 0.958 | 0.976 / 0.973 | 0.955 / 0.951 | 1.00 / 1.00 | 0.95 / 0.96 |
| always OFF | – | 0.793 / 0.785 | 0.500 | 0.690 / 0.668 | 0 | 1 |

**Value of treatment history** (paired bootstrap; all-inputs vs burden-only
tree, depth 3):

| | ΔMAE (burden-only − all), test | robustness | Δagreement (all − burden-only), test | robustness |
|---|---|---|---|---|
| paper_candidate | 0.018 [0.011, 0.024] | 0.023 [0.020, 0.026] | 0.011 [−0.001, 0.022] | 0.014 [0.007, 0.021] |
| experimental_agent | 0.043 [0.036, 0.049] | 0.045 [0.042, 0.048] | 0.048 [0.035, 0.061] | 0.057 [0.051, 0.064] |

* On the robustness set, the on-policy subset (ε = 0: 25 greedy trajectories
  from distinct initial conditions) gives the same agreement as the
  exploratory subsets. Depth-3 tree: 0.933 (paper_candidate) and 0.931
  (experimental_agent), vs 0.92–0.95 for ε > 0; see `metrics.json` →
  `evaluation.robustness_test.*.epsilon_*`. The primary split holds at most one
  greedy trajectory, because all ε = 0 episodes from the config initial
  condition are identical.
* Fidelity on the robustness test (new initial conditions) matches the
  in-distribution test.

## 5. Closed-loop deployment (stringent; interpret with care)

**Setup.** Each student was run as the sole controller (no exploration) from 20
initial conditions: the config initial condition plus 19 log-uniform resistant
fractions in [1e-4, 3e-3]. Episodes are capped at 60 decisions by `max_time`.

| controller | paper_candidate: progressed / mean decisions | experimental_agent: progressed / mean decisions |
|---|---|---|
| PPO (greedy) | 0% / 60.0 | 20% / 59.1 |
| tree all inputs, depth 3 | 100% / 46.0 | 0% / 60.0 |
| tree burden only, depth 3 | 100% / 43.6 | 100% / 4.0 |
| tree all inputs, depth 4 | 100% / 25.0 | 100% / 46.8 |
| two-threshold rule | **0% / 60.0** | **0% / 60.0** |
| tree all inputs, depth 3, padded steps excluded | 0% / 60.0 | 15% / 58.8 |

**Why these numbers swing so much.**
* PPO keeps the tumor in a tight pulse cycle that peaks at `B ≈ 2.2–2.8`, just
  below the progression threshold `B = 3`.
* Flipping a random 5% of PPO's own actions leads to progression within 60
  decisions in 76% (paper_candidate) and 94% (experimental_agent) of episodes.
* A single flipped decision at the wrong time also often leads to progression
  (scratch probe, not saved).
* Closed-loop survival is therefore dominated by whether a few
  near-threshold decisions are reproduced exactly. It swings between 0% and
  100% under small changes to the student, e.g. depth 3 → 4, or
  including/excluding the padded first steps.

**Conclusion.** Closed-loop survival is *not* a smooth measure of how well a
student imitates the policy. Report per-decision fidelity as the primary
result. The two-threshold rule's robust closed-loop survival is noteworthy but
should be presented as an observation, not as proof of equivalence.

## 6. Bootstrap ensemble (consistency across separately trained agents)

**Scope.**
* All 26 `*_s2_best_reward.zip` agents from
  `Bootstrapping/outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/agt_20260605_102549`.
* Each agent was distilled in **its own** training environment (its
  `cfg/<code>_s2.yaml`), with a lighter protocol: 120 primary episodes, 40
  robustness episodes, 30 stability resamples and 10 closed-loop initial
  conditions.
* Groups are kept separate.
* **Burden thresholds cannot be compared across `bootstrap_lv` agents:** each
  has different LV dynamics, so the same normalized burden means different
  things.
* Full table: `bootstrap_ensemble/ensemble_summary.csv`.

| group | n | root split on `B_t` | tree uses any previous-action input | `a_t` at level 1 | fitted continue threshold > start threshold | test agreement, depth 3, median [range] | burden-only − all MAE, robustness, median |
|---|---|---|---|---|---|---|---|
| reference teachers | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 0.927 [0.927–0.928] | 0.034 |
| `methods_repeat` (same LV params, 5 seeds) | 5 | 5/5 (thresholds 1.34–1.50) | 5/5 | 4/5 | 5/5 (start 1.34–1.53, continue 2.20–2.68) | 0.936 [0.901–0.960] | 0.040 |
| `methods_control` | 1 | 0/1 (root = padded-history split) | 1/1 | 1/1 | 1/1 (1.89 / 2.42) | 0.916 | 0.000 |
| `bootstrap_lv` (refit LV params) | 20 | 15/20 (3: padded-history split; 2: older burden frame) | 19/20 | 8/20 | 18/20 | 0.915 [0.844–0.989] | 0.014 (>0.01 in 10/20; <0 in 2/20) |

The constant always-OFF baseline has median agreement 0.77–0.79 in every group.

**Reading.**
* **Agents trained on the paper's LV parameters reproduce the teachers' rule**
  (5 `methods_repeat` seeds): root on `B_t ≈ 1.3–1.5`, then `a_t`, with a
  start threshold of about 1.5 and a higher continue threshold of 2.2–2.7.
  Fidelity is the same, and history helps by a similar amount.
* **The bootstrap-LV agents are more heterogeneous.**
  * Most still treat on high current burden and almost all use treatment
    history in some form. 18/20 have a continue threshold above the start
    threshold, i.e. no continuation bias.
  * But several encode the post-pulse refractory period through *older burden
    frames* (e.g. `B_t-2`, `B_t-3` high ⇒ do not treat) or older actions
    (`a_t-1`, `a_t-2`), rather than through `a_t`.
  * For about half of them, the added value of the action history over a
    burden-only tree is negligible. A burden-only tree can stand in for history
    when the burden trajectory itself reveals a recent pulse.
* **The `methods_control` agent** (same LV params as the paper, current-config
  control) treats the first two decisions unconditionally. Afterwards it uses
  `B_t` together with `B_t-3`, not `a_t`. Its burden-only tree is as good as the
  full tree.

**Bottom line.**
* **Robust across agents:** "treat when current burden is high, but not again
  right after a pulse".
* **Not universal:** that the refractory signal is read from the previous
  *action* specifically rather than from the burden history. Holds for agents
  trained on the paper's LV parameters but not for all bootstrap-LV agents.

## 7. Limitations

* **Model scope.** All trajectories come from the deterministic LV model with
  frozen parameters. No noise, patient sampling or parameter variation was
  applied, beyond the initial resistant fraction in the robustness set.
* **Experimental agent's environment.** It was evaluated in the mini-repo
  stage-2 env because its own training config is not available.
* **Correlated test rows.** Trajectories from one initial condition share long
  prefixes, so test rows are correlated even across trajectory groups. The
  robustness set (distinct initial conditions) gives the same numbers, which
  suggests this does not inflate the fidelity estimates.
* **Exploration coverage.** Exploratory flips enlarge coverage near the PPO's
  orbit only. States far from it, e.g. long ON runs at low burden, are
  sparsely represented.
* **Padded first decision.** The first decision of an episode sees a
  zero-padded history. The tree models it with a dedicated split.
* **What the shallow tree misses.** It does not capture effects of older
  history (`a_t-1…a_t-3`, `B_t-k`). Those effects exist, e.g. low P(ON) at high
  burden right after a pulse, visible in `policy_portrait.png`.

## 8. Suggested manuscript sentence (supported by the above)

> We distilled the trained PPO policy into a shallow (depth-3) regression
> decision tree using dynamically generated LV tumor trajectories. The extracted
> rules depended on current tumor burden and the previous treatment action and
> reproduced the PPO's treatment decisions on held-out trajectories with 93%
> agreement (switching F1 0.91, balanced accuracy 0.87–0.93). Including treatment
> history improved fidelity over a burden-only tree. The learned rule starts
> treatment at about 1.5× the initial burden after an OFF interval, but continues
> treatment only above about 2.2× after an ON interval, i.e. it favors short
> treatment pulses.

**Caveats for this sentence:**
* Name which checkpoint generated the experiments.
* Avoid claiming that the tree replicates the closed-loop outcome.
* For the paper-candidate checkpoint, the history gain is small (+1.1
  percentage points agreement, CI includes 0 on the in-distribution test).
* If the ensemble is mentioned, say the history dependence is consistent across
  seeds on the paper's LV parameters. Across bootstrap-refit LV parameters, the
  refractory signal is sometimes carried by the burden history instead (§6).

## 9. Files

* **Browser pages** (open locally by double-clicking):
  * `distillation_report.html`: full interactive report, both checkpoints.
  * `distillation_figures.html`: figures only, agent used for experiment only.
  * Regenerate with `python analysis/policy_distillation/web/build_pages.py`
    (template and build script in `web/`).
* `scripts/distill_policy.py`: data generation, splits, trees, baselines,
  stability, closed loop.
* `scripts/test_distill_policy.py`: smoke test for observation order,
  P(ON)-argmax vs `predict(deterministic=True)`, tie rule and grouping.
* `scripts/compare_teachers.py`, `scripts/summarize_distillation.py`,
  `scripts/distill_bootstrap_ensemble.py`.
* This folder holds the per-teacher `metrics.json`, `tree_rules.txt`,
  `decision_tree.{png,pdf}`, `closed_loop.csv`,
  `depth_selection_validation.csv`, `uncertainty.json` and
  `observations_and_teacher_labels.csv.gz`, plus `teacher_comparison.json`,
  `policy_portrait.{png,pdf}` and the ensemble summary tables.
* **Reproduce** from `PhysiLearning/LV_Agent_Training_MiniRepo/`, Python 3.10:
  ```
  python -m pip install -r requirements.txt -r requirements-analysis.txt
  python scripts/test_distill_policy.py
  python scripts/distill_policy.py --label paper_candidate
  python scripts/distill_policy.py --label experimental_agent \
      --model ../Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward
  python scripts/distill_policy.py --label paper_candidate_no_padded --exclude-padded-steps   # sensitivity
  python scripts/compare_teachers.py --a outputs/policy_distillation/paper_candidate \
      --b outputs/policy_distillation/experimental_agent
  python scripts/summarize_distillation.py outputs/policy_distillation/paper_candidate \
      outputs/policy_distillation/experimental_agent
  python scripts/distill_bootstrap_ensemble.py --workers 3 --episodes 120 \
      --robustness-episodes 40 --stability-resamples 30 --closed-loop-ics 10
  ```
* **Versions used:** Python 3.10.20, numpy 1.26.2, pandas 2.0.2, torch 2.1.2,
  stable-baselines3 2.2.1, gymnasium 0.28.1, scikit-learn 1.3.2.
