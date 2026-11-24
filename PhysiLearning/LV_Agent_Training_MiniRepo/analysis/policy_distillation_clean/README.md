# Policy distillation, from scratch

A minimal, step-by-step rebuild of `../policy_distillation/`.

**Goal:** explain the trained PPO agent's treatment decisions with a small
decision tree, and make the plot "P(treat) vs. current burden" with the two
thresholds (start ≈ 1.5, continue ≈ 2.3).

**Data:** the decisions the agent actually made while controlling the DPD
(in-silico) tumour simulations (step 2). As an alternative, step 2b generates
decisions in the LV simulator the agent was trained in, as the original analysis
did. Both give a table with the same columns.

## The idea in four sentences

1. The PPO agent is a black box: 8 numbers go in, a probability of treating comes out.
2. For every decision the agent made in the DPD runs, we take the 8 numbers it
   saw and ask it again for its P(treat) in that situation.
3. We fit a decision tree that predicts P(treat) from the 8 numbers. A tree is a
   short list of yes/no questions such as "is the burden above 1.5?", so a person
   can read it.
4. The tree shows that two of the 8 numbers matter most, the current burden and
   the last action. Those two numbers give the two-threshold rule:
   * last action was OFF: start treating when burden ≥ *start* (≈ 1.5)
   * last action was ON: keep treating only while burden > *continue* (≈ 2.3)

## What the agent sees

Burden = cell count ÷ cell count at day 0, so 1.0 means "same size as at the
start". The agent decides on Mondays, Wednesdays and Fridays (days 0, 2, 4, 7,
9, 11, …) and sees the last 4 decision points, oldest first:

| position | name | meaning |
|---|---|---|
| 0 | `burden_t-3` | burden 3 decisions ago |
| 1 | `previous_action_t-3` | action that was running before that measurement |
| 2 | `burden_t-2` | |
| 3 | `previous_action_t-2` | |
| 4 | `burden_t-1` | |
| 5 | `previous_action_t-1` | |
| 6 | `burden_t` | **current burden** |
| 7 | `previous_action_t` | **action that has been running since the last decision** (0 = OFF, 1 = ON) |

`previous_action_t` is the action that is already running. It is *not* the
decision the agent is about to make. At the start of a run there is no history
yet, so the older slots are filled with zeros.

## The data

* **Location:** `/home/mmilas/Documents/cancer/FINAL/data-DB_Agent/outputs/run_*/simulation_*.h5`
  (46 runs, one snapshot per simulated day, 14 GB; outside this repository).
* **Per snapshot** the controller saves `therapy_switch` (ON/OFF), `cells_history`
  (last 4 burdens) and `actions_history` (last 4 actions). We only read these
  small fields. On a decision day, `therapy_switch` already holds that day's
  decision, but the histories are still those from before it (see step 2).
* **Controller code:** `puls_dpd_tissue/puls_dpd_tissue/model.py`, `set_therapy_switch`.
  It calls the agent with `deterministic=True`, i.e. treat if P(treat) > 0.5.
* **Agent:** `puls_dpd_tissue/experiment_fitting/RL-agent/2025_10_01_lv_dont_treat_1_load_best_reward.zip`,
  which has the same weights as `Agent_used_for_experiment/` (the experimental
  agent).
* **Size:** 2234 decisions from 45 runs (9–66 per run, median 51). The agent
  treated in 31% of them. Current burden ranges from 1.0 to 2.46.
* **Restarted runs:** runs 6, 9, 24, 30 and 42 were resumed from a different
  simulation attempt partway through. Step 2 skips the one decision at each such
  seam in runs 9, 24, 30 and 42 (in run 6 the seam is not on a decision day).

## Steps

Each step is one short script. Each one runs on its own and prints or saves its
result. We write them one at a time.

| step | script | what you learn | output |
|---|---|---|---|
| 1 | `01_ask_the_agent.py` | load the agent, hand it one observation, read off P(treat) | printout |
| 2 | `02_load_dpd_decisions.py` | read the logged histories, build the agent's 8-number input for every decision, and check that the agent reproduces the logged decisions | `data/dpd_decisions.csv` |
| 2b | `02b_make_lv_decisions.py` | alternative to step 2: let the agent control the LV simulator it was trained in (`../../src/physilearning/envs/lv.py`, reused unchanged). LV has no randomness, so some actions are flipped (probability 0, 0.05, 0.15 or 0.30 per episode) to reach situations the agent wouldn't create itself. By default every episode starts from the config's initial condition (checked against the original analysis' no-flip episode); with `vary-start`, episode k starts from the resistant fraction of DPD run k (from step 2c) | `data/lv_decisions.csv`, `data/lv_ic_decisions.csv` |
| 2c | `02c_dpd_initial_resistance.py` | count the resistant cells on day 0 of every DPD run (except the artificial runs 0–9) and compare the starting resistant fraction with the LV runs (0.0005) and the original analysis' robustness set (0.0001–0.003). Run before `02b ... vary-start` | `figures/02c_dpd_initial_resistance.png`, `data/dpd_initial_resistance.csv` |
| 3 | `03_policy_portrait.py` | plot P(treat) vs. burden, coloured by last action | the target plot, still without thresholds |
| 3b | `03b_history_groups.py` | split that plot by the last four actions: one panel per exact history, then six groups (`0*00`, `1*00`, `0*10`, `1*10`, `0*01`, `1001 + 0011`) in one plot. Each group is one S-curve with its own threshold | `figures/03b_exact_histories.png`, `figures/03b_history_groups.png` |
| 4 | `04_tree_by_hand.py` | how a tree chooses its first question: try every input and every threshold, keep the one with the smallest error (MSE). Done by hand, then checked against scikit-learn | `figures/04_one_question.png` |
| 4b | `04b_second_question.py` | grow the tree by one level: run step 4's search again inside each of the two groups. Each group gets its own question, and that is where several thresholds come from. Checked against scikit-learn | `figures/04b_second_question.png` |
| 5 | `05_fit_tree.py` | fit a 3-question tree (depth 2) with scikit-learn twice: every decision counts once, and every treatment history counts equally (weight 1 / number of decisions with that history). Read each tree as a rule and count how often it decides like the agent | `figures/05_tree_weights.png` |
| 5b | `05b_more_questions.py` | keep the first question (previous action) and give each side one more question (5 in total), with every history counting equally. Fitted once to P(treat) and once to the agent's 0/1 decision, which can find rare cases such as continuing treatment. Afterwards, questions whose two answers give the same decision are merged | `figures/05b_more_questions.png` |
| 6 | `06_compare_options.py` | compare 12 options: training data (DPD / LV flips / LV flips + DPD starts) × weighting (every decision once / every history equal) × target (P / 0/1 decision). Same tree shape as 5b for all; scored on DPD runs 10–49 the tree wasn't fitted to | `figures/06_compare_options.png`, one tree figure per option in `figures/06_options/` |
| 7 | `07_two_thresholds.py` | draw the chosen tree's start and continue thresholds on the plot from step 3 | final figure |

`common.py` will hold the few functions that several steps share. We add to it
as we go.

**Possible later additions:** the wet-lab agent wells (plates 6 and 7) as a
second, independent test set; the paper-candidate agent for comparison.

## Setup

A separate conda environment called `distill` (Python 3.10, same package versions
as the original analysis):

```bash
conda create -n distill python=3.10
conda activate distill
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then run any step from anywhere, for example:

```bash
conda activate distill
python analysis/policy_distillation_clean/01_ask_the_agent.py
```
