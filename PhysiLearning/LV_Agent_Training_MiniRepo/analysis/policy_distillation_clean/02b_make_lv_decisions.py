"""Step 2b (alternative to step 2): the agent's decisions in the LV simulator.

Instead of reading the DPD runs, we let the agent control the LV (Lotka-Volterra)
tumour model it was trained in, and record every situation it sees.

The LV model has no randomness: if the agent always does what it wants, every
episode is exactly the same trajectory. So, as in the original analysis
(../policy_distillation/), we sometimes flip the agent's action before it is
applied (with probability epsilon). This pushes the tumour into situations the
agent would not create itself. The label of every situation is still the agent's
own P(treat) and decision.

Two versions:
  (default)    every episode starts from the config's resistant fraction (0.0005)
               -> data/lv_decisions.csv
  vary-start   episode k starts from the resistant fraction of DPD run k (cycling
               through the 35 runs of step 2c), same total burden
               -> data/lv_ic_decisions.csv
Same columns as the DPD table from step 2, plus epsilon, flipped, executed_action
and start_resistant_fraction.

Run from anywhere, in the `distill` environment (about a minute each):
    python analysis/policy_distillation_clean/02b_make_lv_decisions.py
    python analysis/policy_distillation_clean/02b_make_lv_decisions.py vary-start
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DPD_START_FILE, FEATURES, LV_DATA_FILE, LV_IC_DATA_FILE, load_agent, p_treat

REPO = Path(__file__).resolve().parents[2]  # LV_Agent_Training_MiniRepo/
sys.path.insert(0, str(REPO / "src"))
from physilearning.envs.lv import LvEnv  # noqa: E402  (the LV simulator, reused unchanged)

CONFIG = REPO / "configs" / "2025_10_lv_real_mela_1_load.yaml"  # the agent's training setup
EPSILONS = (0.0, 0.05, 0.15, 0.30)  # flip probability; episode k uses EPSILONS[k % 4]
N_EPISODES = 250
SEED = 42
VARY_START = sys.argv[1:] == ["vary-start"]

# The original analysis' LV data for the same agent, used once as a check.
OLD_DATA = REPO / "analysis/policy_distillation/experimental_agent/observations_and_teacher_labels.csv.gz"


def run_episode(agent, epsilon, rng, name, resistant_fraction=None):
    """Run one LV episode with the agent in control. Returns one row per decision."""
    env = LvEnv.from_yaml(str(CONFIG))  # a fresh simulator for every episode
    if resistant_fraction is not None:  # same total burden, different share of resistant cells
        total = env.initial_wt + env.initial_mut
        env.initial_wt, env.initial_mut = (1 - resistant_fraction) * total, resistant_fraction * total
    obs, _ = env.reset()                # obs = [burden, previous action] = [1.0, 0]

    # The agent sees the last 4 (burden, action) pairs, oldest first. Before the
    # start there is no history, so those slots are zeros (as in training).
    history = [[0.0, 0], [0.0, 0], [0.0, 0], [float(obs[0]), int(obs[1])]]
    rows = []
    while True:
        situation = [x for pair in history for x in pair]  # flatten to 8 numbers
        p = p_treat(agent, situation)[0]
        decision = int(p > 0.5)                 # what the agent wants
        flipped = bool(rng.random() < epsilon)  # sometimes do the opposite
        action = 1 - decision if flipped else decision

        rows.append({"run": name,
                     "day": env.day - 1,  # LV counts days from 1; shift so the first decision is day 0, as in DPD
                     **dict(zip(FEATURES, situation)),
                     "logged_action": decision, "p_treat": p,
                     "epsilon": epsilon, "flipped": flipped, "executed_action": action,
                     "start_resistant_fraction": env.initial_mut / (env.initial_wt + env.initial_mut)})

        obs, _, progressed, out_of_time, _ = env.step(action)  # simulate 2 or 3 days
        # New measurement: drop the oldest pair, add (new burden, action just applied).
        history = history[1:] + [[float(obs[0]), action]]
        if progressed or out_of_time:  # burden reached 3.0, or the time limit
            return rows


# 1. Run the episodes -----------------------------------------------------------
agent = load_agent()
starts = pd.read_csv(DPD_START_FILE)["fraction"].to_numpy() if VARY_START else [None]
kept, seen = [], set()
for k in range(N_EPISODES):
    rng = np.random.default_rng([SEED, k])  # each episode has its own repeatable coin flips
    start = starts[k % len(starts)]
    rows = run_episode(agent, EPSILONS[k % len(EPSILONS)], rng, f"lv_{k:03d}", start)

    # Same start and same actions -> exactly the same trajectory (no randomness in LV). Keep one copy.
    key = (start, "".join(str(r["executed_action"]) for r in rows))
    if key not in seen:
        seen.add(key)
        kept += rows
table = pd.DataFrame(kept)

# 2. Check against the original analysis (default version only) ---------------------
# Without flips, our episode must be exactly the original's (same agent, same config).
if not VARY_START:
    old = pd.read_csv(OLD_DATA)
    old = old[(old["dataset"] == "primary") & (old["epsilon"] == 0)]
    ours = table[table["epsilon"] == 0]
    same = (len(ours) == len(old)
            and np.allclose(ours[FEATURES].to_numpy(np.float32), old[FEATURES].to_numpy(np.float32), atol=1e-6)
            and np.allclose(ours["p_treat"], old["p_on"], atol=1e-6)
            and (ours["executed_action"].to_numpy() == old["executed_action"].to_numpy()).all())
    print(f"No-flip episode identical to the original analysis' one: {same}")

# 3. Save ------------------------------------------------------------------------------
out = LV_IC_DATA_FILE if VARY_START else LV_DATA_FILE
out.parent.mkdir(exist_ok=True)
table.to_csv(out, index=False)
print(f"{N_EPISODES} episodes -> {table['run'].nunique()} different trajectories, {len(table)} decisions. "
      f"Saved to {out}")
