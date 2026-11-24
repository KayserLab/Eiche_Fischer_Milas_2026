"""Step 2: turn the DPD snapshots into a table with one row per agent decision.

For every decision the agent made in the DPD runs, we rebuild the 8 numbers it
saw, ask it again for P(treat) (as in step 1), and check that "treat if
P(treat) > 0.5" gives the decision that was logged.

Output: data/dpd_decisions.csv

Run from anywhere, in the `distill` environment:
    python analysis/policy_distillation_clean/02_load_dpd_decisions.py
"""
from pathlib import Path
import argparse
import sys

import h5py
import pandas as pd

from common import AGENT_PATH, DATA_FILE, FEATURES, load_agent, p_treat

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.data_paths import SIMULATION_DATABASE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dpd-dir", type=Path, default=SIMULATION_DATABASE / "data-DB_Agent" / "outputs")
parser.add_argument("--agent", type=Path, default=AGENT_PATH)
parser.add_argument("--output", type=Path, default=DATA_FILE)
args = parser.parse_args()
DPD_DIR = args.dpd_dir

# The DPD controller decides on Mondays, Wednesdays and Fridays:
# days 0, 2, 4, 7, 9, 11, 14, ...  i.e. day % 7 is 0, 2 or 4.
DECISION_WEEKDAYS = (0, 2, 4)


def read_snapshot(path):
    """Read the day, the treatment switch and the two history lists from one snapshot file."""
    with h5py.File(path, "r") as f:
        data = f["data"]  # written with the `hickle` package, which keeps the quotes in key names
        day = round(float(data['"current_time"'][()]))
        switch = int(data['"therapy_switch"'][()])     # treatment ON (1) or OFF (0)
        cells = list(data['"cells_history"'][()])      # last 4 burdens (cell count / initial count)
        actions = list(data['"actions_history"'][()])  # last 4 actions (0 = OFF, 1 = ON)
    return day, switch, cells, actions


def decisions_of_run(run_dir):
    """Return (rows, n_skipped): one row per decision in one DPD run.

    On a decision day the DPD code (main.py, record_simulation_state_and_switch_therapy)
    copies the histories into the snapshot, then lets the agent decide, then
    overwrites only `therapy_switch` with the decision, and then writes the file.
    So for a decision on day d:
        snapshot of day d:    therapy_switch = today's decision,
                              histories      = as left by the previous decision
        snapshot of day d+1:  histories now include today's burden and decision
    The agent saw (puls_dpd_tissue/model.py, set_therapy_switch):
        burdens: the day d+1 cell history    (it contains today's burden)
        actions: the day d action history    (today's decision isn't made yet)
    """
    snapshots = {}
    for path in sorted(run_dir.glob("simulation_*.h5")):
        day, switch, cells, actions = read_snapshot(path)
        snapshots[day] = (switch, cells, actions)

    rows, n_skipped = [], 0
    for day in sorted(snapshots):
        if day % 7 not in DECISION_WEEKDAYS or day + 1 not in snapshots:
            continue
        decision, cells_before, actions_before = snapshots[day]
        _, cells_after, actions_after = snapshots[day + 1]

        # A decision shifts both histories by one place: the oldest entry drops out
        # and a new one is added at the end. If that is not what we see, the two
        # snapshots don't belong together, so skip. This happens where a run was
        # resumed from a different simulation attempt (runs 9, 24, 30, 42).
        if cells_after[:3] != cells_before[1:] or actions_after[:3] != actions_before[1:]:
            n_skipped += 1
            continue

        # Interleave into the agent's order: burden, action, burden, action, ...
        situation = []
        for burden, action in zip(cells_after, actions_before):
            situation += [burden, action]

        rows.append({"run": run_dir.name, "day": day,
                     **dict(zip(FEATURES, situation)),
                     "logged_action": decision})
    return rows, n_skipped


# 1. Read all runs -------------------------------------------------------------
run_dirs = sorted(DPD_DIR.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
rows, n_skipped = [], 0
for run_dir in run_dirs:
    run_rows, run_skipped = decisions_of_run(run_dir)
    rows += run_rows
    n_skipped += run_skipped
    if run_skipped:
        print(f"{run_dir.name}: skipped {run_skipped} decision(s) whose snapshots don't fit together")
table = pd.DataFrame(rows)
if table.empty:
    raise RuntimeError(f"No controller decisions found under {DPD_DIR}")

# 2. Ask the agent again, and check it reproduces the logged decisions ----------
agent = load_agent(args.agent)
table["p_treat"] = p_treat(agent, table[FEATURES])
agent_action = (table["p_treat"] > 0.5).astype(int)
agrees = agent_action == table["logged_action"]
print(f"\nThe agent reproduces {agrees.sum()} of {len(table)} logged decisions ({agrees.mean():.2%}).")
if not agrees.all():
    print(table[~agrees])

# 3. Summary and save ------------------------------------------------------------
per_run = table.groupby("run").size()
print(f"\n{len(table)} decisions from {len(per_run)} of {len(run_dirs)} runs "
      f"({per_run.min()}-{per_run.max()} per run, median {per_run.median():.0f}); "
      f"{n_skipped} skipped.")
print(f"The agent treated in {table['logged_action'].mean():.0%} of them.")
print(f"Current burden ranges from {table['burden_t'].min():.2f} to {table['burden_t'].max():.2f}.")
print(f"Previous action OFF: {(table['previous_action_t'] == 0).sum()} rows, "
      f"ON: {(table['previous_action_t'] == 1).sum()} rows.\n")
print(table.head(8).round(3).to_string(index=False))

args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
print(f"\nSaved to {args.output}")
