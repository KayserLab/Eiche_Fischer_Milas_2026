"""Shared helpers for the policy-distillation steps.

Everything that more than one step needs lives here, so each step script stays
short. We add to this file as we go.
"""
import sys
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.data_paths import MODELS, INTERMEDIATE_ANALYSIS_DATA, figure_plot_dir

# The agent that controlled the DPD runs. Its weights are identical to
# PhysiLearning/Agent_used_for_experiment/ (the "experimental agent").
AGENT_PATH = MODELS / 'experimental_agent'

# One row per agent decision in the DPD runs. Written by step 2, read by the later steps.
DATA_FILE = INTERMEDIATE_ANALYSIS_DATA / 'policy_decisions/dpd_decisions.csv'
# Experimental ATAgent2 decisions, written by step 2d.
EXPERIMENTAL_DATA_FILE = INTERMEDIATE_ANALYSIS_DATA / 'policy_decisions/experimental_decisions.csv'
# The same, but for decisions in the LV simulator. Written by step 2b.
LV_DATA_FILE = INTERMEDIATE_ANALYSIS_DATA / 'policy_decisions/lv_decisions.csv'
# LV again, but each episode starts from the resistant fraction of one DPD run (step 2b vary-start).
LV_IC_DATA_FILE = INTERMEDIATE_ANALYSIS_DATA / 'policy_decisions/lv_ic_decisions.csv'
# Starting resistant fraction of every (non-artificial) DPD run. Written by step 2c.
DPD_START_FILE = INTERMEDIATE_ANALYSIS_DATA / 'policy_decisions/dpd_initial_resistance.csv'
FIG_DIR = figure_plot_dir('Figure_4', 'policy_distillation', create=False)

# DPD runs 0-9 are artificial: one tissue, with the resistant clone moved around.
ARTIFICIAL_RUNS = [f"run_{i}" for i in range(10)]

# The datasets the later steps can work on. Pick one with a word on the command
# line, e.g. `python 03_policy_portrait.py lv`. Without one, DPD is used.
DATASETS = {
    # name: (table file, added to figure names, description for plot titles)
    "dpd": (DATA_FILE, "", "the DPD runs"),
    "experimental": (EXPERIMENTAL_DATA_FILE, "_experimental", "the ATAgent2 experiments"),
    "lv": (LV_DATA_FILE, "_lv", "the LV simulator (with random flips)"),
    "lv_ic": (LV_IC_DATA_FILE, "_lv_ic", "the LV simulator (flips + DPD starting fractions)"),
}


def choose_dataset(default="dpd"):
    """Return (table, figure-name suffix, description) for the dataset named on the
    command line, or for the default dataset if none is named."""
    name = sys.argv[1] if len(sys.argv) > 1 else default
    if name not in DATASETS:
        sys.exit(f"Unknown dataset {name!r}. Use one of: {', '.join(DATASETS)}")
    path, suffix, description = DATASETS[name]
    return pd.read_csv(path), suffix, description

# Colours and names for the two groups we compare in every plot.
COLORS = {0: "#2a78d6", 1: "#eb6834"}  # blue, orange (checked to be colour-blind safe)
LABELS = {0: "previous action OFF", 1: "previous action ON"}
GREY = "#898781"
INK, MUTED = "#0b0b0b", "#52514e"  # text colours: main and secondary

# The 8 numbers the agent sees, in the order it sees them (oldest first).
# The README explains each one.
FEATURES = [
    "burden_t-3", "previous_action_t-3",
    "burden_t-2", "previous_action_t-2",
    "burden_t-1", "previous_action_t-1",
    "burden_t",   "previous_action_t",
]


def load_agent(path=AGENT_PATH):
    """Load the saved PPO agent from its .zip file.

    The file also stores two settings that are only used during training (the
    learning-rate and clip-range schedules). They were saved with a newer Python
    version and can't be read here, so we replace them with dummies. We never
    train, so the dummies are never used.
    """
    training_only = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    if isinstance(path, (str, Path)) and Path(path).is_dir():
        path = Path(path)
        for name in ("data", "policy.pth", "pytorch_variables.pth"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"Incomplete unpacked SB3 agent: {path / name}")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for item in path.iterdir():
                if item.is_file():
                    archive.writestr(item.name, item.read_bytes())
        buffer.seek(0)
        path = buffer
    return PPO.load(path, device="cpu", custom_objects=training_only)


def p_treat(agent, situations):
    """Ask the agent for P(treat) in one or more situations.

    `situations` is either one list of 8 numbers, or a table with 8 columns
    (one row per situation). Returns an array with one P(treat) per row.
    """
    x = torch.as_tensor(np.asarray(situations, dtype=np.float32).reshape(-1, len(FEATURES)))
    with torch.no_grad():                                # we only ask, we never train
        hidden = agent.policy.mlp_extractor.policy_net(x)  # 8 numbers -> 64 -> 64
        scores = agent.policy.action_net(hidden)           # 64 -> 2 scores: (OFF, ON)
        probs = torch.softmax(scores, dim=1)               # 2 scores -> 2 probabilities that add up to 1
    return probs[:, 1].numpy()                             # keep P(ON) = P(treat)


def mse_of_question(x, p, threshold):
    """MSE when we ask "is x <= threshold?" and predict each group's average P(treat).
    (Steps 4 and 4b.)"""
    yes, no = p[x <= threshold], p[x > threshold]
    squared_errors = ((yes - yes.mean()) ** 2).sum() + ((no - no.mean()) ** 2).sum()
    return squared_errors / len(p)


def best_question(x, p):
    """Try every threshold for one input. Returns the best threshold and its MSE,
    plus all thresholds tried and their MSE (for figures). (Steps 4 and 4b.)

    If the input has the same value for every decision, it can't split them:
    then the threshold is None and the MSE infinite."""
    values = np.unique(x)                        # sorted, each value once
    thresholds = (values[:-1] + values[1:]) / 2  # halfway between neighbours
    if len(thresholds) == 0:
        return None, np.inf, thresholds, np.array([])
    mses = np.array([mse_of_question(x, p, t) for t in thresholds])
    best = mses.argmin()
    return thresholds[best], mses[best], thresholds, mses


def draw_tree(ax, tree, table, x0=0, y0=0, finish=True, leaf_color=None):
    """Draw a fitted scikit-learn tree (on the 8 FEATURES) as boxes. (Step 5 onwards.)

    Questions are black boxes; follow "yes" to the left. Leaves show the tree's P(treat),
    how many decisions of `table` end there, and the decision (treat if P > 0.5). Leaf
    border: blue if all its decisions came after OFF, orange if after ON, grey if mixed.

    x0, y0 shift the drawing (to put several trees on one axes); with finish=False the
    axes limits are left alone. leaf_color sets one border colour for all leaves.
    Returns the position of the first question, the number of leaves and the depth
    (number of questions on the longest path).
    """
    t = tree.tree_
    leaf_of_row = tree.apply(table[FEATURES])

    # Layout: leaves get x = x0, x0 + 1, ... from left to right; a question sits above the
    # middle of its two answers. y = y0 minus the number of questions asked so far.
    pos = {}

    def place(node, depth, next_x):
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            pos[node] = (next_x, y0 - depth)
            return next_x + 1
        next_x = place(left, depth + 1, next_x)
        next_x = place(right, depth + 1, next_x)
        pos[node] = ((pos[left][0] + pos[right][0]) / 2, y0 - depth)
        return next_x

    n_leaves = place(0, 0, x0) - x0
    box = {"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": INK, "linewidth": 1.1}
    for node, (x, y) in pos.items():
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            rows = leaf_of_row == node
            share_on = table.loc[rows, "previous_action_t"].mean()
            edge = leaf_color or (COLORS[0] if share_on == 0 else COLORS[1] if share_on == 1 else GREY)
            value = t.value[node, 0, 0]
            verdict = "treat" if value > 0.5 else "don't treat"
            ax.text(x, y, f"P = {value:.2f}\n{rows.sum()} decisions\n→ {verdict}", ha="center", va="center",
                    fontsize=9, color=INK, bbox=box | {"edgecolor": edge, "linewidth": 2.4})
            continue
        for child, answer in ((left, "yes"), (right, "no")):
            cx, cy = pos[child]
            ax.plot([x, cx], [y, cy], color=GREY, linewidth=1, zorder=0)
            ax.text((x + cx) / 2, (y + cy) / 2, answer, ha="center", va="center", fontsize=8, color=MUTED,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "linewidth": 0})
        feature = FEATURES[t.feature[node]]
        question = (f"was {feature}\nOFF?" if feature.startswith("previous_action")
                    else f"is {feature}\n≤ {t.threshold[node]:.2f}?")
        ax.text(x, y, question, ha="center", va="center", fontsize=9, color=INK, bbox=box)
    # (Depth from the drawing, not tree.get_depth(): merge_same_decisions doesn't update that.)
    depth = round(max(y0 - y for _, y in pos.values()))
    if finish:
        ax.set_xlim(x0 - 0.7, x0 + n_leaves - 0.3)
        ax.set_ylim(y0 - depth - 0.5, y0 + 0.5)
        ax.axis("off")
    return pos[0], n_leaves, depth


def merge_same_decisions(tree):
    """Turn every question whose two answers lead to the same decision (treat if
    P > 0.5) into a leaf. (Step 5b onwards.)

    Works from the bottom up, so a question can become a leaf after the questions below
    it were merged. The new leaf predicts the question's own average, which
    scikit-learn already stores for every node. No decision changes, only the tree's
    P values. Changes `tree` in place and also returns it.
    """
    t = tree.tree_
    left, right = t.children_left, t.children_right  # writing into these changes the tree

    def treats(node):
        return t.value[node, 0, 0] > 0.5

    def merge(node):
        if left[node] == -1:  # -1 = no children = a leaf
            return
        merge(left[node])
        merge(right[node])
        both_leaves = left[left[node]] == -1 and left[right[node]] == -1
        if both_leaves and treats(left[node]) == treats(right[node]):
            left[node] = right[node] = -1

    merge(0)
    return tree


def count_questions(tree):
    """Number of questions in the tree (counting only nodes that can still be reached)."""
    t = tree.tree_

    def count(node):
        if t.children_left[node] == -1:
            return 0
        return 1 + count(t.children_left[node]) + count(t.children_right[node])

    return count(0)


def histories(table):
    """Each decision's last four actions as text, oldest first, e.g. "0100"."""
    return table[FEATURES[1::2]].astype(int).astype(str).agg("".join, axis=1)


def fit_rule_tree(train, target="p_treat", equal_histories=True, leaves_per_side=3):
    """The tree shape of steps 5b and 6: previous action first (by hand), then one
    scikit-learn tree per side with `leaves_per_side` leaves (grown best first), then
    questions whose two answers give the same decision are merged.

    target: "p_treat" (the agent's P) or "logged_action" (its 0/1 decision).
    equal_histories: every history gets the same total weight (else every decision 1).
    Returns {0: tree after OFF, 1: tree after ON}."""
    from sklearn.tree import DecisionTreeRegressor

    h = histories(train)
    weights = 1 / h.map(h.value_counts()) if equal_histories else pd.Series(1.0, index=train.index)
    sides = {}
    for prev in (0, 1):
        rows = train[train["previous_action_t"] == prev]
        tree = DecisionTreeRegressor(max_leaf_nodes=leaves_per_side, random_state=0)
        tree.fit(rows[FEATURES], rows[target], sample_weight=weights[rows.index])
        sides[prev] = merge_same_decisions(tree)
    return sides


def predict_rule_tree(sides, table):
    """The tree's P(treat) (or share of "treat") for every decision in `table`."""
    predicted = np.empty(len(table))
    for prev, tree in sides.items():
        side = (table["previous_action_t"] == prev).to_numpy()
        if side.any():
            predicted[side] = tree.predict(table.loc[side, FEATURES])
    return predicted


def leaves_in_order(tree):
    """The tree's leaves from left to right, as drawn by draw_tree."""
    t = tree.tree_

    def walk(node):
        if t.children_left[node] == -1:
            return [node]
        return walk(t.children_left[node]) + walk(t.children_right[node])

    return walk(0)


def draw_rule_tree(ax, sides, table):
    """Draw a tree from fit_rule_tree: the first question (previous action) and both
    sides below it. Leaf k (from the left) is drawn at x = k."""
    tops, x, depth = [], 0, 0
    for prev in (0, 1):
        rows = table[table["previous_action_t"] == prev]
        top, n_leaves, side_depth = draw_tree(ax, sides[prev], rows, x0=x, y0=-1, finish=False,
                                              leaf_color=COLORS[prev])
        tops.append(top)
        x += n_leaves
        depth = max(depth, side_depth)
    root_x = (tops[0][0] + tops[1][0]) / 2
    for (cx, cy), answer in zip(tops, ("yes", "no")):
        ax.plot([root_x, cx], [0, cy], color=GREY, linewidth=1, zorder=0)
        ax.text((root_x + cx) / 2, cy / 2, answer, ha="center", va="center", fontsize=8, color=MUTED,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "linewidth": 0})
    ax.text(root_x, 0, "was previous_action_t\nOFF?", ha="center", va="center", fontsize=9, color=INK,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": INK, "linewidth": 1.1})
    ax.set_xlim(-0.5, x - 0.5)
    ax.set_ylim(-1 - depth - 0.5, 0.5)
    ax.axis("off")


def plot_portrait(ax, table):
    """Scatter the agent's P(treat) against the current burden, one colour per
    previous action. csv data (one row per decision)."""
    for prev in (0, 1):
        rows = table[table["previous_action_t"] == prev]
        ax.scatter(rows["burden_t"], rows["p_treat"], s=16, color=COLORS[prev], alpha=0.55,
                   edgecolors="white", linewidths=0.4, label=f"{LABELS[prev]} (n={len(rows)})")

    # The deterministic agent treats when P(treat) > 0.5.
    ax.axhline(0.5, color=GREY, linewidth=0.8, zorder=0)
    ax.text(ax.get_xlim()[1], 0.51, "treat if above ", ha="right", va="bottom", fontsize=8, color=GREY)

    ax.set_xlabel("current burden  $B_t$  (cell count / initial cell count)")
    ax.set_ylabel("agent's P(treat)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
