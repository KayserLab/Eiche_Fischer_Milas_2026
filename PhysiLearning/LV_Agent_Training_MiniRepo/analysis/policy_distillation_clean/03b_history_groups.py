"""Step 3b: split the policy portrait by treatment history.

The blue and orange clouds in step 3 each consist of several S-curves. Here we
look at the decisions separately for each treatment history (the last four
actions), and then for groups of histories.

A history is written oldest -> newest, so the last digit is the action that was
running when the agent decided (blue = ends in 0, orange = ends in 1).
"*" means "0 or 1". Example: "1001" = ON, OFF, OFF, ON, and it fits "**01".

Hollow dots are decisions from the first three decision days of a run (days 0,
2, 4). Their burden history still contains padding zeros.

Outputs (with "_lv" added to the names when run on the LV data):
    figures/03b_exact_histories.png   one panel per exact history (all 16)
    figures/03b_history_groups.png    all groups in one panel (GROUPS below)

Run from anywhere, in the `distill` environment:
    python analysis/policy_distillation_clean/03b_history_groups.py       # DPD
    python analysis/policy_distillation_clean/03b_history_groups.py lv    # LV
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import COLORS, FIG_DIR, GREY, choose_dataset

ACTIONS = ["previous_action_t-3", "previous_action_t-2", "previous_action_t-1", "previous_action_t"]

# All 16 possible histories. Each row of the figure shares the last two actions.
EXACT = ["0010", "1010", "0110", "1110",   # ends in 10
         "0100", "1100", "1000", "0000",   # ends in 00
         "0001", "1001", "0101", "1101",   # ends in 01
         "0011", "1011", "0111", "1111"]   # ends in 11

# The groups: (name, histories it contains, colour). Green -> indigo for "last action
# OFF", amber/red for "last action ON", in the order of the burden at which the agent
# switches. The six colours were checked against each other, also for colour-blind
# readers; the weakest pair (green vs amber) is told apart by shape and label too.
GROUPS = [
    ("0*00",        ["0*00"],         "#5bbe62"),  # green:  starts treating at ~1.5
    ("1*00",        ["1*00"],         "#009c9c"),  # teal:   ~1.8
    ("0*10",        ["0*10"],         "#096acb"),  # blue:   ~1.9
    ("1*10",        ["1*10"],         "#5a31a1"),  # indigo: ~2.2-2.3
    ("0*01",        ["0*01"],         "#e69825"),  # amber:  continues above ~2.3
    ("1001 + 0011", ["1001", "0011"], "#ba2d1f"),  # red:    never continues
]


def matches(history, pattern):
    """True if `history` fits `pattern`, where * stands for 0 or 1."""
    return all(p in ("*", h) for h, p in zip(history, pattern))


def summary(name, rows):
    """Print where this set of decisions sometimes treats and sometimes doesn't."""
    treated = rows.loc[rows["logged_action"] == 1, "burden_t"]
    not_treated = rows.loc[rows["logged_action"] == 0, "burden_t"]
    t_range = f"{treated.min():.2f}-{treated.max():.2f}" if len(treated) else "never"
    n_range = f"{not_treated.min():.2f}-{not_treated.max():.2f}" if len(not_treated) else "never"
    print(f"  {name:12s} n={len(rows):4d}   treated at burdens {t_range:>11}   "
          f"not treated at {n_range:>11}")


def scatter(ax, rows, color, marker="o", **kwargs):
    """Solid markers for normal decisions, hollow ones for the first three decisions of a
    run (days 0, 2, 4), whose input still contains padding zeros instead of a real past."""
    padded = rows["burden_t-3"] == 0
    ax.scatter(rows.loc[~padded, "burden_t"], rows.loc[~padded, "p_treat"], s=12, color=color,
               marker=marker, alpha=0.6, edgecolors="white", linewidths=0.3, **kwargs)
    ax.scatter(rows.loc[padded, "burden_t"], rows.loc[padded, "p_treat"], s=14, marker=marker,
               facecolors="none", edgecolors=color, linewidths=0.8)


def style(ax):
    ax.axhline(0.5, color=GREY, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_exact(table, suffix, source):
    """One panel per exact history, 4 per row."""
    fig, axes = plt.subplots(4, 4, figsize=(13, 12.4), sharex=True, sharey=True)
    print("Every exact history")
    for ax, history in zip(axes.flat, EXACT):
        rows = table[table["history"] == history]
        # All decisions in light grey as a backdrop, this history in colour on top.
        ax.scatter(table["burden_t"], table["p_treat"], s=5, color="#dcdbd5", linewidths=0)
        scatter(ax, rows, COLORS[int(history[-1])])  # blue if the last action was OFF, orange if ON
        style(ax)
        n_padded = (rows["burden_t-3"] == 0).sum()
        ax.set_title(f"{history}   (n = {len(rows)}" + (f", {n_padded} hollow)" if n_padded else ")"),
                     fontsize=9, loc="left")
        if rows.empty:
            ax.text(0.5, 0.65, "never occurs", transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=GREY)
        summary(history, rows)

    for ax in axes[-1]:
        ax.set_xlabel("current burden $B_t$")
    for ax, label in zip(axes[:, 0], ["ends in 10", "ends in 00", "ends in 01", "ends in 11"]):
        ax.set_ylabel(f"{label}\n\nagent's P(treat)")
    fig.suptitle(f"Policy portrait for every exact history, {source} (oldest -> newest action).\n"
                 "Hollow = first 3 decisions of a run (days 0, 2, 4): no real past yet, "
                 "so the agent's input still contains padding zeros.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"03b_exact_histories{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved to {out}\n")


def draw_groups(table, suffix, source):
    """All groups in one panel, one colour each, labelled where each curve crosses 0.5."""
    fig, ax = plt.subplots(figsize=(10.5, 5))
    in_a_group = pd.Series(False, index=table.index)
    print("Groups")
    for i, (name, patterns, color) in enumerate(GROUPS):
        member = table["history"].map(lambda h: any(matches(h, p) for p in patterns))
        in_a_group |= member
        rows = table[member]
        marker = "o" if patterns[0][-1] == "0" else "^"  # circle: last action OFF, triangle: ON
        scatter(ax, rows, color, marker, label=f"{name}   (n = {len(rows)})")

        # Name the curve just right of where it crosses P(treat) = 0.5, or, if it never
        # gets there, at the right end of its flat part. Alternate two heights so that
        # labels of curves crossing at almost the same burden don't overlap.
        near_half = rows.loc[rows["p_treat"].between(0.4, 0.6), "burden_t"]
        flat = rows[rows["p_treat"] < 0.1]
        if len(near_half):
            ax.text(near_half.median() + 0.012, 0.47 - 0.07 * (i % 2), name, ha="left", va="top",
                    fontsize=8, color="#52514e")
        elif len(flat):
            end = flat.nlargest(1, "burden_t").iloc[0]
            ax.text(end["burden_t"] + 0.015, end["p_treat"], name, ha="left", va="center",
                    fontsize=8, color="#52514e")
        summary(name, rows)

    left_out = table[~in_a_group]
    print(f"  not in any group: {len(left_out)} decision(s), "
          f"histories {sorted(left_out['history'].unique())}")

    # Point out the two off-curve clusters of early decisions.
    # (day 0: label above the dot; day 2: label left of its lowest-burden dot.)
    for day, dx, dy, ha, va in ((0, 0, 0.04, "center", "bottom"), (2, -0.015, 0, "right", "center")):
        rows = table[table["day"] == day]
        start = rows.nsmallest(1, "burden_t").iloc[0]
        ax.text(start["burden_t"] + dx, start["p_treat"] + dy, f"day {day}", ha=ha, va=va,
                fontsize=8, color=GREY)

    style(ax)
    ax.set_xlabel("current burden  $B_t$  (cell count / initial cell count)")
    ax.set_ylabel("agent's P(treat)")
    ax.set_ylim(-0.03, 1.03)
    ax.scatter([], [], s=14, facecolors="none", edgecolors=GREY,  # legend entry only
               label="hollow: first 3 decisions of a run\n(days 0, 2, 4). No real past yet,\n"
                     "so the input still contains\npadding zeros.")
    ax.legend(title="history (oldest -> newest action)\ncircles: last action OFF\ntriangles: last action ON",
              title_fontsize=8, fontsize=8, markerscale=2, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, alignment="left")
    ax.set_title(f"Policy portrait by treatment history, {source}   (* = either)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"03b_history_groups{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out}")


table, suffix, source = choose_dataset()
# Write each decision's last four actions as a string, e.g. "1001".
table["history"] = table[ACTIONS].astype(str).agg("".join, axis=1)

FIG_DIR.mkdir(exist_ok=True)
draw_exact(table, suffix, source) # Fig 03b_exact_histories.png
draw_groups(table, suffix, source) # Fig 03b_history_groups.png
