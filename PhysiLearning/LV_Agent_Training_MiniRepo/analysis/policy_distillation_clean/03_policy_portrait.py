"""Step 3: the "policy portrait": P(treat) vs. current burden, coloured by previous action.

Every dot is one decision the agent made, in the DPD runs (table from step 2) or
in the LV simulator (table from step 2b). This is the target plot, still without
the two thresholds (those come in step 6).

Output: figures/03_policy_portrait.png (DPD) or figures/03_policy_portrait_lv.png (LV)

Run from anywhere, in the `distill` environment:
    python analysis/policy_distillation_clean/03_policy_portrait.py       # DPD
    python analysis/policy_distillation_clean/03_policy_portrait.py lv    # LV
"""
import matplotlib.pyplot as plt

from common import FIG_DIR, GREY, LABELS, choose_dataset, plot_portrait

table, suffix, source = choose_dataset()

# 1. The plot -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
plot_portrait(ax, table)

# The first decision of every run has no history yet (zeros) and burden exactly 1.0,
# so all of them land on one spot.
first = table[table["day"] == 0]
# (The text is placed in axes coordinates, near the left edge, which is empty in both datasets.)
ax.annotate(f"first decision\nof every run\n({len(first)} dots, no history)",
            xy=(1.0, first["p_treat"].iloc[0]), xytext=(0.04, 0.30), textcoords="axes fraction",
            fontsize=8, color=GREY, arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.8})

ax.set_title(f"What the agent does in {source}", loc="left")
fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / f"03_policy_portrait{suffix}.png"
fig.savefig(out, dpi=200)
print(f"Saved to {out}\n")

# 2. Where does the current burden alone NOT decide? ----------------------------
# Within each group, find the burden range where the agent sometimes treated and
# sometimes didn't. Inside that range other inputs (older burdens and actions)
# must tip the decision. That is what the tree will have to work out.
for prev in (0, 1):
    rows = table[table["previous_action_t"] == prev]
    treated = rows.loc[rows["logged_action"] == 1, "burden_t"]
    not_treated = rows.loc[rows["logged_action"] == 0, "burden_t"]
    lo, hi = treated.min(), not_treated.max()
    in_overlap = rows["burden_t"].between(lo, hi)
    print(f"{LABELS[prev]}:")
    print(f"  treated at burdens      {treated.min():.2f} - {treated.max():.2f}")
    print(f"  not treated at burdens  {not_treated.min():.2f} - {not_treated.max():.2f}")
    print(f"  -> between {lo:.2f} and {hi:.2f} the burden alone doesn't decide "
          f"({in_overlap.sum()} of {len(rows)} decisions)\n")
