"""Extra ATAgent2 policy portrait marking prediction / treatment mismatches.

Run directly with PyCharm's green Run button, or as part of
03c_experimental_policy_plots.py. Labels use original experimental calendar
days (measurement_day), not zero-based elapsed days.
"""
import matplotlib.pyplot as plt

from common import FIG_DIR, INK, choose_dataset, plot_portrait


def main():
    table, _, _ = choose_dataset("experimental")
    # Recompute from the probability and recorded treatment, rather than trusting
    # a stored agreement flag. Match the teacher's strict P > 0.5 decision rule.
    predicted = (table["p_treat"] > 0.5).astype(int)
    mismatches = table.loc[predicted != table["executed_action"]].copy()
    mismatches["predicted_action"] = predicted.loc[mismatches.index]
    mismatches = mismatches.sort_values(["burden_t", "run", "measurement_day"])

    # Separate annotation space keeps the labels clear of the observations.
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    fig.subplots_adjust(left=0.09, right=0.66, bottom=0.16, top=0.84)
    plot_portrait(ax, table)
    ax.set_title(
        f"ATagent2: {len(mismatches)} recorded treatments differ from the prediction",
        loc="left", fontsize=12, pad=14)
    if len(mismatches):
        ax.scatter(mismatches["burden_t"], mismatches["p_treat"], s=150,
                   facecolors="none", edgecolors="#b2182b", linewidths=2,
                   zorder=5, label="prediction / treatment mismatch")
        for index, (_, row) in enumerate(mismatches.iterrows()):
            plate = row["run"].split("/")[0].replace("_", " ")
            well = row["sample"].split("_")[0]
            expected = "ON" if row["predicted_action"] else "OFF"
            recorded = "ON" if row["executed_action"] else "OFF"
            label = (
                f"{plate}, well {well} — day {int(row['measurement_day'])}\n"
                f"Predicted {expected}; recorded {recorded}\n"
                f"P(treat) = {row['p_treat']:.3f}\n"
                f"Burden = {row['burden_t']:.3f}")
            y = 0.86 - index * 0.72 / max(1, len(mismatches) - 1)
            ax.annotate(
                label, xy=(row["burden_t"], row["p_treat"]),
                xytext=(1.06, y), textcoords="axes fraction",
                ha="left", va="top", fontsize=9, color=INK,
                bbox={"boxstyle": "round,pad=0.6", "fc": "white", "ec": "#b2182b"},
                arrowprops={"arrowstyle": "->", "color": "#b2182b", "lw": 1.2},
                annotation_clip=False, zorder=6)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.text(0.09, 0.035,
             "Red rings: recorded treatment differs from P(treat) > 0.5. Labels use experimental calendar days.",
             fontsize=9, color="#52514e")
    FIG_DIR.mkdir(exist_ok=True)
    output = FIG_DIR / "03d_experimental_mismatches.png"
    fig.savefig(output, dpi=200)
    print(f"Saved to {output}")
    print(mismatches[["run", "measurement_day", "burden_t", "p_treat",
                      "predicted_action", "executed_action"]].to_string(index=False))
    return fig


if __name__ == "__main__":
    main()
    # The aggregate runner displays all figures once after every script finishes.
    import sys
    if "experimental" not in sys.argv[1:]:
        plt.show()
