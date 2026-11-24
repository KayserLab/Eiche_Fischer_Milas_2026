"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script analyzes plate-reader alamarBlue measurements for the alpha2 and
alpha4 conditions and produces the side-by-side Figure 1 comparison plot.

Detailed
--------
Inputs:
- Plate-reader XLSX sheets containing summary and replicate-level values.

Outputs:
- Figure files comparing relative growth behavior between the selected
  conditions.
- Intermediate summary statistics used to annotate or validate the plot.

How it works:
- Loads the relevant XLSX sheets and identifies the replicate groups used in
  the manuscript summary.
- Converts replicate tables into a long format, computes log2 fold changes and
  comparison metrics across time/concentration, and summarizes the resulting
  distributions.
- Produces the final comparison plot plus optional extra interval summaries.

Pipeline role:
- Figure 1 plotting script for the plate-reader growth comparison branch.
"""

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: 2025_01_Plate_Reader_Results.xlsx (alpha2/alpha4 worksheets).
# Scripts: shared/figtools.py for publication sizing and export.
# Outputs: Figure 1 PDF plus CSV tables of replicate metrics and statistics.

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.data_paths import EXP_RAW_DATA, figure_data_dir


# Input file and worksheet configuration
FILE_PATH = str(EXP_RAW_DATA / "alamar_blue" / "2025_01_Plate_Reader_Results.xlsx")
SHEETS = {
    "alpha2": "G418 alpha-2",
    "alpha4": "G418 alpha-4",
}

# These row ranges follow the existing scripts in this folder (0-based indexing).
CONDITION_START_ROW = 281
CONDITION_END_ROW = 294
STD_START_ROW = 296
STD_END_ROW = 309

DAYS = ["0", "2", "5", "7"]
TARGET_CONCENTRATIONS = [0.0, 0.4]
EXCLUDE_LAST_TIMEPOINT = True
OUTPUT_DIR = ft.paper_figure_data_dir(__file__)
NORMALIZED_CELL_COUNT_OUTPUT_DIR = figure_data_dir("Figure_1", "alamar_blue", create=False)
OUTPUT_BASENAME = "alamar_blue_a2a4"
PLOT_DAYS = [2, 5]
SUMMARY_MATCH_TOL = 1e-8
COLOR_SENSITIVE = "#4169e1"  # Royalblue
COLOR_RESISTANT = "#DAA520"  # Goldenrod

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to axes boxes: the physical x-axis and y-axis
# length. Margins are extra space for labels, annotations, and titles.
MAIN_AXES_WIDTHS_MM = (105.0, 57.0)
MAIN_AXES_HEIGHT_MM = 60.0
MAIN_MARGINS_MM = (14.0, 4.0, 18.0, 8.0)  # left, right, bottom, top
MAIN_PANEL_GAP_MM = 8.0
MIXED_AXES_WIDTH_MM = 42.45
MIXED_AXES_HEIGHT_MM = 17.53
# Keep the plot box fixed, but give outside tick labels and annotations room.
MIXED_MARGINS_MM = (16.0, 8.0, 22.0, 8.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300
SHOW_FIGURE = True

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
)


def _load_sheet(file_path: str, sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load mean and std blocks from one worksheet."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    means = df.iloc[CONDITION_START_ROW:CONDITION_END_ROW, [0, 1, 2, 3, 4]].copy()
    means.columns = ["Condition", *DAYS]
    means = means.dropna(subset=["Condition"])
    means["Condition"] = pd.to_numeric(means["Condition"], errors="coerce")
    means = means.dropna(subset=["Condition"])
    means[DAYS] = means[DAYS].astype(float)

    stds = df.iloc[STD_START_ROW:STD_END_ROW, [0, 1, 2, 3, 4]].copy()
    stds.columns = ["Condition", *DAYS]
    stds = stds.dropna(subset=["Condition"])
    stds["Condition"] = pd.to_numeric(stds["Condition"], errors="coerce")
    stds = stds.dropna(subset=["Condition"])
    stds[DAYS] = stds[DAYS].astype(float)
    stds = stds.set_index("Condition")

    return means, stds


def _extract_target_conditions(
    means: pd.DataFrame, stds: pd.DataFrame, line_name: str
) -> pd.DataFrame:
    """Return long-form table for target concentrations and all days."""
    rows = []
    for conc in TARGET_CONCENTRATIONS:
        match = means[np.isclose(means["Condition"], conc, atol=1e-9)]
        if match.empty:
            print(f"[WARN] {line_name}: concentration {conc} mg/ml not found.")
            continue

        mean_row = match.iloc[0]
        std_row = stds.loc[conc] if conc in stds.index else pd.Series({d: np.nan for d in DAYS})
        for day in DAYS:
            rows.append(
                {
                    "cell_line": line_name,
                    "concentration_mg_ml": conc,
                    "day": int(day),
                    "mean_relative_growth": float(mean_row[day]),
                    "std_relative_growth": float(std_row.get(day, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _load_replicate_candidates(file_path: str, sheet_name: str, line_name: str) -> pd.DataFrame:
    """Load replicate rows with normalized values at d2/d5/d7."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    raw = df.iloc[:, [0, 1, 2, 4, 5, 6]].copy()
    raw.columns = ["plate", "condition", "well", "d2", "d5", "d7"]
    raw["plate"] = pd.to_numeric(raw["plate"], errors="coerce")
    raw["condition"] = pd.to_numeric(raw["condition"], errors="coerce")
    raw["well"] = raw["well"].astype(str)
    raw["d2"] = pd.to_numeric(raw["d2"], errors="coerce")
    raw["d5"] = pd.to_numeric(raw["d5"], errors="coerce")
    raw["d7"] = pd.to_numeric(raw["d7"], errors="coerce")

    raw = raw[
        raw["plate"].notna()
        & raw["condition"].notna()
        & raw["well"].str.match(r"^[A-H][0-9]{1,2}$")
        & raw[["d2", "d5", "d7"]].notna().all(axis=1)
    ].copy()
    raw = raw[raw["condition"].isin(TARGET_CONCENTRATIONS)].copy()
    raw["cell_line"] = line_name
    raw["concentration_mg_ml"] = raw["condition"].astype(float)
    raw["replicate_id"] = raw.apply(lambda r: f"p{int(r['plate'])}_{r['well']}", axis=1)
    return raw[
        ["cell_line", "concentration_mg_ml", "replicate_id", "d2", "d5", "d7"]
    ].sort_values(["concentration_mg_ml", "replicate_id"])


def _infer_replicates_used_for_summary(
    candidates: pd.DataFrame, means: pd.DataFrame, stds: pd.DataFrame, line_name: str
) -> pd.DataFrame:
    """
    Infer which replicate subset was used for summary stats by matching mean/std (population SD)
    across d2/d5/d7 for each target concentration.
    """
    selected_parts = []
    for conc in TARGET_CONCENTRATIONS:
        conc_candidates = candidates[np.isclose(candidates["concentration_mg_ml"], conc)].copy()
        if conc_candidates.empty:
            print(f"[WARN] {line_name}: no replicate candidates found for {conc} mg/ml.")
            continue

        mean_match = means[np.isclose(means["Condition"], conc)]
        if mean_match.empty or conc not in stds.index:
            print(f"[WARN] {line_name}: missing summary rows for {conc} mg/ml, using all candidates.")
            selected_parts.append(conc_candidates)
            continue

        target_mean = mean_match.iloc[0]
        target_std = stds.loc[conc]
        idx = conc_candidates.index.to_list()
        best_combo = None
        best_err = np.inf

        for k in range(2, len(idx) + 1):
            for combo in combinations(idx, k):
                sub = conc_candidates.loc[list(combo)]
                d2_vals = sub["d2"].to_numpy()
                d5_vals = sub["d5"].to_numpy()
                d7_vals = sub["d7"].to_numpy()
                err = (
                    abs(np.mean(d2_vals) - float(target_mean["2"]))
                    + abs(np.std(d2_vals, ddof=0) - float(target_std["2"]))
                    + abs(np.mean(d5_vals) - float(target_mean["5"]))
                    + abs(np.std(d5_vals, ddof=0) - float(target_std["5"]))
                    + abs(np.mean(d7_vals) - float(target_mean["7"]))
                    + abs(np.std(d7_vals, ddof=0) - float(target_std["7"]))
                )

                if err < best_err - 1e-14 or (
                    abs(err - best_err) <= 1e-14 and best_combo is not None and len(combo) > len(best_combo)
                ):
                    best_err = err
                    best_combo = combo

        if best_combo is None:
            print(f"[WARN] {line_name}: could not infer subset for {conc} mg/ml; using all candidates.")
            selected_parts.append(conc_candidates)
            continue

        if best_err > SUMMARY_MATCH_TOL:
            print(
                f"[WARN] {line_name}: best subset for {conc} mg/ml has residual error {best_err:.3e}; "
                f"using best match with n={len(best_combo)}."
            )
        else:
            print(f"[INFO] {line_name}: matched summary subset for {conc} mg/ml with n={len(best_combo)}.")

        selected_parts.append(conc_candidates.loc[list(best_combo)].copy())

    if not selected_parts:
        return pd.DataFrame(columns=candidates.columns)
    return pd.concat(selected_parts, ignore_index=True)


def _replicates_wide_to_long(reps_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert selected replicate rows into long format by day."""
    long = reps_wide.melt(
        id_vars=["cell_line", "concentration_mg_ml", "replicate_id"],
        value_vars=["d2", "d5", "d7"],
        var_name="day_label",
        value_name="relative_growth",
    )
    day_map = {"d2": 2, "d5": 5, "d7": 7}
    long["day"] = long["day_label"].map(day_map)
    long["relative_growth"] = pd.to_numeric(long["relative_growth"], errors="coerce")
    return long.dropna(subset=["day", "relative_growth"]).copy()


def _compute_pairwise_metrics(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute alpha4-vs-alpha2 metrics for each concentration/day."""
    alpha2 = comp[comp["cell_line"] == "alpha2"].set_index(["concentration_mg_ml", "day"])
    alpha4 = comp[comp["cell_line"] == "alpha4"].set_index(["concentration_mg_ml", "day"])

    shared_idx = alpha2.index.intersection(alpha4.index)
    rows = []
    for conc, day in shared_idx:
        a2 = alpha2.loc[(conc, day)]
        a4 = alpha4.loc[(conc, day)]

        diff = a4["mean_relative_growth"] - a2["mean_relative_growth"]
        pct_diff = (diff / a2["mean_relative_growth"] * 100.0) if a2["mean_relative_growth"] != 0 else np.nan
        log2_fc = (
            np.log2(a4["mean_relative_growth"] / a2["mean_relative_growth"])
            if a2["mean_relative_growth"] > 0 and a4["mean_relative_growth"] > 0
            else np.nan
        )

        pooled_sd = np.sqrt((a2["std_relative_growth"] ** 2 + a4["std_relative_growth"] ** 2) / 2.0)
        cohens_d = (diff / pooled_sd) if pooled_sd > 0 else np.nan

        rows.append(
            {
                "concentration_mg_ml": conc,
                "day": day,
                "alpha2_mean": a2["mean_relative_growth"],
                "alpha4_mean": a4["mean_relative_growth"],
                "difference_alpha4_minus_alpha2": diff,
                "percent_difference_vs_alpha2": pct_diff,
                "log2_fold_change_alpha4_over_alpha2": log2_fc,
                "cohens_d_approx": cohens_d,
            }
        )

    return pd.DataFrame(rows).sort_values(["concentration_mg_ml", "day"])


def _compute_similarity_by_concentration(comp: pd.DataFrame) -> pd.DataFrame:
    """Similarity summary between alpha2 and alpha4 trajectories per concentration."""
    rows = []
    for conc in TARGET_CONCENTRATIONS:
        sub = comp[comp["concentration_mg_ml"] == conc]
        pivot = sub.pivot(index="day", columns="cell_line", values="mean_relative_growth").dropna()
        if {"alpha2", "alpha4"}.issubset(pivot.columns) and len(pivot) >= 2:
            r = pivot["alpha2"].corr(pivot["alpha4"])
            rmse = float(np.sqrt(np.mean((pivot["alpha4"] - pivot["alpha2"]) ** 2)))
        else:
            r = np.nan
            rmse = np.nan
        rows.append(
            {
                "concentration_mg_ml": conc,
                "pearson_r_across_days": r,
                "rmse_across_days": rmse,
            }
        )
    return pd.DataFrame(rows)


def _compute_log2_fold_change(replicates_long: pd.DataFrame) -> pd.DataFrame:
    """Compute log2 growth relative to baseline and log2(day 5 / day 2)."""
    if replicates_long.empty:
        return pd.DataFrame()

    sub = replicates_long[replicates_long["day"].isin(PLOT_DAYS)].copy()
    if not np.all(np.isfinite(sub["relative_growth"]) & sub["relative_growth"].gt(0)):
        raise ValueError("Log2 fold change requires finite, positive relative-growth values.")
    sub["log2_fold_change"] = np.log2(sub["relative_growth"])

    # Each replicate's interval change is symmetric for reciprocal ratios:
    # doubling is +1, halving is -1, and no change is 0.
    d2 = sub[sub["day"] == 2][
        ["cell_line", "concentration_mg_ml", "replicate_id", "relative_growth"]
    ].rename(columns={"relative_growth": "rg_d2"})
    d5 = sub[sub["day"] == 5][
        ["cell_line", "concentration_mg_ml", "replicate_id", "relative_growth"]
    ].rename(columns={"relative_growth": "rg_d5"})
    d2d5 = d2.merge(d5, on=["cell_line", "concentration_mg_ml", "replicate_id"], how="inner")
    d2d5["log2_fold_change_d2_to_d5"] = np.log2(d2d5["rg_d5"] / d2d5["rg_d2"])
    sub = sub.merge(
        d2d5[["cell_line", "concentration_mg_ml", "replicate_id", "log2_fold_change_d2_to_d5"]],
        on=["cell_line", "concentration_mg_ml", "replicate_id"],
        how="left",
    )
    return sub


def _compute_group_statistics(rel: pd.DataFrame) -> pd.DataFrame:
    """Compare alpha2 vs alpha4 log2 fold changes for each dose/day group."""
    rows = []
    grouped_conditions = [(0.0, 2), (0.0, 5), (0.4, 2), (0.4, 5)]
    for conc, day in grouped_conditions:
        alpha2_vals = rel[
            (np.isclose(rel["concentration_mg_ml"], conc))
            & (rel["day"] == day)
            & (rel["cell_line"] == "alpha2")
        ]["log2_fold_change"].dropna().to_numpy()
        alpha4_vals = rel[
            (np.isclose(rel["concentration_mg_ml"], conc))
            & (rel["day"] == day)
            & (rel["cell_line"] == "alpha4")
        ]["log2_fold_change"].dropna().to_numpy()

        if len(alpha2_vals) >= 2 and len(alpha4_vals) >= 2:
            welch = stats.ttest_ind(alpha2_vals, alpha4_vals, equal_var=False, alternative="two-sided")
            mwu = stats.mannwhitneyu(alpha2_vals, alpha4_vals, alternative="two-sided")
            welch_t = float(welch.statistic)
            welch_p = float(welch.pvalue)
            mwu_u = float(mwu.statistic)
            mwu_p = float(mwu.pvalue)
        else:
            welch_t = np.nan
            welch_p = np.nan
            mwu_u = np.nan
            mwu_p = np.nan

        rows.append(
            {
                "concentration_mg_ml": conc,
                "day": day,
                "metric": "log2_fold_change",
                "n_alpha2": int(len(alpha2_vals)),
                "n_alpha4": int(len(alpha4_vals)),
                "mean_alpha2": float(np.mean(alpha2_vals)) if len(alpha2_vals) else np.nan,
                "mean_alpha4": float(np.mean(alpha4_vals)) if len(alpha4_vals) else np.nan,
                "welch_t_stat": welch_t,
                "welch_p_value": welch_p,
                "mannwhitney_u_stat": mwu_u,
                "mannwhitney_p_value": mwu_p,
            }
        )

    return pd.DataFrame(rows)


def _compute_d2_to_d5_statistics(rel: pd.DataFrame) -> pd.DataFrame:
    """Compare alpha2 vs alpha4 log2(day 5 / day 2), once per replicate."""
    unique_d2d5 = rel[
        ["cell_line", "concentration_mg_ml", "replicate_id", "log2_fold_change_d2_to_d5"]
    ].drop_duplicates()
    rows = []
    for conc in [0.0, 0.4]:
        alpha2_vals = unique_d2d5[
            (np.isclose(unique_d2d5["concentration_mg_ml"], conc))
            & (unique_d2d5["cell_line"] == "alpha2")
        ]["log2_fold_change_d2_to_d5"].dropna().to_numpy()
        alpha4_vals = unique_d2d5[
            (np.isclose(unique_d2d5["concentration_mg_ml"], conc))
            & (unique_d2d5["cell_line"] == "alpha4")
        ]["log2_fold_change_d2_to_d5"].dropna().to_numpy()

        if len(alpha2_vals) >= 2 and len(alpha4_vals) >= 2:
            welch = stats.ttest_ind(alpha2_vals, alpha4_vals, equal_var=False, alternative="two-sided")
            mwu = stats.mannwhitneyu(alpha2_vals, alpha4_vals, alternative="two-sided")
            welch_t = float(welch.statistic)
            welch_p = float(welch.pvalue)
            mwu_u = float(mwu.statistic)
            mwu_p = float(mwu.pvalue)
        else:
            welch_t = np.nan
            welch_p = np.nan
            mwu_u = np.nan
            mwu_p = np.nan

        rows.append(
            {
                "concentration_mg_ml": conc,
                "comparison": "d2_to_d5",
                "metric": "log2_fold_change_d2_to_d5",
                "n_alpha2": int(len(alpha2_vals)),
                "n_alpha4": int(len(alpha4_vals)),
                "mean_alpha2": float(np.mean(alpha2_vals)) if len(alpha2_vals) else np.nan,
                "mean_alpha4": float(np.mean(alpha4_vals)) if len(alpha4_vals) else np.nan,
                "welch_t_stat": welch_t,
                "welch_p_value": welch_p,
                "mannwhitney_u_stat": mwu_u,
                "mannwhitney_p_value": mwu_p,
            }
        )
    return pd.DataFrame(rows)


def _plot_comparison(replicates_long: pd.DataFrame, output_pdf: str) -> None:
    rel = _compute_log2_fold_change(replicates_long)
    if rel.empty:
        raise ValueError("No log2-fold-change data to plot. Ensure selected replicates contain day 2/day 5.")

    color_map = {"alpha2": COLOR_SENSITIVE, "alpha4": COLOR_RESISTANT}
    grouped_conditions = [(0.0, 2), (0.0, 5), (0.4, 2), (0.4, 5)]

    box_data_main = []
    box_positions_main = []
    box_colors_main = []
    group_centers_main = []
    group_labels_main = []
    group_width = 0.34

    for i, (conc, day) in enumerate(grouped_conditions, start=1):
        group_centers_main.append(i)
        group_labels_main.append(f"{conc:.1f} mg/ml\nDay {day}")
        for offset, line in [(-group_width / 2, "alpha2"), (group_width / 2, "alpha4")]:
            vals = rel[
                (np.isclose(rel["concentration_mg_ml"], conc))
                & (rel["day"] == day)
                & (rel["cell_line"] == line)
            ]["log2_fold_change"].dropna().to_numpy()
            if len(vals) > 0:
                box_data_main.append(vals)
                box_positions_main.append(i + offset)
                box_colors_main.append(color_map[line])

    if not box_data_main:
        raise ValueError("No valid log2-fold-change values found for requested dose/day combinations.")

    fig, axes = ft.new_subplots_variable(
        1,
        2,
        ax_widths_mm=MAIN_AXES_WIDTHS_MM,
        ax_heights_mm=(MAIN_AXES_HEIGHT_MM,),
        margins=MAIN_MARGINS_MM,
        wspace_mm=MAIN_PANEL_GAP_MM,
        dpi=DPI,
        apply_style=False,
    )
    ax = axes[0]
    bp = ax.boxplot(
        box_data_main,
        positions=box_positions_main,
        widths=0.28,
        patch_artist=True,
        boxprops={"linewidth": 0.4},
        whiskerprops={"linewidth": 0.4},
        capprops={"linewidth": 0.4},
        medianprops={"linewidth": 0.4, "color": "black"},
    )
    for patch, color in zip(bp["boxes"], box_colors_main):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)

    ax.set_xticks(group_centers_main)
    ax.set_xticklabels(group_labels_main)
    ax.set_ylabel("Log$_2$ fold change\nin cell count")
    ax.set_xlabel("G418 dose and day")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    handles = [
        plt.Line2D([0], [0], color=color_map["alpha2"], lw=8, alpha=1.0, label="alpha2"),
        plt.Line2D([0], [0], color=color_map["alpha4"], lw=8, alpha=1.0, label="alpha4"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    stats_df = _compute_group_statistics(rel)
    y_max = np.nanmax(rel["log2_fold_change"].to_numpy())
    y_min = np.nanmin(rel["log2_fold_change"].to_numpy())
    y_span = max(y_max - y_min, 1e-6)
    for i, (conc, day) in enumerate(grouped_conditions, start=1):
        row = stats_df[
            (np.isclose(stats_df["concentration_mg_ml"], conc)) & (stats_df["day"] == day)
        ]
        if row.empty or pd.isna(row.iloc[0]["welch_p_value"]):
            label = "p=n/a"
        else:
            p = float(row.iloc[0]["welch_p_value"])
            label = f"p={p:.3g}"
        ax.text(i, y_max + 0.06 * y_span, label, ha="center", va="bottom", fontsize=TICK_SIZE_PT)

    # Second panel: log2 fold change between day 2 and day 5.
    ax2 = axes[1]
    rel_d2d5 = rel[
        ["cell_line", "concentration_mg_ml", "replicate_id", "log2_fold_change_d2_to_d5"]
    ].drop_duplicates()
    doses = [0.0, 0.4]
    box_data_d2d5 = []
    box_positions_d2d5 = []
    box_colors_d2d5 = []
    tick_pos_d2d5 = []
    tick_labels_d2d5 = []
    dose_centers_d2d5 = []
    dose_labels_d2d5 = []
    for i, conc in enumerate(doses, start=1):
        dose_centers_d2d5.append(i)
        dose_labels_d2d5.append(f"{conc:.1f} mg/ml")
        for offset, line in [(-group_width / 2, "alpha2"), (group_width / 2, "alpha4")]:
            vals = rel_d2d5[
                (np.isclose(rel_d2d5["concentration_mg_ml"], conc))
                & (rel_d2d5["cell_line"] == line)
            ]["log2_fold_change_d2_to_d5"].dropna().to_numpy()
            if len(vals) > 0:
                box_data_d2d5.append(vals)
                xpos = i + offset
                box_positions_d2d5.append(xpos)
                box_colors_d2d5.append(color_map[line])
                tick_pos_d2d5.append(xpos)
                tick_labels_d2d5.append("Sensitive" if line == "alpha2" else "Resistant")

    if box_data_d2d5:
        bp2 = ax2.boxplot(
            box_data_d2d5,
            positions=box_positions_d2d5,
            widths=0.28,
            patch_artist=True,
            boxprops={"linewidth": 0.4},
            whiskerprops={"linewidth": 0.4},
            capprops={"linewidth": 0.4},
            medianprops={"linewidth": 0.4, "color": "black"},
        )
        for patch, color in zip(bp2["boxes"], box_colors_d2d5):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)
    ax2.set_xticks(tick_pos_d2d5)
    ax2.set_xticklabels(tick_labels_d2d5)
    ax2.set_xlabel("G418 dose")
    ax2.set_ylabel("Log$_2$ fold change\nin cell count")
    ax2.set_title("Day 2 \u2192 Day 5")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    for x, lab in zip(dose_centers_d2d5, dose_labels_d2d5):
        ax2.text(x, -0.18, lab, transform=ax2.get_xaxis_transform(), ha="center", va="top")

    stats_d2d5 = _compute_d2_to_d5_statistics(rel)
    for i, conc in enumerate(doses, start=1):
        row = stats_d2d5[np.isclose(stats_d2d5["concentration_mg_ml"], conc)]
        if row.empty or pd.isna(row.iloc[0]["welch_p_value"]):
            label = "p=n/a"
        else:
            p = float(row.iloc[0]["welch_p_value"])
            label = f"p={p:.3g}"
        ax2.text(i, y_max + 0.06 * y_span, label, ha="center", va="bottom", fontsize=TICK_SIZE_PT)

    d2d5_vals = rel_d2d5["log2_fold_change_d2_to_d5"].dropna().to_numpy()
    all_vals = np.concatenate([rel["log2_fold_change"].dropna().to_numpy(), d2d5_vals]) if len(d2d5_vals) else rel["log2_fold_change"].dropna().to_numpy()
    y_all_min = np.nanmin(all_vals)
    y_all_max = np.nanmax(all_vals)
    y_limit = max(abs(y_all_min), abs(y_all_max), 1e-6) * 1.20
    ax.set_ylim(-y_limit, y_limit)
    ax2.set_ylim(-y_limit, y_limit)

    fig.suptitle("Log2 fold change by dose/day (alpha2 vs alpha4)", y=1.03)
    ft.savefig(fig, output_pdf)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def _plot_extra_mixed_intervals(replicates_long: pd.DataFrame, output_pdf: str) -> None:
    """Plot log2(day 5 / day 2) once per replicate for both treatment states."""
    rel = _compute_log2_fold_change(replicates_long)
    if rel.empty:
        raise ValueError("No log2-fold-change data to plot for mixed intervals.")
    rel_d2d5 = rel[
        ["cell_line", "concentration_mg_ml", "replicate_id", "log2_fold_change_d2_to_d5"]
    ].drop_duplicates()

    color_map = {"alpha2": COLOR_SENSITIVE, "alpha4": COLOR_RESISTANT}
    group_width = 0.46
    doses = [0.0, 0.4]
    box_data = []
    box_positions = []
    box_colors = []
    tick_pos = []
    tick_labels = []
    dose_centers = []
    dose_labels = []
    for i, conc in enumerate(doses, start=1):
        dose_centers.append(i)
        dose_labels.append("Treatment OFF" if np.isclose(conc, 0.0) else "Treatment ON")
        for offset, line in [(-group_width / 2, "alpha2"), (group_width / 2, "alpha4")]:
            vals = rel_d2d5[
                (np.isclose(rel_d2d5["concentration_mg_ml"], conc))
                & (rel_d2d5["cell_line"] == line)
            ]["log2_fold_change_d2_to_d5"].dropna().to_numpy()
            if len(vals) > 0:
                box_data.append(vals)
                xpos = i + offset
                box_positions.append(xpos)
                box_colors.append(color_map[line])
                tick_pos.append(xpos)
                tick_labels.append("Sensitive" if line == "alpha2" else "Resistant")

    if not box_data:
        raise ValueError("No valid values for mixed-interval plot.")

    fig, ax = ft.new_figure(
        width=MIXED_AXES_WIDTH_MM,
        height_mm=MIXED_AXES_HEIGHT_MM,
        margins=MIXED_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.22,
        patch_artist=True,
        boxprops={"linewidth": 0.4},
        whiskerprops={"linewidth": 0.4},
        capprops={"linewidth": 0.4},
        medianprops={"linewidth": 0.4, "color": "black"},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)

    y_values = rel_d2d5["log2_fold_change_d2_to_d5"].dropna().to_numpy()
    y_limit = max(1.0, float(np.max(np.abs(y_values))) * 1.10)
    ax.set_ylim(-y_limit, y_limit)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, symmetric=True))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", pad=1)
    ax.set_ylabel("Log$_2$ fold change\nin cell count", labelpad=1)
    # Tight horizontal bounds to reduce extra whitespace.
    ax.set_xlim(0.55, 2.45)
    # Highlight treatment group (0.4 mg/ml) as right half background.
    x_left, x_right = ax.get_xlim()
    x_mid = (x_left + x_right) / 2.0
    ax.axvspan(x_mid, x_right, color="#bfbfbf", alpha=1.0, zorder=0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(False)
    for x, lab in zip(dose_centers, dose_labels):
        ax.text(x, -0.24, lab, transform=ax.get_xaxis_transform(), ha="center", va="top")
    ft.savefig(fig, output_pdf)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main() -> None:
    plot_output_dir = OUTPUT_DIR
    data_output_dir = NORMALIZED_CELL_COUNT_OUTPUT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    alpha2_means, alpha2_stds = _load_sheet(FILE_PATH, SHEETS["alpha2"])
    alpha4_means, alpha4_stds = _load_sheet(FILE_PATH, SHEETS["alpha4"])

    alpha2_long = _extract_target_conditions(alpha2_means, alpha2_stds, "alpha2")
    alpha4_long = _extract_target_conditions(alpha4_means, alpha4_stds, "alpha4")
    comparison = pd.concat([alpha2_long, alpha4_long], ignore_index=True)
    alpha2_candidates = _load_replicate_candidates(FILE_PATH, SHEETS["alpha2"], "alpha2")
    alpha4_candidates = _load_replicate_candidates(FILE_PATH, SHEETS["alpha4"], "alpha4")
    alpha2_selected = _infer_replicates_used_for_summary(alpha2_candidates, alpha2_means, alpha2_stds, "alpha2")
    alpha4_selected = _infer_replicates_used_for_summary(alpha4_candidates, alpha4_means, alpha4_stds, "alpha4")
    selected_replicates = pd.concat([alpha2_selected, alpha4_selected], ignore_index=True)
    selected_replicates_long = _replicates_wide_to_long(selected_replicates)

    if comparison.empty:
        raise ValueError("No comparison data found. Check concentrations and sheet structure.")
    if selected_replicates_long.empty:
        raise ValueError("No selected replicate data found. Check sheet layout and summary matching.")

    if EXCLUDE_LAST_TIMEPOINT:
        last_day = comparison["day"].max()
        comparison = comparison[comparison["day"] < last_day].copy()
        print(f"\nExcluded last timepoint: day {last_day}")

    pairwise_metrics = _compute_pairwise_metrics(comparison)
    similarity_metrics = _compute_similarity_by_concentration(comparison)
    fold_change = _compute_log2_fold_change(selected_replicates_long)
    group_stats = _compute_group_statistics(fold_change)
    d2_to_d5_stats = _compute_d2_to_d5_statistics(fold_change)

    output_csv_pairwise = data_output_dir / f"{OUTPUT_BASENAME}_pairwise_metrics.csv"
    output_csv_similarity = data_output_dir / f"{OUTPUT_BASENAME}_similarity_metrics.csv"
    output_csv_selected_reps = data_output_dir / f"{OUTPUT_BASENAME}_replicates_selected_for_summary.csv"
    output_csv_fold_change = data_output_dir / f"{OUTPUT_BASENAME}_log2_fold_change.csv"
    output_csv_stats = data_output_dir / f"{OUTPUT_BASENAME}_statistics_alpha2_vs_alpha4_log2_fold_change.csv"
    output_csv_d2_to_d5_stats = data_output_dir / f"{OUTPUT_BASENAME}_statistics_alpha2_vs_alpha4_d2_to_d5_log2_fold_change.csv"
    output_plot_pdf = plot_output_dir / f"{OUTPUT_BASENAME}_normalized_cell_count.pdf"

    pairwise_metrics.to_csv(output_csv_pairwise, index=False)
    similarity_metrics.to_csv(output_csv_similarity, index=False)
    selected_replicates_long.to_csv(output_csv_selected_reps, index=False)
    fold_change.to_csv(output_csv_fold_change, index=False)
    group_stats.to_csv(output_csv_stats, index=False)
    d2_to_d5_stats.to_csv(output_csv_d2_to_d5_stats, index=False)

    print("\nPairwise metrics (alpha4 vs alpha2):")
    print(pairwise_metrics.to_string(index=False))
    print("\nSimilarity by concentration:")
    print(similarity_metrics.to_string(index=False))
    print("\nSelected replicate values used for summary matching:")
    print(selected_replicates_long.to_string(index=False))
    print("\nLog2 fold change (day 2 and day 5):")
    print(fold_change.to_string(index=False))
    print("\nStatistics alpha2 vs alpha4 (log2 fold change per dose/day group):")
    print(group_stats.to_string(index=False))
    print("\nStatistics alpha2 vs alpha4 (day 2 to day 5 log2 fold change):")
    print(d2_to_d5_stats.to_string(index=False))
    print(f"\nSaved: {output_csv_pairwise}")
    print(f"Saved: {output_csv_similarity}")
    print(f"Saved: {output_csv_selected_reps}")
    print(f"Saved: {output_csv_fold_change}")
    print(f"Saved: {output_csv_stats}")
    print(f"Saved: {output_csv_d2_to_d5_stats}")
    print(f"Saved: {output_plot_pdf}")

    _plot_extra_mixed_intervals(selected_replicates_long, str(output_plot_pdf))


if __name__ == "__main__":
    main()
