#!/usr/bin/env python3
"""Create a multipage supplementary PDF of in-vitro versus LV trajectories.

Each page is one retained experimental replicate from the five paper strategies
(CT, AT50, AT100, ATagent, and AT+). The measured sensitive/resistant cell
counts are compared with an LV trajectory from exactly the same initial state,
using the same strategy-specific controllers as
``TTF_All_Strategies_InVitro_DPD_LV.py``. Fully excluded wells are omitted and
truncated wells end at their shared ``experiment_filters`` cutoff.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from analysis.figure_2.Event_Time_Analysis import load_population_counts
from shared.data_paths import figure_data_dir, figure_plot_dir
from shared.experiment_filters import apply_sample_exclusions
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color

STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")
DISPLAY_LABELS = {"LV-Agent": "ATagent", "AT100+": r"AT$^{+}$"}
SENSITIVE_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
PROGRESSION_THRESHOLD = 0.5
AXES_WIDTH_MM = 32.0
AXES_HEIGHT_MM = 23.0
MARGINS_MM = (11.0, 4.0, 10.0, 5.0)
PANELS_PER_PAGE = 9
DPI = 300
SHOW_FIGURE = False

DATA_OUTPUT_DIR = figure_data_dir("Suppl_Figures", "lv_vs_invitro_all_replicates", create=False)
PLOT_OUTPUT = (
    figure_plot_dir("Suppl_Figures", "lv_vs_invitro_all_replicates", create=False)
    / "LV_vs_InVitro_All_Retained_Replicates.pdf"
)
VALUES_OUTPUT = DATA_OUTPUT_DIR / "lv_vs_invitro_all_retained_replicates_trajectories.csv"
AUDIT_OUTPUT = DATA_OUTPUT_DIR / "lv_vs_invitro_all_retained_replicates_audit.csv"

ft.apply_nature_style(
    text_pt=7.0, tick_pt=6.0, legend_pt=5.0,
    overrides={"axes.linewidth": 0.8, "xtick.major.width": 0.8,
               "ytick.major.width": 0.8, "xtick.major.size": 2.5,
               "ytick.major.size": 2.5},
)


def load_ttf_helpers():
    """Load the exact final-LV controller implementation used for the TTF supplement."""
    path = THIS_FILE.parent / "TTF_All_Strategies_InVitro_DPD_LV.py"
    spec = importlib.util.spec_from_file_location("supplementary_ttf_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared LV controller code from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate_lv_trajectory(group: pd.DataFrame, *, strategy: str, helpers, policy) -> pd.DataFrame:
    """Run one controller through the exact filtered in-vitro time horizon."""
    group = group.sort_values("day_tp1")
    baseline = group.iloc[0]
    env = helpers.configure_lv(float(baseline.sus), float(baseline.res))
    observation, _ = env.reset(seed=0)
    observation = np.asarray(observation, dtype=np.float32)
    frames = [np.zeros_like(observation) for _ in range(3)]
    stacked = helpers.stacked_observation(frames, observation)
    previous_action = 0
    horizon = float(group["day_tp1"].max())
    rows = [{"day_tp1": 0.0, "lv_sensitive": float(env.state[0]),
             "lv_resistant": float(env.state[1]), "treatment_on": 0}]
    while float(env.day) - 1.0 < horizon:
        action = helpers.choose_action(
            strategy, env=env, previous_action=previous_action,
            policy=policy, stacked=stacked,
        )
        observation, _reward, _terminated, _truncated, _info = env.step(action)
        previous_action = action
        stacked = helpers.stacked_observation(frames, np.asarray(observation, dtype=np.float32))
        elapsed_day = float(env.day) - 1.0
        if elapsed_day > horizon:
            break
        rows.append({"day_tp1": elapsed_day, "lv_sensitive": float(env.state[0]),
                     "lv_resistant": float(env.state[1]), "treatment_on": action})
    return pd.DataFrame(rows)


def plot_axis(axis: plt.Axes, group: pd.DataFrame, lv: pd.DataFrame, *, strategy: str) -> None:
    initial_total = float(group.iloc[0]["total"])
    axis.plot(group["day_tp1"], group["sus"] / initial_total, color=SENSITIVE_COLOR,
              marker="o", markersize=2.2, linewidth=1.0)
    axis.plot(group["day_tp1"], group["res"] / initial_total, color=RESISTANT_COLOR,
              marker="o", markersize=2.2, linewidth=1.0)
    axis.plot(lv["day_tp1"], lv["lv_sensitive"], color=SENSITIVE_COLOR, linestyle="--", linewidth=1.0)
    axis.plot(lv["day_tp1"], lv["lv_resistant"], color=RESISTANT_COLOR, linestyle="--", linewidth=1.0)
    axis.axhline(PROGRESSION_THRESHOLD, color="#555555", linewidth=0.7, linestyle=":", zorder=0)
    y_max = max(1.05, float(max(group["sus"].max(), group["res"].max()) / initial_total),
                float(max(lv["lv_sensitive"].max(), lv["lv_resistant"].max()))) * 1.08
    axis.set(xlim=(0.0, float(group["day_tp1"].max())), ylim=(0.0, y_max),
             xlabel="Time [days]", ylabel=r"$\eta_{\mathrm{tot}}$")
    record = group.iloc[0]
    axis.set_title(
        f"{record['plate']}  |  {record['sample']}  |  "
        f"{DISPLAY_LABELS.get(strategy, strategy)}",
        color=get_color(strategy, "#222222"), loc="left", fontsize=7, pad=2,
    )
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    counts = load_population_counts(dataset_type="experimental", strategies=STRATEGIES)
    if counts.empty:
        raise RuntimeError("No experimental point-cloud trajectories were found.")
    counts = apply_sample_exclusions(counts, column="sample", cutoff_column="day_tp1", strict=False)
    helpers = load_ttf_helpers()
    policy = helpers.load_experimental_agent()
    PLOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    groups = list(counts.groupby(["plate", "sample", "strategy"], sort=True))
    with PdfPages(PLOT_OUTPUT) as pdf:
        for start in range(0, len(groups), PANELS_PER_PAGE):
            batch = groups[start:start + PANELS_PER_PAGE]
            figure, axes = ft.new_subplots(
                3, 3, ax_width_mm=AXES_WIDTH_MM, ax_height_mm=AXES_HEIGHT_MM,
                margins=MARGINS_MM, wspace_mm=3.0, hspace_mm=4.0, dpi=DPI,
                apply_style=False,
            )
            axes = np.asarray(axes).reshape(-1)
            for axis, (plate, sample, strategy), group in zip(axes, *zip(*batch)):
                group = group.sort_values("day_tp1").copy()
                lv = simulate_lv_trajectory(group, strategy=strategy, helpers=helpers, policy=policy)
                plot_axis(axis, group, lv, strategy=strategy)
                initial_total = float(group.iloc[0]["total"])
                exp_out = group[["day_tp1", "sus", "res"]].copy()
                exp_out["source"] = "In vitro"
                exp_out.rename(columns={"sus": "sensitive_normalized", "res": "resistant_normalized"}, inplace=True)
                exp_out["sensitive_normalized"] /= initial_total
                exp_out["resistant_normalized"] /= initial_total
                lv_out = lv.rename(columns={"lv_sensitive": "sensitive_normalized", "lv_resistant": "resistant_normalized"})
                lv_out = lv_out[["day_tp1", "sensitive_normalized", "resistant_normalized", "treatment_on"]]
                lv_out["source"] = "LV"
                for frame in (exp_out, lv_out):
                    frame["plate"], frame["sample"], frame["strategy"] = plate, sample, strategy
                trajectory_rows.extend([exp_out, lv_out])
                audit_rows.append({"plate": plate, "sample": sample, "strategy": strategy,
                                   "initial_total": initial_total, "last_invitro_day": float(group.day_tp1.max()),
                                   "lv_last_day": float(lv.day_tp1.max()), "n_invitro_points": len(group),
                                   "n_lv_points": len(lv)})
            for axis in axes[len(batch):]:
                axis.set_visible(False)
            figure.legend(handles=[
                Line2D([], [], color=SENSITIVE_COLOR, linewidth=1.0, label="Sensitive"),
                Line2D([], [], color=RESISTANT_COLOR, linewidth=1.0, label="Resistant"),
                Line2D([], [], color="#333333", marker="o", markersize=2.2, linewidth=1.0, label="In vitro"),
                Line2D([], [], color="#333333", linestyle="--", linewidth=1.0, label="LV"),
            ], frameon=False, ncol=4, loc="upper center", fontsize=5, handlelength=1.5,
               columnspacing=0.8, handletextpad=0.35)
            pdf.savefig(figure, dpi=DPI)
            plt.close(figure)
    pd.concat(trajectory_rows, ignore_index=True).to_csv(VALUES_OUTPUT, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_OUTPUT, index=False)
    print(f"[saved] {PLOT_OUTPUT} ({len(audit_rows)} panels across {(len(audit_rows) + PANELS_PER_PAGE - 1) // PANELS_PER_PAGE} pages)")
    print(f"[saved] {VALUES_OUTPUT}")
    print(f"[saved] {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()
