"""Generate the corrected Figure 4d bootstrap policy-behaviour panel.

The panel is rebuilt directly from the preserved raw bootstrap policy-output
table. It uses exactly the 20 ``bootstrap_lv_*`` agents, retains only valid
synthetic observations where ``base + peak_addition <= 3`` (without clipping),
and plots both five exact action histories and two groups defined by their
latest action. The two-group terminal-action panel additionally overlays the
historical experimental agent as matching thinner dashed curves.

Run from the repository root::

    .\\PhysiLearning\\LV_Agent_Training_MiniRepo\\.venv\\Scripts\\python.exe \\
      figures\\Figure_4\\d2_replot_rl_policy_behavior.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.data_paths import figure_data_dir, figure_plot_dir
from shared import figtools as ft

RAW_BOOTSTRAP_PATH = figure_data_dir(
    "Figure_4", "rl_policy_behavior", create=False
) / "bootstrap_agents_policy_behavior.csv"
RAW_EXPERIMENTAL_PATH = figure_data_dir(
    "Figure_4", "rl_policy_behavior", create=False
) / "experimental_agent_policy_behavior.csv"
CANONICAL_OUTPUT_DIR = figure_plot_dir("Figure_4", "rl_policy_behavior", create=False)
CANONICAL_DATA_DIR = figure_data_dir("Figure_4", "rl_policy_behavior", create=False)
# Valid synthetic probes span 0--9: base + peak_addition <= 3 and peak_day <= 3.
BIN_EDGES = np.linspace(0.0, 9.0, 15)
EXACT_HISTORY_ORDER = ("[0,0,0,0]", "[0,0,0,1]", "[0,0,1,1]", "[0,1,1,1]", "[1,1,1,1]")
HISTORY_COLORS = {
    "[0,0,0,0]": "#b22222", "[0,0,0,1]": "#4c9f70", "[0,0,1,1]": "#d67c1c",
    "[0,1,1,1]": "#6b4c9a", "[1,1,1,1]": "#1f4f7a",
}
TERMINAL_ACTION_HISTORY_ORDER = ("[*,*,*,0]", "[*,*,*,1]")
TERMINAL_ACTION_HISTORY_COLORS = {
    "[*,*,*,0]": "#b22222",
    "[*,*,*,1]": "#1f4f7a",
}
# Physical layout: the plotted axes (not the PDF page) must be 20 x 20 mm.
# The extra right margin holds the two legends outside the data rectangle.
AXES_WIDTH_MM = 20.0
AXES_HEIGHT_MM = 20.0
MARGINS_MM = (14.0, 40.0, 10.0, 2.0)
CURVE_LINEWIDTH = 0.9
EXPERIMENTAL_CURVE_LINEWIDTH = 0.65
LEGEND_LINEWIDTH = 0.9


def canonical_history(value: object) -> str:
    """Normalize stored action histories to the Figure 4 bracketed form."""
    digits = [character for character in str(value) if character in "01"]
    if len(digits) != 4:
        raise ValueError(f"Expected a four-action history, got {value!r}.")
    return "[" + ",".join(digits) + "]"


def validate_raw_bootstrap(table: pd.DataFrame) -> None:
    required = {"agent_id", "action_history", "base", "peak_addition", "peak_day", "growth", "chosen_action"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Raw bootstrap policy table is missing required columns: {missing}")
    expected_growth = table["base"].astype(float) + table["peak_day"].astype(float) * table["peak_addition"].astype(float)
    if not np.allclose(table["growth"].astype(float), expected_growth, rtol=0, atol=1e-12):
        raise ValueError("Raw table does not use growth = base + peak_day * peak_addition.")


def prepare_valid_histories(
    raw_bootstrap: pd.DataFrame, *, group_by_terminal_action: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Exclude invalid rows and assign the verified fixed growth bins.

    ``searchsorted(..., side='right')`` sends observations exactly on an
    interior edge to the bin on their right, while retaining the maximum in
    the final bin. This is the verified historical bin convention.
    """
    bootstrap_ids = sorted(raw_bootstrap.loc[raw_bootstrap["agent_id"].str.startswith("bootstrap_lv_"), "agent_id"].unique())
    expected_ids = [f"bootstrap_lv_{index:04d}" for index in range(20)]
    if bootstrap_ids != expected_ids:
        raise ValueError(f"Expected bootstrap_lv_0000 through bootstrap_lv_0019, found {bootstrap_ids}.")
    frame = raw_bootstrap.loc[raw_bootstrap["agent_id"].isin(bootstrap_ids)].copy()
    frame["history"] = frame["action_history"].map(canonical_history)
    # This is a filter; observations are never clipped to the domain boundary.
    frame = frame.loc[(frame["base"] >= 0) & (frame["peak_addition"] >= 0) & (frame["base"] + frame["peak_addition"] <= 3)].copy()
    if group_by_terminal_action:
        frame["history"] = np.where(
            frame["history"].str.endswith(",0]"), "[*,*,*,0]", "[*,*,*,1]",
        )
    else:
        frame = frame.loc[frame["history"].isin(EXACT_HISTORY_ORDER)].copy()
    frame["treatment_on"] = (frame["chosen_action"].astype(int) == 1).astype(float)
    growth = frame["growth"].to_numpy(dtype=float)
    if (growth < BIN_EDGES[0]).any() or (growth > BIN_EDGES[-1]).any():
        raise ValueError("An included growth score lies outside the fixed 0--20 bin range.")
    bin_index = np.searchsorted(BIN_EDGES, growth, side="right") - 1
    # The sole right-most edge (growth == 20) belongs to the final bin. This
    # bin assignment is not a modification of any synthetic observation.
    bin_index[growth == BIN_EDGES[-1]] = len(BIN_EDGES) - 2
    if (bin_index < 0).any() or (bin_index >= len(BIN_EDGES) - 1).any():
        raise ValueError("Could not assign an included growth score to a fixed bin.")
    frame["growth_bin"] = bin_index.astype(int)
    frame["growth_bin_center"] = (BIN_EDGES[frame["growth_bin"].to_numpy()] + BIN_EDGES[frame["growth_bin"].to_numpy() + 1]) / 2
    return frame, bootstrap_ids


def aggregate_bootstrap(valid_rows: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-agent fractions, then the 20-agent mean and sample SD."""
    per_agent = (
        valid_rows.groupby(["history", "agent_id", "growth_bin", "growth_bin_center"], as_index=False)
        .agg(treatment_on_fraction=("treatment_on", "mean"), n_observations_per_agent=("treatment_on", "size"))
    )
    count_variability = per_agent.groupby(["history", "growth_bin"])["n_observations_per_agent"].nunique()
    if (count_variability > 1).any():
        raise ValueError("Bootstrap agents do not share the same valid observation grid within a history/bin.")
    curves = (
        per_agent.groupby(["history", "growth_bin", "growth_bin_center"], as_index=False)
        .agg(
            n_bootstrap_agents=("agent_id", "nunique"),
            n_observations_per_agent=("n_observations_per_agent", "first"),
            n_observations_total=("n_observations_per_agent", "sum"),
            treatment_on_fraction_mean=("treatment_on_fraction", "mean"),
            treatment_on_fraction_sd=("treatment_on_fraction", "std"),
        )
        .sort_values(["history", "growth_bin"])
        .reset_index(drop=True)
    )
    if not (curves["n_bootstrap_agents"] == 20).all():
        raise ValueError("Every displayed bin must contain all 20 bootstrap agents.")
    return curves


def aggregate_experimental_terminal_action(raw_experimental: pd.DataFrame) -> pd.DataFrame:
    """Bin the selected experimental agent using the Figure 4d2 convention."""
    validate_raw_bootstrap(raw_experimental)
    frame = raw_experimental.copy()
    frame["history"] = frame["action_history"].map(canonical_history)
    frame = frame.loc[
        (frame["base"] >= 0)
        & (frame["peak_addition"] >= 0)
        & (frame["base"] + frame["peak_addition"] <= 3)
    ].copy()
    frame["history"] = np.where(
        frame["history"].str.endswith(",0]"), "[*,*,*,0]", "[*,*,*,1]",
    )
    frame["treatment_on"] = (frame["chosen_action"].astype(int) == 1).astype(float)
    growth = frame["growth"].to_numpy(dtype=float)
    if (growth < BIN_EDGES[0]).any() or (growth > BIN_EDGES[-1]).any():
        raise ValueError("An included experimental growth score lies outside the fixed 0--9 bin range.")
    bin_index = np.searchsorted(BIN_EDGES, growth, side="right") - 1
    bin_index[growth == BIN_EDGES[-1]] = len(BIN_EDGES) - 2
    frame["growth_bin"] = bin_index.astype(int)
    frame["growth_bin_center"] = (
        BIN_EDGES[frame["growth_bin"].to_numpy()]
        + BIN_EDGES[frame["growth_bin"].to_numpy() + 1]
    ) / 2
    return (
        frame.groupby(["history", "growth_bin", "growth_bin_center"], as_index=False)
        .agg(
            treatment_on_fraction=("treatment_on", "mean"),
            n_observations=("treatment_on", "size"),
        )
        .sort_values(["history", "growth_bin"])
        .reset_index(drop=True)
    )


def plot_figure(
    curves: pd.DataFrame,
    output_base: Path,
    *,
    history_order: tuple[str, ...],
    history_colors: dict[str, str],
    experimental_curves: pd.DataFrame | None = None,
) -> None:
    """Render the single-panel five-history bootstrap mean ± SD panel."""
    # Arial has no Unicode Latin-letter subscript glyphs.  Use MathText for
    # the baseline shift, while mapping its roman glyphs explicitly to Arial.
    plt.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })
    figure, axis = ft.new_figure(
        width=AXES_WIDTH_MM,
        height_mm=AXES_HEIGHT_MM,
        margins=MARGINS_MM,
        dpi=300,
        apply_style=True,
    )
    for history in history_order:
        subset = curves.loc[curves["history"] == history].sort_values("growth_bin_center")
        color = history_colors[history]
        x = subset["growth_bin_center"].to_numpy()
        mean = subset["treatment_on_fraction_mean"].to_numpy()
        sd = subset["treatment_on_fraction_sd"].fillna(0).to_numpy()
        axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        axis.plot(x, mean, color=color, linewidth=CURVE_LINEWIDTH)
    if experimental_curves is not None:
        for history in history_order:
            subset = experimental_curves.loc[
                experimental_curves["history"] == history
            ].sort_values("growth_bin_center")
            axis.plot(
                subset["growth_bin_center"],
                subset["treatment_on_fraction"],
                color=history_colors[history],
                linestyle="--",
                linewidth=EXPERIMENTAL_CURVE_LINEWIDTH,
            )
    # Keep both axis labels as a single text run in the PDF.  MathText uses
    # separately positioned glyphs, which Illustrator imports as individual
    # objects.
    axis.set(
        xlim=(0, 9), ylim=(0, 1.02),
        xlabel="Growth score",
        ylabel="Treatment probability",
    )
    # Keep the plain-text label in one editable Arial PDF text run.
    axis.yaxis.label.set_fontfamily("Arial")
    axis.set_xticks((0, 3, 6, 9))
    axis.set_yticks(np.linspace(0, 1, 6), labels=("0", "20", "40", "60", "80", "100"))
    axis.spines[["top", "right"]].set_visible(False)
    history_handles = [
        Line2D([0], [0], color=history_colors[history], linewidth=LEGEND_LINEWIDTH, label=history)
        for history in history_order
    ]
    style_handle = Patch(facecolor="#777777", alpha=0.20, label="Bootstrap mean ± 1 SD (n=20)")
    style_handles = [style_handle]
    if experimental_curves is not None:
        style_handles.append(
            Line2D(
                [0], [0], color="#000000", linestyle="--",
                linewidth=EXPERIMENTAL_CURVE_LINEWIDTH, label="ATAgent",
            )
        )
    color_legend = axis.legend(
        handles=history_handles, title="Action history", loc="upper left",
        bbox_to_anchor=(1.02, 1.0), frameon=False, handlelength=1.7,
        borderaxespad=0.0,
    )
    axis.add_artist(color_legend)
    axis.legend(
        handles=style_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
        frameon=False, handlelength=1.5, borderaxespad=0.0,
    )
    ft.format_axes(axis)
    out_of_range_text = ft.audit_text_sizes(figure)
    if out_of_range_text:
        raise ValueError(f"Figure text outside the 5-7 pt guideline: {out_of_range_text}")
    ft.savefig(figure, output_base.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bootstrap", type=Path, default=RAW_BOOTSTRAP_PATH)
    parser.add_argument("--raw-experimental", type=Path, default=RAW_EXPERIMENTAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    parser.add_argument("--data-output-dir", type=Path, default=CANONICAL_DATA_DIR)
    arguments = parser.parse_args()
    raw_bootstrap = pd.read_csv(arguments.raw_bootstrap)
    validate_raw_bootstrap(raw_bootstrap)
    valid_rows, _ = prepare_valid_histories(raw_bootstrap)
    curves = aggregate_bootstrap(valid_rows)
    terminal_action_rows, _ = prepare_valid_histories(
        raw_bootstrap, group_by_terminal_action=True,
    )
    terminal_action_curves = aggregate_bootstrap(terminal_action_rows)
    raw_experimental = pd.read_csv(arguments.raw_experimental)
    experimental_terminal_action_curves = aggregate_experimental_terminal_action(raw_experimental)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = arguments.data_output_dir.resolve()
    data_output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(data_output_dir / "bootstrap_policy_behavior_replot_source_data.csv", index=False)
    terminal_action_curves.to_csv(
        data_output_dir / "bootstrap_policy_behavior_terminal_action_source_data.csv", index=False,
    )
    experimental_terminal_action_curves.to_csv(
        data_output_dir / "experimental_policy_behavior_terminal_action_source_data.csv", index=False,
    )
    plot_figure(
        curves, output_dir / "bootstrap_policy_behavior_replot",
        history_order=EXACT_HISTORY_ORDER, history_colors=HISTORY_COLORS,
    )
    plot_figure(
        terminal_action_curves, output_dir / "bootstrap_policy_behavior_terminal_action_replot",
        history_order=TERMINAL_ACTION_HISTORY_ORDER,
        history_colors=TERMINAL_ACTION_HISTORY_COLORS,
        experimental_curves=experimental_terminal_action_curves,
    )
    print(f"Corrected Figure 4d bootstrap panel written to: {output_dir}")


if __name__ == "__main__":
    main()
