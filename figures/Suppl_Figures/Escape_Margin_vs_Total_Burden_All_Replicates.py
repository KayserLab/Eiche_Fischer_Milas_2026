#!/usr/bin/env python3
"""Plot Figure-2g-style pre-escape trajectories for experimental and DPD runs.

The five rows show CT, AT50, AT100, ATagent, and AT+; each panel shows one fixed
experimental or simulation example. Experimental and simulation examples are
not all matched pairs. Only the negative signed-margin phase before escape is
retained and is mirrored above the x-axis as a positive confinement margin.
Within each panel, one connecting line represents one replicate. Lines use the
canonical strategy colour; points use Figure 2g's linear black-yellow-white scale
for normalised resistant-cell count.

Fully excluded replicates are omitted and partially usable replicates are
stopped at their ``experiment_filters`` cutoff. The PDF and its exact plotted
values are saved in the canonical supplementary figure folders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, figure_plot_dir, intermediate_dir
from shared.experiment_filters import EXCLUDE_SAMPLES, apply_sample_exclusions
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color, get_display_name, normalize_strategy


ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
DATA_OUTPUT = (
    figure_data_dir("Suppl_Figures", "escape_margin_vs_total_burden", create=False)
    / "escape_margin_vs_total_burden_all_retained_replicates.csv"
)
PLOT_OUTPUT = (
    figure_plot_dir("Suppl_Figures", "escape_margin_vs_total_burden", create=False)
    / "Escape_Margin_vs_Total_Burden_All_Replicates.pdf"
)

STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")
# Preserve the ten examples in Supplementary Figure 3. This selection is
# independent of the larger observed-failure example sets in other figures.
SELECTED_EXAMPLES = {
    "experimental": {
        "CT": ("Plate_3", "B2_CT-B2"),
        "AT50": ("Plate_2", "B2_AT50-B2"),
        "AT100": ("Plate_5", "A5_AT100-A5"),
        "LV-Agent": ("Plate_6", "B4_ATAgent2-B4"),
        "AT100+": ("Plate_8", "D3_Dens-D3"),
    },
    "simulation": {
        "CT": ("Plate_CT", "R32_CT-R32"),
        "AT50": ("Plate_AT50", "R11_AT50-R11"),
        "AT100": ("Plate_AT100", "R23_AT100-R23"),
        "LV-Agent": ("Plate_Agent", "R44_Agent-R44"),
        "AT100+": ("Plate_Dens", "R48_Dens-R48"),
    },
}
# Shared display limits for direct Exp-versus-DPD comparison.  As in Figure 2g,
# the negative confinement phase is mirrored above the x-axis and ends at escape.
X_LIMITS = (0.0, 2.7)
Y_LIMITS = (0.0, 0.5)
AXIS_WIDTH_MM = 38.0
AXIS_HEIGHT_MM = 22.0
MARGINS_MM = (18.0, 12.0, 14.0, 12.0)
COLUMN_GAP_MM = 10.0
ROW_GAP_MM = 4.0
LINEWIDTH = 1.0
POINT_SIZE = 14.0
ARROWHEAD_SIZE = 5.0
DPI = 300
SHOW_FIGURE = True

ft.apply_nature_style(
    text_pt=7.0, tick_pt=6.0, legend_pt=5.0,
    overrides={"axes.linewidth": 0.8, "xtick.major.width": 0.8,
               "ytick.major.width": 0.8, "xtick.major.size": 2.5,
               "ytick.major.size": 2.5},
)


def burden_path(plate: str, sample: str) -> Path:
    well, raw_strategy = sample.split("_", 1)
    return EXP_RAW_DATA / plate / "Results" / f"Results_G418_a2a4_{well}_{raw_strategy}.txt"


def load_margins(dataset_type: str) -> pd.DataFrame:
    """Load one source's margin rows; apply manual filters to Exp only."""
    if not ESCAPE_MARGIN_CSV.is_file():
        raise FileNotFoundError(
            f"Canonical escape-margin table not found: {ESCAPE_MARGIN_CSV}. Run "
            "analysis/figure_2/Escape_Margin_Analysis.py first."
        )
    margins = pd.read_csv(ESCAPE_MARGIN_CSV)
    required = {"dataset_type", "plate", "sample", "strategy", "day", "signed_escape_margin"}
    missing = required.difference(margins.columns)
    if missing:
        raise ValueError(f"{ESCAPE_MARGIN_CSV} is missing columns {sorted(missing)}")
    dataset_type = dataset_type.lower()
    if dataset_type not in {"experimental", "simulation"}:
        raise ValueError(f"Unsupported dataset type {dataset_type!r}")
    margins["dataset_type"] = margins["dataset_type"].astype(str).str.lower()
    margins = margins.loc[margins["dataset_type"].eq(dataset_type)].copy()
    margins["strategy"] = margins["strategy"].astype(str).map(normalize_strategy)
    margins["day"] = pd.to_numeric(margins["day"], errors="coerce")
    margins["signed_escape_margin"] = pd.to_numeric(margins["signed_escape_margin"], errors="coerce")
    margins = margins.loc[
        margins["strategy"].isin(STRATEGIES)
        & np.isfinite(margins["day"])
        & np.isfinite(margins["signed_escape_margin"])
    ].copy()
    margins = margins.drop_duplicates(subset=["plate", "sample", "day"], keep="last")
    if dataset_type == "experimental":
        margins = apply_sample_exclusions(
            margins, exclude_map=EXCLUDE_SAMPLES, column="sample", cutoff_column="day", strict=False,
        )
    return margins


def load_plot_values() -> pd.DataFrame:
    """Build normalised values for this figure's ten fixed examples."""
    margins = load_margins("experimental")
    frames: list[pd.DataFrame] = []
    missing_burdens: list[Path] = []
    for (plate, sample, strategy), margin in margins.groupby(["plate", "sample", "strategy"], sort=True):
        path = burden_path(str(plate), str(sample))
        if not path.is_file():
            missing_burdens.append(path)
            continue
        burden = pd.read_csv(path, sep=r"\s+", header=None, names=["token", "sensitive", "resistant"])
        burden["day"] = pd.to_numeric(
            burden["token"].astype(str).str.extract(r"_d(\d+)$", expand=False), errors="coerce"
        )
        burden["sensitive"] = pd.to_numeric(burden["sensitive"], errors="coerce")
        burden["resistant"] = pd.to_numeric(burden["resistant"], errors="coerce")
        burden = burden.dropna(subset=["day", "sensitive", "resistant"]).sort_values("day")
        if burden.empty:
            continue
        baseline = float(burden.iloc[0]["sensitive"] + burden.iloc[0]["resistant"])
        if baseline <= 0:
            continue
        data = margin.merge(burden[["day", "sensitive", "resistant"]], on="day", how="inner")
        if data.empty:
            continue
        data["normalized_total"] = (data["sensitive"] + data["resistant"]) / baseline
        data["normalized_resistant"] = data["resistant"] / baseline
        data["normalized_escape_margin"] = data["signed_escape_margin"] / baseline
        data["initial_total"] = baseline
        data["dataset_type"] = "experimental"
        frames.append(data)
    if missing_burdens:
        print(f"[warning] {len(missing_burdens)} retained replicate(s) had no Results burden file and were skipped.")
    if not frames:
        raise RuntimeError("No matched retained experimental trajectories are available.")
    experimental = pd.concat(frames, ignore_index=True)

    simulation = load_margins("simulation")
    required_counts = {"n_sensitive_retained", "n_resistant_retained"}
    missing_counts = required_counts.difference(simulation.columns)
    if missing_counts:
        raise ValueError(
            f"{ESCAPE_MARGIN_CSV} is missing simulation count columns {sorted(missing_counts)}"
        )
    simulation["initial_total"] = simulation.groupby(["plate", "sample"])[
        ["n_sensitive_retained", "n_resistant_retained"]
    ].transform("first").sum(axis=1)
    simulation = simulation.loc[simulation["initial_total"] > 0].copy()
    simulation["normalized_total"] = (
        simulation["n_sensitive_retained"] + simulation["n_resistant_retained"]
    ) / simulation["initial_total"]
    simulation["normalized_resistant"] = (
        simulation["n_resistant_retained"] / simulation["initial_total"]
    )
    simulation["normalized_escape_margin"] = (
        simulation["signed_escape_margin"] / simulation["initial_total"]
    )
    experimental_days = set(experimental["day"].unique())
    simulation = simulation.loc[simulation["day"].isin(experimental_days)].copy()
    combined = pd.concat([experimental, simulation], ignore_index=True).sort_values(
        ["dataset_type", "strategy", "plate", "sample", "day"]
    )
    identities = pd.DataFrame([
        dict(dataset_type=dataset_type, strategy=strategy, plate=plate, sample=sample)
        for dataset_type, examples in SELECTED_EXAMPLES.items()
        for strategy, (plate, sample) in examples.items()
    ])
    combined = combined.merge(
        identities,
        on=["dataset_type", "strategy", "plate", "sample"],
        how="inner",
    )
    preescape_frames: list[pd.DataFrame] = []
    for _identity, trajectory in combined.groupby(
        ["dataset_type", "strategy", "plate", "sample"], sort=True,
    ):
        trajectory = trajectory.sort_values("day").copy()
        escaped = trajectory.loc[trajectory["normalized_escape_margin"] > 0.0, "day"]
        escape_day = float(escaped.min()) if not escaped.empty else np.nan
        if np.isfinite(escape_day):
            trajectory = trajectory.loc[trajectory["day"] < escape_day].copy()
        trajectory = trajectory.loc[trajectory["normalized_escape_margin"] <= 0.0].copy()
        if trajectory.empty:
            continue
        trajectory["normalized_confinement_margin"] = -trajectory["normalized_escape_margin"]
        trajectory["escape_day"] = escape_day
        preescape_frames.append(trajectory)
    if not preescape_frames:
        raise RuntimeError("No pre-escape confinement trajectories are available.")
    return pd.concat(preescape_frames, ignore_index=True).sort_values(
        ["dataset_type", "strategy", "plate", "sample", "day"]
    )


def add_segment_arrows(axis: plt.Axes, trajectory: pd.DataFrame, color: str) -> None:
    """Draw Figure 2g-style directional arrows between observations."""
    points = trajectory[["normalized_total", "normalized_confinement_margin"]].to_numpy(dtype=float)
    if len(points) < 2:
        return
    for startpoint, endpoint in zip(points[:-1], points[1:]):
        if np.allclose(startpoint, endpoint):
            continue
        arrow_start = startpoint + 0.52 * (endpoint - startpoint)
        arrow_end = startpoint + 0.80 * (endpoint - startpoint)
        axis.annotate(
            "", xy=arrow_end, xytext=arrow_start,
            arrowprops={
                "arrowstyle": "-|>", "color": color, "lw": LINEWIDTH,
                "mutation_scale": ARROWHEAD_SIZE, "shrinkA": 0, "shrinkB": 0,
            },
            zorder=2.05,
        )


def plot(values: pd.DataFrame, output: Path) -> None:
    resistant_max = float(values["normalized_resistant"].max())
    if not np.isfinite(resistant_max) or resistant_max <= 0:
        raise RuntimeError("No positive resistant counts are available for point colours.")
    norm = Normalize(vmin=0.0, vmax=resistant_max)
    cmap = LinearSegmentedColormap.from_list(
        "resistant_black_yellow_white",
        [(0.0, "#000000"), (0.2, "#FFE000"), (1.0, "#FFFFFF")],
    )
    figure, axes = ft.new_subplots(
        len(STRATEGIES), 2, ax_width_mm=AXIS_WIDTH_MM, ax_height_mm=AXIS_HEIGHT_MM,
        margins=MARGINS_MM, wspace_mm=COLUMN_GAP_MM, hspace_mm=ROW_GAP_MM,
        dpi=DPI, sharex=True, sharey=True,
        apply_style=False,
    )
    axes = np.asarray(axes).reshape(len(STRATEGIES), 2)
    for row, strategy in enumerate(STRATEGIES):
        for column, (dataset_type, column_label) in enumerate((
            ("experimental", "Exp example"), ("simulation", "Sim example"),
        )):
            axis = axes[row, column]
            panel = values.loc[
                values["dataset_type"].eq(dataset_type) & values["strategy"].eq(strategy)
            ]
            for (_plate, _sample), trajectory in panel.groupby(["plate", "sample"], sort=True):
                trajectory = trajectory.sort_values("day")
                resistant = trajectory["normalized_resistant"].to_numpy(dtype=float)
                axis.plot(
                    trajectory["normalized_total"],
                    trajectory["normalized_confinement_margin"],
                    color=get_color(strategy, "#777777"), linewidth=LINEWIDTH, zorder=2,
                )
                axis.scatter(trajectory["normalized_total"], trajectory["normalized_confinement_margin"],
                             c=resistant, cmap=cmap, norm=norm, s=POINT_SIZE,
                             edgecolors="black", linewidths=0.2, zorder=2.1)
                add_segment_arrows(axis, trajectory, get_color(strategy, "#777777"))
            axis.axvline(1.0, color="#111111", linewidth=0.55, linestyle=(0, (3, 2)), zorder=1)
            if strategy == "AT50":
                axis.axvline(0.5, color=get_color("AT50"), linewidth=0.7,
                            linestyle=(0, (1, 1.5)), zorder=1)
            axis.set(xlim=X_LIMITS, ylim=Y_LIMITS)
            axis.set_ylabel(
                r"$\eta^{\mathrm{conf}}_{\mathrm{sen}}$",
                fontsize=7, color="#c8a2c8", labelpad=2,
            )
            axis.xaxis.set_major_locator(MultipleLocator(0.5))
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(column_label, fontsize=7, pad=3)
            if column == 1:
                axis.tick_params(axis="y", labelleft=False)
            else:
                axis.text(
                    0.02, 0.94, get_display_name(strategy, superscript=True),
                    transform=axis.transAxes, ha="left", va="top",
                    color=get_color(strategy), fontsize=5.5,
                )
    figure.supxlabel(r"$\eta_{\mathrm{tot}}$", y=0.025, fontsize=7)
    left_mm, right_mm, bottom_mm, top_mm = MARGINS_MM
    panel_span_mm = 2 * AXIS_WIDTH_MM + COLUMN_GAP_MM
    canvas_width_mm = figure.get_figwidth() * 25.4
    canvas_height_mm = figure.get_figheight() * 25.4
    colorbar_ax = figure.add_axes([
        (left_mm + panel_span_mm + 3.0) / canvas_width_mm,
        bottom_mm / canvas_height_mm,
        2.0 / canvas_width_mm,
        (5 * AXIS_HEIGHT_MM + 4 * ROW_GAP_MM) / canvas_height_mm,
    ])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax,
    )
    colorbar.set_label(r"$\eta_{\mathrm{res}}$", fontsize=7, color=FIGURE_COLOR_MAP["yellow"], labelpad=1)
    colorbar.ax.tick_params(labelsize=5, length=2)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_rotation(45)
        tick_label.set_ha("left")
        tick_label.set_rotation_mode("anchor")
    output.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(figure, output)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)


def main() -> None:
    values = load_plot_values()
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(DATA_OUTPUT, index=False)
    plot(values, PLOT_OUTPUT)
    print(f"[saved] {DATA_OUTPUT}")
    print(f"[saved] {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
