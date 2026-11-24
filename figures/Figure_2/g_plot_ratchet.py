"""Single selected-replicate escape-margin panel for Figure 2.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

Each trajectory shows the positive magnitude of the pre-escape confinement
margin. It ends before the first positive escape-margin observation and uses a
treatment-coloured connecting line with smaller points colour-mapped linearly
by resistant-cell count (black to DAA520).
"""

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: canonical all_escape_margin.csv and matching Plate_*/Results/*.txt
# burden trajectories.
# Scripts: analysis/figure_2/Escape_Margin_Analysis.py,
# shared/experiment_filters.py, shared/strategy_colors.py, and shared/figtools.py.
# Outputs: linearly coloured pre-escape confinement-margin PDF and values CSV.

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from shared import figtools as ft
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color, get_display_name
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, intermediate_dir
OUTPUT_DIR = ft.paper_figure_data_dir(__file__)
DATA_OUTPUT_DIR = figure_data_dir("Figure_2", "ratchet", create=False)
OUTPUT_PDF = OUTPUT_DIR / "CT_AT50_AT100_selected_Confinement_Margin_vs_Normalized_Total_Count_resistant_linear.pdf"
OUTPUT_VALUES = DATA_OUTPUT_DIR / "CT_AT50_AT100_selected_Confinement_Margin_vs_Normalized_Total_Count_resistant_linear_values.csv"
EXPERIMENT_ROOT = EXP_RAW_DATA
ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
SELECTED = (
    ("CT", "Plate_3", "B2_CT-B2"),
    ("AT50", "Plate_2", "B2_AT50-B2"),
    ("AT100", "Plate_5", "A5_AT100-A5"),
)
PLOT_ORDER = ("CT", "AT50", "AT100")
X_LIMITS = (0.0, 1.8)
Y_LIMITS = (0.0, 0.32)
AXES_WIDTH_MM = 18.0
AXES_HEIGHT_MM = 35.0
PANEL_GAP_MM = 5.0
# Compact right-side canvas while retaining room for the colour-bar title.
MARGINS_MM = (22.0, 17.5, 15.0, 5.0)

# Match the treatment-threshold guides in Figure 1i/k/m.
THRESHOLD_LINEWIDTH = 1.5
THRESHOLD_DASH_POINTS = 4.0
THRESHOLD_GAP_POINTS = 5.0
THRESHOLD_DOT_POINTS = 0.8
THRESHOLD_DASH_STYLE = (0.0, (THRESHOLD_DASH_POINTS, THRESHOLD_GAP_POINTS))
THRESHOLD_DOT_STYLE = (
    -(THRESHOLD_DASH_POINTS + (THRESHOLD_GAP_POINTS - THRESHOLD_DOT_POINTS) / 2.0),
    (
        THRESHOLD_DOT_POINTS,
        THRESHOLD_DASH_POINTS + THRESHOLD_GAP_POINTS - THRESHOLD_DOT_POINTS,
    ),
)

ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)


def burden_path(plate: str, sample: str) -> Path:
    well, strategy = sample.split("_", 1)
    return EXPERIMENT_ROOT / plate / "Results" / f"Results_G418_a2a4_{well}_{strategy}.txt"


def load_escape_margins() -> pd.DataFrame:
    """Load the canonical experimental escape-margin trajectories."""
    if not ESCAPE_MARGIN_CSV.is_file():
        raise FileNotFoundError(
            f"Canonical escape-margin table not found: {ESCAPE_MARGIN_CSV}. Run "
            "analysis/figure_2/Escape_Margin_Analysis.py first."
        )
    margins = pd.read_csv(ESCAPE_MARGIN_CSV)
    required = {"dataset_type", "plate", "sample", "day", "signed_escape_margin"}
    missing = required.difference(margins.columns)
    if missing:
        raise ValueError(f"{ESCAPE_MARGIN_CSV} is missing columns {sorted(missing)}")
    margins = margins[
        margins["dataset_type"].astype(str).str.lower().eq("experimental")
    ].copy()
    margins["day"] = pd.to_numeric(margins["day"], errors="coerce")
    margins["signed_escape_margin"] = pd.to_numeric(
        margins["signed_escape_margin"], errors="coerce"
    )
    margins = margins.dropna(subset=["day", "signed_escape_margin"])
    return margins.drop_duplicates(
        subset=["plate", "sample", "day"], keep="last"
    ).sort_values(["plate", "sample", "day"])


def load_data() -> pd.DataFrame:
    margins = load_escape_margins()
    frames = []
    for strategy, plate, sample in SELECTED:
        burden = pd.read_csv(burden_path(plate, sample), sep=r"\s+", header=None,
                             names=["token", "sensitive", "resistant"])
        burden["day"] = pd.to_numeric(
            burden["token"].str.extract(r"_d(\d+)$", expand=False), errors="coerce"
        )
        burden = burden.dropna(subset=["day", "sensitive", "resistant"]).sort_values("day")
        margin = margins[(margins["plate"] == plate) & (margins["sample"] == sample)].copy()

        last_available_day = min(float(margin["day"].max()), float(burden["day"].max()))
        cutoff = EXCLUDE_SAMPLES.get(sample, float("inf"))
        if cutoff is None:
            continue
        stop_day = min(float(cutoff), last_available_day)
        margin = margin[margin["day"] <= stop_day].copy()
        burden = burden[burden["day"] <= stop_day].copy()
        baseline = float(burden.iloc[0]["sensitive"] + burden.iloc[0]["resistant"])
        margin["normalized_escape_margin"] = margin["signed_escape_margin"] / baseline
        burden["normalized_total"] = (burden["sensitive"] + burden["resistant"]) / baseline
        burden["normalized_resistant"] = burden["resistant"] / baseline
        data = margin.merge(
            burden[["day", "normalized_resistant", "normalized_total"]],
            on="day", how="inner",
        )
        escaped = data.loc[data["normalized_escape_margin"] > 0.0, "day"]
        escape_day = float(escaped.min()) if not escaped.empty else float("nan")
        if escape_day == escape_day:
            data = data[data["day"] < escape_day].copy()
        # Retain only the confinement phase and mirror it above the x-axis.
        data = data[data["normalized_escape_margin"] <= 0.0].copy()
        data["normalized_confinement_margin"] = -data["normalized_escape_margin"]
        data["escape_day"] = escape_day
        data["strategy"] = strategy
        frames.append(data)
    return pd.concat(frames, ignore_index=True).sort_values(["strategy", "day"])


def add_trajectory(ax, data: pd.DataFrame, norm: Normalize, cmap, *, zorder: float) -> None:
    """Draw a strategy-coloured trajectory with arrows and resistance-coloured points."""
    x = data["normalized_total"].to_numpy(float)
    y = data["normalized_confinement_margin"].to_numpy(float)
    resistant = data["normalized_resistant"].to_numpy(float)
    color = get_color(data["strategy"].iloc[0])
    ax.plot(x, y, color=color, lw=1.0, zorder=zorder)
    for start_x, start_y, end_x, end_y in zip(x[:-1], y[:-1], x[1:], y[1:]):
        if start_x == end_x and start_y == end_y:
            continue
        arrow_start = (start_x + 0.52 * (end_x - start_x), start_y + 0.52 * (end_y - start_y))
        arrow_end = (start_x + 0.80 * (end_x - start_x), start_y + 0.80 * (end_y - start_y))
        ax.annotate(
            "", xy=arrow_end, xytext=arrow_start,
            arrowprops={
                "arrowstyle": "-|>,head_length=0.4,head_width=0.4", "color": color, "lw": 1.0,
                "mutation_scale": 5, "shrinkA": 0, "shrinkB": 0,
            },
            zorder=zorder + 0.05,
        )
    ax.scatter(x, y, c=resistant, cmap=cmap, norm=norm, s=14,
               edgecolors="black", linewidths=.2, zorder=zorder + 0.1)


def main() -> None:
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    norm = Normalize(
        vmin=0.0,
        vmax=float(data["normalized_resistant"].max()),
    )
    # Keep a linear value-to-colour mapping, with the bright yellow reached at
    # 20% of the displayed resistance range before blending to white.
    cmap = LinearSegmentedColormap.from_list(
        "resistant_black_yellow_white",
        [(0.0, "#000000"), (0.2, "#FFE000"), (1.0, "#FFFFFF")],
    )
    # Keep the 35 mm height used by Figure 2b--d.  Each panel is 20% wider than
    # before (18 mm rather than 15 mm) and the 5 mm gaps separate all x-axes.
    # The 15 mm right margin remains reserved for the colour bar and label.
    fig, axes = ft.new_subplots(
        1, 3, ax_width_mm=AXES_WIDTH_MM, ax_height_mm=AXES_HEIGHT_MM,
        margins=MARGINS_MM, wspace_mm=PANEL_GAP_MM,
        sharex=True, sharey=True, apply_style=False,
    )
    for index, strategy in enumerate(PLOT_ORDER):
        group = data[data["strategy"] == strategy].sort_values("day")
        if group.empty:
            raise ValueError(f"No pre-escape confinement observations for {strategy}")
        ax = axes[index]
        add_trajectory(ax, group, norm, cmap, zorder=2.0)
        ax.axvline(
            1.0, color=get_color(strategy), lw=THRESHOLD_LINEWIDTH,
            ls=THRESHOLD_DASH_STYLE, dash_capstyle="butt", zorder=1,
        )
        if strategy in {"AT50", "AT100"}:
            ax.axvline(
                0.5 if strategy == "AT50" else 1.0,
                color=get_color(strategy), lw=THRESHOLD_LINEWIDTH,
                ls=THRESHOLD_DOT_STYLE, dash_capstyle="round", zorder=1.1,
            )
        ax.set(xlim=X_LIMITS, ylim=Y_LIMITS)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title(get_display_name(strategy, superscript=True), color=get_color(strategy), fontsize=7, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#000000")
        ax.tick_params(axis="y", colors="#000000")
        if index:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel(
        r"$\eta^{\mathrm{conf}}_{\mathrm{sen}}$",
        fontsize=7,
        color="#c8a2c8",
    )
    left_mm, right_mm, bottom_mm, top_mm = MARGINS_MM
    panel_span_mm = len(PLOT_ORDER) * AXES_WIDTH_MM + (len(PLOT_ORDER) - 1) * PANEL_GAP_MM
    canvas_width_mm = fig.get_figwidth() * 25.4
    canvas_height_mm = fig.get_figheight() * 25.4
    fig.supxlabel(
        r"Normalized count, $\eta_{\mathrm{tot}}$",
        x=(left_mm + panel_span_mm / 2.0) / canvas_width_mm,
        y=(bottom_mm / 2.0) / canvas_height_mm,
        fontsize=7,
    )

    # Position the colour bar within the reserved right margin, relative to
    # the changed three-panel geometry rather than hard-coded canvas values.
    colorbar_left_mm = left_mm + panel_span_mm + 2.0
    colorbar_ax = fig.add_axes([
        colorbar_left_mm / canvas_width_mm,
        bottom_mm / canvas_height_mm,
        2.0 / canvas_width_mm,
        AXES_HEIGHT_MM / canvas_height_mm,
    ])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label(
        r"Normalized resistant count, $\eta_{\mathrm{res}}$",
        fontsize=7,
        rotation=270,
        va="bottom",
        color=FIGURE_COLOR_MAP["yellow"],
        labelpad=2,
    )
    colorbar.set_ticks([0.0, 0.1, 0.2, 0.3])
    colorbar.set_ticklabels(["0", "0.1", "0.2", "0.3"])
    colorbar.ax.tick_params(labelsize=5, length=2)

    ft.savefig(fig, OUTPUT_PDF)
    plt.close(fig)
    data.to_csv(OUTPUT_VALUES, index=False)
    print(f"[saved] {OUTPUT_PDF}")
    print(f"[saved] {OUTPUT_VALUES}")


if __name__ == "__main__":
    main()
