"""Export a standalone legend for ``g_plot_selected_well_trajectories.py``."""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: none; all legend labels/styles are code-defined.
# Scripts: g_plot_selected_well_trajectories.py, shared/strategy_colors.py, shared/figtools.py.
# Outputs: standalone horizontal and vertical trajectory-legend PDFs.

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.strategy_colors import get_color
from g_plot_selected_well_trajectories import (
    RESISTANT_COLOR,
    SENSITIVE_COLOR,
    THRESHOLD_DOT_POINTS,
    THRESHOLD_LINEWIDTH,
    TOTAL_COLOR,
)


DEFAULT_OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "Selected_Well_Trajectories"
OUTPUT_NAME = "Selected_Well_Trajectories_Legend.pdf"
VERTICAL_OUTPUT_NAME = "Selected_Well_Trajectories_Legend_Vertical.pdf"
LEGEND_WIDTH_IN = 180.0 / 72.0
LEGEND_HEIGHT_IN = 35.6 / 72.0
TEXT_SIZE = 6
LINEWIDTH = 1.6
THRESHOLD_STATE_LABEL_FONTSIZE = 5
# Kept local so the compact legend remains visibly dashed without changing
# the established dash cycle in the trajectory panels.
LEGEND_THRESHOLD_ON_DASH_STYLE = (0.0, (2.5, 1.5))
# The plot's dotted line spans a full panel; this denser equivalent preserves
# the same round-dot appearance in the compact legend sample.
LEGEND_THRESHOLD_DOT_STYLE = (0.0, (0.8, 3.2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the selected-well trajectory legend.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold-color-split",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Use an AT50/AT100 split across or through each threshold-line sample.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def add_line(ax: plt.Axes, x: float, y: float, color: str, *, simulated: bool = False) -> None:
    x_start, x_end = x - 0.043, x + 0.043
    if simulated:
        # Explicit dashes ensure the visible DPD sample has the same endpoints
        # as the solid Exp sample, rather than ending in a dash gap.
        dash_segments = (
            (x_start, x_start + 0.022),
            (x - 0.011, x + 0.011),
            (x_end - 0.022, x_end),
        )
        for segment_start, segment_end in dash_segments:
            ax.plot(
                [segment_start, segment_end], [y, y], color=color,
                linewidth=LINEWIDTH, solid_capstyle="butt",
            )
    else:
        ax.plot(
            [x_start, x_end], [y, y], color=color,
            linewidth=LINEWIDTH, solid_capstyle="butt",
        )


def add_state_line(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    on: bool,
    ct_color: str | None,
    at50_color: str,
    at100_color: str,
    split: str,
) -> None:
    linestyle = LEGEND_THRESHOLD_ON_DASH_STYLE if on else LEGEND_THRESHOLD_DOT_STYLE
    dash_capstyle = "butt" if on else "round"
    colors = [at50_color, at100_color] if not on else [ct_color, at50_color, at100_color]
    if any(color is None for color in colors):
        raise ValueError("A CT color is required for the treatment-on threshold legend.")
    colors = [str(color) for color in colors]

    if split == "horizontal":
        x_positions = [x - 0.043 + idx * (0.086 / len(colors)) for idx in range(len(colors) + 1)]
        segments = tuple(
            (x_positions[idx], x_positions[idx + 1], y, color)
            for idx, color in enumerate(colors)
        )
        linewidth = THRESHOLD_LINEWIDTH
    else:
        # Stack CT, AT50, and AT100 from top to bottom for treatment-on;
        # treatment-off contains only AT50 and AT100.
        offsets = {2: (0.011, -0.011), 3: (0.014, 0.0, -0.014)}[len(colors)]
        segments = tuple(
            (x - 0.043, x + 0.043, y + offset, color)
            for offset, color in zip(offsets, colors)
        )
        # Preserve the plot's threshold width for the dotted OFF marker; at
        # half-width its small round dots disappear in this compact legend.
        linewidth = THRESHOLD_LINEWIDTH if not on else THRESHOLD_LINEWIDTH / len(colors)

    for x_start, x_end, y_position, color in segments:
        ax.plot(
            [x_start, x_end], [y_position, y_position], color=color, linewidth=linewidth,
            linestyle=linestyle, dash_capstyle=dash_capstyle,
        )


def add_off_dots(ax: plt.Axes, x: float, y: float, *, at50_color: str, at100_color: str) -> None:
    """Draw three OFF dots, with a half-AT50/half-AT100 center dot."""
    # Draw each dot as the same short, round-capped stroke used by the plot's
    # dotted threshold.  The center stroke is clipped into two color halves.
    dot_length = THRESHOLD_DOT_POINTS / (LEGEND_WIDTH_IN * 72.0)
    left_x, center_x, right_x = x - 0.027, x, x + 0.027

    def draw_dot(center: float, color: str):
        return ax.plot(
            [center - dot_length / 2.0, center + dot_length / 2.0],
            [y, y],
            color=color,
            linewidth=THRESHOLD_LINEWIDTH,
            solid_capstyle="round",
            dash_capstyle="round",
        )[0]

    draw_dot(left_x, at50_color)
    draw_dot(center_x, at50_color)
    right_half = draw_dot(center_x, at100_color)
    right_half.set_clip_path(
        Rectangle((center_x, y - 0.05), 0.05, 0.10, transform=ax.transData)
    )
    draw_dot(right_x, at100_color)


def plot_legend(output_path: Path, show: bool, threshold_color_split: str) -> None:
    fig = plt.figure(figsize=(LEGEND_WIDTH_IN, LEGEND_HEIGHT_IN))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    columns = {
        "Total": 0.180,
        "Sensitive": 0.360,
        "Resistant": 0.540,
        "threshold_on": 0.720,
        "threshold_off": 0.900,
    }
    header_y, experiment_y, simulation_y = 0.77, 0.49, 0.24
    ct_color = get_color("CT", "grey")
    at50_color = get_color("AT50", "grey")
    at100_color = get_color("AT100", "grey")
    for label in ("Total", "Sensitive", "Resistant"):
        x = columns[label]
        ax.text(x, header_y, label, ha="center", va="center", fontsize=TEXT_SIZE)
    ax.text(0.810, 0.82, "Threshold", ha="center", va="center", fontsize=TEXT_SIZE)
    ax.text(columns["threshold_on"], 0.65, "ON", ha="center", va="center", fontsize=THRESHOLD_STATE_LABEL_FONTSIZE)
    ax.text(columns["threshold_off"], 0.65, "OFF", ha="center", va="center", fontsize=THRESHOLD_STATE_LABEL_FONTSIZE)
    ax.text(0.023, experiment_y, "Exp", ha="left", va="center", fontsize=TEXT_SIZE)
    ax.text(0.023, simulation_y, "Sim", ha="left", va="center", fontsize=TEXT_SIZE)

    add_line(ax, columns["Total"], experiment_y, TOTAL_COLOR)
    add_line(ax, columns["Total"], simulation_y, TOTAL_COLOR, simulated=True)
    add_line(ax, columns["Sensitive"], experiment_y, SENSITIVE_COLOR)
    add_line(ax, columns["Sensitive"], simulation_y, SENSITIVE_COLOR, simulated=True)
    add_line(ax, columns["Resistant"], experiment_y, RESISTANT_COLOR)
    add_line(ax, columns["Resistant"], simulation_y, RESISTANT_COLOR, simulated=True)
    add_state_line(
        ax, columns["threshold_on"], experiment_y,
        on=True, ct_color=ct_color, at50_color=at50_color, at100_color=at100_color,
        split=threshold_color_split,
    )
    add_state_line(
        ax, columns["threshold_on"], simulation_y,
        on=True, ct_color=ct_color, at50_color=at50_color, at100_color=at100_color,
        split=threshold_color_split,
    )
    add_off_dots(ax, columns["threshold_off"], experiment_y, at50_color=at50_color, at100_color=at100_color)
    add_off_dots(ax, columns["threshold_off"], simulation_y, at50_color=at50_color, at100_color=at100_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_name = OUTPUT_NAME if args.threshold_color_split == "horizontal" else VERTICAL_OUTPUT_NAME
    output_path = args.output_dir / output_name
    plot_legend(output_path, show=args.show, threshold_color_split=args.threshold_color_split)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
