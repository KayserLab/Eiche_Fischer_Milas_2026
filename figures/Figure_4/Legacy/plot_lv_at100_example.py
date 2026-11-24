"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream/related script reference(s):
# - import matplotlib.py
# =============================================================================
Plot a synthetic AT100 trajectory generated from the fitted LV environment.

This script intentionally does not read experimental replicate data. It uses the
paper-facing fitted LV parameters from the PhysiLearning mini-repo and plots the
result with the same compact trajectory styling used for selected wells.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.strategy_colors import FIGURE_COLOR_MAP


PARAMS_PATH = (
    REPO_ROOT
    / "PhysiLearning"
    / "LV_Agent_Training_MiniRepo"
    / "reference"
    / "fitted_lv_params.json"
)
DEFAULT_OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "LV_AT100_Example"

TOTAL_COLOR = "black"
SENSITIVE_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
TREATMENT_COLOR = "#bfbfbf"
SERIES_LINEWIDTH = 1.3
TOTAL_LINEWIDTH = 1.6
MARKER_SIZE = 2.2
LINESTYLE = "-"

SIZE_SCALE = 1.15
HEIGHT_SCALE = 1.15

PLOT_BOX_WIDTH_CM = 2.4 * SIZE_SCALE
PLOT_BOX_HEIGHT_CM = 1.6 * SIZE_SCALE * HEIGHT_SCALE
LEFT_MARGIN_CM = 1.3 * SIZE_SCALE
RIGHT_MARGIN_CM = 0.3 * SIZE_SCALE
BOTTOM_MARGIN_CM = 1.0 * SIZE_SCALE * HEIGHT_SCALE
TOP_MARGIN_CM = 0.6 * SIZE_SCALE * HEIGHT_SCALE
FIGURE_AXES_WIDTH_MM = PLOT_BOX_WIDTH_CM * 10.0
FIGURE_AXES_HEIGHT_MM = PLOT_BOX_HEIGHT_CM * 10.0
FIGURE_MARGINS_MM = (
    LEFT_MARGIN_CM * 10.0,
    RIGHT_MARGIN_CM * 10.0,
    BOTTOM_MARGIN_CM * 10.0,
    TOP_MARGIN_CM * 10.0,
)

AXIS_FONTSIZE = 7
TITLE_FONTSIZE = 7
TICK_FONTSIZE = 6
TICK_LABEL_PAD = 1
TICK_LENGTH = 2
XLABEL_PAD = 2
YLABEL_PAD = 2
LEGEND_FONTSIZE = 5

ft.apply_nature_style(
    text_pt=AXIS_FONTSIZE,
    tick_pt=TICK_FONTSIZE,
    legend_pt=LEGEND_FONTSIZE,
    overrides={
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
    },
)


@dataclass(frozen=True)
class LvParams:
    growth_rate_wt: float
    growth_rate_mut: float
    death_rate_wt: float
    death_rate_mut: float
    treat_death_rate_wt: float
    treat_death_rate_mut: float
    carrying_capacity: float
    competition_wt: float
    competition_mut: float
    timestep_size: float
    ramp_time_up: float
    ramp_time_down: float
    min_death_during_treat: float
    on_treat_threshold: float
    off_treat_threshold: float
    initial_wt: float
    initial_mut: float
    max_tumor_size: float
    max_time: int
    treatment_time_step: int


@dataclass(frozen=True)
class SyntheticTrajectory:
    days: list[int]
    sensitive: list[float]
    resistant: list[float]
    treatment_intervals: list[tuple[int, int]]

    @property
    def total(self) -> list[float]:
        return [s + r for s, r in zip(self.sensitive, self.resistant)]


class LvAt100Simulator:
    """Small, self-contained mirror of the mini-repo's ramped LV dynamics."""

    def __init__(self, params: LvParams) -> None:
        self.params = params
        self.sensitive = float(params.initial_wt)
        self.resistant = float(params.initial_mut)
        self.initial_total = self.sensitive + self.resistant
        self.threshold_burden = params.max_tumor_size * self.initial_total
        self.time = 0
        self.day = 1
        self.growth_fraction = 1.0
        self.death_fraction = 0.0
        self.time_on_treatment = 0.0
        self.time_off_treatment = 100.0
        self.last_action = 0
        self.treatment_history = [0]

    def should_treat_at100(self) -> int:
        return int((self.sensitive + self.resistant) >= self.initial_total)

    def next_day_and_inner_steps(self) -> tuple[int, int]:
        if self.day == 5 or (self.day > 5 and (self.day - 5) % 7 == 0):
            return self.day + 3, int(self.params.treatment_time_step * 1.5)
        return self.day + 2, int(self.params.treatment_time_step)

    def grow_one_population(self, index: int, action: int) -> float:
        p = self.params
        step_up = p.timestep_size * 2.0 / p.ramp_time_up
        step_down = p.timestep_size * 2.0 / p.ramp_time_down

        if action == 1:
            if self.time_off_treatment > 0:
                self.time_off_treatment = 0.0
            self.time_on_treatment += p.timestep_size
        else:
            if self.time_on_treatment > 0:
                self.time_on_treatment = 0.0
            self.time_off_treatment += p.timestep_size

        if (
            (action == 1 and self.time_on_treatment > p.on_treat_threshold)
            or (action == 0 and self.time_off_treatment < p.off_treat_threshold)
        ):
            if self.growth_fraction > 0:
                self.growth_fraction = max(0.0, self.growth_fraction - step_down)
            else:
                self.death_fraction = min(1.0, self.death_fraction + step_down)
        else:
            if self.death_fraction > 0:
                self.death_fraction = max(0.0, self.death_fraction - step_up)
            else:
                self.growth_fraction = min(1.0, self.growth_fraction + step_up)

        values = [self.sensitive, self.resistant]
        growth_rates = [
            p.growth_rate_wt * self.growth_fraction,
            p.growth_rate_mut,
        ]
        death_rates = [p.death_rate_wt, p.death_rate_mut]
        treat_death_rates = [
            p.treat_death_rate_wt * self.death_fraction,
            p.treat_death_rate_mut,
        ]
        competition = [p.competition_wt, p.competition_mut]
        j = 1 - index
        treat_vector = [1, 0] if self.death_fraction > 0 else [0, 0]

        new_size = values[index] * (
            1
            + p.timestep_size
            * growth_rates[index]
            * (1 - (values[index] + values[j] * competition[j]) / p.carrying_capacity)
            - treat_death_rates[index] * treat_vector[index] * p.timestep_size
            - death_rates[index] * p.timestep_size
        ) - treat_vector[index] * p.min_death_during_treat * p.timestep_size

        return max(0.0, 0.0 if new_size < 1.0e-5 else float(new_size))

    def step(self, action: int) -> int:
        next_day, inner_steps = self.next_day_and_inner_steps()
        for _ in range(inner_steps):
            self.time += 1
            self.last_action = action
            self.sensitive = self.grow_one_population(0, action)
            self.resistant = self.grow_one_population(1, action)
            self.treatment_history.append(action)
        self.day = next_day
        return next_day

    def run(self) -> SyntheticTrajectory:
        days = [self.day]
        sensitive = [self.sensitive]
        resistant = [self.resistant]
        treatment_intervals: list[tuple[int, int]] = []

        while self.time < self.params.max_time and self.sensitive + self.resistant < self.threshold_burden:
            start_day = self.day
            action = self.should_treat_at100()
            next_day = self.step(action)
            if action:
                treatment_intervals.append((start_day, next_day))
            days.append(self.day)
            sensitive.append(self.sensitive)
            resistant.append(self.resistant)

        return SyntheticTrajectory(
            days=days,
            sensitive=sensitive,
            resistant=resistant,
            treatment_intervals=treatment_intervals,
        )


def load_params(path: Path) -> LvParams:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return LvParams(**raw)


def add_treatment_intervals(
    ax: plt.Axes,
    intervals: list[tuple[int, int]],
    max_day: float,
    ymin: float,
    ymax: float,
) -> None:
    for start, end in intervals:
        ax.axvspan(
            start,
            min(end, max_day),
            ymin=ymin,
            ymax=ymax,
            facecolor=TREATMENT_COLOR,
            lw=0,
            alpha=1.0,
            zorder=0,
        )


def plot_trajectory(
    trajectory: SyntheticTrajectory,
    output_dir: Path,
    max_y: float | None,
    show: bool,
) -> Path:
    denominator = trajectory.total[0]
    total_norm = [value / denominator for value in trajectory.total]
    sensitive_norm = [value / denominator for value in trajectory.sensitive]
    resistant_norm = [value / denominator for value in trajectory.resistant]
    upper_y = float(max_y) if max_y is not None else max(1.25, max(total_norm) * 1.05)
    x_axis_max = trajectory.days[-1]

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        apply_style=False,
    )

    add_treatment_intervals(
        ax,
        trajectory.treatment_intervals,
        max_day=x_axis_max,
        ymin=0.0,
        ymax=1.0,
    )

    ax.plot(
        trajectory.days,
        total_norm,
        color=TOTAL_COLOR,
        marker="x",
        markersize=MARKER_SIZE,
        linewidth=TOTAL_LINEWIDTH,
        linestyle=LINESTYLE,
        label="Total",
    )
    ax.plot(
        trajectory.days,
        sensitive_norm,
        color=SENSITIVE_COLOR,
        marker="x",
        markersize=MARKER_SIZE,
        linewidth=SERIES_LINEWIDTH,
        linestyle=LINESTYLE,
        label="Sensitive",
    )
    ax.plot(
        trajectory.days,
        resistant_norm,
        color=RESISTANT_COLOR,
        marker="x",
        markersize=MARKER_SIZE,
        linewidth=SERIES_LINEWIDTH,
        linestyle=LINESTYLE,
        label="Resistant",
    )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_xlim(trajectory.days[0], x_axis_max)
    ax.set_ylim(0, upper_y)
    ax.set_xlabel("Time [days]", fontsize=AXIS_FONTSIZE, labelpad=XLABEL_PAD)
    ax.set_ylabel("Cells (normalized)", fontsize=AXIS_FONTSIZE, labelpad=YLABEL_PAD)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE, pad=TICK_LABEL_PAD, length=TICK_LENGTH)
    ax.tick_params(axis="x", which="minor", labelsize=0, length=TICK_LENGTH)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, pad=TICK_LABEL_PAD, length=TICK_LENGTH)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "LV_AT100_example_norm"
    ft.set_transparent_background(fig)
    ft.savefig(fig, output_base.with_suffix(".pdf"))

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a synthetic AT100 example from the fitted LV parameters."
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=PARAMS_PATH,
        help=f"Path to fitted LV parameter JSON. Default: {PARAMS_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported plot. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-y",
        type=float,
        default=3.2,
        help="Fixed y-axis maximum. Use <=0 for auto-scaling. Default: 3.2",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    trajectory = LvAt100Simulator(params).run()
    max_y = None if args.max_y <= 0 else args.max_y
    output_base = plot_trajectory(
        trajectory=trajectory,
        output_dir=args.output_dir,
        max_y=max_y,
        show=args.show,
    )
    print(f"[info] parameter_source={args.params}")
    print(f"[info] points={len(trajectory.days)} final_day={trajectory.days[-1]}")
    print(f"[saved] {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
