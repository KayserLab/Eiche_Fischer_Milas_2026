"""Plot all experimental wells and their established matched simulation runs.

Each source is exported as a two-panel figure.  Every line is one well/run,
coloured by strategy.  The left panel shows area-averaged total density; the
right panel shows pooled relative resistant growth per day.  Simulation runs
use the established Figure 3 matching cohort.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name, normalize_strategy
from analysis.figure_3 import well_density_intervals as source


OUTPUT_DIR = figure_plot_dir("Suppl_Figures", "all_well_density_trajectories", create=True)
DATA_DIR = figure_data_dir("Suppl_Figures", "all_well_density_trajectories", create=True)
GATE = (1.0e-8, float("inf"))


def ordered_strategies(data: pd.DataFrame) -> list[str]:
    present = {normalize_strategy(value) for value in data["strategy"]}
    return [strategy for strategy in STRATEGY_ORDER if strategy in present]


def plot_source(data: pd.DataFrame, output: Path, *, simulation: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), sharex=True)
    linestyle = "--" if simulation else "-"
    for (_plate, _well), group in data.groupby(["plate", "well"], sort=True):
        group = group.sort_values("mid_day")
        strategy = normalize_strategy(str(group["strategy"].iloc[0]))
        color = get_color(strategy, "#666666")
        axes[0].fill_between(
            group["mid_day"],
            group["density_area_mean"] - group["density_area_std"],
            group["density_area_mean"] + group["density_area_std"],
            color=color, alpha=0.13, linewidth=0, zorder=1,
        )
        axes[0].plot(group["mid_day"], group["density_area_mean"], color=color,
                     linestyle=linestyle, linewidth=0.8, alpha=0.38)
        axes[1].fill_between(
            group["mid_day"],
            group["net_relative_resistant_growth"]
            - group["net_relative_resistant_growth_std"],
            group["net_relative_resistant_growth"]
            + group["net_relative_resistant_growth_std"],
            color=color, alpha=0.13, linewidth=0, zorder=1,
        )
        axes[1].plot(group["mid_day"], group["net_relative_resistant_growth"], color=color,
                     linestyle=linestyle, linewidth=0.8, alpha=0.38)
    axes[1].axhline(0, color="#222222", linewidth=0.7, zorder=0)
    axes[0].set_ylabel(r"$\langle\rho_\mathrm{tot}\rangle_A$ [cells/1000 µm$^2$]")
    axes[1].set_ylabel(r"$\Delta\langle\tilde{\rho}_\mathrm{res}\rangle_A/\Delta t$ [day$^{-1}$]")
    for axis in axes:
        axis.set_xlabel("Time [days]")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", pad=2)
    handles = [Line2D([], [], color=get_color(strategy), linewidth=1.4,
                      label=get_display_name(strategy))
               for strategy in ordered_strategies(data)]
    axes[1].legend(handles=handles, frameon=False, fontsize=6, ncol=2,
                   handlelength=1.4, loc="upper right")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    ft.savefig(fig, output)
    plt.close(fig)


def main() -> None:
    experimental = source.scan_intervals(source.DEFAULT_RUN_DIR, GATE)
    simulation = source.scan_matched_simulation_intervals(source.DEFAULT_SIMULATION_RUN_DIR, GATE)
    for name, data, is_simulation in (
        ("experimental", experimental, False),
        ("matched_simulation", simulation, True),
    ):
        data.to_csv(DATA_DIR / f"all_well_density_trajectories_{name}.csv", index=False)
        plot_source(data, OUTPUT_DIR / f"all_well_density_trajectories_{name}.pdf",
                    simulation=is_simulation)


if __name__ == "__main__":
    main()
