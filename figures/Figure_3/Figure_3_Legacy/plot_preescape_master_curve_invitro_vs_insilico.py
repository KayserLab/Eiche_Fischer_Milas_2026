#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot side-by-side pre-escape master curves for in-vitro and simulation data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shared import figtools as ft
import plot_density_change_datapoints_coarsegrained as base
import plot_binned_density_change_strategy_lines_coarsegrained as plotter
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name


GATE = (2.0e-5, 4.375e-4)
GATE_TOKEN = "gate_2e-05_0p0004375"
GATE_LABEL = "2e-5 to 0.0004375 cells µm$^{-2}$"
DEFAULT_IN_VITRO_RUN = Path(
    r"D:\2025_08_CoreSeed_Large_Testdataset\output"
    r"\run_20260630_145305_320_32"
)
DEFAULT_IN_SILICO_RUN = Path(
    r"F:\2025_08_CoreSeed_Simulation_Density_Output"
    r"\run_sim_20260709_145640_320_32"
)
DEFAULT_IN_VITRO_ESCAPE = Path(
    r"D:\2025_08_CoreSeed_Large_Testdataset\Figures"
    r"\Distance_to_front_analysis\frontcontact_summary_ALL.csv"
)
DEFAULT_IN_SILICO_ESCAPE = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\Maja"
    r"\combined_results.csv"
)
DEFAULT_IN_VITRO_TABLE = Path(
    r"F:\CoreSeed_InVitro_PreEscape_320\run_kernel_320_coarse_32"
    r"\summary_plots\gate_matrix_with_n\before_escape\relative"
) / GATE_TOKEN / "binned_strategy_statistics.tsv"
DEFAULT_IN_SILICO_TABLE = Path(
    r"F:\CoreSeed_InSilico_PreEscape_320\run_kernel_320_coarse_32"
    r"\summary_plots\gate_matrix_with_n\before_escape\relative"
) / GATE_TOKEN / "binned_strategy_statistics.tsv"
DEFAULT_IN_VITRO_MASTER_TABLE = DEFAULT_IN_VITRO_TABLE.with_name(
    "pooled_all_strategies_statistics.tsv"
)
DEFAULT_IN_SILICO_MASTER_TABLE = DEFAULT_IN_SILICO_TABLE.with_name(
    "pooled_all_strategies_statistics.tsv"
)
DEFAULT_OUTPUT = Path.home() / "Desktop" / (
    "pre_escape_master_curves_invitro_vs_insilico_"
    "320um_gate_2e-05_0p0004375.pdf"
)
REQUIRED_COLUMNS = {
    "phase",
    "change_type",
    "strategy",
    "x_bin_center_cells_per_1000_um2",
    "n_pixels",
    "median",
    "median_ci95_low",
    "median_ci95_high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-vitro-table", type=Path, default=DEFAULT_IN_VITRO_TABLE)
    parser.add_argument("--in-silico-table", type=Path, default=DEFAULT_IN_SILICO_TABLE)
    parser.add_argument(
        "--in-vitro-master-table",
        type=Path,
        default=DEFAULT_IN_VITRO_MASTER_TABLE,
    )
    parser.add_argument(
        "--in-silico-master-table",
        type=Path,
        default=DEFAULT_IN_SILICO_MASTER_TABLE,
    )
    parser.add_argument("--in-vitro-run", type=Path, default=DEFAULT_IN_VITRO_RUN)
    parser.add_argument("--in-silico-run", type=Path, default=DEFAULT_IN_SILICO_RUN)
    parser.add_argument(
        "--in-vitro-escape",
        type=Path,
        default=DEFAULT_IN_VITRO_ESCAPE,
    )
    parser.add_argument(
        "--in-silico-escape",
        type=Path,
        default=DEFAULT_IN_SILICO_ESCAPE,
    )
    parser.add_argument(
        "--recompute-master",
        action="store_true",
        help="Recompute pooled pixel histograms instead of using cached master tables.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-bin-pixels",
        type=int,
        default=100,
        help="Minimum qualifying pixels per plotted strategy/bin. Default: 100.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Summary table does not exist: {path}")
    data = pd.read_csv(path, sep="\t")
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    data = data[(data["phase"] == "before_escape") & (data["change_type"] == "rel")]
    if data.empty:
        raise RuntimeError(f"No pre-escape relative-growth rows in {path}")
    return data


def compute_pooled_master(
    *,
    run_dir: Path,
    escape_path: Path,
    output_table: Path,
    expect_simulation: bool,
) -> pd.DataFrame:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Density run does not exist: {run_dir}")
    if not escape_path.is_file():
        raise FileNotFoundError(f"Escape table does not exist: {escape_path}")

    config = base.load_config(base.DEFAULT_CONFIG_PATH)
    if float(config.get("radius_um", 0)) != 320.0:
        raise ValueError("The active density configuration is not a 320-um kernel.")
    if int(config.get("coarsening_factor", 0)) != 32:
        raise ValueError("The active density configuration is not coarse-32.")

    plotter.DEFAULT_GATES = (GATE,)
    plotter.CHANGE_TYPES = ("rel",)
    strategies = plotter.strategies_in_run(run_dir)
    resolution_args = argparse.Namespace(
        experimental_escape_summary=(
            None if expect_simulation else os.fspath(escape_path)
        ),
        simulation_escape_csv=(
            os.fspath(escape_path) if expect_simulation else None
        ),
    )
    cutoffs, simulation_run, escape_source = plotter.resolve_escape_cutoffs(
        run_dir,
        config,
        resolution_args,
    )
    if bool(simulation_run) != bool(expect_simulation):
        raise ValueError(
            f"Unexpected run type for {run_dir}: simulation={simulation_run}"
        )

    settings = plotter.plot_settings(config, "rel")
    x_bin_size = float(plotter.DEFAULT_X_BIN_SIZE)
    x_max = float(settings["xlim"][1])
    n_x_bins = int(np.ceil(x_max / x_bin_size))
    x_edges = np.arange(n_x_bins + 1, dtype=float) * x_bin_size
    aggregator = plotter.Aggregator.create(
        n_strategies=len(strategies),
        x_edges=x_edges,
        ylim=settings["ylim"],
        n_y_bins=int(plotter.DEFAULT_Y_HIST_BINS),
    )
    aggregators = {
        ("before_escape", GATE_TOKEN, "rel"): aggregator,
    }
    completed = plotter.read_and_accumulate(
        run_dir,
        config,
        aggregators,
        x_edges,
        strategies,
        ("before_escape",),
        cutoffs,
        simulation_run,
    )

    pooled = plotter.Aggregator.create(
        n_strategies=1,
        x_edges=x_edges,
        ylim=settings["ylim"],
        n_y_bins=int(plotter.DEFAULT_Y_HIST_BINS),
    )
    pooled.count[0] = aggregator.count.sum(axis=0)
    pooled.total[0] = aggregator.total.sum(axis=0)
    pooled.total_sq[0] = aggregator.total_sq.sum(axis=0)
    pooled.histogram[0] = aggregator.histogram.sum(axis=0)
    summary = plotter.summarize(
        pooled,
        ["All strategies"],
        GATE,
        "rel",
        "before_escape",
    )
    output_table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_table, sep="\t", index=False)
    print(
        f"[master] {run_dir.name}: {completed} intervals; "
        f"escape={escape_source}; saved={output_table}"
    )
    return summary


def load_or_compute_master(
    *,
    table_path: Path,
    run_dir: Path,
    escape_path: Path,
    expect_simulation: bool,
    recompute: bool,
) -> pd.DataFrame:
    if recompute or not table_path.is_file():
        return compute_pooled_master(
            run_dir=run_dir,
            escape_path=escape_path,
            output_table=table_path,
            expect_simulation=expect_simulation,
        )
    print(f"[master] using cached table: {table_path}")
    return load_table(table_path)


def plot_dataset(
    ax: plt.Axes,
    data: pd.DataFrame,
    master: pd.DataFrame,
    *,
    title: str,
    min_bin_pixels: int,
) -> list[str]:
    plotted_strategies: list[str] = []
    for strategy in STRATEGY_ORDER:
        sub = data[
            (data["strategy"] == strategy)
            & (data["n_pixels"] >= min_bin_pixels)
        ].sort_values("x_bin_center_cells_per_1000_um2")
        if sub.empty:
            continue
        plotted_strategies.append(strategy)
        x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        ci_low = sub["median_ci95_low"].to_numpy(dtype=float)
        ci_high = sub["median_ci95_high"].to_numpy(dtype=float)
        color = get_color(strategy, "#000000")
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            median,
            color=color,
            marker="o",
            markersize=2.8,
            linewidth=1.25,
            markeredgewidth=0,
            label=get_display_name(strategy),
            zorder=3,
        )

    master_sub = master[
        master["n_pixels"] >= min_bin_pixels
    ].sort_values("x_bin_center_cells_per_1000_um2")
    if master_sub.empty:
        raise RuntimeError(f"No pooled master bins remain for {title}")
    master_x = master_sub[
        "x_bin_center_cells_per_1000_um2"
    ].to_numpy(dtype=float)
    master_median = master_sub["median"].to_numpy(dtype=float)
    master_low = master_sub["median_ci95_low"].to_numpy(dtype=float)
    master_high = master_sub["median_ci95_high"].to_numpy(dtype=float)
    ax.fill_between(
        master_x,
        master_low,
        master_high,
        color="black",
        alpha=0.16,
        linewidth=0,
        zorder=8,
    )
    ax.plot(
        master_x,
        master_median,
        color="black",
        marker="o",
        markersize=3.6,
        linewidth=2.4,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="All strategies pooled",
        zorder=10,
    )

    ax.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=0)
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(-0.2, 1.35)
    ax.set_xlabel(r"Total density [cells (1000 µm$^2$)$^{-1}$]")
    ft.format_axes(ax)
    return plotted_strategies


def main() -> None:
    args = parse_args()
    if args.min_bin_pixels < 1:
        raise ValueError("--min-bin-pixels must be at least 1")

    in_vitro = load_table(args.in_vitro_table.expanduser())
    in_silico = load_table(args.in_silico_table.expanduser())
    in_vitro_master = load_or_compute_master(
        table_path=args.in_vitro_master_table.expanduser(),
        run_dir=args.in_vitro_run.expanduser(),
        escape_path=args.in_vitro_escape.expanduser(),
        expect_simulation=False,
        recompute=bool(args.recompute_master),
    )
    in_silico_master = load_or_compute_master(
        table_path=args.in_silico_master_table.expanduser(),
        run_dir=args.in_silico_run.expanduser(),
        escape_path=args.in_silico_escape.expanduser(),
        expect_simulation=True,
        recompute=bool(args.recompute_master),
    )

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=6,
        overrides={
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "figure.titlesize": 8,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    vitro_strategies = plot_dataset(
        axes[0],
        in_vitro,
        in_vitro_master,
        title="In vitro",
        min_bin_pixels=int(args.min_bin_pixels),
    )
    silico_strategies = plot_dataset(
        axes[1],
        in_silico,
        in_silico_master,
        title="In silico",
        min_bin_pixels=int(args.min_bin_pixels),
    )
    axes[0].set_ylabel(r"$\Delta\tilde{\rho}_\mathrm{res}/\Delta t$")
    axes[0].text(-0.16, 1.05, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.10, 1.05, "b", transform=axes[1].transAxes, fontweight="bold")

    legend_strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if strategy in set(vitro_strategies) | set(silico_strategies)
    ]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=get_color(strategy, "#000000"),
            marker="o",
            markersize=3,
            linewidth=1.25,
            markeredgewidth=0,
            label=get_display_name(strategy),
        )
        for strategy in legend_strategies
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markersize=3.6,
            linewidth=2.4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label="All strategies pooled",
        )
    )
    fig.suptitle(
        f"Before escape | 320 µm kernel | resistant-density gate {GATE_LABEL}",
        y=0.985,
    )
    fig.text(
        0.5,
        0.91,
        (
            "Colored: strategy medians; black: pooled all-pixel median; "
            f"95% CI; n_px >= {int(args.min_bin_pixels):,}"
        ),
        ha="center",
        va="center",
        fontsize=6,
        color="#444444",
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.82, wspace=0.18)

    output = args.output.expanduser()
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)
    print(f"[saved] {output}")
    print(f"[in vitro] {', '.join(vitro_strategies)}")
    print(f"[in silico] {', '.join(silico_strategies)}")


if __name__ == "__main__":
    main()
