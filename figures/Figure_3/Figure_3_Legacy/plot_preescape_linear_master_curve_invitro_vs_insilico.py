#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Plot pre-escape linear-growth master curves for in vitro and in silico data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name


GATE_TOKEN = "gate_2e-05_0p0004375"
GATE_LABEL = "2e-5 to 0.0004375 cells µm$^{-2}$"
CHANGE_TYPE = "linear_rel_per_day"
ANALYSIS_SUBDIR = (
    Path("summary_plots")
    / "gate_matrix_with_n"
    / "before_escape"
    / "linear_relative_per_day"
    / GATE_TOKEN
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


def default_analysis_dir(dataset: str, kernel_um: int) -> Path:
    if dataset not in {"InVitro", "InSilico"}:
        raise ValueError(dataset)
    return (
        Path(fr"F:\CoreSeed_{dataset}_PreEscape_{kernel_um}")
        / f"run_kernel_{kernel_um}_coarse_{kernel_um // 10}"
        / ANALYSIS_SUBDIR
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-um", type=int, choices=(160, 320), default=320)
    parser.add_argument("--in-vitro-table", type=Path, default=None)
    parser.add_argument("--in-silico-table", type=Path, default=None)
    parser.add_argument("--in-vitro-master-table", type=Path, default=None)
    parser.add_argument("--in-silico-master-table", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--min-bin-pixels",
        type=int,
        default=100,
        help="Minimum qualifying pixels per plotted strategy/bin. Default: 100.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    vitro_dir = default_analysis_dir("InVitro", int(args.kernel_um))
    silico_dir = default_analysis_dir("InSilico", int(args.kernel_um))
    vitro_table = (
        args.in_vitro_table
        if args.in_vitro_table is not None
        else vitro_dir / "binned_strategy_statistics.tsv"
    )
    silico_table = (
        args.in_silico_table
        if args.in_silico_table is not None
        else silico_dir / "binned_strategy_statistics.tsv"
    )
    vitro_master = (
        args.in_vitro_master_table
        if args.in_vitro_master_table is not None
        else vitro_dir / "pooled_all_strategies_statistics.tsv"
    )
    silico_master = (
        args.in_silico_master_table
        if args.in_silico_master_table is not None
        else silico_dir / "pooled_all_strategies_statistics.tsv"
    )
    output = (
        args.output
        if args.output is not None
        else Path.home()
        / "Desktop"
        / (
            "pre_escape_master_curves_invitro_vs_insilico_"
            f"{int(args.kernel_um)}um_gate_2e-05_0p0004375.pdf"
        )
    )
    return tuple(
        path.expanduser()
        for path in (vitro_table, silico_table, vitro_master, silico_master, output)
    )


def load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Summary table does not exist: {path}")
    data = pd.read_csv(path, sep="\t")
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    data = data[
        (data["phase"] == "before_escape")
        & (data["change_type"] == CHANGE_TYPE)
    ]
    if data.empty:
        raise RuntimeError(f"No pre-escape linear-growth rows in {path}")
    return data


def filtered_rows(data: pd.DataFrame, min_bin_pixels: int) -> pd.DataFrame:
    return data[data["n_pixels"] >= min_bin_pixels].sort_values(
        "x_bin_center_cells_per_1000_um2"
    )


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
        sub = filtered_rows(data[data["strategy"] == strategy], min_bin_pixels)
        if sub.empty:
            continue
        plotted_strategies.append(strategy)
        x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        ci_low = sub["median_ci95_low"].to_numpy(dtype=float)
        ci_high = sub["median_ci95_high"].to_numpy(dtype=float)
        color = get_color(strategy, "#000000")
        ax.fill_between(
            x,
            ci_low,
            ci_high,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
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

    master_sub = filtered_rows(master, min_bin_pixels)
    if master_sub.empty:
        raise RuntimeError(f"No pooled master bins remain for {title}")
    master_x = master_sub["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
    master_median = master_sub["median"].to_numpy(dtype=float)
    master_low = master_sub["median_ci95_low"].to_numpy(dtype=float)
    master_high = master_sub["median_ci95_high"].to_numpy(dtype=float)
    ax.fill_between(
        master_x,
        master_low,
        master_high,
        color="black",
        alpha=0.18,
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
    ax.set_xlabel(r"Total density [cells (1000 µm$^2$)$^{-1}$]")
    ft.format_axes(ax)
    return plotted_strategies


def shared_y_limits(
    tables: tuple[pd.DataFrame, ...],
    min_bin_pixels: int,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for table in tables:
        sub = filtered_rows(table, min_bin_pixels)
        for column in ("median_ci95_low", "median", "median_ci95_high"):
            finite = sub[column].to_numpy(dtype=float)
            values.append(finite[np.isfinite(finite)])
    concatenated = np.concatenate([item for item in values if item.size])
    low = float(np.min(concatenated))
    high = float(np.max(concatenated))
    span = max(high - low, 0.25)
    return min(-0.12, low - 0.08 * span), max(0.55, high + 0.10 * span)


def print_ci_diagnostics(label: str, master: pd.DataFrame, min_bin_pixels: int) -> None:
    sub = filtered_rows(master, min_bin_pixels)
    widths = (
        sub["median_ci95_high"].to_numpy(dtype=float)
        - sub["median_ci95_low"].to_numpy(dtype=float)
    )
    print(
        f"[{label} pooled CI] bins={len(sub)}; "
        f"median_width={np.median(widths):.6g}; max_width={np.max(widths):.6g}; "
        f"median_n={np.median(sub['n_pixels']):.0f}; "
        f"total_n={int(sub['n_pixels'].sum()):,}"
    )


def main() -> None:
    args = parse_args()
    if args.min_bin_pixels < 1:
        raise ValueError("--min-bin-pixels must be at least 1")
    (
        in_vitro_path,
        in_silico_path,
        in_vitro_master_path,
        in_silico_master_path,
        output,
    ) = resolve_paths(args)
    in_vitro = load_table(in_vitro_path)
    in_silico = load_table(in_silico_path)
    in_vitro_master = load_table(in_vitro_master_path)
    in_silico_master = load_table(in_silico_master_path)

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
    axes[0].set_ylim(
        *shared_y_limits(
            (in_vitro, in_silico, in_vitro_master, in_silico_master),
            int(args.min_bin_pixels),
        )
    )
    axes[0].set_ylabel("Linear resistant growth [fraction day$^{-1}$]")
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
        f"Before escape | {int(args.kernel_um)} µm kernel | gate {GATE_LABEL}",
        y=0.985,
    )
    fig.text(
        0.5,
        0.91,
        (
            "Linear growth = (rho_next/rho_previous - 1)/delta_days; "
            "colored: strategy medians; black: pooled median; "
            f"95% pixel-level CI; n_px >= {int(args.min_bin_pixels):,}"
        ),
        ha="center",
        va="center",
        fontsize=5.8,
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

    output.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)
    print(f"[saved] {output}")
    print(f"[in vitro] {', '.join(vitro_strategies)}")
    print(f"[in silico] {', '.join(silico_strategies)}")
    print_ci_diagnostics("in vitro", in_vitro_master, int(args.min_bin_pixels))
    print_ci_diagnostics("in silico", in_silico_master, int(args.min_bin_pixels))


if __name__ == "__main__":
    main()
