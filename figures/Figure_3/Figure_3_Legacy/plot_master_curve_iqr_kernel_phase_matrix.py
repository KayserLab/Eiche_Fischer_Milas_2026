#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""Write the 3-kernel x 2-gate x 3-phase linear-growth master-curve set."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR, FIGURE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shared import figtools as ft
from analysis.figure_3 import master_curve_iqr_common as common
from shared.strategy_colors import get_color, get_display_name


DEFAULT_CACHE_ROOT = Path(r"F:\CoreSeed_Master_Curve_Linear_IQR_Cache")
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "Desktop" / "CoreSeed_Linear_Growth_Master_Curves_IQR"
)
KERNELS = (320, 160, 40)
FIXED_YLIM = (-1.0, 4.0)
PHASE_LABELS = {
    "overall": "All timepoints",
    "before_escape": "Before escape",
    "after_escape": "After escape",
}
GATE_LABELS = {
    "ungated": (
        "No resistant-density gate "
        "(all finite pixels with previous resistant density > 1e-10)"
    ),
    "gate_2e-05_0p0004375": (
        r"Previous resistant density 2e-5 to 4.375e-4 cells µm$^{-2}$"
    ),
}
GATE_FOLDERS = {
    "ungated": "ungated",
    "gate_2e-05_0p0004375": "gate_2e-05_to_0p0004375",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--min-bin-pixels",
        type=int,
        default=100,
        help="Minimum pixels in a plotted bin. Default 100.",
    )
    return parser.parse_args()


def cache_paths(cache_root: Path, dataset: str, kernel_um: int) -> list[Path]:
    dataset_root = cache_root / dataset
    if dataset == "in_vitro" and kernel_um == 40:
        return [
            dataset_root / "kernel_40um_ct_at50_at100.npz",
            dataset_root / "kernel_40um_streamed_other_strategies.npz",
        ]
    return [dataset_root / f"kernel_{kernel_um}um_all_phases_iqr.npz"]


def load_kernel_bundle(
    cache_root: Path,
    dataset: str,
    kernel_um: int,
) -> common.AggregationBundle:
    paths = cache_paths(cache_root, dataset, kernel_um)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing cache(s): " + ", ".join(str(path) for path in missing)
        )
    return common.merge_bundles(paths)


def rows_for_plot(data: pd.DataFrame, min_bin_pixels: int) -> pd.DataFrame:
    return data[
        (data["n_pixels"] >= min_bin_pixels)
        & np.isfinite(data["median"])
        & np.isfinite(data["q1"])
        & np.isfinite(data["q3"])
    ].sort_values("x_bin_center_cells_per_1000_um2")


def plot_dataset(
    ax: plt.Axes,
    strategy_data: pd.DataFrame,
    pooled_data: pd.DataFrame,
    *,
    title: str,
    min_bin_pixels: int,
) -> list[str]:
    plotted: list[str] = []
    for strategy in common.ALLOWED_STRATEGIES:
        sub = rows_for_plot(
            strategy_data[strategy_data["strategy"] == strategy],
            min_bin_pixels,
        )
        if sub.empty:
            continue
        plotted.append(strategy)
        x = sub["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
        q1 = sub["q1"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        q3 = sub["q3"].to_numpy(dtype=float)
        color = get_color(strategy, "#666666")
        ax.fill_between(
            x,
            q1,
            q3,
            color=color,
            alpha=0.075,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            median,
            color=color,
            marker="o",
            markersize=2.2,
            linewidth=1.0,
            markeredgewidth=0,
            zorder=2,
        )

    master = rows_for_plot(pooled_data, min_bin_pixels)
    if master.empty:
        raise RuntimeError(f"No pooled master-curve bins remain for {title}")
    x = master["x_bin_center_cells_per_1000_um2"].to_numpy(dtype=float)
    q1 = master["q1"].to_numpy(dtype=float)
    median = master["median"].to_numpy(dtype=float)
    q3 = master["q3"].to_numpy(dtype=float)
    counts = master["n_pixels"].to_numpy(dtype=np.int64)
    ax.fill_between(
        x,
        q1,
        q3,
        color="black",
        alpha=0.14,
        linewidth=0,
        zorder=7,
    )
    ax.plot(
        x,
        median,
        color="black",
        marker="o",
        markersize=3.5,
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=9,
    )
    for index, (x_value, y_value, count) in enumerate(zip(x, median, counts)):
        upward = index % 2 == 0
        if y_value <= FIXED_YLIM[0] + 0.25:
            upward = True
        elif y_value >= FIXED_YLIM[1] - 0.25:
            upward = False
        ax.annotate(
            f"n={common.compact_count(int(count))}",
            xy=(x_value, y_value),
            xytext=(0, 5 if upward else -7),
            textcoords="offset points",
            ha="center",
            va="bottom" if upward else "top",
            fontsize=4.0,
            color="black",
            zorder=12,
            clip_on=True,
        )

    ax.axhline(0.0, color="#B8B8B8", linewidth=0.6, zorder=0)
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_xlim(0.0, 3.0)
    ax.set_xlabel(r"Total density [cells (1000 µm$^2$)$^{-1}$]")
    ft.format_axes(ax)
    return plotted


def shared_y_limits(
    tables: tuple[pd.DataFrame, ...],
    min_bin_pixels: int,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for table in tables:
        sub = rows_for_plot(table, min_bin_pixels)
        for column in ("q1", "median", "q3"):
            finite = sub[column].to_numpy(dtype=float)
            values.append(finite[np.isfinite(finite)])
    nonempty = [value for value in values if value.size]
    if not nonempty:
        raise RuntimeError("No finite IQR values available for plot limits")
    combined = np.concatenate(nonempty)
    low = float(np.min(combined))
    high = float(np.max(combined))
    span = max(high - low, 0.2)
    return low - 0.12 * span, high + 0.18 * span


def add_identity_columns(
    data: pd.DataFrame,
    *,
    dataset: str,
    kernel_um: int,
) -> pd.DataFrame:
    result = data.copy()
    result.insert(0, "dataset", dataset)
    result.insert(1, "kernel_um", int(kernel_um))
    result.insert(2, "growth_definition", "(rho_next/rho_previous - 1)/delta_days")
    result.insert(3, "uncertainty_display", "IQR (q1 to q3; descriptive spread, not CI)")
    return result


def write_tables(
    output_dir: Path,
    *,
    dataset: str,
    kernel_um: int,
    strategy_data: pd.DataFrame,
    pooled_data: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = output_dir / f"{dataset}_strategy_binned_statistics.tsv"
    pooled_path = output_dir / f"{dataset}_pooled_master_binned_statistics.tsv"
    add_identity_columns(
        strategy_data,
        dataset=dataset,
        kernel_um=kernel_um,
    ).to_csv(strategy_path, sep="\t", index=False)
    add_identity_columns(
        pooled_data,
        dataset=dataset,
        kernel_um=kernel_um,
    ).to_csv(pooled_path, sep="\t", index=False)


def plot_one(
    output_dir: Path,
    *,
    kernel_um: int,
    gate_name: str,
    phase: str,
    vitro_strategy: pd.DataFrame,
    vitro_pooled: pd.DataFrame,
    silico_strategy: pd.DataFrame,
    silico_pooled: pd.DataFrame,
    min_bin_pixels: int,
) -> Path:
    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=5.6,
        overrides={
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "figure.titlesize": 8,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55), sharex=True, sharey=True)
    vitro_strategies = plot_dataset(
        axes[0],
        vitro_strategy,
        vitro_pooled,
        title="In vitro",
        min_bin_pixels=min_bin_pixels,
    )
    silico_strategies = plot_dataset(
        axes[1],
        silico_strategy,
        silico_pooled,
        title="In silico",
        min_bin_pixels=min_bin_pixels,
    )
    axes[0].set_ylim(*FIXED_YLIM)
    axes[0].set_ylabel("Linear resistant growth [fraction day$^{-1}$]")
    axes[0].text(-0.16, 1.05, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.10, 1.05, "b", transform=axes[1].transAxes, fontweight="bold")

    strategies = [
        strategy
        for strategy in common.ALLOWED_STRATEGIES
        if strategy in set(vitro_strategies) | set(silico_strategies)
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=get_color(strategy, "#666666"),
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            markeredgewidth=0,
            label=get_display_name(strategy),
        )
        for strategy in strategies
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markersize=3.5,
            linewidth=2.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label="All included strategies pooled",
        )
    )
    fig.suptitle(
        f"{PHASE_LABELS[phase]} | {kernel_um} µm kernel",
        y=0.986,
    )
    fig.text(
        0.5,
        0.925,
        GATE_LABELS[gate_name],
        ha="center",
        va="center",
        fontsize=5.8,
        color="#333333",
    )
    fig.text(
        0.5,
        0.885,
        (
            "Median with IQR; black = pooled pixels; labels give n pixels; "
            "Pulse and KPP-Agent excluded; "
            r"linear growth = ($\rho_{next}/\rho_{previous}-1$)/$\Delta$days"
        ),
        ha="center",
        va="center",
        fontsize=5.4,
        color="#444444",
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.255, top=0.82, wspace=0.18)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "master_curve_in_vitro_vs_in_silico.pdf"
    ft.savefig(fig, output_path, transparent=False)
    plt.close(fig)
    return output_path


def summarize_bundle(
    bundle: common.AggregationBundle,
    gate_name: str,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregator = bundle.aggregators[(gate_name, phase)]
    strategy = common.summarize_iqr(
        aggregator,
        bundle.strategies,
        gate_name=gate_name,
        phase=phase,
    )
    pooled = common.summarize_iqr(
        common.pooled_aggregator(aggregator),
        ["All included strategies pooled"],
        gate_name=gate_name,
        phase=phase,
    )
    return strategy, pooled


def write_readme(
    output_root: Path,
    cache_root: Path,
    min_bin_pixels: int,
    outputs: list[Path],
) -> None:
    text = (
        "Linear resistant-growth master curves: in vitro vs in silico\n"
        "=============================================================\n\n"
        "Layout: kernel_[320|160|40]um / [ungated|gate...] / "
        "[overall|before_escape|after_escape].\n"
        "Each leaf folder contains one PDF plus the exact binned statistics "
        "used to draw its in-vitro and in-silico strategy and pooled curves.\n\n"
        "Growth: (rho_next/rho_previous - 1) / delta_days, i.e. average "
        "linear fractional growth per 24 h. Pixels with non-finite values or "
        "previous resistant density <= 1e-10 are undefined and excluded.\n"
        "X uses total density at the first frame of each interval. The standard "
        "gate uses resistant density at that same first frame.\n"
        "Uncertainty display: median with interquartile range (Q1-Q3). This is "
        "descriptive pixel-level spread, not a confidence interval, so it does "
        "not collapse merely because n is very large.\n"
        "Black curves pool every included pixel from CT, NT, AT50, AT100, "
        "ATagent, and AT⁺. Pulse and KPP-Agent are excluded from all curves.\n"
        "All PDF panels use fixed Y limits from -1 to 4 fraction day$^{-1}$.\n"
        "Before escape requires next_day <= the last confined day (or TTE for "
        "simulations). After escape requires first_day >= escape day/TTE. "
        "Intervals crossing escape are in overall only. Overall also includes "
        "wells/runs without a finite escape time.\n"
        f"Minimum plotted bin count: {min_bin_pixels}.\n"
        f"Replot-ready aggregate histogram caches: {cache_root}\n"
        f"PDF count: {len(outputs)}\n"
    )
    (output_root / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.expanduser()
    output_root = args.output_root.expanduser()
    if args.min_bin_pixels < 1:
        raise ValueError("--min-bin-pixels must be at least 1")
    output_root.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for kernel_um in KERNELS:
        print(f"[load] kernel={kernel_um} um", flush=True)
        vitro = load_kernel_bundle(cache_root, "in_vitro", kernel_um)
        silico = load_kernel_bundle(cache_root, "in_silico", kernel_um)
        for gate_name, _gate in common.GATES:
            for phase in common.PHASES:
                output_dir = (
                    output_root
                    / f"kernel_{kernel_um}um"
                    / GATE_FOLDERS[gate_name]
                    / phase
                )
                vitro_strategy, vitro_pooled = summarize_bundle(
                    vitro,
                    gate_name,
                    phase,
                )
                silico_strategy, silico_pooled = summarize_bundle(
                    silico,
                    gate_name,
                    phase,
                )
                write_tables(
                    output_dir,
                    dataset="in_vitro",
                    kernel_um=kernel_um,
                    strategy_data=vitro_strategy,
                    pooled_data=vitro_pooled,
                )
                write_tables(
                    output_dir,
                    dataset="in_silico",
                    kernel_um=kernel_um,
                    strategy_data=silico_strategy,
                    pooled_data=silico_pooled,
                )
                output = plot_one(
                    output_dir,
                    kernel_um=kernel_um,
                    gate_name=gate_name,
                    phase=phase,
                    vitro_strategy=vitro_strategy,
                    vitro_pooled=vitro_pooled,
                    silico_strategy=silico_strategy,
                    silico_pooled=silico_pooled,
                    min_bin_pixels=int(args.min_bin_pixels),
                )
                outputs.append(output)
                manifest_rows.append(
                    {
                        "kernel_um": kernel_um,
                        "gate": gate_name,
                        "phase": phase,
                        "pdf": str(output),
                        "in_vitro_pooled_pixels": int(vitro_pooled["n_pixels"].sum()),
                        "in_silico_pooled_pixels": int(silico_pooled["n_pixels"].sum()),
                    }
                )
                print(f"[saved] {output}", flush=True)
        del vitro, silico

    pd.DataFrame(manifest_rows).to_csv(
        output_root / "output_manifest.tsv",
        sep="\t",
        index=False,
    )
    write_readme(
        output_root,
        cache_root,
        int(args.min_bin_pixels),
        outputs,
    )
    print(f"[done] {len(outputs)} PDFs in {output_root}")


if __name__ == "__main__":
    main()
