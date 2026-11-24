#!/usr/bin/env python3
"""Plot total cell counts from Results_today files.

Panel 1 uses raw Dens wells from day 3. Panel 2 pools all other strategies
from day 1. Panel 3 shows the same non-Dens day-1 rows split by strategy.
Bars show means with sample standard deviations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

Path(os.environ.setdefault("MPLCONFIGDIR", r"C:\tmp\matplotlib-cache")).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from shared import figtools as ft
from shared.strategy_colors import STRATEGY_ORDER, get_color, get_display_name, normalize_strategy


DEFAULT_DATASET_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
DEFAULT_OUTPUT_STEM = "results_today_total_counts_dens_day3_other_day1"
RESULT_ID_RE = re.compile(
    r"^.+_(?P<well>[A-D][1-6])_(?P<strategy>.+)-(?P=well)_d(?P<day>\d+)$"
)


@dataclass(frozen=True)
class ResultRow:
    plate: str
    well: str
    raw_strategy: str
    strategy: str
    day: int
    total_count: int
    resistant_count: int


@dataclass(frozen=True)
class Summary:
    strategy: str
    day: int
    n: int
    mean: float
    sd: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot total cell counts from Results_today: raw Dens at day 3, "
            "pooled non-Dens rows at day 1, and non-Dens strategies at day 1."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.fspath(DEFAULT_DATASET_DIR),
        help=f"Dataset root. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=os.fspath(PROJECT_ROOT / "figures" / "summary"),
        help="Output directory. Default: figures/summary.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output file stem. Default: {DEFAULT_OUTPUT_STEM}",
    )
    return parser.parse_args()


def parse_row(line: str, *, plate: str, expected_day: int) -> ResultRow:
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"Expected 3 columns in {plate}: {line!r}")

    match = RESULT_ID_RE.match(parts[0])
    if match is None:
        raise ValueError(f"Could not parse result id in {plate}: {parts[0]}")

    day = int(match.group("day"))
    if day != expected_day:
        raise ValueError(f"Expected day {expected_day}, got day {day}: {parts[0]}")

    raw_strategy = match.group("strategy")
    return ResultRow(
        plate=plate,
        well=match.group("well"),
        raw_strategy=raw_strategy,
        strategy=normalize_strategy(raw_strategy),
        day=day,
        total_count=int(parts[1]),
        resistant_count=int(parts[2]),
    )


def read_day(dataset_dir: Path, day: int) -> list[ResultRow]:
    files = sorted(dataset_dir.glob(f"Plate_*/Results_today/Results_d{day:02d}.txt"))
    if not files:
        raise FileNotFoundError(f"No Results_d{day:02d}.txt files found under {dataset_dir}")

    rows: list[ResultRow] = []
    for path in files:
        plate = path.parents[1].name
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(parse_row(line, plate=plate, expected_day=day))
    return rows


def ordered_strategies(strategies: set[str]) -> list[str]:
    ordered = [strategy for strategy in STRATEGY_ORDER if strategy in strategies]
    ordered.extend(sorted(strategies.difference(ordered)))
    return ordered


def summarize(rows: list[ResultRow]) -> list[Summary]:
    summaries: list[Summary] = []
    for strategy in ordered_strategies({row.strategy for row in rows}):
        values = np.array([row.total_count for row in rows if row.strategy == strategy], dtype=float)
        summaries.append(
            Summary(
                strategy=strategy,
                day=rows[0].day,
                n=int(values.size),
                mean=float(np.mean(values)),
                sd=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            )
        )
    return summaries


def summarize_pooled(rows: list[ResultRow], *, label: str) -> Summary:
    values = np.array([row.total_count for row in rows], dtype=float)
    return Summary(
        strategy=label,
        day=rows[0].day,
        n=int(values.size),
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    )


def write_csv(path: Path, summaries: list[Summary], rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "day", "n", "mean_total_count", "sd_total_count"])
        for summary in summaries:
            writer.writerow([summary.strategy, summary.day, summary.n, summary.mean, summary.sd])
        writer.writerow([])
        writer.writerow(["plate", "well", "raw_strategy", "strategy", "day", "total_count", "resistant_count"])
        for row in rows:
            writer.writerow(
                [
                    row.plate,
                    row.well,
                    row.raw_strategy,
                    row.strategy,
                    row.day,
                    row.total_count,
                    row.resistant_count,
                ]
            )


def draw_panel(ax: plt.Axes, summaries: list[Summary], rows: list[ResultRow], title: str) -> None:
    strategies = [summary.strategy for summary in summaries]
    x = np.arange(len(strategies), dtype=float)
    means = np.array([summary.mean for summary in summaries], dtype=float)
    sds = np.array([summary.sd for summary in summaries], dtype=float)

    ax.bar(
        x,
        means,
        yerr=sds,
        capsize=2.5,
        color=[get_color(strategy, "#777777") for strategy in strategies],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        zorder=2,
    )

    rng = np.random.default_rng(7)
    for idx, strategy in enumerate(strategies):
        values = np.array([row.total_count for row in rows if row.strategy == strategy], dtype=float)
        ax.scatter(
            np.full(values.size, x[idx]) + rng.uniform(-0.16, 0.16, values.size),
            values,
            s=8,
            facecolor="white",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.75,
            zorder=3,
        )
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([get_display_name(strategy) for strategy in strategies], rotation=45, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=1)


def plot(dens_rows: list[ResultRow], other_rows: list[ResultRow], out_base: Path) -> None:
    dens_summary = summarize(dens_rows)
    pooled_other_summary = [summarize_pooled(other_rows, label="All non-Dens")]
    other_summaries = summarize(other_rows)
    all_summaries = dens_summary + pooled_other_summary + other_summaries

    ft.apply_nature_style(
        text_pt=7,
        tick_pt=6,
        legend_pt=6,
        overrides={
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "ytick.major.size": 2.5,
        },
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.0, 3.2),
        dpi=300,
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 3.4]},
        constrained_layout=True,
    )

    draw_panel(axes[0], dens_summary, dens_rows, "AT100+ / Dens, day 3")
    draw_panel(axes[1], pooled_other_summary, other_rows, "All non-Dens, day 1")
    draw_panel(axes[2], other_summaries, other_rows, "Non-Dens by strategy, day 1")
    axes[0].set_ylabel("Total cell count")

    ymax = max(summary.mean + summary.sd for summary in all_summaries) * 1.18
    for ax in axes:
        ax.set_ylim(0, ymax)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, out_base.with_suffix(".pdf"))
    ft.savefig(fig, out_base.with_suffix(".png"), dpi=300, transparent=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    out_base = Path(args.output_dir).expanduser() / args.output_stem

    day1_rows = read_day(dataset_dir, 1)
    day3_rows = read_day(dataset_dir, 3)
    dens_rows = [row for row in day3_rows if row.raw_strategy == "Dens"]
    other_rows = [row for row in day1_rows if row.raw_strategy != "Dens"]

    if not dens_rows:
        raise RuntimeError("No raw Dens rows found in day-3 Results_today files")
    if not other_rows:
        raise RuntimeError("No non-Dens rows found in day-1 Results_today files")

    summaries = summarize(dens_rows) + [summarize_pooled(other_rows, label="All non-Dens")] + summarize(other_rows)
    write_csv(out_base.with_suffix(".csv"), summaries, dens_rows + other_rows)
    plot(dens_rows, other_rows, out_base)

    print(f"Read {len(dens_rows)} Dens rows from day 3")
    print(f"Read {len(other_rows)} non-Dens rows from day 1")
    for summary in summaries:
        print(
            f"{get_display_name(summary.strategy)} day {summary.day}: "
            f"n={summary.n}, mean={summary.mean:.1f}, sd={summary.sd:.1f}"
        )
    print(f"Saved: {out_base.with_suffix('.pdf')}")
    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
