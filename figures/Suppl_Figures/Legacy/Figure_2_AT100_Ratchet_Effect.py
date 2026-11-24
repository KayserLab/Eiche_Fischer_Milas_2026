"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - \\]+?
# - D:\2025_08_Co
# Upstream/related script reference(s):
# - and shared/figtools.py
# - shared/experiment_filters.py
# - import matplotlib.py
# - plot_all_well_confinement_escape_trajectories.py
# =============================================================================
TL;DR
-----
This script visualizes the AT100 ratchet-effect trajectories as a supplementary
plot.

Detailed
--------
Inputs:
- Raw `Results/*.txt` replicate trajectories.
- Shared replicate exclusion/truncation rules.

Outputs:
- Two selected Figure 2 relationship figures: confinement margin versus total
  count for Plate 1/B4, and normalized total count versus resistant count for
  Plate 5/A4.

How it works:
- Loads replicate trajectories and normalizes total burden to baseline.
- Filters to the relevant AT100 branch and applies the shared exclusion logic.
- Draws per-replicate trajectory segments in the manuscript style used for the
  ratchet-effect illustration.

Pipeline role:
- Supplementary trajectory plot connected to the Figure 2 treatment-dynamics
  story.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental Results/*.txt plus confinement/escape CSVs resolved by
# plot_all_well_confinement_escape_trajectories.py.
# Scripts: plot_all_well_confinement_escape_trajectories.py,
# shared/experiment_filters.py, and shared/figtools.py.
# Outputs: selected AT100 ratchet-effect supplementary panels.

import re
import shutil
import sys
import tempfile
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared import experiment_filters
from shared.data_paths import EXP_RAW_DATA, figure_plot_dir


FIGURE_2_DIR = REPO_ROOT / "figures" / "Figure_2"
if str(FIGURE_2_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_2_DIR))
from figures.Suppl_Figures.Legacy import plot_all_well_confinement_escape_trajectories as confinement_source

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 50.0
FIGURE_AXES_HEIGHT_MM = 35.0
FIGURE_MARGINS_MM = (12.0, 3.0, 12.0, 3.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
DPI = 300

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    },
)
DATA_ROOT = EXP_RAW_DATA
PLATES = [f"Plate_{i}" for i in range(1, 6)]
TARGET_STRATEGY = "AT100"
OUT_DIR = figure_plot_dir("Suppl_Figures", "ratchet_effect", create=False)

RESULTS_RE = re.compile(
    r"^Results_G418_a2a4_(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$",
    re.IGNORECASE,
)
TOKEN_DAY_RE = re.compile(r"_d(?P<day>\d{1,3})\b", re.IGNORECASE)
IGNORE_FILENAME_PARTS = ("overexposed", "seg")
TIME_CMAP = LinearSegmentedColormap.from_list("blue_orange_time", ["#2C7BB6", "#F28E2B"])



@dataclass(frozen=True)
class ResultRow:
    plate: str
    sample: str
    strategy: str
    day: int
    sensitive: int
    resistant: int


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def save_figure(fig: plt.Figure, output: Path) -> None:
    """Render locally before copying to the network figure directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.gettempdir()) / f"{output.stem}_{uuid4().hex}.pdf"
    try:
        ft.savefig(fig, temporary)
        shutil.copy2(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_day(token: str) -> int:
    match = TOKEN_DAY_RE.search(token)
    if not match:
        raise ValueError(f"Cannot parse day from token: {token}")
    return int(match.group("day"))


def iter_result_rows() -> Iterable[ResultRow]:
    for plate in PLATES:
        results_dir = DATA_ROOT / plate / "Results"
        if not results_dir.is_dir():
            continue

        for path in sorted(results_dir.glob("*.txt")):
            if any(tag in path.name.lower() for tag in IGNORE_FILENAME_PARTS):
                continue

            match = RESULTS_RE.match(path.name)
            if not match:
                continue

            strategy = str(match.group("strategy"))
            if strategy != TARGET_STRATEGY:
                continue

            well = match.group("well").upper()
            sample = f"{well}_{strategy}-{well}"

            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    try:
                        yield ResultRow(
                            plate=plate,
                            sample=sample,
                            strategy=strategy,
                            day=parse_day(parts[0]),
                            sensitive=int(parts[1]),
                            resistant=int(parts[2]),
                        )
                    except ValueError:
                        continue


def normalize_to_baseline_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["total"] > 0].copy()

    base0 = (
        df[df["day"] == 0]
        .groupby(["plate", "sample"], as_index=False)["total"]
        .first()
        .rename(columns={"total": "base0"})
    )
    base_earliest = (
        df.sort_values(["plate", "sample", "day"])
        .groupby(["plate", "sample"], as_index=False)
        .first()[["plate", "sample", "total"]]
        .rename(columns={"total": "base_earliest"})
    )

    out = (
        df.merge(base0, on=["plate", "sample"], how="left")
        .merge(base_earliest, on=["plate", "sample"], how="left")
    )
    out["base_total"] = out["base0"].fillna(out["base_earliest"])
    out = out[out["base_total"] > 0].copy()
    out["sensitive_norm"] = out["sensitive"] / out["base_total"]
    out["resistant_norm"] = out["resistant"] / out["base_total"]
    out["total_norm"] = out["total"] / out["base_total"]
    return out


def load_results() -> pd.DataFrame:
    df = pd.DataFrame([row.__dict__ for row in iter_result_rows()])
    if df.empty:
        raise RuntimeError("No AT100 rows loaded from Results/*.txt.")

    df["total"] = df["sensitive"] + df["resistant"]
    df = experiment_filters.apply_sample_exclusions(
        df,
        exclude_map=experiment_filters.EXCLUDE_SAMPLES,
        column="sample",
        cutoff_column="day",
        strict=False,
    )
    df = normalize_to_baseline_total(df)
    confinement = confinement_source.load_confinement(
        confinement_source.DEFAULT_CONFINEMENT_CSV,
        include_excluded=False,
    ).rename(columns={"confinement_thickness": "confinement_margin"})
    df = df.merge(
        confinement,
        on=["plate", "sample", "day"],
        how="left",
        validate="one_to_one",
    )
    df = df.sort_values(["plate", "sample", "day"]).reset_index(drop=True)
    return df


def padded_limits(values: np.ndarray, pad_frac: float = 0.04) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    lo = min(0.0, lo)
    if hi == lo:
        span = 1.0 if hi == 0 else abs(hi) * 0.05
        return lo, hi + span
    return lo, hi + (hi - lo) * pad_frac


def positive_padded_limits(values: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float]:
    """Return strictly positive limits suitable for a logarithmic axis."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0, 10.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi == lo:
        return lo / 1.2, hi * 1.2
    factor = (hi / lo) ** pad_frac
    return lo / factor, hi * factor


def build_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def make_plot(rep_df: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float], out_pdf: Path) -> None:
    rep_df = rep_df.sort_values("day").reset_index(drop=True)
    x = rep_df["resistant_norm"].to_numpy(dtype=float)
    y = rep_df["sensitive_norm"].to_numpy(dtype=float)
    progress = np.linspace(0.0, 1.0, len(rep_df))

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    x_line = np.linspace(xlim[0], xlim[1], 200)
    y_line = 1.0 - x_line
    ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.0)

    if len(rep_df) >= 2:
        seg_mask = (
            np.isfinite(x[:-1]) & np.isfinite(y[:-1]) &
            np.isfinite(x[1:]) & np.isfinite(y[1:]) &
            (x[:-1] > 0) & (x[1:] > 0)
        )
        if np.any(seg_mask):
            seg_x0 = x[:-1][seg_mask]
            seg_y0 = y[:-1][seg_mask]
            seg_x1 = x[1:][seg_mask]
            seg_y1 = y[1:][seg_mask]
            segments = np.stack(
                [
                    np.column_stack([seg_x0, seg_y0]),
                    np.column_stack([seg_x1, seg_y1]),
                ],
                axis=1,
            )
            seg_progress = (progress[:-1][seg_mask] + progress[1:][seg_mask]) / 2.0
            lc = LineCollection(
                segments,
                cmap=TIME_CMAP,
                norm=Normalize(vmin=0.0, vmax=1.0),
                linewidths=1.4,
            )
            lc.set_array(seg_progress)
            ax.add_collection(lc)

    pt_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    ax.scatter(
        x[pt_mask],
        y[pt_mask],
        c=progress[pt_mask],
        cmap=TIME_CMAP,
        norm=Normalize(vmin=0.0, vmax=1.0),
        s=20,
        edgecolors="black",
        linewidths=0.25,
        zorder=3,
    )

    ax.set_xlabel("Normalized resistant cell count")
    ax.set_ylabel("Normalized sensitive cell count")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{rep_df['plate'].iloc[0]}: {rep_df['sample'].iloc[0]}", fontsize=7)
    style_axis(ax)

    save_figure(fig, out_pdf)
    plt.close(fig)
    print(f"[save] {out_pdf}")


def make_relationship_plot(
    rep_df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str,
    out_pdf: Path,
    x_log: bool = False,
    y_log: bool = False,
    horizontal_reference: float | None = None,
    vertical_reference: float | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Draw one time-coloured count relationship for a selected replicate."""
    data = rep_df.sort_values("day").dropna(subset=[x_column, y_column]).copy()
    if x_log:
        data = data[data[x_column] > 0].copy()
    if y_log:
        data = data[data[y_column] > 0].copy()
    if data.empty:
        print(f"[skip] no finite {y_column} values for {out_pdf.name}")
        return
    x = data[x_column].to_numpy(dtype=float)
    y = data[y_column].to_numpy(dtype=float)
    progress = np.linspace(0.0, 1.0, len(data))

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    if len(data) >= 2:
        segments = build_segments(x, y)
        lc = LineCollection(
            segments,
            cmap=TIME_CMAP,
            norm=Normalize(vmin=0.0, vmax=1.0),
            linewidths=1.4,
        )
        lc.set_array((progress[:-1] + progress[1:]) / 2.0)
        ax.add_collection(lc)
    # Keep a marker and its label together in the draw order: a later point
    # therefore covers an earlier point and its day label when they overlap.
    for index, (x_value, y_value, day, point_progress) in enumerate(
        zip(x, y, data["day"], progress)
    ):
        marker_zorder = 3 + 2 * index
        ax.scatter(
            [x_value], [y_value], color=TIME_CMAP(point_progress),
            s=20, edgecolors="black", linewidths=0.25, zorder=marker_zorder,
        )
        ax.annotate(
            str(int(round(float(day)))), (x_value, y_value),
            xytext=(0, 0), textcoords="offset points",
            ha="center", va="center", fontsize=3.0, color="white",
            zorder=marker_zorder + 1,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*(x_limits if x_limits is not None else (
        positive_padded_limits(x) if x_log else padded_limits(x)
    )))
    ax.set_ylim(*(positive_padded_limits(y) if y_log else padded_limits(y)))
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if horizontal_reference is not None:
        ax.axhline(horizontal_reference, color="black", linestyle="--", linewidth=0.8, zorder=2)
    if vertical_reference is not None:
        ax.axvline(vertical_reference, color="black", linestyle="--", linewidth=0.8, zorder=2)
    style_axis(ax)
    save_figure(fig, out_pdf)
    plt.close(fig)
    print(f"[save] {out_pdf}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    print("[counts] filtered AT100 replicates:")
    print(f"  replicates={int(df['sample'].nunique())}, rows={len(df)}")

    confinement_example = df[
        (df["plate"] == "Plate_1") & (df["sample"] == "B4_AT100-B4")
    ].copy()
    if confinement_example.empty:
        raise RuntimeError("Selected confinement example Plate_1/B4_AT100-B4 was not found.")
    make_relationship_plot(
        confinement_example,
        x_column="total",
        y_column="confinement_margin",
        xlabel="Total cell count",
        ylabel="Confinement margin",
        out_pdf=OUT_DIR / "Plate_1_B4_AT100-B4_Confinement_Margin_vs_Total_Count.pdf",
        y_log=True,
        vertical_reference=float(confinement_example["base_total"].iloc[0]),
        x_limits=(10_000.0, float(np.max(confinement_example["total"])) * 1.04),
    )

    total_example = df[
        (df["plate"] == "Plate_5") & (df["sample"] == "A4_AT100-A4")
    ].copy()
    if total_example.empty:
        raise RuntimeError("Selected total-count example Plate_5/A4_AT100-A4 was not found.")
    make_relationship_plot(
        total_example,
        x_column="resistant",
        y_column="total_norm",
        xlabel="Resistant cell count",
        ylabel="Normalized total cell count",
        out_pdf=OUT_DIR / "Plate_5_A4_AT100-A4_Total_Count_vs_Resistant_Count.pdf",
        x_log=True,
        horizontal_reference=1.0,
    )


if __name__ == "__main__":
    main()
