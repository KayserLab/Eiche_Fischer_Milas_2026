"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - \\]+?
# - D:\2025_08_Co
# Upstream/related script reference(s):
# - analysis/supplementary/Figure_5_Escape_Day.py
# - and shared/figtools.py
# - shared/experiment_filters.py
# - import matplotlib.py
# - shared/strategy_colors.py
# =============================================================================
TL;DR
-----
This script compares resistant growth immediately before escape with the
preceding sensitive-cell burden.

Detailed
--------
Inputs:
- Raw `Results/*.txt` replicate trajectories.
- Escape-front summary data.
- Shared replicate exclusion/truncation rules.

Outputs:
- Figure 2 overlays, scatter plots, and density-contour views of pre-escape
  resistant growth versus prior sensitive burden.

How it works:
- Loads replicate trajectories and normalizes total-burden information.
- Joins those trajectories to replicate-level escape timing.
- Extracts the interval immediately preceding escape and computes resistant
  growth versus the previous sensitive count for that interval.
- Visualizes the relationship with multiple complementary plot styles.

Pipeline role:
- Figure 2 plotting script for the pre-escape growth-versus-burden question.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Plate_*/Results/*.txt and the escape-day summary used by
# analysis/supplementary/Figure_5_Escape_Day.py.
# Scripts: analysis/supplementary/Figure_5_Escape_Day.py, shared/experiment_filters.py, shared/strategy_colors.py,
# and shared/figtools.py.
# Outputs: pre-escape growth-versus-sensitive-count supplementary PDFs.

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared import experiment_filters
from shared.data_paths import EXP_RAW_DATA
from analysis.legacy import Figure_5_Escape_Day as figure4_escape
from shared.strategy_colors import COLOR_MAP


DPI = 300

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to axes boxes: the physical x-axis and y-axis
# length. Margins are extra space for labels, ticks, legends, and titles.
GRID_AXES_WIDTH_MM = 40.0
GRID_AXES_HEIGHT_MM = 40.0
GRID_MARGINS_MM = (15.0, 5.0, 15.0, 5.0)  # left, right, bottom, top
GRID_PANEL_GAP_MM = 8.0
OVERLAY_AXES_WIDTH_MM = 40.0
OVERLAY_AXES_HEIGHT_MM = 40.0
OVERLAY_MARGINS_MM = (19.32, 2.48, 9.35, 1.22)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
STRATEGIES = ["NT", "CT", "AT50", "AT100"]
DATA_ROOT = EXP_RAW_DATA
PLATES = [f"Plate_{i}" for i in range(1, 6)]
OUT_PLOT_DIR = ft.paper_figure_data_dir(__file__)
OUT_PDF = OUT_PLOT_DIR / "Figure_2_PreEscape_Resistant_Growth_vs_Previous_Sensitive_Count.pdf"
OUT_PDF_REL_OVERLAY = OUT_PLOT_DIR / "Figure_2_PreEscape_Relative_Resistant_Growth_vs_Previous_Sensitive_Count_CrossOverlay.pdf"

RESULTS_RE = re.compile(
    r"^Results_G418_a2a4_(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$",
    re.IGNORECASE,
)
TOKEN_DAY_RE = re.compile(r"_d(?P<day>\d{1,3})\b", re.IGNORECASE)
IGNORE_FILENAME_PARTS = ("overexposed", "seg")
GRID_N = 140
CONTOUR_QUANTILES = (0.80, 0.90, 0.95)
FILL_ALPHA_MIN = 0.10
FILL_ALPHA_MAX = 0.75
OUTLINE_LW = 0.8
LEGEND_FONTSIZE = 5.5
CROSS_MARKERSIZE = 5.2
CROSS_MARKEREDGEWIDTH = 1.0
CROSS_ELINEWIDTH = 1.0

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_FONTSIZE,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    },
)


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
    ax.tick_params(direction="out", pad=2, length=2.5, width=0.8, labelsize=6)
    ax.grid(alpha=0.25)


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
            if strategy not in STRATEGIES:
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
    return out


def load_results() -> pd.DataFrame:
    df = pd.DataFrame([row.__dict__ for row in iter_result_rows()])
    if df.empty:
        raise RuntimeError("No rows loaded from Results/*.txt.")

    df["total"] = df["sensitive"] + df["resistant"]
    df = experiment_filters.apply_sample_exclusions(
        df,
        exclude_map=experiment_filters.EXCLUDE_SAMPLES,
        column="sample",
        cutoff_column="day",
        strict=False,
    )
    df = normalize_to_baseline_total(df)
    df = df.sort_values(["strategy", "plate", "sample", "day"]).reset_index(drop=True)
    return df


def load_escape_summary() -> pd.DataFrame:
    summary = pd.read_csv(figure4_escape.SUM_CSV)
    if "replicate" not in summary.columns and "sample" in summary.columns:
        summary["replicate"] = summary["sample"]
    if "status" in summary.columns:
        summary = summary[summary["status"].astype(str).str.lower() == "ok"].copy()

    summary["strategy"] = summary["replicate"].astype(str).map(figure4_escape.parse_strategy)
    if "escape_day" in summary.columns:
        summary["escape_day"] = figure4_escape.safe_numeric(summary["escape_day"])
    elif "first_escape_day" in summary.columns:
        summary["escape_day"] = figure4_escape.safe_numeric(summary["first_escape_day"])
    else:
        raise ValueError("Escape summary is missing escape_day/first_escape_day.")

    keep_cols = [
        col
        for col in [
            "plate",
            "replicate",
            "strategy",
            "escape_day",
            "escape_detected",
        ]
        if col in summary.columns
    ]
    return summary[keep_cols].copy()


def build_pre_escape_interval_df(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    nt_seen: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    for _, row in summary_df.iterrows():
        plate = str(row["plate"])
        sample = str(row["replicate"])
        strategy = str(row["strategy"])
        status, cutoff = figure4_escape.get_filter_rule(sample)
        escape_day = pd.to_numeric(pd.Series([row.get("escape_day")]), errors="coerce").iloc[0]

        if "escape_detected" in row.index:
            escaped = str(row["escape_detected"]).strip().lower() in {"true", "1", "yes"}
        else:
            escaped = bool(np.isfinite(escape_day))

        if status == "exclude":
            continue
        if status == "truncate" and cutoff is not None and (not np.isfinite(escape_day) or float(escape_day) > float(cutoff)):
            continue
        if not (escaped and np.isfinite(escape_day)):
            continue

        g = results_df[
            (results_df["plate"].astype(str) == plate)
            & (results_df["sample"].astype(str) == sample)
            & (results_df["strategy"].astype(str) == strategy)
        ].copy()
        if len(g) < 2:
            continue
        g = g.sort_values("day").reset_index(drop=True)

        for i in range(1, len(g)):
            prev_day = float(pd.to_numeric(g.loc[i - 1, "day"], errors="coerce"))
            day = float(pd.to_numeric(g.loc[i, "day"], errors="coerce"))
            if not (np.isfinite(prev_day) and np.isfinite(day)):
                continue
            if day <= prev_day:
                continue
            if day > float(escape_day):
                continue

            sensitive_prev = float(pd.to_numeric(g.loc[i - 1, "sensitive_norm"], errors="coerce"))
            resistant_prev = float(pd.to_numeric(g.loc[i - 1, "resistant_norm"], errors="coerce"))
            resistant_curr = float(pd.to_numeric(g.loc[i, "resistant_norm"], errors="coerce"))
            if not all(np.isfinite(v) for v in [sensitive_prev, resistant_prev, resistant_curr]):
                continue

            dt = day - prev_day
            drdt = (resistant_curr - resistant_prev) / dt
            rel_drdt = np.nan
            if resistant_prev > 0:
                rel_drdt = drdt / resistant_prev

            rows.append(
                {
                    "plate": plate,
                    "sample": sample,
                    "strategy": strategy,
                    "prev_day": prev_day,
                    "day": day,
                    "escape_day": float(escape_day),
                    "sensitive_norm_prev": sensitive_prev,
                    "resistant_norm_prev": resistant_prev,
                    "resistant_norm_curr": resistant_curr,
                    "dt": dt,
                    "drdt": drdt,
                    "rel_drdt_over_rprev": rel_drdt,
                }
            )

    nt_df = results_df[results_df["strategy"].astype(str) == "NT"].copy()
    for (plate, sample, strategy), g in nt_df.groupby(["plate", "sample", "strategy"], sort=True):
        nt_seen.add((str(plate), str(sample), str(strategy)))
        if len(g) < 2:
            continue
        g = g.sort_values("day").reset_index(drop=True)

        for i in range(1, len(g)):
            prev_day = float(pd.to_numeric(g.loc[i - 1, "day"], errors="coerce"))
            day = float(pd.to_numeric(g.loc[i, "day"], errors="coerce"))
            if not (np.isfinite(prev_day) and np.isfinite(day)):
                continue
            if day <= prev_day:
                continue

            sensitive_prev = float(pd.to_numeric(g.loc[i - 1, "sensitive_norm"], errors="coerce"))
            resistant_prev = float(pd.to_numeric(g.loc[i - 1, "resistant_norm"], errors="coerce"))
            resistant_curr = float(pd.to_numeric(g.loc[i, "resistant_norm"], errors="coerce"))
            if not all(np.isfinite(v) for v in [sensitive_prev, resistant_prev, resistant_curr]):
                continue

            dt = day - prev_day
            drdt = (resistant_curr - resistant_prev) / dt
            rel_drdt = np.nan
            if resistant_prev > 0:
                rel_drdt = drdt / resistant_prev

            rows.append(
                {
                    "plate": str(plate),
                    "sample": str(sample),
                    "strategy": str(strategy),
                    "prev_day": prev_day,
                    "day": day,
                    "escape_day": np.nan,
                    "sensitive_norm_prev": sensitive_prev,
                    "resistant_norm_prev": resistant_prev,
                    "resistant_norm_curr": resistant_curr,
                    "dt": dt,
                    "drdt": drdt,
                    "rel_drdt_over_rprev": rel_drdt,
                }
            )

    return pd.DataFrame(rows)


def finite_limits(values: np.ndarray, pad_frac: float = 0.04, lower_zero: bool = False) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0) if lower_zero else (-1.0, 1.0)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lower_zero:
        lo = 0.0
    if hi == lo:
        span = 1.0 if hi == 0 else abs(hi) * 0.05
        return lo - (0.0 if lower_zero else span), hi + span
    pad = (hi - lo) * pad_frac
    lo_out = lo if lower_zero else lo - pad
    return lo_out, hi + pad


def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_n: int = GRID_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return None

    xx = np.linspace(xlim[0], xlim[1], grid_n)
    yy = np.linspace(ylim[0], ylim[1], grid_n)
    xg, yg = np.meshgrid(xx, yy)
    pts = np.vstack([x, y])
    positions = np.vstack([xg.ravel(), yg.ravel()])

    try:
        kde = gaussian_kde(pts)
        z = kde(positions).reshape(xg.shape)
        return xg, yg, z
    except LinAlgError:
        jitter = 1e-6
        pts_j = np.vstack([
            x + np.random.normal(0, jitter, size=x.size),
            y + np.random.normal(0, jitter, size=y.size),
        ])
        kde = gaussian_kde(pts_j)
        z = kde(positions).reshape(xg.shape)
        return xg, yg, z


def filled_contours_on_ax(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    base_color: str,
    zorder: int,
) -> bool:
    density = compute_density_grid(x, y, xlim=xlim, ylim=ylim, grid_n=GRID_N)
    if density is None:
        return False
    xg, yg, z = density
    vals = z[np.isfinite(z) & (z > 0)]
    if vals.size == 0:
        return False

    core_levels = np.quantile(vals, CONTOUR_QUANTILES)
    levels = np.concatenate((core_levels, [vals.max() * 1.001]))
    n_regions = len(levels) - 1
    base_rgba = np.array(mcolors.to_rgba(base_color))
    colors = []
    for i in range(n_regions):
        alpha = FILL_ALPHA_MIN + (FILL_ALPHA_MAX - FILL_ALPHA_MIN) * (i / max(1, n_regions - 1))
        rgba = base_rgba.copy()
        rgba[-1] = alpha
        colors.append(tuple(rgba))

    ax.contourf(
        xg,
        yg,
        z,
        levels=levels,
        colors=colors,
        antialiased=True,
        zorder=zorder,
    )
    ax.contour(
        xg,
        yg,
        z,
        levels=core_levels,
        colors=[base_color],
        linewidths=OUTLINE_LW,
        zorder=zorder + 1,
    )
    return True


def plot_strategy_crosses(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str) -> None:
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy].copy()
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue

        x_med = float(np.median(x))
        x_q25 = float(np.quantile(x, 0.25))
        x_q75 = float(np.quantile(x, 0.75))
        y_med = float(np.median(y))
        y_q25 = float(np.quantile(y, 0.25))
        y_q75 = float(np.quantile(y, 0.75))

        ax.errorbar(
            x_med,
            y_med,
            xerr=[[x_med - x_q25], [x_q75 - x_med]],
            yerr=[[y_med - y_q25], [y_q75 - y_med]],
            fmt="x",
            markersize=CROSS_MARKERSIZE,
            markeredgewidth=CROSS_MARKEREDGEWIDTH,
            color=COLOR_MAP[strategy],
            ecolor=COLOR_MAP[strategy],
            elinewidth=CROSS_ELINEWIDTH,
            capsize=0,
            label=strategy,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7)
    style_axis(ax)


def plot_strategy_scatter(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str) -> None:
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy].copy()
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue

        ax.scatter(
            x,
            y,
            s=10,
            color=COLOR_MAP[strategy],
            alpha=0.28,
            edgecolors="none",
            label=f"{strategy} (n={x.size})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7)
    style_axis(ax)


def plot_strategy_contours(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for zorder, strategy in enumerate(STRATEGIES, start=1):
        sub = df[df["strategy"] == strategy].copy()
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        filled_contours_on_ax(
            ax,
            x,
            y,
            xlim=xlim,
            ylim=ylim,
            base_color=COLOR_MAP[strategy],
            zorder=zorder,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7)
    style_axis(ax)


def print_summary(df: pd.DataFrame) -> None:
    print("[counts] pre-escape intervals:")
    print("  Treated strategies use only intervals ending on or before the true escape day.")
    print("  NT uses the full filtered trajectory and all intervals count as pre-escape.")
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        n_rep = int(sub["sample"].nunique()) if not sub.empty else 0
        n_drdt = int(np.isfinite(pd.to_numeric(sub["drdt"], errors="coerce")).sum())
        n_rel = int(np.isfinite(pd.to_numeric(sub["rel_drdt_over_rprev"], errors="coerce")).sum())
        print(f"  {strategy}: replicates={n_rep}, dR/dt points={n_drdt}, relative dR/dt/R points={n_rel}")

    for y_col, label in [
        ("drdt", "dR/dt"),
        ("rel_drdt_over_rprev", "relative dR/dt/R_prev"),
    ]:
        print(f"[stats] Strategy medians and IQRs for {label} vs previous-day normalized sensitive count")
        for strategy in STRATEGIES:
            sub = df[df["strategy"] == strategy].copy()
            x = pd.to_numeric(sub["sensitive_norm_prev"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size == 0:
                print(f"  {strategy}: n=0")
                continue
            print(
                f"  {strategy}: "
                f"x median={float(np.median(x)):.4f}, x IQR=({float(np.quantile(x, 0.25)):.4f}, {float(np.quantile(x, 0.75)):.4f}); "
                f"y median={float(np.median(y)):.4f}, y IQR=({float(np.quantile(y, 0.25)):.4f}, {float(np.quantile(y, 0.75)):.4f})"
            )


def make_relative_overlay_figure(rel_df: pd.DataFrame, out_pdf: Path) -> None:
    plot_df = rel_df.copy()

    fig, ax = ft.new_figure(
        width=OVERLAY_AXES_WIDTH_MM,
        height_mm=OVERLAY_AXES_HEIGHT_MM,
        margins=OVERLAY_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    plot_strategy_crosses(
        ax,
        plot_df,
        x_col="sensitive_norm_prev",
        y_col="rel_drdt_over_rprev",
        xlabel="Normalized sensitive count",
        ylabel="Resistant growth rate",
        title="",
    )

    ax.set_xlim(0.0, 2.7)
    ax.set_ylim(-0.05, 0.65)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

    ft.savefig(fig, out_pdf)
    plt.close(fig)
    print(f"[save] {out_pdf}")


def make_figure(interval_df: pd.DataFrame, out_pdf: Path) -> None:
    rel_df = interval_df[np.isfinite(pd.to_numeric(interval_df["rel_drdt_over_rprev"], errors="coerce"))].copy()
    xlim = finite_limits(interval_df["sensitive_norm_prev"].to_numpy(dtype=float), pad_frac=0.04, lower_zero=True)
    drdt_ylim = finite_limits(interval_df["drdt"].to_numpy(dtype=float), pad_frac=0.05, lower_zero=False)
    rel_ylim = finite_limits(rel_df["rel_drdt_over_rprev"].to_numpy(dtype=float), pad_frac=0.05, lower_zero=False)

    fig, axes = ft.new_subplots(
        3,
        2,
        ax_width_mm=GRID_AXES_WIDTH_MM,
        ax_height_mm=GRID_AXES_HEIGHT_MM,
        margins=GRID_MARGINS_MM,
        wspace_mm=GRID_PANEL_GAP_MM,
        hspace_mm=GRID_PANEL_GAP_MM,
        dpi=DPI,
        apply_style=False,
    )

    plot_strategy_crosses(
        axes[0, 0],
        interval_df,
        x_col="sensitive_norm_prev",
        y_col="drdt",
        xlabel="Previous-day normalized sensitive count",
        ylabel="dR/dt [1/day]",
        title="Median + IQR",
    )
    plot_strategy_crosses(
        axes[0, 1],
        rel_df,
        x_col="sensitive_norm_prev",
        y_col="rel_drdt_over_rprev",
        xlabel="Previous-day normalized sensitive count",
        ylabel="dR/dt / R_prev [1/day]",
        title="Median + IQR",
    )
    plot_strategy_scatter(
        axes[1, 0],
        interval_df,
        x_col="sensitive_norm_prev",
        y_col="drdt",
        xlabel="Previous-day normalized sensitive count",
        ylabel="dR/dt [1/day]",
        title="Scatter",
    )
    plot_strategy_scatter(
        axes[1, 1],
        rel_df,
        x_col="sensitive_norm_prev",
        y_col="rel_drdt_over_rprev",
        xlabel="Previous-day normalized sensitive count",
        ylabel="dR/dt / R_prev [1/day]",
        title="Scatter",
    )
    plot_strategy_contours(
        axes[2, 0],
        interval_df,
        x_col="sensitive_norm_prev",
        y_col="drdt",
        xlabel="Previous-day normalized sensitive count",
        ylabel="dR/dt [1/day]",
        title="Contours",
        xlim=xlim,
        ylim=drdt_ylim,
    )
    plot_strategy_contours(
        axes[2, 1],
        rel_df,
        x_col="sensitive_norm_prev",
        y_col="rel_drdt_over_rprev",
        xlabel="Previous-day normalized sensitive count",
        ylabel="dR/dt / R_prev [1/day]",
        title="Contours",
        xlim=xlim,
        ylim=rel_ylim,
    )

    axes[0, 0].set_xlim(*xlim)
    axes[1, 0].set_xlim(*xlim)
    axes[2, 0].set_xlim(*xlim)
    axes[0, 1].set_xlim(*xlim)
    axes[1, 1].set_xlim(*xlim)
    axes[2, 1].set_xlim(*xlim)
    axes[0, 0].set_ylim(*drdt_ylim)
    axes[1, 0].set_ylim(*drdt_ylim)
    axes[2, 0].set_ylim(*drdt_ylim)
    axes[0, 1].set_ylim(*rel_ylim)
    axes[1, 1].set_ylim(*rel_ylim)
    axes[2, 1].set_ylim(*rel_ylim)

    axes[0, 0].set_xlim(0.0, 3.0)
    axes[0, 1].set_xlim(0.0, 3.0)
    axes[0, 0].set_ylim(0.0, 0.01)
    axes[0, 1].set_ylim(0.0, 0.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=5, loc="upper right")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        axes[1, 0].legend(handles, labels, frameon=False, fontsize=5, loc="upper right")

    ft.savefig(fig, out_pdf)
    plt.close(fig)
    print(f"[save] {out_pdf}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = load_results()
    escape_summary = load_escape_summary()
    interval_df = build_pre_escape_interval_df(results_df, escape_summary)
    if interval_df.empty:
        raise RuntimeError("No strict pre-escape interval datapoints were built.")
    rel_df = interval_df[np.isfinite(pd.to_numeric(interval_df["rel_drdt_over_rprev"], errors="coerce"))].copy()

    make_figure(interval_df, OUT_PDF)
    make_relative_overlay_figure(rel_df, OUT_PDF_REL_OVERLAY)
    print_summary(interval_df)


if __name__ == "__main__":
    main()
