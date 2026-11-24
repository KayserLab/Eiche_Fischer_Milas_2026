"""
TL;DR
-----
This script relates local density measurements around resistant clones to
sensitive-cell burden and phase behavior. It builds the Figure 2 density-versus-
sensitive-number views from several upstream analysis tables.

Detailed
--------
Inputs:
- Figure 2h density summaries.
- Escape-day information and sensitive-count summaries.
- Replicate exclusion/truncation rules from `shared/experiment_filters.py`.

Outputs:
- Figure panels and intermediate merged tables linking density context to
  sensitive-cell burden across behavioral phases.

How it works:
- Loads escape timing, sensitive counts, dominant-clone summaries, and local
  density measurements from upstream CSVs.
- Applies replicate-level exclusions/truncations to every merged branch.
- Builds phase-aware analysis tables and visualizes them with scatter and
  binned-median summaries.

Pipeline role:
- Figure 2 density-context plotting script built on top of Figure 2h outputs.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Figure2h ring-density CSV, escape summary, Results/*.txt, and
# plots_clones count tables.
# Scripts: analysis/figure_4/Figure_2h_analysis.py and analysis/basic/escape/Escape_Front_Write_Tiffs.py
# (upstream), shared/experiment_filters.py, and shared/figtools.py.
# Outputs: merged density-context tables and correlation panels.

import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import figtools as ft
from shared import experiment_filters


BASE_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
PLATES = [f"Plate_{i}" for i in range(1, 9)]

RESULTS_DIRNAME = "Results"
PLOTS_CLONES_DIRNAME = "plots_clones"

ESCAPE_SUMMARY_CSV = BASE_DIR / "Figures" / "Escape_Quantification" / "escape_front_batch_summary_all_res_within_sens.csv"
FIGURE3_DATA_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
DENSITY_TIMESERIES_CSV = FIGURE3_DATA_DIR / "Figure2h_ring_change_vs_ring_total_density_timeseries_epanechnikov_40pxboundary.csv"
TOTAL_DENSITY_COL = "Mean_Total_Density_cells_per_um2"
LEGACY_TOTAL_DENSITY_COL = "Mean_Total_Density"

OUT_DIR = ft.paper_figure_data_dir(__file__)
OUT_DATA_DIR = BASE_DIR / "Figures" / "Figure_3" / "Data"

OUT_MERGED_CSV = OUT_DATA_DIR / "Density_to_sennumber_correlation_datapoints.csv"
OUT_FIG_MEDIAN_SEN_PDF = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive.pdf"
OUT_FIG_MEDIAN_SEN_PNG = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive.png"
OUT_FIG_MEDIAN_SEN_BINNED_PDF = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_binned.pdf"
OUT_FIG_MEDIAN_SEN_BINNED_PNG = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_binned.png"
OUT_FIG_MEDIAN_SEN_BINNED_5_PDF = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_binned_5bins.pdf"
OUT_FIG_MEDIAN_SEN_BINNED_5_PNG = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_binned_5bins.png"
OUT_FIG_MEDIAN_SEN_COMBINED_PDF = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_combined.pdf"
OUT_FIG_MEDIAN_SEN_COMBINED_PNG = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_combined.png"
OUT_FIG_MEDIAN_SEN_COMBINED_5_PDF = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_combined_5bins.pdf"
OUT_FIG_MEDIAN_SEN_COMBINED_5_PNG = OUT_DIR / "Density_to_sennumber_correlation_median_vs_sensitive_combined_5bins.png"

EXCLUDED_STRATEGIES_FOR_PLOTS = {"ATAgent1", "Pulse"}

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis
# length. Margins are extra space for tick labels, labels, and legends.
FIGURE_AXES_WIDTH_MM = 40.0
FIGURE_AXES_HEIGHT_MM = 40.0
FIGURE_MARGINS_MM = (15.44, 2.41, 8.23, 0.99)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.5
DPI = 300
PHASE_STYLE = {
    "before": {"color": "#1f77b4", "label": "Before"},
    "after": {"color": "#d62728", "label": "After"},
}
SCATTER_MARKER_SIZE = 6
SCATTER_ALPHA = 0.20
BINNED_MARKER_SIZE = 3.2
BINNED_LINEWIDTH = 1.0
BINNED_CONTOUR_LINEWIDTH = 1.45
N_BINS = 10
CHANGE_TYPE = "rel"
DENSITY_SCALE_FACTOR = 1.0
DENSITY_DISPLAY_SCALE = 1000.0
DENSITY_AXIS_LABEL = "Boundary density [cells/1000 µm²]"
DENSITY_DISPLAY_YLIM = (0.0, 2.5)

ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
)

TOKEN_DAY_RE = re.compile(r"_d(?P<day>\d{1,3})\b", re.IGNORECASE)
FNAME_RE = re.compile(
    r"(?P<prefix>.*_)?(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def parse_strategy(sample: str) -> str:
    try:
        return sample.split("_", 1)[1].split("-", 1)[0]
    except Exception:
        return "UNKNOWN"


def timepoint_to_day(tp: int) -> int:
    if tp < 1:
        raise ValueError(f"timepoint must be >=1, got {tp}")
    day = 1
    for k in range(2, tp + 1):
        step = 2 if ((k - 2) % 3) in (0, 1) else 3
        day += step
    return day


def parse_well_strategy_from_filename(fname: str) -> tuple[str, str]:
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Could not parse well/strategy from filename: {fname}")
    well = m.group("well").upper()
    strategy = m.group("strategy")
    return well, strategy


def parse_day_from_token(token: str) -> int:
    m = TOKEN_DAY_RE.search(token)
    if not m:
        raise ValueError(f"Could not parse day token: {token}")
    return int(m.group("day"))


def sample_from_clone_counts_raw_filename(name: str) -> str | None:
    suffix = "_clone_counts_raw.csv"
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def load_escape_days() -> pd.DataFrame:
    if not ESCAPE_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing escape summary CSV: {ESCAPE_SUMMARY_CSV}")

    log(f"Loading escape summary: {ESCAPE_SUMMARY_CSV}")
    esc = pd.read_csv(ESCAPE_SUMMARY_CSV)
    required = {"plate", "sample", "first_escape_day"}
    missing = required - set(esc.columns)
    if missing:
        raise KeyError(f"Escape summary missing columns: {sorted(missing)}")

    esc["first_escape_day"] = pd.to_numeric(esc["first_escape_day"], errors="coerce")
    if "escape_detected" in esc.columns:
        esc = esc[esc["escape_detected"] == True].copy()

    esc = esc[np.isfinite(esc["first_escape_day"])].copy()
    esc = esc.rename(columns={"first_escape_day": "escape_day"})
    esc = esc[["plate", "sample", "escape_day"]].drop_duplicates(subset=["plate", "sample"], keep="first")
    log(f"Escape rows kept: {len(esc)}")
    return esc


def load_sensitive_counts() -> pd.DataFrame:
    log("Loading sensitive counts from Results/*.txt across plates")
    rows: list[dict] = []

    files: list[tuple[str, Path]] = []
    for plate in PLATES:
        results_dir = BASE_DIR / plate / RESULTS_DIRNAME
        if not results_dir.is_dir():
            log(f"[warn] missing Results dir: {results_dir}")
            continue
        for f in sorted(results_dir.glob("*.txt")):
            files.append((plate, f))

    log(f"Results files discovered: {len(files)}")
    for plate, fpath in tqdm(files, desc="Results files", unit="file"):
        try:
            well, strategy = parse_well_strategy_from_filename(fpath.name)
        except Exception:
            continue
        sample = f"{well}_{strategy}-{well}"
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    day = parse_day_from_token(parts[0])
                    sensitive = int(parts[1])
                    resistant = int(parts[2])
                except Exception:
                    continue
                rows.append(
                    {
                        "plate": plate,
                        "sample": sample,
                        "strategy": strategy,
                        "day": day,
                        "sensitive_cells": sensitive,
                        "resistant_cells": resistant,
                        "total_cells": sensitive + resistant,
                    }
                )

    if not rows:
        raise RuntimeError("No sensitive count rows parsed from Results files.")

    df = pd.DataFrame(rows)
    log(f"Sensitive rows before filtering: {len(df)}")
    df = experiment_filters.apply_sample_exclusions(df, column="sample", cutoff_column="day", strict=False)
    log(f"Sensitive rows after filtering:  {len(df)}")
    df = df.sort_values(["plate", "sample", "day"]).drop_duplicates(subset=["plate", "sample", "day"], keep="last")
    return df


def build_total_baseline_table(sen: pd.DataFrame) -> pd.DataFrame:
    sen = sen.copy()
    sen["day"] = pd.to_numeric(sen["day"], errors="coerce")
    sen["total_cells"] = pd.to_numeric(sen["total_cells"], errors="coerce")
    sen = sen.dropna(subset=["day", "total_cells"]).copy()

    base_day1 = (
        sen[sen["day"] == 1]
        .sort_values(["plate", "sample", "day"])
        .groupby(["plate", "sample"], as_index=False)["total_cells"]
        .first()
        .rename(columns={"total_cells": "baseline_total_day1"})
    )
    base_earliest = (
        sen.sort_values(["plate", "sample", "day"])
        .groupby(["plate", "sample"], as_index=False)
        .first()[["plate", "sample", "day", "total_cells"]]
        .rename(columns={"day": "baseline_day_fallback", "total_cells": "baseline_total_fallback"})
    )

    baseline = base_earliest.merge(base_day1, how="left", on=["plate", "sample"])
    baseline["baseline_total"] = baseline["baseline_total_day1"].fillna(baseline["baseline_total_fallback"])
    baseline["baseline_day"] = np.where(
        np.isfinite(pd.to_numeric(baseline["baseline_total_day1"], errors="coerce")),
        1,
        baseline["baseline_day_fallback"],
    )
    baseline = baseline[np.isfinite(pd.to_numeric(baseline["baseline_total"], errors="coerce"))].copy()
    baseline = baseline[baseline["baseline_total"] > 0].copy()

    used_day1 = int(np.isfinite(pd.to_numeric(baseline["baseline_total_day1"], errors="coerce")).sum())
    used_fallback = int(len(baseline) - used_day1)
    log(f"Baseline totals built for {len(baseline)} samples (day 1: {used_day1}, fallback earliest day: {used_fallback})")
    return baseline[["plate", "sample", "baseline_total", "baseline_day"]]


def load_dominant_clone_by_timepoint() -> pd.DataFrame:
    log("Loading dominant clone per sample/timepoint from *_clone_counts_raw.csv")
    rows: list[dict] = []
    n_files = 0

    for plate in PLATES:
        pdir = BASE_DIR / plate / PLOTS_CLONES_DIRNAME
        if not pdir.is_dir():
            continue
        files = sorted(pdir.glob("*_clone_counts_raw.csv"))
        n_files += len(files)

        for fpath in tqdm(files, desc=f"{plate} clone-count files", unit="file", leave=False):
            sample = sample_from_clone_counts_raw_filename(fpath.name)
            if sample is None:
                continue
            try:
                df = pd.read_csv(fpath, index_col=0)
            except Exception:
                continue
            clone_cols = [c for c in df.columns if str(c).startswith("clone_")]
            if not clone_cols:
                continue

            for raw_tp, series in df[clone_cols].iterrows():
                vals = pd.to_numeric(series, errors="coerce").dropna()
                if vals.empty:
                    continue
                max_val = vals.max()
                winners = sorted([c for c, v in vals.items() if v == max_val], key=lambda c: int(str(c).replace("clone_", "")))
                if not winners:
                    continue
                winner = winners[0]
                try:
                    dom_label = int(str(winner).replace("clone_", ""))
                except Exception:
                    continue
                tp = int(raw_tp) + 1  # raw index is 0-based timepoint
                day = timepoint_to_day(tp)
                rows.append(
                    {
                        "plate": plate,
                        "sample": sample,
                        "timepoint": tp,
                        "day": day,
                        "dominant_clone_label": dom_label,
                    }
                )

    log(f"Clone-count files scanned: {n_files}")
    if not rows:
        raise RuntimeError("No dominant clone rows built from clone count files.")

    dom = pd.DataFrame(rows)
    log(f"Dominant-clone rows before filtering: {len(dom)}")
    dom = experiment_filters.apply_sample_exclusions(dom, column="sample", cutoff_column="day", strict=False)
    log(f"Dominant-clone rows after filtering:  {len(dom)}")
    dom = dom.drop_duplicates(subset=["plate", "sample", "timepoint"], keep="last")
    return dom


def load_density_around_clone() -> pd.DataFrame:
    if not DENSITY_TIMESERIES_CSV.exists():
        raise FileNotFoundError(f"Missing density timeseries CSV: {DENSITY_TIMESERIES_CSV}")

    log(f"Loading density timeseries: {DENSITY_TIMESERIES_CSV}")
    d = pd.read_csv(DENSITY_TIMESERIES_CSV)
    required = {"plate", "sample", "clone_label", "timepoint"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"Density CSV missing columns: {sorted(missing)}")
    source_density_col = TOTAL_DENSITY_COL if TOTAL_DENSITY_COL in d.columns else LEGACY_TOTAL_DENSITY_COL
    if source_density_col not in d.columns:
        raise KeyError(
            f"Density CSV missing density column: expected {TOTAL_DENSITY_COL!r} or {LEGACY_TOTAL_DENSITY_COL!r}"
        )

    d["timepoint"] = pd.to_numeric(d["timepoint"], errors="coerce")
    d["clone_label"] = pd.to_numeric(d["clone_label"], errors="coerce")
    d["Mean_Total_Density"] = pd.to_numeric(d[source_density_col], errors="coerce")
    d = d.dropna(subset=["timepoint", "clone_label", "Mean_Total_Density"]).copy()
    if "change_type" in d.columns:
        d = d[d["change_type"].astype(str) == CHANGE_TYPE].copy()
    d["timepoint"] = d["timepoint"].astype(int)
    d["clone_label"] = d["clone_label"].astype(int)
    d["day"] = d["timepoint"].map(timepoint_to_day)
    if "strategy" not in d.columns:
        d["strategy"] = d["sample"].astype(str).map(parse_strategy)

    grouped = (
        d.groupby(["plate", "sample", "strategy", "clone_label", "timepoint", "day"], as_index=False)["Mean_Total_Density"]
        .agg(density_mean="mean", density_median="median")
    )
    log(f"Density rows after sector aggregation: {len(grouped)}")
    grouped = experiment_filters.apply_sample_exclusions(grouped, column="sample", cutoff_column="day", strict=False)
    log(f"Density rows after filtering:          {len(grouped)}")
    return grouped


def build_analysis_table() -> pd.DataFrame:
    esc = load_escape_days()
    sen = load_sensitive_counts()
    baseline = build_total_baseline_table(sen)
    dom = load_dominant_clone_by_timepoint()
    den = load_density_around_clone()

    log("Merging dominant-clone labels with clone-level local densities")
    merged = dom.merge(
        den,
        how="left",
        left_on=["plate", "sample", "timepoint", "day", "dominant_clone_label"],
        right_on=["plate", "sample", "timepoint", "day", "clone_label"],
    )
    merged = merged.drop(columns=["clone_label"], errors="ignore")
    log(f"Rows after dominant-density merge: {len(merged)}")

    log("Merging sensitive counts by (plate, sample, day)")
    merged = merged.merge(
        sen[["plate", "sample", "strategy", "day", "sensitive_cells", "resistant_cells", "total_cells"]],
        how="left",
        on=["plate", "sample", "day"],
        suffixes=("", "_sens"),
    )

    merged["strategy"] = merged["strategy"].fillna(merged["strategy_sens"])
    merged = merged.drop(columns=["strategy_sens"], errors="ignore")

    log("Merging escape day by (plate, sample)")
    merged = merged.merge(esc, how="left", on=["plate", "sample"])

    log("Merging replicate baseline totals from day 1")
    merged = merged.merge(baseline, how="left", on=["plate", "sample"])

    # Keep rows with all required values for scatter plotting.
    keep_cols = ["density_mean", "density_median", "sensitive_cells", "escape_day", "baseline_total"]
    merged = merged.dropna(subset=keep_cols).copy()
    merged["sensitive_cells"] = pd.to_numeric(merged["sensitive_cells"], errors="coerce")
    if "resistant_cells" not in merged.columns:
        merged["resistant_cells"] = np.nan
    merged["resistant_cells"] = pd.to_numeric(merged["resistant_cells"], errors="coerce")
    if "total_cells" not in merged.columns:
        merged["total_cells"] = merged["sensitive_cells"] + merged["resistant_cells"]
    merged["total_cells"] = pd.to_numeric(merged["total_cells"], errors="coerce")
    merged["baseline_total"] = pd.to_numeric(merged["baseline_total"], errors="coerce")
    merged = merged[np.isfinite(merged["sensitive_cells"])].copy()
    merged = merged[np.isfinite(merged["total_cells"])].copy()
    merged = merged[np.isfinite(merged["baseline_total"]) & (merged["baseline_total"] > 0)].copy()
    merged["sensitive_cells_norm"] = merged["sensitive_cells"] / merged["baseline_total"]

    merged["phase"] = np.where(merged["day"] < merged["escape_day"], "before", "after")
    merged = experiment_filters.apply_sample_exclusions(merged, column="sample", cutoff_column="day", strict=False)

    merged = merged.sort_values(["plate", "sample", "day", "timepoint"]).reset_index(drop=True)
    log(f"Final analysis datapoints: {len(merged)}")
    return merged


def padded_limits(values: np.ndarray, *, lower_zero: bool = False) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if v_max > v_min:
        pad = 0.05 * (v_max - v_min)
    else:
        pad = 0.5
    lo = v_min - pad
    hi = v_max + pad
    if lower_zero:
        lo = max(0.0, lo)
    return lo, hi


def prepare_density_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("density_mean", "density_median"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col] / DENSITY_SCALE_FACTOR
        out[col] = out[col] * DENSITY_DISPLAY_SCALE
    return out


def scatter_overlay_by_phase(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    *,
    fixed_ylim: tuple[float, float] | None = None,
) -> None:
    df = df[~df["strategy"].astype(str).isin(EXCLUDED_STRATEGIES_FOR_PLOTS)].copy()
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    for phase in ["before", "after"]:
        sub = df[df["phase"] == phase].copy()
        if sub.empty:
            continue
        style = PHASE_STYLE[phase]
        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            s=SCATTER_MARKER_SIZE,
            alpha=SCATTER_ALPHA,
            color=style["color"],
            edgecolors="none",
            label=style["label"],
        )

    x_limits = padded_limits(pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float), lower_zero=True)
    y_limits = padded_limits(pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float), lower_zero=True)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=5.5, loc="best")

    ft.savefig(fig, out_pdf)
    ft.savefig(fig, out_png)
    plt.close(fig)
    log(f"Saved plot: {out_pdf}")
    log(f"Saved plot: {out_png}")


def binned_median_lineplot_by_phase(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    *,
    fixed_ylim: tuple[float, float] | None = None,
    n_bins: int = N_BINS,
) -> None:
    df = df[~df["strategy"].astype(str).isin(EXCLUDED_STRATEGIES_FOR_PLOTS)].copy()
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    x_vals = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    x_vals = x_vals[np.isfinite(x_vals)]
    if x_vals.size == 0:
        raise RuntimeError("No finite x values available for binned median plot.")

    quantile_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    bin_edges = np.quantile(x_vals, quantile_edges)
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        x0 = float(x_vals[0])
        bin_edges = np.array([x0 - 0.5, x0 + 0.5], dtype=float)

    for phase in ["before", "after"]:
        sub = df[df["phase"] == phase].copy()
        if sub.empty:
            continue
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna(subset=[x_col, y_col]).copy()
        if sub.empty:
            continue
        sub["x_bin"] = pd.cut(sub[x_col], bins=bin_edges, include_lowest=True, duplicates="drop")
        grouped = (
            sub.groupby("x_bin", observed=False)
            .agg(
                x_median=(x_col, "median"),
                y_median=(y_col, "median"),
                n=(y_col, "size"),
            )
            .reset_index(drop=True)
        )
        grouped = grouped.dropna(subset=["x_median", "y_median"]).copy()
        if grouped.empty:
            continue
        style = PHASE_STYLE[phase]
        line, = ax.plot(
            grouped["x_median"].to_numpy(dtype=float),
            grouped["y_median"].to_numpy(dtype=float),
            marker="o",
            markersize=BINNED_MARKER_SIZE,
            linewidth=BINNED_LINEWIDTH,
            color=style["color"],
            alpha=0.95,
            label=style["label"],
        )
        line.set_path_effects(
            [
                pe.Stroke(linewidth=BINNED_CONTOUR_LINEWIDTH, foreground="black"),
                pe.Normal(),
            ]
        )

    x_limits = padded_limits(pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float), lower_zero=True)
    y_limits = padded_limits(pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float), lower_zero=True)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=5.5, loc="best")

    ft.savefig(fig, out_pdf)
    ft.savefig(fig, out_png)
    plt.close(fig)
    log(f"Saved plot: {out_pdf}")
    log(f"Saved plot: {out_png}")


def scatter_and_binned_lineplot_by_phase(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    *,
    fixed_ylim: tuple[float, float] | None = None,
    n_bins: int = N_BINS,
) -> None:
    df = df[~df["strategy"].astype(str).isin(EXCLUDED_STRATEGIES_FOR_PLOTS)].copy()
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    x_vals = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    x_vals = x_vals[np.isfinite(x_vals)]
    if x_vals.size == 0:
        raise RuntimeError("No finite x values available for combined scatter+median plot.")

    quantile_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    bin_edges = np.quantile(x_vals, quantile_edges)
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        x0 = float(x_vals[0])
        bin_edges = np.array([x0 - 0.5, x0 + 0.5], dtype=float)

    for phase in ["before", "after"]:
        sub = df[df["phase"] == phase].copy()
        if sub.empty:
            continue
        style = PHASE_STYLE[phase]
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna(subset=[x_col, y_col]).copy()
        if sub.empty:
            continue

        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            s=SCATTER_MARKER_SIZE,
            alpha=SCATTER_ALPHA,
            color=style["color"],
            edgecolors="none",
        )

        sub["x_bin"] = pd.cut(sub[x_col], bins=bin_edges, include_lowest=True, duplicates="drop")
        grouped = (
            sub.groupby("x_bin", observed=False)
            .agg(
                x_median=(x_col, "median"),
                y_median=(y_col, "median"),
            )
            .reset_index(drop=True)
        )
        grouped = grouped.dropna(subset=["x_median", "y_median"]).copy()
        if grouped.empty:
            continue
        line, = ax.plot(
            grouped["x_median"].to_numpy(dtype=float),
            grouped["y_median"].to_numpy(dtype=float),
            marker="o",
            markersize=BINNED_MARKER_SIZE,
            linewidth=BINNED_LINEWIDTH,
            color=style["color"],
            alpha=0.98,
            label=style["label"],
        )
        line.set_path_effects(
            [
                pe.Stroke(linewidth=BINNED_CONTOUR_LINEWIDTH, foreground="black"),
                pe.Normal(),
            ]
        )

    x_limits = padded_limits(pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float), lower_zero=True)
    y_limits = padded_limits(pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float), lower_zero=True)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=5.5, loc="best")

    ft.savefig(fig, out_pdf)
    ft.savefig(fig, out_png)
    plt.close(fig)
    log(f"Saved plot: {out_pdf}")
    log(f"Saved plot: {out_png}")


def main() -> None:
    log("=== Density_to_sennumber_correlation: start ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = build_analysis_table()
    if df.empty:
        log("[warn] No datapoints available after merge/filtering. Nothing to plot.")
        return

    df.to_csv(OUT_MERGED_CSV, index=False)
    log(f"Saved merged datapoints CSV: {OUT_MERGED_CSV}")
    plot_df = prepare_density_display_df(df)

    log("Counts by phase and strategy:")
    summary = df.groupby(["phase", "strategy"]).size().rename("n").reset_index()
    for _, r in summary.iterrows():
        log(f"  phase={r['phase']}, strategy={r['strategy']}, n={int(r['n'])}")

    scatter_overlay_by_phase(
        df=plot_df,
        x_col="sensitive_cells_norm",
        x_label="Normalized sensitive count",
        y_col="density_median",
        y_label=DENSITY_AXIS_LABEL,
        out_pdf=OUT_FIG_MEDIAN_SEN_PDF,
        out_png=OUT_FIG_MEDIAN_SEN_PNG,
        fixed_ylim=DENSITY_DISPLAY_YLIM,
    )
    binned_median_lineplot_by_phase(
        df=plot_df,
        x_col="sensitive_cells_norm",
        x_label="Normalized sensitive count",
        y_col="density_median",
        y_label=DENSITY_AXIS_LABEL,
        out_pdf=OUT_FIG_MEDIAN_SEN_BINNED_PDF,
        out_png=OUT_FIG_MEDIAN_SEN_BINNED_PNG,
        fixed_ylim=DENSITY_DISPLAY_YLIM,
    )
    binned_median_lineplot_by_phase(
        df=plot_df,
        x_col="sensitive_cells_norm",
        x_label="Normalized sensitive count",
        y_col="density_median",
        y_label=DENSITY_AXIS_LABEL,
        out_pdf=OUT_FIG_MEDIAN_SEN_BINNED_5_PDF,
        out_png=OUT_FIG_MEDIAN_SEN_BINNED_5_PNG,
        fixed_ylim=DENSITY_DISPLAY_YLIM,
        n_bins=5,
    )
    scatter_and_binned_lineplot_by_phase(
        df=plot_df,
        x_col="sensitive_cells_norm",
        x_label="Normalized sensitive count",
        y_col="density_median",
        y_label=DENSITY_AXIS_LABEL,
        out_pdf=OUT_FIG_MEDIAN_SEN_COMBINED_PDF,
        out_png=OUT_FIG_MEDIAN_SEN_COMBINED_PNG,
        fixed_ylim=DENSITY_DISPLAY_YLIM,
    )
    scatter_and_binned_lineplot_by_phase(
        df=plot_df,
        x_col="sensitive_cells_norm",
        x_label="Normalized sensitive count",
        y_col="density_median",
        y_label=DENSITY_AXIS_LABEL,
        out_pdf=OUT_FIG_MEDIAN_SEN_COMBINED_5_PDF,
        out_png=OUT_FIG_MEDIAN_SEN_COMBINED_5_PNG,
        fixed_ylim=DENSITY_DISPLAY_YLIM,
        n_bins=5,
    )

    log("=== Density_to_sennumber_correlation: done ===")


if __name__ == "__main__":
    main()
