"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
TL;DR
-----
This script combines the density-experiment and CT density-correlation
branches into one comparative Figure 3 plot.

Detailed
--------
Inputs:
- Density-experiment analysis CSVs from
  `analysis/legacy/Density_Experiment_Density_Correlation_analysis.py`.
- CT analysis CSVs from
  `analysis/legacy/Density_Experiments_Density_Correlation_analysis_CT.py`.
- Shared replicate exclusion/truncation rules.

Outputs:
- A combined comparison plot summarizing density-versus-growth trajectories
  across both branches.

How it works:
- Loads the time-resolved and per-clone summaries for both the density
  experiment and CT analyses.
- Reconstructs comparable trajectory sets and bins them along the density axis.
- Produces one shared summary visualization so the two analysis branches can be
  compared on the same plotting logic.

Pipeline role:
- Figure 3 combined plotting endpoint for the density experiment versus CT
  comparison.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared import experiment_filters


EXP_BASE_DIR = Path(r"C:\Users\labadmin-kayser\Experimental_data\2026_01_Density_Experiment")
CT_BASE_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = ft.paper_figure_data_dir(__file__)
OUT_BASENAME = OUT_DIR / "Density_Experiment_mediangrowth_over_density_both"

EXP_IN_TIME_CSV = EXP_BASE_DIR / r"Figures\Figure_2\Data\Density_Experiment_ring_total_density_vs_abs_clone_growth.csv"
CT_IN_TIME_CSV = CT_BASE_DIR / r"Figures\Figure_2\Data\CT_Density_Correlation_ring_total_density_vs_abs_clone_growth.csv"

DENSITY_DISPLAY_SCALE = 1000.0
DENSITY_AXIS_LABEL = "Boundary Density [cells (1000 µm$^2$)$^{-1}$]"
BINS = np.arange(0.0, 5.6, 0.5)
XLIM = (0.0, 5.5)
XTICKS = np.arange(0.0, 5.1, 1.0)
YLIM = (0.0, 1.25)
MIN_ARROW_SEGMENT_CM = 0.5
DPI = 300

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 14.0
FIGURE_AXES_HEIGHT_MM = 38.0
FIGURE_PANEL_GAP_MM = 2.5
FIGURE_MARGINS_MM = (12.0, 1.0, 11.0, 8.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.0

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

EXP_COLOR = "#222222"
EXP_FILL = "#8A8A8A"
CT_COLOR = "#1f77b4"
CT_FILL = "#8FBCE6"

CT_RESULTS_RE = re.compile(
    r"^Results_G418_a2a4_(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$",
    re.IGNORECASE,
)
CT_TOKEN_DAY_RE = re.compile(r"_d(?P<day>\d{1,3})\b", re.IGNORECASE)
CT_IGNORE_FILENAME_PARTS = ("overexposed", "seg")


def _density_x_col_name(df: pd.DataFrame, ring_px: int = 40, summary: str = "median") -> str:
    preferred = f"ring{int(ring_px)}_total_density_{summary}_12sectors"
    if preferred in df.columns:
        return preferred
    fallback = (
        "ring_total_density_mean_12sectors"
        if summary == "mean"
        else "ring_total_density_median_12sectors"
    )
    if fallback in df.columns:
        return fallback
    raise ValueError(f"Missing density x column: {preferred}")


def _scale_density_values(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.to_numeric(values, errors="coerce") * DENSITY_DISPLAY_SCALE


def _format_tick_label(value: float) -> str:
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    abs_value = abs(float(value))
    if abs_value < 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if abs_value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _forward_rate(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.asarray(t[1:] - t[:-1], dtype=float)
    dy = np.asarray(y[1:] - y[:-1], dtype=float)
    out = np.full(dt.shape, np.nan, dtype=float)
    mask = np.isfinite(dt) & np.isfinite(dy) & (dt > 0)
    out[mask] = dy[mask] / dt[mask]
    return out


def _load_density_experiment_time_data() -> pd.DataFrame:
    if not EXP_IN_TIME_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {EXP_IN_TIME_CSV}")

    df = pd.read_csv(EXP_IN_TIME_CSV)
    required = {"plate", "sample", "clone_label", "timepoint"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Density experiment CSV missing columns: {missing}")

    xcol = _density_x_col_name(df, ring_px=40, summary="median")
    for col in ["timepoint", "clone_label", xcol]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["plate", "sample", "timepoint", "clone_label", xcol]).copy()
    df["_x_density"] = _scale_density_values(df[xcol])
    return df


def _extract_condition_from_filename(fname: str) -> float | None:
    match = re.search(r"_(\d+\.\d+)_", fname)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _extract_sample_from_filename(fname: str) -> str | None:
    stem = Path(fname).stem
    match = re.search(r"([A-Z]\d+_\d+\.\d+_\d+-[A-Z]\d+)$", stem)
    return match.group(1) if match else None


def _extract_day(day_token: str) -> int:
    text = str(day_token)
    match = re.search(r"(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return int(text[-2:])


def _read_results_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["DayToken", "Sensitive", "Resistant"])
    if df.empty:
        return df
    df["Day"] = df["DayToken"].apply(_extract_day).astype(int)
    df["Sensitive"] = pd.to_numeric(df["Sensitive"], errors="coerce")
    df["Resistant"] = pd.to_numeric(df["Resistant"], errors="coerce")
    df = df.dropna(subset=["Day", "Sensitive", "Resistant"]).copy()
    return df.sort_values("Day").reset_index(drop=True)[["Day", "Sensitive", "Resistant"]]


def _iter_density_experiment_results_files():
    plates = sorted(
        [path.name for path in EXP_BASE_DIR.iterdir() if path.is_dir() and path.name.lower().startswith("plate")]
    )
    for plate in plates:
        results_dir = EXP_BASE_DIR / plate / "Results"
        if not results_dir.is_dir():
            continue
        for filename in os.listdir(results_dir):
            if filename.endswith(".txt") and "overexposed" not in filename.lower():
                yield plate, results_dir / filename


def _load_density_experiment_replicate_timeseries() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for plate, path in _iter_density_experiment_results_files():
        condition = _extract_condition_from_filename(path.name)
        sample = _extract_sample_from_filename(path.name)
        if condition is None or sample is None:
            continue

        rule = experiment_filters.EXCLUDE_SAMPLES.get(sample, "KEEP")
        if rule is None:
            continue

        df = _read_results_file(path)
        if df.empty:
            continue
        df["plate"] = plate
        df["sample"] = sample
        if rule != "KEEP":
            df = df[df["Day"] <= float(rule)].copy()
        if len(df) >= 3:
            rows.append(df[["plate", "sample", "Day", "Resistant"]].copy())

    if not rows:
        return pd.DataFrame(columns=["plate", "sample", "Day", "Resistant"])
    return pd.concat(rows, ignore_index=True).sort_values(["plate", "sample", "Day"]).reset_index(drop=True)


def _build_density_experiment_raw_all_trajectories() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    local_density_df = _load_density_experiment_time_data()
    rep_df = _load_density_experiment_replicate_timeseries()
    if local_density_df.empty or rep_df.empty:
        return []

    density_map: dict[tuple[str, str, int], np.ndarray] = {}
    for (plate, sample, timepoint), group in local_density_df.groupby(["plate", "sample", "timepoint"], sort=False):
        values = pd.to_numeric(group["_x_density"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            density_map[(str(plate), str(sample), int(timepoint))] = values

    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rep_ids = rep_df[["plate", "sample"]].drop_duplicates().sort_values(["plate", "sample"])
    for _, row in rep_ids.iterrows():
        plate = str(row["plate"])
        sample = str(row["sample"])
        sub = rep_df[(rep_df["plate"] == plate) & (rep_df["sample"] == sample)].sort_values("Day").copy()
        if len(sub) < 3:
            continue

        t = sub["Day"].to_numpy(dtype=float)
        resistant = sub["Resistant"].to_numpy(dtype=float)
        growth_abs = _forward_rate(resistant, t)
        growth_rel = np.full_like(growth_abs, np.nan, dtype=float)
        mask = np.isfinite(growth_abs) & np.isfinite(resistant[:-1]) & (resistant[:-1] > 0)
        growth_rel[mask] = growth_abs[mask] / resistant[:-1][mask]

        x_lookup = t[:-1].astype(int)
        x_rep: list[float] = []
        y_rep: list[float] = []
        tp_rep: list[float] = []
        for i, tp in enumerate(x_lookup):
            xvals = density_map.get((plate, sample, int(tp)))
            if xvals is None or xvals.size == 0:
                continue
            xv = float(np.nanmedian(xvals))
            yv = float(growth_rel[i]) if np.isfinite(growth_rel[i]) else np.nan
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            x_rep.append(xv)
            y_rep.append(yv)
            tp_rep.append(float(tp))
        if x_rep:
            trajectories.append(
                (
                    np.asarray(x_rep, dtype=float),
                    np.asarray(y_rep, dtype=float),
                    np.asarray(tp_rep, dtype=float),
                )
            )
    return trajectories


def _ct_x_col_name(df: pd.DataFrame, ring_px: int = 40, summary: str = "median") -> str:
    preferred = f"ring{int(ring_px)}_total_density_{summary}_12sectors"
    if preferred not in df.columns:
        raise ValueError(f"Missing CT density x column: {preferred}")
    return preferred


def _load_ct_time_data() -> pd.DataFrame:
    if not CT_IN_TIME_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {CT_IN_TIME_CSV}")

    df = pd.read_csv(CT_IN_TIME_CSV)
    required = {"plate", "sample", "clone_label", "timepoint"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CT CSV missing columns: {missing}")

    xcol = _ct_x_col_name(df, ring_px=40, summary="median")
    for col in ["timepoint", "clone_label", xcol]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["plate", "sample", "timepoint", "clone_label", xcol]).copy()
    df["_x_density"] = _scale_density_values(df[xcol])
    return df


def _sample_strategy(sample: str) -> str:
    try:
        return str(sample).split("_", 1)[1].split("-", 1)[0]
    except Exception:
        return ""


def _timepoint_to_day(tp: int) -> int:
    if tp < 1:
        raise ValueError(f"_timepoint_to_day expects tp >= 1, got {tp}")
    day = 1
    for k in range(2, tp + 1):
        step = 2 if ((k - 2) % 3) in (0, 1) else 3
        day += step
    return day


def _parse_ct_day(token: str) -> int:
    match = CT_TOKEN_DAY_RE.search(token)
    if not match:
        raise ValueError(f"Cannot parse day from token: {token}")
    return int(match.group("day"))


def _load_ct_replicate_timeseries() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    plates = sorted(
        [path.name for path in CT_BASE_DIR.iterdir() if path.is_dir() and path.name.lower().startswith("plate")]
    )
    for plate in plates:
        plots_dir = CT_BASE_DIR / plate / "plots_clones"
        if not plots_dir.is_dir():
            continue
        for csv_path in sorted(plots_dir.glob("*_clone_counts_raw.csv")):
            sample = csv_path.stem.replace("_clone_counts_raw", "")
            if _sample_strategy(sample) != "CT":
                continue
            rule = experiment_filters.EXCLUDE_SAMPLES.get(sample, "KEEP")
            if rule is None:
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            clone_cols = [col for col in df.columns if str(col).startswith("clone_")]
            if not clone_cols:
                continue
            df[clone_cols] = df[clone_cols].apply(pd.to_numeric, errors="coerce")
            resistant = df[clone_cols].sum(axis=1, min_count=1)
            n_timepoints = len(df.index)
            if n_timepoints == 0:
                continue
            timepoints = np.arange(1, n_timepoints + 1, dtype=int)
            days = np.asarray([_timepoint_to_day(int(tp)) for tp in timepoints], dtype=float)
            out = pd.DataFrame(
                {
                    "plate": plate,
                    "sample": sample,
                    "Timepoint": timepoints,
                    "Day": days,
                    "Resistant": resistant.to_numpy(dtype=float),
                }
            )
            if rule != "KEEP":
                out = out[out["Day"] <= float(rule)].copy()
            if len(out) >= 3:
                rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["plate", "sample", "Timepoint", "Day", "Resistant"])
    return pd.concat(rows, ignore_index=True).sort_values(["plate", "sample", "Timepoint"]).reset_index(drop=True)


def _build_ct_all_clones_trajectories() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    local_density_df = _load_ct_time_data()
    rep_df = _load_ct_replicate_timeseries()
    if local_density_df.empty or rep_df.empty:
        return []

    density_map: dict[tuple[str, str, int], np.ndarray] = {}
    for (plate, sample, timepoint), group in local_density_df.groupby(["plate", "sample", "timepoint"], sort=False):
        values = pd.to_numeric(group["_x_density"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            density_map[(str(plate), str(sample), int(timepoint))] = values

    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rep_ids = rep_df[["plate", "sample"]].drop_duplicates().sort_values(["plate", "sample"])
    for _, row in rep_ids.iterrows():
        plate = str(row["plate"])
        sample = str(row["sample"])
        sub = rep_df[(rep_df["plate"] == plate) & (rep_df["sample"] == sample)].sort_values("Day").copy()
        if len(sub) < 3:
            continue

        t = sub["Day"].to_numpy(dtype=float)
        resistant = sub["Resistant"].to_numpy(dtype=float)
        growth_rel = np.full(len(t) - 1, np.nan, dtype=float)
        growth_abs = _forward_rate(resistant, t)
        mask = np.isfinite(growth_abs) & np.isfinite(resistant[:-1]) & (resistant[:-1] > 0)
        growth_rel[mask] = growth_abs[mask] / resistant[:-1][mask]

        x_left_timepoint = sub["Timepoint"].to_numpy(dtype=int)[:-1]
        x_rep: list[float] = []
        y_rep: list[float] = []
        tp_rep: list[float] = []
        for i, tp in enumerate(x_left_timepoint):
            xvals = density_map.get((plate, sample, int(tp)))
            if xvals is None or xvals.size == 0:
                continue
            xv = float(np.nanmedian(xvals))
            yv = float(growth_rel[i]) if np.isfinite(growth_rel[i]) else np.nan
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            x_rep.append(xv)
            y_rep.append(yv)
            tp_rep.append(float(tp))
        if x_rep:
            trajectories.append(
                (
                    np.asarray(x_rep, dtype=float),
                    np.asarray(y_rep, dtype=float),
                    np.asarray(tp_rep, dtype=float),
                )
            )
    return trajectories


def _summarize_binned(
    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    bins: np.ndarray,
) -> pd.DataFrame:
    rows: list[tuple[float, float]] = []
    for x, y, _tp in trajectories:
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        rows.extend((float(xv), float(yv)) for xv, yv in zip(x[mask], y[mask]))

    if not rows:
        return pd.DataFrame(columns=["x_center", "y_center", "y_q25", "y_q75"])

    df = pd.DataFrame(rows, columns=["x", "y"])
    df["x_bin"] = pd.cut(df["x"], bins=np.asarray(bins, dtype=float), include_lowest=True, right=True)
    df = df.dropna(subset=["x_bin"])
    if df.empty:
        return pd.DataFrame(columns=["x_center", "y_center", "y_q25", "y_q75"])

    agg = (
        df.groupby("x_bin", observed=True, as_index=False)
        .agg(
            y_center=("y", "median"),
            y_q25=("y", lambda values: np.nanquantile(values, 0.25)),
            y_q75=("y", lambda values: np.nanquantile(values, 0.75)),
        )
    )
    agg["x_center"] = agg["x_bin"].apply(lambda interval: float(interval.mid))
    return agg.sort_values("x_center").reset_index(drop=True)


def _plot_summary(
    ax: plt.Axes,
    summary: pd.DataFrame,
    line_color: str,
    fill_color: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if summary.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = pd.to_numeric(summary["x_center"], errors="coerce").to_numpy(dtype=float)
    y_center = pd.to_numeric(summary["y_center"], errors="coerce").to_numpy(dtype=float)
    y_q25 = pd.to_numeric(summary["y_q25"], errors="coerce").to_numpy(dtype=float)
    y_q75 = pd.to_numeric(summary["y_q75"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y_center) & np.isfinite(y_q25) & np.isfinite(y_q75)
    if not np.any(mask):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    ax.fill_between(x[mask], y_q25[mask], y_q75[mask], color=fill_color, alpha=0.35, zorder=1)
    ax.plot(x[mask], y_center[mask], "o-", color=line_color, linewidth=1.2, markersize=3.2, label=label, zorder=2)
    return x[mask], y_center[mask]


def _draw_mid_segment_chevrons(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    line_color: str,
    direction: str,
) -> None:
    if x.size < 2:
        return
    if direction not in {"ltr", "rtl"}:
        raise ValueError("direction must be 'ltr' or 'rtl'")

    trans = ax.transData
    inv = ax.transData.inverted()
    head_len_px = 9.0
    head_half_width_px = 4.5
    tip_fraction = 0.62
    min_segment_len_px = float(MIN_ARROW_SEGMENT_CM / 2.54) * float(ax.figure.dpi)

    for i in range(x.size - 1):
        p0 = trans.transform((float(x[i]), float(y[i])))
        p1 = trans.transform((float(x[i + 1]), float(y[i + 1])))
        start, end = (p0, p1) if direction == "ltr" else (p1, p0)

        vec = end - start
        norm = float(np.hypot(vec[0], vec[1]))
        if not np.isfinite(norm) or norm <= 0 or norm < min_segment_len_px:
            continue
        unit = vec / norm
        normal = np.asarray([-unit[1], unit[0]], dtype=float)
        tip = start + tip_fraction * vec
        base_center = tip - head_len_px * unit
        left = base_center + head_half_width_px * normal
        right = base_center - head_half_width_px * normal

        tip_d = inv.transform(tip)
        left_d = inv.transform(left)
        right_d = inv.transform(right)
        ax.plot([left_d[0], tip_d[0]], [left_d[1], tip_d[1]], color=line_color, linewidth=1.0, zorder=3)
        ax.plot([right_d[0], tip_d[0]], [right_d[1], tip_d[1]], color=line_color, linewidth=1.0, zorder=3)


def main() -> None:
    exp_traj = _build_density_experiment_raw_all_trajectories()
    ct_traj = _build_ct_all_clones_trajectories()
    if not exp_traj:
        raise RuntimeError("No density-experiment trajectories available for combined plot.")
    if not ct_traj:
        raise RuntimeError("No CT trajectories available for combined plot.")

    exp_summary = _summarize_binned(exp_traj, BINS)
    ct_summary = _summarize_binned(ct_traj, BINS)
    if exp_summary.empty or ct_summary.empty:
        raise RuntimeError("Binned summaries are empty for combined plot.")

    fig, axes = ft.new_subplots(
        nrows=1,
        ncols=2,
        ax_width_mm=FIGURE_AXES_WIDTH_MM,
        ax_height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        wspace_mm=FIGURE_PANEL_GAP_MM,
        sharex=True,
        sharey=True,
        squeeze=False,
        dpi=DPI,
        apply_style=False,
    )
    ax_nt, ax_ct = axes[0]

    exp_x, exp_y = _plot_summary(ax_nt, exp_summary, line_color=EXP_COLOR, fill_color=EXP_FILL, label="NT")
    ct_x, ct_y = _plot_summary(ax_ct, ct_summary, line_color=CT_COLOR, fill_color=CT_FILL, label="CT")

    for ax, title in [(ax_nt, "NT\n(sparsely seeded)"), (ax_ct, "CT\n(colonies)")]:
        ax.set_title(title, pad=2.0)
        ax.set_xlim(float(XLIM[0]), float(XLIM[1]))
        ax.set_ylim(float(YLIM[0]), float(YLIM[1]))
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([_format_tick_label(float(x)) for x in XTICKS])
        ax.grid(True, linewidth=0.5, alpha=0.25)

    ax_nt.set_ylabel("Resistant growth rate")
    ax_ct.tick_params(axis="y", labelleft=False)
    fig.text(0.977, 0.09, DENSITY_AXIS_LABEL, ha="right", va="center", fontsize=TEXT_SIZE_PT)
    fig.canvas.draw()

    _draw_mid_segment_chevrons(ax_nt, exp_x, exp_y, line_color=EXP_COLOR, direction="ltr")
    _draw_mid_segment_chevrons(ax_ct, ct_x, ct_y, line_color=CT_COLOR, direction="rtl")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, OUT_BASENAME.with_suffix(".pdf"))
    ft.savefig(fig, OUT_BASENAME.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
