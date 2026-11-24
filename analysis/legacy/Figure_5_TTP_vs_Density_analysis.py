#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TL;DR
-----
This script builds the replicate-level density-versus-TTP table for Figure 4.
It combines sector-level local density measurements from Figure 2h with
replicate progression timing, retaining only density observations before TTP
or through the canonical experiment end for non-progressors.

Detailed
--------
Inputs:
- Ring-density summaries produced by `Figure_2h_analysis.py`.
- Replicate-level total-burden trajectories from `Results/*.txt`.
- Shared replicate exclusion and truncation rules from `shared/experiment_filters.py`.

Outputs:
- A replicate-level CSV consumed by
  `figures/Figure_5/Figure_5_TTP_vs_Density.py`.

How it works:
- Reads raw results files to compute replicate-level TTP based on normalized
  resistant burden crossing the progression threshold.
- Applies canonical strategy-specific experiment-end days:
  96 for CT/AT50/AT100 and 82 for Dens/ATAgent strategies.
- Excludes wells truncated before TTP or before their canonical experiment end.
- Restricts density rows to the observable pre-TTP window.
- Uses only total-density context from the Figure 2h table; density-change
  variants are collapsed before aggregation.
- Aggregates density in two steps:
  1. Within each replicate and timepoint, collapse all available sector-level
     rows to one timepoint summary using the median local density.
  2. Within each replicate, collapse those timepoint summaries to one
     replicate summary using the median again.

Why this multilayer aggregation is used:
- Pooling all sector rows directly gives more weight to clones and timepoints
  that simply contribute more rows.
- In particular, long-lived CT replicates accumulate many late low-density
  observations, which can drive the replicate summary downward mainly because
  they lived longer, not because their typical local density was lower at a
  comparable stage.
- The timepoint-equal scheme keeps the replicate as the experimental unit,
  uses all observed clones within that replicate, and gives each observed
  pre-TTP timepoint equal influence on the replicate summary.
- This still summarizes the full pre-TTP history, but it avoids the strongest
  row-count weighting artifact of the pooled-row approach.

Pipeline role:
- Figure 4 support analysis that bridges Figure 2h local-density measurements
  to the TTP comparison panel using replicate-level summaries.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.experiment_filters import EXCLUDE_SAMPLES


# =========================
# Configuration
# =========================
PLATES: List[str] = [
    "Plate_1",
    "Plate_2",
    "Plate_3",
    "Plate_4",
    "Plate_5",
    "Plate_6",
    "Plate_7",
    "Plate_8",
]

RESULTS_SUBDIR = "Results"

BASE_DIR = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
FIGURE3_DATA_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_3_Data")
FIG2H_CSV = FIGURE3_DATA_DIR / "Figure2h_ring_change_vs_ring_total_density_timeseries_epanechnikov_40pxboundary.csv"

OUT_DIR = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_5_Data")
OUT_CSV = OUT_DIR / "Figure5c_density_vs_TTP.csv"

RES_NORM_THRESHOLD = 1.0
MIN_CLONE_LABEL = 2

CANONICAL_END_DAY_96 = {"CT", "AT50", "AT100"}
CANONICAL_END_DAY_82 = {"Dens", "AT100+"}


def plate_path(plate: str) -> Path:
    return BASE_DIR / plate


def log_stage(message: str) -> None:
    print(f"[stage] {message}")


# =========================
# Helpers
# =========================
def infer_strategy_from_sample(sample: str) -> str:
    try:
        return sample.split("_", 1)[1].split("-", 1)[0]
    except Exception:
        return "NA"


def canonical_end_day(strategy: str) -> float:
    s = str(strategy).strip()
    if s in CANONICAL_END_DAY_96:
        return 96.0
    if s in CANONICAL_END_DAY_82 or s.upper().startswith("ATAGENT"):
        return 82.0
    print(f"[WARN] Unknown strategy for canonical end day: {s!r}")
    return np.nan


def timepoint_to_day(tp: int) -> int:
    if tp < 1:
        raise ValueError(f"timepoint_to_day expects tp >= 1, got {tp}")
    day = 1
    for k in range(2, tp + 1):
        step = 2 if ((k - 2) % 3) in (0, 1) else 3
        day += step
    return day


def pick_column(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find any {what} column in DataFrame. "
        f"Tried: {candidates}, have: {list(df.columns)}"
    )


def load_density_rows() -> pd.DataFrame:
    if not FIG2H_CSV.is_file():
        raise FileNotFoundError(f"2h CSV not found at: {FIG2H_CSV}")

    print(f"[2h] Loading ring-density timeseries from: {FIG2H_CSV}")
    df = pd.read_csv(FIG2H_CSV)
    print(f"[2h] df2h: {len(df)} rows, columns={list(df.columns)}")

    plate_col = pick_column(df, ["plate", "Plate"], "plate")
    sample_col = pick_column(df, ["sample", "Sample"], "sample")
    clone_col = pick_column(df, ["clone_label", "Clone_Label", "clone"], "clone_label")
    timepoint_col = pick_column(df, ["timepoint", "t_index", "frame_index", "frame", "t"], "timepoint")
    density_col = pick_column(
        df,
        [
            "Mean_Total_Density_cells_per_um2",
            "Mean_Total_Density",
            "mean_total_density",
            "MeanTotalDensity",
        ],
        "ring total density",
    )
    strategy_col = "strategy" if "strategy" in df.columns else None
    day_col = "day" if "day" in df.columns else None
    sector_col = "sector" if "sector" in df.columns else None

    # Figure 4 uses only total-density context. Figure 2h stores one copy of
    # the same total-density value per density-change kind, so collapse those
    # bookkeeping duplicates before any replicate-level summaries are computed.
    dedup_cols = [plate_col, sample_col, clone_col, timepoint_col, density_col]
    if sector_col is not None:
        dedup_cols.append(sector_col)
    if day_col is not None:
        dedup_cols.append(day_col)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols).copy()
    if len(df) != before_dedup:
        print(
            f"[2h] collapsed {before_dedup - len(df)} duplicate total-density "
            "row(s) across density-change variants"
        )

    out = pd.DataFrame(
        {
            "plate": df[plate_col].astype(str),
            "sample": df[sample_col].astype(str),
            "clone_label": pd.to_numeric(df[clone_col], errors="coerce"),
            "timepoint": pd.to_numeric(df[timepoint_col], errors="coerce"),
            "ring_total_density": pd.to_numeric(df[density_col], errors="coerce"),
        }
    )

    if strategy_col is not None:
        out["strategy"] = df[strategy_col].astype(str)
    else:
        out["strategy"] = out["sample"].map(infer_strategy_from_sample)

    if day_col is not None:
        out["day"] = pd.to_numeric(df[day_col], errors="coerce")
    else:
        out["day"] = np.nan

    out = out.dropna(subset=["clone_label", "timepoint", "ring_total_density"]).copy()
    out["clone_label"] = out["clone_label"].astype(int)
    out["timepoint"] = out["timepoint"].astype(int)
    out = out[out["clone_label"] >= MIN_CLONE_LABEL].copy()

    missing_day = ~np.isfinite(out["day"])
    if missing_day.any():
        out.loc[missing_day, "day"] = out.loc[missing_day, "timepoint"].map(timepoint_to_day)
    out["day"] = out["day"].astype(int)

    print(
        f"[2h] usable density rows: {len(out)} "
        f"across {out[['plate', 'sample']].drop_duplicates().shape[0]} replicate(s)"
    )
    return out


def extract_sample_from_results_filename(fname: str) -> str:
    base = fname.replace("Results_", "").replace(".txt", "")
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[-2] + "_" + parts[-1]
    return base


def load_results_table(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["DayRaw", "Green", "Red"],
    )
    if data.empty:
        return data

    data["Red"] = data["Red"].where(
        ~data["DayRaw"].astype(str).str.contains("MTDr", na=False),
        data["Red"] + data["Green"],
    )
    data["Green"] = data["Green"].where(
        ~data["DayRaw"].astype(str).str.contains("MTDr", na=False),
        0,
    )

    data["Red"] = data["Red"].where(
        ~data["DayRaw"].astype(str).str.contains("ncr", na=False),
        data["Red"] + data["Green"],
    )
    data["Green"] = data["Green"].where(
        ~data["DayRaw"].astype(str).str.contains("ncr", na=False),
        0,
    )

    day_match = data["DayRaw"].astype(str).str.extract(r"d(\d+)$")
    data["day"] = pd.to_numeric(day_match[0], errors="coerce")
    data = data.dropna(subset=["day"]).copy()
    if data.empty:
        return data

    data["day"] = data["day"].astype(int)
    data["Total"] = pd.to_numeric(data["Green"], errors="coerce") + pd.to_numeric(data["Red"], errors="coerce")
    data = data.sort_values("day").reset_index(drop=True)
    return data


def summarize_replicate_timing(
    plate: str,
    sample: str,
    strategy: str,
    data: pd.DataFrame,
) -> dict:
    row = {
        "plate": plate,
        "sample": sample,
        "strategy": strategy,
        "canonical_end_day": np.nan,
        "manual_cutoff_day": np.nan,
        "last_observed_day": np.nan,
        "usable_last_day": np.nan,
        "ttp_day": np.nan,
        "event": np.nan,
        "included": False,
        "status": "invalid",
        "baseline_total": np.nan,
    }

    cutoff = EXCLUDE_SAMPLES.get(sample, "KEEP")
    if cutoff is None:
        row["status"] = "fully_excluded"
        return row

    canonical_day = canonical_end_day(strategy)
    row["canonical_end_day"] = canonical_day
    if not np.isfinite(canonical_day):
        row["status"] = "unknown_strategy_end_day"
        return row

    if data.empty:
        row["status"] = "empty_results"
        return row

    norm_idx = 1 if ("Dens" in sample and len(data) > 1) else 0
    baseline_total = float(data["Total"].iloc[norm_idx])
    row["baseline_total"] = baseline_total
    if baseline_total <= 0 or not np.isfinite(baseline_total):
        row["status"] = "invalid_baseline"
        return row

    manual_cutoff_day = np.nan
    if cutoff != "KEEP":
        try:
            manual_cutoff_day = float(cutoff)
        except (TypeError, ValueError):
            manual_cutoff_day = np.nan
    row["manual_cutoff_day"] = manual_cutoff_day

    last_observed_day = float(pd.to_numeric(data["day"], errors="coerce").max())
    row["last_observed_day"] = last_observed_day

    usable_last_day = min(last_observed_day, canonical_day)
    if np.isfinite(manual_cutoff_day):
        usable_last_day = min(usable_last_day, manual_cutoff_day)
    row["usable_last_day"] = usable_last_day

    data = data.copy()
    data["res_norm"] = pd.to_numeric(data["Red"], errors="coerce") / baseline_total
    hits = data.loc[
        (pd.to_numeric(data["day"], errors="coerce") <= usable_last_day)
        & (pd.to_numeric(data["res_norm"], errors="coerce") >= RES_NORM_THRESHOLD),
        "day",
    ]

    if not hits.empty:
        row["ttp_day"] = float(hits.iloc[0])
        row["event"] = 1
        row["included"] = True
        row["status"] = "progressed"
        return row

    if usable_last_day >= canonical_day:
        row["ttp_day"] = float(canonical_day)
        row["event"] = 0
        row["included"] = True
        row["status"] = "end_survivor"
        return row

    row["status"] = "truncated_before_ttp_or_end"
    return row


def build_replicate_timing_table() -> pd.DataFrame:
    rows: List[dict] = []

    for plate in PLATES:
        results_dir = plate_path(plate) / RESULTS_SUBDIR
        if not results_dir.is_dir():
            print(f"[TTP] No Results dir in {results_dir}, skipping.")
            continue

        print(f"[TTP] Reading Results from: {results_dir}")
        for fname in sorted(os.listdir(results_dir)):
            if not (fname.startswith("Results_") and fname.endswith(".txt")):
                continue
            if "overexposed" in fname.lower() or "seg" in fname.lower():
                continue

            sample = extract_sample_from_results_filename(fname)
            strategy = infer_strategy_from_sample(sample)
            path = results_dir / fname

            try:
                data = load_results_table(path)
            except Exception as exc:
                print(f"[TTP] ERROR reading {path}: {exc}")
                continue

            row = summarize_replicate_timing(
                plate=plate,
                sample=sample,
                strategy=strategy,
                data=data,
            )
            rows.append(row)

    if not rows:
        raise RuntimeError("No replicate timing rows could be built from Results files.")

    timing = pd.DataFrame(rows)
    status_counts = timing["status"].value_counts(dropna=False).to_dict()
    print(f"[TTP] replicate statuses: {status_counts}")
    kept = timing[timing["included"] == True].copy()
    print(f"[TTP] retained replicates for density summary: {len(kept)} / {len(timing)}")
    return timing


def summarize_density_before_ttp(
    density_rows: pd.DataFrame,
    timing_df: pd.DataFrame,
) -> pd.DataFrame:
    keep_cols = [
        "plate",
        "sample",
        "strategy",
        "canonical_end_day",
        "manual_cutoff_day",
        "last_observed_day",
        "usable_last_day",
        "ttp_day",
        "event",
        "status",
    ]
    kept_timing = timing_df[timing_df["included"] == True][keep_cols].copy()
    if kept_timing.empty:
        raise RuntimeError("No retained replicates available after TTP filtering.")

    merged = density_rows.merge(
        kept_timing,
        on=["plate", "sample"],
        how="inner",
        suffixes=("", "_timing"),
    )
    if merged.empty:
        raise RuntimeError("No density rows matched retained replicate timing rows.")

    same_day_progressors = (
        pd.to_numeric(merged["event"], errors="coerce").fillna(0).astype(int) == 1
    )
    keep_density = np.where(
        same_day_progressors,
        pd.to_numeric(merged["day"], errors="coerce") < pd.to_numeric(merged["ttp_day"], errors="coerce"),
        pd.to_numeric(merged["day"], errors="coerce") <= pd.to_numeric(merged["ttp_day"], errors="coerce"),
    )
    merged = merged[keep_density].copy()

    if merged.empty:
        raise RuntimeError("No density rows remain after pre-TTP filtering.")

    # First collapse each replicate-timepoint to one robust density value so
    # every observed pre-TTP timepoint contributes equally downstream.
    per_timepoint = (
        merged
        .groupby(
            [
                "plate",
                "sample",
                "strategy_timing",
                "ttp_day",
                "event",
                "status",
                "canonical_end_day",
                "manual_cutoff_day",
                "last_observed_day",
                "usable_last_day",
                "day",
                "timepoint",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            timepoint_median_density=("ring_total_density", "median"),
            timepoint_mean_density=("ring_total_density", "mean"),
            n_density_rows_timepoint=("ring_total_density", "size"),
        )
    )

    raw_row_counts = (
        merged
        .groupby(
            [
                "plate",
                "sample",
                "strategy_timing",
                "ttp_day",
                "event",
                "status",
                "canonical_end_day",
                "manual_cutoff_day",
                "last_observed_day",
                "usable_last_day",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_density_rows=("ring_total_density", "size"),
        )
    )

    grouped = (
        per_timepoint
        .groupby(
            [
                "plate",
                "sample",
                "strategy_timing",
                "ttp_day",
                "event",
                "status",
                "canonical_end_day",
                "manual_cutoff_day",
                "last_observed_day",
                "usable_last_day",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            median_density=("timepoint_median_density", "median"),
            mean_density=("timepoint_median_density", "mean"),
            n_timepoints=("timepoint_median_density", "size"),
            density_first_day=("day", "min"),
            density_last_day=("day", "max"),
        )
    )

    grouped = grouped.merge(
        raw_row_counts,
        on=[
            "plate",
            "sample",
            "strategy_timing",
            "ttp_day",
            "event",
            "status",
            "canonical_end_day",
            "manual_cutoff_day",
            "last_observed_day",
            "usable_last_day",
        ],
        how="left",
    )
    grouped = grouped.rename(columns={"strategy_timing": "strategy"})
    grouped = grouped.sort_values(["strategy", "plate", "sample"]).reset_index(drop=True)

    print(
        f"[density] replicate summaries: {len(grouped)} rows "
        f"across {grouped['strategy'].nunique()} strategy/strategies"
    )
    return grouped


# =========================
# Main analysis
# =========================
def main() -> None:
    log_stage("Figure 4 analysis: replicate-level density vs TTP")

    density_rows = load_density_rows()
    timing_df = build_replicate_timing_table()
    df_out = summarize_density_before_ttp(density_rows=density_rows, timing_df=timing_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"[save] Wrote {len(df_out)} rows to: {OUT_CSV}")
    print("[done] Figure 4 density-vs-TTP analysis complete.")


if __name__ == "__main__":
    main()
