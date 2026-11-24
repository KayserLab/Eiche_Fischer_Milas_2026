"""
TL;DR
-----
This script builds the analysis table for growth as a function of escape grade.
It uses the newer escape-area branch and writes the CSVs consumed by the
current Figure 2 growth-over-escape plotting script.

Detailed
--------
Inputs:
- `Escape_Fig_Build_escape_areas_timeseries_new.csv`

Outputs:
- `Escape_Fig_Growth_over_Grade_of_Escape_data_new.csv`
- QC CSVs for negative-growth rows and fraction-greater-than-one rows.

How it works:
- Reads the newer escape-area timeseries as the canonical upstream source.
- Derives per-interval growth quantities together with escape-grade measures.
- Writes one main analysis table plus QC tables that preserve excluded or
  suspicious rows for later inspection.

Pipeline role:
- Analysis-side producer for the Figure 2 growth-versus-escape-grade branch.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = DATA_ROOT / r"Figures\Escape_Quantification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical upstream input: newer boundary-aware builder output.
BUILD_TS = OUT_DIR / "Escape_Fig_Build_escape_areas_timeseries_new.csv"

# Keep the filename expected by the current Figure 2 plotting branch.
OUT_CSV = OUT_DIR / "Escape_Fig_Growth_over_Grade_of_Escape_data_new.csv"
OUT_NEG_CSV = OUT_DIR / "Escape_Fig_Growth_over_Grade_of_Escape_negative_growth_rows_new.csv"
OUT_FRAC_GT1_CSV = OUT_DIR / "Escape_Fig_Growth_over_Grade_of_Escape_fraction_gt1_rows_new.csv"

SAMPLE_RE = re.compile(r"^[A-H]\d+_([^-]+)-[A-H]\d+$", re.IGNORECASE)


def parse_strategy(sample: str) -> str:
    m = SAMPLE_RE.match(str(sample))
    return m.group(1) if m else "UNKNOWN"


def main() -> None:
    if not BUILD_TS.exists():
        raise FileNotFoundError(
            f"Missing builder output: {BUILD_TS}\nRun Escape_Fig_Build_escape_areas_new.py first."
        )

    df = pd.read_csv(BUILD_TS)
    if df.empty:
        raise RuntimeError("Builder timeseries CSV is empty.")

    if "strategy" not in df.columns and "sample" in df.columns:
        df["strategy"] = df["sample"].map(parse_strategy)

    num_cols = ["delta_res_count_per_day", "escape_front_length_px", "escape_front_fraction", "boundary_length_px"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["delta_res_count_per_day", "escape_front_length_px", "escape_front_fraction"]).copy()
    print(f"[info] rows used in growth-over-grade analysis: {len(df)}")

    df.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV} rows={len(df)}")

    neg_df = df[df["delta_res_count_per_day"] < 0].copy().sort_values(
        ["delta_res_count_per_day", "plate", "sample", "t"]
    )
    neg_df.to_csv(OUT_NEG_CSV, index=False)
    print(f"[save] {OUT_NEG_CSV} rows={len(neg_df)}")

    frac_gt1_df = df[df["escape_front_fraction"] > 1.0].copy().sort_values(
        ["escape_front_fraction", "plate", "sample", "t"],
        ascending=[False, True, True, True],
    )
    frac_gt1_df.to_csv(OUT_FRAC_GT1_CSV, index=False)
    print(f"[save] {OUT_FRAC_GT1_CSV} rows={len(frac_gt1_df)}")


if __name__ == "__main__":
    main()
