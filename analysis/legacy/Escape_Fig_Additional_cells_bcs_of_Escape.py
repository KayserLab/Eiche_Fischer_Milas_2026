"""
TL;DR
-----
This script quantifies how many additional resistant cells are associated with
escape. It reads the newer escape-area timeseries and writes the canonical
analysis tables used by the corresponding Figure 2 replot script.

Detailed
--------
Inputs:
- `Escape_Fig_Build_escape_areas_timeseries_new.csv`

Outputs:
- Figure-ready CSV summaries for extra resistant-cell burden attributable to
  escape.
- Statistical summary values describing the distribution by strategy.

How it works:
- Reads the newer escape-area branch rather than the legacy builder outputs.
- Aggregates replicate-level escape-associated resistant-cell counts.
- Computes compact descriptive statistics and strategy-wise comparison values.
- Writes the tables consumed by the replotting script.

Pipeline role:
- Analysis-side producer for the “additional cells because of escape” Figure 2
  panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


DATA_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
OUT_DIR = DATA_ROOT / r"Figures\Escape_Quantification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical upstream inputs: use the newer boundary-aware builder outputs.
BUILD_TS = OUT_DIR / "Escape_Fig_Build_escape_areas_timeseries_new.csv"
BUILD_SUM = OUT_DIR / "Escape_Fig_Build_escape_areas_summary_new.csv"

# Keep historical filenames so downstream consumers do not need to change.
OUT_TS = OUT_DIR / "Escape_Fig_Additional_cells_bcs_of_Escape_timeseries.csv"
OUT_SUM = OUT_DIR / "Escape_Fig_Additional_cells_bcs_of_Escape_summary.csv"
OUT_TESTS = OUT_DIR / "Escape_Fig_Additional_cells_bcs_of_Escape_pairwise_tests.csv"

PLOT_STRATEGIES = ["CT", "AT50", "AT100"]


def describe_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else np.nan,
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        "min": float(np.min(arr)) if len(arr) else np.nan,
        "max": float(np.max(arr)) if len(arr) else np.nan,
    }


def main() -> None:
    if not BUILD_TS.exists() or not BUILD_SUM.exists():
        raise FileNotFoundError(
            "Missing builder outputs.\n"
            f"Expected:\n- {BUILD_TS}\n- {BUILD_SUM}\n"
            "Run Escape_Fig_Build_escape_areas_new.py first."
        )

    ts = pd.read_csv(BUILD_TS)
    sm = pd.read_csv(BUILD_SUM)
    if ts.empty or sm.empty:
        raise RuntimeError("Builder outputs are empty.")

    ts.to_csv(OUT_TS, index=False)
    sm.to_csv(OUT_SUM, index=False)
    print(f"[save] {OUT_TS} rows={len(ts)}")
    print(f"[save] {OUT_SUM} rows={len(sm)}")

    plot_df = sm[sm["strategy"].isin(PLOT_STRATEGIES)].copy()
    plot_df["escape_growth_fraction_total"] = pd.to_numeric(
        plot_df["escape_growth_fraction_total"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["escape_growth_fraction_total"])
    print(f"[info] rows in summary: {len(sm)}")
    print(f"[info] rows used for stats: {len(plot_df)}")

    if plot_df.empty:
        print("[warn] no rows for selected strategies; skipping pairwise tests.")
        return

    order = [s for s in PLOT_STRATEGIES if s in set(plot_df["strategy"])]
    vals = [
        plot_df.loc[plot_df["strategy"] == s, "escape_growth_fraction_total"].to_numpy(dtype=float)
        for s in order
    ]

    print("[info] group summaries for escape_growth_fraction_total")
    for s, arr in zip(order, vals):
        stats = describe_values(arr)
        print(
            f"[group] {s} | n={int(stats['n'])} | "
            f"mean={stats['mean']:.6f} | median={stats['median']:.6f} | "
            f"std={stats['std']:.6f} | min={stats['min']:.6f} | max={stats['max']:.6f}"
        )

    test_rows: list[dict[str, float | int | str]] = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a_name = order[i]
            b_name = order[j]
            a = vals[i]
            b = vals[j]
            a_stats = describe_values(a)
            b_stats = describe_values(b)

            if len(a) == 0 or len(b) == 0:
                u_stat = np.nan
                p_value = np.nan
            else:
                res = mannwhitneyu(a, b, alternative="two-sided")
                u_stat = float(res.statistic)
                p_value = float(res.pvalue)

            rank_biserial = (
                float((2.0 * u_stat) / (len(a) * len(b)) - 1.0)
                if len(a) > 0 and len(b) > 0 and np.isfinite(u_stat)
                else np.nan
            )

            test_rows.append(
                {
                    "group_a": a_name,
                    "group_b": b_name,
                    "test": "Mann-Whitney U",
                    "alternative": "two-sided",
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "mean_a": a_stats["mean"],
                    "mean_b": b_stats["mean"],
                    "median_a": a_stats["median"],
                    "median_b": b_stats["median"],
                    "std_a": a_stats["std"],
                    "std_b": b_stats["std"],
                    "min_a": a_stats["min"],
                    "min_b": b_stats["min"],
                    "max_a": a_stats["max"],
                    "max_b": b_stats["max"],
                    "u_statistic": u_stat,
                    "p_value_two_sided": p_value,
                    "rank_biserial_from_u": rank_biserial,
                    "values_a": ";".join(f"{float(x):.6f}" for x in a),
                    "values_b": ";".join(f"{float(x):.6f}" for x in b),
                }
            )

            print(
                f"[test] {a_name} vs {b_name} | "
                f"n=({len(a)},{len(b)}) | U={u_stat:.6f} | p={p_value:.6g}"
            )

    pd.DataFrame(test_rows).to_csv(OUT_TESTS, index=False)
    print(f"[save] {OUT_TESTS}")


if __name__ == "__main__":
    main()
