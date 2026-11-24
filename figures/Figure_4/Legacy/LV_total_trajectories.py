"""
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - \\]+?
# - D:\2025_08_Co
# Upstream/related script reference(s):
# - import matplotlib.py
# =============================================================================
TL;DR
-----
This script plots normalized total-burden trajectories for the LV-agent branch.

Detailed
--------
Inputs:
- Raw `Results/*.txt` files for the configured plates.
- Shared replicate exclusion/truncation rules and strategy colors.

Outputs:
- A Figure 4 trajectory plot for LV-agent replicates and related printed
  summaries.

How it works:
- Parses replicate result trajectories and maps samples to strategy labels.
- Normalizes total burden to an early baseline day.
- Selects the LV-agent branch and visualizes its replicate trajectories over
  time, while also printing selected absolute/relative summaries.

Pipeline role:
- Figure 4 support plot focused on LV-agent trajectory behavior.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# --- project helpers ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import figtools as ft
from shared.experiment_filters import apply_sample_exclusions
from shared.strategy_colors import get_color


def strategy_color(strategy: str, default: str = "#000000") -> str:
    return get_color(strategy, default)


# =============================================================================
# EDITABLE FIGURE PARAMETERS
# =============================================================================
# Width and height refer to the axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 20.0
FIGURE_AXES_HEIGHT_MM = 20.0
FIGURE_MARGINS_MM = (8.0, 4.0, 8.0, 4.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.0
DPI = 300
SHOW_FIGURE = True

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

# =============================================================================
# User config
# =============================================================================
DATA_ROOT = r"D:\2025_08_CoreSeed_Large_Testdataset"
PLATES = ["Plate_1", "Plate_2", "Plate_3", "Plate_4", "Plate_5"]
STRATEGIES = {"NT", "CT", "Pulse", "AT100"}

OUT_DIR = ft.paper_figure_data_dir(__file__)
OUT_PDF = OUT_DIR / "LV_total_trajectories.pdf"
OUT_JPG = OUT_DIR / "LV_total_trajectories.jpg"

REPL_ALPHA = 0.35
REPL_LW = 1.0

# Console debug
PRINT_NT_VALUES = True
EXCLUDE_FILENAME_PATTERNS = ("overexposed", "seg")

# Only these plate/well pairs are included in the plot (from fitting set).
FIT_PLATE_WELLS = {
    (1, "B4"), (1, "B5"), (1, "C4"), (1, "C6"), (1, "D4"),
    (2, "A1"), (2, "A6"), (2, "C1"), (2, "C2"), (2, "D2"), (2, "D5"),
    (3, "B2"), (3, "B3"), (3, "D2"), (3, "D4"),
    (4, "A4"), (4, "A5"), (4, "B1"), (4, "B6"), (4, "C2"), (4, "C5"),
    (5, "A2"), (5, "A5"), (5, "B3"), (5, "B5"),
    (5, "C1"), (5, "C3"), (5, "C4"), (5, "C5"), (5, "C6"),
    (5, "D2"), (5, "D5"),
}


# =============================================================================
# Regex / parsing
# =============================================================================

# Example line token (from Results/*.txt):
# G418_a2a4_A1_NT-A1_d01 21806 0
TOKEN_DAY_RE = re.compile(r"_d(?P<day>\d{1,3})\b", re.IGNORECASE)

# Example filename:
# Results_G418_a2a4_A1_NT-A1.txt
# or: G418_a2a4_A1_NT-A1.txt
FNAME_RE = re.compile(
    r"(?P<prefix>.*_)?(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$",
    re.IGNORECASE
)


@dataclass
class Row:
    plate: str
    well: str
    sample: str      # replicate id (includes strategy + well)
    strategy: str
    day: int
    sen: int
    res: int
    total: int


def plate_path(plate: str) -> str:
    return os.path.join(DATA_ROOT, plate)


def plate_to_int(plate: str) -> int:
    m = re.search(r"(\d+)$", str(plate))
    if not m:
        raise ValueError(f"Cannot parse plate number from plate name: {plate}")
    return int(m.group(1))


def parse_day_from_token(token: str) -> int:
    m = TOKEN_DAY_RE.search(token)
    if not m:
        raise ValueError(f"Cannot parse day from token: {token}")
    return int(m.group("day"))


def parse_well_strategy_from_filename(fname: str) -> Tuple[str, str]:
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    well = m.group("well").upper()
    strategy = m.group("strategy")
    return well, strategy


def load_results_rows_from_plate(plate: str) -> List[Row]:
    """
    Load all replicate files from:
      <DATA_ROOT>/<plate>/Results/*.txt
    """
    results_dir = os.path.join(plate_path(plate), "Results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Missing Results dir: {results_dir}")

    rows: List[Row] = []

    txt_files = [
        f for f in sorted(os.listdir(results_dir))
        if f.lower().endswith(".txt")
        and not any(p in f.lower() for p in EXCLUDE_FILENAME_PATTERNS)
    ]
    print(f"[{plate}] Results files: {len(txt_files)}")

    for fname in txt_files:
        fpath = os.path.join(results_dir, fname)

        try:
            well, strategy = parse_well_strategy_from_filename(fname)
        except Exception as e:
            print(f"[SKIP] {plate} file '{fname}' (cannot parse well/strategy): {e}")
            continue

        # Keep sample naming consistent with your previous scripts
        sample = f"{well}_{strategy}-{well}"

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                token, sen_s, res_s = parts[0], parts[1], parts[2]
                try:
                    day = parse_day_from_token(token)
                    sen, res = int(sen_s), int(res_s)
                    total = sen + res
                    rows.append(Row(plate, well, sample, strategy, day, sen, res, total))
                except Exception as e:
                    print(f"[PARSE FAIL] {plate} {fname}:{line_no} token='{token}' ({e})")
                    continue

    return rows


def build_fit_wells_by_strategy(df: pd.DataFrame) -> Dict[str, List[Tuple[int, str]]]:
    out: Dict[str, List[Tuple[int, str]]] = {}
    if df.empty:
        return out
    for strategy, g in df.groupby("strategy", sort=True):
        vals = sorted(
            {(int(p), str(w)) for p, w in zip(g["plate_num"], g["well"])},
            key=lambda x: (x[0], x[1]),
        )
        out[str(strategy)] = vals
    return out


# =============================================================================
# Normalization (robust)
# =============================================================================
def normalize_total_prefer_day0_else_earliest(df: pd.DataFrame) -> pd.DataFrame:
    """
    base_total = total(day==0) if available, else total(earliest available day) per replicate.
    y(t) = total(t) / base_total
    """
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
    out["y"] = out["total"] / out["base_total"]

    used_day0 = int(out["base0"].notna().groupby([out["plate"], out["sample"]]).any().sum())
    used_earliest = int(out["base0"].isna().groupby([out["plate"], out["sample"]]).all().sum())
    print("\n=== NORMALIZATION BASELINE SUMMARY ===")
    print(f"Replicates using day==0 baseline: {used_day0}")
    print(f"Replicates using earliest-day fallback: {used_earliest}")

    return out


def print_nt_absolute_and_relative(df_norm: pd.DataFrame) -> None:
    nt = df_norm[df_norm["strategy"] == "NT"].copy()
    if nt.empty:
        print("\n=== NT VALUES (absolute + relative) ===")
        print("No NT data after filtering/exclusions/normalization.")
        return

    nt = nt.sort_values(["plate", "sample", "day"])
    print("\n=== NT VALUES (absolute + relative) ===")
    print("plate\tsample\tday\ttotal_abs\tbase_total\ty_rel")
    for (plate, sample), g in nt.groupby(["plate", "sample"], sort=False):
        for _, r in g.iterrows():
            print(
                f"{plate}\t{sample}\t{int(r['day'])}\t"
                f"{int(r['total'])}\t{float(r['base_total']):.6g}\t{float(r['y']):.6g}"
            )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("\n=== LOADING FROM Results/ (NOT Results_today) ===")

    all_rows: List[Row] = []
    for plate in PLATES:
        all_rows.extend(load_results_rows_from_plate(plate))

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    if df.empty:
        raise RuntimeError("No rows loaded from Results/. Check folder names and filename patterns.")

    print("\n=== STRATEGY COUNTS (raw loaded) ===")
    print(df["strategy"].value_counts())

    # Restrict to wells used for fitting.
    df["plate_num"] = df["plate"].map(plate_to_int)
    df["well"] = df["well"].astype(str).str.upper()
    df = df[df.apply(lambda r: (int(r["plate_num"]), str(r["well"])) in FIT_PLATE_WELLS, axis=1)].copy()
    print("\n=== ROWS AFTER FIT_PLATE_WELLS FILTER ===")
    print(f"Rows kept: {len(df)}")

    # Filter to strategies we care about
    df = df[df["strategy"].isin(STRATEGIES)].copy()
    print("\n=== STRATEGY COUNTS (after STRATEGIES filter) ===")
    print(df["strategy"].value_counts())

    # Apply exclusions/truncations
    df = apply_sample_exclusions(df, column="sample", cutoff_column="day", strict=False)
    print("\n=== STRATEGY COUNTS (after exclusions) ===")
    print(df["strategy"].value_counts())

    fit_wells_by_strategy = build_fit_wells_by_strategy(df)
    print("\n=== FIT_WELLS_BY_STRATEGY (used in plot) ===")
    for strategy in sorted(fit_wells_by_strategy):
        print(f"{strategy}: {fit_wells_by_strategy[strategy]}")

    # Normalize
    df = normalize_total_prefer_day0_else_earliest(df)

    # NT debug print
    if PRINT_NT_VALUES:
        print_nt_absolute_and_relative(df)

    # Determine which strategies are actually plotted (keep your preferred order)
    plotted_strategies = list(df["strategy"].drop_duplicates())
    preferred = ["NT", "CT", "Pulse", "AT100"]
    plotted_strategies = [s for s in preferred if s in plotted_strategies] + [
        s for s in plotted_strategies if s not in preferred
    ]

    # Use shared color map (fallback to black if missing)
    COLOR_MAP = {s: strategy_color(s, default="#000000") for s in plotted_strategies}

    print("\n=== COLOR MAP USED (shared strategy colors) ===")
    for s in plotted_strategies:
        print(f"{s}: {COLOR_MAP[s]}")

    # Plot replicates
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

    df = df.sort_values(["strategy", "plate", "sample", "day"])
    for (plate, sample, strategy), g in df.groupby(["plate", "sample", "strategy"], sort=False):
        ax.plot(
            g["day"].to_numpy(),
            g["y"].to_numpy(),
            color=COLOR_MAP[strategy],
            alpha=REPL_ALPHA,
            linewidth=REPL_LW,
        )

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Relative Cell Number")
    ax.set_ylim(0, 2.5)
    ax.set_xlim(1, 19)

    # Legend (top right) - one size smaller
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color=COLOR_MAP[s], lw=1.5) for s in plotted_strategies]
    ax.legend(proxies, plotted_strategies, frameon=False, loc="upper right", fontsize=5.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    ft.savefig(fig, OUT_PDF)
    ft.savefig(fig, OUT_JPG)
    print(f"\n[OK] Saved: {OUT_PDF}")
    print(f"[OK] Saved: {OUT_JPG}")

    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
