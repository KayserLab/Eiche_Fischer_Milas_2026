"""
TL;DR
-----
This script compares escape timing across strategies, using replicate-level
escape-day rules that now handle truncation and end-of-experiment cases
explicitly.

Detailed
--------
Inputs:
- Escape-summary data from the escape-front branch.
- Shared replicate exclusion/truncation rules and strategy colors.

Outputs:
- A Figure 4 escape-day comparison panel and associated pairwise statistics.

How it works:
- Loads replicate escape-day information and harmonizes strategy labels.
- Removes replicates that were truncated before escape could ever be observed,
  while treating experiment-end non-escapers according to the current rule.
- Applies pairwise survival-style comparison logic and plots the resulting
  strategy distributions.

Pipeline role:
- Figure 4 plotting script for escape timing across strategies.
"""

# Figure_4_Escape_Day.py  (UPDATED: import shared colors + apply as marker edges + xtick label color)

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: escape-front batch summary CSV and per-well observation metadata.
# Scripts: analysis/basic/escape/Escape_Front_Write_Tiffs.py (upstream),
# shared/experiment_filters.py, shared/strategy_colors.py, and shared/figtools.py.
# Outputs: escape-day comparison panel, summary CSV, and pairwise statistics.

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from shared import experiment_filters

# Stats (pairwise tests)
try:
    from scipy.stats import chi2
except ImportError as e:
    raise ImportError(
        "This script requires scipy for log-rank tests. "
        "Install with: pip install scipy"
    ) from e


# =============================================================================
# IMPORT SHARED STRATEGY COLORS
# =============================================================================
from shared import figtools as ft
from shared.strategy_colors import get_color as shared_strategy_color, get_display_name


# =============================================================================
# EDITABLE FIGURE PARAMETERS
# =============================================================================
# Width and height refer to the axes box: the physical x-axis and y-axis length.
FIGURE_AXES_WIDTH_MM = 30.0
FIGURE_AXES_HEIGHT_MM = 35.0
FIGURE_MARGINS_MM = (11.0, 1.0, 14.0, 1.0)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 6.0
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


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    return ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )

# =============================================================================
# Paths
# =============================================================================
# Loading stays EXACTLY as before
from shared.data_paths import figure_plot_dir, intermediate_dir

OUT_ROOT = intermediate_dir("experimental", "figure_2", "escape_quantification", create=False)
SUM_CSV = OUT_ROOT / "escape_front_batch_summary_all_res_within_sens.csv"
EXPERIMENT_END_DAY = 96.0

# Saving is grouped by the supplementary figure branch.
PUB_DIR = figure_plot_dir("Suppl_Figures", "escape_day")
PUB_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = PUB_DIR / "Figure_5_Escape_Day.pdf"


# =============================================================================
# Strategy parsing
# =============================================================================
WELL_STRAT_RE = re.compile(r"^([A-H]\d{1,2})_([^-_]+)-([A-H]\d{1,2})$")


def parse_strategy(replicate: str) -> str:
    m = WELL_STRAT_RE.match(str(replicate))
    if not m:
        return "UNKNOWN"
    return m.group(2)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def get_filter_rule(sample: str) -> tuple[str, float | None]:
    """
    Uses experiment_filters.EXCLUDE_SAMPLES:
      - None    -> excluded
      - number  -> truncation cutoff day
    """
    rules = getattr(experiment_filters, "EXCLUDE_SAMPLES", {})
    if sample not in rules:
        return "keep", None
    val = rules[sample]
    if val is None:
        return "exclude", None
    try:
        return "truncate", float(val)
    except Exception:
        return "keep", None


def apply_experiment_filters_escape_day(
    df: pd.DataFrame,
    sample_col: str = "replicate",
    escape_col: str = "escape_day",
) -> pd.DataFrame:
    """
    Apply shared/experiment_filters.py to escape-day data after non-escapers have
    already been resolved to a real plotting value where possible.

    Rule:
    - fully excluded samples are dropped
    - truncated samples are kept only if escape_day is finite and <= cutoff
    - samples without a truncation cutoff are kept
    """
    if sample_col not in df.columns:
        print(f"[exclude WARN] Missing '{sample_col}' column; skipping experiment_filters.")
        return df.copy()

    keep_mask = pd.Series(True, index=df.index)
    excluded_rows = 0
    truncated_rows = 0

    for sample, idx in df.groupby(sample_col).groups.items():
        status, cutoff = get_filter_rule(str(sample))
        if status == "exclude":
            keep_mask.loc[idx] = False
            excluded_rows += len(idx)
            continue

        if status != "truncate" or cutoff is None:
            continue

        esc = pd.to_numeric(df.loc[idx, escape_col], errors="coerce")
        bad = (~np.isfinite(esc)) | (esc > float(cutoff))
        drop_idx = esc.index[bad.fillna(True)]
        keep_mask.loc[drop_idx] = False
        truncated_rows += int(bad.fillna(True).sum())

    out = df[keep_mask].copy()
    print(
        f"[exclude] After experiment_filters: kept {len(out)}/{len(df)} rows "
        f"(excluded={excluded_rows}, truncated={truncated_rows})"
    )
    return out


def apply_non_escaper_frame_completeness_filter(
    df: pd.DataFrame,
    value_col: str = "escape_day",
    escape_detected_col: str = "escape_detected",
    n_total_col: str = "n_frames_total",
    n_processed_col: str = "n_frames_processed",
    non_escaper_value: float = EXPERIMENT_END_DAY,
) -> pd.DataFrame:
    """
    Keep non-escapers in downstream plotting only if n_frames_total == n_frames_processed.
    Non-escapers failing this criterion are excluded by setting value_col=NaN.
    """
    required = {value_col, escape_detected_col, n_total_col, n_processed_col}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        print(
            "[filter WARN] Skipping non-escaper frame-completeness filter; "
            f"missing columns: {missing}"
        )
        return df.copy()

    out = df.copy()
    esc = (
        out[escape_detected_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    n_total = pd.to_numeric(out[n_total_col], errors="coerce")
    n_proc = pd.to_numeric(out[n_processed_col], errors="coerce")
    complete = np.isfinite(n_total) & np.isfinite(n_proc) & (n_total == n_proc)

    non_esc = ~esc
    keep_non_esc = non_esc & complete
    drop_mask = non_esc & (~complete)
    out.loc[keep_non_esc, value_col] = float(non_escaper_value)
    out.loc[drop_mask, value_col] = np.nan
    dropped = int(drop_mask.sum())
    kept_non_esc = int(keep_non_esc.sum())
    print(
        "[filter] Non-escapers kept only if complete frames: "
        f"kept {kept_non_esc} row(s), excluded {dropped} row(s)"
    )
    return out


# =============================================================================
# Plot 3 config
# =============================================================================
DISPLAY_LABEL = {
    "CT": get_display_name("CT"),
    "AT50": get_display_name("AT50"),
    "AT100": get_display_name("AT100"),
    "ATAgent2": get_display_name("ATAgent2"),
    "Dens": get_display_name("Dens"),
}

ORDER = ["CT", "AT50", "AT100", "ATAgent2", "Dens"]

BRACKET_LW = 0.9
BRACKET_H_FRAC = 0.04
BRACKET_TOP_MARGIN_FRAC = 0.08
BRACKET_STEP_FRAC = 0.10
STARS_Y_OFFSET_H_FRAC = 0.45
STARS_FONTSIZE = 5
POINT_JITTER = 0.08
POINT_SIZE = 8
POINT_ALPHA = 0.85
MEDIAN_COLOR = "#ff7f0e"
MEDIAN_LINE_HALF_WIDTH = 0.12


# =============================================================================
# Stats helpers
# =============================================================================
def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def is_significant(p: float) -> bool:
    return bool(np.isfinite(p) and p < 5e-2)


def add_sig_bracket(ax, x1: float, x2: float, y: float, stars: str) -> None:
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if (y1 > y0) else 1.0
    h = BRACKET_H_FRAC * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=BRACKET_LW, color="black", clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h * STARS_Y_OFFSET_H_FRAC, stars, ha="center", va="bottom", fontsize=STARS_FONTSIZE, color="black")


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return pvals

    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty_like(ranked)

    running_max = 0.0
    for i, p in enumerate(ranked):
        factor = m - i
        val = factor * p
        running_max = max(running_max, val)
        adj[i] = min(1.0, running_max)

    out = np.empty_like(adj)
    out[order] = adj
    return out


def logrank_pvalue(
    durations_a: np.ndarray,
    events_a: np.ndarray,
    durations_b: np.ndarray,
    events_b: np.ndarray,
) -> float:
    """
    Two-sample log-rank test p-value (chi-square with 1 df).
    durations_*: observed/censor times
    events_*: 1=event occurred, 0=right-censored
    """
    t_a = np.asarray(durations_a, dtype=float)
    t_b = np.asarray(durations_b, dtype=float)
    e_a = np.asarray(events_a, dtype=bool)
    e_b = np.asarray(events_b, dtype=bool)

    valid_a = np.isfinite(t_a)
    valid_b = np.isfinite(t_b)
    t_a = t_a[valid_a]
    t_b = t_b[valid_b]
    e_a = e_a[valid_a]
    e_b = e_b[valid_b]

    if t_a.size < 2 or t_b.size < 2:
        return np.nan

    event_times = np.unique(np.concatenate([t_a[e_a], t_b[e_b]]))
    if event_times.size == 0:
        return np.nan

    oe_sum = 0.0
    var_sum = 0.0
    for t in event_times:
        n1 = int(np.sum(t_a >= t))
        n2 = int(np.sum(t_b >= t))
        n = n1 + n2
        if n <= 1:
            continue

        d1 = int(np.sum((t_a == t) & e_a))
        d2 = int(np.sum((t_b == t) & e_b))
        d = d1 + d2
        if d == 0:
            continue

        e1 = d * (n1 / n)
        var = (n1 * n2 * d * (n - d)) / (n * n * (n - 1))
        oe_sum += (d1 - e1)
        var_sum += var

    if var_sum <= 0:
        return np.nan

    chisq = (oe_sum * oe_sum) / var_sum
    return float(chi2.sf(chisq, df=1))


def pairwise_strategy_logrank_escape_day(
    df: pd.DataFrame,
    value_col: str = "escape_day",
    event_col: str = "escaped",
) -> pd.DataFrame:
    strategies = [s for s in ORDER if s in set(df["strategy"].unique())]
    results: list[dict] = []

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            g1 = df.loc[df["strategy"] == s1, [value_col, event_col]].copy()
            g2 = df.loc[df["strategy"] == s2, [value_col, event_col]].copy()
            g1[value_col] = pd.to_numeric(g1[value_col], errors="coerce")
            g2[value_col] = pd.to_numeric(g2[value_col], errors="coerce")
            g1 = g1[np.isfinite(g1[value_col])].copy()
            g2 = g2[np.isfinite(g2[value_col])].copy()

            a = g1[value_col].to_numpy(dtype=float)
            b = g2[value_col].to_numpy(dtype=float)
            ea = g1[event_col].to_numpy(dtype=bool) if len(g1) else np.array([], dtype=bool)
            eb = g2[event_col].to_numpy(dtype=bool) if len(g2) else np.array([], dtype=bool)

            if a.size < 2 or b.size < 2:
                p = np.nan
            else:
                p = logrank_pvalue(a, ea, b, eb)

            results.append({
                "strategy_1": s1,
                "strategy_2": s2,
                "n1": int(a.size),
                "n2": int(b.size),
                "events1": int(np.sum(ea)) if ea.size else 0,
                "events2": int(np.sum(eb)) if eb.size else 0,
                "p_raw": p,
            })

    out = pd.DataFrame(results)
    if len(out):
        p = out["p_raw"].to_numpy(dtype=float)
        adj = np.full_like(p, np.nan, dtype=float)
        finite = np.isfinite(p)
        adj[finite] = holm_adjust(p[finite])
        out["p_holm"] = adj

    print("\n=== Pairwise log-rank tests (escape_day) ===")
    if len(out):
        print(out.to_string(index=False))
    else:
        print("(No valid pairs.)")

    return out


# =============================================================================
# Main (Plot 3 ONLY)
# =============================================================================
def main() -> None:
    print("=== REAL DATA: Plot 3 ONLY ===")

    if not SUM_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUM_CSV}")

    summ = pd.read_csv(SUM_CSV)
    print(f"[INFO] Loaded summary: {len(summ):,} rows")

    if "replicate" not in summ.columns and "sample" in summ.columns:
        summ["replicate"] = summ["sample"]
    if "replicate" not in summ.columns:
        raise ValueError("Summary CSV missing required column: replicate/sample")

    if "status" in summ.columns:
        before = len(summ)
        summ = summ[summ["status"].astype(str).str.lower() == "ok"].copy()
        print(f"[filter] Kept status=ok rows: {len(summ)}/{before}")

    summ["strategy"] = summ["replicate"].astype(str).map(parse_strategy)

    if "escape_day" in summ.columns:
        summ["escape_day"] = safe_numeric(summ["escape_day"])
    elif "first_escape_day" in summ.columns:
        summ["escape_day"] = safe_numeric(summ["first_escape_day"])
    else:
        raise ValueError("Summary CSV missing required column: escape_day/first_escape_day")

    summ = apply_non_escaper_frame_completeness_filter(
        summ,
        value_col="escape_day",
        escape_detected_col="escape_detected",
        n_total_col="n_frames_total",
        n_processed_col="n_frames_processed",
        non_escaper_value=EXPERIMENT_END_DAY,
    )
    summ = apply_experiment_filters_escape_day(
        summ,
        sample_col="replicate",
        escape_col="escape_day",
    )
    if "escape_detected" in summ.columns:
        summ["escaped"] = (
            summ["escape_detected"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    else:
        summ["escaped"] = np.isfinite(summ["escape_day"])

    p3 = summ.copy()
    p3 = p3[p3["strategy"].isin(ORDER)].copy()
    p3 = p3[p3["strategy"].astype(str).str.upper() != "ATAGENT1"].copy()
    p3 = p3[np.isfinite(p3["escape_day"])].copy()

    if len(p3) == 0:
        print("[WARN] Plot3: no rows after filtering/finite checks.")
        return

    present = [s for s in ORDER if s in set(p3["strategy"].unique())]
    groups = [
        p3.loc[p3["strategy"] == s, "escape_day"].dropna().to_numpy(dtype=float)
        for s in present
    ]
    labels = [f"{DISPLAY_LABEL[s]}\n(n={len(g)})" for s, g in zip(present, groups)]
    pairwise_df = pairwise_strategy_logrank_escape_day(
        p3,
        value_col="escape_day",
        event_col="escaped",
    )

    fig, ax = new_figure()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    ax.set_xticks(np.arange(1, len(present) + 1))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Day of Escape")
    ax.set_title("")

    for tick_label, strat in zip(ax.get_xticklabels(), present):
        tick_label.set_color(shared_strategy_color(strat, "#000000"))
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)

    # Jittered scatter + per-group median line (match other escape plot style).
    rng = np.random.default_rng(7)
    for idx, vals in enumerate(groups, start=1):
        vals_arr = np.asarray(vals, dtype=float)
        if vals_arr.size == 0:
            continue
        strat = present[idx - 1]
        col = shared_strategy_color(strat, "#888888")
        x = idx + rng.uniform(-POINT_JITTER, POINT_JITTER, size=vals_arr.size)
        ax.scatter(
            x,
            vals_arr,
            s=POINT_SIZE,
            c=col,
            alpha=POINT_ALPHA,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
        med = float(np.median(vals_arr))
        ax.hlines(
            med,
            idx - MEDIAN_LINE_HALF_WIDTH,
            idx + MEDIAN_LINE_HALF_WIDTH,
            colors=MEDIAN_COLOR,
            linewidth=1.1,
            zorder=4,
        )

    # Significance indications from raw p-values, only for significant comparisons.
    significant_pairs: list[dict] = []
    if len(pairwise_df):
        pos_map = {s: i + 1 for i, s in enumerate(present)}
        for _, r in pairwise_df.iterrows():
            s1 = str(r["strategy_1"])
            s2 = str(r["strategy_2"])
            if s1 not in pos_map or s2 not in pos_map:
                continue
            p_plot = float(r.get("p_holm", np.nan))
            if not is_significant(p_plot):
                continue
            significant_pairs.append(
                {
                    "x1": float(pos_map[s1]),
                    "x2": float(pos_map[s2]),
                    "p": p_plot,
                    "stars": p_to_stars(p_plot),
                }
            )

    if significant_pairs:
        y_vals = np.concatenate([g for g in groups if len(g) > 0])
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        yr = max(1.0, y_max - y_min)
        top_margin = BRACKET_TOP_MARGIN_FRAC * yr
        step = BRACKET_STEP_FRAC * yr
        bracket_h = BRACKET_H_FRAC * yr
        needed_above = top_margin + (len(significant_pairs) - 1) * step + bracket_h * 1.8
        ax.set_ylim(y_min - 0.03 * yr, y_max + needed_above)

        for k, row in enumerate(significant_pairs):
            y_bracket = ax.get_ylim()[1] - top_margin - k * step
            add_sig_bracket(ax, row["x1"], row["x2"], y_bracket, row["stars"])

    ft.savefig(fig, OUT_PDF)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"[OK] Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
