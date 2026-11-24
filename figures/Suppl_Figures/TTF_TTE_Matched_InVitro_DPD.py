#!/usr/bin/env python3
"""Compare in-vitro and DPD event times for exact initial-condition matches.

A pair requires equality of the experimental sample ID and the simulation's
recorded initial_condition, AND equality of treatment strategy. No nearest-well
or cross-strategy substitutions are allowed. Each simulation run is one point.

TTF: first acquisition day with resistant cells >= 50% of initial total.
TTE: first acquisition day with signed_escape_margin > 0.
Both use the acquisition-day calendar (baseline = day 1). Only pairs with the
event observed in both sources are plotted or used in agreement statistics.
Unreached events remain in the audit table as right-censored records.
Manual experimental exclusions/cutoffs are respected.

Inputs are read-only. Tables and PDFs default to figure_data and figure_plots,
respectively, under Suppl_Figures/ttf_tte_matched_analysis on the share.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import figtools as ft
from analysis.figure_2.Event_Time_Analysis import (
    ESCAPE_MARGIN_CSV, EXPERIMENT_ROOT, SIMULATION_ROOT,
    EXPERIMENT_H5_SUFFIX, SIMULATION_H5_SUFFIX,
    _read_population_file, load_escape_margins,
)
from shared.data_paths import COMBINED_RESULTS_CSV, figure_plot_dir, figure_data_dir
from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import get_color, normalize_strategy

STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")
LABELS = {"CT": "CT", "AT50": "AT50", "AT100": "AT100",
          "LV-Agent": "ATagent", "AT100+": r"AT$^{+}$"}
EVENTS = ("TTF", "TTE")


def sample_strategy(sample: str) -> str:
    return normalize_strategy(sample.split("_", 1)[1].split("-", 1)[0])


def discover_pairs(metadata: pd.DataFrame, experiment_root: Path,
                   simulation_root: Path) -> pd.DataFrame:
    """Join exact seed IDs and strategies, retaining rejected runs in the audit."""
    metadata = metadata[["protocol", "run", "initial_condition"]].drop_duplicates().copy()
    metadata["run"] = metadata["run"].astype(str).str.replace(r"\.0$", "", regex=True)
    metadata["strategy"] = metadata["protocol"].map(normalize_strategy)
    if metadata.duplicated(["strategy", "run"]).any():
        raise ValueError("Conflicting initial-condition metadata for a simulation run")
    lookup = metadata.set_index(["strategy", "run"])
    experiments: dict[tuple[str, str], Path] = {}
    for path in sorted(experiment_root.glob(f"Plate_*/hdf5/*{EXPERIMENT_H5_SUFFIX}")):
        sample = path.name.removeprefix("G418_a2a4_").removesuffix(EXPERIMENT_H5_SUFFIX)
        key = (sample_strategy(sample), sample)
        if key in experiments:
            raise ValueError(f"Ambiguous experimental sample ID: {sample}")
        experiments[key] = path
    rows = []
    for path in sorted(simulation_root.glob(f"Plate_*/hdf5/*{SIMULATION_H5_SUFFIX}")):
        strategy = normalize_strategy(path.parent.parent.name.removeprefix("Plate_"))
        if strategy not in STRATEGIES:
            continue
        sample = path.name.removesuffix(SIMULATION_H5_SUFFIX)
        match = re.match(r"^R(\d+)_", sample)
        if not match or int(match[1]) < 10:
            continue
        run = str(int(match[1]))
        seed = ""
        exp_path = None
        reason = ""
        if (strategy, run) not in lookup.index:
            reason = "missing_initial_condition_metadata"
        else:
            seed = str(lookup.loc[(strategy, run), "initial_condition"])
            exp_path = experiments.get((strategy, seed))
            if exp_path is None:
                reason = "no_exact_experimental_seed_with_same_strategy"
            elif seed in EXCLUDE_SAMPLES and EXCLUDE_SAMPLES[seed] is None:
                reason = "experimental_sample_excluded"
        rows.append({
            "strategy": strategy, "run": run, "initial_condition": seed,
            "exp_plate": exp_path.parent.parent.name if exp_path else "",
            "exp_sample": seed, "sim_plate": path.parent.parent.name,
            "sim_sample": sample, "exp_input": str(exp_path) if exp_path else "",
            "sim_input": str(path), "match_status": reason or "exact_match",
        })
    if not rows:
        raise RuntimeError("No simulation inputs were found")
    return pd.DataFrame(rows)


def failure_record(path: Path, kind: str, plate: str, sample: str,
                   strategy: str, horizon: float) -> dict:
    if kind == "experimental" and sample in EXCLUDE_SAMPLES and EXCLUDE_SAMPLES[sample] is None:
        return {"reason": "experimental_sample_excluded"}
    counts = _read_population_file(path, dataset_type=kind, plate=plate,
                                   sample=sample, strategy=strategy)
    if counts.empty:
        return {"reason": "no_population_observations"}
    counts = counts.sort_values("day")
    if float(counts.iloc[0]["day"]) != 1:
        return {"reason": "missing_day_1_baseline"}
    baseline = float(counts.iloc[0]["total"])
    if not np.isfinite(baseline) or baseline <= 0:
        return {"reason": "invalid_initial_total"}
    limit = horizon
    cutoff = EXCLUDE_SAMPLES.get(sample)
    if kind == "experimental" and cutoff is not None:
        limit = min(limit, float(cutoff))
    counts = counts[counts["day"] <= limit]
    if counts.empty:
        return {"reason": "no_population_observations_within_cutoff"}
    hits = counts.loc[counts["res"] / baseline >= 0.5, "day"]
    observed = not hits.empty
    time = float(hits.min() if observed else counts["day"].max())
    if kind == "simulation":
        actual = set(counts.loc[counts["day"] <= time, "day"].astype(int))
        if set(range(1, int(time) + 1)) - actual:
            return {"reason": "missing_daily_population_observations"}
    return {"time": time, "observed": observed, "reason": "",
            "initial_total": baseline, "last_day": float(counts["day"].max())}


def escape_record(margins: pd.DataFrame, kind: str, plate: str,
                  sample: str, strategy: str, horizon: float) -> dict:
    if kind == "experimental" and sample in EXCLUDE_SAMPLES and EXCLUDE_SAMPLES[sample] is None:
        return {"reason": "experimental_sample_excluded"}
    group = margins.loc[
        margins["dataset_type"].eq(kind) & margins["plate"].eq(plate)
        & margins["sample"].eq(sample) & margins["strategy"].eq(strategy)
    ].sort_values("day")
    limit = horizon
    cutoff = EXCLUDE_SAMPLES.get(sample)
    if kind == "experimental" and cutoff is not None:
        limit = min(limit, float(cutoff))
    group = group[group["day"] <= limit]
    if group.empty:
        return {"reason": "no_escape_margin_observations_within_cutoff"}
    hits = group.loc[group["signed_escape_margin"] > 0, "day"]
    observed = not hits.empty
    return {"time": float(hits.min() if observed else group["day"].max()),
            "observed": observed, "reason": "",
            "last_day": float(group["day"].max())}


def event_pairs(matches: pd.DataFrame, margins: pd.DataFrame,
                horizon: float) -> pd.DataFrame:
    rows = []
    for pair in matches.loc[matches["match_status"].eq("exact_match")].to_dict("records"):
        for event in EVENTS:
            row = {**pair, "event": event}
            reasons = []
            for prefix, kind in (("exp", "experimental"), ("sim", "simulation")):
                args = (kind, pair[f"{prefix}_plate"], pair[f"{prefix}_sample"],
                        pair["strategy"], horizon)
                record = (failure_record(Path(pair[f"{prefix}_input"]), *args)
                          if event == "TTF" else escape_record(margins, *args))
                row.update({f"{prefix}_{key}": value for key, value in record.items()})
                if record["reason"]:
                    reasons.append(f"{prefix}: {record['reason']}")
            if not reasons and not (row.get("exp_observed") and row.get("sim_observed")):
                reasons.append("event_unobserved_in_one_or_both_sources")
            row["plot_status"] = "; ".join(reasons) or "included"
            rows.append(row)
    if not rows:
        raise RuntimeError("No exact same-strategy pairs survived the exclusion rules")
    return pd.DataFrame(rows)


def pearson_statistics(x, y) -> dict[str, float]:
    """Pearson r and its unadjusted, two-sided normal-null p-value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"pearson_r": np.nan, "pearson_p_two_sided": np.nan}
    result = pearsonr(x, y)
    return {"pearson_r": float(result.statistic),
            "pearson_p_two_sided": float(result.pvalue)}


def pearson_correlation(x, y) -> float:
    """Report r only for at least three finite, nonconstant matched pairs."""
    return pearson_statistics(x, y)["pearson_r"]


def format_pearson_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def agreement_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in EVENTS:
        for strategy in (*STRATEGIES, "All points"):
            group = pairs.loc[pairs["event"].eq(event)]
            if strategy != "All points":
                group = group.loc[group["strategy"].eq(strategy)]
            included = group.loc[group["plot_status"].isin(
                ("included", "event_unobserved_in_one_or_both_sources"))]
            complete = included.loc[included["exp_observed"].eq(True)
                                    & included["sim_observed"].eq(True)]
            delta = complete["sim_time"] - complete["exp_time"]
            rows.append({
                "event": event, "strategy": strategy, "n_exact_pairs": len(group),
                "n_plotted": len(complete), "n_both_events_observed": len(complete),
                "n_pairs_with_censoring": len(included) - len(complete),
                "mean_sim_minus_exp_days": delta.mean(),
                "median_sim_minus_exp_days": delta.median(),
                "mae_days": delta.abs().mean(),
                "rmse_days": np.sqrt((delta ** 2).mean()),
                **pearson_statistics(complete["exp_time"], complete["sim_time"]),
            })
    return pd.DataFrame(rows)


def plot_pairs(pairs: pd.DataFrame, output: Path) -> None:
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.6))
    valid = pairs.loc[pairs["plot_status"].eq("included")
                      & pairs["exp_observed"].eq(True)
                      & pairs["sim_observed"].eq(True)]
    if valid.empty:
        raise RuntimeError("No pairs have the event observed in both sources")
    max_time = float(valid[["exp_time", "sim_time"]].max().max())
    upper = max(20.0, np.ceil(max_time * 1.12 / 20.0) * 20.0)
    for ax, event, letter in zip(axes, EVENTS, ("A", "B")):
        data = valid.loc[valid["event"].eq(event)]
        ax.plot([0, upper], [0, upper], "--", color="0.55", lw=0.8, zorder=1)
        for row in data.to_dict("records"):
            color = get_color(row["strategy"])
            x, y = row["exp_time"], row["sim_time"]
            ax.scatter(x, y, s=21, facecolors=color, edgecolors="black",
                       linewidths=0.55, alpha=0.9, zorder=3)
        ax.set(xlim=(0, upper), ylim=(0, upper),
               xlabel=f"{event} in vitro [days]", ylabel=f"{event} in silico [days]")
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{letter}   {event}", loc="left", fontweight="bold", fontsize=8)
        pooled = pearson_statistics(data["exp_time"], data["sim_time"])
        pooled_r = f"{pooled['pearson_r']:.2f}" if np.isfinite(pooled["pearson_r"]) else "NA"
        ax.text(0.04, 0.96,
                f"All points: r={pooled_r}, p={format_pearson_p(pooled['pearson_p_two_sided'])}, n={len(data)}",
                transform=ax.transAxes, va="top", fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
        ax.text(0, -0.32, "Pearson within strategy (observed pairs):", transform=ax.transAxes,
                va="top", fontsize=6, clip_on=False)
        present_strategies = [s for s in STRATEGIES if data["strategy"].eq(s).any()]
        for index, strategy in enumerate(present_strategies):
            group = data.loc[data["strategy"].eq(strategy)]
            stat = pearson_statistics(group["exp_time"], group["sim_time"])
            r = stat["pearson_r"]
            r_label = f"{r:.2f}" if np.isfinite(r) else "NA"
            ax.text(0, -0.40 - index * 0.073,
                    f"{LABELS[strategy]}: r={r_label}, p={format_pearson_p(stat['pearson_p_two_sided'])}, n={len(group)}",
                    transform=ax.transAxes, va="top", fontsize=6,
                    color=get_color(strategy), clip_on=False)
    present = [s for s in STRATEGIES if valid["strategy"].eq(s).any()]
    handles = [Line2D([], [], marker="o", linestyle="none", color=get_color(s),
                      markeredgecolor="black", markeredgewidth=0.5,
                      markersize=4, label=LABELS[s]) for s in present]
    handles.append(Line2D([], [], linestyle="--", color="0.55", lw=0.8, label="Agreement"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=len(handles), frameon=False, columnspacing=1.4)
    fig.text(0.5, 0.017, "All points pools treatment strategies; p-values are unadjusted, two-sided Pearson tests.",
             ha="center", fontsize=6)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.43, top=0.94, wspace=0.35)
    ft.savefig(fig, output.with_suffix(".pdf"), transparent=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=COMBINED_RESULTS_CSV)
    parser.add_argument("--escape-margin-csv", type=Path, default=ESCAPE_MARGIN_CSV)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--simulation-root", type=Path, default=SIMULATION_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=figure_data_dir("Suppl_Figures", "ttf_tte_matched_analysis",
                                                "ttf_tte_matched", create=False))
    parser.add_argument("--output-plot-dir", type=Path,
                        default=figure_plot_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False),
                        help="PDF destination; --output-dir stores analysis tables.")
    parser.add_argument("--horizon-days", type=float, default=300.0)
    args = parser.parse_args()
    if not np.isfinite(args.horizon_days) or args.horizon_days <= 0:
        raise ValueError("--horizon-days must be finite and positive")
    metadata = pd.read_csv(args.metadata_csv, usecols=["protocol", "run", "initial_condition"],
                           dtype=str)
    matches = discover_pairs(metadata, args.experiment_root, args.simulation_root)
    margins = load_escape_margins(args.escape_margin_csv)
    pairs = event_pairs(matches, margins, args.horizon_days)
    summary = agreement_summary(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output_dir / "matching_audit.csv", index=False)
    pairs.to_csv(args.output_dir / "matched_event_times.csv", index=False)
    summary.to_csv(args.output_dir / "agreement_summary.csv", index=False)
    plot_pairs(pairs, args.output_plot_dir / "TTF_TTE_Matched_InVitro_DPD.pdf")
    print(matches.groupby(["strategy", "match_status"]).size().to_string())
    print(summary.to_string(index=False))
    print(f"Saved PDF to {args.output_plot_dir}")
    print(f"Saved figure and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
