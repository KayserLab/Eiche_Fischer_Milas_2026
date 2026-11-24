#!/usr/bin/env python3
"""Assess initial-condition imbalance between retained experimental cohorts.

Group seeds by the treatment assigned to their original experimental well,
then compare the groups under a COMMON simulated strategy. CT is the primary
reference; other simulated strategies provide a sensitivity analysis.
Observed events and valid right-censoring bounds are analysed separately from
invalid records. Outputs follow the shared supplementary plot/data layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, CensoredData, ecdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir
from figures.Suppl_Figures.TTF_TTE_Matched_InVitro_DPD import EVENTS, LABELS
from figures.Suppl_Figures.plot_TTF_TTE_Matched_Simulation_Strategies import STRATEGIES
from figures.Suppl_Figures.plot_TTF_TTE_Matched_Simulation_Strategies import benjamini_hochberg
from shared.strategy_colors import get_color

GROUPS = ("CT", "AT50", "AT100", "AT100+")
BASE_DATA = figure_data_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False)
PLOT_DIR = figure_plot_dir("Suppl_Figures", "ttf_tte_matched_analysis", create=False)


def join_experimental_cohorts(pairs, simulations):
    columns = ["initial_condition", "strategy", "exp_plate", "exp_sample"]
    if not pairs["match_status"].eq("exact_match").all():
        raise ValueError("Experimental cohort membership must use exact matches")
    if not pairs["initial_condition"].eq(pairs["exp_sample"]).all():
        raise ValueError("Seed ID does not match original experimental sample")
    cohorts = pairs[columns].drop_duplicates().rename(columns={"strategy": "experimental_strategy"})
    if cohorts["initial_condition"].duplicated().any():
        raise ValueError("A seed belongs to multiple retained experimental wells")
    if simulations.duplicated(["event", "strategy", "initial_condition"]).any():
        raise ValueError("Duplicate simulated seed/strategy/event outcome")
    joined = simulations.merge(cohorts, on="initial_condition", how="inner", validate="many_to_one")
    joined["valid_endpoint"] = (
        joined["event_status"].isin(["observed", "event_not_observed"])
        & np.isfinite(joined["time"]) & joined["time"].ge(0))
    valid = joined.loc[joined["valid_endpoint"]]
    if not valid["observed"].eq(valid["event_status"].eq("observed")).all():
        raise ValueError("Event flags disagree with endpoint status")
    return cohorts, joined


def km_curve(times, observed):
    times, observed = np.asarray(times, float), np.asarray(observed, bool)
    sample = CensoredData(uncensored=times[observed], right=times[~observed])
    sf = ecdf(sample).sf
    return np.asarray(sf.quantiles), np.asarray(sf.probabilities)


def km_summary(times, observed, tau):
    times = np.asarray(times, float)
    t, survival = km_curve(times, observed)
    hits = np.flatnonzero(survival <= 0.5)
    median = float(t[hits[0]]) if len(hits) else np.nan
    # No positive-survival tail is extrapolated past the final available bound.
    if tau > times.max() and (not len(survival) or survival[-1] > 0):
        rmst = np.nan
    else:
        knots = np.r_[0.0, t[(t > 0) & (t < tau)], float(tau)]
        i = np.searchsorted(t, knots[:-1], side="right") - 1
        levels = np.where(i >= 0, survival[np.maximum(i, 0)], 1.0)
        rmst = float(np.sum(np.diff(knots) * levels))
    return median, rmst


def multigroup_logrank(times, observed, labels):
    """Standard tied-event logrank covariance; one independent seed per group."""
    times, observed, labels = np.asarray(times, float), np.asarray(observed, bool), np.asarray(labels)
    levels = sorted(set(labels))
    k = len(levels)
    if k < 2:
        return {"logrank_chi2": np.nan, "df": 0, "p_logrank": np.nan}
    scores = np.zeros(k)
    covariance = np.zeros((k, k))
    for time in np.unique(times[observed]):
        risk = np.array([np.sum((labels == label) & (times >= time)) for label in levels], float)
        events = np.array([np.sum((labels == label) & (times == time) & observed)
                           for label in levels], float)
        n, d = risk.sum(), events.sum()
        probabilities = risk / n
        scores += events - d * probabilities
        if n > 1:
            covariance += d * (n - d) / (n - 1) * (
                np.diag(probabilities) - np.outer(probabilities, probabilities))
    degrees = int(np.linalg.matrix_rank(covariance))
    if degrees == 0:
        return {"logrank_chi2": np.nan, "df": 0, "p_logrank": np.nan}
    statistic = max(0.0, float(scores @ np.linalg.pinv(covariance) @ scores))
    return {"logrank_chi2": statistic, "df": degrees,
            "p_logrank": float(chi2.sf(statistic, degrees))}


def analyse(joined):
    summaries, tests = [], []
    for event in EVENTS:
        for reference in STRATEGIES:
            data = joined.loc[joined["event"].eq(event) & joined["strategy"].eq(reference)
                              & joined["valid_endpoint"]]
            if data.empty:
                continue
            # Common integration horizon within this event/reference comparison.
            tau = float(data.groupby("experimental_strategy")["time"].max().min())
            for group in GROUPS:
                sample = data.loc[data["experimental_strategy"].eq(group)]
                if sample.empty:
                    continue
                median, rmst = km_summary(sample["time"], sample["observed"], tau)
                summaries.append({
                    "event": event, "simulated_reference_strategy": reference,
                    "experimental_strategy": group, "n_initial_conditions": len(sample),
                    "n_events": int(sample["observed"].sum()),
                    "n_right_censored": int((~sample["observed"]).sum()),
                    "km_median_days": median, "rmst_horizon_days": tau,
                    "restricted_mean_days": rmst,
                })
            tests.append({
                "event": event, "simulated_reference_strategy": reference,
                "n_initial_conditions": len(data),
                "n_experimental_groups": data["experimental_strategy"].nunique(),
                **multigroup_logrank(data["time"], data["observed"], data["experimental_strategy"])
            })
    tests = pd.DataFrame(tests)
    tests["q_bh_all_references_and_events"] = benjamini_hochberg(tests["p_logrank"].to_numpy())
    return pd.DataFrame(summaries), tests


def plot_ct_reference(joined, tests, output):
    ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6))
    for ax, event, letter in zip(axes, EVENTS, ("A", "B")):
        data = joined.loc[joined["event"].eq(event) & joined["strategy"].eq("CT")
                          & joined["valid_endpoint"]]
        for group in GROUPS:
            sample = data.loc[data["experimental_strategy"].eq(group)]
            if sample.empty:
                continue
            t, survival = km_curve(sample["time"], sample["observed"])
            color = get_color(group)
            ax.step(np.r_[0.0, t], np.r_[1.0, survival], where="post", color=color, lw=1.1,
                    label=f"{LABELS[group]}-assigned (n={len(sample)})")
            censored_times = sample.loc[~sample["observed"], "time"].to_numpy()
            if len(censored_times):
                indices = np.searchsorted(t, censored_times, side="right") - 1
                censored_survival = np.where(indices >= 0, survival[np.maximum(indices, 0)], 1)
                ax.plot(censored_times, censored_survival, linestyle="none", marker="|",
                        markersize=4.5, markeredgewidth=0.9, color=color)
        row = tests.loc[tests["event"].eq(event)
                        & tests["simulated_reference_strategy"].eq("CT")].iloc[0]
        ax.text(0.97, 0.97, f"Cohort logrank p={row['p_logrank']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set(xlim=(0, data["time"].max() * 1.05), ylim=(-0.02, 1.09),
               xlabel=f"Simulated {event} under CT [days]",
               ylabel="Fraction without event")
        ax.set_title(f"{letter}   {event}: CT simulated for every cohort",
                     loc="left", fontsize=7, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=4, frameon=False, columnspacing=1.4)
    fig.text(0.5, 0.052, "Colors identify the original experimental assignment; every trajectory here uses simulated CT.",
             ha="center", fontsize=6)
    fig.text(0.5, 0.02, "Vertical ticks mark right-censoring; unobserved endpoints are not treated as event times.",
             ha="center", fontsize=6)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.88, bottom=0.29, wspace=0.34)
    ft.savefig(fig, output, transparent=False)
    plt.close(fig)


def write_methods(path, cohorts, summary, tests):
    ct = summary.loc[summary["simulated_reference_strategy"].eq("CT")]
    text = f"""# Experimental cohort initial-condition comparison

Question: did the initial conditions assigned to some experimental treatments
start with a worse simulated prognosis than those assigned to other treatments?

## Population and matching

Use the {len(cohorts)} distinct retained experimental wells with exact seed-ID
matches in the existing experiment-simulation table. Experimental exclusions
are inherited from that matched cohort. Group membership is the treatment
actually assigned to the original experimental well, not the treatment used
in the counterfactual simulation. No experiment-ATagent cohort is available
in this matched initial-condition set, so no conclusion about its allocation
is possible. This is a comparison of the represented matched wells, not proof
of balance across every experimental replicate.

## Common-treatment reference

The primary comparison simulates CT for all four experimental seed groups.
That holds simulated treatment fixed while comparing initial-condition groups.
Compare within each reference strategy only. Other simulated strategies are
reported as sensitivity analyses; they are not pooled as independent replicates.
Some reference strategies lack particular experimental seed groups (notably
simulated AT+ lacks AT50-origin seeds); their tests cover only available groups.

TTF/TTE use acquisition days and the existing definitions. Shorter event times
mean worse control under the reference simulation. Use valid observed events
and valid right-censoring bounds; discard invalid event/baseline records.
Censoring means only that the event was not reached by the last available day.
The analysis assumes censoring does not depend on unobserved future event time
within each cohort. Technical truncation could violate that assumption.
These are model-based, exploratory prognosis comparisons, not a causal test
of initial-condition allocation or a direct measurement of experimental prognosis.

## Estimation and testing

Kaplan-Meier curves use scipy.stats.ecdf(CensoredData). The median is the first
time estimated event-free fraction <= 0.5 (not the median of observed failures).
A missing median means it was not reached. Restricted mean event-free time
integrates the KM curve from day 0 to a common horizon, separately for each
event/reference: the minimum across groups of their maximum endpoint day.
The horizon and RMST are descriptive and differ between reference strategies;
do not compare RMST across different horizons. No CI is claimed for these medians
or RMSTs.

The global tied-event logrank compares experimental groups within each simulated
reference. At each event time, expected events are d * n_group/n_total and
covariance is d*(n_total-d)/(n_total-1) * (diag(p)-p*p.T).
Use the covariance pseudoinverse and its rank for the chi-squared approximation.
Global p-values are unadjusted; the CSV additionally applies BH adjustment
across all {len(tests)} event/reference tests. Comparisons share seeds, so those
tests are dependent and exploratory. No pairwise significance claims are made.
Absence of significance does not establish equivalence or successful allocation.
Survival curve differences need not satisfy proportional hazards; logrank power
may be poor for crossing curves.

## CT benchmark results

```text
{ct.to_string(index=False)}
```

## References

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html
- https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logrank.html
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-pairs-csv", type=Path,
                        default=BASE_DATA / "ttf_tte_matched/matched_event_times.csv")
    parser.add_argument("--simulation-events-csv", type=Path,
                        default=BASE_DATA / "ttf_tte_matched_simulation_strategies/simulation_event_times.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DATA / "experimental_cohort_initial_conditions")
    parser.add_argument("--output-plot-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()
    cohorts, joined = join_experimental_cohorts(pd.read_csv(args.matched_pairs_csv),
                                                pd.read_csv(args.simulation_events_csv))
    summary, tests = analyse(joined)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(args.output_dir / "experimental_cohort_membership.csv", index=False)
    joined.to_csv(args.output_dir / "counterfactual_simulation_endpoint_audit.csv", index=False)
    summary.to_csv(args.output_dir / "cohort_prognosis_by_simulated_reference.csv", index=False)
    tests.to_csv(args.output_dir / "global_cohort_logrank_tests.csv", index=False)
    ct = joined.loc[joined["strategy"].eq("CT") & joined["valid_endpoint"]]
    ct.to_csv(args.output_dir / "common_CT_reference_endpoints.csv", index=False)
    plot_ct_reference(joined, tests, args.output_plot_dir / "TTF_TTE_Experimental_Cohorts_Common_CT.pdf")
    write_methods(args.output_dir / "METHODS.md", cohorts, summary, tests)
    print("CT benchmark:")
    print(summary.loc[summary["simulated_reference_strategy"].eq("CT")].to_string(index=False))
    print("Global cohort comparisons under each simulated reference:")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
