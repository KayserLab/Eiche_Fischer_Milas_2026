"""Statistics that respect simulation runs sharing an experiment-derived initial condition.

Every experiment-derived simulation run (run >= 10) was initialised from one original
experimental colony, recorded as ``initial_condition`` (full experimental sample ID, e.g.
``D5_AT50-D5``) in ``combined_results.csv``. Runs of different strategies started from the
same initial condition are not independent, so simulation comparisons use:

* two-sided log-rank tests stratified by initial condition on the shared initial conditions,
  retaining censored records;
* bootstrap CIs that resample matched initial-condition blocks (all compared strategies of a
  sampled initial condition enter together);
* cluster bootstrap CIs for correlations that resample whole initial-condition clusters;
* paired inference on matched initial conditions for two-strategy run-level summaries.

Experimental colonies are independent units and keep the unmatched procedures.
This module only reads ``combined_results.csv`` (initial-condition metadata).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon

from shared.data_paths import COMBINED_RESULTS_CSV

BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260827

# Strategy labels used by the different scripts -> protocol label in combined_results.csv.
PROTOCOL_BY_STRATEGY = {
    "CT": "CT", "AT50": "AT50", "AT100": "AT100",
    "LV-Agent": "AGENT", "ATAgent2": "AGENT", "Agent": "AGENT", "AGENT": "AGENT",
    "AT100+": "DENS", "Dens": "DENS", "DENS": "DENS",
}
INVALID_INITIAL_CONDITIONS = {"", "unknown", "artificial", "nan"}


def run_number(value) -> int | None:
    """Return the run number from ``R25_CT-R25``, ``run_25``, ``CT_run_25`` or ``25``."""
    text = str(value)
    match = re.match(r"^R(\d+)_", text) or re.search(r"run_(\d+)$", text) or re.fullmatch(r"(\d+)(?:\.0)?", text)
    return int(match.group(1)) if match else None


@lru_cache(maxsize=4)
def _initial_condition_table(path: str) -> dict[tuple[str, int], str]:
    table = pd.read_csv(path, usecols=["protocol", "run", "initial_condition"], dtype=str)
    table["protocol"] = table["protocol"].str.strip().str.upper()
    table["run"] = pd.to_numeric(table["run"], errors="coerce")
    table = table.dropna(subset=["run"]).drop_duplicates(["protocol", "run"])
    lookup = {}
    for row in table.itertuples(index=False):
        value = str(row.initial_condition).strip()
        lookup[(row.protocol, int(row.run))] = "" if value.lower() in INVALID_INITIAL_CONDITIONS else value
    return lookup


def initial_condition(strategy: str, run, path: Path = COMBINED_RESULTS_CSV) -> str:
    """Initial-condition ID of one simulation trajectory (empty string if unknown)."""
    protocol = PROTOCOL_BY_STRATEGY.get(str(strategy), str(strategy).upper())
    number = run_number(run)
    if number is None:
        return ""
    return _initial_condition_table(str(path)).get((protocol, number), "")


def attach_initial_conditions(frame: pd.DataFrame, *, strategy_col: str = "strategy",
                              run_col: str = "sample", out_col: str = "initial_condition") -> pd.DataFrame:
    out = frame.copy()
    out[out_col] = [initial_condition(s, r) for s, r in zip(out[strategy_col], out[run_col])]
    return out


def matched_records(a: pd.DataFrame, b: pd.DataFrame, *, key: str = "initial_condition") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two strategy tables to their shared initial conditions, aligned row by row."""
    for name, frame in (("a", a), ("b", b)):
        valid = frame[key].astype(str).str.len() > 0
        if frame.loc[valid, key].duplicated().any():
            raise ValueError(f"Duplicate {key} within strategy table {name}; cannot match uniquely")
    shared = sorted(set(a.loc[a[key] != "", key]) & set(b.loc[b[key] != "", key]))
    return (a.set_index(key).loc[shared].reset_index(), b.set_index(key).loc[shared].reset_index())


# ------------------------------------------------------------------ survival
def kaplan_meier_median(durations, events) -> float:
    """First time at which the Kaplan--Meier survival estimate is <= 0.5 (NaN if never)."""
    t, e = np.asarray(durations, float), np.asarray(events, bool)
    keep = np.isfinite(t); t, e = t[keep], e[keep]
    survival = 1.0
    for time in np.unique(t[e]):
        at_risk = int(np.sum(t >= time))
        if at_risk:
            survival *= 1.0 - int(np.sum((t == time) & e)) / at_risk
        if survival <= 0.5:
            return float(time)
    return float("nan")


def _logrank_terms(ta, ea, tb, eb) -> tuple[float, float]:
    ta, tb = np.asarray(ta, float), np.asarray(tb, float)
    ea, eb = np.asarray(ea, bool), np.asarray(eb, bool)
    observed_minus_expected = variance = 0.0
    for time in np.unique(np.concatenate([ta[ea], tb[eb]])):
        na, nb = int(np.sum(ta >= time)), int(np.sum(tb >= time)); n = na + nb
        da, db = int(np.sum((ta == time) & ea)), int(np.sum((tb == time) & eb)); d = da + db
        if n <= 1 or d == 0:
            continue
        observed_minus_expected += da - d * na / n
        variance += na * nb * d * (n - d) / (n * n * (n - 1))
    return observed_minus_expected, variance


def logrank_test(ta, ea, tb, eb) -> dict:
    """Ordinary two-sided two-sample log-rank test (chi-square, 1 df)."""
    oe, var = _logrank_terms(ta, ea, tb, eb)
    stat = oe * oe / var if var > 0 else float("nan")
    return {"test": "two-sided log-rank", "chi2": stat, "df": 1,
            "p": float(chi2.sf(stat, 1)) if np.isfinite(stat) else float("nan")}


def stratified_logrank_test(a: pd.DataFrame, b: pd.DataFrame, *, time: str, event: str,
                            stratum: str = "initial_condition") -> dict:
    """Two-sided log-rank test stratified by initial condition, on shared initial conditions.

    Observed-minus-expected events and their variances are summed over strata; censored
    records are retained. Only initial conditions present in both groups contribute.
    """
    ma, mb = matched_records(a, b, key=stratum)
    oe = var = 0.0
    for ra, rb in zip(ma.itertuples(index=False), mb.itertuples(index=False)):
        o, v = _logrank_terms([getattr(ra, time)], [getattr(ra, event)], [getattr(rb, time)], [getattr(rb, event)])
        oe += o; var += v
    stat = oe * oe / var if var > 0 else float("nan")
    return {"test": "two-sided log-rank stratified by initial condition", "chi2": stat, "df": 1,
            "p": float(chi2.sf(stat, 1)) if np.isfinite(stat) else float("nan"),
            "n_shared_initial_conditions": len(ma),
            "events_a": int(np.sum(ma[event].astype(bool))), "events_b": int(np.sum(mb[event].astype(bool)))}


def _bootstrap_km_medians(t, e, indices) -> np.ndarray:
    """KM medians of bootstrap samples given by an index matrix (NaN where undefined)."""
    t, e = np.asarray(t, float), np.asarray(e, bool)
    bt, be = t[indices], e[indices]
    reps = indices.shape[0]
    survival, medians = np.ones(reps), np.full(reps, np.nan)
    for day in np.unique(t[e]):
        at_risk = np.sum(bt >= day, axis=1)
        hits = np.sum((bt == day) & be, axis=1)
        survival *= 1.0 - np.divide(hits, at_risk, out=np.zeros(reps), where=at_risk > 0)
        medians[np.isnan(medians) & (survival <= 0.5)] = day
    return medians


def _percentile(differences: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(differences)
    if not valid.any():
        return float("nan"), float("nan"), int(differences.size)
    low, high = np.quantile(differences[valid], [0.025, 0.975])
    return float(low), float(high), int((~valid).sum())


def km_median_difference_ci_independent(ref_t, ref_e, comp_t, comp_e, rng, reps: int = BOOTSTRAP_REPEATS) -> dict:
    """Comparison-minus-reference KM-median difference; percentile bootstrap resampling
    records independently within each group (reference drawn first). Resamples with an
    undefined KM median in either group are discarded and counted."""
    ref_t, comp_t = np.asarray(ref_t, float), np.asarray(comp_t, float)
    ref = _bootstrap_km_medians(ref_t, ref_e, rng.integers(len(ref_t), size=(reps, len(ref_t))))
    comp = _bootstrap_km_medians(comp_t, comp_e, rng.integers(len(comp_t), size=(reps, len(comp_t))))
    low, high, undefined = _percentile(comp - ref)
    return {"effect": kaplan_meier_median(comp_t, comp_e) - kaplan_meier_median(ref_t, ref_e),
            "ci_low": low, "ci_high": high, "undefined_resamples": undefined, "resamples": reps,
            "ci_method": "percentile bootstrap; records resampled independently within each group"}


def km_median_difference_ci_matched(ref: pd.DataFrame, comp: pd.DataFrame, *, time: str, event: str,
                                    rng, reps: int = BOOTSTRAP_REPEATS) -> dict:
    """KM-median difference on shared initial conditions; percentile bootstrap resampling
    matched initial-condition blocks jointly (both strategies of a block enter together)."""
    ma, mb = matched_records(ref, comp)
    idx = rng.integers(len(ma), size=(reps, len(ma)))
    a = _bootstrap_km_medians(ma[time], ma[event], idx)
    b = _bootstrap_km_medians(mb[time], mb[event], idx)
    low, high, undefined = _percentile(b - a)
    return {"effect": kaplan_meier_median(mb[time], mb[event]) - kaplan_meier_median(ma[time], ma[event]),
            "median_ref": kaplan_meier_median(ma[time], ma[event]),
            "median_comp": kaplan_meier_median(mb[time], mb[event]),
            "n_shared_initial_conditions": len(ma),
            "ci_low": low, "ci_high": high, "undefined_resamples": undefined, "resamples": reps,
            "ci_method": "percentile bootstrap; matched initial-condition blocks resampled jointly"}


def survival_contrasts(experimental: pd.DataFrame, simulation: pd.DataFrame, *, time: str, event: str,
                       strategies: tuple[str, ...], seed: int = BOOTSTRAP_SEED,
                       reps: int = BOOTSTRAP_REPEATS) -> list[dict]:
    """Fig. 2d,e contrasts (comparison minus reference) for every strategy pair.

    Effect: difference of Kaplan--Meier medians with right-censored follow-up.
    Experiments (independent colonies): ordinary log-rank; CI resamples colonies within
    each strategy. Simulations: log-rank stratified by initial condition and KM medians on
    the shared initial conditions; CI resamples matched initial-condition blocks.
    One generator per endpoint; experimental pairs are drawn before simulated pairs.
    """
    from itertools import combinations

    rng = np.random.default_rng(seed)
    rows = []
    for source, table in (("Experiment", experimental), ("Simulation", simulation)):
        for ref, comp in combinations(strategies, 2):
            a, b = table[table["strategy"] == ref], table[table["strategy"] == comp]
            if len(a) < 2 or len(b) < 2:
                continue
            row = {"source": source, "reference": ref, "comparison": comp,
                   "n_reference": len(a), "n_comparison": len(b),
                   "events_reference": int(a[event].astype(bool).sum()),
                   "events_comparison": int(b[event].astype(bool).sum()),
                   "km_median_reference_all": kaplan_meier_median(a[time], a[event]),
                   "km_median_comparison_all": kaplan_meier_median(b[time], b[event])}
            if source == "Experiment":
                effect = km_median_difference_ci_independent(a[time], a[event], b[time], b[event], rng, reps)
                effect.update(median_ref=row["km_median_reference_all"], median_comp=row["km_median_comparison_all"],
                              n_shared_initial_conditions=None)
                test = logrank_test(a[time], a[event], b[time], b[event])
            else:
                effect = km_median_difference_ci_matched(a, b, time=time, event=event, rng=rng, reps=reps)
                test = stratified_logrank_test(a, b, time=time, event=event)
                row["events_reference_matched"], row["events_comparison_matched"] = test["events_a"], test["events_b"]
            row.update({("delta_days" if k == "effect" else k): v for k, v in effect.items()})
            row.update({"test": test["test"], "chi2": test["chi2"], "df": test["df"], "p": test["p"],
                        "adjustment": "none"})
            rows.append(row)
    return rows


def format_survival_contrast(row: dict) -> str:
    matched = (f"; shared ICs={row['n_shared_initial_conditions']} (events {row['events_reference_matched']}/"
               f"{row['events_comparison_matched']})" if row["n_shared_initial_conditions"] else "")
    return (f"    {row['reference']} vs {row['comparison']} ({row['comparison']} - {row['reference']}): "
            f"n={row['n_reference']}/{row['n_comparison']}, events={row['events_reference']}/"
            f"{row['events_comparison']}{matched}; KM medians={row['median_ref']:.4g}/{row['median_comp']:.4g} "
            f"days; delta={row['delta_days']:.4g} days; 95% CI={row['ci_low']:.4g}--{row['ci_high']:.4g} "
            f"({row['undefined_resamples']}/{row['resamples']} undefined resamples dropped); "
            f"{row['test']}: chi2={row['chi2']:.6g}, df=1, p={row['p']:.6g} (unadjusted)")


# ------------------------------------------------------------------ correlation
def pearson_r(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    return float(np.sum(xc * yc) / den) if den > 0 else float("nan")


def cluster_bootstrap_pearson_ci(x, y, clusters, rng, reps: int = BOOTSTRAP_REPEATS) -> dict:
    """Percentile CI for Pearson r resampling whole clusters with replacement and keeping
    every row of each sampled cluster. Degenerate resamples (zero variance) are dropped."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    labels = pd.Series(clusters).astype(str).to_numpy()
    unique = np.unique(labels)
    members = [np.flatnonzero(labels == c) for c in unique]
    draws = rng.integers(len(unique), size=(reps, len(unique)))
    values = np.full(reps, np.nan)
    for i, row in enumerate(draws):
        idx = np.concatenate([members[j] for j in row])
        values[i] = pearson_r(x[idx], y[idx])
    low, high, dropped = _percentile(values)
    return {"r": pearson_r(x, y), "ci_low": low, "ci_high": high, "n_rows": len(x),
            "n_clusters": len(unique), "degenerate_resamples": dropped, "resamples": reps,
            "ci_method": "percentile cluster bootstrap; whole initial-condition clusters resampled"}


# ------------------------------------------------------------------ paired two-strategy summaries
def paired_comparison(reference: pd.DataFrame, comparison: pd.DataFrame, value: str, rng,
                      reps: int = BOOTSTRAP_REPEATS) -> dict:
    """Matched-initial-condition comparison of one run-level summary (comparison minus reference).

    Effect (retained definition): difference of group medians, median(comparison) -
    median(reference), computed on the matched pairs. Also reported: median of within-pair
    differences. CIs resample matched pairs jointly. Test: two-sided Wilcoxon signed-rank on
    within-pair differences; W+ is the sum of ranks of positive (comparison > reference)
    differences.
    """
    ma, mb = matched_records(reference, comparison)
    a, b = ma[value].to_numpy(float), mb[value].to_numpy(float)
    d = b - a
    idx = rng.integers(len(a), size=(reps, len(a)))
    group_diff = np.median(b[idx], axis=1) - np.median(a[idx], axis=1)
    pair_median = np.median(d[idx], axis=1)
    test = wilcoxon(b, a, alternative="two-sided")
    nonzero = d[d != 0]
    ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
    return {
        "n_pairs": len(d), "median_reference": float(np.median(a)), "median_comparison": float(np.median(b)),
        "difference_of_group_medians": float(np.median(b) - np.median(a)),
        "difference_of_group_medians_ci": [float(q) for q in np.quantile(group_diff, [0.025, 0.975])],
        "median_within_pair_difference": float(np.median(d)),
        "median_within_pair_difference_ci": [float(q) for q in np.quantile(pair_median, [0.025, 0.975])],
        "ci_method": "percentile bootstrap; matched initial-condition pairs resampled jointly",
        "resamples": reps,
        "test": "two-sided Wilcoxon signed-rank on within-pair differences (comparison - reference)",
        "W_plus": float(ranks[nonzero > 0].sum()), "W_minus": float(ranks[nonzero < 0].sum()),
        "scipy_statistic_min_W": float(test.statistic), "p": float(test.pvalue),
        "zero_differences": int(np.sum(d == 0)),
    }
