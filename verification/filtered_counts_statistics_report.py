"""Collect the final manuscript statistics after the filtered-count TTF harmonization.

Recomputes, from the same inputs and functions used by the figure scripts:
  Fig. 2d,e  KM-median differences, CIs and log-rank tests (+ historical-method audit)
  Fig. 2f    Pearson r with CIs
  Fig. 4h,i  cohorts, KM medians, pairwise log-rank tests (Bonferroni x10)
  Fig. 4j    ATagent vs AT+ effects/tests, plotted means/SDs
  Fig. 3h,i  completed density audit (validity check only)
and verifies the supplementary consumers of simulated TTF and the experimental time origin.

Reads the data root; writes only to outputs/filtered_counts_statistics_20261110/ in the repo.
Run after analysis/figure_4/ef_Figure_4_prepare_plot_data.py and the figure scripts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "filtered_counts_statistics_20261110"
OUT.mkdir(parents=True, exist_ok=True)
BACKUP = ROOT / "tmp" / "filtered_counts_backup_20261110" / "files"

from shared.data_paths import DATA_ROOT, figure_data_dir  # noqa: E402
from analysis.figure_2.Event_Time_Analysis import (  # noqa: E402
    derive_progression, derive_tte, load_escape_margins, load_population_counts,
    prepare_progression_statistics, simulation_failure_times, ESCAPE_MARGIN_CSV,
)
from analysis import shared_initial_condition_statistics as ics  # noqa: E402


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tte_mod = load_script("fig2d_tte", "figures/Figure_2/b_TTE CT & AT100.py")
ttf_mod = load_script("fig2e_ttf", "figures/Figure_2/c_TTF CT & AT100.py")
cor_mod = load_script("fig2f_cor", "figures/Figure_2/d_plot_ttf_vs_tte.py")
from figures.Figure_4 import e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico as km  # noqa: E402
from figures.Figure_4 import f_Figure_4_Cumulative_Treatment_average_burden_in_vitro_and_in_silico as burden  # noqa: E402

FIG2 = ("CT", "AT50", "AT100")
ALL_SIM = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")
KM_KEY = {"CT": "CT", "AT50": "AT50", "AT100": "AT100", "LV-Agent": "ATAgent2", "AT100+": "Dens"}
DISPLAY = {"CT": "CT", "AT50": "AT50", "AT100": "AT100", "ATAgent2": "ATagent", "Dens": "AT+",
           "LV-Agent": "ATagent", "AT100+": "AT+"}
MANUSCRIPT_FIG2 = {("TTE", "Experiment", "AT50"): (16, 0, 23), ("TTE", "Experiment", "AT100"): (23, 7, 49),
                   ("TTE", "Simulation", "AT50"): (4, 2, 14), ("TTE", "Simulation", "AT100"): (58, 42, 75),
                   ("TTF", "Experiment", "AT50"): (14, 0, 21), ("TTF", "Experiment", "AT100"): (19, 2, 33),
                   ("TTF", "Simulation", "AT50"): (11, 8, 17), ("TTF", "Simulation", "AT100"): (55, 47, 67)}
report: dict = {"data_root": str(DATA_ROOT), "checks": []}


def check(name: str, ok: bool, detail=None) -> None:
    report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f" | {detail}"), flush=True)


# =============================================================== inputs (loaded once)
print("loading inputs", flush=True)
margins = load_escape_margins(ESCAPE_MARGIN_CSV)
exp_counts_all = load_population_counts(dataset_type="experimental", strategies=ALL_SIM)
sim_counts_all = load_population_counts(dataset_type="simulation", strategies=ALL_SIM)
canonical = simulation_failure_times(ALL_SIM, counts=sim_counts_all)
canonical["key"] = canonical["strategy"].map(KM_KEY)
canonical["sample_key"] = canonical["key"] + "_run_" + canonical["run"].astype(str)
canonical["initial_condition"] = [ics.initial_condition(k, r) for k, r in zip(canonical["key"], canonical["run"])]
canonical.to_csv(OUT / "canonical_simulated_ttf.csv", index=False)
check("every canonical simulation run has an initial condition", canonical["initial_condition"].ne("").all())
first_days = sim_counts_all.groupby(["strategy", "sample"])["day"].min()
report["runs_without_day1_baseline"] = [f"{k[0]} {k[1]} (first day {v:g})" for k, v in first_days.items() if v != 1]
check("day-1 rule: only AT+ R23 (B5_AT100-B5) lacks a day-1 snapshot and it is excluded",
      report["runs_without_day1_baseline"] == ["AT100+ R23_Dens-R23 (first day 82)"] and "Dens_run_23" not in set(canonical["sample_key"]),
      report["runs_without_day1_baseline"])
check("canonical cohorts CT/AT50/AT100/ATagent/AT+ = 36/35/36/35/23",
      canonical.groupby("key").size().to_dict() == {"AT100": 36, "AT50": 35, "ATAgent2": 35, "CT": 36, "Dens": 23},
      canonical.groupby("key").size().to_dict())

# explicit endpoint checks (derived from inputs)
for key, expected_strategy, run in (("CT_run_25", "CT", 25), ("ATAgent2_run_20", "LV-Agent", 20)):
    row = canonical.loc[canonical["sample_key"] == key].iloc[0]
    counts = sim_counts_all[(sim_counts_all["strategy"] == expected_strategy) &
                            (sim_counts_all["sample"].str.match(rf"^R{run}_"))].sort_values("day")
    n0 = float(counts["total"].iloc[0])
    day = row["ttp"]
    before = counts[counts["day"] == day - 1].iloc[0]; at = counts[counts["day"] == day].iloc[0]
    report.setdefault("endpoint_examples", []).append({
        "trajectory": key, "initial_condition": row["initial_condition"], "filtered_ttf_day": day,
        "event": bool(row["progressed"]), "denominator_day1_filtered_total": n0,
        "day_before": {"day": float(before["day"]), "res": float(before["res"]), "fraction": float(before["res"] / n0)},
        "failure_day": {"day": float(at["day"]), "res": float(at["res"]), "fraction": float(at["res"] / n0)}})
check("CT_run_25 filtered TTF derived as day 26",
      canonical.loc[canonical.sample_key == "CT_run_25", "ttp"].item() == 26)
check("ATagent run_20 filtered TTF derived as day 89",
      canonical.loc[canonical.sample_key == "ATAgent2_run_20", "ttp"].item() == 89)

# =============================================================== Fig. 2d,e
print("Fig. 2d,e", flush=True)
tte_exp = tte_mod.prepare_tte_statistics(derive_tte(margins, dataset_type="experimental", strategies=FIG2, apply_truncation=True))
tte_sim = tte_mod.load_simulation_tte(margins)
ttf_exp = prepare_progression_statistics(exp_counts_all[exp_counts_all.strategy.isin(FIG2)], apply_truncation=True)
ttf_sim = ics.attach_initial_conditions(canonical[canonical.strategy.isin(FIG2)].copy(), strategy_col="strategy", run_col="sample")
check("Fig. 2e simulation table == canonical endpoint (same rows, times, events)",
      len(ttf_sim) == 107 and set(ttf_sim["sample"]) == set(ttf_mod.load_simulation_ttp()["sample"]))
fig2 = {"TTE": tte_mod.tte_contrasts(tte_exp, tte_sim), "TTF": ttf_mod.ttf_contrasts(ttf_exp, ttf_sim)}
for endpoint, rows in fig2.items():
    table_sim = tte_sim if endpoint == "TTE" else ttf_sim
    time, event = ("tte", "escaped") if endpoint == "TTE" else ("ttp", "progressed")
    for row in rows:
        row["endpoint"] = endpoint
        if row["source"] == "Simulation":
            a = table_sim[table_sim.strategy == row["reference"]]; b = table_sim[table_sim.strategy == row["comparison"]]
            row["unstratified_logrank_p_same_runs"] = ics.logrank_test(a[time], a[event], b[time], b[event])["p"]
            row["matched_medians_equal_plotted_medians"] = bool(
                row["median_ref"] == row["km_median_reference_all"] and row["median_comp"] == row["km_median_comparison_all"])
        ms = MANUSCRIPT_FIG2.get((endpoint, row["source"], row["comparison"])) if row["reference"] == "CT" else None
        row["manuscript_delta_ci"] = ms


def historical_published_method(tables: dict, time: str, event: str) -> list[dict]:
    """tmp/figure2_results_comparison.py: KM medians, records resampled independently within
    each strategy for BOTH sources, one generator per endpoint, experiments first."""
    rng = np.random.default_rng(ics.BOOTSTRAP_SEED); out = []
    for source in ("Experiment", "Simulation"):
        t = tables[source]
        for ref, comp in combinations(FIG2, 2):
            a, b = t[t.strategy == ref], t[t.strategy == comp]
            r = ics.km_median_difference_ci_independent(a[time], a[event], b[time], b[event], rng)
            out.append({"source": source, "reference": ref, "comparison": comp, "delta": r["effect"],
                        "ci": [r["ci_low"], r["ci_high"]], "undefined": r["undefined_resamples"]})
    return out


def historical_event_only(tables: dict, time: str, event: str) -> list[dict]:
    """Git revision b8ea40e (27 Aug - 6 Oct 2026): plain medians of observed events only."""
    rng = np.random.default_rng(ics.BOOTSTRAP_SEED); out = []
    for source in ("Experiment", "Simulation"):
        t = tables[source]; t = t[t[event].astype(bool)]
        for ref, comp in combinations(FIG2, 2):
            a, b = t[t.strategy == ref][time].to_numpy(float), t[t.strategy == comp][time].to_numpy(float)
            ab = rng.choice(a, size=(ics.BOOTSTRAP_REPEATS, a.size)); bb = rng.choice(b, size=(ics.BOOTSTRAP_REPEATS, b.size))
            lo, hi = np.quantile(np.median(bb, 1) - np.median(ab, 1), [.025, .975])
            out.append({"source": source, "reference": ref, "comparison": comp,
                        "delta": float(np.median(b) - np.median(a)), "ci": [float(lo), float(hi)]})
    return out


report["fig2_historical"] = {}
# historical runs used the pre-harmonization Fig. 2e row order (prepare_progression_statistics order)
ttf_sim_historical_order = prepare_progression_statistics(sim_counts_all[sim_counts_all.strategy.isin(FIG2)],
                                                          apply_truncation=False)
for endpoint, tables, time, event in (("TTE", {"Experiment": tte_exp, "Simulation": tte_sim}, "tte", "escaped"),
                                      ("TTF", {"Experiment": ttf_exp, "Simulation": ttf_sim_historical_order}, "ttp", "progressed")):
    pub = historical_published_method(tables, time, event); old = historical_event_only(tables, time, event)
    for kind, rows in (("published_km_independent", pub), ("event_only_b8ea40e", old)):
        for r in rows:
            ms = MANUSCRIPT_FIG2.get((endpoint, r["source"], r["comparison"])) if r["reference"] == "CT" else None
            r["manuscript"] = ms
            r["matches_manuscript"] = None if ms is None else bool(
                round(r["delta"]) == ms[0] and round(r["ci"][0]) == ms[1] and round(r["ci"][1]) == ms[2])
        report["fig2_historical"][f"{endpoint}_{kind}"] = rows
pub_all = [r for k, v in report["fig2_historical"].items() if k.endswith("published_km_independent") for r in v if r["manuscript"]]
old_all = [r for k, v in report["fig2_historical"].items() if k.endswith("event_only_b8ea40e") for r in v if r["manuscript"]]
check("historical KM/independent-bootstrap method reproduces all 8 manuscript effects+CIs",
      all(r["matches_manuscript"] for r in pub_all), f"{sum(r['matches_manuscript'] for r in pub_all)}/8")
check("historical event-only method reproduces none of the 8", not any(r["matches_manuscript"] for r in old_all),
      f"{sum(r['matches_manuscript'] for r in old_all)}/8")
report["fig2de"] = fig2

# =============================================================== Fig. 2f
print("Fig. 2f", flush=True)
def events_table(endpoint: str, kind: str) -> pd.DataFrame:
    if endpoint == "TTE":
        t = derive_tte(margins, dataset_type=kind, strategies=FIG2, apply_truncation=kind == "experimental")
        return t[t.escaped.astype(bool)][["plate", "sample", "strategy", "tte", "source"]]
    counts = exp_counts_all[exp_counts_all.strategy.isin(FIG2)] if kind == "experimental" else \
        sim_counts_all[sim_counts_all.strategy.isin(FIG2)]
    t = derive_progression(counts, apply_truncation=kind == "experimental")
    return t[t.progressed.astype(bool)][["plate", "sample", "strategy", "ttf", "source"]]


invitro = events_table("TTE", "experimental").merge(events_table("TTF", "experimental"),
                                                    on=["plate", "sample", "strategy", "source"], validate="one_to_one")
insilico = events_table("TTE", "simulation").merge(events_table("TTF", "simulation"),
                                                   on=["plate", "sample", "strategy", "source"], validate="one_to_one")
insilico = ics.attach_initial_conditions(insilico, strategy_col="strategy", run_col="sample")
invitro["initial_condition"] = invitro["sample"]
merged = pd.concat([invitro, insilico]).dropna(subset=["tte", "ttf"]).sort_values(["source", "strategy", "sample"])
sim_ttf_check = insilico.merge(canonical[["strategy", "sample", "ttp", "progressed"]], on=["strategy", "sample"])
check("Fig. 2f simulated TTF == canonical endpoint", bool((sim_ttf_check.ttf == sim_ttf_check.ttp).all() and sim_ttf_check.progressed.all()))
saved_2f = pd.read_csv(figure_data_dir("Figure_2", "ttf_vs_tte", create=False) / "TTF_vs_TTE_Pearson_CT_AT50_AT100.csv")
cmp2f = merged.merge(saved_2f, on=["plate", "sample", "strategy", "source"], how="outer", indicator=True, suffixes=("", "_saved"))
check("regenerated Fig. 2f table == recomputed pairs", bool(cmp2f._merge.eq("both").all() and (cmp2f.tte == cmp2f.tte_saved).all()
                                                       and (cmp2f.ttf == cmp2f.ttf_saved).all()), f"rows={len(cmp2f)}")
report["fig2f"] = cor_mod.manuscript_correlations(merged)
report["fig2f_strategy_rows"] = merged.groupby(["source", "strategy"]).size().reset_index(name="n").to_dict("records")

# =============================================================== Fig. 4h,i
print("Fig. 4h,i", flush=True)
vitro_groups, vitro_rows = km.load_prepared_groups(km.CSV_PATH)
silico_groups, silico_rows = km.load_insilico_groups()
plotted = [s for s in km.INCLUDE_STRATEGIES if km._rows_for_strategy(vitro_groups, s) or km._rows_for_strategy(silico_groups, s)]
fig4 = {"order": plotted, "cohorts": [], "tests": {}}
for label, groups in (("Exp", vitro_groups), ("Sim", silico_groups)):
    for s in plotted:
        rows = km._rows_for_strategy(groups, s)
        t = [r["time"] for r in rows]; e = [r["event"] for r in rows]
        fig4["cohorts"].append({"dataset": label, "strategy": DISPLAY[s], "n": len(rows), "events": int(sum(e)),
                                "censored": int(len(e) - sum(e)), "km_median": ics.kaplan_meier_median(t, e)})
    fig4["tests"][label] = km.pairwise_logrank_rows(groups, plotted, stratified=label == "Sim")
silico_df = pd.DataFrame(silico_rows)
cmp4i = silico_df.merge(canonical[["sample_key", "ttp", "progressed"]], on="sample_key", how="outer", indicator=True)
check("Fig. 4i in-silico table == canonical endpoint (all strategies)",
      bool(cmp4i._merge.eq("both").all() and (cmp4i.time == cmp4i.ttp).all() and (cmp4i.event == cmp4i.progressed.astype(int)).all()),
      f"rows={len(cmp4i)}")
agent = silico_df[silico_df.strategy == "ATAgent2"]
check("agent outcome cohort = 35 runs (23 failures, 12 censored), Agent/run_35 excluded",
      len(agent) == 35 and agent.event.sum() == 23 and "ATAgent2_run_35" not in set(agent.sample_key))
check("AT+/run_35 retained", "Dens_run_35" in set(silico_df.sample_key),
      silico_df.loc[silico_df.sample_key == "Dens_run_35", ["initial_condition", "time", "event"]].to_dict("records"))
# unstratified tests on the same (new) table, and on the pre-harmonization table, for change attribution
old_km = BACKUP / "figure_data/Figure_4/kaplan_meier/time_to_failure_by_run_rel_0.5_in_silico.csv"
old_groups, _ = km.load_prepared_groups(old_km)
fig4["tests_sim_unstratified_new"] = km.pairwise_logrank_rows(silico_groups, plotted, stratified=False)
fig4["tests_sim_unstratified_old"] = km.pairwise_logrank_rows(old_groups, plotted, stratified=False)
old_df = pd.read_csv(old_km)
changed = silico_df.merge(old_df[["sample_key", "time", "event"]], on="sample_key", suffixes=("", "_old"))
fig4["removed_runs"] = sorted(set(old_df.sample_key) - set(silico_df.sample_key))
fig4["added_runs"] = sorted(set(silico_df.sample_key) - set(old_df.sample_key))
fig4["changed_runs"] = changed[(changed.time != changed.time_old) | (changed.event != changed.event_old)][
    ["sample_key", "initial_condition", "time_old", "time", "event_old", "event"]].to_dict("records")
report["fig4hi"] = fig4

# =============================================================== Fig. 4j
print("Fig. 4j", flush=True)
report["fig4j"] = {"tests": burden.atagent_vs_atplus_statistics(burden.VITRO_REPLICATE_CSV, "Exp")
                   + burden.atagent_vs_atplus_statistics(burden.SILICO_REPLICATE_CSV, "Sim")}
report["fig4j"]["plotted_means_sd"] = (
    pd.concat([pd.read_csv(burden.VITRO_SUMMARY_CSV).assign(dataset="Exp"),
               pd.read_csv(burden.SILICO_SUMMARY_CSV).assign(dataset="Sim")])
    [["dataset", "strategy", "n_replicates", "mean_normalized_burden", "sd_normalized_burden",
      "mean_treatment_pct", "sd_treatment_pct"]].to_dict("records"))
new_rep = pd.read_csv(burden.SILICO_REPLICATE_CSV)
old_rep = pd.read_csv(BACKUP / "figure_data/Figure_4/cumulative_treatment_burden/cumulative_treatment_average_burden_in_silico_per_replicate.csv")
cj = new_rep.merge(old_rep, on="sample_key", suffixes=("", "_old"))
report["fig4j"]["removed_runs"] = sorted(set(old_rep.sample_key) - set(new_rep.sample_key))
check("Fig. 4j simulated rows = Fig. 4i cohort (AT+ run 23 excluded)", set(new_rep.sample_key) == set(silico_df.sample_key),
      report["fig4j"]["removed_runs"])
report["fig4j"]["changed_runs"] = cj[(cj.n_burden_days != cj.n_burden_days_old)][
    ["sample_key", "cap_day_old", "cap_day", "average_normalized_total_burden_old", "average_normalized_total_burden",
     "days_under_treatment_pct_old", "days_under_treatment_pct"]].to_dict("records")
caps = new_rep.set_index("sample_key")["cap_day"]
cap_ok = all(int(caps[k]) == int(np.ceil(v)) for k, v in zip(silico_df.sample_key, silico_df.time) if k in caps.index)
check("Fig. 4j simulated follow-up caps == regenerated Fig. 4i failure/censoring days", cap_ok)
old_rep_unpaired = []
for col in ("average_normalized_total_burden", "days_under_treatment_pct"):
    for name, rep in (("old", old_rep), ("new", new_rep)):
        a = rep.loc[rep.strategy == "ATAgent2", col].to_numpy(float); b = rep.loc[rep.strategy == "Dens", col].to_numpy(float)
        t = mannwhitneyu(a, b, alternative="two-sided")
        old_rep_unpaired.append({"outcome": col, "table": name, "U_agent": float(t.statistic), "p": float(t.pvalue),
                                 "median_difference": float(np.median(b) - np.median(a))})
report["fig4j"]["unpaired_sim_old_vs_new"] = old_rep_unpaired
vit_new = pd.read_csv(burden.VITRO_REPLICATE_CSV)
vit_old = pd.read_csv(BACKUP / "figure_data/Figure_4/cumulative_treatment_burden/cumulative_treatment_average_burden_in_vitro_per_replicate.csv")
check("in-vitro Fig. 4j table unchanged by regeneration",
      vit_new.drop(columns="results_file").equals(vit_old.drop(columns="results_file")))
for name in ("time_to_failure_by_well_rel_0.5.csv", "time_to_failure_by_well_rel_0.5_no_trunc.csv"):
    new = pd.read_csv(km.DATA_DIR / name); old = pd.read_csv(BACKUP / "figure_data/Figure_4/kaplan_meier" / name)
    check(f"in-vitro {name} unchanged by regeneration", new.drop(columns="filepath").sort_values("sample_key").reset_index(drop=True)
          .equals(old.drop(columns="filepath").sort_values("sample_key").reset_index(drop=True)))

# =============================================================== experimental time origin
print("time origin", flush=True)
day0, labels = [], []
for path in km.find_result_files(km.ROOTS):
    header = km.parse_header_from_path(path)
    if not header or header[2].upper() not in {s.upper() for s in km.INCLUDE_STRATEGIES}:
        continue
    records = km.parse_file(path)
    if records:
        day0.append({"plate": header[0], "sample_key": f"{header[1]}_{header[2]}-{header[1]}", "day0": records[0].day})
day0 = pd.DataFrame(day0)
exp_events = prepare_progression_statistics(exp_counts_all, apply_truncation=True)
fig4h = pd.DataFrame(vitro_rows)
joined = fig4h.merge(exp_events, left_on=["plate", "sample_key"], right_on=["plate", "sample"], how="outer", indicator=True)
both = joined[joined._merge == "both"]
mismatch = both[(both.time != both.ttp) | (both.event != both.progressed.astype(int))]
report["time_origin"] = {
    "n_results_files": len(day0), "first_record_day_counts": day0.day0.value_counts().to_dict(),
    "fig4h_wells": len(fig4h), "hdf5_wells": len(exp_events), "joined_both": len(both),
    "only_fig4h": joined.loc[joined._merge == "left_only", "sample_key"].tolist(),
    "only_hdf5": joined.loc[joined._merge == "right_only", "sample"].tolist(),
    "mismatches": mismatch[["plate", "sample_key", "strategy_x", "time", "event", "ttp", "progressed", "last_observed_day"]].to_dict("records"),
}
check("all in-vitro Results files start at acquisition day 1 (Fig. 4h adds 1 back -> acquisition days)",
      set(day0.day0) == {1}, report["time_origin"]["first_record_day_counts"])
check("Fig. 4h in-vitro times/events == HDF5 acquisition-day TTF (Fig. 2e / Suppl. Fig. 4 pipeline)",
      mismatch.empty, f"{len(mismatch)} mismatching wells of {len(both)}")

# =============================================================== supplementary consumers
print("supplementary consumers", flush=True)
supp = {}
def compare_sim(frame: pd.DataFrame, name: str, strategy: str, sample: str, time: str, event: str) -> None:
    f = frame.rename(columns={strategy: "strategy_c", sample: "sample_c"})
    m = f.merge(canonical, left_on=["strategy_c", "sample_c"], right_on=["strategy", "sample"], how="left")
    ok_time = (m[time] == m["ttp"]) | (~m[event].astype(bool) & ~m["progressed"].astype(bool))
    bad = m[~ok_time | (m[event].astype(bool) != m["progressed"].astype(bool))]
    supp[name] = {"rows": len(m), "unmatched": int(m["ttp"].isna().sum()), "mismatches": bad[["strategy_c", "sample_c", time, event, "ttp", "progressed"]].to_dict("records")}
    check(f"{name}: simulated TTF == canonical endpoint", bad.empty and not m["ttp"].isna().any(), f"rows={len(m)}, mismatches={len(bad)}")

s4 = pd.read_csv(figure_data_dir("Suppl_Figures", "ttf_all_strategies", create=False) / "ttf_all_strategies_invitro_dpd_lv_per_replicate.csv")
compare_sim(s4[s4.source == "Sim"], "Suppl. Fig. 4 (TTF all strategies)", "strategy", "sample", "ttf_days", "failed")
supp["Suppl. Fig. 4 (TTF all strategies)"]["n_by_strategy"] = s4[s4.source == "Sim"].groupby("strategy").size().to_dict()
check("Suppl. Fig. 4 simulated cohort == canonical cohort (165 runs, AT+ run 23 excluded)",
      set(zip(s4.loc[s4.source == "Sim", "strategy"], s4.loc[s4.source == "Sim", "sample"])) == set(zip(canonical.strategy, canonical["sample"])),
      supp["Suppl. Fig. 4 (TTF all strategies)"]["n_by_strategy"])
mat_dir = figure_data_dir("Suppl_Figures", "ttf_tte_matched_analysis", "ttf_tte_matched_simulation_strategies", create=False)
ms = pd.read_csv(mat_dir / "simulation_event_times.csv")
ms = ms[(ms.event == "TTF") & ms.event_status.isin(["observed", "event_not_observed"])]
compare_sim(ms, "Matched-strategy TTF correlations", "strategy", "sample", "time", "observed")
mg = pd.read_csv(figure_data_dir("Suppl_Figures", "escape_margin_over_time", create=False) / "escape_margin_25_examples_per_source_selection.csv")
mg = mg[mg.dataset_type == "simulation"].copy(); mg["failed"] = mg["ttf_days"].notna()
compare_sim(mg, "Suppl. margin trajectories (failure markers)", "strategy", "sample", "ttf_days", "failed")
report["supplementary"] = supp

# =============================================================== Fig. 3h,i
print("Fig. 3h,i", flush=True)
manifest = json.loads((ROOT / "tmp/fig3_claim_audit/manifest.json").read_text())
current = Path(manifest["current_source"])
sha = hashlib.sha256(current.read_bytes()).hexdigest()
check("Fig. 3h,i diagnostic HDF5 unchanged since the density audit (sha256)", sha == manifest["current_sha256"])
fig3 = pd.read_csv(ROOT / "tmp/effect_size_ci_audit_20261110/fig3_paired_contrast_ci.csv")
report["fig3hi"] = fig3[["panel", "dataset", "strategy", "unit", "n_paired", "mean_paired_contrast_day^-1",
                         "ci95_low", "ci95_high", "W_plus", "p_holm_existing"]].to_dict("records")
report["fig3_ttf_dependence"] = "current workflow uses all timepoints (no escape/failure truncation); unaffected by TTF"

(OUT / "statistics_results.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print(f"saved {OUT / 'statistics_results.json'}", flush=True)
print("FAILED CHECKS:", [c["check"] for c in report["checks"] if not c["ok"]], flush=True)
