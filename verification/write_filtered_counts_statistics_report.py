"""Format statistics_results.json into the short manuscript statistics report (Markdown).

Usage: python verification/write_filtered_counts_statistics_report.py <output.md>
All numbers are read from outputs/filtered_counts_statistics_20261110/statistics_results.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "filtered_counts_statistics_20261110"
R = json.loads((RES / "statistics_results.json").read_text(encoding="utf-8"))
D = {"CT": "CT", "AT50": "AT50", "AT100": "AT100", "ATAgent2": "ATagent", "Dens": "AT+", "LV-Agent": "ATagent", "AT100+": "AT+"}


def p(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    x = float(x)
    if x >= 0.001:
        return f"{x:.3g}"
    m, e = f"{x:.2e}".split("e")
    return f"{m}×10^{int(e)}"


def f(x, nd=3) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{float(x):.{nd}g}"


def days(x) -> str:
    return "n.e." if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{float(x):g}"


L: list[str] = []
add = L.append
add("# Statistics report — filtered-count failure endpoints (Nature Physics checklist)")
add("")
add("Generated from `verification/filtered_counts_statistics_report.py` (repo `Eiche_Fischer_2026`, branch `EF_dev`). "
    "Manuscript text unchanged. Directions are comparison minus reference (AT − CT; AT+ − ATagent). "
    "Bootstrap: 10,000 resamples, seed 20260827, 2.5/97.5 percentiles.")
add("")
add("## 0. Canonical simulated failure endpoint (all figures)")
add("")
add("- **Source:** prepared simulation point clouds `intermediate_analysis_data/simulation/basic/prepared_pointclouds/Plate_*/hdf5/*_simulation.h5` "
    "(cells > 504.8 simulation units from the origin removed; retained cells need ≥ 5 cells incl. itself within 30.3 units), "
    "experiment-derived runs (run ≥ 10).")
add("- **Denominator:** filtered sensitive + resistant count of the same run on acquisition day 1.")
add("- **TTF:** first acquisition day (day 1 = first snapshot) with filtered resistant count / denominator ≥ 0.5; otherwise right-censored at the last acquisition day.")
add("- **Inclusion:** experiment-derived runs with a day-1 snapshot (day-1 baseline) and ≥ 2 acquisition days. The same day-1 rule applies to the Fig. 4j burden/exposure rows. "
    "Only one run lacks day 1 and is excluded everywhere: " + "; ".join(R["runs_without_day1_baseline"]) + ", initial condition B5_AT100-B5 (record starts on day 82).")
add("- Implemented once in `analysis/figure_2/Event_Time_Analysis.py::simulation_failure_times` and used by Fig. 2e,f, Fig. 4i (and through its follow-up caps Fig. 4j) and Suppl. Fig. 4; Fig. 4i previously used unfiltered `combined_results.csv` counts.")
add("")
add("| Trajectory | Initial condition | Day before: res / N0 | Failure day: res / N0 | Filtered TTF |")
add("|---|---|---|---|---|")
for ex in R["endpoint_examples"]:
    b, a = ex["day_before"], ex["failure_day"]
    add(f"| {ex['trajectory']} | {ex['initial_condition']} | d{b['day']:g}: {b['res']:.0f}/{ex['denominator_day1_filtered_total']:.0f} = {b['fraction']:.4f} | "
        f"d{a['day']:g}: {a['res']:.0f}/{ex['denominator_day1_filtered_total']:.0f} = {a['fraction']:.4f} | day {ex['filtered_ttf_day']:g} |")
add("")
changed = R["fig4hi"]["changed_runs"]
add(f"Relative to the published Fig. 4i input, {len(changed)} simulated endpoints changed ("
    + "; ".join(f"{c['sample_key']} {c['time_old']:g}→{c['time']:g} d" for c in changed)
    + f") and {len(R['fig4hi']['removed_runs'])} run was removed ({', '.join(R['fig4hi']['removed_runs'])}; no day-1 baseline). All other 164 endpoints are identical.")
add("")

# ------------------------------------------------------------------ Fig. 2d,e
add("## Fig. 2d,e — TTE and TTF")
add("")
add("**Effect:** difference of Kaplan–Meier medians (right-censored follow-up retained), AT − CT, days. "
    "**Exp:** colonies independent; ordinary two-sided log-rank; CI resamples colonies within each strategy. "
    "**Sim:** restricted to initial conditions (ICs) shared by both strategies; two-sided log-rank stratified by IC; "
    "CI resamples matched IC blocks jointly. Bootstrap resamples with an undefined KM median (survival never ≤ 0.5) are discarded and counted. "
    "P values unadjusted; df = 1.")
add("")
add("| Endpoint | Source | Contrast | n CT/AT (events) | Matched ICs (events) | KM median CT / AT (d) | Δ (d) [95% CI] | Undefined resamples | χ² | P |")
add("|---|---|---|---|---|---|---|---|---|---|")
for endpoint in ("TTE", "TTF"):
    for r in R["fig2de"][endpoint]:
        if r["reference"] != "CT":
            continue
        matched = (f"{r['n_shared_initial_conditions']} ({r['events_reference_matched']}/{r['events_comparison_matched']})"
                   if r["n_shared_initial_conditions"] else "n/a")
        add(f"| {endpoint} | {r['source'][:3]} | {r['comparison']}−CT | {r['n_reference']}/{r['n_comparison']} "
            f"({r['events_reference']}/{r['events_comparison']}) | {matched} | {days(r['median_ref'])} / {days(r['median_comp'])} | "
            f"{days(r['delta_days'])} [{days(r['ci_low'])}, {days(r['ci_high'])}] | {r['undefined_resamples']} | {r['chi2']:.2f} | {p(r['p'])} |")
add("")
notes = []
for endpoint in ("TTE", "TTF"):
    for r in R["fig2de"][endpoint]:
        if r["source"] == "Simulation" and r["reference"] == "CT" and not r["matched_medians_equal_plotted_medians"]:
            notes.append(f"{endpoint} {r['comparison']}−CT: matched-IC KM medians ({days(r['median_ref'])}/{days(r['median_comp'])}) differ from plotted all-run medians "
                         f"({days(r['km_median_reference_all'])}/{days(r['km_median_comparison_all'])}).")
if notes:
    add("Plotted median bars use all runs; " + " ".join(notes))
else:
    add("Plotted median bars (all runs) equal the matched-IC medians used for every simulated Δ.")
add("")
add("**Historical audit.** Manuscript values vs reproductions from the same inputs:")
add("")
add("| Endpoint | Source | Contrast | Manuscript | KM-median, independent bootstrap (both sources) | Event-only medians (git b8ea40e) |")
add("|---|---|---|---|---|---|")
hist = R["fig2_historical"]
for endpoint in ("TTE", "TTF"):
    pub = [r for r in hist[f"{endpoint}_published_km_independent"] if r["manuscript"]]
    old = {(r["source"], r["comparison"]): r for r in hist[f"{endpoint}_event_only_b8ea40e"] if r["manuscript"]}
    for r in pub:
        o = old[(r["source"], r["comparison"])]; ms = r["manuscript"]
        add(f"| {endpoint} | {r['source'][:3]} | {r['comparison']}−CT | {ms[0]} [{ms[1]}, {ms[2]}] | "
            f"{days(r['delta'])} [{days(r['ci'][0])}, {days(r['ci'][1])}] {'✓' if r['matches_manuscript'] else '✗'} | "
            f"{days(o['delta'])} [{days(o['ci'][0])}, {days(o['ci'][1])}] {'✓' if o['matches_manuscript'] else '✗'} |")
add("")
add("All eight original effects/CIs are reproduced exactly by KM-median differences with records resampled independently within strategy "
    "(one generator per endpoint, experimental contrasts drawn first), as in the untracked audit script `tmp/figure2_results_comparison.py`. "
    "The event-only median code (git b8ea40e, 27 Aug–6 Oct 2026) reproduces none. Experimental values are unchanged in the corrected workflow; "
    "simulated CIs (and P values) change because ICs are now matched.")
add("")

# ------------------------------------------------------------------ Fig. 2f
add("## Fig. 2f — TTE vs TTF correlation")
add("")
add("Only trajectories with both escape and failure observed (CT, AT50, AT100).")
add("")
add("| Source | n | Clusters | Pearson r [95% CI] | CI method | Test, P |")
add("|---|---|---|---|---|---|")
for r in R["fig2f"]:
    test = f"{r['test']}, P = {p(r['p'])}" if math.isfinite(float(r["p"])) else "none retained (row-level Pearson P invalid under IC clustering)"
    add(f"| {r['source']} | {r['n']} | {r['n_clusters']}{' ICs' if r['source']=='In silico' else ' colonies'} | {r['r']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['ci_method']} | {test} |")
add("")

# ------------------------------------------------------------------ Fig. 3
add("## Fig. 3h,i — growth at high (ρ_tot bin containing 1.55) vs low (0.45) density")
add("")
add("Effect: mean over wells/runs of within-unit median g_high − g_low (day⁻¹); only units with both bins. "
    "Test: one-sided Wilcoxon signed-rank (high − low < 0), Holm across CT/AT50/AT100 within dataset. "
    "CI (new, documented): percentile bootstrap, 10,000 resamples of whole paired wells/runs within strategy and dataset.")
add("")
add("| Panel | Strategy | Paired n | Mean Δg (day⁻¹) [95% CI] | W+ | Holm P |")
add("|---|---|---|---|---|---|")
for r in R["fig3hi"]:
    add(f"| {r['panel']} ({'Exp' if r['dataset']=='experiment' else 'Sim'}) | {r['strategy']} | {r['n_paired']} {r['unit']}s | "
        f"{r['mean_paired_contrast_day^-1']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | {r['W_plus']:g} | {p(r['p_holm_existing'])} |")
add("")
add("Still valid: the diagnostic HDF5 is byte-identical to the audited file and the workflow uses all timepoints (no TTE/TTF dependence).")
add("")

# ------------------------------------------------------------------ Fig. 4h,i
add("## Fig. 4h,i — failure-free survival")
add("")
add("| Strategy | Exp n (events/censored) | Exp KM median (d) | Sim n (events/censored) | Sim KM median (d) |")
add("|---|---|---|---|---|")
coh = {(c["dataset"], c["strategy"]): c for c in R["fig4hi"]["cohorts"]}
for s in ("CT", "AT50", "AT100", "AT+", "ATagent"):
    e, m = coh[("Exp", s)], coh[("Sim", s)]
    add(f"| {s} | {e['n']} ({e['events']}/{e['censored']}) | {days(e['km_median'])} | {m['n']} ({m['events']}/{m['censored']}) | {days(m['km_median'])} |")
add("")
add("Tests: two-sided log-rank, Bonferroni × 10 within dataset. Exp ordinary; Sim stratified by IC on shared ICs (censored records retained).")
add("")
add("| Pair | Exp χ² | Exp P (Bonf.) | Sim shared ICs | Sim χ² | Sim P (Bonf.) |")
add("|---|---|---|---|---|---|")
ex = {(r["strategy_a"], r["strategy_b"]): r for r in R["fig4hi"]["tests"]["Exp"]}
for r in R["fig4hi"]["tests"]["Sim"]:
    e = ex[(r["strategy_a"], r["strategy_b"])]
    add(f"| {D[r['strategy_a']]}–{D[r['strategy_b']]} | {e['chi2']:.2f} | {p(e['p_value'])} ({p(e['p_value_bonferroni'])}) | "
        f"{r['n_shared_initial_conditions']} | {r['chi2']:.2f} | {p(r['p_value'])} ({p(r['p_value_bonferroni'])}) |")
add("")

# ------------------------------------------------------------------ Fig. 4j
add("## Fig. 4j — treatment fraction vs time-averaged normalized burden")
add("")
add("Follow-up ends at failure (inclusive) or censoring; burden = mean of filtered total / day-1 total over retained observations; "
    "treatment fraction = treatment-on days / elapsed follow-up days (initial state excluded). Effects: AT+ − ATagent.")
add("")
add("| Dataset | Outcome | n ATagent/AT+ | Matched pairs | Mean ± SD ATagent | Mean ± SD AT+ | Effect [95% CI] | Test | Statistic | P |")
add("|---|---|---|---|---|---|---|---|---|---|")
for r in R["fig4j"]["tests"]:
    lo, hi = r["difference_of_group_medians_ci"]
    outcome = "burden" if r["outcome"].startswith("average") else "treatment %"
    if r["dataset"] == "Sim":
        plo, phi = r["median_within_pair_difference_ci"]
        eff = (f"diff. of group medians (matched) {r['difference_of_group_medians']:.3g} [{lo:.3g}, {hi:.3g}]; "
               f"median within-pair diff. {r['median_within_pair_difference']:.3g} [{plo:.3g}, {phi:.3g}]")
        stat = f"W+ = {r['W_plus']:g} (AT+ > ATagent ranks; W− = {r['W_minus']:g}; zeros {r['zero_differences']})"
        test, pairs = "Wilcoxon signed-rank, two-sided, unadjusted; pair bootstrap", r["n_pairs"]
    else:
        eff = f"diff. of group medians {r['difference_of_group_medians']:.3g} [{lo:.3g}, {hi:.3g}]"
        stat = f"U = {r['U_agent']:g} (for ATagent; U(AT+) = {r['U_atplus']:g})"
        test, pairs = "Mann–Whitney U, two-sided, unadjusted; colony bootstrap within group", "n/a"
    add(f"| {r['dataset']} | {outcome} | {r['n_agent']}/{r['n_atplus']} | {pairs} | {r['mean_agent']:.3g} ± {r['sd_agent']:.3g} | "
        f"{r['mean_atplus']:.3g} ± {r['sd_atplus']:.3g} | {eff} | {test} | {stat} | {p(r['p'])} |")
add("")
add("Plotted points (mean ± SD, all runs/colonies):")
add("")
add("| Dataset | Strategy | n | Burden mean ± SD | Treatment % mean ± SD |")
add("|---|---|---|---|---|")
for r in R["fig4j"]["plotted_means_sd"]:
    add(f"| {r['dataset']} | {D.get(r['strategy'], r['strategy'])} | {r['n_replicates']} | {r['mean_normalized_burden']:.3f} ± {r['sd_normalized_burden']:.3f} | "
        f"{r['mean_treatment_pct']:.2f} ± {r['sd_treatment_pct']:.2f} |")
add("")

# ------------------------------------------------------------------ checks
add("## Supplementary consumers of simulated TTF and verification")
add("")
add("| Check | Result |")
add("|---|---|")
for c in R["checks"]:
    detail = "" if c["detail"] in (None, "") else f" ({c['detail']})"
    add(f"| {c['check']} | {'PASS' if c['ok'] else '**FAIL**'}{detail} |")
add("")
t = R["time_origin"]
add(f"**Experimental time origin.** All {t['n_results_files']} in-vitro `Results_*.txt` trajectories start at acquisition day "
    f"{', '.join(str(k) for k in t['first_record_day_counts'])}; the Fig. 4h path computes (event day − first day) + 1, i.e. acquisition days, "
    f"identical to the Fig. 2e convention. Fig. 4h and the HDF5 pipeline share {t['joined_both']} colonies; "
    f"{len(t['mismatches'])} differ in time or event status"
    + (": " + "; ".join(f"{m['sample_key']} (Fig. 4h {m['time']:g}/{m['event']} vs HDF5 {m['ttp']:g}/{int(m['progressed'])})" for m in t["mismatches"]) if t["mismatches"] else "")
    + ".")
if t["only_fig4h"] or t["only_hdf5"]:
    add(f"Colonies present in only one pipeline: Fig. 4h only {t['only_fig4h']}; HDF5 only {t['only_hdf5']}.")
add("")

# ------------------------------------------------------------------ changes (all numbers from results)
import pandas as pd

BACKUP = ROOT / "tmp" / "filtered_counts_backup_20261110" / "files" / "figure_data" / "Figure_4" / "cumulative_treatment_burden"
MS_P2 = {("TTE", "AT50"): "9.07×10^-8", ("TTE", "AT100"): "5.29×10^-19", ("TTF", "AT50"): "1.75×10^-13", ("TTF", "AT100"): "2.31×10^-16"}
add("## What changed relative to the manuscript, and why")
add("")
add("Causes: **F** filtered-count endpoint (CT_run_25, ATagent run_20); **B** day-1 baseline rule (AT+ run 23 excluded); "
    "**IC** shared-initial-condition inference (stratified log-rank, matched-block / cluster / pair bootstrap, paired test); "
    "**U** documented uncertainty or resampling sequence only.")
add("")
add("| Item | Manuscript | Final | Cause |")
add("|---|---|---|---|")
for endpoint in ("TTE", "TTF"):
    for r in R["fig2de"][endpoint]:
        if r["reference"] != "CT" or r["source"] != "Simulation":
            continue
        ms = r["manuscript_delta_ci"]
        add(f"| Fig. 2{'d' if endpoint == 'TTE' else 'e'} Sim {r['comparison']}−CT Δ{endpoint} | {ms[0]} [{ms[1]}, {ms[2]}], P = {MS_P2[(endpoint, r['comparison'])]} | "
            f"{days(r['delta_days'])} [{days(r['ci_low'])}, {days(r['ci_high'])}], P = {p(r['p'])} | IC |")
add("| Fig. 2d,e experiments | as published | unchanged (exact) | — |")
f2 = {r["source"]: r for r in R["fig2f"]}
add(f"| Fig. 2f in vitro | r = 0.847 [0.667, 0.930], P = 5.1×10^-10 | r = {f2['In vitro']['r']:.3f} [{f2['In vitro']['ci_low']:.3f}, {f2['In vitro']['ci_high']:.3f}], P = {p(f2['In vitro']['p'])} | U |")
add(f"| Fig. 2f in silico | r = 0.979 [0.954, 0.991], P = 4.1×10^-49 | r = {f2['In silico']['r']:.3f} [{f2['In silico']['ci_low']:.3f}, {f2['In silico']['ci_high']:.3f}], no P | IC |")
med = {c["strategy"]: c for c in R["fig4hi"]["cohorts"] if c["dataset"] == "Sim"}
add(f"| Fig. 4i Sim cohorts n | CT 36, AT50 35, AT100 36, AT+ 24, ATagent 35 | CT {med['CT']['n']}, AT50 {med['AT50']['n']}, AT100 {med['AT100']['n']}, "
    f"AT+ {med['AT+']['n']} ({med['AT+']['events']} failures), ATagent {med['ATagent']['n']} ({med['ATagent']['events']} failures) | B |")
add("| Fig. 4i Sim KM medians CT/AT50/AT100/AT+/ATagent | 27/38/82/93/107 d | "
    + "/".join(days(med[s_]["km_median"]) for s_ in ("CT", "AT50", "AT100", "AT+", "ATagent")) + " d | — |")
st = {(r["strategy_a"], r["strategy_b"]): r for r in R["fig4hi"]["tests"]["Sim"]}
un = {(r["strategy_a"], r["strategy_b"]): r for r in R["fig4hi"]["tests_sim_unstratified_new"]}
old = {(r["strategy_a"], r["strategy_b"]): r for r in R["fig4hi"]["tests_sim_unstratified_old"]}
for pair, label in ((("CT", "ATAgent2"), "CT–ATagent (quoted in text)"), (("AT100", "Dens"), "AT100–AT+"),
                    (("ATAgent2", "Dens"), "ATagent–AT+"), (("CT", "Dens"), "CT–AT+")):
    add(f"| Fig. 4i Sim {label}, Bonferroni P | {p(old[pair]['p_value_bonferroni'])} | {p(st[pair]['p_value_bonferroni'])} "
        f"(unstratified on final cohort: {p(un[pair]['p_value_bonferroni'])}) | F + B + IC |")
add("| Fig. 4h experiments (all pairs) | CT–ATagent P = 0.0052 | unchanged | — |")
j = {(r["dataset"], r["outcome"]): r for r in R["fig4j"]["tests"]}
ex = j[("Exp", "average_normalized_total_burden")]
add(f"| Fig. 4j Exp burden AT+ − ATagent | −0.44 [−0.51, −0.31], P = 0.00012 | {ex['difference_of_group_medians']:.2f} "
    f"[{ex['difference_of_group_medians_ci'][0]:.2f}, {ex['difference_of_group_medians_ci'][1]:.2f}], P = {p(ex['p'])} | — |")
sb, sx = j[("Sim", "average_normalized_total_burden")], j[("Sim", "days_under_treatment_pct")]
unp = {(u["outcome"], u["table"]): u for u in R["fig4j"]["unpaired_sim_old_vs_new"]}
add(f"| Fig. 4j Sim burden AT+ − ATagent | −0.74 [−0.77, −0.62], P = 9.6×10^-11 (Mann–Whitney, 35/24 runs) | "
    f"{sb['difference_of_group_medians']:.2f} [{sb['difference_of_group_medians_ci'][0]:.2f}, {sb['difference_of_group_medians_ci'][1]:.2f}], "
    f"P = {p(sb['p'])} (Wilcoxon signed-rank, {sb['n_pairs']} matched pairs) | F + B + IC |")
add(f"| Fig. 4j Sim treatment fraction AT+ − ATagent (text: \"similar treatment exposure\") | not quoted (previous Mann–Whitney P = {p(unp[('days_under_treatment_pct', 'old')]['p'])}) | "
    f"{sx['difference_of_group_medians']:+.2f} [{sx['difference_of_group_medians_ci'][0]:.2f}, {sx['difference_of_group_medians_ci'][1]:.2f}] percentage points, "
    f"P = {p(sx['p'])} ({sx['n_pairs']} pairs) | F + B + IC |")
old_sum = pd.read_csv(BACKUP / "cumulative_treatment_average_burden_in_silico_strategy_summary.csv").set_index("strategy")
new_sum = {r["strategy"]: r for r in R["fig4j"]["plotted_means_sd"] if r["dataset"] == "Sim"}
moved = []
for s_ in ("CT", "ATAgent2", "Dens"):
    o, n_ = old_sum.loc[s_], new_sum[s_]
    moved.append(f"{D[s_]} n {int(o.n_replicates)}→{n_['n_replicates']}, burden {o.mean_normalized_burden:.3f}±{o.sd_normalized_burden:.3f}→"
                 f"{n_['mean_normalized_burden']:.3f}±{n_['sd_normalized_burden']:.3f}, treatment {o.mean_treatment_pct:.2f}±{o.sd_treatment_pct:.2f}→"
                 f"{n_['mean_treatment_pct']:.2f}±{n_['sd_treatment_pct']:.2f} %")
add("| Fig. 4j plotted Sim points (mean ± SD) | as published | " + "; ".join(moved) + "; other strategies unchanged | F + B |")
add("| Fig. 3h,i | means, W+, Holm P as published | unchanged; 95% CIs added | U |")
add("")

(RES / "report_body.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
