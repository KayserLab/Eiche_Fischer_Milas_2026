#!/usr/bin/env python3
"""Build the two standalone HTML pages for the policy-distillation results.

Reads the committed analysis outputs in the parent folder and writes
  ../distillation_report.html   full interactive report (both checkpoints)
  ../distillation_figures.html  figures only, agent used for experiment only
Both open directly in a browser (fonts load from Google Fonts when online).

    python analysis/policy_distillation/web/build_pages.py   # from the mini-repo root
Requires numpy and pandas (the mini-repo environment).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent  # analysis/policy_distillation
TEMPLATE = (HERE / "template.html").read_text(encoding="utf-8")


def standalone(fragment):
    """Wrap a page fragment (title, links, style, content, script) as a full document."""
    i = fragment.index('<div class="wrap">')
    head, body = fragment[:i], fragment[i:]
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>img{max-width:100%}[hidden]{display:none!important}body{margin:0}</style>\n'
            + head + '</head>\n<body>\n' + body + '</body>\n</html>\n')


# ---------------------------------------------------------------- page data
out = {"teachers": {}}
for t in ["paper_candidate", "experimental_agent"]:
    m = json.loads((A/t/"metrics.json").read_text())
    u = json.loads((A/t/"uncertainty.json").read_text())
    df = pd.read_csv(A/t/"observations_and_teacher_labels.csv.gz")
    ev = df[df.split.isin(["test","robustness_test"]) & (df.step >= 3)]
    rng = np.random.default_rng(0)
    pts = ev.sample(n=min(1400, len(ev)), random_state=1)
    portrait = [[round(r.burden_t,3), round(r.p_on,3), int(r.previous_action_t)] for r in pts.itertuples()]
    # greedy trajectory at config IC
    g = df[(df.dataset=="primary") & (df.epsilon==0)].sort_values("step")
    greedy = [[int(r.step), round(r.burden_t,3), int(r.executed_action), round(r.p_on,3)] for r in g.itertuples()]
    # a perturbed eps=0.05 episode that progressed, with few flips
    p = df[(df.dataset=="primary") & (df.epsilon==0.05) & (df.episode_terminated)]
    best = None
    for ep, e in p.groupby("episode"):
        e = e.sort_values("step"); nf = int(e.flipped.sum())
        if nf >= 1 and (best is None or (nf, len(e)) < (best[0], best[1])):
            best = (nf, len(e), e)
    pert = [[int(r.step), round(r.burden_t,3), int(r.executed_action), int(r.flipped)] for r in best[2].itertuples()]
    evals = {}
    for split in ["test","robustness_test"]:
        evals[split] = {k: {kk: v["all"][kk] for kk in ["probability_MAE","hard_action_agreement","balanced_accuracy","switching_F1","switch_on_recall","switch_off_recall","teacher_on_fraction","teacher_switch_fraction"]}
                        for k, v in m["evaluation"][split].items()}
    cl = {k: {kk: v[kk] for kk in ["progressed_fraction","mean_steps","mean_schedule_agreement_with_ppo","mean_teacher_agreement_on_own_states"]} for k, v in m["closed_loop"].items()}
    out["teachers"][t] = {
        "sha": m["policy_weights_sha256"][:12], "model": m["model"],
        "data": m["data"], "dups": m["duplicate_episodes_removed"],
        "splits": m["primary_tree_splits"], "burden_splits": m["burden_only_tree_splits"],
        "rules": m["fitted_rules"], "stability": m["split_stability"],
        "eval": evals, "uncertainty": u, "closed_loop": cl,
        "eps_prog": m["ppo_progression_fraction_by_epsilon"],
        "portrait": portrait, "greedy": greedy, "perturbed": pert,
        "depth_sweep": pd.read_csv(A/t/"depth_selection_validation.csv").round(4).to_dict(orient="records"),
    }
    # leaf values for tree diagram: refit not needed; read leaves from tree_rules text
    out["teachers"][t]["tree_text"] = (A/t/"tree_rules.txt").read_text().split("\n\n\n")[0]
np_ = {}
for t in ["paper_candidate_no_padded","experimental_agent_no_padded"]:
    m = json.loads((A/t/"metrics.json").read_text())
    np_[t] = {"agree": m["evaluation"]["test"]["tree_all_features_depth3"]["all"]["hard_action_agreement"],
              "cl": m["closed_loop"]["tree_all_features_depth3"]["progressed_fraction"]}
out["no_padded"] = np_
out["comparison"] = json.loads((A/"teacher_comparison.json").read_text())
ens = pd.read_csv(A/"bootstrap_ensemble/ensemble_summary.csv")
cols = ["agent_code","group","bootstrap_success","root_feature","root_threshold","level1_features","uses_previous_action","hysteresis_start","hysteresis_stay",
        "test_all_features_agreement","test_burden_only_agreement","robust_all_features_agreement","robust_burden_only_agreement","test_all_features_MAE","test_burden_only_MAE","robust_all_features_MAE","robust_burden_only_MAE","robust_majority_agreement"]
out["ensemble"] = json.loads(ens[cols].round(4).to_json(orient="records"))

REPORT = TEMPLATE.replace("__DATA__", json.dumps(out, separators=(",", ":")))

# ------------------------------------- figures-only page, experiment agent only
t = TEMPLATE
style = t[t.index("<style>"):t.index("</style>")+8]
script = t[t.index("<script>")+8:t.index("</script>")]

# ---- data: experiment agent only ----
D = json.loads(json.dumps(out))
D["teachers"] = {"experimental_agent": D["teachers"]["experimental_agent"]}
D["ensemble"] = [r for r in D["ensemble"] if r["agent_code"] != "paper_candidate"]
for k in ("comparison", "no_padded"): D.pop(k, None)

# ---- script patches ----
def sub(old, new):
    global script
    assert old in script, old[:60]
    script = script.replace(old, new)
sub('''  const items = [["experimental_agent", "test"], ["experimental_agent", "robustness_test"], ["paper_candidate", "test"], ["paper_candidate", "robustness_test"]];''',
    '''  const items = [["experimental_agent", "test"], ["experimental_agent", "robustness_test"]];''')
sub('''const box = document.getElementById("figHist"), W = 960, H = 250;''', '''const box = document.getElementById("figHist"), W = 960, H = 160;''')
sub('''`${NAMES[t]} · ${s === "test" ? "test" : "new starts"}`''', '''s === "test" ? "Test trajectories" : "New initial conditions"''')
sub('''const [label, c] = GROUP[gname], big''', '''const [label, c] = GROUP[gname], big''')
sub('''const GROUP = { reference_teacher: ["Main checkpoints", "--g-ref"],''', '''const GROUP = { reference_teacher: ["Agent used for experiment", "--g-ref"],''')
sub('''      const pc = r.agent_code === "paper_candidate";
      txt(G, pc ? cx - 12 : cx + 12, pc ? cy - 10 : cy + 16, NAMES[r.agent_code], { class: "lab-strong halo", "text-anchor": pc ? "end" : "start" });''',
    '''      txt(G, cx + 12, cy + 16, NAMES[r.agent_code], { class: "lab-strong halo", "text-anchor": "start" });''')
def cut(a, b):
    global script
    i = script.index(a); j = script.index(b, i)
    script = script[:i] + script[j:]
cut("/* ---------- headline ---------- */", "/* ---------- pulse cycle ---------- */")
cut("/* ---------- explorer ---------- */", "/* ---------- portrait ---------- */")
cut("function renderCL() {", "/* ---------- ensemble ---------- */")
cut("/* ---------- provenance ---------- */", "/* ---------- wiring ---------- */")
sub('const NAMES = { experimental_agent: "Agent used for experiment", paper_candidate: "Paper candidate" };', 'const NAMES = { experimental_agent: "Agent used for experiment" };')
# drop ensemble table rendering
i = script.index('  // table\n  const groups'); j = script.index('  document.getElementById("tblEns").innerHTML = h + "</tbody>";\n}') + len('  document.getElementById("tblEns").innerHTML = h + "</tbody>";\n')
script = script[:i] + script[j:]
# wiring
i = script.index("/* ---------- wiring ---------- */")
script = script[:i] + '''/* ---------- wiring ---------- */
function segment(id, key, after) {
  document.querySelectorAll(`#${id} button`).forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(`#${id} button`).forEach(o => o.setAttribute("aria-pressed", o === b ? "true" : "false"));
    state[key] = b.dataset[Object.keys(b.dataset)[0]]; renderFid();
  }));
}
segment("splitSeg", "split"); segment("metricSeg", "metric");
function renderAll() { renderCycle(); renderTree(); renderPortrait(); renderFid(); renderHist(); renderPert(); renderEps(); renderEns(); }
renderAll();
const mq = window.matchMedia("(prefers-color-scheme: dark)");
mq.addEventListener && mq.addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
'''
script = script.replace("const D = __DATA__;", "const D = " + json.dumps(D, separators=(",", ":")) + ";")

extra_css = '''<style>
.figs { display: grid; gap: 40px; padding-block: 8px 0; }
.fighead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }
.fighead .no { font-family: var(--f-mono); font-size: var(--step--1); color: var(--accent); letter-spacing: .06em; text-transform: uppercase; }
.fighead h2 { font-size: var(--step-1); }
header.mini { padding-block: 40px 22px; display: grid; gap: 8px; border-bottom: 1px solid var(--rule); margin-bottom: 28px; }
header.mini h1 { font-size: var(--step-3); }
.controls-row { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
.controls-row .lbl { font-size: var(--step--1); color: var(--ink-3); }
</style>'''

body = '''<div class="wrap">
<header class="mini">
  <div class="eyebrow">Policy distillation · agent used for experiment · policy.pth b136b65433d5</div>
  <h1>Distillation Figures</h1>
  <p class="meta-line">Depth-3 CART tree fitted to the agent's P(treatment ON). All panels show the checkpoint in Agent_used_for_experiment.</p>
</header>
<div class="figs">

<figure>
  <div class="fighead"><span class="no">Figure 1</span><h2>Pulse cycle under the agent's control</h2></div>
  <div class="fig" id="figCycle"></div>
  <div class="legend" id="legCycle"></div>
  <figcaption>Burden at each decision point from the configured starting tumour (resistant fraction 5×10⁻⁴). Shaded: treated intervals. Dashed: fitted start and continue thresholds. Red: progression at 3×.</figcaption>
</figure>

<figure>
  <div class="fighead"><span class="no">Figure 2</span><h2>Extracted decision tree</h2></div>
  <div class="fig" id="figTree" style="overflow-x:auto"></div>
  <figcaption>Leaf values are mean agent P(ON); orange leaves mean treat. The grey branch only isolates the first decision of an episode (zero-padded history).</figcaption>
</figure>

<figure>
  <div class="fighead"><span class="no">Figure 3</span><h2>Treatment probability against current burden</h2></div>
  <div class="fig" id="figPortrait"></div>
  <div class="legend" id="legPortrait"></div>
  <figcaption>1,400 sampled held-out decisions, coloured by whether the last interval was treated. Dashed: fitted two-threshold rule.</figcaption>
</figure>

<figure>
  <div class="fighead"><span class="no">Figure 4</span><h2>Fidelity of the stand-in models</h2></div>
  <div class="controls-row">
    <span class="lbl">Held-out set</span>
    <div class="seg" id="splitSeg">
      <button type="button" id="s-test" data-s="test" aria-pressed="true">Test trajectories</button>
      <button type="button" id="s-rob" data-s="robustness_test" aria-pressed="false">New initial conditions</button>
    </div>
    <span class="lbl">Metric</span>
    <div class="seg" id="metricSeg">
      <button type="button" id="m-agree" data-m="hard_action_agreement" aria-pressed="true">Agreement</button>
      <button type="button" id="m-bal" data-m="balanced_accuracy" aria-pressed="false">Balanced accuracy</button>
      <button type="button" id="m-sw" data-m="switching_F1" aria-pressed="false">Switching F1</button>
    </div>
  </div>
  <div class="fig" id="figFid"></div>
  <figcaption>Agreement with the agent's deterministic decisions. Dashed red: always-OFF baseline.</figcaption>
</figure>

<figure>
  <div class="fighead"><span class="no">Figure 5</span><h2>Gain from treatment history</h2></div>
  <div class="fig" id="figHist"></div>
  <figcaption>All-input tree minus burden-only tree, with 95% intervals from 2,000 paired trajectory bootstraps. Left: reduction in P(ON) error. Right: agreement gain in percentage points.</figcaption>
</figure>

<div class="two">
  <figure>
    <div class="fighead"><span class="no">Figure 6a</span><h2>One flipped action</h2></div>
    <div class="fig" id="figPert"></div>
    <div class="legend" id="legPert"></div>
    <figcaption>Unperturbed run versus the same run with a single flipped decision.</figcaption>
  </figure>
  <figure>
    <div class="fighead"><span class="no">Figure 6b</span><h2>Fragility to flipped actions</h2></div>
    <div class="fig" id="figEps"></div>
    <figcaption>Share of the agent's own runs that progress within 60 decisions, by fraction of actions flipped.</figcaption>
  </figure>
</div>

<figure>
  <div class="fighead"><span class="no">Figure 7</span><h2>Thresholds across the agent ensemble</h2></div>
  <div class="fig" id="figEns"></div>
  <div class="legend" id="legEns"></div>
  <figcaption>Fitted start and continue thresholds per agent. Above the diagonal: harder to continue than to start (pulsing). Bootstrap-refit agents live in their own tumour models, so their thresholds are not directly comparable.</figcaption>
</figure>

</div>
</div>'''
head = '<title>Distillation Figures</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n' + t[t.index('<link rel="preconnect"'):t.index("<style>")]
FIGS = head + style + "\n" + extra_css + "\n" + body + "\n<script>" + script + "</script>\n"

(A / "distillation_report.html").write_text(standalone(REPORT), encoding="utf-8")
(A / "distillation_figures.html").write_text(standalone(FIGS), encoding="utf-8")
print("wrote", A / "distillation_report.html")
print("wrote", A / "distillation_figures.html")
