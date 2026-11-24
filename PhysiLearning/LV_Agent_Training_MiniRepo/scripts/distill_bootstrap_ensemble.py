#!/usr/bin/env python3
"""Repeat the CART distillation for every agent of the bootstrap ensemble.

Each agent is rolled out in the LV environment it was trained on (its own
stage-2 config from the ensemble's `cfg/` folder) and distilled with
scripts/distill_policy.py. Agents keep their training group label
(bootstrap_lv / methods_repeat / methods_control); groups are never pooled
silently. A per-agent summary table is written next to the per-agent outputs.

    python scripts/distill_bootstrap_ensemble.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_policy as dp  # noqa: E402

DEFAULT_RUN = (dp.MINIREPO.parent / "Bootstrapping/outputs/bootstrap_n20_methods_seed_maxiter1000"
               / "agent_training/agt_20260605_102549")


def distil_agent(job: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    argv = ["--model", job["model"], "--config", job["config"], "--label", job["agent_code"],
            "--outdir", job["outdir"], *job["extra"]]
    args = dp.parse_args(argv)
    if not (Path(job["outdir"]) / "metrics.json").exists() or job["overwrite"]:
        dp.run(args)
    return job


def summarise(metrics: dict, group: str, extra: dict) -> dict:
    d = metrics["primary_depth"]
    ev = metrics["evaluation"]
    splits = metrics["primary_tree_splits"]
    stab = metrics["split_stability"]["all_features"]
    row = {
        **extra,
        "group": group,
        "label": metrics["label"],
        "policy_weights_sha256": metrics["policy_weights_sha256"][:12],
        "root_feature": splits[0]["feature"] if splits else None,
        "root_threshold": splits[0]["threshold"] if splits else None,
        "level1_features": "|".join(s["feature"] for s in splits if s["depth"] == 1),
        "tree_features": "|".join(sorted({s["feature"] for s in splits})),
        "uses_previous_action": any(s["feature"].startswith("previous_action") for s in splits),
        "root_feature_stability": stab["root_feature_frequency"].get(splits[0]["feature"], 0.0)
        if splits else None,
        "hysteresis_start": metrics["fitted_rules"]["hysteresis"]["start"],
        "hysteresis_stay": metrics["fitted_rules"]["hysteresis"]["stay"],
        "teacher_on_fraction_test": ev["test"][f"tree_all_features_depth{d}"]["all"]["teacher_on_fraction"],
    }
    for split in ("test", "robustness_test"):
        tag = "test" if split == "test" else "robust"
        for student in ("all_features", "burden_only", "action_only"):
            a = ev[split][f"tree_{student}_depth{d}"]["all"]
            row[f"{tag}_{student}_MAE"] = a["probability_MAE"]
            row[f"{tag}_{student}_agreement"] = a["hard_action_agreement"]
            row[f"{tag}_{student}_switchF1"] = a["switching_F1"]
        row[f"{tag}_hysteresis_agreement"] = ev[split]["rule_hysteresis_fitted"]["all"]["hard_action_agreement"]
        row[f"{tag}_majority_agreement"] = ev[split]["constant_majority_action"]["all"]["hard_action_agreement"]
    cl = metrics["closed_loop"]
    for name in ("ppo", f"tree_all_features_depth{d}", f"tree_burden_only_depth{d}", "rule_hysteresis_fitted"):
        short = name.replace(f"_depth{d}", "")
        row[f"closed_loop_{short}_progressed"] = cl[name]["progressed_fraction"]
        row[f"closed_loop_{short}_steps"] = cl[name]["mean_steps"]
        row[f"closed_loop_{short}_schedule_agreement"] = cl[name]["mean_schedule_agreement_with_ppo"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--outdir", type=Path,
                        default=dp.MINIREPO / "outputs/policy_distillation/bootstrap_ensemble")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reference", type=Path, nargs="*", default=[
        dp.MINIREPO / "outputs/policy_distillation/paper_candidate",
        dp.MINIREPO / "outputs/policy_distillation/experimental_agent"],
        help="distill_policy output dirs of the main teachers, added as labelled rows.")
    args, extra = parser.parse_known_args()

    manifest = pd.read_csv(args.run_dir / "agent_training_manifest.csv")
    jobs = []
    for row in manifest.itertuples(index=False):
        code = row.agent_code
        jobs.append({
            "agent_code": code, "group": row.group,
            "bootstrap_success": row.bootstrap_success,
            "model": str(args.run_dir / "models" / code / f"{code}_s2_best_reward.zip"),
            "config": str(args.run_dir / "cfg" / f"{code}_s2.yaml"),
            "outdir": str(args.outdir / code), "extra": extra, "overwrite": args.overwrite,
        })
    with Pool(args.workers) as pool:
        for job in pool.imap_unordered(distil_agent, jobs):
            print("done", job["agent_code"], flush=True)

    rows = []
    for ref in args.reference:
        if (ref / "metrics.json").exists():
            rows.append(summarise(json.loads((ref / "metrics.json").read_text()), "reference_teacher",
                                  {"agent_code": ref.name, "bootstrap_success": None}))
    for job in jobs:
        metrics = json.loads((Path(job["outdir"]) / "metrics.json").read_text())
        rows.append(summarise(metrics, job["group"], {"agent_code": job["agent_code"],
                                                      "bootstrap_success": job["bootstrap_success"]}))
    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "ensemble_summary.csv", index=False)

    numeric = [c for c in table.columns if table[c].dtype.kind in "fb" and c != "bootstrap_success"]
    group_summary = table.groupby("group")[numeric].agg(["mean", "std", "min", "max"])
    group_summary.to_csv(args.outdir / "ensemble_group_summary.csv")
    feature_counts = {g: {"root_feature": t["root_feature"].value_counts().to_dict(),
                          "uses_previous_action": float(t["uses_previous_action"].mean()),
                          "n": int(len(t))}
                      for g, t in table.groupby("group")}
    (args.outdir / "ensemble_feature_selection.json").write_text(json.dumps(feature_counts, indent=2))
    print(json.dumps(feature_counts, indent=2))
    print(f"Saved to {args.outdir}")


if __name__ == "__main__":
    main()
