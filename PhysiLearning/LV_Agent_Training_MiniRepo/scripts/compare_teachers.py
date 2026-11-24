#!/usr/bin/env python3
"""Compare two PPO checkpoints by weight hash and by policy output agreement.

Both teachers are queried on the pooled observations recorded by
scripts/distill_policy.py for each of them (i.e. on states visited under either
policy), so agreement is measured on realistic histories, not synthetic probes.

    python scripts/compare_teachers.py \
        --a outputs/policy_distillation/paper_candidate \
        --b outputs/policy_distillation/experimental_agent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_policy as dp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=Path, required=True, help="distill_policy output dir of teacher A")
    parser.add_argument("--b", type=Path, required=True, help="distill_policy output dir of teacher B")
    parser.add_argument("--out", type=Path, default=dp.MINIREPO / "outputs/policy_distillation/teacher_comparison.json")
    args = parser.parse_args()

    meta = {k: json.loads((d / "metrics.json").read_text()) for k, d in (("a", args.a), ("b", args.b))}
    models = {k: dp.load_teacher(dp.MINIREPO.parent / m["model"]) for k, m in meta.items()}
    report = {"teachers": {k: {"label": m["label"], "model": m["model"],
                               "policy_weights_sha256": m["policy_weights_sha256"]}
                           for k, m in meta.items()},
              "identical_policy_weights": meta["a"]["policy_weights_sha256"] == meta["b"]["policy_weights_sha256"],
              "agreement": {}}
    for source, d in (("a", args.a), ("b", args.b)):
        df = pd.read_csv(d / "observations_and_teacher_labels.csv")
        obs = df[dp.FEATURES].drop_duplicates().to_numpy(dtype=np.float32)
        p = {k: dp.teacher_probabilities(m, obs) for k, m in models.items()}
        act = {k: dp.hard_action(v) for k, v in p.items()}
        report["agreement"][f"states_visited_under_{meta[source]['label']}"] = {
            "n_unique_observations": int(len(obs)),
            "hard_action_agreement": float(np.mean(act["a"] == act["b"])),
            "mean_abs_p_on_difference": float(np.mean(np.abs(p["a"] - p["b"]))),
            "on_fraction": {meta[k]["label"]: float(v.mean()) for k, v in act.items()},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
