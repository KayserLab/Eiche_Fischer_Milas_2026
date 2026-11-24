#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from refit_core import (
    ORIGINAL_SEED,
    PARAMETER_ORDER,
    REFERENCE_DIR,
    evaluate_params,
    fit_from_seed,
    load_experiments,
    load_original_saved,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parameter_table(original_saved: list[float], fit_original: list[float], fit_provided: list[float]) -> list[dict]:
    rows = []
    for idx, name in enumerate(PARAMETER_ORDER):
        rows.append(
            {
                "parameter": name,
                "original_saved": float(original_saved[idx]),
                "fit_from_original_seed": float(fit_original[idx]),
                "fit_from_provided_seed": float(fit_provided[idx]),
            }
        )
    return rows


def load_reference_scores() -> dict[str, dict]:
    rows = json.loads((REFERENCE_DIR / "april27_score_matrix.json").read_text(encoding="utf-8"))
    return {row["param_set"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the focused current/new_env LV refit comparison from original_seed and provided_seed.")
    parser.add_argument("--maxiter", type=int, default=1000, help="Max Nelder-Mead iterations for each fit.")
    parser.add_argument("--outdir", default=None, help="Output directory. Default: outputs/refit_compare_<timestamp>/")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else (REPO_ROOT / "outputs" / f"refit_compare_{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments()
    original_saved = load_original_saved()
    provided_seed = list(original_saved)

    print(f"Writing outputs to {outdir}")
    print(f"Starting focused refit comparison with maxiter={args.maxiter}")

    original_saved_objective, original_saved_rows = evaluate_params(original_saved, experiments)

    fit_original = fit_from_seed(ORIGINAL_SEED, experiments, outdir, "fit_from_original_seed", maxiter=args.maxiter)
    fit_provided = fit_from_seed(provided_seed, experiments, outdir, "fit_from_provided_seed", maxiter=args.maxiter)

    fit_original_params = [float(v) for v in fit_original.x.tolist()]
    fit_provided_params = [float(v) for v in fit_provided.x.tolist()]

    fit_original_objective, fit_original_rows = evaluate_params(fit_original_params, experiments)
    fit_provided_objective, fit_provided_rows = evaluate_params(fit_provided_params, experiments)

    reference_scores = load_reference_scores()

    summary_rows = [
        {
            "scenario": "original_saved",
            "objective": float(original_saved_objective),
            "param_l2_to_original_saved": 0.0,
            "april27_reference_objective": float(reference_scores["original_saved"]["new_env"]),
            "objective_delta_vs_april27_reference": float(original_saved_objective - reference_scores["original_saved"]["new_env"]),
        },
        {
            "scenario": "fit_from_original_seed",
            "objective": float(fit_original_objective),
            "param_l2_to_original_saved": float(np.linalg.norm(np.asarray(fit_original_params) - np.asarray(original_saved))),
            "april27_reference_objective": float(reference_scores["new_original_seed_best"]["new_env"]),
            "objective_delta_vs_april27_reference": float(fit_original_objective - reference_scores["new_original_seed_best"]["new_env"]),
        },
        {
            "scenario": "fit_from_provided_seed",
            "objective": float(fit_provided_objective),
            "param_l2_to_original_saved": float(np.linalg.norm(np.asarray(fit_provided_params) - np.asarray(original_saved))),
            "april27_reference_objective": float(reference_scores["new_provided_seed_best"]["new_env"]),
            "objective_delta_vs_april27_reference": float(fit_provided_objective - reference_scores["new_provided_seed_best"]["new_env"]),
        },
    ]

    write_csv(outdir / "parameter_table.csv", build_parameter_table(original_saved, fit_original_params, fit_provided_params))
    write_csv(outdir / "objective_summary.csv", summary_rows)
    write_csv(outdir / "per_experiment_original_saved.csv", original_saved_rows)
    write_csv(outdir / "per_experiment_fit_from_original_seed.csv", fit_original_rows)
    write_csv(outdir / "per_experiment_fit_from_provided_seed.csv", fit_provided_rows)

    payload = {
        "parameter_order": PARAMETER_ORDER,
        "original_seed": ORIGINAL_SEED,
        "provided_seed": provided_seed,
        "original_saved": original_saved,
        "results": {
            "original_saved": {
                "objective": float(original_saved_objective),
            },
            "fit_from_original_seed": {
                "success": bool(fit_original.success),
                "status": int(fit_original.status),
                "message": str(fit_original.message),
                "fun": float(fit_original.fun),
                "nit": int(fit_original.nit),
                "nfev": int(fit_original.nfev),
                "optimized_params": fit_original_params,
            },
            "fit_from_provided_seed": {
                "success": bool(fit_provided.success),
                "status": int(fit_provided.status),
                "message": str(fit_provided.message),
                "fun": float(fit_provided.fun),
                "nit": int(fit_provided.nit),
                "nfev": int(fit_provided.nfev),
                "optimized_params": fit_provided_params,
            },
        },
        "april27_reference_scores": reference_scores,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = [
        "Focused LV refit comparison",
        f"Output dir: {outdir}",
        "",
        "This mini-repo reproduces only the current/new_env refit comparison discussed for the paper.",
        "It does not include old_env or methods-adjusted fitting in the runnable path.",
        "",
        f"original_saved objective: {original_saved_objective:.12g}",
        f"fit_from_original_seed objective: {fit_original_objective:.12g}",
        f"fit_from_provided_seed objective: {fit_provided_objective:.12g}",
        "",
        f"fit_from_original_seed L2 to original_saved: {summary_rows[1]['param_l2_to_original_saved']:.12g}",
        f"fit_from_provided_seed L2 to original_saved: {summary_rows[2]['param_l2_to_original_saved']:.12g}",
        "",
        "Reference April 27 new_env objectives:",
        f"- original_saved: {reference_scores['original_saved']['new_env']:.12g}",
        f"- new_original_seed_best: {reference_scores['new_original_seed_best']['new_env']:.12g}",
        f"- new_provided_seed_best: {reference_scores['new_provided_seed_best']['new_env']:.12g}",
    ]
    (outdir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Finished. Outputs written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
