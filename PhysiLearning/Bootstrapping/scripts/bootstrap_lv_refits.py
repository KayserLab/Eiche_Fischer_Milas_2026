#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSILEARNING_ROOT = REPO_ROOT.parent
FITTING_REPO = PHYSILEARNING_ROOT / "LV_Fitting_MiniRepo"
AGENT_PARAMS_PATH = PHYSILEARNING_ROOT / "LV_Agent_Training_MiniRepo" / "reference" / "fitted_lv_params.json"
DEFAULT_REFIT_SUMMARY = FITTING_REPO / "outputs" / "refit_compare_full" / "summary.json"

for path in (FITTING_REPO / "scripts", FITTING_REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refit_core import (  # noqa: E402
    ORIGINAL_SEED,
    PARAMETER_ORDER,
    evaluate_params,
    fit_from_seed,
    load_experiments,
    load_original_saved,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_by_treatment(experiments: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for record in experiments:
        grouped[record.treatment_type].append(record)
    return dict(sorted(grouped.items()))


def sample_experiments(experiments: list, rng: np.random.Generator, *, stratified: bool) -> tuple[list, list[dict]]:
    selected = []
    rows = []
    if stratified:
        for treatment_type, records in group_by_treatment(experiments).items():
            indices = rng.integers(0, len(records), size=len(records))
            for within_order, idx in enumerate(indices.tolist()):
                record = records[idx]
                selected.append(record)
                rows.append(
                    {
                        "draw_order": len(rows),
                        "treatment_type": treatment_type,
                        "within_treatment_draw_order": within_order,
                        "source_index_within_treatment": idx,
                        "experiment": record.rel_path,
                    }
                )
    else:
        indices = rng.integers(0, len(experiments), size=len(experiments))
        for draw_order, idx in enumerate(indices.tolist()):
            record = experiments[idx]
            selected.append(record)
            rows.append(
                {
                    "draw_order": draw_order,
                    "treatment_type": record.treatment_type,
                    "within_treatment_draw_order": "",
                    "source_index_within_treatment": "",
                    "experiment": record.rel_path,
                }
            )
    return selected, rows


def load_refit_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing full refit summary: {path}. Run the fitting mini-repo full refit first, "
            "or use --seed-mode provided_saved/original_seed."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_agent_training_frozen_seed(path: Path = AGENT_PARAMS_PATH) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing agent-training LV parameter reference: {path}")
    params = json.loads(path.read_text(encoding="utf-8"))
    return [
        float(params["growth_rate_wt"]),
        float(params["growth_rate_mut"]),
        float(params["competition_wt"]),
        float(params["death_rate_wt"]),
        float(params["treat_death_rate_wt"]),
        float(params["ramp_time_up"]),
        float(params["ramp_time_down"]),
        float(params["carrying_capacity"]),
        float(params["on_treat_threshold"]),
        float(params["off_treat_threshold"]),
        float(params["min_death_during_treat"]),
    ]


def choose_seed(seed_mode: str, refit_summary_path: Path) -> list[float]:
    if seed_mode == "provided_saved":
        return load_original_saved()
    if seed_mode == "original_seed":
        return [float(v) for v in ORIGINAL_SEED]
    if seed_mode == "agent_training_frozen":
        return load_agent_training_frozen_seed()

    summary = load_refit_summary(refit_summary_path)
    if seed_mode == "full_provided_best":
        return [float(v) for v in summary["results"]["fit_from_provided_seed"]["optimized_params"]]
    if seed_mode == "full_original_best":
        return [float(v) for v in summary["results"]["fit_from_original_seed"]["optimized_params"]]
    raise ValueError(f"Unknown seed mode: {seed_mode}")


def summarize_parameters(rows: list[dict], ci: float) -> list[dict]:
    if not rows:
        return []
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    out = []
    for name in PARAMETER_ORDER:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        out.append(
            {
                "parameter": name,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q250": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q750": float(np.quantile(values, 0.75)),
                "q975": float(np.quantile(values, 0.975)),
                f"ci{int(ci * 100)}_lower": float(np.quantile(values, lo_q)),
                f"ci{int(ci * 100)}_upper": float(np.quantile(values, hi_q)),
                "max": float(np.max(values)),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified replicate-level bootstrap LV refits.")
    parser.add_argument("--n-bootstrap", type=int, default=20, help="Number of bootstrap refits.")
    parser.add_argument("--maxiter", type=int, default=1000, help="Nelder-Mead maxiter per bootstrap refit.")
    parser.add_argument("--random-seed", type=int, default=20260602, help="Random seed for bootstrap sampling.")
    parser.add_argument(
        "--seed-mode",
        choices=["full_provided_best", "full_original_best", "provided_saved", "original_seed", "agent_training_frozen"],
        default="full_provided_best",
        help="Optimizer start vector for each bootstrap refit.",
    )
    parser.add_argument("--refit-summary", default=str(DEFAULT_REFIT_SUMMARY), help="Full refit summary JSON for full_* seed modes.")
    parser.add_argument("--unstratified", action="store_true", help="Sample all experiments without treatment strata.")
    parser.add_argument("--ci", type=float, default=0.95, help="Central interval width for parameter ranges.")
    parser.add_argument("--outdir", default=None, help="Output directory. Default: outputs/lv_bootstrap_<timestamp>.")
    parser.add_argument("--start-index", type=int, default=0, help="First bootstrap index label.")
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be >= 1")
    if not (0.0 < args.ci < 1.0):
        raise ValueError("--ci must be between 0 and 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else REPO_ROOT / "outputs" / f"lv_bootstrap_{timestamp}"
    if not outdir.is_absolute():
        outdir = (REPO_ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # The fitting runtime writes relative Training/ paths internally.
    os.chdir(FITTING_REPO)

    experiments = load_experiments()
    treatment_counts = {key: len(value) for key, value in group_by_treatment(experiments).items()}
    seed = choose_seed(args.seed_mode, Path(args.refit_summary))
    rng = np.random.default_rng(args.random_seed)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "bootstrap_unit": "experiment_replicate",
        "sampling": "unstratified" if args.unstratified else "stratified_by_treatment",
        "treatment_group_counts": treatment_counts,
        "n_bootstrap": int(args.n_bootstrap),
        "maxiter": int(args.maxiter),
        "random_seed": int(args.random_seed),
        "seed_mode": args.seed_mode,
        "seed_vector": seed,
        "parameter_order": PARAMETER_ORDER,
        "fitting_repo": str(FITTING_REPO),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    result_rows = []
    sampled_rows = []
    for offset in range(args.n_bootstrap):
        bootstrap_index = args.start_index + offset
        label = f"bootstrap_{bootstrap_index:04d}"
        run_dir = outdir / "runs" / label
        sampled_experiments, sampled = sample_experiments(experiments, rng, stratified=not args.unstratified)
        for row in sampled:
            row["bootstrap_index"] = bootstrap_index
            sampled_rows.append(row)
        write_csv(run_dir / "sampled_experiments.csv", sampled)

        print(f"[{label}] fitting {len(sampled_experiments)} sampled experiments with maxiter={args.maxiter}")
        fit = fit_from_seed(seed, sampled_experiments, run_dir, label, maxiter=args.maxiter)
        params = [float(v) for v in fit.x.tolist()]
        full_objective, _ = evaluate_params(params, experiments)

        row = {
            "bootstrap_index": bootstrap_index,
            "label": label,
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "bootstrap_objective": float(fit.fun),
            "full_data_objective": float(full_objective),
            "nit": int(fit.nit),
            "nfev": int(fit.nfev),
        }
        row.update({name: float(params[idx]) for idx, name in enumerate(PARAMETER_ORDER)})
        result_rows.append(row)

        write_csv(outdir / "bootstrap_results.csv", result_rows)
        write_csv(outdir / "bootstrap_sampled_experiments.csv", sampled_rows)
        write_csv(outdir / "parameter_intervals.csv", summarize_parameters(result_rows, args.ci))

    report = [
        "LV bootstrap refit summary",
        f"Output dir: {outdir}",
        "Bootstrap unit: experiment replicate",
        f"Sampling: {'unstratified' if args.unstratified else 'stratified by treatment'}",
        f"n_bootstrap: {args.n_bootstrap}",
        f"maxiter: {args.maxiter}",
        f"seed_mode: {args.seed_mode}",
        "",
        "Files:",
        "- metadata.json",
        "- bootstrap_results.csv",
        "- bootstrap_sampled_experiments.csv",
        "- parameter_intervals.csv",
        "- runs/bootstrap_*/progress/*.json",
    ]
    (outdir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Finished bootstrap run. Outputs written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
