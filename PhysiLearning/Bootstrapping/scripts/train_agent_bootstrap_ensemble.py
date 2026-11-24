#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSILEARNING_ROOT = REPO_ROOT.parent
AGENT_REPO = PHYSILEARNING_ROOT / "LV_Agent_Training_MiniRepo"
AGENT_SRC = AGENT_REPO / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from physilearning.train import Trainer  # noqa: E402

METHODS_BOOTSTRAP_DIR = REPO_ROOT / "outputs" / "bootstrap_n20_methods_seed_maxiter1000"
DEFAULT_BOOTSTRAP_RESULTS = METHODS_BOOTSTRAP_DIR / "bootstrap_results.csv"
METHODS_PARAMS_PATH = PHYSILEARNING_ROOT / "LV_Agent_Training_MiniRepo" / "reference" / "fitted_lv_params.json"
STAGE1_CONFIG = AGENT_REPO / "configs" / "2025_10_lv_real_mela_1.yaml"
STAGE2_CONFIG = AGENT_REPO / "configs" / "2025_10_lv_real_mela_1_load.yaml"

PARAMETER_COLUMNS = [
    "growth_rate_sus",
    "growth_rate_res",
    "competition",
    "random_death",
    "death_treat_sus",
    "ramp_time_up",
    "ramp_time_down",
    "carrying_capacity",
    "on_treat_threshold",
    "off_treat_threshold",
    "min_treat",
]


def machine_resources() -> dict:
    resources = {
        "cpu_count_logical": os.cpu_count(),
        "ram_total_gb": None,
        "ram_available_gb": None,
        "torch_version": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "cuda_devices": [],
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        resources["ram_total_gb"] = round(vm.total / 1024**3, 2)
        resources["ram_available_gb"] = round(vm.available / 1024**3, 2)
    except Exception:
        pass
    try:
        import torch

        resources["torch_version"] = torch.__version__
        resources["cuda_available"] = bool(torch.cuda.is_available())
        resources["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            resources["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return resources


@dataclass
class AgentJob:
    agent_order: int
    agent_id: str
    group: str
    seed: int
    params: dict[str, float]
    bootstrap_index: str = ""
    bootstrap_success: str = ""
    bootstrap_status: str = ""
    bootstrap_objective: str = ""
    full_data_objective: str = ""


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_methods_params() -> dict[str, float]:
    params = json.loads(METHODS_PARAMS_PATH.read_text(encoding="utf-8"))
    return {
        "growth_rate_sus": float(params["growth_rate_wt"]),
        "growth_rate_res": float(params["growth_rate_mut"]),
        "competition": float(params["competition_wt"]),
        "random_death": float(params["death_rate_wt"]),
        "death_treat_sus": float(params["treat_death_rate_wt"]),
        "ramp_time_up": float(params["ramp_time_up"]),
        "ramp_time_down": float(params["ramp_time_down"]),
        "carrying_capacity": float(params["carrying_capacity"]),
        "on_treat_threshold": float(params["on_treat_threshold"]),
        "off_treat_threshold": float(params["off_treat_threshold"]),
        "min_treat": float(params["min_death_during_treat"]),
    }


def bootstrap_params(row: dict) -> dict[str, float]:
    missing = [name for name in PARAMETER_COLUMNS if name not in row or row[name] == ""]
    if missing:
        raise KeyError(f"Bootstrap row missing parameter columns: {missing}")
    return {name: float(row[name]) for name in PARAMETER_COLUMNS}


def apply_lv_params(config: dict, params: dict[str, float]) -> dict:
    cfg = dict(config)
    env = cfg["env"]
    lv = env["LvEnv"]

    env["growth_rate_wt"] = float(params["growth_rate_sus"])
    env["growth_rate_mut"] = float(params["growth_rate_res"])
    env["death_rate_wt"] = float(params["random_death"])
    env["death_rate_mut"] = float(params["random_death"])
    env["treat_death_rate_wt"] = float(params["death_treat_sus"])
    env["treat_death_rate_mut"] = 0.0

    lv["competition_wt"] = float(params["competition"])
    lv["competition_mut"] = 1.0
    lv["ramp_time_up"] = float(params["ramp_time_up"])
    lv["ramp_time_down"] = float(params["ramp_time_down"])
    lv["carrying_capacity"] = float(params["carrying_capacity"])
    lv["on_treat_threshold"] = float(params["on_treat_threshold"])
    lv["off_treat_threshold"] = float(params["off_treat_threshold"])
    lv["min_death_during_treat"] = float(params["min_treat"])
    return cfg


def build_config(base_config: Path, params: dict[str, float], save_prefix: str, timesteps: int, seed: int, training_root: Path, load_model: Path | None) -> dict:
    cfg = apply_lv_params(load_yaml(base_config), params)
    model_cfg = cfg["learning"]["model"]
    model_cfg["total_timesteps"] = int(timesteps)
    model_cfg["model_save_prefix"] = save_prefix
    model_cfg["seed"] = int(seed)
    model_cfg["training_root"] = str(training_root)
    model_cfg.setdefault("model_kwargs", {})
    model_cfg["model_kwargs"]["seed"] = int(seed)
    if load_model is None:
        model_cfg["load"]["enable_loading"] = 0
        model_cfg["load"]["last_model"] = 0
    else:
        model_cfg["load"]["enable_loading"] = 1
        model_cfg["load"]["last_model"] = 0
        model_cfg["load"]["saved_model_name"] = str(load_model)
    return cfg


def clean_monitor(training_root: Path) -> None:
    logs_dir = training_root / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    empty_monitor = AGENT_REPO / "Training" / "Logs" / "empty_monitor.csv"
    monitor_path = logs_dir / "monitor.csv"
    if monitor_path.exists():
        monitor_path.unlink()
    if empty_monitor.exists():
        shutil.copy2(empty_monitor, monitor_path)


def expected_model(prefix: str, suffix: str, training_root: Path) -> Path:
    return training_root / "SavedModels" / f"{prefix}_{suffix}.zip"


def set_process_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def configure_worker_threads(torch_num_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(torch_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_num_threads)
    try:
        import torch

        torch.set_num_threads(torch_num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass


def run_training(config_path: Path, seed: int, training_root: Path, torch_num_threads: int) -> None:
    os.chdir(AGENT_REPO)
    configure_worker_threads(torch_num_threads)
    set_process_seed(seed)
    clean_monitor(training_root)
    trainer = Trainer(str(config_path))
    trainer.learn()


def copy_model_if_present(source: Path, destination_dir: Path) -> str:
    if not source.exists():
        return ""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return str(destination)


def train_agent_worker(payload: dict) -> dict:
    result = {
        "agent_id": payload["agent_id"],
        "status": "training_stage1",
        "error": "",
        "traceback": "",
    }
    status_path = Path(payload["worker_status_path"])
    write_json(status_path, result)
    try:
        training_root = Path(payload["training_root"])
        stage1_config = Path(payload["stage1_config"])
        stage2_config = Path(payload["stage2_config"])
        model_dir = Path(payload["model_dir"])
        seed = int(payload["seed"])
        torch_num_threads = int(payload["torch_num_threads"])
        stage1_prefix = payload["stage1_prefix"]
        stage2_prefix = payload["stage2_prefix"]

        run_training(stage1_config, seed, training_root, torch_num_threads)
        result["copied_stage1_best_model"] = copy_model_if_present(expected_model(stage1_prefix, "best_reward", training_root), model_dir)
        result["copied_stage1_final_model"] = copy_model_if_present(expected_model(stage1_prefix, "final", training_root), model_dir)

        result["status"] = "training_stage2"
        write_json(status_path, result)
        run_training(stage2_config, seed, training_root, torch_num_threads)
        result["copied_stage2_best_model"] = copy_model_if_present(expected_model(stage2_prefix, "best_reward", training_root), model_dir)
        result["copied_stage2_final_model"] = copy_model_if_present(expected_model(stage2_prefix, "final", training_root), model_dir)
        result["status"] = "finished"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(status_path, result)
    return result


def build_jobs(rows: list[dict], n_bootstrap_agents: int, n_methods_repeats: int, n_control_agents: int, bootstrap_seed_base: int, methods_seed_base: int, control_seed_base: int) -> list[AgentJob]:
    jobs: list[AgentJob] = []
    selected_rows = rows[:n_bootstrap_agents]
    for row in selected_rows:
        bootstrap_index = int(row["bootstrap_index"])
        jobs.append(
            AgentJob(
                agent_order=len(jobs),
                agent_id=f"bootstrap_lv_{bootstrap_index:04d}",
                group="bootstrap_lv",
                seed=int(bootstrap_seed_base + bootstrap_index),
                params=bootstrap_params(row),
                bootstrap_index=str(bootstrap_index),
                bootstrap_success=str(row.get("success", "")),
                bootstrap_status=str(row.get("status", "")),
                bootstrap_objective=str(row.get("bootstrap_objective", "")),
                full_data_objective=str(row.get("full_data_objective", "")),
            )
        )

    methods_params = load_methods_params()
    for repeat_index in range(n_methods_repeats):
        jobs.append(
            AgentJob(
                agent_order=len(jobs),
                agent_id=f"methods_repeat_{repeat_index:04d}",
                group="methods_repeat",
                seed=int(methods_seed_base + repeat_index),
                params=methods_params,
            )
        )

    for control_index in range(n_control_agents):
        jobs.append(
            AgentJob(
                agent_order=len(jobs),
                agent_id=f"methods_control_{control_index:04d}",
                group="methods_control",
                seed=int(control_seed_base + control_index),
                params=methods_params,
            )
        )
    return jobs


def agent_code(job: AgentJob) -> str:
    if job.group == "bootstrap_lv":
        return f"b{int(job.bootstrap_index):04d}"
    if job.group == "methods_repeat":
        return f"mr{job.agent_id.rsplit('_', 1)[-1]}"
    if job.group == "methods_control":
        return f"mc{job.agent_id.rsplit('_', 1)[-1]}"
    return f"a{job.agent_order:04d}"


def manifest_row(job: AgentJob, outdir: Path, run_label: str, stage1_config: Path, stage2_config: Path, training_root: Path, status: str, error: str = "") -> dict:
    code = agent_code(job)
    stage1_prefix = f"{code}_s1"
    stage2_prefix = f"{code}_s2"
    row = {
        "agent_order": job.agent_order,
        "agent_id": job.agent_id,
        "agent_code": code,
        "group": job.group,
        "seed": job.seed,
        "bootstrap_index": job.bootstrap_index,
        "bootstrap_success": job.bootstrap_success,
        "bootstrap_status": job.bootstrap_status,
        "bootstrap_objective": job.bootstrap_objective,
        "full_data_objective": job.full_data_objective,
        "status": status,
        "error": error,
        "stage1_config": str(stage1_config),
        "stage2_config": str(stage2_config),
        "training_root": str(training_root),
        "training_root_stage1_best_model": str(expected_model(stage1_prefix, "best_reward", training_root)),
        "training_root_stage1_final_model": str(expected_model(stage1_prefix, "final", training_root)),
        "training_root_stage2_best_model": str(expected_model(stage2_prefix, "best_reward", training_root)),
        "training_root_stage2_final_model": str(expected_model(stage2_prefix, "final", training_root)),
        "copied_models_dir": str(outdir / "models" / code),
        "worker_status_json": str(outdir / "worker_status" / f"{code}.json"),
    }
    for name in PARAMETER_COLUMNS:
        row[name] = float(job.params[name])
    return row


def write_run_plan(outdir: Path, args: argparse.Namespace, jobs: list[AgentJob]) -> None:
    rows_by_group = {}
    for job in jobs:
        rows_by_group[job.group] = rows_by_group.get(job.group, 0) + 1
    plan = [
        "Agent bootstrap training plan",
        "==============================",
        "",
        "Goal",
        "Train a sensitivity ensemble of PPO agents using the completed methods-seed LV bootstrap.",
        "The ensemble separates LV-parameter sensitivity from basic RL-training stochasticity.",
        "",
        "Design",
        f"- bootstrap_lv agents: {rows_by_group.get('bootstrap_lv', 0)}; one agent per bootstrapped LV parameter vector.",
        f"- methods_repeat agents: {rows_by_group.get('methods_repeat', 0)}; repeated seeds on the methods-section LV parameter vector.",
        f"- methods_control agents: {rows_by_group.get('methods_control', 0)}; current-config control agent(s) on the methods-section LV parameter vector.",
        "- All bootstrap LV fits are used, including optimizer runs that reached maxiter, because the kept LV bootstrap result set contains all 20 rows.",
        "",
        "Training procedure",
        "- Stage 1 uses the canonical agent config 2025_10_lv_real_mela_1.yaml.",
        "- Stage 1 reward_shaping is dont_treat.",
        "- Stage 2 uses the canonical agent config 2025_10_lv_real_mela_1_load.yaml.",
        "- Stage 2 reward_shaping is ttp and loads the matching stage-1 best_reward model.",
        f"- Stage 1 timesteps: {args.stage1_timesteps}.",
        f"- Stage 2 timesteps: {args.stage2_timesteps}.",
        "- PPO architecture, observation setup, treatment schedule, and all non-LV hyperparameters are inherited from the mini-repo configs.",
        "",
        "Observation/noise decision",
        "- The current reference config does not expose sensitive/resistant cells separately; it exposes total burden plus previous action with four-frame stacking.",
        "- The current reference config does not add Gaussian observation noise.",
        "- We therefore keep the current config unchanged and plan to revise the methods text rather than introduce a new training regime.",
        "",
        "Traceability",
        f"- Bootstrap results source: {args.bootstrap_results}.",
        f"- Methods parameter source: {METHODS_PARAMS_PATH}.",
        f"- Stage 1 base config: {STAGE1_CONFIG}.",
        f"- Stage 2 base config: {STAGE2_CONFIG}.",
        "- Runtime configs are written under cfg/.",
        "- Each agent uses an isolated training_root under r/<agent_code> so parallel workers do not share monitor/log/model files.",
        "- Runtime paths and model prefixes are intentionally short to avoid Windows MAX_PATH failures in TensorBoard event-file writing.",
        "- Models are produced by the preserved training runtime under each isolated training_root and copied into models/<agent_id>/ after each successful stage.",
        "- agent_training_manifest.csv records seeds, parameter vectors, config paths, model paths, and status.",
        "",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "AGENT_BOOTSTRAPPING_PLAN.txt").write_text("\n".join(plan), encoding="utf-8")


def assert_outdir_available(outdir: Path, resume: bool) -> None:
    if outdir.exists() and any(outdir.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {outdir}. "
            "Use --resume to continue an existing run or choose a new --outdir."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the agreed agent ensemble on methods-seed LV bootstrap results.")
    parser.add_argument("--bootstrap-results", default=str(DEFAULT_BOOTSTRAP_RESULTS), help="bootstrap_results.csv from the kept methods-seed LV bootstrap.")
    parser.add_argument("--n-bootstrap-agents", type=int, default=20, help="Number of bootstrap LV rows to train.")
    parser.add_argument("--n-methods-repeats", type=int, default=5, help="Repeated-seed methods-parameter agents.")
    parser.add_argument("--n-control-agents", type=int, default=1, help="Control agents on methods parameters with current config.")
    parser.add_argument("--stage1-timesteps", type=int, default=100000)
    parser.add_argument("--stage2-timesteps", type=int, default=160000)
    parser.add_argument("--bootstrap-seed-base", type=int, default=20260600)
    parser.add_argument("--methods-seed-base", type=int, default=20260700)
    parser.add_argument("--control-seed-base", type=int, default=20260800)
    parser.add_argument("--outdir", default=None, help="Output directory. Default: kept bootstrap output/agent_training/<timestamp>.")
    parser.add_argument("--run-label", default=None, help="Short label for generated config/model prefixes. Default: agt_<timestamp>.")
    parser.add_argument("--parallel-workers", type=int, default=8, help="Number of agents to train concurrently. Default: 8.")
    parser.add_argument("--torch-num-threads", type=int, default=2, help="Torch CPU threads per worker. Default: 2.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs/manifest/plan without calling Trainer.learn().")
    parser.add_argument("--resume", action="store_true", help="Allow an existing non-empty output directory and skip agents with copied stage2 final models.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to the next agent if a training stage fails.")
    args = parser.parse_args()

    bootstrap_results = Path(args.bootstrap_results)
    if not bootstrap_results.is_absolute():
        bootstrap_results = (Path.cwd() / bootstrap_results).resolve()
    args.bootstrap_results = str(bootstrap_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_label or f"agt_{timestamp}"
    outdir = Path(args.outdir) if args.outdir else METHODS_BOOTSTRAP_DIR / "agent_training" / run_label
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()
    assert_outdir_available(outdir, args.resume)

    rows = load_csv(bootstrap_results)
    if len(rows) < args.n_bootstrap_agents:
        raise ValueError(f"Requested {args.n_bootstrap_agents} bootstrap agents but only found {len(rows)} bootstrap rows.")
    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be >= 1")
    if args.torch_num_threads < 1:
        raise ValueError("--torch-num-threads must be >= 1")
    resources = machine_resources()
    requested_threads = args.parallel_workers * args.torch_num_threads
    if resources["cpu_count_logical"] and requested_threads > resources["cpu_count_logical"]:
        print(
            f"Warning: requested worker thread budget is {requested_threads} "
            f"({args.parallel_workers} workers x {args.torch_num_threads} Torch threads) "
            f"for {resources['cpu_count_logical']} logical CPUs. Consider lowering workers or threads if the machine becomes sluggish."
        )

    jobs = build_jobs(
        rows,
        args.n_bootstrap_agents,
        args.n_methods_repeats,
        args.n_control_agents,
        args.bootstrap_seed_base,
        args.methods_seed_base,
        args.control_seed_base,
    )

    config_dir = outdir / "cfg"
    manifest_path = outdir / "agent_training_manifest.csv"
    metadata_path = outdir / "metadata.json"
    outdir.mkdir(parents=True, exist_ok=True)
    write_run_plan(outdir, args, jobs)

    manifest_rows: list[dict] = []
    if args.resume and manifest_path.exists():
        manifest_rows = load_csv(manifest_path)

    completed_agent_ids = set()
    if args.resume:
        for job in jobs:
            code = agent_code(job)
            copied_stage2_final = outdir / "models" / code / f"{code}_s2_final.zip"
            if copied_stage2_final.exists():
                completed_agent_ids.add(job.agent_id)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "bootstrap_results": str(bootstrap_results),
        "methods_params_path": str(METHODS_PARAMS_PATH),
        "stage1_config": str(STAGE1_CONFIG),
        "stage2_config": str(STAGE2_CONFIG),
        "n_bootstrap_agents": args.n_bootstrap_agents,
        "n_methods_repeats": args.n_methods_repeats,
        "n_control_agents": args.n_control_agents,
        "stage1_timesteps": args.stage1_timesteps,
        "stage2_timesteps": args.stage2_timesteps,
        "bootstrap_seed_base": args.bootstrap_seed_base,
        "methods_seed_base": args.methods_seed_base,
        "control_seed_base": args.control_seed_base,
        "run_label": run_label,
        "parallel_workers": args.parallel_workers,
        "torch_num_threads": args.torch_num_threads,
        "machine_resources": resources,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "agent_repo": str(AGENT_REPO),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rows_by_agent_id = {row.get("agent_id"): row for row in manifest_rows}
    worker_payloads = []

    for job in tqdm(jobs, desc="prepare agent configs", unit="agent"):
        if job.agent_id in completed_agent_ids:
            continue

        code = agent_code(job)
        stage1_prefix = f"{code}_s1"
        stage2_prefix = f"{code}_s2"
        stage1_config_path = config_dir / f"{stage1_prefix}.yaml"
        stage2_config_path = config_dir / f"{stage2_prefix}.yaml"
        training_root = outdir / "r" / code
        stage1_best_for_loading = training_root / "SavedModels" / f"{stage1_prefix}_best_reward"

        stage1_config = build_config(STAGE1_CONFIG, job.params, stage1_prefix, args.stage1_timesteps, job.seed, training_root, load_model=None)
        stage2_config = build_config(STAGE2_CONFIG, job.params, stage2_prefix, args.stage2_timesteps, job.seed, training_root, load_model=stage1_best_for_loading)
        write_yaml(stage1_config_path, stage1_config)
        write_yaml(stage2_config_path, stage2_config)

        row = manifest_row(job, outdir, run_label, stage1_config_path, stage2_config_path, training_root, status="prepared")
        rows_by_agent_id[job.agent_id] = row
        manifest_rows = list(rows_by_agent_id.values())
        write_csv(manifest_path, manifest_rows)

        if args.dry_run:
            row["status"] = "dry_run_prepared"
            write_csv(manifest_path, manifest_rows)
            continue

        worker_payloads.append(
            {
                "agent_id": job.agent_id,
                "seed": job.seed,
                "stage1_config": str(stage1_config_path),
                "stage2_config": str(stage2_config_path),
                "training_root": str(training_root),
                "model_dir": str(outdir / "models" / code),
                "worker_status_path": str(outdir / "worker_status" / f"{code}.json"),
                "torch_num_threads": args.torch_num_threads,
                "stage1_prefix": stage1_prefix,
                "stage2_prefix": stage2_prefix,
            }
        )

    if not args.dry_run and worker_payloads:
        if args.parallel_workers == 1:
            for payload in tqdm(worker_payloads, desc="train agents", unit="agent"):
                result = train_agent_worker(payload)
                row = rows_by_agent_id[result["agent_id"]]
                row.update(result)
                write_csv(manifest_path, list(rows_by_agent_id.values()))
                if result["status"] == "failed" and not args.continue_on_error:
                    raise RuntimeError(result["error"])
        else:
            with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
                future_to_agent = {executor.submit(train_agent_worker, payload): payload["agent_id"] for payload in worker_payloads}
                for future in tqdm(as_completed(future_to_agent), total=len(future_to_agent), desc="train agents", unit="agent"):
                    agent_id = future_to_agent[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "agent_id": agent_id,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    row = rows_by_agent_id[result["agent_id"]]
                    row.update(result)
                    write_csv(manifest_path, list(rows_by_agent_id.values()))
                    if result["status"] == "failed" and not args.continue_on_error:
                        raise RuntimeError(result["error"])

    print(f"Finished agent ensemble orchestration. Outputs written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
