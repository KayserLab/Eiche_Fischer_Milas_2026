#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSILEARNING_ROOT = REPO_ROOT.parent
AGENT_REPO = PHYSILEARNING_ROOT / "LV_Agent_Training_MiniRepo"
AGENT_SRC = AGENT_REPO / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from physilearning.train import Trainer  # noqa: E402

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

STAGE1_CONFIG = AGENT_REPO / "configs" / "2025_10_lv_real_mela_1.yaml"
STAGE2_CONFIG = AGENT_REPO / "configs" / "2025_10_lv_real_mela_1_load.yaml"
REFERENCE_PRETRAIN = AGENT_REPO / "reference" / "saved_models" / "2025_10_lv_real_mela_1_best_reward.zip"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def clean_monitor() -> None:
    logs_dir = AGENT_REPO / "Training" / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    empty_monitor = logs_dir / "empty_monitor.csv"
    monitor_path = logs_dir / "monitor.csv"
    if monitor_path.exists():
        monitor_path.unlink()
    shutil.copy2(empty_monitor, monitor_path)


def apply_lv_params(config: dict, row: dict) -> dict:
    cfg = dict(config)
    env = cfg["env"]
    lv = env["LvEnv"]

    growth_sus = float(row["growth_rate_sus"])
    growth_res = float(row["growth_rate_res"])
    random_death = float(row["random_death"])

    env["growth_rate_wt"] = growth_sus
    env["growth_rate_mut"] = growth_res
    env["death_rate_wt"] = random_death
    env["death_rate_mut"] = random_death
    env["treat_death_rate_wt"] = float(row["death_treat_sus"])
    env["treat_death_rate_mut"] = 0.0

    lv["competition_wt"] = float(row["competition"])
    lv["competition_mut"] = 1.0
    lv["ramp_time_up"] = float(row["ramp_time_up"])
    lv["ramp_time_down"] = float(row["ramp_time_down"])
    lv["carrying_capacity"] = float(row["carrying_capacity"])
    lv["on_treat_threshold"] = float(row["on_treat_threshold"])
    lv["off_treat_threshold"] = float(row["off_treat_threshold"])
    lv["min_death_during_treat"] = float(row["min_treat"])
    return cfg


def build_config(base_config: Path, row: dict, save_prefix: str, timesteps: int, load_model: Path | None) -> dict:
    cfg = apply_lv_params(load_yaml(base_config), row)
    model_cfg = cfg["learning"]["model"]
    model_cfg["total_timesteps"] = int(timesteps)
    model_cfg["model_save_prefix"] = save_prefix
    if load_model is None:
        model_cfg["load"]["enable_loading"] = 0
        model_cfg["load"]["last_model"] = 0
    else:
        model_cfg["load"]["enable_loading"] = 1
        model_cfg["load"]["last_model"] = 0
        model_cfg["load"]["saved_model_name"] = str(load_model)
    return cfg


def run_training(config_path: Path) -> None:
    os.chdir(AGENT_REPO)
    clean_monitor()
    trainer = Trainer(str(config_path))
    trainer.learn()


def expected_model(prefix: str, suffix: str) -> Path:
    return AGENT_REPO / "Training" / "SavedModels" / f"{prefix}_{suffix}.zip"


def selected_bootstrap_rows(rows: list[dict], n_agents: int | None, only_success: bool) -> list[dict]:
    usable = []
    for row in rows:
        if only_success and str(row.get("success", "")).lower() != "true":
            continue
        missing = [col for col in PARAMETER_COLUMNS if col not in row or row[col] == ""]
        if missing:
            raise KeyError(f"bootstrap row missing parameter columns: {missing}")
        usable.append(row)
    if n_agents is not None:
        usable = usable[:n_agents]
    return usable


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO agents on bootstrapped LV parameter fits.")
    parser.add_argument("--bootstrap-results", required=True, help="bootstrap_results.csv from bootstrap_lv_refits.py")
    parser.add_argument("--n-agents", type=int, default=None, help="Number of bootstrap rows to train. Default: all rows.")
    parser.add_argument("--mode", choices=["two_stage", "stage2_reference"], default="two_stage")
    parser.add_argument("--stage1-timesteps", type=int, default=100000)
    parser.add_argument("--stage2-timesteps", type=int, default=160000)
    parser.add_argument("--suffix", default="bootstrap_lv", help="Prefix suffix for generated model names.")
    parser.add_argument("--only-successful-fits", action="store_true", help="Skip bootstrap LV fits whose optimizer success is false.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs/manifest but do not train.")
    parser.add_argument("--outdir", default=None, help="Output directory. Default: outputs/agent_training_<timestamp>.")
    args = parser.parse_args()

    bootstrap_results = Path(args.bootstrap_results)
    if not bootstrap_results.is_absolute():
        bootstrap_results = (Path.cwd() / bootstrap_results).resolve()
    rows = selected_bootstrap_rows(load_csv(bootstrap_results), args.n_agents, args.only_successful_fits)
    if not rows:
        raise RuntimeError("No bootstrap rows selected for agent training.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else REPO_ROOT / "outputs" / f"agent_training_{timestamp}"
    if not outdir.is_absolute():
        outdir = (REPO_ROOT / outdir).resolve()
    config_dir = outdir / "runtime_configs"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for idx, row in enumerate(rows):
        bootstrap_index = int(row["bootstrap_index"])
        label = f"{args.suffix}_{bootstrap_index:04d}"
        stage1_prefix = f"{label}_stage1"
        stage2_prefix = f"{label}_stage2"

        stage1_config_path = config_dir / f"{stage1_prefix}.yaml"
        stage2_config_path = config_dir / f"{stage2_prefix}.yaml"

        if args.mode == "two_stage":
            stage1_config = build_config(STAGE1_CONFIG, row, stage1_prefix, args.stage1_timesteps, load_model=None)
            write_yaml(stage1_config_path, stage1_config)
            stage1_load_model = AGENT_REPO / "Training" / "SavedModels" / f"{stage1_prefix}_best_reward"
        else:
            stage1_config_path = None
            stage1_load_model = REFERENCE_PRETRAIN

        stage2_config = build_config(STAGE2_CONFIG, row, stage2_prefix, args.stage2_timesteps, load_model=stage1_load_model)
        write_yaml(stage2_config_path, stage2_config)

        manifest = {
            "agent_order": idx,
            "bootstrap_index": bootstrap_index,
            "bootstrap_label": row.get("label", f"bootstrap_{bootstrap_index:04d}"),
            "mode": args.mode,
            "stage1_config": str(stage1_config_path) if stage1_config_path else "",
            "stage2_config": str(stage2_config_path),
            "stage1_best_model": str(expected_model(stage1_prefix, "best_reward")) if args.mode == "two_stage" else str(REFERENCE_PRETRAIN),
            "stage1_final_model": str(expected_model(stage1_prefix, "final")) if args.mode == "two_stage" else "",
            "stage2_best_model": str(expected_model(stage2_prefix, "best_reward")),
            "stage2_final_model": str(expected_model(stage2_prefix, "final")),
        }
        for col in PARAMETER_COLUMNS:
            manifest[col] = float(row[col])
        manifest_rows.append(manifest)
        write_csv(outdir / "agent_training_manifest.csv", manifest_rows)

        print(f"[{label}] prepared configs")
        if args.dry_run:
            continue

        if args.mode == "two_stage":
            print(f"[{label}] training stage 1 for {args.stage1_timesteps} timesteps")
            run_training(stage1_config_path)
        print(f"[{label}] training stage 2 for {args.stage2_timesteps} timesteps")
        run_training(stage2_config_path)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "bootstrap_results": str(bootstrap_results),
        "n_selected_agents": len(rows),
        "mode": args.mode,
        "stage1_timesteps": args.stage1_timesteps,
        "stage2_timesteps": args.stage2_timesteps,
        "dry_run": bool(args.dry_run),
        "agent_repo": str(AGENT_REPO),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Finished agent training orchestration. Outputs written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
