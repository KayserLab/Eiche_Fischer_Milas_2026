from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physilearning.evaluate import Evaluation
from physilearning.train import Trainer


def ensure_repo_root() -> None:
    os.chdir(REPO_ROOT)


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_runtime_config(
    base_config: Path,
    *,
    timesteps: int | None = None,
    save_prefix: str | None = None,
    load_model: str | None = None,
) -> Path:
    config = load_config(base_config)
    if timesteps is not None:
        config["learning"]["model"]["total_timesteps"] = int(timesteps)
    if save_prefix is not None:
        config["learning"]["model"]["model_save_prefix"] = save_prefix
    if load_model is not None:
        config["learning"]["model"]["load"]["saved_model_name"] = str(load_model)
        config["learning"]["model"]["load"]["enable_loading"] = 1

    runtime_dir = REPO_ROOT / "outputs" / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out_path = runtime_dir / f"{save_prefix or base_config.stem}.yaml"
    with out_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return out_path


def clean_monitor() -> None:
    logs_dir = REPO_ROOT / "Training" / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = logs_dir / "monitor.csv"
    empty_monitor = logs_dir / "empty_monitor.csv"
    if monitor_path.exists():
        monitor_path.unlink()
    shutil.copy2(empty_monitor, monitor_path)


def run_training(config_path: Path) -> None:
    ensure_repo_root()
    clean_monitor()
    trainer = Trainer(str(config_path))
    trainer.learn()


def evaluate_model(config_path: Path, model_path: Path, outdir: Path, episodes: int) -> None:
    ensure_repo_root()
    outdir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(str(config_path))
    trainer.setup_env()
    evaluation = Evaluation(trainer.env, config_file=str(config_path))
    evaluation.run_environment(
        model_name=str(model_path),
        num_episodes=int(episodes),
        save_path=str(outdir),
        save_name="lv_agent_eval",
        fixed_therapy=False,
    )
    trainer.env.close()
