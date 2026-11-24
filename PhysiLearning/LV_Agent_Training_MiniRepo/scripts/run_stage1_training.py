from __future__ import annotations

import argparse
from pathlib import Path

from training_common import REPO_ROOT, run_training, write_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-1 PPO pretraining on the fitted LV system.")
    parser.add_argument(
        "--config",
        default="configs/2025_10_lv_real_mela_1.yaml",
        help="Base stage-1 config relative to the mini-repo root.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Optional override for total_timesteps.")
    parser.add_argument("--suffix", default="", help="Optional suffix appended to model_save_prefix.")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    base_prefix = Path(args.config).stem
    save_prefix = f"{base_prefix}_{args.suffix}" if args.suffix else base_prefix
    runtime_config = write_runtime_config(config_path, timesteps=args.timesteps, save_prefix=save_prefix)
    run_training(runtime_config)
    print(f"Runtime config: {runtime_config}")
    print(f"Expected best model: Training/SavedModels/{save_prefix}_best_reward.zip")
    print(f"Expected final model: Training/SavedModels/{save_prefix}_final.zip")


if __name__ == "__main__":
    main()
