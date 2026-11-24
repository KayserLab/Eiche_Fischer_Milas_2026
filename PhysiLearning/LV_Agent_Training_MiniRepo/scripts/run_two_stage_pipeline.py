from __future__ import annotations

import argparse
from pathlib import Path

from training_common import REPO_ROOT, run_training, write_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical two-stage PPO training pipeline on the fitted LV system.")
    parser.add_argument("--suffix", default="", help="Optional suffix appended to both model_save_prefix values.")
    parser.add_argument("--stage1-timesteps", type=int, default=None, help="Optional override for stage-1 total_timesteps.")
    parser.add_argument("--stage2-timesteps", type=int, default=None, help="Optional override for stage-2 total_timesteps.")
    args = parser.parse_args()

    stage1_prefix = "2025_10_lv_real_mela_1"
    stage2_prefix = "2025_10_lv_real_mela_1_load"
    if args.suffix:
        stage1_prefix = f"{stage1_prefix}_{args.suffix}"
        stage2_prefix = f"{stage2_prefix}_{args.suffix}"

    stage1_config = write_runtime_config(
        REPO_ROOT / "configs" / "2025_10_lv_real_mela_1.yaml",
        timesteps=args.stage1_timesteps,
        save_prefix=stage1_prefix,
    )
    run_training(stage1_config)

    stage1_best = REPO_ROOT / "Training" / "SavedModels" / f"{stage1_prefix}_best_reward"
    stage2_config = write_runtime_config(
        REPO_ROOT / "configs" / "2025_10_lv_real_mela_1_load.yaml",
        timesteps=args.stage2_timesteps,
        save_prefix=stage2_prefix,
        load_model=str(stage1_best),
    )
    run_training(stage2_config)

    print(f"Stage-1 runtime config: {stage1_config}")
    print(f"Stage-2 runtime config: {stage2_config}")
    print(f"Expected stage-1 best model: Training/SavedModels/{stage1_prefix}_best_reward.zip")
    print(f"Expected stage-2 best model: Training/SavedModels/{stage2_prefix}_best_reward.zip")


if __name__ == "__main__":
    main()
