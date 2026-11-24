from __future__ import annotations

import argparse
from pathlib import Path

from training_common import REPO_ROOT, run_training, write_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-2 PPO fine-tuning on the fitted LV system.")
    parser.add_argument(
        "--config",
        default="configs/2025_10_lv_real_mela_1_load.yaml",
        help="Base stage-2 config relative to the mini-repo root.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Optional override for total_timesteps.")
    parser.add_argument("--suffix", default="", help="Optional suffix appended to model_save_prefix.")
    parser.add_argument(
        "--load-model",
        default="",
        help="Optional explicit model path to load instead of the default stage-1 best model.",
    )
    parser.add_argument(
        "--use-reference-pretrain",
        action="store_true",
        help="Load the preserved reference pretrain model from reference/saved_models/.",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    base_prefix = Path(args.config).stem
    save_prefix = f"{base_prefix}_{args.suffix}" if args.suffix else base_prefix

    if args.load_model:
        load_model = REPO_ROOT / args.load_model
    elif args.use_reference_pretrain:
        load_model = REPO_ROOT / "reference" / "saved_models" / "2025_10_lv_real_mela_1_best_reward.zip"
    else:
        stage1_prefix = "2025_10_lv_real_mela_1"
        if args.suffix:
            stage1_prefix = f"{stage1_prefix}_{args.suffix}"
        load_model = REPO_ROOT / "Training" / "SavedModels" / f"{stage1_prefix}_best_reward"

    runtime_config = write_runtime_config(
        config_path,
        timesteps=args.timesteps,
        save_prefix=save_prefix,
        load_model=str(load_model),
    )
    run_training(runtime_config)
    print(f"Runtime config: {runtime_config}")
    print(f"Loaded model: {load_model}")
    print(f"Expected best model: Training/SavedModels/{save_prefix}_best_reward.zip")
    print(f"Expected final model: Training/SavedModels/{save_prefix}_final.zip")


if __name__ == "__main__":
    main()
