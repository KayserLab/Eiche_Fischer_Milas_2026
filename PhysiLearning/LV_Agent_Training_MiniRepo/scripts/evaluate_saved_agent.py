from __future__ import annotations

import argparse
from pathlib import Path

from training_common import REPO_ROOT, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO agent on the fitted LV system.")
    parser.add_argument(
        "--config",
        default="configs/2025_10_lv_real_mela_1_load.yaml",
        help="Config relative to the mini-repo root used to construct the evaluation env.",
    )
    parser.add_argument(
        "--model",
        default="reference/saved_models/2025_10_lv_real_mela_1_load_best_reward.zip",
        help="Model path relative to the mini-repo root.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of deterministic evaluation episodes.")
    parser.add_argument(
        "--outdir",
        default="outputs/reference_agent_eval",
        help="Output directory relative to the mini-repo root.",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    model_path = REPO_ROOT / args.model
    outdir = REPO_ROOT / args.outdir
    evaluate_model(config_path, model_path, outdir, args.episodes)
    print(f"Evaluated model: {model_path}")
    print(f"Trajectory outputs: {outdir}")


if __name__ == "__main__":
    main()
