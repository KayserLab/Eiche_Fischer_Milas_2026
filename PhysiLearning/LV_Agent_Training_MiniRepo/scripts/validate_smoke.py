#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_CHECKSUM_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".sha256",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_CHECKSUM_NAMES = {".python-version"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_text_checksum_path(path: Path) -> bool:
    return path.name in TEXT_CHECKSUM_NAMES or path.suffix.lower() in TEXT_CHECKSUM_SUFFIXES


def _sha256_lf_normalized_text(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256_bytes(data)


def verify_checksums() -> None:
    manifest = REPO_ROOT / "checksums.sha256"
    if not manifest.exists():
        raise AssertionError(f"Missing checksum manifest: {manifest}")
    for line_number, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        expected_hash, rel_path = line.split("  ", 1)
        path = REPO_ROOT / rel_path
        if not path.exists():
            raise AssertionError(f"Checksum manifest line {line_number}: missing file {rel_path}")
        observed_hash = _sha256_bytes(path.read_bytes())
        if observed_hash != expected_hash:
            if _is_text_checksum_path(path) and _sha256_lf_normalized_text(path) == expected_hash:
                print(f"Checksum verification used LF-normalized text for {rel_path}.")
                continue
            raise AssertionError(f"Checksum mismatch for {rel_path}: {observed_hash} != {expected_hash}")
    print("Checksum verification passed.")


def run(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_validation_outputs(suffix: str, eval_outdir: Path) -> None:
    remove_path(eval_outdir)
    for directory in [
        REPO_ROOT / "outputs" / "runtime_configs",
        REPO_ROOT / "Training" / "Configs",
        REPO_ROOT / "Training" / "SavedModels",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    prefixes = [
        f"2025_10_lv_real_mela_1_{suffix}",
        f"2025_10_lv_real_mela_1_load_{suffix}",
    ]
    for prefix in prefixes:
        remove_path(REPO_ROOT / "outputs" / "runtime_configs" / f"{prefix}.yaml")
        remove_path(REPO_ROOT / "Training" / "Configs" / f"{prefix}.yaml")
        remove_path(REPO_ROOT / "Training" / "SavedModels" / f"{prefix}_best_reward.zip")
        remove_path(REPO_ROOT / "Training" / "SavedModels" / f"{prefix}_final.zip")


def require_file(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing expected file: {path}")


def validate_reference_eval(eval_outdir: Path, episodes: int) -> None:
    run(
        [
            sys.executable,
            "scripts/evaluate_saved_agent.py",
            "--episodes",
            str(episodes),
            "--outdir",
            str(eval_outdir.relative_to(REPO_ROOT)),
        ]
    )
    h5_files = list(eval_outdir.glob("*.h5"))
    if not h5_files:
        raise AssertionError(f"No .h5 trajectory files written in {eval_outdir}")
    for path in h5_files:
        print(f"Found evaluation trajectory: {path}")


def validate_training_smoke(suffix: str, timesteps: int) -> None:
    stage1_prefix = f"2025_10_lv_real_mela_1_{suffix}"
    stage2_prefix = f"2025_10_lv_real_mela_1_load_{suffix}"

    run([sys.executable, "scripts/run_stage1_training.py", "--timesteps", str(timesteps), "--suffix", suffix])
    require_file(REPO_ROOT / "outputs" / "runtime_configs" / f"{stage1_prefix}.yaml")
    require_file(REPO_ROOT / "Training" / "Configs" / f"{stage1_prefix}.yaml")
    require_file(REPO_ROOT / "Training" / "SavedModels" / f"{stage1_prefix}_final.zip")
    require_file(REPO_ROOT / "Training" / "SavedModels" / f"{stage1_prefix}_best_reward.zip")

    run([sys.executable, "scripts/run_stage2_finetune.py", "--timesteps", str(timesteps), "--suffix", suffix])
    require_file(REPO_ROOT / "outputs" / "runtime_configs" / f"{stage2_prefix}.yaml")
    require_file(REPO_ROOT / "Training" / "Configs" / f"{stage2_prefix}.yaml")
    require_file(REPO_ROOT / "Training" / "SavedModels" / f"{stage2_prefix}_final.zip")
    require_file(REPO_ROOT / "Training" / "SavedModels" / f"{stage2_prefix}_best_reward.zip")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LV agent training mini-repo on a fresh machine.")
    parser.add_argument("--episodes", type=int, default=1, help="Reference-agent evaluation episodes.")
    parser.add_argument("--timesteps", type=int, default=64, help="Smoke-training timesteps per stage.")
    parser.add_argument("--suffix", default="validation_smoke", help="Suffix for generated smoke-training models.")
    parser.add_argument("--skip-training", action="store_true", help="Only evaluate the preserved reference agent.")
    args = parser.parse_args()

    verify_checksums()
    eval_outdir = REPO_ROOT / "outputs" / "reference_agent_eval_smoke"
    clean_validation_outputs(args.suffix, eval_outdir)
    validate_reference_eval(eval_outdir, args.episodes)
    if not args.skip_training:
        validate_training_smoke(args.suffix, args.timesteps)

    print("LV agent training mini-repo validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
