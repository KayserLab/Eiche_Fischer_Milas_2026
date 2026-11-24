#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

EXPECTED_MAXITER_1 = {
    "original_saved": 0.103785613408,
    "fit_from_original_seed": 0.393310461058,
    "fit_from_provided_seed": 0.103639653674,
}
EXPECTED_MAXITER_3 = {
    "original_saved": 0.10378561340793453,
    "fit_from_original_seed": 0.3821832095141169,
    "fit_from_provided_seed": 0.10363965367398319,
}

REQUIRED_OUTPUTS = [
    "parameter_table.csv",
    "objective_summary.csv",
    "summary.json",
    "report.txt",
    "per_experiment_original_saved.csv",
    "per_experiment_fit_from_original_seed.csv",
    "per_experiment_fit_from_provided_seed.csv",
    "progress/fit_from_original_seed.status.json",
    "progress/fit_from_provided_seed.status.json",
]


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


def run_refit(maxiter: int, outdir: Path) -> None:
    if outdir.exists():
        shutil.rmtree(outdir)
    command = [
        sys.executable,
        "scripts/run_refit_comparison.py",
        "--maxiter",
        str(maxiter),
        "--outdir",
        str(outdir),
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_objectives(outdir: Path) -> dict[str, float]:
    path = outdir / "objective_summary.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["scenario"]: float(row["objective"]) for row in csv.DictReader(handle)}


def check_outputs(outdir: Path) -> None:
    missing = [rel for rel in REQUIRED_OUTPUTS if not (outdir / rel).exists()]
    if missing:
        raise AssertionError(f"Missing expected files in {outdir}: {missing}")


def check_objectives(outdir: Path, expected: dict[str, float], atol: float) -> None:
    observed = read_objectives(outdir)
    for scenario, expected_value in expected.items():
        if scenario not in observed:
            raise AssertionError(f"{outdir}: missing scenario {scenario!r}")
        delta = abs(observed[scenario] - expected_value)
        print(
            f"{outdir.name}: {scenario}: observed={observed[scenario]:.17g} "
            f"expected={expected_value:.17g} delta={delta:.3g}"
        )
        if delta > atol:
            raise AssertionError(
                f"{outdir}: {scenario} differs by {delta:.3g}, tolerance is {atol:.3g}"
            )


def validate_run(maxiter: int, outdir: Path, expected: dict[str, float], atol: float) -> None:
    run_refit(maxiter, outdir)
    check_outputs(outdir)
    check_objectives(outdir, expected, atol)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LV fitting mini-repo on a fresh machine.")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute objective tolerance.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the maxiter=1 smoke run.")
    parser.add_argument("--skip-capped", action="store_true", help="Skip the maxiter=3 capped validation run.")
    args = parser.parse_args()

    verify_checksums()
    if not args.skip_smoke:
        validate_run(1, REPO_ROOT / "outputs" / "validation_maxiter1", EXPECTED_MAXITER_1, args.atol)
    if not args.skip_capped:
        validate_run(3, REPO_ROOT / "outputs" / "validation_maxiter3", EXPECTED_MAXITER_3, args.atol)

    print("LV fitting mini-repo validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
