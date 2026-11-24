#!/usr/bin/env python3
"""Uncertainty and figure for distill_policy.py outputs.

For each output directory, refits the primary trees on the stored training
split (deterministic, so identical to distill_policy.py; this is asserted),
then computes paired trajectory-bootstrap 95% intervals on the test and
robustness splits for:
  - all-feature vs burden-only tree (P(ON) MAE and hard-action agreement),
  - the fitted two-threshold hysteresis rule.
Writes `uncertainty.json` into each directory and a combined policy-portrait
figure (teacher P(ON) against current burden, split by previous action).

    python scripts/summarize_distillation.py outputs/policy_distillation/paper_candidate \
        outputs/policy_distillation/experimental_agent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_policy as dp  # noqa: E402

PREV_OFF, PREV_ON = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED = "#0b0b0b", "#52514e"


def refit(df: pd.DataFrame, metrics: dict) -> dict:
    s = metrics["settings"]
    train = df[df["split"] == "train"]
    trees = {name: dp.fit_tree(train, dp.FEATURE_SETS[name], s["depth"], s["min_leaf_fraction"], s["seed"])
             for name in ("all_features", "burden_only")}
    splits = dp.tree_splits(trees["all_features"], dp.FEATURES)
    expected = metrics["primary_tree_splits"]
    assert [(a["feature"], round(a["threshold"], 6)) for a in splits] == \
        [(b["feature"], round(b["threshold"], 6)) for b in expected], "refit differs from stored tree"
    return trees


def paired_bootstrap(d: pd.DataFrame, trees: dict, hyst: dict, n_boot: int, seed: int) -> dict:
    p = d["p_on"].to_numpy()
    teacher = d["teacher_action"].to_numpy()
    pred = {k: dp.tree_predict(t, d, dp.FEATURE_SETS[k]) for k, t in trees.items()}
    per_row = {
        "MAE_all": np.abs(p - pred["all_features"]),
        "MAE_burden": np.abs(p - pred["burden_only"]),
        "agree_all": (dp.hard_action(pred["all_features"]) == teacher).astype(float),
        "agree_burden": (dp.hard_action(pred["burden_only"]) == teacher).astype(float),
        "agree_hysteresis": (dp.hysteresis_rule(d, **hyst) == teacher).astype(float),
    }
    frame = pd.DataFrame(per_row)
    frame["traj"] = d["trajectory_id"].to_numpy()
    sums = frame.groupby("traj").sum()
    counts = frame.groupby("traj").size().to_numpy()
    sums_np = sums.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums_np), size=(n_boot, len(sums_np)))
    boot = sums_np[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    boot = pd.DataFrame(boot, columns=sums.columns)
    boot["MAE_burden_minus_all"] = boot["MAE_burden"] - boot["MAE_all"]
    boot["agree_all_minus_burden"] = boot["agree_all"] - boot["agree_burden"]
    point = frame.drop(columns="traj").mean()
    point["MAE_burden_minus_all"] = point["MAE_burden"] - point["MAE_all"]
    point["agree_all_minus_burden"] = point["agree_all"] - point["agree_burden"]
    return {k: {"estimate": float(point[k]),
                "ci95": [float(np.quantile(boot[k], 0.025)), float(np.quantile(boot[k], 0.975))]}
            for k in boot.columns} | {"n_trajectories": int(len(sums_np)), "n_rows": int(len(d))}


def portrait(ax, df: pd.DataFrame, metrics: dict, title: str) -> None:
    d = df[df["split"].isin(["test", "robustness_test"]) & (df["step"] >= dp.N_STACK - 1)]
    for prev, color, label in ((0, PREV_OFF, "previous action OFF"), (1, PREV_ON, "previous action ON")):
        g = d[d["previous_action_t"] == prev]
        ax.scatter(g["burden_t"], g["p_on"], s=9, alpha=0.35, color=color, linewidths=0,
                   label=f"{label} (n={len(g)})")
    h = metrics["fitted_rules"]["hysteresis"]
    ax.axvline(h["start"], color=PREV_OFF, lw=2, ls="--")
    ax.axvline(h["stay"], color=PREV_ON, lw=2, ls="--")
    ax.text(h["start"] - 0.03, 1.05, f"start if $B_t$ ≥ {h['start']:.2f}", color=INK, ha="right", fontsize=8)
    ax.text(h["stay"] + 0.03, 1.05, f"continue if $B_t$ > {h['stay']:.2f}", color=INK, ha="left", fontsize=8)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0.4, 3.25)
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("current normalised burden  $B_t$", color=INK)
    ax.set_ylabel("teacher P(treatment ON)", color=INK)
    ax.set_title(f"{title}: dashed = fitted two-threshold rule", color=INK, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--figure", type=Path,
                        default=dp.MINIREPO / "outputs/policy_distillation/policy_portrait")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(args.dirs), figsize=(5.2 * len(args.dirs), 4.2), squeeze=False)
    for ax, d in zip(axes[0], args.dirs):
        metrics = json.loads((d / "metrics.json").read_text())
        df = pd.read_csv(d / "observations_and_teacher_labels.csv")
        trees = refit(df, metrics)
        hyst = metrics["fitted_rules"]["hysteresis"]
        out = {split: paired_bootstrap(df[df["split"] == split], trees, hyst, args.n_boot,
                                       metrics["settings"]["seed"])
               for split in ("test", "robustness_test")}
        (d / "uncertainty.json").write_text(json.dumps(out, indent=2))
        print(d.name, json.dumps({s: {k: (round(v["estimate"], 3), [round(x, 3) for x in v["ci95"]])
                                      for k, v in r.items() if isinstance(v, dict)}
                                  for s, r in out.items()}, indent=1))
        portrait(ax, df, metrics, metrics["label"])
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure.with_suffix(".png"), dpi=170)
    fig.savefig(args.figure.with_suffix(".pdf"))
    print("figure:", args.figure.with_suffix(".png"))


if __name__ == "__main__":
    main()
