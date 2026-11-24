#!/usr/bin/env python3
"""Distil a frozen LV PPO policy into a shallow CART regression tree.

This is ordinary CART policy distillation (supervised regression of a decision
tree onto the teacher's P(treatment ON)); it is NOT the iterative, value-weighted
VIPER algorithm. The script never retrains PPO, never writes into
`reference/`, `configs/` or any checkpoint directory, and writes all results
below `--outdir` (default: `outputs/policy_distillation/<label>`; `outputs/`
is git-ignored).

Observation order (VecFrameStack(n_stack=4, channels_order='last') over the
LvEnv base observation [total_burden, previous_action]), oldest to newest:

    [burden_t-3, previous_action_t-3, burden_t-2, previous_action_t-2,
     burden_t-1, previous_action_t-1, burden_t,   previous_action_t]

`previous_action_t` is the action that was administered during the interval
that ended with the measurement `burden_t`, i.e. the action already applied
when the teacher makes its next decision. It is NOT the decision being made.

Datasets
--------
primary     LV dynamics from the given config (unchanged initial condition).
            Episodes mix greedy PPO control and epsilon-flipped executed
            actions (eps in {0, 0.05, 0.15, 0.30}); every visited observation is
            labelled with the unchanged teacher P(ON). Because the LV model is
            deterministic, all eps=0 episodes are identical; trajectories are
            therefore grouped by the hash of their executed action sequence so
            that exact duplicates never straddle the train/val/test split.
robustness  Test-only set with the same exploration schedule but a varied
            initial resistant fraction (log-uniform), i.e. genuinely different
            initial conditions. The config on disk is not modified.

Run from the mini-repo root with its Python 3.10 environment, e.g.

    python scripts/distill_policy.py --episodes 16 --robustness-episodes 8 \
        --stability-resamples 5 --outdir outputs/policy_distillation_smoke
    python scripts/distill_policy.py --episodes 250 --depth 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

MINIREPO = Path(__file__).resolve().parents[1]
SRC = MINIREPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from physilearning.envs.lv import LvEnv  # noqa: E402

FEATURES = [
    "burden_t-3", "previous_action_t-3",
    "burden_t-2", "previous_action_t-2",
    "burden_t-1", "previous_action_t-1",
    "burden_t", "previous_action_t",
]
FEATURE_SETS = {
    "all_features": FEATURES,
    "burden_only": FEATURES[0::2],
    "action_only": FEATURES[1::2],
}
EXPLORATION_RATES = (0.0, 0.05, 0.15, 0.30)
N_STACK = 4
DEFAULT_MODEL = MINIREPO / "reference/saved_models/2025_10_lv_real_mela_1_load_best_reward.zip"
DEFAULT_CONFIG = MINIREPO / "configs/2025_10_lv_real_mela_1_load.yaml"


# --------------------------------------------------------------------------
# Teacher loading and querying
# --------------------------------------------------------------------------
def sha256_path(path: Path) -> dict:
    """SHA-256 of a checkpoint zip, or of each member of an extracted SB3 dir."""
    if path.is_file():
        return {"file": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        member.name: hashlib.sha256(member.read_bytes()).hexdigest()
        for member in sorted(path.iterdir())
        if member.is_file()
    }


def policy_weights_sha256(path: Path) -> str:
    """SHA-256 of policy.pth, comparable between zip and extracted-dir checkpoints."""
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            return hashlib.sha256(archive.read("policy.pth")).hexdigest()
    return hashlib.sha256((path / "policy.pth").read_bytes()).hexdigest()


@contextmanager
def materialize_checkpoint(path: Path) -> Iterator[Path]:
    """Yield a loadable zip path for a zip or an extracted SB3 save directory."""
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    with tempfile.TemporaryDirectory(prefix="sb3_teacher_") as tmp_dir:
        zip_path = Path(tmp_dir) / f"{path.name}.zip"
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in sorted(path.iterdir()):
                if member.is_file():
                    archive.write(member, arcname=member.name)
        yield zip_path


def load_teacher(path: Path) -> PPO:
    custom_objects = {"clip_range": lambda _: 0.0, "lr_schedule": lambda _: 0.0}
    with materialize_checkpoint(path) as load_path, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PPO.load(str(load_path), device="cpu", custom_objects=custom_objects)
    if tuple(model.observation_space.shape) != (len(FEATURES),):
        raise ValueError(f"Unexpected observation space {model.observation_space} in {path}")
    if getattr(model.action_space, "n", None) != 2:
        raise ValueError(f"Expected binary action space, got {model.action_space} in {path}")
    return model


def teacher_probabilities(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Return P(ON) for a batch of stacked observations, shape (n,)."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, len(FEATURES))
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
    if probs.shape != (len(obs), 2) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"Unexpected PPO probabilities: shape {probs.shape}")
    return probs[:, 1].astype(float)


def hard_action(p_on: np.ndarray) -> np.ndarray:
    """Deterministic decision from P(ON). Ties go to OFF, as in categorical argmax."""
    return (np.asarray(p_on) > 0.5).astype(int)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class StateInfo(gymnasium.Wrapper):
    """Expose the hidden LV state in `info` (DummyVecEnv auto-resets on done)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self.env.unwrapped.state
        info = dict(info)
        info["lv_state"] = (float(state[0]), float(state[1]))
        info["lv_time"] = int(self.env.unwrapped.time)
        return obs, reward, terminated, truncated, info


def make_env(config_path: Path, initial_mut_fraction: float | None = None) -> VecFrameStack:
    """Training-equivalent env: DummyVecEnv + VecFrameStack(4, 'last'), no Monitor."""

    def factory():
        env = LvEnv.from_yaml(str(config_path))
        if initial_mut_fraction is not None:
            # Keep total (normalised) initial burden, change only its composition.
            total = env.initial_wt + env.initial_mut
            env.initial_wt = (1.0 - initial_mut_fraction) * total
            env.initial_mut = initial_mut_fraction * total
        return StateInfo(env)

    env = VecFrameStack(DummyVecEnv([factory]), n_stack=N_STACK, channels_order="last")
    if tuple(env.observation_space.shape) != (len(FEATURES),):
        raise ValueError(f"Unexpected stacked observation shape {env.observation_space.shape}")
    return env


def default_initial_mut_fraction(config_path: Path) -> float:
    env = LvEnv.from_yaml(str(config_path))
    return float(env.initial_mut / (env.initial_wt + env.initial_mut))


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def rollout(
    env: VecFrameStack,
    controller: Callable[[np.ndarray], int],
    teacher: PPO | None,
    epsilon: float,
    rng: np.random.Generator,
    max_steps: int,
) -> tuple[list[dict], dict]:
    """Run one episode. `controller` picks the intended action; with probability
    `epsilon` the executed action is flipped. Each visited observation is labelled
    with the teacher's P(ON) (if a teacher is given)."""
    rows: list[dict] = []
    obs = env.reset()
    state = tuple(float(x) for x in env.venv.envs[0].unwrapped.state[:2])
    outcome = {"terminated": False, "truncated": False, "steps": 0, "return": 0.0}
    for step in range(max_steps):
        x = np.asarray(obs[0], dtype=np.float32)
        intended = int(controller(x))
        flipped = bool(rng.random() < epsilon)
        executed = 1 - intended if flipped else intended
        row = {"step": step, "intended_action": intended, "executed_action": executed,
               "flipped": flipped, "sensitive": state[0], "resistant": state[1]}
        row.update(dict(zip(FEATURES, map(float, x))))
        if teacher is not None:
            row["p_on"] = float(teacher_probabilities(teacher, x)[0])
        rows.append(row)
        obs, reward, done, info = env.step(np.array([executed], dtype=np.int64))
        outcome["return"] += float(reward[0])
        outcome["steps"] = step + 1
        state = info[0]["lv_state"]
        if bool(done[0]):
            truncated = bool(info[0].get("TimeLimit.truncated", False))
            outcome["truncated"] = truncated
            outcome["terminated"] = not truncated
            break
    burden = state[0] + state[1]
    outcome["final_burden"] = burden
    outcome["final_resistant_fraction"] = state[1] / burden if burden > 0 else float("nan")
    return rows, outcome


def teacher_controller(teacher: PPO) -> Callable[[np.ndarray], int]:
    return lambda x: int(hard_action(teacher_probabilities(teacher, x))[0])


def generate_dataset(
    teacher: PPO,
    config_path: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    dataset: str,
    mut_fraction_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Collect genuine LV trajectories under PPO control with epsilon-flips.
    Teacher labels are always the unchanged PPO P(ON) of the visited observation."""
    frames = []
    controller = teacher_controller(teacher)
    for episode in range(episodes):
        rng = np.random.default_rng([seed, episode, 0 if dataset == "primary" else 1])
        eps = EXPLORATION_RATES[episode % len(EXPLORATION_RATES)]
        mut_fraction = None
        if mut_fraction_range is not None:
            lo, hi = np.log10(mut_fraction_range[0]), np.log10(mut_fraction_range[1])
            mut_fraction = float(10 ** rng.uniform(lo, hi))
        env = make_env(config_path, mut_fraction)
        try:
            rows, outcome = rollout(env, controller, teacher, eps, rng, max_steps)
        finally:
            env.close()
        frame = pd.DataFrame(rows)
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "episode", episode)
        frame.insert(2, "epsilon", eps)
        frame.insert(3, "initial_mut_fraction",
                     mut_fraction if mut_fraction is not None else default_initial_mut_fraction(config_path))
        # Group identical trajectories together (deterministic dynamics).
        key = f"{frame['initial_mut_fraction'].iloc[0]:.12g}|" + "".join(
            map(str, frame["executed_action"]))
        frame.insert(4, "trajectory_id", hashlib.sha1(key.encode()).hexdigest()[:12])
        frame["teacher_action"] = hard_action(frame["p_on"].to_numpy())
        frame["episode_terminated"] = outcome["terminated"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Students: trees and simple rule baselines
# --------------------------------------------------------------------------
def fit_tree(df: pd.DataFrame, features: list[str], depth: int, min_leaf: float, seed: int,
             sample_weight: np.ndarray | None = None) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(df[features].to_numpy(dtype=np.float32), df["p_on"].to_numpy(dtype=float),
             sample_weight=sample_weight)
    return tree


def tree_predict(tree: DecisionTreeRegressor, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(tree.predict(df[features].to_numpy(dtype=np.float32)), 0.0, 1.0)


def hysteresis_rule(df: pd.DataFrame, start: float, stay: float) -> np.ndarray:
    """`effective`-style rule (cf. evaluate.fixed_at): if previously OFF, start
    when burden_t >= start; if previously ON, continue while burden_t > stay.
    Burden is normalised to the initial burden, so thresholds are fold-changes."""
    b, prev = df["burden_t"].to_numpy(), df["previous_action_t"].to_numpy().astype(int)
    return np.where(prev == 0, b >= start, b > stay).astype(int)


def fit_hysteresis_rule(df: pd.DataFrame) -> dict:
    """Grid-search the two thresholds for maximal hard-action agreement on `df`."""
    grid = np.round(np.arange(0.5, 3.01, 0.01), 2)
    target = df["teacher_action"].to_numpy()
    b, prev = df["burden_t"].to_numpy(), df["previous_action_t"].to_numpy().astype(int)
    off, on = prev == 0, prev == 1
    # Each threshold only affects its own branch, so optimise them independently.
    acc_start = [np.sum((b[off] >= g) == target[off]) for g in grid]
    acc_stay = [np.sum((b[on] > g) == target[on]) for g in grid]
    return {"start": float(grid[int(np.argmax(acc_start))]),
            "stay": float(grid[int(np.argmax(acc_stay))])}


def fit_burden_threshold(df: pd.DataFrame) -> float:
    grid = np.round(np.arange(0.5, 3.01, 0.01), 2)
    target, b = df["teacher_action"].to_numpy(), df["burden_t"].to_numpy()
    return float(grid[int(np.argmax([np.mean((b >= g) == target) for g in grid]))])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def f1(true: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum(true & pred)
    denom = 2 * tp + np.sum(true & ~pred) + np.sum(~true & pred)
    return float(2 * tp / denom) if denom else float("nan")


def score(df: pd.DataFrame, student_action: np.ndarray, student_p: np.ndarray | None = None) -> dict:
    """Fidelity of a student's decisions to the teacher on the rows of `df`."""
    p = df["p_on"].to_numpy()
    teacher = df["teacher_action"].to_numpy().astype(int)
    student = np.asarray(student_action).astype(int)
    prev = df["previous_action_t"].to_numpy().astype(int)
    teacher_switch, student_switch = teacher != prev, student != prev
    out = {
        "n": int(len(df)),
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "probability_MAE": float(np.mean(np.abs(p - student_p))) if student_p is not None else None,
        "hard_action_agreement": float(np.mean(teacher == student)),
        "balanced_accuracy": float(np.mean([np.mean(student[teacher == c] == c)
                                            for c in (0, 1) if np.any(teacher == c)])),
        "on_F1": f1(teacher == 1, student == 1),
        "switching_F1": f1(teacher_switch, student_switch),
        "switch_on_recall": float(np.mean(student[(prev == 0) & (teacher == 1)] == 1))
        if np.any((prev == 0) & (teacher == 1)) else None,
        "switch_off_recall": float(np.mean(student[(prev == 1) & (teacher == 0)] == 0))
        if np.any((prev == 1) & (teacher == 0)) else None,
        "teacher_on_fraction": float(np.mean(teacher)),
        "teacher_switch_fraction": float(np.mean(teacher_switch)),
        "student_on_fraction": float(np.mean(student)),
    }
    return out


def score_by_epsilon(df: pd.DataFrame, action: np.ndarray, p: np.ndarray | None) -> dict:
    out = {"all": score(df, action, p)}
    for eps in EXPLORATION_RATES:
        mask = np.isclose(df["epsilon"].to_numpy(), eps)
        if mask.any():
            out[f"epsilon_{eps:g}"] = score(df[mask], action[mask], None if p is None else p[mask])
    return out


# --------------------------------------------------------------------------
# Closed-loop evaluation: deploy the tree as the controller
# --------------------------------------------------------------------------
def tree_controller(tree: DecisionTreeRegressor, features: list[str]) -> Callable[[np.ndarray], int]:
    idx = [FEATURES.index(f) for f in features]
    return lambda x: int(hard_action(np.clip(tree.predict(x[idx].reshape(1, -1)), 0.0, 1.0))[0])


def hysteresis_controller(start: float, stay: float) -> Callable[[np.ndarray], int]:
    return lambda x: int(x[6] >= start) if x[7] < 0.5 else int(x[6] > stay)


def closed_loop(teacher: PPO, controllers: dict[str, Callable[[np.ndarray], int]], config_path: Path,
                mut_fractions: list[float], max_steps: int) -> tuple[pd.DataFrame, dict]:
    """Deploy greedy PPO and each student as the sole controller from the same
    initial conditions (no exploration) and compare schedules and outcomes."""
    rng = np.random.default_rng(0)  # unused with epsilon = 0
    rows = []
    for frac in mut_fractions:
        ppo_actions = None
        for name, ctrl in {"ppo": teacher_controller(teacher), **controllers}.items():
            env = make_env(config_path, frac)
            try:
                steps, outcome = rollout(env, ctrl, teacher, 0.0, rng, max_steps)
            finally:
                env.close()
            actions = np.array([r["executed_action"] for r in steps])
            p_on = np.array([r["p_on"] for r in steps])
            if name == "ppo":
                ppo_actions = actions
            n = min(len(actions), len(ppo_actions))
            rows.append({
                "controller": name, "initial_mut_fraction": frac,
                "steps": outcome["steps"], "progressed": outcome["terminated"],
                "return": outcome["return"], "on_fraction": float(actions.mean()),
                "n_switches": int(np.sum(np.diff(np.r_[0, actions]) != 0)),
                "final_burden": outcome["final_burden"],
                "final_resistant_fraction": outcome["final_resistant_fraction"],
                "schedule_agreement_with_ppo": float(np.mean(actions[:n] == ppo_actions[:n])),
                "first_divergence_step": int(np.argmax(actions[:n] != ppo_actions[:n]))
                if np.any(actions[:n] != ppo_actions[:n]) else None,
                # agreement with the teacher on the states this controller itself visits
                "teacher_agreement_on_own_states": float(np.mean(hard_action(p_on) == actions)),
                "schedule": "".join(map(str, actions)),
            })
    table = pd.DataFrame(rows)
    summary = {}
    for name, g in table.groupby("controller", sort=False):
        summary[name] = {
            "n_initial_conditions": int(len(g)),
            "progressed_fraction": float(g["progressed"].mean()),
            "mean_steps": float(g["steps"].mean()),
            "mean_return": float(g["return"].mean()),
            "mean_on_fraction": float(g["on_fraction"].mean()),
            "mean_schedule_agreement_with_ppo": float(g["schedule_agreement_with_ppo"].mean()),
            "mean_teacher_agreement_on_own_states": float(g["teacher_agreement_on_own_states"].mean()),
            "config_initial_condition": json.loads(g.iloc[[0]].to_json(orient="records"))[0],
        }
    return table, summary


# --------------------------------------------------------------------------
# Tree description and stability
# --------------------------------------------------------------------------
def tree_splits(tree: DecisionTreeRegressor, features: list[str]) -> list[dict]:
    t = tree.tree_
    out, stack = [], [(0, 0, "root")]
    while stack:
        node, depth, path = stack.pop()
        if t.children_left[node] == -1:
            continue
        out.append({"node": int(node), "depth": depth, "path": path,
                    "feature": features[t.feature[node]], "threshold": float(t.threshold[node])})
        stack.append((t.children_right[node], depth + 1, path + "R"))
        stack.append((t.children_left[node], depth + 1, path + "L"))
    return sorted(out, key=lambda s: (s["depth"], s["path"]))


def split_stability(train: pd.DataFrame, features: list[str], depth: int, min_leaf: float,
                    seed: int, resamples: int) -> dict:
    """Refit on trajectory-level bootstrap resamples of the training set and record
    how often each feature is chosen at the root / within the top two levels."""
    ids = train["trajectory_id"].unique()
    rng = np.random.default_rng(seed)
    root, top2, root_thr = [], [], []
    for _ in range(resamples):
        draw = pd.Series(rng.choice(ids, size=len(ids), replace=True)).value_counts()
        weight = train["trajectory_id"].map(draw).fillna(0).to_numpy(dtype=float)
        if weight.sum() == 0 or len(draw) < 2:
            continue
        tree = fit_tree(train, features, depth, min_leaf, seed, sample_weight=weight)
        splits = tree_splits(tree, features)
        if not splits:
            continue
        root.append(splits[0]["feature"])
        root_thr.append(splits[0]["threshold"])
        top2.append(sorted({s["feature"] for s in splits if s["depth"] <= 1}))
    n = len(root)
    feats = sorted(set(root) | {f for s in top2 for f in s})
    return {
        "n_resamples": n,
        "root_feature_frequency": {f: root.count(f) / n for f in sorted(set(root))} if n else {},
        "top_two_levels_feature_frequency": {f: sum(f in s for s in top2) / n for f in feats} if n else {},
        "root_threshold_quantiles": dict(zip(["q05", "q50", "q95"],
                                             map(float, np.quantile(root_thr, [0.05, 0.5, 0.95]))))
        if n else {},
    }


def plot_trees(trees: dict[str, tuple[DecisionTreeRegressor, list[str]]], path_stem: Path,
               title: str) -> None:
    n = len(trees)
    fig, axes = plt.subplots(n, 1, figsize=(18, 7.5 * n), squeeze=False)
    for ax, (name, (tree, features)) in zip(axes[:, 0], trees.items()):
        plot_tree(tree, feature_names=features, filled=True, rounded=True, precision=3,
                  fontsize=9, ax=ax, impurity=False, proportion=True)
        ax.set_title(f"{title} — {name} (leaf value = mean teacher P(ON))", fontsize=12)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".pdf"))
    fig.savefig(path_stem.with_suffix(".png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def split_groups(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Trajectory-grouped 60/20/20 train/validation/test split."""
    groups = df["trajectory_id"].to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trval, test = next(first.split(df, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    tr, val = next(second.split(df.iloc[trval], groups=groups[trval]))
    parts = {"train": df.iloc[trval[tr]], "validation": df.iloc[trval[val]], "test": df.iloc[test]}
    ids = [set(p["trajectory_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "Group leakage"
    return parts


def uniqueness(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "episodes": int(df["episode"].nunique()),
        "unique_trajectories": int(df["trajectory_id"].nunique()),
        "unique_observations": int(df[FEATURES].round(6).drop_duplicates().shape[0]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="SB3 PPO checkpoint (.zip or extracted save directory).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="LV env config used to generate trajectories (read only).")
    parser.add_argument("--label", default=None, help="Name for this teacher (output subfolder).")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Default: outputs/policy_distillation/<label>.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--robustness-episodes", type=int, default=100)
    parser.add_argument("--mut-fraction-range", type=float, nargs=2, default=(1e-4, 3e-3))
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=3, help="Pre-specified primary depth.")
    parser.add_argument("--min-leaf-fraction", type=float, default=0.02)
    parser.add_argument("--stability-resamples", type=int, default=100)
    parser.add_argument("--closed-loop-ics", type=int, default=20,
                        help="Number of robustness initial conditions for closed-loop deployment.")
    parser.add_argument("--exclude-padded-steps", action="store_true",
                        help="Sensitivity analysis: drop the first 3 decisions, whose stacked "
                             "history is zero-padded by VecFrameStack.")
    args = parser.parse_args(argv)
    if args.episodes < 12 or args.depth < 1 or not 0 < args.min_leaf_fraction < 0.5:
        parser.error("Need >=12 episodes, depth>=1 and 0<min-leaf-fraction<0.5")
    if args.label is None:
        args.label = args.model.stem if args.model.suffix == ".zip" else args.model.name
    if args.outdir is None:
        args.outdir = MINIREPO / "outputs" / "policy_distillation" / args.label
    return args


def run(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(args.model)

    primary = generate_dataset(teacher, args.config, args.episodes, args.seed, args.max_steps, "primary")
    robust = generate_dataset(teacher, args.config, args.robustness_episodes, args.seed,
                              args.max_steps, "robustness", tuple(args.mut_fraction_range))
    # Reference (all episodes, before de-duplication): how often does PPO itself
    # progress when a fraction epsilon of its actions is flipped?
    episode_outcomes = primary.groupby("episode").agg(
        epsilon=("epsilon", "first"), progressed=("episode_terminated", "first"))
    robustness_outcomes = robust.groupby("episode").agg(
        epsilon=("epsilon", "first"), progressed=("episode_terminated", "first"))
    perturbation_reference = {
        "primary": episode_outcomes.groupby("epsilon")["progressed"].mean().to_dict(),
        "robustness": robustness_outcomes.groupby("epsilon")["progressed"].mean().to_dict(),
    }

    # The LV model is deterministic: identical executed-action sequences give
    # identical trajectories. Keep one copy of each so duplicates neither inflate
    # a split nor weight the fit; record how often each trajectory occurred.
    duplicates = {}
    for name in ("primary", "robust"):
        frame = primary if name == "primary" else robust
        first = frame.groupby("trajectory_id")["episode"].min()
        mult = frame.groupby("trajectory_id")["episode"].nunique()
        frame = frame[frame["episode"].isin(first.values)].copy()
        frame["trajectory_multiplicity"] = frame["trajectory_id"].map(mult)
        duplicates[name] = int(mult.sum() - len(mult))
        if name == "primary":
            primary = frame
        else:
            robust = frame
    if args.exclude_padded_steps:
        primary = primary[primary["step"] >= N_STACK - 1].copy()
        robust = robust[robust["step"] >= N_STACK - 1].copy()
    if len(primary) < 100:
        raise RuntimeError("Too few observations; inspect the environment rollout.")
    parts = split_groups(primary, args.seed)
    parts["robustness_test"] = robust
    split_of = {tid: name for name in ("train", "validation", "test")
                for tid in parts[name]["trajectory_id"].unique()}
    primary["split"] = primary["trajectory_id"].map(split_of)
    robust["split"] = "robustness_test"
    pd.concat([primary, robust], ignore_index=True).to_csv(
        args.outdir / "observations_and_teacher_labels.csv", index=False)

    train, val = parts["train"], parts["validation"]
    min_leaf = args.min_leaf_fraction

    # Depth sweep on validation only. The primary depth is fixed a priori.
    sweep = []
    for name, feats in FEATURE_SETS.items():
        for depth in range(1, 7):
            tree = fit_tree(train, feats, depth, min_leaf, args.seed)
            p = tree_predict(tree, val, feats)
            s = score(val, hard_action(p), p)
            sweep.append({"feature_set": name, "depth": depth, "n_leaves": int(tree.get_n_leaves()),
                          "val_probability_MAE": s["probability_MAE"],
                          "val_hard_action_agreement": s["hard_action_agreement"],
                          "val_switching_F1": s["switching_F1"]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(args.outdir / "depth_selection_validation.csv", index=False)
    cand = sweep[(sweep.feature_set == "all_features") & sweep.depth.between(2, 4)]
    selected_depth = int(cand.sort_values(["val_probability_MAE", "depth"]).iloc[0]["depth"])

    depths = sorted({args.depth, selected_depth})
    trees = {(name, d): fit_tree(train, feats, d, min_leaf, args.seed)
             for name, feats in FEATURE_SETS.items() for d in depths}

    hyst = fit_hysteresis_rule(train)
    thr = fit_burden_threshold(train)
    majority = int(train["teacher_action"].mean() > 0.5)

    evaluation = {}
    for split_name in ("test", "robustness_test"):
        d = parts[split_name]
        res = {}
        for (name, depth), tree in trees.items():
            p = tree_predict(tree, d, FEATURE_SETS[name])
            res[f"tree_{name}_depth{depth}"] = score_by_epsilon(d, hard_action(p), p)
        res["rule_hysteresis_fitted"] = score_by_epsilon(d, hysteresis_rule(d, **hyst), None)
        res["rule_burden_threshold_fitted"] = score_by_epsilon(
            d, (d["burden_t"].to_numpy() >= thr).astype(int), None)
        res["constant_majority_action"] = score_by_epsilon(d, np.full(len(d), majority), None)
        evaluation[split_name] = res

    primary_tree = trees[("all_features", args.depth)]
    burden_tree = trees[("burden_only", args.depth)]
    stability = {name: split_stability(train, FEATURE_SETS[name], args.depth, min_leaf, args.seed,
                                       args.stability_resamples)
                 for name in ("all_features", "burden_only")}

    rng = np.random.default_rng([args.seed, 99])
    lo, hi = np.log10(args.mut_fraction_range[0]), np.log10(args.mut_fraction_range[1])
    ics = [default_initial_mut_fraction(args.config)] + list(
        10 ** rng.uniform(lo, hi, size=max(args.closed_loop_ics - 1, 0)))
    controllers = {f"tree_{name}_depth{depth}": tree_controller(tree, FEATURE_SETS[name])
                   for (name, depth), tree in trees.items()}
    controllers.update({f"tree_all_features_depth{d}": tree_controller(
        fit_tree(train, FEATURES, d, min_leaf, args.seed), FEATURES) for d in range(1, 7)})
    controllers["rule_hysteresis_fitted"] = hysteresis_controller(**hyst)
    controllers["rule_burden_threshold_fitted"] = lambda x: int(x[6] >= thr)
    closed_table, closed = closed_loop(teacher, controllers, args.config, ics, args.max_steps)
    closed_table.to_csv(args.outdir / "closed_loop.csv", index=False)

    rules = []
    for (name, depth), tree in trees.items():
        rules.append(f"### {name}, depth {depth} (fit on train split, target = teacher P(ON))\n"
                     + export_text(tree, feature_names=FEATURE_SETS[name], decimals=3,
                                   show_weights=False))
    rules.append(f"### fitted hysteresis rule: ON if (prev OFF and burden_t >= {hyst['start']}) "
                 f"or (prev ON and burden_t > {hyst['stay']})")
    rules.append(f"### fitted burden-only threshold rule: ON if burden_t >= {thr}")
    (args.outdir / "tree_rules.txt").write_text("\n\n".join(rules) + "\n", encoding="utf-8")

    plot_trees({"all features": (primary_tree, FEATURE_SETS["all_features"]),
                "burden only": (burden_tree, FEATURE_SETS["burden_only"])},
               args.outdir / "decision_tree", f"{args.label}, depth {args.depth}")

    report = {
        "method": "CART regression distillation of PPO P(ON) (not VIPER)",
        "label": args.label,
        "model": str(args.model.resolve().relative_to(MINIREPO.parent) if args.model.resolve().is_relative_to(MINIREPO.parent) else args.model),
        "model_sha256": sha256_path(args.model),
        "policy_weights_sha256": policy_weights_sha256(args.model),
        "config": str(args.config.resolve().relative_to(MINIREPO.parent) if args.config.resolve().is_relative_to(MINIREPO.parent) else args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__,
                     "gymnasium": gymnasium.__version__, "scikit_learn": sklearn.__version__},
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "exploration_rates": list(EXPLORATION_RATES),
        "data": {name: uniqueness(part) for name, part in parts.items()},
        "duplicate_episodes_removed": duplicates,
        "primary_depth": args.depth,
        "validation_selected_depth_2to4": selected_depth,
        "fitted_rules": {"hysteresis": hyst, "burden_threshold": thr, "majority_action": majority},
        "primary_tree_splits": tree_splits(primary_tree, FEATURE_SETS["all_features"]),
        "burden_only_tree_splits": tree_splits(burden_tree, FEATURE_SETS["burden_only"]),
        "evaluation": evaluation,
        "split_stability": stability,
        "closed_loop": closed,
        "ppo_progression_fraction_by_epsilon": perturbation_reference,
    }
    (args.outdir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def headline(report: dict) -> str:
    d = report["primary_depth"]
    lines = [f"Teacher: {report['label']}  (policy.pth sha256 {report['policy_weights_sha256'][:12]})",
             f"Data: {report['data']}", "",
             f"{'student':<40} {'split':<16} {'MAE':>6} {'agree':>6} {'balAcc':>6} {'swF1':>6}"]
    for split in ("test", "robustness_test"):
        for name, res in report["evaluation"][split].items():
            if "depth" in name and f"depth{d}" not in name:
                continue
            a = res["all"]
            mae = f"{a['probability_MAE']:.3f}" if a["probability_MAE"] is not None else "  -  "
            lines.append(f"{name:<40} {split:<16} {mae:>6} {a['hard_action_agreement']:.3f} "
                         f"{a['balanced_accuracy']:.3f} {a['switching_F1']:.3f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = run(args)
    print(headline(report))
    print("\n" + (args.outdir / "tree_rules.txt").read_text(encoding="utf-8"))
    print("Saved to", args.outdir)


if __name__ == "__main__":
    main()
