"""
Behavioral analysis for trained PPO melanoma treatment policies.

This script probes trained Stable-Baselines3 PPO policies directly with
synthetic 4-step observation histories. The primary behavioral output is now
the hard deterministic action chosen by the policy, i.e. the action that would
be used for a deterministic treatment schedule. The exact policy probability
P(treatment = ON | observation) is still saved as a diagnostic column.

This distinction is intentional:
    - probabilities describe the smooth learned policy distribution
    - hard actions describe the deployed deterministic decision rule
    - repeated deterministic decisions do not estimate probabilities; they
      only repeat the same argmax decision

The script does not retrain agents and does not modify saved checkpoints.

Observation construction
------------------------
The repository's LV PPO agents were trained on a 4-frame VecFrameStack of a
2-value base observation:
    [total_tumor_burden, previous_action]

Runtime inspection in this repo confirmed that the stacked observation is
ordered from oldest to newest frame, flattened as:
    [burden_t-3, action_t-3,
     burden_t-2, action_t-2,
     burden_t-1, action_t-1,
     burden_t,   action_t]

The analysis therefore constructs synthetic observations in exactly that order.

Growth construction
-------------------
This version uses the single-peak history proposed in the companion RL-agent
exploration code:
    - start with a flat 4-step burden history at `base`
    - introduce one isolated peak of height `base + peak_addition`
    - `peak_day` in {0, 1, 2, 3} is the frame index in the stacked history
      peak_day = 0 -> oldest frame
      peak_day = 3 -> newest / most recent frame

The derived scalar follows the colleague's definition:
    growth = base + peak_day * peak_addition

This is not equivalent to the previous persistent-jump construction. It
compresses peak magnitude and recency into one display axis, so grouped curves
are binned over growth before plotting.

Scaling / normalization
-----------------------
The trained LV agents expect normalized tumor burden values. The learned policy
was trained on normalized observations with a burden-space upper bound of about
3, but this script defaults to probing `base` and `peak_addition` from 0 to 5
because the current use case is explicit stress testing. The constructed burden
history is still clipped to the model's observation-space bounds before being
fed into the policy.

How to run
----------
From the project root:

    python PhysiLearning/Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py --mode experimental
    python PhysiLearning/Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py --mode bootstrap
    python PhysiLearning/Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py --mode both

Optional paths:

    python PhysiLearning/Agent_Behaviour_Analysis/rl_policy_behavior_analysis.py ^
        --mode both ^
        --experimental-agent-path PhysiLearning/Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward ^
        --bootstrap-agent-dir PhysiLearning/Bootstrapping/outputs/bootstrap_n20_methods_seed_maxiter1000/agent_training/agt_20260605_102549/models ^
        --output-dir PhysiLearning/Agent_Behaviour_Analysis/outputs/rl_policy_behavior

Outputs
-------
The script writes:
    - CSV files for experimental-agent results
    - CSV files for per-bootstrap-agent results
    - a bootstrap summary CSV with mean / SD / SEM
    - publication-style PNG/PDF/SVG plots

By default, CSV source data and figures are written to the canonical Figure 4
folders on the shared data drive, rather than into this repository.

Important repository-specific note
----------------------------------
The default experimental checkpoint path is inferred from the repository folder
name `PhysiLearning/Agent_used_for_experiment/...`. The repository does not
contain an explicit provenance file stating that this is the exact
schedule-generating agent, so that association should still be verified by a
human before publication-critical use.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.data_paths import MODELS, figure_data_dir, figure_plot_dir


STACK_LENGTH = 4
BASE_OBS_DIM = 2
EXPECTED_OBS_SHAPE = (STACK_LENGTH * BASE_OBS_DIM,)
TREATMENT_ON_ACTION = 1

DEFAULT_BASE_MIN = 0.0
DEFAULT_BASE_MAX = 5.0
DEFAULT_PEAK_ADDITION_MIN = 0.0
DEFAULT_PEAK_ADDITION_MAX = 5.0
DEFAULT_PEAK_DAY_VALUES = (0, 1, 2, 3)
DEFAULT_N_GROWTH_BINS = 14
AGGREGATE_GROUP_ORDER = (
    "[0,0,0,0]",
    "[0,0,0,1]",
    "[0,0,1,1]",
    "[0,1,1,1]",
    "[1,1,1,1]",
)
GROWTH_XLIM = (0.0, 20.0)
GROWTH_XTICKS = [0, 5, 10, 15, 20]


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    agent_code: str
    agent_type: str
    agent_group: str
    path: Path


def parse_args() -> argparse.Namespace:
    repo_root = infer_repo_root()
    default_bootstrap_dir = infer_default_bootstrap_models_dir(repo_root)

    parser = argparse.ArgumentParser(
        description="Probe trained PPO melanoma policies with synthetic observation histories."
    )
    parser.add_argument(
        "--mode",
        choices=("experimental", "bootstrap", "both"),
        default="both",
        help="Which agent set to analyze.",
    )
    parser.add_argument(
        "--experimental-agent-path",
        type=Path,
        default=MODELS / 'experimental_agent',
        help="Path to the experimental SB3 checkpoint. Can be a .zip file or an extracted SB3 save directory.",
    )
    parser.add_argument(
        "--bootstrap-agent-dir",
        type=Path,
        default=default_bootstrap_dir,
        help="Directory containing bootstrap agent subdirectories such as b0000/.../b0000_s2_best_reward.zip.",
    )
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        default=figure_data_dir("Figure_4", "rl_policy_behavior", create=False),
        help="Canonical Figure 4 directory for CSV source data.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=figure_plot_dir("Figure_4", "rl_policy_behavior", create=False),
        help="Canonical Figure 4 directory for PNG, PDF, and SVG figures.",
    )
    parser.add_argument(
        "--n-base",
        type=int,
        default=11,
        help="Number of base burden values between the configured min/max.",
    )
    parser.add_argument(
        "--n-p-addition",
        "--n-peak-addition",
        dest="n_p_addition",
        type=int,
        default=11,
        help="Number of peak-addition values between the configured min/max.",
    )
    parser.add_argument(
        "--n-growth-bins",
        type=int,
        default=DEFAULT_N_GROWTH_BINS,
        help=(
            "Number of bins used when collapsing the colleague-style growth "
            "score into grouped curves."
        ),
    )
    parser.add_argument(
        "--action-history-mode",
        choices=("extremes", "representative", "all"),
        default="all",
        help="Which exact binary action histories to probe.",
    )
    return parser.parse_args()


def infer_repo_root() -> Path:
    this_file = Path(__file__).resolve()
    if this_file.parent.name == "Agent_Behaviour_Analysis":
        return this_file.parents[2]
    return this_file.parents[1]


def infer_default_bootstrap_models_dir(repo_root: Path) -> Path:
    return MODELS / 'bootstrap'


def historical_bootstrap_models_dir(repo_root: Path) -> Path:
    agent_training_root = (
        repo_root
        / "PhysiLearning"
        / "Bootstrapping"
        / "outputs"
        / "bootstrap_n20_methods_seed_maxiter1000"
        / "agent_training"
    )
    candidates = sorted(
        [path for path in agent_training_root.glob("agt_*") if (path / "models").is_dir()]
    )
    if not candidates:
        return agent_training_root / "agt_MISSING" / "models"
    return candidates[-1] / "models"


def select_action_histories(mode: str) -> list[tuple[int, int, int, int]]:
    if mode == "extremes":
        return [(0, 0, 0, 0), (1, 1, 1, 1)]
    if mode == "representative":
        return [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (1, 1, 1, 1)]
    return [tuple(int(bit) for bit in f"{value:04b}") for value in range(16)]


def format_action_history(action_history: Sequence[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in action_history) + "]"


def action_history_ends_with(action_history_str: str, value: int) -> bool:
    return action_history_str.endswith(f",{value}]")


def add_action_history_groups(df: pd.DataFrame) -> pd.DataFrame:
    grouped_frames: list[pd.DataFrame] = []

    for action_history in AGGREGATE_GROUP_ORDER:
        exact_history = df.loc[df["action_history"] == action_history].copy()
        if not exact_history.empty:
            exact_history["action_history_group"] = action_history
            grouped_frames.append(exact_history)

    if not grouped_frames:
        return pd.DataFrame(columns=[*df.columns, "action_history_group"])

    result = pd.concat(grouped_frames, ignore_index=True)
    result["action_history_group"] = pd.Categorical(
        result["action_history_group"],
        categories=list(AGGREGATE_GROUP_ORDER),
        ordered=True,
    )
    return result


def infer_manifest_path_from_models_dir(models_dir: Path) -> Path | None:
    bundled = models_dir / 'agent_training_manifest.csv'
    if bundled.is_file():
        return bundled
    parent = models_dir.parent
    manifest = parent / "agent_training_manifest.csv"
    if manifest.exists():
        return manifest
    return None


def discover_bootstrap_agents(models_dir: Path) -> list[AgentSpec]:
    manifest_path = infer_manifest_path_from_models_dir(models_dir)
    agents: list[AgentSpec] = []

    if manifest_path and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        for row in manifest.itertuples(index=False):
            model_path = models_dir / row.agent_code / f"{row.agent_code}_s2_best_reward.zip"
            agents.append(
                AgentSpec(
                    agent_id=str(row.agent_id),
                    agent_code=str(row.agent_code),
                    agent_type="bootstrap_ensemble",
                    agent_group=str(row.group),
                    path=model_path,
                )
            )
        return agents

    for model_path in sorted(models_dir.glob("*/*_s2_best_reward.zip")):
        agent_code = model_path.parent.name
        if agent_code.startswith("b"):
            agent_group = "bootstrap_lv"
        elif agent_code.startswith("mr"):
            agent_group = "methods_repeat"
        elif agent_code.startswith("mc"):
            agent_group = "methods_control"
        else:
            agent_group = "unknown"
        agents.append(
            AgentSpec(
                agent_id=agent_code,
                agent_code=agent_code,
                agent_type="bootstrap_ensemble",
                agent_group=agent_group,
                path=model_path,
            )
        )
    return agents


@contextmanager
def materialize_sb3_checkpoint(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    expected_members = {
        "data",
        "policy.pth",
        "policy.optimizer.pth",
        "pytorch_variables.pth",
        "_stable_baselines3_version",
        "system_info.txt",
    }
    actual_members = {member.name for member in path.iterdir() if member.is_file()}
    missing = expected_members - actual_members
    if missing:
        raise FileNotFoundError(
            f"Extracted SB3 checkpoint directory is incomplete: {path}. Missing: {sorted(missing)}"
        )

    with tempfile.TemporaryDirectory(prefix="sb3_agent_") as tmp_dir:
        zip_path = Path(tmp_dir) / f"{path.name}.zip"
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in sorted(path.iterdir()):
                if member.is_file():
                    archive.write(member, arcname=member.name)
        yield zip_path


def load_ppo_model(path: Path) -> PPO:
    custom_objects = {
        "clip_range": lambda _: 0.0,
        "lr_schedule": lambda _: 0.0,
    }
    with materialize_sb3_checkpoint(path) as load_path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PPO.load(str(load_path), device="cpu", custom_objects=custom_objects)
    validate_model_interface(model, path)
    return model


def validate_model_interface(model: PPO, source_path: Path) -> None:
    if tuple(model.observation_space.shape) != EXPECTED_OBS_SHAPE:
        raise ValueError(
            f"Unexpected observation shape for {source_path}: "
            f"{model.observation_space.shape}. Expected {EXPECTED_OBS_SHAPE}. "
            "Update the observation constructor before using this script."
        )
    if getattr(model.action_space, "n", None) != 2:
        raise ValueError(
            f"Unexpected action space for {source_path}: {model.action_space}. "
            "This script expects a binary discrete policy."
        )


def get_burden_bounds(model: PPO) -> tuple[float, float]:
    low = np.asarray(model.observation_space.low, dtype=np.float32)
    high = np.asarray(model.observation_space.high, dtype=np.float32)
    burden_low = float(np.min(low[0::2]))
    burden_high = float(np.max(high[0::2]))
    return burden_low, burden_high


def build_cell_number_history(
    base: float,
    peak_day: int,
    peak_addition: float,
    burden_low: float,
    burden_high: float,
) -> np.ndarray:
    history = np.full(STACK_LENGTH, base, dtype=np.float32)

    # Colleague-style single-peak probe: one frame is elevated above the
    # constant base. This tests sensitivity to an isolated historical peak
    # rather than to a persistent growth step. Index 0 is the oldest frame and
    # index 3 is the newest frame, matching the flattened VecFrameStack order.
    history[int(peak_day)] += peak_addition

    return np.clip(history, burden_low, burden_high)


def build_stacked_observation(
    cell_number_history: Sequence[float],
    action_history: Sequence[int],
) -> np.ndarray:
    if len(cell_number_history) != STACK_LENGTH or len(action_history) != STACK_LENGTH:
        raise ValueError("Both cell_number_history and action_history must have length 4.")

    obs = np.empty(EXPECTED_OBS_SHAPE[0], dtype=np.float32)

    # Verified runtime order for VecFrameStack(n_stack=4, channels_order='last'):
    # [burden_t-3, action_t-3, burden_t-2, action_t-2, burden_t-1, action_t-1, burden_t, action_t]
    for idx, (burden, action) in enumerate(zip(cell_number_history, action_history)):
        obs[2 * idx] = float(burden)
        obs[2 * idx + 1] = float(action)

    return obs


def extract_policy_outputs(model: PPO, obs_vector: np.ndarray) -> tuple[float, int]:
    """Return exact ON probability plus the deterministic hard action.

    The hard action is computed as the argmax of the categorical policy
    probabilities. That is the quantity used by deterministic deployment and
    by `model.predict(..., deterministic=True)`. Repeating that deterministic
    decision would still yield only 0 or 1, so the probability is kept as a
    separate diagnostic rather than estimated by repeated hard decisions.
    """
    obs_batch = np.asarray(obs_vector, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        distribution = model.policy.get_distribution(obs_tensor)
        base_distribution = getattr(distribution, "distribution", distribution)
        probs = getattr(base_distribution, "probs", None)
        if probs is None:
            raise RuntimeError(
                "Could not extract categorical probabilities from PPO policy distribution."
            )
        probs = probs.detach().cpu().numpy()

    if probs.shape != (1, 2):
        raise RuntimeError(f"Unexpected policy probability shape: {probs.shape}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError(f"Policy probabilities do not sum to 1: {probs}")

    treatment_probability = float(probs[0, TREATMENT_ON_ACTION])
    chosen_action = int(np.argmax(probs[0]))
    return treatment_probability, chosen_action


def analyze_agent(
    agent: AgentSpec,
    action_histories: Sequence[Sequence[int]],
    base_values: np.ndarray,
    peak_addition_values: np.ndarray,
) -> pd.DataFrame:
    model = load_ppo_model(agent.path)
    burden_low, burden_high = get_burden_bounds(model)
    total_probes = (
        len(action_histories)
        * len(base_values)
        * len(DEFAULT_PEAK_DAY_VALUES)
        * len(peak_addition_values)
    )

    rows: list[dict[str, object]] = []
    with tqdm(
        total=total_probes,
        desc=f"{agent.agent_code} probes",
        leave=False,
        dynamic_ncols=True,
    ) as probe_bar:
        for action_history in action_histories:
            action_history_str = format_action_history(action_history)
            probe_bar.set_postfix_str(f"history={action_history_str}")
            for base in base_values:
                for peak_day in DEFAULT_PEAK_DAY_VALUES:
                    for peak_addition in peak_addition_values:
                        cell_history = build_cell_number_history(
                            base=base,
                            peak_day=peak_day,
                            peak_addition=peak_addition,
                            burden_low=burden_low,
                            burden_high=burden_high,
                        )
                        obs = build_stacked_observation(
                            cell_number_history=cell_history,
                            action_history=action_history,
                        )
                        probability, chosen_action = extract_policy_outputs(model, obs)
                        growth = float(base + peak_day * peak_addition)
                        rows.append(
                            {
                                "agent_id": agent.agent_id,
                                "agent_code": agent.agent_code,
                                "agent_type": agent.agent_type,
                                "agent_group": agent.agent_group,
                                "action_history": action_history_str,
                                "base": float(base),
                                "peak_day": int(peak_day),
                                "peak_addition": float(peak_addition),
                                "growth": growth,
                                "chosen_action": chosen_action,
                                "treatment_probability": probability,
                            }
                        )
                        probe_bar.update(1)
    return pd.DataFrame(rows)


def aggregate_bootstrap_summary(all_bootstrap_results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        all_bootstrap_results.groupby(
            ["action_history", "base", "peak_day", "peak_addition", "growth"],
            as_index=False,
        )
        .agg(
            chosen_action_count=("chosen_action", "count"),
            fraction_chosen_treatment_on_mean=("chosen_action", "mean"),
            fraction_chosen_treatment_on_sd=("chosen_action", "std"),
            treatment_probability_mean=("treatment_probability", "mean"),
            treatment_probability_sd=("treatment_probability", "std"),
        )
    )
    grouped = grouped.rename(columns={"chosen_action_count": "n_agents"})
    grouped["fraction_chosen_treatment_on_sem"] = (
        grouped["fraction_chosen_treatment_on_sd"] / np.sqrt(grouped["n_agents"].clip(lower=1))
    )
    grouped["treatment_probability_sem"] = (
        grouped["treatment_probability_sd"] / np.sqrt(grouped["n_agents"].clip(lower=1))
    )
    return grouped


def add_growth_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Collapse the colleague-style scalar growth score into display bins.

    The single-peak construction maps many different `(base, peak_day,
    peak_addition)` combinations onto a nonuniform scalar axis. Binning keeps
    the line plots readable while preserving the colleague's growth definition
    in the raw CSV output.
    """
    if n_bins < 1:
        raise ValueError("n_growth_bins must be at least 1.")

    result = df.copy()
    growth = result["growth"].to_numpy(dtype=float)
    growth_min = float(np.nanmin(growth))
    growth_max = float(np.nanmax(growth))
    if np.isclose(growth_min, growth_max):
        result["growth_bin"] = 0
        result["growth"] = growth_min
        result["growth_bin_left"] = growth_min
        result["growth_bin_right"] = growth_max
        return result

    edges = np.linspace(growth_min, growth_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.searchsorted(edges, growth, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    result["growth_bin"] = bin_index
    result["growth"] = centers[bin_index]
    result["growth_bin_left"] = edges[bin_index]
    result["growth_bin_right"] = edges[bin_index + 1]
    return result


def collapse_experimental_for_growth_groups(
    experimental_results: pd.DataFrame,
    n_growth_bins: int,
) -> pd.DataFrame:
    grouped = add_action_history_groups(experimental_results)
    grouped = add_growth_bins(grouped, n_growth_bins)
    return (
        grouped.groupby(
            ["action_history_group", "growth", "growth_bin", "growth_bin_left", "growth_bin_right"],
            observed=True,
            as_index=False,
        )
        .agg(
            n_decisions=("chosen_action", "count"),
            fraction_chosen_treatment_on=("chosen_action", "mean"),
            treatment_probability_mean=("treatment_probability", "mean"),
        )
        .sort_values(["action_history_group", "growth"])
    )


def summarize_bootstrap_for_growth_groups(
    all_bootstrap_results: pd.DataFrame,
    n_growth_bins: int,
) -> pd.DataFrame:
    grouped = add_action_history_groups(all_bootstrap_results)
    grouped = add_growth_bins(grouped, n_growth_bins)
    per_agent_curve = (
        grouped.groupby(
            [
                "agent_id",
                "action_history_group",
                "growth",
                "growth_bin",
                "growth_bin_left",
                "growth_bin_right",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            fraction_chosen_treatment_on=("chosen_action", "mean"),
            treatment_probability_mean=("treatment_probability", "mean"),
        )
        .sort_values(["agent_id", "action_history_group", "growth"])
    )
    summary = (
        per_agent_curve.groupby(
            ["action_history_group", "growth", "growth_bin", "growth_bin_left", "growth_bin_right"],
            observed=True,
            as_index=False,
        )
        .agg(
            n_agents=("fraction_chosen_treatment_on", "count"),
            fraction_chosen_treatment_on_mean=("fraction_chosen_treatment_on", "mean"),
            fraction_chosen_treatment_on_sd=("fraction_chosen_treatment_on", "std"),
            treatment_probability_mean=("treatment_probability_mean", "mean"),
            treatment_probability_sd=("treatment_probability_mean", "std"),
        )
    )
    summary["fraction_chosen_treatment_on_sem"] = (
        summary["fraction_chosen_treatment_on_sd"] / np.sqrt(summary["n_agents"].clip(lower=1))
    )
    summary["treatment_probability_sem"] = (
        summary["treatment_probability_sd"] / np.sqrt(summary["n_agents"].clip(lower=1))
    )
    return summary


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 5.5),
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.facecolor": "#fbfbf8",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
        }
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight")


def plot_experimental_curves(
    experimental_results: pd.DataFrame,
    output_dir: Path,
    n_growth_bins: int,
) -> None:
    curve_df = collapse_experimental_for_growth_groups(experimental_results, n_growth_bins)
    fig, ax = plt.subplots()
    style_map = {
        "[0,0,0,0]": {"color": "#1f4e79", "linestyle": "-", "linewidth": 3.0},
        "[0,0,0,1]": {"color": "#4c9f70", "linestyle": "--", "linewidth": 2.4},
        "[0,0,1,1]": {"color": "#d97b29", "linestyle": "--", "linewidth": 2.4},
        "[0,1,1,1]": {"color": "#6a4c93", "linestyle": "--", "linewidth": 2.4},
        "[1,1,1,1]": {"color": "#9b2226", "linestyle": "-", "linewidth": 3.0},
    }
    histories = [group for group in AGGREGATE_GROUP_ORDER if group in set(curve_df["action_history_group"])]
    for history in histories:
        subset = curve_df.loc[curve_df["action_history_group"] == history]
        ax.plot(
            subset["growth"],
            subset["fraction_chosen_treatment_on"],
            label=history,
            **style_map[history],
        )
    ax.set_title("Experimental Agent Chosen Actions")
    ax.set_xlabel("Growth score")
    ax.set_ylabel("Fraction choosing treatment ON")
    ax.set_xlim(*GROWTH_XLIM)
    ax.set_xticks(GROWTH_XTICKS)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="History group", ncol=1)
    save_figure(fig, output_dir / "experimental_policy_behavior")
    plt.close(fig)


def plot_bootstrap_curves(
    bootstrap_results: pd.DataFrame,
    output_dir: Path,
    n_growth_bins: int,
) -> None:
    summary = summarize_bootstrap_for_growth_groups(bootstrap_results, n_growth_bins)
    fig, ax = plt.subplots()
    style_map = {
        "[0,0,0,0]": {"color": "#1f4e79", "linestyle": "-", "linewidth": 3.0},
        "[0,0,0,1]": {"color": "#4c9f70", "linestyle": "--", "linewidth": 2.4},
        "[0,0,1,1]": {"color": "#d97b29", "linestyle": "--", "linewidth": 2.4},
        "[0,1,1,1]": {"color": "#6a4c93", "linestyle": "--", "linewidth": 2.4},
        "[1,1,1,1]": {"color": "#9b2226", "linestyle": "-", "linewidth": 3.0},
    }
    histories = [group for group in AGGREGATE_GROUP_ORDER if group in set(summary["action_history_group"])]
    for history in histories:
        subset = summary.loc[summary["action_history_group"] == history]
        style = style_map[history]
        ax.plot(
            subset["growth"],
            subset["fraction_chosen_treatment_on_mean"],
            label=history,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax.fill_between(
            subset["growth"],
            subset["fraction_chosen_treatment_on_mean"] - subset["fraction_chosen_treatment_on_sd"],
            subset["fraction_chosen_treatment_on_mean"] + subset["fraction_chosen_treatment_on_sd"],
            color=style["color"],
            alpha=0.18,
        )
    ax.set_title("Bootstrap Agent Chosen Actions")
    ax.set_xlabel("Growth score")
    ax.set_ylabel("Fraction choosing treatment ON")
    ax.set_xlim(*GROWTH_XLIM)
    ax.set_xticks(GROWTH_XTICKS)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="History group", ncol=1)
    save_figure(fig, output_dir / "bootstrap_policy_behavior")
    plt.close(fig)


def plot_comparison(
    experimental_results: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    output_dir: Path,
    n_growth_bins: int,
) -> None:
    exp_curve = collapse_experimental_for_growth_groups(experimental_results, n_growth_bins)
    boot_curve = summarize_bootstrap_for_growth_groups(bootstrap_results, n_growth_bins)

    fig, ax = plt.subplots()
    style_map = {
        "[0,0,0,0]": {"color": "#1f4e79"},
        "[0,0,0,1]": {"color": "#4c9f70"},
        "[0,0,1,1]": {"color": "#d97b29"},
        "[0,1,1,1]": {"color": "#6a4c93"},
        "[1,1,1,1]": {"color": "#9b2226"},
    }
    histories = sorted(
        set(exp_curve["action_history_group"].drop_duplicates()).intersection(
            boot_curve["action_history_group"].drop_duplicates()
        )
    )

    histories = [group for group in AGGREGATE_GROUP_ORDER if group in histories]
    for history in histories:
        color = style_map[history]["color"]
        exp_subset = exp_curve.loc[exp_curve["action_history_group"] == history]
        boot_subset = boot_curve.loc[boot_curve["action_history_group"] == history]

        ax.plot(
            exp_subset["growth"],
            exp_subset["fraction_chosen_treatment_on"],
            color=color,
            linestyle="-",
            label=f"{history} experimental",
        )
        ax.plot(
            boot_subset["growth"],
            boot_subset["fraction_chosen_treatment_on_mean"],
            color=color,
            linestyle="--",
            label=f"{history} bootstrap mean",
        )
        ax.fill_between(
            boot_subset["growth"],
            boot_subset["fraction_chosen_treatment_on_mean"] - boot_subset["fraction_chosen_treatment_on_sd"],
            boot_subset["fraction_chosen_treatment_on_mean"] + boot_subset["fraction_chosen_treatment_on_sd"],
            color=color,
            alpha=0.15,
        )

    ax.set_title("Experimental vs Bootstrap Chosen Actions")
    ax.set_xlabel("Growth score")
    ax.set_ylabel("Fraction choosing treatment ON")
    ax.set_xlim(*GROWTH_XLIM)
    ax.set_xticks(GROWTH_XTICKS)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=2 if len(histories) > 2 else 1)
    save_figure(fig, output_dir / "experimental_vs_bootstrap_policy_behavior")
    plt.close(fig)


def ensure_mode_paths(args: argparse.Namespace) -> None:
    if args.mode in {"experimental", "both"} and not args.experimental_agent_path.exists():
        raise FileNotFoundError(
            "Experimental agent checkpoint was not found. "
            f"Expected: {args.experimental_agent_path}"
        )
    if args.mode in {"bootstrap", "both"} and not args.bootstrap_agent_dir.exists():
        raise FileNotFoundError(
            "Bootstrap agent directory was not found. "
            f"Expected: {args.bootstrap_agent_dir}"
        )


def print_discovery_summary(
    experimental_path: Path | None,
    bootstrap_agents: Sequence[AgentSpec],
) -> None:
    if experimental_path is not None:
        print(f"Experimental agent path: {experimental_path}")
    if bootstrap_agents:
        group_counts = (
            pd.Series([agent.agent_group for agent in bootstrap_agents]).value_counts().to_dict()
        )
        print(
            f"Ensemble agents discovered: {len(bootstrap_agents)} "
            f"({bootstrap_agents[0].path.parent.parent})"
        )
        print(f"Ensemble group counts: {group_counts}")
        print(
            "Ensemble agent codes: "
            + ", ".join(agent.agent_code for agent in bootstrap_agents[:10])
            + (" ..." if len(bootstrap_agents) > 10 else "")
        )


def main() -> None:
    args = parse_args()
    ensure_mode_paths(args)
    args.data_output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    base_values = np.linspace(DEFAULT_BASE_MIN, DEFAULT_BASE_MAX, args.n_base, dtype=np.float32)
    peak_addition_values = np.linspace(
        DEFAULT_PEAK_ADDITION_MIN,
        DEFAULT_PEAK_ADDITION_MAX,
        args.n_p_addition,
        dtype=np.float32,
    )
    action_histories = select_action_histories(args.action_history_mode)
    probes_per_agent = (
        len(action_histories)
        * len(base_values)
        * len(DEFAULT_PEAK_DAY_VALUES)
        * len(peak_addition_values)
    )

    experimental_results: pd.DataFrame | None = None
    bootstrap_results: list[pd.DataFrame] = []

    print(f"CSV output directory: {args.data_output_dir}")
    print(f"Figure output directory: {args.plot_output_dir}")
    print(
        "Grid: "
        f"{len(base_values)} base values x "
        f"{len(peak_addition_values)} peak-addition values x "
        f"{len(DEFAULT_PEAK_DAY_VALUES)} peak-day values x "
        f"{len(action_histories)} action histories"
    )
    print(f"Probes per agent: {probes_per_agent}")
    print(f"Grouped curves use {args.n_growth_bins} bins over growth = base + peak_day * peak_addition.")

    if args.mode in {"experimental", "both"}:
        print("Analyzing experimental agent...")
        warnings.warn(
            "The default experimental-agent path is inferred from the folder "
            "'PhysiLearning/Agent_used_for_experiment'. Verify that this is the "
            "exact schedule-generating checkpoint before publication-critical use."
        )
        experimental_agent = AgentSpec(
            agent_id="experimental_agent",
            agent_code=Path(args.experimental_agent_path).stem,
            agent_type="experimental",
            agent_group="experimental",
            path=args.experimental_agent_path,
        )
        experimental_results = analyze_agent(
            agent=experimental_agent,
            action_histories=action_histories,
            base_values=base_values,
            peak_addition_values=peak_addition_values,
        )
        experimental_results.to_csv(
            args.data_output_dir / "experimental_agent_policy_behavior.csv",
            index=False,
        )
        collapse_experimental_for_growth_groups(experimental_results, args.n_growth_bins).to_csv(
            args.data_output_dir / "experimental_agent_policy_behavior_grouped_curves.csv",
            index=False,
        )
        plot_experimental_curves(experimental_results, args.plot_output_dir, args.n_growth_bins)

    bootstrap_agents: list[AgentSpec] = []
    if args.mode in {"bootstrap", "both"}:
        bootstrap_agents = discover_bootstrap_agents(args.bootstrap_agent_dir)
        if not bootstrap_agents:
            raise FileNotFoundError(
                "No bootstrap agents were discovered. "
                f"Checked: {args.bootstrap_agent_dir}"
            )

        print_discovery_summary(
            args.experimental_agent_path if args.mode in {"experimental", "both"} else None,
            bootstrap_agents,
        )

        with tqdm(
            bootstrap_agents,
            desc="Ensemble agents",
            leave=True,
            dynamic_ncols=True,
        ) as agents_bar:
            for agent in agents_bar:
                agents_bar.set_postfix_str(f"group={agent.agent_group}")
                if not agent.path.exists():
                    warnings.warn(f"Skipping missing bootstrap agent: {agent.path}")
                    continue
                try:
                    result = analyze_agent(
                        agent=agent,
                        action_histories=action_histories,
                        base_values=base_values,
                        peak_addition_values=peak_addition_values,
                    )
                except Exception as exc:  # pragma: no cover - intentional robustness path
                    warnings.warn(f"Skipping {agent.agent_code} due to loading/analysis error: {exc}")
                    continue
                bootstrap_results.append(result)

        if not bootstrap_results:
            raise RuntimeError("All bootstrap agents failed to load or analyze.")

        bootstrap_all = pd.concat(bootstrap_results, ignore_index=True)
        bootstrap_all.to_csv(
            args.data_output_dir / "bootstrap_agents_policy_behavior.csv",
            index=False,
        )
        bootstrap_summary = aggregate_bootstrap_summary(bootstrap_all)
        bootstrap_summary.to_csv(
            args.data_output_dir / "bootstrap_agents_policy_behavior_summary.csv",
            index=False,
        )
        summarize_bootstrap_for_growth_groups(bootstrap_all, args.n_growth_bins).to_csv(
            args.data_output_dir / "bootstrap_agents_policy_behavior_grouped_curves.csv",
            index=False,
        )
        plot_bootstrap_curves(bootstrap_all, args.plot_output_dir, args.n_growth_bins)

        if experimental_results is not None:
            plot_comparison(experimental_results, bootstrap_all, args.plot_output_dir, args.n_growth_bins)

    elif experimental_results is not None:
        print_discovery_summary(args.experimental_agent_path, [])

    print(f"CSV outputs written to: {args.data_output_dir}")
    print(f"Figure outputs written to: {args.plot_output_dir}")
    print(f"Action histories analyzed: {[format_action_history(h) for h in action_histories]}")
    print(
        "Synthetic burden defaults: "
        f"base in [{DEFAULT_BASE_MIN}, {DEFAULT_BASE_MAX}], "
        f"peak_addition in [{DEFAULT_PEAK_ADDITION_MIN}, {DEFAULT_PEAK_ADDITION_MAX}], "
        f"peak_day in {list(DEFAULT_PEAK_DAY_VALUES)}"
    )
    print("Growth score: growth = base + peak_day * peak_addition.")
    print("Primary curve metric: fraction of deterministic decisions choosing treatment ON.")
    print("Action 1 is interpreted as treatment ON, matching physilearning.envs.lv.LvEnv.step().")


if __name__ == "__main__":
    main()
