"""Create Figure 4d experimental trajectories and escape-margin PDFs.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

Outputs
-------
1. All retained ATAgent experimental trajectories.
2. All retained AT+ experimental trajectories.
3. ATAgent escape margins over time.
4. AT+ escape margins over time.

Every panel has an exact 30 x 20 mm data axes. Sensitive and resistant cell
numbers are normalized by the initial total.
"""
from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: in-vitro Results/*.txt, combined_results.csv, simulation manifest/HDF5,
# LV YAML config, trained PPO model, and fitted environment inputs.
# Scripts: PhysiLearning LV environment code, shared/experiment_filters.py, shared/figtools.py.
# Outputs: four normalized experimental PDFs and an audit CSV.

import argparse
import csv
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import MultipleLocator
from stable_baselines3 import PPO


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.experiment_filters import EXCLUDE_SAMPLES, apply_sample_exclusions
from shared.data_paths import FITTING, MODELS, COMBINED_RESULTS_CSV, EXP_RAW_DATA, figure_data_dir, intermediate_dir
from shared.strategy_colors import FIGURE_COLOR_MAP, get_color, normalize_strategy


LV_REPO = REPO_ROOT / "PhysiLearning" / "LV_Agent_Training_MiniRepo"
LV_SRC = LV_REPO / "src"
if str(LV_SRC) not in sys.path:
    sys.path.insert(0, str(LV_SRC))

from physilearning.envs.lv import LvEnv


DATA_ROOT = EXP_RAW_DATA
COMBINED_DPD_CSV = COMBINED_RESULTS_CSV
DPD_RUN_ROOT = intermediate_dir("simulation", "basic", "density_maps", create=False)
DPD_MANIFEST = DPD_RUN_ROOT / "simulation_density_run_manifest.tsv"
LV_CONFIG = FITTING / 'PhysiLearning/LV_Agent_Training_MiniRepo/configs/2025_10_lv_real_mela_1_load.yaml'
EXPERIMENTAL_AGENT_MODEL = MODELS / 'experimental_agent'
OUTPUT_DIR = ft.paper_figure_data_dir(__file__) / "Example_Trajectories"
DATA_OUTPUT_DIR = figure_data_dir("Figure_4", "example_trajectories", create=False)
ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"

MIN_FINAL_DAY = 50.0
MIN_DPD_RUN = 10
RUN_TO_DENS_SAMPLE = {42: "B2_Dens-B2", 43: "B3_Dens-B3", 44: "B4_Dens-B4", 45: "C2_Dens-C2", 46: "C3_Dens-C3", 47: "C4_Dens-C4", 48: "D3_Dens-D3", 49: "D5_Dens-D5"}
ESCAPE_MARGIN_EXAMPLES = {
    "LV-Agent": ("Plate_7", "C4_ATAgent2-C4"),
    "AT100+": ("Plate_8", "D3_Dens-D3"),
}
AXES_WIDTH_MM = 25.0
AXES_HEIGHT_MM = 15.0
PANEL_GAP_MM = 8.0
SINGLE_MARGINS_MM = (8.5, 6.5, 10.0, 3.0)
TRAJECTORY_MARGINS_MM = (8.5, 6.5, 10.0, 6.0)

SENSITIVE_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
CONFINEMENT_COLOR = "#c8a2c8"
ESCAPE_COLOR = "#f15a29"
LINEWIDTH = 0.9
PROGRESSION_THRESHOLD = 0.5
FAILURE_REFERENCE_COLOR = "#bfbfbf"
FAILURE_REFERENCE_LINEWIDTH = 0.75
FAILURE_REFERENCE_LINESTYLE = (0.0, (4.0, 5.0))
MARGIN_LIMITS = (0.0, 0.6)
COUNT_LIMITS = (0.0, 2.5)
ESCAPE_MARGIN_X_LIMITS = (0.0, 70.0)
MARGIN_LINESTYLE = "-"
AXIS_LINEWIDTH = 0.8
EVENT_MARKER_SIZE = 3.4
TREATMENT_LINEWIDTH = 1.5
TEXT_PT = 7.0
TICK_PT = 6.0
LEGEND_PT = 5.0
DPI = 300

ft.apply_nature_style(
    text_pt=TEXT_PT,
    tick_pt=TICK_PT,
    legend_pt=LEGEND_PT,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    },
)


@dataclass(frozen=True)
class Trajectory:
    days: np.ndarray
    sensitive: np.ndarray
    resistant: np.ndarray
    actions: np.ndarray
    label: str

    @property
    def total(self) -> np.ndarray:
        return self.sensitive + self.resistant

    def normalized(self) -> "Trajectory":
        denominator = float(self.total[0])
        if denominator <= 0:
            raise ValueError(f"Non-positive initial total in {self.label}")
        return Trajectory(
            days=self.days.copy(),
            sensitive=self.sensitive / denominator,
            resistant=self.resistant / denominator,
            actions=self.actions.copy(),
            label=self.label,
        )

    @property
    def stability_score(self) -> float:
        """RMS deviation of normalized total burden from its baseline."""
        normalized_total = self.total / float(self.total[0])
        return float(np.sqrt(np.mean((normalized_total - 1.0) ** 2)))


def numeric_suffix(path: Path | str) -> int:
    match = re.search(r"(\d+)(?!.*\d)", Path(path).stem)
    return int(match.group(1)) if match else -1


def parse_result_file(path: Path, stop_day: float | None = None) -> Trajectory:
    days: list[float] = []
    sensitive: list[float] = []
    resistant: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 3:
                continue
            # Match the measurement suffix (``_dNN``), not a D-row well name
            # such as ``D3`` embedded earlier in the record identifier.
            day_match = re.search(r"_d(\d+)\b", parts[0], flags=re.IGNORECASE)
            if not day_match:
                continue
            day = float(day_match.group(1))
            if stop_day is not None and day > stop_day:
                continue
            try:
                sensitive_count = float(parts[1])
                resistant_count = float(parts[2])
            except ValueError:
                continue
            days.append(day)
            sensitive.append(sensitive_count)
            resistant.append(resistant_count)
    if not days:
        raise ValueError(f"No trajectory records found in {path}")
    return Trajectory(
        np.asarray(days),
        np.asarray(sensitive),
        np.asarray(resistant),
        np.zeros(len(days), dtype=int),
        str(path),
    )


def parse_schedule(path: Path) -> dict[str, set[int]]:
    schedule: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split("\t", maxsplit=1)
            if not parts or not parts[0]:
                continue
            treated: set[int] = set()
            if len(parts) == 2:
                for token in parts[1].split():
                    token = token.strip().lower()
                    if not token or token.endswith("s"):
                        continue
                    token = token.removesuffix("d")
                    try:
                        treated.add(int(float(token)))
                    except ValueError:
                        pass
            schedule[parts[0].strip()] = treated
    return schedule


def schedule_treatment_intervals(
    data_root: Path, plate: str, sample: str,
) -> list[tuple[float, float]]:
    """Return consecutive one-day treatment intervals for an experiment."""
    treated_days = sorted(parse_schedule(data_root / plate / "treatment_schedule.txt").get(sample, set()))
    if not treated_days:
        return []
    intervals: list[tuple[float, float]] = []
    start = end = treated_days[0]
    for day in treated_days[1:]:
        if day == end + 1:
            end = day
        else:
            intervals.append((float(start), float(end + 1)))
            start = end = day
    intervals.append((float(start), float(end + 1)))
    return intervals


def draw_treatment_timing(
    ax: plt.Axes,
    intervals: Iterable[tuple[float, float]],
    color: str,
    *,
    x_limits: tuple[float, float],
    final_day: float,
    alpha: float = 1.0,
) -> None:
    """Draw treatment exposure as compact segments immediately above an axis."""
    for start, stop in intervals:
        start = max(start, x_limits[0])
        stop = min(stop, x_limits[1], final_day)
        if stop <= start:
            continue
        ax.plot(
            [start, stop], [1.045, 1.045], transform=ax.get_xaxis_transform(),
            color=color, lw=TREATMENT_LINEWIDTH, solid_capstyle="butt",
            alpha=alpha, clip_on=False, zorder=8,
        )


def experimental_candidates(data_root: Path, strategy: str) -> list[Trajectory]:
    candidates: list[Trajectory] = []
    pattern = f"Results_G418_a2a4_*_{strategy}-*.txt"
    for path in sorted(data_root.glob(f"Plate_*/Results/{pattern}")):
        match = re.search(
            rf"Results_G418_a2a4_(?P<well>[A-H]\d+)_{re.escape(strategy)}-",
            path.name,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        well = match.group("well").upper()
        replicate_key = f"{well}_{strategy}-{well}"
        if replicate_key in EXCLUDE_SAMPLES and EXCLUDE_SAMPLES[replicate_key] is None:
            continue
        cutoff = EXCLUDE_SAMPLES.get(replicate_key)
        trajectory = parse_result_file(path, stop_day=float(cutoff) if cutoff else None)
        if trajectory.days[-1] <= MIN_FINAL_DAY:
            continue
        treated_days = parse_schedule(path.parents[1] / "treatment_schedule.txt").get(
            replicate_key, set()
        )
        actions = np.asarray([int(int(day) in treated_days) for day in trajectory.days])
        candidates.append(
            Trajectory(
                trajectory.days,
                trajectory.sensitive,
                trajectory.resistant,
                actions,
                f"{path.parents[1].name} | {replicate_key}",
            )
        )
    return candidates


def retained_trajectories(data_root: Path, strategies: tuple[str, ...]) -> list[Trajectory]:
    """Return every non-excluded experimental replicate through its filter cutoff."""
    trajectories: list[Trajectory] = []
    for strategy in strategies:
        pattern = f"Results_G418_a2a4_*_{strategy}-*.txt"
        for path in sorted(data_root.glob(f"Plate_*/Results/{pattern}")):
            match = re.search(rf"Results_G418_a2a4_(?P<well>[A-H]\d+)_{re.escape(strategy)}-", path.name, flags=re.IGNORECASE)
            if not match:
                continue
            well = match.group("well").upper()
            sample = f"{well}_{strategy}-{well}"
            cutoff = EXCLUDE_SAMPLES.get(sample)
            if sample in EXCLUDE_SAMPLES and cutoff is None:
                continue
            trajectory = parse_result_file(path, stop_day=float(cutoff) if cutoff else None)
            trajectories.append(Trajectory(trajectory.days, trajectory.sensitive, trajectory.resistant,
                                           trajectory.actions, f"{path.parents[1].name} | {sample}"))
    if not trajectories:
        raise RuntimeError(f"No retained trajectories found for {strategies}.")
    return trajectories


def choose_stable(candidates: Iterable[Trajectory], description: str) -> Trajectory:
    eligible = [trajectory for trajectory in candidates if trajectory.total[0] > 0]
    if not eligible:
        raise RuntimeError(f"No >{MIN_FINAL_DAY:g}-day candidates found for {description}")
    return min(eligible, key=lambda trajectory: trajectory.stability_score)


def choose_mapped_dens_pair(experimental: Iterable[Trajectory], dpd: Iterable[Trajectory]) -> tuple[Trajectory, Trajectory]:
    experiments = list(experimental)
    simulations = list(dpd)
    pairs = []
    for run, sample in RUN_TO_DENS_SAMPLE.items():
        exp = next((item for item in experiments if sample in item.label), None)
        sim = next((item for item in simulations if item.label.endswith(f"run {run}")), None)
        if exp is not None and sim is not None and exp.total[0] > 0 and sim.total[0] > 0:
            pairs.append((exp.stability_score + sim.stability_score, exp, sim))
    if not pairs:
        raise RuntimeError("No eligible mapped >50-day AT100+ pair found")
    _score, exp, sim = min(pairs, key=lambda item: item[0])
    return exp, sim


def load_dpd_summary_candidates(path: Path, protocol: str, display: str) -> list[Trajectory]:
    data = pd.read_csv(path)
    required = {"protocol", "run", "day", "normalized_tot", "normalized_sus"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"DPD table is missing columns: {sorted(missing)}")
    # ``combined_results.csv`` now stores this state as ``treatment``.
    # Retain compatibility with older tables that used ``treatment_on``.
    if "treatment_on" not in data.columns:
        if "treatment" not in data.columns:
            raise ValueError("DPD table is missing a treatment/treatment_on column")
        data["treatment_on"] = data["treatment"]
    data["protocol"] = data["protocol"].astype(str).str.strip().str.upper()
    for column in ("run", "day", "normalized_tot", "normalized_sus"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[(data["protocol"] == protocol.upper()) & (data["run"] >= MIN_DPD_RUN)]
    candidates: list[Trajectory] = []
    for run, group in data.groupby("run"):
        group = group.sort_values("day").drop_duplicates("day", keep="first").dropna(
            subset=["day", "normalized_tot", "normalized_sus"]
        )
        if group.empty or float(group["day"].iloc[-1]) <= MIN_FINAL_DAY:
            continue
        sensitive = group["normalized_sus"].to_numpy(dtype=float)
        total = group["normalized_tot"].to_numpy(dtype=float)
        candidates.append(
            Trajectory(
                group["day"].to_numpy(dtype=float),
                sensitive,
                total - sensitive,
                group["treatment_on"].astype(bool).to_numpy(dtype=int),
                f"DPD {display} run {int(run)}",
            )
        )
    return candidates


def stored_count(node) -> int:
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.ndim else int(node[()])
    for name in ("block0_values", "values"):
        if name in node:
            return int(node[name].shape[0])
    raise ValueError(f"Cannot count cells in {node.name}")


def treatment_state(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        key = next(key for key in data if key.strip('"') == "therapy_switch")
        return int(bool(np.asarray(data[key])[()]))


def load_dpd_dens_candidates(manifest_path: Path) -> list[Trajectory]:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    manifest = manifest[manifest["strategy"].str.upper().eq("DENS")].copy()
    manifest["run_number"] = pd.to_numeric(
        manifest["run_id"].str.extract(r"(\d+)$", expand=False), errors="coerce"
    )
    manifest = manifest[manifest["run_number"] >= MIN_DPD_RUN]
    processed_dir = manifest_path.parent / "Plate_Dens" / "hdf5"
    candidates: list[Trajectory] = []
    for record in manifest.itertuples(index=False):
        processed = processed_dir / f"{record.well_identifier}_simulation.h5"
        if not processed.exists():
            continue
        days: list[float] = []
        sensitive: list[float] = []
        resistant: list[float] = []
        with h5py.File(processed, "r") as handle:
            day_keys = sorted(
                set(handle["sus"]) & set(handle["res"]), key=numeric_suffix
            )
            for day_key in day_keys:
                days.append(float(numeric_suffix(Path(day_key))))
                sensitive.append(float(stored_count(handle["sus"][day_key])))
                resistant.append(float(stored_count(handle["res"][day_key])))
        if not days or days[-1] <= MIN_FINAL_DAY:
            continue
        raw_files = sorted(
            Path(record.source_run_dir).glob("simulation_*.h5"), key=numeric_suffix
        )
        actions = [treatment_state(path) for path in raw_files[: len(days)]]
        if len(actions) < len(days):
            actions.extend([0] * (len(days) - len(actions)))
        candidates.append(
            Trajectory(
                np.asarray(days),
                np.asarray(sensitive),
                np.asarray(resistant),
                np.asarray(actions),
                f"DPD AT100+ run {int(record.run_number)}",
            )
        )
    return candidates


def configure_lv(initial_sensitive: float, initial_resistant: float) -> LvEnv:
    with LV_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    total = initial_sensitive + initial_resistant
    if total <= 0:
        raise ValueError("LV initial total must be positive")
    config["env"]["initial_wt"] = float(initial_sensitive / total)
    config["env"]["initial_mut"] = float(initial_resistant / total)

    runtime_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
            runtime_path = Path(handle.name)
        return LvEnv.from_yaml(str(runtime_path))
    finally:
        if runtime_path is not None:
            runtime_path.unlink(missing_ok=True)


def stacked_observation(frames: list[np.ndarray], observation: np.ndarray) -> np.ndarray:
    frames.append(np.asarray(observation, dtype=np.float32).reshape(-1))
    del frames[:-4]
    return np.concatenate(frames, axis=0)


def load_experimental_agent() -> PPO:
    """Load the preserved experimental PPO checkpoint.

    The checkpoint is stored as an extracted Stable-Baselines3 archive rather
    than as a ``.zip`` file.  Stable-Baselines3 loads zip archives, so package
    the existing files temporarily without altering the preserved artifact.
    """
    if EXPERIMENTAL_AGENT_MODEL.is_file():
        return PPO.load(str(EXPERIMENTAL_AGENT_MODEL))
    if not EXPERIMENTAL_AGENT_MODEL.is_dir():
        raise FileNotFoundError(f"Experimental agent checkpoint not found: {EXPERIMENTAL_AGENT_MODEL}")

    required_files = {
        "data",
        "policy.pth",
        "policy.optimizer.pth",
        "pytorch_variables.pth",
        "_stable_baselines3_version",
        "system_info.txt",
    }
    available_files = {path.name for path in EXPERIMENTAL_AGENT_MODEL.iterdir() if path.is_file()}
    missing = required_files - available_files
    if missing:
        raise FileNotFoundError(
            f"Experimental agent checkpoint is incomplete: {EXPERIMENTAL_AGENT_MODEL}. "
            f"Missing: {sorted(missing)}"
        )

    with tempfile.TemporaryDirectory(prefix="experimental_ppo_") as temp_dir:
        archive_path = Path(temp_dir) / f"{EXPERIMENTAL_AGENT_MODEL.name}.zip"
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(EXPERIMENTAL_AGENT_MODEL.iterdir()):
                if path.is_file():
                    archive.write(path, arcname=path.name)
        return PPO.load(str(archive_path))


def run_lv(
    experimental: Trajectory,
    *,
    controller: str,
) -> Trajectory:
    env = configure_lv(float(experimental.sensitive[0]), float(experimental.resistant[0]))
    observation, _ = env.reset(seed=0)
    observation = np.asarray(observation, dtype=np.float32)
    frames = [np.zeros_like(observation) for _ in range(3)]
    stacked = stacked_observation(frames, observation)
    model = load_experimental_agent() if controller == "agent" else None

    days = [float(env.day)]
    sensitive = [float(env.state[0])]
    resistant = [float(env.state[1])]
    actions = [0]
    previous_action = 0
    final_day = float(experimental.days[-1])

    while float(env.day) < final_day:
        if controller == "agent":
            predicted, _ = model.predict(stacked, deterministic=True)
            action = int(np.asarray(predicted).reshape(-1)[0])
        elif controller == "at100_plus":
            burden = float(env.state[0] + env.state[1])
            action = int(burden >= 1.0 and previous_action == 0)
        else:
            raise ValueError(f"Unknown controller: {controller}")

        # Figure 3.5 convention: the action chosen at this observation applies
        # throughout the complete following 2- or 3-day growth interval.
        actions[-1] = action
        observation, _reward, terminated, truncated, _info = env.step(action)
        days.append(float(env.day))
        sensitive.append(float(env.state[0]))
        resistant.append(float(env.state[1]))
        actions.append(0)
        previous_action = action
        stacked = stacked_observation(frames, np.asarray(observation, dtype=np.float32))
        # Continue to the requested experimental horizon even if the training
        # environment reports its progression/episode termination threshold.

    label = "LV environment | LV-Agent" if controller == "agent" else "LV environment | AT100+"
    return Trajectory(
        np.asarray(days),
        np.asarray(sensitive),
        np.asarray(resistant),
        np.asarray(actions),
        label,
    )


def treatment_intervals(trajectory: Trajectory) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for index, action in enumerate(trajectory.actions):
        if not action:
            continue
        start = float(trajectory.days[index])
        if index + 1 < len(trajectory.days):
            end = float(trajectory.days[index + 1])
        else:
            end = start + 1.0
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def truncate_at_progression(trajectory: Trajectory) -> Trajectory:
    normalized = trajectory.normalized()
    crossings = np.flatnonzero(normalized.resistant > PROGRESSION_THRESHOLD)
    stop = int(crossings[0]) + 1 if crossings.size else len(trajectory.days)
    return Trajectory(
        trajectory.days[:stop], trajectory.sensitive[:stop],
        trajectory.resistant[:stop], trajectory.actions[:stop], trajectory.label,
    )


def plot_on_axis(ax: plt.Axes, trajectory: Trajectory) -> Trajectory:
    truncated = truncate_at_progression(trajectory)
    normalized = truncated.normalized()
    ax.plot(normalized.days, normalized.sensitive, color=SENSITIVE_COLOR,
            marker=None, linewidth=LINEWIDTH, linestyle="-")
    ax.plot(normalized.days, normalized.resistant, color=RESISTANT_COLOR,
            marker=None, linewidth=LINEWIDTH, linestyle="-")
    return truncated


def finish_axis(ax: plt.Axes, x_max: float) -> None:
    ax.axhline(PROGRESSION_THRESHOLD, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time [d]")
    ax.set_ylabel(r"Normalized count $\eta$")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 3)
    ax.set_yticks((1, 2, 3))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(
        handles=[
            Line2D([], [], color=SENSITIVE_COLOR, linewidth=LINEWIDTH, label="sen"),
            Line2D([], [], color=RESISTANT_COLOR, linewidth=LINEWIDTH, label="res"),
        ],
        frameon=False, ncol=1, loc="upper left", handlelength=1.6,
        columnspacing=0.8, handletextpad=0.35,
    )
    ft.format_axes(ax)


def save_single(
    trajectory: Trajectory,
    output_path: Path,
    x_max: float,
) -> None:
    fig, ax = ft.new_figure(
        width=AXES_WIDTH_MM,
        height_mm=AXES_HEIGHT_MM,
        margins=TRAJECTORY_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    plot_on_axis(ax, trajectory)
    finish_axis(ax, x_max)
    ft.savefig(fig, output_path)
    plt.close(fig)


def save_all_trajectories(trajectories: list[Trajectory], output_path: Path, title: str) -> None:
    fig, ax = ft.new_figure(width=AXES_WIDTH_MM, height_mm=AXES_HEIGHT_MM,
                            margins=TRAJECTORY_MARGINS_MM, dpi=DPI, apply_style=False)
    for trajectory in trajectories:
        normalized = trajectory.normalized()
        ax.plot(normalized.days, normalized.sensitive, color=SENSITIVE_COLOR,
                linewidth=LINEWIDTH, alpha=0.62, marker=None)
        ax.plot(normalized.days, normalized.resistant, color=RESISTANT_COLOR,
                linewidth=LINEWIDTH, alpha=0.62, marker=None)
    ax.axhline(PROGRESSION_THRESHOLD, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.set(xlim=(0, 70), ylim=(0, 3), xlabel="Time [d]", ylabel=r"Normalized count, $\eta$")
    ax.xaxis.set_major_locator(MultipleLocator(35.0))
    ax.legend(handles=[Line2D([], [], color=SENSITIVE_COLOR, linewidth=LINEWIDTH, label="sen"),
                       Line2D([], [], color=RESISTANT_COLOR, linewidth=LINEWIDTH, label="res")],
              frameon=False, loc="upper left", handlelength=1.6, handletextpad=0.35)
    ft.format_axes(ax)
    ft.savefig(fig, output_path)
    plt.close(fig)


def load_escape_margins() -> pd.DataFrame:
    table = pd.read_csv(ESCAPE_MARGIN_CSV)
    required = {
        "dataset_type", "plate", "sample", "strategy", "day",
        "confinement_count", "escaped_resistant_count",
        "n_sensitive_retained", "n_resistant_retained",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{ESCAPE_MARGIN_CSV} is missing columns: {sorted(missing)}")
    table = table.loc[table["dataset_type"].astype(str).str.lower().eq("experimental")].copy()
    table["strategy"] = table["strategy"].astype(str).map(normalize_strategy)
    for column in (
        "day", "confinement_count", "escaped_resistant_count",
        "n_sensitive_retained", "n_resistant_retained",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["day", "n_sensitive_retained", "n_resistant_retained"])
    table[["confinement_count", "escaped_resistant_count"]] = table[
        ["confinement_count", "escaped_resistant_count"]
    ].fillna(0.0)
    table = apply_sample_exclusions(table, exclude_map=EXCLUDE_SAMPLES, column="sample", cutoff_column="day", strict=False)
    table = table.sort_values(["plate", "sample", "day"])
    table["initial_total"] = table.groupby(["plate", "sample"])[["n_sensitive_retained", "n_resistant_retained"]].transform("first").sum(axis=1)
    table = table.loc[table["initial_total"] > 0].copy()
    table["normalized_confinement_margin"] = table["confinement_count"] / table["initial_total"]
    table["normalized_escape_margin"] = table["escaped_resistant_count"] / table["initial_total"]
    table["normalized_sensitive"] = table["n_sensitive_retained"] / table["initial_total"]
    table["normalized_resistant"] = table["n_resistant_retained"] / table["initial_total"]
    return table


def _event_day(trajectory: pd.DataFrame, column: str, threshold: float) -> float:
    event = trajectory.loc[trajectory[column].ge(threshold), "day"]
    return float(event.iloc[0]) if not event.empty else np.nan


def _threshold_crossing_day(
    days: np.ndarray, values: np.ndarray, threshold: float,
) -> float:
    """Return the first upward threshold crossing, linearly interpolated."""
    for index, value in enumerate(values):
        if value < threshold:
            continue
        if index == 0:
            return float(days[0])
        previous_value = values[index - 1]
        previous_day = days[index - 1]
        if previous_value >= threshold or np.isclose(value, previous_value):
            return float(days[index])
        fraction = (threshold - previous_value) / (value - previous_value)
        return float(previous_day + fraction * (days[index] - previous_day))
    return np.nan


def _draw_event_marker(
    ax: plt.Axes,
    day: float,
    marker: object,
    color: str,
    top_extent: float,
    additional_down_pt: float = 0.0,
) -> None:
    if not np.isfinite(day) or not ESCAPE_MARGIN_X_LIMITS[0] <= day <= ESCAPE_MARGIN_X_LIMITS[1]:
        return
    offset = ScaledTranslation(
        0.0,
        -(EVENT_MARKER_SIZE * top_extent + additional_down_pt) / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.plot(
        [day], [0.0], marker=marker, markersize=EVENT_MARKER_SIZE,
        markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.4,
        linestyle="None", transform=ax.get_xaxis_transform() + offset,
        clip_on=False, zorder=7,
    )


def _add_component_ylabels(fig: plt.Figure, ax: plt.Axes) -> None:
    """Add the Figure 2f colour-coded component labels."""
    position = ax.get_position()
    left_x = position.x0 - 0.155
    right_x = position.x1 + 0.1025
    center_y = (position.y0 + position.y1) / 2
    for x, y, label, color, rotation in (
        (left_x, center_y - 0.160, r"$\eta^{\mathrm{conf}}_{\mathrm{sen}}$", CONFINEMENT_COLOR, 90),
        (left_x, center_y + 0.160, r"$\eta^{\mathrm{esc}}_{\mathrm{res}}$", ESCAPE_COLOR, 90),
        (right_x, center_y - 0.160, r"$\eta_{\mathrm{sen}}$", SENSITIVE_COLOR, 270),
        (right_x, center_y + 0.160, r"$\eta_{\mathrm{res}}$", RESISTANT_COLOR, 270),
    ):
        fig.text(x, y, label, rotation=rotation, ha="center", va="center",
                 color=color, fontsize=6.0)
    for x in (left_x, right_x):
        fig.add_artist(Line2D(
            [x - 0.018, x + 0.018], [center_y, center_y], color="#000000", lw=1.0,
            solid_capstyle="butt", transform=fig.transFigure, clip_on=False,
        ))


def save_escape_margins(table: pd.DataFrame, strategy: str, output_path: Path, title: str) -> int:
    plate, sample = ESCAPE_MARGIN_EXAMPLES[strategy]
    panel = table.loc[
        table["strategy"].eq(strategy)
        & table["plate"].eq(plate)
        & table["sample"].eq(sample)
    ].copy()
    if panel.empty:
        raise RuntimeError(f"Missing selected escape-margin example: {strategy} {plate}/{sample}")
    fig, ax = ft.new_figure(width=AXES_WIDTH_MM, height_mm=AXES_HEIGHT_MM,
                            margins=SINGLE_MARGINS_MM, dpi=DPI, apply_style=False)
    count_ax = ax.twinx()
    count_ax.set_position(ax.get_position())
    for _identity, trajectory in panel.groupby(["plate", "sample"], sort=True):
        trajectory = trajectory.sort_values("day")
        ax.plot(trajectory["day"], trajectory["normalized_confinement_margin"],
                color=CONFINEMENT_COLOR, linewidth=1.2, linestyle=MARGIN_LINESTYLE,
                alpha=1.0, marker=None, zorder=3)
        ax.plot(trajectory["day"], trajectory["normalized_escape_margin"],
                color=ESCAPE_COLOR, linewidth=1.2, linestyle=MARGIN_LINESTYLE,
                alpha=1.0, marker=None, zorder=3)
        count_ax.plot(trajectory["day"], trajectory["normalized_sensitive"],
                      color=SENSITIVE_COLOR, linewidth=1.0, alpha=1.0, zorder=2)
        count_ax.plot(trajectory["day"], trajectory["normalized_resistant"],
                      color=RESISTANT_COLOR, linewidth=1.0, alpha=1.0, zorder=2)
        resistant_crossing_day = _threshold_crossing_day(
            trajectory["day"].to_numpy(float),
            trajectory["normalized_resistant"].to_numpy(float),
            PROGRESSION_THRESHOLD,
        )
        if np.isfinite(resistant_crossing_day) and ESCAPE_MARGIN_X_LIMITS[0] <= resistant_crossing_day <= ESCAPE_MARGIN_X_LIMITS[1]:
            count_ax.plot(
                [resistant_crossing_day, ESCAPE_MARGIN_X_LIMITS[1]],
                [PROGRESSION_THRESHOLD, PROGRESSION_THRESHOLD],
                color=FAILURE_REFERENCE_COLOR, linewidth=FAILURE_REFERENCE_LINEWIDTH,
                linestyle=FAILURE_REFERENCE_LINESTYLE, dash_capstyle="butt", zorder=0,
            )
        plate, sample = _identity
        draw_treatment_timing(
            ax, schedule_treatment_intervals(DATA_ROOT, str(plate), str(sample)),
            get_color(strategy, "#000000"), x_limits=ESCAPE_MARGIN_X_LIMITS,
            final_day=float(trajectory["day"].max()),
        )
        escape_day = _event_day(trajectory, "normalized_escape_margin", np.nextafter(0.0, 1.0))
        failure_day = _event_day(trajectory, "normalized_resistant", PROGRESSION_THRESHOLD)
        marker_color = get_color(strategy, "#000000")
        overlap = np.isfinite(escape_day) and np.isfinite(failure_day) and np.isclose(escape_day, failure_day)
        _draw_event_marker(
            ax, escape_day, "^", marker_color, 0.52,
            additional_down_pt=EVENT_MARKER_SIZE if overlap else 0.0,
        )
        _draw_event_marker(ax, failure_day, (8, 0, 22.5), marker_color, 0.49)
    count = panel[["plate", "sample"]].drop_duplicates().shape[0]
    count_ax.set_ylim(COUNT_LIMITS)
    count_ax.spines[["left", "bottom", "top"]].set_visible(False)
    count_ax.spines["right"].set_linewidth(AXIS_LINEWIDTH)
    count_ax.tick_params(axis="y", labelsize=5)
    count_ax.set_zorder(0)
    ax.patch.set_visible(False)
    ax.set(
        xlim=ESCAPE_MARGIN_X_LIMITS, ylim=MARGIN_LIMITS, xlabel="Time [d]",
    )
    ax.xaxis.set_major_locator(MultipleLocator(35.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color="#000000", marker="^", markersize=EVENT_MARKER_SIZE,
                   markerfacecolor="#000000", markeredgewidth=0.4,
                   linestyle="None", label="Escape"),
            Line2D([], [], color="#000000", marker=(8, 0, 22.5), markersize=EVENT_MARKER_SIZE,
                   markerfacecolor="#000000", markeredgewidth=0.4,
                   linestyle="None", label="Failure"),
        ],
        frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0.0, fontsize=4.5, ncol=2, columnspacing=0.8,
        handlelength=1.0,
    )
    ft.format_axes(ax)
    _add_component_ylabels(fig, ax)
    ft.savefig(fig, output_path)
    plt.close(fig)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--combined-dpd-csv", type=Path, default=COMBINED_DPD_CSV)
    parser.add_argument("--dpd-manifest", type=Path, default=DPD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Required input not found: {args.data_root}")

    atagent = retained_trajectories(args.data_root, ("ATAgent2",))
    at_plus = retained_trajectories(args.data_root, ("Dens",))
    margins = load_escape_margins()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "01_ATAgent_trajectories.pdf",
        args.output_dir / "02_AT100_plus_trajectories.pdf",
        args.output_dir / "03_ATAgent_escape_margin_over_time.pdf",
        args.output_dir / "04_AT100_plus_escape_margin_over_time.pdf",
    ]
    save_all_trajectories(atagent, outputs[0], "ATagent")
    save_all_trajectories(at_plus, outputs[1], r"AT$^{+}$")
    atagent_margin_count = save_escape_margins(margins, "LV-Agent", outputs[2], "ATagent")
    at_plus_margin_count = save_escape_margins(margins, "AT100+", outputs[3], r"AT$^{+}$")

    audit = pd.DataFrame([
        {"panel": 1, "source": "experiment", "identifier": "ATagent", "n_replicates": len(atagent)},
        {"panel": 2, "source": "experiment", "identifier": "AT+", "n_replicates": len(at_plus)},
        {"panel": 3, "source": "escape_margin", "identifier": "ATagent", "n_replicates": atagent_margin_count},
        {"panel": 4, "source": "escape_margin", "identifier": "AT+", "n_replicates": at_plus_margin_count},
    ])
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(DATA_OUTPUT_DIR / "selected_trajectory_audit.tsv", sep="\t", index=False)

    print(f"[retained] ATagent={len(atagent)}, AT+={len(at_plus)}")
    for output in outputs:
        print(f"[saved] {output}")


if __name__ == "__main__":
    main()
