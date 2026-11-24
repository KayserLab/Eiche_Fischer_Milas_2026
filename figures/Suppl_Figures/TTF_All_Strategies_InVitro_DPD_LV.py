#!/usr/bin/env python3
"""Compare TTF across experimental, simulated, and fitted-GLV trajectories.

For each retained in-vitro replicate this script runs the *final fitted LvEnv*
used for the paper agent, from that replicate's observed day-1 sensitive and
resistant counts.  It evaluates the five study strategies: CT, AT50, AT100,
ATAgent (the frozen PPO policy), and AT+.

TTF is the first observation at which resistant cells reach 50% of the initial
total cell count, using acquisition days (the first observation is day 1).
The experimental and DPD selection rules match Figure 2's CT/AT100 panel:
experimental exclusions and cutoffs use acquisition days, and DPD runs start
at run 10. Only observed failures appear as points. Non-failures are retained
as right-censored records at their last observation for Kaplan--Meier medians,
including manually truncated experiments and incomplete DPD runs. GLV uses
the same plotting and censoring rules. Each source is followed to day 300.

Inputs
------
* Canonical experimental ``*_total_poisson_padded.h5`` point clouds.
* Canonical prepared simulation ``*_simulation.h5`` point clouds.
* The frozen GLV YAML configuration and PPO checkpoint used for Figure 4.
* ``experiment_filters.EXCLUDE_SAMPLES`` (full exclusions and truncations).

Outputs
-------
* One supplementary TTF PDF comparing all three sources.
* Replicate-level values, source/strategy summary, an LV run audit CSV, and a
  reusable GLV cache in ``intermediate_analysis_data``.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from analysis.figure_2.Event_Time_Analysis import (
    load_population_counts,
    prepare_progression_statistics,
    kaplan_meier_median,
    simulation_failure_times,
)
from shared.data_paths import FITTING, MODELS, figure_data_dir, figure_plot_dir, intermediate_dir
from shared.experiment_filters import apply_sample_exclusions
from shared.strategy_colors import get_color, get_display_name

LV_REPO = REPO_ROOT / "PhysiLearning" / "LV_Agent_Training_MiniRepo"
LV_SRC = LV_REPO / "src"
if str(LV_SRC) not in sys.path:
    sys.path.insert(0, str(LV_SRC))

from physilearning.envs.lv import LvEnv
from stable_baselines3 import PPO


# -----------------------------------------------------------------------------
# Scientific and figure settings
# -----------------------------------------------------------------------------
STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")
DISPLAY_LABELS = {
    "CT": "CT",
    "AT50": get_display_name("AT50", superscript=True),
    "AT100": get_display_name("AT100", superscript=True),
    "LV-Agent": get_display_name("LV-Agent", superscript=True),
    "AT100+": r"AT$^{+}$",
}
SOURCE_ORDER = ("Exp", "Sim", "GLV")
SOURCE_OFFSETS = {"Exp": -0.24, "Sim": 0.0, "GLV": 0.24}
SOURCE_MARKERS = {"Exp": "o", "Sim": "x", "GLV": "^"}
HORIZON_DAYS = 300.0
PLOT_Y_MAX_DAYS = 200.0
FAILURE_THRESHOLD = 0.5
MIN_DPD_RUN = 10
GLV_CACHE_VERSION = 2  # acquisition-day event times

LV_CONFIG = FITTING / 'PhysiLearning/LV_Agent_Training_MiniRepo/configs/2025_10_lv_real_mela_1_load.yaml'
EXPERIMENTAL_AGENT_MODEL = MODELS / 'experimental_agent'

DATA_DIR = figure_data_dir("Suppl_Figures", "ttf_all_strategies", create=False)
PLOT_DIR = figure_plot_dir("Suppl_Figures", "ttf_all_strategies", create=False)
GLV_CACHE_DIR = intermediate_dir("combined", "ttf_all_strategies_glv", create=False)

AXES_WIDTH_MM = 52.0
AXES_HEIGHT_MM = 38.0
MARGINS_MM = (11.0, 2.0, 9.0, 4.0)
POINT_SIZE = 11.0
POINT_JITTER = 0.055
MEDIAN_HALF_WIDTH = 0.065
MEDIAN_COLOR = "#222222"
DPI = 300
SHOW_FIGURE = True

ft.apply_nature_style(
    text_pt=7.0, tick_pt=6.0, legend_pt=5.0,
    overrides={"axes.linewidth": 0.8, "xtick.major.width": 0.8,
               "ytick.major.width": 0.8, "xtick.major.size": 2.5,
               "ytick.major.size": 2.5},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=float, default=HORIZON_DAYS)
    parser.add_argument("--output-data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument(
        "--recompute-glv", action="store_true",
        help="Ignore a valid GLV cache and calculate the 300-day GLV runs again.",
    )
    parser.add_argument("--hide-figure", action="store_true")
    return parser.parse_args()


def observed_ttf_rows(*, dataset_type: str, horizon_days: float) -> pd.DataFrame:
    """Use the CT/AT100 panel's event and censoring rules for every strategy."""
    counts = load_population_counts(
        dataset_type=dataset_type, strategies=STRATEGIES,
        simulation_min_run=MIN_DPD_RUN,
    )
    if counts.empty:
        raise RuntimeError(f"No {dataset_type} population trajectories were found.")
    counts = counts.loc[counts["day"] <= horizon_days].copy()
    if dataset_type == "simulation":
        # Canonical simulated endpoint (Figs. 2e, 4i): includes the day-1 baseline and
        # follow-up inclusion rules; output columns kept as before.
        events = simulation_failure_times(
            STRATEGIES, threshold=FAILURE_THRESHOLD, simulation_min_run=MIN_DPD_RUN, counts=counts,
        ).drop(columns=["n_acquisition_days", "first_acquisition_day", "input_file", "run"])
    else:
        events = prepare_progression_statistics(
            counts, apply_truncation=True, threshold=FAILURE_THRESHOLD,
        )
    events = events.rename(columns={
        "ttp": "ttf_days", "progressed": "failed",
        "last_observed_day": "last_available_day",
    })
    events["source"] = "Exp" if dataset_type == "experimental" else "Sim"
    events["censor_day"] = events["last_available_day"].where(~events["failed"])
    return events.drop(columns=["ttf"])


def configure_lv(
    initial_sensitive: float, initial_resistant: float, *, horizon_days: float,
) -> LvEnv:
    """Create one GLV environment, allocating enough storage for its horizon."""
    total = initial_sensitive + initial_resistant
    if total <= 0:
        raise ValueError("LV initial total must be positive")
    with LV_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["env"]["initial_wt"] = float(initial_sensitive / total)
    config["env"]["initial_mut"] = float(initial_resistant / total)
    # The frozen 140-day configuration allocates max_time=112000, or 800 fine
    # integration steps per day.  Extend only this storage/time limit so the
    # unchanged fitted dynamics and policy can be evaluated through the plot
    # horizon.  Four days of buffer avoids an environment boundary write after
    # the last observation interval.
    fine_steps_per_day = float(config["env"]["treatment_time_step"]) / 2.0
    required_max_time = int(np.ceil((horizon_days + 4.0) * fine_steps_per_day))
    config["env"]["max_time"] = max(int(config["env"]["max_time"]), required_max_time)
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
    """Load the extracted PPO archive without modifying the preserved checkpoint."""
    if EXPERIMENTAL_AGENT_MODEL.is_file():
        return PPO.load(str(EXPERIMENTAL_AGENT_MODEL))
    required = {"data", "policy.pth", "policy.optimizer.pth", "pytorch_variables.pth",
                "_stable_baselines3_version", "system_info.txt"}
    available = {item.name for item in EXPERIMENTAL_AGENT_MODEL.glob("*") if item.is_file()}
    missing = required - available
    if missing:
        raise FileNotFoundError(
            f"Incomplete PPO checkpoint at {EXPERIMENTAL_AGENT_MODEL}: {sorted(missing)}"
        )
    with tempfile.TemporaryDirectory(prefix="supplementary_ttf_ppo_") as temp_dir:
        archive_path = Path(temp_dir) / f"{EXPERIMENTAL_AGENT_MODEL.name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(EXPERIMENTAL_AGENT_MODEL.iterdir()):
                if item.is_file():
                    archive.write(item, arcname=item.name)
        return PPO.load(str(archive_path))


def choose_action(
    strategy: str, *, env: LvEnv, previous_action: int, policy: PPO | None,
    stacked: np.ndarray,
) -> int:
    """Paper protocol rules; action holds over the following LV observation interval."""
    burden = float(env.state[0] + env.state[1])
    if strategy == "CT":
        return 1
    if strategy == "AT50":
        return int(burden >= 1.0 or (bool(previous_action) and burden > 0.5))
    if strategy == "AT100":
        return int(burden >= 1.0)
    if strategy == "AT100+":
        return int(burden >= 1.0 and not previous_action)
    if strategy == "LV-Agent":
        if policy is None:
            raise RuntimeError("LV-Agent requested without a PPO policy")
        predicted, _ = policy.predict(stacked, deterministic=True)
        return int(np.asarray(predicted).reshape(-1)[0])
    raise ValueError(f"Unknown LV strategy: {strategy}")


def simulate_lv_ttf(
    baseline: pd.Series, *, strategy: str, horizon_days: float, policy: PPO | None,
) -> dict[str, object]:
    """Evolve one observed baseline, stopping immediately after failure."""
    env = configure_lv(
        float(baseline.sus), float(baseline.res), horizon_days=horizon_days,
    )
    observation, _ = env.reset(seed=0)
    observation = np.asarray(observation, dtype=np.float32)
    frames = [np.zeros_like(observation) for _ in range(3)]
    stacked = stacked_observation(frames, observation)
    initial_total = float(baseline.total)
    previous_action = 0
    failed = float(env.state[1]) >= FAILURE_THRESHOLD

    def result(*, failed: bool, ttf_days: float, last_day: float) -> dict[str, object]:
        return {
            "source": "GLV", "dataset_type": "lv", "plate": baseline.plate,
            "sample": str(baseline["sample"]), "strategy": strategy,
            "ttf_days": ttf_days, "failed": failed,
            "censor_day": np.nan if failed else last_day,
            "initial_total": initial_total, "last_available_day": last_day,
            "initial_sensitive": float(baseline.sus),
            "initial_resistant": float(baseline.res),
            "run_horizon_days": horizon_days,
            "glv_cache_version": GLV_CACHE_VERSION,
            "glv_config": str(LV_CONFIG),
            "glv_model": str(EXPERIMENTAL_AGENT_MODEL),
        }

    if failed:
        return result(failed=True, ttf_days=1.0, last_day=1.0)
    last_observed_day = float(env.day)
    while float(env.day) < horizon_days:
        action = choose_action(
            strategy, env=env, previous_action=previous_action,
            policy=policy, stacked=stacked,
        )
        observation, _reward, _terminated, _truncated, _info = env.step(action)
        previous_action = action
        stacked = stacked_observation(frames, np.asarray(observation, dtype=np.float32))
        acquisition_day = float(env.day)
        if acquisition_day > horizon_days:
            break
        last_observed_day = acquisition_day
        if float(env.state[1]) >= FAILURE_THRESHOLD:
            return result(failed=True, ttf_days=acquisition_day, last_day=acquisition_day)

    return result(
        failed=False, ttf_days=last_observed_day, last_day=last_observed_day,
    )


def lv_baselines(experimental_counts: pd.DataFrame) -> pd.DataFrame:
    """Return one retained experimental baseline for each GLV simulation."""
    filtered = apply_sample_exclusions(
        experimental_counts, column="sample", cutoff_column="day", strict=False
    )
    return (
        filtered.sort_values("day").groupby(["plate", "sample", "strategy"], as_index=False)
        .first()[["plate", "sample", "strategy", "sus", "res", "total"]]
    )


def glv_cache_path(horizon_days: float) -> Path:
    """Return the reusable intermediate-data cache path for one GLV horizon."""
    horizon_label = f"{horizon_days:g}".replace(".", "p")
    return GLV_CACHE_DIR / f"ttf_all_strategies_glv_horizon_{horizon_label}_days.csv"


def load_valid_glv_cache(
    path: Path, *, baselines: pd.DataFrame, horizon_days: float,
) -> pd.DataFrame | None:
    """Load a cache only when it contains the complete requested GLV run set."""
    if not path.is_file():
        return None
    try:
        cache = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        print(f"[cache] ignoring unreadable GLV cache {path}: {error}")
        return None
    required = {
        "source", "dataset_type", "plate", "sample", "strategy", "ttf_days",
        "failed", "censor_day", "initial_total", "last_available_day",
        "initial_sensitive", "initial_resistant", "run_horizon_days",
        "glv_cache_version", "glv_config", "glv_model",
    }
    missing = required.difference(cache.columns)
    if missing:
        print(f"[cache] ignoring GLV cache missing columns: {sorted(missing)}")
        return None
    expected_keys = {
        (str(row.plate), str(row.sample), strategy)
        for row in baselines.itertuples(index=False) for strategy in STRATEGIES
    }
    actual_keys = {
        (str(row.plate), str(row.sample), str(row.strategy))
        for row in cache.itertuples(index=False)
    }
    valid_horizon = np.allclose(
        pd.to_numeric(cache["run_horizon_days"], errors="coerce"), horizon_days
    )
    valid_version = pd.to_numeric(
        cache["glv_cache_version"], errors="coerce"
    ).eq(GLV_CACHE_VERSION).all()
    valid_provenance = (
        cache["glv_config"].eq(str(LV_CONFIG)).all()
        and cache["glv_model"].eq(str(EXPERIMENTAL_AGENT_MODEL)).all()
    )
    if (
        len(cache) != len(expected_keys)
        or actual_keys != expected_keys
        or not valid_horizon
        or not valid_version
        or not valid_provenance
    ):
        print(f"[cache] ignoring incomplete or incompatible GLV cache: {path}")
        return None
    cache["failed"] = cache["failed"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if cache["failed"].isna().any():
        print(f"[cache] ignoring GLV cache with invalid failure flags: {path}")
        return None
    print(f"[cache] using GLV results from {path}")
    return cache


def lv_ttf_rows(baselines: pd.DataFrame, *, horizon_days: float) -> pd.DataFrame:
    """Run all five GLV controls from every retained experimental baseline."""
    policy = load_experimental_agent()
    rows: list[dict[str, object]] = []
    total_runs = len(baselines) * len(STRATEGIES)
    with tqdm(total=total_runs, desc="LV TTF runs", unit="run") as progress:
        for baseline in baselines.itertuples(index=False):
            baseline_series = pd.Series(baseline._asdict())
            for strategy in STRATEGIES:
                rows.append(simulate_lv_ttf(
                    baseline_series, strategy=strategy, horizon_days=horizon_days, policy=policy,
                ))
                progress.update(1)
    return pd.DataFrame(rows)


def load_or_compute_glv_rows(
    experimental_counts: pd.DataFrame, *, horizon_days: float, recompute: bool,
) -> tuple[pd.DataFrame, Path]:
    """Reuse complete GLV results or calculate and cache them once."""
    baselines = lv_baselines(experimental_counts)
    cache_path = glv_cache_path(horizon_days)
    if not recompute:
        cached = load_valid_glv_cache(
            cache_path, baselines=baselines, horizon_days=horizon_days,
        )
        if cached is not None:
            return cached, cache_path

    for required in (LV_CONFIG, EXPERIMENTAL_AGENT_MODEL):
        if not required.exists():
            raise FileNotFoundError(f"Required GLV input not found: {required}")
    print(f"[cache] calculating GLV results through day {horizon_days:g}")
    rows = lv_ttf_rows(baselines, horizon_days=horizon_days)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(cache_path, index=False)
    print(f"[saved] {cache_path}")
    return rows, cache_path


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        for source in SOURCE_ORDER:
            group = values[(values.strategy == strategy) & (values.source == source)]
            ttf = group.loc[group["failed"], "ttf_days"].dropna()
            rows.append({
                "strategy": strategy, "source": source, "n": len(group),
                "n_failed": int(group["failed"].sum()),
                "n_censored": int((~group["failed"]).sum()),
                "ttf_median_days": kaplan_meier_median(group["ttf_days"], group["failed"]),
                "ttf_mean_days": float(ttf.mean()) if len(ttf) else np.nan,
            })
    return pd.DataFrame(rows)


def plot(values: pd.DataFrame, path: Path, *, y_max_days: float) -> None:
    fig, ax = ft.new_figure(
        width=AXES_WIDTH_MM, height_mm=AXES_HEIGHT_MM, margins=MARGINS_MM,
        dpi=DPI, apply_style=False,
    )
    rng = np.random.default_rng(20260826)
    centers = np.arange(1, len(STRATEGIES) + 1, dtype=float)
    for center, strategy in zip(centers, STRATEGIES):
        color = get_color(strategy, "#777777")
        for source in SOURCE_ORDER:
            group = values[(values.strategy == strategy) & (values.source == source)].copy()
            if group.empty:
                continue
            # Only failures are points; all retained rows inform the KM median.
            failures = group.loc[group["failed"] & group["ttf_days"].notna()]
            x = center + SOURCE_OFFSETS[source] + rng.uniform(-POINT_JITTER, POINT_JITTER, len(failures))
            y = failures["ttf_days"].to_numpy(dtype=float)
            ax.scatter(x, y, s=POINT_SIZE, marker=SOURCE_MARKERS[source],
                       facecolors=color, edgecolors="black", linewidths=0.3, zorder=3)
            median = kaplan_meier_median(group["ttf_days"], group["failed"])
            if len(y) and np.isfinite(median):
                ax.hlines(median, center + SOURCE_OFFSETS[source] - MEDIAN_HALF_WIDTH,
                          center + SOURCE_OFFSETS[source] + MEDIAN_HALF_WIDTH,
                          color=MEDIAN_COLOR, linewidth=1.0, zorder=4)

    ax.set_xlim(0.5, len(STRATEGIES) + 0.5)
    ax.set_ylim(0, y_max_days+5)
    ax.set_yticks(np.arange(0, y_max_days + 1, 20))
    ax.set_xticks(centers)
    ax.set_xticklabels([DISPLAY_LABELS.get(item, get_display_name(item)) for item in STRATEGIES])
    for tick, strategy in zip(ax.get_xticklabels(), STRATEGIES):
        tick.set_color(get_color(strategy, "#777777"))
    ax.set_ylabel("Time to failure, TTF [d]")
    ax.grid(axis="y", alpha=0.18, linewidth=0.4)
    source_handles = [Line2D([0], [0], marker=SOURCE_MARKERS[source], color="none",
                             markerfacecolor="#777777", markeredgecolor="black",
                             markersize=4.2, label=source) for source in SOURCE_ORDER]
    ax.legend(handles=source_handles, loc="upper left", frameon=False,
              fontsize=5, handletextpad=0.4, labelspacing=0.3)
    ft.savefig(fig, path)
    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.horizon_days <= 0:
        raise ValueError("--horizon-days must be positive")
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)

    experimental_counts = load_population_counts(
        dataset_type="experimental", strategies=STRATEGIES
    )
    if experimental_counts.empty:
        raise RuntimeError("No experimental trajectories available for LV baselines.")
    invitro = observed_ttf_rows(dataset_type="experimental", horizon_days=args.horizon_days)
    dpd = observed_ttf_rows(dataset_type="simulation", horizon_days=args.horizon_days)
    lv, cache_path = load_or_compute_glv_rows(
        experimental_counts, horizon_days=args.horizon_days,
        recompute=args.recompute_glv,
    )
    values = pd.concat([invitro, dpd, lv], ignore_index=True, sort=False)
    summary = summarize(values)

    values_path = args.output_data_dir / "ttf_all_strategies_invitro_dpd_lv_per_replicate.csv"
    summary_path = args.output_data_dir / "ttf_all_strategies_invitro_dpd_lv_summary.csv"
    lv_path = args.output_data_dir / "ttf_all_strategies_lv_run_audit.csv"
    pdf_path = args.output_plot_dir / "TTF_All_Strategies_InVitro_DPD_LV.pdf"
    values.to_csv(values_path, index=False)
    summary.to_csv(summary_path, index=False)
    lv.to_csv(lv_path, index=False)
    plot(values, pdf_path, y_max_days=min(args.horizon_days, PLOT_Y_MAX_DAYS))
    print(f"[saved] {values_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {lv_path}")
    print(f"[saved] {pdf_path}")
    print(f"[cache] {cache_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
