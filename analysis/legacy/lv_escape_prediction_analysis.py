#!/usr/bin/env python3
"""
Analyze how the fitted LV agent-training environment predicts resistant-cell
counts and fractions before and after experimental escape.

All outputs are written next to this script under `lv_escape_prediction_outputs`.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
OUT_DIR = SCRIPT_PATH.parent / "lv_escape_prediction_outputs"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"

DATA_DIR = (
    REPO_ROOT
    / "PhysiLearning"
    / "LV_Fitting_MiniRepo"
    / "data"
    / "experimental_data_elias"
    / "2025_stem_paper_final_final"
)
CONFIG_PATH = (
    REPO_ROOT
    / "PhysiLearning"
    / "LV_Agent_Training_MiniRepo"
    / "configs"
    / "2025_10_lv_real_mela_1_load.yaml"
)
PARAM_PATH = (
    REPO_ROOT
    / "PhysiLearning"
    / "LV_Agent_Training_MiniRepo"
    / "reference"
    / "fitted_lv_params.json"
)
ESCAPE_SUMMARY_CSV = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Data\escape_front_batch_summary_all_res_within_sens.csv"
)

STRATEGY_ORDER = ["CT", "AT50", "AT100", "Pulse", "NT"]
STRATEGY_COLORS = {
    "CT": "#3b73b9",
    "AT50": "#52a65a",
    "AT100": "#d8842f",
    "Pulse": "#9b5cbd",
    "NT": "#666666",
}
PSEUDOCOUNT = 0.5


@dataclass
class LvParams:
    growth_rate_wt: float
    growth_rate_mut: float
    death_rate_wt: float
    death_rate_mut: float
    treat_death_rate_wt: float
    treat_death_rate_mut: float
    carrying_capacity: float
    competition_wt: float
    competition_mut: float
    timestep_size: float
    ramp_time_up: float
    ramp_time_down: float
    min_death_during_treat: float
    on_treat_threshold: float
    off_treat_threshold: float
    treatment_time_step: int


@dataclass
class LvState:
    sensitive: float
    resistant: float
    drug: float = 0.0
    day: int = 1
    time: int = 0
    time_on_treatment: float = 0.0
    time_off_treatment: float = 100.0
    growth_fraction: float = 1.0
    death_fraction: float = 0.0


def parse_strategy(sample: str) -> str:
    try:
        return sample.split("_", 1)[1].split("-", 1)[0]
    except Exception:
        return ""


def parse_plate_name(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"Plate \d+", part):
            return part
    raise ValueError(f"Could not infer plate from {path}")


def parse_sample_from_result_file(path: Path) -> str:
    stem = path.stem
    prefix = "Results_G418_a2a4_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    raise ValueError(f"Unexpected result filename: {path.name}")


def load_exclude_samples() -> dict[str, float | None]:
    sys.path.insert(0, str(REPO_ROOT))
    from shared import experiment_filters  # noqa: PLC0415

    return dict(experiment_filters.EXCLUDE_SAMPLES)


def load_lv_params() -> LvParams:
    with PARAM_PATH.open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return LvParams(
        growth_rate_wt=float(params["growth_rate_wt"]),
        growth_rate_mut=float(params["growth_rate_mut"]),
        death_rate_wt=float(params["death_rate_wt"]),
        death_rate_mut=float(params["death_rate_mut"]),
        treat_death_rate_wt=float(params["treat_death_rate_wt"]),
        treat_death_rate_mut=float(params["treat_death_rate_mut"]),
        carrying_capacity=float(params["carrying_capacity"]),
        competition_wt=float(params["competition_wt"]),
        competition_mut=float(params["competition_mut"]),
        timestep_size=float(params["timestep_size"]),
        ramp_time_up=float(params["ramp_time_up"]),
        ramp_time_down=float(params["ramp_time_down"]),
        min_death_during_treat=float(params["min_death_during_treat"]),
        on_treat_threshold=float(params["on_treat_threshold"]),
        off_treat_threshold=float(params["off_treat_threshold"]),
        treatment_time_step=int(config["env"]["treatment_time_step"]),
    )


def load_treatment_schedules() -> dict[str, set[int]]:
    schedules: dict[str, set[int]] = {}
    for path in sorted(DATA_DIR.glob("Plate */treatment_schedule.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                sample = parts[0]
                schedules[sample] = {int(v) for v in parts[1:]}
    return schedules


def load_count_data(exclude_samples: dict[str, float | None]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(DATA_DIR.glob("Plate */Results/Results_G418_a2a4_*.txt")):
        sample = parse_sample_from_result_file(path)
        strategy = parse_strategy(sample)
        if strategy not in STRATEGY_ORDER:
            continue
        if exclude_samples.get(sample, "keep") is None:
            continue

        df = pd.read_csv(
            path,
            sep=r"\s+",
            names=["raw_name", "sensitive_count", "resistant_count"],
            engine="python",
        )
        df["day"] = df["raw_name"].str.extract(r"_d(\d+)$").astype(int)
        cutoff = exclude_samples.get(sample)
        if cutoff is not None and cutoff != "keep":
            df = df[df["day"] <= float(cutoff)].copy()
        if df.empty:
            continue

        for obs_index, (_, row) in enumerate(df.sort_values("day").iterrows(), start=1):
            rows.append(
                {
                    "plate": parse_plate_name(path),
                    "sample": sample,
                    "strategy": strategy,
                    "obs_index": obs_index,
                    "day": int(row["day"]),
                    "sensitive_count": float(row["sensitive_count"]),
                    "resistant_count": float(row["resistant_count"]),
                    "source_file": str(path.relative_to(REPO_ROOT)),
                }
            )

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError(f"No count data loaded from {DATA_DIR}")
    data["total_count"] = data["sensitive_count"] + data["resistant_count"]
    data["resistant_fraction"] = data["resistant_count"] / data["total_count"]
    return data


def load_escape_table(exclude_samples: dict[str, float | None]) -> pd.DataFrame:
    if not ESCAPE_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Escape summary CSV not found: {ESCAPE_SUMMARY_CSV}")
    df = pd.read_csv(ESCAPE_SUMMARY_CSV)
    rows = []
    for sample, g in df.groupby("sample"):
        sample = str(sample)
        strategy = str(g["strategy"].dropna().iloc[0]) if "strategy" in g else parse_strategy(sample)
        escape_detected = (
            g.get("escape_detected", pd.Series(False, index=g.index))
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
            .any()
        )
        escape_values = pd.to_numeric(g["first_escape_day"], errors="coerce")
        escape_values = escape_values[np.isfinite(escape_values) & (escape_values >= 0)]
        raw_escape_day = float(escape_values.min()) if escape_detected and len(escape_values) else np.nan

        rule = exclude_samples.get(sample, "keep")
        filtered_escape_day = raw_escape_day
        filter_status = "keep"
        if rule is None:
            filter_status = "excluded"
            filtered_escape_day = np.nan
        elif rule != "keep":
            cutoff = float(rule)
            filter_status = f"truncated_at_{cutoff:g}"
            if not np.isfinite(raw_escape_day) or raw_escape_day > cutoff:
                filtered_escape_day = np.nan

        rows.append(
            {
                "sample": sample,
                "strategy": strategy,
                "escape_detected_raw": bool(np.isfinite(raw_escape_day)),
                "escape_day_raw": raw_escape_day,
                "escape_day": filtered_escape_day,
                "filter_status_for_escape": filter_status,
            }
        )
    return pd.DataFrame(rows)


def grow_one_clone(
    state: LvState,
    params: LvParams,
    clone_index: int,
    other_index: int,
) -> float:
    if state.drug == 1:
        if state.time_off_treatment > 0:
            state.time_off_treatment = 0
        state.time_on_treatment += params.timestep_size
    else:
        if state.time_on_treatment > 0:
            state.time_on_treatment = 0
        state.time_off_treatment += params.timestep_size

    step_up = params.timestep_size * 2 / params.ramp_time_up
    step_down = params.timestep_size * 2 / params.ramp_time_down

    if (state.drug == 1 and state.time_on_treatment > params.on_treat_threshold) or (
        state.drug == 0 and state.time_off_treatment < params.off_treat_threshold
    ):
        if state.growth_fraction > 0:
            state.growth_fraction = max(0.0, state.growth_fraction - step_down)
        else:
            state.death_fraction = min(1.0, state.death_fraction + step_down)
    else:
        if state.death_fraction > 0:
            state.death_fraction = max(0.0, state.death_fraction - step_up)
        else:
            state.growth_fraction = min(1.0, state.growth_fraction + step_up)

    sensitive = state.sensitive
    resistant = state.resistant
    clone = sensitive if clone_index == 0 else resistant
    other = resistant if other_index == 1 else sensitive

    growth_rates = [
        params.growth_rate_wt * state.growth_fraction,
        params.growth_rate_mut,
    ]
    treatment_death_rates = [
        params.treat_death_rate_wt * state.death_fraction,
        params.treat_death_rate_mut,
    ]
    basal_death_rates = [params.death_rate_wt, params.death_rate_mut]
    competition = [params.competition_wt, params.competition_mut]
    treat = [1, 0] if state.death_fraction > 0 else [0, 0]

    new_size = clone * (
        1
        + params.timestep_size
        * growth_rates[clone_index]
        * (1 - (clone + other * competition[other_index]) / params.carrying_capacity)
        - treatment_death_rates[clone_index] * treat[clone_index] * params.timestep_size
        - basal_death_rates[clone_index] * params.timestep_size
    ) - treat[clone_index] * params.min_death_during_treat * params.timestep_size

    if new_size < 1.0e-5:
        new_size = 0.0
    return float(new_size)


def step_lv(state: LvState, params: LvParams, action: int) -> LvState:
    if state.day == 5 or (state.day > 5 and (state.day - 5) % 7 == 0):
        interval_steps = int(params.treatment_time_step * 1.5)
        state.day += 3
    else:
        interval_steps = int(params.treatment_time_step)
        state.day += 2

    for _ in range(interval_steps):
        state.time += 1
        state.drug = float(action)
        state.sensitive = grow_one_clone(state, params, clone_index=0, other_index=1)
        state.resistant = grow_one_clone(state, params, clone_index=1, other_index=0)
        if state.sensitive <= 0 and state.resistant <= 0:
            state.sensitive = 0.0
            state.resistant = 0.0
    return state


def simulate_sample(sample_df: pd.DataFrame, treated_indices: set[int], params: LvParams) -> pd.DataFrame:
    sample_df = sample_df.sort_values("day").copy()
    initial_total = float(sample_df["total_count"].iloc[0])
    state = LvState(
        sensitive=float(sample_df["sensitive_count"].iloc[0] / initial_total),
        resistant=float(sample_df["resistant_count"].iloc[0] / initial_total),
    )

    predictions = [
        {
            "obs_index": int(sample_df["obs_index"].iloc[0]),
            "pred_sensitive_norm": state.sensitive,
            "pred_resistant_norm": state.resistant,
            "action_from_previous": np.nan,
        }
    ]

    for position in range(1, len(sample_df)):
        action_index = int(sample_df["obs_index"].iloc[position - 1])
        action = 1 if action_index in treated_indices else 0
        state = step_lv(state, params, action)
        predictions.append(
            {
                "obs_index": int(sample_df["obs_index"].iloc[position]),
                "pred_sensitive_norm": state.sensitive,
                "pred_resistant_norm": state.resistant,
                "action_from_previous": action,
            }
        )

    pred = pd.DataFrame(predictions)
    pred["pred_total_norm"] = pred["pred_sensitive_norm"] + pred["pred_resistant_norm"]
    pred["pred_resistant_fraction"] = np.where(
        pred["pred_total_norm"] > 0,
        pred["pred_resistant_norm"] / pred["pred_total_norm"],
        np.nan,
    )
    pred["initial_total_count"] = initial_total
    pred["pred_sensitive_count"] = pred["pred_sensitive_norm"] * initial_total
    pred["pred_resistant_count"] = pred["pred_resistant_norm"] * initial_total
    pred["pred_total_count"] = pred["pred_total_norm"] * initial_total
    return sample_df.merge(pred, on="obs_index", how="left")


def build_prediction_table(
    count_df: pd.DataFrame,
    schedules: dict[str, set[int]],
    escape_df: pd.DataFrame,
    params: LvParams,
) -> pd.DataFrame:
    frames = []
    for sample, sample_df in count_df.groupby("sample", sort=True):
        frames.append(simulate_sample(sample_df, schedules.get(sample, set()), params))
    out = pd.concat(frames, ignore_index=True)
    out = out.merge(
        escape_df[
            [
                "sample",
                "escape_detected_raw",
                "escape_day_raw",
                "escape_day",
                "filter_status_for_escape",
            ]
        ],
        on="sample",
        how="left",
    )
    out["phase"] = "no_escape_observed"
    escaped = np.isfinite(out["escape_day"])
    out.loc[escaped & (out["day"] < out["escape_day"]), "phase"] = "pre_escape"
    out.loc[escaped & (out["day"] >= out["escape_day"]), "phase"] = "post_escape"

    for variable in ["sensitive", "resistant", "total"]:
        out[f"{variable}_count_error"] = out[f"pred_{variable}_count"] - out[f"{variable}_count"]
        out[f"{variable}_count_abs_error"] = out[f"{variable}_count_error"].abs()
        out[f"{variable}_count_log2_ratio"] = np.log2(
            (out[f"pred_{variable}_count"] + PSEUDOCOUNT)
            / (out[f"{variable}_count"] + PSEUDOCOUNT)
        )
    out["resistant_fraction_error"] = out["pred_resistant_fraction"] - out["resistant_fraction"]
    out["resistant_fraction_abs_error"] = out["resistant_fraction_error"].abs()
    out["days_relative_to_escape"] = out["day"] - out["escape_day"]
    return out


def rmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr**2)))


def summarize_points(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(["strategy", "phase"], dropna=False):
        strategy, phase = keys
        rows.append(
            {
                "strategy": strategy,
                "phase": phase,
                "n_points": int(len(g)),
                "n_samples": int(g["sample"].nunique()),
                "mean_resistant_fraction_error": float(g["resistant_fraction_error"].mean()),
                "median_resistant_fraction_error": float(g["resistant_fraction_error"].median()),
                "mae_resistant_fraction": float(g["resistant_fraction_abs_error"].mean()),
                "rmse_resistant_fraction": rmse(g["resistant_fraction_error"]),
                "mean_resistant_count_log2_ratio": float(g["resistant_count_log2_ratio"].mean()),
                "median_resistant_count_log2_ratio": float(g["resistant_count_log2_ratio"].median()),
                "mae_resistant_count": float(g["resistant_count_abs_error"].mean()),
                "rmse_resistant_count": rmse(g["resistant_count_error"]),
                "mean_total_count_log2_ratio": float(g["total_count_log2_ratio"].mean()),
                "median_total_count_log2_ratio": float(g["total_count_log2_ratio"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "phase"])


def summarize_replicates(pred: pd.DataFrame, full_pred: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if full_pred is None:
        full_pred = pred
    initial_lookup = (
        full_pred.sort_values(["sample", "obs_index"])
        .groupby("sample", as_index=True)
        .first()[["initial_total_count", "resistant_fraction"]]
        .rename(columns={"resistant_fraction": "initial_resistant_fraction"})
    )
    for sample, g in pred.groupby("sample", sort=True):
        initial = initial_lookup.loc[sample]
        row = {
            "sample": sample,
            "strategy": str(g["strategy"].iloc[0]),
            "n_points": int(len(g)),
            "initial_total_count": float(initial["initial_total_count"]),
            "initial_resistant_fraction": float(initial["initial_resistant_fraction"]),
            "escape_day": float(g["escape_day"].iloc[0])
            if np.isfinite(g["escape_day"].iloc[0])
            else np.nan,
            "escape_day_raw": float(g["escape_day_raw"].iloc[0])
            if np.isfinite(g["escape_day_raw"].iloc[0])
            else np.nan,
            "filter_status_for_escape": str(g["filter_status_for_escape"].iloc[0]),
        }
        for phase in ["pre_escape", "post_escape", "no_escape_observed"]:
            sub = g[g["phase"] == phase]
            row[f"n_{phase}"] = int(len(sub))
            row[f"mean_res_frac_error_{phase}"] = (
                float(sub["resistant_fraction_error"].mean()) if len(sub) else np.nan
            )
            row[f"mean_res_count_log2_ratio_{phase}"] = (
                float(sub["resistant_count_log2_ratio"].mean()) if len(sub) else np.nan
            )
            row[f"rmse_res_frac_{phase}"] = rmse(sub["resistant_fraction_error"])
            row[f"rmse_res_count_{phase}"] = rmse(sub["resistant_count_error"])
        row["shift_res_frac_error_post_minus_pre"] = (
            row["mean_res_frac_error_post_escape"] - row["mean_res_frac_error_pre_escape"]
        )
        row["shift_res_count_log2_ratio_post_minus_pre"] = (
            row["mean_res_count_log2_ratio_post_escape"]
            - row["mean_res_count_log2_ratio_pre_escape"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "sample"])


def describe_vector(values: pd.Series) -> dict[str, float | int]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def pvalue_or_nan(func, values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    try:
        return float(func(values).pvalue)
    except ValueError:
        return np.nan


def build_stat_tests(rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        (
            "resistant_fraction_error",
            "mean_res_frac_error_pre_escape",
            "mean_res_frac_error_post_escape",
            "shift_res_frac_error_post_minus_pre",
        ),
        (
            "resistant_count_log2_ratio",
            "mean_res_count_log2_ratio_pre_escape",
            "mean_res_count_log2_ratio_post_escape",
            "shift_res_count_log2_ratio_post_minus_pre",
        ),
    ]
    for metric, pre_col, post_col, shift_col in metrics:
        both = rep.dropna(subset=[pre_col, post_col]).copy()
        shift = both[shift_col].to_numpy(dtype=float)
        rows.append(
            {
                "test": "paired_post_minus_pre",
                "metric": metric,
                **describe_vector(pd.Series(shift)),
                "wilcoxon_p": pvalue_or_nan(lambda v: stats.wilcoxon(v), shift),
                "ttest_1samp_p": pvalue_or_nan(lambda v: stats.ttest_1samp(v, 0.0), shift),
                "n_positive_shift": int(np.sum(shift > 0)),
                "n_negative_shift": int(np.sum(shift < 0)),
                "binomial_two_sided_p_positive_vs_negative": float(
                    stats.binomtest(
                        int(np.sum(shift > 0)),
                        int(np.sum(shift > 0) + np.sum(shift < 0)),
                        p=0.5,
                    ).pvalue
                )
                if int(np.sum(shift > 0) + np.sum(shift < 0)) > 0
                else np.nan,
            }
        )

        for phase, col in [("pre_escape", pre_col), ("post_escape", post_col)]:
            vals = rep[col].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "test": f"one_sample_bias_{phase}",
                    "metric": metric,
                    **describe_vector(pd.Series(vals)),
                    "wilcoxon_p": pvalue_or_nan(lambda v: stats.wilcoxon(v), vals),
                    "ttest_1samp_p": pvalue_or_nan(lambda v: stats.ttest_1samp(v, 0.0), vals),
                    "n_positive_shift": int(np.sum(vals > 0)),
                    "n_negative_shift": int(np.sum(vals < 0)),
                    "binomial_two_sided_p_positive_vs_negative": float(
                        stats.binomtest(
                            int(np.sum(vals > 0)),
                            int(np.sum(vals > 0) + np.sum(vals < 0)),
                            p=0.5,
                        ).pvalue
                    )
                    if int(np.sum(vals > 0) + np.sum(vals < 0)) > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def ordered_strategies(values: Iterable[str]) -> list[str]:
    present = set(values)
    return [s for s in STRATEGY_ORDER if s in present] + sorted(present - set(STRATEGY_ORDER))


def plot_predicted_vs_observed(pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    specs = [
        ("resistant_fraction", "pred_resistant_fraction", "Resistant fraction"),
        ("resistant_count", "pred_resistant_count", "Resistant count"),
    ]
    for ax, (obs_col, pred_col, label) in zip(axes, specs):
        for strategy in ordered_strategies(pred["strategy"]):
            sub = pred[pred["strategy"] == strategy]
            ax.scatter(
                sub[obs_col],
                sub[pred_col],
                s=18,
                alpha=0.7,
                label=strategy,
                color=STRATEGY_COLORS.get(strategy, "#333333"),
                edgecolor="none",
            )
        finite = pred[[obs_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        lo = float(finite.min().min())
        hi = float(finite.max().max())
        if obs_col.endswith("count"):
            lo = max(PSEUDOCOUNT, lo)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, alpha=0.7)
        ax.set_xlabel(f"Observed {label.lower()}")
        ax.set_ylabel(f"Predicted {label.lower()}")
        ax.set_title(label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "predicted_vs_observed.png", dpi=300)
    plt.close(fig)


def plot_pre_post_bias(rep: pd.DataFrame) -> None:
    escaped = rep.dropna(
        subset=[
            "mean_res_frac_error_pre_escape",
            "mean_res_frac_error_post_escape",
            "mean_res_count_log2_ratio_pre_escape",
            "mean_res_count_log2_ratio_post_escape",
        ]
    ).copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    plot_specs = [
        (
            "mean_res_frac_error_pre_escape",
            "mean_res_frac_error_post_escape",
            "Mean resistant-fraction error",
        ),
        (
            "mean_res_count_log2_ratio_pre_escape",
            "mean_res_count_log2_ratio_post_escape",
            "Mean resistant-count log2(pred/obs)",
        ),
    ]
    for ax, (pre_col, post_col, ylabel) in zip(axes, plot_specs):
        values = [escaped[pre_col].dropna(), escaped[post_col].dropna()]
        ax.boxplot(values, positions=[0, 1], widths=0.45, showfliers=False)
        for _, row in escaped.iterrows():
            color = STRATEGY_COLORS.get(row["strategy"], "#333333")
            ax.plot([0, 1], [row[pre_col], row[post_col]], color=color, alpha=0.45, lw=1)
            ax.scatter([0, 1], [row[pre_col], row[post_col]], color=color, s=20, zorder=3)
        ax.axhline(0, color="black", lw=1, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"])
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pre_post_bias_by_replicate.png", dpi=300)
    plt.close(fig)


def plot_relative_to_escape(pred: pd.DataFrame) -> None:
    escaped = pred[np.isfinite(pred["days_relative_to_escape"])].copy()
    if escaped.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    specs = [
        ("resistant_fraction_error", "Resistant-fraction error"),
        ("resistant_count_log2_ratio", "Resistant-count log2(pred/obs)"),
    ]
    for ax, (col, label) in zip(axes, specs):
        for strategy in ordered_strategies(escaped["strategy"]):
            sub = escaped[escaped["strategy"] == strategy]
            ax.scatter(
                sub["days_relative_to_escape"],
                sub[col],
                s=16,
                alpha=0.45,
                color=STRATEGY_COLORS.get(strategy, "#333333"),
                label=strategy,
            )
        bins = np.arange(
            math.floor(escaped["days_relative_to_escape"].min() / 7) * 7,
            math.ceil(escaped["days_relative_to_escape"].max() / 7) * 7 + 8,
            7,
        )
        escaped["rel_bin"] = pd.cut(escaped["days_relative_to_escape"], bins=bins, include_lowest=True)
        binned = escaped.groupby("rel_bin", observed=False).agg(
            x=("days_relative_to_escape", "mean"),
            y=(col, "mean"),
            n=(col, "size"),
        )
        binned = binned[binned["n"] >= 2]
        ax.plot(binned["x"], binned["y"], color="black", lw=2, label="7-day bin mean")
        ax.axhline(0, color="black", lw=1, alpha=0.5)
        ax.axvline(0, color="#b00020", lw=1.2, alpha=0.8)
        ax.set_xlabel("Days relative to escape")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "errors_relative_to_escape.png", dpi=300)
    plt.close(fig)


def plot_strategy_time_courses(pred: pd.DataFrame) -> None:
    strategies = ordered_strategies(pred["strategy"])
    fig, axes = plt.subplots(len(strategies), 1, figsize=(8, 2.2 * len(strategies)), sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies):
        sub = pred[pred["strategy"] == strategy]
        grouped = sub.groupby("day").agg(
            obs=("resistant_fraction", "mean"),
            pred=("pred_resistant_fraction", "mean"),
            n=("sample", "nunique"),
        )
        ax.plot(grouped.index, grouped["obs"], marker="o", lw=1.5, label="Observed")
        ax.plot(grouped.index, grouped["pred"], marker="s", lw=1.5, label="LV predicted")
        ax.set_ylabel(strategy)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Day")
    fig.supylabel("Mean resistant fraction")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "strategy_mean_resistant_fraction_time_courses.png", dpi=300)
    plt.close(fig)


def plot_example_trajectories(pred: pd.DataFrame) -> None:
    escaped_samples = (
        pred[np.isfinite(pred["escape_day"])]
        .groupby("sample")
        .agg(escape_day=("escape_day", "first"), strategy=("strategy", "first"))
        .sort_values("escape_day")
        .head(12)
        .index.tolist()
    )
    if not escaped_samples:
        return
    ncols = 3
    nrows = int(math.ceil(len(escaped_samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, sample in zip(axes, escaped_samples):
        sub = pred[pred["sample"] == sample].sort_values("day")
        color = STRATEGY_COLORS.get(str(sub["strategy"].iloc[0]), "#333333")
        ax.plot(sub["day"], sub["resistant_fraction"], marker="o", lw=1.5, color=color, label="Observed")
        ax.plot(
            sub["day"],
            sub["pred_resistant_fraction"],
            marker="s",
            lw=1.5,
            color="black",
            label="LV predicted",
        )
        escape_day = float(sub["escape_day"].iloc[0])
        ax.axvline(escape_day, color="#b00020", lw=1, alpha=0.8)
        ax.set_title(f"{sample} (escape day {escape_day:g})", fontsize=9)
        ax.set_xlabel("Day")
        ax.set_ylabel("Res. fraction")
        ax.set_ylim(bottom=0)
    for ax in axes[len(escaped_samples) :]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "example_escaped_replicate_trajectories.png", dpi=300)
    plt.close(fig)


def plot_error_heatmap(rep: pd.DataFrame) -> None:
    plot_df = rep.copy()
    plot_df["sample_label"] = plot_df["sample"]
    plot_df = plot_df.sort_values(["strategy", "escape_day", "sample"])
    values = plot_df[
        [
            "mean_res_frac_error_pre_escape",
            "mean_res_frac_error_post_escape",
            "shift_res_frac_error_post_minus_pre",
            "mean_res_count_log2_ratio_pre_escape",
            "mean_res_count_log2_ratio_post_escape",
            "shift_res_count_log2_ratio_post_minus_pre",
        ]
    ].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.18 * len(plot_df))))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(values)), vmax=np.nanmax(np.abs(values)))
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["sample_label"], fontsize=6)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(
        ["frac pre", "frac post", "frac shift", "log2 pre", "log2 post", "log2 shift"],
        rotation=45,
        ha="right",
    )
    fig.colorbar(im, ax=ax, label="Signed error / shift")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "per_replicate_error_heatmap.png", dpi=300)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows).copy()
    formatted = table.astype(object).where(pd.notna(table), "NA")
    for col in formatted.columns:
        formatted[col] = formatted[col].map(
            lambda v: fmt(float(v), 4) if isinstance(v, (float, np.floating)) else str(v)
        )
    headers = [str(c) for c in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in formatted.columns) + " |")
    return "\n".join(lines)


def write_markdown_report(
    pred: pd.DataFrame,
    eval_pred: pd.DataFrame,
    point_summary: pd.DataFrame,
    rep_summary: pd.DataFrame,
    tests: pd.DataFrame,
    params: LvParams,
) -> None:
    n_samples = pred["sample"].nunique()
    n_escaped = rep_summary["escape_day"].notna().sum()
    paired_frac = tests[
        (tests["test"] == "paired_post_minus_pre")
        & (tests["metric"] == "resistant_fraction_error")
    ].iloc[0]
    paired_log = tests[
        (tests["test"] == "paired_post_minus_pre")
        & (tests["metric"] == "resistant_count_log2_ratio")
    ].iloc[0]
    pre_frac = tests[
        (tests["test"] == "one_sample_bias_pre_escape")
        & (tests["metric"] == "resistant_fraction_error")
    ].iloc[0]
    post_frac = tests[
        (tests["test"] == "one_sample_bias_post_escape")
        & (tests["metric"] == "resistant_fraction_error")
    ].iloc[0]
    strategy_notes = []
    for strategy in STRATEGY_ORDER:
        sub = point_summary[point_summary["strategy"] == strategy]
        if sub.empty:
            continue
        pre = sub[sub["phase"] == "pre_escape"]
        post = sub[sub["phase"] == "post_escape"]
        if not pre.empty and not post.empty:
            strategy_notes.append(
                f"- {strategy}: point-level mean resistant-fraction error changed from "
                f"{fmt(float(pre['mean_resistant_fraction_error'].iloc[0]))} pre-escape to "
                f"{fmt(float(post['mean_resistant_fraction_error'].iloc[0]))} post-escape."
            )

    lines = [
        "# LV Prediction Before and After Escape",
        "",
        "## Scope",
        "",
        (
            f"This analysis used {n_samples} replicate count trajectories after `experiment_filters` "
            f"and {n_escaped} replicates with an escape day still inside the filtered observation window. "
            f"Error summaries use {len(eval_pred)} post-initialization observations; the day-1 initialization "
            "row is excluded because prediction equals observation by construction."
        ),
        "",
        "Inputs:",
        "",
        f"- Count data: `{DATA_DIR.relative_to(REPO_ROOT)}`",
        f"- Escape summary: `{ESCAPE_SUMMARY_CSV}`",
        f"- Fitted LV parameters used for agent training: `{PARAM_PATH.relative_to(REPO_ROOT)}`",
        f"- Agent-training config: `{CONFIG_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Model Replay",
        "",
        (
            "Each replicate was initialized from its day-1 sensitive and resistant counts, normalized by "
            "that replicate's initial total, matching the LV fitting workflow. The fitted LV environment "
            "was replayed with the actual per-replicate treatment schedule from `treatment_schedule.txt`. "
            "Predicted normalized values were multiplied by the replicate's initial total for count-scale "
            "comparisons."
        ),
        "",
        "Key LV parameters used:",
        "",
        f"- growth rates: sensitive={params.growth_rate_wt}, resistant={params.growth_rate_mut}",
        f"- treatment death sensitive={params.treat_death_rate_wt}, resistant={params.treat_death_rate_mut}",
        f"- carrying capacity={params.carrying_capacity}, competition_wt={params.competition_wt}, competition_mut={params.competition_mut}",
        f"- ramp up={params.ramp_time_up}, ramp down={params.ramp_time_down}, treatment step={params.treatment_time_step}",
        "",
        "## Main Results",
        "",
        (
            "Primary signed resistant-fraction residuals are absolute fraction differences, defined as "
            "`predicted fraction - observed fraction`. For example, +0.10 means +10 percentage points, "
            "not +10 percent relative error. For resistant counts, `log2(predicted / observed)` with a "
            "0.5 pseudocount is used as the main relative count-scale bias measure, so positive values "
            "mean multiplicative overprediction and negative values mean multiplicative underprediction."
        ),
        "",
        (
            f"Per-replicate resistant-fraction bias before escape: median={fmt(pre_frac['median'])}, "
            f"mean={fmt(pre_frac['mean'])}, Wilcoxon p={fmt(pre_frac['wilcoxon_p'])}."
        ),
        (
            f"Per-replicate resistant-fraction bias after escape: median={fmt(post_frac['median'])}, "
            f"mean={fmt(post_frac['mean'])}, Wilcoxon p={fmt(post_frac['wilcoxon_p'])}."
        ),
        (
            f"Paired post-minus-pre shift in resistant-fraction error: n={int(paired_frac['n'])}, "
            f"median={fmt(paired_frac['median'])}, mean={fmt(paired_frac['mean'])}, "
            f"Wilcoxon p={fmt(paired_frac['wilcoxon_p'])}."
        ),
        (
            f"Paired post-minus-pre shift in resistant-count log2 ratio: n={int(paired_log['n'])}, "
            f"median={fmt(paired_log['median'])}, mean={fmt(paired_log['mean'])}, "
            f"Wilcoxon p={fmt(paired_log['wilcoxon_p'])}."
        ),
        "",
        "Scientific readout:",
        "",
        (
            "For resistant fraction, the fitted LV environment does not show a before-overestimate/after-underestimate "
            "pattern. The dominant pattern is positive fraction bias before escape and a further positive "
            "post-escape shift in the paired escaped replicates."
        ),
        (
            "For resistant absolute cell counts, the inference is less clean. Across all pre-escape points "
            "the model often overpredicts resistant counts, but across all post-escape points the median "
            "count log2 ratio is close to zero and slightly negative. In the paired escaped-replicate subset, "
            "the post-minus-pre count-ratio shift is positive. This is best read as a paired-subset tendency, "
            "not as a global statement that counts are always more overpredicted after escape."
        ),
        "",
        "Strategy-level notes:",
        "",
        *(strategy_notes if strategy_notes else ["- No strategies had both pre- and post-escape observations."]),
        "",
        "Interpretation guide:",
        "",
        "- Positive resistant-fraction error: LV predicts too large a resistant fraction.",
        "- Negative resistant-fraction error: LV predicts too small a resistant fraction.",
        "- Positive resistant-count log2(pred/obs): LV predicts too many resistant cells multiplicatively.",
        "- Negative resistant-count log2(pred/obs): LV predicts too few resistant cells multiplicatively.",
        "- Positive post-minus-pre shift: LV becomes more overpredictive, or less underpredictive, after escape.",
        "- Negative post-minus-pre shift: LV becomes more underpredictive, or less overpredictive, after escape.",
        "",
        "## Visual Checks",
        "",
        "![Predicted vs observed](plots/predicted_vs_observed.png)",
        "",
        "![Pre/post bias](plots/pre_post_bias_by_replicate.png)",
        "",
        "![Errors relative to escape](plots/errors_relative_to_escape.png)",
        "",
        "![Mean strategy time courses](plots/strategy_mean_resistant_fraction_time_courses.png)",
        "",
        "![Escaped replicate examples](plots/example_escaped_replicate_trajectories.png)",
        "",
        "![Per-replicate heatmap](plots/per_replicate_error_heatmap.png)",
        "",
        "## Output Tables",
        "",
        "- `tables/prediction_long_table.csv`: one row per sample/day with observations, predictions, errors, escape phase, and treatment action.",
        "- `tables/point_error_summary_by_strategy_phase.csv`: point-level summaries by strategy and escape phase.",
        "- `tables/replicate_error_summary.csv`: per-replicate summaries and pre/post shifts.",
        "- `tables/statistical_tests.csv`: paired and one-sample bias tests.",
        "",
        "## Aggregate Summary",
        "",
        markdown_table(point_summary),
        "",
        "## Statistical Tests",
        "",
        markdown_table(tests),
        "",
        "## Caveats",
        "",
        (
            "The LV state is relative to the replicate-specific initial total. Count-scale predictions "
            "therefore assume the model's normalized burden can be back-transformed by multiplying with "
            "the observed initial total. This is consistent with the fitting code, but count residuals "
            "should be interpreted as model-scale extrapolations rather than independent absolute cell-count calibration."
        ),
        (
            "The resistant-fraction analysis is more robust to total-count scale calibration, but it does "
            "not test whether the model gets the total population size right. Fraction and absolute-count "
            "claims should therefore be kept separate."
        ),
        (
            "Samples whose escape happened only after an `experiment_filters` truncation cutoff are treated "
            "as having no analyzable post-escape period, because the post-cutoff observations are excluded "
            "from this analysis."
        ),
        "",
    ]
    (OUT_DIR / "LV_escape_prediction_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    exclude_samples = load_exclude_samples()
    params = load_lv_params()
    schedules = load_treatment_schedules()
    counts = load_count_data(exclude_samples)
    escape = load_escape_table(exclude_samples)
    pred = build_prediction_table(counts, schedules, escape, params)
    eval_pred = pred[pred["obs_index"] > 1].copy()

    point_summary = summarize_points(eval_pred)
    rep_summary = summarize_replicates(eval_pred, full_pred=pred)
    tests = build_stat_tests(rep_summary)

    save_table(pred, "prediction_long_table.csv")
    save_table(point_summary, "point_error_summary_by_strategy_phase.csv")
    save_table(rep_summary, "replicate_error_summary.csv")
    save_table(tests, "statistical_tests.csv")
    save_table(escape, "escape_table_after_filter_logic.csv")

    plot_predicted_vs_observed(eval_pred)
    plot_pre_post_bias(rep_summary)
    plot_relative_to_escape(eval_pred)
    plot_strategy_time_courses(pred)
    plot_example_trajectories(pred)
    plot_error_heatmap(rep_summary)

    write_markdown_report(pred, eval_pred, point_summary, rep_summary, tests, params)
    print(f"Wrote analysis outputs to {OUT_DIR}")
    print(f"Samples: {pred['sample'].nunique()}, observations: {len(pred)}, error-summary observations: {len(eval_pred)}")
    print(f"Escaped within filtered window: {rep_summary['escape_day'].notna().sum()}")


if __name__ == "__main__":
    main()
