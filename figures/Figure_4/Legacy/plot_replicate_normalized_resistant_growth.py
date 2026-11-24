"""Fit an experimental CT growth ceiling and normalize resistant-cell growth.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream/related script reference(s):
# - import matplotlib.py
# - plot_replicate_resistant_invitro_dpd_lv.py
# =============================================================================

The reference curve is the cross-replicate median of the experimental CT
resistant-cell trajectories in the same 27-replicate DPD intersection used by
``plot_replicate_resistant_invitro_dpd_lv.py``.  Only the longest observation
window in which every matched CT replicate contributes is fitted.  The ODE

    dN/dt = r_max * N / sqrt(1 + N/N_c)

is integrated numerically and fitted directly to N(t) by nonlinear least
squares.  Every positive consecutive-count interval is then assigned

    g = ln(N_next/N_previous) / delta_days
    G = g / r_max.

Negative G and G greater than one are retained.  Intervals touching a zero LV
population are undefined rather than silently imputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SCRIPT_DIR = THIS_FILE.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.figure_4 import plot_replicate_resistant_invitro_dpd_lv as comparison
from shared.strategy_colors import get_color

DEFAULT_OUTPUT = comparison.DEFAULT_OUTPUT
SOURCE_ORDER = ("In vitro", "DPD", "LV")
STRATEGY_ORDER = ("CT", "AT50", "AT100")
SOURCE_COLORS = {"In vitro": "#222222", "DPD": "#4C78A8", "LV": "#E45756"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=comparison.EXPERIMENT_ROOT)
    parser.add_argument("--dpd-counts", type=Path, default=comparison.DPD_COUNTS)
    parser.add_argument("--escape-summary", type=Path, default=comparison.ESCAPE_SUMMARY)
    parser.add_argument("--schedule-root", type=Path, default=comparison.SCHEDULE_ROOT)
    parser.add_argument("--params", type=Path, default=comparison.PARAMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_comparison_data(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    experimental, _audit = comparison.apply_filters(
        comparison.read_experiments(args.experiment_root)
    )
    return comparison.build_comparisons(
        experimental,
        comparison.read_dpd(args.dpd_counts),
        comparison.read_schedules(args.schedule_root),
        json.loads(args.params.read_text(encoding="utf-8")),
        comparison.read_escape_times(args.escape_summary),
    )


def integrate_growth_ode(
    elapsed_days: np.ndarray,
    r_max: float,
    n_c: float,
    n0: float,
) -> np.ndarray:
    times = np.asarray(elapsed_days, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("elapsed_days must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError("elapsed_days must be finite, nonnegative, and sorted")
    if r_max <= 0 or n_c <= 0 or n0 <= 0:
        raise ValueError("r_max, N_c, and N0 must be positive")
    if times[-1] == 0:
        return np.full(times.shape, float(n0), dtype=float)

    solution = solve_ivp(
        lambda _time, population: (
            r_max * population / np.sqrt(1.0 + population / n_c)
        ),
        (0.0, float(times[-1])),
        [float(n0)],
        t_eval=times,
        rtol=1.0e-9,
        atol=1.0e-10,
        max_step=0.1,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return np.asarray(solution.y[0], dtype=float)


def prepare_ct_reference_input(long_data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ct = long_data[
        long_data["strategy"].eq("CT") & long_data["source"].eq("In vitro")
    ].copy()
    if ct.empty:
        raise ValueError("No matched experimental CT trajectories found")
    ct["replicate_key"] = ct["plate"].astype(str) + "/" + ct["sample"].astype(str)
    ct["elapsed_day"] = ct["day"] - ct.groupby("replicate_key")["day"].transform("min")
    n_replicates = int(ct["replicate_key"].nunique())

    reference = (
        ct.groupby("elapsed_day", as_index=False)
        .agg(
            median_resistant_count=("resistant_count", "median"),
            mean_resistant_count=("resistant_count", "mean"),
            q25_resistant_count=("resistant_count", lambda values: values.quantile(0.25)),
            q75_resistant_count=("resistant_count", lambda values: values.quantile(0.75)),
            n_replicates=("replicate_key", "nunique"),
        )
        .sort_values("elapsed_day")
        .reset_index(drop=True)
    )
    reference["used_for_fit"] = reference["n_replicates"].eq(n_replicates)
    used_days = reference.loc[reference["used_for_fit"], "elapsed_day"]
    if used_days.empty or float(used_days.iloc[0]) != 0:
        raise ValueError("The complete CT cohort does not contain the baseline observation")
    fit_end = float(used_days.max())
    if reference.loc[reference["elapsed_day"] <= fit_end, "used_for_fit"].eq(False).any():
        raise ValueError("Complete-cohort CT observations are not a contiguous initial window")
    reference["used_for_fit"] &= reference["elapsed_day"] <= fit_end
    return reference, n_replicates


def fit_ct_reference(
    reference: pd.DataFrame,
    n_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_input = reference[reference["used_for_fit"]].copy()
    times = fit_input["elapsed_day"].to_numpy(dtype=float)
    observed = fit_input["median_resistant_count"].to_numpy(dtype=float)
    lower = np.log([1.0e-5, 1.0e-3, 1.0e-3])
    upper = np.log([5.0, 1.0e10, 1.0e7])
    initial = np.log([0.25, 3000.0, max(float(observed[0]), 1.0)])

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        r_max, n_c, n0 = np.exp(log_parameters)
        return integrate_growth_ode(times, r_max, n_c, n0) - observed

    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=2000,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    if not fit.success:
        raise RuntimeError(f"CT nonlinear least-squares fit failed: {fit.message}")
    r_max, n_c, n0 = np.exp(fit.x)
    fitted_at_observations = integrate_growth_ode(times, r_max, n_c, n0)
    residuals = observed - fitted_at_observations
    sum_squares = float(np.sum(residuals ** 2))
    total_squares = float(np.sum((observed - observed.mean()) ** 2))
    r_squared = 1.0 - sum_squares / total_squares if total_squares > 0 else np.nan
    hit_bound = bool(
        np.any(np.isclose(fit.x, lower, atol=1.0e-4))
        or np.any(np.isclose(fit.x, upper, atol=1.0e-4))
    )

    parameters = pd.DataFrame([
        {
            "reference_source": "In vitro",
            "reference_strategy": "CT",
            "population": "resistant",
            "aggregation": "cross-replicate median",
            "cohort": "same matched replicates as the 27 DPD comparison plots",
            "fit_residual_scale": "absolute resistant cell counts",
            "n_ct_replicates": n_replicates,
            "n_fit_timepoints": len(times),
            "fit_start_elapsed_day": float(times.min()),
            "fit_end_elapsed_day": float(times.max()),
            "r_max_per_day": float(r_max),
            "N_c_resistant_cells": float(n_c),
            "N0_resistant_cells": float(n0),
            "rmse_resistant_cells": float(np.sqrt(np.mean(residuals ** 2))),
            "r_squared": float(r_squared),
            "optimizer_cost": float(fit.cost),
            "optimizer_nfev": int(fit.nfev),
            "optimizer_hit_parameter_bound": hit_bound,
        }
    ])
    curve_times = np.linspace(float(times.min()), float(times.max()), 321)
    curve = pd.DataFrame(
        {
            "elapsed_day": curve_times,
            "fitted_resistant_count": integrate_growth_ode(
                curve_times, float(r_max), float(n_c), float(n0)
            ),
        }
    )
    return parameters, curve


def calculate_interval_growth(
    long_data: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    r_max = float(parameters.iloc[0]["r_max_per_day"])
    n_c = float(parameters.iloc[0]["N_c_resistant_cells"])
    rows: list[dict[str, object]] = []
    group_columns = ["plate", "sample", "strategy", "dpd_run", "source"]
    for keys, group in long_data.groupby(group_columns, sort=True):
        group = group.sort_values("day").reset_index(drop=True)
        baseline_day = float(group.iloc[0]["day"])
        for index in range(len(group) - 1):
            start = group.iloc[index]
            end = group.iloc[index + 1]
            day_start = float(start["day"])
            day_end = float(end["day"])
            delta_days = day_end - day_start
            n_start = float(start["resistant_count"])
            n_end = float(end["resistant_count"])
            valid = bool(
                np.isfinite(delta_days)
                and delta_days > 0
                and np.isfinite(n_start)
                and np.isfinite(n_end)
                and n_start > 0
                and n_end > 0
            )
            if valid:
                growth_rate = float(np.log(n_end / n_start) / delta_days)
                normalized_growth = growth_rate / r_max
                ct_rate_at_start = r_max / np.sqrt(1.0 + n_start / n_c)
                growth_relative_to_ct_curve = growth_rate / ct_rate_at_start
                invalid_reason = ""
            else:
                growth_rate = np.nan
                normalized_growth = np.nan
                ct_rate_at_start = np.nan
                growth_relative_to_ct_curve = np.nan
                if not np.isfinite(delta_days) or delta_days <= 0:
                    invalid_reason = "nonpositive_or_nonfinite_time_interval"
                elif not np.isfinite(n_start) or not np.isfinite(n_end):
                    invalid_reason = "nonfinite_resistant_count"
                else:
                    invalid_reason = "nonpositive_resistant_count"
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "interval_index": index + 1,
                    "day_start": day_start,
                    "day_end": day_end,
                    "day_midpoint": 0.5 * (day_start + day_end),
                    "elapsed_day_start": day_start - baseline_day,
                    "elapsed_day_end": day_end - baseline_day,
                    "elapsed_day_midpoint": 0.5 * (day_start + day_end) - baseline_day,
                    "delta_days": delta_days,
                    "start_resistant_count": n_start,
                    "end_resistant_count": n_end,
                    "geometric_midpoint_resistant_count": (
                        float(np.sqrt(n_start * n_end)) if valid else np.nan
                    ),
                    "per_cell_growth_rate_per_day": growth_rate,
                    "r_max_reference_per_day": r_max,
                    "normalized_growth": normalized_growth,
                    "ct_model_per_cell_rate_at_start_per_day": ct_rate_at_start,
                    "growth_relative_to_ct_model_at_start": growth_relative_to_ct_curve,
                    "valid_normalized_growth": valid,
                    "invalid_reason": invalid_reason,
                }
            )
    return pd.DataFrame(rows)


def shared_growth_limits(growth: pd.DataFrame) -> tuple[float, float]:
    values = growth.loc[growth["valid_normalized_growth"], "normalized_growth"]
    values = values[np.isfinite(values)]
    if values.empty:
        raise ValueError("No finite normalized-growth intervals found")
    lower = min(float(values.min()), 0.0)
    upper = max(float(values.max()), 1.0)
    span = max(upper - lower, 1.0)
    lower = np.floor((lower - 0.06 * span) * 2.0) / 2.0
    upper = np.ceil((upper + 0.06 * span) * 2.0) / 2.0
    return float(lower), float(upper)


def plot_replicate_growth(
    comparisons: list[dict[str, object]],
    growth: pd.DataFrame,
    output: Path,
) -> None:
    y_limits = shared_growth_limits(growth)
    x_max = float(max(int(item["last_day"]) for item in comparisons))
    for item in comparisons:
        subset = growth[
            growth["plate"].eq(item["plate"])
            & growth["sample"].eq(item["sample"])
        ]
        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        for source in SOURCE_ORDER:
            values = subset[subset["source"].eq(source)].sort_values("day_midpoint")
            ax.plot(
                values["day_midpoint"],
                values["normalized_growth"],
                color=SOURCE_COLORS[source],
                marker="o" if source == "In vitro" else None,
                ms=2.2,
                lw=1.2,
                label=source,
            )
        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
        ax.axhline(
            1.0,
            color="#555555",
            linewidth=1.0,
            linestyle="--",
            label="G = 1 (CT r_max)",
        )
        if item["escape_marker_shown"]:
            ax.axvline(
                float(item["escape_day"]),
                color="#777777",
                linestyle="--",
                linewidth=1.0,
                label="In vitro escape",
            )
        ax.set(
            xlim=(0.0, x_max),
            ylim=y_limits,
            xlabel="Time [days]",
            ylabel="Normalized resistant growth, G",
        )
        ax.set_title(
            f'{item["plate"]} - {item["sample"]} - DPD R{item["run"]}',
            fontsize=9,
        )
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        stem = f'{item["plate"]}_{item["sample"]}_R{item["run"]}'
        fig.savefig(output / f"{stem}_normalized_growth.pdf")
        plt.close(fig)


def plot_ct_fit(
    long_data: pd.DataFrame,
    reference: pd.DataFrame,
    curve: pd.DataFrame,
    parameters: pd.DataFrame,
    output_pdf: Path,
) -> None:
    ct = long_data[
        long_data["strategy"].eq("CT") & long_data["source"].eq("In vitro")
    ].copy()
    ct["replicate_key"] = ct["plate"].astype(str) + "/" + ct["sample"].astype(str)
    ct["elapsed_day"] = ct["day"] - ct.groupby("replicate_key")["day"].transform("min")
    fit_input = reference[reference["used_for_fit"]]
    fit_end = float(fit_input["elapsed_day"].max())
    r_max = float(parameters.iloc[0]["r_max_per_day"])
    n_c = float(parameters.iloc[0]["N_c_resistant_cells"])
    n0 = float(parameters.iloc[0]["N0_resistant_cells"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax in axes:
        for _key, group in ct.groupby("replicate_key"):
            shown = group[group["elapsed_day"] <= fit_end].sort_values("elapsed_day")
            ax.plot(
                shown["elapsed_day"],
                shown["resistant_count"],
                color="#BBBBBB",
                linewidth=0.7,
                alpha=0.65,
            )
        ax.plot(
            fit_input["elapsed_day"],
            fit_input["median_resistant_count"],
            color="#222222",
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            label="Experimental CT median",
        )
        ax.plot(
            curve["elapsed_day"],
            curve["fitted_resistant_count"],
            color="#E45756",
            linewidth=1.5,
            label="Square-root-limited fit",
        )
        ax.set_xlabel("Elapsed time [days]")
        ax.set_xlim(0.0, fit_end)
    axes[0].set_ylabel("Resistant cell count")
    axes[0].set_title("Linear scale", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_title("Log scale", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"Experimental CT reference: r_max={r_max:.4f}/day, "
        f"N_c={n_c:.1f}, N0={n0:.1f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_growth_summary(growth: pd.DataFrame, output_pdf: Path) -> None:
    y_limits = shared_growth_limits(growth)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.2), sharey=True)
    for row, source in enumerate(SOURCE_ORDER):
        source_data = growth[
            growth["source"].eq(source) & growth["valid_normalized_growth"]
        ]
        for strategy in STRATEGY_ORDER:
            values = source_data[source_data["strategy"].eq(strategy)]
            color = get_color(strategy, "#888888")
            axes[row, 0].scatter(
                values["day_midpoint"], values["normalized_growth"],
                s=7, alpha=0.55, color=color, edgecolors="none", label=strategy,
            )
            axes[row, 1].scatter(
                values["start_resistant_count"], values["normalized_growth"],
                s=7, alpha=0.55, color=color, edgecolors="none", label=strategy,
            )
        for column in range(2):
            axes[row, column].axhline(0.0, color="#888888", linewidth=0.7, linestyle=":")
            axes[row, column].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
            axes[row, column].set_ylim(*y_limits)
        axes[row, 0].set_ylabel(f"{source}\nNormalized growth, G")
        axes[row, 1].set_xscale("log")
    axes[0, 0].set_title("G versus interval midpoint time", fontsize=9)
    axes[0, 1].set_title("G versus starting resistant count", fontsize=9)
    axes[-1, 0].set_xlabel("Time [days]")
    axes[-1, 1].set_xlabel("Starting resistant cell count")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_pdf)
    plt.close(fig)


def summarize_growth(growth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, strategy), group in growth.groupby(["source", "strategy"], sort=False):
        valid = group.loc[group["valid_normalized_growth"], "normalized_growth"]
        rows.append(
            {
                "source": source,
                "strategy": strategy,
                "n_intervals_total": len(group),
                "n_intervals_valid": len(valid),
                "n_intervals_invalid": int((~group["valid_normalized_growth"]).sum()),
                "normalized_growth_median": float(valid.median()),
                "normalized_growth_q25": float(valid.quantile(0.25)),
                "normalized_growth_q75": float(valid.quantile(0.75)),
                "normalized_growth_min": float(valid.min()),
                "normalized_growth_max": float(valid.max()),
            }
        )
    return pd.DataFrame(rows)


def write_audit(
    path: Path,
    parameters: pd.DataFrame,
    growth: pd.DataFrame,
) -> None:
    row = parameters.iloc[0]
    invalid = growth.loc[~growth["valid_normalized_growth"]]
    lines = [
        "Normalized resistant growth audit",
        "",
        "Reference population: resistant cells",
        "Reference source and treatment: experimental in-vitro CT",
        f"Matched CT replicates: {int(row['n_ct_replicates'])}",
        "Aggregation: median resistant count at each experimental observation",
        "Fit window rule: longest initial window in which every matched CT replicate contributes",
        f"Fit elapsed-day window: {row['fit_start_elapsed_day']:g} to {row['fit_end_elapsed_day']:g}",
        f"Fit timepoints: {int(row['n_fit_timepoints'])}",
        "Fit method: numerical solve_ivp integration and nonlinear least squares on N(t)",
        "ODE: dN/dt = r_max * N / sqrt(1 + N/N_c)",
        f"r_max [1/day]: {row['r_max_per_day']:.12g}",
        f"N_c [resistant cells]: {row['N_c_resistant_cells']:.12g}",
        f"N0 [resistant cells]: {row['N0_resistant_cells']:.12g}",
        f"R-squared on fitted median counts: {row['r_squared']:.12g}",
        "Interval rate: g = ln(N_next/N_previous) / delta_days",
        "Requested normalization: G = g / r_max",
        "Time uses numeric calendar-day differences; experimental and LV gaps are 2 or 3 days",
        "LV treatment: CT always on; AT50 50%/100% hysteresis; AT100 on at total >= initial total",
        "LV treatment is simulated independently of the in-vitro schedule",
        "G is not clipped: negative values and values greater than one are retained",
        f"Undefined intervals: {len(invalid)} (nonpositive/nonfinite counts or invalid time gaps)",
        "The optional growth_relative_to_ct_model_at_start column additionally accounts for N_c",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons, long_data = load_comparison_data(args)
    reference, n_replicates = prepare_ct_reference_input(long_data)
    parameters, curve = fit_ct_reference(reference, n_replicates)
    growth = calculate_interval_growth(long_data, parameters)

    parameters.to_csv(
        args.output_dir / "ct_sqrt_limited_reference_fit_parameters.csv", index=False
    )
    reference.to_csv(
        args.output_dir / "ct_sqrt_limited_reference_fit_input.csv", index=False
    )
    curve.to_csv(
        args.output_dir / "ct_sqrt_limited_reference_fit_curve.csv", index=False
    )
    growth.to_csv(
        args.output_dir / "replicate_normalized_resistant_growth_values.csv", index=False
    )
    summarize_growth(growth).to_csv(
        args.output_dir / "replicate_normalized_resistant_growth_summary.csv", index=False
    )
    write_audit(
        args.output_dir / "normalized_resistant_growth_audit.txt", parameters, growth
    )
    plot_replicate_growth(comparisons, growth, args.output_dir)
    plot_ct_fit(
        long_data,
        reference,
        curve,
        parameters,
        args.output_dir / "ct_sqrt_limited_reference_fit.pdf",
    )
    plot_growth_summary(
        growth,
        args.output_dir / "normalized_resistant_growth_summary.pdf",
    )

    row = parameters.iloc[0]
    print(
        f"Fitted experimental CT reference: r_max={row['r_max_per_day']:.8g}/day, "
        f"N_c={row['N_c_resistant_cells']:.8g}, N0={row['N0_resistant_cells']:.8g}, "
        f"R2={row['r_squared']:.8g}"
    )
    print(
        f"Wrote {len(comparisons)} normalized-growth replicate PDFs and diagnostics "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
