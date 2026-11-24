"""Run CT, AT50, AT100, and AT100+ LV counterfactuals from filtered in-vitro baselines."""
# The model is the frozen LV system used to train the paper agent. Actions are
# chosen only at experimental observation times (raw days 1, 3, 5, 8, ...),
# and each action is held for the complete following 2- or 3-day interval.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - D:\2025_08_CoreSeed_Large_Testdataset
# - Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_4_Data
# Upstream/related script reference(s):
# - figures/Figure_1/TTP CT & AT100.py
# - analysis/plotting/plot_replicate_resistant_invitro_dpd_lv.py
# =============================================================================
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
TTP_SCRIPT = REPO_ROOT / "figures" / "Figure_1" / "TTP CT & AT100.py"
TRAJECTORY_SCRIPT = (
    REPO_ROOT / "figures" / "Figure_4"
    / "plot_replicate_resistant_invitro_dpd_lv.py"
)
PARAMS_PATH = (
    REPO_ROOT / "PhysiLearning" / "LV_Agent_Training_MiniRepo"
    / "reference" / "fitted_lv_params.json"
)
BASELINE_ROOT = Path(r"D:\2025_08_CoreSeed_Large_Testdataset")
FITTING_DATA_ROOT = (
    REPO_ROOT / "PhysiLearning" / "LV_Fitting_MiniRepo" / "data"
    / "experimental_data_elias" / "2025_stem_paper_final_final"
)
DEFAULT_OUTPUT = Path(
    r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_4_Data"
)
PROTOCOLS = ("CT", "AT50", "AT100", "AT100+")
END_DAY = 96
SOURCE_ORDER = ("In vitro", "DPD", "LV")
SOURCE_OFFSETS = {"In vitro": -0.24, "DPD": 0.0, "LV": 0.24}
SOURCE_MARKERS = {"In vitro": "o", "DPD": "s", "LV": "^"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def filtered_baselines(tte_module, baseline_root: Path) -> pd.DataFrame:
    rows, seen = [], set()
    for plate in tte_module.PLATES:
        h5_dir = baseline_root / plate / "hdf5"
        if not h5_dir.exists():
            continue
        for path in sorted(h5_dir.glob("G418_a2a4_*_total.h5")):
            sample = re.sub(r"_total$", "", re.sub(r"^G418_a2a4_", "", path.stem))
            if sample in seen:
                continue
            strategy = tte_module.parse_strategy(sample)
            if strategy not in tte_module.STRATEGIES_KEEP:
                continue
            if tte_module.EXCLUDE_SAMPLES.get(sample, "keep") is None:
                continue
            with h5py.File(path, "r") as h5:
                if "sus" not in h5 or "res" not in h5:
                    continue
                counts = {}
                for population in ("sus", "res"):
                    keys = sorted(h5[population].keys(), key=tte_module.numeric_sort_key)
                    if not keys:
                        counts[population] = np.nan
                        continue
                    obj = h5[population][keys[0]]
                    if isinstance(obj, h5py.Dataset):
                        counts[population] = tte_module.count_coords_dataset(obj[()])
                    elif "values" in obj:
                        counts[population] = tte_module.count_coords_dataset(obj["values"][()])
                    elif "block0_values" in obj:
                        counts[population] = tte_module.count_coords_dataset(obj["block0_values"][()])
                    else:
                        counts[population] = 0
            seen.add(sample)
            rows.append({
                "plate": plate,
                "sample": sample, "invitro_strategy": strategy,
                "sus": counts["sus"], "res": counts["res"],
                "total": counts["sus"] + counts["res"],
            })
    baseline = pd.DataFrame(rows)
    if baseline.empty:
        raise ValueError("TTP script found no eligible in-vitro baselines")
    valid = (
        np.isfinite(baseline["sus"])
        & np.isfinite(baseline["res"])
        & np.isfinite(baseline["total"])
        & (baseline["sus"] >= 0)
        & (baseline["res"] > 0)
        & (baseline["total"] > 0)
    )
    return baseline.loc[valid].reset_index(drop=True)

def choose_action(protocol: str, simulator, previous_action: bool) -> bool:
    """Choose the action for the next complete observation interval."""
    burden = simulator.s + simulator.r
    if protocol == "CT":
        return True
    if protocol == "AT50":
        # Zhang-style hysteresis: restart at baseline, then continue until
        # the observed burden reaches 50% of baseline.
        if burden >= 1.0:
            return True
        return bool(previous_action and burden > 0.5)
    if protocol == "AT100":
        return bool(burden >= 1.0)
    if protocol == "AT100+":
        # Threshold-triggered treatment for one observation interval only.
        return bool(burden >= 1.0 and not previous_action)
    raise ValueError(protocol)


def simulate_counterfactuals(
    baselines: pd.DataFrame, params: dict[str, float], ramped_lv_class
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    trajectory_rows = []
    for base in baselines.itertuples(index=False):
        initial_total = float(base.total)
        for protocol in PROTOCOLS:
            sim = ramped_lv_class(
                params, float(base.sus), float(base.res)
            )
            ttp = 0.0 if sim.r >= 1.0 else float(END_DAY)
            progressed = bool(sim.r >= 1.0)
            previous_action = False
            observation_index = 1
            treated_intervals = 0
            treated_calendar_days = 0

            while True:
                elapsed_day = sim.day - 1
                if not progressed and sim.r >= 1.0:
                    ttp = float(elapsed_day)
                    progressed = True

                next_calendar_day = sim.next_observation_day(sim.day)
                next_elapsed_day = next_calendar_day - 1
                has_next_interval = next_elapsed_day <= END_DAY
                action = (
                    choose_action(protocol, sim, previous_action)
                    if has_next_interval
                    else np.nan
                )
                resistant_count = sim.r * initial_total
                trajectory_rows.append(
                    {
                        "plate": base.plate,
                        "sample": base.sample,
                        "invitro_strategy": base.invitro_strategy,
                        "protocol": protocol,
                        "observation_index": observation_index,
                        "calendar_day": sim.day,
                        "day": elapsed_day,
                        "sensitive_count": sim.s * initial_total,
                        "resistant_count": resistant_count,
                        "total_count": (sim.s + sim.r) * initial_total,
                        "sensitive_normalized_to_initial_total": sim.s,
                        "resistant_normalized_to_initial_total": sim.r,
                        "total_normalized_to_initial_total": sim.s + sim.r,
                        "action_for_next_interval": action,
                        "next_calendar_day": (
                            next_calendar_day if has_next_interval else np.nan
                        ),
                    }
                )

                if not has_next_interval:
                    break
                interval_days = next_calendar_day - sim.day
                sim.advance_interval(int(action), next_calendar_day)
                previous_action = bool(action)
                treated_intervals += int(action)
                treated_calendar_days += int(action) * interval_days
                observation_index += 1

            result_rows.append(
                {
                    "plate": base.plate,
                    "sample": base.sample,
                    "invitro_strategy": base.invitro_strategy,
                    "protocol": protocol,
                    "initial_sensitive": float(base.sus),
                    "initial_resistant": float(base.res),
                    "initial_total": initial_total,
                    "ttp": ttp,
                    "progressed": progressed,
                    "treated_intervals": treated_intervals,
                    "treated_calendar_days": treated_calendar_days,
                    "last_observation_day": sim.day - 1,
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(trajectory_rows)


def compare_baseline_sources(
    baselines: pd.DataFrame,
    fitting_data_root: Path,
) -> pd.DataFrame:
    """Audit current HDF5 inputs against the frozen LV-fitting count copy."""
    rows = []
    for base in baselines.itertuples(index=False):
        fitting_path = (
            fitting_data_root
            / str(base.plate).replace("_", " ")
            / "Results"
            / f"Results_G418_a2a4_{base.sample}.txt"
        )
        fitting_sensitive = np.nan
        fitting_resistant = np.nan
        status = "missing_fitting_file"
        if fitting_path.exists():
            table = pd.read_csv(
                fitting_path,
                sep=r"\s+",
                header=None,
                names=["day_token", "sensitive", "resistant"],
            )
            if not table.empty:
                fitting_sensitive = float(table.iloc[0]["sensitive"])
                fitting_resistant = float(table.iloc[0]["resistant"])
                status = "ok"
        fitting_total = fitting_sensitive + fitting_resistant
        rows.append(
            {
                "plate": base.plate,
                "sample": base.sample,
                "invitro_strategy": base.invitro_strategy,
                "current_hdf5_sensitive": float(base.sus),
                "current_hdf5_resistant": float(base.res),
                "current_hdf5_total": float(base.total),
                "frozen_fitting_sensitive": fitting_sensitive,
                "frozen_fitting_resistant": fitting_resistant,
                "frozen_fitting_total": fitting_total,
                "same_sensitive": bool(float(base.sus) == fitting_sensitive),
                "same_resistant": bool(float(base.res) == fitting_resistant),
                "current_to_fitting_resistant_ratio": (
                    float(base.res) / fitting_resistant
                    if np.isfinite(fitting_resistant) and fitting_resistant > 0
                    else np.nan
                ),
                "current_resistant_fraction": float(base.res) / float(base.total),
                "fitting_resistant_fraction": (
                    fitting_resistant / fitting_total
                    if np.isfinite(fitting_total) and fitting_total > 0
                    else np.nan
                ),
                "status": status,
                "fitting_source_path": str(fitting_path),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=PARAMS_PATH)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument(
        "--fitting-data-root", type=Path, default=FITTING_DATA_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def summarize_protocols(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in PROTOCOLS:
        group = results.loc[results["protocol"] == protocol]
        rows.append(
            {
                "protocol": protocol,
                "n": len(group),
                "n_progressed": int(group["progressed"].sum()),
                "n_censored_at_day_96": int((~group["progressed"]).sum()),
                "ttp_median": float(group["ttp"].median()),
                "ttp_q25": float(group["ttp"].quantile(0.25)),
                "ttp_q75": float(group["ttp"].quantile(0.75)),
                "ttp_min": float(group["ttp"].min()),
                "ttp_max": float(group["ttp"].max()),
                "median_treated_intervals": float(
                    group["treated_intervals"].median()
                ),
                "median_treated_calendar_days": float(
                    group["treated_calendar_days"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_original_ttp_sources(ttp_module) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the exact in-vitro and DPD loading path from the Figure 1 script."""
    ttp_module.STRATEGIES_KEEP = ["CT", "AT50", "AT100"]
    raw = ttp_module.collect_all_samples()
    if raw.empty:
        raise ValueError("TTP CT & AT100.py found no in-vitro trajectories")
    normalized = ttp_module.normalize_to_day1_total(raw)
    filtered = ttp_module.apply_experiment_filters(
        normalized, use_truncation_days=True
    )
    invitro = ttp_module.compute_ttp(
        filtered, use_truncation_days=True
    ).copy()
    invitro["source"] = "In vitro"
    invitro["source_id"] = invitro["sample"].astype(str)

    dpd = ttp_module.load_simulation_ttp().copy().reset_index(drop=True)
    dpd["source"] = "DPD"
    dpd["source_id"] = [f"DPD_{index + 1:03d}" for index in range(len(dpd))]
    dpd["progressed"] = dpd["ttp"].notna()
    return invitro, dpd


def build_ttp_comparison_values(
    invitro: pd.DataFrame,
    dpd: pd.DataFrame,
    lv_results: pd.DataFrame,
) -> pd.DataFrame:
    invitro_values = invitro.rename(columns={"sample": "replicate_id"})[
        ["source", "replicate_id", "strategy", "ttp", "progressed"]
    ].copy()
    dpd_values = dpd.rename(columns={"source_id": "replicate_id"})[
        ["source", "replicate_id", "strategy", "ttp", "progressed"]
    ].copy()
    lv_values = lv_results.rename(
        columns={"sample": "replicate_id", "protocol": "strategy"}
    )[["replicate_id", "strategy", "ttp", "progressed"]].copy()
    lv_values.insert(0, "source", "LV")
    values = pd.concat(
        [invitro_values, dpd_values, lv_values], ignore_index=True
    )
    values["included_in_plot"] = values["ttp"].notna()
    return values


def summarize_ttp_sources(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SOURCE_ORDER:
        for protocol in PROTOCOLS:
            group = values[
                values["source"].eq(source)
                & values["strategy"].eq(protocol)
            ]
            plotted = group.loc[group["included_in_plot"], "ttp"]
            rows.append(
                {
                    "source": source,
                    "protocol": protocol,
                    "n_rows": len(group),
                    "n_plotted": len(plotted),
                    "n_missing_or_truncated": int((~group["included_in_plot"]).sum()),
                    "ttp_median": float(plotted.median()) if len(plotted) else np.nan,
                    "ttp_q25": float(plotted.quantile(0.25)) if len(plotted) else np.nan,
                    "ttp_q75": float(plotted.quantile(0.75)) if len(plotted) else np.nan,
                    "ttp_min": float(plotted.min()) if len(plotted) else np.nan,
                    "ttp_max": float(plotted.max()) if len(plotted) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def invitro_pairwise_statistics(
    values: pd.DataFrame,
    protocols: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    invitro = values[
        values["source"].eq("In vitro") & values["included_in_plot"]
    ]
    for first, second in combinations(protocols, 2):
        first_values = invitro.loc[invitro["strategy"].eq(first), "ttp"]
        second_values = invitro.loc[invitro["strategy"].eq(second), "ttp"]
        if len(first_values) >= 2 and len(second_values) >= 2:
            p_value = float(
                mannwhitneyu(
                    first_values.to_numpy(),
                    second_values.to_numpy(),
                    alternative="two-sided",
                ).pvalue
            )
        else:
            p_value = np.nan
        rows.append(
            {
                "source": "In vitro",
                "protocol_1": first,
                "protocol_2": second,
                "n_1": len(first_values),
                "n_2": len(second_values),
                "p_value": p_value,
                "stars": (
                    ttp_p_to_stars(p_value) if np.isfinite(p_value) else "n/a"
                ),
            }
        )
    return pd.DataFrame(rows)


def ttp_p_to_stars(p_value: float) -> str:
    if p_value < 1.0e-4:
        return "****"
    if p_value < 1.0e-3:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_protocol_subset(
    ttp_module,
    values: pd.DataFrame,
    protocols: tuple[str, ...],
    output_pdf: Path,
) -> None:
    axis_width = 44.0 if len(protocols) == 4 else 35.0
    fig, ax = ttp_module.ft.new_figure(
        width=axis_width,
        height_mm=ttp_module.FIGURE_AXES_HEIGHT_MM,
        margins=ttp_module.FIGURE_MARGINS_MM,
        dpi=ttp_module.DPI,
        apply_style=False,
    )
    centers = np.arange(1, len(protocols) + 1, dtype=float)
    ax.set_xticks(centers)
    ax.set_xticklabels(protocols)
    for tick_label, protocol in zip(ax.get_xticklabels(), protocols):
        tick_label.set_color(ttp_module.get_color(protocol, "#888888"))
    ax.set_ylabel("Time to progression [days]")
    ax.set_ylim(*ttp_module.YLIM)
    ax.grid(True, axis="y", alpha=ttp_module.GRID_ALPHA)
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", pad=1)

    rng = np.random.default_rng(7)
    for center, protocol in zip(centers, protocols):
        counts = []
        for source in SOURCE_ORDER:
            source_values = values.loc[
                values["strategy"].eq(protocol)
                & values["source"].eq(source)
                & values["included_in_plot"],
                "ttp",
            ].to_numpy(dtype=float)
            counts.append(len(source_values))
            if not len(source_values):
                continue
            offset = SOURCE_OFFSETS[source]
            x_values = (
                center
                + offset
                + rng.uniform(
                    -ttp_module.POINT_JITTER * 0.55,
                    ttp_module.POINT_JITTER * 0.55,
                    size=len(source_values),
                )
            )
            ax.scatter(
                x_values,
                source_values,
                s=ttp_module.POINT_SIZE,
                c=ttp_module.get_color(protocol, "#888888"),
                alpha=ttp_module.POINT_ALPHA,
                edgecolors="black",
                linewidths=0.3,
                marker=SOURCE_MARKERS[source],
                zorder=3,
            )
            median = float(np.median(source_values))
            ax.hlines(
                median,
                center + offset - 0.065,
                center + offset + 0.065,
                colors=ttp_module.MEDIAN_COLOR,
                linewidth=1.0,
                zorder=4,
            )
        ax.text(
            center,
            -0.105,
            "n=" + "/".join(str(count) for count in counts),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=ttp_module.N_FONTSIZE,
            clip_on=False,
        )

    statistics = invitro_pairwise_statistics(values, protocols)
    significant = statistics[
        statistics["p_value"].notna() & statistics["p_value"].lt(0.05)
    ].reset_index(drop=True)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for index, row in significant.iterrows():
        first_index = protocols.index(str(row["protocol_1"]))
        second_index = protocols.index(str(row["protocol_2"]))
        y_bracket = (
            y_max
            - ttp_module.BRACKET_TOP_MARGIN_FRAC * y_range
            - index * ttp_module.BRACKET_STEP_FRAC * y_range
        )
        ttp_module.add_sig_bracket(
            ax,
            centers[first_index] + SOURCE_OFFSETS["In vitro"],
            centers[second_index] + SOURCE_OFFSETS["In vitro"],
            y_bracket,
            stars=str(row["stars"]),
        )

    legend_handles = [
        Line2D(
            [0], [0], marker=SOURCE_MARKERS[source], color="none",
            markerfacecolor="#777777", markeredgecolor="black",
            markeredgewidth=0.3, markersize=4.0, label=source,
        )
        for source in SOURCE_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=5,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    ax.text(
        0.01, 0.02, "n order: In vitro / DPD / LV",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=4.5,
    )
    ttp_module.ft.savefig(fig, output_pdf)
    plt.close(fig)


def write_ttp_comparison_audit(
    path: Path,
    ttp_module,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "In-vitro, DPD, and LV TTP comparison audit",
        "",
        f"In-vitro HDF5 root: {ttp_module.ROOT_DIR}",
        "In-vitro values: collect_all_samples -> normalize_to_day1_total -> apply_experiment_filters(use_truncation_days=True) -> compute_ttp",
        f"DPD source: {ttp_module.SIMULATION_SUMMARY_CSV}",
        "DPD values: load_simulation_ttp from TTP CT & AT100.py",
        "LV values: corrected counterfactual simulations in this script",
        "Plot count order: In vitro / DPD / LV",
        "Significance brackets: pairwise two-sided Mann-Whitney U tests on in-vitro values only, matching the Figure 1 convention",
        "AT100+: no in-vitro or DPD values exist; LV only",
        "",
        summary.to_csv(index=False).strip(),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit(
    path: Path,
    params_path: Path,
    baseline_root: Path,
    n_baselines: int,
    params: dict[str, float],
    baseline_source_audit: pd.DataFrame,
) -> None:
    n_resistant_differences = int(
        (~baseline_source_audit["same_resistant"]).sum()
    )
    lines = [
        "LV TTP model and protocol audit",
        "",
        f"Parameter source: {params_path.resolve()}",
        "Parameter lineage: frozen system embedded in both canonical paper-agent training configs",
        f"Baseline source: {baseline_root.resolve()}",
        f"Filtered positive-resistant baselines: {n_baselines}",
        f"Current-vs-frozen-fitting resistant baseline differences: {n_resistant_differences}/{n_baselines}",
        "Simulation input choice: current HDF5 baselines used by TTP CT & AT100.py",
        "Initialization: observed sensitive and resistant counts divided by their replicate-specific initial total",
        "Back-transformation: normalized LV populations multiplied by that same initial total",
        "Observation calendar: raw days 1, 3, 5, 8, ...; reported TTP uses elapsed days 0, 2, 4, 7, ...",
        f"Standard 2-day interval: {params['treatment_time_step']} LvEnv inner loops",
        f"Three-day weekly interval: {int(params['treatment_time_step'] * 1.5)} LvEnv inner loops",
        "Progression: first observation with resistant / initial total >= 1; otherwise censored at day 96",
        "CT: treatment in every interval",
        "AT50: treat from baseline; stop at total <= 0.5; restart at total >= 1.0",
        "AT100: treat an interval when observed total >= 1.0",
        "AT100+: treat an interval when observed total >= 1.0 and the preceding interval was off",
        "Dynamics: sequential sensitive/resistant LvEnv grow calls, including two historical ramp updates per inner loop",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ttp_module = load_module("paper_ttp", TTP_SCRIPT)
    trajectory_module = load_module("replicate_lv", TRAJECTORY_SCRIPT)
    ttp_module.SHOW_FIGURE = False
    ttp_module.FIGURE_MARGINS_MM = (9.30, 4.0, 5.74, 3.28)
    baselines = filtered_baselines(ttp_module, args.baseline_root)
    baseline_source_audit = compare_baseline_sources(
        baselines, args.fitting_data_root
    )
    params = json.loads(args.params.read_text(encoding="utf-8"))
    results, trajectories = simulate_counterfactuals(
        baselines, params, trajectory_module.RampedLv
    )
    invitro_ttp, dpd_ttp = load_original_ttp_sources(ttp_module)
    comparison_values = build_ttp_comparison_values(
        invitro_ttp, dpd_ttp, results
    )
    comparison_summary = summarize_ttp_sources(comparison_values)
    comparison_statistics = invitro_pairwise_statistics(
        comparison_values, ("CT", "AT50", "AT100")
    )

    results.to_csv(
        args.output_dir / "lv_ttp_all_invitro_initial_conditions.csv",
        index=False,
    )
    trajectories.to_csv(
        args.output_dir / "lv_ttp_all_invitro_trajectories.csv",
        index=False,
    )
    baseline_source_audit.to_csv(
        args.output_dir / "lv_ttp_baseline_input_audit.csv",
        index=False,
    )
    summary = summarize_protocols(results)
    summary.to_csv(
        args.output_dir / "lv_ttp_protocol_summary.csv",
        index=False,
    )
    comparison_values.to_csv(
        args.output_dir / "ttp_invitro_dpd_lv_values.csv",
        index=False,
    )
    comparison_summary.to_csv(
        args.output_dir / "ttp_invitro_dpd_lv_summary.csv",
        index=False,
    )
    comparison_statistics.to_csv(
        args.output_dir / "ttp_invitro_pairwise_statistics.csv",
        index=False,
    )
    write_audit(
        args.output_dir / "lv_ttp_model_and_protocol_audit.txt",
        args.params,
        args.baseline_root,
        len(baselines),
        params,
        baseline_source_audit,
    )
    write_ttp_comparison_audit(
        args.output_dir / "ttp_invitro_dpd_lv_audit.txt",
        ttp_module,
        comparison_summary,
    )

    # Replace the earlier three-protocol figure with corrected calculations,
    # and add the requested four-protocol version beside it.
    plot_protocol_subset(
        ttp_module,
        comparison_values,
        ("CT", "AT50", "AT100"),
        args.output_dir
        / "lv_ttp_CT_AT50_AT100_all_invitro_initial_conditions.pdf",
    )
    plot_protocol_subset(
        ttp_module,
        comparison_values,
        PROTOCOLS,
        args.output_dir
        / "lv_ttp_CT_AT50_AT100_AT100plus_all_invitro_initial_conditions.pdf",
    )
    print(summary.to_string(index=False))
    print(comparison_summary.to_string(index=False))
    print(
        f"Wrote {len(results)} LV protocol runs from {len(baselines)} "
        f"in-vitro initial conditions to {args.output_dir}"
    )


if __name__ == "__main__":
    main()




