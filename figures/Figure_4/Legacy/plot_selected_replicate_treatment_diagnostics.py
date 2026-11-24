"""Create treatment-aware LV, DPD, and in-vitro trajectory diagnostics.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Absolute input path(s) configured in this script:
# - Y:\Membe
# Upstream/related script reference(s):
# - import matplotlib.py
# - plot_replicate_resistant_invitro_dpd_lv.py
# =============================================================================

The three treatment histories intentionally come from different sources:
in-vitro uses the original calendar-day schedule, DPD uses its saved treatment
status table, and LV is newly simulated under the named adaptive protocol.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
TRAJECTORY_SCRIPT = THIS_FILE.with_name("plot_replicate_resistant_invitro_dpd_lv.py")
DEFAULT_OUTPUT = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Figures_Core_Seed\Figure_4_Data")
SELECTED = (("Plate_4", "C4_CT-C4", 36), ("Plate_4", "D4_AT50-D4", 17),
            ("Plate_2", "C6_AT100-C6", 28))
SENSITIVE_COLOR, RESISTANT_COLOR, TREATMENT_COLOR = "#0072B2", "#D55E00", "#B8B8B8"


def load_trajectory_module():
    spec = importlib.util.spec_from_file_location("replicate_trajectory", TRAJECTORY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {TRAJECTORY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--dpd-counts", type=Path)
    parser.add_argument("--dpd-treatment-status", type=Path)
    parser.add_argument("--params", type=Path)
    return parser.parse_args()


def read_original_invitro_schedules(experiment_root: Path) -> dict[tuple[str, str], set[int]]:
    schedules = {}
    for path in sorted(experiment_root.glob("Plate_*/treatment_schedule.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split("\t", 1)
            if not parts or not parts[0]:
                continue
            days = set()
            if len(parts) == 2:
                for token in parts[1].split():
                    match = re.match(r"(\d+)", token)
                    if match:
                        days.add(int(match.group(1)))
            schedules[(path.parent.name, parts[0])] = days
    return schedules


def daily_spans(days: set[int], last_day: float) -> list[tuple[float, float]]:
    return [(float(day), min(float(day + 1), float(last_day)))
            for day in sorted(days) if day <= last_day]


def read_dpd_treatment_status(path: Path) -> pd.DataFrame:
    status = pd.read_csv(path, sep="\t")
    required = {"plate", "well", "first_day", "is_treatment"}
    if not required.issubset(status.columns):
        raise ValueError(f"DPD treatment status lacks columns {sorted(required - set(status.columns))}")
    status["first_day"] = pd.to_numeric(status["first_day"], errors="raise")
    if status["is_treatment"].dtype != bool:
        status["is_treatment"] = status["is_treatment"].astype(str).str.lower().eq("true")
    return status


def dpd_saved_spans(status: pd.DataFrame, strategy: str, run: int, last_day: float) -> list[tuple[float, float]]:
    well = f"R{run}_{strategy}-R{run}"
    rows = status[(status["plate"] == f"Plate_{strategy}") & (status["well"] == well)].sort_values("first_day")
    if rows.empty:
        raise ValueError(f"No saved DPD treatment history for {well}")
    return [(float(day), min(float(day + 1), last_day))
            for day in rows.loc[rows["is_treatment"], "first_day"] if day <= last_day]


def choose_lv_action(strategy: str, simulator, previous_action: bool) -> bool:
    total = simulator.s + simulator.r
    if strategy == "CT":
        return True
    if strategy == "AT100":
        return total >= 1.0
    if strategy == "AT50":
        if previous_action and total <= 0.5:
            return False
        if not previous_action and total >= 1.0:
            return True
        return previous_action
    raise ValueError(f"Unsupported LV protocol: {strategy}")


def simulate_adaptive_lv(module, params: dict[str, float], sensitive: float,
                         resistant: float, strategy: str, last_day: int) -> pd.DataFrame:
    sim = module.RampedLv(params, sensitive, resistant)
    rows = [{"day": sim.day, "lv_sensitive": sim.s * sim.scale,
             "lv_resistant": sim.r * sim.scale, "treatment_on": np.nan}]
    previous_action = True
    while sim.day < last_day:
        next_day = sim.next_observation_day(sim.day)
        if next_day > last_day:
            break
        action = choose_lv_action(strategy, sim, previous_action)
        start_day = sim.day
        sim.advance_interval(int(action), next_day)
        rows[-1]["treatment_on"] = int(action)
        rows[-1]["interval_end_day"] = next_day
        rows.append({"day": sim.day, "lv_sensitive": sim.s * sim.scale,
                     "lv_resistant": sim.r * sim.scale, "treatment_on": np.nan})
        previous_action = action
    result = pd.DataFrame(rows)
    if "interval_end_day" not in result:
        result["interval_end_day"] = np.nan
    return result


def lv_spans(lv: pd.DataFrame) -> list[tuple[float, float]]:
    rows = lv[lv["treatment_on"].eq(1) & lv["interval_end_day"].notna()]
    return list(zip(rows["day"].astype(float), rows["interval_end_day"].astype(float)))


def shade(ax: plt.Axes, spans: list[tuple[float, float]]) -> None:
    for start, end in spans:
        ax.axvspan(start, end, color=TREATMENT_COLOR, alpha=0.32, linewidth=0, zorder=0)


def positive_limits(series: list[pd.Series]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float) for item in series])
    values = values[np.isfinite(values) & (values > 0)]
    return max(1.0, float(values.min()) / 1.6), float(values.max()) * 1.6


def plot_diagnostic(record: dict[str, object], output_dir: Path) -> Path:
    exp, dpd, lv = (record[key].sort_values("day") for key in ("experimental", "dpd", "lv"))
    panels = (
        ("LV", lv["day"], lv["lv_sensitive"], lv["lv_resistant"], lv_spans(lv)),
        ("DPD", dpd["day"], dpd["n_sensitive_clean"], dpd["n_resistant_total"], record["dpd_spans"]),
        ("In vitro", exp["day"], exp["sensitive"], exp["resistant"], record["invitro_spans"]),
    )
    ylim = positive_limits([exp["sensitive"], exp["resistant"], dpd["n_sensitive_clean"],
                            dpd["n_resistant_total"], lv["lv_sensitive"], lv["lv_resistant"]])
    xmax = max(float(exp["day"].max()), float(dpd["day"].max()), float(lv["day"].max()))
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.2), sharex=True, sharey=True)
    for ax, (label, days, sensitive, resistant, spans) in zip(axes, panels):
        shade(ax, spans)
        marker = "o" if label == "In vitro" else None
        ax.plot(days, sensitive, color=SENSITIVE_COLOR, lw=1.7, marker=marker, ms=3, zorder=2)
        ax.plot(days, resistant, color=RESISTANT_COLOR, lw=1.7, marker=marker, ms=3, zorder=2)
        ax.set_yscale("log")
        ax.set(xlim=(1, xmax), ylim=ylim, ylabel="Cell count")
        ax.set_title(label, loc="left", fontsize=10, weight="bold")
        ax.grid(axis="y", which="major", color="#E5E5E5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time [days]")
    axes[0].legend(handles=[Line2D([], [], color=SENSITIVE_COLOR, lw=1.8, label="Sensitive"),
                            Line2D([], [], color=RESISTANT_COLOR, lw=1.8, label="Resistant"),
                            Patch(facecolor=TREATMENT_COLOR, alpha=0.32, edgecolor="none",
                                  label="Treatment on")],
                   frameon=False, ncol=3, loc="upper right")
    fig.suptitle(f'{record["plate"]} - {record["sample"]} - DPD R{record["run"]}',
                 x=0.08, ha="left", fontsize=12, weight="bold")
    fig.text(0.08, 0.01,
             "Gray: adaptive LV action; saved DPD action; original in-vitro calendar-day schedule.",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0.04, 0.035, 0.99, 0.955))
    path = output_dir / f'{record["plate"]}_{record["sample"]}_R{record["run"]}_treatment_diagnostic.pdf'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    args, module = parse_args(), load_trajectory_module()
    experiment_root = args.experiment_root or module.EXPERIMENT_ROOT
    dpd_counts = args.dpd_counts or module.DPD_COUNTS
    dpd_status_path = args.dpd_treatment_status or (
        dpd_counts.parents[1] / "summary_plots" / "requested_variations_20260716"
        / "simulation_treatment_status.tsv"
    )
    params_path = args.params or module.PARAMS_PATH
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experimental, _ = module.apply_filters(module.read_experiments(experiment_root))
    dpd_all = module.read_dpd(dpd_counts)
    invitro_schedules = read_original_invitro_schedules(experiment_root)
    dpd_status = read_dpd_treatment_status(dpd_status_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    outputs, audits, values = [], [], []
    for plate, sample, run in SELECTED:
        exp = experimental[(experimental["plate"] == plate) & (experimental["sample"] == sample)].copy()
        if exp.empty:
            raise ValueError(f"Filtered in-vitro replicate not found: {plate} / {sample}")
        strategy = str(exp["strategy"].iloc[0])
        dpd = dpd_all[(dpd_all["run"] == run) & (dpd_all["strategy"] == strategy)].copy()
        if dpd.empty:
            raise ValueError(f"DPD trajectory not found: R{run} / {strategy}")
        exp_days = invitro_schedules.get((plate, sample), set())
        exp_spans = daily_spans(exp_days, float(exp["day"].max()))
        saved_dpd_spans = dpd_saved_spans(dpd_status, strategy, run, float(dpd["day"].max()))
        last_day = int(max(exp["day"].max(), dpd["day"].max()))
        first = exp.sort_values("day").iloc[0]
        lv = simulate_adaptive_lv(module, params, float(first["sensitive"]),
                                  float(first["resistant"]), strategy, last_day)
        record = dict(plate=plate, sample=sample, strategy=strategy, run=run,
                      experimental=exp, dpd=dpd, lv=lv,
                      invitro_spans=exp_spans, dpd_spans=saved_dpd_spans)
        outputs.append(plot_diagnostic(record, args.output_dir))
        audits.append({
            "plate": plate, "sample": sample, "strategy": strategy, "dpd_run": run,
            "invitro_action_source": str(experiment_root / plate / "treatment_schedule.txt"),
            "invitro_treated_calendar_days": " ".join(map(str, sorted(exp_days))),
            "dpd_action_source": str(dpd_status_path),
            "dpd_treated_calendar_days": " ".join(str(int(start)) for start, _ in saved_dpd_spans),
            "lv_action_source": f"simulated_{strategy}_policy",
            "lv_treated_interval_starts": " ".join(str(int(start)) for start, _ in lv_spans(lv)),
            "invitro_last_day": int(exp["day"].max()), "dpd_last_day": int(dpd["day"].max()),
            "lv_last_day": int(lv["day"].max()),
        })
        for source, frame, sc, rc in (("LV", lv, "lv_sensitive", "lv_resistant"),
                                      ("DPD", dpd, "n_sensitive_clean", "n_resistant_total"),
                                      ("In vitro", exp, "sensitive", "resistant")):
            values.append(pd.DataFrame({"plate": plate, "sample": sample, "strategy": strategy,
                                        "dpd_run": run, "source": source, "day": frame["day"],
                                        "sensitive_count": frame[sc], "resistant_count": frame[rc]}))
    pd.DataFrame(audits).to_csv(args.output_dir / "selected_replicate_treatment_diagnostic_audit.csv", index=False)
    pd.concat(values, ignore_index=True).to_csv(args.output_dir / "selected_replicate_treatment_diagnostic_values.csv", index=False)
    print("\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
