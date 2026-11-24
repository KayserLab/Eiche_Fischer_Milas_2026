"""Compare resistant-cell trajectories from in vitro, DPD, and fitted LV data.

DPD R10--R49 initial conditions are mapped to experimental wells using
Data_Maja/initial_condition_DB/mapping.png. Experimental exclusions and
per-well truncations always come from experiment_filters.EXCLUDE_SAMPLES.

The fitted LV counterfactual follows the paper-agent training environment
exactly. Day 1 is the baseline observation and later observations follow the
1, 3, 5, 8, ... calendar. Treatment is selected independently from the
in-vitro schedule: CT is always on, AT50 uses 50%/100% hysteresis, and AT100
is on whenever total burden is at least the initial burden.

Each source keeps its complete available filtered horizon.  The resistant-cell
progression threshold is the replicate's initial total cell count, matching the
executable definition in ``TTP CT & AT100.py``.
"""
from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental Results/*.txt/treatment schedules, simulation escaped-cell
# counts, escape summary, fitted_lv_params.json, and the encoded run mapping.
# Scripts: shared/experiment_filters.py and shared/strategy_colors.py.
# Outputs: matched in-vitro/DPD/LV trajectory figures and audit tables.

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.experiment_filters import EXCLUDE_SAMPLES
from shared.strategy_colors import normalize_strategy
from shared.data_paths import FITTING, EXP_RAW_DATA, figure_data_dir, figure_plot_dir, intermediate_dir

EXPERIMENT_ROOT = EXP_RAW_DATA
DPD_COUNTS = intermediate_dir("simulation", "figure_2", "escape_quantification", create=False) / "escape_cell_counts_all_simulations.csv"
ESCAPE_SUMMARY = intermediate_dir("simulation", "figure_2", create=False) / "escape_front_batch_summary_all_res_within_sens.csv"
SCHEDULE_ROOT = FITTING / "PhysiLearning" / "LV_Fitting_MiniRepo" / "data" / (
    "experimental_data_elias"
) / "2025_stem_paper_final_final"
PARAMS_PATH = FITTING / "PhysiLearning" / "LV_Agent_Training_MiniRepo" / (
    "reference"
) / "fitted_lv_params.json"
DEFAULT_OUTPUT = figure_plot_dir("Figure_4", "replicate_trajectories", create=False)
DEFAULT_DATA_OUTPUT = figure_data_dir("Figure_4", "replicate_trajectories", create=False)
OVERWRITE = False  # Set True to replace existing PDFs and CSV exports from PyCharm.

# Transcription of Data_Maja/Data_Maja/initial_condition_DB/mapping.png.
RUN_TO_SAMPLE = {
    10: "A1_AT50-A1", 11: "B2_AT50-B2", 12: "B6_AT50-B6",
    13: "C2_AT50-C2", 14: "C5_AT50-C5", 15: "D1_AT50-D1",
    16: "D3_AT50-D3", 17: "D4_AT50-D4", 18: "D5_AT50-D5",
    19: "D6_AT50-D6", 20: "A2_AT100-A2", 21: "A5_AT100-A5",
    22: "B3_AT100-B3", 23: "B5_AT100-B5", 24: "B6_AT100-B6",
    25: "C1_AT100-C1", 26: "C2_AT100-C2", 27: "C4_AT100-C4",
    28: "C6_AT100-C6", 29: "D4_AT100-D4", 30: "A1_CT-A1",
    31: "A3_CT-A3", 32: "B2_CT-B2", 33: "B3_CT-B3", 34: "C1_CT-C1",
    35: "C2_CT-C2", 36: "C4_CT-C4", 37: "C5_CT-C5", 38: "D2_CT-D2",
    39: "D3_CT-D3", 40: "D4_CT-D4", 41: "D5_CT-D5",
    42: "B2_Dens-B2", 43: "B3_Dens-B3", 44: "B4_Dens-B4",
    45: "C2_Dens-C2", 46: "C3_Dens-C3", 47: "C4_Dens-C4",
    48: "D3_Dens-D3", 49: "D5_Dens-D5",
}
RUN_TO_PLATE = {
    11: "Plate_2", 12: "Plate_2", 13: "Plate_3", 14: "Plate_3", 15: "Plate_4", 16: "Plate_4", 17: "Plate_4", 18: "Plate_4", 19: "Plate_4",
    20: "Plate_5", 21: "Plate_5", 22: "Plate_1", 23: "Plate_1", 24: "Plate_1", 25: "Plate_2", 26: "Plate_2", 27: "Plate_2", 28: "Plate_2", 29: "Plate_3",
    30: "Plate_2", 31: "Plate_2", 32: "Plate_3", 33: "Plate_3", 34: "Plate_4", 35: "Plate_4", 36: "Plate_4", 37: "Plate_4", 38: "Plate_5", 39: "Plate_5", 40: "Plate_5", 41: "Plate_5",
}
RESULT_RE = re.compile(
    r"^Results_G418_a2a4_(?P<sample>[A-H]\d+_(?P<strategy>[^-]+)-[A-H]\d+)\.txt$"
)
RUN_RE = re.compile(r"^R(?P<run>\d+)_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    p.add_argument("--dpd-counts", type=Path, default=DPD_COUNTS)
    p.add_argument("--escape-summary", type=Path, default=ESCAPE_SUMMARY)
    p.add_argument("--schedule-root", type=Path, default=SCHEDULE_ROOT)
    p.add_argument("--params", type=Path, default=PARAMS_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--data-output-dir", type=Path, default=DEFAULT_DATA_OUTPUT)
    p.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                   help="Replace existing PDFs and CSV exports.")
    return p.parse_args()


def read_experiments(root: Path) -> pd.DataFrame:
    frames = []
    mapped_pairs = {(RUN_TO_PLATE[r], s) for r, s in RUN_TO_SAMPLE.items() if r in RUN_TO_PLATE}
    for plate_dir in sorted(root.glob("Plate_*")):
        for path in sorted((plate_dir / "Results").glob("Results_G418_a2a4_*.txt")):
            match = RESULT_RE.match(path.name)
            if not match:
                continue
            sample = match.group("sample")
            if (plate_dir.name, sample) not in mapped_pairs:
                continue
            table = pd.read_csv(
                path, sep=r"\s+", header=None,
                names=["day_token", "sensitive", "resistant"],
            )
            table["day"] = pd.to_numeric(
                table["day_token"].str.extract(r"_d(\d+)$", expand=False),
                errors="coerce",
            )
            table["plate"] = plate_dir.name
            table["sample"] = sample
            table["strategy"] = normalize_strategy(match.group("strategy"))
            table["source_path"] = str(path)
            frames.append(table)
    if not frames:
        raise ValueError(f"No mapped experimental result files found below {root}")
    data = pd.concat(frames, ignore_index=True)
    for col in ("day", "sensitive", "resistant"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["day", "sensitive", "resistant"])


def apply_filters(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = []
    kept = []
    for (plate, sample), group in data.groupby(["plate", "sample"], sort=True):
        cutoff = EXCLUDE_SAMPLES.get(sample, np.inf)
        if cutoff is None:
            audit.append({"plate": plate, "sample": sample, "status": "excluded",
                          "cutoff_elapsed_day": np.nan,
                          "last_kept_calendar_day": np.nan})
            continue
        group = group.copy()
        group["day_tp1"] = group["day"] - float(group["day"].min())
        group = group[group["day_tp1"] <= float(cutoff)].copy()
        status = "truncated" if np.isfinite(cutoff) else "included"
        if group.empty or (group["resistant"] <= 0).any():
            audit.append({"plate": plate, "sample": sample,
                          "status": "excluded_nonpositive_resistant",
                          "cutoff_elapsed_day": cutoff,
                          "last_kept_calendar_day": np.nan})
            continue
        audit.append({"plate": plate, "sample": sample, "status": status,
                      "cutoff_elapsed_day": cutoff,
                      "last_kept_calendar_day": float(group["day"].max())})
        kept.append(group)
    if not kept:
        raise ValueError("No experimental replicates remain after filtering")
    return pd.concat(kept, ignore_index=True), pd.DataFrame(audit)


def read_schedules(root: Path) -> dict[tuple[str, str], set[int]]:
    schedules: dict[tuple[str, str], set[int]] = {}
    for path in sorted(root.glob("Plate */treatment_schedule.txt")):
        plate = "Plate_" + path.parent.name.split()[-1]
        for raw in path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split("\t", 1)
            if not parts:
                continue
            interval_indices = set()
            if len(parts) == 2:
                for token in parts[1].split():
                    match = re.match(r"(\d+)", token)
                    if match:
                        interval_indices.add(int(match.group(1)))
            schedules[(plate, parts[0])] = interval_indices
    return schedules


def read_dpd(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["run"] = pd.to_numeric(
        data["sample"].astype(str).str.extract(RUN_RE, expand=False),
        errors="coerce",
    )
    data["strategy"] = data["strategy"].astype(str).map(normalize_strategy)
    return data


def read_escape_times(path: Path) -> dict[tuple[str, str], float]:
    data = pd.read_csv(path)
    detected = data["escape_detected"].astype(str).str.lower().eq("true")
    ok = data["status"].astype(str).str.lower().eq("ok")
    data["escape_day"] = pd.to_numeric(data["first_escape_day"], errors="coerce")
    data = data[detected & ok & (data["escape_day"] >= 0)].copy()
    result = {}
    for row in data.itertuples(index=False):
        cutoff = EXCLUDE_SAMPLES.get(str(row.sample), np.inf)
        if cutoff is not None and float(row.escape_day) <= float(cutoff):
            result[(str(row.plate), str(row.sample))] = float(row.escape_day)
    return result


class RampedLv:
    """Exact, dependency-light mirror of the fitted agent-training LvEnv."""

    def __init__(
        self,
        params: dict[str, float],
        sensitive: float,
        resistant: float,
        start_day: int = 1,
    ):
        self.p = params
        total = sensitive + resistant
        if not np.isfinite(total) or total <= 0:
            raise ValueError("LV initial total must be positive and finite")
        self.scale = total
        self.s = sensitive / total
        self.r = resistant / total
        self.gf, self.df = 1.0, 0.0
        self.on_time, self.off_time = 0.0, 100.0
        self.day = int(start_day)
        self.time = 0
        self.last_action = 0

    def grow_one_population(self, index: int, action: int) -> float:
        p, dt = self.p, self.p["timestep_size"]
        if action:
            self.off_time = 0.0
            self.on_time += dt
        else:
            self.on_time = 0.0
            self.off_time += dt
        active = (
            (action and self.on_time > p["on_treat_threshold"])
            or (not action and self.off_time < p["off_treat_threshold"])
        )
        up, down = dt * 2 / p["ramp_time_up"], dt * 2 / p["ramp_time_down"]
        if active:
            if self.gf > 0:
                self.gf = max(0.0, self.gf - down)
            else:
                self.df = min(1.0, self.df + down)
        elif self.df > 0:
            self.df = max(0.0, self.df - up)
        else:
            self.gf = min(1.0, self.gf + up)

        values = [self.s, self.r]
        growth_rates = [p["growth_rate_wt"] * self.gf, p["growth_rate_mut"]]
        death_rates = [p["death_rate_wt"], p["death_rate_mut"]]
        treatment_death_rates = [
            p["treat_death_rate_wt"] * self.df,
            p["treat_death_rate_mut"],
        ]
        competition = [p["competition_wt"], p["competition_mut"]]
        treatment_vector = [1, 0] if self.df > 0 else [0, 0]
        other = 1 - index

        new_size = values[index] * (
            1
            + dt
            * growth_rates[index]
            * (
                1
                - (values[index] + values[other] * competition[other])
                / p["carrying_capacity"]
            )
            - treatment_death_rates[index] * treatment_vector[index] * dt
            - death_rates[index] * dt
        ) - treatment_vector[index] * p["min_death_during_treat"] * dt
        return max(0.0, 0.0 if new_size < 1.0e-5 else float(new_size))

    def inner(self, action: int) -> None:
        """Advance one historical LvEnv inner loop."""
        self.time += 1
        self.last_action = int(action)
        # LvEnv updates populations sequentially and updates the treatment
        # ramp once inside each population's grow() call.
        self.s = self.grow_one_population(0, int(action))
        self.r = self.grow_one_population(1, int(action))

    def steps_for_interval(self, day_span: int) -> int:
        return int(round(float(self.p["treatment_time_step"]) * day_span / 2.0))

    def advance_interval(self, action: int, next_day: int) -> None:
        day_span = int(next_day) - self.day
        if day_span not in (2, 3):
            raise ValueError(f"Expected a 2- or 3-day observation interval, got {day_span}")
        for _ in range(self.steps_for_interval(day_span)):
            self.inner(int(action))
        self.day = int(next_day)

    @staticmethod
    def next_observation_day(day: int) -> int:
        if day == 5 or (day > 5 and (day - 5) % 7 == 0):
            return day + 3
        return day + 2

    def simulate(
        self,
        last_day: int,
        treatment_intervals: set[int],
    ) -> pd.DataFrame:
        """Replay interval-indexed treatment through the last raw calendar day."""
        rows = [{
            "day": self.day,
            "lv_sensitive": self.s * self.scale,
            "lv_resistant": self.r * self.scale,
        }]
        interval_index = 1
        while self.day < int(last_day):
            next_day = self.next_observation_day(self.day)
            if next_day > int(last_day):
                break
            self.advance_interval(
                action=int(interval_index in treatment_intervals),
                next_day=next_day,
            )
            rows.append({
                "day": self.day,
                "lv_sensitive": self.s * self.scale,
                "lv_resistant": self.r * self.scale,
            })
            interval_index += 1
        return pd.DataFrame(rows)

    def simulate_protocol(self, last_day: int, protocol: str) -> pd.DataFrame:
        """Simulate CT, AT50, or AT100 adaptively through the requested day."""
        rows = [{
            "day": self.day,
            "lv_sensitive": self.s * self.scale,
            "lv_resistant": self.r * self.scale,
            "treatment_on": 0,
        }]
        previous_action = True
        while self.day < int(last_day):
            next_day = self.next_observation_day(self.day)
            if next_day > int(last_day):
                break
            total = self.s + self.r
            if protocol == "CT":
                action = True
            elif protocol == "AT100":
                action = total >= 1.0
            elif protocol == "AT50":
                if previous_action and total <= 0.5:
                    action = False
                elif not previous_action and total >= 1.0:
                    action = True
                else:
                    action = previous_action
            else:
                raise ValueError(f"Unsupported LV protocol: {protocol}")
            rows[-1]["treatment_on"] = int(action)
            self.advance_interval(int(action), next_day)
            rows.append({
                "day": self.day,
                "lv_sensitive": self.s * self.scale,
                "lv_resistant": self.r * self.scale,
                "treatment_on": 0,
            })
            previous_action = action
        return pd.DataFrame(rows)


def build_comparisons(
    experimental: pd.DataFrame, dpd: pd.DataFrame,
    schedules: dict[tuple[str, str], set[int]], params: dict[str, float],
    escape_times: dict[tuple[str, str], float],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    sample_to_run = {sample: run for run, sample in RUN_TO_SAMPLE.items()}
    comparisons, long_frames = [], []
    for (plate, sample), exp in experimental.groupby(["plate", "sample"], sort=True):
        exp = exp.sort_values("day")
        strategy = str(exp["strategy"].iloc[0])
        run = sample_to_run[sample]
        sim = dpd[(dpd["run"] == run) & (dpd["strategy"] == strategy)].copy()
        sim = sim.sort_values("day")
        if sim.empty or (sim["n_resistant_total"] <= 0).any():
            continue
        experimental_last_day = int(exp["day"].max())
        dpd_last_day = int(sim["day"].max())
        comparison_last_day = int(max(experimental_last_day, dpd_last_day))
        lv = RampedLv(
            params, float(exp.iloc[0]["sensitive"]), float(exp.iloc[0]["resistant"])
        ).simulate_protocol(comparison_last_day, strategy)
        escape_day = escape_times.get((plate, sample), np.nan)
        escape_marker_shown = bool(
            np.isfinite(escape_day)
            and float(exp["day"].min()) <= float(escape_day) <= experimental_last_day
        )
        progression_threshold = float(
            exp.iloc[0]["sensitive"] + exp.iloc[0]["resistant"]
        )
        comparisons.append({
            "plate": plate, "sample": sample, "strategy": strategy, "run": run,
            "experimental": exp, "dpd": sim, "lv": lv,
            "experimental_last_day": experimental_last_day,
            "dpd_last_day": dpd_last_day,
            "lv_last_day": int(lv["day"].max()),
            "last_day": comparison_last_day,
            "escape_day": escape_day,
            "escape_marker_shown": escape_marker_shown,
            "escape_marker_reason": (
                "shown_within_experimental_horizon"
                if escape_marker_shown
                else ("no_detected_escape" if not np.isfinite(escape_day)
                      else "outside_filtered_experimental_horizon")
            ),
            "progression_resistant_count": progression_threshold,
        })
        long_frames.extend([
            pd.DataFrame({"plate": plate, "sample": sample, "strategy": strategy,
                          "dpd_run": run, "source": "In vitro", "day": exp["day"],
                          "resistant_count": exp["resistant"]}),
            pd.DataFrame({"plate": plate, "sample": sample, "strategy": strategy,
                          "dpd_run": run, "source": "DPD", "day": sim["day"],
                          "resistant_count": sim["n_resistant_total"]}),
            pd.DataFrame({"plate": plate, "sample": sample, "strategy": strategy,
                          "dpd_run": run, "source": "LV", "day": lv["day"],
                          "resistant_count": lv["lv_resistant"]}),
        ])
    if not comparisons:
        raise ValueError("No matched, positive-valued replicate comparisons found")
    return comparisons, pd.concat(long_frames, ignore_index=True)


def plot_all(comparisons: list[dict[str, object]], output: Path) -> None:
    all_values = np.concatenate([
        np.asarray(c[source][column], dtype=float)
        for c in comparisons
        for source, column in (("experimental", "resistant"),
                               ("dpd", "n_resistant_total"), ("lv", "lv_resistant"))
    ] + [
        np.asarray([c["progression_resistant_count"]], dtype=float)
        for c in comparisons
    ])
    ymin, ymax = float(all_values.min()), float(all_values.max())
    xmin, xmax = 0.0, float(max(int(c["last_day"]) for c in comparisons))
    colors = {"In vitro": "#222222", "DPD": "#4C78A8", "LV": "#E45756"}
    for c in comparisons:
        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        ax.plot(c["experimental"]["day"], c["experimental"]["resistant"],
                color=colors["In vitro"], marker="o", ms=2.2, lw=1.2,
                label="In vitro")
        ax.plot(c["dpd"]["day"], c["dpd"]["n_resistant_total"],
                color=colors["DPD"], lw=1.2, label="DPD")
        ax.plot(c["lv"]["day"], c["lv"]["lv_resistant"],
                color=colors["LV"], lw=1.2, label="LV")
        ax.axhline(
            float(c["progression_resistant_count"]), color="#555555",
            linestyle=":", linewidth=1.0, label="Progression threshold",
        )
        if c["escape_marker_shown"]:
            ax.axvline(float(c["escape_day"]), color="#777777", linestyle="--",
                       linewidth=1.0, label="In vitro escape")
        ax.set(xlim=(xmin, xmax), ylim=(1.0, ymax * 1.18),
               xlabel="Time [days]", ylabel="Resistant cell count")
        ax.set_yscale("log")
        ax.set_title(f'{c["plate"]} - {c["sample"]} - DPD R{c["run"]}', fontsize=9)
        ax.legend(frameon=False)
        fig.tight_layout()
        stem = f'{c["plate"]}_{c["sample"]}_R{c["run"]}'
        fig.savefig(output / f"{stem}.pdf")
        ax.set_yscale("linear")
        ax.set_ylim(0.0, ymax * 1.05)
        fig.tight_layout()
        fig.savefig(output / f"{stem}_linear.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    expected_data = [
        args.data_output_dir / "replicate_resistant_comparison_values.csv",
        args.data_output_dir / "replicate_filter_audit.csv",
        args.data_output_dir / "replicate_mapping_used.csv",
    ]
    existing = [path for path in expected_data if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists. Set OVERWRITE = True or pass --overwrite to replace: "
            + ", ".join(map(str, existing))
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_output_dir.mkdir(parents=True, exist_ok=True)
    experimental, audit = apply_filters(read_experiments(args.experiment_root))
    comparisons, long_data = build_comparisons(
        experimental, read_dpd(args.dpd_counts), {},
        json.loads(args.params.read_text(encoding="utf-8")),
        read_escape_times(args.escape_summary),
    )
    plot_all(comparisons, args.output_dir)
    long_data.to_csv(args.data_output_dir / "replicate_resistant_comparison_values.csv",
                     index=False)
    audit.to_csv(args.data_output_dir / "replicate_filter_audit.csv", index=False)
    pd.DataFrame([
        {k: c[k] for k in (
            "plate", "sample", "strategy", "run", "experimental_last_day",
            "dpd_last_day", "lv_last_day", "last_day", "escape_day",
            "escape_marker_shown", "escape_marker_reason",
            "progression_resistant_count",
        )}
        for c in comparisons
    ]).to_csv(args.data_output_dir / "replicate_mapping_used.csv", index=False)
    print(f"Wrote {len(comparisons)} log and {len(comparisons)} linear replicate PDFs to {args.output_dir}")


if __name__ == "__main__":
    main()
