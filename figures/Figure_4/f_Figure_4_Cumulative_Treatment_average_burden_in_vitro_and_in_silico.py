#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Inputs and outputs are resolved from the canonical data root through
# shared/data_paths.py. Run analysis/figure_4/ef_Figure_4_prepare_plot_data.py first.
# =============================================================================
"""Create Figure 4 cumulative-treatment versus average-burden plots.

Set ``PLOT_TARGET`` below to ``"in_vitro"``, ``"in_silico"``, or ``"both"``.
The script plots prepared summaries beneath the configured data root. Its
public ``build_*`` functions are also used by
``ef_Figure_4_prepare_plot_data.py`` to rebuild those summaries from the
deposited experimental and simulation inputs.
"""
from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: prepared treatment/burden and KM CSVs; rebuilding uses in-vitro
# Results/*.txt, simulation_density_run_manifest.tsv, HDF5, and combined_results.csv.
# Scripts: ef_Figure_4_prepare_plot_data.py (producer), shared/experiment_filters.py,
# shared/strategy_colors.py, and shared/figtools.py.
# Outputs: cumulative-treatment versus average-burden PDFs and source CSVs.

import csv
import math
import os
import re
import sys
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle
from scipy.stats import mannwhitneyu

# Choose one: "in_vitro", "in_silico", or "both".
PLOT_TARGET = "both"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.experiment_filters import EXCLUDE_SAMPLES
from shared import figtools as ft
from shared.strategy_colors import get_color, get_display_name
from shared.data_paths import COMBINED_RESULTS_CSV, EXP_RAW_DATA, figure_data_dir, intermediate_dir, resolve_stored_data_path

FIGURE_AXES_WIDTH_MM, FIGURE_AXES_HEIGHT_MM = 30.0, 30.0
FIGURE_MARGINS_MM = (11.0, 1.0, 10.0, 1.0)
OVERLAID_MARGINS_MM = (11.0, 1.0, 10.0, 5.0)
TEXT_SIZE_PT, TICK_SIZE_PT, LEGEND_SIZE_PT, DPI = 7.0, 6.0, 6.0, 300
SHOW_FIGURE = True
X_AXIS_LABEL = r"Normalized count, $\eta_{\mathrm{tot}}$"
Y_AXIS_LABEL = "Treatment Fraction"
ft.apply_nature_style(text_pt=TEXT_SIZE_PT, tick_pt=TICK_SIZE_PT,
                      legend_pt=LEGEND_SIZE_PT,
                      overrides={"font.family": "Arial", "mathtext.fontset": "custom",
                                 "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                                 "mathtext.bf": "Arial:bold", "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                                 "ytick.major.width": 0.8, "xtick.major.size": 2.5,
                                 "ytick.major.size": 2.5})

VITRO_ROOTS = [EXP_RAW_DATA]
SIMULATION_MANIFEST = intermediate_dir(
    "simulation", "basic", "prepared_pointclouds", create=False
) / "simulation_pointcloud_run_manifest.tsv"
PUB_DIR = ft.paper_figure_data_dir(__file__)
PUB_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = figure_data_dir("Figure_4", "cumulative_treatment_burden")
INCLUDE_STRATEGIES = ["AT50", "AT100", "CT", "ATAgent2", "Dens"]
DISPLAY_NAME = {s: get_display_name(s) for s in INCLUDE_STRATEGIES}

VITRO_REPLICATE_CSV = DATA_DIR / "cumulative_treatment_average_burden_in_vitro_per_replicate.csv"
VITRO_SUMMARY_CSV = DATA_DIR / "cumulative_treatment_average_burden_in_vitro_strategy_summary.csv"
SILICO_REPLICATE_CSV = DATA_DIR / "cumulative_treatment_average_burden_in_silico_per_replicate.csv"
SILICO_SUMMARY_CSV = DATA_DIR / "cumulative_treatment_average_burden_in_silico_strategy_summary.csv"
VITRO_KM_CSV = figure_data_dir("Figure_4", "kaplan_meier") / "time_to_failure_by_well_rel_0.5.csv"
SILICO_KM_CSV = figure_data_dir("Figure_4", "kaplan_meier") / "time_to_failure_by_run_rel_0.5_in_silico.csv"

_DIGITS = re.compile(r"(\d+)")
_STRATEGY = re.compile(r"_([^_-]+)-")
_RESULT = re.compile(r"(?P<plate>Plate_\d+).*?[\\/]Results[\\/]Results_[^_]+_[^_]+_(?P<well>[A-H]\d+)_(?P<strategy>[A-Za-z0-9]+)-", re.I)
_DAY = re.compile(r"(?:^|[_-])d(\d{1,2})$", re.I)
Record = namedtuple("Record", "day total resistant")
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260827


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _bootstrap_median_difference_ci(
    reference: np.ndarray,
    comparison: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return comparison-minus-reference median difference and 95% CI."""
    difference = float(np.median(comparison) - np.median(reference))
    reference_boot = rng.choice(reference, size=(BOOTSTRAP_REPEATS, reference.size), replace=True)
    comparison_boot = rng.choice(comparison, size=(BOOTSTRAP_REPEATS, comparison.size), replace=True)
    differences = np.median(comparison_boot, axis=1) - np.median(reference_boot, axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return difference, float(low), float(high)


ATAGENT_ATPLUS_OUTCOMES = (
    ("average_normalized_total_burden", "average normalized total burden"),
    ("days_under_treatment_pct", "treatment fraction (% of follow-up days)"),
)


def atagent_vs_atplus_statistics(path: Path, label: str) -> list[dict]:
    """ATagent-versus-AT+ run/colony-level comparisons (AT+ minus ATagent).

    Exp (independent colonies): effect = median(AT+) - median(ATagent); percentile bootstrap
    resampling colonies within each group; two-sided Mann--Whitney U (U for ATagent).
    Sim: runs are matched by initial condition; effect = median(AT+) - median(ATagent) on the
    matched pairs (median of within-pair differences also reported); percentile bootstrap
    resampling matched pairs; two-sided Wilcoxon signed-rank on within-pair differences.
    Unadjusted P values.
    """
    from analysis.shared_initial_condition_statistics import initial_condition, paired_comparison

    table = pd.read_csv(path)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for column, display in ATAGENT_ATPLUS_OUTCOMES:
        agent_rows = table.loc[table["strategy"] == "ATAgent2"].dropna(subset=[column])
        atplus_rows = table.loc[table["strategy"] == "Dens"].dropna(subset=[column])
        agent, atplus = agent_rows[column].to_numpy(float), atplus_rows[column].to_numpy(float)
        if agent.size < 2 or atplus.size < 2:
            continue
        base = {"dataset": label, "outcome": column, "display": display,
                "n_agent": int(agent.size), "n_atplus": int(atplus.size),
                "mean_agent": float(agent.mean()), "sd_agent": float(agent.std(ddof=1)),
                "mean_atplus": float(atplus.mean()), "sd_atplus": float(atplus.std(ddof=1)),
                "median_agent_all": float(np.median(agent)), "median_atplus_all": float(np.median(atplus))}
        if label == "Sim":
            frames = []
            for rows in (agent_rows, atplus_rows):
                frame = rows[["sample_key", "strategy", column]].copy()
                frame["initial_condition"] = [initial_condition(s, k) for s, k in zip(frame["strategy"], frame["sample_key"])]
                frames.append(frame)
            paired = paired_comparison(frames[0], frames[1], column, rng, reps=BOOTSTRAP_REPEATS)
            base.update(paired)
        else:
            difference, low, high = _bootstrap_median_difference_ci(agent, atplus, rng)
            test = mannwhitneyu(agent, atplus, alternative="two-sided")
            base.update({"difference_of_group_medians": difference, "difference_of_group_medians_ci": [low, high],
                         "ci_method": "percentile bootstrap; colonies resampled independently within each group",
                         "resamples": BOOTSTRAP_REPEATS,
                         "test": "two-sided Mann--Whitney U (asymptotic, tie and continuity corrected)",
                         "U_agent": float(test.statistic), "U_atplus": float(agent.size * atplus.size - test.statistic),
                         "p": float(test.pvalue)})
        results.append(base)
    return results


def print_atagent_vs_atplus_statistics(path: Path, label: str) -> None:
    """Print ATagent-versus-AT+ statistics without changing plots."""
    if not path.is_file():
        print(f"[MANUSCRIPT STATS] Figure 4j {label}: source table missing ({path})")
        return
    print(f"[MANUSCRIPT STATS] Figure 4j ATagent vs AT+ ({label}; AT+ minus ATagent; unadjusted):")
    for r in atagent_vs_atplus_statistics(path, label):
        low, high = r["difference_of_group_medians_ci"]
        line = (f"  {r['display']}: n={r['n_agent']}/{r['n_atplus']}; means(SD)={r['mean_agent']:.4g}"
                f"({r['sd_agent']:.3g})/{r['mean_atplus']:.4g}({r['sd_atplus']:.3g}); ")
        if label == "Sim":
            plow, phigh = r["median_within_pair_difference_ci"]
            line += (f"matched pairs={r['n_pairs']}; difference of group medians={r['difference_of_group_medians']:.4g} "
                     f"[{low:.4g}, {high:.4g}]; median within-pair difference={r['median_within_pair_difference']:.4g} "
                     f"[{plow:.4g}, {phigh:.4g}]; Wilcoxon signed-rank W+={r['W_plus']:.6g}, p={r['p']:.6g}")
        else:
            line += (f"difference of group medians={r['difference_of_group_medians']:.4g} [{low:.4g}, {high:.4g}]; "
                     f"Mann--Whitney U(ATagent)={r['U_agent']:.6g}, p={r['p']:.6g}")
        print(line)


def summarize_strategy_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    output = []
    for strategy in INCLUDE_STRATEGIES:
        values = grouped.get(strategy, [])
        x = np.asarray([r["average_normalized_total_burden"] for r in values], float)
        y = np.asarray([r["days_under_treatment_pct"] for r in values], float)
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if not len(x) or not len(y):
            continue
        qx = np.percentile(x, [25, 50, 75]); qy = np.percentile(y, [25, 50, 75])
        output.append({"strategy": strategy, "display_strategy": DISPLAY_NAME.get(strategy, strategy),
                       "color": get_color(strategy, default="#000000"), "n_replicates": int(min(len(x), len(y))),
                       "mean_normalized_burden": float(x.mean()), "sd_normalized_burden": float(x.std(ddof=1)) if len(x)>1 else 0.,
                       "median_normalized_burden": float(qx[1]), "q1_normalized_burden": float(qx[0]), "q3_normalized_burden": float(qx[2]),
                       "iqr_normalized_burden": float(qx[2]-qx[0]), "mean_treatment_pct": float(y.mean()),
                       "sd_treatment_pct": float(y.std(ddof=1)) if len(y)>1 else 0., "median_treatment_pct": float(qy[1]),
                       "q1_treatment_pct": float(qy[0]), "q3_treatment_pct": float(qy[2]), "iqr_treatment_pct": float(qy[2]-qy[0])})
    return output


def _walk_files(roots: Iterable[Path], name: str | None = None) -> list[Path]:
    found = []
    for root in roots:
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                if (name and filename == name) or (not name and directory.lower().endswith("results") and filename.lower().startswith("results_") and filename.lower().endswith(".txt")):
                    found.append(Path(directory) / filename)
    return found


def _km_caps(path: Path) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    plate_sample, sample = {}, {}
    if not path.is_file(): return plate_sample, sample
    for row in csv.DictReader(path.open(encoding="utf-8")):
        try: value = float(row["time"])
        except (KeyError, ValueError, TypeError): continue
        if np.isfinite(value):
            key = str(row.get("sample_key", "")); plate = str(row.get("plate", ""))
            plate_sample[(plate, key)] = value; sample[key] = min(sample.get(key, value), value)
    return plate_sample, sample


def _effective_cap(plate: str, sample: str, by_pair: dict, by_sample: dict) -> tuple[bool, float | None]:
    rule = EXCLUDE_SAMPLES.get(sample, "KEEP")
    if rule is None: return False, None
    try: cap = None if rule == "KEEP" else float(rule)
    except (ValueError, TypeError): cap = None
    km = by_pair.get((plate, sample), by_sample.get(sample))
    return True, km if cap is None else (min(cap, km) if km is not None else cap)


def _parse_records(path: Path) -> list[Record]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split(); match = _DAY.search(cells[0].rstrip(",;") if cells else "")
        if match and len(cells) >= 3:
            try:
                sensitive, resistant = int(cells[1]), int(cells[2]); records.append(Record(int(match.group(1)), sensitive+resistant, resistant))
            except ValueError: pass
    return sorted(records)


def build_invitro_per_replicate_rows() -> list[dict]:
    """Rebuild experimental burden/treatment rows from schedules and Results files."""
    by_pair, by_sample = _km_caps(VITRO_KM_CSV); treatment = {}
    include = {s.upper() for s in INCLUDE_STRATEGIES}
    for schedule in _walk_files(VITRO_ROOTS, "treatment_schedule.txt"):
        plate = schedule.parent.name
        for line in schedule.read_text(encoding="utf-8").splitlines():
            cells = line.split()
            if not cells: continue
            sample = cells[0]; match = _STRATEGY.search(sample)
            if not match or match.group(1).upper() not in include: continue
            keep, cap = _effective_cap(plate, sample, by_pair, by_sample)
            if not keep: continue
            starts, stop = [], None
            for token in cells[1:]:
                number = _DIGITS.search(token)
                if number: (starts.append if "s" not in token else lambda _: None)(int(number.group(1)))
                if number and "s" in token: stop = int(number.group(1))
            end = min(float(stop if stop is not None else (max(starts)+1 if starts else 0)), cap) if cap is not None else float(stop if stop is not None else (max(starts)+1 if starts else 0))
            imaging = max(0, math.ceil(end)-1)
            if imaging: treatment[(plate, sample)] = (len({d for d in starts if 1 <= d < end}), imaging, cap)
    rows = []
    for path in _walk_files(VITRO_ROOTS):
        match = _RESULT.search(str(path))
        if not match: continue
        plate, well, strategy = match["plate"], match["well"].upper(), match["strategy"]
        sample = f"{well}_{strategy}-{well}"
        if (plate, sample) not in treatment: continue
        records = _parse_records(path); _keep, cap = _effective_cap(plate, sample, by_pair, by_sample)
        if cap is not None: records = [r for r in records if r.day <= cap]
        # Every strategy, including AT+ (Dens), is referenced to its own day-1
        # total. Replicates lacking a positive day-1 measurement are excluded.
        base = next((r for r in records if r.day == 1 and r.total > 0), None)
        if base is None: continue
        values = np.asarray([r.total / base.total for r in records if r.total > 0], float)
        if not len(values): continue
        treated, imaging, cap = treatment[(plate, sample)]
        rows.append({"plate": plate, "sample_key": sample, "strategy": strategy, "display_strategy": DISPLAY_NAME.get(strategy, strategy),
                     "treated_days": treated, "imaging_days": imaging, "fraction_days_treated": treated/imaging,
                     "days_under_treatment_pct": 100*treated/imaging, "average_normalized_total_burden": float(values.mean()),
                     "normalization_base_day": base.day, "normalization_base_total": base.total, "n_burden_days": len(values),
                     "first_burden_day": records[0].day, "last_burden_day": records[-1].day, "cap_day": cap or "", "results_file": str(path)})
    return rows


def _stored_count(node) -> int:
    if isinstance(node, h5py.Dataset): return int(node.shape[0]) if node.ndim else int(node[()])
    for name in ("block0_values", "values"):
        if name in node: return int(node[name].shape[0])
    raise ValueError(f"Cannot count cells in {node.name}")


def build_insilico_per_replicate_rows() -> list[dict]:
    """Rebuild simulation burden/treatment rows from the simulation manifest."""
    if not SIMULATION_MANIFEST.is_file(): raise FileNotFoundError(SIMULATION_MANIFEST)
    manifest = pd.read_csv(SIMULATION_MANIFEST, sep="\t", dtype=str)
    mapping = {"CT":"CT", "AT50":"AT50", "AT100":"AT100", "AGENT":"ATAgent2", "DENS":"Dens"}
    manifest["source"] = manifest.strategy.str.strip().str.upper(); manifest["strategy"] = manifest.source.map(mapping)
    manifest["run"] = pd.to_numeric(manifest.run_id.str.extract(r"(\d+)$", expand=False), errors="coerce")
    manifest = manifest[manifest.strategy.notna() & (manifest.run >= 10)]
    caps = {}
    if SILICO_KM_CSV.is_file():
        for row in pd.read_csv(SILICO_KM_CSV).itertuples(index=False): caps[str(row.sample_key)] = int(math.ceil(float(row.time)))
    saved_treatment = {}
    if COMBINED_RESULTS_CSV.is_file():
        combined = pd.read_csv(COMBINED_RESULTS_CSV)
        treatment_column = (
            "treatment_on" if "treatment_on" in combined.columns else "treatment"
        )
        if treatment_column not in combined.columns:
            raise ValueError(
                f"Simulation overview lacks a treatment-state column: {COMBINED_RESULTS_CSV}"
            )
        for (protocol, run), frame in combined.groupby(["protocol", "run"]):
            try:
                run_number = int(run)
            except (TypeError, ValueError):
                # Non-numeric labels (for example NT control wells) are not
                # simulation runs represented by the prepared point-cloud manifest.
                continue
            treatment = pd.to_numeric(
                frame.sort_values("day").drop_duplicates("day")[treatment_column],
                errors="coerce",
            ).fillna(0).astype(bool).to_numpy()
            saved_treatment[(str(protocol).upper(), run_number)] = treatment
    rows = []
    for item in manifest.itertuples(index=False):
        processed = resolve_stored_data_path(item.prepared_hdf5_path)
        raw = sorted(resolve_stored_data_path(item.source_run_dir).glob("simulation_*.h5"))
        if not processed.is_file() or not raw: continue
        with h5py.File(processed, "r") as handle:
            days = sorted(set(handle["sus"]) & set(handle["res"])); totals = np.asarray([_stored_count(handle["sus"][d])+_stored_count(handle["res"][d]) for d in days], float)
        if not len(totals) or totals[0] <= 0: continue
        # Day-1 inclusion rule (as for the survival endpoint): burden is normalized to the
        # day-1 total, so runs whose first saved snapshot is not acquisition day 1 are excluded.
        first_day = int(_DIGITS.search(days[0]).group(1))
        if first_day != 1:
            print(f"[exclude] Sim {item.strategy}_run_{int(item.run)}: no day-1 baseline (first saved day {first_day})")
            continue
        key = f"{item.strategy}_run_{int(item.run)}"; n = min(len(raw), len(totals), caps.get(key, len(totals)))
        treated = saved_treatment.get((item.source, int(item.run)))
        if treated is None or len(treated) < n:
            def state(path: Path) -> bool:
                with h5py.File(path, "r") as h: return bool(np.asarray(h["data"][next(k for k in h["data"] if k.strip('\"') == "therapy_switch")])[()])
            with ThreadPoolExecutor(max_workers=16) as pool: treated = np.asarray(list(pool.map(state, raw[:n])))
        exposure = treated[1:n]
        rows.append({"plate": f"Plate_{item.source}", "sample_key": key, "strategy": item.strategy, "display_strategy": DISPLAY_NAME.get(item.strategy, item.strategy),
                     "treated_days": int(exposure.sum()), "imaging_days": int(len(exposure)), "fraction_days_treated": float(exposure.mean()) if len(exposure) else 0.,
                     "days_under_treatment_pct": float(100*exposure.mean()) if len(exposure) else 0., "average_normalized_total_burden": float(np.mean(totals[:n]/totals[0]),),
                     "normalization_base_day": 1, "normalization_base_total": float(totals[0]), "n_burden_days": n, "first_burden_day": 1, "last_burden_day": n,
                     "cap_day": caps.get(key, ""), "results_file": str(processed)})
    return rows


def _load_summary(path: Path) -> list[dict]:
    if not path.is_file(): raise FileNotFoundError(f"Prepared strategy summary not found: {path}. Run ef_Figure_4_prepare_plot_data.py first.")
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def _limits(values, low, high):
    left, right = min(v-l for v,l in zip(values, low)), max(v+h for v,h in zip(values, high)); pad = .07*(right-left or 1)
    return left-pad, right+pad


def _set_treatment_fraction_ticks(ax) -> None:
    """Use compact, non-percent treatment-fraction tick labels."""
    ax.set_yticks(np.linspace(0, 1, 6), labels=("0", "0.2", "0.4", "0.6", "0.8", "1"))


def _save_figure(fig, pdf_path: Path) -> None:
    """Save the publication PDF."""
    ft.savefig(fig, pdf_path)


def _add_outcome_guide(ax) -> None:
    """Draw editable vector faces in the empty plot corners."""
    bad_color, good_color = "#c83e3e", "#27864b"
    for (cx, cy), color, happy in (
        ((0.89, 0.88), bad_color, False),
        ((0.12, 0.12), good_color, True),
    ):
        ax.add_patch(Circle(
            (cx, cy), radius=0.065, transform=ax.transAxes,
            facecolor="white", edgecolor=color, linewidth=0.9, zorder=4,
        ))
        for eye_x in (cx - 0.022, cx + 0.022):
            ax.add_patch(Circle(
                (eye_x, cy + 0.020), radius=0.0065, transform=ax.transAxes,
                facecolor=color, edgecolor="none", zorder=5,
            ))
        ax.add_patch(Arc(
            (cx, cy - (0.008 if happy else 0.038)),
            width=0.075, height=0.050,
            theta1=200 if happy else 20, theta2=340 if happy else 160,
            transform=ax.transAxes, color=color, linewidth=0.9, zorder=5,
        ))


def _plot_one(rows: list[dict], dataset: str, statistic: str) -> None:
    prefix = f"cumulative_treatment_average_burden_{dataset}_{'mean_sd' if statistic == 'mean' else 'median_iqr'}"
    fig, ax = ft.new_figure(width=FIGURE_AXES_WIDTH_MM, height_mm=FIGURE_AXES_HEIGHT_MM, margins=FIGURE_MARGINS_MM, dpi=DPI, apply_style=False)
    values=[]; lows=[]; highs=[]
    for row in rows:
        x = float(row[f"{statistic}_normalized_burden"]); y = float(row[f"{statistic}_treatment_pct"]) / 100
        if statistic == "mean": xl=xh=float(row["sd_normalized_burden"]); yl=yh=float(row["sd_treatment_pct"]) / 100
        else: xl=x-float(row["q1_normalized_burden"]); xh=float(row["q3_normalized_burden"])-x; yl=y-float(row["q1_treatment_pct"]) / 100; yh=float(row["q3_treatment_pct"]) / 100-y
        ax.errorbar(x,y,xerr=np.asarray([[xl],[xh]]),yerr=np.asarray([[yl],[yh]]),fmt="x",markersize=6,markeredgewidth=1.2,capsize=2.5,elinewidth=.9,color=row["color"]); values.append(x); lows.append(xl); highs.append(xh)
    ax.set(xlim=_limits(values,lows,highs), ylim=(-0.05,1.05), xlabel=X_AXIS_LABEL, ylabel=Y_AXIS_LABEL)
    _set_treatment_fraction_ticks(ax)
    ax.xaxis.label.set_fontfamily("Arial")
    ax.yaxis.label.set_fontfamily("Arial")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend([Line2D([0],[0],marker="x",linestyle="",color=r["color"]) for r in rows], [r["display_strategy"] for r in rows], frameon=False, loc="best", fontsize=LEGEND_SIZE_PT)
    _save_figure(fig, PUB_DIR / f"{prefix}.pdf")
    if SHOW_FIGURE and plt.get_backend().lower() != "agg": plt.show()
    plt.close(fig)


def _plot_both_side_by_side(vitro_rows: list[dict], silico_rows: list[dict], statistic: str) -> None:
    """Plot the two datasets in separate, side-by-side axes."""
    fig, axes = ft.new_subplots(
        1, 2, ax_width_mm=30, ax_height_mm=30, margins=(22, 22, 14, 14),
        wspace_mm=6, dpi=DPI, sharex=True, sharey=True, apply_style=False,
    )
    all_values=[]; all_low=[]; all_high=[]
    for ax, rows, label in zip(axes, (vitro_rows, silico_rows), ("In vitro", "In silico")):
        for row in rows:
            x=float(row[f"{statistic}_normalized_burden"]); y=float(row[f"{statistic}_treatment_pct"]) / 100
            xl=x-float(row["q1_normalized_burden"]) if statistic=="median" else float(row["sd_normalized_burden"]); xh=float(row["q3_normalized_burden"])-x if statistic=="median" else float(row["sd_normalized_burden"])
            yl=y-float(row["q1_treatment_pct"]) / 100 if statistic=="median" else float(row["sd_treatment_pct"]) / 100; yh=float(row["q3_treatment_pct"]) / 100-y if statistic=="median" else float(row["sd_treatment_pct"]) / 100
            ax.errorbar(x, y, xerr=np.asarray([[xl], [xh]]), yerr=np.asarray([[yl], [yh]]), fmt="x", color=row["color"], capsize=2.5)
            all_values.append(x); all_low.append(xl); all_high.append(xh)
        ax.set(title=label, ylim=(-0.05,1.05), xlabel=X_AXIS_LABEL)
    axes[0].set_ylabel(Y_AXIS_LABEL)
    _set_treatment_fraction_ticks(axes[0])
    for ax in axes:
        ax.xaxis.label.set_fontfamily("Arial")
    axes[0].yaxis.label.set_fontfamily("Arial")
    axes[0].set_xlim(*_limits(all_values, all_low, all_high)); axes[1].set_ylabel("")
    _save_figure(fig, PUB_DIR / f"cumulative_treatment_average_burden_in_vitro_and_in_silico_{'mean_sd' if statistic=='mean' else 'median_iqr'}.pdf")
    plt.close(fig)


def _plot_both_overlaid(vitro_rows: list[dict], silico_rows: list[dict], statistic: str) -> None:
    """Overlay in-vitro and in-silico summaries on one shared data axes."""
    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM, height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=OVERLAID_MARGINS_MM, dpi=DPI, apply_style=False,
    )
    all_values=[]; all_low=[]; all_high=[]
    for rows, marker in ((vitro_rows, "o"), (silico_rows, "x")):
        for row in rows:
            x=float(row[f"{statistic}_normalized_burden"]); y=float(row[f"{statistic}_treatment_pct"]) / 100
            xl=x-float(row["q1_normalized_burden"]) if statistic=="median" else float(row["sd_normalized_burden"]); xh=float(row["q3_normalized_burden"])-x if statistic=="median" else float(row["sd_normalized_burden"])
            yl=y-float(row["q1_treatment_pct"]) / 100 if statistic=="median" else float(row["sd_treatment_pct"]) / 100; yh=float(row["q3_treatment_pct"]) / 100-y if statistic=="median" else float(row["sd_treatment_pct"]) / 100
            ax.errorbar(
                x, y, xerr=np.asarray([[xl], [xh]]), yerr=np.asarray([[yl], [yh]]),
                fmt="none", color=row["color"], capsize=2.5,
                capthick=0.9, elinewidth=0.9, alpha=0.9, zorder=1,
            )
            marker_kwargs = (
                {"color": row["color"], "linewidths": 1.2}
                if marker == "x"
                else {"facecolors": "white", "edgecolors": row["color"], "linewidths": 1.0}
            )
            ax.scatter([x], [y], marker=marker, s=8, zorder=2, **marker_kwargs)
            all_values.append(x); all_low.append(xl); all_high.append(xh)
    ax.set(
        ylim=(-0.05, 1.05), xlabel=X_AXIS_LABEL,
        ylabel=Y_AXIS_LABEL,
    )
    _set_treatment_fraction_ticks(ax)
    ax.xaxis.label.set_fontfamily("Arial")
    ax.yaxis.label.set_fontfamily("Arial")
    ax.set_xlim(*_limits(all_values, all_low, all_high))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", markersize=3.5, markerfacecolor="white", color="#333333", linestyle="", label="Exp"),
            Line2D([0], [0], marker="x", markersize=3.5, color="#333333", linestyle="", label="Sim"),
        ],
        frameon=False, loc="lower right",
        ncol=1, borderaxespad=0.5,
        fontsize=5.0, handlelength=1.0, handletextpad=0.3,
    )
    _add_outcome_guide(ax)
    _save_figure(fig, PUB_DIR / f"cumulative_treatment_average_burden_in_vitro_and_in_silico_overlaid_{'mean_sd' if statistic=='mean' else 'median_iqr'}.pdf")
    plt.close(fig)


def main() -> None:
    if PLOT_TARGET not in {"in_vitro", "in_silico", "both"}: raise ValueError("PLOT_TARGET must be 'in_vitro', 'in_silico', or 'both'")
    vitro = _load_summary(VITRO_SUMMARY_CSV) if PLOT_TARGET in {"in_vitro", "both"} else []
    silico = _load_summary(SILICO_SUMMARY_CSV) if PLOT_TARGET in {"in_silico", "both"} else []
    if PLOT_TARGET in {"in_vitro", "both"}:
        print_atagent_vs_atplus_statistics(VITRO_REPLICATE_CSV, "Exp")
    if PLOT_TARGET in {"in_silico", "both"}:
        print_atagent_vs_atplus_statistics(SILICO_REPLICATE_CSV, "Sim")
    for statistic in ("mean", "median"):
        if PLOT_TARGET == "in_vitro": _plot_one(vitro, "in_vitro", statistic)
        elif PLOT_TARGET == "in_silico": _plot_one(silico, "in_silico", statistic)
        else:
            _plot_both_side_by_side(vitro, silico, statistic)
            _plot_both_overlaid(vitro, silico, statistic)


if __name__ == "__main__": main()
