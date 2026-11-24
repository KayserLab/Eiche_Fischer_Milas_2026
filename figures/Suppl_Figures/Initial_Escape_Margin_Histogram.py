#!/usr/bin/env python3
"""Plot initial confinement margins for experimental and simulation runs.

Input: canonical ``all_escape_margin.csv`` from
``analysis/figure_2/Escape_Margin_Analysis.py``. The first valid timepoint of
each retained trajectory from the five study strategies (CT, AT50, AT100,
ATAgent, and AT+) is retained. Experimental trajectories are filtered with the
shared exclusion/truncation rules. Simulation trajectories are not subject to
those experimental curation rules: they have no tracked clone labels and their
resistant cells have already passed the simulation's upstream point filter.
Every retained initial frame is in the confinement phase, so the histogram
shows the normalized positive confinement component,
``eta_conf_sen = confinement_count / initial_total``.

Outputs:
* per-replicate source table in ``figure_data/Suppl_Figures``;
* histogram PDF in ``figure_plots/Suppl_Figures``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared import experiment_filters
from shared import figtools as ft
from shared.data_paths import figure_data_dir, figure_plot_dir, intermediate_dir
from shared.strategy_colors import normalize_strategy


INPUT_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
DATA_OUTPUT = (
    figure_data_dir("Suppl_Figures", "initial_escape_margin", create=False)
    / "initial_escape_margin_per_replicate.csv"
)
PLOT_OUTPUT = (
    figure_plot_dir("Suppl_Figures", "initial_escape_margin", create=False)
    / "Initial_Escape_Margin_Histogram.pdf"
)
BIN_WIDTH = 0.05
STUDY_STRATEGIES = ("CT", "AT50", "AT100", "LV-Agent", "AT100+")

ft.apply_nature_style(text_pt=7.0, tick_pt=6.0, legend_pt=6.0)


def load_initial_margins(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical escape-margin table not found: {path}")
    table = pd.read_csv(path)
    required = {
        "dataset_type", "plate", "sample", "strategy", "day",
        "signed_escape_margin", "confinement_count",
        "n_sensitive_retained", "n_resistant_retained",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    table = table.copy()
    table["dataset_type"] = table["dataset_type"].astype(str).str.lower()
    table = table.loc[table["dataset_type"].isin(("experimental", "simulation"))].copy()
    table["strategy"] = table["strategy"].astype(str).map(normalize_strategy)
    table = table.loc[table["strategy"].isin(STUDY_STRATEGIES)].copy()
    table["day"] = pd.to_numeric(table["day"], errors="coerce")
    for column in (
        "signed_escape_margin", "confinement_count",
        "n_sensitive_retained", "n_resistant_retained",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.loc[
        np.isfinite(table["day"])
        & np.isfinite(table["signed_escape_margin"])
        & np.isfinite(table["confinement_count"])
        & np.isfinite(table["n_sensitive_retained"])
        & np.isfinite(table["n_resistant_retained"])
    ].copy()
    experimental = experiment_filters.apply_sample_exclusions(
        table.loc[table["dataset_type"].eq("experimental")].copy(),
        exclude_map=experiment_filters.EXCLUDE_SAMPLES,
        column="sample",
        cutoff_column="day",
        strict=False,
    )
    simulation = table.loc[table["dataset_type"].eq("simulation")].copy()
    filtered = pd.concat((experimental, simulation), ignore_index=True)
    if filtered.empty:
        raise RuntimeError("No escape-margin rows remain after filtering.")

    initial = (
        filtered.sort_values(["dataset_type", "plate", "sample", "day"])
        .groupby(["dataset_type", "plate", "sample"], as_index=False)
        .first()
    )
    initial["initial_total"] = (
        initial["n_sensitive_retained"] + initial["n_resistant_retained"]
    )
    initial = initial.loc[initial["initial_total"] > 0].copy()
    initial["initial_confinement_margin"] = (
        initial["confinement_count"] / initial["initial_total"]
    )
    return initial.rename(
        columns={"day": "initial_day", "signed_escape_margin": "initial_escape_margin"}
    )


def plot_histogram(initial: pd.DataFrame, output: Path) -> None:
    values = initial["initial_confinement_margin"].to_numpy(dtype=float)
    if values.size == 0:
        raise RuntimeError("No finite initial confinement-margin values are available to plot.")
    lower = 0.0
    upper = float(np.ceil(values.max() / BIN_WIDTH) * BIN_WIDTH)
    bins = np.arange(lower, upper + BIN_WIDTH, BIN_WIDTH)

    figure, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(100.0 / 25.4, 110.0 / 25.4), dpi=300,
    )
    for axis, dataset_type, title in zip(
        axes,
        ("experimental", "simulation"),
        ("Experimental", "Simulation"),
        strict=True,
    ):
        subset = initial.loc[initial["dataset_type"].eq(dataset_type)]
        subset_values = subset["initial_confinement_margin"].to_numpy(dtype=float)
        axis.hist(subset_values, bins=bins, color="#4C78A8", alpha=0.8)
        axis.set_ylabel("Number of replicates")
        axis.set_title(f"{title} (n={len(subset_values)})")
        axis.set_xlim(lower, upper)
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel(r"Initial $\eta^{\mathrm{conf}}_{\mathrm{sen}}$")
    figure.subplots_adjust(left=0.16, right=0.96, bottom=0.12, top=0.95, hspace=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    ft.savefig(figure, output)
    plt.close(figure)


def main() -> None:
    initial = load_initial_margins(INPUT_CSV)
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    initial.to_csv(DATA_OUTPUT, index=False)
    plot_histogram(initial, PLOT_OUTPUT)
    print(f"[saved] {DATA_OUTPUT} ({len(initial)} replicates)")
    print(f"[saved] {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
