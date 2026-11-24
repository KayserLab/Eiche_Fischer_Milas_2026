"""Plot experimental time to failure (TTF) against time to escape (TTE).
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

TTF is the first observation at which resistant cells reach 50% of the
replicate's initial total cell count. TTE is the detected escape day. Pearson's
correlation is calculated for CT, AT50, and AT100 replicates with both events
observed within the shared experimental filtering/truncation rules.

Save in-vitro-only and combined plots, each with and without a dashed ordinary
least-squares fit (TTF = intercept + slope * TTE). Each line fits all points in
its panel, including both sources in the combined panel. Statistics describe
observed event pairs only; they do not account for censored replicates.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: canonical all_escape_margin.csv, experimental
# *_total_poisson_padded.h5, and simulation *_simulation.h5 files.
# Scripts: analysis/figure_2/Event_Time_Analysis.py, shared/strategy_colors.py,
# shared/experiment_filters.py (through the shared analysis), and shared/figtools.py.
# Outputs: merged TTF/TTE CSV and correlation plot.

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.data_paths import figure_data_dir
from shared.strategy_colors import get_color, get_display_name, normalize_strategy
from analysis.shared_initial_condition_statistics import (
    attach_initial_conditions,
    cluster_bootstrap_pearson_ci,
)
from analysis.figure_2.Event_Time_Analysis import (
    ESCAPE_MARGIN_CSV,
    derive_progression,
    derive_tte,
    load_escape_margins,
    load_population_counts,
)


STRATEGIES = ("CT", "AT50", "AT100")
# These are raw labels used in the in-vitro source files.  They are converted
# to the canonical plotting labels below (Dens -> AT100+, Agent -> LV-Agent).
INVITRO_EXTENDED_STRATEGIES = ("CT", "AT50", "AT100", "AT100+", "LV-Agent")
FAILURE_THRESHOLD = 0.5
OUTPUT_DIR = ft.paper_figure_data_dir(__file__)
DATA_OUTPUT_DIR = figure_data_dir("Figure_2", "ttf_vs_tte", create=False)
OUTPUT_PDF = "TTF_vs_TTE_Pearson_CT_AT50_AT100.pdf"
OUTPUT_INVITRO_PDF = "TTF_vs_TTE_Pearson_CT_AT50_AT100_invitro.pdf"
OUTPUT_INVITRO_EXTENDED_PDF = "TTF_vs_TTE_Pearson_CT_AT50_AT100_AT100plus_Agent_invitro.pdf"
OUTPUT_CSV = "TTF_vs_TTE_Pearson_CT_AT50_AT100.csv"
AXES_WIDTH_MM = 15.0
AXES_HEIGHT_MM = 35.0
MARGINS_MM = (12, 4, 10, 4)
YLIM = (0, 160)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260827

ft.apply_nature_style(text_pt=7, tick_pt=6, legend_pt=6)


def build_ttf_table(
    raw_strategies: tuple[str, ...] = STRATEGIES,
    *,
    dataset_type: str = "experimental",
    counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if counts is None:
        counts = load_population_counts(
            dataset_type=dataset_type,
            strategies=tuple(raw_strategies),
        )
    if counts.empty:
        raise ValueError("No total-burden replicate data were loaded for TTF.")
    table = derive_progression(
        counts,
        apply_truncation=dataset_type == "experimental",
        threshold=FAILURE_THRESHOLD,
    )
    table = table[
        table["strategy"].isin(raw_strategies) & table["progressed"].astype(bool)
    ].copy()
    table = table[["plate", "sample", "strategy", "ttf", "source"]]
    table["strategy"] = table["strategy"].map(normalize_strategy)
    return table


def build_tte_table(
    raw_strategies: tuple[str, ...] = STRATEGIES,
    *,
    dataset_type: str = "experimental",
    margins: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if margins is None:
        margins = load_escape_margins(ESCAPE_MARGIN_CSV)
    table = derive_tte(
        margins,
        dataset_type=dataset_type,
        strategies=tuple(raw_strategies),
        apply_truncation=dataset_type == "experimental",
    )
    table = table[
        table["strategy"].isin(raw_strategies) & table["escaped"].astype(bool)
    ].copy()
    table = table[["plate", "sample", "strategy", "tte", "source"]]
    table["strategy"] = table["strategy"].map(normalize_strategy)
    return table


def pearson_stats(data: pd.DataFrame) -> tuple[float, float]:
    if len(data) < 3 or data["tte"].nunique() < 2 or data["ttf"].nunique() < 2:
        return np.nan, np.nan
    result = pearsonr(data["tte"].to_numpy(float), data["ttf"].to_numpy(float))
    return float(result.statistic), float(result.pvalue)


def pearson_bootstrap_ci(data: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for Pearson's r, resampling replicate pairs."""
    values = data.loc[:, ["tte", "ttf"]].dropna().to_numpy(dtype=float)
    if values.shape[0] < 3:
        return np.nan, np.nan
    sampled_indices = rng.integers(0, values.shape[0], size=(BOOTSTRAP_REPEATS, values.shape[0]))
    samples = values[sampled_indices]
    x = samples[:, :, 0]
    y = samples[:, :, 1]
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    valid = denominator > 0
    correlations = np.full(BOOTSTRAP_REPEATS, np.nan)
    correlations[valid] = np.sum(x_centered[valid] * y_centered[valid], axis=1) / denominator[valid]
    correlations = correlations[np.isfinite(correlations)]
    if correlations.size == 0:
        return np.nan, np.nan
    low, high = np.quantile(correlations, [0.025, 0.975])
    return float(low), float(high)


def linear_fit(data: pd.DataFrame):
    """OLS of TTF on TTE with a freely estimated intercept, matching Pearson r."""
    if len(data) < 3 or data["tte"].nunique() < 2 or data["ttf"].nunique() < 2:
        return None
    return linregress(data["tte"].to_numpy(float), data["ttf"].to_numpy(float))


def manuscript_correlations(data: pd.DataFrame) -> list[dict]:
    """Source-specific Pearson r with 95% percentile-bootstrap CIs (one generator per source).

    In vitro: colonies are independent, so (TTE, TTF) pairs are resampled and the ordinary
    Pearson t-test is reported. In silico: strategies started from the same experimental
    colony share an initial condition, so whole initial-condition clusters are resampled
    (all eligible strategy rows of a sampled cluster retained); no row-level P is reported
    because the ordinary Pearson test assumes independent rows.
    """
    results = []
    for source in ("In vitro", "In silico"):
        group = data[data["source"] == source]
        if len(group) < 3:
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        r, p = pearson_stats(group)
        if source == "In silico":
            ci = cluster_bootstrap_pearson_ci(group["tte"], group["ttf"], group["initial_condition"], rng,
                                              reps=BOOTSTRAP_REPEATS)
            results.append({"source": source, "n": len(group), "n_clusters": ci["n_clusters"], "r": r,
                            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "ci_method": ci["ci_method"],
                            "degenerate_resamples": ci["degenerate_resamples"], "p": float("nan"),
                            "test": "none (rows clustered by initial condition)",
                            "row_level_pearson_p_not_valid": p})
        else:
            low, high = pearson_bootstrap_ci(group, rng)
            results.append({"source": source, "n": len(group), "n_clusters": len(group), "r": r,
                            "ci_low": low, "ci_high": high,
                            "ci_method": "percentile bootstrap; (TTE, TTF) pairs of independent colonies resampled",
                            "degenerate_resamples": None, "p": p,
                            "test": "two-sided Pearson correlation t-test"})
    return results


def print_statistics(data: pd.DataFrame) -> None:
    """Report manuscript correlations and descriptive pooled/strategy-specific fits."""
    print("Statistics use replicates with both TTE and TTF observed only.")
    for res in manuscript_correlations(data):
        p_text = f"p={res['p']:.6g} ({res['test']})" if np.isfinite(res["p"]) else f"P not reported: {res['test']}"
        print(
            f"[MANUSCRIPT Pearson {res['source']}] n={res['n']} (clusters={res['n_clusters']}), "
            f"r={res['r']:.6g}, 95% CI=[{res['ci_low']:.6g}, {res['ci_high']:.6g}] "
            f"({res['ci_method']}); {p_text}"
        )
    groups = []
    if data["source"].nunique() > 1:
        groups.append(("In vitro + In silico (pooled)", data))
    groups.extend(data.groupby("source", sort=True))
    groups.extend(
        (f"{source} {strategy}", group)
        for (source, strategy), group in data.groupby(["source", "strategy"], sort=True)
    )
    print("Descriptive OLS: TTF = intercept + slope * TTE (days); row-level, not cluster-adjusted.")
    for label, group in groups:
        fit = linear_fit(group)
        if fit is None:
            print(f"[OLS {label}] undefined: need >=3 pairs and variation in both variables.")
            continue
        print(
            f"[OLS {label}] n={len(group)}, r={fit.rvalue:.6g}, TTF={fit.intercept:.6g} + "
            f"{fit.slope:.6g} * TTE, R^2={fit.rvalue**2:.6g}"
        )


def plot(
    data: pd.DataFrame,
    output: Path,
    *,
    show_fit: bool = False,
) -> None:
    fig, ax = ft.new_figure(
        width=AXES_WIDTH_MM, height_mm=AXES_HEIGHT_MM,
        margins=MARGINS_MM, apply_style=False,
    )
    markers = {"In vitro": "o", "In silico": "x"}
    source_labels = {"In vitro": "Exp", "In silico": "Sim"}
    for (source, strategy), group in data.groupby(["source", "strategy"], sort=True):
        ax.scatter(
            group["tte"], group["ttf"], s=18, marker=markers[str(source)],
            label=f"{source_labels[str(source)]}: {get_display_name(strategy, superscript=True)}",
            color=get_color(strategy, "#666666"),
            edgecolor="black" if source == "In vitro" else None,
            linewidth=0.8 if source == "In silico" else 0.35,
            zorder=2,
        )

    if show_fit:
        fit = linear_fit(data)
        if fit is not None:
            # Limit the descriptive fit to the observed TTE range; do not extrapolate.
            x = np.array([data["tte"].min(), data["tte"].max()], dtype=float)
            ax.plot(
                x, fit.intercept + fit.slope * x, color="black",
                linestyle="--", linewidth=0.9, zorder=3,
            )

    ax.set_xlabel("TTE [d]")
    ax.set_ylabel("TTF [d]")
    ax.set_xlim(left=0)
    ax.set_xticks((0, 50, 100))
    ax.set_ylim(*YLIM)
    ax.spines[["top", "right"]].set_visible(False)
    correlation_labels = []
    for source in ("In vitro", "In silico"):
        group = data[data["source"] == source]
        if group.empty:
            continue
        r, _ = pearson_stats(group)
        r_text = f"{r:.2f}" if np.isfinite(r) else "undefined"
        source_text = source_labels[source]
        correlation_labels.append(rf"{source_text}: $r$ = {r_text}")
    ax.text(
        1.0, 1.0, "\n".join(correlation_labels), transform=ax.transAxes,
        ha="right", va="top", multialignment="right", fontsize=5,
    )
    ft.savefig(fig, output)
    plt.close(fig)


def save_variants(data: pd.DataFrame, filename: str) -> None:
    """Save PDF versions with and without the pooled fit."""
    for show_fit in (False, True):
        output = OUTPUT_DIR / filename
        if show_fit:
            output = output.with_stem(output.stem + "_dashed_fit")
        plot(data, output, show_fit=show_fit)
        print(f"[saved] {output}")


def save_standard_outputs(merged: pd.DataFrame) -> None:
    merged = merged.loc[np.isfinite(merged[["tte", "ttf"]]).all(axis=1)].copy()
    if merged.empty:
        raise ValueError("No finite matched TTE/TTF pairs are available.")
    invitro = merged[merged["source"] == "In vitro"].copy()
    if invitro.empty:
        raise ValueError("No in-vitro matched TTE/TTF pairs are available.")
    print_statistics(merged)
    save_variants(merged, OUTPUT_PDF)
    save_variants(invitro, OUTPUT_INVITRO_PDF)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Re-render the saved matched-replicate table without rereading source data.",
    )
    parser.add_argument(
        "--invitro-extended", action="store_true",
        help="Save the in-vitro-only plot including AT100+ and Agent.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        csv_path = DATA_OUTPUT_DIR / OUTPUT_CSV
        if not csv_path.is_file():
            raise FileNotFoundError(f"Saved correlation table not found: {csv_path}")
        merged = pd.read_csv(csv_path)
        if "initial_condition" not in merged:
            simulated = merged["source"].eq("In silico")
            merged.loc[simulated, "initial_condition"] = attach_initial_conditions(
                merged.loc[simulated], strategy_col="strategy", run_col="sample")["initial_condition"]
            merged.loc[~simulated, "initial_condition"] = merged.loc[~simulated, "sample"]
        save_standard_outputs(merged)
        return
    margins = load_escape_margins(ESCAPE_MARGIN_CSV)
    if args.invitro_extended:
        extended_counts = load_population_counts(
            dataset_type="experimental",
            strategies=INVITRO_EXTENDED_STRATEGIES,
        )
        extended_ttf = build_ttf_table(
            INVITRO_EXTENDED_STRATEGIES,
            dataset_type="experimental",
            counts=extended_counts,
        )
        extended_tte = build_tte_table(
            INVITRO_EXTENDED_STRATEGIES,
            dataset_type="experimental",
            margins=margins,
        )
        extended = extended_tte.merge(
            extended_ttf, on=["plate", "sample", "strategy", "source"], how="inner",
            validate="one_to_one",
        )
        # Retain only the requested canonical strategies, in case a source
        # contains additional aliases.
        extended = extended[extended["strategy"].isin(INVITRO_EXTENDED_STRATEGIES)].copy()
        if extended.empty:
            raise ValueError("No extended in-vitro replicates have both observed TTE and TTF events.")
        extended = extended.loc[np.isfinite(extended[["tte", "ttf"]]).all(axis=1)].copy()
        if extended.empty:
            raise ValueError("No finite extended in-vitro matched TTE/TTF pairs are available.")
        print_statistics(extended)
        save_variants(extended, OUTPUT_INVITRO_EXTENDED_PDF)
        return
    experimental_counts = load_population_counts(
        dataset_type="experimental", strategies=STRATEGIES,
    )
    ttf = build_ttf_table(
        STRATEGIES, dataset_type="experimental", counts=experimental_counts,
    )
    tte = build_tte_table(
        STRATEGIES, dataset_type="experimental", margins=margins,
    )
    invitro = tte.merge(
        ttf, on=["plate", "sample", "strategy", "source"], how="inner",
        validate="one_to_one",
    )
    simulation_counts = load_population_counts(
        dataset_type="simulation", strategies=STRATEGIES,
    )
    simulation_ttf = build_ttf_table(
        STRATEGIES, dataset_type="simulation", counts=simulation_counts,
    )
    simulation_tte = build_tte_table(
        STRATEGIES, dataset_type="simulation", margins=margins,
    )
    insilico = simulation_tte.merge(
        simulation_ttf, on=["plate", "sample", "strategy", "source"], how="inner",
        validate="one_to_one",
    )
    insilico = attach_initial_conditions(insilico, strategy_col="strategy", run_col="sample")
    invitro["initial_condition"] = invitro["sample"]  # independent colonies: own cluster
    merged = pd.concat([invitro, insilico], ignore_index=True)
    merged = merged.dropna(subset=["tte", "ttf"]).sort_values(["source", "strategy", "sample"])
    if merged.empty:
        raise ValueError("No replicates have both observed TTE and TTF events.")

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(DATA_OUTPUT_DIR / OUTPUT_CSV, index=False)
    save_standard_outputs(merged)
    print(f"[saved] {DATA_OUTPUT_DIR / OUTPUT_CSV}")


if __name__ == "__main__":
    main()
