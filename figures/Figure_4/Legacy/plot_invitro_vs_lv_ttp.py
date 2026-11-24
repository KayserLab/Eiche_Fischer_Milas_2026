#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream/related script reference(s):
# - AT100.py
# - import matplotlib.py
# - plot_lv_ttp_all_invitro_initial_conditions.py
# - plot_replicate_resistant_invitro_dpd_lv.py
# =============================================================================
'''Compare matched in-vitro and LV time to progression for all baselines.'''
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import figtools as ft
from shared.strategy_colors import get_color


TTP_SCRIPT = REPO_ROOT / 'figures' / 'Figure_2' / 'TTP CT & AT100.py'
LV_TTP_SCRIPT = REPO_ROOT / 'figures' / 'Figure_4' / 'plot_lv_ttp_all_invitro_initial_conditions.py'
TRAJECTORY_SCRIPT = REPO_ROOT / 'figures' / 'Figure_4' / 'plot_replicate_resistant_invitro_dpd_lv.py'
OUTPUT_DIR = ft.paper_figure_data_dir(__file__)
OUTPUT_PDF = OUTPUT_DIR / 'In_vitro_vs_LV_TTP_matched_replicates.pdf'
OUTPUT_PNG = OUTPUT_DIR / 'In_vitro_vs_LV_TTP_matched_replicates.png'
OUTPUT_CSV = OUTPUT_DIR / 'In_vitro_vs_LV_TTP_matched_replicates.csv'
OUTPUT_STATS_CSV = OUTPUT_DIR / 'In_vitro_vs_LV_TTP_matched_replicates_statistics.csv'

AXES_WIDTH_MM = 40.0
AXES_HEIGHT_MM = 40.0
MARGINS_MM = (14.0, 3.0, 12.0, 3.0)
PROTOCOLS = ('CT', 'AT50', 'AT100')
N_BOOTSTRAPS = 10_000
RNG_SEED = 20260809


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_matched_values() -> pd.DataFrame:
    ttp_module = load_module('invitro_ttp', TTP_SCRIPT)
    lv_ttp_module = load_module('lv_ttp', LV_TTP_SCRIPT)
    trajectory_module = load_module('lv_trajectory', TRAJECTORY_SCRIPT)

    ttp_module.STRATEGIES_KEEP = list(PROTOCOLS)
    raw = ttp_module.collect_all_samples()
    normalized = ttp_module.normalize_to_day1_total(raw)
    filtered = ttp_module.apply_experiment_filters(normalized, use_truncation_days=True)
    invitro = ttp_module.compute_ttp(filtered, use_truncation_days=True).rename(
        columns={'ttp': 'invitro_ttp', 'progressed': 'invitro_progressed'}
    )

    baselines = lv_ttp_module.filtered_baselines(ttp_module, lv_ttp_module.BASELINE_ROOT)
    params = json.loads(lv_ttp_module.PARAMS_PATH.read_text(encoding='utf-8'))
    lv_results, _trajectories = lv_ttp_module.simulate_counterfactuals(
        baselines, params, trajectory_module.RampedLv
    )
    lv = lv_results[lv_results['protocol'].isin(PROTOCOLS)].rename(
        columns={'protocol': 'strategy', 'ttp': 'lv_ttp', 'progressed': 'lv_progressed'}
    )[['sample', 'strategy', 'lv_ttp', 'lv_progressed', 'initial_total']]

    merged = invitro.merge(lv, how='inner', on=['sample', 'strategy'])
    merged['invitro_ttp'] = pd.to_numeric(merged['invitro_ttp'], errors='coerce')
    merged['lv_ttp'] = pd.to_numeric(merged['lv_ttp'], errors='coerce')
    merged = merged.dropna(subset=['invitro_ttp', 'lv_ttp']).copy()
    if merged.empty:
        raise RuntimeError('No matched finite in-vitro/LV time-to-progression pairs were found.')
    return merged.sort_values(['strategy', 'sample']).reset_index(drop=True)


def fisher_z_interval(r_value: float, n: int) -> tuple[float, float]:
    if n <= 3:
        return np.nan, np.nan
    half_width = stats.norm.ppf(0.975) / np.sqrt(n - 3)
    return tuple(np.tanh(np.arctanh(np.clip(r_value, -0.999999, 0.999999)) + np.array([-half_width, half_width])))


def bootstrap_r_interval(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    values = []
    for _ in range(N_BOOTSTRAPS):
        index = rng.integers(0, len(x), len(x))
        if np.std(x[index]) and np.std(y[index]):
            values.append(float(np.corrcoef(x[index], y[index])[0, 1]))
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def regression_band(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = stats.linregress(x, y)
    grid = np.linspace(np.min(x), np.max(x), 200)
    fitted = fit.intercept + fit.slope * grid
    residuals = y - (fit.intercept + fit.slope * x)
    residual_se = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
    se_mean = residual_se * np.sqrt(1 / len(x) + (grid - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    return grid, fitted, stats.t.ppf(0.975, len(x) - 2) * se_mean


def correlation_statistics(values: pd.DataFrame) -> pd.DataFrame:
    x = values['invitro_ttp'].to_numpy(float)
    y = values['lv_ttp'].to_numpy(float)
    r_value, p_value = stats.pearsonr(x, y)
    fisher_low, fisher_high = fisher_z_interval(float(r_value), len(x))
    bootstrap_low, bootstrap_high = bootstrap_r_interval(x, y)
    return pd.DataFrame([{
        'n': len(x), 'pearson_r': r_value, 'pearson_p': p_value,
        'fisher_z_ci_low': fisher_low, 'fisher_z_ci_high': fisher_high,
        'bootstrap_ci_low': bootstrap_low, 'bootstrap_ci_high': bootstrap_high,
    }])


def plot(values: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = ft.new_figure(width=AXES_WIDTH_MM, height_mm=AXES_HEIGHT_MM,
                            margins=MARGINS_MM, dpi=300, apply_style=True)
    for strategy in PROTOCOLS:
        sub = values[values['strategy'] == strategy]
        if sub.empty:
            continue
        progressed = sub['lv_progressed'].astype(bool)
        ax.scatter(sub.loc[progressed, 'invitro_ttp'], sub.loc[progressed, 'lv_ttp'],
                   s=12, color=get_color(strategy, '#777777'), alpha=0.75, edgecolors='black', linewidths=0.25,
                   label=strategy, zorder=3)
        if (~progressed).any():
            ax.scatter(sub.loc[~progressed, 'invitro_ttp'], sub.loc[~progressed, 'lv_ttp'],
                       s=14, marker='s', facecolors='none', edgecolors=get_color(strategy, '#777777'), linewidths=0.8,
                       label=f'{strategy} (LV censored)', zorder=3)
    x = values['invitro_ttp'].to_numpy(float)
    y = values['lv_ttp'].to_numpy(float)
    grid, fitted, half_band = regression_band(x, y)
    ax.fill_between(grid, fitted - half_band, fitted + half_band, color='#333333', alpha=0.12,
                    linewidth=0, zorder=1)
    ax.plot(grid, fitted, color='#333333', linewidth=1.0, zorder=2, label='OLS fit')
    x_max = float(values['invitro_ttp'].max())
    y_max = float(values['lv_ttp'].max())
    ax.set_xlim(0, x_max * 1.05 if x_max > 0 else 1.0)
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
    ax.set_xlabel('In-vitro time to progression [days]')
    ax.set_ylabel('LV time to progression [days]')
    row = summary.iloc[0]
    ax.text(0.98, 0.02,
            f'Pearson r={row.pearson_r:.2f}, p={row.pearson_p:.3g}, n={int(row.n)}\n'
            f'Fisher 95% CI [{row.fisher_z_ci_low:.2f}, {row.fisher_z_ci_high:.2f}]\n'
            f'Bootstrap 95% CI [{row.bootstrap_ci_low:.2f}, {row.bootstrap_ci_high:.2f}]',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=5.0)
    ax.legend(frameon=False, fontsize=5.0, loc='upper left', handletextpad=0.35, labelspacing=0.25)
    ft.format_axes(ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, OUTPUT_PDF, dpi=300, bbox_inches=None, pad_inches=0)
    ft.savefig(fig, OUTPUT_PNG, dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    values = build_matched_values()
    summary = correlation_statistics(values)
    values.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(OUTPUT_STATS_CSV, index=False)
    plot(values, summary)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'Matched pairs: {len(values)}')


if __name__ == '__main__':
    main()
