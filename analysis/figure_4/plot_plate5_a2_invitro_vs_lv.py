#!/usr/bin/env python3
'''Publication trajectory panel for the Plate 5 A2 AT100 replicate.

Plots matched in-vitro and fitted-LV trajectories on the same normalized,
linear scale. Colour identifies cell type and line style identifies source.
'''
from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: Plate 5 A2 Results trajectory, treatment schedule,
# all_escape_margin.csv, and fitted_lv_params.json.
# Scripts: plot_replicate_resistant_invitro_dpd_lv.py and shared/figtools.py.
# Outputs: compact Plate 5 A2 matched in-vitro/LV PDFs.

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
for path in (REPO_ROOT, THIS_FILE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared import figtools as ft
from shared.data_paths import figure_plot_dir, intermediate_dir
from analysis.figure_4 import plot_replicate_resistant_invitro_dpd_lv as matched
from shared.strategy_colors import FIGURE_COLOR_MAP


PLATE = 'Plate_5'
SAMPLE = 'A2_AT100-A2'
OUTPUT_DIR = figure_plot_dir('Figure_4', 'Example_Trajectories', create=False)
OUTPUT_PDF = OUTPUT_DIR / 'Plate_5_A2_AT100-A2_in_vitro_and_LV.pdf'
OUTPUT_FULL_X_PDF = OUTPUT_DIR / 'Plate_5_A2_AT100-A2_in_vitro_and_LV_full_x_scale.pdf'
OVERWRITE = False  # Set True to replace the PDFs when running with PyCharm.

AXES_WIDTH_MM = 30.0
AXES_HEIGHT_MM = 20.0
MARGINS_MM = (12.0, 3.0, 10.0, 3.0)
SENSITIVE_COLOR = FIGURE_COLOR_MAP["blue"]
RESISTANT_COLOR = FIGURE_COLOR_MAP["yellow"]
PROGRESSION_THRESHOLD = 0.5
TREATMENT_COLOR = '#C8C8C8'
LV_TREATMENT_HATCH = '.......'
LV_TREATMENT_HATCH_LINEWIDTH = 0.7
# Canonical output of analysis/figure_2/Escape_Margin_Analysis.py.
ESCAPE_MARGIN_CSV = intermediate_dir(
    'combined', 'figure_2', create=False,
) / 'all_escape_margin.csv'

ft.apply_nature_style(
    text_pt=6.0, tick_pt=5.0, legend_pt=5.0,
    overrides={'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8},
)


def read_escape_day() -> float | None:
    if not ESCAPE_MARGIN_CSV.is_file():
        raise FileNotFoundError(
            f'Canonical escape-margin table not found: {ESCAPE_MARGIN_CSV}'
        )
    data = pd.read_csv(ESCAPE_MARGIN_CSV)
    required = {'dataset_type', 'plate', 'sample', 'day', 'signed_escape_margin'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'{ESCAPE_MARGIN_CSV} is missing columns {sorted(missing)}')
    data['day'] = pd.to_numeric(data['day'], errors='coerce')
    data['signed_escape_margin'] = pd.to_numeric(
        data['signed_escape_margin'], errors='coerce'
    )
    escaped = data[
        data['dataset_type'].astype(str).str.lower().eq('experimental')
        & data['plate'].astype(str).eq(PLATE)
        & data['sample'].astype(str).eq(SAMPLE)
        & (data['signed_escape_margin'] > 0)
        & np.isfinite(data['day'])
    ]
    if escaped.empty:
        return None
    escape_day = float(escaped['day'].min())
    cutoff = matched.EXCLUDE_SAMPLES.get(SAMPLE, np.inf)
    if cutoff is None or escape_day > float(cutoff):
        return None
    return escape_day


def load_trajectories():
    experimental, _audit = matched.apply_filters(matched.read_experiments(matched.EXPERIMENT_ROOT))
    exp = experimental[(experimental['plate'] == PLATE) & (experimental['sample'] == SAMPLE)].copy()
    if exp.empty:
        raise RuntimeError(f'No retained in-vitro trajectory for {PLATE} / {SAMPLE}.')
    exp = exp.sort_values('day')
    strategy = str(exp['strategy'].iloc[0])
    params = json.loads(matched.PARAMS_PATH.read_text(encoding='utf-8'))
    lv = matched.RampedLv(
        params, float(exp.iloc[0]['sensitive']), float(exp.iloc[0]['resistant'])
    ).simulate_protocol(int(exp['day'].max()), strategy)
    escape_day = read_escape_day()
    initial_total = float(exp.iloc[0]['sensitive'] + exp.iloc[0]['resistant'])
    return exp, lv, escape_day, initial_total


def read_invitro_treatment_days() -> set[int]:
    schedule_path = matched.EXPERIMENT_ROOT / PLATE / 'treatment_schedule.txt'
    for line in schedule_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split('\t', 1)
        if parts and parts[0] == SAMPLE:
            return {
                int(match.group(1))
                for token in parts[1].split()
                if (match := re.match(r'(\d+)', token))
            }
    raise ValueError(f'Treatment schedule entry not found for {PLATE} / {SAMPLE}.')


def shade_invitro_treatment(ax: plt.Axes, days: set[int], stop_day: float) -> None:
    valid_days = sorted(day for day in days if day <= stop_day)
    if not valid_days:
        return
    start = previous = valid_days[0]
    for day in valid_days[1:] + [10**9]:
        if day != previous + 1:
            ax.axvspan(
                start,
                min(previous + 1, stop_day),
                ymin=0.5,
                ymax=1.0,
                color=TREATMENT_COLOR,
                linewidth=0,
                zorder=0,
            )
            start = day
        previous = day


def shade_lv_treatment(ax: plt.Axes, trajectory) -> None:
    intervals: list[tuple[float, float]] = []
    for current, following in zip(trajectory.iloc[:-1].itertuples(), trajectory.iloc[1:].itertuples()):
        if not current.treatment_on:
            continue
        start, end = float(current.day), float(following.day)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    for start, end in intervals:
        treatment_span = ax.axvspan(
            start,
            end,
            ymin=0.0,
            ymax=0.5,
            facecolor='none',
            edgecolor=TREATMENT_COLOR,
            hatch=LV_TREATMENT_HATCH,
            lw=0,
            alpha=1.0,
            zorder=0,
        )
        treatment_span.set_hatch_linewidth(LV_TREATMENT_HATCH_LINEWIDTH)


def truncate_one_timepoint_after_progression(table, resistant_column: str, initial_total: float):
    '''Keep the crossing observation and exactly one following observation.'''
    normalized_resistant = table[resistant_column].to_numpy(float) / initial_total
    crossings = (normalized_resistant >= PROGRESSION_THRESHOLD).nonzero()[0]
    if crossings.size:
        return table.iloc[:min(int(crossings[0]) + 2, len(table))].copy()
    return table.copy()


def plot(*, full_x_scale: bool, output_pdf: Path) -> None:
    exp, lv, escape_day, initial_total = load_trajectories()
    if not full_x_scale:
        exp = truncate_one_timepoint_after_progression(exp, 'resistant', initial_total)
        lv = lv[lv['day'] <= float(exp['day'].max())].copy()
    treatment_days = read_invitro_treatment_days()
    fig, ax = ft.new_figure(
        width=AXES_WIDTH_MM, height_mm=AXES_HEIGHT_MM, margins=MARGINS_MM,
        dpi=300, apply_style=False,
    )
    shade_invitro_treatment(ax, treatment_days, float(exp['day'].max()))
    shade_lv_treatment(ax, lv)
    ax.plot(exp['day'], exp['sensitive'] / initial_total, color=SENSITIVE_COLOR,
            linewidth=1.0, marker='o', markersize=2.2, label='In vitro sensitive')
    ax.plot(exp['day'], exp['resistant'] / initial_total, color=RESISTANT_COLOR,
            linewidth=1.0, marker='o', markersize=2.2, label='In vitro resistant')
    ax.plot(lv['day'], lv['lv_sensitive'] / initial_total, color=SENSITIVE_COLOR,
            linewidth=1.0, linestyle=':', label='LV sensitive')
    ax.plot(lv['day'], lv['lv_resistant'] / initial_total, color=RESISTANT_COLOR,
            linewidth=1.0, linestyle=':', label='LV resistant')
    ax.axhline(PROGRESSION_THRESHOLD, color='#555555', linewidth=0.8, linestyle=':', zorder=0)

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Normalized cell count')
    x_axis_max = max(float(exp['day'].max()), float(lv['day'].max())) if full_x_scale else (
        float(escape_day) if escape_day is not None else max(float(exp['day'].max()), float(lv['day'].max()))
    )
    ax.set_xlim(0, x_axis_max)
    ax.set_ylim(0, max(1.05, float(max(
        exp['sensitive'].max(), exp['resistant'].max(), lv['lv_sensitive'].max(), lv['lv_resistant'].max()
    ) / initial_total) * 1.08))
    handles = [
        Line2D([], [], color='#333333', linewidth=1.0, marker='o', markersize=2.2, label='Exp'),
        Line2D([], [], color='#333333', linewidth=1.0, linestyle=':', label='GLV'),
    ]
    ax.legend(handles=handles, frameon=False, ncol=1, loc='upper left', handlelength=1.6,
              columnspacing=0.8, handletextpad=0.35)
    ft.format_axes(ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ft.savefig(fig, output_pdf, dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    existing = [path for path in (OUTPUT_PDF, OUTPUT_FULL_X_PDF) if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            'Output already exists. Set OVERWRITE = True to replace: ' + ', '.join(map(str, existing))
        )
    plot(full_x_scale=False, output_pdf=OUTPUT_PDF)
    print(f'Saved: {OUTPUT_PDF}')
    plot(full_x_scale=True, output_pdf=OUTPUT_FULL_X_PDF)
    print(f'Saved: {OUTPUT_FULL_X_PDF}')
