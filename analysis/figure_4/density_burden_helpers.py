"""Burden loading, statistics and plotting helpers for current Figure 4a.

The HDF5 entry point supplies density values; this module does not load TIFFs.
"""
from __future__ import annotations
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats
from shared import figtools as ft
from shared import experiment_filters
from shared.data_paths import EXP_RAW_DATA, figure_data_dir, figure_plot_dir, intermediate_dir

EXPERIMENT_ROOT = EXP_RAW_DATA
ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
RESULT_RE = re.compile(
    r'^Results_G418_a2a4_(?P<well>[A-H]\d{1,2})_(?P<strategy>[^/\\]+?)-(?P=well)\.txt$',
    re.IGNORECASE,
)
DAY_RE = re.compile(r'_d(?P<day>\d{1,3})\b', re.IGNORECASE)
ESCAPE_DEFINITION = 'first_signed_escape_margin_gt_zero'

# Final Figure 4 assets are maintained with the core-figure seed files.
OUTPUT_DIR = figure_plot_dir("Figure_4", "whole_well_density", create=False)
DATA_OUTPUT_DIR = figure_data_dir("Figure_4", "whole_well_density", create=False)
EXCLUDED_STRATEGIES = {'ATAGENT1'}

AXES_SIZE_MM = 20.0
MARGINS_MM = (18.0, 3.0, 13.0, 4.0)
N_BOOTSTRAPS = 10_000
RNG_SEED = 20260809
X_LIMITS = (0.0, 2.5)
Y_LIMITS = (0.0, 0.0005)
WHOLE_WELL_DENSITY_LABEL = (
    "⟨" r'$\rho_{\mathrm{tot}}$' "⟩" r'$_A$'
    r' $[\mathrm{cells}\,\mathrm{µm}^{-2}]$'
)
DENSITY_COLUMN = 'whole_well_density_cells_per_um2'
PHASE_STYLE = {
    'before': {'color': '#1f77b4', 'label': 'Before escape'},
    'after': {'color': '#d62728', 'label': 'After escape'},
}


def load_sensitive_counts() -> pd.DataFrame:
    '''Read susceptible/resistant trajectories directly from Plate_*/Results.'''
    rows: list[dict[str, object]] = []
    for plate_dir in sorted(EXPERIMENT_ROOT.glob('Plate_*')):
        for path in sorted((plate_dir / 'Results').glob('Results_G418_a2a4_*.txt')):
            match = RESULT_RE.match(path.name)
            if not match:
                continue
            well = match.group('well').upper()
            strategy = match.group('strategy')
            sample = f'{well}_{strategy}-{well}'
            for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
                parts = line.split()
                day_match = DAY_RE.search(parts[0]) if len(parts) >= 3 else None
                if day_match is None:
                    continue
                try:
                    sensitive = int(parts[1])
                    resistant = int(parts[2])
                except ValueError:
                    continue
                rows.append({
                    'plate': plate_dir.name,
                    'sample': sample,
                    'strategy': strategy,
                    'day': int(day_match.group('day')),
                    'sensitive_cells': sensitive,
                    'resistant_cells': resistant,
                    'total_cells': sensitive + resistant,
                })
    if not rows:
        raise RuntimeError(f'No Results trajectories found below {EXPERIMENT_ROOT}.')
    data = pd.DataFrame(rows)
    data = experiment_filters.apply_sample_exclusions(
        data, column='sample', cutoff_column='day', strict=False,
    )
    data = data[~data['strategy'].astype(str).str.upper().isin(EXCLUDED_STRATEGIES)].copy()
    return data.sort_values(['plate', 'sample', 'day']).drop_duplicates(
        ['plate', 'sample', 'day'], keep='last',
    )



def build_total_baseline_table(counts: pd.DataFrame) -> pd.DataFrame:
    '''Use day 1 total count, falling back to the earliest available day.'''
    data = counts.copy()
    data['day'] = pd.to_numeric(data['day'], errors='coerce')
    data['total_cells'] = pd.to_numeric(data['total_cells'], errors='coerce')
    data = data.dropna(subset=['day', 'total_cells'])
    day_one = (
        data[data['day'] == 1]
        .sort_values(['plate', 'sample', 'day'])
        .groupby(['plate', 'sample'], as_index=False)['total_cells'].first()
        .rename(columns={'total_cells': 'baseline_total_day1'})
    )
    earliest = (
        data.sort_values(['plate', 'sample', 'day'])
        .groupby(['plate', 'sample'], as_index=False).first()
        [['plate', 'sample', 'day', 'total_cells']]
        .rename(columns={'day': 'baseline_day_fallback', 'total_cells': 'baseline_total_fallback'})
    )
    baseline = earliest.merge(day_one, how='left', on=['plate', 'sample'])
    baseline['baseline_total'] = baseline['baseline_total_day1'].fillna(
        baseline['baseline_total_fallback']
    )
    baseline['baseline_day'] = np.where(
        baseline['baseline_total_day1'].notna(), 1, baseline['baseline_day_fallback'],
    )
    baseline = baseline[
        np.isfinite(pd.to_numeric(baseline['baseline_total'], errors='coerce'))
        & (baseline['baseline_total'] > 0)
    ]
    return baseline[['plate', 'sample', 'baseline_total', 'baseline_day']]



def load_escape_days() -> pd.DataFrame:
    '''Derive experimental escape day from the canonical signed-margin table.'''
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
        & (data['signed_escape_margin'] > 0)
        & data['day'].notna()
    ]
    return (
        escaped.groupby(['plate', 'sample'], as_index=False)['day'].min()
        .rename(columns={'day': 'escape_day'})
    )



def cluster_bootstrap_r_interval(
    data: pd.DataFrame,
    x_column: str,
    rng: np.random.Generator,
    y_column: str = DENSITY_COLUMN,
) -> tuple[float, float]:
    """95% percentile interval for Pearson's r, resampling whole colonies.

    All valid time points from a sampled colony are retained together.  This
    preserves within-colony dependence instead of treating repeated time
    points as independent biological replicates.
    """
    required = ['plate', 'sample', x_column, y_column]
    valid = data.loc[:, required].dropna().copy()
    valid["cluster"] = valid["plate"].astype(str) + "::" + valid["sample"].astype(str)
    clusters = [group for _, group in valid.groupby("cluster", sort=False)]
    if len(clusters) < 2:
        return np.nan, np.nan

    values: list[float] = []
    for _ in range(N_BOOTSTRAPS):
        sampled = [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))]
        resample = pd.concat(sampled, ignore_index=True)
        x = resample[x_column].to_numpy(float)
        y = resample[y_column].to_numpy(float)
        if x.size >= 3 and np.std(x) and np.std(y):
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)



def regression_band(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = stats.linregress(x, y)
    x_grid = np.linspace(np.min(x), np.max(x), 200)
    y_fit = fit.intercept + fit.slope * x_grid
    residual = y - (fit.intercept + fit.slope * x)
    residual_se = np.sqrt(np.sum(residual ** 2) / (len(x) - 2))
    se_mean = residual_se * np.sqrt(1 / len(x) + (x_grid - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    return x_grid, y_fit, stats.t.ppf(0.975, len(x) - 2) * se_mean



def phase_statistics(
    df: pd.DataFrame,
    x_column: str,
    y_column: str = DENSITY_COLUMN,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []
    for phase in ('before', 'after'):
        sub = df[df['phase'] == phase]
        sub = sub.dropna(subset=['plate', 'sample', x_column, y_column]).copy()
        x = sub[x_column].to_numpy(float)
        y = sub[y_column].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        sub = sub.loc[valid].copy()
        x, y = x[valid], y[valid]
        n_colonies = int(sub[['plate', 'sample']].drop_duplicates().shape[0])
        if len(x) < 3 or not np.std(x) or not np.std(y):
            rows.append({'phase': phase, 'n_observations': len(x), 'n_colonies': n_colonies,
                         'pearson_r': np.nan, 'cluster_bootstrap_ci_low': np.nan,
                         'cluster_bootstrap_ci_high': np.nan})
            continue
        r_value = float(stats.pearsonr(x, y).statistic)
        bootstrap_low, bootstrap_high = cluster_bootstrap_r_interval(
            sub, x_column, rng, y_column
        )
        rows.append({'phase': phase, 'n_observations': len(x), 'n_colonies': n_colonies,
                     'pearson_r': r_value, 'cluster_bootstrap_ci_low': bootstrap_low,
                     'cluster_bootstrap_ci_high': bootstrap_high})
    return pd.DataFrame(rows)



def load_or_compute_statistics(
    df: pd.DataFrame,
    x_column: str,
    output_csv: Path,
    y_column: str = DENSITY_COLUMN,
) -> pd.DataFrame:
    """Reuse cached statistics so a plotting-only rerun stays fast."""
    required_columns = {'phase', 'n_observations', 'n_colonies', 'pearson_r',
                        'cluster_bootstrap_ci_low', 'cluster_bootstrap_ci_high'}
    if output_csv.is_file():
        cached = pd.read_csv(output_csv)
        expected_counts = {}
        for phase in ('before', 'after'):
            sub = df[df['phase'] == phase].dropna(
                subset=['plate', 'sample', x_column, y_column]
            ).copy()
            finite = np.isfinite(sub[x_column].to_numpy(float)) & np.isfinite(
                sub[y_column].to_numpy(float)
            )
            sub = sub.loc[finite]
            expected_counts[phase] = (
                len(sub), int(sub[['plate', 'sample']].drop_duplicates().shape[0])
            )
        cache_matches_data = required_columns.issubset(cached.columns) and all(
            len(row := cached[cached['phase'] == phase]) == 1
            and int(row.iloc[0]['n_observations']) == counts[0]
            and int(row.iloc[0]['n_colonies']) == counts[1]
            for phase, counts in expected_counts.items()
        )
        if cache_matches_data:
            print(f'Loaded cached statistics: {output_csv}')
            return cached
        print(f'[warn] Statistics cache does not match the filtered datapoints; recomputing it.')

    summary = phase_statistics(df, x_column, y_column)
    summary.to_csv(output_csv, index=False)
    print(f'Computed and cached statistics: {output_csv}')
    return summary



def add_statistics_text(ax: plt.Axes, summary: pd.DataFrame) -> None:
    text = []
    '''
    for row in summary.itertuples(index=False):
        text.append(f'{row.phase.capitalize()}: r={row.pearson_r:.2f}, '
                    f'colonies={int(row.n_colonies)}, observations={int(row.n_observations)}\n'
                    f'Cluster-bootstrap 95% [{row.cluster_bootstrap_ci_low:.2f}, '
                    f'{row.cluster_bootstrap_ci_high:.2f}]')
    ax.text(0.02, 0.98, '\n'.join(text), transform=ax.transAxes, va='top', ha='left', fontsize=5.0)
    '''


def plot(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    output_pdf: Path,
    title: str | None = None,
    phases: tuple[str, ...] = ('before', 'after'),
    axes_size_mm: float = AXES_SIZE_MM,
    axes_width_mm: float | None = None,
    axes_height_mm: float | None = None,
    margins_mm: tuple[float, float, float, float] = MARGINS_MM,
    y_limits: tuple[float, float] = Y_LIMITS,
    y_label: str = WHOLE_WELL_DENSITY_LABEL,
    phase_colors: dict[str, str] | None = None,
    output_svg: Path | None = None,
    y_column: str = DENSITY_COLUMN,
    x_limits: tuple[float, float] | None = None,
    show_regression: bool = True,
    y_tick_exponent: int = -4,
    mean_curve_bins: int | None = None,
    mean_curve_label: str | None = None,
    x_major_ticks: tuple[float, ...] | None = None,
    x_minor_ticks: tuple[float, ...] | None = None,
    y_major_ticks: tuple[float, ...] | None = None,
    y_tick_label_rotation: float | None = None,
) -> None:
    fig, ax = ft.new_subplots(
        ax_width_mm=axes_width_mm if axes_width_mm is not None else axes_size_mm,
        ax_height_mm=axes_height_mm if axes_height_mm is not None else axes_size_mm,
        margins=margins_mm,
        dpi=300,
        apply_style=True,
    )
    for phase in phases:
        style = PHASE_STYLE[phase].copy()
        if phase_colors and phase in phase_colors:
            style['color'] = phase_colors[phase]
        sub = df[df['phase'] == phase]
        x = sub[x_column].to_numpy(float)
        y = sub[y_column].to_numpy(float)
        ax.scatter(x, y, s=5.0, alpha=0.14, color=style['color'], edgecolors='none', rasterized=True, zorder=1)
        if show_regression and len(x) >= 3 and np.std(x) and np.std(y):
            x_grid, y_fit, half_band = regression_band(x, y)
            ax.fill_between(x_grid, y_fit - half_band, y_fit + half_band, color=style['color'], alpha=0.18, linewidth=0, zorder=0.8)
            ax.plot(x_grid, y_fit, color=style['color'], linewidth=1.2, label=style['label'], zorder=1.2)
        if mean_curve_bins is not None:
            limits = X_LIMITS if x_limits is None else x_limits
            finite = np.isfinite(x) & np.isfinite(y)
            in_range = finite & (x >= limits[0]) & (x <= limits[1])
            x_selected = x[in_range]
            y_selected = y[in_range]
            edges = np.linspace(limits[0], limits[1], mean_curve_bins + 1)
            indices = np.digitize(x_selected, edges[1:-1], right=False)
            mean_x, mean_y = [], []
            for bin_index in range(mean_curve_bins):
                values = indices == bin_index
                if np.any(values):
                    mean_x.append(float(np.mean(x_selected[values])))
                    mean_y.append(float(np.mean(y_selected[values])))
            if mean_x:
                ax.plot(
                    mean_x,
                    mean_y,
                    color=style['color'],
                    linewidth=1.2,
                    label=mean_curve_label if phase == phases[0] else None,
                    zorder=1.3,
                )
    add_statistics_text(ax, summary[summary['phase'].isin(phases)])
    ax.set_xlim(*(X_LIMITS if x_limits is None else x_limits))
    if x_major_ticks is not None:
        ax.set_xticks(x_major_ticks)
    if x_minor_ticks is not None:
        ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_ylim(*y_limits)
    if y_major_ticks is not None:
        ax.set_yticks(y_major_ticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if "⟨" in y_label or "⟩" in y_label:
        # Arial renders the scientific MathText; Segoe supplies ⟨ and ⟩.
        ax.yaxis.label.set_fontfamily('Segoe UI Symbol')
    for spine in ax.spines.values():
        spine.set_zorder(4)
    y_tick_scale = 10.0 ** y_tick_exponent
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                '0' if np.isclose(value, 0.0)
                else (
                    rf'${value / y_tick_scale:g} \times '
                    rf'10^{{{y_tick_exponent}}}$'
                )
            )
        )
    )
    if y_tick_label_rotation is not None:
        for label in ax.get_yticklabels():
            label.set_rotation(y_tick_label_rotation)
            label.set_horizontalalignment('right')
            label.set_verticalalignment('center')
    if title:
        ax.set_title(title, loc='right', fontsize=6.5)
    if mean_curve_label is not None:
        ax.legend(
            frameon=False, loc='upper left', fontsize=5.0, handlelength=1.5,
            handletextpad=0.4, borderaxespad=0.4, borderpad=0.0,
        )
    ft.savefig(fig, output_pdf, dpi=300, bbox_inches=None, pad_inches=0)
    if output_svg is not None:
        fig.savefig(output_svg, format='svg', transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


