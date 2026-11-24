#!/usr/bin/env python3
'''Correlate eta_tot(t0) with area-average total density at t0.

For every experimental timepoint t0 in CT, AT50, and AT100, this script:

* calculates the fixed-grid total and resistant density fields at t0;
* retains grid locations that are valid at t0 and whose resistant density at
  t0 meets the configured Epanechnikov-kernel threshold; and
* spatially averages rho_tot(t0) over those locations.

The density is cached for all valid, experiment-filtered timepoints. Each row
is labelled as before or after escape, allowing plots to select phases without
recalculating density fields.
'''
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ANALYSIS_DIR = PROJECT_ROOT / 'analysis' / 'figure_3' / 'HDF5_analysis'
for path in (PROJECT_ROOT, ANALYSIS_DIR, SCRIPT_DIR):
    if os.fspath(path) not in sys.path:
        sys.path.insert(0, os.fspath(path))

# Read the selected radius before importing the HDF5 modules.
KERNEL_RADII_UM = (120, 160)
DEFAULT_KERNEL_RADIUS_UM = 120
ANALYSIS_STRATEGIES = ('CT', 'AT50', 'AT100')
KERNEL_RADIUS_UM = float(DEFAULT_KERNEL_RADIUS_UM)
if __name__ == '__main__':
    kernel_parser = argparse.ArgumentParser(add_help=False)
    kernel_parser.add_argument(
        '--kernel-radius-um', type=int, choices=KERNEL_RADII_UM,
        default=DEFAULT_KERNEL_RADIUS_UM,
    )
    KERNEL_RADIUS_UM = float(kernel_parser.parse_known_args()[0].kernel_radius_um)
MIN_NEIGHBORS = 14
MIN_RESISTANT_NUMERATOR = 2.25
KERNEL_QUERY_CHUNK_SIZE = 100
os.environ['HDF5_KERNEL_RADIUS_UM'] = f'{KERNEL_RADIUS_UM:g}'
os.environ['HDF5_KERNEL_QUERY_CHUNK_SIZE'] = str(KERNEL_QUERY_CHUNK_SIZE)
os.environ['MAJA_KERNEL_RADIUS_UM'] = f'{KERNEL_RADIUS_UM:g}'
os.environ['MAJA_KERNEL_QUERY_CHUNK_SIZE'] = str(KERNEL_QUERY_CHUNK_SIZE)

from analysis.figure_3.HDF5_analysis import (
    compute_flux_vs_density_experiment as hdf5_density,
)
from analysis.figure_4 import density_burden_helpers as base

actual_min_resistant_numerator = (
    float(hdf5_density.MIN_RES_DENSITY)
    * float(hdf5_density.prolif_bin_area)
)
if not np.isclose(hdf5_density.prolif_radius_um, KERNEL_RADIUS_UM):
    raise RuntimeError(
        f'Expected a {KERNEL_RADIUS_UM:g}-um HDF5 kernel, got '
        f'{hdf5_density.prolif_radius_um:g} um.'
    )
if int(hdf5_density.prolif_min_neighbors) != MIN_NEIGHBORS:
    raise RuntimeError(
        f'Expected prolif_min_neighbors={MIN_NEIGHBORS}, got '
        f'{hdf5_density.prolif_min_neighbors}.'
    )
if not np.isclose(
    actual_min_resistant_numerator,
    MIN_RESISTANT_NUMERATOR,
):
    raise RuntimeError(
        f'Expected the resistant-support numerator '
        f'{MIN_RESISTANT_NUMERATOR:g}, got '
        f'{actual_min_resistant_numerator:g}.'
    )
kernel_module = sys.modules[hdf5_density._kernel_density_field.__module__]
actual_query_chunk_size = int(kernel_module.KERNEL_QUERY_CHUNK_SIZE)
if actual_query_chunk_size != KERNEL_QUERY_CHUNK_SIZE:
    raise RuntimeError(
        f'Expected KD-tree query chunks of {KERNEL_QUERY_CHUNK_SIZE:,} grid '
        f'points, got {actual_query_chunk_size:,}.'
    )
DENSITY_COLUMN = 'area_average_total_density_cells_per_um2'
DENSITY_DEFINITION = (
    f'hdf5_figure3_t0_snapshot_area_average_total_density_kernel{KERNEL_RADIUS_UM:g}um_'
    'minneighbors14_t0_resistant_numerator2p25_all_valid_timepoints'
)
PHASE_DEFINITION = 'before_if_timepoint_precedes_escape'
OUTPUT_PDF = (
    base.OUTPUT_DIR
    / f'rhoTot_vs_etaTot_before_k{KERNEL_RADIUS_UM:g}_n14_r2p25.pdf'
)
OUTPUT_DATA_CSV = (
    base.DATA_OUTPUT_DIR
    / f'rhoTot_vs_etaTot_k{KERNEL_RADIUS_UM:g}_n14_r2p25.csv'
)
OUTPUT_STATS_CSV = (
    base.DATA_OUTPUT_DIR
    / f'rhoTot_vs_etaTot_stats_k{KERNEL_RADIUS_UM:g}_n14_r2p25.csv'
)
DENSITY_CHECKPOINT_CSV = OUTPUT_DATA_CSV.with_name(
    f'{OUTPUT_DATA_CSV.stem}_density_checkpoint.csv'
)
Y_LIMITS = (0.0, 0.0023)
AXES_WIDTH_MM = 25.0
AXES_HEIGHT_MM = 30.0
MARGINS_MM = (21.0, 3.0, 13.0, 4.0)


def _requested_days(counts: pd.DataFrame) -> dict[tuple[str, str], set[float]]:
    '''Return the filtered Results days needed for every colony.'''
    requested: dict[tuple[str, str], set[float]] = {}
    for (plate, sample), group in counts.groupby(['plate', 'sample'], sort=False):
        days = pd.to_numeric(group['day'], errors='coerce').to_numpy(float)
        requested[(str(plate), str(sample))] = {
            float(day) for day in days if np.isfinite(day)
        }
    return requested


def _hdf5_paths() -> dict[tuple[str, str], Path]:
    '''Index the filtered experimental HDF5 HDF5 inputs by plate and sample.'''
    paths: dict[tuple[str, str], Path] = {}
    for strategy_runs in hdf5_density.discover_runs().values():
        for raw_path in strategy_runs.values():
            path = Path(raw_path)
            plate = hdf5_density._plate_name(path)
            sample, _rule = hdf5_density._sample_and_rule(path)
            key = (str(plate), str(sample))
            if key in paths:
                raise ValueError(
                    f'Duplicate HDF5 HDF5 inputs for {plate}/{sample}: '
                    f'{paths[key]} and {path}'
                )
            paths[key] = path
    return paths


def _figure3_snapshot_density_summary(
    *,
    total: np.ndarray,
    resistant: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int] | None:
    '''Return the area-average total density for one snapshot.'''
    supported = (
        valid
        & np.isfinite(total)
        & np.isfinite(resistant)
        & (resistant >= hdf5_density.MIN_RES_DENSITY)
    )
    if not np.any(supported):
        return None
    values = total[supported] / float(hdf5_density.px_to_micron) ** 2
    ddof = 1 if values.size > 1 else 0
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=ddof)),
        'n_pixels': int(np.count_nonzero(supported)),
    }


def _write_density_checkpoint(rows: list[dict[str, object]]) -> None:
    checkpoint = pd.DataFrame(rows)
    # A shared-drive CSV can briefly be held by Excel, Explorer, or indexing.
    # Keep the calculation running: the next successful checkpoint includes all
    # rows accumulated in memory.
    for attempt in range(5):
        temporary = DENSITY_CHECKPOINT_CSV.with_name(
            f'{DENSITY_CHECKPOINT_CSV.stem}.{uuid.uuid4().hex}.tmp'
        )
        try:
            checkpoint.to_csv(temporary, index=False)
            os.replace(temporary, DENSITY_CHECKPOINT_CSV)
            return
        except PermissionError:
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(1)
    print(
        '[checkpoint] shared-drive file is locked; will retry at the next '
        'completed snapshot.'
    )


def load_hdf5_snapshot_area_average_density(
    counts: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    '''Calculate the experimental area-average density at each requested t0.'''
    requested = _requested_days(counts)
    paths = _hdf5_paths()
    rows: list[dict[str, object]] = []
    completed_days: set[tuple[str, str, float]] = set()

    if force:
        DENSITY_CHECKPOINT_CSV.unlink(missing_ok=True)
    elif DENSITY_CHECKPOINT_CSV.is_file():
        checkpoint = pd.read_csv(DENSITY_CHECKPOINT_CSV)
        checkpoint_required = {
            'plate', 'sample', 'day', DENSITY_COLUMN,
            'density_definition',
        }
        if (
            checkpoint_required.issubset(checkpoint.columns)
            and checkpoint['density_definition'].astype(str).eq(
                DENSITY_DEFINITION
            ).all()
        ):
            rows = checkpoint.to_dict(orient='records')
            completed_days = {
                (str(plate), str(sample), float(day))
                for plate, sample, day in checkpoint[
                    ['plate', 'sample', 'day']
                ].drop_duplicates().itertuples(index=False, name=None)
            }
            print(
                f'Resuming from {len(completed_days)} completed experimental '
                f'snapshots: {DENSITY_CHECKPOINT_CSV}'
            )
        else:
            print(
                f'[warn] Ignoring incompatible density checkpoint: '
                f'{DENSITY_CHECKPOINT_CSV}'
            )

    available = sorted(set(requested).intersection(paths))
    missing = sorted(set(requested).difference(paths))
    for plate, sample in missing:
        print(f'[warn] HDF5 HDF5 input missing: {plate}/{sample}')

    print(
        f'[HDF5 t0 density] experimental HDF5 only; '
        f'KD-tree query chunks={KERNEL_QUERY_CHUNK_SIZE:,}',
        flush=True,
    )

    density_conversion = 1.0 / float(hdf5_density.px_to_micron) ** 2
    resistant_threshold_physical = (
        float(hdf5_density.MIN_RES_DENSITY) * density_conversion
    )
    grid_spacing_um = (
        float(hdf5_density._SIM_GRID_SPACING)
        * float(hdf5_density.px_to_micron)
    )

    for well_index, key in enumerate(available, start=1):
        plate, sample = key
        wanted_days = requested[key]
        pending_days = {
            day for day in wanted_days
            if (plate, sample, day) not in completed_days
        }
        if not pending_days:
            print(
                f'[HDF5 t0 density] well {well_index}/{len(available)} '
                f'already cached: {plate}/{sample}',
                flush=True,
            )
            continue
        path = paths[key]
        last_count_day = max(pending_days)

        with h5py.File(path, 'r') as handle:
            grid_points, grid_shape = hdf5_density._fixed_well_grid(handle)
            days = [
                (day, day_key)
                for day, day_key in hdf5_density._sorted_days(handle)
                if float(day) <= last_count_day
            ]
            for day_index, (day, day_key) in enumerate(days, start=1):
                current_day = float(day)
                if current_day not in pending_days:
                    continue
                print(
                    f'[HDF5 t0 density] well '
                    f'{well_index}/{len(available)}, '
                    f'day {day_index}/{len(days)}: '
                    f'{plate}/{sample} d{day}',
                    flush=True,
                )
                total, _susceptible, resistant, valid = (
                    hdf5_density.local_densities(
                        handle, day_key, grid_points, grid_shape
                    )
                )
                summary = _figure3_snapshot_density_summary(
                    total=total,
                    resistant=resistant,
                    valid=valid,
                )
                if summary is not None:
                    rows.append({
                        'plate': plate,
                        'sample': sample,
                        'timepoint': f'{current_day:g}',
                        'day': current_day,
                        DENSITY_COLUMN: float(summary['mean']),
                        'area_average_total_density_std_cells_per_um2': (
                            float(summary['std'])
                        ),
                        'density_pixels': int(summary['n_pixels']),
                        'density_grid_pixels': int(total.size),
                        'density_grid_spacing_um': grid_spacing_um,
                        (
                            'resistant_support_threshold_cells_per_um2'
                        ): resistant_threshold_physical,
                        'density_definition': DENSITY_DEFINITION,
                    })
                    completed_days.add((plate, sample, current_day))
                    _write_density_checkpoint(rows)
                    print(
                        f'[checkpoint] saved {plate}/{sample} d{current_day:g}',
                        flush=True,
                    )

    if not rows:
        raise RuntimeError(
            'No t0 snapshot-density values were calculated.'
        )
    density = pd.DataFrame(rows)
    return (
        density.sort_values(['plate', 'sample', 'day'])
        .drop_duplicates(['plate', 'sample', 'day'], keep='last')
        .reset_index(drop=True)
    )


def build_analysis_table(*, force: bool = False) -> pd.DataFrame:
    '''Join snapshot density at t0 to eta_tot(t0) and escape timing.'''
    counts = base.load_sensitive_counts()
    counts = counts.loc[counts['strategy'].str.upper().isin(ANALYSIS_STRATEGIES)]
    baseline = base.build_total_baseline_table(counts)
    escape = base.load_escape_days()
    counts_with_escape = counts.merge(
        escape[['plate', 'sample']],
        how='inner',
        on=['plate', 'sample'],
    )
    density = load_hdf5_snapshot_area_average_density(
        counts_with_escape,
        force=force,
    )
    merged = density.merge(
        counts[
            [
                'plate', 'sample', 'strategy', 'day',
                'sensitive_cells', 'total_cells',
            ]
        ],
        how='inner',
        on=['plate', 'sample', 'day'],
    ).merge(
        baseline[['plate', 'sample', 'baseline_total']],
        how='inner',
        on=['plate', 'sample'],
    ).merge(
        escape,
        how='inner',
        on=['plate', 'sample'],
    )
    for column in (
        'sensitive_cells', 'total_cells', 'baseline_total',
        DENSITY_COLUMN,
    ):
        merged[column] = pd.to_numeric(merged[column], errors='coerce')
    merged = merged.dropna(
        subset=[
            'total_cells', 'baseline_total', DENSITY_COLUMN,
            'day', 'escape_day',
        ]
    )
    merged = merged[merged['baseline_total'] > 0].copy()
    merged['sensitive_cells_normalized'] = (
        merged['sensitive_cells'] / merged['baseline_total']
    )
    merged['total_cells_normalized'] = (
        merged['total_cells'] / merged['baseline_total']
    )
    merged['phase'] = np.where(
        merged['day'] < merged['escape_day'],
        'before',
        'after',
    )
    merged['escape_definition'] = base.ESCAPE_DEFINITION
    merged['phase_definition'] = PHASE_DEFINITION
    merged['eta_total_timepoint'] = 'snapshot_t0'
    if merged.empty:
        raise RuntimeError(
            'No rows remained after merging snapshot density, total counts, '
            'and escape timing.'
        )
    return merged.sort_values(
        ['phase', 'plate', 'sample', 'day']
    ).reset_index(drop=True)


def load_or_build_analysis_table(*, force: bool = False) -> pd.DataFrame:
    '''Reuse a method-tagged cache unless a rebuild was requested.'''
    required = {
        'plate', 'sample', 'strategy', 'day',
        'total_cells_normalized', DENSITY_COLUMN, 'phase',
        'escape_definition', 'phase_definition', 'density_definition',
        'density_pixels',
    }
    if not force:
        for cache_path in (OUTPUT_DATA_CSV,):
            if not cache_path.is_file():
                continue
            cached = pd.read_csv(cache_path)
            current = (
                required.issubset(cached.columns)
                and cached['escape_definition'].astype(str).eq(
                    base.ESCAPE_DEFINITION
                ).all()
                and cached['phase_definition'].astype(str).eq(
                    PHASE_DEFINITION
                ).all()
                and cached['density_definition'].astype(str).eq(
                    DENSITY_DEFINITION
                ).all()
                and not cached['strategy'].astype(str).str.upper().isin(
                    base.EXCLUDED_STRATEGIES
                ).any()
            )
            if current:
                selected = cached.loc[cached['strategy'].str.upper().isin(ANALYSIS_STRATEGIES)].copy()
                if not selected.empty:
                    print(f'Loaded {len(selected)} cached CT, AT50, and AT100 datapoints: {cache_path}')
                    return selected
            if cache_path == OUTPUT_DATA_CSV:
                print(
                    f'[warn] eta_tot datapoint cache is outdated; '
                    f'rebuilding {OUTPUT_DATA_CSV}'
                )

    result = build_analysis_table(force=force)
    result.to_csv(OUTPUT_DATA_CSV, index=False)
    print(f'Built and cached eta_tot datapoints: {OUTPUT_DATA_CSV}')
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot cached t0 density against normalized total count.'
    )
    parser.add_argument(
        '--kernel-radius-um', type=int, choices=KERNEL_RADII_UM,
        default=DEFAULT_KERNEL_RADIUS_UM,
        help=f'Kernel radius in micrometres (default: {DEFAULT_KERNEL_RADIUS_UM}).',
    )
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Discard the t0 density checkpoint and rebuild the analysis.',
    )
    parser.add_argument(
        '--plot-phases',
        nargs='+',
        choices=('before', 'after'),
        default=('before',),
        help='Cached phases to draw; defaults to before.',
    )
    parser.add_argument(
        '--plot-strategies',
        nargs='+',
        type=str.upper, choices=ANALYSIS_STRATEGIES,
        help='Optional subset of CT, AT50, and AT100 (default: use all three).',
    )
    return parser.parse_args()


def output_pdf_for_selection(
    phases: tuple[str, ...],
    strategies: tuple[str, ...],
) -> Path:
    if phases == ('before',) and not strategies:
        return OUTPUT_PDF
    phase_token = '_'.join(phases)
    strategy_token = '' if not strategies else '_' + '_'.join(strategies)
    return base.OUTPUT_DIR / (
        f'rhoTot_vs_etaTot_{phase_token}{strategy_token}_k{KERNEL_RADIUS_UM:g}_n14_r2p25.pdf'
    )


def main() -> None:
    arguments = parse_arguments()
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_phases = tuple(arguments.plot_phases)
    plot_strategies = tuple(arguments.plot_strategies or ())
    output_pdf = output_pdf_for_selection(plot_phases, plot_strategies)
    data = load_or_build_analysis_table(force=arguments.recompute)
    plot_data = data
    if plot_strategies:
        plot_data = data[
            data['strategy'].astype(str).str.upper().isin(plot_strategies)
        ].copy()
        if plot_data.empty:
            available = sorted(data['strategy'].dropna().astype(str).unique())
            raise ValueError(
                f'No cached datapoints match {plot_strategies}; '
                f'available strategies are {available}.'
            )
        print(
            'Plotting pooled strategies: ' + ', '.join(plot_strategies)
            + f' ({len(plot_data)} cached datapoints)'
        )

    if arguments.recompute or plot_strategies:
        summary = base.phase_statistics(
            plot_data,
            'total_cells_normalized',
            DENSITY_COLUMN,
        )
        if arguments.recompute and not plot_strategies:
            summary.to_csv(OUTPUT_STATS_CSV, index=False)
            print(f'Computed and cached eta_tot statistics: {OUTPUT_STATS_CSV}')
    else:
        summary = base.load_or_compute_statistics(
            plot_data,
            'total_cells_normalized',
            OUTPUT_STATS_CSV,
            DENSITY_COLUMN,
        )
    base.plot(
        plot_data,
        summary,
        x_column='total_cells_normalized',
        x_label=r'Global cell count, $\eta_{\mathrm{tot}}$',
        output_pdf=output_pdf,
        phases=plot_phases,
        axes_width_mm=AXES_WIDTH_MM,
        axes_height_mm=AXES_HEIGHT_MM,
        margins_mm=MARGINS_MM,
        y_limits=Y_LIMITS,
        y_label=(
            'Local density, '
            + base.WHOLE_WELL_DENSITY_LABEL.replace(' $[', '\n$[', 1)
        ),
        phase_colors={'before': '#000000', 'after': '#d62728'},
        y_column=DENSITY_COLUMN,
        x_limits=(0.0, 1.6),
        show_regression=False,
        y_tick_exponent=-3,
        y_major_ticks=(0.001, 0.002),
        mean_curve_bins=15,
        mean_curve_label='Mean',
        x_major_ticks=(0.0, 1.0),
        x_minor_ticks=(0.5, 1.5),
    )
    print(f'Saved: {output_pdf}')
    print(f'Saved statistics: {OUTPUT_STATS_CSV}')


if __name__ == '__main__':
    main()
