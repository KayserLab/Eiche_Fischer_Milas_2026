"""Read-only audit of the commented Fig. 3 contrasts (no manuscript edits).

Run from the repository root. Outputs go to tmp/fig3_claim_audit by default.
The current HDF5 stores within-unit medians, not means. Its contrast is
therefore explicitly labelled as a mean of paired within-unit medians.
Legacy mean contrasts are reconstructed independently from per-unit counts
and sums, never copied from the previously exported statistics table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from shared.data_paths import DATA_ROOT
from figures.Figure_3 import plot_hdf5_master_curves as current
from analysis.figure_3.HDF5_analysis.constants import (
    AREA_SCALE, prolif_n_rho_bins, prolif_rho_max,
    is_non_artificial_run, validate_kernel_settings,
)

STRATEGIES = ('CT', 'AT50', 'AT100')


def summarize(units, workflow, dataset, strategy, low_bin, high_bin):
    lo = np.array([u['low_value'] for u in units], dtype=float)
    hi = np.array([u['high_value'] for u in units], dtype=float)
    paired = np.isfinite(lo) & np.isfinite(hi)
    differences = hi[paired] - lo[paired]
    result = wilcoxon(differences, alternative='less', method='auto') if len(differences) else None
    if len(differences):
        exact = wilcoxon(differences, alternative='less', method='exact')
        assert result.pvalue == exact.pvalue
    return dict(
        workflow=workflow, dataset=dataset, strategy=strategy,
        low_bin_left=low_bin[0], low_bin_right=low_bin[1],
        high_bin_left=high_bin[0], high_bin_right=high_bin[1],
        low_bin_center=np.mean(low_bin), high_bin_center=np.mean(high_bin),
        n_available=len(units), n_low=int(np.isfinite(lo).sum()),
        n_high=int(np.isfinite(hi).sum()), n_paired=int(paired.sum()),
        paired_low_mean=float(lo[paired].mean()),
        paired_high_mean=float(hi[paired].mean()),
        mean_paired_contrast=float(differences.mean()),
        median_paired_contrast=float(np.median(differences)),
        across_unit_low_median=float(np.nanmedian(lo)),
        across_unit_high_median=float(np.nanmedian(hi)),
        difference_across_unit_medians=float(np.nanmedian(hi)-np.nanmedian(lo)),
        negative_pairs=int((differences < 0).sum()),
        positive_pairs=int((differences > 0).sum()),
        zero_pairs=int((differences == 0).sum()),
        tied_absolute_differences=len(differences)-len(np.unique(np.abs(differences))),
        wilcoxon_statistic=float(result.statistic) if result else np.nan,
        p_raw=float(result.pvalue) if result else np.nan,
        effect_units='day^-1',
    )


def audit_current(output, kernel_radius_um=160.0):
    path = current.default_selection_input(kernel_radius_um)
    records, unit_records = [], []
    with h5py.File(path, 'r') as f:
        validate_kernel_settings(f.attrs, kernel_radius_um, source=path)
        for dataset in ('experiment', 'simulation'):
            loaded = current.load_first_endpoint_run_curves(
                path, dataset, 'all_timepoints', 'relative', kernel_radius_um)
            native_edges = np.linspace(0, prolif_rho_max, prolif_n_rho_bins + 1)
            if dataset == 'experiment':
                native_edges /= AREA_SCALE
            edges = native_edges * current.density_scale(dataset)
            indices = np.searchsorted(edges, [0.45, 1.55], side='right')-1
            il, ih = indices
            assert np.allclose(current.density_axis(dataset), (edges[:-1]+edges[1:])/2)
            print('CURRENT BINS', dataset, indices.tolist(), edges[il:il+2], edges[ih:ih+2])
            for strategy in STRATEGIES:
                g = f[dataset][strategy]
                names = g['run_names'].asstr()[:]
                curve = np.asarray(g['all_timepoints/t0_density/median_rel'])
                counts = np.asarray(g['all_timepoints/t0_density/count_rel'])
                keep = np.array([dataset != 'simulation' or is_non_artificial_run(n) for n in names])
                assert np.all(np.isfinite(curve) == (counts >= 20))
                assert np.allclose(curve[keep], loaded[strategy], equal_nan=True)
                units = []
                for name, values, count in zip(names[keep], curve[keep], counts[keep]):
                    row = dict(workflow='current_t0_all_timepoints_within_unit_median',
                               dataset=dataset, strategy=strategy, unit=name,
                               low_value=float(values[il]), high_value=float(values[ih]),
                               n_low_observations=int(count[il]), n_high_observations=int(count[ih]))
                    row['paired_contrast'] = row['high_value']-row['low_value']
                    units.append(row)
                unit_records.extend(units)
                records.append(summarize(units, units[0]['workflow'], dataset, strategy,
                                         edges[il:il+2], edges[ih:ih+2]))
    output['current_source'] = str(path)
    with path.open('rb') as stream:
        output['current_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return records, unit_records


def validate_legacy_exports(results):
    """Cross-check the independent reconstruction against original exports."""
    root = DATA_ROOT/'figure_data/Figure_3/master_curve_160'
    for suffix, cohort in (('', 'escaping_runs'), ('_all_runs', 'all_runs')):
        path = root/('plots_before_escape_gate_matrix'+suffix)/'no_gating/both_t0_t1_gt_1e-08/mean_curve_contrast_statistics.tsv'
        saved = pd.read_csv(path, sep='\t')
        subset = results[results.workflow == 'legacy_before_escape_symmetric_mean_'+cohort]
        for row in subset.itertuples():
            ds = 'in_vitro' if row.dataset == 'experiment' else 'in_silico'
            old = saved[(saved.dataset == ds) & (saved.strategy == row.strategy)].iloc[0]
            assert row.n_paired == old.n_independent_units
            for a, b in ((row.mean_paired_contrast, old.mean_contrast_day_minus_1),
                         (row.p_raw, old.wilcoxon_one_sided_p),
                         (row.p_holm, old.wilcoxon_one_sided_p_holm)):
                assert np.isclose(a, b, rtol=1e-12, atol=1e-16), (row.strategy, a, b)
    print('CHECKED: all 12 legacy effects, paired counts, raw p and Holm p reproduce saved exports')


def audit_legacy():
    base = DATA_ROOT/'intermediate_analysis_data/combined/figure_3/master_curve_160'
    records, unit_records = [], []
    for suffix in ('', '_all_runs'):
        root = base/('aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p1'+suffix)
        workflow = 'legacy_before_escape_symmetric_mean'+('_all_runs' if suffix else '_escaping_runs')
        for dataset, folder in (('experiment','in_vitro_wells'), ('simulation','in_silico_runs')):
            groups = {s: [] for s in STRATEGIES}
            for path in sorted((root/folder).rglob('*.npz')):
                with np.load(path, allow_pickle=False) as d:
                    labels = [str(s) for s in d['strategies']]
                    meta = json.loads(str(d['metadata_json'][0]))
                    strategy = str(meta.get('strategy', labels[0]))
                    if strategy not in groups:
                        continue
                    slugs = [str(s) for s in d['condition_slugs']]
                    ci = slugs.index('before_escape__both_gt_1e-08')
                    edges = d['x_edges']
                    centers = (edges[:-1]+edges[1:])/2
                    il = int(np.flatnonzero(np.isclose(centers, .45))[0])
                    ih = int(np.flatnonzero(np.isclose(centers, 1.55))[0])
                    count, total = d[f'c{ci}_count'][0], d[f'c{ci}_total'][0]
                    row = dict(workflow=workflow, dataset=dataset, strategy=strategy,
                               unit=str(path.relative_to(root/folder)),
                               low_value=float(total[il]/count[il]) if count[il] >= 100 else np.nan,
                               high_value=float(total[ih]/count[ih]) if count[ih] >= 100 else np.nan,
                               n_low_observations=int(count[il]), n_high_observations=int(count[ih]))
                    row['paired_contrast'] = row['high_value']-row['low_value']
                    groups[strategy].append(row)
            for strategy, units in groups.items():
                unit_records.extend(units)
                records.append(summarize(units, workflow, dataset, strategy,
                                         edges[il:il+2], edges[ih:ih+2]))
    return records, unit_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO/'tmp/fig3_claim_audit')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(data_root=str(DATA_ROOT), numpy=np.__version__, scipy=scipy.__version__,
                    current_effect='Mean across paired wells/runs of high-bin minus low-bin within-unit medians',
                    legacy_effect='Mean across paired wells/runs of high-bin minus low-bin within-unit means',
                    test='One-sided paired Wilcoxon signed-rank; alternative=less; method=auto',
                    correction='Holm across CT, AT50, AT100 separately within dataset and workflow')
    records, units = audit_current(manifest)
    legacy_records, legacy_units = audit_legacy()
    records.extend(legacy_records)
    units.extend(legacy_units)
    results = pd.DataFrame(records)
    for _, indices in results.groupby(['workflow','dataset']).groups.items():
        p = results.loc[indices, 'p_raw'].to_numpy()
        order = np.argsort(p)
        adjusted = np.empty(len(p))
        adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
        results.loc[indices, 'p_holm'] = adjusted
    validate_legacy_exports(results)
    results.to_csv(args.output_dir/'contrast_summary.tsv',sep='\t',index=False)
    pd.DataFrame(units).to_csv(args.output_dir/'unit_level_contrasts.tsv',sep='\t',index=False)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(results[['workflow','dataset','strategy','n_low','n_high','n_paired',
                   'paired_low_mean','paired_high_mean','mean_paired_contrast',
                   'difference_across_unit_medians','p_raw','p_holm']].to_string(index=False))
    print('OUTPUT', args.output_dir)


if __name__ == '__main__':
    main()
