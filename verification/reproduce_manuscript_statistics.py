"""Export the current executable manuscript tests and their exact input rows.

Calls the same maintained functions as the live figure scripts. It contains no
historical sample-size assertions, backup paths or fixed manuscript p-values.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.data_paths import DATA_ROOT
from analysis.figure_2.Event_Time_Analysis import (
    ESCAPE_MARGIN_CSV, derive_tte, load_escape_margins, load_population_counts,
    prepare_progression_statistics,
)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    out = DATA_ROOT / 'Statistics'
    inputs = out / 'Test_Inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'running', 'data_root': '.', 'analyses': [],
        'scope': 'current executable figure tests; individual replicate inputs and additional plot statistics indexed',
        'figure4a_kernel_radius_um': 120, 'cohort_sizes': 'derived separately for each test'}

    def save(name, rows, implementation, **details):
        rows = clean(rows)
        target = out / f'{name}.json'
        target.write_text(json.dumps({'implementation': implementation, **details, 'results': rows},
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
        manifest['analyses'].append({'id': name, 'results': target.relative_to(DATA_ROOT).as_posix(),
            'implementation': implementation, 'tests': len(rows), **details})
        print(f'Statistics: {name}, {len(rows)} results', flush=True)

    tte = load('release_stats_tte', 'figures/Figure_2/b_TTE CT & AT100.py')
    ttf = load('release_stats_ttf', 'figures/Figure_2/c_TTF CT & AT100.py')
    cor = load('release_stats_correlation', 'figures/Figure_2/d_plot_ttf_vs_tte.py')
    km = load('release_stats_km', 'figures/Figure_4/e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico.py')
    burden = load('release_stats_burden', 'figures/Figure_4/f_Figure_4_Cumulative_Treatment_average_burden_in_vitro_and_in_silico.py')
    strategies = tuple(tte.STRATEGIES_KEEP)
    margins = load_escape_margins(ESCAPE_MARGIN_CSV)
    exp_tte = tte.prepare_tte_statistics(derive_tte(margins, dataset_type='experimental',
                                                   strategies=strategies, apply_truncation=True))
    sim_tte = tte.load_simulation_tte(margins)
    exp_ttf = prepare_progression_statistics(load_population_counts(dataset_type='experimental',
                                                strategies=strategies), apply_truncation=True)
    sim_ttf = ttf.load_simulation_ttp()
    for name, frame in [('figure2_tte_experimental', exp_tte), ('figure2_tte_simulation', sim_tte),
                        ('figure2_ttf_experimental', exp_ttf), ('figure2_ttf_simulation', sim_ttf)]:
        frame.to_csv(inputs / f'{name}.csv', index=False)
    save('figure2_tte', tte.tte_contrasts(exp_tte, sim_tte),
         'figures/Figure_2/b_TTE CT & AT100.py:tte_contrasts',
         bootstrap_seed=tte.BOOTSTRAP_SEED, bootstrap_repeats=tte.BOOTSTRAP_REPEATS,
         experimental_unit='independent colony', simulation_unit='matched initial-condition block')
    save('figure2_ttf', ttf.ttf_contrasts(exp_ttf, sim_ttf),
         'figures/Figure_2/c_TTF CT & AT100.py:ttf_contrasts',
         bootstrap_seed=ttf.BOOTSTRAP_SEED, bootstrap_repeats=ttf.BOOTSTRAP_REPEATS,
         experimental_unit='independent colony', simulation_unit='matched initial-condition block')
    correlation = pd.read_csv(cor.DATA_OUTPUT_DIR / cor.OUTPUT_CSV)
    correlation.to_csv(inputs / 'figure2_correlation_pairs.csv', index=False)
    save('figure2_correlation', cor.manuscript_correlations(correlation),
         'figures/Figure_2/d_plot_ttf_vs_tte.py:manuscript_correlations',
         eligible_pairs='both TTE and TTF observed, as in the plotted correlation table')
    vitro, _ = km.load_prepared_groups(km.CSV_PATH)
    silico, _ = km.load_insilico_groups()
    for source, groups, stratified in [('experimental', vitro, False), ('simulation', silico, True)]:
        pd.DataFrame([dict(row, plotted_strategy=strategy) for strategy in km.INCLUDE_STRATEGIES
                      for row in km._rows_for_strategy(groups, strategy)]).to_csv(
                          inputs / f'figure4_survival_{source}.csv', index=False)
        save(f'figure4_survival_{source}', km.pairwise_logrank_rows(groups, km.INCLUDE_STRATEGIES, stratified),
             'figures/Figure_4/e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico.py:pairwise_logrank_rows',
             multiple_comparisons='Bonferroni across the available strategy pairs within this dataset',
             pairing='shared initial-condition strata' if stratified else 'independent colonies')
    for source, path, label in [('experimental', burden.VITRO_REPLICATE_CSV, 'Exp'),
                                ('simulation', burden.SILICO_REPLICATE_CSV, 'Sim')]:
        pd.read_csv(path).to_csv(inputs / f'figure4_burden_{source}.csv', index=False)
        results = burden.atagent_vs_atplus_statistics(path, label)
        if not results:
            raise ValueError(f'Missing eligible ATagent/AT+ burden contrasts: {source}')
        save(f'figure4_burden_{source}', results,
             'figures/Figure_4/f_Figure_4_Cumulative_Treatment_average_burden_in_vitro_and_in_silico.py:atagent_vs_atplus_statistics',
             bootstrap_seed=burden.BOOTSTRAP_SEED, bootstrap_repeats=burden.BOOTSTRAP_REPEATS,
             multiple_comparisons='unadjusted, matching the manuscript implementation')
    figure3 = load('release_stats_figure3', 'verification/audit_figure3_claims.py')
    for radius in (120, 160):
        description = {}
        rows, units = figure3.audit_current(description, kernel_radius_um=radius)
        result = pd.DataFrame(rows)
        for _, index in result.groupby(['workflow', 'dataset']).groups.items():
            p = result.loc[index, 'p_raw'].to_numpy()
            order = np.argsort(p)
            corrected = np.empty(len(p))
            corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
            result.loc[index, 'p_holm'] = corrected
        pd.DataFrame(units).to_csv(inputs / f'figure3_density_growth_paired_units_{radius}um.csv', index=False)
        save(f'figure3_density_growth_{radius}um', result.to_dict('records'),
             'verification/audit_figure3_claims.py:audit_current', kernel_radius_um=radius,
             effect='mean across paired wells/runs of high-bin minus low-bin within-unit medians',
             test='one-sided paired Wilcoxon signed-rank, alternative=less, method=auto',
             multiple_comparisons='Holm across CT/AT50/AT100 within each dataset and radius variant')
    # Index statistics produced by maintained plotting/analysis code, including
    # Figure 4a density-phase comparisons and all supplementary test tables.
    additional = []
    for path in sorted((DATA_ROOT / 'figure_data').rglob('*.csv')):
        header = pd.read_csv(path, nrows=0)
        columns = list(header.columns)
        if any(c.lower() in {'p', 'p_raw', 'p_value', 'pvalue', 'p_adjusted', 'p_holm',
                             'p_value_bonferroni', 'pearson_r', 'spearman_r'} for c in columns):
            with path.open('rb') as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            additional.append({'path': path.relative_to(DATA_ROOT).as_posix(),
                               'sha256': digest, 'columns': columns})
    manifest.update(status='computed', additional_generated_test_tables=additional)
    (out / 'statistics_manifest.json').write_text(json.dumps(clean(manifest), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Exported {len(manifest["analyses"])} current statistical analyses and {len(additional)} additional test tables.', flush=True)


if __name__ == '__main__':
    main()
