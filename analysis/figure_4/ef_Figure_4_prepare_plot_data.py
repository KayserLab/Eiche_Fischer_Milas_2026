'''Prepare the plot-ready tables used by the Figure 4 survival and burden panels.
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================

All inputs and outputs resolve beneath ``EICHE_FISCHER_DATA_ROOT`` through the
imported plotting modules. Run this preparation before the Kaplan-Meier and
cumulative-treatment/burden plotting scripts.
'''

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: raw in-vitro Results/*.txt, simulation HDF5/manifest, and combined
# simulation results configured in the two imported Figure 4 scripts.
# Scripts: e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico.py and
# f_Figure_4_Cumulative_Treatment_average_burden_in_vitro_and_in_silico.py.
# Outputs: canonical Figure 4 KM and treatment/burden CSV tables.

from figures.Figure_4 import f_Figure_4_Cumulative_Treatment_average_burden_in_vitro_and_in_silico as burden, \
    e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico as km

# Set True to refresh the Figure 4 prepared CSV tables from PyCharm.
OVERWRITE = True


def prepare_in_vitro_km() -> None:
    result_files = km.find_result_files(km.ROOTS)
    if not result_files:
        raise RuntimeError(f'No in-vitro Results_*.txt files found under {km.ROOTS}')
    include = [strategy.upper() for strategy in km.INCLUDE_STRATEGIES]
    _groups, truncated = km.build_groups(result_files, include, True)
    _groups, untruncated = km.build_groups(result_files, include, False)
    if not truncated or not untruncated:
        raise RuntimeError('No usable in-vitro Kaplan-Meier rows were produced.')
    km.write_summary_csvs(
        truncated, km.CSV_PATH, km.FAIL_CSV_PATH
    )
    km.write_summary_csvs(
        untruncated, km.CSV_PATH_NO_TRUNC,
        km.FAIL_CSV_PATH_NO_TRUNC,
    )


def prepare_in_vitro_burden() -> None:
    rows = burden.build_invitro_per_replicate_rows()
    summary = burden.summarize_strategy_rows(rows)
    if not rows or not summary:
        raise RuntimeError('No usable in-vitro treatment/burden rows were produced.')
    burden.save_csv(rows, burden.VITRO_REPLICATE_CSV)
    burden.save_csv(summary, burden.VITRO_SUMMARY_CSV)
    print(f'[OK] Saved {len(rows)} in-vitro burden rows: {burden.VITRO_REPLICATE_CSV}')
    print(f'[OK] Saved {len(summary)} in-vitro strategy rows: {burden.VITRO_SUMMARY_CSV}')


def prepare_in_silico() -> None:
    _groups, km_rows = km.build_insilico_groups()
    km.write_insilico_summary(km_rows)
    print(f'[OK] Saved {len(km_rows)} in-silico KM rows: {km.SILICO_SUMMARY_CSV}')
    rows = burden.build_insilico_per_replicate_rows()
    summary = burden.summarize_strategy_rows(rows)
    if not rows or not summary:
        raise RuntimeError('No usable in-silico treatment/burden rows were produced.')
    burden.save_csv(rows, burden.SILICO_REPLICATE_CSV)
    burden.save_csv(summary, burden.SILICO_SUMMARY_CSV)
    print(f'[OK] Saved {len(rows)} in-silico burden rows: {burden.SILICO_REPLICATE_CSV}')
    print(f'[OK] Saved {len(summary)} in-silico strategy rows: {burden.SILICO_SUMMARY_CSV}')


def main() -> None:
    outputs = [
        km.CSV_PATH, km.FAIL_CSV_PATH, km.CSV_PATH_NO_TRUNC, km.FAIL_CSV_PATH_NO_TRUNC,
        burden.VITRO_REPLICATE_CSV, burden.VITRO_SUMMARY_CSV,
        km.SILICO_SUMMARY_CSV, burden.SILICO_REPLICATE_CSV, burden.SILICO_SUMMARY_CSV,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            "Output already exists. Set OVERWRITE = True to replace: "
            + ", ".join(map(str, existing))
        )
    prepare_in_vitro_km()
    prepare_in_vitro_burden()
    prepare_in_silico()


if __name__ == '__main__':
    main()
