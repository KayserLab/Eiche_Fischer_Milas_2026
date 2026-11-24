"""Original drawing entry points used by the compact reviewer data release."""

# Calls receive frozen figure-source values; original main programs are not run.
DRAW = {
 'fig1_assay': ('_plot_comparison', '_plot_extra_mixed_intervals'),
 'fig1_trajectories': ('plot_single_sample', 'plot_stacked_samples'),
 'fig2_tte': ('plot_tte_boxplot',),
 'fig2_ttf': ('plot_boxplot',),
 'fig2_ttf_vs_tte': ('plot',),
 'fig2_examples': ('make_plot', 'make_split_signed_plot', 'make_separate_margin_plot'),
 'fig2_examples_side_by_side': ('make_separate_margin_plot',),
 'fig3_master_120': ('draw_plot',),
 'fig3_master_160': ('draw_plot',),
 'fig3_cell_size': ('plot_main_figure', 'plot_verification_figure'),
 'fig3_maps': ('draw_at50_two_timepoint_comparison',),
 'fig3_trajectories': ('plot_trajectory',),
 'fig4_lv_example': ('plot_trajectory',),
 'fig4_agent_examples': ('save_all_trajectories', 'save_escape_margins'),
 'fig4_policy': ('plot_figure',),
 'fig4_burden': ('_plot_one', '_plot_both_side_by_side', '_plot_both_overlaid'),
 'suppl_initial_margin': ('plot_histogram',),
 'suppl_escape_time': ('plot_examples',),
 'suppl_escape_burden': ('plot',),
 'suppl_resistant_maps': ('draw_resistant_density_maps',),
 'suppl_all_ttf': ('plot',),
 'suppl_km_atplus': ('plot',),
 'matched_events': ('plot_pairs',),
 'simulation_events': ('plot_matched_and_correlations',),
 'suppl_rank_agreement': ('plot_ranks', 'plot_agreement'),
 'suppl_initial_conditions': ('plot_ct_reference',),
 'suppl_mean_ranks': ('plot_scores',),
 'suppl_day1_counts': ('plot_counts',),
}

# These few renderers have drawing code inside main. Replace only their input
# readers with frozen values; their calculations and drawing code stay intact.
MAIN_INPUTS = {
 'fig1_legend': (),
 'fig2_ratchet': ('load_data',),
 'fig4_density': ('load_or_build_analysis_table',),
 'fig4_survival': ('load_prepared_groups', 'load_insilico_groups'),
 'fig4_portrait': ('load_decisions',),
 'suppl_relative_growth': ('load_tables',),
}

def plotting_steps(steps):
    return [s for s in steps if s.name in DRAW or s.name in MAIN_INPUTS or s.name == 'fig1_movie_frames']
