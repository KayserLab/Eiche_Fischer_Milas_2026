# Fig. 3 claim audit, 4 November 2026

The commented numerical sentence is **not up to date with the active Fig. 3 workflow**. The direction of the density-growth contrast remains supported, but the original numerical effects, replicate ranges and simulation p-value bound cannot be reused for the current analysis. No manuscript text or production analysis has been changed.

## Claim-by-claim verdict

| Claim | Verdict |
|---|---|
| High local density is associated with reduced relative resistant growth for CT, AT50 and AT100, in both datasets | Supported: all six current paired contrasts are negative and significant after Holm correction. This is an association, not a causal intervention. |
| In-vitro contrast of -0.27 to -0.37 day^-1, n=9-14, all Holm p <=0.00195 | Outdated numerical summary. The current, explicitly defined contrast below is -0.156874 to -0.309749 day^-1, n=13-15. Its largest Holm p is 0.000305176, so the quoted p upper bound remains conservative, but the effect and n ranges do not describe the current result. Even the saved legacy mean analysis gives -0.263051 to -0.342846 day^-1, n=11-14. |
| In-silico contrast of -0.60 to -0.78 day^-1, n=26-32, all p <=1.5e-8 | Reproduces in the legacy escaping-run mean analysis, including after Holm correction. It does not describe the active analysis: the current contrast is -0.359224 to -0.445834 day^-1, n=25-34; the largest raw and Holm p are 2.980232e-8. |
| High density means resistant stasis for every strategy | Too strong as a universal quantitative claim. Reduced growth is supported; no equivalence/stasis test is performed. Current high-bin strategy medians remain positive for AT50 and AT100 in both datasets. |
| Strategies occupy different density ranges with different probability densities | Legacy code provides descriptive, pooled observation-weighted histograms/KDEs. The active Fig. 3 workflow stores counts but does not normalize/compare the strategy occupancy distributions or test differences. The legacy PDF display is disabled. Thus this is not a demonstrated comparison in the current figure. |
| Density-occupancy differences explain distinct escape dynamics | Not established by this code. There is no analysis relating per-well/run density occupancy to measured TTE or escape risk, or testing whether occupancy differences account for strategy differences in escape. Selecting observations before TTE is not such a test. |

## Sources and scope

- Overleaf source: `C:\Users\meiche\Downloads\2D_melanoma_colonies____Adaptive_Therapy__Copy_ (4).zip`, modified 2026-11-04 11:26:19 CET. ZIP SHA256: `fa695105a515ebc7d2a037db318a467ae3817a15790cdf1e62439fcd86b72598`.
- The quoted contrast and occupancy statements are commented at `main.tex:149-150`. The active Fig. 3 caption is at `main.tex:131-137`. The active Supplementary Methods describe the current definition at `SI.tex:100-175`; the old definition is preserved in a comment block afterwards.
- Code base commit: `953d2020da0f0240f9cb4c0dd6a2bbb7c7249cbc`; this audit uses the working-tree code. Pre-existing unrelated untracked files were left alone.
- Current data: `Y:\Members\MaxE\Publications\STEM Paper\Data\figure_data\Figure_3\HDF5_analysis\flux_vs_density\kernel_radius_160um\selection_diagnostics\master_curve_selection_diagnostics_radius_160um.h5`.
- Current HDF5 SHA256: `82c732d6bb6777164d60eaa28ba6129312fd9cb4b175bd07432fde24084602c1`. It passes the current kernel-settings validator, which recognizes this exact audited input. This is not blanket acceptance of stale HDF5 files.
- Numerical results were recalculated from unit-level HDF5 curves and independent-unit legacy NPZ counts/sums. This audit did not regenerate every density field from raw point clouds. It did not copy the old exported inferential results as its calculation.

## Current analysis: definition and actual bins

The active defaults are 160-um Epanechnikov kernels, all time points, relative growth and total density at the first endpoint. See [constants.py](../analysis/figure_3/HDF5_analysis/constants.py), [compute_master_curves.py](../analysis/figure_3/HDF5_analysis/compute_master_curves.py) and [plot_hdf5_master_curves.py](../figures/Figure_3/plot_hdf5_master_curves.py).

At each valid grid-point/interval the outcome is

`g = [rho_res(t1)-rho_res(t0)] / [rho_res(t0)*(t1-t0)]`, in day^-1,

and the density coordinate is `rho_tot(t0)`. Experimental elapsed times are used; simulations use daily intervals. Both endpoints require at least 14 kernel neighbours and finite fields, and starting resistant density must be at least `2.25/(pi*160^2)` cells/um^2. Experimental exclusions/truncations apply; simulation run IDs must exceed 9.

There are 30 equal density bins, with physical width 0.062969813 cells/(1000 um^2). Neither requested density is an exact bin centre. I used the bins **containing** the requested values, with no interpolation, rebinning or extrapolation:

| Target density | Actual interval, cells/(1000 um^2) | Centre | Zero-based index |
|---|---|---|---|
| 0.45 | [0.440788691, 0.503758504) | 0.472273598 | 7 |
| 1.55 | [1.511275512, 1.574245325) | 1.542760419 | 24 |

The bins are the same in experiments and simulations after physical-unit conversion. A well/run needs at least 20 grid-point/interval observations in a bin. Only units valid at **both** bins enter a paired contrast. These are spatial and temporal observations, not independent biological replicates. The whole high-density bin is outside the current plot's x-limit of 1.5, although its source data are available.

Related caption inconsistency: `main.tex:135` quotes a resistant-support threshold of 0.05 cells/(1000 um^2). The current 160-um rule is approximately **0.028 cells/(1000 um^2)**. The value approximately 0.05 follows from the same numerator at 120 um. The active map renderer also passes `MIN_RESISTANT_NUMERATOR/(pi*radius**2)` and the native support mask to the outline, so the caption should be checked against the radius of the assembled panels. No caption edit was made.

**Important estimand distinction:** the current HDF5 stores within-unit medians and quartiles, not within-unit means. To preserve the unit of inference, the audited current effect is

`mean over paired units [median(g in high bin) - median(g in low bin)]`.

It is a mean of paired within-unit median contrasts. It must not be labelled as the original mean-growth contrast. The original within-unit means cannot be recovered exactly from medians, quartiles and counts. Reproducing that estimand under the new selection rules would require extending/recomputing the raw-field aggregation.

The active coloured strategy curves instead show the **median across units** at each bin. Their endpoint difference is a separate descriptive statistic, supplied below. The black curve is a mean across strategy curves; that does not make the underlying per-strategy quantities means.

## Current six contrasts and inference

Each well/run is one paired observation. I reapplied the original inference design to the current within-unit median curves: **one-sided Wilcoxon signed-rank test**, alternative high-minus-low < 0. Holm adjustment covers CT, AT50 and AT100 **separately within each dataset**, not all six tests together. No current production Fig. 3 entry point already performs this contrast inference; these are newly calculated audit results. The Wilcoxon test concerns the location of paired differences, not specifically their arithmetic mean.

All six have no zero differences or tied absolute differences. SciPy 1.17.1 `method='auto'` agrees exactly with `method='exact'` for every result.

| Dataset | Strategy | n low | n high | n both | Mean paired contrast, day^-1 | Raw p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| In vitro | CT | 13 | 13 | 13 | -0.156874 | 0.0001220703125 | 0.000244140625 |
| In vitro | AT50 | 15 | 15 | 15 | -0.309749 | 0.000030517578125 | 0.000091552734375 |
| In vitro | AT100 | 14 | 14 | 14 | -0.220762 | 0.00030517578125 | 0.00030517578125 |
| In silico | CT | 36 | 26 | 26 | -0.445834 | 1.490116119e-8 | 2.980232239e-8 |
| In silico | AT50 | 34 | 35 | 34 | -0.359224 | 5.820766091e-11 | 1.746229827e-10 |
| In silico | AT100 | 25 | 35 | 25 | -0.396380 | 2.980232239e-8 | 2.980232239e-8 |

In vitro n denotes wells; in silico n denotes runs. The total available cohorts before bin support are 13/15/14 wells and 36/35/36 runs for CT/AT50/AT100. All pairs are negative except two AT100 experimental wells. The signed-rank statistics are 0 for five groups and 3 for experimental AT100.

For direct comparison with the current strategy curves, the median-across-unit values are:

| Dataset | Strategy | Low-bin median, day^-1 | High-bin median, day^-1 | High minus low, day^-1 |
|---|---|---:|---:|---:|
| In vitro | CT | 0.173267 | 0.008733 | -0.164534 |
| In vitro | AT50 | 0.194162 | 0.061090 | -0.133072 |
| In vitro | AT100 | 0.237494 | 0.160066 | -0.077428 |
| In silico | CT | 0.324156 | -0.063632 | -0.387787 |
| In silico | AT50 | 0.363971 | 0.117844 | -0.246127 |
| In silico | AT100 | 0.368555 | 0.188941 | -0.179614 |

These use all available units separately at each bin, as the plotter does. They are not the paired mean effects tested in the preceding table; those p-values must not be attached to these differences as if they were the same estimand.

## Reconstruction of the original mean workflow

The legacy workflow uses total density averaged over endpoints, symmetric relative growth `(rho_res1-rho_res0)/[mean(rho_res0,rho_res1)*elapsed_days]`, pre-escape intervals, 0.10-wide bins, a minimum of 100 observations per unit/bin, and resistant density at both endpoints above the legacy 1e-8 threshold. It also has different spatial preprocessing/support rules. It is not the active t0-density, all-timepoints analysis.

The actual legacy bins are [0.40,0.50) and [1.50,1.60), centred exactly on 0.45 and 1.55 cells/(1000 um^2). The legacy effect is the mean across paired units of their high-bin **mean** minus low-bin **mean**, in day^-1. The test and within-dataset Holm family are the same as above.

Reconstructed from `aggregation_cache_figure2_masks_before_escape_all_both_endpoints_xbin_0p1`, using the `before_escape__both_gt_1e-08` condition:

| Dataset | Strategy | n low | n high | n both | Mean contrast, day^-1 | Raw p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| In vitro | CT | 13 | 13 | 13 | -0.278617 | 0.0001220703125 | 0.0003662109375 |
| In vitro | AT50 | 14 | 15 | 14 | -0.263051 | 0.0001220703125 | 0.0003662109375 |
| In vitro | AT100 | 11 | 13 | 11 | -0.342846 | 0.00048828125 | 0.00048828125 |
| In silico | CT | 45 | 26 | 26 | -0.603438 | 1.490116119e-8 | 1.490116119e-8 |
| In silico | AT50 | 41 | 34 | 32 | -0.778561 | 4.656612873e-10 | 1.396983862e-9 |
| In silico | AT100 | 33 | 29 | 29 | -0.775850 | 1.862645149e-9 | 3.725290298e-9 |

These reproduce every effect, paired n, raw p and adjusted p in the saved legacy `plots_before_escape_gate_matrix/no_gating/both_t0_t1_gt_1e-08/mean_curve_contrast_statistics.tsv` export. The manuscript's simulation numbers identify this historical escaping-run cohort, which also includes run IDs excluded by the current >9 rule. The experimental effect and n ranges quoted in the comment do not match this export.

The separate legacy `_all_runs` caches give simulation effects CT -0.603438 (n=26), AT50 -0.792337 (n=33), AT100 -0.795855 (n=31); experimental results are unchanged. Those reproduce the corresponding `_all_runs` export too. Full raw/adjusted p-values and marginal n for that variant are in the audit TSV.

## Density occupancy and escape: what the code actually does

1. **Active workflow:** `compute_master_curves.py` stores `count_rel` per well/run and density bin. `plot_hdf5_master_curves.py` loads the growth medians and displays median/IQR growth curves. It does not compute normalized occupancy probabilities, per-unit time fractions in density ranges, pairwise strategy occupancy tests or an occupancy-to-escape model. A saved count is not itself a comparison of how often strategies visit density ranges.
2. **Legacy descriptive comparison:** `Figure_3_Legacy/plot_master_curve_sim_exp.py:790` implements `kde_from_binned_counts`, a unit-area Gaussian KDE based on pooled density-bin observation counts, with bandwidth 0.10 cells/(1000 um^2). It can draw CT, AT50 and AT100 distributions. These describe fractions of retained grid-point/interval observations; they are not equal-weighted well/run probabilities or elapsed-time-weighted residence times. Its current `SHOW_DENSITY_PDF` flag is false. Legacy AT50/AT100 joint contours likewise provide descriptive marginals, not inferential occupancy comparisons.
3. **Escape use:** pre-TTE variants select observations using the escape day. The legacy low-density diagnostic records days before escape and summarizes selected low-density pixels. Another old trajectory selector defines an 'irreversible escape' pattern using density/growth thresholds themselves. None of these tests whether measured density-occupancy differences predict independently measured spatial escape or explain differences among the three strategies. The Fig. 4 density-versus-total-burden analysis is a different question; its escape-phase selection is not an occupancy-to-escape analysis either.
4. **Manuscript references:** the commented occupancy sentence points to Fig. 3e,g, which are representative maps/trajectories in the current caption. They do not show strategy density PDFs. The current Fig. 3 caption and active Supplementary Methods describe growth curves, with no occupancy comparison.

The defensible conclusion from the current analysis is a negative local density-growth association across all strategies. The proposed density-occupancy explanation of escape remains a mechanistic interpretation that this analysis does not directly verify.

## Reproduction and audit files

Script: [verification/audit_figure3_claims.py](../verification/audit_figure3_claims.py).

Run it using the project's configured Python interpreter from the repository root. It writes only audit outputs under `tmp/fig3_claim_audit` and reads the laboratory source data.

- [contrast_summary.tsv](../tmp/fig3_claim_audit/contrast_summary.tsv): all 18 current/legacy combinations, exact bins, marginal and paired n, effects and p-values.
- [unit_level_contrasts.tsv](../tmp/fig3_claim_audit/unit_level_contrasts.tsv): each well/run ID, both bin counts, both outcomes and paired contrast, including noncontributing units.
- [manifest.json](../tmp/fig3_claim_audit/manifest.json): current input hash, versions, effect and correction definitions.
- [archive_provenance.json](../tmp/fig3_claim_audit/archive_provenance.json): Overleaf archive identity/hash and code commit.

Verification passed: data match the active plotter's loader, finite current bins agree with the 20-observation rule, physical bin centres agree with the plotter, automatic Wilcoxon p-values equal exact p-values, and the 12 reconstructed legacy results match the original exports to numerical precision.
