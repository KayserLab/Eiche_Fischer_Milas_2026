# Supplementary Figure 6: failure-only example selection

Updated 12 November 2026 following the author's request: replace only examples without observed failure, and consider the full dynamics rather than time to failure alone. Exactly 13 panels changed (7 in vitro, 6 in silico). The 37 previously failing examples remain in their original columns. CT is unchanged in both figures.

## Scope and interpretation

All available canonical population files for the five plotted strategies and all matching margin records were inventoried. The census contains 333 records: 120 experimental and 213 simulation records. Canonical filters fully exclude 57 experimental records; 47 simulation records are pilot runs below R10; Dens R23 lacks a valid day-1 baseline. This leaves 63 experimental and 165 simulation trajectories for numerical comparison. Existing manual cutoffs are applied before evaluating failure.

The displayed sample is conditional on observed failure. It cannot represent the frequency of censored/no-failure outcomes in the complete cohort. With 37 panels fixed, this is a constrained representative selection, not a random sample or an unrestricted optimum. Full-cohort records, including censored outcomes and exclusion reasons, remain in the audit.

| Strategy | In-vitro valid / failures | In-silico valid / failures |
|---|---:|---:|
| CT | 13 / 13 | 36 / 28 |
| AT50 | 15 / 12 | 35 / 23 |
| AT100 | 14 / 8 | 36 / 21 |
| LV-Agent | 11 / 5 | 35 / 23 |
| AT100+ | 10 / 8 | 23 / 13 |

## Deliberate comparison of dynamics

Sensitive, resistant and total populations are read afresh from canonical HDF5 datasets. All five curves, including confinement and escaped-resistant counts, use the initial canonical total as their denominator. Failure is the first acquired day with normalized resistant count >= 0.5. Escape is the first positive signed escape margin. Event times are not interpolated.

Candidate sets are evaluated within each source and strategy, with the old observed-failure panels locked. The numerical comparison uses these feature blocks:

- 30%: sensitive, resistant and total trajectory shapes at 21 relative-time positions from baseline through failure.
- 25%: confinement and escape-component trajectory shapes over the same interval.
- 15%: observed minima, maxima and excess total variation (rebounds/cycling beyond net change) for all five curves before failure.
- 10%: all five post-failure curves at 11 relative-time positions through the last valid observation.
- 20%: failure time, escape time, their signed lag, and whether escape was observed.

Time resampling is used only for curve comparison; it does not change plotted acquisition days or event times. Relative-time shapes separate curve form from duration, while the timing block preserves absolute event timing. Each feature is converted to its empirical percentile within the strategy's observed-failure cohort. Missing post-failure data or unobserved escape time receive the neutral 0.5 feature rank; the escape-observed indicator retains censoring information. Post-failure duration differs between records, so that block has limited weight.

The distance is the square root of the weighted mean squared feature-rank difference, averaging features within each block first. Every combination of the required number of replacements is scored using equal weights for (a) energy distance between the five-example empirical distribution and the full observed-failure cohort and (b) mean distance from each cohort member to its nearest displayed example. This balances distributional coverage and representative shapes. Previously failing examples remain fixed. New experimental candidates with unresolved yellow/purple review flags or a documented central hole are vetoed; the canonical retained cohort remains the numerical reference.

The original Comments_on_wells.xlsx values and colour flags were read again. Full-cohort diagnostic sheets were generated, and the selected curves and close alternatives were visually compared. The optimal numerical sets were retained after that review.

## Replacements

| Source | Strategy | Column | Previous example | Replacement | Escape day | Failure day |
|---|---|---:|---|---|---:|---:|
| experimental | AT50 | 5 | Plate_2/B1_AT50-B1 | Plate_3/C4_AT50-C4 | 24 | 45 |
| experimental | AT100 | 4 | Plate_5/A2_AT100-A2 | Plate_2/C4_AT100-C4 | 24 | 36 |
| experimental | AT100 | 5 | Plate_3/D4_AT100-D4 | Plate_5/A4_AT100-A4 | 38 | 59 |
| experimental | LV-Agent | 3 | Plate_7/A2_ATAgent2-A2 | Plate_7/C2_ATAgent2-C2 | 22 | 43 |
| experimental | LV-Agent | 4 | Plate_6/B4_ATAgent2-B4 | Plate_6/B3_ATAgent2-B3 | 54 | 59 |
| experimental | LV-Agent | 5 | Plate_7/C1_ATAgent2-C1 | Plate_7/A4_ATAgent2-A4 | 64 | 61 |
| experimental | AT100+ | 5 | Plate_8/B3_Dens-B3 | Plate_8/D3_Dens-D3 | 52 | 61 |
| simulation | AT50 | 5 | Plate_AT50/R47_AT50-R47 | Plate_AT50/R27_AT50-R27 | 18 | 35 |
| simulation | AT100 | 4 | Plate_AT100/R28_AT100-R28 | Plate_AT100/R42_AT100-R42 | 38 | 54 |
| simulation | AT100 | 5 | Plate_AT100/R29_AT100-R29 | Plate_AT100/R13_AT100-R13 | 67 | 78 |
| simulation | LV-Agent | 4 | Plate_Agent/R18_Agent-R18 | Plate_Agent/R26_Agent-R26 | 119 | 103 |
| simulation | LV-Agent | 5 | Plate_Agent/R36_Agent-R36 | Plate_Agent/R28_Agent-R28 | unobserved | 113 |
| simulation | AT100+ | 5 | Plate_Dens/R43_Dens-R43 | Plate_Dens/R33_Dens-R33 | 87 | 90 |

## Why these dynamics complement the retained panels

- In-vitro AT50, P3 C4: adds repeated sensitive-population recovery followed by strong resistant expansion and loss of sensitive cells. Its TTF is close to existing examples, but the complete post-escape trajectory adds information; the more extreme late TTF of P3 C5 did not provide the best overall set coverage.
- In-vitro AT100, P2 C4 and P5 A4: combine early confinement loss and resistant takeover with a longer multi-cycle course and gradual escape growth. P5 A4 also provides a substantial post-failure trajectory. Together they improve coverage alongside the fixed early, intermediate and late examples.
- In-vitro LV-Agent, P7 C2, P6 B3 and P7 A4: these are the only remaining three failures, so the row now contains all five observed failures in the retained agent cohort. P7 C2 has a peripheral resistant clone and marked sensitive collapse; P6 B3 retains strong sensitive growth/cycling; P7 A4 has early suppression and late resistant expansion. P6 B3 is documented as having a 2D fast-growing sensitive subpopulation. It is retained by the canonical filters but must not be described as a clean generic example. Failure precedes the signed-margin escape event in P7 A4.
- In-vitro AT100+, P8 D3: contributes substantial sensitive cycling, early loss of confinement and a prolonged gradual resistant rise, with failure at day 61 and escape before failure. P8 B2 is a close alternative with earlier failure; the balanced score favours D3.
- In-silico AT50, R27: represents the central pattern of sensitive suppression and rebound, confinement loss, and subsequent resistant takeover. Several nearby runs have similar event times; R27 is preferred by the combined curve descriptors.
- In-silico AT100, R42 and R13: add an earlier, smaller-amplitude cycling course and an intermediate multi-cycle course. R42 and R37 are close alternatives in both timing and curve shape; R42 is favoured with the main weighting.
- In-silico LV-Agent, R26 and R28: R26 supplies prolonged cycling and sensitive decline with failure before escape. R28 supplies sustained sensitive recovery and resistant progression without observed signed-margin escape through day 150, although failure occurs on day 113. This preserves a genuine distinction between the two event definitions. The alternative pair R10/R23 is close when timing is given less weight.
- In-silico AT100+, R33: adds sustained small-amplitude cycling and gradual resistant growth with escape shortly before failure. R31 is a close alternative, but the combined shape and coverage score favours R33.

## Sensitivity and limitations

The exhaustive search was repeated with timing weights of 10% and 30%, reallocating the difference equally to population and margin shapes. Experimental AT50/AT100 and simulated AT50/AT100+ retain the same replacements in both alternatives. Experimental AT100+ switches to P8 B2 at 10%; simulated LV-Agent switches to R10/R23 at 10%; simulated AT100 switches R42 to R37 at 30%. These close alternatives show that the choices are defensible examples rather than uniquely determined representatives. The 20% timing solution is used consistently; weights were not tuned per strategy.

The five experimental LV-Agent failures are the complete observed-failure subset, including the documented P6 B3 caveat. The fixed CT rows cannot be improved by this request, since all their original examples already failed. Recorded resistant-cell washing and manual cutoffs remain in the audit; no post-cutoff failure was used. Simulation and experimental columns are independently selected, not matched replicate pairs.

## Complete current selection

| Source | Strategy | Column | Plate/sample | Escape day | Failure day | Last valid day | Action |
|---|---|---:|---|---:|---:|---:|---|
| experimental | CT | 1 | Plate_4/C5_CT-C5 | 17 | 24 | 43 | kept |
| experimental | CT | 2 | Plate_3/B2_CT-B2 | 19 | 31 | 45 | kept |
| experimental | CT | 3 | Plate_4/C2_CT-C2 | 17 | 36 | 50 | kept |
| experimental | CT | 4 | Plate_4/C1_CT-C1 | 24 | 43 | 50 | kept |
| experimental | CT | 5 | Plate_2/A3_CT-A3 | 26 | 36 | 47 | kept |
| experimental | AT50 | 1 | Plate_2/B6_AT50-B6 | 22 | 36 | 57 | kept |
| experimental | AT50 | 2 | Plate_2/B2_AT50-B2 | 40 | 45 | 50 | kept |
| experimental | AT50 | 3 | Plate_4/D5_AT50-D5 | 40 | 47 | 54 | kept |
| experimental | AT50 | 4 | Plate_4/D6_AT50-D6 | 68 | 73 | 96 | kept |
| experimental | AT50 | 5 | Plate_3/C4_AT50-C4 | 24 | 45 | 78 | replaced |
| experimental | AT100 | 1 | Plate_2/C1_AT100-C1 | 26 | 38 | 73 | kept |
| experimental | AT100 | 2 | Plate_5/A5_AT100-A5 | 50 | 50 | 59 | kept |
| experimental | AT100 | 3 | Plate_1/B6_AT100-B6 | 68 | 64 | 96 | kept |
| experimental | AT100 | 4 | Plate_2/C4_AT100-C4 | 24 | 36 | 40 | replaced |
| experimental | AT100 | 5 | Plate_5/A4_AT100-A4 | 38 | 59 | 92 | replaced |
| experimental | LV-Agent | 1 | Plate_7/C5_ATAgent2-C5 | 15 | 33 | 36 | kept |
| experimental | LV-Agent | 2 | Plate_7/C4_ATAgent2-C4 | 43 | 59 | 68 | kept |
| experimental | LV-Agent | 3 | Plate_7/C2_ATAgent2-C2 | 22 | 43 | 68 | replaced |
| experimental | LV-Agent | 4 | Plate_6/B3_ATAgent2-B3 | 54 | 59 | 61 | replaced |
| experimental | LV-Agent | 5 | Plate_7/A4_ATAgent2-A4 | 64 | 61 | 71 | replaced |
| experimental | AT100+ | 1 | Plate_8/C5_Dens-C5 | 15 | 33 | 38 | kept |
| experimental | AT100+ | 2 | Plate_8/C2_Dens-C2 | 36 | 45 | 54 | kept |
| experimental | AT100+ | 3 | Plate_8/C3_Dens-C3 | 57 | 57 | 64 | kept |
| experimental | AT100+ | 4 | Plate_8/D5_Dens-D5 | 66 | 59 | 66 | kept |
| experimental | AT100+ | 5 | Plate_8/D3_Dens-D3 | 52 | 61 | 66 | replaced |
| simulation | CT | 1 | Plate_CT/R11_CT-R11 | 17 | 23 | 29 | kept |
| simulation | CT | 2 | Plate_CT/R39_CT-R39 | 16 | 25 | 26 | kept |
| simulation | CT | 3 | Plate_CT/R14_CT-R14 | 18 | 26 | 35 | kept |
| simulation | CT | 4 | Plate_CT/R34_CT-R34 | 19 | 28 | 41 | kept |
| simulation | CT | 5 | Plate_CT/R36_CT-R36 | 19 | 30 | 33 | kept |
| simulation | AT50 | 1 | Plate_AT50/R12_AT50-R12 | 18 | 28 | 34 | kept |
| simulation | AT50 | 2 | Plate_AT50/R11_AT50-R11 | 20 | 36 | 45 | kept |
| simulation | AT50 | 3 | Plate_AT50/R19_AT50-R19 | 32 | 42 | 52 | kept |
| simulation | AT50 | 4 | Plate_AT50/R15_AT50-R15 | 45 | 51 | 59 | kept |
| simulation | AT50 | 5 | Plate_AT50/R27_AT50-R27 | 18 | 35 | 41 | replaced |
| simulation | AT100 | 1 | Plate_AT100/R20_AT100-R20 | 41 | 53 | 62 | kept |
| simulation | AT100 | 2 | Plate_AT100/R26_AT100-R26 | 77 | 82 | 83 | kept |
| simulation | AT100 | 3 | Plate_AT100/R34_AT100-R34 | 108 | 109 | 110 | kept |
| simulation | AT100 | 4 | Plate_AT100/R42_AT100-R42 | 38 | 54 | 61 | replaced |
| simulation | AT100 | 5 | Plate_AT100/R13_AT100-R13 | 67 | 78 | 86 | replaced |
| simulation | LV-Agent | 1 | Plate_Agent/R20_Agent-R20 | 86 | 89 | 99 | kept |
| simulation | LV-Agent | 2 | Plate_Agent/R27_Agent-R27 | 105 | 107 | 115 | kept |
| simulation | LV-Agent | 3 | Plate_Agent/R16_Agent-R16 | 133 | 124 | 135 | kept |
| simulation | LV-Agent | 4 | Plate_Agent/R26_Agent-R26 | 119 | 103 | 131 | replaced |
| simulation | LV-Agent | 5 | Plate_Agent/R28_Agent-R28 | unobserved | 113 | 150 | replaced |
| simulation | AT100+ | 1 | Plate_Dens/R37_Dens-R37 | 54 | 67 | 74 | kept |
| simulation | AT100+ | 2 | Plate_Dens/R48_Dens-R48 | 59 | 67 | 74 | kept |
| simulation | AT100+ | 3 | Plate_Dens/R49_Dens-R49 | 84 | 93 | 102 | kept |
| simulation | AT100+ | 4 | Plate_Dens/R24_Dens-R24 | 115 | 108 | 119 | kept |
| simulation | AT100+ | 5 | Plate_Dens/R33_Dens-R33 | 87 | 90 | 99 | replaced |

## Reproduction and artifacts

- Full raw-data census and deterministic selection: `analysis/figure_2/Select_Failing_Escape_Margin_Examples.py`. Subsequent descriptor/selection runs can use `--cached`; the original selection snapshot is preserved and can be supplied with `--baseline`.
- Audit directory: `Data/verification/supp6_failure_reselection_20261112/`. It contains all record dispositions (`cohort_audit.csv`), normalized trajectories (`cohort_values.csv`), dynamic descriptors (`dynamic_features.csv`), the selected replacements (`replacement_selection.csv`), top alternatives and weighting sensitivity (`selection_scores.csv`), and per-cohort diagnostic PNGs.
- The `before/` directory preserves the previous script, notes, values and both PDFs. `original_selection.csv` freezes the 37 locked slots and 13 replaceable slots.
- Plotter: `figures/Suppl_Figures/Escape_Margin_Over_Time_By_Strategy.py`. The plotting constants freeze the reviewed selection; plotting does not silently reselect examples. A guard rejects any selected trajectory without observed failure.
- The fresh cohort values were used to replot via `--replot-from Data/verification/supp6_failure_reselection_20261112/selected_values_for_replot.csv`, avoiding a redundant raw-data scan.
- Final PDFs keep their existing names in `Data/figure_plots/Suppl_Figures/escape_margin_over_time/`. Values and event/selection audits keep their existing names in the corresponding `figure_data` directory.
- Numerical verification: `verification/verify_supp6_failure_examples.py`. It checks 50 observed failures, exactly 13 replacements, every locked plate/sample/column, event times and all five curves against the full census, manual cutoffs, page dimensions, panel counts, embedded fonts and vector-only PDF output.
- Layout remains 5 by 5 panels on 180 by 185 mm pages, with fixed original axes and colours. Display clipping limits remain 0-0.8 for margins and 0-3 for populations; complete values are retained in the exported tables.

Numerical and PDF-structure verification passed on 12 November 2026. Both final PDFs were rendered with Poppler at 200 dpi and visually checked for panel labels, event markers, spacing and page clipping; the intended fixed curve-display limits are preserved.
