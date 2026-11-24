# Matched in-vitro / in-silico TTF and TTE

Run from the repository root after authorizing access to the configured external
data root:

```powershell
.\.venv\Scripts\python.exe figures\Suppl_Figures\TTF_TTE_Matched_InVitro_DPD.py
```

The script reads the canonical combined simulation metadata, prepared population
HDF5s, experimental population HDF5s, and combined escape-margin table. It does
not change those inputs. By default, the PDF goes to
`figure_plots/Suppl_Figures/ttf_tte_matched_analysis/` under the configured
data root (the laboratory share). Tables go to
`figure_data/Suppl_Figures/ttf_tte_matched_analysis/ttf_tte_matched/` under the
same data root. Use `--output-plot-dir` to override the PDF destination and
`--output-dir` to override the table destination.

- `TTF_TTE_Matched_InVitro_DPD.pdf`: two panels with editable vector artwork.
- `matching_audit.csv`: each simulation's recorded seed, treatment strategy,
  experimental and simulation IDs, input paths, and match/exclusion status.
- `matched_event_times.csv`: exact pairs, event times or censoring bounds,
  event-observed flags, and reasons for unusable event data.
- `agreement_summary.csv`: counts and descriptive agreement statistics, computed
  only for pairs with both events observed.

A pair must match the complete experimental sample ID in the simulation's
`initial_condition` metadata and the normalized treatment strategy. Equal well
coordinates alone, nearest-well matches, or equal seeds under different
treatments are insufficient. There are no substitute matches. Unknown metadata,
ambiguous experimental IDs, and conflicting run metadata are never silently
resolved. Runs below 10 are artificial initial conditions and are omitted.

TTF is the first observation where resistant count is at least half of the
initial total. TTE is the first observation with positive signed escape margin.
Both are measured on acquisition days (day 1 is the baseline). This follows the
shared `Event_Time_Analysis.py` calendar; the older supplementary unpaired TTF
script instead uses elapsed days starting at zero.

The existing experimental exclusion table applies to each paired experimental
sample. Numeric cutoffs limit observations to the last valid acquisition day.
Each panel includes only pairs where that event was observed in both the
experiment and simulation. Censored pairs remain in the exported audit table
with an exclusion status and are omitted from the figure. Simulation population data require
the day-1 baseline and uninterrupted daily observations through failure or
censoring. Missing margin data are audited and omitted from that panel.

Colors identify treatment strategies consistently in both panels. Each point
represents one experimental well paired to one simulation run; if multiple
runs have the same seed, their experimental coordinate repeats. Agreement
statistics describe observed matched event times.
Pearson r is annotated separately for each strategy and is included in the
summary CSV, using the same observed pairs shown in the scatter plot. It is
not reported for fewer than three pairs or for constant event times. There is
also an explicitly labelled All points coefficient pooling the observed pairs
across treatments. Both within-strategy and pooled annotations report unadjusted
two-sided Pearson p-values from scipy.stats.pearsonr. Pooled association can
reflect between-strategy differences; correlation is not absolute agreement.

A synthetic check of exact matching, event thresholds, day conventions,
censoring, missing simulation days, and conflicting metadata can run without
accessing the study data:

```powershell
.\.venv\Scripts\python.exe verification\check_matched_event_times.py
```
