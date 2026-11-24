# Nature Communications submission checklist

This repository checklist complements, but does not replace, the journal's forms and manuscript instructions. Nature Communications follows the Nature Portfolio policies for reporting and the availability of data, code, materials, and protocols. The journal also provides a dedicated code/software checklist, a bio/life-sciences reporting summary, and a light-microscopy reporting table on its [submission resources page](https://www.nature.com/ncomms/submit/resources).

## Repository and code archive

- [ ] Remove legacy directories from the final archival release, or explicitly exclude them from the release tag.
- [ ] Freeze a versioned release corresponding exactly to the submitted manuscript.
- [ ] Archive that release with a permanent DOI and cite it in the manuscript.
- [ ] Add a code license, repository citation metadata, author/contact information, and the final code DOI to the root README.
- [ ] Export exact resolved dependency versions for all three documented environments.
- [ ] Confirm that a fresh checkout contains no local paths except the intentional laboratory fallback in `shared/data_paths.py` and provenance-only YAML fields.
- [ ] Test the default plot reconstruction and the two optional examples from a clean extraction; distinguish this from full raw-cohort reanalysis.
- [ ] Record operating system, hardware, wall-clock runtime, and peak storage for expensive stages.

Nature Portfolio requires central custom code to be available to editors and reviewers and asks for a separate Code Availability statement explaining access and restrictions. Releasing a fixed, citable snapshot is consistent with the journal's current [reporting and code-availability policy](https://www.nature.com/ncomms/editorial-policies).

## Figshare data archive

- [ ] Preserve the structure in [`DATA_MANIFEST.md`](DATA_MANIFEST.md).
- [ ] Verify that every deposited input is actually used or clearly label auxiliary material.
- [ ] Include archive-level and file-level SHA-256 checksums.
- [ ] Add a data license, creators, version, description, keywords, and links between the data DOI, code DOI, and manuscript.
- [ ] Test reviewer access before submission, including any private-review link.
- [ ] Cite the dataset in the manuscript reference list as well as in the Data Availability statement.
- [ ] State the figure-source starting boundary and raw-cohort omissions explicitly; identify the one CZI/day and complete simulation snapshot as optional examples.

Nature Portfolio requires a Data Availability statement that makes access to the minimum dataset needed to interpret, verify, and extend the work transparent. Nature Communications specifically identifies Figshare as a suitable general repository in its editorial on [simplifying data and code sharing](https://www.nature.com/articles/s41467-024-50517-4).

## Figure source data and reporting files

- [ ] Prepare source-data tables for every main and supplementary graph, including the plotted points and values underlying summary statistics.
- [ ] Map each source-data sheet/file unambiguously to its figure and panel.
- [ ] Verify that `n` denotes biological versus technical replicates consistently in captions, Methods, and reporting forms.
- [ ] Complete the current bio/life-sciences reporting summary.
- [ ] Complete the journal's light-microscopy reporting table where applicable.
- [ ] Cross-check sample exclusions, statistical tests, and cohort definitions against the fixed analysis and manuscript text.

The journal has long emphasized supplying source data underlying graphs and making raw data accessible to reviewers; see its reproducibility editorial, [“Reproducibility: let's get it right from the start”](https://www.nature.com/articles/s41467-018-06012-8).

## Suggested manuscript statements

Replace bracketed fields only after the archives are frozen.

**Data Availability (draft; author review required).** Figure-source tables and arrays for the main and supplementary plots are available on Figshare at [DATA DOI], together with one native microscopy image/day and one complete simulation snapshot as processing examples. The default deposited workflow redraws plots from these frozen figure inputs; it does not repeat the complete raw-cohort analysis. Access to the additional cohort data is described at [AUTHOR TO SPECIFY].

**Code Availability.** The custom analysis, figure-generation, model-fitting, and reinforcement-learning code used in this study is available at [CODE DOI] and [repository URL], release [VERSION]. Installation and full-reproduction instructions are provided in the archived repository. The data archive location is configured through a single environment variable; no scientific parameters are changed by this path configuration.

These drafts must be reconciled with the final deposit contents and the manuscript before submission.
