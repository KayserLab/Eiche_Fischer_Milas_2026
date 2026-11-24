# Deposited data

The data ZIP is deposited separately from this code. Its root contains:

- `README.md`: directory map and starting boundary.
- `Documentation/DATA_GUIDE.md`: schemas, units and explicit omissions.
- `FILE_MANIFEST.tsv`, `MANIFEST.json`: deposited data/resource files.
- `CHECKSUMS.sha256`: payload checksums.
- `DATASET_SUMMARY.json`: measured counts and sizes.

Supply its extracted `Data/` directory as described in the
[repository README](../README.md). The compact public release includes frozen
full-cohort figure-source tables/arrays under `Plot_Source/`, companion CSV
tables and treatment schedules, fixed resources, selected lossless movie
pages, one native CZI image/day and one complete simulation snapshot with its
matching reduced file. It does not include the full raw/prepared cohort or a
microscopy timeline. The default route redraws plots; the original full
analysis requires additional data and explicit `--mode full`.
Outputs under `Data/Reference/` are comparison material.

`Reference/Plot_Renderer_Outputs/` records the original renderers' outputs for
the matching compact release. It tests faithful replotting. Older untouched
author outputs are kept separately under `Reference/Author_Supplied_Outputs/`;
agreement with that manuscript baseline is a separate comparison.

Archive-integrity and full scientific-execution reports are separate.
Use the matching code version and verified ZIP; the archive's own manifest
defines its deposited contents.
