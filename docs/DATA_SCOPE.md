# Code and Analysis Data

The repository contains simulation and analysis code with the compact inputs
needed by the documented workflows.

## Included

- Simulation engines, runners, selectors, plotting scripts and notebook sources.
- Molecular models, parameter manifests and 28 small seed/host inputs.
- Per-run terminal statistics, condition summaries, first-passage and residence
  tables, and plot-ready growth curves.
- Representative structures, entropy geometries and four BTB classifier examples.
- Experimental DLS, NMR and other source tables.
- Source versions, file hashes and row-to-run indices.

Copies used by different archived scripts retain their original paths.
A repeated copy of a dataset is not an additional experimental or simulation replicate.

## Simulation Archives

Bulk event histories, production checkpoints, multi-block MOL2 streams and
full per-run structure collections are stored separately. The compact BTB
classification workflow checks 284 row/run mappings and 7,978 recorded
assignments, with geometric classification tests on four included examples.
Use `python -m tools.audit_fig5d_lineage --full-source-archive PATH` to
reclassify the complete external cohort.

[The retention manifest](../metadata/data_retention.json) identifies external
bulk payloads by hash. Molecular starting inputs and representative structures
remain included. See [code entry points](CODE_INDEX.md) for analysis commands.
