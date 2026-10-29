# Figure and Table Source Archive

This folder preserves 670 compact files selected from the supplied
`MOL_figure_source_bundles_20261022_v4` distribution. Files are byte-identical to
the inspected copies. Original figure/table and revision directory names are
retained so different revisions are not silently substituted.

This is a source archive, not a single executable simulation installation.
Archived scripts can contain author-machine paths, depend on omitted models,
or write into their configured workspace. Do not run arbitrary scripts here.
Use the root `verify_publication.py` for safe local checks. The base package's
`reproduce.py` covers groups 01-06 only.

The root `metadata/publication_sources.csv` is the selection ledger. `COPY`
means byte verification, not proof of final-artwork reproduction. Other decisions
record external binaries, raw experiment candidates, unreviewed notebooks or
hash mismatches; these payloads were not silently included. A recorded historical
SHA-256 for an external file is not a claim that this release downloaded it.

Table directories also contain `SI_displayed.csv` where provided. Such files are
Word-table transcriptions, not experimental raw data. Literature tables,
condition summaries, particle-level derived records and original source code
remain distinguished by the manifest's `role` and `scope` columns.

The common event selector, molecular runner, seeded runner, surface-domain
runner and finite-bath code are under
`SI_tables_sources_20261023/Table_S28/code/`. Their presence does not supply
the missing `code/candidate_corrected` engines or model files. The verifier
compares available runner/selector hashes to six executed UiO-66 result records.

No new license or authorship attribution is assigned to these copies. Author
approval and the release checklist are required before public redistribution.
