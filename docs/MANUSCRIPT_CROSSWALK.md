# Dated Manuscript Cross-References

Use the root [contents](../README.md#contents) and [analysis index](CODE_INDEX.md)
to navigate by scientific topic. A dataset is identified by its source files,
hashes, simulation conditions and analysis definitions, not by a figure number.

## Archived Mappings

[`FIGURE_MAP.csv`](../FIGURE_MAP.csv) is the cross-reference snapshot accumulated
through November 3, 2026. It uses the manuscript/SI numbering checked during that
period. The legacy column name `current_reference` means current at that review,
not current in every later manuscript. Its row-level confirmations likewise refer
to the reviewed snapshots; they are not an automatically updated task list.

The associated records are:

- [`release_document_baseline.json`](../metadata/release_document_baseline.json):
  document hashes and accepted-artwork identification from the November 2 review,
  with older baselines retained.
- [`s20_reference.json`](../metadata/s20_reference.json): the November 3 snapshot
  of nine controlled modulator-geometry comparison rows and their source hashes.
- [`table_source_limits.csv`](../metadata/table_source_limits.csv): dated numerical
  verification scope and source assessments, with historical table identifiers.
- [`HANDOFF_20261022.md`](HANDOFF_20261022.md): an earlier handoff, not current
  publication instructions or a live list of missing data.

These snapshots are retained to explain past comparisons. They do not assert
that the latest Word files, captions or figure arrangement have been checked.

## Handling Revisions

- Renumbering or moving a figure/table does not require renaming source datasets,
  scripts or outputs. Historical labels inside those names remain valid paths.
- A changed parameter, dataset, statistic or scientific claim requires checking
  the corresponding analysis. Content changes are different from layout changes.
- If a journal-specific crosswalk is needed for a final submission, create a new
  dated mapping tied to those exact document hashes. Do not silently reinterpret
  an older mapping as the new version.
- Keep published-value regression tests and source hashes. A test against a
  frozen table remains useful evidence even after that table is renumbered.
- Cite the repository release/commit and relevant scientific topic in the paper.
  Exact internal file names need only appear where they help a reader reproduce
  an analysis, not throughout the scientific narrative.

Archived source-folder READMEs, notebooks and scripts may retain figure-specific
wording. Their original content is part of provenance; the topic-based root
README and analysis index provide the current reader-facing organization.
