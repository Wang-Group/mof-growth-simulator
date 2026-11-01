# External Data Archive Plan

This is a selection plan, not evidence of a Zenodo upload or an assigned DOI.
The code candidate contains compact inputs and derived tables; external payloads
remain at their supplied source locations. Resolve author permissions and the
publication checklist before public release.

Use `metadata/publication_sources.csv` as the machine-readable ledger. Its
`source_relative` paths are relative to `MOL_figure_source_bundles_20261022_v4`.
The recorded bytes and SHA-256 identify the inspected versions. `COPY` records
belong to GitHub; the other decisions require the handling below.

| Decision | Records | Proposed handling |
|---|---:|---|
| EXTERNAL_LARGE_FILE | 3 | Original BTB, HKUST-1 and PCN-240 trajectory pickles; select for a raw-trajectory archive with exact plot/run mapping |
| EXTERNAL_BINARY_OR_ARTWORK | 23 | Review individually: necessary smaller checkpoints/raw trajectories may belong in Zenodo; exported artwork is not a substitute for raw data |
| EXTERNAL_NOTEBOOK_OUTPUTS_NOT_REVIEWED | 4 | Review source cells and embedded outputs first; do not publish credentials, personal output or unrestricted execution history |
| EXTERNAL_RAW_EXPERIMENT | 213 | NMR acquisition candidates; confirm the exact acquisitions behind the paper's table rows before selecting complete acquisition folders |
| VERSION_VARIANT_ARCHIVED | 3 | Now included separately in group 10 with both original and shared-revision provenance; final figure adoption still needs confirmation |

These are file counts, not independent simulations or experiments. External
records total about 606 MB in this selection; this is not the size of a complete
research archive. The two corrected engines and eight models are now included
in group 09; required molecular seeds/hosts/checkpoints may expand the final
archive. Group 11 retains April 3 sources separately from the April 10 rerun;
its small saved JSON traces need not be treated as missing external payloads.
The old 26.4 GB long-growth tree is
not automatically required in full, but compact summaries alone do not preserve
every event or molecular state.

The subsequent group 12 audit located and remotely hash-verified **975 checkpoint
files plus 975 associated result JSONs**, totalling **545,387,724 bytes**.
Their exact project-relative paths and hashes are in
`12_runtime_provenance/remote_checkpoint_verification.json`; continuation and
paired-branch relationships are in `checkpoint_lineage.csv`. These payloads were
not downloaded into the Git branch. Do not simply add this total to the older
606 MB ledger without checking overlap. Small seed/host pickles and compressed
parameter manifests are already packaged; they need not be requested again.

Group 14 adds three historical BTB trajectory hash records totaling 352,537,047
bytes. They remain on NAS and were not deserialized or copied into GitHub. Their
filename indices conflict with the parent directory, so archive selection must
retain that ambiguity rather than silently assign classification rows. Nine
small compressed large-assembly structures and the two recovered code sources
are already packaged. Check overlap before adding this count to older ledgers.

## Proposed Datasets

1. **Historical trajectory sources:** original pickles, run-folder parameters,
   exact source hashes, structure references and time-axis conventions. The
   compact BTB step segments remain convenient plotting inputs, not a claim to
   retain every original event timestamp.
2. **Current molecular campaigns:** executed corrected engines, model/seed/host
   provenance, condition grids, per-trace records and necessary checkpoints.
   Distinguish independent repeats from continuations and unsuccessful/censored
   runs. Include a manifest linking raw inputs to the GitHub processed tables.
3. **Experimental and entropy-calculation inputs:** author-confirmed raw
   acquisitions and the actual geometry,
   solvent/model inputs and scripts behind the solution-entropy calculation.
   Include Gaussian inputs/outputs only if they were actually used. S6's source
   notebooks, mapped geometries and model inputs were recovered on NAS and
   locally recalculated on November 1. The small sources are now in group 08
   for GitHub, with a safe offline S6 reproduction entry; do not request them
   again as missing Zenodo payloads.

Table S8's existing 12-row time table has been confirmed by the author as the
original record. Its source request is closed; this archive plan does not request
another observation sheet or a newly created Excel file for S8.

## Release Procedure

Select the raw files into a dedicated staging directory, verify byte sizes and
SHA-256, and create archives as a separate step. Verify archive member names and
extracted hashes before transfer. Record dataset version, authors, licenses,
software requirements and the corresponding Git commit/tag. Upload only after
approval, then insert the actual returned DOI and GitHub release URL into both
records. No DOI placeholder should be presented as a published identifier.

Pickle files are historical Python serialization and must be treated as trusted
research inputs only: loading an untrusted pickle can execute code. Supply
portable CSV exports and documented exporters where practical.
