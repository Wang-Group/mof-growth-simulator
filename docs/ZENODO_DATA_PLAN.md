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
| HASH_MISMATCH_REVIEW_REQUIRED | 3 | Obtain version confirmation and update the provenance ledger; do not publish as the earlier hash-matched version |

These are file counts, not independent simulations or experiments. External
records total about 606 MB in this selection; this is not the size of a complete
research archive. Missing corrected engines/models and required molecular
checkpoints may expand the final archive. The old 26.4 GB long-growth tree is
not automatically required in full, but compact summaries alone do not preserve
every event or molecular state.

## Proposed Datasets

1. **Historical trajectory sources:** original pickles, run-folder parameters,
   exact source hashes, structure references and time-axis conventions. The
   compact BTB step segments remain convenient plotting inputs, not a claim to
   retain every original event timestamp.
2. **Current molecular campaigns:** executed corrected engines, model/seed/host
   provenance, condition grids, per-trace records and necessary checkpoints.
   Distinguish independent repeats from continuations and unsuccessful/censored
   runs. Include a manifest linking raw inputs to the GitHub processed tables.
3. **Experimental and quantum-chemical inputs:** only author-confirmed raw
   acquisitions and Gaussian inputs/outputs supporting the published analyses.
   Derived entropy spreadsheets or Word-table transcriptions are not originals.

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
