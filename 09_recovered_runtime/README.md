# Recovered Corrected Engines and Models

Ten files were read from the 5090 project
`/home/wave/MOL_growth_recalculation_20260830/` on 2026-10-31 using authorized
read-only access. Original project-relative paths and file bytes are retained.
No server settings, jobs or source files were changed. Credentials are excluded.

`code/candidate_corrected/` contains the UiO-66 and MOL engines.
`models/` contains eight inputs: explicit-H BDC, BTC, BTBa/BTBb, Zr6/Zr12 and
their AA variants. `BDC_explicit_H.mol` has 16 atoms including four explicit H;
the older hydrogen-free BDC model is not interchangeable.

All file sizes and SHA-256 values match the supplied historical remote index.
The UiO-66 engine SHA-256 is
`4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9`, matching
the six executed-result records in `07_publication_sources/SI_tables_sources_20261023/Table_S7/`.
This is byte-level provenance, not evidence that all campaigns have been rerun.

The corresponding common/seeded/surface/finite-bath runners remain under
`07_publication_sources/SI_tables_sources_20261023/Table_S28/code/`. Their
historical absolute paths have not been rewritten. A complete campaign replay
still needs exact seeds/hosts/checkpoints, conditions and continuation rules.
Do not execute archival code merely because it is now present.

Run `python verify_publication.py` from the repository root for hashes, engine
syntax, MOL atom counts and recorded engine/runner/selector correspondence.
Full per-file provenance is in `metadata/recovered_sources_20261031.json`.
