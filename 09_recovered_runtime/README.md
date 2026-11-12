# Molecular Engines and Models

`code/candidate_corrected/` contains the UiO-66 and MOL engines.
`models/` contains eight inputs: explicit-H BDC, BTC, BTBa/BTBb, Zr6/Zr12
and their AA variants. `BDC_explicit_H.mol` has 16 atoms including four
explicit hydrogens; it is distinct from the older hydrogen-free BDC model.

Source bytes and project-relative paths are preserved. The UiO-66 engine hash
is `4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9`,
matching the six executed-result records in
`../07_publication_sources/SI_tables_sources_20261023/Table_S7/`.
Full per-file hashes are in `../metadata/recovered_sources_20261031.json`.

## Running Calculations

Common, seeded, surface and finite-bath runners are in
`../07_publication_sources/SI_tables_sources_20261023/Table_S28/code/`.
Configure their input/output paths using the corresponding campaign manifest.
Seeds, parameters and source-version mappings are in [group 12](../12_runtime_provenance/);
continuation runs additionally require the specified external checkpoint.

From the repository root, `python verify_publication.py` checks source hashes,
engine syntax, molecular atom counts and recorded engine/runner/selector versions.
