# Calculation Source Archive

This folder preserves 670 compact files from the supplied
`MOL_figure_source_bundles_20261022_v4` distribution. Original directory names
and source bytes are retained to distinguish calculation versions.

Use the [topic-based index](../docs/CODE_INDEX.md) to find analysis entry points.
The source ledger is `../metadata/publication_sources.csv`; its paths, roles,
sizes and hashes distinguish original code, experimental data, derived
statistics and transcribed tables.

## Simulation Code

The common event selector and molecular, seeded, surface-domain and finite-bath
runners are under `SI_tables_sources_20261023/Table_S28/code/`.
Their corresponding engines and models are in [group 09](../09_recovered_runtime/),
and versioned parameters and seed inputs are in [group 12](../12_runtime_provenance/).

Configure model, input and output paths before running archived production
scripts. Some retain their original machine-specific paths. Portable analysis
commands are listed in the root README and code index.

## Verification

From the repository root:

```bash
python verify_publication.py --report reproduced_local/publication_check.json
```

This checks source hashes and recalculates the listed numerical results.
Code and project data use the licenses described in [LICENSING.md](../LICENSING.md);
existing source attribution is preserved.
