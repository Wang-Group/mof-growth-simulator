# Entropy Sources And S6 Reproduction

The 13 CSV/XLSX files in `data/` were extracted byte-for-byte from the supplied
`Entropy calculation/data.zip`. See `source_manifest.json` for the archive and
member hashes. No table was edited, refitted or relabelled.

These are supplied calculation/processing tables associated with the entropy
discussion. The 2026-10-31 audit matched all 18 displayed S6 numbers to the six
reference-species rows of `entropy_results.csv`; the shared standalone CSV has
the same bytes. See [the value audit](../13_table_source_audit/README.md).

The November 1 NAS follow-up recovered the entropy and molecular-properties
notebooks, mapped the six geometries and located the solvent/model inputs.
Local recalculation matched all 18 displayed values; BTB maps to `BTBa_new.xyz`.
`source/` now includes all six supplied code/notebook files and 33 MOL/XYZ
geometries (5,099,439 bytes). `calculation_manifest.json` records original and
packaged SHA-256 values. Python and geometry files are byte-identical; notebook
cell sources are unchanged but outputs, execution counts and metadata are
cleared. The original 13 data tables are unchanged, not duplicated.

## Reproduce Table S6

From the repository root:

```bash
python tools/reproduce_entropy.py
```

This writes `s6_recalculated.csv` and `s6_verification.json` under
`reproduced_local/entropy/`. It verifies all source hashes, recomputes the six
geometry mappings and reproduces 18 SI values at displayed precision. It uses
the original notebook's rounded molecular inputs, symmetry numbers and solvent
parameters at 393 K. It does not silently replace rounded inertia values with
full-precision values. `verify_publication.py` runs this same calculation check
without writing outputs or accessing NAS.

## Archived Code Boundaries

- `calculate_entropy.ipynb` provides the S6 function and original inputs.
- `calculate_molecular_properties.ipynb` supplies the geometry calculations.
- `Entropy.py` and `main.py` are the historical batch implementation.
- `data_fitting.ipynb` fits larger-assembly entropy trends from the supplied CSVs; its optional environment additionally requires scikit-learn. That fit is not part of the S6 verification.
- `mol_xyz_files/pre-process.ipynb` is historical preprocessing and can overwrite XYZ files. Do not run it in the source archive.

Use the safe entry above, not Run All on the archived notebooks. The old batch
runner has workstation paths, Linux signal handling, and dependencies on
upstream MOL2 folders not included here. It is preserved for provenance,
not presented as a portable one-click rerun. Original outputs remain external;
reproducing the supplied entropy model is not independent validation of all
its physical approximations.
