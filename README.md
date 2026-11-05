# MOF/MOL Growth: Code and Source Data

Research code and compact data for molecular assembly, nucleation, established
growth, layer morphology and defect repair in metal-organic frameworks and layers.

This publication branch contains curated KMC analysis inputs, molecular structures,
plotting and fitting code, experimental source tables, and versioned simulation
sources. It is separate from the original development history on `main`.
Browse the [contents by scientific topic](#contents) and the
[analysis entry points](docs/CODE_INDEX.md). These descriptions do not depend on
manuscript figure numbers. Source folders and scripts retain historical names
to preserve provenance and working commands; those names do not identify the
latest manuscript layout. [Dated manuscript cross-references](docs/MANUSCRIPT_CROSSWALK.md)
are secondary provenance records, not the repository's contents index.

Code is licensed under MIT and project-produced data under CC BY 4.0, subject to
the [scope and third-party notices](LICENSING.md). The final release identifier
and selected large-data Zenodo archive are still to be assigned. MS/SI documents
are supplied separately and are not included here.

## Reproduce the Analyses

Use Python 3.11 or newer and the versions in [requirements.txt](requirements.txt).
Install Arial for the original figure typography.

```bash
python -m pip install -r requirements.txt
python reproduce.py --output-dir reproduced_local
python verify_package.py --output-dir reproduced_local
python verify_publication.py --report reproduced_local/publication_check.json
python -m unittest discover -s tests -v
```

The lightweight renderer covers groups 01-06; it does not launch KMC. Additional
offline entries are:

```bash
python 10_plot_versions/render_s7.py
python tools/reproduce_entropy.py
python tools/reproduce_s20.py
python -m tools.audit_fig5d_lineage
python 16_fig5c_reconstruction/render.py
python -m tools.audit_fig5c
python -m tools.verify_submission_sources
```

These analysis/verification commands use packaged inputs, not shared drives or
SSH access. Fresh outputs go to `reproduced_local/`; supplied reference plots
remain in `reproduced/`. See [CODE_INDEX.md](docs/CODE_INDEX.md) for source locations
and the separate April 3 mixed-seed plot-only entry. The default mixed-seed renderer
retains the distinct April 10 comparison dataset.

The [controlled modulator-geometry guide](docs/S20_REPRODUCTION.md) documents
recalculation from 27,648 per-trajectory records, comparison with nine saved
matched-condition results, and summaries for all 864 assay-condition groups.
The check against a dated SI table is retained as an additional regression test.
No original source file or manuscript is overwritten.

## Contents

| Scientific topic | Folders | Data and analysis content |
|---|---|---|
| UiO-66 first passage and small-assembly structures | [01](01_uio66_long_growth/) | 24 trajectories, nine first-passage thresholds, low-entity residence times and last-saved entity-13 structures |
| Experimental DLS | [02](02_dls/) | Original size trace, apparent-progress processing, Gualtieri fit, parameters and diagnostic plots |
| Historical BTB trajectories and motifs | [03](03_btb_trajectory/), [04](04_btb_motifs/), [15](15_fig5d_trajectory_lineage/) | Growth-curve segments, motif frequencies, 284 row/run mappings and 3,688 entity 1-13 snapshots |
| Mixed-node seeds and vacancy repair | [05](05_mixed_zr6_zr12/), [06](06_vacancy_repair/), [11](11_s21_april03/) | Zr6/Zr12 evolution, defect-filling curves and initial/intermediate structures; April 3 and April 10 families kept separate |
| Nucleation, seeded growth and surface-domain simulations | [07](07_publication_sources/), [09](09_recovered_runtime/), [12](12_runtime_provenance/) | Response grids, endpoint statistics, finite-bath calculations, executed engines, models, seeds and checkpoint lineage |
| Controlled modulator geometry | [source records](07_publication_sources/SI_tables_sources_20261023/Table_S20/matched_source_files/) | Twelve identities, 27,648 terminal records, 864 assay-condition groups and matched assembly/growth statistics |
| Ring-closure entropy | [08](08_entropy_inputs/) | Six species, original calculation code, molecular geometries and eighteen reproduced entropy values |
| BTB kinetic formation boundary | [16](16_fig5c_reconstruction/) | 1,260-condition success fractions, interpolation, P=0.5 isosurface and grid-convergence checks |
| Modulator identity and exchange measurements | [17](17_submission_sources/) | Literature-derived identity tables, checked regressions and ten NMR acquisitions with source mappings |
| Finite-cluster structure and scattering | [07](07_publication_sources/), [17](17_submission_sources/figure6c/) | Archived PXRD analyses and a matched N=2000 BTB comparison with four trajectories per condition |
| Source versions and historical audits | [10](10_plot_versions/), [13](13_table_source_audit/), [14](14_fig5_source_followup/) | Archived plot versions, source correspondence and dated search records, not an active missing-data list |

## Methods and Scope

- **Time coordinates:** the 24-trajectory UiO-66 analysis uses raw algorithm time
  `u = stored time_s / 0.175`. Historical representative UiO-66, BTB, HKUST-1 and
  PCN-240 trajectory plots use a 0.2405 ms site-opening reference scale.
  A reference scale is not an independently
  calibrated absolute growth time across frameworks or model variants.
- **Structures:** the 24 entity-13 structures are last-saved snapshots with nine
  reconstructed topologies, not first-passage structures or a unique deterministic
  nucleus. They are distinct from the historical UiO motif collection. BTB motif
  classification uses the last occurrence before the first maximum and the
  archived composition/radial-histogram rule.
- **DLS:** the fit uses 40 points from 117-1170 s, including preparation time.
  R2 = 0.972658 applies to processed monotonic apparent progress, not raw diameter
  or crystallinity. The 0-600 s panel is a display window; nG is selected from 1, 2, 3.
- **Model variants:** fixed-coordinate and precursor-depleting simulations retain
  their own executed mappings, endpoints and clocks. Continuations and paired
  branches are not counted as new independent starting particles.
- **BTB kinetic boundary:** the release implementation extracts the P=0.5 isosurface from the
  recorded 1260-condition success fractions. Interpolation coordinates, grid
  stability and implementation provenance are documented in group 16.
- **NMR:** the exchange dataset preserves author/software linewidth measurements. The
  row-to-acquisition mapping and original spectra are packaged; the local check
  validates source identity and arithmetic, not an independent spectral refit.

Archived production scripts preserve their executed versions and can require
external checkpoints. The documented lightweight entries reproduce the stated
analyses; they do not re-execute every production campaign. Hashes, numerical
checks and test results are reported separately so their coverage is explicit.

## Version and Publication

The [recorded document baseline](metadata/release_document_baseline.json) is a
dated audit snapshot, not a claim to track the latest manuscript. Renumbering a
figure does not change a dataset or invalidate a calculation. Source hashes and
executed conditions identify the data; [CHECKSUMS.sha256](CHECKSUMS.sha256)
covers immutable packaged files. [Publication preparation](docs/PUBLICATION_RELEASE.md)
lists only the remaining release actions; [provenance notes](docs/PROVENANCE_NOTES.md)
retain the relevant historical distinctions.

Default branch: `publication-release-20261029`. This independent-history publication
branch does not replace `main`. Original authorship and third-party notices remain
with their sources. Cite the repository and exact release tag or commit used;
the paper and dataset DOI should be added when assigned, not guessed.
