# MOF/MOL Growth: Code and Source Data

Computational sources and experimental data supporting **How monocarboxylic acids
modulate metal-organic framework synthesis**.

This publication branch contains curated KMC analysis inputs, molecular structures,
plotting and fitting code, experimental source tables, and versioned simulation
sources. It is separate from the original development history on `main`.
The [figure map](FIGURE_MAP.csv) connects folders to the current MS/SI numbering;
historical filenames can use earlier figure numbers.

**Release preparation:** the November 2 document/source corrections are recorded.
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
python -m tools.audit_fig5d_lineage
python 16_fig5c_reconstruction/render.py
python -m tools.audit_fig5c
python -m tools.verify_submission_sources
```

These analysis/verification commands use packaged inputs, not shared drives or
SSH access. Fresh outputs go to `reproduced_local/`; supplied reference plots
remain in `reproduced/`. See [CODE_INDEX.md](docs/CODE_INDEX.md) for source locations
and the separate April 3 S21 plot-only entry. The default mixed-seed renderer
retains the distinct April 10 comparison dataset.

## Contents

| Topic | Folders | Current use |
|---|---|---|
| UiO-66 first passage and structures | [01](01_uio66_long_growth/) | 24 trajectories, nine thresholds, entity-13 structures; Fig. 3D and S7 |
| Experimental DLS | [02](02_dls/) | Original CSV, apparent-progress processing and Gualtieri fit; Fig. 3E |
| Historical BTB trajectories and motifs | [03](03_btb_trajectory/), [04](04_btb_motifs/), [15](15_fig5d_trajectory_lineage/) | Fig. 5A/D; 284 row/run mappings and figure-size snapshots |
| Mixed seeds and vacancy repair | [05](05_mixed_zr6_zr12/), [06](06_vacancy_repair/), [11](11_s21_april03/) | Distinct historical repair datasets; April 3 is the current S21 source |
| Figure/table-organized calculation sources | [07](07_publication_sources/), [09](09_recovered_runtime/), [12](12_runtime_provenance/) | Response surfaces, seeded/surface growth, finite baths, engines, seeds and checkpoint lineage |
| Entropy | [08](08_entropy_inputs/) | Six species, original code and geometries; eighteen reproduced S6 values |
| Figure versions and provenance | [10](10_plot_versions/), [13](13_table_source_audit/), [14](14_fig5_source_followup/) | Archived version/source records; dated search records are not the current task list |
| BTB kinetic boundary | [16](16_fig5c_reconstruction/) | Fig. 5C interpolation, isosurface extraction and grid-convergence checks |
| Submission supplements | [17](17_submission_sources/) | SI S7 identity CSVs, ten S4 NMR acquisitions and mapping, N=2000 Fig. 6C reference panel |

## Methods and Scope

- **Time coordinates:** Fig. 3D/S7 use raw algorithm time `u = stored time_s / 0.175`.
  The current MS specifies a 0.2405 ms site-opening reference scale for the
  historical Fig. 3A/5A/7 traces. A reference scale is not an independently
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
- **Fig. 5C:** the release implementation extracts the P=0.5 isosurface from the
  recorded 1260-condition success fractions. Interpolation coordinates, grid
  stability and implementation provenance are documented in group 16.
- **NMR:** Table S4 preserves author/software linewidth measurements. The
  row-to-acquisition mapping and original spectra are packaged; the local check
  validates source identity and arithmetic, not an independent spectral refit.

Archived production scripts preserve their executed versions and can require
external checkpoints. The documented lightweight entries reproduce the stated
analyses; they do not re-execute every production campaign. Hashes, numerical
checks and test results are reported separately so their coverage is explicit.

## Version and Publication

The current [document baseline](metadata/release_document_baseline.json) records
the reviewed MS/SI hashes and accepted Fig. 6 image. [CHECKSUMS.sha256](CHECKSUMS.sha256)
covers immutable packaged files. [Publication preparation](docs/PUBLICATION_RELEASE.md)
lists only the remaining release actions; [provenance notes](docs/PROVENANCE_NOTES.md)
retain the relevant historical distinctions.

Branch: `codex/publication-release-20261029`. This independent-history publication
branch does not replace `main`. Original authorship and third-party notices remain
with their sources. Cite the repository and exact release tag or commit used;
the paper and dataset DOI should be added when assigned, not guessed.
