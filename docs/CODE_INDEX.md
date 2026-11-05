# Analysis Entry Points by Scientific Topic

Use the safe entries below first. Archived code is retained for provenance and
may need external inputs; its presence is not a claim of full production replay.
No manuscript, credentials, large trajectory directories or unreviewed notebook
outputs belong in this branch. The independent release branch must not replace
the historical `main` branch.

Topics below describe the inputs, calculation and verification scope independently
of manuscript numbering. Paths retain their historical names for compatibility;
for example, a script containing `s20` still reproduces the same controlled
modulator comparison after the paper renumbers that table. Document-specific
labels are explained in the [dated cross-reference guide](MANUSCRIPT_CROSSWALK.md).

| Topic | Code location or entry | Verified scope |
|---|---|---|
| Lightweight figure reproduction | `python reproduce.py` | Groups 01-06 only; reads compact local tables |
| UiO-66 long growth and entity-13 structures | `01_uio66_long_growth/source/` and `scripts/` | Historical simulator and analysis sources, 24-run compact inputs; not a new KMC run |
| DLS/Gualtieri fit | `02_dls/fit_uio66_dls_gualtieri.py` | Original DLS input and processed-progress refit |
| Historical BTB growth | `03_btb_trajectory/source/` and `original/` | Executed source and original plot convention retained |
| BTB motif frequencies | `reproduce.py` and `04_btb_motifs/` | Original frequencies and representatives unchanged; all 284 rows now linked through group 15 |
| BTB motif trajectory lineage | `python -m tools.audit_fig5d_lineage`, sources in `15_fig5d_trajectory_lineage/` | Offline replay of 3688 entity 1-13 snapshots; full 7978-assignment recovery audit and raw-stream spot checks recorded |
| Historical BTB source search | `python -m tools.audit_fig5_sources`, archives in `14_fig5_source_followup/source/` | November 1 partial evidence; superseded for motif lineage by group 15; retained as a dated source-search record |
| BTB kinetic boundary reconstruction | `python 16_fig5c_reconstruction/render.py`; `python -m tools.audit_fig5c` | Same 1260-condition probabilities; projected historical-boundary and four-grid convergence checks; not original module recovery |
| Mixed-seed and vacancy repair | `05_mixed_zr6_zr12/original/`, `06_vacancy_repair/original/` | Historical code; use lightweight renderer rather than replaying the archived mixed-seed script |
| Current response/seeded/surface/finite-bath campaigns | `07_publication_sources/SI_tables_sources_20261023/Table_S28/code/` | Supplied runners/selectors and processed input grids |
| Corrected engines and molecular models | `09_recovered_runtime/` | Executed versions matched by recorded SHA-256 |
| Campaign-specific source versions and small seeds | `12_runtime_provenance/` | Exact code/seed references and checkpoint lineage; large checkpoints external |
| Ring-closure entropy | `python tools/reproduce_entropy.py`, sources in `08_entropy_inputs/source/` | Six geometry mappings and 18 values reproduced, no batch-writing cells |
| UiO-66 first-passage and residence-time panels | `python 10_plot_versions/render_s7.py` | Raw algorithm time u; seconds scripts retained only as history |
| Controlled modulator-geometry comparison | `python tools/reproduce_s20.py`; [method and outputs](S20_REPRODUCTION.md) | 27,648 endpoints; 864 groups recomputed; nine matched rows checked against saved precision and a dated SI snapshot; no time rescaling |
| April 3 mixed-node seed growth | `python 11_s21_april03/plot_saved_trace.py --help` | Plot-only entry; April 3 variants remain separate from April 10 |
| Exchange and modulator-identity sources | `python -m tools.verify_submission_sources` | 99 source files; ten NMR acquisitions; seven identity regressions; recorded document/artwork snapshot |
| Whole candidate verification | `python verify_package.py`, `python verify_publication.py` | Source hashes and expressly listed numerical checks, not all archived scripts or every bootstrap |

## Remaining Coverage Limits

The historical BTB-boundary module `analyze_figure6_current_parameter_phase_v1.py` remains
unlocated in the bounded searches recorded in the publication checklist. Its
supplied plotter and data are included. Group 16 now provides an explicitly labelled
independent reconstruction with validated projected-boundary and mesh convergence;
do not call it the executed original. Group 14 records the expanded 5090/NAS search. Group 15
now links all 284 BTB motif rows, preserves the co-located block-based classifier and
extraction notebooks, and resolves the three NAS repeat-index conflicts by hash.
Its fixed-representative replay is not graph-isomorphism validation or RNG replay.

Large checkpoints/raw trajectories require a selected external archive such as
Zenodo. A manifest or a NAS path is not an uploaded dataset. Code MIT and project
data CC BY 4.0 are confirmed; see `LICENSING.md`. Final release/archive identifiers
remain to be assigned. Current actions are in `docs/PUBLICATION_RELEASE.md`;
historical-module provenance is not an unresolved source-data request.
