# Code Entry Points And Coverage

Use the safe entries below first. Archived code is retained for provenance and
may need external inputs; its presence is not a claim of full production replay.
No manuscript, credentials, large trajectory directories or unreviewed notebook
outputs belong in this branch. The independent release branch must not replace
the historical `main` branch.

| Topic | Code location or entry | Verified scope |
|---|---|---|
| Lightweight figure reproduction | `python reproduce.py` | Groups 01-06 only; reads compact local tables |
| UiO-66 long growth and entity-13 structures | `01_uio66_long_growth/source/` and `scripts/` | Historical simulator and analysis sources, 24-run compact inputs; not a new KMC run |
| DLS/Gualtieri fit | `02_dls/fit_uio66_dls_gualtieri.py` | Original DLS input and processed-progress refit |
| Historical BTB growth | `03_btb_trajectory/source/` and `original/` | Executed source and original plot convention retained |
| BTB motif frequencies | `reproduce.py` and `04_btb_motifs/` | Original frequencies and representatives unchanged; all 284 rows now linked through group 15 |
| Fig. 5D trajectory lineage | `python -m tools.audit_fig5d_lineage`, sources in `15_fig5d_trajectory_lineage/` | Offline replay of 3688 figure-size snapshots; full 7978-assignment recovery audit and raw-stream spot checks recorded |
| Earlier Fig. 5 provenance search | `python -m tools.audit_fig5_sources`, archives in `14_fig5_source_followup/source/` | November 1 historical partial evidence; superseded for panel D by group 15, retained for panel C search scope |
| Mixed-seed and vacancy repair | `05_mixed_zr6_zr12/original/`, `06_vacancy_repair/original/` | Historical code; use lightweight renderer rather than replaying the archived mixed-seed script |
| Current response/seeded/surface/finite-bath campaigns | `07_publication_sources/SI_tables_sources_20261023/Table_S28/code/` | Supplied runners/selectors and processed input grids |
| Corrected engines and molecular models | `09_recovered_runtime/` | Executed versions matched by recorded SHA-256 |
| Campaign-specific source versions and small seeds | `12_runtime_provenance/` | Exact code/seed references and checkpoint lineage; large checkpoints external |
| S6 entropy | `python tools/reproduce_entropy.py`, sources in `08_entropy_inputs/source/` | Six geometry mappings and 18 values reproduced, no batch-writing cells |
| S7 algorithm-time panels | `python 10_plot_versions/render_s7.py` | Adopted raw-u entry; seconds scripts retained only as history |
| April 3 S21 saved trace | `python 11_s21_april03/plot_saved_trace.py --help` | Plot-only entry; April 3 variants remain separate from April 10 |
| Whole candidate verification | `python verify_package.py`, `python verify_publication.py` | Source hashes and expressly listed numerical checks, not all archived scripts or every bootstrap |

## Remaining Coverage Limits

The exact Fig. 5C module `analyze_figure6_current_parameter_phase_v1.py` remains
unlocated in the bounded searches recorded in the publication checklist. Its
supplied plotter and data are included; do not invent a replacement and call it
the executed original. Group 14 records the expanded 5090/NAS search. Group 15
now links all Fig. 5D rows, preserves the co-located block-based classifier and
extraction notebooks, and resolves the three NAS repeat-index conflicts by hash.
Its fixed-representative replay is not graph-isomorphism validation or RNG replay.

Large checkpoints/raw trajectories require a selected external archive such as
Zenodo. A manifest or a NAS path is not an uploaded dataset. Licensing, author
credit and final release/archive identifiers remain author release decisions.
