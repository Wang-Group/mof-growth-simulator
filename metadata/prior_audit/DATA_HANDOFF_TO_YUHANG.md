# Data coordination for the October manuscript and SI

Review date: 2026-10-22. This note identifies the evidence available on Yibin's PC and the shared project drive, and the remaining records needed to reconcile the latest paper with its data. Responsibility assignments below are proposed coordination tasks, not messages already sent or tasks accepted by another author.

## Current document baseline

The current title is **How monocarboxylic acids modulate metal-organic framework synthesis**. The main quantitative story now uses direct-coordinate calculations of primary nucleation, established growth, surface-domain formation, and BTB layer growth. Historical recipe-based trajectories remain in the paper as separate sources. The August handoff alone does not cover the October manuscript.

The two source documents are:

- `C:/Users/yibinjiang/Documents/xwechat_files/wxid_w2m1w39nju0q22_a138/msg/file/2026-10/manuscript_defect_section_to_SI_20261018.docx`
- `C:/Users/yibinjiang/Documents/xwechat_files/wxid_w2m1w39nju0q22_a138/msg/file/2026-10/supporting_information_defect_section_to_SI_20261019.docx`

Reading method: the working final view includes tracked insertions and excludes tracked deletions and moved-from content. The source files were not edited, and no revisions were accepted. The MS contains 13 insertion and 41 deletion elements; the SI contains 39 insertion and 9 deletion elements. Neither file contains Word comments. The deleted main-text defect section was not treated as current manuscript text. Its supporting analyses remain in SI Section S6.

Source SHA-256:

```text
MS 682d0418f02450818162cae6682a6d95cf493ca769bafdb3174c5ad67115caa2
SI 6d78cd1f7a28cb1dfd39780b8ef747ee24526b8e76fcef06d200e3c4c070eb75
```

This is a data/provenance review with inspection of selected embedded figures, not a complete page-layout or scientific validation of every new calculation. XML paragraph identifiers in the extraction files are search aids, not Word page numbers.

## Evidence available on our side

Repository root: `C:/Users/yibinjiang/Documents/GitHub/mof-growth-simulator`.

| Current use | Evidence recovered | Verification and remaining work |
|---|---|---|
| Main Figure 3D and SI Figure S7 | `UiO-66/Long_Growth_Curated/data/first_passage_24.csv`, residence/representative-trace tables, source snapshot, and all 24 local raw trajectories | All 24 raw files exist and their sizes match the manifest. All 24 entity-13 snapshots exist. All six source files match their archived hashes. Figure 3D's plotting data match the curated table at all 216 values across 24 runs and nine thresholds. |
| Main Figure 3D-E artwork | `crystallization_model_fits/render_uio66_entity13_dls_linkage.py`, `output/uio66_entity13_dls_linkage.png`, and `output/uio66_induction_vs_dls_first_passage_curves.csv` | The saved two-panel artwork matches the two panels incorporated into the current main figure. The script documents the 24-run aggregation, medians, quartiles, scatter, and DLS overlay. The unmodified final figure assembly still needs to be registered by its author. |
| Main Figure 3E and SI S3.2 | `UiO-66/DLS_Gualtieri/` | Original `260316.csv`, processing and fitting code, fitted curves, parameters, and PNG/SVG are available. A fresh in-memory refit reproduced all stored parameters and fit quality. |
| Entity-13 motif family for the 24-run study | `UiO-66/Long_Growth_Curated/data/uio66_entity13_structure_*.csv` and `structures/entity13_groups/` | Nine reconstructed topologies across 24 last-saved snapshots; largest class 11/24. Composition is 5 Zr6 + 8 BDC in 21/24 and 4 Zr6 + 9 BDC in 3/24. These are last-saved structures, not proven first-passage structures. |
| Main Figure 5A historical BTB trajectory | Shared-drive pickle and plotting script identified below | Raw file exists and has been hashed. Its generated PNG matches the characteristic 620-690 ms segment and 4, 7, 10 plateaus. The executed time-scale command and complete simulation executable remain to be recovered. |
| Main Figure 5D historical BTB motif percentages | Shared-drive `unique_structures_result.csv` and `unique_structures/` | 284 rows, 71 distinct encoded conditions. Nonmissing denominators are 284 for entities 2-10 and 283 for entities 1 and 11-13. Recomputed displayed percentages match. Confirm the structure-index-to-image mapping and original successful-run selection before finalizing methods. |
| SI Figures S21-S22 mixed Zr6/Zr12 growth | `UiO-66/UiO_66_from_amorphous/Zr6_Zr12_mix/` | Recovered run summary confirms 37 Zr6 + 37 Zr12 + 98 BDC = 172 initial entities, four repeats, and zero retained Zr12 in all four final structures. The regenerated seed and original historical artwork must remain distinguished. |
| SI Figures S23-S24 original-vacancy repair | `UiO-66/UiO_66_from_amorphous/Missing_BDC_Zr6/` | Source table confirms 54/54 Zr and 132/591 BDC vacancies filled at N=2800; 54/54 and 584/591 at N=100001. This matches the current SI. No new occupancy reconstruction from the large pickles was performed in this audit. |
| SI Figure S25 illustrative repair | `UiO-66/UiO_66_from_amorphous/initialize_structures1/` and the August curated handoff | Supporting files and documented trajectories exist. Match the exact two plotted trajectories before marking this panel fully reconciled. |
| Historical concentration/morphology/preassociation records in Table S25 | `MOL_nucleation/`, `MOL_geometry/`, `figure7_zr_ligand_data/`, and `zr_ligand_one_shot_minimal_summary/` | Preserve as separately identified historical evidence. The old preassociation Figure 7 is not the current Figure 7. The old ML/phase-diagram data are not the current direct-coordinate Figure 5B-C grid. |
| Main Figure 7 and SI S42-S45 | HKUST-1 and PCN-240 simulation code; historical assembled artwork on the shared drive | Exact plotted trace, motif-count export, per-size denominators and executed clock remain unverified in this audit. Code presence alone does not resolve the figure source. |

## Newly recovered provenance that should reach Yuhang

### Figure 3D is traceable to the 24-run family

Search the SI for:

> Main panel: inherited first-passage plot, exact trace list unresolved.

Also search for:

> Exact trace/frequency export for A and first-passage cohort for D are unmatched.

The existing local plotting script reads `uio66_induction_vs_dls_first_passage_curves.csv` and constructs Figure 3D's exact threshold series: 5, 9, 13, 20, 30, 100, 300, 500, 1000. All 24 x 9 values are identical to `Long_Growth_Curated/data/first_passage_24.csv`, with maximum absolute difference 0.0. The saved artwork visually matches the main-panel scatter, median curve and interquartile band.

Suggested source-register update after the final composite is acknowledged by its editor:

> Main Figure 3D was generated from the curated 24-trajectory UiO-66 long-growth family using render_uio66_entity13_dls_linkage.py. The first-passage values match those supporting Figure S7. Main Figure 3A and the separate 20-run motif study retain their own provenance and are not assigned to this cohort.

This corrects provenance only. It does not validate a physical clock. Figure 3D still shows the earlier x0.175 axis convention, whereas Figure S7 has been returned to raw algorithmic time. Coordinate their labels and Table S27 before re-exporting either panel.

### Figure 5A has a specific raw trajectory

Shared source folder:

```text
Z:/光催化小组/0-papers-in-progress/2025-宋宇航-MOL_growth_mechanism/Figures/Figures/Figure3/Zr_23.51473516375187_FA_14486.64624330687_L_28.22501429085254_Ratio_0.6_Step_1e15.0_Index_0_SC_1.449714838985524_KC_1.72260712693314_2025-08-18_22_52_29/
```

Files:

- `entities_number_3124.8659233585645_index2.pkl` (125,694,297 bytes)
- `plot_crystal_growth_annotations.py`
- `annotated_plots/entities_number_3124.8659233585645_index2_entity_vs_time.png`

Raw-file SHA-256: `e1a0c3f933be7be169f13c594b62e48ed9e2866e7aab356ab7769bc68a93b2d4`.

The folder encodes Zr=23.514735 mM, cap=14486.646243 mM, linker=28.225014 mM, water/DMF=0.6, cS=1.449714839, cK=1.722607127. These are recovered filename inputs, pending confirmation against the executed simulation header. The `index2` pickle suffix must not be assumed to be a repeat ID: its meaning is not established here.

Search Table S27 for:

> Source-scaled elapsed ms after subtraction of the original 620 ms offset.

The current embedded Figure 5A still displays a broken axis with 620-690 ms labels. That does not match a statement that 620 ms was subtracted. Coordinate the final artwork and this sentence.

### Figure 5D percentages have recoverable denominators

Source:

```text
Z:/光催化小组/0-papers-in-progress/2025-宋宇航-MOL_growth_mechanism/Figures/Figures/Figure3/unique_structures_result.csv
```

Examples of exact count recovery:

| Entity | Structure index | Count / nonmissing denominator | Percentage in figure |
|---:|---:|---:|---:|
| 1 | 1 | 130/283 | 45.9% |
| 6 | 1 | 269/284 | 94.7% |
| 7 | 1 | 266/284 | 93.7% |
| 9 | 1 | 262/284 | 92.3% |
| 10 | 1 | 256/284 | 90.1% |
| 11 | 1 | 145/283 | 51.2% |
| 13 | 1 | 240/283 | 84.8% |

These are conditional historical motif frequencies pooled across encoded conditions. They are not 32-repeat probabilities from the new grid. `btb_historical_motif_frequencies.csv` in this review folder records every recovered class for entities 1-13. Before editing the manuscript, establish the selection rule and representative MOL2 mapping with Yuhang.

## Time and fit conventions to preserve

For Figure S7, use `u = stored_scaled_time / 0.175` for every trajectory, first-passage, interval and residence column. Do not apply that conversion to DLS experimental seconds. Keep the historical KMC clock distinct from the current direct-coordinate model clock.

| Quantity | Median in raw algorithmic time u |
|---|---:|
| First passage to N=5 | 1.6008 |
| First passage to N=9 | 426.0339 |
| First passage to N=13 | 26789.2407 |
| First passage to N=30 | 31908.0695 |
| Paired N=9 to N=13 interval | 26531.5825 |
| Paired N=13 to N=30 interval | 51.6103 |
| Paired N=30 to N=1000 interval | 125.1358 |

The three representative Figure S7 labels round to I11=3496, I12=33331, I14=153479 in u, matching the current embedded figure. Paired interval medians are not differences of marginal medians. One run, I03, has a long N=13-to-30 delay; first reaching 13 is not a guarantee of immediate growth.

The fresh DLS refit reproduces nG=2, aN=215.538827 s, bN=2.512205 s, kG=0.001927994842 s^-1, R2=0.9726582834, and RMSE=0.0517498735. It uses all 40 points from 117 to 1170 s, although the displayed panel ends at 600 s. R2 is computed on the processed monotonic apparent-progress variable. The script searches nG in {1,2,3}, selects 2, then refines the continuous parameters; do not describe this implementation as an unrestricted fitted exponent or as nG fixed a priori. The 90 s preparation period is already included in the last two raw-CSV columns.

## Records needed from the newer calculation workspace

The SI says these calculations were executed on a remote 5090 workstation. Their owner is not established by the two Word files. Coordinate with Yuhang to obtain the actual workspace or release bundle; do not equate it with the older institutional archive at 10.26.14.64.

The following named source families were not found in the inspected repository, including ignored files. The August package therefore cannot yet be presented as the complete software/data release for this manuscript.

- `simulation_review_gate_20260831/code/proposed/`: `common_kmc_selector.py`, `common_molecular_runner.py`, and the actual dependency/input versions.
- `common_response_surface_v2_final_condition_summary.csv`, `common_response_surface_v2_final_trace_metrics.csv`, and `common_response_surface_v2_final_pooled_committor.csv`: current response grids and Figure 5 variance decomposition.
- `seeded_propagation_v1_final_*.csv`, `seeded_molecular_runner.py`, `surface_secondary_nucleation_runner.py`, and the three `figure5_surface_vs_seeded_growth_sc1_v1_*` source tables: current main Figure 4.
- `finite_bath_population_runner.py`, `chemistry_mapping.py`, `figure8_absolute_outcomes_v1_*`: finite-inventory BTB concentration comparisons.
- The N=10000 UiO-66 and matched N=2000 BTB structure/PXRD bundles, including saved structures, pattern tables and analysis scripts.
- The newer explicit-cap geometry and defect-intermediate campaigns: launch manifest, original IDs, continuation ledger, endpoint and censoring tables, and occupancy-analysis sources. Our older repair examples do not supply these ensembles.
- `panel_source_correspondence.csv`, `LEGACY_PANEL_CORRESPONDENCE.md`, `all_catalogued_calculation_families.csv`, and `kmc_trace_catalog_20260827`: current provenance records that need the recovered mappings above.
- `Table_S5_1_literature_catalogue.xlsx` and its evidence sources for the expanded literature analysis. This is a separate source family from our KMC archive.

For every family, retain executed source hash, inputs, seed/host structure hash, original run IDs, stopping rules, time definition, exclusions, and figure-building command. A complete final cohort must not count a continuation as an additional independent replicate.

## Suggested coordination order

1. Our side supplies the verified Figure 3D/S7 linkage, DLS source and fit, Figure 5A raw-file location, Figure 5D count table, and older repair data. Use this note and its accompanying checks as the first reconciliation list.
2. Yuhang confirms which current composite files are authoritative and provides the trace/structure identifiers behind Figure 3A, S8-S10 and Figure 7/S42-S45. Keep the historical 20-run motif study separate from the 24-run long-growth cohort.
3. The newer calculation owner provides the 5090 source/data manifests. Reconcile those with SI Tables S23-S29 before updating the GitHub/Zenodo release candidate.
4. Update the source register and the common time labels, then make document edits in one coordinated copy. The original Word files were left intact during this review.

Two additional manuscript edits should be raised during that reconciliation:

- Search MS for `growing one N = 10,000 product` and reconcile with SI Table S7: the rho_off=30 trajectory is capped at N=2922. The sentence should not imply every condition reached 10000.
- The final-view SI contains both `S4.4. Size-resolved finite cluster PXRD of UiO-66 products` and `S4.4. Established-seed growth and structural validation`. This is a real duplicate heading, not deleted text. Renumber its references with the document editor.

## Audit files and limits

- `local_evidence_checks.json`: fresh DLS reproduction, long-growth file/source checks, structural counts, and repair values.
- `legacy_panel_checks.json`: precise source paths, sizes and hashes for the recovered historical panels.
- `figure3D_source_comparison.csv`: the 216 numerical comparisons.
- `btb_historical_motif_frequencies.csv`: recomputed class counts and per-size denominators.
- `MS_review.json`, `SI_review.json`, and final/original text exports: revision-aware reading records.
- Three Python scripts reproduce the extraction and evidence checks. No KMC production simulations were rerun.

Repository HEAD at inspection: `2a9b189`. Existing uncommitted changes were retained. This review added files only in the separate review folder, not to the repository or the two Word documents. No message or file was sent to another person, and no code or data was published.
