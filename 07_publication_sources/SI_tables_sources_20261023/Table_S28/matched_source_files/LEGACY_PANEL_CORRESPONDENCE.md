# Historical motif panels: source and parameter correspondence

Read-only source investigation, 16 September 2026. Project-relative paths below refer to the MOL-growth modelling workspace. This note does not regenerate a trajectory or revise the author-selected scientific interpretation.

## Panel-level findings

| Current panel | Actual source established | Statistic and sampling unit | Inputs/conversion established | Remaining gap |
|---|---|---|---|---|
| Figure 4A | Complete Figure 4 image SHA-256 `98c11e0928bcb4f8dbd629c8dac40e603e0eb9564694d0d3eea074315f56709e`; exact image8 in `manuscript_v11.docx`, not a current 32-replicate grid output. | Representative UiO-66 trajectory plus recurring motifs. Original `SI-260515.docx`, body item 207, reports a 20-independent-run motif study. | Original SI reports entropy coefficient 0.789, equilibrium coefficient 1.330, metal 22.915 mM, cap 1473 mM, BDC 34.560 mM, H2O/DMF 0. These are recipe-era inputs, not new-model coordinates. | Exact plotted trajectory ID, raw frequency counts, plotting script, and panel-specific clock factor have not been linked to this artwork. The separate curated 24-run dataset must not be silently substituted for these 20 runs. |
| Figure 5A | `figure_rebuild_20260901/figure6_archived_base.png`, cropped by `figure6_redesign_20260904_v4/build_figure6_legible_v4.py`, lines 60–90; current Figure 5 SHA-256 `cbd546965e892a55e1123987842062d878227b19dbb03cbbb516f8a332397fa1`. | Representative historical BTB-MOL trajectory; original v11 highlights N=4, 7, 10. | Presentation change only: displayed time is original plotted milliseconds minus 620 ms. | Exact trajectory ID and its recipe/clock metadata have not been established. Neither current direct-coordinate grid values nor the old 100-condition screen can be assigned to this one trajectory without a match. |
| Figure 5D | Same historical composite, cropped without recalculating frequencies by the builder, lines 184–191. | Original `manuscript_v11.docx` body item 100 defines **the last structure observed at each entity number among trajectories that subsequently continued to grow**. Percentages are therefore size-conditioned motif frequencies in the successful-growth subset, not unconditional nucleation probabilities, time occupancies, or equilibrium populations. | The conditional sampling rule is documented. Original `SI-260515.docx` section 4.2 supplies artwork but no exact run list or parameter settings for this motif subset. | Per-size denominators/counts, run IDs, and condition membership remain unresolved. Do not label this as 32 trajectories per condition or cite later grid outputs as its source. |
| Figure 5B,C (contrast) | `common_response_surface_v2_final_condition_summary.csv`; builder lines 28 and 51–58. | 1,260 direct-coordinate BTB conditions, 32 trajectories per condition; finite-window N=30 success fraction. B is factorial variance decomposition, C its interpolated P=0.5 boundary. | Nine closure levels, fourteen atomic ratios, ten removal coefficients; documented in SI S8.1 and `BTB_variance_recomputed.csv`. | This dataset supplies B,C only. It supplies neither trajectory A nor motif frequencies D. |
| SI Figure S17 (separate supporting ensemble) | `analysis_workspace/source_package/publication_handoff_2026-08-05/zenodo_dataset/data/uio66_long_growth/data/first_passage_24.csv` and associated curated tables/source snapshot. | 24 curated trajectories, all reaching N=1000; conditional first-passage, residence, and motif comparisons. | Exact recipe and legacy coordinates below. Stored seconds are raw algorithmic time multiplied by 0.175; corrected S17 divides by 0.175. | Matching recipe proximity does not prove these are the same trajectories or counts as Figure 4A. |

The source figure chain is independently documented in `analysis_workspace/rewrite/figure_theory_audit_20260830/FIGURE_SOURCE_AUDIT.md`. The current image identifiers and captions are in `analysis_workspace/rewrite/recalculation_campaign_20260830/student_handoff_20260915_v5/02_current_figures/CURRENT_FIGURE_INDEX.csv`. Its companion `ASSET_SOURCE_MATCHES.csv` identifies the exact current Figure 5 byte source as `manuscript_revision_v29_figure_consolidation_20260910/source_figures/Figure6.png` (also packaged as `current_author_figures/Figure_6.png`). The named v4 builder establishes the historical-panel versus current-grid lineage; its own rendered composite has a different hash and must not be claimed to be byte-identical to the author-carried current composite. The archived historical base itself has SHA-256 `bb8eb97de49e5d09f1ef6bd2e1151ef53f17dfde7c30b549df899fee467b9fe1`.

## What the earlier conversion actually recovered

`analysis_workspace/rewrite/kmc_trace_catalog_20260827/trace_parameters.csv` replays **the formulas executed by each archived program**. Its derived `R_nonlink_over_link_t0`, `pZr_external_code_raw`, `log_E_n*` and `trajectory_time_scale_s` are original-code coordinates. The catalogue does not assert an exact transformation to the present site-resolved selector. Earlier curve interpretation is documented in `analysis_workspace/rewrite/LEGACY_KMC_CALCULATION_REUSE_CATALOG.md` and `analysis_workspace/rewrite/REpzr_growth_analysis_20260828/`.

For the separate curated UiO-66 24-run family, the exact preserved launcher contains:

- atomic metal = 22.9154705859894 mM;
- BDC = 34.57984361265395 mM;
- cap = 1473.06341756944 mM;
- original entropy coefficient = 0.789387907185137;
- original equilibrium coefficient = 1.32975172557788;
- original removal/addition code ratio R(0) = 10.622146147729712;
- original external-event node selector p = 0.01778493215288533;
- plateau ln E = 24.39208633202073; ln E(13) = 24.557744425026694.

Source: `.../github_release/source_snapshot/UiO-66/Long_Growth_Curated/source/UiO66_growth_main_case00_snapshot.py`, lines 1–9, 110–137, and catalogue row `uio_long::00`. The 34.57984 mM BDC value is not identical to the 34.560 mM value printed for the original 20-run motif study. Preserve both provenance records instead of silently correcting one from the other. Applying the archived initial formula to the **rounded 20-run recipe** gives R(0) approximately 10.62584 and p approximately 0.0177946, conditional on that executable family being the one used; these are not a verified panel-specific conversion.

## Equations and the limit of a single-coordinate conversion

For pure M6 supply in the archived driver, let q be atomic metal concentration divided by linker concentration, z the number of carboxylates per linker (2 for BDC, 3 for BTB), cK the original equilibrium coefficient, and w the H2O/DMF mixing input. At zero accumulated base the recovered formulas are

    K_eff,old = 1.64 cK / [1 + 0.01[H2O]/[cap] + 0.01[DMF]/[cap]]
    R_old(0) = [cap] / [z[linker] K_eff,old]
    p_old = exp(-3.6)(2q/z) / [1 + exp(-3.6)(2q/z)]
    ln E_old(N) = 30.9 cS + 0.35[1-ln(N+1)/ln(150)] for N < 150
    ln E_old(N) = 30.9 cS for N >= 150.

The literal piecewise expression in the preserved drivers is reported; it is not replaced by a clipped approximation. The original bath uses [DMF]=12900/(1+w) mM and [H2O]=55500w/(1+w) mM. Some archived families periodically recompute R from a DMF-decomposition/base model. The exact initial coordinate is therefore not automatically the coordinate at every later event.

Sources: archived MOL `mol_growth_main_20250811.py`, lines 28–114 and 152–153; UiO curated launcher as above. The MOL driver uses its own BA/formic-acid deprotonation coefficient (10^(4.19-3.75)); do not copy the UiO acid coefficient into it when reconstructing time dependence.

More importantly, the **meaning of p changes between selectors**. Let F_L be free linker sites, F_M free metal sites, F=F_L+F_M, B existing links and C ready pairs. When both free-site pools exist, the old selector executes

    A_node,old = F p_old
    A_linker,old = F(1-p_old)
    A_remove,old = B R_old
    A_close,old = C E_old(N).

It chooses the incoming identity first and then uniformly chooses a site from the compatible pool. If that pool is empty, it falls back to the other pool. The present direct-coordinate selector instead executes

    A_node,new = F_L p_new
    A_linker,new = F_M(1-p_new)
    A_remove,new = B rho_off
    A_close,new = C exp(30.9 s_close),

with p_new=q/(q+6), no incompatible-pool fallback, and no N-dependent closure term.

Thus substituting p_old into q=6p/(1-p), setting rho_off=R_old and s_close=cS is **not an exact equivalence of trajectory dynamics**. For a single graph state with both pools nonempty one can define

    a_Z = F p_old/F_L;  a_L = F(1-p_old)/F_M;  a_sum = a_Z+a_L
    p_effective = a_Z/a_sum
    rho_effective = R_old/a_sum
    s_effective(N) = ln[E_old(N)/a_sum]/30.9.

These are algebraically derived **state-local** equivalents after rescaling every propensity by a_sum; they change as the graph changes. The corresponding local clock would obey d tau = a_sum du, not one global milliseconds multiplier. This correspondence also ceases to reproduce the old empty-pool fallback. It is a diagnostic explanation, not a relabeling instruction or a current-model reanalysis of the artwork.

Evidence: archived `MOL_Assembly_Large_Correction_20250811.py`, `next_thing_to_do`, lines 390–470; present `analysis_workspace/rewrite/simulation_review_gate_20260831/code/proposed/common_kmc_selector.py`, lines 29–71, 154–187, 230–263 and 269–281.

## Recommended references within the revised paper

- Cite the new SI panel-correspondence table for Figure 4A and Figure 5A,D; retain their original dataset labels and conditional motif definition.
- Cite the current direct-coordinate source table and S8.1 only for Figure 5B,C.
- Cite the curated 24-run table only for its own first-passage/residence analysis (S17), not as the unverified numerical source of historical motif percentages.
- Do not assign new-model constants or recompute artwork frequencies from later calculations. Resolving a missing raw motif/run correspondence requires the original producer's trace list and frequency export; mark this in the author-response file.
