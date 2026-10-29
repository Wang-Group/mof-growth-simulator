# Preliminary defect results and manuscript/SI readiness

## Decision

The modified model now provides usable evidence that missing interior BDC sites and singly attached interior BDC can remain during growth. The paper can report this capability and its measurement sensitivity. These results do not yet close the modulator-identity, defect-versus-modulator-dose, or preferential-core-etching questions. They are a frozen, completion-selected subset, not the completed 36-condition factorial experiment.

## Frozen dataset and provenance

The cohort is the 322 available N ≥ 1000 structures at 19:20 Taipei time on 14 September 2026: 209 original endpoints and 113 first-passage N = 1000 continuations. Each trajectory appears once. Of the 209 original endpoints, 177 reached 50,000 events and 32 stopped at a wall-time cap. The continuation endpoints are not equivalent to 50,000-event endpoints. Only 12 of 36 planned conditions are represented. The continuation launch order also favors larger starting structures; availability is not random sampling. All underlying growth runs remain on 5090 and were not changed for this analysis.

The complete original cohort contains 1152 trajectories; 1120 reached 50,000 events and 32 remained capped after continuation. Conditioning on N ≥ 1000 selects only part of that population. Neither the fraction of selected particles containing defects nor a pooled site fraction is an unconditional experimental defect yield.

Remote source root: `/home/wave/MOL_growth_recalculation_20260830/`. Exact per-sample result and checkpoint paths and hashes are in `selection.json`, `existing_322.json`, and `samples/*.json`. `SHA256.json` verifies the 322 analysis exports and frozen selection; `transfer_verification.json` records local verification. Only compact derived data were retrieved; molecular checkpoints remain on the cluster.

## What the model actually adds

The extension represents a singly attached, physically displaced BDC together with a receiving acetate or solvent-proxy cap. It uses acetate, not benzoate, and permits all twelve Zr₆ connection sites. The solvent proxy and acetate states compete with equal total proposal fractions in this campaign. The penalty Δ = U_D/(RT) takes 0, 2, 4, 6, 8 and 10; ρ₀ takes 0.1, 1 and 3; fixed and finite building-unit pools complete the 36 conditions, with 32 replicates per condition. s_close = 1. The seed contains 16 Zr₆ nodes and 60 BDC linkers; N counts these entities, not atoms. The initial free pool contains 1000 Zr₆ and 6000 BDC, corresponding to atomic Zr/BDC = 1.

For a geometrically permitted move, the extra acceptance factor is exp[−max(0, Δ(n_D,target − n_D,source))]. Thus Δ is an intermediate-state energy penalty, not an independently calculated transition-state barrier. Entering the intermediate can occur during both bond opening and vacancy refilling. Increasing Δ need not monotonically increase the residual defect fraction. Even Δ = 0 retains the explicit intermediate geometry and hard steric exclusion; it is not the previous engine with its intermediate omitted.

In the finite pool, building-unit availability decreases while the reference removal coefficient is fixed. The instantaneous total removal/addition propensity ratio is not ρ₀: it also depends on available species, bonds and sites. Detached monomers can return to the pool, whereas removed oligomers remain outside the tracked aggregate. This is not a complete equilibrium simulation of all fragments. These details must be stated in a separate SI extension, not retroactively attributed to the original nucleation/growth simulations.

## Defect identification and robustness

A central Zr₆ orientation defines a reference lattice. Missing BDC sites are counted only where both neighboring nodes are present, their registration is unambiguous, and the site is inside the inferred particle envelope with a surface buffer of 0.35 nearest-neighbor node spacings. A present, singly attached BDC is a different category; it is not counted as an empty linker site. Linker occupancy uses canonical entity coordinates, while physical intermediate displacement is used in the simulation's steric checks. The explicit intermediate count n_D is also distinct from all graph-degree-one BDC linkers.

Each structure was analyzed at registration tolerances 0.5, 1 and 2 Å and, at 2 Å, envelope parameters α = 0.85, 1.05 and 1.25. These are analysis settings, not changes to the simulation's 2 Å collision or 0.5 Å connection thresholds. Fractions are calculated per trajectory; medians do not pool thousands of lattice sites as independent replicates. Conditions with fewer than 20 eligible interior linker sites are excluded from fraction summaries. Zero eligible sites means undefined, not zero defects.

At the original 0.5 Å registration, 215/322 structures contain apparent missing interior BDC sites. At 2 Å and α = 1.05, 291/322 contain missing sites and 183/322 contain singly attached interior BDC. 154/322 remain positive for missing BDC under all five measurement settings. Full node registration is obtained in 213/322 structures at 2 Å. Consequently, the existence of retained defects is more secure than an absolute defect concentration in every morphology.

### Condition-level descriptive results

| Endpoint group | Δ | ρ₀ | Pool | n | Median N | Median missing BDC sites (%) |
|---|---:|---:|---|---:|---:|---:|
| original_endpoint | 0 | 0.1 | finite | 32 | 4898 | 3.49 |
| original_endpoint | 0 | 0.1 | fixed | 32 | 11088 | 1.60 |
| original_endpoint | 0 | 1.0 | finite | 32 | 1278 | 3.65 |
| original_endpoint | 0 | 1.0 | fixed | 32 | 1627 | 3.90 |
| original_endpoint | 2 | 0.1 | finite | 27 | 1227 | 2.22 |
| original_endpoint | 2 | 0.1 | fixed | 32 | 5390 | 3.31 |
| original_endpoint | 4 | 0.1 | fixed | 22 | 1252 | 2.31 |
| size_followon | 0 | 3.0 | finite | 12 | 1000 | 3.90 |
| size_followon | 0 | 3.0 | fixed | 20 | 1000 | 3.86 |
| size_followon | 2 | 0.1 | finite | 3 | 1000 | 0.94 |
| size_followon | 2 | 1.0 | fixed | 32 | 1000 | 4.24 |
| size_followon | 4 | 0.1 | fixed | 10 | 1000 | 1.19 |
| size_followon | 6 | 0.1 | fixed | 30 | 1000 | 2.70 |
| size_followon | 8 | 0.1 | fixed | 6 | 1000 | 3.04 |

These are available-case summaries. Rows from different endpoint groups must not be treated as a dose response or fitted as a complete factorial experiment.

### A directly interpretable subset

At Δ = 0 and ρ₀ = 1, all 32 original trajectories in each pool mode reach the common 50,000-event endpoint and are represented. Median N is 1627 for fixed and 1278 for finite pools. Median missing-interior-BDC fractions are 3.90% and 3.65%, respectively, at 2 Å and α = 1.05. These are equal-event, not equal-size or equal-physical-time comparisons. They establish retained vacancies in both pool treatments; they do not isolate a pure concentration effect independently of size and growth history.

The complete 32-trajectory Δ = 2, ρ₀ = 1, fixed-pool continuation group reaches N = 1000. 32/32 retain missing interior BDC sites, with median 4.24% (interquartile range 2.94–5.87%). This is a useful common-size result, but comparison with a different Δ at an original 50,000-event endpoint would confound the intermediate penalty with endpoint and size.

Importantly, defects also remain at Δ = 0. The results therefore do not show that a positive extra energy penalty is necessary for defect retention. Explicit steric intermediate geometry remains active at Δ = 0; attributing the change relative to the old engine specifically to that geometry requires a matched old-versus-new ablation. The added penalty cannot be called the established cause simply because the new model produces defects.

### Radial distribution and core etching

Inner and outer bins are r/R < 0.5 and 0.5 ≤ r/R ≤ 1, with R the largest observed node-center radius about the reference center. These are fractions among eligible interior linker sites, not raw counts or shell-volume densities. The following paired particle comparisons require at least 10 eligible sites in both bins. Intervals bootstrap particles (10,000 resamples), not individual sites.

| Pool | Registration tolerance (Å) | Paired particles | Median inner−outer difference (percentage points) | Bootstrap 95% interval |
|---|---:|---:|---:|---|
| fixed | 0.5 | 16 | 0.92 | -2.31 to 2.49 |
| finite | 0.5 | 13 | 0.47 | -3.34 to 2.86 |
| fixed | 1 | 32 | -1.37 | -2.74 to -0.01 |
| finite | 1 | 32 | -0.12 | -1.23 to 1.08 |
| fixed | 2 | 32 | -1.77 | -3.23 to -0.10 |
| finite | 2 | 32 | 0.16 | -0.84 to 0.76 |

At the 2 Å reference tolerance, the finite-pool median paired inner-minus-outer difference is +0.16 percentage points (95% bootstrap interval −0.84 to +0.76), offering no clear inner enrichment. The fixed-pool difference is −1.77 points (−3.23 to −0.10): the outer eligible region has more missing-linker sites in this subset. This direction is also present at 1 Å, whereas the stricter 0.5 Å setting leaves only 16 fixed-pool and 13 finite-pool particles eligible for the paired comparison and gives broad intervals. These results do not support a general statement that the modeled core contains more defects than its shell.

The radial result must be read together with its tolerance sensitivity and unequal eligible populations. A static radial occupancy profile does not demonstrate preferential etching: an unoccupied site may never have filled, and no etchant exposure or location-resolved dissolution rate is recorded here. Core-first etching therefore remains an open mechanistic question, not a conclusion from this dataset.

There are 78 candidate missing-node positions across 63 structures with all twelve nearest-neighbor lattice positions occupied at the reference setting. This stronger local criterion is useful for selecting structures for inspection, but still does not establish when a node was lost or a missing-cluster concentration. More weakly supported missing-node candidates are particularly sensitive to the inferred envelope around branched particles.

## What changes in the paper, and what does not

1. Add a short, clearly identified model-extension result in the UiO-66 discussion, with detailed model definitions and robustness in the SI. Do not replace the earlier nucleation/growth results: they use a different engine and endpoint.
2. Correct SI paragraphs 291 and 295, which describe replaceable caps as unimplemented. A validated acetate/solvent intermediate extension now exists; this does not establish a completed substituted-benzoate comparison. Retain the prospective identity-specific section as pending until its own results are checked.
3. Keep the older imposed-vacancy repair examples in S6.7 separate from this spontaneously incomplete growth ensemble. The present analysis is not a measured repair rate and does not supersede the old trajectories.
4. The latest main-text conclusion calls defect repair a next quantitative test. Retained-defect capability is now demonstrated, while quantitative repair kinetics remain a next test. A proposed narrowly scoped clarification is supplied separately.
5. Preserve the author's experimental interpretation, citations, equations and comments. These runs do not resolve main-text comment 0 (paragraph 70), requesting a quantitative experimental nucleation-versus-growth comparison. They also do not answer comment 1 (paragraph 89), requesting a detailed SI explanation of the BTB-MOL variance partition and first-order/total-effect parameter importance. Both comments remain open; neither concerns the present defect-counting calculation.
6. Do not restore the deleted broad defect/etching section or add literature conclusions based only on this snapshot. No new bibliography numbers are assigned here.

## Remaining work needed for a stronger defect claim

Finish the scheduled common-size cohorts or explicitly analyze the entire original 1152-run cohort with eligibility/censoring reported. Record time-resolved site histories to distinguish never-filled vacancies, actual removal, and repair. Compare Δ at a matched endpoint and report size/morphology dependence, rather than assuming a monotonic barrier effect. Check representative candidate missing-node structures visually and against their graph histories. These are distinct from completing aromatic modulator identity calculations.

## Files

`preliminary_defects.png/pdf/svg`: same-event comparison, radial paired profiles and envelope sensitivity. `coverage.png/pdf/svg`: completion coverage, not a physical phase diagram. `condition_statistics.csv`, `diagnostic_counts.csv`, `radial_counts.csv`, `node_support.csv`: per-condition/per-particle evidence. `interpretation_statistics.json`: bootstrap and robustness summaries. `manuscript_SI_proposed_insertions.md`: proposed text, not an overwrite of the author's files. `manuscript_inventory.json`: reviewed document hashes and comment inventory.
