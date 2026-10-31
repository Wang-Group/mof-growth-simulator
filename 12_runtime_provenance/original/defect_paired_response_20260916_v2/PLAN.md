# Paired established-crystal defect response

## Question and scope

Does the added singly attached-state energy change vacancy retention/repair during further growth of the same already-large crystal? How does finite precursor supply alter this response and its radial distribution?

The previous unequal-size endpoints cannot isolate the penalty effect. This prospective campaign branches **all 32** N=1000 products from the completed Delta=0, rho=3 fixed-pool cohort into the same factorial conditions. No particle is selected by defect content. This is a paired response experiment on an established-crystal ensemble, NOT de novo nucleation, an equilibrium defect population, or an ablation of explicit geometry. Initial defects and growth history are shared across conditions. Changing the parameters at branch time is a declared intervention; no claim is made that the starting geometry equilibrated under the new conditions.

## Design

- Added dimensionless intermediate-state penalty: 0 and 2.
- Initial removal coefficient: 1 and 3.
- Supply: fixed concentrations; finite free pool 250 Zr6 + 1500 BDC; finite free pool 100 Zr6 + 600 BDC.
- 12 conditions x 32 paired starting particles = 384 trajectories.
- All conditions start with atomic free Zr/BDC=1, s_close=1 and solvent preference=0.5. Acetate and the compact solvent surrogate, all twelve node directions and explicit-H BDC are unchanged.
- Initial species coefficients are 1/7 and 6/7 for all pools. Reducing finite inventory accelerates depletion per incorporation, NOT an imposed exponential decay or a different initial concentration. This is an effective smaller-reservoir comparison.
- Fixed control uses nominal inventory 100/600 but replenishes it; detached oligomers remain outside the tracked assembly and monomers recycle, following the reviewed rules.
- All molecular event functions remain frozen. New code handles parameter assignment, clock reset, snapshots and queue control only.

## Sampling and bounds

Save the common initial structure and exact model-time snapshots at elapsed tau=0.25,0.5,1,2,5,10,20 using the reviewed residual-wait clock. Primary matched times are 0.5,1,2; later times are secondary and can be censored by the size ceiling. Record first crossing N=1250 and 1500 separately, without stopping growth there. Save terminal structure for every outcome. Maximum 200000 additional events, two hours of growth/export allowance per trajectory, and N=2500 safety ceiling. Resource-stopped runs are censored, never classified as zero-defect or no-growth outcomes. A watchdog permits checkpoint export, then terminates an unresponsive process; no automatic relaunch.

At most 64 growth workers combined with the retained eight v1 pilot runs, each with one numerical-library thread; start at 8 for two minutes. At most 2 read-only analysis workers, sharing slots with the pilot. Preserve at least 24 GiB free memory and 100 GiB disk space; stop new launches after three invalid runs. No existing campaign is interrupted or overwritten. The v1 queue was held after eight starts because its first low-removal trace grew from 1000 to 1086 entities by tau=0.215, motivating early matched observation times before fast branches encounter the size ceiling. Those pilots are not extra independent replicates and are excluded from production summaries.

## Analysis specified before outcomes

Primary comparisons are within-starting-particle changes at equal elapsed model time, with paired differences across penalty and pool treatments, not independent samples for each branch. Report completion/censoring for all 32 starts. Censored trajectories are not extrapolated to unobserved times. Secondary size-crossing comparisons are explicitly conditional on crossing the size threshold.

Use the starting particle's central-node lattice orientation for every snapshot to retain spatial correspondence. Save eligible interior linker rows with occupancy, degree and radial coordinate at registration tolerances 0.5,1,2 Angstrom. Separate missing positions from singly attached occupied positions; preserve undefined denominators and registration fractions. Track retained, filled and newly empty lattice positions only where eligible at BOTH observations, and also show the full current-interior denominator to expose growth selection. Inner/outer profiles use local eligible-site denominators, not raw counts. Report both normalized current radius and absolute radius relative to the common initial center.

This experiment can resolve response to the added penalty on matched starting structures, but cannot by itself prove that the explicit crowded intermediate is the unique origin of defects. A cap-free/explicit-intermediate model ablation remains a separate, unrun question.

## Provenance

All source checkpoints, parent manifest, frozen molecular code, launcher and new scripts are SHA-256 recorded. Source data stay under the existing 5090 campaigns. New production artifacts are confined to `/home/wave/MOL_growth_recalculation_20260830/defect_paired_response_20260916_v2`. No manuscript scientific interpretation is changed before results exist. Run `summarize.py` after snapshot analysis to produce all-outcome accounting, particle-level vacancy/radial tables and paired penalty contrasts. V1 pilot results remain separate.
