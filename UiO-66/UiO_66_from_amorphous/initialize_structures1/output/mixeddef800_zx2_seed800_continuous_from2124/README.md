# UiO-66 Mixed-Defect Continuous Same-Condition Trajectory

This directory contains the current retained **continuous** mixed-defect UiO-66 trajectory used for the single-condition repair story.

The retained logic is:

1. start from the artificial mixed-defect seed `UiO-66_R3_Mixeddef_0.40_seed800`,
2. place that seed in one fixed Zr-rich growth condition,
3. continue under the same unchanged chemistry until a defect-containing nucleus forms and later grows further.

This gives a continuous chain:

- mixed-defect seed `2124`
- defect-containing nucleus `2960`
- latest retained endpoint `3651`

## Fixed Growth Condition

The chemistry is unchanged throughout the retained trajectory:

- `Zr x2`
- linker `x1`
- capping agent `x1`
- `exchange_rxn_time = 0.1 s`
- static dissolution (`DISSOLUTION_UPDATE_INTERVAL_STEPS = None`)

Numerically:

- `Zr_conc = 45.8309411719788`
- `Linker_conc = 69.1596872253079`
- `Capping_agent_conc = 1473.06341756944`

## Stored Structures

Pristine reference:

- `../../data/UiO-66_15x15x15_sphere_R3.mol2`

Artificial mixed-defect seed:

- `./references/UiO-66_R3_Mixeddef_0.40_seed800.pkl`
- `./references/UiO-66_R3_Mixeddef_0.40_seed800.mol2`

Selected same-condition nucleus milestone:

- `./references/assembly_2026-04-03_10-39-56_entity_number_2960.pkl`
- `./references/assembly_2026-04-03_10-39-56_entity_number_2960.mol2`

Latest retained endpoint:

- `./assembly_final_entity_number_3651.pkl`
- `./assembly_final_entity_number_3651.mol2`

## Rendered Figures

Structure comparison:

- `seed_initial_final_structure_triptych.png`
- `seed_initial_final_structure_triptych.svg`

Repair progression:

- `repair_progress_vs_growth.png`
- `repair_progress_vs_growth.svg`
- `repair_progress_vs_growth.csv`

Resume-segment trajectory files:

- `segment02_resume_from_3280_entities_number.pkl`
- `segment02_resume_from_3280_entities_number_seconds.pkl`

The structure triptych shows:

- left = artificial mixed-defect seed
- middle = same-condition nucleus milestone at `2960`
- right = latest retained endpoint at `3651`

## Reference Counts

Pristine `R3` reference:

- total entities: `2769`
- linker entities: `2448`
- `Zr` entities: `321`

Original mixed-defect seed:

- total entities: `2124`
- linker entities: `1857`
- `Zr` entities: `267`
- missing pristine sites: `591` linker sites and `54` `Zr` sites

## Continuous-Trajectory Metrics

At the `2960` nucleus milestone:

- `BDC fill = 0.804`
- `Zr fill = 0.981`

At the latest retained endpoint `3651`:

- `final_total = 3651`
- `final_bdc = 3065`
- `final_zr = 586`
- `filled_bdc_sites = 514 / 591`
- `filled_zr_sites = 54 / 54`
- `BDC fill = 0.870`
- `Zr fill = 1.000`

Best linker-site recovery observed so far along this retained continuous trajectory:

- `best BDC fill = 0.876` at `3500` entities

First complete recovery of original missing `Zr` sites along the plotted progression:

- `Zr fill = 1.000` by about `3080` entities in the retained milestone plot

## Interpretation

This continuous single-condition line supports the following claims:

- the artificial mixed-defect seed can evolve under one fixed Zr-rich condition into a larger defect-containing nucleus,
- the same unchanged condition can fully recover the original missing `Zr` sites,
- substantial original linker-site recovery occurs concurrently,
- the structure continues net growth well beyond the pristine reference size.

Important limit for wording:

- this retained continuous line does **not** yet show full recovery of the original missing linker sites,
- and at the current latest endpoint the original missing linker sites are only partially recovered (`BDC fill = 0.870`, best observed `0.876` at `3500`).

## Continuation / Resume Note

This directory is one continuous same-condition story, but the simulation was resumed from a saved `3280` snapshot to finish the longer trajectory within practical runtime.

No chemistry change was introduced at the resume point:

- same seed family
- same condition
- same output directory
- only the save interval was coarsened in the resumed segment to reduce checkpoint overhead

Logs preserved for the two execution segments:

- `segment01_seed_to3280.log`
- `resume_from_3280.log`

Preferred full-trajectory summary file:

- `repair_progress_vs_growth.csv`

This CSV is the cleanest continuous milestone table for the retained trajectory because it is reconstructed from the saved structure snapshots across both execution segments.

## Recommended Wording

Manuscript-safe short wording:

> Starting from an artificial mixed-defect UiO-66 seed, one fixed Zr-rich growth condition produces a larger defect-containing nucleus, fully repairs the original missing Zr sites, recovers most original missing linker sites, and drives continued growth beyond the pristine reference size.
