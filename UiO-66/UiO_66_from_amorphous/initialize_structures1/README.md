# UiO-66 Defect-Nucleus Workspace

This directory contains the active UiO-66 amorphous / defect-seeded growth workspace.

It now keeps one cleaned single-condition defect-repair result and one separate mixed-nucleus output tree:

- `output/mixeddef800_zx2_seed800_continuous_from2124/`
- `output/mixed_nuclei/`

The first directory is the current retained continuous mixed-defect repair case.
The second directory contains separate nucleus-composition studies, including ongoing `Zr6/Zr12`-mixed work, and should be interpreted independently.

## Current Main Result

The current preferred story is:

1. start from an artificial mixed-defect UiO-66 seed,
2. keep the external chemistry unchanged,
3. continue the trajectory long enough to observe formation of a defect-containing nucleus, defect repair, and further growth.

The fixed condition used for the retained case is:

- `Zr x2`
- linker `x1`
- capping agent `x1`
- `exchange_rxn_time = 0.1 s`
- static dissolution (`DISSOLUTION_UPDATE_INTERVAL_STEPS = None`)

Numerically:

- `Zr_conc = 45.8309411719788`
- `Linker_conc = 69.1596872253079`
- `Capping_agent_conc = 1473.06341756944`

## Structures Used

Pristine reference:

- `data/UiO-66_15x15x15_sphere_R3.mol2`

Retained references for the current case:

- `output/mixeddef800_zx2_seed800_continuous_from2124/references/UiO-66_R3_Mixeddef_0.40_seed800.pkl`
- `output/mixeddef800_zx2_seed800_continuous_from2124/references/UiO-66_R3_Mixeddef_0.40_seed800.mol2`
- `output/mixeddef800_zx2_seed800_continuous_from2124/references/assembly_2026-04-03_10-39-56_entity_number_2960.pkl`
- `output/mixeddef800_zx2_seed800_continuous_from2124/references/assembly_2026-04-03_10-39-56_entity_number_2960.mol2`

Current endpoint:

- `output/mixeddef800_zx2_seed800_continuous_from2124/assembly_final_entity_number_3651.pkl`
- `output/mixeddef800_zx2_seed800_continuous_from2124/assembly_final_entity_number_3651.mol2`

Quick-look comparison figures:

- `output/mixeddef800_zx2_seed800_continuous_from2124/seed_initial_final_structure_triptych.png`
- `output/mixeddef800_zx2_seed800_continuous_from2124/seed_initial_final_structure_triptych.svg`
- `output/mixeddef800_zx2_seed800_continuous_from2124/repair_progress_vs_growth.png`
- `output/mixeddef800_zx2_seed800_continuous_from2124/repair_progress_vs_growth.svg`
- `output/mixeddef800_zx2_seed800_continuous_from2124/repair_progress_vs_growth.csv`

Detailed case note:

- `output/mixeddef800_zx2_seed800_continuous_from2124/README.md`

## Quantitative Summary

Pristine `R3` reference:

- total entities: `2769`
- linker entities: `2448`
- `Zr` entities: `321`

Original mixed-defect seed:

- total entities: `2124`
- linker entities: `1857`
- `Zr` entities: `267`
- missing pristine sites: `591` linker sites and `54` `Zr` sites

Chosen defect-containing nucleus milestone in the continuous trajectory:

- total entities: `2960`

Latest analyzed endpoint:

- `final_total = 3651`
- `final_bdc = 3065`
- `final_zr = 586`
- `filled_bdc_sites = 514 / 591`
- `filled_zr_sites = 54 / 54`
- `BDC fill = 0.870`
- `Zr fill = 1.000`
- `final_total = 3651 > 2769` pristine

Best linker-site recovery observed so far along this retained continuous trajectory:

- `best BDC fill = 0.876` at `3500` entities

Interpretation that is safe to reuse:

- the same unchanged growth condition supports complete recovery of the original missing `Zr` sites,
- the same trajectory also recovers most original missing linker sites,
- the nucleus continues net growth beyond the pristine reference size.

Important limit:

- this exact retained case does not yet support the statement that both defect classes are fully healed under one fixed condition, because `BDC fill` is not yet `1.000`.

## Relevant Scripts

Core scripts used for this line of work:

- `generate_defects.py`
- `generate_defect_seeds.py`
- `run_seeded_growth.py`
- `run_defect_growth_matrix.py`
- `plot_same_condition_repair_progress.py`
- `plot_structure_triptych.py`
- `UiO66_growth_main_conc.py`
- `UiO66_Assembly_Large_Correction_conc.py`

## Environment Note

Use 64-bit Python for this workspace:

- `C:\Users\yibinjiang\AppData\Local\Programs\Python\Python311\python.exe`

Do not rely on the default `python` on this machine if it resolves to a 32-bit install.

## Cleanup Status

To keep the repository output tree readable, older exploratory defect-growth output directories were removed after the retained single-condition case was assembled.

The current single-condition result directory was made self-contained by moving its original mixed-defect seed and its selected initial defect nucleus into:

- `output/mixeddef800_zx2_seed800_continuous_from2124/references/`

## Recommended Wording

Short manuscript-safe wording:

> Under one fixed Zr-rich growth condition, a residual mixed-defect UiO-66 nucleus continues to evolve, fully repairs the original missing Zr sites, recovers most original missing linker sites, and grows beyond the pristine reference size.
